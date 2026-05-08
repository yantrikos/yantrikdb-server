//! Raft cluster assembly — wires log_storage + state_machine +
//! HttpRaftNetworkFactory + Raft into a working cluster, **and**
//! enforces the production invariants the individual components can't
//! enforce on their own.
//!
//! ## Production invariants (fail-fast at assembly time)
//!
//! 1. **mTLS is required when cluster mode is OpenRaft.** If
//!    [`RaftClusterMode::OpenRaft`] is requested but `cluster_tls`
//!    isn't fully specified, [`build_raft_cluster`] returns
//!    [`AssemblyError::MtlsRequired`] before any sockets are opened.
//!    This means an operator can't accidentally ship plaintext cluster
//!    traffic — a misconfigured server refuses to start at all.
//! 2. **Dev-mode is allowed but loud.** A deployment that sets
//!    `cluster_tls.dev_mode = true` AND OpenRaft mode is permitted (so
//!    dev clusters can run with self-signed certs) but emits a warning
//!    log at assembly time.
//! 3. **Disabled mode is plaintext-OK.** When mode is
//!    [`RaftClusterMode::Disabled`] (single-node), the assembly
//!    function isn't called at all — the existing `LocalSqliteCommitter`
//!    is used directly. The gate is "openraft enabled".
//!
//! ## What this module does NOT include
//!
//! - The actual server.rs wiring that calls [`build_raft_cluster`] —
//!   that's a follow-up when the live cluster mode flag exists.
//! - Snapshot transport optimization (chunking, bincode) — see RFC 010
//!   PR-4 review notes.
//! - Linearizable reads via `Raft::ensure_linearizable()`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use openraft::{Config, Raft};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::committer::RaftCommitter;
use super::http_network::HttpRaftNetworkFactory;
use super::log_storage::SqliteRaftLogStorage;
use super::state_machine::YantrikStateMachine;
use super::types::{YantrikNode, YantrikNodeId, YantrikRaftTypeConfig};
use crate::commit::MutationCommitter;
use crate::security::cluster_tls::{ClusterTlsConfig, ClusterTlsError};

/// Whether this server runs in cluster mode. `Disabled` means
/// single-node (existing `LocalSqliteCommitter`); `OpenRaft` means a
/// real Raft cluster with mTLS-required production gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RaftClusterMode {
    /// Single-node — no cluster transport, no Raft. Application code
    /// holds an `Arc<LocalSqliteCommitter>` and serves writes locally.
    Disabled,
    /// 3+ node cluster via openraft. Requires fully-specified
    /// `cluster_tls` (or `dev_mode = true` with a warning log).
    OpenRaft,
}

impl Default for RaftClusterMode {
    fn default() -> Self {
        RaftClusterMode::Disabled
    }
}

/// What backs the HTTP handler write path. RFC 010 PR-6.5 boot invariant
/// gate: `OpenRaft` cluster mode REQUIRES `RaftSubmitter` here. Any
/// other combination is rejected at assembly time so the cluster cannot
/// regress to "cosmetic openraft" mode (writes land locally, replication
/// reports healthy but moves zero application bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandlerWritePath {
    /// Handlers call into a `LocalSqliteSubmitter` (or directly into
    /// `engine.record()` on the legacy unmigrated path). Single-node
    /// only — pairs exclusively with `RaftClusterMode::Disabled`.
    LocalSqlite,
    /// Handlers call into a `RaftSubmitter` that routes through openraft
    /// consensus. Pairs exclusively with `RaftClusterMode::OpenRaft`.
    RaftSubmitter,
}

impl Default for HandlerWritePath {
    fn default() -> Self {
        HandlerWritePath::LocalSqlite
    }
}

#[derive(Debug, Error)]
pub enum AssemblyError {
    /// `cluster.mode = "openraft"` was requested but `cluster_tls`
    /// isn't fully specified. Fail-fast — refuses to start the server.
    #[error(
        "openraft mode requires fully-specified cluster_tls (cert_path, key_path, ca_path) \
         to prevent accidental plaintext cluster traffic; missing field: {missing}"
    )]
    MtlsRequired { missing: &'static str },

    /// `cluster_tls` failed to build into rustls / reqwest configs.
    #[error("cluster_tls build failed: {0}")]
    ClusterTls(#[from] ClusterTlsError),

    /// Building the reqwest client for inter-node Raft RPCs failed.
    #[error("reqwest client build failed: {0}")]
    ReqwestBuild(String),

    /// openraft `Raft::new` returned a fatal error during assembly.
    /// Indicates a fundamental misconfiguration — surface to the
    /// operator unchanged.
    #[error("openraft Raft::new fatal: {0}")]
    RaftNew(String),

    /// Reading a cluster_tls cert/key/CA file as bytes for reqwest
    /// failed.
    #[error("read PEM file `{path}`: {source}")]
    PemRead {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// `cluster.mode = "openraft"` was requested but the handler write
    /// path is configured as something other than `RaftSubmitter`. This
    /// is the cosmetic-openraft regression gate: openraft can be assembled
    /// with a non-Raft handler path, but if the binary boots in that
    /// state it lies in `/v1/health` (reports `healthy: true`) while
    /// every write bypasses replication. Refuse at boot.
    #[error(
        "openraft mode requires handler_write_path = \"raft_submitter\"; got {actual:?}. \
         Configure cluster.handler_write_path = \"raft_submitter\", or set \
         cluster.raft_mode = \"disabled\" for single-node deployments."
    )]
    WritePathMismatch {
        actual: HandlerWritePath,
        expected: HandlerWritePath,
    },

    /// `cluster.mode = "openraft"` was requested but the cluster has
    /// fewer than 2 declared peers. A 1-peer "cluster" can't form a
    /// quorum, can't survive a single-node failure, and is almost
    /// certainly a misconfiguration. Refuse at boot.
    #[error(
        "openraft mode requires at least 2 peers (got {have}). \
         A 1-peer cluster has no quorum semantics; configure additional \
         peers in cluster.peers or set cluster.raft_mode = \"disabled\"."
    )]
    InsufficientPeers { have: usize, need: usize },
}

/// Inputs to [`build_raft_cluster`].
pub struct RaftAssemblyConfig {
    pub mode: RaftClusterMode,
    /// Local node id within the cluster.
    pub node_id: YantrikNodeId,
    /// HTTP address other peers reach this node at (e.g.
    /// `https://10.0.0.5:7100`). Stored in the membership log.
    pub node_addr: String,
    /// Cluster TLS config — required when `mode == OpenRaft`. Ignored
    /// when `mode == Disabled`.
    pub cluster_tls: Option<ClusterTlsConfig>,
    /// Full cluster voter set (typically including this node's address).
    /// PR-6.5 boot invariant: `OpenRaft` mode requires `peers.len() >= 2`.
    /// Empty / 1-element peer lists are misconfiguration and rejected
    /// at boot.
    pub peers: Vec<String>,
    /// What backs the HTTP handler write path. PR-6.5 boot invariant:
    /// `OpenRaft` mode requires `RaftSubmitter` here.
    pub write_path: HandlerWritePath,
    /// Per-RPC timeout for the reqwest client.
    pub request_timeout: Duration,
    /// openraft heartbeat / election tuning.
    pub openraft_config: Config,
}

impl RaftAssemblyConfig {
    /// Reasonable defaults for production: 200ms heartbeat, 800ms-1.6s
    /// election timeout, 10s RPC timeout.
    pub fn production_defaults(node_id: YantrikNodeId, node_addr: String) -> Self {
        Self {
            mode: RaftClusterMode::OpenRaft,
            node_id,
            node_addr,
            cluster_tls: None,                           // operator MUST supply
            peers: Vec::new(),                           // operator MUST supply (PR-6.5 gate)
            write_path: HandlerWritePath::RaftSubmitter, // openraft requires it
            request_timeout: Duration::from_secs(10),
            openraft_config: Config {
                cluster_name: "yantrikdb".into(),
                heartbeat_interval: 200,
                election_timeout_min: 800,
                election_timeout_max: 1600,
                ..Default::default()
            },
        }
    }

    /// Validate the assembly config against the PR-6.5 boot invariants.
    /// Called from [`build_raft_cluster`] before any sockets are opened.
    /// Exposed pub(crate) so tests can hit the gate without spinning up
    /// the full Raft + reqwest stack.
    ///
    /// Invariants enforced (in order):
    /// 1. `OpenRaft` mode requires `RaftSubmitter` handler write path.
    /// 2. `OpenRaft` mode requires `peers.len() >= 2` for real quorum.
    /// 3. `cluster_tls` checks happen later inside [`build_raft_cluster`]
    ///    via [`validate_cluster_tls_for_openraft`].
    pub(crate) fn validate(&self) -> Result<(), AssemblyError> {
        if self.mode == RaftClusterMode::OpenRaft {
            if self.write_path != HandlerWritePath::RaftSubmitter {
                return Err(AssemblyError::WritePathMismatch {
                    actual: self.write_path,
                    expected: HandlerWritePath::RaftSubmitter,
                });
            }
            if self.peers.len() < 2 {
                return Err(AssemblyError::InsufficientPeers {
                    have: self.peers.len(),
                    need: 2,
                });
            }
        }
        Ok(())
    }
}

/// Result of a successful assembly.
pub struct RaftAssembly {
    pub raft: Arc<Raft<YantrikRaftTypeConfig>>,
    pub committer: RaftCommitter,
    /// Snapshot of the bound network factory — `RaftCommitter` already
    /// closed over its own copy; this is here so callers can build
    /// additional clients (e.g. for join-cluster CLI flows).
    pub network_factory: HttpRaftNetworkFactory,
}

/// Validate the cluster_tls config for openraft mode. Returns the
/// specific missing field so the error message is actionable.
fn validate_cluster_tls_for_openraft(
    cluster_tls: Option<&ClusterTlsConfig>,
) -> Result<&ClusterTlsConfig, AssemblyError> {
    let cfg = cluster_tls.ok_or(AssemblyError::MtlsRequired {
        missing: "cluster_tls (entire section)",
    })?;
    if cfg.cert_path.is_none() {
        return Err(AssemblyError::MtlsRequired {
            missing: "cert_path",
        });
    }
    if cfg.key_path.is_none() {
        return Err(AssemblyError::MtlsRequired {
            missing: "key_path",
        });
    }
    if cfg.ca_path.is_none() {
        return Err(AssemblyError::MtlsRequired { missing: "ca_path" });
    }
    Ok(cfg)
}

/// Build a reqwest client that does mTLS using the cluster certs.
fn build_reqwest_client_for_cluster(
    cfg: &ClusterTlsConfig,
    request_timeout: Duration,
) -> Result<reqwest::Client, AssemblyError> {
    let cert_path = cfg.cert_path.as_ref().expect("validated above");
    let key_path = cfg.key_path.as_ref().expect("validated above");
    let ca_path = cfg.ca_path.as_ref().expect("validated above");

    let cert_pem = std::fs::read(cert_path).map_err(|e| AssemblyError::PemRead {
        path: cert_path.clone(),
        source: e,
    })?;
    let key_pem = std::fs::read(key_path).map_err(|e| AssemblyError::PemRead {
        path: key_path.clone(),
        source: e,
    })?;
    let ca_pem = std::fs::read(ca_path).map_err(|e| AssemblyError::PemRead {
        path: ca_path.clone(),
        source: e,
    })?;

    // reqwest's Identity::from_pem accepts a single bundle (cert + key).
    // We concatenate so the operator can store them separately without
    // having to keep a "bundled" file in sync.
    let mut bundle = cert_pem.clone();
    bundle.push(b'\n');
    bundle.extend_from_slice(&key_pem);

    let identity = reqwest::Identity::from_pem(&bundle)
        .map_err(|e| AssemblyError::ReqwestBuild(format!("Identity::from_pem: {e}")))?;
    let ca_cert = reqwest::Certificate::from_pem(&ca_pem)
        .map_err(|e| AssemblyError::ReqwestBuild(format!("Certificate::from_pem: {e}")))?;

    let mut builder = reqwest::Client::builder()
        .timeout(request_timeout)
        .identity(identity)
        .add_root_certificate(ca_cert)
        // Don't auto-trust system roots: cluster traffic uses the
        // explicit cluster CA.
        .tls_built_in_root_certs(false);

    if cfg.dev_mode {
        tracing::warn!(
            "cluster_tls.dev_mode = true — accepting self-signed peer certs. \
             NEVER set this in production."
        );
        builder = builder.danger_accept_invalid_certs(true);
    }

    builder.build().map_err(|e| {
        // Surface the source chain — the bare reqwest::Error is just
        // "builder error" with the actual cause buried in the source chain.
        let mut chain = format!("build: {e}");
        let mut src = std::error::Error::source(&e);
        while let Some(s) = src {
            chain.push_str(&format!(" / {s}"));
            src = s.source();
        }
        AssemblyError::ReqwestBuild(chain)
    })
}

/// Wire a Raft cluster from its constituent parts. Enforces the
/// mTLS gate (production invariant #1).
///
/// The `local` committer is the same `MutationCommitter` the state
/// machine apply path drives — typically an `Arc<LocalSqliteCommitter>`.
/// The `RaftCommitter` then routes writes through openraft and reads
/// through `local` (stale-OK semantics).
pub async fn build_raft_cluster(
    cfg: RaftAssemblyConfig,
    log_storage: SqliteRaftLogStorage,
    local: Arc<dyn MutationCommitter>,
) -> Result<RaftAssembly, AssemblyError> {
    // PR-6.5 boot invariants: write-path coupling + peer count. Run
    // BEFORE the cluster_tls / cert IO checks so a misconfigured
    // handler path fails fast even if certs are missing.
    cfg.validate()?;
    let cluster_tls = validate_cluster_tls_for_openraft(cfg.cluster_tls.as_ref())?;
    let client = build_reqwest_client_for_cluster(cluster_tls, cfg.request_timeout)?;
    let network_factory = HttpRaftNetworkFactory::new(client, cfg.request_timeout);

    let validated_config = Arc::new(
        cfg.openraft_config
            .validate()
            .map_err(|e| AssemblyError::RaftNew(format!("openraft Config::validate: {e}")))?,
    );

    // RFC 010 PR-6.4 — state machine apply path needs an Applier so
    // every commit also writes engine state, not just the commit log.
    // For now build_raft_cluster constructs a `LocalApplier` placeholder
    // that returns NotYetWired for engine-mutating variants; the
    // production wiring (EngineApplier with TenantPool-backed resolver)
    // is plumbed from main.rs in a follow-up commit. The state machine
    // tolerates NotYetWired and logs a warning so this transitional
    // state is operator-visible.
    let applier: Arc<dyn crate::commit::Applier> = Arc::new(crate::commit::LocalApplier::new());
    let state_machine = YantrikStateMachine::new(local.clone(), applier);
    let raft = Raft::<YantrikRaftTypeConfig>::new(
        cfg.node_id,
        validated_config,
        network_factory.clone(),
        log_storage,
        state_machine,
    )
    .await
    .map_err(|e| AssemblyError::RaftNew(format!("{e}")))?;
    let raft = Arc::new(raft);

    let committer = RaftCommitter::new(raft.clone(), local);

    Ok(RaftAssembly {
        raft,
        committer,
        network_factory,
    })
}

/// Convenience helper: initialize a brand-new single-node cluster on
/// the given assembly. Used during cluster bootstrap (`yantrikdb
/// cluster init`). For joining an existing cluster, callers use
/// `Raft::add_learner` + `Raft::change_membership` against the existing
/// leader instead.
pub async fn initialize_single_node(
    assembly: &RaftAssembly,
    node_addr: String,
) -> Result<
    (),
    openraft::error::RaftError<
        YantrikNodeId,
        openraft::error::InitializeError<YantrikNodeId, YantrikNode>,
    >,
> {
    let me = {
        let metrics = assembly.raft.metrics().borrow().clone();
        metrics.id
    };
    let mut nodes = BTreeMap::new();
    nodes.insert(me, YantrikNode::new(node_addr));
    assembly.raft.initialize(nodes).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_tls() -> ClusterTlsConfig {
        ClusterTlsConfig::default()
    }

    fn tls_with(cert: Option<&str>, key: Option<&str>, ca: Option<&str>) -> ClusterTlsConfig {
        ClusterTlsConfig {
            cert_path: cert.map(std::path::PathBuf::from),
            key_path: key.map(std::path::PathBuf::from),
            ca_path: ca.map(std::path::PathBuf::from),
            dev_mode: false,
            rotate_check_secs: 60,
        }
    }

    #[test]
    fn openraft_mode_rejects_missing_cluster_tls_section() {
        let err = validate_cluster_tls_for_openraft(None).unwrap_err();
        match err {
            AssemblyError::MtlsRequired { missing } => assert!(missing.contains("cluster_tls")),
            other => panic!("expected MtlsRequired, got {other:?}"),
        }
    }

    #[test]
    fn openraft_mode_rejects_empty_tls_config() {
        let cfg = empty_tls();
        let err = validate_cluster_tls_for_openraft(Some(&cfg)).unwrap_err();
        match err {
            AssemblyError::MtlsRequired { missing } => assert_eq!(missing, "cert_path"),
            other => panic!("expected MtlsRequired, got {other:?}"),
        }
    }

    #[test]
    fn openraft_mode_rejects_missing_key() {
        let cfg = tls_with(Some("/tmp/cert.pem"), None, Some("/tmp/ca.pem"));
        let err = validate_cluster_tls_for_openraft(Some(&cfg)).unwrap_err();
        match err {
            AssemblyError::MtlsRequired { missing } => assert_eq!(missing, "key_path"),
            other => panic!("expected MtlsRequired, got {other:?}"),
        }
    }

    #[test]
    fn openraft_mode_rejects_missing_ca() {
        let cfg = tls_with(Some("/tmp/cert.pem"), Some("/tmp/key.pem"), None);
        let err = validate_cluster_tls_for_openraft(Some(&cfg)).unwrap_err();
        match err {
            AssemblyError::MtlsRequired { missing } => assert_eq!(missing, "ca_path"),
            other => panic!("expected MtlsRequired, got {other:?}"),
        }
    }

    #[test]
    fn openraft_mode_accepts_fully_specified_tls() {
        let cfg = tls_with(
            Some("/tmp/cert.pem"),
            Some("/tmp/key.pem"),
            Some("/tmp/ca.pem"),
        );
        validate_cluster_tls_for_openraft(Some(&cfg))
            .expect("fully-specified config must pass validation");
    }

    #[tokio::test]
    async fn build_raft_cluster_fails_on_missing_cluster_tls() {
        // The whole point of the gate. Even if everything else is ready
        // to go, no cluster_tls means the assembly refuses.
        let local = Arc::new(crate::commit::LocalSqliteCommitter::open_in_memory().unwrap())
            as Arc<dyn MutationCommitter>;
        let log_storage = SqliteRaftLogStorage::open_in_memory();
        let cfg = RaftAssemblyConfig {
            mode: RaftClusterMode::OpenRaft,
            node_id: YantrikNodeId::new(1),
            node_addr: "https://127.0.0.1:7100".into(),
            cluster_tls: None,
            peers: vec![
                "https://127.0.0.1:7100".into(),
                "https://127.0.0.1:7101".into(),
            ],
            write_path: HandlerWritePath::RaftSubmitter,
            request_timeout: Duration::from_secs(1),
            openraft_config: Config::default(),
        };
        match build_raft_cluster(cfg, log_storage, local).await {
            Err(AssemblyError::MtlsRequired { .. }) => {}
            Err(other) => panic!("expected MtlsRequired, got {other:?}"),
            Ok(_) => panic!("expected MtlsRequired, assembly succeeded"),
        }
    }

    #[tokio::test]
    async fn build_raft_cluster_fails_on_unreadable_cert_files() {
        // Paths exist syntactically but point at non-existent files.
        // Validation passes (all three paths supplied), but
        // build_reqwest_client_for_cluster errors out on read.
        let local = Arc::new(crate::commit::LocalSqliteCommitter::open_in_memory().unwrap())
            as Arc<dyn MutationCommitter>;
        let log_storage = SqliteRaftLogStorage::open_in_memory();
        let cluster_tls = tls_with(
            Some("/nonexistent/cert.pem"),
            Some("/nonexistent/key.pem"),
            Some("/nonexistent/ca.pem"),
        );
        let cfg = RaftAssemblyConfig {
            mode: RaftClusterMode::OpenRaft,
            node_id: YantrikNodeId::new(1),
            node_addr: "https://127.0.0.1:7100".into(),
            cluster_tls: Some(cluster_tls),
            peers: vec![
                "https://127.0.0.1:7100".into(),
                "https://127.0.0.1:7101".into(),
            ],
            write_path: HandlerWritePath::RaftSubmitter,
            request_timeout: Duration::from_secs(1),
            openraft_config: Config::default(),
        };
        match build_raft_cluster(cfg, log_storage, local).await {
            Err(AssemblyError::PemRead { .. }) => {}
            Err(other) => panic!("expected PemRead, got {other:?}"),
            Ok(_) => panic!("expected PemRead, assembly succeeded"),
        }
    }

    #[test]
    fn cluster_mode_default_is_disabled() {
        // Operators must opt INTO cluster mode. A fresh server config
        // with no cluster section runs single-node — no plaintext gate
        // can be tripped by accident.
        assert_eq!(RaftClusterMode::default(), RaftClusterMode::Disabled);
    }

    #[test]
    fn production_defaults_demand_explicit_tls() {
        // production_defaults() returns mode=OpenRaft + cluster_tls=None.
        // Operator must explicitly supply cluster_tls before assembly
        // succeeds — the defaults exist as a starting template, not a
        // ready-to-run config.
        let d = RaftAssemblyConfig::production_defaults(
            YantrikNodeId::new(1),
            "https://10.0.0.1:7100".into(),
        );
        assert_eq!(d.mode, RaftClusterMode::OpenRaft);
        assert!(
            d.cluster_tls.is_none(),
            "production_defaults must NOT bake in any cluster_tls — operator supplies it"
        );
    }

    // ── PR-6.5 boot invariant tests ─────────────────────────────────

    fn cfg_for(
        mode: RaftClusterMode,
        write_path: HandlerWritePath,
        peers: Vec<String>,
    ) -> RaftAssemblyConfig {
        RaftAssemblyConfig {
            mode,
            node_id: YantrikNodeId::new(1),
            node_addr: "https://10.0.0.1:7100".into(),
            cluster_tls: Some(tls_with(
                Some("/tmp/cert.pem"),
                Some("/tmp/key.pem"),
                Some("/tmp/ca.pem"),
            )),
            peers,
            write_path,
            request_timeout: Duration::from_secs(1),
            openraft_config: Config::default(),
        }
    }

    fn three_peer_set() -> Vec<String> {
        vec![
            "https://10.0.0.1:7100".into(),
            "https://10.0.0.2:7100".into(),
            "https://10.0.0.3:7100".into(),
        ]
    }

    #[test]
    fn pr_6_5_openraft_with_localsqlite_write_path_is_rejected() {
        // The cosmetic-openraft regression gate. If this test ever
        // accepts the misconfiguration, the whole point of PR 6.5 is
        // gone — refuse the boot, not eventually surface a 503 in
        // /v1/health.
        let cfg = cfg_for(
            RaftClusterMode::OpenRaft,
            HandlerWritePath::LocalSqlite,
            three_peer_set(),
        );
        match cfg.validate() {
            Err(AssemblyError::WritePathMismatch { actual, expected }) => {
                assert_eq!(actual, HandlerWritePath::LocalSqlite);
                assert_eq!(expected, HandlerWritePath::RaftSubmitter);
            }
            other => panic!("expected WritePathMismatch, got {other:?}"),
        }
    }

    #[test]
    fn pr_6_5_openraft_with_empty_peers_is_rejected() {
        let cfg = cfg_for(
            RaftClusterMode::OpenRaft,
            HandlerWritePath::RaftSubmitter,
            vec![],
        );
        match cfg.validate() {
            Err(AssemblyError::InsufficientPeers { have, need }) => {
                assert_eq!(have, 0);
                assert_eq!(need, 2);
            }
            other => panic!("expected InsufficientPeers, got {other:?}"),
        }
    }

    #[test]
    fn pr_6_5_openraft_with_one_peer_is_rejected() {
        // 1-peer "cluster" has no quorum semantics — almost certainly a
        // misconfiguration where the operator forgot to add the others.
        let cfg = cfg_for(
            RaftClusterMode::OpenRaft,
            HandlerWritePath::RaftSubmitter,
            vec!["https://10.0.0.1:7100".into()],
        );
        assert!(matches!(
            cfg.validate(),
            Err(AssemblyError::InsufficientPeers { have: 1, need: 2 })
        ));
    }

    #[test]
    fn pr_6_5_openraft_with_two_peers_passes() {
        // Two-voter cluster (e.g. .140 + .141 in the homelab) is
        // intentionally permitted as the minimum viable cluster.
        let cfg = cfg_for(
            RaftClusterMode::OpenRaft,
            HandlerWritePath::RaftSubmitter,
            vec![
                "https://10.0.0.1:7100".into(),
                "https://10.0.0.2:7100".into(),
            ],
        );
        cfg.validate()
            .expect("two-peer openraft cluster must validate");
    }

    #[test]
    fn pr_6_5_disabled_mode_does_not_demand_peers() {
        // Single-node mode is plaintext-OK, peers-OK, write-path-OK in
        // any combination. The gate only fires when openraft is on.
        let cfg = cfg_for(
            RaftClusterMode::Disabled,
            HandlerWritePath::LocalSqlite,
            vec![],
        );
        cfg.validate()
            .expect("single-node mode must validate without peers");
    }

    #[test]
    fn pr_6_5_disabled_mode_with_raft_submitter_is_currently_permitted() {
        // Operator declared cluster.write_path = "raft_submitter" but
        // mode = "disabled" — this is a no-op declaration in single-node
        // mode (RaftSubmitter has no Raft to submit through). PR 6.5
        // doesn't reject it because nothing's broken on disk; future
        // PRs may surface a warning log if the combination becomes
        // ambiguous in practice.
        let cfg = cfg_for(
            RaftClusterMode::Disabled,
            HandlerWritePath::RaftSubmitter,
            vec![],
        );
        cfg.validate()
            .expect("Disabled+RaftSubmitter is permitted (no-op declaration)");
    }

    #[tokio::test]
    async fn pr_6_5_build_raft_cluster_runs_validate_first() {
        // The load-bearing wiring assertion: build_raft_cluster MUST
        // run validate() before reading TLS files. Otherwise a misconfigured
        // write_path could surface as a confusing PEM read error instead
        // of the actionable WritePathMismatch.
        let local = Arc::new(crate::commit::LocalSqliteCommitter::open_in_memory().unwrap())
            as Arc<dyn MutationCommitter>;
        let log_storage = SqliteRaftLogStorage::open_in_memory();
        let cfg = cfg_for(
            RaftClusterMode::OpenRaft,
            HandlerWritePath::LocalSqlite, // mismatched
            three_peer_set(),
        );
        match build_raft_cluster(cfg, log_storage, local).await {
            Err(AssemblyError::WritePathMismatch { .. }) => {}
            Err(other) => panic!("expected WritePathMismatch, got {other:?}"),
            Ok(_) => panic!("expected WritePathMismatch, assembly succeeded"),
        }
    }

    #[test]
    fn handler_write_path_default_is_local_sqlite() {
        // Backwards-compat: any existing config that doesn't specify
        // write_path keeps its single-node behavior. Operators must
        // explicitly opt INTO RaftSubmitter when enabling openraft.
        assert_eq!(HandlerWritePath::default(), HandlerWritePath::LocalSqlite);
    }

    #[test]
    fn production_defaults_pair_openraft_with_raft_submitter() {
        // production_defaults() pairs OpenRaft mode with RaftSubmitter
        // write path so the template is internally consistent.
        // Operator still needs to fill cluster_tls + peers before
        // validation passes.
        let d = RaftAssemblyConfig::production_defaults(
            YantrikNodeId::new(1),
            "https://10.0.0.1:7100".into(),
        );
        assert_eq!(d.write_path, HandlerWritePath::RaftSubmitter);
        assert!(d.peers.is_empty(), "operator MUST supply peers");
    }
}
