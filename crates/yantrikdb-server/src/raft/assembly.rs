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
            cluster_tls: None, // operator MUST supply
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
        return Err(AssemblyError::MtlsRequired {
            missing: "ca_path",
        });
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

    let identity = reqwest::Identity::from_pem(&bundle).map_err(|e| {
        AssemblyError::ReqwestBuild(format!("Identity::from_pem: {e}"))
    })?;
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

    builder
        .build()
        .map_err(|e| AssemblyError::ReqwestBuild(format!("build: {e}")))
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
    let cluster_tls = validate_cluster_tls_for_openraft(cfg.cluster_tls.as_ref())?;
    let client = build_reqwest_client_for_cluster(cluster_tls, cfg.request_timeout)?;
    let network_factory = HttpRaftNetworkFactory::new(client, cfg.request_timeout);

    let validated_config = Arc::new(cfg.openraft_config.validate().map_err(|e| {
        AssemblyError::RaftNew(format!("openraft Config::validate: {e}"))
    })?);

    let state_machine = YantrikStateMachine::new(local.clone());
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
pub async fn initialize_single_node(assembly: &RaftAssembly, node_addr: String) -> Result<(), openraft::error::RaftError<YantrikNodeId, openraft::error::InitializeError<YantrikNodeId, YantrikNode>>> {
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

    fn tls_with(
        cert: Option<&str>,
        key: Option<&str>,
        ca: Option<&str>,
    ) -> ClusterTlsConfig {
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
}
