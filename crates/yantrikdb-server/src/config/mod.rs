use serde::Deserialize;
use std::path::{Path, PathBuf};

pub mod live_reload;
pub mod tenant_overrides;
pub mod versioned;
pub mod watch;

pub use live_reload::{ReloadError, ReloadOutcome, Reloadable};
pub use tenant_overrides::{
    InMemoryTenantConfigStore, OverrideValue, TenantConfigError, TenantConfigOverride,
    TenantConfigStore,
};
pub use versioned::{ConfigDelta, ConfigVersion, VersionedConfig};
pub use watch::{ConfigWatch, ConfigWatchSender};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ServerConfig {
    pub server: ServerSection,
    pub tls: TlsSection,
    pub encryption: EncryptionSection,
    pub embedding: EmbeddingSection,
    pub background: BackgroundSection,
    /// RFC 027 / v0.8.24: autonomous-hygiene cadence. The server drives the
    /// engine's `run_maintenance_cycle()` on a per-tenant schedule so closing
    /// loops (conflict burn-down, trigger prune, importance recalibration,
    /// entity backfill, auto-relate) is structural, not voluntary.
    pub maintenance: MaintenanceSection,
    pub limits: LimitsSection,
    pub cluster: ClusterSection,
    /// RFC 014-A: cluster-transport mTLS. Optional in legacy cluster
    /// mode; production gate for RFC 010 PR-4 openraft.
    pub cluster_tls: crate::security::ClusterTlsConfig,
    /// RFC 028: native replication. Active when
    /// `cluster.raft_mode = "yrp"`.
    pub yrp: YrpSection,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerSection {
    pub wire_port: u16,
    pub http_port: u16,
    pub data_dir: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingSection {
    pub strategy: EmbeddingStrategy,
    pub dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct TlsSection {
    pub cert_path: Option<PathBuf>,
    pub key_path: Option<PathBuf>,
}

impl TlsSection {
    pub fn is_enabled(&self) -> bool {
        self.cert_path.is_some() && self.key_path.is_some()
    }
}

// ── Encryption ─────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct EncryptionSection {
    /// Path to a 32-byte master key file (raw bytes).
    /// If unset and `auto_generate` is true, one is created on first startup.
    pub key_path: Option<PathBuf>,

    /// If true, generate a fresh key file at `key_path` (or `data_dir/master.key`)
    /// when none exists. Default true for ease of setup.
    pub auto_generate: bool,

    /// Master key value as hex string (64 chars). Takes precedence over key_path.
    /// Useful for env-driven config.
    pub key_hex: Option<String>,
}

impl EncryptionSection {
    /// Whether encryption is enabled (any key source configured).
    ///
    /// Not currently called — reserved for future startup banner /
    /// /v1/admin/status surfacing of at-rest encryption state.
    #[allow(dead_code)]
    pub fn is_enabled(&self) -> bool {
        self.key_path.is_some() || self.key_hex.is_some() || self.auto_generate
    }

    /// Resolve the master key from this configuration. Generates one if needed.
    pub fn resolve_key(&self, data_dir: &Path) -> anyhow::Result<Option<[u8; 32]>> {
        // Priority 0: env var override (issue #6 — env-friendly setup).
        // Documented as the env-equivalent of `[encryption] key_hex`. Takes
        // precedence over TOML so an operator can rotate the key without
        // editing the file.
        if let Ok(hex_str) = std::env::var("YANTRIKDB_ENCRYPTION_KEY_HEX") {
            let bytes = hex::decode(hex_str.trim())
                .map_err(|e| anyhow::anyhow!("invalid YANTRIKDB_ENCRYPTION_KEY_HEX: {}", e))?;
            if bytes.len() != 32 {
                anyhow::bail!(
                    "YANTRIKDB_ENCRYPTION_KEY_HEX must decode to exactly 32 bytes (got {})",
                    bytes.len()
                );
            }
            let mut key = [0u8; 32];
            key.copy_from_slice(&bytes);
            tracing::info!("encryption: enabled via YANTRIKDB_ENCRYPTION_KEY_HEX env var");
            return Ok(Some(key));
        }

        // Priority 1: explicit hex value in TOML
        if let Some(ref hex_str) = self.key_hex {
            let bytes = hex::decode(hex_str)
                .map_err(|e| anyhow::anyhow!("invalid encryption.key_hex: {}", e))?;
            if bytes.len() != 32 {
                anyhow::bail!("encryption.key_hex must decode to exactly 32 bytes");
            }
            let mut key = [0u8; 32];
            key.copy_from_slice(&bytes);
            return Ok(Some(key));
        }

        // Priority 2: explicit key file
        let path = match &self.key_path {
            Some(p) => p.clone(),
            None if self.auto_generate => data_dir.join("master.key"),
            None => return Ok(None),
        };

        if path.exists() {
            let bytes = std::fs::read(&path)?;
            if bytes.len() != 32 {
                anyhow::bail!(
                    "key file at {} must be exactly 32 bytes (got {})",
                    path.display(),
                    bytes.len()
                );
            }
            let mut key = [0u8; 32];
            key.copy_from_slice(&bytes);
            return Ok(Some(key));
        }

        // Auto-generate
        if self.auto_generate {
            use rand::RngCore;
            let mut key = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut key);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, key)?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
            }

            tracing::info!(
                path = %path.display(),
                "auto-generated encryption master key"
            );
            return Ok(Some(key));
        }

        Ok(None)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingStrategy {
    /// Use [`crate::embedder::FastEmbedder`] (fastembed + ONNX Runtime +
    /// `Qdrant/all-MiniLM-L6-v2-onnx` downloaded from HuggingFace on
    /// first run). Output dim = 384. Requires network at first start
    /// and an ONNX Runtime shared library at runtime.
    Builtin,
    /// Use [`yantrikdb::embedder::BundledEmbedder`] — the engine's
    /// `potion-base-2M` static embedder. Output dim = 64. Zero network,
    /// zero ONNX dependency, ~7 MB baked into the binary via
    /// `include_bytes!`. Designed for air-gapped Docker, edge deploys,
    /// and any context where the first-run HuggingFace fetch is
    /// undesirable. Quality is ~89% of MiniLM at the recall@5 cost.
    /// Default for the Docker image since `yantrikdb-server v0.8.16`
    /// (issue #35).
    Bundled,
    /// No server-side embedder; clients must compute embeddings and
    /// supply them on every `record` / `recall` call.
    ClientOnly,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BackgroundSection {
    pub consolidation_interval_minutes: u64,
    pub decay_sweep_interval_minutes: u64,
    /// RFC 010 PR-6.4: pause consolidation/conflict-scan/pattern-mining
    /// enrichment work when the engine reports `count_pending_ops()`
    /// above this threshold. `None` = auto-scale from engine.delta_max()
    /// at `(delta_max * 75 / 100).max(ENRICHMENT_PAUSE_THRESHOLD_FLOOR)`.
    ///
    /// Decay loop, snapshot scheduler, WAL checkpoint, and health probes
    /// are NOT paused — those are correctness-adjacent (memory aging is
    /// a function of wall-clock time, not engine load).
    #[serde(default)]
    pub enrichment_pause_threshold: Option<u64>,
}

/// RFC 027 / v0.8.24 — the sleep cycle. The server drives the engine's
/// `run_maintenance_cycle()` on a per-tenant timer so the close mechanisms
/// the engine already has (conflict burn-down, trigger prune, importance
/// recalibration, entity backfill, auto-relate) run structurally, with no
/// agent in the loop. Defaults are on + light: an existing deployment gains
/// hygiene on upgrade with no config change, and the heavy corpus-rewriting
/// passes stay opt-in.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MaintenanceSection {
    /// Master switch. Default `true` — hygiene is the point.
    pub enabled: bool,
    /// Cadence per tenant, in seconds. Default 600 (10 minutes).
    pub interval_secs: u64,
    /// Base delay before a tenant's first cycle, in seconds. Per-tenant jitter
    /// is added on top so N tenants don't stampede on startup. Default 120.
    pub initial_delay_secs: u64,
    /// Skip a tick if this node is catching up replication (openraft mode).
    /// Default `true` — don't pile maintenance mutations onto a node that is
    /// still applying the backlog.
    pub pause_during_replication_catchup: bool,
    /// Heavy pass: split oversized episodic dumps into atomic facts. Opt-in.
    pub run_split_oversized: bool,
    /// Heavy pass: repair leaked tool-call artifacts in the corpus. Opt-in.
    pub run_repair_artifacts: bool,
    /// Cap for the trigger prune (engine default 64).
    pub max_pending_triggers: usize,
    /// Cap on edges upserted per auto-relate pass (engine default 500).
    pub max_auto_relate_edges: usize,
    /// Minimum plaintext length for the split pass (engine default 1500).
    pub split_min_chars: usize,
}

impl Default for MaintenanceSection {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_secs: 600,
            initial_delay_secs: 120,
            pause_during_replication_catchup: true,
            run_split_oversized: false,
            run_repair_artifacts: false,
            max_pending_triggers: 64,
            max_auto_relate_edges: 500,
            split_min_chars: 1500,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LimitsSection {
    pub max_databases: usize,
    pub max_connections: usize,
}

// ── Cluster / Replication ──────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ClusterSection {
    /// Unique node identifier (used for HLC and Raft).
    /// 0 means single-node mode (no replication).
    pub node_id: u32,

    /// Role for this node in the cluster.
    pub role: NodeRole,

    /// Port for inter-peer cluster traffic (separate from client wire port).
    /// Defaults to 7440.
    pub cluster_port: u16,

    /// Address other peers should use to reach this node (host:cluster_port).
    /// If unset, derived from cluster_port + hostname.
    pub advertise_addr: Option<String>,

    /// List of peer nodes in the cluster.
    pub peers: Vec<PeerConfig>,

    /// Heartbeat interval in milliseconds (default 1000ms = 1s).
    pub heartbeat_interval_ms: u64,

    /// Election timeout in milliseconds (default 5000ms = 5s).
    /// If a follower doesn't hear from leader for this long, election starts.
    pub election_timeout_ms: u64,

    /// Shared cluster secret for authenticating peer connections.
    /// All nodes in a cluster must share the same secret.
    pub cluster_secret: Option<String>,

    /// Replication mode: async (default) or sync.
    pub replication_mode: ReplicationMode,

    /// RFC 010 PR-4: which Raft engine drives the cluster. Defaults to
    /// `Disabled` (single-node, no Raft). `OpenRaft` activates the
    /// production-grade openraft engine — and triggers the mTLS
    /// production gate at server startup (see
    /// [`crate::raft::build_raft_cluster`]).
    pub raft_mode: crate::raft::RaftClusterMode,
}

/// RFC 028 `[yrp]` section — native-replication knobs. Only read when
/// `cluster.raft_mode = "yrp"`. Node identity comes from
/// `cluster.node_id`; `peers` lists EVERY cluster member including this
/// node (self is matched by node_id).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YrpSection {
    /// Immutable cluster identity (RFC 028 §3). Must be identical and
    /// non-zero on every member; alien state quarantines at boot.
    pub cluster_id: u64,
    /// Driver tick period. All election/heartbeat timing is counted in
    /// these ticks.
    pub tick_ms: u64,
    /// Randomized election timeout range, in ticks.
    pub election_ticks_min: u32,
    pub election_ticks_max: u32,
    /// Leader heartbeat cadence, in ticks.
    pub heartbeat_ticks: u32,
    /// Log compaction: compact once the durably-applied span exceeds
    /// this many entries. 0 = disabled — the production default until
    /// Phase C ships engine-checkpoint transfer for beyond-GC stragglers.
    pub compact_after_entries: u64,
    /// Entries a LEADER retains above its compaction base so transient
    /// follower lag is served from the log, not a snapshot.
    pub leader_retain_entries: u64,
    /// All cluster members (including this node).
    pub peers: Vec<YrpPeerConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct YrpPeerConfig {
    pub node_id: u64,
    /// HTTP base url peers reach this member at (e.g. "http://10.0.0.2:7438").
    /// YRP wire messages ride the HTTP plane (`POST /v1/yrp/msg`).
    pub addr: String,
    /// Witness: votes in elections, never stores data, never counts
    /// toward commit durability (RFC 028 §4). At most one per cluster.
    #[serde(default)]
    pub witness: bool,
}

impl Default for YrpSection {
    fn default() -> Self {
        Self {
            cluster_id: 0,
            tick_ms: 50,
            election_ticks_min: 10,
            election_ticks_max: 20,
            heartbeat_ticks: 2,
            compact_after_entries: 0,
            leader_retain_entries: 512,
            peers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeRole {
    /// Standalone node, no replication. Default.
    Single,
    /// Full data node that can become primary or secondary via election.
    Voter,
    /// Read-only replica that consumes oplog but never votes or accepts writes.
    ReadReplica,
    /// Witness — vote-only node, no data storage. Tiebreaker for 2-node clusters.
    Witness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicationMode {
    /// Writes return immediately, replicas catch up asynchronously.
    Async,
    /// Writes block until quorum of secondaries ack.
    Sync,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PeerConfig {
    /// Peer's wire protocol address (host:port).
    pub addr: String,
    /// Peer's role in the cluster.
    pub role: NodeRole,
}

impl Default for ClusterSection {
    fn default() -> Self {
        Self {
            node_id: 0,
            role: NodeRole::Single,
            cluster_port: 7440,
            advertise_addr: None,
            peers: Vec::new(),
            heartbeat_interval_ms: 1000,
            election_timeout_ms: 5000,
            cluster_secret: None,
            replication_mode: ReplicationMode::Async,
            raft_mode: crate::raft::RaftClusterMode::Disabled,
        }
    }
}

impl ClusterSection {
    /// Whether replication is enabled (i.e. not single-node mode).
    pub fn is_clustered(&self) -> bool {
        self.role != NodeRole::Single
    }

    /// Total voter count (this node + voter peers, excluding witness/read replicas).
    ///
    /// Not currently called — reserved for quorum diagnostics and config
    /// validation on startup.
    #[allow(dead_code)]
    pub fn voter_count(&self) -> usize {
        let self_voter = matches!(self.role, NodeRole::Voter) as usize;
        let peer_voters = self
            .peers
            .iter()
            .filter(|p| p.role == NodeRole::Voter)
            .count();
        self_voter + peer_voters
    }

    /// Total quorum members (voters + witnesses) for elections.
    pub fn quorum_members(&self) -> usize {
        let self_member = matches!(self.role, NodeRole::Voter | NodeRole::Witness) as usize;
        let peer_members = self
            .peers
            .iter()
            .filter(|p| matches!(p.role, NodeRole::Voter | NodeRole::Witness))
            .count();
        self_member + peer_members
    }

    /// Quorum size needed for elections (N/2 + 1).
    pub fn quorum_size(&self) -> usize {
        let total = self.quorum_members();
        total / 2 + 1
    }
}

impl Default for ServerSection {
    fn default() -> Self {
        Self {
            wire_port: 7437,
            http_port: 7438,
            data_dir: PathBuf::from("./data"),
        }
    }
}

impl Default for EmbeddingSection {
    fn default() -> Self {
        Self {
            strategy: EmbeddingStrategy::Builtin,
            dim: 384,
        }
    }
}

impl Default for BackgroundSection {
    fn default() -> Self {
        Self {
            consolidation_interval_minutes: 30,
            decay_sweep_interval_minutes: 60,
            enrichment_pause_threshold: None, // auto-scale from engine.delta_max()
        }
    }
}

impl Default for LimitsSection {
    fn default() -> Self {
        Self {
            max_databases: 100,
            max_connections: 1000,
        }
    }
}

impl ServerConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: ServerConfig = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn data_dir(&self) -> &Path {
        &self.server.data_dir
    }

    pub fn control_db_path(&self) -> PathBuf {
        self.server.data_dir.join("control.db")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #6 regression: YANTRIKDB_ENCRYPTION_KEY_HEX env var must be
    /// honored as the env-equivalent of `[encryption] key_hex` in TOML.
    /// Pre-fix: env var was silently ignored, encryption disabled.
    ///
    /// All env-var assertions consolidated into ONE test because env vars
    /// are process-global and Rust runs tests in parallel by default.
    /// Splitting these into multiple #[test]s race on the shared var.
    #[test]
    fn encryption_env_var_handling() {
        const VAR: &str = "YANTRIKDB_ENCRYPTION_KEY_HEX";
        let cfg_default = EncryptionSection {
            key_path: None,
            auto_generate: false,
            key_hex: None,
        };

        // Case 1: 64 hex chars (32 bytes) — env var produces the right key.
        let key_hex = "f".repeat(64);
        std::env::set_var(VAR, &key_hex);
        let resolved = cfg_default.resolve_key(Path::new("/tmp")).unwrap();
        assert_eq!(resolved, Some([0xffu8; 32]), "env var should produce key");

        // Case 2: invalid hex — error mentions the env var name.
        std::env::set_var(VAR, "not-hex-at-all");
        let err = cfg_default.resolve_key(Path::new("/tmp")).unwrap_err();
        assert!(err.to_string().contains(VAR));

        // Case 3: valid hex but wrong byte length — error mentions length.
        std::env::set_var(VAR, "ab"); // 1 byte
        let err = cfg_default.resolve_key(Path::new("/tmp")).unwrap_err();
        assert!(err.to_string().contains("32 bytes"));

        // Case 4: env var takes precedence over TOML — set both, env wins.
        std::env::set_var(VAR, &key_hex);
        let cfg_with_toml = EncryptionSection {
            key_path: None,
            auto_generate: false,
            key_hex: Some("0".repeat(64)), // would resolve to all-zeros
        };
        let resolved = cfg_with_toml.resolve_key(Path::new("/tmp")).unwrap();
        assert_eq!(
            resolved,
            Some([0xffu8; 32]),
            "env var must beat TOML key_hex"
        );

        std::env::remove_var(VAR);
    }

    /// v0.8.16 issue #35: the `bundled` strategy must parse from TOML so
    /// the docker default config (`docker/yantrikdb.toml`) works on the
    /// no-network startup path. Also pins that `builtin` stays the
    /// default when no `[embedding]` section is present — backwards
    /// compat for existing single-binary deployments that already have
    /// dim=384 MiniLM embeddings on disk.
    #[test]
    fn embedding_strategy_variants_parse() {
        // `builtin` (existing, default).
        let toml = r#"[embedding]
strategy = "builtin"
dim = 384
"#;
        let cfg: EmbeddingSection = toml::from_str(toml)
            .and_then(|v: toml::Value| v["embedding"].clone().try_into())
            .unwrap();
        assert!(matches!(cfg.strategy, EmbeddingStrategy::Builtin));
        assert_eq!(cfg.dim, 384);

        // `bundled` (new in v0.8.16) — pins serde `rename_all = "snake_case"`.
        let toml = r#"[embedding]
strategy = "bundled"
dim = 64
"#;
        let cfg: EmbeddingSection = toml::from_str(toml)
            .and_then(|v: toml::Value| v["embedding"].clone().try_into())
            .unwrap();
        assert!(matches!(cfg.strategy, EmbeddingStrategy::Bundled));
        assert_eq!(cfg.dim, 64);

        // `client_only` (existing).
        let toml = r#"[embedding]
strategy = "client_only"
dim = 384
"#;
        let cfg: EmbeddingSection = toml::from_str(toml)
            .and_then(|v: toml::Value| v["embedding"].clone().try_into())
            .unwrap();
        assert!(matches!(cfg.strategy, EmbeddingStrategy::ClientOnly));

        // Default = builtin + dim=384 (backwards compat for configs with
        // no [embedding] section).
        let default = EmbeddingSection::default();
        assert!(matches!(default.strategy, EmbeddingStrategy::Builtin));
        assert_eq!(default.dim, 384);
    }
}
