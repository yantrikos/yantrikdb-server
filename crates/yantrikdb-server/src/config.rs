use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub server: ServerSection,
    pub tls: TlsSection,
    pub encryption: EncryptionSection,
    pub embedding: EmbeddingSection,
    pub background: BackgroundSection,
    pub limits: LimitsSection,
    pub cluster: ClusterSection,
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
pub struct TlsSection {
    pub cert_path: Option<PathBuf>,
    pub key_path: Option<PathBuf>,
}

impl Default for TlsSection {
    fn default() -> Self {
        Self {
            cert_path: None,
            key_path: None,
        }
    }
}

impl TlsSection {
    pub fn is_enabled(&self) -> bool {
        self.cert_path.is_some() && self.key_path.is_some()
    }
}

// ── Encryption ─────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
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

impl Default for EncryptionSection {
    fn default() -> Self {
        Self {
            key_path: None,
            auto_generate: true,
            key_hex: None,
        }
    }
}

impl EncryptionSection {
    /// Whether encryption is enabled (any key source configured).
    pub fn is_enabled(&self) -> bool {
        self.key_path.is_some() || self.key_hex.is_some() || self.auto_generate
    }

    /// Resolve the master key from this configuration. Generates one if needed.
    pub fn resolve_key(&self, data_dir: &Path) -> anyhow::Result<Option<[u8; 32]>> {
        // Priority 1: explicit hex value
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
    Builtin,
    ClientOnly,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BackgroundSection {
    pub consolidation_interval_minutes: u64,
    pub decay_sweep_interval_minutes: u64,
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
        }
    }
}

impl ClusterSection {
    /// Whether replication is enabled (i.e. not single-node mode).
    pub fn is_clustered(&self) -> bool {
        self.role != NodeRole::Single
    }

    /// Total voter count (this node + voter peers, excluding witness/read replicas).
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

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server: ServerSection::default(),
            tls: TlsSection::default(),
            encryption: EncryptionSection::default(),
            embedding: EmbeddingSection::default(),
            background: BackgroundSection::default(),
            limits: LimitsSection::default(),
            cluster: ClusterSection::default(),
        }
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
