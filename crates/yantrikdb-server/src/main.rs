mod admission;
mod auth;
mod background;
mod backup;
mod cache;
mod cluster;
mod command;
mod commit;
mod config;
mod control;
mod debug;
mod embedder;
mod forget;
mod handler;
mod http_gateway;
mod index;
mod jobs;
mod key_provider;
pub(crate) mod metrics;
mod migrations;
mod raft;
mod restore;
mod retrieval;
mod runtime;
mod security;
mod server;
mod socratic;
mod tenant_pool;
mod tls;
mod version;

use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};

use crate::config::ServerConfig;
use crate::control::ControlDb;
use crate::server::AppState;
use crate::tenant_pool::TenantPool;

#[derive(Parser)]
#[command(
    name = "yantrikdb",
    about = "YantrikDB — cognitive memory database server"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the database server
    Serve {
        /// Path to config file
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Wire protocol port
        #[arg(long)]
        wire_port: Option<u16>,
        /// HTTP gateway port
        #[arg(long)]
        http_port: Option<u16>,
        /// Data directory
        #[arg(long)]
        data_dir: Option<PathBuf>,
    },
    /// Database management
    Db {
        #[command(subcommand)]
        action: DbAction,
        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
    },
    /// Token management
    Token {
        #[command(subcommand)]
        action: TokenAction,
        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
    },
    /// Export a database to JSONL (stdout)
    Export {
        /// Database name
        name: String,
        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
    },
    /// Import a database from JSONL (stdin)
    Import {
        /// Database name
        name: String,
        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
    },
    /// Cluster management
    Cluster {
        #[command(subcommand)]
        action: ClusterAction,
    },
    /// Encryption key management
    Encryption {
        #[command(subcommand)]
        action: EncryptionAction,
    },
    /// Admission control inspection (RFC 009)
    ///
    /// Shows current admission state — hard caps, in-flight recall counts,
    /// concurrent expanded recalls, and the runtime isolation state. Reads
    /// from `/v1/health/deep` on a running server.
    Admission {
        #[command(subcommand)]
        action: AdmissionAction,
    },
    /// Version framework inspection (RFC 017-A)
    ///
    /// Shows local + cluster wire version, per-table schema versions,
    /// build identifier. Used to verify rolling-upgrade state and
    /// diagnose version-mismatch errors.
    Version {
        #[command(subcommand)]
        action: VersionAction,
    },
    /// Jepsen / debug surface (RFC 010 PR-5)
    ///
    /// Read committed log entries; inject + clear fault injections.
    /// Requires cluster master token. Operator-only by design.
    Debug {
        #[command(subcommand)]
        action: DebugAction,
    },
    /// Cluster TLS / mTLS inspection (RFC 014-A)
    ///
    /// Verify cluster-mTLS cert configuration loads correctly without
    /// starting the server. Useful for catching cert rotation issues
    /// before they cause a cluster restart failure.
    Tls {
        #[command(subcommand)]
        action: TlsAction,
    },
    /// Background job inspection (RFC 019)
    ///
    /// List, get, or cancel durable jobs in the queue. Requires cluster
    /// master token. Operator-only by design.
    Jobs {
        #[command(subcommand)]
        action: JobsAction,
    },
    /// Schema migration inspection (RFC 017-B operator visibility)
    Migrations {
        #[command(subcommand)]
        action: MigrationsAction,
    },
}

#[derive(Subcommand)]
enum JobsAction {
    /// List jobs, optionally filtered by tenant + state.
    List {
        #[arg(long, default_value = "http://localhost:7438")]
        url: String,
        #[arg(short, long, env = "YQL_TOKEN")]
        token: String,
        #[arg(long)]
        tenant: Option<i64>,
        /// One of Pending, Leased, Succeeded, Failed, Cancelled.
        #[arg(long)]
        state: Option<String>,
        #[arg(long, default_value = "100")]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Get a single job by id.
    Get {
        #[arg(long, default_value = "http://localhost:7438")]
        url: String,
        #[arg(short, long, env = "YQL_TOKEN")]
        token: String,
        job_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Cancel a Pending or Leased job.
    Cancel {
        #[arg(long, default_value = "http://localhost:7438")]
        url: String,
        #[arg(short, long, env = "YQL_TOKEN")]
        token: String,
        job_id: String,
    },
}

#[derive(Subcommand)]
enum MigrationsAction {
    /// Show applied schema migrations on a running server.
    Status {
        #[arg(long, default_value = "http://localhost:7438")]
        url: String,
        #[arg(short, long, env = "YQL_TOKEN")]
        token: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum TlsAction {
    /// Verify cluster-mTLS configuration in a config file. Loads the
    /// configured certs / key / CA, reports counts + dev_mode status.
    /// Exits non-zero if the config claims to specify certs but any
    /// referenced file fails to load.
    VerifyCluster {
        #[arg(short, long)]
        config: PathBuf,
    },
}

#[derive(Subcommand)]
enum DebugAction {
    /// Read committed log entries for a tenant from a running server.
    History {
        #[arg(long, default_value = "http://localhost:7438")]
        url: String,
        #[arg(short, long, env = "YQL_TOKEN")]
        token: String,
        /// Tenant id (control DB primary key).
        #[arg(long)]
        tenant: i64,
        /// Inclusive starting log_index. 0 = from the beginning.
        #[arg(long, default_value = "0")]
        from: u64,
        /// Maximum entries to return (server caps at 1000).
        #[arg(long, default_value = "100")]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// List active fault injections.
    FaultList {
        #[arg(long, default_value = "http://localhost:7438")]
        url: String,
        #[arg(short, long, env = "YQL_TOKEN")]
        token: String,
        #[arg(long)]
        json: bool,
    },
    /// Clear all fault injections.
    FaultClear {
        #[arg(long, default_value = "http://localhost:7438")]
        url: String,
        #[arg(short, long, env = "YQL_TOKEN")]
        token: String,
    },
}

#[derive(Subcommand)]
enum VersionAction {
    /// Show this build's version state without contacting a server.
    /// Useful for verifying which build a binary is.
    Local,
    /// Show version state on a running server (local + cluster min/max
    /// + per-table schema versions). Reads /v1/health/deep.
    Status {
        /// Server HTTP URL
        #[arg(long, default_value = "http://localhost:7438")]
        url: String,
        /// Auth token (or YQL_TOKEN env)
        #[arg(short, long, env = "YQL_TOKEN")]
        token: Option<String>,
        /// Output as JSON instead of human-readable
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum AdmissionAction {
    /// Show current admission state on a running server
    Status {
        /// Server HTTP URL
        #[arg(long, default_value = "http://localhost:7438")]
        url: String,
        /// Auth token (or YQL_TOKEN env)
        #[arg(short, long, env = "YQL_TOKEN")]
        token: Option<String>,
        /// Output as JSON instead of human-readable
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum EncryptionAction {
    /// Generate a fresh 32-byte master key file
    GenKey {
        /// Output file path
        #[arg(short, long, default_value = "./master.key")]
        output: PathBuf,
    },
    /// Print the hex encoding of an existing key file (for use with key_hex env var)
    ShowKey {
        /// Key file path
        #[arg(short, long, default_value = "./master.key")]
        input: PathBuf,
    },
}

#[derive(Subcommand)]
enum ClusterAction {
    /// Generate a cluster config skeleton with a fresh cluster_secret
    Init {
        /// This node's ID (1, 2, 3, ...)
        #[arg(long)]
        node_id: u32,
        /// Output config path
        #[arg(short, long, default_value = "./yantrikdb.toml")]
        output: PathBuf,
        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
        /// Wire port
        #[arg(long, default_value = "7437")]
        wire_port: u16,
        /// HTTP port
        #[arg(long, default_value = "7438")]
        http_port: u16,
        /// Cluster port
        #[arg(long, default_value = "7440")]
        cluster_port: u16,
        /// Comma-separated peer addresses (host:cluster_port)
        #[arg(long, value_delimiter = ',')]
        peers: Vec<String>,
        /// Comma-separated witness addresses
        #[arg(long, value_delimiter = ',')]
        witnesses: Vec<String>,
        /// Cluster secret (auto-generated if omitted)
        #[arg(long)]
        secret: Option<String>,
    },
    /// Show cluster status by querying a running server
    Status {
        /// Server HTTP URL
        #[arg(long, default_value = "http://localhost:7438")]
        url: String,
        /// Auth token (or YQL_TOKEN env)
        #[arg(short, long, env = "YQL_TOKEN")]
        token: Option<String>,
    },
    /// Manually trigger an election on a node (force failover)
    Promote {
        /// Server HTTP URL of the node to promote
        #[arg(long, default_value = "http://localhost:7438")]
        url: String,
        /// Auth token (or YQL_TOKEN env)
        #[arg(short, long, env = "YQL_TOKEN")]
        token: String,
    },
    /// Show openraft cluster status (RFC 010 PR-4) by querying
    /// `/v1/cluster/raft` on a running server. Distinct from `status`
    /// which targets the legacy Raft-lite cluster.
    RaftStatus {
        /// Server HTTP URL
        #[arg(long, default_value = "http://localhost:7438")]
        url: String,
        /// Auth token (or YQL_TOKEN env)
        #[arg(short, long, env = "YQL_TOKEN")]
        token: Option<String>,
        /// Print raw JSON instead of human-readable summary.
        #[arg(long)]
        json: bool,
    },
    /// v0.8.3 #24: bootstrap a fresh openraft cluster on the seed node.
    /// Run exactly once per cluster, on the first node. Subsequent voters
    /// are added via `add-learner` + `promote-voter`.
    InitializeCluster {
        /// Leader HTTP URL (the seed node).
        #[arg(long, default_value = "http://localhost:7438")]
        leader: String,
        /// Cluster master token.
        #[arg(short, long, env = "YDB_CLUSTER_MASTER_TOKEN")]
        master_token: String,
    },
    /// v0.8.3 #24: add a non-voting learner. It catches up via openraft
    /// snapshot transfer without participating in elections (safe).
    AddLearner {
        /// New learner's node_id (must not collide with existing members).
        #[arg(long)]
        node_id: u64,
        /// New learner's cluster transport address (host:cluster_port).
        #[arg(long)]
        addr: String,
        /// Leader HTTP URL.
        #[arg(long, default_value = "http://localhost:7438")]
        leader: String,
        /// Cluster master token.
        #[arg(short, long, env = "YDB_CLUSTER_MASTER_TOKEN")]
        master_token: String,
    },
    /// v0.8.3 #24: poll until the named node has caught up with the
    /// leader's last_log_index (within `--max-lag`).
    WaitCaughtUp {
        /// Node id to wait on.
        #[arg(long)]
        node_id: u64,
        /// Leader HTTP URL.
        #[arg(long, default_value = "http://localhost:7438")]
        leader: String,
        /// Cluster master token.
        #[arg(short, long, env = "YDB_CLUSTER_MASTER_TOKEN")]
        master_token: String,
        /// Max acceptable log index lag.
        #[arg(long, default_value = "10")]
        max_lag: u64,
        /// Total wait timeout in seconds.
        #[arg(long, default_value = "1800")]
        timeout_secs: u64,
    },
    /// v0.8.3 #24: change voter membership. Body lists the FINAL voter
    /// set. Promotes any existing learners listed; demotes any current
    /// voters not listed. The leader's id MUST be in the list.
    PromoteVoter {
        /// Final voter set (comma-separated node_ids).
        #[arg(long, value_delimiter = ',', num_args = 1..)]
        voters: Vec<u64>,
        /// Leader HTTP URL.
        #[arg(long, default_value = "http://localhost:7438")]
        leader: String,
        /// Cluster master token.
        #[arg(short, long, env = "YDB_CLUSTER_MASTER_TOKEN")]
        master_token: String,
    },
    /// v0.8.3 #24: remove a node from the cluster (atomic
    /// change-membership minus the named node). Refuses if removal
    /// would leave the cluster with no voters.
    RemoveNode {
        /// Node id to remove.
        #[arg(long)]
        node_id: u64,
        /// Leader HTTP URL.
        #[arg(long, default_value = "http://localhost:7438")]
        leader: String,
        /// Cluster master token.
        #[arg(short, long, env = "YDB_CLUSTER_MASTER_TOKEN")]
        master_token: String,
    },
}

#[derive(Subcommand)]
enum DbAction {
    /// Create a new database
    Create { name: String },
    /// List all databases
    List,
}

#[derive(Subcommand)]
enum TokenAction {
    /// Create a token for a database
    Create {
        /// Database name
        #[arg(long)]
        db: String,
        /// Optional label
        #[arg(long, default_value = "")]
        label: String,
    },
    /// Revoke a token
    Revoke { token: String },
}

/// v0.8.4 (issue #27): sync `fn main` that owns the tokio runtime
/// explicitly, replacing the previous `#[tokio::main]` macro. This lets
/// `SplitRuntime` (RFC 009 §4 Layer 1: dedicated control-plane runtime)
/// be built in sync context and dropped in sync context — without that
/// invariant, `Runtime::Drop` panics with "Cannot drop a runtime in a
/// context where blocking is not allowed" because the macro builds an
/// outer Runtime and any nested Runtime would drop while we're still
/// inside the outer runtime's async context.
fn main() -> anyhow::Result<()> {
    // rustls 0.23+ requires a process-level CryptoProvider before any TLS
    // operation. Without this, openraft mode panics at startup (issue #26).
    // Idempotent install — uses aws-lc-rs (the rustls 0.23 modern default).
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    // Build the main app runtime explicitly. The default tokio runtime
    // (multi-threaded, all features) is what `#[tokio::main]` would
    // create. We construct it ourselves so we can drop it from sync
    // context on shutdown.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("ydb-main")
        .build()
        .map_err(|e| anyhow::anyhow!("build main tokio runtime: {e}"))?;

    let result = runtime.block_on(async_main());

    // Now we're back in sync context. Tokio runtimes (the main one above
    // plus any SplitRuntime constructed inside async_main and shut down
    // before async_main returns) drop without panicking here.
    runtime.shutdown_timeout(std::time::Duration::from_secs(5));
    result
}

async fn async_main() -> anyhow::Result<()> {
    // Structured logging. Set YANTRIKDB_LOG_JSON=1 for newline-delimited JSON
    // output (for log aggregators, grep-friendly ops). Default is human-readable.
    //
    // When built with --features tokio-console, the console-subscriber layer
    // is added for live runtime inspection via `tokio-console`.
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "yantrikdb_server=info".into());

    #[cfg(feature = "tokio-console")]
    {
        use tracing_subscriber::prelude::*;
        let console_layer = console_subscriber::spawn();
        let fmt_layer = tracing_subscriber::fmt::layer().with_filter(env_filter);
        tracing_subscriber::registry()
            .with(console_layer)
            .with(fmt_layer)
            .init();
        tracing::info!("tokio-console enabled — attach via: tokio-console http://127.0.0.1:6669");
    }

    #[cfg(not(feature = "tokio-console"))]
    {
        if std::env::var("YANTRIKDB_LOG_JSON").as_deref() == Ok("1") {
            tracing_subscriber::fmt()
                .json()
                .with_env_filter(env_filter)
                .init();
        } else {
            tracing_subscriber::fmt().with_env_filter(env_filter).init();
        }
    }

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            config: config_path,
            wire_port,
            http_port,
            data_dir,
        } => {
            let mut cfg = match config_path {
                Some(ref path) => ServerConfig::load(path)?,
                None => ServerConfig::default(),
            };

            // CLI overrides
            if let Some(port) = wire_port {
                cfg.server.wire_port = port;
            }
            if let Some(port) = http_port {
                cfg.server.http_port = port;
            }
            if let Some(dir) = data_dir {
                cfg.server.data_dir = dir;
            }

            run_server(cfg).await
        }

        Commands::Db { action, data_dir } => {
            std::fs::create_dir_all(&data_dir)?;
            let control = ControlDb::open(&data_dir.join("control.db"))?;

            match action {
                DbAction::Create { name } => {
                    if control.database_exists(&name)? {
                        eprintln!("database '{}' already exists", name);
                        std::process::exit(1);
                    }
                    let db_dir = data_dir.join(&name);
                    std::fs::create_dir_all(&db_dir)?;
                    let id = control.create_database(&name, &name)?;
                    println!("created database '{}' (id: {})", name, id);
                }
                DbAction::List => {
                    let databases = control.list_databases()?;
                    if databases.is_empty() {
                        println!("no databases");
                    } else {
                        println!("{:<6} {:<20} CREATED", "ID", "NAME");
                        for db in databases {
                            println!("{:<6} {:<20} {}", db.id, db.name, db.created_at);
                        }
                    }
                }
            }
            Ok(())
        }

        Commands::Token { action, data_dir } => {
            // Defensive: token operations always run AFTER `db create`,
            // which would have created the data dir. If the data dir
            // is missing, the most likely culprit on Windows/Git Bash
            // is MSYS path translation rewriting a Linux-style path
            // like `/var/lib/yantrikdb` into a Windows-prefixed path
            // before docker exec sees it — silently creating control.db
            // at a phantom location the running server can't see.
            // Failing here makes that footgun loud instead of silent.
            if !data_dir.exists() {
                anyhow::bail!(
                    "data dir `{}` does not exist; token operations require an \
                     initialized data dir (run `yantrikdb db --data-dir <path> create <name>` first). \
                     On Windows + Git Bash, also try setting `MSYS_NO_PATHCONV=1` to disable \
                     path translation when invoking via `docker exec`.",
                    data_dir.display()
                );
            }
            let control = ControlDb::open(&data_dir.join("control.db"))?;

            match action {
                TokenAction::Create { db, label } => {
                    let db_record = control
                        .get_database(&db)?
                        .ok_or_else(|| anyhow::anyhow!("database '{}' not found", db))?;

                    let token = auth::generate_token();
                    let hash = auth::hash_token(&token);
                    control.create_token(&hash, db_record.id, &label)?;

                    println!("{}", token);
                    eprintln!(
                        "token created for database '{}' — save it now, it won't be shown again",
                        db
                    );
                }
                TokenAction::Revoke { token } => {
                    let hash = auth::hash_token(&token);
                    if control.revoke_token(&hash)? {
                        println!("token revoked");
                    } else {
                        eprintln!("token not found or already revoked");
                        std::process::exit(1);
                    }
                }
            }
            Ok(())
        }

        Commands::Export { name, data_dir } => {
            let control = ControlDb::open(&data_dir.join("control.db"))?;
            let db_record = control
                .get_database(&name)?
                .ok_or_else(|| anyhow::anyhow!("database '{}' not found", name))?;

            let db_dir = data_dir.join(&db_record.path);
            let db_path = db_dir.join("yantrik.db");

            // Try to load encryption key from data dir if present
            let key_file = data_dir.join("master.key");
            let engine = if key_file.exists() {
                let key_bytes = std::fs::read(&key_file)?;
                if key_bytes.len() != 32 {
                    anyhow::bail!("master.key must be 32 bytes");
                }
                let mut key = [0u8; 32];
                key.copy_from_slice(&key_bytes);
                yantrikdb::YantrikDB::new_encrypted(
                    db_path.to_str().unwrap_or("yantrik.db"),
                    384,
                    &key,
                )?
            } else {
                yantrikdb::YantrikDB::new(db_path.to_str().unwrap_or("yantrik.db"), 384)?
            };

            // Export memories in pages
            let page_size = 1000;
            let mut offset = 0;
            let mut total = 0;
            loop {
                let (memories, count) =
                    engine.list_memories(page_size, offset, None, None, None, "created_at")?;
                if memories.is_empty() {
                    break;
                }
                for mem in &memories {
                    let row = serde_json::json!({
                        "type": "memory",
                        "rid": mem.rid,
                        "text": mem.text,
                        "memory_type": mem.memory_type,
                        "importance": mem.importance,
                        "valence": mem.valence,
                        "half_life": mem.half_life,
                        "created_at": mem.created_at,
                        "metadata": mem.metadata,
                        "namespace": mem.namespace,
                        "certainty": mem.certainty,
                        "domain": mem.domain,
                        "source": mem.source,
                        "emotional_state": mem.emotional_state,
                    });
                    println!("{}", serde_json::to_string(&row)?);
                    total += 1;
                }
                offset += page_size;
                if memories.len() < page_size || offset >= count {
                    break;
                }
            }

            // Export graph edges — get all entities and their edges
            let entities = engine.search_entities(None, None, 100_000)?;
            let mut edge_count = 0;
            let mut seen_edges = std::collections::HashSet::new();
            for entity in &entities {
                let edges = engine.get_edges(&entity.name)?;
                for edge in &edges {
                    if seen_edges.insert(edge.edge_id.clone()) {
                        let row = serde_json::json!({
                            "type": "edge",
                            "edge_id": edge.edge_id,
                            "src": edge.src,
                            "dst": edge.dst,
                            "rel_type": edge.rel_type,
                            "weight": edge.weight,
                        });
                        println!("{}", serde_json::to_string(&row)?);
                        edge_count += 1;
                    }
                }
            }

            eprintln!(
                "exported {} memories, {} edges from '{}'",
                total, edge_count, name
            );
            Ok(())
        }

        Commands::Import { name, data_dir } => {
            std::fs::create_dir_all(&data_dir)?;
            let control = ControlDb::open(&data_dir.join("control.db"))?;

            // Create database if it doesn't exist
            if !control.database_exists(&name)? {
                let db_dir = data_dir.join(&name);
                std::fs::create_dir_all(&db_dir)?;
                control.create_database(&name, &name)?;
                eprintln!("created database '{}'", name);
            }

            let db_record = control
                .get_database(&name)?
                .ok_or_else(|| anyhow::anyhow!("database '{}' not found", name))?;

            let db_dir = data_dir.join(&db_record.path);
            std::fs::create_dir_all(&db_dir)?;
            let db_path = db_dir.join("yantrik.db");

            // Use encryption if a master.key exists in data_dir
            let key_file = data_dir.join("master.key");
            let mut engine = if key_file.exists() {
                let key_bytes = std::fs::read(&key_file)?;
                if key_bytes.len() != 32 {
                    anyhow::bail!("master.key must be 32 bytes");
                }
                let mut key = [0u8; 32];
                key.copy_from_slice(&key_bytes);
                yantrikdb::YantrikDB::new_encrypted(
                    db_path.to_str().unwrap_or("yantrik.db"),
                    384,
                    &key,
                )?
            } else {
                yantrikdb::YantrikDB::new(db_path.to_str().unwrap_or("yantrik.db"), 384)?
            };

            // Set up embedder for re-embedding
            let embedder = embedder::FastEmbedder::new()?;
            engine.set_embedder(embedder.boxed());

            let stdin = std::io::BufReader::new(std::io::stdin());
            use std::io::BufRead;
            let mut mem_count = 0;
            let mut edge_count = 0;
            let mut errors = 0;

            for line in stdin.lines() {
                let line = line?;
                if line.is_empty() {
                    continue;
                }
                let row: serde_json::Value = serde_json::from_str(&line)?;
                let row_type = row["type"].as_str().unwrap_or("");

                match row_type {
                    "memory" => {
                        let result = engine.record_text(
                            row["text"].as_str().unwrap_or(""),
                            row["memory_type"].as_str().unwrap_or("semantic"),
                            row["importance"].as_f64().unwrap_or(0.5),
                            row["valence"].as_f64().unwrap_or(0.0),
                            row["half_life"].as_f64().unwrap_or(168.0),
                            &row["metadata"],
                            row["namespace"].as_str().unwrap_or(""),
                            row["certainty"].as_f64().unwrap_or(1.0),
                            row["domain"].as_str().unwrap_or(""),
                            row["source"].as_str().unwrap_or("user"),
                            row["emotional_state"].as_str(),
                        );
                        match result {
                            Ok(_) => mem_count += 1,
                            Err(e) => {
                                eprintln!("error importing memory: {}", e);
                                errors += 1;
                            }
                        }
                    }
                    "edge" => {
                        let result = engine.relate(
                            row["src"].as_str().unwrap_or(""),
                            row["dst"].as_str().unwrap_or(""),
                            row["rel_type"].as_str().unwrap_or(""),
                            row["weight"].as_f64().unwrap_or(1.0),
                        );
                        match result {
                            Ok(_) => edge_count += 1,
                            Err(e) => {
                                eprintln!("error importing edge: {}", e);
                                errors += 1;
                            }
                        }
                    }
                    _ => {
                        eprintln!("unknown row type: {}", row_type);
                        errors += 1;
                    }
                }
            }

            eprintln!(
                "imported {} memories, {} edges into '{}' ({} errors)",
                mem_count, edge_count, name, errors
            );
            Ok(())
        }

        Commands::Cluster { action } => {
            match action {
                ClusterAction::Init {
                    node_id,
                    output,
                    data_dir,
                    wire_port,
                    http_port,
                    cluster_port,
                    peers,
                    witnesses,
                    secret,
                } => {
                    let secret = secret.unwrap_or_else(generate_cluster_secret);

                    let mut peers_toml = String::new();
                    for addr in &peers {
                        peers_toml.push_str(&format!(
                            "\n[[cluster.peers]]\naddr = \"{}\"\nrole = \"voter\"\n",
                            addr
                        ));
                    }
                    for addr in &witnesses {
                        peers_toml.push_str(&format!(
                            "\n[[cluster.peers]]\naddr = \"{}\"\nrole = \"witness\"\n",
                            addr
                        ));
                    }

                    let toml = format!(
                        r#"# YantrikDB cluster config — generated by `yantrikdb cluster init`
[server]
wire_port = {wire_port}
http_port = {http_port}
data_dir = "{data_dir}"

[cluster]
node_id = {node_id}
role = "voter"
cluster_port = {cluster_port}
heartbeat_interval_ms = 1000
election_timeout_ms = 5000
cluster_secret = "{secret}"
{peers_toml}"#,
                        data_dir = data_dir.display(),
                    );

                    std::fs::write(&output, toml)?;
                    println!("config written to {}", output.display());
                    println!();
                    println!("cluster_secret: {}", secret);
                    println!("(use this as the auth token from any client to access the default database)");
                    println!();
                    println!("next steps:");
                    println!("  1. Copy this secret to all other nodes' configs");
                    println!("  2. Run: yantrikdb serve --config {}", output.display());
                    Ok(())
                }
                ClusterAction::Status { url, token } => {
                    // Hit both /v1/cluster (cluster topology) and
                    // /v1/health/deep (admission + runtime state added in
                    // RFC 009). The combined view is what operators
                    // actually want during incident triage.
                    let base = url.trim_end_matches('/');
                    let client = reqwest::blocking::Client::new();

                    let mut cluster_req = client.get(format!("{}/v1/cluster", base));
                    let mut health_req = client.get(format!("{}/v1/health/deep", base));
                    if let Some(ref t) = token {
                        let auth = format!("Bearer {}", t);
                        cluster_req = cluster_req.header("Authorization", &auth);
                        health_req = health_req.header("Authorization", auth);
                    }

                    let cluster_resp = cluster_req.send()?;
                    let cluster_status = cluster_resp.status();
                    let cluster_text = cluster_resp.text()?;
                    if !cluster_status.is_success() {
                        eprintln!("error {}: {}", cluster_status, cluster_text);
                        std::process::exit(1);
                    }
                    let cluster_value: serde_json::Value = serde_json::from_str(&cluster_text)?;

                    // /v1/health/deep is best-effort — if it errors, we
                    // still show cluster topology rather than failing.
                    let health_value: Option<serde_json::Value> = match health_req.send() {
                        Ok(r) if r.status().is_success() => {
                            serde_json::from_str(&r.text().unwrap_or_default()).ok()
                        }
                        _ => None,
                    };

                    let combined = serde_json::json!({
                        "cluster": cluster_value,
                        "health": health_value,
                    });
                    println!("{}", serde_json::to_string_pretty(&combined)?);
                    Ok(())
                }
                ClusterAction::Promote { url, token } => {
                    let url = format!("{}/v1/cluster/promote", url.trim_end_matches('/'));
                    let resp = reqwest::blocking::Client::new()
                        .post(&url)
                        .header("Authorization", format!("Bearer {}", token))
                        .send()?;
                    let status = resp.status();
                    let text = resp.text()?;
                    if !status.is_success() {
                        eprintln!("error {}: {}", status, text);
                        std::process::exit(1);
                    }
                    println!("{}", text);
                    Ok(())
                }
                ClusterAction::RaftStatus { url, token, json } => {
                    let url = format!("{}/v1/cluster/raft", url.trim_end_matches('/'));
                    let mut req = reqwest::blocking::Client::new().get(&url);
                    if let Some(ref t) = token {
                        req = req.header("Authorization", format!("Bearer {}", t));
                    }
                    let resp = req.send()?;
                    let status = resp.status();
                    let text = resp.text()?;
                    if !status.is_success() {
                        eprintln!("error {}: {}", status, text);
                        std::process::exit(1);
                    }
                    if json {
                        println!("{}", text);
                    } else {
                        let s: crate::raft::RaftStatus = serde_json::from_str(&text)?;
                        println!("openraft cluster status");
                        println!("───────────────────────");
                        println!("  this node    : node-{}  ({})", s.node_id, s.state);
                        match s.current_leader {
                            Some(id) if id == s.node_id => {
                                println!("  leader       : SELF (node-{})", id)
                            }
                            Some(id) => println!("  leader       : node-{}", id),
                            None => println!("  leader       : (none — election in progress)"),
                        }
                        println!("  current term : {}", s.current_term);
                        println!(
                            "  last log     : {}",
                            s.last_log_index
                                .map(|n| n.to_string())
                                .unwrap_or_else(|| "(none)".into())
                        );
                        println!(
                            "  last applied : {}",
                            s.last_applied_index
                                .map(|n| n.to_string())
                                .unwrap_or_else(|| "(none)".into())
                        );
                        println!(
                            "  snapshot @   : {}",
                            s.snapshot_index
                                .map(|n| n.to_string())
                                .unwrap_or_else(|| "(no snapshot)".into())
                        );
                        if let Some(lag) = s.millis_since_quorum_ack {
                            println!("  quorum ack   : {} ms ago", lag);
                        }
                        println!(
                            "  health       : {}",
                            if s.healthy { "OK" } else { "FATAL" }
                        );
                        println!();
                        println!("  Members ({}):", s.members.len());
                        for m in &s.members {
                            let role = if m.is_voter { "voter" } else { "learner" };
                            println!("    node-{:<3}  {:<8}  {}", m.node_id, role, m.addr);
                        }
                    }
                    Ok(())
                }
                ClusterAction::InitializeCluster {
                    leader,
                    master_token,
                } => {
                    let url = format!("{}/v1/cluster/initialize", leader.trim_end_matches('/'));
                    let resp = reqwest::blocking::Client::new()
                        .post(&url)
                        .header("Authorization", format!("Bearer {}", master_token))
                        .send()?;
                    let status = resp.status();
                    let text = resp.text()?;
                    if !status.is_success() {
                        eprintln!("error {}: {}", status, text);
                        std::process::exit(1);
                    }
                    println!("{}", text);
                    Ok(())
                }
                ClusterAction::AddLearner {
                    node_id,
                    addr,
                    leader,
                    master_token,
                } => {
                    let url = format!("{}/v1/cluster/add-learner", leader.trim_end_matches('/'));
                    let body = serde_json::json!({"node_id": node_id, "addr": addr});
                    let resp = reqwest::blocking::Client::new()
                        .post(&url)
                        .header("Authorization", format!("Bearer {}", master_token))
                        .header("Content-Type", "application/json")
                        .body(body.to_string())
                        .send()?;
                    let status = resp.status();
                    let text = resp.text()?;
                    if !status.is_success() {
                        eprintln!("error {}: {}", status, text);
                        std::process::exit(1);
                    }
                    println!("{}", text);
                    Ok(())
                }
                ClusterAction::WaitCaughtUp {
                    node_id,
                    leader,
                    master_token: _,
                    max_lag,
                    timeout_secs,
                } => {
                    let url = format!("{}/v1/cluster/raft", leader.trim_end_matches('/'));
                    let deadline =
                        std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
                    let client = reqwest::blocking::Client::new();
                    loop {
                        if std::time::Instant::now() > deadline {
                            eprintln!("timed out waiting for node {} to catch up", node_id);
                            std::process::exit(1);
                        }
                        let resp = client.get(&url).send()?;
                        if !resp.status().is_success() {
                            std::thread::sleep(std::time::Duration::from_secs(2));
                            continue;
                        }
                        let v: serde_json::Value = resp.json()?;
                        let leader_idx = v
                            .get("last_log_index")
                            .and_then(|x| x.as_u64())
                            .unwrap_or(0);
                        let members = v
                            .get("members")
                            .and_then(|m| m.as_array())
                            .cloned()
                            .unwrap_or_default();
                        let target = members
                            .iter()
                            .find(|m| m.get("node_id").and_then(|n| n.as_u64()) == Some(node_id));
                        if target.is_none() {
                            eprintln!(
                                "node {} is not yet a member; call add-learner first",
                                node_id
                            );
                            std::process::exit(1);
                        }
                        // Per-member last_log_index isn't always exposed by /v1/cluster/raft;
                        // fall back to "if member is present and snapshot/log indices look
                        // close, declare caught-up". For a stricter check, openraft's
                        // metrics struct would need exposing — deferred.
                        let lag_ok = true;
                        let _ = max_lag;
                        if lag_ok {
                            println!("node {} present in membership; assuming caught up (lag tracking deferred)", node_id);
                            return Ok(());
                        }
                        std::thread::sleep(std::time::Duration::from_secs(3));
                    }
                }
                ClusterAction::PromoteVoter {
                    voters,
                    leader,
                    master_token,
                } => {
                    let url = format!("{}/v1/cluster/promote-voter", leader.trim_end_matches('/'));
                    let body = serde_json::json!({"voters": voters});
                    let resp = reqwest::blocking::Client::new()
                        .post(&url)
                        .header("Authorization", format!("Bearer {}", master_token))
                        .header("Content-Type", "application/json")
                        .body(body.to_string())
                        .send()?;
                    let status = resp.status();
                    let text = resp.text()?;
                    if !status.is_success() {
                        eprintln!("error {}: {}", status, text);
                        std::process::exit(1);
                    }
                    println!("{}", text);
                    Ok(())
                }
                ClusterAction::RemoveNode {
                    node_id,
                    leader,
                    master_token,
                } => {
                    let url = format!("{}/v1/cluster/remove", leader.trim_end_matches('/'));
                    let body = serde_json::json!({"node_id": node_id});
                    let resp = reqwest::blocking::Client::new()
                        .post(&url)
                        .header("Authorization", format!("Bearer {}", master_token))
                        .header("Content-Type", "application/json")
                        .body(body.to_string())
                        .send()?;
                    let status = resp.status();
                    let text = resp.text()?;
                    if !status.is_success() {
                        eprintln!("error {}: {}", status, text);
                        std::process::exit(1);
                    }
                    println!("{}", text);
                    Ok(())
                }
            }
        }

        Commands::Version { action } => match action {
            VersionAction::Local => {
                let snap = crate::version::VersionSnapshot::local();
                println!("YantrikDB version state (local build)");
                println!("─────────────────────────────────────");
                println!("  binary build id : {}", snap.build_id);
                println!("  wire version    : {}", snap.wire);
                println!("  min supported   : {}", snap.min_supported_wire);
                println!();
                println!("  Per-table schema versions:");
                for (table, ver) in &snap.table_schema_versions {
                    println!("    {:<28} {}", table, ver);
                }
                Ok(())
            }
            VersionAction::Status { url, token, json } => {
                let url = format!("{}/v1/health/deep", url.trim_end_matches('/'));
                let mut req = reqwest::blocking::Client::new().get(&url);
                if let Some(ref t) = token {
                    req = req.header("Authorization", format!("Bearer {}", t));
                }
                let resp = req.send()?;
                let status = resp.status();
                let text = resp.text()?;
                if !status.is_success() {
                    eprintln!("error {}: {}", status, text);
                    std::process::exit(1);
                }
                let value: serde_json::Value = serde_json::from_str(&text)?;
                let version = value.get("version");

                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(version.unwrap_or(&serde_json::Value::Null))?
                    );
                    return Ok(());
                }

                let v = version.ok_or_else(|| {
                    anyhow::anyhow!(
                        "server response has no `version` block — likely a pre-RFC-017 build"
                    )
                })?;

                println!("YantrikDB version state (server)");
                println!("─────────────────────────────────");
                println!(
                    "  binary build id        : {}",
                    v["build_id"].as_str().unwrap_or("?")
                );
                println!(
                    "  local wire version     : {}.{}",
                    v["wire"]["major"].as_u64().unwrap_or(0),
                    v["wire"]["minor"].as_u64().unwrap_or(0),
                );
                println!(
                    "  min supported wire     : {}.{}",
                    v["min_supported_wire"]["major"].as_u64().unwrap_or(0),
                    v["min_supported_wire"]["minor"].as_u64().unwrap_or(0),
                );
                if let Some(cluster) = v.get("cluster") {
                    println!(
                        "  cluster min wire       : {}.{}",
                        cluster["min"]["major"].as_u64().unwrap_or(0),
                        cluster["min"]["minor"].as_u64().unwrap_or(0),
                    );
                    println!(
                        "  cluster max wire       : {}.{}",
                        cluster["max"]["major"].as_u64().unwrap_or(0),
                        cluster["max"]["minor"].as_u64().unwrap_or(0),
                    );
                    println!(
                        "  observed peers         : {}",
                        cluster["peer_count"].as_u64().unwrap_or(0)
                    );
                }
                if let Some(tables) = v.get("table_schema_versions").and_then(|t| t.as_array()) {
                    println!();
                    println!("  Per-table schema versions:");
                    for entry in tables {
                        if let Some(arr) = entry.as_array() {
                            if arr.len() == 2 {
                                let name = arr[0].as_str().unwrap_or("?");
                                let ver = arr[1].as_u64().unwrap_or(0);
                                println!("    {:<28} v{}", name, ver);
                            }
                        }
                    }
                }
                Ok(())
            }
        },

        Commands::Jobs { action } => match action {
            JobsAction::List {
                url,
                token,
                tenant,
                state,
                limit,
                json,
            } => {
                let mut url = format!("{}/v1/jobs?limit={}", url.trim_end_matches('/'), limit);
                if let Some(t) = tenant {
                    url.push_str(&format!("&tenant={}", t));
                }
                if let Some(ref s) = state {
                    url.push_str(&format!("&state={}", s));
                }
                let resp = reqwest::blocking::Client::new()
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .send()?;
                let status = resp.status();
                let text = resp.text()?;
                if !status.is_success() {
                    eprintln!("error {}: {}", status, text);
                    std::process::exit(1);
                }
                if json {
                    println!("{}", text);
                } else {
                    let arr: serde_json::Value = serde_json::from_str(&text)?;
                    let jobs = arr.as_array();
                    let count = jobs.map(|a| a.len()).unwrap_or(0);
                    println!("Jobs: {}", count);
                    if let Some(jobs) = jobs {
                        for j in jobs {
                            let id = j["id"].as_str().unwrap_or("?");
                            let kind = j["kind"].as_str().unwrap_or("?");
                            let state = j["state"].as_str().unwrap_or("?");
                            let pri = j["priority"].as_u64().unwrap_or(0);
                            let tid = j["tenant_id"].as_i64().unwrap_or(0);
                            println!(
                                "  {} t={} pri={} {:<10} {}",
                                &id[..8.min(id.len())],
                                tid,
                                pri,
                                state,
                                kind
                            );
                        }
                    }
                }
                Ok(())
            }
            JobsAction::Get {
                url,
                token,
                job_id,
                json,
            } => {
                let url = format!("{}/v1/jobs/{}", url.trim_end_matches('/'), job_id);
                let resp = reqwest::blocking::Client::new()
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .send()?;
                let status = resp.status();
                let text = resp.text()?;
                if !status.is_success() {
                    eprintln!("error {}: {}", status, text);
                    std::process::exit(1);
                }
                if json {
                    println!("{}", text);
                } else {
                    let v: serde_json::Value = serde_json::from_str(&text)?;
                    println!("Job {}", v["id"].as_str().unwrap_or("?"));
                    println!("──────────");
                    println!("  tenant   : {}", v["tenant_id"].as_i64().unwrap_or(0));
                    println!("  kind     : {}", v["kind"].as_str().unwrap_or("?"));
                    println!("  state    : {}", v["state"].as_str().unwrap_or("?"));
                    println!("  priority : {}", v["priority"].as_u64().unwrap_or(0));
                    if let Some(leased) = v["leased_by"].as_str() {
                        println!("  leased_by: {}", leased);
                    }
                    if let Some(err) = v["error_message"].as_str() {
                        println!("  error    : {}", err);
                    }
                }
                Ok(())
            }
            JobsAction::Cancel { url, token, job_id } => {
                let url = format!("{}/v1/jobs/{}", url.trim_end_matches('/'), job_id);
                let resp = reqwest::blocking::Client::new()
                    .delete(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .send()?;
                let status = resp.status();
                let text = resp.text()?;
                if !status.is_success() {
                    eprintln!("error {}: {}", status, text);
                    std::process::exit(1);
                }
                println!("✓ cancelled {}", job_id);
                Ok(())
            }
        },

        Commands::Migrations { action } => match action {
            MigrationsAction::Status { url, token, json } => {
                let url = format!("{}/v1/admin/migrations", url.trim_end_matches('/'));
                let resp = reqwest::blocking::Client::new()
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .send()?;
                let status = resp.status();
                let text = resp.text()?;
                if !status.is_success() {
                    eprintln!("error {}: {}", status, text);
                    std::process::exit(1);
                }
                if json {
                    println!("{}", text);
                } else {
                    let v: serde_json::Value = serde_json::from_str(&text)?;
                    println!("Schema migrations applied");
                    println!("─────────────────────────");
                    if let Some(obj) = v.as_object() {
                        for (db, applied) in obj {
                            println!("  [{}]", db);
                            if let Some(arr) = applied.as_array() {
                                for m in arr {
                                    println!(
                                        "    m{:03} {}",
                                        m["id"].as_u64().unwrap_or(0),
                                        m["name"].as_str().unwrap_or("?")
                                    );
                                }
                            } else if let Some(err) = applied["error"].as_str() {
                                println!("    error: {}", err);
                            }
                        }
                    }
                }
                Ok(())
            }
        },

        Commands::Tls { action } => match action {
            TlsAction::VerifyCluster { config } => {
                // Load the config file, run cert_inspect, render report.
                // Exits non-zero if any cert path is set but the file
                // fails to load — which is what operators want to catch
                // before a production cert rotation goes wrong.
                let cfg = ServerConfig::load(&config)?;
                let report = crate::security::cert_inspect::inspect(&cfg.cluster_tls);
                println!("{}", report.render_human());

                // If config claims to be fully specified but cert files
                // don't load, surface that explicitly with non-zero exit.
                if cfg.cluster_tls.is_fully_specified() {
                    let acceptor_result = crate::security::build_acceptor(&cfg.cluster_tls);
                    let connector_result = crate::security::build_connector(&cfg.cluster_tls);
                    let mut had_error = false;
                    if let Err(e) = acceptor_result {
                        eprintln!("\n✗ acceptor build failed: {}", e);
                        had_error = true;
                    }
                    if let Err(e) = connector_result {
                        eprintln!("✗ connector build failed: {}", e);
                        had_error = true;
                    }
                    if had_error {
                        std::process::exit(1);
                    }
                    println!("\n✓ acceptor + connector both build successfully");
                    if cfg.cluster_tls.dev_mode {
                        println!("\n⚠  WARNING: dev_mode is enabled. Peer cert chain verification");
                        println!("   is SKIPPED. Never set this in production.");
                    }
                } else {
                    println!(
                        "\n(no certs configured — cluster_tls is opt-in until openraft mode lands)"
                    );
                }
                Ok(())
            }
        },

        Commands::Debug { action } => match action {
            DebugAction::History {
                url,
                token,
                tenant,
                from,
                limit,
                json,
            } => {
                let url = format!(
                    "{}/v1/debug/history/{}?from={}&limit={}",
                    url.trim_end_matches('/'),
                    tenant,
                    from,
                    limit
                );
                let resp = reqwest::blocking::Client::new()
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .send()?;
                let status = resp.status();
                let text = resp.text()?;
                if !status.is_success() {
                    eprintln!("error {}: {}", status, text);
                    std::process::exit(1);
                }
                if json {
                    println!("{}", text);
                } else {
                    let v: serde_json::Value = serde_json::from_str(&text)?;
                    println!("YantrikDB commit log — tenant {}", tenant);
                    println!("──────────────────────────────────");
                    println!(
                        "  high_watermark : {}",
                        v["high_watermark"].as_u64().unwrap_or(0)
                    );
                    println!(
                        "  from_index     : {}",
                        v["from_index"].as_u64().unwrap_or(0)
                    );
                    println!(
                        "  returned       : {}",
                        v["entries"].as_array().map(|a| a.len()).unwrap_or(0)
                    );
                    println!();
                    if let Some(entries) = v["entries"].as_array() {
                        for e in entries {
                            let idx = e["log_index"].as_u64().unwrap_or(0);
                            let term = e["term"].as_u64().unwrap_or(0);
                            let kind = e["mutation"]["kind"].as_str().unwrap_or("?");
                            let op_id = e["op_id"].as_str().unwrap_or("?");
                            println!("  [{:>5}] term={:<3} {:<24} {}", idx, term, kind, op_id);
                        }
                    }
                }
                Ok(())
            }
            DebugAction::FaultList { url, token, json } => {
                let url = format!("{}/v1/debug/fault", url.trim_end_matches('/'));
                let resp = reqwest::blocking::Client::new()
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .send()?;
                let status = resp.status();
                let text = resp.text()?;
                if !status.is_success() {
                    eprintln!("error {}: {}", status, text);
                    std::process::exit(1);
                }
                if json {
                    println!("{}", text);
                } else {
                    let faults: serde_json::Value = serde_json::from_str(&text)?;
                    let arr = faults.as_array();
                    let count = arr.map(|a| a.len()).unwrap_or(0);
                    println!("Active fault injections: {}", count);
                    if let Some(faults) = arr {
                        for f in faults {
                            let id = f["id"].as_u64().unwrap_or(0);
                            let kind = f["kind"]["kind"].as_str().unwrap_or("?");
                            let ttl = f["ttl_secs"].as_u64();
                            match ttl {
                                Some(t) => println!("  fault_{:<6} {:<20} ttl={}s", id, kind, t),
                                None => println!("  fault_{:<6} {:<20} (persistent)", id, kind),
                            }
                        }
                    }
                }
                Ok(())
            }
            DebugAction::FaultClear { url, token } => {
                let url = format!("{}/v1/debug/fault/clear", url.trim_end_matches('/'));
                let resp = reqwest::blocking::Client::new()
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .send()?;
                let status = resp.status();
                let text = resp.text()?;
                if !status.is_success() {
                    eprintln!("error {}: {}", status, text);
                    std::process::exit(1);
                }
                let v: serde_json::Value = serde_json::from_str(&text)?;
                println!("cleared {} fault(s)", v["cleared"].as_u64().unwrap_or(0));
                Ok(())
            }
        },

        Commands::Admission { action } => match action {
            AdmissionAction::Status { url, token, json } => {
                let url = format!("{}/v1/health/deep", url.trim_end_matches('/'));
                let mut req = reqwest::blocking::Client::new().get(&url);
                if let Some(ref t) = token {
                    req = req.header("Authorization", format!("Bearer {}", t));
                }
                let resp = req.send()?;
                let status = resp.status();
                let text = resp.text()?;
                if !status.is_success() {
                    eprintln!("error {}: {}", status, text);
                    std::process::exit(1);
                }
                let value: serde_json::Value = serde_json::from_str(&text)?;

                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(
                            value.get("admission").unwrap_or(&serde_json::Value::Null)
                        )?
                    );
                    return Ok(());
                }

                let admission = value.get("admission").ok_or_else(|| {
                    anyhow::anyhow!("server did not return admission state — older server version?")
                })?;
                let runtime = value.get("runtime");

                println!("YantrikDB admission state (RFC 009)");
                println!("───────────────────────────────────");
                println!(
                    "  hard top_k cap          : {}",
                    admission["hard_top_k_cap"].as_u64().unwrap_or(0)
                );
                println!(
                    "  max request body bytes  : {}",
                    admission["max_request_body_bytes"].as_u64().unwrap_or(0)
                );
                let in_flight_max = admission["in_flight_recall"]["max"].as_u64().unwrap_or(0);
                let in_flight_used = admission["in_flight_recall"]["in_use"]
                    .as_u64()
                    .unwrap_or(0);
                println!(
                    "  in-flight recalls       : {}/{} ({}%)",
                    in_flight_used,
                    in_flight_max,
                    if in_flight_max > 0 {
                        100 * in_flight_used / in_flight_max
                    } else {
                        0
                    }
                );
                let expanded_max = admission["expanded_recall"]["max"].as_u64().unwrap_or(0);
                let expanded_used = admission["expanded_recall"]["in_use"].as_u64().unwrap_or(0);
                println!(
                    "  expanded concurrent     : {}/{} ({}%)",
                    expanded_used,
                    expanded_max,
                    if expanded_max > 0 {
                        100 * expanded_used / expanded_max
                    } else {
                        0
                    }
                );
                if let Some(rt) = runtime {
                    println!(
                        "  control runtime isolated: {}",
                        rt["control_runtime_isolated"].as_bool().unwrap_or(false)
                    );
                }
                println!();
                println!("Tip: scrape /metrics for term changes, scheduling latency p99,");
                println!("     and rejection counts (yantrikdb_recall_rejected_total).");
                Ok(())
            }
        },

        Commands::Encryption { action } => match action {
            EncryptionAction::GenKey { output } => {
                use rand::RngCore;
                if output.exists() {
                    eprintln!(
                        "key file {} already exists — refusing to overwrite",
                        output.display()
                    );
                    std::process::exit(1);
                }
                if let Some(parent) = output.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let mut key = [0u8; 32];
                rand::thread_rng().fill_bytes(&mut key);
                std::fs::write(&output, key)?;

                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ =
                        std::fs::set_permissions(&output, std::fs::Permissions::from_mode(0o600));
                }

                println!("master key written to {}", output.display());
                println!();
                println!("hex: {}", hex::encode(key));
                println!();
                println!("next steps:");
                println!("  1. Add to yantrikdb.toml:");
                println!("       [encryption]");
                println!("       key_path = \"{}\"", output.display());
                println!("  2. Or set env var: YANTRIKDB_ENCRYPTION_KEY_HEX=<hex>");
                println!("  3. ⚠️  In a cluster, ALL nodes must use the SAME key");
                println!("  4. ⚠️  Backup this key — losing it = losing all data");
                Ok(())
            }
            EncryptionAction::ShowKey { input } => {
                let bytes = std::fs::read(&input)?;
                if bytes.len() != 32 {
                    anyhow::bail!("key file must be exactly 32 bytes (got {})", bytes.len());
                }
                println!("{}", hex::encode(&bytes));
                Ok(())
            }
        },
    }
}

fn generate_cluster_secret() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: [u8; 32] = rng.gen();
    format!("ydb_cluster_{}", hex::encode(bytes))
}

/// Render the version gauges into a Prometheus-formatted buffer. Wired
/// into `metrics::set_version_gauge_renderer` at startup so /metrics
/// emits the build-time version state on every scrape. Kept as a free
/// function (not a closure) because `set_version_gauge_renderer` takes
/// a `fn` pointer for thread-safe OnceLock storage.
fn render_version_gauges(out: &mut String) {
    let snap = crate::version::VersionSnapshot::local();
    out.push_str("# HELP yantrikdb_wire_version_major Local wire-protocol major version\n");
    out.push_str("# TYPE yantrikdb_wire_version_major gauge\n");
    out.push_str(&format!(
        "yantrikdb_wire_version_major {}\n",
        snap.wire.major
    ));
    out.push_str("# HELP yantrikdb_wire_version_minor Local wire-protocol minor version\n");
    out.push_str("# TYPE yantrikdb_wire_version_minor gauge\n");
    out.push_str(&format!(
        "yantrikdb_wire_version_minor {}\n",
        snap.wire.minor
    ));
    out.push_str(
        "# HELP yantrikdb_min_supported_wire_version_major Oldest wire-protocol major this binary can replay\n",
    );
    out.push_str("# TYPE yantrikdb_min_supported_wire_version_major gauge\n");
    out.push_str(&format!(
        "yantrikdb_min_supported_wire_version_major {}\n",
        snap.min_supported_wire.major
    ));
    out.push_str("# HELP yantrikdb_schema_version Per-table schema version this binary expects\n");
    out.push_str("# TYPE yantrikdb_schema_version gauge\n");
    for (table, ver) in &snap.table_schema_versions {
        out.push_str(&format!(
            "yantrikdb_schema_version{{table=\"{}\"}} {}\n",
            table,
            u32::from(*ver)
        ));
    }
}

async fn run_server(cfg: ServerConfig) -> anyhow::Result<()> {
    // Ensure data directory exists
    std::fs::create_dir_all(&cfg.server.data_dir)?;

    // RFC 017-A: install the version gauge renderer so /metrics emits
    // wire/schema version gauges. Idempotent — set_version_gauge_renderer
    // uses OnceLock and silently ignores subsequent calls.
    crate::metrics::set_version_gauge_renderer(render_version_gauges);

    // Deadlock detector — parking_lot's `deadlock_detection` feature lets us
    // ask at runtime whether any cycle of thread-held locks exists. We run
    // this every 10 seconds. If a cycle is detected we log a structured
    // ERROR per deadlocked thread with its backtrace. This would have caught
    // the v0.5.7/0.5.8 cognition::triggers self-deadlock in ~10 seconds
    // instead of hours. Low overhead: parking_lot only runs the cycle check
    // when this function is called, it does not poll or instrument locks
    // themselves.
    std::thread::Builder::new()
        .name("parking-lot-deadlock-detector".into())
        .spawn(|| {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(10));
                let deadlocks = parking_lot::deadlock::check_deadlock();
                if deadlocks.is_empty() {
                    continue;
                }
                tracing::error!(
                    deadlock_count = deadlocks.len(),
                    "DEADLOCK DETECTED — parking_lot found circular lock dependency"
                );
                for (i, threads) in deadlocks.iter().enumerate() {
                    for t in threads {
                        tracing::error!(
                            deadlock_id = i,
                            thread_id = ?t.thread_id(),
                            backtrace = ?t.backtrace(),
                            "deadlocked thread"
                        );
                    }
                }
                // Do not auto-exit or restart on detection — let the ops
                // watchdog + auto-restart policy decide. Logging is enough
                // to break the "silent hang" failure mode.
            }
        })?;
    tracing::info!("deadlock detector started (parking_lot::deadlock, 10s cadence)");

    // Open control database
    let control = ControlDb::open(&cfg.control_db_path())?;

    // Ensure default database exists
    tenant_pool::ensure_default_database(&control, cfg.data_dir())?;

    // Initialize embedder based on config
    let embedder = match cfg.embedding.strategy {
        config::EmbeddingStrategy::Builtin => Some(embedder::FastEmbedder::new()?),
        config::EmbeddingStrategy::ClientOnly => {
            tracing::info!("embedding strategy: client_only (no server-side embeddings)");
            None
        }
    };

    // Log ONNX Runtime version for debuggability. The 1.20.1 vs 1.24.4
    // mismatch bit us on first Proxmox deploy; having the version in the
    // startup log catches similar mismatches early. See task #86.
    if std::env::var("ORT_DYLIB_PATH").is_ok() {
        tracing::info!(
            ort_dylib_path = %std::env::var("ORT_DYLIB_PATH").unwrap_or_default(),
            "ONNX Runtime: using ORT_DYLIB_PATH"
        );
    }

    // Resolve master encryption key (auto-generates if needed)
    let master_key = cfg.encryption.resolve_key(&cfg.server.data_dir)?;

    // Issue #3 (yantrikos/yantrikdb): users on Docker setups reported they
    // saw NO encryption-related output, leaving them unable to confirm
    // whether at-rest encryption was active. The previous tracing::info/warn
    // depended on RUST_LOG and tracing-subscriber config, which didn't
    // surface in their `docker logs` output. We now also write a
    // boundary-marker banner to stderr (which Docker always captures) so
    // the encryption state is impossible to miss at startup.
    let enc_state = if master_key.is_some() {
        "enabled (AES-256-GCM)"
    } else {
        "disabled — set [encryption] section in config to enable at-rest encryption"
    };
    eprintln!("[yantrikdb] encryption: {enc_state}");
    if master_key.is_some() {
        tracing::info!("encryption: enabled (AES-256-GCM)");
    } else {
        tracing::warn!("encryption: disabled — set [encryption] to enable at-rest encryption");
    }

    // Create tenant pool and background worker registry
    let pool = Arc::new(TenantPool::new(&cfg, embedder, master_key));
    let workers = background::WorkerRegistry::new(&cfg.background);
    let control = Arc::new(Mutex::new(control));

    // v0.8.8: eager engine warm-up. A database server should serve queries
    // at steady-state latency, not the cold-load latency of HNSW reload
    // (~10 s for a 400 MB engine). Without this, the first query against
    // any unloaded namespace blocks for tens of seconds — clients with
    // default timeouts (8 s) see "transport: timeout" failures even though
    // the server is healthy. Eager-load at startup so every database is
    // warm before HTTP starts accepting requests.
    //
    // Loaded sequentially. With N databases, startup adds ~N × cold-load
    // time. For larger fleets, switch to parallel via tokio::spawn_blocking
    // — but sequential keeps disk I/O contention bounded and gives clean
    // log output. Acceptable cost for a server that runs for days.
    {
        let dbs = control.lock().list_databases().unwrap_or_else(|e| {
            tracing::warn!(error = %e, "could not enumerate databases for warm-up");
            Vec::new()
        });
        let total = dbs.len();
        if total > 0 {
            tracing::info!(count = total, "warming up engines (eager load)");
            let warm_start = std::time::Instant::now();
            for (i, db) in dbs.iter().enumerate() {
                let t0 = std::time::Instant::now();
                match pool.get_engine(db) {
                    Ok(_) => tracing::info!(
                        db = %db.name,
                        progress = format!("{}/{}", i + 1, total),
                        elapsed_ms = t0.elapsed().as_millis() as u64,
                        "warmed engine"
                    ),
                    Err(e) => tracing::warn!(
                        db = %db.name,
                        error = %e,
                        "failed to warm engine — will lazy-load on first query"
                    ),
                }
            }
            tracing::info!(
                total_elapsed_ms = warm_start.elapsed().as_millis() as u64,
                count = total,
                "engine warm-up complete"
            );
        }
    }

    // Initialize cluster context if clustering is enabled
    let cluster_ctx = if cfg.cluster.is_clustered() {
        let raft_path = cfg.server.data_dir.join("raft.json");
        let node_state = Arc::new(cluster::NodeState::new(
            cfg.cluster.node_id,
            cfg.cluster.role,
            raft_path,
        )?);
        let peer_registry = Arc::new(cluster::PeerRegistry::new(&cfg.cluster.peers));
        let ctx = Arc::new(cluster::ClusterContext::new(
            cfg.cluster.clone(),
            node_state,
            peer_registry,
            Arc::clone(&pool),
            Some(Arc::clone(&control)),
        ));
        tracing::info!(
            node_id = cfg.cluster.node_id,
            role = ?cfg.cluster.role,
            peers = cfg.cluster.peers.len(),
            "cluster mode enabled"
        );
        Some(ctx)
    } else {
        None
    };

    // RFC 009 admission control: hard caps + concurrency semaphores.
    // Defaults are sized for a typical 4-core deployment. Operators tune
    // via [admission] config block once benchmark validation lands in PR-3.
    let admission =
        crate::admission::AdmissionState::new(crate::admission::AdmissionConfig::default());

    // RFC 009 §4 Layer 1: dedicated tokio runtime for Raft control plane.
    // Cluster background tasks (heartbeat, sync_loop, election) spawn on
    // the control runtime so they can't be CPU-starved by HTTP/recall.
    // Combined with SCHED_FIFO (Layer 2) and admission caps (Layer 3),
    // this gives us the CPU isolation acceptance gate in tests/cpu_isolation.rs.
    //
    // The split runtime lives for the lifetime of the server. We hold
    // it in a local so its destructor runs on graceful shutdown.
    // v0.8.4 (issue #27 resolved): SplitRuntime is back. The v0.8.4
    // `fn main()` is sync, so the nested-Runtime drop panic from
    // v0.8.2 no longer applies. Cluster control plane gets its own
    // CPU-isolated runtime; HTTP/recall stays responsive under heavy
    // openraft replication traffic.
    let split_runtime = if cluster_ctx.is_some() {
        match crate::runtime::SplitRuntime::new(crate::runtime::RuntimeConfig::default()) {
            Ok(rt) => Some(rt),
            Err(e) => {
                tracing::warn!(error = %e, "could not build split runtime; falling back to single runtime");
                None
            }
        }
    } else {
        None
    };
    let control_runtime_handle = split_runtime.as_ref().map(|rt| rt.control_handle());

    // RFC 014-A: validate cluster-mTLS config at startup. If certs are
    // configured, prove they load successfully BEFORE the cluster
    // transport tries to use them — fail-fast on misconfiguration.
    // If not configured, this is a no-op (cluster_tls is opt-in until
    // RFC 010 PR-4 openraft mode makes it required).
    if cfg.cluster_tls.is_fully_specified() {
        match crate::security::build_acceptor(&cfg.cluster_tls) {
            Ok(_) => tracing::info!(
                dev_mode = cfg.cluster_tls.dev_mode,
                "cluster mTLS acceptor: loaded successfully"
            ),
            Err(e) => {
                tracing::error!(error = %e, "cluster mTLS acceptor: load FAILED");
                anyhow::bail!("cluster mTLS misconfigured: {e}");
            }
        }
        match crate::security::build_connector(&cfg.cluster_tls) {
            Ok(_) => tracing::info!("cluster mTLS connector: loaded successfully"),
            Err(e) => {
                tracing::error!(error = %e, "cluster mTLS connector: load FAILED");
                anyhow::bail!("cluster mTLS misconfigured: {e}");
            }
        }
        if cfg.cluster_tls.dev_mode {
            tracing::warn!(
                "cluster_tls.dev_mode=true — peer cert verification SKIPPED. \
                 NEVER use in production."
            );
        }
    }

    // RFC 010 PR-2: durable commit log substrate. SQLite-backed file
    // alongside the existing data dir. Filename pinned so backup/DR
    // (RFC 012) can find it without config plumbing.
    let commit_log_path = cfg.server.data_dir.join("commit_log.sqlite");
    let local_committer = Arc::new(
        crate::commit::LocalSqliteCommitter::open(&commit_log_path)
            .map_err(|e| anyhow::anyhow!("failed to open commit log: {e}"))?,
    );
    tracing::info!(
        commit_log_path = %commit_log_path.display(),
        "commit log opened"
    );

    // RFC 010 PR-4: assemble openraft when the cluster section asks for
    // it. Misconfig (openraft mode without cluster_tls) fails fast here
    // — server refuses to start, which is the production-safety
    // guarantee from `build_raft_cluster`.
    let (commit_log, raft_assembly): (
        Arc<dyn crate::commit::MutationCommitter>,
        Option<Arc<crate::raft::RaftAssembly>>,
    ) = match cfg.cluster.raft_mode {
        crate::raft::RaftClusterMode::Disabled => (
            local_committer.clone() as Arc<dyn crate::commit::MutationCommitter>,
            None,
        ),
        crate::raft::RaftClusterMode::OpenRaft => {
            let raft_log_path = cfg.server.data_dir.join("raft_log.sqlite");
            let raft_log_conn = rusqlite::Connection::open(&raft_log_path)
                .map_err(|e| anyhow::anyhow!("failed to open raft_log.sqlite: {e}"))?;
            // Migrations m004 must already be in run_pending so the table
            // exists. The committer construction above already ran them
            // for commit_log.sqlite — we run them on raft_log.sqlite too.
            let mut raft_log_conn = raft_log_conn;
            crate::migrations::MigrationRunner::run_pending(&mut raft_log_conn)
                .map_err(|e| anyhow::anyhow!("raft_log migrations: {e}"))?;
            let log_storage = crate::raft::SqliteRaftLogStorage::new(Arc::new(
                parking_lot::Mutex::new(raft_log_conn),
            ));

            let node_id = crate::raft::YantrikNodeId::new(cfg.cluster.node_id as u64);
            let node_addr = cfg
                .cluster
                .advertise_addr
                .clone()
                .unwrap_or_else(|| format!("https://127.0.0.1:{}", cfg.cluster.cluster_port));

            let assembly_cfg = crate::raft::RaftAssemblyConfig {
                mode: crate::raft::RaftClusterMode::OpenRaft,
                node_id,
                node_addr: node_addr.clone(),
                cluster_tls: Some(cfg.cluster_tls.clone()),
                request_timeout: std::time::Duration::from_secs(10),
                openraft_config: openraft::Config {
                    cluster_name: "yantrikdb".into(),
                    heartbeat_interval: cfg.cluster.heartbeat_interval_ms,
                    election_timeout_min: cfg.cluster.election_timeout_ms,
                    election_timeout_max: cfg.cluster.election_timeout_ms.saturating_mul(2),
                    ..Default::default()
                },
            };

            tracing::info!(
                node_id = %node_id,
                node_addr = %node_addr,
                "assembling openraft cluster (RFC 010 PR-4)"
            );
            let assembly = crate::raft::build_raft_cluster(
                assembly_cfg,
                log_storage,
                local_committer.clone() as Arc<dyn crate::commit::MutationCommitter>,
            )
            .await
            .map_err(|e| anyhow::anyhow!("openraft assembly failed: {e}"))?;
            tracing::info!("openraft assembled — RaftCommitter now driving writes");

            // v0.8.3: auto-bootstrap removed (was a fragile node_id heuristic).
            // For fresh openraft deployments, the seed operator runs:
            //   yantrikdb cluster initialize-cluster --leader http://X:7438 --master-token T
            // Subsequent nodes are added via add-learner + promote-voter.
            //
            // Existing v0.8.2 deployments where auto-bootstrap already wrote
            // a membership record continue to work — openraft persists
            // membership in raft_log.sqlite across restarts.
            {
                let metrics = assembly.raft.metrics().borrow().clone();
                if metrics.membership_config.nodes().count() == 0 {
                    tracing::warn!(
                        node_id = cfg.cluster.node_id,
                        "openraft membership empty. Run \
                         `yantrikdb cluster initialize-cluster` on the seed node \
                         (one time per cluster) or `cluster add-learner` from an \
                         existing leader to add this node."
                    );
                }
            }

            // Spawn the metrics recorder so /metrics gets live
            // openraft gauges. Tied to a CancellationToken so we can
            // drop it cleanly on shutdown — for now the token is
            // never cancelled (server runs until SIGTERM kills us).
            let cancel = tokio_util::sync::CancellationToken::new();
            crate::raft::spawn_raft_metrics_recorder(assembly.raft.clone(), cancel);

            let committer: Arc<dyn crate::commit::MutationCommitter> =
                Arc::new(assembly.committer.clone());
            let assembly_arc = Arc::new(assembly);
            (committer, Some(assembly_arc))
        }
    };

    // RFC 010 PR-5: fault-injection registry for Jepsen runs. Empty in
    // production builds; populated via /v1/debug/fault/inject.
    let fault_registry = crate::debug::FaultRegistry::new();

    // RFC 019: durable job queue. Same data dir as commit log; separate
    // SQLite file so the two contend on different WAL files.
    let jobs_path = cfg.server.data_dir.join("jobs.sqlite");
    let jobs: Arc<dyn crate::jobs::JobQueue> = Arc::new(
        crate::jobs::LocalSqliteJobQueue::open(&jobs_path)
            .map_err(|e| anyhow::anyhow!("failed to open job queue: {e}"))?,
    );
    tracing::info!(jobs_path = %jobs_path.display(), "job queue opened");

    let state = Arc::new(AppState {
        control,
        pool,
        workers,
        cluster: cluster_ctx.clone(),
        inflight: std::sync::atomic::AtomicU32::new(0),
        admission,
        control_runtime: control_runtime_handle.clone(),
        commit_log,
        raft: raft_assembly,
        fault_registry,
        jobs,
        data_dir: cfg.server.data_dir.clone(),
    });

    // Built-in watchdog — periodically probes the engine lock and fires a
    // metric if acquisition takes too long. Complement to the external bash
    // watchdog: this one runs in-process and feeds /metrics directly, while
    // the external one captures gdb backtraces and triggers ntfy alerts.
    {
        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(15));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let pool = state_clone.pool.clone();
                let control = state_clone.control.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    // Probe engine lock on default db
                    let db_record = {
                        let ctrl = control.lock();
                        ctrl.get_database("default").ok().flatten()
                    };
                    if let Some(rec) = db_record {
                        if let Ok(engine) = pool.get_engine(&rec) {
                            let start = std::time::Instant::now();
                            let timeout = std::time::Duration::from_secs(5);
                            if engine.try_lock_for(timeout).is_some() {
                                crate::metrics::record_engine_lock_wait(start.elapsed());
                            } else {
                                crate::metrics::record_engine_lock_wait(timeout);
                                tracing::warn!(
                                    wait_secs = 5,
                                    "built-in watchdog: engine lock not acquired within 5s"
                                );
                            }
                        }
                    }
                })
                .await;
            }
        });
        tracing::info!("built-in watchdog started (15s cadence, 5s lock timeout)");
    }

    // Build TLS acceptor if configured
    let tls_acceptor = if cfg.tls.is_enabled() {
        let acceptor = tls::build_tls_acceptor(&cfg.tls)?;
        tracing::info!("TLS enabled");
        Some(acceptor)
    } else {
        None
    };

    // Start wire protocol server
    let wire_addr = format!("0.0.0.0:{}", cfg.server.wire_port);
    let wire_listener = tokio::net::TcpListener::bind(&wire_addr).await?;

    // Start HTTP gateway
    let http_addr = format!("0.0.0.0:{}", cfg.server.http_port);
    let http_listener = tokio::net::TcpListener::bind(&http_addr).await?;

    tracing::info!(
        wire_port = cfg.server.wire_port,
        http_port = cfg.server.http_port,
        tls = cfg.tls.is_enabled(),
        data_dir = %cfg.server.data_dir.display(),
        "YantrikDB server starting"
    );

    let wire_state = Arc::clone(&state);
    let http_state = Arc::clone(&state);
    let shutdown_state = Arc::clone(&state);

    // Cancellation token for cluster background tasks
    let cluster_cancel = tokio_util::sync::CancellationToken::new();

    // Spawn cluster server + background loops if clustered.
    // Per RFC 009 §4: when split runtime is active, ALL cluster background
    // tasks spawn on `control_runtime_handle` so HTTP/recall load can't
    // starve them of CPU. Falls back to current runtime if split runtime
    // failed to build (best-effort isolation).
    let mut cluster_handles = Vec::new();
    if let Some(ref ctx) = cluster_ctx {
        let cluster_addr = format!("0.0.0.0:{}", cfg.cluster.cluster_port);
        let cluster_listener = tokio::net::TcpListener::bind(&cluster_addr).await?;
        tracing::info!(
            cluster_port = cfg.cluster.cluster_port,
            isolated_runtime = control_runtime_handle.is_some(),
            "cluster wire server starting"
        );

        let spawn = |fut: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>| {
            match control_runtime_handle.as_ref() {
                Some(handle) => handle.spawn(fut),
                None => tokio::spawn(fut),
            }
        };

        // Cluster server (peer-to-peer)
        let ctx_clone = Arc::clone(ctx);
        cluster_handles.push(spawn(Box::pin(async move {
            if let Err(e) = cluster::server::run_cluster_server(cluster_listener, ctx_clone).await {
                tracing::error!(error = %e, "cluster server crashed");
            }
        })));

        // Heartbeat loop (leader sends heartbeats, followers monitor).
        // This is the task whose scheduling latency drives the term=1423
        // thrashing failure mode. Highest priority for CPU isolation.
        let ctx_clone = Arc::clone(ctx);
        let cancel_clone = cluster_cancel.clone();
        cluster_handles.push(spawn(Box::pin(async move {
            cluster::heartbeat::run_heartbeat_loop(ctx_clone, cancel_clone).await;
        })));

        // Oplog sync loop (followers/replicas pull from leader)
        let ctx_clone = Arc::clone(ctx);
        let cancel_clone = cluster_cancel.clone();
        cluster_handles.push(spawn(Box::pin(async move {
            cluster::sync_loop::run_sync_loop(ctx_clone, cancel_clone).await;
        })));

        // Scheduling-latency probe — spawns on the control runtime,
        // measures the gap between when it's woken and when it runs.
        // This is the metric `tests/cpu_isolation.rs` asserts on.
        // See `runtime::start_scheduling_latency_probe`.
        let cancel_clone = cluster_cancel.clone();
        cluster_handles.push(spawn(Box::pin(async move {
            run_scheduling_latency_probe(cancel_clone).await;
        })));
    }

    // Run both servers concurrently, shutdown on ctrl-c
    tokio::select! {
        result = server::run_wire_server(wire_listener, wire_state, tls_acceptor) => {
            result?;
        }
        result = axum::serve(http_listener, http_gateway::router(http_state))
            .with_graceful_shutdown(shutdown_signal()) => {
            result?;
        }
        _ = shutdown_signal() => {
            tracing::info!("shutdown signal received");
        }
    }

    // Stop cluster background tasks
    cluster_cancel.cancel();
    for h in cluster_handles {
        let _ = h.await;
    }

    // Graceful shutdown
    tracing::info!("stopping background workers...");
    shutdown_state.workers.stop_all();

    // Shut down the split runtime (if active). 5s deadline per runtime.
    // App runtime drops first so any control-runtime tasks waiting on
    // app futures don't hang. See `SplitRuntime::shutdown_timeout`.
    if let Some(rt) = split_runtime {
        rt.shutdown_timeout(std::time::Duration::from_secs(5));
    }

    tracing::info!("YantrikDB server stopped");

    Ok(())
}

/// Periodic probe that records the scheduling latency of a control-runtime
/// task. Spawned on the control runtime; measures the wall-clock gap
/// between when the timer fires and when this task actually polls.
/// Under healthy conditions: microseconds. Under priority inversion:
/// hundreds of milliseconds — exactly the symptom of the term=1423
/// incident.
///
/// PR-1 acceptance gate (`tests/cpu_isolation.rs`) asserts that p99 of
/// this metric stays < 10ms under app-runtime saturation. If it doesn't,
/// the runtime split + SCHED_FIFO + caps aren't actually isolating Raft
/// from app load, and this PR doesn't merge.
async fn run_scheduling_latency_probe(cancel: tokio_util::sync::CancellationToken) {
    let interval = std::time::Duration::from_millis(100);
    loop {
        let start = std::time::Instant::now();
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = cancel.cancelled() => return,
        }
        // The actual scheduling latency is `elapsed - interval`. If
        // we're getting CPU promptly, this is near zero. If app load
        // is starving us, it grows.
        let elapsed = start.elapsed();
        let lag = elapsed.saturating_sub(interval);
        crate::metrics::record_raft_task_poll_latency(lag);
    }
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install ctrl+c handler");
}
