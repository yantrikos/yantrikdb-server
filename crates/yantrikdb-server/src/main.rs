mod auth;
mod background;
mod cluster;
mod command;
mod config;
mod control;
mod embedder;
mod handler;
mod http_gateway;
mod server;
mod tenant_pool;
mod tls;

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "yantrikdb_server=info".into()),
        )
        .init();

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
                    let url = format!("{}/v1/cluster", url.trim_end_matches('/'));
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
                    println!("{}", serde_json::to_string_pretty(&value)?);
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
            }
        }

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

async fn run_server(cfg: ServerConfig) -> anyhow::Result<()> {
    // Ensure data directory exists
    std::fs::create_dir_all(&cfg.server.data_dir)?;

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

    // Resolve master encryption key (auto-generates if needed)
    let master_key = cfg.encryption.resolve_key(&cfg.server.data_dir)?;
    if master_key.is_some() {
        tracing::info!("encryption: enabled (AES-256-GCM)");
    } else {
        tracing::warn!("encryption: disabled — set [encryption] to enable at-rest encryption");
    }

    // Create tenant pool and background worker registry
    let pool = Arc::new(TenantPool::new(&cfg, embedder, master_key));
    let workers = background::WorkerRegistry::new(&cfg.background);
    let control = Arc::new(Mutex::new(control));

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

    let state = Arc::new(AppState {
        control,
        pool,
        workers,
        cluster: cluster_ctx.clone(),
    });

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

    // Spawn cluster server + background loops if clustered
    let mut cluster_handles = Vec::new();
    if let Some(ref ctx) = cluster_ctx {
        let cluster_addr = format!("0.0.0.0:{}", cfg.cluster.cluster_port);
        let cluster_listener = tokio::net::TcpListener::bind(&cluster_addr).await?;
        tracing::info!(
            cluster_port = cfg.cluster.cluster_port,
            "cluster wire server starting"
        );

        // Cluster server (peer-to-peer)
        let ctx_clone = Arc::clone(ctx);
        cluster_handles.push(tokio::spawn(async move {
            if let Err(e) = cluster::server::run_cluster_server(cluster_listener, ctx_clone).await {
                tracing::error!(error = %e, "cluster server crashed");
            }
        }));

        // Heartbeat loop (leader sends heartbeats, followers monitor)
        let ctx_clone = Arc::clone(ctx);
        let cancel_clone = cluster_cancel.clone();
        cluster_handles.push(tokio::spawn(async move {
            cluster::heartbeat::run_heartbeat_loop(ctx_clone, cancel_clone).await;
        }));

        // Oplog sync loop (followers/replicas pull from leader)
        let ctx_clone = Arc::clone(ctx);
        let cancel_clone = cluster_cancel.clone();
        cluster_handles.push(tokio::spawn(async move {
            cluster::sync_loop::run_sync_loop(ctx_clone, cancel_clone).await;
        }));
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
    tracing::info!("YantrikDB server stopped");

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install ctrl+c handler");
}
