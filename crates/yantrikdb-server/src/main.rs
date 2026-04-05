mod auth;
mod background;
mod command;
mod config;
mod control;
mod embedder;
mod handler;
mod http_gateway;
mod server;
mod tenant_pool;
mod tls;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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
                        println!("{:<6} {:<20} {}", "ID", "NAME", "CREATED");
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
            let engine = yantrikdb::YantrikDB::new(db_path.to_str().unwrap_or("yantrik.db"), 384)?;

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
            let mut engine =
                yantrikdb::YantrikDB::new(db_path.to_str().unwrap_or("yantrik.db"), 384)?;

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
    }
}

async fn run_server(cfg: ServerConfig) -> anyhow::Result<()> {
    // Ensure data directory exists
    std::fs::create_dir_all(&cfg.server.data_dir)?;

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

    // Create tenant pool and background worker registry
    let pool = TenantPool::new(&cfg, embedder);
    let workers = background::WorkerRegistry::new(&cfg.background);

    let state = Arc::new(AppState {
        control: Mutex::new(control),
        pool,
        workers,
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
