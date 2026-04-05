mod auth;
mod command;
mod config;
mod control;
mod embedder;
mod handler;
mod http_gateway;
mod server;
mod tenant_pool;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use clap::{Parser, Subcommand};

use crate::config::ServerConfig;
use crate::control::ControlDb;
use crate::server::AppState;
use crate::tenant_pool::TenantPool;

#[derive(Parser)]
#[command(name = "yantrikdb", about = "YantrikDB — cognitive memory database server")]
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
        Commands::Serve { config: config_path, wire_port, http_port, data_dir } => {
            let mut cfg = match config_path {
                Some(ref path) => ServerConfig::load(path)?,
                None => ServerConfig::default(),
            };

            // CLI overrides
            if let Some(port) = wire_port { cfg.server.wire_port = port; }
            if let Some(port) = http_port { cfg.server.http_port = port; }
            if let Some(dir) = data_dir { cfg.server.data_dir = dir; }

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
                    let db_record = control.get_database(&db)?
                        .ok_or_else(|| anyhow::anyhow!("database '{}' not found", db))?;

                    let token = auth::generate_token();
                    let hash = auth::hash_token(&token);
                    control.create_token(&hash, db_record.id, &label)?;

                    println!("{}", token);
                    eprintln!("token created for database '{}' — save it now, it won't be shown again", db);
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
        config::EmbeddingStrategy::Builtin => {
            Some(embedder::FastEmbedder::new()?)
        }
        config::EmbeddingStrategy::ClientOnly => {
            tracing::info!("embedding strategy: client_only (no server-side embeddings)");
            None
        }
    };

    // Create tenant pool
    let pool = TenantPool::new(&cfg, embedder);

    let state = Arc::new(AppState { control: Mutex::new(control), pool });

    // Start wire protocol server
    let wire_addr = format!("0.0.0.0:{}", cfg.server.wire_port);
    let wire_listener = tokio::net::TcpListener::bind(&wire_addr).await?;

    // Start HTTP gateway
    let http_addr = format!("0.0.0.0:{}", cfg.server.http_port);
    let http_listener = tokio::net::TcpListener::bind(&http_addr).await?;

    tracing::info!(
        wire_port = cfg.server.wire_port,
        http_port = cfg.server.http_port,
        data_dir = %cfg.server.data_dir.display(),
        "YantrikDB server starting"
    );

    let wire_state = Arc::clone(&state);
    let http_state = Arc::clone(&state);

    // Run both servers concurrently
    tokio::select! {
        result = server::run_wire_server(wire_listener, wire_state) => {
            result?;
        }
        result = axum::serve(http_listener, http_gateway::router(http_state)) => {
            result?;
        }
    }

    Ok(())
}
