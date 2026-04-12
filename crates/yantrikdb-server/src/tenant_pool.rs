//! Tenant engine pool — lazy-load YantrikDB instances per database.
//!
//! Each tenant gets an isolated YantrikDB engine backed by its own SQLite file.
//! Engines are cached in memory and shared across connections to the same database.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yantrikdb::YantrikDB;

use crate::config::ServerConfig;
use crate::control::{ControlDb, DatabaseRecord};
use crate::embedder::FastEmbedder;

pub struct TenantPool {
    engines: Mutex<HashMap<i64, Arc<Mutex<YantrikDB>>>>,
    data_dir: PathBuf,
    embedding_dim: usize,
    embedder: Option<FastEmbedder>,
    master_key: Option<[u8; 32]>,
}

impl TenantPool {
    pub fn new(
        config: &ServerConfig,
        embedder: Option<FastEmbedder>,
        master_key: Option<[u8; 32]>,
    ) -> Self {
        Self {
            engines: Mutex::new(HashMap::new()),
            data_dir: config.server.data_dir.clone(),
            embedding_dim: config.embedding.dim,
            embedder,
            master_key,
        }
    }

    /// Whether encryption is enabled for engines created by this pool.
    ///
    /// Not currently called — reserved for /v1/admin/status surfacing of
    /// encryption state and for startup diagnostics.
    #[allow(dead_code)]
    pub fn is_encrypted(&self) -> bool {
        self.master_key.is_some()
    }

    /// Get or create an engine for the given database.
    pub fn get_engine(&self, db_record: &DatabaseRecord) -> anyhow::Result<Arc<Mutex<YantrikDB>>> {
        let mut engines = self.engines.lock();

        if let Some(engine) = engines.get(&db_record.id) {
            return Ok(Arc::clone(engine));
        }

        // Create the database directory if needed
        let db_dir = self.data_dir.join(&db_record.path);
        std::fs::create_dir_all(&db_dir)?;

        let db_path = db_dir.join("yantrik.db");
        let mut engine = if let Some(ref key) = self.master_key {
            YantrikDB::new_encrypted(
                db_path.to_str().unwrap_or("yantrik.db"),
                self.embedding_dim,
                key,
            )?
        } else {
            YantrikDB::new(db_path.to_str().unwrap_or("yantrik.db"), self.embedding_dim)?
        };

        // Set the shared embedder if available
        if let Some(ref emb) = self.embedder {
            engine.set_embedder(emb.boxed());
        }

        let engine = Arc::new(Mutex::new(engine));
        engines.insert(db_record.id, Arc::clone(&engine));

        tracing::info!(db_name = %db_record.name, db_id = db_record.id, "loaded engine");

        Ok(engine)
    }

    /// Remove an engine from the pool (e.g. on database drop).
    ///
    /// Not currently called — reserved for the planned /v1/admin/drop
    /// endpoint which tears down a tenant cleanly.
    #[allow(dead_code)]
    pub fn evict(&self, db_id: i64) {
        let mut engines = self.engines.lock();
        engines.remove(&db_id);
    }

    /// Number of loaded engines.
    pub fn loaded_count(&self) -> usize {
        self.engines.lock().len()
    }

    /// Get the data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

/// Ensure a "default" database exists in control.db and return its record.
pub fn ensure_default_database(
    control: &ControlDb,
    data_dir: &Path,
) -> anyhow::Result<DatabaseRecord> {
    if let Some(db) = control.get_database("default")? {
        return Ok(db);
    }

    let path = "default";
    let db_dir = data_dir.join(path);
    std::fs::create_dir_all(&db_dir)?;

    let id = control.create_database("default", path)?;
    Ok(DatabaseRecord {
        id,
        name: "default".into(),
        path: path.into(),
        created_at: String::new(),
    })
}
