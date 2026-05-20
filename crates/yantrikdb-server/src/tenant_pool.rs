//! Tenant engine pool — lazy-load YantrikDB instances per database.
//!
//! Each tenant gets an isolated YantrikDB engine backed by its own SQLite file.
//! Engines are cached in memory and shared across connections to the same database.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yantrikdb::engine::materializer::{
    recommended_worker_count, spawn_all_workers, AllWorkerGuards,
};
use yantrikdb::YantrikDB;

use crate::commit::{ApplyError, EngineResolver, TenantId};
use crate::config::ServerConfig;
use crate::control::{ControlDb, DatabaseRecord};
use crate::embedder::ServerEmbedder;

pub struct TenantPool {
    engines: Mutex<HashMap<i64, Arc<YantrikDB>>>,
    /// Per-engine background worker guards (materializer + compactor).
    /// Held for the lifetime of the engine; dropped when the engine is
    /// evicted from the pool, which signals workers to shut down.
    /// Without these the engine's delta tier never compacts, and writes
    /// wedge at `delta_max` (default 256) — see v0.7.18 release notes.
    worker_guards: Mutex<HashMap<i64, AllWorkerGuards>>,
    data_dir: PathBuf,
    embedding_dim: usize,
    embedder: Option<ServerEmbedder>,
    master_key: Option<[u8; 32]>,
}

impl TenantPool {
    pub fn new(
        config: &ServerConfig,
        embedder: Option<ServerEmbedder>,
        master_key: Option<[u8; 32]>,
    ) -> Self {
        Self {
            engines: Mutex::new(HashMap::new()),
            worker_guards: Mutex::new(HashMap::new()),
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
    pub fn get_engine(&self, db_record: &DatabaseRecord) -> anyhow::Result<Arc<YantrikDB>> {
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

        // v0.8.9: drop the server-side Mutex<YantrikDB>. YantrikDB is
        // Send+Sync (asserted in engine library); all top-level methods
        // take &self with internal locks. The outer Mutex was dead
        // serialization that prevented concurrent recall — a single AGI
        // could clog a CPU core. Now: Arc<YantrikDB> direct, recalls
        // parallelize through engine's read connection pool.
        let engine = Arc::new(engine);
        engines.insert(db_record.id, Arc::clone(&engine));

        // v0.7.18: spawn the engine's materializer + compactor workers.
        // Without these the delta tier never drains and writes wedge at
        // delta_max (default 256). Bundle returns a guard whose Drop
        // signals shutdown — store under db_id with engine-lifetime.
        let guard = spawn_all_workers(&engine, recommended_worker_count());
        self.worker_guards.lock().insert(db_record.id, guard);

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
        // Drop the worker guard alongside the engine — Drop signals the
        // materializer + compactor to shut down for this tenant.
        self.worker_guards.lock().remove(&db_id);
    }

    /// Number of loaded engines.
    pub fn loaded_count(&self) -> usize {
        self.engines.lock().len()
    }

    /// Borrow the configured embedder, if any. The HTTP layer uses this
    /// to pre-embed `/v1/remember` payloads before delegating to the
    /// engine — see issue #19. Pre-embedding lets us fail-fast when
    /// the model service hiccups instead of silently writing a row
    /// with `embedding=NULL` that then poisons the namespace's
    /// similarity-recall path.
    pub fn embedder(&self) -> Option<&ServerEmbedder> {
        self.embedder.as_ref()
    }

    /// Get the data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

/// RFC 010 PR-6.4 — Resolves `TenantId` to `Arc<YantrikDB>` for
/// [`crate::commit::EngineApplier`].
///
/// The Raft state machine carries `TenantId` (i64 = control-DB primary
/// key) alongside every committed mutation; the applier needs the engine
/// for that tenant to write the mutation into engine state. This adapter
/// looks the tenant up in the control DB on cache miss, then hands off
/// to [`TenantPool::get_engine`] which lazy-loads the engine if needed.
///
/// Resolution failures (control-DB lookup error, tenant not found,
/// engine open failure) surface as [`ApplyError::EngineFailure`] —
/// catastrophic at apply time because the entry is already durable in
/// the log. The state machine treats these as divergence risk.
pub struct TenantPoolEngineResolver {
    pool: Arc<TenantPool>,
    control: Arc<Mutex<ControlDb>>,
}

impl TenantPoolEngineResolver {
    pub fn new(pool: Arc<TenantPool>, control: Arc<Mutex<ControlDb>>) -> Self {
        Self { pool, control }
    }
}

impl EngineResolver for TenantPoolEngineResolver {
    fn resolve(&self, tenant_id: TenantId) -> Result<Arc<YantrikDB>, ApplyError> {
        let id = tenant_id.0;
        let db_record = {
            let control = self.control.lock();
            control
                .get_database_by_id(id)
                .map_err(|e| ApplyError::EngineFailure {
                    message: format!("control DB lookup tenant_id={id}: {e}"),
                })?
                .ok_or_else(|| ApplyError::EngineFailure {
                    message: format!(
                        "tenant_id={id} not found in control DB — followers must \
                         replicate the database row before applying mutations against \
                         it (RFC 010 PR-6 control-plane replication contract)"
                    ),
                })?
        };
        self.pool
            .get_engine(&db_record)
            .map_err(|e| ApplyError::EngineFailure {
                message: format!("tenant_pool.get_engine(tenant_id={id}): {e}"),
            })
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
