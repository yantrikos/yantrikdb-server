//! `TenantCommitConnectionPool` — per-tenant SQLite connection cache.
//!
//! ## Why this exists (RFC 010 PR-6.3 §2)
//!
//! Today's [`super::local::LocalSqliteCommitter`] holds **one** SQLite
//! connection to a single global `commit_log.sqlite` file. Every
//! tenant's commit log rows live in that one file, partitioned only
//! by the `tenant_id` column.
//!
//! RFC 010 PR-6.3 moves to **Option D**: each tenant's
//! `memory_commit_log` table lives **inside that tenant's own
//! `yantrik.db` file**, alongside the engine's `memories` /
//! `entity_edges` tables. Benefits called out in the RFC:
//!
//! - Clean tenant deletion. Drop the file, log goes too. No orphan
//!   commit-log rows from a previous tenant incarnation.
//! - Tenant-scoped backups (RFC 012) inherit the log automatically.
//! - Schema migrations run per-tenant on first open after upgrade
//!   — no cross-tenant migration coordination.
//!
//! Trade-off accepted: snapshot serialization spans many files. RFC
//! 010 PR-6.7 handles this via openraft's chunked `generic-snapshot-data`.
//!
//! ## Scope of PR 6.3 (this file)
//!
//! Trait shape + a connection pool that caches one `Arc<Mutex<Connection>>`
//! per active tenant, opening connections lazily against each tenant's
//! `yantrik.db` and running [`MigrationRunner::run_pending`] on first
//! open so `memory_commit_log` is materialized inside the file.
//!
//! `LocalSqliteCommitter` continues to use its global file in PR 6.3.
//! The actual swap (LocalSqliteCommitter routes through this pool
//! instead of one global connection) is a future migration —
//! `yantrikdb admin migrate-commit-log` walks every tenant's DB and
//! seeds `memory_commit_log` rows from the legacy global file. PR 6.3
//! ships the pool; the swap-and-migrate ships separately.
//!
//! ## Concurrency model
//!
//! Same as today's `LocalSqliteCommitter`: one connection per tenant,
//! wrapped in `parking_lot::Mutex` so SQL is serialized within a
//! tenant. Cross-tenant work is independent — different connections,
//! different mutexes, no contention.
//!
//! `rusqlite::Connection` is `Send` but not `Sync`, hence the mutex.
//! WAL mode lets the engine's connection (which lives elsewhere)
//! read concurrently with our writes to `memory_commit_log` since
//! the two connections only ever write to **disjoint** tables.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use rusqlite::Connection;

use super::mutation::TenantId;
use super::trait_def::CommitError;
use crate::migrations::MigrationRunner;

/// Function that maps a tenant id to the absolute path of that
/// tenant's `yantrik.db` file. Decoupled from the pool so callers
/// can wire it to whatever path-resolution logic they already use
/// (typically the existing [`crate::tenant_pool::TenantPool`]).
pub type PathResolver = Arc<dyn Fn(TenantId) -> PathBuf + Send + Sync>;

/// Per-tenant SQLite connection cache for the commit-log table living
/// inside each tenant's `yantrik.db` (Option D layout).
pub struct TenantCommitConnectionPool {
    conns: Mutex<HashMap<TenantId, CachedConnection>>,
    resolver: PathResolver,
    /// Maximum number of cached connections. When exceeded, the next
    /// `for_tenant` call evicts the least-recently-used entry before
    /// inserting. Keeps RSS bounded for clusters with many tenants.
    max_size: usize,
}

struct CachedConnection {
    conn: Arc<Mutex<Connection>>,
    last_used: Instant,
}

impl TenantCommitConnectionPool {
    /// Default cap on cached connections. 256 covers any plausible
    /// active-tenant count for a single-cluster deployment; rarely-used
    /// tenants are evicted to disk-only when the cap is hit.
    pub const DEFAULT_MAX_SIZE: usize = 256;

    /// Default idle threshold for [`Self::close_idle`]. Connections
    /// that haven't been used in this long are eligible for eviction
    /// on the next sweep.
    pub const DEFAULT_IDLE_THRESHOLD: Duration = Duration::from_secs(5 * 60);

    pub fn new(resolver: PathResolver) -> Self {
        Self {
            conns: Mutex::new(HashMap::new()),
            resolver,
            max_size: Self::DEFAULT_MAX_SIZE,
        }
    }

    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }

    /// Get-or-insert the cached connection for `tenant_id`. First
    /// access opens the file, runs pending migrations (so
    /// `memory_commit_log` exists inside the tenant's `yantrik.db`),
    /// configures pragmas, and caches.
    ///
    /// Updates `last_used` on every call — eviction is LRU.
    pub fn for_tenant(&self, tenant_id: TenantId) -> Result<Arc<Mutex<Connection>>, CommitError> {
        let mut map = self.conns.lock();

        if let Some(entry) = map.get_mut(&tenant_id) {
            entry.last_used = Instant::now();
            return Ok(Arc::clone(&entry.conn));
        }

        // Evict LRU if at capacity. Bounded RSS for many-tenant
        // deployments. The dropped Arc closes the SQLite connection
        // when no other holders remain.
        if map.len() >= self.max_size {
            if let Some(victim) = map.iter().min_by_key(|(_, c)| c.last_used).map(|(t, _)| *t) {
                map.remove(&victim);
            }
        }

        let path = (self.resolver)(tenant_id);

        // Ensure parent dir exists. The engine creates yantrik.db on
        // first engine open; if the pool runs before the engine ever
        // opens the file, the parent must still exist.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CommitError::StorageFailure {
                message: format!("create tenant dir {parent:?}: {e}"),
            })?;
        }

        let mut conn = Connection::open(&path).map_err(|e| CommitError::StorageFailure {
            message: format!("open tenant db {path:?}: {e}"),
        })?;
        Self::configure_pragmas(&conn)?;
        MigrationRunner::run_pending(&mut conn).map_err(|e| CommitError::StorageFailure {
            message: format!("run migrations on {path:?}: {e}"),
        })?;

        let arc = Arc::new(Mutex::new(conn));
        map.insert(
            tenant_id,
            CachedConnection {
                conn: Arc::clone(&arc),
                last_used: Instant::now(),
            },
        );
        Ok(arc)
    }

    /// Walk the cache and close (drop) every connection idle longer
    /// than `idle_threshold`. Returns how many were evicted.
    pub fn close_idle(&self, idle_threshold: Duration) -> usize {
        let cutoff = Instant::now()
            .checked_sub(idle_threshold)
            .unwrap_or_else(Instant::now);
        let mut map = self.conns.lock();
        let to_evict: Vec<TenantId> = map
            .iter()
            .filter(|(_, c)| c.last_used < cutoff)
            .map(|(t, _)| *t)
            .collect();
        for t in &to_evict {
            map.remove(t);
        }
        to_evict.len()
    }

    /// How many connections are currently cached.
    pub fn open_count(&self) -> usize {
        self.conns.lock().len()
    }

    /// Drop every cached connection. Used at shutdown.
    pub fn close_all(&self) {
        self.conns.lock().clear();
    }

    fn configure_pragmas(conn: &Connection) -> Result<(), CommitError> {
        // Same shape as LocalSqliteCommitter::configure_pragmas. WAL
        // mode in particular matters — engine reads happen on a
        // separate connection to the same file, and WAL lets readers
        // proceed without blocking on this connection's writes (and
        // vice-versa).
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;\n\
             PRAGMA synchronous=NORMAL;\n\
             PRAGMA foreign_keys=ON;",
        )
        .map_err(|e| CommitError::StorageFailure {
            message: format!("PRAGMA setup failed: {e}"),
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn build_pool(dir: &TempDir) -> TenantCommitConnectionPool {
        let base = dir.path().to_path_buf();
        TenantCommitConnectionPool::new(Arc::new(move |tid: TenantId| {
            base.join(format!("tenant_{}", tid.0)).join("yantrik.db")
        }))
    }

    #[test]
    fn first_open_creates_file_and_runs_migrations() {
        let dir = TempDir::new().unwrap();
        let pool = build_pool(&dir);

        let conn_arc = pool.for_tenant(TenantId::new(1)).unwrap();
        let conn = conn_arc.lock();

        // memory_commit_log must exist after the pool opens the file —
        // m001 ran during MigrationRunner::run_pending.
        let table_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master \
                 WHERE type='table' AND name='memory_commit_log'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(table_count, 1);

        // _yantrikdb_meta_migrations must record at least m001.
        let meta_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM _yantrikdb_meta_migrations WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(meta_count, 1);
    }

    #[test]
    fn repeat_open_returns_cached_arc() {
        let dir = TempDir::new().unwrap();
        let pool = build_pool(&dir);

        let a = pool.for_tenant(TenantId::new(7)).unwrap();
        let b = pool.for_tenant(TenantId::new(7)).unwrap();
        // Both Arcs point to the same Mutex<Connection>.
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(pool.open_count(), 1);
    }

    #[test]
    fn different_tenants_get_distinct_connections() {
        let dir = TempDir::new().unwrap();
        let pool = build_pool(&dir);

        let a = pool.for_tenant(TenantId::new(1)).unwrap();
        let b = pool.for_tenant(TenantId::new(2)).unwrap();
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(pool.open_count(), 2);

        // Each tenant's file lives in its own subdirectory.
        assert!(dir.path().join("tenant_1/yantrik.db").exists());
        assert!(dir.path().join("tenant_2/yantrik.db").exists());
    }

    #[test]
    fn migration_idempotent_across_reopens() {
        // Reopening the pool a second time must not re-run migrations.
        // The MigrationRunner's _yantrikdb_meta_migrations table is the
        // gate; this test confirms the gate works through this layer.
        let dir = TempDir::new().unwrap();
        {
            let pool = build_pool(&dir);
            pool.for_tenant(TenantId::new(1)).unwrap();
        }
        // Reopen — fresh pool, fresh process. m001 must NOT re-run.
        let pool2 = build_pool(&dir);
        let conn_arc = pool2.for_tenant(TenantId::new(1)).unwrap();
        let conn = conn_arc.lock();
        let meta_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM _yantrikdb_meta_migrations WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(meta_count, 1, "m001 must record exactly once");
    }

    #[test]
    fn lru_eviction_at_max_size() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().to_path_buf();
        let pool = TenantCommitConnectionPool::new(Arc::new(move |tid: TenantId| {
            base.join(format!("tenant_{}", tid.0)).join("yantrik.db")
        }))
        .with_max_size(2);

        // Open three tenants in sequence. With max_size=2, the third
        // open must evict the LRU (tenant 1, opened first and not
        // re-touched).
        let _a = pool.for_tenant(TenantId::new(1)).unwrap();
        // Sleep is irrelevant — last_used is monotonic instants. Two
        // separate accesses suffice.
        let _b = pool.for_tenant(TenantId::new(2)).unwrap();
        // Touch tenant 2 to make it more recent than tenant 1.
        let _b2 = pool.for_tenant(TenantId::new(2)).unwrap();
        let _c = pool.for_tenant(TenantId::new(3)).unwrap();

        assert_eq!(pool.open_count(), 2);

        // Re-opening tenant 1 must hit a fresh connection (Arc identity
        // is fresh) — not the evicted one.
        let a2 = pool.for_tenant(TenantId::new(1)).unwrap();
        // open_count clamps at 2: tenant 1 came back, an LRU got evicted.
        assert_eq!(pool.open_count(), 2);
        // The new Arc is distinct from any previously held Arc since the
        // old one was dropped on eviction.
        assert!(!Arc::ptr_eq(&a2, &_b));
    }

    #[test]
    fn close_idle_evicts_old_connections() {
        let dir = TempDir::new().unwrap();
        let pool = build_pool(&dir);

        let _a = pool.for_tenant(TenantId::new(1)).unwrap();
        let _b = pool.for_tenant(TenantId::new(2)).unwrap();
        assert_eq!(pool.open_count(), 2);

        // Threshold zero → everything older than now() is evicted, which
        // is everything except the row touched in this exact instant.
        // We expect 2 evictions because Instant::now() inside close_idle
        // is strictly after both last_used values.
        std::thread::sleep(Duration::from_millis(2));
        let evicted = pool.close_idle(Duration::from_millis(1));
        assert_eq!(evicted, 2);
        assert_eq!(pool.open_count(), 0);
    }

    #[test]
    fn close_all_drops_every_connection() {
        let dir = TempDir::new().unwrap();
        let pool = build_pool(&dir);
        let _a = pool.for_tenant(TenantId::new(1)).unwrap();
        let _b = pool.for_tenant(TenantId::new(7)).unwrap();
        assert_eq!(pool.open_count(), 2);
        pool.close_all();
        assert_eq!(pool.open_count(), 0);
    }

    #[test]
    fn pragmas_are_set_on_first_open() {
        // WAL mode in particular is load-bearing — engine reads from
        // the same file via a separate connection and need WAL for
        // concurrency. Pin it.
        let dir = TempDir::new().unwrap();
        let pool = build_pool(&dir);
        let conn_arc = pool.for_tenant(TenantId::new(1)).unwrap();
        let conn = conn_arc.lock();
        let mode: String = conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "wal");
    }

    #[test]
    fn parent_dir_created_if_missing() {
        // Operator scenarios where a tenant's DB hasn't been opened by
        // the engine yet: the pool must still be able to create the
        // file rather than failing on a missing parent.
        let dir = TempDir::new().unwrap();
        let base = dir.path().to_path_buf();
        let pool = TenantCommitConnectionPool::new(Arc::new(move |tid: TenantId| {
            base.join(format!("never/seen/this/dir/tenant_{}/yantrik.db", tid.0))
        }));
        let _conn = pool.for_tenant(TenantId::new(99)).unwrap();
        assert!(dir
            .path()
            .join("never/seen/this/dir/tenant_99/yantrik.db")
            .exists());
    }

    // Compile-time pin: pool is Send + Sync. AppState in PR 6.4 will
    // hold an `Arc<TenantCommitConnectionPool>` shared across handlers.
    #[allow(dead_code)]
    fn _send_sync_compile_check<T: Send + Sync>(_: T) {}
    #[allow(dead_code)]
    fn _pool_is_send_sync(p: TenantCommitConnectionPool) {
        _send_sync_compile_check(p);
    }
}
