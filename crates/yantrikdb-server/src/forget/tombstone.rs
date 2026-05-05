//! `TombstoneIndex` — in-memory index of tombstoned `(tenant_id, rid)`
//! pairs, implementing [`crate::cache::TombstoneProvider`] over the
//! `memory_commit_log` table.
//!
//! See module docs in [`super`] for the design rationale.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use rusqlite::Connection;

use crate::cache::TombstoneProvider;
use crate::commit::{CommitError, CommitObserver, MemoryMutation, TenantId};

/// In-memory index of `(tenant_id, rid)` pairs that have been tombstoned.
///
/// Cheap to clone via `Arc`. Concurrent reads use a `parking_lot::RwLock`
/// so the cache hot path doesn't block on writers — `record` (a single
/// `HashSet::insert` per call) is short enough that contention with
/// readers is negligible.
pub struct TombstoneIndex {
    inner: RwLock<HashMap<TenantId, HashSet<String>>>,
}

impl TombstoneIndex {
    /// Empty index. Useful for tests that don't need hydration.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }

    /// Hydrate the index from the commit log. Scans every
    /// `TombstoneMemory` entry across all tenants and inserts the rids.
    /// `PurgeMemory` entries cancel earlier tombstones (the rid is
    /// physically gone — no further filtering needed).
    ///
    /// Linear in the number of tombstone+purge entries; called once at
    /// startup before any recall traffic is admitted.
    pub fn from_commit_log(conn: &Connection) -> Result<Arc<Self>, CommitError> {
        let mut stmt = conn
            .prepare(
                "SELECT tenant_id, op_kind, payload
                 FROM memory_commit_log
                 WHERE op_kind IN ('TombstoneMemory', 'PurgeMemory')
                 ORDER BY tenant_id ASC, log_index ASC",
            )
            .map_err(|e| CommitError::StorageFailure {
                message: format!("prepare tombstone scan failed: {e}"),
            })?;

        let mut tenants: HashMap<TenantId, HashSet<String>> = HashMap::new();

        let rows = stmt
            .query_map([], |row| {
                let tenant_id_raw: i64 = row.get(0)?;
                let op_kind: String = row.get(1)?;
                let payload: Vec<u8> = row.get(2)?;
                Ok((tenant_id_raw, op_kind, payload))
            })
            .map_err(|e| CommitError::StorageFailure {
                message: format!("tombstone scan query failed: {e}"),
            })?;

        for row_result in rows {
            let (tenant_id_raw, op_kind, payload) =
                row_result.map_err(|e| CommitError::StorageFailure {
                    message: format!("tombstone scan row failed: {e}"),
                })?;
            let tenant_id = TenantId::new(tenant_id_raw);
            let mutation: MemoryMutation =
                serde_json::from_slice(&payload).map_err(|e| CommitError::StorageFailure {
                    message: format!(
                        "tombstone payload deserialize failed for tenant {tenant_id}: {e}"
                    ),
                })?;
            match (op_kind.as_str(), mutation) {
                ("TombstoneMemory", MemoryMutation::TombstoneMemory { rid, .. }) => {
                    tenants.entry(tenant_id).or_default().insert(rid);
                }
                ("PurgeMemory", MemoryMutation::PurgeMemory { rid, .. }) => {
                    if let Some(set) = tenants.get_mut(&tenant_id) {
                        set.remove(&rid);
                    }
                }
                // op_kind column and payload kind disagreed — payload is
                // the source of truth, but we surface this as a hard error
                // because it indicates corruption or grammar drift that
                // would silently skew the index.
                (kind, payload_mut) => {
                    return Err(CommitError::StorageFailure {
                        message: format!(
                            "op_kind/payload mismatch in commit log: column={kind} payload={}",
                            payload_mut.variant_name()
                        ),
                    });
                }
            }
        }

        Ok(Arc::new(Self {
            inner: RwLock::new(tenants),
        }))
    }

    /// Record a freshly-committed tombstone. Called by the committer
    /// after a successful `TombstoneMemory` commit so subsequent reads
    /// see the tombstone immediately.
    pub fn record(&self, tenant_id: TenantId, rid: String) {
        self.inner.write().entry(tenant_id).or_default().insert(rid);
    }

    /// Drop a rid from the index after a `PurgeMemory` commit succeeds.
    /// Idempotent: removing a rid that's not present is a no-op.
    pub fn record_purge(&self, tenant_id: TenantId, rid: &str) {
        if let Some(set) = self.inner.write().get_mut(&tenant_id) {
            set.remove(rid);
        }
    }

    /// Number of tombstoned rids for a tenant. Used by `/metrics`.
    pub fn tenant_size(&self, tenant_id: TenantId) -> usize {
        self.inner
            .read()
            .get(&tenant_id)
            .map(|set| set.len())
            .unwrap_or(0)
    }

    /// Total tombstoned rids across all tenants. Used by `/metrics`.
    pub fn total_size(&self) -> usize {
        self.inner.read().values().map(|set| set.len()).sum()
    }

    /// Number of distinct tenants with at least one tombstone. Used by
    /// `/metrics`.
    pub fn tenant_count(&self) -> usize {
        self.inner.read().len()
    }
}

impl Default for TombstoneIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TombstoneProvider for TombstoneIndex {
    async fn is_tombstoned(&self, tenant_id: TenantId, rid: &str) -> bool {
        self.inner
            .read()
            .get(&tenant_id)
            .map(|set| set.contains(rid))
            .unwrap_or(false)
    }
}

impl CommitObserver for TombstoneIndex {
    /// Hot path on every commit: branch on the variant and update the
    /// in-memory set. Called by `LocalSqliteCommitter` after the SQL
    /// transaction has committed durably.
    fn after_commit(&self, tenant_id: TenantId, mutation: &MemoryMutation) {
        if let Some(rid) = mutation.tombstoned_rid() {
            self.record(tenant_id, rid.to_string());
        }
        if let Some(rid) = mutation.purged_rid() {
            self.record_purge(tenant_id, rid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::{CommitOptions, LocalSqliteCommitter, MutationCommitter, OpId};
    use crate::migrations::MigrationRunner;

    fn tombstone(rid: &str) -> MemoryMutation {
        MemoryMutation::TombstoneMemory {
            rid: rid.into(),
            reason: Some("test".into()),
            requested_at_unix_micros: 0,
        }
    }

    fn upsert(rid: &str) -> MemoryMutation {
        MemoryMutation::UpsertMemory {
            rid: rid.into(),
            text: "x".into(),
            memory_type: "semantic".into(),
            importance: 0.5,
            valence: 0.0,
            half_life: 168.0,
            namespace: "default".into(),
            certainty: 1.0,
            domain: "general".into(),
            source: "user".into(),
            emotional_state: None,
            embedding: None,
            extracted_entities: vec![],
            created_at_unix_micros: None,
            embedding_model: None,
            metadata: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn empty_index_reports_nothing_tombstoned() {
        let idx = TombstoneIndex::new();
        assert!(!idx.is_tombstoned(TenantId::new(1), "mem_a").await);
        assert_eq!(idx.total_size(), 0);
        assert_eq!(idx.tenant_count(), 0);
    }

    #[tokio::test]
    async fn record_then_query_returns_true() {
        let idx = TombstoneIndex::new();
        idx.record(TenantId::new(1), "mem_a".into());
        assert!(idx.is_tombstoned(TenantId::new(1), "mem_a").await);
        assert!(!idx.is_tombstoned(TenantId::new(1), "mem_b").await);
    }

    #[tokio::test]
    async fn tenant_scope_is_respected() {
        // A tombstone in tenant 1 is invisible from tenant 2's view —
        // critical for multi-tenant isolation.
        let idx = TombstoneIndex::new();
        idx.record(TenantId::new(1), "mem_a".into());
        assert!(idx.is_tombstoned(TenantId::new(1), "mem_a").await);
        assert!(!idx.is_tombstoned(TenantId::new(2), "mem_a").await);
    }

    #[tokio::test]
    async fn record_purge_removes_from_index() {
        let idx = TombstoneIndex::new();
        idx.record(TenantId::new(1), "mem_a".into());
        assert!(idx.is_tombstoned(TenantId::new(1), "mem_a").await);
        idx.record_purge(TenantId::new(1), "mem_a");
        assert!(!idx.is_tombstoned(TenantId::new(1), "mem_a").await);
        // Idempotent — removing again is a no-op, not a panic.
        idx.record_purge(TenantId::new(1), "mem_a");
    }

    #[tokio::test]
    async fn any_tombstoned_walks_the_list() {
        let idx = TombstoneIndex::new();
        idx.record(TenantId::new(1), "mem_b".into());
        let rids = vec![
            "mem_a".to_string(),
            "mem_b".to_string(),
            "mem_c".to_string(),
        ];
        assert!(idx.any_tombstoned(TenantId::new(1), &rids).await);
        // None tombstoned → false.
        let clean = vec!["mem_a".to_string(), "mem_c".to_string()];
        assert!(!idx.any_tombstoned(TenantId::new(1), &clean).await);
    }

    #[tokio::test]
    async fn metrics_track_size_and_tenant_count() {
        let idx = TombstoneIndex::new();
        idx.record(TenantId::new(1), "mem_a".into());
        idx.record(TenantId::new(1), "mem_b".into());
        idx.record(TenantId::new(2), "mem_c".into());
        assert_eq!(idx.tenant_size(TenantId::new(1)), 2);
        assert_eq!(idx.tenant_size(TenantId::new(2)), 1);
        assert_eq!(idx.tenant_size(TenantId::new(99)), 0);
        assert_eq!(idx.total_size(), 3);
        assert_eq!(idx.tenant_count(), 2);
    }

    #[tokio::test]
    async fn from_commit_log_hydrates_existing_tombstones() {
        // End-to-end: write a memory through the committer, tombstone it,
        // tear down the committer, rebuild the index from the persistent
        // log, and verify the tombstone is still seen.
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("commit.db");

        let committer = LocalSqliteCommitter::open(&db_path).unwrap();
        let tenant = TenantId::new(1);
        committer
            .commit(tenant, upsert("mem_a"), CommitOptions::default())
            .await
            .unwrap();
        committer
            .commit(tenant, tombstone("mem_a"), CommitOptions::default())
            .await
            .unwrap();
        committer
            .commit(tenant, tombstone("mem_b"), CommitOptions::default())
            .await
            .unwrap();
        // Drop the committer so the WAL is checkpointed.
        drop(committer);

        // Rebuild only the index, from a fresh connection.
        let mut conn = rusqlite::Connection::open(&db_path).unwrap();
        MigrationRunner::run_pending(&mut conn).unwrap();
        let idx = TombstoneIndex::from_commit_log(&conn).unwrap();

        assert!(idx.is_tombstoned(tenant, "mem_a").await);
        assert!(idx.is_tombstoned(tenant, "mem_b").await);
        assert!(!idx.is_tombstoned(tenant, "mem_c").await);
        assert_eq!(idx.tenant_size(tenant), 2);
    }

    #[tokio::test]
    async fn from_commit_log_handles_empty_log() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("commit.db");
        let committer = LocalSqliteCommitter::open(&db_path).unwrap();
        // Commit nothing. Just bring up the DB so migrations run.
        drop(committer);

        let mut conn = rusqlite::Connection::open(&db_path).unwrap();
        MigrationRunner::run_pending(&mut conn).unwrap();
        let idx = TombstoneIndex::from_commit_log(&conn).unwrap();
        assert_eq!(idx.total_size(), 0);
        assert_eq!(idx.tenant_count(), 0);
    }

    #[tokio::test]
    async fn from_commit_log_skips_non_tombstone_entries() {
        // Hydration must NOT pick up Upsert/Update entries — only
        // Tombstone (and Purge for cancellation).
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("commit.db");
        let committer = LocalSqliteCommitter::open(&db_path).unwrap();
        committer
            .commit(TenantId::new(1), upsert("mem_a"), CommitOptions::default())
            .await
            .unwrap();
        committer
            .commit(TenantId::new(1), upsert("mem_b"), CommitOptions::default())
            .await
            .unwrap();
        drop(committer);

        let mut conn = rusqlite::Connection::open(&db_path).unwrap();
        MigrationRunner::run_pending(&mut conn).unwrap();
        let idx = TombstoneIndex::from_commit_log(&conn).unwrap();
        assert_eq!(idx.total_size(), 0);
        assert!(!idx.is_tombstoned(TenantId::new(1), "mem_a").await);
    }

    #[tokio::test]
    async fn from_commit_log_isolates_per_tenant() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("commit.db");
        let committer = LocalSqliteCommitter::open(&db_path).unwrap();
        committer
            .commit(
                TenantId::new(1),
                tombstone("mem_a"),
                CommitOptions::default(),
            )
            .await
            .unwrap();
        committer
            .commit(
                TenantId::new(2),
                tombstone("mem_b"),
                CommitOptions::default(),
            )
            .await
            .unwrap();
        drop(committer);

        let mut conn = rusqlite::Connection::open(&db_path).unwrap();
        MigrationRunner::run_pending(&mut conn).unwrap();
        let idx = TombstoneIndex::from_commit_log(&conn).unwrap();
        assert!(idx.is_tombstoned(TenantId::new(1), "mem_a").await);
        assert!(!idx.is_tombstoned(TenantId::new(1), "mem_b").await);
        assert!(idx.is_tombstoned(TenantId::new(2), "mem_b").await);
        assert!(!idx.is_tombstoned(TenantId::new(2), "mem_a").await);
        assert_eq!(idx.tenant_count(), 2);
    }

    #[tokio::test]
    async fn dyn_dispatch_works() {
        // Index must work behind `Arc<dyn TombstoneProvider>` so
        // TombstoneAwareCache can hold it without knowing the concrete type.
        let idx = TombstoneIndex::new();
        idx.record(TenantId::new(1), "mem_a".into());
        let dyn_provider: Arc<dyn TombstoneProvider> = Arc::new(idx);
        assert!(dyn_provider.is_tombstoned(TenantId::new(1), "mem_a").await);
        assert!(!dyn_provider.is_tombstoned(TenantId::new(1), "mem_b").await);

        // Behind dyn, generic helpers still work.
        let rids = vec!["mem_a".to_string()];
        assert!(dyn_provider.any_tombstoned(TenantId::new(1), &rids).await);
    }

    #[tokio::test]
    async fn purge_after_tombstone_via_log_clears_index() {
        // Hydration should reflect: tombstone then purge → no entry.
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("commit.db");
        let committer = LocalSqliteCommitter::open(&db_path).unwrap();

        // Tombstone then purge by writing the rows directly. PurgeMemory
        // is grammar-level only at this point — the committer apply path
        // will reject it (RFC 011 PR-3 ships the apply path). We bypass
        // by inserting straight via SQL so the hydration logic is what's
        // under test.
        let tenant = TenantId::new(1);
        committer
            .commit(tenant, tombstone("mem_a"), CommitOptions::default())
            .await
            .unwrap();
        drop(committer);

        // Manually insert a PurgeMemory entry so hydration can see the
        // cancellation. This simulates the future RFC 011 PR-3 apply path
        // having committed it.
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let purge = MemoryMutation::PurgeMemory {
            rid: "mem_a".into(),
            purge_epoch: 1,
        };
        let payload = serde_json::to_vec(&purge).unwrap();
        let op_id = OpId::new_random();
        conn.execute(
            "INSERT INTO memory_commit_log (
                tenant_id, log_index, term, op_id, op_kind, payload,
                wire_version_major, wire_version_minor,
                schema_table, schema_version,
                committed_at_unix_micros, applied_at_unix_micros
             ) VALUES (?1, ?2, 0, ?3, 'PurgeMemory', ?4, 1, 0, 'memory_commit_log', 1, 0, NULL)",
            rusqlite::params![tenant.0, 99_i64, op_id.to_string(), payload],
        )
        .unwrap();
        drop(conn);

        // Hydrate and verify Tombstone followed by Purge leaves the rid
        // out of the index.
        let mut conn = rusqlite::Connection::open(&db_path).unwrap();
        MigrationRunner::run_pending(&mut conn).unwrap();
        let idx = TombstoneIndex::from_commit_log(&conn).unwrap();
        assert!(!idx.is_tombstoned(tenant, "mem_a").await);
        assert_eq!(idx.tenant_size(tenant), 0);
    }
}
