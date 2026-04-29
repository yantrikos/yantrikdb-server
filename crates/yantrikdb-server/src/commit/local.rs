//! `LocalSqliteCommitter` — single-node MutationCommitter, SQLite-backed.
//!
//! ## What changed in PR-2 (this PR)
//!
//! PR-1 backed the committer with in-memory `Vec<CommittedEntry>`. PR-2
//! replaces that with persistent SQLite storage in the `memory_commit_log`
//! table (created by migration m001). The trait surface, semantics, and
//! every PR-1 test still apply unchanged — this is the value of having
//! defined the trait first.
//!
//! ## Concurrency model
//!
//! `rusqlite::Connection` is `Send` but not `Sync`. We wrap it in
//! `Arc<parking_lot::Mutex<Connection>>` and serialize all SQL through
//! that mutex. Per-tenant log writes are intrinsically serial (each
//! commit needs the next log_index for its tenant) so this is the right
//! shape — there's no per-tenant gain to be had from parallelism that
//! doesn't compromise log_index monotonicity.
//!
//! ## Async wrapping
//!
//! All SQL work happens inside `tokio::task::spawn_blocking` so the
//! tokio runtime worker isn't blocked on disk IO. The async_trait
//! method body acquires the connection mutex inside the blocking task,
//! does its SQL, and returns the result.
//!
//! ## File path vs in-memory
//!
//! - [`LocalSqliteCommitter::open`] takes a file path (production).
//! - [`LocalSqliteCommitter::open_in_memory`] uses `:memory:` for tests.
//!
//! Both run migrations at construction so the table layout is ready.
//!
//! ## Replay safety
//!
//! On restart, `high_watermark` walks the table to find max(log_index)
//! per tenant. The `(tenant_id, op_id)` UNIQUE index enforces
//! idempotency at the DB level — a duplicate insert with same op_id
//! returns a constraint violation, which the committer catches and
//! converts into the appropriate `CommitError::OpIdCollision` /
//! existing-receipt response.

use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};

use super::mutation::{MemoryMutation, OpId, TenantId};
use super::trait_def::{
    CommitError, CommitOptions, CommitReceipt, CommittedEntry, MutationCommitter,
};
use crate::migrations::MigrationRunner;
use crate::version::VersionedEvent;

/// Single-node SQLite-backed committer.
pub struct LocalSqliteCommitter {
    conn: Arc<Mutex<Connection>>,
    /// Optional observers notified after every successful commit. Used
    /// by [`crate::forget::TombstoneIndex`] (RFC 011-B) to stay in
    /// lock-step with the durable log; held as `Arc<dyn CommitObserver>`
    /// so the commit layer doesn't take a hard dependency on each
    /// consumer's concrete type.
    observers: Vec<Arc<dyn super::CommitObserver>>,
}

impl LocalSqliteCommitter {
    /// Open or create a SQLite database at the given path. Runs any
    /// pending migrations before returning. The connection is configured
    /// in WAL mode for better concurrency under read load.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CommitError> {
        let mut conn = Connection::open(path).map_err(|e| CommitError::StorageFailure {
            message: format!("failed to open SQLite: {e}"),
        })?;
        Self::configure_pragmas(&conn)?;
        Self::run_migrations(&mut conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            observers: Vec::new(),
        })
    }

    /// Open an in-memory SQLite for tests. Migrations run as usual.
    pub fn open_in_memory() -> Result<Self, CommitError> {
        let mut conn =
            Connection::open_in_memory().map_err(|e| CommitError::StorageFailure {
                message: format!("failed to open in-memory SQLite: {e}"),
            })?;
        Self::configure_pragmas(&conn)?;
        Self::run_migrations(&mut conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            observers: Vec::new(),
        })
    }

    /// Attach a [`super::CommitObserver`] so its `after_commit` hook
    /// fires after every successful commit. Builder-style. Used by
    /// [`crate::forget::TombstoneIndex`] (RFC 011-B) to stay in
    /// lock-step with the durable log.
    pub fn with_observer(mut self, observer: Arc<dyn super::CommitObserver>) -> Self {
        self.observers.push(observer);
        self
    }

    fn configure_pragmas(conn: &Connection) -> Result<(), CommitError> {
        // WAL mode: better read concurrency. Synchronous=NORMAL: durability
        // strong enough for our write path (we replicate via the commit
        // log; even a power loss between fsyncs is recoverable from peers).
        // foreign_keys=ON: enforce referential integrity if any future
        // schema adds FKs.
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

    fn run_migrations(conn: &mut Connection) -> Result<(), CommitError> {
        MigrationRunner::run_pending(conn).map_err(|e| CommitError::StorageFailure {
            message: format!("migration runner failed: {e}"),
        })?;
        Ok(())
    }

    /// Return the total number of entries across all tenants. Used by
    /// /metrics + tests. Goes through SQL.
    pub fn total_entries(&self) -> Result<u64, CommitError> {
        let conn = self.conn.lock();
        conn.query_row("SELECT COUNT(*) FROM memory_commit_log", [], |row| {
            row.get::<_, i64>(0)
        })
        .map(|n| n as u64)
        .map_err(|e| CommitError::StorageFailure {
            message: format!("count query failed: {e}"),
        })
    }

    /// Number of distinct tenants with at least one entry. Used by /metrics.
    pub fn active_tenant_count(&self) -> Result<u64, CommitError> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT COUNT(DISTINCT tenant_id) FROM memory_commit_log",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|n| n as u64)
        .map_err(|e| CommitError::StorageFailure {
            message: format!("tenant count query failed: {e}"),
        })
    }

    /// Look up a previously committed entry by op_id (for idempotency).
    /// Returns None if no such entry exists.
    fn lookup_by_op_id(
        conn: &Connection,
        tenant_id: TenantId,
        op_id: OpId,
    ) -> Result<Option<CommittedEntry>, CommitError> {
        let op_id_str = op_id.to_string();
        let row = conn
            .query_row(
                "SELECT log_index, term, op_kind, payload, committed_at_unix_micros, applied_at_unix_micros
                 FROM memory_commit_log
                 WHERE tenant_id = ?1 AND op_id = ?2",
                params![tenant_id.0, op_id_str],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)? as u64,
                        row.get::<_, i64>(1)? as u64,
                        row.get::<_, String>(2)?,
                        row.get::<_, Vec<u8>>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, Option<i64>>(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| CommitError::StorageFailure {
                message: format!("op_id lookup failed: {e}"),
            })?;
        let (log_index, term, _op_kind, payload, committed_micros, applied_micros) = match row {
            Some(r) => r,
            None => return Ok(None),
        };
        let mutation: MemoryMutation =
            serde_json::from_slice(&payload).map_err(|e| CommitError::StorageFailure {
                message: format!("payload deserialize failed for op_id {op_id}: {e}"),
            })?;
        Ok(Some(CommittedEntry {
            op_id,
            tenant_id,
            term,
            log_index,
            mutation,
            committed_at: micros_to_systime(committed_micros),
            applied_at: applied_micros.map(micros_to_systime),
        }))
    }

    /// Compute the next log_index for a tenant. Used for the CAS check.
    fn next_log_index(conn: &Connection, tenant_id: TenantId) -> Result<u64, CommitError> {
        let max: Option<i64> = conn
            .query_row(
                "SELECT MAX(log_index) FROM memory_commit_log WHERE tenant_id = ?1",
                params![tenant_id.0],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| CommitError::StorageFailure {
                message: format!("max log_index query failed: {e}"),
            })?
            .flatten();
        Ok(max.map(|n| (n + 1) as u64).unwrap_or(1))
    }
}

#[async_trait]
impl MutationCommitter for LocalSqliteCommitter {
    async fn commit(
        &self,
        tenant_id: TenantId,
        mutation: MemoryMutation,
        opts: CommitOptions,
    ) -> Result<CommitReceipt, CommitError> {
        // Reject grammar-only variants up front. Same as PR-1 — log
        // grammar accepts them but apply path doesn't yet.
        if !mutation.is_implemented() {
            return Err(CommitError::NotYetImplemented {
                variant: mutation.variant_name(),
                planned_rfc: mutation.planned_rfc(),
            });
        }

        let conn = Arc::clone(&self.conn);
        let mutation_for_blocking = mutation.clone();
        let opts_for_blocking = opts;
        let observers = self.observers.clone();

        // SQL work happens inside spawn_blocking so the tokio runtime
        // doesn't block on disk IO.
        let result = tokio::task::spawn_blocking(move || -> Result<CommitReceipt, CommitError> {
            let mut conn = conn.lock();

            // Client-provided op_id (RFC 010 PR-3) wins; fall back to a
            // fresh UUIDv7 for server-internal commits.
            let op_id = opts_for_blocking.op_id.unwrap_or_else(OpId::new_random);

            // Idempotency check. Client-provided op_ids exercise this
            // path routinely (HTTP retries, network blips).
            if let Some(existing) = Self::lookup_by_op_id(&conn, tenant_id, op_id)? {
                if existing.mutation == mutation_for_blocking {
                    return Ok(CommitReceipt {
                        op_id: existing.op_id,
                        tenant_id: existing.tenant_id,
                        term: existing.term,
                        log_index: existing.log_index,
                        committed_at: existing.committed_at,
                        applied_at: existing.applied_at,
                    });
                } else {
                    return Err(CommitError::OpIdCollision {
                        op_id,
                        tenant_id,
                        existing_index: existing.log_index,
                    });
                }
            }

            // Compare-and-swap: if expected_log_index is set, verify it.
            let next_index = Self::next_log_index(&conn, tenant_id)?;
            if let Some(expected) = opts_for_blocking.expected_log_index {
                if expected != next_index {
                    return Err(CommitError::UnexpectedLogIndex {
                        tenant_id,
                        expected,
                        actual: next_index,
                    });
                }
            }

            let now = SystemTime::now();
            let now_micros = systime_to_micros(now);
            let applied_micros = if opts_for_blocking.wait_for_apply {
                Some(now_micros)
            } else {
                None
            };

            let wire = mutation_for_blocking.wire_version();
            let (schema_table, schema_version_int) = match mutation_for_blocking.schema_version() {
                Some((tbl, ver)) => (Some(tbl.to_string()), Some(u32::from(ver) as i64)),
                None => (None, None),
            };
            let payload = serde_json::to_vec(&mutation_for_blocking).map_err(|e| {
                CommitError::StorageFailure {
                    message: format!("payload serialize failed: {e}"),
                }
            })?;

            // INSERT inside an explicit transaction so a concurrent reader
            // never sees a half-written row + so we can roll back cleanly
            // if the UNIQUE op_id index trips.
            let tx = conn.transaction().map_err(|e| CommitError::StorageFailure {
                message: format!("begin transaction failed: {e}"),
            })?;
            let inserted = tx.execute(
                "INSERT INTO memory_commit_log (
                    tenant_id, log_index, term,
                    op_id, op_kind, payload,
                    wire_version_major, wire_version_minor,
                    schema_table, schema_version,
                    committed_at_unix_micros, applied_at_unix_micros
                 ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
                params![
                    tenant_id.0,
                    next_index as i64,
                    0_i64, // term: 0 in PR-2 (no Raft yet); RaftCommitter (PR-4) sets real terms
                    op_id.to_string(),
                    mutation_for_blocking.variant_name(),
                    payload,
                    wire.major as i64,
                    wire.minor as i64,
                    schema_table,
                    schema_version_int,
                    now_micros,
                    applied_micros,
                ],
            );
            match inserted {
                Ok(_) => {}
                Err(e) => {
                    // Detect UNIQUE constraint violation on op_id (race
                    // between two commits picking the same op_id —
                    // astronomically unlikely with UUIDv7 but possible
                    // on a cosmic-ray day).
                    let msg = e.to_string();
                    if msg.contains("UNIQUE") {
                        return Err(CommitError::OpIdCollision {
                            op_id,
                            tenant_id,
                            existing_index: 0, // unknown without re-query; not critical
                        });
                    }
                    return Err(CommitError::StorageFailure {
                        message: format!("INSERT failed: {e}"),
                    });
                }
            }
            tx.commit().map_err(|e| CommitError::StorageFailure {
                message: format!("commit transaction failed: {e}"),
            })?;

            // Round-trip the timestamp through micros_to_systime so the
            // receipt matches what a later `lookup_by_op_id` (or any
            // read_range) returns. SQLite stores at microsecond
            // precision; without this rounding, a client retry would
            // see a slightly-different `committed_at` from the original
            // call — confusing for replay-comparison tests and for any
            // downstream system that hashes the receipt.
            Ok(CommitReceipt {
                op_id,
                tenant_id,
                term: 0,
                log_index: next_index,
                committed_at: micros_to_systime(now_micros),
                applied_at: applied_micros.map(micros_to_systime),
            })
        })
        .await
        .map_err(|e| CommitError::StorageFailure {
            message: format!("spawn_blocking join failed: {e}"),
        })??;

        // RFC 011-B: notify observers after the SQL commit succeeds so a
        // failed commit never poisons in-memory derived state. Cheap and
        // synchronous — heavy work belongs on the InvalidationBus.
        for observer in &observers {
            observer.after_commit(tenant_id, &mutation);
        }

        Ok(result)
    }

    async fn read_range(
        &self,
        tenant_id: TenantId,
        from_index: u64,
        limit: usize,
    ) -> Result<Vec<CommittedEntry>, CommitError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<Vec<CommittedEntry>, CommitError> {
            let conn = conn.lock();
            let mut stmt = conn
                .prepare(
                    "SELECT log_index, term, op_id, op_kind, payload,
                            committed_at_unix_micros, applied_at_unix_micros
                     FROM memory_commit_log
                     WHERE tenant_id = ?1 AND log_index >= ?2
                     ORDER BY log_index ASC
                     LIMIT ?3",
                )
                .map_err(|e| CommitError::StorageFailure {
                    message: format!("prepare read_range failed: {e}"),
                })?;
            let from_index_clamped = from_index.max(1);
            let rows = stmt
                .query_map(
                    params![tenant_id.0, from_index_clamped as i64, limit as i64],
                    |row| {
                        let log_index = row.get::<_, i64>(0)? as u64;
                        let term = row.get::<_, i64>(1)? as u64;
                        let op_id_str: String = row.get(2)?;
                        let _op_kind: String = row.get(3)?;
                        let payload: Vec<u8> = row.get(4)?;
                        let committed_micros: i64 = row.get(5)?;
                        let applied_micros: Option<i64> = row.get(6)?;
                        Ok((log_index, term, op_id_str, payload, committed_micros, applied_micros))
                    },
                )
                .map_err(|e| CommitError::StorageFailure {
                    message: format!("read_range query failed: {e}"),
                })?;
            let mut out = Vec::with_capacity(limit);
            for row_result in rows {
                let (log_index, term, op_id_str, payload, committed_micros, applied_micros) =
                    row_result.map_err(|e| CommitError::StorageFailure {
                        message: format!("read_range row failed: {e}"),
                    })?;
                let op_id_uuid =
                    op_id_str
                        .parse::<uuid7::Uuid>()
                        .map_err(|e| CommitError::StorageFailure {
                            message: format!("op_id parse failed: {e}"),
                        })?;
                let mutation: MemoryMutation = serde_json::from_slice(&payload).map_err(|e| {
                    CommitError::StorageFailure {
                        message: format!(
                            "payload deserialize failed at log_index {log_index}: {e}"
                        ),
                    }
                })?;
                out.push(CommittedEntry {
                    op_id: OpId::from_uuid(op_id_uuid),
                    tenant_id,
                    term,
                    log_index,
                    mutation,
                    committed_at: micros_to_systime(committed_micros),
                    applied_at: applied_micros.map(micros_to_systime),
                });
            }
            Ok(out)
        })
        .await
        .map_err(|e| CommitError::StorageFailure {
            message: format!("spawn_blocking join failed: {e}"),
        })?
    }

    async fn high_watermark(&self, tenant_id: TenantId) -> Result<u64, CommitError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<u64, CommitError> {
            let conn = conn.lock();
            let max: Option<i64> = conn
                .query_row(
                    "SELECT MAX(log_index) FROM memory_commit_log WHERE tenant_id = ?1",
                    params![tenant_id.0],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| CommitError::StorageFailure {
                    message: format!("high_watermark query failed: {e}"),
                })?
                .flatten();
            Ok(max.map(|n| n as u64).unwrap_or(0))
        })
        .await
        .map_err(|e| CommitError::StorageFailure {
            message: format!("spawn_blocking join failed: {e}"),
        })?
    }

    async fn list_active_tenants(&self) -> Result<Vec<TenantId>, CommitError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<Vec<TenantId>, CommitError> {
            let conn = conn.lock();
            let mut stmt = conn
                .prepare(
                    "SELECT DISTINCT tenant_id FROM memory_commit_log ORDER BY tenant_id ASC",
                )
                .map_err(|e| CommitError::StorageFailure {
                    message: format!("list_active_tenants prepare failed: {e}"),
                })?;
            let rows = stmt
                .query_map([], |row| row.get::<_, i64>(0))
                .map_err(|e| CommitError::StorageFailure {
                    message: format!("list_active_tenants query failed: {e}"),
                })?;
            let mut out = Vec::new();
            for row in rows {
                let id = row.map_err(|e| CommitError::StorageFailure {
                    message: format!("list_active_tenants row failed: {e}"),
                })?;
                out.push(TenantId::new(id));
            }
            Ok(out)
        })
        .await
        .map_err(|e| CommitError::StorageFailure {
            message: format!("spawn_blocking join failed: {e}"),
        })?
    }
}

fn systime_to_micros(t: SystemTime) -> i64 {
    t.duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

fn micros_to_systime(micros: i64) -> SystemTime {
    if micros < 0 {
        std::time::UNIX_EPOCH
    } else {
        std::time::UNIX_EPOCH + std::time::Duration::from_micros(micros as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upsert(text: &str) -> MemoryMutation {
        MemoryMutation::UpsertMemory {
            rid: format!("mem_{}", text),
            text: text.into(),
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
            metadata: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn first_commit_assigns_log_index_1() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let r = c
            .commit(TenantId::new(1), upsert("hello"), CommitOptions::new())
            .await
            .unwrap();
        assert_eq!(r.log_index, 1);
        assert_eq!(r.term, 0);
        assert_eq!(r.tenant_id, TenantId::new(1));
        assert!(r.applied_at.is_some());
    }

    #[tokio::test]
    async fn log_index_is_monotonic_per_tenant() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let t = TenantId::new(1);
        for i in 1..=5 {
            let r = c
                .commit(t, upsert(&format!("msg{}", i)), CommitOptions::new())
                .await
                .unwrap();
            assert_eq!(r.log_index, i);
        }
    }

    #[tokio::test]
    async fn log_indices_are_per_tenant_independent() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let r1 = c
            .commit(TenantId::new(1), upsert("a"), CommitOptions::new())
            .await
            .unwrap();
        let r2 = c
            .commit(TenantId::new(1), upsert("b"), CommitOptions::new())
            .await
            .unwrap();
        let r3 = c
            .commit(TenantId::new(2), upsert("c"), CommitOptions::new())
            .await
            .unwrap();
        assert_eq!(r1.log_index, 1);
        assert_eq!(r2.log_index, 2);
        assert_eq!(r3.log_index, 1);
    }

    #[tokio::test]
    async fn high_watermark_tracks_last_committed() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let t = TenantId::new(1);
        assert_eq!(c.high_watermark(t).await.unwrap(), 0);
        c.commit(t, upsert("a"), CommitOptions::new()).await.unwrap();
        assert_eq!(c.high_watermark(t).await.unwrap(), 1);
        c.commit(t, upsert("b"), CommitOptions::new()).await.unwrap();
        assert_eq!(c.high_watermark(t).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn read_range_returns_empty_for_unknown_tenant() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let entries = c.read_range(TenantId::new(999), 0, 10).await.unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn read_range_respects_from_and_limit() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let t = TenantId::new(1);
        for i in 1..=10 {
            c.commit(t, upsert(&format!("m{}", i)), CommitOptions::new())
                .await
                .unwrap();
        }
        let entries = c.read_range(t, 3, 4).await.unwrap();
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].log_index, 3);
        assert_eq!(entries[3].log_index, 6);
    }

    #[tokio::test]
    async fn read_range_clamps_at_end() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let t = TenantId::new(1);
        for i in 1..=3 {
            c.commit(t, upsert(&format!("m{}", i)), CommitOptions::new())
                .await
                .unwrap();
        }
        let entries = c.read_range(t, 1, 100).await.unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[tokio::test]
    async fn read_range_from_beyond_end_returns_empty() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let t = TenantId::new(1);
        c.commit(t, upsert("a"), CommitOptions::new()).await.unwrap();
        let entries = c.read_range(t, 100, 10).await.unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn unimplemented_variant_returns_not_yet_implemented() {
        // TombstoneMemory implemented in RFC 011-B. Use TenantConfigPatch
        // (RFC 021) which remains grammar-only until that PR ships.
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let cfg = MemoryMutation::TenantConfigPatch {
            key: "k".into(),
            value: serde_json::Value::Null,
        };
        let err = c
            .commit(TenantId::new(1), cfg, CommitOptions::new())
            .await
            .unwrap_err();
        assert!(matches!(err, CommitError::NotYetImplemented { .. }));
        assert_eq!(c.high_watermark(TenantId::new(1)).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn purge_memory_returns_not_yet_implemented_until_rfc_011() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let purge = MemoryMutation::PurgeMemory {
            rid: "x".into(),
            purge_epoch: 0,
        };
        let err = c
            .commit(TenantId::new(1), purge, CommitOptions::new())
            .await
            .unwrap_err();
        assert!(matches!(err, CommitError::NotYetImplemented { .. }));
    }

    #[tokio::test]
    async fn expected_log_index_match_succeeds() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let t = TenantId::new(1);
        let r = c
            .commit(t, upsert("a"), CommitOptions::new().expecting_index(1))
            .await
            .unwrap();
        assert_eq!(r.log_index, 1);
    }

    #[tokio::test]
    async fn expected_log_index_mismatch_fails() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let t = TenantId::new(1);
        c.commit(t, upsert("a"), CommitOptions::new()).await.unwrap();
        let err = c
            .commit(t, upsert("b"), CommitOptions::new().expecting_index(1))
            .await
            .unwrap_err();
        assert!(matches!(err, CommitError::UnexpectedLogIndex { .. }));
        assert_eq!(c.high_watermark(t).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn no_wait_opts_leaves_applied_at_none() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let r = c
            .commit(TenantId::new(1), upsert("a"), CommitOptions::new().no_wait())
            .await
            .unwrap();
        assert!(r.applied_at.is_none());
    }

    #[tokio::test]
    async fn total_entries_and_active_tenant_count_track_state() {
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        assert_eq!(c.total_entries().unwrap(), 0);
        assert_eq!(c.active_tenant_count().unwrap(), 0);
        c.commit(TenantId::new(1), upsert("a"), CommitOptions::new())
            .await
            .unwrap();
        c.commit(TenantId::new(1), upsert("b"), CommitOptions::new())
            .await
            .unwrap();
        c.commit(TenantId::new(2), upsert("c"), CommitOptions::new())
            .await
            .unwrap();
        assert_eq!(c.total_entries().unwrap(), 3);
        assert_eq!(c.active_tenant_count().unwrap(), 2);
    }

    #[tokio::test]
    async fn dyn_dispatch_works() {
        let c: Arc<dyn MutationCommitter> = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let r = c
            .commit(TenantId::new(1), upsert("a"), CommitOptions::new())
            .await
            .unwrap();
        assert_eq!(r.log_index, 1);
    }

    #[tokio::test]
    async fn client_provided_op_id_is_idempotent() {
        // RFC 010 PR-3 contract: HTTP clients can retry safely by
        // passing the same op_id. Re-commit returns the original receipt
        // without appending a duplicate row.
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let t = TenantId::new(1);
        let op_id = OpId::new_random();
        let m = upsert("idempotent");

        let r1 = c
            .commit(t, m.clone(), CommitOptions::new().with_op_id(op_id))
            .await
            .unwrap();
        let r2 = c
            .commit(t, m.clone(), CommitOptions::new().with_op_id(op_id))
            .await
            .unwrap();

        // Same op_id, same payload → same receipt (no duplicate append).
        assert_eq!(r1.op_id, r2.op_id);
        assert_eq!(r1.log_index, r2.log_index);
        assert_eq!(r1.committed_at, r2.committed_at);
        assert_eq!(c.high_watermark(t).await.unwrap(), 1, "no duplicate append");
        assert_eq!(c.total_entries().unwrap(), 1);
    }

    #[tokio::test]
    async fn client_provided_op_id_with_different_payload_returns_collision() {
        // Same op_id + different payload = client bug. Don't silently
        // overwrite; raise OpIdCollision so the client can fix.
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let t = TenantId::new(1);
        let op_id = OpId::new_random();

        c.commit(t, upsert("first"), CommitOptions::new().with_op_id(op_id))
            .await
            .unwrap();

        let err = c
            .commit(t, upsert("second"), CommitOptions::new().with_op_id(op_id))
            .await
            .unwrap_err();
        assert!(matches!(err, CommitError::OpIdCollision { .. }));
        // Original entry untouched.
        assert_eq!(c.high_watermark(t).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn op_id_isolation_across_tenants() {
        // Same op_id on different tenants is fine — they're separate
        // log spaces. Tested explicitly because the (tenant_id, op_id)
        // UNIQUE index is the enforcement boundary.
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let op_id = OpId::new_random();
        let r1 = c
            .commit(
                TenantId::new(1),
                upsert("a"),
                CommitOptions::new().with_op_id(op_id),
            )
            .await
            .unwrap();
        let r2 = c
            .commit(
                TenantId::new(2),
                upsert("a"),
                CommitOptions::new().with_op_id(op_id),
            )
            .await
            .unwrap();
        // Both succeed; per-tenant log_index=1 each.
        assert_eq!(r1.log_index, 1);
        assert_eq!(r2.log_index, 1);
        assert_eq!(r1.op_id, r2.op_id);
        assert_ne!(r1.tenant_id, r2.tenant_id);
    }

    #[tokio::test]
    async fn read_range_round_trips_full_mutation() {
        // PR-2 specific: payloads serialize to SQLite and back; verify
        // the round-trip preserves all fields, especially nested
        // metadata and the optional embedding.
        let c = LocalSqliteCommitter::open_in_memory().unwrap();
        let t = TenantId::new(1);
        let m = MemoryMutation::UpsertMemory {
            rid: "mem_complex".into(),
            text: "complex payload".into(),
            memory_type: "episodic".into(),
            importance: 0.73,
            valence: -0.5,
            half_life: 96.0,
            namespace: "tenant-x".into(),
            certainty: 0.9,
            domain: "specialized".into(),
            source: "extractor-v2".into(),
            emotional_state: Some("focused".into()),
            embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
            metadata: serde_json::json!({"tag": "test", "score": 42}),
        };
        c.commit(t, m.clone(), CommitOptions::new()).await.unwrap();
        let entries = c.read_range(t, 0, 10).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].mutation, m);
    }
}
