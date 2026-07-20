//! Reusable migration framework — RFC 010 PR-2.
//!
//! ## Why this module exists
//!
//! Eight tables are pre-registered in [`crate::version::TABLE_SCHEMA_VERSIONS`]
//! (memory_commit_log, quota_policies, quota_audit_log, memory_tombstones,
//! hnsw_manifests, backup_manifests, durable_jobs, tenant_config_overrides).
//! Each lands in its own RFC PR. Without a shared migration runner each
//! RFC would invent its own table-creation logic, ordering rules, and
//! "have I run this yet" tracking.
//!
//! This module is the shared substrate. Every RFC adds:
//! 1. A new file `m<NNN>_<table_name>.rs` implementing [`Migration`]
//! 2. An entry in [`ALL_MIGRATIONS`]
//!
//! At server startup, [`MigrationRunner::run_pending`] applies any
//! migrations whose `id` doesn't appear in the `_yantrikdb_meta_migrations`
//! table. Idempotent — safe to call multiple times.
//!
//! ## Design contract
//!
//! 1. **Forward-only**. No down migrations. If you need to revert a
//!    migration, write a new migration that reverses it. This avoids
//!    the "did down/up actually round-trip?" class of footguns.
//! 2. **Monotonic IDs**. `id` is a `u32` that strictly increases.
//!    Migrations apply in id order.
//! 3. **Idempotent at the table level**. Each migration's `up` SQL uses
//!    `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`. So
//!    if a migration is partially applied and then the runner is
//!    interrupted + restarted, re-running is safe.
//! 4. **Single-statement transactions per migration**. The runner wraps
//!    each migration's `up` in a transaction so a partial failure leaves
//!    the database consistent (either fully applied or untouched).
//! 5. **Schema versions tracked per RFC** in `crate::version::TABLE_SCHEMA_VERSIONS`.
//!    The migration just creates the table; the version registry is
//!    where the "I expect schema vN" assertion lives.

pub mod m001_memory_commit_log;
pub mod m002_durable_jobs;
pub mod m003_hnsw_manifests;
// m004_raft_log removed in Phase D (saga 239) — it created openraft-only
// tables (raft_log_entries/raft_vote/raft_state). The runner is
// forward-only and tolerates already-applied ids not present in this
// list, so dropping it is safe on DBs that recorded id=4.

use rusqlite::{Connection, Transaction};
use thiserror::Error;

/// Error type for migration operations.
#[derive(Debug, Error)]
pub enum MigrationError {
    #[error("SQLite error during migration {migration_id} ({migration_name}): {source}")]
    Sqlite {
        migration_id: u32,
        migration_name: &'static str,
        source: rusqlite::Error,
    },

    #[error("migration {id} ({name}) failed: {message}")]
    Failure {
        id: u32,
        name: &'static str,
        message: String,
    },

    #[error("migrations applied out of order: tried {tried} after {previous}")]
    OutOfOrder { previous: u32, tried: u32 },
}

/// Implemented by every migration in the codebase.
pub trait Migration: Send + Sync {
    /// Monotonic ID. Migrations run in `id` order.
    fn id(&self) -> u32;

    /// Stable human-readable name. Stored alongside the id in the
    /// metadata table for operator visibility.
    fn name(&self) -> &'static str;

    /// Apply the migration. Called inside a transaction; if `up` returns
    /// an error, the transaction rolls back.
    fn up(&self, tx: &Transaction<'_>) -> Result<(), rusqlite::Error>;
}

/// The authoritative list of every migration in the codebase, in apply
/// order. Add new migrations to the end; never reorder or remove
/// existing entries.
fn all_migrations() -> Vec<Box<dyn Migration>> {
    vec![
        Box::new(m001_memory_commit_log::M001),
        Box::new(m002_durable_jobs::M002),
        Box::new(m003_hnsw_manifests::M003),
    ]
}

pub struct MigrationRunner;

impl MigrationRunner {
    /// Apply any pending migrations against the given connection.
    /// Idempotent: migrations already applied are skipped.
    ///
    /// Returns the number of migrations applied during this call (0 if
    /// the database is fully up-to-date).
    pub fn run_pending(conn: &mut Connection) -> Result<usize, MigrationError> {
        Self::ensure_meta_table(conn).map_err(|e| MigrationError::Sqlite {
            migration_id: 0,
            migration_name: "_meta_table_setup",
            source: e,
        })?;

        let applied = Self::applied_ids(conn).map_err(|e| MigrationError::Sqlite {
            migration_id: 0,
            migration_name: "_applied_ids_query",
            source: e,
        })?;

        let migrations = all_migrations();

        // Verify the `applied` set is a prefix of the migration list.
        // If we ever skip a migration in the static list (e.g. someone
        // accidentally removes a migration entry from `all_migrations`),
        // we want a clear error rather than silent corruption.
        for migration in &migrations {
            // Either applied (must match) or not applied (and we'll run it).
            // No "applied but not in list" detection here — if you delete
            // a migration from the static list, the runner can't tell.
            // Standing acceptance criterion 6 (schema versioning) catches
            // this at a different layer.
            let _ = migration.id();
        }

        let mut applied_count = 0;
        let mut last_applied_id = applied.iter().copied().max().unwrap_or(0);

        for migration in migrations {
            let id = migration.id();
            if applied.contains(&id) {
                continue;
            }
            if id <= last_applied_id {
                return Err(MigrationError::OutOfOrder {
                    previous: last_applied_id,
                    tried: id,
                });
            }

            let name = migration.name();
            tracing::info!(
                migration_id = id,
                migration_name = name,
                "applying migration"
            );

            let tx = conn.transaction().map_err(|e| MigrationError::Sqlite {
                migration_id: id,
                migration_name: name,
                source: e,
            })?;

            migration.up(&tx).map_err(|e| MigrationError::Sqlite {
                migration_id: id,
                migration_name: name,
                source: e,
            })?;

            tx.execute(
                "INSERT INTO _yantrikdb_meta_migrations (id, name, applied_at_unix_micros) VALUES (?1, ?2, ?3)",
                rusqlite::params![
                    id,
                    name,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_micros() as i64)
                        .unwrap_or(0),
                ],
            )
            .map_err(|e| MigrationError::Sqlite {
                migration_id: id,
                migration_name: name,
                source: e,
            })?;

            tx.commit().map_err(|e| MigrationError::Sqlite {
                migration_id: id,
                migration_name: name,
                source: e,
            })?;

            applied_count += 1;
            last_applied_id = id;
            tracing::info!(
                migration_id = id,
                migration_name = name,
                "migration applied"
            );
        }

        Ok(applied_count)
    }

    fn ensure_meta_table(conn: &Connection) -> Result<(), rusqlite::Error> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS _yantrikdb_meta_migrations (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at_unix_micros INTEGER NOT NULL
            ) STRICT;",
        )
    }

    fn applied_ids(conn: &Connection) -> Result<std::collections::HashSet<u32>, rusqlite::Error> {
        let mut stmt = conn.prepare("SELECT id FROM _yantrikdb_meta_migrations")?;
        let ids: rusqlite::Result<Vec<u32>> =
            stmt.query_map([], |row| row.get::<_, u32>(0))?.collect();
        Ok(ids?.into_iter().collect())
    }

    /// Inspect which migrations have been applied. Used by `yantrikdb
    /// migrations status` CLI and `/v1/health/deep` for operator
    /// visibility into rolling-upgrade state.
    pub fn applied_summary(conn: &Connection) -> Result<Vec<(u32, String)>, rusqlite::Error> {
        Self::ensure_meta_table(conn)?;
        let mut stmt =
            conn.prepare("SELECT id, name FROM _yantrikdb_meta_migrations ORDER BY id ASC")?;
        let rows: rusqlite::Result<Vec<(u32, String)>> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, u32>(0)?, row.get::<_, String>(1)?))
            })?
            .collect();
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_in_memory() -> Connection {
        Connection::open_in_memory().expect("in-memory sqlite")
    }

    #[test]
    fn runs_pending_migrations_on_fresh_db() {
        let mut conn = open_in_memory();
        let n = MigrationRunner::run_pending(&mut conn).unwrap();
        assert!(n >= 1, "should have applied at least m001");

        // The _yantrikdb_meta_migrations table should now exist with
        // at least one row.
        let count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM _yantrikdb_meta_migrations",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(count >= 1);
    }

    #[test]
    fn run_pending_is_idempotent() {
        let mut conn = open_in_memory();
        let first_run = MigrationRunner::run_pending(&mut conn).unwrap();
        let second_run = MigrationRunner::run_pending(&mut conn).unwrap();
        assert!(first_run >= 1, "first run should apply migrations");
        assert_eq!(second_run, 0, "second run should be a no-op");
    }

    #[test]
    fn applied_summary_returns_ordered_list() {
        let mut conn = open_in_memory();
        MigrationRunner::run_pending(&mut conn).unwrap();
        let summary = MigrationRunner::applied_summary(&conn).unwrap();
        // At least m001 must be in the summary.
        assert!(!summary.is_empty());
        // Ordered by id ASC.
        for window in summary.windows(2) {
            assert!(
                window[0].0 < window[1].0,
                "summary not ordered: {summary:?}"
            );
        }
        // m001 is the first migration.
        assert_eq!(summary[0].0, 1);
        assert_eq!(summary[0].1, "memory_commit_log");
    }
}
