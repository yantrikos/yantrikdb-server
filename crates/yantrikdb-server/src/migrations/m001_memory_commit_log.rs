//! Migration 001 — `memory_commit_log` table.
//!
//! This is the durable backing for [`crate::commit::LocalSqliteCommitter`].
//! Every mutation written through the committer ends up as a row here,
//! tagged with its (tenant_id, log_index) primary key, the originating
//! `op_id` for idempotency, the wire/schema versions for replay safety,
//! and the serialized payload.
//!
//! ## Schema
//!
//! ```sql
//! CREATE TABLE memory_commit_log (
//!     tenant_id INTEGER NOT NULL,
//!     log_index INTEGER NOT NULL,
//!     term INTEGER NOT NULL DEFAULT 0,
//!     op_id TEXT NOT NULL,
//!     op_kind TEXT NOT NULL,
//!     payload BLOB NOT NULL,
//!     wire_version_major INTEGER NOT NULL,
//!     wire_version_minor INTEGER NOT NULL,
//!     schema_table TEXT,
//!     schema_version INTEGER,
//!     committed_at_unix_micros INTEGER NOT NULL,
//!     applied_at_unix_micros INTEGER,
//!     PRIMARY KEY (tenant_id, log_index)
//! );
//! ```
//!
//! Plus a `UNIQUE (tenant_id, op_id)` index for idempotency.
//!
//! ## Why STRICT mode
//!
//! SQLite STRICT tables reject row inserts whose value doesn't match
//! the column's declared type. Without STRICT, `INSERT (..., 'banana')`
//! into an INTEGER column silently succeeds and stores the string. We
//! want type errors loud, so the `STRICT` modifier is mandatory on every
//! YantrikDB table going forward.

use rusqlite::{Error, Transaction};

use super::Migration;

pub struct M001;

impl Migration for M001 {
    fn id(&self) -> u32 {
        1
    }

    fn name(&self) -> &'static str {
        "memory_commit_log"
    }

    fn up(&self, tx: &Transaction<'_>) -> Result<(), Error> {
        tx.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memory_commit_log (
                tenant_id              INTEGER NOT NULL,
                log_index              INTEGER NOT NULL,
                term                   INTEGER NOT NULL DEFAULT 0,
                op_id                  TEXT    NOT NULL,
                op_kind                TEXT    NOT NULL,
                payload                BLOB    NOT NULL,
                wire_version_major     INTEGER NOT NULL,
                wire_version_minor     INTEGER NOT NULL,
                schema_table           TEXT,
                schema_version         INTEGER,
                committed_at_unix_micros INTEGER NOT NULL,
                applied_at_unix_micros   INTEGER,
                PRIMARY KEY (tenant_id, log_index)
            ) STRICT;

            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_commit_log_op_id
                ON memory_commit_log (tenant_id, op_id);

            CREATE INDEX IF NOT EXISTS idx_memory_commit_log_op_kind
                ON memory_commit_log (tenant_id, op_kind);
            "#,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn migration_creates_table_and_indexes() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        M001.up(&tx).unwrap();
        tx.commit().unwrap();

        // Table exists.
        let table_count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_commit_log'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(table_count, 1);

        // Both indexes exist.
        let idx_count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' \
                 AND name IN ('idx_memory_commit_log_op_id', 'idx_memory_commit_log_op_kind')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(idx_count, 2);
    }

    #[test]
    fn migration_is_idempotent() {
        let mut conn = Connection::open_in_memory().unwrap();
        for _ in 0..3 {
            let tx = conn.transaction().unwrap();
            M001.up(&tx).unwrap();
            tx.commit().unwrap();
        }
        // Should still have exactly one table.
        let table_count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_commit_log'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(table_count, 1);
    }

    #[test]
    fn strict_table_rejects_wrong_types() {
        // STRICT mode is non-negotiable for YantrikDB schema. Pin it
        // with a regression test so any future migration that drops
        // STRICT fails this check.
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        M001.up(&tx).unwrap();
        tx.commit().unwrap();

        // Attempt to insert a string into the INTEGER tenant_id column.
        // STRICT mode should reject this.
        let result = conn.execute(
            "INSERT INTO memory_commit_log (
                tenant_id, log_index, op_id, op_kind, payload,
                wire_version_major, wire_version_minor,
                committed_at_unix_micros
             ) VALUES ('not_an_integer', 1, 'op', 'kind', X'00', 1, 0, 0)",
            [],
        );
        assert!(
            result.is_err(),
            "STRICT mode should reject string in INTEGER column"
        );
    }
}
