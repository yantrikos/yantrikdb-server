//! Migration 002 — `durable_jobs` table for RFC 019.
//!
//! Backing storage for [`crate::jobs::LocalSqliteJobQueue`]. Substrate
//! consumed by HNSW delete queue (RFC 011), snapshot creation (RFC 012),
//! index reconciliation (RFC 013), re-embedding (RFC 013-B), cache
//! warming (RFC 015). Without this shared table, each RFC would invent
//! its own broken worker loop.
//!
//! ## Schema
//!
//! ```sql
//! CREATE TABLE durable_jobs (
//!     id                            TEXT    PRIMARY KEY,
//!     tenant_id                     INTEGER NOT NULL,
//!     kind                          TEXT    NOT NULL,
//!     payload                       BLOB    NOT NULL,
//!     state                         TEXT    NOT NULL,
//!     priority                      INTEGER NOT NULL DEFAULT 5,
//!     created_at_unix_micros        INTEGER NOT NULL,
//!     leased_by                     TEXT,
//!     leased_at_unix_micros         INTEGER,
//!     lease_expires_at_unix_micros  INTEGER,
//!     completed_at_unix_micros      INTEGER,
//!     outcome                       TEXT,
//!     error_message                 TEXT
//! ) STRICT;
//! ```

use rusqlite::{Error, Transaction};

use super::Migration;

pub struct M002;

impl Migration for M002 {
    fn id(&self) -> u32 {
        2
    }

    fn name(&self) -> &'static str {
        "durable_jobs"
    }

    fn up(&self, tx: &Transaction<'_>) -> Result<(), Error> {
        tx.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS durable_jobs (
                id                            TEXT    PRIMARY KEY,
                tenant_id                     INTEGER NOT NULL,
                kind                          TEXT    NOT NULL,
                payload                       BLOB    NOT NULL,
                state                         TEXT    NOT NULL,
                priority                      INTEGER NOT NULL DEFAULT 5,
                created_at_unix_micros        INTEGER NOT NULL,
                leased_by                     TEXT,
                leased_at_unix_micros         INTEGER,
                lease_expires_at_unix_micros  INTEGER,
                completed_at_unix_micros      INTEGER,
                outcome                       TEXT,
                error_message                 TEXT
            ) STRICT;

            -- Composite index for the most-frequent query: pick the
            -- next Pending job by (priority DESC, created_at ASC) for
            -- a given tenant.
            CREATE INDEX IF NOT EXISTS idx_durable_jobs_pickup
                ON durable_jobs (state, tenant_id, priority DESC, created_at_unix_micros);

            -- Index for lease-expiry sweeps. Covers the WHERE clause of
            -- `expire_stale_leases`.
            CREATE INDEX IF NOT EXISTS idx_durable_jobs_lease_expires
                ON durable_jobs (state, lease_expires_at_unix_micros);

            -- Index for kind-filtered queries (e.g. "give me a Pending
            -- hnsw_delete job"). Used when workers ask for specific kinds.
            CREATE INDEX IF NOT EXISTS idx_durable_jobs_kind_state
                ON durable_jobs (kind, state);
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
        M002.up(&tx).unwrap();
        tx.commit().unwrap();

        let table_count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='durable_jobs'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(table_count, 1);

        let idx_count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' \
                 AND name LIKE 'idx_durable_jobs_%'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(idx_count, 3);
    }

    #[test]
    fn strict_table_rejects_wrong_types() {
        // Pin STRICT mode (standing acceptance criterion from RFC 010 PR-2).
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        M002.up(&tx).unwrap();
        tx.commit().unwrap();

        // INTEGER tenant_id rejects strings.
        let result = conn.execute(
            "INSERT INTO durable_jobs (
                id, tenant_id, kind, payload, state, priority, created_at_unix_micros
             ) VALUES ('j', 'not_an_integer', 'k', X'00', 'Pending', 5, 0)",
            [],
        );
        assert!(
            result.is_err(),
            "STRICT mode should reject string in INTEGER column"
        );
    }
}
