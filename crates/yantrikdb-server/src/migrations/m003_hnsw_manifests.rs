//! Migration 003 — `hnsw_manifests` table.
//!
//! Per-tenant HNSW index manifest. One row per (tenant_id, embedding_model)
//! tuple — supports RFC 013 Phase 2's shadow-index dual-read during
//! embedding model migration. Until 013-B ships, only one manifest per
//! tenant exists at a time.
//!
//! ## Why a manifest at all
//!
//! v0.5.2 had a class of bug where replicated memories were not
//! searchable: a follower applied the oplog (memories landed in SQLite),
//! but its HNSW index was not rebuilt from those entries. The manifest
//! fixes this by tracking the **commit-log watermark** the index is
//! known to be consistent up to. On startup, if
//! `manifest.source_log_watermark < memory_commit_log.high_watermark`,
//! the reconciler replays the missing entries before serving recall.
//!
//! ## Schema
//!
//! ```sql
//! CREATE TABLE hnsw_manifests (
//!     tenant_id              INTEGER NOT NULL,
//!     embedding_model        TEXT    NOT NULL,
//!     index_generation       INTEGER NOT NULL,
//!     source_log_start       INTEGER NOT NULL,
//!     source_log_watermark   INTEGER NOT NULL,
//!     vector_dim             INTEGER NOT NULL,
//!     distance_metric        TEXT    NOT NULL,
//!     deleted_count_pending  INTEGER NOT NULL DEFAULT 0,
//!     checksum               TEXT,
//!     created_at_unix_micros INTEGER NOT NULL,
//!     updated_at_unix_micros INTEGER NOT NULL,
//!     PRIMARY KEY (tenant_id, embedding_model)
//! ) STRICT;
//! ```
//!
//! Composite PK is `(tenant_id, embedding_model)` so a tenant can hold
//! a primary + shadow manifest during model migration without breaking
//! the unique-per-tenant invariant the recall path assumes.

use rusqlite::{Error, Transaction};

use super::Migration;

pub struct M003;

impl Migration for M003 {
    fn id(&self) -> u32 {
        3
    }

    fn name(&self) -> &'static str {
        "hnsw_manifests"
    }

    fn up(&self, tx: &Transaction<'_>) -> Result<(), Error> {
        tx.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS hnsw_manifests (
                tenant_id              INTEGER NOT NULL,
                embedding_model        TEXT    NOT NULL,
                index_generation       INTEGER NOT NULL,
                source_log_start       INTEGER NOT NULL,
                source_log_watermark   INTEGER NOT NULL,
                vector_dim             INTEGER NOT NULL,
                distance_metric        TEXT    NOT NULL,
                deleted_count_pending  INTEGER NOT NULL DEFAULT 0,
                checksum               TEXT,
                created_at_unix_micros INTEGER NOT NULL,
                updated_at_unix_micros INTEGER NOT NULL,
                PRIMARY KEY (tenant_id, embedding_model)
            ) STRICT;

            -- Per-tenant lookup is the hot path (recall + reconciler);
            -- the PK already covers it but a separate index on
            -- (tenant_id) helps when the embedding_model is unknown
            -- (e.g. listing all manifests for a tenant).
            CREATE INDEX IF NOT EXISTS idx_hnsw_manifests_tenant
                ON hnsw_manifests (tenant_id);
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
    fn migration_creates_table_and_index() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        M003.up(&tx).unwrap();
        tx.commit().unwrap();

        let table_count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='hnsw_manifests'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(table_count, 1);

        let idx_count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' \
                 AND name='idx_hnsw_manifests_tenant'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(idx_count, 1);
    }

    #[test]
    fn migration_is_idempotent() {
        let mut conn = Connection::open_in_memory().unwrap();
        for _ in 0..3 {
            let tx = conn.transaction().unwrap();
            M003.up(&tx).unwrap();
            tx.commit().unwrap();
        }
        let table_count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='hnsw_manifests'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(table_count, 1);
    }

    #[test]
    fn strict_table_rejects_wrong_types() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        M003.up(&tx).unwrap();
        tx.commit().unwrap();

        // Insert a string into the INTEGER vector_dim column.
        let result = conn.execute(
            "INSERT INTO hnsw_manifests (
                tenant_id, embedding_model, index_generation,
                source_log_start, source_log_watermark,
                vector_dim, distance_metric,
                deleted_count_pending, checksum,
                created_at_unix_micros, updated_at_unix_micros
             ) VALUES (1, 'test', 0, 0, 0, 'not_int', 'cosine', 0, NULL, 0, 0)",
            [],
        );
        assert!(
            result.is_err(),
            "STRICT mode should reject string in INTEGER column"
        );
    }
}
