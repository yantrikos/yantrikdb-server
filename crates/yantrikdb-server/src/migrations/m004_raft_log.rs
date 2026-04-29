//! Migration 004 — openraft log storage tables.
//!
//! Three tables, all STRICT:
//! - **`raft_log_entries`** — cluster-global Raft log. Distinct from
//!   `memory_commit_log` (which is per-tenant application data
//!   populated by the state-machine apply path in PR-4-c). Each row
//!   carries the openraft framing (term + leader + payload variant)
//!   and the serialized `openraft::Entry<YantrikRaftTypeConfig>`.
//! - **`raft_vote`** — singleton row holding the persisted vote.
//!   The CHECK constraint pins the row to id=1 so callers can't
//!   accidentally insert a second record.
//! - **`raft_state`** — key-value store for openraft's misc pointers
//!   (currently `last_purged_log_id` and `committed_log_id`). A
//!   key-value table is right here because openraft has no fixed
//!   schema for these — they're optional state with serde-friendly
//!   serialization.
//!
//! ## Why a separate Raft log table from `memory_commit_log`
//!
//! The Raft log is cluster-global with openraft framing (membership
//! entries, blank entries for term promotion, normal entries carrying
//! `YantrikLogEntry`). `memory_commit_log` is per-tenant application
//! data — populated by the state-machine apply path (PR-4-c). The
//! Raft log can be purged after entries are applied + snapshotted,
//! while `memory_commit_log` retains entries for client replay,
//! tombstone reconciliation, and HNSW manifest watermark tracking.
//! Combining them would couple unrelated lifecycles.

use rusqlite::{Error, Transaction};

use super::Migration;

pub struct M004;

impl Migration for M004 {
    fn id(&self) -> u32 {
        4
    }

    fn name(&self) -> &'static str {
        "raft_log_storage"
    }

    fn up(&self, tx: &Transaction<'_>) -> Result<(), Error> {
        tx.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS raft_log_entries (
                log_index            INTEGER PRIMARY KEY,
                term                 INTEGER NOT NULL,
                leader_node_id       INTEGER NOT NULL,
                payload              BLOB    NOT NULL
            ) STRICT;

            -- Singleton vote row. CHECK constraint guarantees only one row.
            CREATE TABLE IF NOT EXISTS raft_vote (
                id                   INTEGER PRIMARY KEY CHECK (id = 1),
                payload              BLOB    NOT NULL,
                updated_at_unix_micros INTEGER NOT NULL
            ) STRICT;

            -- Key/value pointers (last_purged_log_id, committed_log_id).
            CREATE TABLE IF NOT EXISTS raft_state (
                key                  TEXT    PRIMARY KEY,
                value                BLOB    NOT NULL,
                updated_at_unix_micros INTEGER NOT NULL
            ) STRICT;
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
    fn migration_creates_three_tables() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        M004.up(&tx).unwrap();
        tx.commit().unwrap();

        let count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' \
                 AND name IN ('raft_log_entries', 'raft_vote', 'raft_state')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn migration_is_idempotent() {
        let mut conn = Connection::open_in_memory().unwrap();
        for _ in 0..3 {
            let tx = conn.transaction().unwrap();
            M004.up(&tx).unwrap();
            tx.commit().unwrap();
        }
        let count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='raft_log_entries'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn vote_table_singleton_check_rejects_second_row() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        M004.up(&tx).unwrap();
        tx.commit().unwrap();
        // First insert with id=1 OK.
        conn.execute(
            "INSERT INTO raft_vote (id, payload, updated_at_unix_micros) VALUES (1, X'00', 0)",
            [],
        )
        .unwrap();
        // Second insert with id=2 must fail (CHECK).
        let r = conn.execute(
            "INSERT INTO raft_vote (id, payload, updated_at_unix_micros) VALUES (2, X'00', 0)",
            [],
        );
        assert!(r.is_err(), "vote table CHECK must reject id != 1");
    }

    #[test]
    fn strict_table_rejects_wrong_types_in_log() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        M004.up(&tx).unwrap();
        tx.commit().unwrap();
        // String into the term INTEGER column.
        let r = conn.execute(
            "INSERT INTO raft_log_entries (log_index, term, leader_node_id, payload) \
             VALUES (1, 'not_int', 0, X'00')",
            [],
        );
        assert!(r.is_err(), "STRICT must reject string in INTEGER column");
    }
}
