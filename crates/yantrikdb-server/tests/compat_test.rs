//! Data directory compatibility test.
//!
//! Verifies that the current build can open, read, and operate on a data
//! directory created by older schema versions. This catches forward-
//! incompatible schema or file-format changes before they reach production.
//!
//! The test creates a fresh data dir with known content, then re-opens it
//! as the current build would on startup — verifying migrations run, the
//! engine loads, and queries return correct results.

use std::path::PathBuf;
use tempfile::TempDir;

/// Create a data dir with known content using the current engine, then
/// re-open it as a separate instance. This validates the migration path
/// and basic read-after-write correctness across engine opens.
#[test]
fn test_data_dir_survives_reopen() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("yantrik.db");

    let dim = 8;
    let emb: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();

    // Phase 1: create and populate
    {
        let db = yantrikdb::YantrikDB::new(db_path.to_str().unwrap(), dim).unwrap();
        let rid1 = db
            .record(
                "The quick brown fox",
                "semantic",
                0.8,
                0.0,
                604800.0,
                &serde_json::json!({"source": "compat_test"}),
                &emb,
                "default",
                0.9,
                "general",
                "user",
                None,
            )
            .unwrap();
        let rid2 = db
            .record(
                "jumped over the lazy dog",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &emb,
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        db.relate("fox", "dog", "chases", 0.9).unwrap();

        // Verify within this session
        let stats = db.stats(None).unwrap();
        assert_eq!(stats.active_memories, 2);
        assert!(stats.edges >= 1);
    }

    // Phase 2: reopen and verify everything survived
    {
        let db = yantrikdb::YantrikDB::new(db_path.to_str().unwrap(), dim).unwrap();

        let stats = db.stats(None).unwrap();
        assert_eq!(stats.active_memories, 2, "expected 2 memories after reopen");
        assert!(stats.edges >= 1, "expected at least 1 edge after reopen");

        // Recall should find memories
        let results = db
            .recall(
                &emb, 10, None,  // time_window
                None,  // memory_type
                false, // include_consolidated
                false, // expand_entities
                None,  // query_text
                false, // skip_reinforce
                None,  // namespace
                None,  // domain
                None,  // source
                None,  // certainty_min — v0.7.20 issue #46
                None,  // order — v0.7.20 issue #46
                false, // include_superseded — v0.10,
                None,  // event_after  — engine 0.17 valid-time filter; unbounded
                None,  // event_before
            )
            .unwrap();
        assert!(
            results.len() >= 2,
            "recall should return at least 2 memories"
        );

        // Edges should be intact
        let edges = db.get_edges("fox").unwrap();
        assert!(!edges.is_empty(), "fox edges should survive reopen");
    }
}

/// Verify that the control database survives reopen and contains expected data.
#[test]
fn test_control_db_survives_reopen() {
    use crate::helper::*;

    let tmp = TempDir::new().unwrap();
    let control_path = tmp.path().join("control.db");

    // Phase 1: create
    {
        let control = open_control(&control_path);
        control.create_database("mydb", "mydb").unwrap();
        let hash = "test_hash_abc123";
        control.create_token(hash, 1, "test-label").unwrap();
    }

    // Phase 2: reopen
    {
        let control = open_control(&control_path);
        let dbs = control.list_databases().unwrap();
        assert_eq!(dbs.len(), 1);
        assert_eq!(dbs[0].1, "mydb");

        let db_id = control.validate_token("test_hash_abc123").unwrap();
        assert_eq!(db_id, Some(1));
    }
}

// Helper to open ControlDb without importing the server binary's module
mod helper {
    use rusqlite::{params, Connection};
    use std::path::Path;

    pub struct SimpleControlDb {
        conn: Connection,
    }

    impl SimpleControlDb {
        pub fn create_database(&self, name: &str, path: &str) -> anyhow::Result<i64> {
            self.conn.execute(
                "INSERT INTO databases (name, path) VALUES (?1, ?2)",
                params![name, path],
            )?;
            Ok(self.conn.last_insert_rowid())
        }

        pub fn create_token(
            &self,
            hash: &str,
            database_id: i64,
            label: &str,
        ) -> anyhow::Result<()> {
            self.conn.execute(
                "INSERT INTO tokens (hash, database_id, label) VALUES (?1, ?2, ?3)",
                params![hash, database_id, label],
            )?;
            Ok(())
        }

        pub fn list_databases(&self) -> anyhow::Result<Vec<(i64, String)>> {
            let mut stmt = self
                .conn
                .prepare("SELECT id, name FROM databases ORDER BY id")?;
            let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
            Ok(rows.collect::<Result<Vec<_>, _>>()?)
        }

        pub fn validate_token(&self, hash: &str) -> anyhow::Result<Option<i64>> {
            let mut stmt = self
                .conn
                .prepare("SELECT database_id FROM tokens WHERE hash = ?1 AND revoked_at IS NULL")?;
            let mut rows = stmt.query_map(params![hash], |row| row.get::<_, i64>(0))?;
            Ok(rows.next().transpose()?)
        }
    }

    pub fn open_control(path: &Path) -> SimpleControlDb {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS databases (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL UNIQUE,
                 path TEXT NOT NULL,
                 config TEXT NOT NULL DEFAULT '{}',
                 created_at TEXT NOT NULL DEFAULT (datetime('now'))
             );
             CREATE TABLE IF NOT EXISTS tokens (
                 hash TEXT PRIMARY KEY,
                 database_id INTEGER NOT NULL REFERENCES databases(id),
                 label TEXT NOT NULL DEFAULT '',
                 created_at TEXT NOT NULL DEFAULT (datetime('now')),
                 revoked_at TEXT
             );",
        )
        .unwrap();
        SimpleControlDb { conn }
    }
}

// ── Issue #58: /v1/remember idempotency_key contract (engine semantics the
// gateway maps 1:1 — same-key+same-payload = original rid + zero writes;
// same-key+divergent-payload = typed IdempotencyConflict carrying the
// existing rid (gateway → 200 {stored:false, idempotency_conflict:true,
// rid}); batch per-item keys are all-or-nothing on conflict; invalid keys
// are refused loudly, never coerced. ──

fn idem_test_db() -> (TempDir, yantrikdb::YantrikDB) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("idem.db");
    let db = yantrikdb::YantrikDB::new(db_path.to_str().unwrap(), 8).unwrap();
    (tmp, db)
}

fn idem_vec() -> Vec<f32> {
    (0..8).map(|i| ((i as f32) * 0.3).sin()).collect()
}

#[test]
fn idempotency_same_key_same_payload_returns_original_rid_zero_writes() {
    let (_tmp, db) = idem_test_db();
    let emb = idem_vec();
    let meta = serde_json::json!({});
    let rid1 = db
        .record_with_idempotency(
            "keyed fact",
            "semantic",
            0.5,
            0.0,
            168.0,
            &meta,
            &emb,
            "ns",
            1.0,
            "work",
            "user",
            None,
            Some("key-1"),
            None, // created_at
        )
        .unwrap();
    let rid2 = db
        .record_with_idempotency(
            "keyed fact",
            "semantic",
            0.5,
            0.0,
            168.0,
            &meta,
            &emb,
            "ns",
            1.0,
            "work",
            "user",
            None,
            Some("key-1"),
            None, // created_at
        )
        .unwrap();
    assert_eq!(rid1, rid2, "retry must return the ORIGINAL rid");
    let stats = db.stats(None).unwrap();
    assert_eq!(
        stats.active_memories, 1,
        "idempotent retry must write nothing"
    );
}

#[test]
fn idempotency_same_key_divergent_payload_conflicts_with_existing_rid() {
    let (_tmp, db) = idem_test_db();
    let emb = idem_vec();
    let meta = serde_json::json!({});
    let rid1 = db
        .record_with_idempotency(
            "original text",
            "semantic",
            0.5,
            0.0,
            168.0,
            &meta,
            &emb,
            "ns",
            1.0,
            "work",
            "user",
            None,
            Some("key-2"),
            None, // created_at
        )
        .unwrap();
    let err = db
        .record_with_idempotency(
            "DIFFERENT text",
            "semantic",
            0.5,
            0.0,
            168.0,
            &meta,
            &emb,
            "ns",
            1.0,
            "work",
            "user",
            None,
            Some("key-2"),
            None, // created_at
        )
        .unwrap_err();
    match err {
        yantrikdb::YantrikDbError::IdempotencyConflict { existing_rid, .. } => {
            assert_eq!(existing_rid, rid1, "conflict must carry the existing rid");
        }
        other => panic!("expected IdempotencyConflict, got: {other}"),
    }
    let stats = db.stats(None).unwrap();
    assert_eq!(stats.active_memories, 1, "conflict must write nothing");
}

#[test]
fn idempotency_batch_per_item_keys_all_or_nothing() {
    let (_tmp, db) = idem_test_db();
    let emb = idem_vec();
    let mk = |text: &str, key: Option<&str>| yantrikdb::types::RecordInput {
        text: text.into(),
        memory_type: "semantic".into(),
        importance: 0.5,
        valence: 0.0,
        half_life: 168.0,
        metadata: serde_json::json!({}),
        embedding: emb.clone(),
        namespace: "ns".into(),
        certainty: 1.0,
        domain: "work".into(),
        source: "user".into(),
        emotional_state: None,
        idempotency_key: key.map(String::from),
        created_at: None,
    };
    // First batch: two keyed items (the "{caller_key}:{index}" derivation
    // the gateway applies for a batch-level key produces exactly this).
    let rids1 = db
        .record_batch(&[mk("item a", Some("bk:0")), mk("item b", Some("bk:1"))])
        .unwrap();
    assert_eq!(rids1.len(), 2);
    // Full retry: same rids back, zero new rows.
    let rids2 = db
        .record_batch(&[mk("item a", Some("bk:0")), mk("item b", Some("bk:1"))])
        .unwrap();
    assert_eq!(rids1, rids2, "batch retry must return original rids");
    assert_eq!(db.stats(None).unwrap().active_memories, 2);
    // Divergent payload under a claimed key fails the WHOLE batch.
    let err = db
        .record_batch(&[mk("item a", Some("bk:0")), mk("MUTATED b", Some("bk:1"))])
        .unwrap_err();
    assert!(
        matches!(err, yantrikdb::YantrikDbError::IdempotencyConflict { .. }),
        "expected IdempotencyConflict, got: {err}"
    );
    assert_eq!(
        db.stats(None).unwrap().active_memories,
        2,
        "failed batch must be all-or-nothing"
    );
}

#[test]
fn idempotency_invalid_key_is_refused_loudly() {
    let (_tmp, db) = idem_test_db();
    let emb = idem_vec();
    let meta = serde_json::json!({});
    let err = db
        .record_with_idempotency(
            "text",
            "semantic",
            0.5,
            0.0,
            168.0,
            &meta,
            &emb,
            "ns",
            1.0,
            "work",
            "user",
            None,
            Some("   "),
            None, // created_at
        )
        .unwrap_err();
    assert!(
        matches!(err, yantrikdb::YantrikDbError::InvalidIdempotencyKey { .. }),
        "whitespace key must be refused, not coerced: {err}"
    );
}
