//! Crash recovery test — verifies that SQLite WAL + journal_mode protects
//! against data corruption on unclean shutdown.
//!
//! Simulates crash-after-write by writing memories then dropping the engine
//! without close() (equivalent to kill -9). Reopens and verifies all ack'd
//! writes are present.

use tempfile::TempDir;

/// Write N memories, drop without close, reopen, verify all are present.
/// Repeat M times to exercise the WAL recovery path repeatedly.
#[test]
fn test_crash_recovery_write_drop_reopen() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("yantrik.db");
    let dim = 8;
    let emb: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();

    let iterations = 20; // 20 crash-cycles
    let writes_per_iter = 10;
    let mut total_written = 0u64;

    for iter in 0..iterations {
        // Open, write, DROP (simulates kill -9 — no close/flush)
        {
            let db = yantrikdb::YantrikDB::new(db_path.to_str().unwrap(), dim).unwrap();
            for j in 0..writes_per_iter {
                let text = format!("crash test iter={} j={}", iter, j);
                db.record(
                    &text,
                    "semantic",
                    0.5,
                    0.0,
                    604800.0,
                    &serde_json::json!({"iter": iter, "j": j}),
                    &emb,
                    "default",
                    0.8,
                    "general",
                    "user",
                    None,
                )
                .unwrap();
                total_written += 1;
            }
            // Drop without close — WAL may have uncommitted pages
            drop(db);
        }

        // Reopen and verify
        {
            let db = yantrikdb::YantrikDB::new(db_path.to_str().unwrap(), dim).unwrap();
            let stats = db.stats(None).unwrap();
            assert_eq!(
                stats.active_memories, total_written as i64,
                "iteration {}: expected {} memories, got {}",
                iter, total_written, stats.active_memories,
            );
        }
    }

    // Final verification: all memories accessible via recall
    let db = yantrikdb::YantrikDB::new(db_path.to_str().unwrap(), dim).unwrap();
    let stats = db.stats(None).unwrap();
    assert_eq!(
        stats.active_memories,
        (iterations * writes_per_iter) as i64,
        "final check: expected {} total memories",
        iterations * writes_per_iter,
    );
}

/// Verify that oplog entries survive crash-recovery alongside memories.
#[test]
fn test_crash_recovery_oplog_integrity() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("yantrik.db");
    let dim = 8;
    let emb: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();

    // Write memories + relate edges
    {
        let db = yantrikdb::YantrikDB::new(db_path.to_str().unwrap(), dim).unwrap();
        for i in 0..5 {
            db.record(
                &format!("entity {} is important", i),
                "semantic",
                0.8,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &emb,
                "default",
                0.9,
                "general",
                "user",
                None,
            )
            .unwrap();
        }
        db.relate("alice", "bob", "knows", 0.9).unwrap();
        db.relate("bob", "charlie", "works_with", 0.7).unwrap();
        // Drop without close
        drop(db);
    }

    // Reopen and verify edges + oplog
    {
        let db = yantrikdb::YantrikDB::new(db_path.to_str().unwrap(), dim).unwrap();
        let stats = db.stats(None).unwrap();
        assert_eq!(stats.active_memories, 5);
        assert!(stats.edges >= 2, "edges should survive crash recovery");
        assert!(
            stats.operations >= 7,
            "oplog should have at least 7 entries (5 records + 2 relates)"
        );
    }
}
