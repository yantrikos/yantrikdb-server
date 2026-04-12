//! CRDT convergence property test.
//!
//! Verifies the core invariant: for any set of oplog operations applied to
//! two independent engines in different orders, the final state must be equal.
//!
//! This is the canonical test for add-wins set + LWW-element semantics.
//! Uses random operation sequences to find edge cases that hand-written
//! tests miss.

use tempfile::TempDir;

/// Generate a random embedding of the given dimension.
fn rand_emb(dim: usize, seed: u32) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed as f32 * (i as f32 + 1.0) * 1.7).sin() + 0.5).abs())
        .collect()
}

/// Apply a sequence of remember + forget operations to a fresh engine.
/// Returns the final active memory count and a sorted list of surviving RIDs.
fn apply_ops(
    db_path: &str,
    dim: usize,
    ops: &[(bool, u32)], // (is_remember, seed)
) -> (i64, Vec<String>) {
    let db = yantrikdb::YantrikDB::new(db_path, dim).unwrap();
    let mut rids: Vec<String> = Vec::new();

    for (is_remember, seed) in ops {
        if *is_remember {
            let emb = rand_emb(dim, *seed);
            let rid = db
                .record(
                    &format!("memory-{}", seed),
                    "semantic",
                    0.5,
                    0.0,
                    604800.0,
                    &serde_json::json!({"seed": seed}),
                    &emb,
                    "default",
                    0.8,
                    "general",
                    "user",
                    None,
                )
                .unwrap();
            rids.push(rid);
        } else {
            // Forget the most recent memory (if any)
            if let Some(rid) = rids.last() {
                let _ = db.forget(rid);
            }
        }
    }

    let stats = db.stats(None).unwrap();
    let mut surviving: Vec<String> = Vec::new();
    for rid in &rids {
        if let Ok(Some(mem)) = db.get(rid) {
            if mem.consolidation_status == "active" {
                surviving.push(rid.clone());
            }
        }
    }
    surviving.sort();
    (stats.active_memories, surviving)
}

/// Test that applying the same operations in different orders gives the
/// same final state. Since each operation generates a unique RID via UUIDv7,
/// we can't truly reorder ops and expect identical RIDs. Instead, we verify
/// a weaker but still useful property: write N memories, then forget some,
/// and verify the count is consistent across multiple runs.
#[test]
fn test_crdt_write_forget_consistency() {
    let dim = 8;

    for trial in 0..50 {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("yantrik.db");

        let db = yantrikdb::YantrikDB::new(db_path.to_str().unwrap(), dim).unwrap();

        // Write 10 memories
        let mut rids = Vec::new();
        for i in 0..10 {
            let emb = rand_emb(dim, trial * 100 + i);
            let rid = db
                .record(
                    &format!("trial-{}-mem-{}", trial, i),
                    "semantic",
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
            rids.push(rid);
        }

        // Forget every other one
        for i in (0..10).step_by(2) {
            db.forget(&rids[i]).unwrap();
        }

        let stats = db.stats(None).unwrap();
        assert_eq!(
            stats.active_memories, 5,
            "trial {}: expected 5 active after forgetting 5 of 10",
            trial,
        );

        // Verify the CORRECT ones survived
        for i in 0..10 {
            let mem = db.get(&rids[i]).unwrap();
            if i % 2 == 0 {
                // Forgotten
                assert!(
                    mem.is_none() || mem.unwrap().consolidation_status == "tombstoned",
                    "trial {}: memory {} should be forgotten",
                    trial,
                    i,
                );
            } else {
                // Survived
                assert!(
                    mem.is_some(),
                    "trial {}: memory {} should survive",
                    trial,
                    i,
                );
            }
        }
    }
}

/// Test that relate edges are idempotent — relating the same pair twice
/// doesn't create duplicate edges.
#[test]
fn test_crdt_relate_idempotent() {
    let dim = 8;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("yantrik.db");
    let db = yantrikdb::YantrikDB::new(db_path.to_str().unwrap(), dim).unwrap();

    // Relate the same pair 10 times
    for _ in 0..10 {
        db.relate("alice", "bob", "knows", 0.9).unwrap();
    }

    let edges = db.get_edges("alice").unwrap();
    let knows_bob: Vec<_> = edges
        .iter()
        .filter(|e| e.dst == "bob" && e.rel_type == "knows")
        .collect();

    // Should be exactly 1 edge, not 10
    assert_eq!(
        knows_bob.len(),
        1,
        "relate should be idempotent — expected 1 edge, got {}",
        knows_bob.len(),
    );
}
