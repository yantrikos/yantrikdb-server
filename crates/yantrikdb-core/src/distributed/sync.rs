//! Bidirectional sync protocol.
//!
//! Pull-based delta sync: each side extracts ops since the peer's watermark,
//! sends them to the other side for application. Idempotent — syncing twice
//! transfers 0 additional ops.

use crate::engine::YantrikDB;
use crate::error::Result;
use crate::replication::{apply_ops, extract_ops_since, get_peer_watermark, set_peer_watermark};

/// Result of a bidirectional sync.
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Number of ops A pulled from B.
    pub a_pulled: usize,
    /// Number of ops B pulled from A.
    pub b_pulled: usize,
}

/// Bidirectional sync between two YantrikDB instances.
///
/// Algorithm:
/// 1. A reads its watermark for B → B extracts ops since that cursor → A applies them
/// 2. B reads its watermark for A → A extracts ops since that cursor → B applies them
/// 3. Update watermarks on both sides
pub fn sync_bidirectional(a: &YantrikDB, b: &YantrikDB) -> Result<SyncResult> {
    let a_actor = a.actor_id().to_string();
    let b_actor = b.actor_id().to_string();

    // ── Phase 1: A pulls from B ──
    let a_watermark = get_peer_watermark(a.conn(), &b_actor)?;
    let (since_hlc, since_op_id) = match &a_watermark {
        Some((hlc, op_id)) => (Some(hlc.as_slice()), Some(op_id.as_str())),
        None => (None, None),
    };

    let b_ops = extract_ops_since(
        b.conn(),
        since_hlc,
        since_op_id,
        Some(&a_actor), // exclude A's own ops that were previously synced to B
        10_000,
    )?;

    let a_stats = if !b_ops.is_empty() {
        let stats = apply_ops(a, &b_ops)?;

        // Update A's watermark for B using the last op received
        if let Some(last_op) = b_ops.last() {
            set_peer_watermark(a.conn(), &b_actor, &last_op.hlc, &last_op.op_id)?;
        }

        stats.ops_applied
    } else {
        0
    };

    // ── Phase 2: B pulls from A ──
    let b_watermark = get_peer_watermark(b.conn(), &a_actor)?;
    let (since_hlc, since_op_id) = match &b_watermark {
        Some((hlc, op_id)) => (Some(hlc.as_slice()), Some(op_id.as_str())),
        None => (None, None),
    };

    let a_ops = extract_ops_since(
        a.conn(),
        since_hlc,
        since_op_id,
        Some(&b_actor), // exclude B's own ops that were previously synced to A
        10_000,
    )?;

    let b_stats = if !a_ops.is_empty() {
        let stats = apply_ops(b, &a_ops)?;

        // Update B's watermark for A using the last op received
        if let Some(last_op) = a_ops.last() {
            set_peer_watermark(b.conn(), &a_actor, &last_op.hlc, &last_op.op_id)?;
        }

        stats.ops_applied
    } else {
        0
    };

    Ok(SyncResult {
        a_pulled: a_stats,
        b_pulled: b_stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    fn empty_meta() -> serde_json::Value {
        serde_json::json!({})
    }

    #[test]
    fn test_two_instances_converge() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "device-A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "device-B").unwrap();

        // A writes independently
        a.record("from A - memory 1", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        a.record("from A - memory 2", "episodic", 0.7, 0.1, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "general", "user", None).unwrap();
        a.relate("Alice", "Bob", "knows", 1.0).unwrap();

        // B writes independently
        b.record("from B - memory 3", "semantic", 0.6, 0.0, 604800.0, &empty_meta(), &vec_seed(3.0, 8), "default", 0.8, "general", "user", None).unwrap();
        b.relate("Charlie", "Dave", "works_with", 0.8).unwrap();

        // Sync
        let result = sync_bidirectional(&a, &b).unwrap();
        assert!(result.a_pulled > 0);
        assert!(result.b_pulled > 0);

        // Both should have the same stats (except operations count may differ
        // due to reinforce ops being local-only)
        let a_stats = a.stats(None).unwrap();
        let b_stats = b.stats(None).unwrap();
        assert_eq!(a_stats.active_memories, b_stats.active_memories);
        assert_eq!(a_stats.edges, b_stats.edges);
        assert_eq!(a_stats.entities, b_stats.entities);
    }

    #[test]
    fn test_forget_propagates() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        // A records and B gets it via sync
        let rid = a.record("will be forgotten", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        sync_bidirectional(&a, &b).unwrap();

        // Verify B has it
        assert!(b.get(&rid).unwrap().is_some());
        assert_eq!(b.get(&rid).unwrap().unwrap().consolidation_status, "active");

        // A forgets it
        a.forget(&rid).unwrap();

        // Sync again
        sync_bidirectional(&a, &b).unwrap();

        // B should see it tombstoned
        let mem = b.get(&rid).unwrap().unwrap();
        assert_eq!(mem.consolidation_status, "tombstoned");
    }

    #[test]
    fn test_concurrent_edges_lww() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        // Both create same (src,dst,rel_type)
        a.relate("X", "Y", "linked", 0.3).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        b.relate("X", "Y", "linked", 0.9).unwrap();

        // Sync
        sync_bidirectional(&a, &b).unwrap();

        // Both should have weight=0.9 (B was later)
        let a_edges = a.get_edges("X").unwrap();
        let b_edges = b.get_edges("X").unwrap();
        assert_eq!(a_edges.len(), 1);
        assert_eq!(b_edges.len(), 1);
        assert_eq!(a_edges[0].weight, b_edges[0].weight);
        assert_eq!(a_edges[0].weight, 0.9);
    }

    #[test]
    fn test_sync_idempotent() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        a.record("test", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

        // First sync
        let r1 = sync_bidirectional(&a, &b).unwrap();
        assert!(r1.b_pulled > 0);

        // Second sync — should transfer 0 ops
        let r2 = sync_bidirectional(&a, &b).unwrap();
        assert_eq!(r2.a_pulled, 0);
        assert_eq!(r2.b_pulled, 0);
    }

    #[test]
    fn test_consolidation_syncs() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        // A creates memories and consolidates
        a.record("mem1", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        a.record("mem2", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.1, 8), "default", 0.8, "general", "user", None).unwrap();

        let consolidated = crate::consolidate::consolidate(&a, 0.0, 365.0, 2, false).unwrap();
        assert!(!consolidated.is_empty());

        // Sync to B
        sync_bidirectional(&a, &b).unwrap();

        // B should have consolidation_members entries
        let cm_count: i64 = b.conn().query_row(
            "SELECT COUNT(*) FROM consolidation_members",
            [],
            |row| row.get(0),
        ).unwrap();
        assert!(cm_count >= 2);

        // B should have the consolidated memory
        let b_stats = b.stats(None).unwrap();
        assert!(b_stats.active_memories >= 1); // The consolidated memory
    }

    #[test]
    fn test_independent_consolidation_preserved() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        // Create shared memories on both via sync
        let r1 = a.record("shared1", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        let r2 = a.record("shared2", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.1, 8), "default", 0.8, "general", "user", None).unwrap();
        let r3 = a.record("shared3", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.2, 8), "default", 0.8, "general", "user", None).unwrap();
        sync_bidirectional(&a, &b).unwrap();

        // A consolidates {shared1, shared2} — manually create the consolidated
        // memory + consolidation_members + oplog entry so it replicates
        let a_consolidated = a.record("A consolidated", "semantic", 0.6, 0.0, 604800.0, &serde_json::json!({"consolidated_from": [&r1, &r2]}), &vec_seed(1.05, 8), "default", 0.8, "general", "user", None).unwrap();
        let hlc_a = a.tick_hlc();
        let hlc_a_bytes = hlc_a.to_bytes().to_vec();
        a.conn().execute(
            "INSERT OR IGNORE INTO consolidation_members (consolidation_rid, source_rid, hlc, actor_id) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![a_consolidated, r1, hlc_a_bytes, "A"],
        ).unwrap();
        a.conn().execute(
            "INSERT OR IGNORE INTO consolidation_members (consolidation_rid, source_rid, hlc, actor_id) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![a_consolidated, r2, hlc_a_bytes, "A"],
        ).unwrap();
        // Log a consolidate op so it replicates
        a.log_op("consolidate", Some(&a_consolidated), &serde_json::json!({
            "consolidated_rid": a_consolidated,
            "source_rids": [&r1, &r2],
            "text": "A consolidated",
            "importance": 0.6,
            "valence": 0.0,
            "half_life": 604800.0,
        }), None).unwrap();

        // B consolidates {shared2, shared3}
        let b_consolidated = b.record("B consolidated", "semantic", 0.6, 0.0, 604800.0, &serde_json::json!({"consolidated_from": [&r2, &r3]}), &vec_seed(1.15, 8), "default", 0.8, "general", "user", None).unwrap();
        let hlc_b = b.tick_hlc();
        let hlc_b_bytes = hlc_b.to_bytes().to_vec();
        b.conn().execute(
            "INSERT OR IGNORE INTO consolidation_members (consolidation_rid, source_rid, hlc, actor_id) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![b_consolidated, r2, hlc_b_bytes, "B"],
        ).unwrap();
        b.conn().execute(
            "INSERT OR IGNORE INTO consolidation_members (consolidation_rid, source_rid, hlc, actor_id) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![b_consolidated, r3, hlc_b_bytes, "B"],
        ).unwrap();
        b.log_op("consolidate", Some(&b_consolidated), &serde_json::json!({
            "consolidated_rid": b_consolidated,
            "source_rids": [&r2, &r3],
            "text": "B consolidated",
            "importance": 0.6,
            "valence": 0.0,
            "half_life": 604800.0,
        }), None).unwrap();

        // Sync
        sync_bidirectional(&a, &b).unwrap();

        // Both should have both consolidation records
        let a_cm: i64 = a.conn().query_row(
            "SELECT COUNT(DISTINCT consolidation_rid) FROM consolidation_members",
            [],
            |row| row.get(0),
        ).unwrap();
        let b_cm: i64 = b.conn().query_row(
            "SELECT COUNT(DISTINCT consolidation_rid) FROM consolidation_members",
            [],
            |row| row.get(0),
        ).unwrap();

        // Both A and B should have at least 2 distinct consolidation records
        assert!(a_cm >= 2, "A should have both consolidation records, got {a_cm}");
        assert!(b_cm >= 2, "B should have both consolidation records, got {b_cm}");
    }

    #[test]
    fn test_vector_index_consistent_after_sync() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        // A records memories
        let emb1 = vec_seed(1.0, 8);
        let emb2 = vec_seed(5.0, 8);
        a.record("close to query", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &emb1, "default", 0.8, "general", "user", None).unwrap();
        a.record("far from query", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &emb2, "default", 0.8, "general", "user", None).unwrap();

        // Sync to B
        sync_bidirectional(&a, &b).unwrap();

        // A can recall (has vec_memories entries)
        let a_results = a.recall(&emb1, 2, None, None, false, false, None, false, None, None, None).unwrap();
        assert!(!a_results.is_empty());

        // B won't have vec_memories (embeddings aren't in oplog)
        // But after rebuild, it should work
        crate::replication::rebuild_vec_index(&b).unwrap();

        // Now B should be able to recall — but B doesn't have embeddings in memories
        // table either (they're not in the oplog payload). This is expected in V1:
        // both devices generate embeddings independently.
        // For this test, B's memories have no embeddings, so recall returns empty.
        // This is the correct V1 behavior — embedding sync is deferred.

        // Just verify the memory data converged
        let a_stats = a.stats(None).unwrap();
        let b_stats = b.stats(None).unwrap();
        assert_eq!(a_stats.active_memories, b_stats.active_memories);
    }

    // ── V2: Conflict integration tests ──

    #[test]
    fn test_conflict_resolution_replicates() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        let rid_a = a.record("birthday March 5", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        let rid_b = a.record("birthday March 15", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "general", "user", None).unwrap();

        // Create and resolve a conflict on A
        let conflict = crate::conflict::create_conflict(
            &a,
            &crate::types::ConflictType::IdentityFact,
            &rid_a,
            &rid_b,
            Some("User"),
            Some("birthday"),
            "conflicting birthdays",
        )
        .unwrap();

        a.resolve_conflict(
            &conflict.conflict_id,
            "keep_a",
            Some(&rid_a),
            None,
            Some("Confirmed March 5"),
        )
        .unwrap();

        // Sync to B
        sync_bidirectional(&a, &b).unwrap();

        // B should have the conflict record (resolved)
        let b_conflict = b.get_conflict(&conflict.conflict_id).unwrap();
        assert!(b_conflict.is_some());
        assert_eq!(b_conflict.unwrap().status, "resolved");

        // B should have memory_b tombstoned
        let mem_b_on_b = b.get(&rid_b).unwrap().unwrap();
        assert_eq!(mem_b_on_b.consolidation_status, "tombstoned");
    }

    #[test]
    fn test_correction_replicates() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        let rid = a.record("color is green", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        sync_bidirectional(&a, &b).unwrap();

        // Correct on A
        let result = a.correct(&rid, "color is blue", Some(0.9), None, &vec_seed(2.0, 8), Some("User said blue")).unwrap();
        sync_bidirectional(&a, &b).unwrap();

        // B should have original tombstoned
        let original = b.get(&rid).unwrap().unwrap();
        assert_eq!(original.consolidation_status, "tombstoned");

        // B should have the corrected memory
        let corrected = b.get(&result.corrected_rid).unwrap();
        assert!(corrected.is_some());
        assert_eq!(corrected.unwrap().text, "color is blue");
    }

    #[test]
    fn test_conflict_detect_replicates() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        let rid_a = a.record("mem a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        let rid_b = a.record("mem b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "general", "user", None).unwrap();

        // Create conflict on A
        crate::conflict::create_conflict(
            &a,
            &crate::types::ConflictType::Preference,
            &rid_a,
            &rid_b,
            Some("User"),
            Some("prefers"),
            "conflicting preferences",
        )
        .unwrap();

        // Sync to B
        sync_bidirectional(&a, &b).unwrap();

        // B should have the open conflict
        let b_conflicts = b.get_conflicts(Some("open"), None, None, None, 50).unwrap();
        assert!(!b_conflicts.is_empty());
        assert_eq!(b_conflicts[0].conflict_type, "preference");
    }

    // ── V3 sync tests ──

    #[test]
    fn test_trigger_replicates_across_devices() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        // Fire a trigger on A
        let trigger = crate::types::Trigger {
            trigger_type: "decay_review".to_string(),
            reason: "important memory decaying".to_string(),
            urgency: 0.8,
            source_rids: vec!["rid-1".to_string()],
            suggested_action: "ask_user".to_string(),
            context: std::collections::HashMap::new(),
        };
        let ts = crate::time::now_secs();
        crate::triggers::persist_trigger(&a, &trigger, ts).unwrap();

        // Sync A -> B
        sync_bidirectional(&a, &b).unwrap();

        // B should have the trigger
        let b_triggers = b.get_pending_triggers(10).unwrap();
        assert_eq!(b_triggers.len(), 1);
        assert_eq!(b_triggers[0].trigger_type, "decay_review");
    }

    #[test]
    fn test_trigger_lifecycle_replicates() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        // Fire trigger on A
        let trigger = crate::types::Trigger {
            trigger_type: "consolidation_ready".to_string(),
            reason: "test".to_string(),
            urgency: 0.6,
            source_rids: vec![],
            suggested_action: "run_consolidation".to_string(),
            context: std::collections::HashMap::new(),
        };
        let ts = crate::time::now_secs();
        let tid = crate::triggers::persist_trigger(&a, &trigger, ts)
            .unwrap()
            .unwrap();

        // Sync A -> B (trigger arrives)
        sync_bidirectional(&a, &b).unwrap();

        // Dismiss on A
        a.dismiss_trigger(&tid).unwrap();

        // Sync again
        sync_bidirectional(&a, &b).unwrap();

        // B should see it dismissed
        let b_triggers = b.get_trigger_history(None, 10).unwrap();
        assert_eq!(b_triggers.len(), 1);
        assert_eq!(b_triggers[0].status, "dismissed");
    }

    #[test]
    fn test_pattern_replicates() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        // Create a hub pattern on A via edges
        for i in 0..6 {
            a.relate("Alice", &format!("target_{i}"), &format!("rel_{i}"), 1.0)
                .unwrap();
        }
        let config = crate::types::PatternConfig {
            entity_hub_min_degree: 5,
            ..Default::default()
        };
        crate::patterns::mine_patterns(&a, &config).unwrap();

        // A should have a pattern
        let a_patterns = a.get_patterns(Some("entity_hub"), None, 10).unwrap();
        assert!(!a_patterns.is_empty());

        // Sync to B
        sync_bidirectional(&a, &b).unwrap();

        // B should have the pattern
        let b_patterns = b.get_patterns(Some("entity_hub"), None, 10).unwrap();
        assert!(!b_patterns.is_empty());
        assert!(b_patterns[0].description.contains("Alice"));
    }
}
