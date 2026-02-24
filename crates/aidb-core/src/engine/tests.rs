use rusqlite::params;

use crate::hlc::HLCTimestamp;
use crate::types::*;

use super::AIDB;

fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.iter().map(|x| x / norm).collect()
}

fn empty_meta() -> serde_json::Value {
    serde_json::json!({})
}

#[test]
fn test_new_and_stats() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let s = db.stats(None).unwrap();
    assert_eq!(s.active_memories, 0);
    assert_eq!(s.edges, 0);
}

#[test]
fn test_actor_id_auto_generated() {
    let db = AIDB::new(":memory:", 8).unwrap();
    assert_eq!(db.actor_id().len(), 36); // UUIDv7
}

#[test]
fn test_actor_id_explicit() {
    let db = AIDB::new_with_actor(":memory:", 8, "device-A").unwrap();
    assert_eq!(db.actor_id(), "device-A");
}

#[test]
fn test_record_and_get() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db.record("hello world", "episodic", 0.8, 0.0, 604800.0, &empty_meta(), &emb, "default").unwrap();
    assert_eq!(rid.len(), 36);

    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.text, "hello world");
    assert_eq!(mem.memory_type, "episodic");
    assert_eq!(mem.importance, 0.8);
    assert_eq!(mem.consolidation_status, "active");
}

#[test]
fn test_record_updates_stats() {
    let db = AIDB::new(":memory:", 8).unwrap();
    db.record("one", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    db.record("two", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default").unwrap();
    assert_eq!(db.stats(None).unwrap().active_memories, 2);
}

#[test]
fn test_recall_basic() {
    let db = AIDB::new(":memory:", 8).unwrap();
    db.record("the cat sat on the mat", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    db.record("dogs are loyal friends", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(5.0, 8), "default").unwrap();
    db.record("cats love warm places", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.1, 8), "default").unwrap();

    let results = db.recall(&vec_seed(1.0, 8), 2, None, None, false, false, None, false, None).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_recall_empty() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let results = db.recall(&vec_seed(1.0, 8), 5, None, None, false, false, None, false, None).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_relate_and_get_edges() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let eid = db.relate("Alice", "Bob", "knows", 1.0).unwrap();
    assert_eq!(eid.len(), 36);

    let edges = db.get_edges("Alice").unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].src, "Alice");
    assert_eq!(edges[0].dst, "Bob");
}

#[test]
fn test_forget() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rid = db.record("forget me", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    assert!(db.forget(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.consolidation_status, "tombstoned");
}

#[test]
fn test_forget_nonexistent() {
    let db = AIDB::new(":memory:", 8).unwrap();
    assert!(!db.forget("nonexistent").unwrap());
}

#[test]
fn test_decay_fresh() {
    let db = AIDB::new(":memory:", 8).unwrap();
    db.record("fresh", "episodic", 0.9, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    let decayed = db.decay(0.01).unwrap();
    assert!(decayed.is_empty());
}

#[test]
fn test_oplog_has_hlc() {
    let db = AIDB::new(":memory:", 8).unwrap();
    db.record("test", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();

    let hlc_bytes: Vec<u8> = db.conn.query_row(
        "SELECT hlc FROM oplog ORDER BY rowid DESC LIMIT 1",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(hlc_bytes.len(), 16);

    let ts = HLCTimestamp::from_bytes(&hlc_bytes).unwrap();
    assert!(ts.millis > 0);
}

#[test]
fn test_oplog_has_embedding_hash() {
    let db = AIDB::new(":memory:", 8).unwrap();
    db.record("test", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();

    // The record op should have an embedding_hash
    let hash: Vec<u8> = db.conn.query_row(
        "SELECT embedding_hash FROM oplog WHERE op_type = 'record' LIMIT 1",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(hash.len(), 32); // BLAKE3 output is 32 bytes
}

#[test]
fn test_oplog_enriched_payload() {
    let db = AIDB::new(":memory:", 8).unwrap();
    db.record("test payload", "semantic", 0.7, 0.3, 1000.0, &serde_json::json!({"key": "val"}), &vec_seed(1.0, 8), "default").unwrap();

    let payload_str: String = db.conn.query_row(
        "SELECT payload FROM oplog WHERE op_type = 'record' LIMIT 1",
        [],
        |row| row.get(0),
    ).unwrap();
    let payload: serde_json::Value = serde_json::from_str(&payload_str).unwrap();

    assert_eq!(payload["type"], "semantic");
    assert_eq!(payload["text"], "test payload");
    assert_eq!(payload["importance"], 0.7);
    assert_eq!(payload["valence"], 0.3);
    assert_eq!(payload["half_life"], 1000.0);
    assert!(payload["rid"].is_string());
    assert!(payload["created_at"].is_number());
    assert!(payload["metadata"]["key"] == "val");
}

#[test]
fn test_schema_v3_has_conflicts_table() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let count: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='conflicts'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_resolve_keep_a() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rid_a = db.record("birthday March 5", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    let rid_b = db.record("birthday March 15", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default").unwrap();

    let conflict = crate::conflict::create_conflict(
        &db, &crate::types::ConflictType::IdentityFact, &rid_a, &rid_b,
        Some("User"), Some("birthday"), "conflicting birthdays",
    ).unwrap();

    let result = db.resolve_conflict(&conflict.conflict_id, "keep_a", Some(&rid_a), None, Some("User confirmed March 5")).unwrap();
    assert!(result.loser_tombstoned);

    let mem_b = db.get(&rid_b).unwrap().unwrap();
    assert_eq!(mem_b.consolidation_status, "tombstoned");

    let resolved = db.get_conflict(&conflict.conflict_id).unwrap().unwrap();
    assert_eq!(resolved.status, "resolved");
    assert_eq!(resolved.strategy.as_deref(), Some("keep_a"));
}

#[test]
fn test_resolve_keep_both() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rid_a = db.record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    let rid_b = db.record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default").unwrap();

    let conflict = crate::conflict::create_conflict(
        &db, &crate::types::ConflictType::Minor, &rid_a, &rid_b, None, None, "test",
    ).unwrap();
    let result = db.resolve_conflict(&conflict.conflict_id, "keep_both", None, None, None).unwrap();
    assert!(!result.loser_tombstoned);

    let mem_a = db.get(&rid_a).unwrap().unwrap();
    let mem_b = db.get(&rid_b).unwrap().unwrap();
    assert_eq!(mem_a.consolidation_status, "active");
    assert_eq!(mem_b.consolidation_status, "active");
}

#[test]
fn test_correct_memory() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rid = db.record("favorite color is green", "episodic", 0.7, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();

    let result = db.correct(
        &rid, "favorite color is blue", Some(0.9), None, &vec_seed(2.0, 8),
        Some("User corrected their favorite color"),
    ).unwrap();

    assert!(result.original_tombstoned);

    let original = db.get(&rid).unwrap().unwrap();
    assert_eq!(original.consolidation_status, "tombstoned");

    let corrected = db.get(&result.corrected_rid).unwrap().unwrap();
    assert_eq!(corrected.text, "favorite color is blue");
    assert_eq!(corrected.importance, 0.9);
}

#[test]
fn test_get_conflicts_filtered() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rid_a = db.record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    let rid_b = db.record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default").unwrap();
    let rid_c = db.record("c", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(3.0, 8), "default").unwrap();

    crate::conflict::create_conflict(
        &db, &crate::types::ConflictType::IdentityFact, &rid_a, &rid_b,
        Some("User"), Some("birthday"), "test 1",
    ).unwrap();
    crate::conflict::create_conflict(
        &db, &crate::types::ConflictType::Preference, &rid_b, &rid_c,
        Some("User"), Some("prefers"), "test 2",
    ).unwrap();

    let all = db.get_conflicts(None, None, None, None, 50).unwrap();
    assert_eq!(all.len(), 2);

    let identity_only = db.get_conflicts(None, Some("identity_fact"), None, None, 50).unwrap();
    assert_eq!(identity_only.len(), 1);

    let critical = db.get_conflicts(None, None, None, Some("critical"), 50).unwrap();
    assert_eq!(critical.len(), 1);
}

#[test]
fn test_dismiss_conflict() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rid_a = db.record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    let rid_b = db.record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default").unwrap();

    let conflict = crate::conflict::create_conflict(
        &db, &crate::types::ConflictType::Minor, &rid_a, &rid_b, None, None, "test",
    ).unwrap();

    db.dismiss_conflict(&conflict.conflict_id, Some("Not really a conflict")).unwrap();

    let c = db.get_conflict(&conflict.conflict_id).unwrap().unwrap();
    assert_eq!(c.status, "dismissed");
}

#[test]
fn test_stats_include_conflicts() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let s = db.stats(None).unwrap();
    assert_eq!(s.open_conflicts, 0);
    assert_eq!(s.resolved_conflicts, 0);

    let rid_a = db.record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    let rid_b = db.record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default").unwrap();
    crate::conflict::create_conflict(
        &db, &crate::types::ConflictType::Minor, &rid_a, &rid_b, None, None, "test",
    ).unwrap();

    let s = db.stats(None).unwrap();
    assert_eq!(s.open_conflicts, 1);
    assert_eq!(s.resolved_conflicts, 0);
}

// ── V3 Cognition tests ──

#[test]
fn test_schema_v4_has_trigger_log_and_patterns() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let count: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('trigger_log', 'patterns')",
        [], |row| row.get(0),
    ).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn test_think_empty_db() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let config = ThinkConfig {
        run_consolidation: false,
        run_conflict_scan: false,
        run_pattern_mining: false,
        ..Default::default()
    };
    let result = db.think(&config).unwrap();
    assert!(result.triggers.is_empty());
    assert_eq!(result.consolidation_count, 0);
    assert_eq!(result.conflicts_found, 0);
    assert!(result.duration_ms < 5000);
}

#[test]
fn test_think_with_decayed_memories() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rid = db.record("important deadline", "episodic", 0.9, 0.0, 100.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();

    // Backdate last_access
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
    db.conn().execute(
        "UPDATE memories SET last_access = ?1 WHERE rid = ?2",
        rusqlite::params![ts - 10000.0, rid],
    ).unwrap();

    let config = ThinkConfig {
        run_consolidation: false,
        run_conflict_scan: false,
        run_pattern_mining: false,
        ..Default::default()
    };
    let result = db.think(&config).unwrap();
    assert!(!result.triggers.is_empty());
    assert_eq!(result.triggers[0].trigger_type, "decay_review");
}

#[test]
fn test_think_records_last_think_at() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let config = ThinkConfig {
        run_consolidation: false,
        run_conflict_scan: false,
        run_pattern_mining: false,
        ..Default::default()
    };
    db.think(&config).unwrap();

    let val: String = db.conn().query_row(
        "SELECT value FROM meta WHERE key = 'last_think_at'",
        [], |row| row.get(0),
    ).unwrap();
    let ts: f64 = val.parse().unwrap();
    assert!(ts > 0.0);
}

#[test]
fn test_trigger_lifecycle() {
    let db = AIDB::new(":memory:", 8).unwrap();

    // Create a trigger via persistence
    let trigger = crate::types::Trigger {
        trigger_type: "decay_review".to_string(),
        reason: "test".to_string(),
        urgency: 0.8,
        source_rids: vec!["rid-1".to_string()],
        suggested_action: "test".to_string(),
        context: std::collections::HashMap::new(),
    };
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
    let tid = crate::triggers::persist_trigger(&db, &trigger, ts).unwrap().unwrap();

    // Verify pending
    let pending = db.get_pending_triggers(10).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].status, "pending");

    // Deliver
    assert!(db.deliver_trigger(&tid).unwrap());
    let history = db.get_trigger_history(None, 10).unwrap();
    assert_eq!(history[0].status, "delivered");

    // Acknowledge
    assert!(db.acknowledge_trigger(&tid).unwrap());

    // Act
    assert!(db.act_on_trigger(&tid).unwrap());
    let history = db.get_trigger_history(None, 10).unwrap();
    assert_eq!(history[0].status, "acted");
}

#[test]
fn test_stats_include_triggers_and_patterns() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let s = db.stats(None).unwrap();
    assert_eq!(s.pending_triggers, 0);
    assert_eq!(s.active_patterns, 0);
}

// ── Graph-augmented recall: invariant & regression tests ──

#[test]
fn test_entity_type_stored_on_relate() {
    let db = AIDB::new(":memory:", 8).unwrap();
    db.relate("Sarah", "data pipeline", "leads", 1.0).unwrap();
    db.relate("FAISS", "recommendation engine", "used_in", 1.0).unwrap();
    db.relate("Mike", "ONNX", "built_with", 1.0).unwrap();

    // Sarah → person, FAISS → tech, data pipeline → unknown, Mike → person, ONNX → tech
    let etype: String = db.conn.query_row(
        "SELECT entity_type FROM entities WHERE name = 'Sarah'", [], |r| r.get(0),
    ).unwrap();
    assert_eq!(etype, "person");

    let etype: String = db.conn.query_row(
        "SELECT entity_type FROM entities WHERE name = 'FAISS'", [], |r| r.get(0),
    ).unwrap();
    assert_eq!(etype, "tech");

    let etype: String = db.conn.query_row(
        "SELECT entity_type FROM entities WHERE name = 'data pipeline'", [], |r| r.get(0),
    ).unwrap();
    assert_eq!(etype, "unknown");

    let etype: String = db.conn.query_row(
        "SELECT entity_type FROM entities WHERE name = 'Mike'", [], |r| r.get(0),
    ).unwrap();
    assert_eq!(etype, "person");
}

#[test]
fn test_recall_deterministic_with_skip_reinforce() {
    let db = AIDB::new(":memory:", 8).unwrap();
    for i in 0..10 {
        db.record(&format!("memory {i}"), "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(i as f32, 8), "default").unwrap();
    }
    let query = vec_seed(3.0, 8);

    let r1 = db.recall(&query, 5, None, None, false, false, None, true, None).unwrap();
    let r2 = db.recall(&query, 5, None, None, false, false, None, true, None).unwrap();
    let r3 = db.recall(&query, 5, None, None, false, false, None, true, None).unwrap();

    // Same RIDs in same order every time
    let rids1: Vec<&str> = r1.iter().map(|r| r.rid.as_str()).collect();
    let rids2: Vec<&str> = r2.iter().map(|r| r.rid.as_str()).collect();
    let rids3: Vec<&str> = r3.iter().map(|r| r.rid.as_str()).collect();
    assert_eq!(rids1, rids2);
    assert_eq!(rids2, rids3);

    // Scores very close (tiny drift from wall-clock recency between calls)
    for i in 0..5 {
        assert!((r1[i].score - r2[i].score).abs() < 1e-4,
            "score drift too large between calls: {} vs {}", r1[i].score, r2[i].score);
    }
}

#[test]
fn test_reinforce_mutates_but_skip_does_not() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rid = db.record("test", "episodic", 0.5, 0.0, 1000.0,
        &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    let original_hl = db.get(&rid).unwrap().unwrap().half_life;

    // skip_reinforce=true should NOT change half_life
    db.recall(&vec_seed(1.0, 8), 1, None, None, false, false, None, true, None).unwrap();
    let after_skip = db.get(&rid).unwrap().unwrap().half_life;
    assert!((original_hl - after_skip).abs() < 1e-10);

    // skip_reinforce=false SHOULD change half_life
    db.recall(&vec_seed(1.0, 8), 1, None, None, false, false, None, false, None).unwrap();
    let after_reinforce = db.get(&rid).unwrap().unwrap().half_life;
    assert!(after_reinforce > original_hl);
}

#[test]
fn test_graph_expansion_off_no_graph_results() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let r1 = db.record("Alice discussed plan", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    let r2 = db.record("Bob reviewed code", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(5.0, 8), "default").unwrap();
    db.relate("Alice", "Bob", "knows", 1.0).unwrap();
    db.link_memory_entity(&r1, "Alice").unwrap();
    db.link_memory_entity(&r2, "Bob").unwrap();

    // expand_entities=false: no graph_proximity should be set
    let results = db.recall(&vec_seed(1.0, 8), 10, None, None, false, false,
        Some("Alice"), false, None).unwrap();
    for r in &results {
        assert!((r.scores.graph_proximity - 0.0).abs() < 1e-10,
            "graph_proximity should be 0.0 when expansion is disabled");
    }
}

#[test]
fn test_graph_expansion_on_boosts_connected_memory() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let r1 = db.record("Alice discussed the project plan", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    let r2 = db.record("Bob reviewed the code", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(5.0, 8), "default").unwrap();
    db.relate("Alice", "Bob", "knows", 1.0).unwrap();
    db.link_memory_entity(&r1, "Alice").unwrap();
    db.link_memory_entity(&r2, "Bob").unwrap();

    // expand_entities=true with query mentioning "Alice"
    let results = db.recall(&vec_seed(1.0, 8), 10, None, None, false, true,
        Some("What is Alice working on?"), true, None).unwrap();

    // The Alice memory should have graph_proximity > 0
    let alice_result = results.iter().find(|r| r.rid == r1).unwrap();
    assert!(alice_result.scores.graph_proximity > 0.0,
        "Alice memory should have graph proximity when expansion is on");
}

#[test]
fn test_backfill_uses_word_boundaries() {
    let db = AIDB::new(":memory:", 8).unwrap();
    // Create an entity "data"
    db.relate("data", "pipeline", "part_of", 1.0).unwrap();

    // Create memories: one with "data" as a word, one with "database" (contains "data")
    let r1 = db.record("the data is clean", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    let r2 = db.record("the database is fast", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(2.0, 8), "default").unwrap();

    let _count = db.backfill_memory_entities().unwrap();

    // Check: r1 should be linked to "data", r2 should NOT
    let linked_to_data: Vec<String> = db.conn.prepare(
        "SELECT memory_rid FROM memory_entities WHERE entity_name = 'data'"
    ).unwrap().query_map([], |row| row.get(0)).unwrap()
        .collect::<std::result::Result<Vec<_>, _>>().unwrap();

    assert!(linked_to_data.contains(&r1), "memory with 'data' as word should be linked");
    assert!(!linked_to_data.contains(&r2), "memory with 'database' should NOT be linked (word boundary)");
}

#[test]
fn test_recall_scores_bounded() {
    // All recall scores should be non-negative and reasonably bounded
    let db = AIDB::new(":memory:", 8).unwrap();
    for i in 0..10 {
        db.record(&format!("memory {i}"), "episodic",
            (i as f64) * 0.1, // importance 0..0.9
            ((i as f64) - 5.0) * 0.2, // valence -1.0..0.8
            604800.0, &empty_meta(), &vec_seed(i as f32, 8), "default").unwrap();
    }

    let results = db.recall(&vec_seed(5.0, 8), 10, None, None, false, false, None, true, None).unwrap();
    for r in &results {
        assert!(r.score >= 0.0, "score should be non-negative, got {}", r.score);
        assert!(r.score < 5.0, "score should be reasonably bounded, got {}", r.score);
        assert!(r.scores.similarity >= -1.0 && r.scores.similarity <= 1.0);
        assert!(r.scores.decay >= 0.0 && r.scores.decay <= 1.0);
        assert!(r.scores.recency >= 0.0 && r.scores.recency <= 1.0);
    }
}

#[test]
fn test_link_memory_entity_idempotent() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rid = db.record("test", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    db.relate("Alice", "Bob", "knows", 1.0).unwrap();

    // Link twice — should not error or create duplicates
    db.link_memory_entity(&rid, "Alice").unwrap();
    db.link_memory_entity(&rid, "Alice").unwrap();

    let count: i64 = db.conn.query_row(
        "SELECT COUNT(*) FROM memory_entities WHERE memory_rid = ?1 AND entity_name = 'Alice'",
        params![rid], |r| r.get(0),
    ).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_schema_v5_has_memory_entities() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let count: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_entities'",
        [], |r| r.get(0),
    ).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_recall_top_k_respected_with_graph_expansion() {
    let db = AIDB::new(":memory:", 8).unwrap();
    // Create a web of interconnected memories
    for i in 0..20 {
        let rid = db.record(&format!("memory about topic {i}"), "episodic",
            0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(i as f32, 8), "default").unwrap();
        let entity = format!("Entity{i}");
        db.relate(&entity, &format!("Entity{}", (i + 1) % 20), "related_to", 1.0).unwrap();
        db.link_memory_entity(&rid, &entity).unwrap();
    }

    let results = db.recall(&vec_seed(0.0, 8), 5, None, None, false, true,
        Some("Entity0 topic"), true, None).unwrap();

    // top_k=5 must be respected even with graph expansion
    assert!(results.len() <= 5, "results should not exceed top_k=5, got {}", results.len());
}

// ── V4: Storage & Performance tests ──

#[test]
fn test_schema_v6_has_storage_tier() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rid = db.record("tier test", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.storage_tier, "hot");
}

#[test]
fn test_schema_v7_has_fts5_and_join_tables() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let conn = db.conn();

    // FTS5 virtual table exists — insert then search
    let _rid = db.record("The quick brown fox jumps over the lazy dog", "episodic",
        0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();

    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'quick brown'",
        [], |row| row.get(0),
    ).unwrap();
    assert_eq!(count, 1, "FTS5 should index inserted memory");

    // Join tables exist
    let _: i64 = conn.query_row(
        "SELECT COUNT(*) FROM trigger_source_rids", [], |row| row.get(0),
    ).unwrap();
    let _: i64 = conn.query_row(
        "SELECT COUNT(*) FROM pattern_evidence", [], |row| row.get(0),
    ).unwrap();
    let _: i64 = conn.query_row(
        "SELECT COUNT(*) FROM pattern_entities", [], |row| row.get(0),
    ).unwrap();

    // Schema version is 8
    let ver: String = conn.query_row(
        "SELECT value FROM meta WHERE key = 'schema_version'", [], |row| row.get(0),
    ).unwrap();
    assert_eq!(ver, "8");
}

#[test]
fn test_fts5_search_multiple_memories() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let conn = db.conn();

    db.record("Alice loves Rust programming", "semantic",
        0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();
    db.record("Bob prefers Python scripting", "semantic",
        0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(0.5, 8), "default").unwrap();
    db.record("Alice and Bob work on Rust projects", "episodic",
        0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(0.3, 8), "default").unwrap();

    // Search for "Rust" should match 2 memories
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'rust'",
        [], |row| row.get(0),
    ).unwrap();
    assert_eq!(count, 2, "FTS5 should find 2 memories containing 'rust'");

    // Search for "Alice" should match 2 memories
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'alice'",
        [], |row| row.get(0),
    ).unwrap();
    assert_eq!(count, 2, "FTS5 should find 2 memories containing 'alice'");

    // Search for "Python" should match 1
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'python'",
        [], |row| row.get(0),
    ).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_archive_memory() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rid = db.record("to archive", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();

    // Archive
    assert!(db.archive(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.storage_tier, "cold");

    // Verify removed from vec_memories (recall should not find it)
    let results = db.recall(&vec_seed(1.0, 8), 10, None, None, false, false, None, true, None).unwrap();
    assert!(results.iter().all(|r| r.rid != rid), "archived memory should not appear in recall");

    // Stats should show archived
    assert_eq!(db.stats(None).unwrap().archived_memories, 1);
}

#[test]
fn test_hydrate_memory() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(2.0, 8);
    let rid = db.record("to hydrate", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &emb, "default").unwrap();

    // Archive then hydrate
    db.archive(&rid).unwrap();
    assert!(db.hydrate(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.storage_tier, "hot");

    // Should be back in recall
    let results = db.recall(&emb, 10, None, None, false, false, None, true, None).unwrap();
    assert!(results.iter().any(|r| r.rid == rid), "hydrated memory should appear in recall");

    // Stats
    assert_eq!(db.stats(None).unwrap().archived_memories, 0);
}

#[test]
fn test_archive_idempotent() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rid = db.record("idempotent", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default").unwrap();

    assert!(db.archive(&rid).unwrap());
    assert!(!db.archive(&rid).unwrap()); // Already cold
}

#[test]
fn test_record_batch() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let inputs: Vec<RecordInput> = (0..10).map(|i| RecordInput {
        text: format!("batch memory {i}"),
        memory_type: "episodic".to_string(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: serde_json::json!({}),
        embedding: vec_seed(i as f32, 8),
        namespace: "default".to_string(),
    }).collect();

    let rids = db.record_batch(&inputs).unwrap();
    assert_eq!(rids.len(), 10);

    // All retrievable
    for rid in &rids {
        assert!(db.get(rid).unwrap().is_some());
    }
    assert_eq!(db.stats(None).unwrap().active_memories, 10);
}

#[test]
fn test_record_batch_empty() {
    let db = AIDB::new(":memory:", 8).unwrap();
    let rids = db.record_batch(&[]).unwrap();
    assert!(rids.is_empty());
}

#[test]
fn test_evict() {
    let db = AIDB::new(":memory:", 8).unwrap();
    // Seed 20 memories
    for i in 0..20 {
        db.record(&format!("evict mem {i}"), "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(i as f32, 8), "default").unwrap();
    }
    assert_eq!(db.stats(None).unwrap().active_memories, 20);

    // Evict to keep 10
    let archived = db.evict(10).unwrap();
    assert_eq!(archived.len(), 10);

    let stats = db.stats(None).unwrap();
    assert_eq!(stats.archived_memories, 10);

    // Recall should only find hot memories
    let results = db.recall(&vec_seed(0.0, 8), 20, None, None, false, false, None, true, None).unwrap();
    for r in &results {
        assert!(!archived.contains(&r.rid), "evicted memory should not be in recall");
    }
}

#[test]
fn test_evict_no_action_when_under_limit() {
    let db = AIDB::new(":memory:", 8).unwrap();
    for i in 0..5 {
        db.record(&format!("small db {i}"), "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(i as f32, 8), "default").unwrap();
    }
    let archived = db.evict(10).unwrap();
    assert!(archived.is_empty());
}
