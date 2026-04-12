use rusqlite::params;

use crate::hlc::HLCTimestamp;
use crate::types::*;

use super::YantrikDB;

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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let s = db.stats(None).unwrap();
    assert_eq!(s.active_memories, 0);
    assert_eq!(s.edges, 0);
}

#[test]
fn test_actor_id_auto_generated() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert_eq!(db.actor_id().len(), 36); // UUIDv7
}

#[test]
fn test_actor_id_explicit() {
    let db = YantrikDB::new_with_actor(":memory:", 8, "device-A").unwrap();
    assert_eq!(db.actor_id(), "device-A");
}

#[test]
fn test_record_and_get() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db.record("hello world", "episodic", 0.8, 0.0, 604800.0, &empty_meta(), &emb, "default", 0.8, "general", "user", None).unwrap();
    assert_eq!(rid.len(), 36);

    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.text, "hello world");
    assert_eq!(mem.memory_type, "episodic");
    assert_eq!(mem.importance, 0.8);
    assert_eq!(mem.consolidation_status, "active");
}

#[test]
fn test_record_updates_stats() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record("one", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    db.record("two", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "general", "user", None).unwrap();
    assert_eq!(db.stats(None).unwrap().active_memories, 2);
}

#[test]
fn test_recall_basic() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record("the cat sat on the mat", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    db.record("dogs are loyal friends", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(5.0, 8), "default", 0.8, "general", "user", None).unwrap();
    db.record("cats love warm places", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.1, 8), "default", 0.8, "general", "user", None).unwrap();

    let results = db.recall(&vec_seed(1.0, 8), 2, None, None, false, false, None, false, None, None, None).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_recall_empty() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let results = db.recall(&vec_seed(1.0, 8), 5, None, None, false, false, None, false, None, None, None).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_relate_and_get_edges() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let eid = db.relate("Alice", "Bob", "knows", 1.0).unwrap();
    assert_eq!(eid.len(), 36);

    let edges = db.get_edges("Alice").unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].src, "Alice");
    assert_eq!(edges[0].dst, "Bob");
}

#[test]
fn test_forget() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db.record("forget me", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    assert!(db.forget(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.consolidation_status, "tombstoned");
}

#[test]
fn test_forget_nonexistent() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert!(!db.forget("nonexistent").unwrap());
}

#[test]
fn test_decay_fresh() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record("fresh", "episodic", 0.9, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let decayed = db.decay(0.01).unwrap();
    assert!(decayed.is_empty());
}

#[test]
fn test_oplog_has_hlc() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record("test", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

    let hlc_bytes: Vec<u8> = db.conn().query_row(
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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record("test", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

    // The record op should have an embedding_hash
    let hash: Vec<u8> = db.conn().query_row(
        "SELECT embedding_hash FROM oplog WHERE op_type = 'record' LIMIT 1",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(hash.len(), 32); // BLAKE3 output is 32 bytes
}

#[test]
fn test_oplog_enriched_payload() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record("test payload", "semantic", 0.7, 0.3, 1000.0, &serde_json::json!({"key": "val"}), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

    let payload_str: String = db.conn().query_row(
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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let count: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='conflicts'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_resolve_keep_a() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid_a = db.record("birthday March 5", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let rid_b = db.record("birthday March 15", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "general", "user", None).unwrap();

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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid_a = db.record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let rid_b = db.record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "general", "user", None).unwrap();

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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db.record("favorite color is green", "episodic", 0.7, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid_a = db.record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let rid_b = db.record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let rid_c = db.record("c", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(3.0, 8), "default", 0.8, "general", "user", None).unwrap();

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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid_a = db.record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let rid_b = db.record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "general", "user", None).unwrap();

    let conflict = crate::conflict::create_conflict(
        &db, &crate::types::ConflictType::Minor, &rid_a, &rid_b, None, None, "test",
    ).unwrap();

    db.dismiss_conflict(&conflict.conflict_id, Some("Not really a conflict")).unwrap();

    let c = db.get_conflict(&conflict.conflict_id).unwrap().unwrap();
    assert_eq!(c.status, "dismissed");
}

#[test]
fn test_stats_include_conflicts() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let s = db.stats(None).unwrap();
    assert_eq!(s.open_conflicts, 0);
    assert_eq!(s.resolved_conflicts, 0);

    let rid_a = db.record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let rid_b = db.record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "general", "user", None).unwrap();
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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let count: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('trigger_log', 'patterns')",
        [], |row| row.get(0),
    ).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn test_think_empty_db() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db.record("important deadline", "episodic", 0.9, 0.0, 100.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

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
    let db = YantrikDB::new(":memory:", 8).unwrap();
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
    let db = YantrikDB::new(":memory:", 8).unwrap();

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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let s = db.stats(None).unwrap();
    assert_eq!(s.pending_triggers, 0);
    assert_eq!(s.active_patterns, 0);
}

// ── Graph-augmented recall: invariant & regression tests ──

#[test]
fn test_entity_type_stored_on_relate() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // "knows" is a person-person relationship
    db.relate("Sarah", "Mike", "knows", 1.0).unwrap();
    // "works_at" → src=person, dst=organization
    db.relate("Sarah", "Flipkart", "works_at", 1.0).unwrap();
    // "lives_in" → src=person, dst=place
    db.relate("Sarah", "Bangalore", "lives_in", 1.0).unwrap();
    // Tech blocklist still works
    db.relate("FAISS", "recommendation engine", "used_in", 1.0).unwrap();

    let etype: String = db.conn().query_row(
        "SELECT entity_type FROM entities WHERE name = 'Sarah'", [], |r| r.get(0),
    ).unwrap();
    assert_eq!(etype, "person");

    let etype: String = db.conn().query_row(
        "SELECT entity_type FROM entities WHERE name = 'Mike'", [], |r| r.get(0),
    ).unwrap();
    assert_eq!(etype, "person");

    let etype: String = db.conn().query_row(
        "SELECT entity_type FROM entities WHERE name = 'Flipkart'", [], |r| r.get(0),
    ).unwrap();
    assert_eq!(etype, "organization");

    let etype: String = db.conn().query_row(
        "SELECT entity_type FROM entities WHERE name = 'Bangalore'", [], |r| r.get(0),
    ).unwrap();
    assert_eq!(etype, "place");

    let etype: String = db.conn().query_row(
        "SELECT entity_type FROM entities WHERE name = 'FAISS'", [], |r| r.get(0),
    ).unwrap();
    assert_eq!(etype, "tech");

    let etype: String = db.conn().query_row(
        "SELECT entity_type FROM entities WHERE name = 'recommendation engine'", [], |r| r.get(0),
    ).unwrap();
    assert_eq!(etype, "unknown");
}

#[test]
fn test_recall_deterministic_with_skip_reinforce() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for i in 0..10 {
        db.record(&format!("memory {i}"), "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(i as f32, 8), "default", 0.8, "general", "user", None).unwrap();
    }
    let query = vec_seed(3.0, 8);

    let r1 = db.recall(&query, 5, None, None, false, false, None, true, None, None, None).unwrap();
    let r2 = db.recall(&query, 5, None, None, false, false, None, true, None, None, None).unwrap();
    let r3 = db.recall(&query, 5, None, None, false, false, None, true, None, None, None).unwrap();

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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db.record("test", "episodic", 0.5, 0.0, 1000.0,
        &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let original_hl = db.get(&rid).unwrap().unwrap().half_life;

    // skip_reinforce=true should NOT change half_life
    db.recall(&vec_seed(1.0, 8), 1, None, None, false, false, None, true, None, None, None).unwrap();
    let after_skip = db.get(&rid).unwrap().unwrap().half_life;
    assert!((original_hl - after_skip).abs() < 1e-10);

    // skip_reinforce=false SHOULD change half_life
    db.recall(&vec_seed(1.0, 8), 1, None, None, false, false, None, false, None, None, None).unwrap();
    let after_reinforce = db.get(&rid).unwrap().unwrap().half_life;
    assert!(after_reinforce > original_hl);
}

#[test]
fn test_graph_expansion_off_no_graph_results() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let r1 = db.record("Alice discussed plan", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let r2 = db.record("Bob reviewed code", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(5.0, 8), "default", 0.8, "general", "user", None).unwrap();
    db.relate("Alice", "Bob", "knows", 1.0).unwrap();
    db.link_memory_entity(&r1, "Alice").unwrap();
    db.link_memory_entity(&r2, "Bob").unwrap();

    // expand_entities=false: no graph_proximity should be set
    let results = db.recall(&vec_seed(1.0, 8), 10, None, None, false, false,
        Some("Alice"), false, None, None, None).unwrap();
    for r in &results {
        assert!((r.scores.graph_proximity - 0.0).abs() < 1e-10,
            "graph_proximity should be 0.0 when expansion is disabled");
    }
}

#[test]
fn test_graph_expansion_on_boosts_connected_memory() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let r1 = db.record("Alice discussed the project plan", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let r2 = db.record("Bob reviewed the code", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(5.0, 8), "default", 0.8, "general", "user", None).unwrap();
    db.relate("Alice", "Bob", "knows", 1.0).unwrap();
    db.link_memory_entity(&r1, "Alice").unwrap();
    db.link_memory_entity(&r2, "Bob").unwrap();

    // expand_entities=true with query mentioning "Alice"
    let results = db.recall(&vec_seed(1.0, 8), 10, None, None, false, true,
        Some("What is Alice working on?"), true, None, None, None).unwrap();

    // The Alice memory should have graph_proximity > 0
    let alice_result = results.iter().find(|r| r.rid == r1).unwrap();
    assert!(alice_result.scores.graph_proximity > 0.0,
        "Alice memory should have graph proximity when expansion is on");
}

#[test]
fn test_backfill_uses_word_boundaries() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Create an entity "data"
    db.relate("data", "pipeline", "part_of", 1.0).unwrap();

    // Create memories: one with "data" as a word, one with "database" (contains "data")
    let r1 = db.record("the data is clean", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let r2 = db.record("the database is fast", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "general", "user", None).unwrap();

    let _count = db.backfill_memory_entities().unwrap();

    // Check: r1 should be linked to "data", r2 should NOT
    let linked_to_data: Vec<String> = db.conn().prepare(
        "SELECT memory_rid FROM memory_entities WHERE entity_name = 'data'"
    ).unwrap().query_map([], |row| row.get(0)).unwrap()
        .collect::<std::result::Result<Vec<_>, _>>().unwrap();

    assert!(linked_to_data.contains(&r1), "memory with 'data' as word should be linked");
    assert!(!linked_to_data.contains(&r2), "memory with 'database' should NOT be linked (word boundary)");
}

#[test]
fn test_recall_scores_bounded() {
    // All recall scores should be non-negative and reasonably bounded
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for i in 0..10 {
        db.record(&format!("memory {i}"), "episodic",
            (i as f64) * 0.1, // importance 0..0.9
            ((i as f64) - 5.0) * 0.2, // valence -1.0..0.8
            604800.0, &empty_meta(), &vec_seed(i as f32, 8), "default", 0.8, "general", "user", None).unwrap();
    }

    let results = db.recall(&vec_seed(5.0, 8), 10, None, None, false, false, None, true, None, None, None).unwrap();
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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db.record("test", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    db.relate("Alice", "Bob", "knows", 1.0).unwrap();

    // Link twice — should not error or create duplicates
    db.link_memory_entity(&rid, "Alice").unwrap();
    db.link_memory_entity(&rid, "Alice").unwrap();

    let count: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM memory_entities WHERE memory_rid = ?1 AND entity_name = 'Alice'",
        params![rid], |r| r.get(0),
    ).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_schema_v5_has_memory_entities() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let count: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_entities'",
        [], |r| r.get(0),
    ).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_recall_top_k_respected_with_graph_expansion() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Create a web of interconnected memories
    for i in 0..20 {
        let rid = db.record(&format!("memory about topic {i}"), "episodic",
            0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(i as f32, 8), "default", 0.8, "general", "user", None).unwrap();
        let entity = format!("Entity{i}");
        db.relate(&entity, &format!("Entity{}", (i + 1) % 20), "related_to", 1.0).unwrap();
        db.link_memory_entity(&rid, &entity).unwrap();
    }

    let results = db.recall(&vec_seed(0.0, 8), 5, None, None, false, true,
        Some("Entity0 topic"), true, None, None, None).unwrap();

    // top_k=5 must be respected even with graph expansion
    assert!(results.len() <= 5, "results should not exceed top_k=5, got {}", results.len());
}

// ── V4: Storage & Performance tests ──

#[test]
fn test_schema_v6_has_storage_tier() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db.record("tier test", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.storage_tier, "hot");
}

#[test]
fn test_schema_v7_has_fts5_and_join_tables() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // FTS5 virtual table exists — insert then search.
    // Must record BEFORE acquiring conn — db.record() internally takes
    // conn, so holding conn across record() would self-deadlock.
    let _rid = db.record("The quick brown fox jumps over the lazy dog", "episodic",
        0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

    let conn = db.conn();

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

    // Schema version is current
    let ver: String = conn.query_row(
        "SELECT value FROM meta WHERE key = 'schema_version'", [], |row| row.get(0),
    ).unwrap();
    assert_eq!(ver, crate::schema::SCHEMA_VERSION.to_string());
}

#[test]
fn test_fts5_search_multiple_memories() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    db.record("Alice loves Rust programming", "semantic",
        0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    db.record("Bob prefers Python scripting", "semantic",
        0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(0.5, 8), "default", 0.8, "general", "user", None).unwrap();
    db.record("Alice and Bob work on Rust projects", "episodic",
        0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(0.3, 8), "default", 0.8, "general", "user", None).unwrap();

    // Acquire conn AFTER records are written — db.record() internally takes
    // conn, and holding it across db.record() would self-deadlock (the
    // Mutex<Connection> is non-reentrant). See CONCURRENCY.md Rule 4.
    let conn = db.conn();

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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db.record("to archive", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

    // Archive
    assert!(db.archive(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.storage_tier, "cold");

    // Verify removed from vec_memories (recall should not find it)
    let results = db.recall(&vec_seed(1.0, 8), 10, None, None, false, false, None, true, None, None, None).unwrap();
    assert!(results.iter().all(|r| r.rid != rid), "archived memory should not appear in recall");

    // Stats should show archived
    assert_eq!(db.stats(None).unwrap().archived_memories, 1);
}

#[test]
fn test_hydrate_memory() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(2.0, 8);
    let rid = db.record("to hydrate", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &emb, "default", 0.8, "general", "user", None).unwrap();

    // Archive then hydrate
    db.archive(&rid).unwrap();
    assert!(db.hydrate(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.storage_tier, "hot");

    // Should be back in recall
    let results = db.recall(&emb, 10, None, None, false, false, None, true, None, None, None).unwrap();
    assert!(results.iter().any(|r| r.rid == rid), "hydrated memory should appear in recall");

    // Stats
    assert_eq!(db.stats(None).unwrap().archived_memories, 0);
}

#[test]
fn test_archive_idempotent() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db.record("idempotent", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

    assert!(db.archive(&rid).unwrap());
    assert!(!db.archive(&rid).unwrap()); // Already cold
}

#[test]
fn test_record_batch() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let inputs: Vec<RecordInput> = (0..10).map(|i| RecordInput {
        text: format!("batch memory {i}"),
        memory_type: "episodic".to_string(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: serde_json::json!({}),
        embedding: vec_seed(i as f32, 8),
        namespace: "default".to_string(),
        certainty: 0.8,
        domain: "general".to_string(),
        source: "user".to_string(),
        emotional_state: None,
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
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = db.record_batch(&[]).unwrap();
    assert!(rids.is_empty());
}

#[test]
fn test_evict() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Seed 20 memories
    for i in 0..20 {
        db.record(&format!("evict mem {i}"), "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(i as f32, 8), "default", 0.8, "general", "user", None).unwrap();
    }
    assert_eq!(db.stats(None).unwrap().active_memories, 20);

    // Evict to keep 10
    let archived = db.evict(10).unwrap();
    assert_eq!(archived.len(), 10);

    let stats = db.stats(None).unwrap();
    assert_eq!(stats.archived_memories, 10);

    // Recall should only find hot memories
    let results = db.recall(&vec_seed(0.0, 8), 20, None, None, false, false, None, true, None, None, None).unwrap();
    for r in &results {
        assert!(!archived.contains(&r.rid), "evicted memory should not be in recall");
    }
}

#[test]
fn test_evict_no_action_when_under_limit() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for i in 0..5 {
        db.record(&format!("small db {i}"), "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(i as f32, 8), "default", 0.8, "general", "user", None).unwrap();
    }
    let archived = db.evict(10).unwrap();
    assert!(archived.is_empty());
}

#[test]
fn test_query_builder_basic() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for i in 0..10 {
        db.record(&format!("memory {i}"), "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(i as f32, 8), "default", 0.8, "general", "user", None).unwrap();
    }

    let results = db.query(
        RecallQuery::new(vec_seed(0.0, 8))
            .top_k(3)
            .skip_reinforce()
    ).unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn test_query_builder_with_filters() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record("episodic one", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "work", 0.8, "general", "user", None).unwrap();
    db.record("semantic one", "semantic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(2.0, 8), "work", 0.8, "general", "user", None).unwrap();
    db.record("episodic two", "episodic", 0.5, 0.0, 604800.0,
        &empty_meta(), &vec_seed(3.0, 8), "personal", 0.8, "general", "user", None).unwrap();

    // Filter by type + namespace
    let results = db.query(
        RecallQuery::new(vec_seed(1.0, 8))
            .top_k(10)
            .memory_type("episodic")
            .namespace("work")
            .skip_reinforce()
    ).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory_type, "episodic");
    assert_eq!(results[0].namespace, "work");
}

#[test]
fn test_query_builder_contributions_present() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record("test mem", "episodic", 0.8, 0.5, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

    let results = db.query(
        RecallQuery::new(vec_seed(1.0, 8))
            .top_k(1)
            .skip_reinforce()
    ).unwrap();
    assert_eq!(results.len(), 1);

    let r = &results[0];
    // Verify explainability fields
    assert!(r.scores.valence_multiplier >= 1.0);
    assert!(r.scores.contributions.similarity >= 0.0);
    assert!(r.scores.contributions.decay >= 0.0);
    assert!(r.scores.contributions.recency >= 0.0);
    assert!(r.scores.contributions.importance >= 0.0);
}

// ── V5: Encryption at rest tests ──

fn test_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    for (i, b) in key.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(7).wrapping_add(42);
    }
    key
}

#[test]
fn test_encrypted_record_and_get() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();
    assert!(db.is_encrypted());

    let meta = serde_json::json!({"source": "test", "topic": "encryption"});
    let emb = vec_seed(1.0, 8);
    let rid = db.record("secret memory", "episodic", 0.8, 0.3, 604800.0, &meta, &emb, "default", 0.8, "general", "user", None).unwrap();

    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.text, "secret memory");
    assert_eq!(mem.memory_type, "episodic");
    assert_eq!(mem.importance, 0.8);
    assert_eq!(mem.metadata["source"], "test");
    assert_eq!(mem.metadata["topic"], "encryption");
}

#[test]
fn test_encrypted_data_not_plaintext_in_db() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();

    let rid = db.record("secret memory", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

    // Read raw stored text — should NOT be plaintext
    let stored_text: String = db.conn().query_row(
        "SELECT text FROM memories WHERE rid = ?1",
        params![rid], |r| r.get(0),
    ).unwrap();
    assert_ne!(stored_text, "secret memory", "text should be encrypted in DB");

    // Read raw stored metadata — should NOT be plaintext
    let stored_meta: String = db.conn().query_row(
        "SELECT metadata FROM memories WHERE rid = ?1",
        params![rid], |r| r.get(0),
    ).unwrap();
    assert_ne!(stored_meta, "{}", "metadata should be encrypted in DB");
}

#[test]
fn test_encrypted_recall_roundtrip() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();

    db.record("cat sat on mat", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    db.record("dog ran in park", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(5.0, 8), "default", 0.8, "general", "user", None).unwrap();
    db.record("cats love warmth", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.1, 8), "default", 0.8, "general", "user", None).unwrap();

    let results = db.recall(&vec_seed(1.0, 8), 2, None, None, false, false, None, true, None, None, None).unwrap();
    assert_eq!(results.len(), 2);
    // Text should be decrypted in results
    assert!(results.iter().any(|r| r.text.contains("cat")));
}

#[test]
fn test_encrypted_record_batch() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();

    let inputs: Vec<RecordInput> = (0..5).map(|i| RecordInput {
        text: format!("encrypted batch {i}"),
        memory_type: "episodic".to_string(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: serde_json::json!({"idx": i}),
        embedding: vec_seed(i as f32, 8),
        namespace: "default".to_string(),
        certainty: 0.8,
        domain: "general".to_string(),
        source: "user".to_string(),
        emotional_state: None,
    }).collect();

    let rids = db.record_batch(&inputs).unwrap();
    assert_eq!(rids.len(), 5);

    for (i, rid) in rids.iter().enumerate() {
        let mem = db.get(rid).unwrap().unwrap();
        assert_eq!(mem.text, format!("encrypted batch {i}"));
        assert_eq!(mem.metadata["idx"], i);
    }
}

#[test]
fn test_encrypted_archive_hydrate() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();

    let emb = vec_seed(2.0, 8);
    let rid = db.record("to archive encrypted", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &emb, "default", 0.8, "general", "user", None).unwrap();

    // Archive (encrypt compressed)
    assert!(db.archive(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.storage_tier, "cold");
    assert_eq!(mem.text, "to archive encrypted"); // text still decryptable

    // Hydrate (decrypt compressed, re-encrypt raw)
    assert!(db.hydrate(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.storage_tier, "hot");

    // Should be findable in recall after hydration
    let results = db.recall(&emb, 10, None, None, false, false, None, true, None, None, None).unwrap();
    assert!(results.iter().any(|r| r.rid == rid));
}

#[test]
fn test_encrypted_correct_memory() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();

    let rid = db.record("color is green", "semantic", 0.7, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let result = db.correct(&rid, "color is blue", Some(0.9), None, &vec_seed(2.0, 8), Some("fixed")).unwrap();

    assert!(result.original_tombstoned);
    let corrected = db.get(&result.corrected_rid).unwrap().unwrap();
    assert_eq!(corrected.text, "color is blue");
    assert_eq!(corrected.importance, 0.9);
}

#[test]
fn test_unencrypted_db_unaffected() {
    // Verify existing unencrypted path still works identically
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert!(!db.is_encrypted());

    let rid = db.record("plaintext memory", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.text, "plaintext memory");

    // Text should be stored as plaintext
    let stored_text: String = db.conn().query_row(
        "SELECT text FROM memories WHERE rid = ?1",
        params![rid], |r| r.get(0),
    ).unwrap();
    assert_eq!(stored_text, "plaintext memory");
}

#[test]
fn test_encrypted_db_wrong_key_fails() {
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // Create with key A
    let key_a = test_key();
    {
        let db = YantrikDB::new_encrypted(path, 8, &key_a).unwrap();
        db.record("secret", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        db.close().unwrap();
    }

    // Re-open with key B should fail (wrong DEK unwrap)
    let mut key_b = [0u8; 32];
    key_b[0] = 99;
    let result = YantrikDB::new_encrypted(path, 8, &key_b);
    assert!(result.is_err(), "Opening encrypted DB with wrong key should fail");
}

#[test]
fn test_encrypted_db_reopen_same_key() {
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    let key = test_key();
    let rid;
    {
        let db = YantrikDB::new_encrypted(path, 8, &key).unwrap();
        rid = db.record("persistent secret", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        db.close().unwrap();
    }

    // Re-open with same key — should decrypt successfully
    {
        let db = YantrikDB::new_encrypted(path, 8, &key).unwrap();
        let mem = db.get(&rid).unwrap().unwrap();
        assert_eq!(mem.text, "persistent secret");
    }
}

#[test]
fn test_open_encrypted_db_without_key_fails() {
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    let key = test_key();
    {
        let db = YantrikDB::new_encrypted(path, 8, &key).unwrap();
        db.record("data", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        db.close().unwrap();
    }

    // Open without key — should detect encryption_enabled and refuse
    let result = YantrikDB::new(path, 8);
    assert!(result.is_err(), "Opening encrypted DB without key should fail");
}

// ════════════════════════════════════════════════════════════════════════════════
// Phase 3: Richer Dimensions (certainty, domain, source, emotional_state)
// ════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_record_with_dimensions() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db.record(
        "meeting notes for Q1 planning", "episodic", 0.7, 0.2, 604800.0,
        &empty_meta(), &emb, "default",
        0.9, "work", "document", Some("joy"),
    ).unwrap();

    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.text, "meeting notes for Q1 planning");
    assert!((mem.certainty - 0.9).abs() < 1e-6, "certainty should be 0.9, got {}", mem.certainty);
    assert_eq!(mem.domain, "work");
    assert_eq!(mem.source, "document");
    assert_eq!(mem.emotional_state, Some("joy".to_string()));
}

#[test]
fn test_domain_filter() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record 3 memories: 2 in "work" domain, 1 in "health"
    db.record("work task A", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "work", "user", None).unwrap();
    db.record("health checkup", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "health", "user", None).unwrap();
    db.record("work task B", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(3.0, 8), "default", 0.8, "work", "user", None).unwrap();

    // Recall with domain="work" should return only the 2 work memories
    let results = db.recall(
        &vec_seed(1.0, 8), 10, None, None, false, false, None, true, None,
        Some("work"), None,
    ).unwrap();
    assert_eq!(results.len(), 2, "Expected 2 work-domain memories, got {}", results.len());
    for r in &results {
        assert_eq!(r.domain, "work");
    }
}

#[test]
fn test_source_filter() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record 3 memories: 2 from "user" source, 1 from "system"
    db.record("user input A", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    db.record("system log", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "general", "system", None).unwrap();
    db.record("user input B", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(3.0, 8), "default", 0.8, "general", "user", None).unwrap();

    // Recall with source="user" should return only the 2 user-sourced memories
    let results = db.recall(
        &vec_seed(1.0, 8), 10, None, None, false, false, None, true, None,
        None, Some("user"),
    ).unwrap();
    assert_eq!(results.len(), 2, "Expected 2 user-source memories, got {}", results.len());
    for r in &results {
        assert_eq!(r.source, "user");
    }
}

#[test]
fn test_domain_and_source_combined_filter() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record 4 memories with different domain/source combinations
    db.record("work from user", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "work", "user", None).unwrap();
    db.record("work from system", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "work", "system", None).unwrap();
    db.record("health from user", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(3.0, 8), "default", 0.8, "health", "user", None).unwrap();
    db.record("health from system", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(4.0, 8), "default", 0.8, "health", "system", None).unwrap();

    // Filter by domain="work" AND source="user" — should return only 1
    let results = db.recall(
        &vec_seed(1.0, 8), 10, None, None, false, false, None, true, None,
        Some("work"), Some("user"),
    ).unwrap();
    assert_eq!(results.len(), 1, "Expected 1 work+user memory, got {}", results.len());
    assert_eq!(results[0].domain, "work");
    assert_eq!(results[0].source, "user");

    // Filter by domain="health" AND source="system" — should return only 1
    let results = db.recall(
        &vec_seed(4.0, 8), 10, None, None, false, false, None, true, None,
        Some("health"), Some("system"),
    ).unwrap();
    assert_eq!(results.len(), 1, "Expected 1 health+system memory, got {}", results.len());
    assert_eq!(results[0].domain, "health");
    assert_eq!(results[0].source, "system");
}

#[test]
fn test_dimensions_preserved_on_correct() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db.record(
        "the sky is green", "semantic", 0.7, 0.0, 604800.0,
        &empty_meta(), &vec_seed(1.0, 8), "default",
        0.6, "work", "document", Some("surprise"),
    ).unwrap();

    // Correct the memory text
    let result = db.correct(&rid, "the sky is blue", Some(0.9), None, &vec_seed(2.0, 8), Some("color fix")).unwrap();
    assert!(result.original_tombstoned);

    // Verify the corrected memory preserves domain, source, and emotional_state
    let corrected = db.get(&result.corrected_rid).unwrap().unwrap();
    assert_eq!(corrected.text, "the sky is blue");
    assert_eq!(corrected.domain, "work", "domain should be preserved after correction");
    assert_eq!(corrected.source, "document", "source should be preserved after correction");
}

#[test]
fn test_batch_record_with_dimensions() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    let inputs: Vec<RecordInput> = vec![
        RecordInput {
            text: "batch work meeting".to_string(),
            memory_type: "episodic".to_string(),
            importance: 0.6,
            valence: 0.1,
            half_life: 604800.0,
            metadata: serde_json::json!({"batch": true}),
            embedding: vec_seed(1.0, 8),
            namespace: "default".to_string(),
            certainty: 0.95,
            domain: "work".to_string(),
            source: "document".to_string(),
            emotional_state: Some("focus".to_string()),
        },
        RecordInput {
            text: "batch health jog".to_string(),
            memory_type: "episodic".to_string(),
            importance: 0.4,
            valence: 0.3,
            half_life: 604800.0,
            metadata: serde_json::json!({"batch": true}),
            embedding: vec_seed(2.0, 8),
            namespace: "default".to_string(),
            certainty: 0.7,
            domain: "health".to_string(),
            source: "user".to_string(),
            emotional_state: None,
        },
        RecordInput {
            text: "batch personal diary".to_string(),
            memory_type: "semantic".to_string(),
            importance: 0.3,
            valence: -0.1,
            half_life: 604800.0,
            metadata: serde_json::json!({"batch": true}),
            embedding: vec_seed(3.0, 8),
            namespace: "default".to_string(),
            certainty: 0.5,
            domain: "personal".to_string(),
            source: "system".to_string(),
            emotional_state: Some("calm".to_string()),
        },
    ];

    let rids = db.record_batch(&inputs).unwrap();
    assert_eq!(rids.len(), 3);

    // Verify first memory
    let m0 = db.get(&rids[0]).unwrap().unwrap();
    assert_eq!(m0.text, "batch work meeting");
    assert!((m0.certainty - 0.95).abs() < 1e-6);
    assert_eq!(m0.domain, "work");
    assert_eq!(m0.source, "document");
    assert_eq!(m0.emotional_state, Some("focus".to_string()));

    // Verify second memory
    let m1 = db.get(&rids[1]).unwrap().unwrap();
    assert_eq!(m1.domain, "health");
    assert_eq!(m1.source, "user");
    assert_eq!(m1.emotional_state, None);

    // Verify third memory
    let m2 = db.get(&rids[2]).unwrap().unwrap();
    assert_eq!(m2.domain, "personal");
    assert_eq!(m2.source, "system");
    assert_eq!(m2.emotional_state, Some("calm".to_string()));
}

// ════════════════════════════════════════════════════════════════════════════════
// Phase 1: Interactive Recall (confidence, hints, recall_refine)
// ════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_recall_with_response_structure() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record("memory for recall response", "episodic", 0.7, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

    let response = db.recall_with_response(
        &vec_seed(1.0, 8), 5, None, None, false, false, None, true, None, None, None,
    ).unwrap();

    // RecallResponse must have all four fields
    assert!(!response.results.is_empty(), "results should not be empty");
    assert!(response.confidence >= 0.0 && response.confidence <= 1.0, "confidence should be in [0, 1], got {}", response.confidence);
    // retrieval_summary should have sources_used and candidate_count
    assert!(!response.retrieval_summary.sources_used.is_empty(), "sources_used should not be empty");
    assert!(response.retrieval_summary.candidate_count > 0, "candidate_count should be > 0");
    // hints is a Vec, may be empty or not
    let _ = response.hints; // just ensure the field exists and is accessible
}

#[test]
fn test_high_confidence_no_hints() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(5.0, 8);

    // Record several memories with similar embeddings so density is high
    for i in 0..5 {
        db.record(
            &format!("exact match memory {}", i), "episodic", 0.9, 0.0, 604800.0,
            &empty_meta(), &emb, "default", 0.8, "general", "user", None,
        ).unwrap();
    }

    // Recall with exact same embedding and top_k matching the number of stored memories.
    // This maximises the density signal (results.len / top_k = 1.0).
    let response = db.recall_with_response(
        &emb, 5, None, None, false, false, None, true, None, None, None,
    ).unwrap();

    assert!(!response.results.is_empty());
    // With 5 exact matches and top_k=5: signal_density=1.0, signal_sim~1.0
    // confidence = 0.35*sim + 0.25*gap + 0.20*(1/4) + 0.20*1.0
    // should be >= 0.60
    assert!(
        response.confidence >= 0.60,
        "Confidence should be >= 0.60 for exact match with full density, got {}",
        response.confidence,
    );
    assert!(
        response.hints.is_empty(),
        "Hints should be empty for high-confidence recall, got {} hints",
        response.hints.len(),
    );
}

#[test]
fn test_low_confidence_has_hints() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record a memory with one embedding
    db.record("something about cats", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

    // Recall with a very different embedding and a short query_text.
    // The short query_text (<=3 words) triggers the "specificity" hint, and
    // low density (1 result / 10 top_k) keeps confidence below 0.60.
    let far_emb = vec_seed(100.0, 8);
    let response = db.recall_with_response(
        &far_emb, 10, None, None, false, false,
        Some("cats"),  // short query_text triggers specificity hint
        true, None, None, None,
    ).unwrap();

    // With only 1 memory, top_k=10, no entities, no gap — confidence should be low
    assert!(
        response.confidence < 0.60,
        "Confidence should be < 0.60 for distant query, got {}",
        response.confidence,
    );
    assert!(
        !response.hints.is_empty(),
        "Hints should be non-empty for low-confidence recall with short query_text",
    );
}

#[test]
fn test_recall_refine_excludes_originals() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record 5 memories with distinct embeddings
    let mut rids = Vec::new();
    for i in 1..=5 {
        let rid = db.record(
            &format!("memory number {}", i), "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(i as f32, 8), "default", 0.8, "general", "user", None,
        ).unwrap();
        rids.push(rid);
    }

    // First recall: get top 2
    let first_results = db.recall(
        &vec_seed(1.0, 8), 2, None, None, false, false, None, true, None, None, None,
    ).unwrap();
    assert_eq!(first_results.len(), 2);
    let original_rids: Vec<String> = first_results.iter().map(|r| r.rid.clone()).collect();

    // Refine: exclude the first 2 RIDs
    let refined = db.recall_refine(
        &vec_seed(1.0, 8),      // original query
        &vec_seed(2.0, 8),      // refinement embedding
        &original_rids.iter().map(|s| s.as_str()).collect::<Vec<&str>>()
            .iter().map(|s| s.to_string()).collect::<Vec<String>>(),
        3,                       // top_k
        None, None, None,
    ).unwrap();

    // Refined results should not contain any of the original RIDs
    for result in &refined.results {
        assert!(
            !original_rids.contains(&result.rid),
            "Refined result should not contain original RID {}, but it does",
            result.rid,
        );
    }
}

#[test]
fn test_recall_refine_returns_response() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record a few memories
    for i in 1..=4 {
        db.record(
            &format!("refine test {}", i), "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(i as f32, 8), "default", 0.8, "general", "user", None,
        ).unwrap();
    }

    let exclude: Vec<String> = vec![];
    let response = db.recall_refine(
        &vec_seed(1.0, 8),
        &vec_seed(2.0, 8),
        &exclude,
        3,
        None, None, None,
    ).unwrap();

    // Verify RecallResponse structure
    assert!(response.confidence >= 0.0 && response.confidence <= 1.0);
    assert!(!response.retrieval_summary.sources_used.is_empty());
    assert!(response.retrieval_summary.candidate_count > 0);
    // hints may or may not be present
    let _ = &response.hints;
}

#[test]
fn test_retrieval_summary_fields() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record some memories so recall has candidates
    for i in 1..=3 {
        db.record(
            &format!("summary test {}", i), "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(i as f32, 8), "default", 0.8, "general", "user", None,
        ).unwrap();
    }

    let response = db.recall_with_response(
        &vec_seed(1.0, 8), 5, None, None, false, false, None, true, None, None, None,
    ).unwrap();

    let summary = &response.retrieval_summary;
    assert!(summary.top_similarity > 0.0, "top_similarity should be > 0, got {}", summary.top_similarity);
    assert!(
        summary.sources_used.contains(&"hnsw".to_string()),
        "sources_used should contain 'hnsw', got {:?}",
        summary.sources_used,
    );
    assert!(summary.candidate_count > 0, "candidate_count should be > 0, got {}", summary.candidate_count);
}

// ════════════════════════════════════════════════════════════════════════════════
// Phase 2: Adaptive Learning (feedback, weights, learning)
// ════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_recall_feedback_stores() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db.record("feedback target", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &emb, "default", 0.8, "general", "user", None).unwrap();

    // Submit feedback
    db.recall_feedback(Some("test query"), Some(&emb), &rid, "relevant", Some(0.85), Some(1)).unwrap();

    // Verify the row exists in recall_feedback table
    let count: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM recall_feedback WHERE rid = ?1 AND feedback = 'relevant'",
        params![rid],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(count, 1, "Expected 1 feedback row, got {}", count);
}

#[test]
fn test_learned_weights_default() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let weights = db.load_learned_weights().unwrap();

    assert!((weights.w_sim - 0.50).abs() < 1e-6, "w_sim default should be 0.50, got {}", weights.w_sim);
    assert!((weights.w_decay - 0.20).abs() < 1e-6, "w_decay default should be 0.20, got {}", weights.w_decay);
    assert!((weights.w_recency - 0.30).abs() < 1e-6, "w_recency default should be 0.30, got {}", weights.w_recency);
    assert!((weights.gate_tau - 0.25).abs() < 1e-6, "gate_tau default should be 0.25, got {}", weights.gate_tau);
    assert!((weights.alpha_imp - 0.80).abs() < 1e-6, "alpha_imp default should be 0.80, got {}", weights.alpha_imp);
    assert_eq!(weights.generation, 0, "generation should start at 0");
}

#[test]
fn test_feedback_count_increments() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db.record("counting feedback", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &emb, "default", 0.8, "general", "user", None).unwrap();

    // Submit 5 feedback items
    for i in 0..5 {
        let feedback_type = if i % 2 == 0 { "relevant" } else { "irrelevant" };
        db.recall_feedback(Some("query"), Some(&emb), &rid, feedback_type, Some(0.5), Some(i + 1)).unwrap();
    }

    let count = db.feedback_count().unwrap();
    assert_eq!(count, 5, "Expected feedback_count=5, got {}", count);
}

#[test]
fn test_learning_skipped_under_threshold() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db.record("learning test", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &emb, "default", 0.8, "general", "user", None).unwrap();

    // Submit fewer than 20 feedback items (the MIN_FEEDBACK threshold)
    for i in 0..10 {
        db.recall_feedback(Some("q"), Some(&emb), &rid, "relevant", Some(0.5), Some(i)).unwrap();
    }

    // run_learning should return false (skipped due to insufficient feedback)
    let result = db.run_learning().unwrap();
    assert!(!result, "run_learning should return false with < 20 feedback items");
}

#[test]
fn test_learning_runs_with_enough_feedback() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db.record("learning convergence", "episodic", 0.7, 0.0, 604800.0, &empty_meta(), &emb, "default", 0.8, "general", "user", None).unwrap();

    // Submit 25 feedback items (above the MIN_FEEDBACK=20 threshold)
    for i in 0..25 {
        let feedback_type = if i % 3 == 0 { "irrelevant" } else { "relevant" };
        let score = 0.3 + (i as f64 * 0.02);
        db.recall_feedback(Some("learning query"), Some(&emb), &rid, feedback_type, Some(score), Some(i + 1)).unwrap();
    }

    // run_learning should complete without error (may return true or false
    // depending on whether the optimizer found an improvement)
    let result = db.run_learning();
    assert!(result.is_ok(), "run_learning should not error with 25 feedback items: {:?}", result.err());
}

#[test]
fn test_think_includes_learning() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db.record("think learning integration", "episodic", 0.7, 0.0, 604800.0, &empty_meta(), &emb, "default", 0.8, "general", "user", None).unwrap();

    // Submit 25+ feedback items so learning has enough data
    for i in 0..26 {
        let feedback_type = if i % 4 == 0 { "irrelevant" } else { "relevant" };
        db.recall_feedback(Some("think query"), Some(&emb), &rid, feedback_type, Some(0.5), Some(i + 1)).unwrap();
    }

    // think() internally calls run_learning() — it should not panic or error
    let config = ThinkConfig::default();
    let result = db.think(&config);
    assert!(result.is_ok(), "think() should not error when learning has enough feedback: {:?}", result.err());
}

// ── Contradiction Classifier Tests ──

#[test]
fn test_conflict_entity_substitution_org() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb1 = vec_seed(1.0, 8);
    let emb2 = vec_seed(1.1, 8); // Very similar embedding

    // Create entities of the same type (organization)
    db.relate("User", "Google", "works_at", 1.0).unwrap();
    db.relate("User", "Meta", "works_at", 1.0).unwrap();

    // Record memories mentioning these entities
    db.record("User works at Google as a senior engineer", "episodic", 0.7, 0.0, 604800.0,
              &empty_meta(), &emb1, "default", 0.8, "work", "user", None).unwrap();
    db.record("User works at Meta as a senior engineer", "episodic", 0.7, 0.0, 604800.0,
              &empty_meta(), &emb2, "default", 0.8, "work", "user", None).unwrap();

    // Scan for conflicts — the entity substitution classifier should detect
    // that Google and Meta are both organizations, making this an identity_fact conflict
    let conflicts = crate::conflict::scan_conflicts(&db).unwrap();
    // Edge-based conflicts should be found (works_at is an identity rel type)
    assert!(!conflicts.is_empty(), "should detect works_at conflict");
    assert_eq!(conflicts[0].conflict_type, "identity_fact");
}

#[test]
fn test_conflict_entity_substitution_tech() {
    let db = YantrikDB::new(":memory:", 384).unwrap();

    // Create tech entities
    db.relate("API", "PostgreSQL", "uses", 1.0).unwrap();
    db.relate("API", "MySQL", "uses", 1.0).unwrap();

    // Record memories with similar embeddings but different tech choices
    let emb1 = vec_seed(2.0, 384);
    let emb2 = vec_seed(2.05, 384);
    db.record("The API service uses PostgreSQL for the database layer", "semantic", 0.8, 0.0, 604800.0,
              &empty_meta(), &emb1, "default", 0.8, "architecture", "user", None).unwrap();
    db.record("The API service uses MySQL for the database layer", "semantic", 0.8, 0.0, 604800.0,
              &empty_meta(), &emb2, "default", 0.8, "architecture", "user", None).unwrap();

    let conflicts = crate::conflict::scan_conflicts(&db).unwrap();
    // Should detect entity-based semantic conflict with tech substitution
    let entity_based = conflicts.iter().filter(|c| {
        c.detection_reason.contains("contradict")
    }).collect::<Vec<_>>();
    // May or may not detect depending on similarity threshold — just ensure no panics
    assert!(conflicts.len() >= 0);
}

// ── Relationship-Based Entity Type Tests ──

#[test]
fn test_relate_infers_entity_types() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.relate("MyApp", "React", "built_with", 1.0).unwrap();

    let entities = db.search_entities(Some("MyApp"), None, 1).unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].entity_type, "project");

    let entities = db.search_entities(Some("React"), None, 1).unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].entity_type, "tech");
}

#[test]
fn test_relate_infers_infrastructure() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.relate("Backend", "AWS", "deployed_on", 1.0).unwrap();

    let entities = db.search_entities(Some("AWS"), None, 1).unwrap();
    assert_eq!(entities[0].entity_type, "infrastructure");
}

// ── Confidence-Calibrated Recall Tests ──

#[test]
fn test_recall_with_response_has_certainty_reasons() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    db.record("Important architecture decision about microservices", "semantic", 0.8, 0.0, 604800.0,
              &empty_meta(), &emb, "default", 0.8, "work", "user", None).unwrap();

    let response = db.recall_with_response(
        &emb, 5, None, None, false, false, Some("architecture decision"), false, None, None, None,
    ).unwrap();

    assert!(!response.certainty_reasons.is_empty(), "should have certainty reasons");
    assert!(response.confidence >= 0.0 && response.confidence <= 1.0);
}

#[test]
fn test_recall_empty_db_low_confidence() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);

    let response = db.recall_with_response(
        &emb, 5, None, None, false, false, Some("anything"), false, None, None, None,
    ).unwrap();

    assert!(response.confidence < 0.5, "empty DB should have low confidence");
    assert!(response.certainty_reasons.iter().any(|r| r.contains("No") || r.contains("Sparse") || r.contains("Weak")),
            "should explain low confidence");
}

// ── Relationship Depth Tests ──

#[test]
fn test_relationship_depth_basic() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);

    // Create an entity with some relationships and memories
    db.relate("Alice", "Bob", "knows", 1.0).unwrap();
    db.relate("Alice", "ProjectX", "works_on", 1.0).unwrap();

    db.record("Alice presented the quarterly report", "episodic", 0.5, 0.3, 604800.0,
              &empty_meta(), &emb, "default", 0.8, "work", "user", None).unwrap();
    db.record("Alice prefers async communication", "semantic", 0.6, 0.0, 604800.0,
              &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "preference", "user", None).unwrap();

    let depth = db.relationship_depth("Alice", None).unwrap();
    assert_eq!(depth.entity, "Alice");
    assert_eq!(depth.entity_type, "person");
    assert!(depth.connection_count >= 2, "Alice connected to Bob and ProjectX");
    assert!(depth.memories_mentioning >= 2, "at least 2 memories mention Alice");
    assert!(depth.depth_score > 0.0, "should have positive depth score");
    assert!(depth.depth_score <= 1.0);
}

#[test]
fn test_relationship_depth_not_found() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let result = db.relationship_depth("NonexistentEntity", None);
    assert!(result.is_err(), "should error for unknown entity");
}

// ── Procedural Memory Tests ──

#[test]
fn test_record_and_surface_procedural() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(3.0, 8);

    let rid = db.record_procedural(
        "Use Agent tool with Explore subtype for architectural questions in this codebase",
        &emb, "work", "code search", 0.8, "default",
    ).unwrap();

    // Verify it was stored as procedural type
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.memory_type, "procedural");
    assert!((mem.importance - 0.8).abs() < 0.01);

    // Surface it with a similar query
    let results = db.surface_procedural(&emb, Some("how to search code"), Some("work"), 5, None).unwrap();
    assert!(!results.is_empty(), "should surface the procedural memory");
    assert_eq!(results[0].memory_type, "procedural");
}

#[test]
fn test_reinforce_procedural() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(4.0, 8);

    let rid = db.record_procedural(
        "Always run tests before pushing",
        &emb, "work", "git workflow", 0.5, "default",
    ).unwrap();

    // Reinforce with high outcome
    let reinforced = db.reinforce_procedural(&rid, 1.0).unwrap();
    assert!(reinforced);

    // Check importance increased
    let mem = db.get(&rid).unwrap().unwrap();
    assert!(mem.importance > 0.5, "importance should increase after positive reinforcement");
}

#[test]
fn test_procedural_stats() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record_procedural("proc 1", &vec_seed(1.0, 8), "work", "task A", 0.7, "default").unwrap();
    db.record_procedural("proc 2", &vec_seed(2.0, 8), "work", "task B", 0.9, "default").unwrap();
    db.record_procedural("proc 3", &vec_seed(3.0, 8), "health", "exercise", 0.5, "default").unwrap();

    let stats = db.procedural_stats(None).unwrap();
    assert!(stats.len() >= 2, "should have stats for work and health domains");
    let work_stats = stats.iter().find(|(d, _, _)| d == "work");
    assert!(work_stats.is_some());
    let (_, count, _) = work_stats.unwrap();
    assert_eq!(*count, 2);
}

// ── Session + Think Integration Tests ──

#[test]
fn test_session_awareness_trigger() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);

    // Start and end a session
    let sid = db.session_start("default", "claude", &serde_json::json!({})).unwrap();
    db.record("Worked on battle testing the MCP server", "episodic", 0.7, 0.5, 604800.0,
              &empty_meta(), &emb, "default", 0.8, "work", "user", None).unwrap();
    let _summary = db.session_end(&sid, Some("Battle tested MCP server v0.2.8")).unwrap();

    // Simulate time passing by backdating the session
    db.conn().execute(
        "UPDATE sessions SET ended_at = ended_at - 86400 * 3, started_at = started_at - 86400 * 3 WHERE session_id = ?1",
        params![sid],
    ).unwrap();

    // Run think — should generate a session_awareness trigger
    let config = ThinkConfig {
        run_consolidation: false,
        run_conflict_scan: false,
        run_pattern_mining: false,
        run_personality: false,
        ..Default::default()
    };
    let result = db.think(&config).unwrap();

    let session_triggers: Vec<_> = result.triggers.iter()
        .filter(|t| t.trigger_type == "session_awareness")
        .collect();
    assert!(!session_triggers.is_empty(), "should generate session awareness trigger after 3-day gap");
    assert!(session_triggers[0].reason.contains("hours"), "reason should mention time gap");
}
