//! Conflict detection and resolution.
//!
//! Rule-based detection engine for semantic contradictions across synced memories.
//! Conflicts are first-class data: stored in their own table, queryable, auditable,
//! and replicated via the oplog.

use rusqlite::params;

use crate::engine::AIDB;
use crate::error::Result;
use crate::types::{Conflict, ConflictType};

/// Rel types that indicate unique-value identity facts (should not have multiple values).
const IDENTITY_REL_TYPES: &[&str] = &[
    "birthday", "age", "lives_in", "works_at", "email", "phone",
    "full_name", "spouse", "hometown",
];

/// Rel types that indicate preferences (concurrent differences are suspicious).
const PREFERENCE_REL_TYPES: &[&str] = &[
    "prefers", "favorite", "likes", "dislikes",
];

/// Classify a conflict type from the rel_type.
fn classify_conflict(rel_type: &str) -> ConflictType {
    if IDENTITY_REL_TYPES.contains(&rel_type) {
        ConflictType::IdentityFact
    } else if PREFERENCE_REL_TYPES.contains(&rel_type) {
        ConflictType::Preference
    } else {
        ConflictType::Minor
    }
}

/// Check if a conflict already exists for this (memory_a, memory_b) pair.
/// Checks both orderings.
fn conflict_exists(db: &AIDB, rid_a: &str, rid_b: &str) -> Result<bool> {
    let conn = db.conn();
    let exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM conflicts
         WHERE (memory_a = ?1 AND memory_b = ?2)
            OR (memory_a = ?2 AND memory_b = ?1)",
        params![rid_a, rid_b],
        |row| row.get(0),
    )?;
    Ok(exists)
}

/// Find the oplog target_rid for a relate op with given (src, dst, rel_type).
fn find_memory_for_edge(
    conn: &rusqlite::Connection,
    src: &str,
    dst: &str,
    rel_type: &str,
) -> Result<Option<String>> {
    let result = conn.query_row(
        "SELECT target_rid FROM oplog
         WHERE op_type = 'relate'
           AND json_extract(payload, '$.src') = ?1
           AND json_extract(payload, '$.dst') = ?2
           AND json_extract(payload, '$.rel_type') = ?3
         ORDER BY hlc DESC LIMIT 1",
        params![src, dst, rel_type],
        |row| row.get::<_, Option<String>>(0),
    );

    match result {
        Ok(rid) => Ok(rid),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Create a conflict record and log it to the oplog for replication.
pub fn create_conflict(
    db: &AIDB,
    conflict_type: &ConflictType,
    memory_a: &str,
    memory_b: &str,
    entity: Option<&str>,
    rel_type: Option<&str>,
    detection_reason: &str,
) -> Result<Conflict> {
    let conflict_id = uuid7::uuid7().to_string();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let priority = conflict_type.default_priority();
    let hlc_ts = db.tick_hlc();
    let hlc_bytes = hlc_ts.to_bytes().to_vec();
    let actor_id = db.actor_id().to_string();

    let conn = db.conn();
    conn.execute(
        "INSERT OR IGNORE INTO conflicts
         (conflict_id, conflict_type, priority, status, memory_a, memory_b,
          entity, rel_type, detected_at, detected_by, detection_reason,
          hlc, origin_actor)
         VALUES (?1, ?2, ?3, 'open', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            conflict_id,
            conflict_type.as_str(),
            priority,
            memory_a,
            memory_b,
            entity,
            rel_type,
            ts,
            actor_id,
            detection_reason,
            hlc_bytes,
            actor_id,
        ],
    )?;

    // Log to oplog for replication
    db.log_op(
        "conflict_detect",
        Some(&conflict_id),
        &serde_json::json!({
            "conflict_id": conflict_id,
            "conflict_type": conflict_type.as_str(),
            "priority": priority,
            "memory_a": memory_a,
            "memory_b": memory_b,
            "entity": entity,
            "rel_type": rel_type,
            "detected_at": ts,
            "detected_by": actor_id,
            "detection_reason": detection_reason,
        }),
        None,
    )?;

    Ok(Conflict {
        conflict_id,
        conflict_type: conflict_type.as_str().to_string(),
        priority: priority.to_string(),
        status: "open".to_string(),
        memory_a: memory_a.to_string(),
        memory_b: memory_b.to_string(),
        entity: entity.map(String::from),
        rel_type: rel_type.map(String::from),
        detected_at: ts,
        detected_by: actor_id,
        detection_reason: detection_reason.to_string(),
        resolved_at: None,
        resolved_by: None,
        strategy: None,
        winner_rid: None,
        resolution_note: None,
    })
}

/// Detect edge-based contradictions for a newly materialized edge.
/// Called from materialize_relate in replication.rs during sync.
pub fn detect_edge_conflicts(
    db: &AIDB,
    src: &str,
    dst: &str,
    rel_type: &str,
    incoming_target_rid: Option<&str>,
) -> Result<Vec<Conflict>> {
    let conn = db.conn();
    let mut conflicts = Vec::new();

    // Only check identity and preference rel_types
    let is_identity = IDENTITY_REL_TYPES.contains(&rel_type);
    let is_preference = PREFERENCE_REL_TYPES.contains(&rel_type);
    if !is_identity && !is_preference {
        return Ok(conflicts);
    }

    // Find existing edges with same (src, rel_type) but different dst
    let mut stmt = conn.prepare(
        "SELECT edge_id, dst FROM edges
         WHERE src = ?1 AND rel_type = ?2 AND dst != ?3 AND tombstoned = 0",
    )?;

    let existing: Vec<(String, String)> = stmt
        .query_map(params![src, rel_type, dst], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    for (_edge_id, existing_dst) in existing {
        let conflict_type = classify_conflict(rel_type);

        // Find memory rids for the edges via oplog
        let memory_a = find_memory_for_edge(conn, src, &existing_dst, rel_type)?;
        let memory_b = incoming_target_rid
            .map(String::from)
            .or_else(|| find_memory_for_edge(conn, src, dst, rel_type).ok().flatten());

        if let (Some(ref mem_a), Some(ref mem_b)) = (&memory_a, &memory_b) {
            if !conflict_exists(db, mem_a, mem_b)? {
                let conflict = create_conflict(
                    db,
                    &conflict_type,
                    mem_a,
                    mem_b,
                    Some(src),
                    Some(rel_type),
                    &format!(
                        "Entity '{}' has conflicting {} values: '{}' vs '{}'",
                        src, rel_type, existing_dst, dst
                    ),
                )?;
                conflicts.push(conflict);
            }
        }
    }

    Ok(conflicts)
}

/// Full-database conflict scan. Finds all edge-based contradictions
/// and concurrent consolidation conflicts.
pub fn scan_conflicts(db: &AIDB) -> Result<Vec<Conflict>> {
    let conn = db.conn();
    let mut conflicts = Vec::new();

    // Scan for contradicting edges: same (src, rel_type) with different dst values
    let mut stmt = conn.prepare(
        "SELECT src, rel_type, GROUP_CONCAT(DISTINCT dst) as dsts, COUNT(DISTINCT dst) as cnt
         FROM edges
         WHERE tombstoned = 0
         GROUP BY src, rel_type
         HAVING cnt > 1",
    )?;

    let rows: Vec<(String, String, String)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    for (src, rel_type, dsts_csv) in rows {
        let is_identity = IDENTITY_REL_TYPES.contains(&rel_type.as_str());
        let is_preference = PREFERENCE_REL_TYPES.contains(&rel_type.as_str());
        if !is_identity && !is_preference {
            continue;
        }

        let dsts: Vec<&str> = dsts_csv.split(',').collect();
        if dsts.len() < 2 {
            continue;
        }

        // Create pairwise conflicts
        for i in 0..dsts.len() {
            for j in (i + 1)..dsts.len() {
                let mem_a = find_memory_for_edge(conn, &src, dsts[i].trim(), &rel_type)?;
                let mem_b = find_memory_for_edge(conn, &src, dsts[j].trim(), &rel_type)?;

                if let (Some(ref a), Some(ref b)) = (&mem_a, &mem_b) {
                    if !conflict_exists(db, a, b)? {
                        let conflict_type = classify_conflict(&rel_type);
                        let conflict = create_conflict(
                            db,
                            &conflict_type,
                            a,
                            b,
                            Some(&src),
                            Some(&rel_type),
                            &format!(
                                "Entity '{}' has conflicting {} values: '{}' vs '{}'",
                                src,
                                rel_type,
                                dsts[i].trim(),
                                dsts[j].trim()
                            ),
                        )?;
                        conflicts.push(conflict);
                    }
                }
            }
        }
    }

    // Scan for concurrent consolidation conflicts
    let mut cm_stmt = conn.prepare(
        "SELECT cm1.consolidation_rid, cm2.consolidation_rid, cm1.source_rid
         FROM consolidation_members cm1
         JOIN consolidation_members cm2
           ON cm1.source_rid = cm2.source_rid
          AND cm1.consolidation_rid < cm2.consolidation_rid",
    )?;

    let cm_rows: Vec<(String, String, String)> = cm_stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let mut seen_pairs = std::collections::HashSet::new();
    for (rid_a, rid_b, shared_source) in cm_rows {
        let pair = if rid_a < rid_b {
            (rid_a.clone(), rid_b.clone())
        } else {
            (rid_b.clone(), rid_a.clone())
        };
        if seen_pairs.contains(&pair) {
            continue;
        }
        seen_pairs.insert(pair);

        if !conflict_exists(db, &rid_a, &rid_b)? {
            let conflict = create_conflict(
                db,
                &ConflictType::Consolidation,
                &rid_a,
                &rid_b,
                None,
                None,
                &format!(
                    "Concurrent consolidation: both '{}' and '{}' consumed source '{}'",
                    rid_a, rid_b, shared_source
                ),
            )?;
            conflicts.push(conflict);
        }
    }

    Ok(conflicts)
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
    fn test_create_conflict() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid_a = db
            .record(
                "User likes coffee",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0, 8),
                "default",
            )
            .unwrap();
        let rid_b = db
            .record(
                "User likes tea",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(2.0, 8),
                "default",
            )
            .unwrap();

        let conflict = create_conflict(
            &db,
            &ConflictType::Preference,
            &rid_a,
            &rid_b,
            Some("User"),
            Some("prefers"),
            "User has conflicting preference: coffee vs tea",
        )
        .unwrap();

        assert_eq!(conflict.status, "open");
        assert_eq!(conflict.conflict_type, "preference");
        assert_eq!(conflict.priority, "high");
        assert_eq!(conflict.memory_a, rid_a);
        assert_eq!(conflict.memory_b, rid_b);
    }

    #[test]
    fn test_conflict_dedup() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid_a = db
            .record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default")
            .unwrap();
        let rid_b = db
            .record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default")
            .unwrap();

        assert!(!conflict_exists(&db, &rid_a, &rid_b).unwrap());
        create_conflict(&db, &ConflictType::Minor, &rid_a, &rid_b, None, None, "test").unwrap();
        assert!(conflict_exists(&db, &rid_a, &rid_b).unwrap());
        assert!(conflict_exists(&db, &rid_b, &rid_a).unwrap()); // reversed order
    }

    #[test]
    fn test_classify_conflict() {
        assert_eq!(classify_conflict("birthday"), ConflictType::IdentityFact);
        assert_eq!(classify_conflict("works_at"), ConflictType::IdentityFact);
        assert_eq!(classify_conflict("favorite"), ConflictType::Preference);
        assert_eq!(classify_conflict("prefers"), ConflictType::Preference);
        assert_eq!(classify_conflict("random_rel"), ConflictType::Minor);
    }

    #[test]
    fn test_scan_contradicting_edges() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.relate("User", "Google", "works_at", 1.0).unwrap();
        db.relate("User", "Meta", "works_at", 1.0).unwrap();

        let conflicts = scan_conflicts(&db).unwrap();
        assert!(!conflicts.is_empty());
        assert_eq!(conflicts[0].conflict_type, "identity_fact");
        assert_eq!(conflicts[0].entity.as_deref(), Some("User"));
    }

    #[test]
    fn test_scan_no_conflict_for_non_identity_edges() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.relate("User", "Alice", "friends_with", 1.0).unwrap();
        db.relate("User", "Bob", "friends_with", 1.0).unwrap();

        let conflicts = scan_conflicts(&db).unwrap();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_conflict_type_default_priorities() {
        assert_eq!(ConflictType::IdentityFact.default_priority(), "critical");
        assert_eq!(ConflictType::Preference.default_priority(), "high");
        assert_eq!(ConflictType::Temporal.default_priority(), "high");
        assert_eq!(ConflictType::Consolidation.default_priority(), "medium");
        assert_eq!(ConflictType::Minor.default_priority(), "low");
    }
}
