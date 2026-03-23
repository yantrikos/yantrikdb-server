//! Conflict detection and resolution.
//!
//! Rule-based detection engine for semantic contradictions across synced memories.
//! Conflicts are first-class data: stored in their own table, queryable, auditable,
//! and replicated via the oplog.

use rusqlite::params;

use crate::engine::YantrikDB;
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

/// Entity types that, when substituted in otherwise-identical sentences, indicate
/// a factual contradiction (identity-level conflict).
const IDENTITY_ENTITY_TYPES: &[&str] = &[
    "organization", "place", "person",
];

/// Entity types where substitution indicates a preference contradiction.
const PREFERENCE_ENTITY_TYPES: &[&str] = &[
    "tech",
];

/// Temporal keywords whose presence (when differing) suggests a temporal conflict.
const TEMPORAL_KEYWORDS: &[&str] = &[
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "morning", "afternoon", "evening", "night",
    "today", "tomorrow", "yesterday",
    "2024", "2025", "2026", "2027",
    "q1", "q2", "q3", "q4",
];

/// Date-like regex patterns for temporal substitution detection.
const DATE_PATTERNS: &[&str] = &[
    // Already handled by TEMPORAL_KEYWORDS: month names, day names, etc.
    // These catch numeric dates that TEMPORAL_KEYWORDS miss.
];

/// Check if a token looks like a date component (numeric date parts).
fn is_date_like(token: &str) -> bool {
    // ISO date: 2024-01-15 (split into parts: 2024, 01, 15)
    // Already covered by year keywords for 4-digit years.
    // Catch day/month numbers: 1-31
    if let Ok(n) = token.parse::<u32>() {
        return (1..=31).contains(&n);
    }
    // Ordinals: 1st, 2nd, 3rd, 15th, etc.
    if token.len() >= 3 && token.ends_with("st") || token.ends_with("nd")
        || token.ends_with("rd") || token.ends_with("th")
    {
        let num_part = &token[..token.len() - 2];
        if let Ok(n) = num_part.parse::<u32>() {
            return (1..=31).contains(&n);
        }
    }
    false
}

/// Map substitution category names to ConflictType.
/// Identity-like categories produce IdentityFact; everything else produces Preference.
const IDENTITY_CATEGORIES: &[&str] = &[
    "cloud_providers",
];

/// Check substitution_members table for category-based conflict.
fn check_category_substitution(
    conn: &rusqlite::Connection,
    diff_a: &[&String],
    diff_b: &[&String],
) -> Option<(ConflictType, String)> {
    // Try each pair of diff tokens
    let mut stmt = match conn.prepare_cached(
        "SELECT c.name, c.conflict_mode
         FROM substitution_members m1
         JOIN substitution_members m2 ON m1.category_id = m2.category_id
         JOIN substitution_categories c ON c.id = m1.category_id
         WHERE m1.token_normalized = ?1 AND m2.token_normalized = ?2
           AND m1.status = 'active' AND m2.status = 'active'
           AND m1.confidence >= 0.6 AND m2.confidence >= 0.6
           AND c.status = 'active'
         LIMIT 1"
    ) {
        Ok(s) => s,
        Err(_) => return None,
    };

    for token_a in diff_a {
        for token_b in diff_b {
            if let Ok((cat_name, _conflict_mode)) = stmt.query_row(
                params![token_a.as_str(), token_b.as_str()],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            ) {
                let conflict_type = if IDENTITY_CATEGORIES.contains(&cat_name.as_str()) {
                    ConflictType::IdentityFact
                } else {
                    ConflictType::Preference
                };
                let desc = format!(
                    "{} category substitution: {{{}}} vs {{{}}}",
                    cat_name, token_a, token_b,
                );
                return Some((conflict_type, desc));
            }
        }
    }

    // Try multi-word: join all diff tokens and check
    if diff_a.len() >= 2 || diff_b.len() >= 2 {
        let joined_a: String = diff_a.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ");
        let joined_b: String = diff_b.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ");
        if let Ok((cat_name, _)) = stmt.query_row(
            params![joined_a, joined_b],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        ) {
            let conflict_type = if IDENTITY_CATEGORIES.contains(&cat_name.as_str()) {
                ConflictType::IdentityFact
            } else {
                ConflictType::Preference
            };
            let desc = format!(
                "{} category substitution: {{{}}} vs {{{}}}",
                cat_name, joined_a, joined_b,
            );
            return Some((conflict_type, desc));
        }
    }

    None
}

/// Detect entity substitution in two memory texts.
///
/// Detection flow (in priority order):
/// 1. Temporal keywords (month names, days, years, etc.)
/// 2. Date-like numeric patterns (ordinals, day numbers)
/// 3. Substitution category lookup (learned + seed categories)
/// 4. Entity table lookup (legacy fallback)
fn classify_entity_substitution(
    conn: &rusqlite::Connection,
    text_a: &str,
    text_b: &str,
) -> (ConflictType, Option<String>) {
    let words_a: std::collections::HashSet<String> = text_a
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();
    let words_b: std::collections::HashSet<String> = text_b
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();

    let diff_a: Vec<&String> = words_a.difference(&words_b).collect();
    let diff_b: Vec<&String> = words_b.difference(&words_a).collect();

    // ── Step 1: Temporal keyword substitution ──
    let temporal_a = diff_a.iter().any(|w| TEMPORAL_KEYWORDS.contains(&w.as_str()));
    let temporal_b = diff_b.iter().any(|w| TEMPORAL_KEYWORDS.contains(&w.as_str()));
    if temporal_a && temporal_b {
        let diff_desc = format!(
            "temporal substitution: {{{}}} vs {{{}}}",
            diff_a.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
            diff_b.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
        );
        return (ConflictType::Temporal, Some(diff_desc));
    }

    // ── Step 2: Date-like numeric patterns ──
    let date_a = diff_a.iter().any(|w| is_date_like(w));
    let date_b = diff_b.iter().any(|w| is_date_like(w));
    if (temporal_a || date_a) && (temporal_b || date_b) {
        let diff_desc = format!(
            "date substitution: {{{}}} vs {{{}}}",
            diff_a.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
            diff_b.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
        );
        return (ConflictType::Temporal, Some(diff_desc));
    }

    // ── Step 3: Substitution category lookup (seed + learned) ──
    if let Some((conflict_type, desc)) = check_category_substitution(conn, &diff_a, &diff_b) {
        return (conflict_type, Some(desc));
    }

    // ── Step 4: Entity table lookup (legacy fallback) ──
    let mut entity_types_a: Vec<String> = Vec::new();
    let mut entity_types_b: Vec<String> = Vec::new();
    let mut entity_names_a: Vec<String> = Vec::new();
    let mut entity_names_b: Vec<String> = Vec::new();

    if let Ok(mut stmt) = conn.prepare_cached(
        "SELECT name, entity_type FROM entities WHERE LOWER(name) = ?1"
    ) {
        for word in &diff_a {
            if let Ok(etype) = stmt.query_row(params![word.as_str()], |row| row.get::<_, String>(1)) {
                entity_types_a.push(etype);
                entity_names_a.push(word.to_string());
            }
        }
        for word in &diff_b {
            if let Ok(etype) = stmt.query_row(params![word.as_str()], |row| row.get::<_, String>(1)) {
                entity_types_b.push(etype);
                entity_names_b.push(word.to_string());
            }
        }
    }

    // Multi-word entity matching
    if entity_types_a.is_empty() && diff_a.len() >= 2 {
        let joined: String = diff_a.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ");
        if let Ok(mut stmt) = conn.prepare_cached(
            "SELECT name, entity_type FROM entities WHERE LOWER(name) = ?1"
        ) {
            if let Ok((name, etype)) = stmt.query_row(params![joined], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                entity_types_a.push(etype);
                entity_names_a.push(name);
            }
        }
    }
    if entity_types_b.is_empty() && diff_b.len() >= 2 {
        let joined: String = diff_b.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ");
        if let Ok(mut stmt) = conn.prepare_cached(
            "SELECT name, entity_type FROM entities WHERE LOWER(name) = ?1"
        ) {
            if let Ok((name, etype)) = stmt.query_row(params![joined], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                entity_types_b.push(etype);
                entity_names_b.push(name);
            }
        }
    }

    // Check if both sides have entities of the same type
    for type_a in &entity_types_a {
        for type_b in &entity_types_b {
            if type_a == type_b {
                let diff_desc = format!(
                    "{} substitution: {{{}}} vs {{{}}}",
                    type_a,
                    entity_names_a.join(", "),
                    entity_names_b.join(", "),
                );

                if IDENTITY_ENTITY_TYPES.contains(&type_a.as_str()) {
                    return (ConflictType::IdentityFact, Some(diff_desc));
                }
                if PREFERENCE_ENTITY_TYPES.contains(&type_a.as_str()) {
                    return (ConflictType::Preference, Some(diff_desc));
                }
                return (ConflictType::Minor, Some(diff_desc));
            }
        }
    }

    // No substitution detected
    (ConflictType::Minor, None)
}

/// Map a substitution category name to the appropriate ConflictType.
pub(crate) fn category_to_conflict_type(cat_name: &str) -> ConflictType {
    if IDENTITY_CATEGORIES.contains(&cat_name) {
        ConflictType::IdentityFact
    } else {
        ConflictType::Preference
    }
}

/// Check if a conflict already exists for this (memory_a, memory_b) pair.
/// Checks both orderings.
pub(crate) fn conflict_exists(db: &YantrikDB, rid_a: &str, rid_b: &str) -> Result<bool> {
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
    db: &YantrikDB,
    conflict_type: &ConflictType,
    memory_a: &str,
    memory_b: &str,
    entity: Option<&str>,
    rel_type: Option<&str>,
    detection_reason: &str,
) -> Result<Conflict> {
    let conflict_id = crate::id::new_id();
    let ts = crate::time::now_secs();
    let priority = conflict_type.default_priority();
    let hlc_ts = db.tick_hlc();
    let hlc_bytes = hlc_ts.to_bytes().to_vec();
    let actor_id = db.actor_id().to_string();

    db.conn().execute(
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
    db: &YantrikDB,
    src: &str,
    dst: &str,
    rel_type: &str,
    incoming_target_rid: Option<&str>,
) -> Result<Vec<Conflict>> {
    let mut conflicts = Vec::new();

    // Only check identity and preference rel_types
    let is_identity = IDENTITY_REL_TYPES.contains(&rel_type);
    let is_preference = PREFERENCE_REL_TYPES.contains(&rel_type);
    if !is_identity && !is_preference {
        return Ok(conflicts);
    }

    // Collect data while holding the conn lock, then release before calling
    // conflict_exists/create_conflict (which also acquire the lock).
    let edge_data: Vec<(String, Option<String>, Option<String>)> = {
        let conn = db.conn();
        let mut stmt = conn.prepare(
            "SELECT edge_id, dst FROM edges
             WHERE src = ?1 AND rel_type = ?2 AND dst != ?3 AND tombstoned = 0",
        )?;

        let existing: Vec<(String, String)> = stmt
            .query_map(params![src, rel_type, dst], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        existing.into_iter().map(|(_edge_id, existing_dst)| {
            let memory_a = find_memory_for_edge(&conn, src, &existing_dst, rel_type).ok().flatten();
            let memory_b = incoming_target_rid
                .map(String::from)
                .or_else(|| find_memory_for_edge(&conn, src, dst, rel_type).ok().flatten());
            (existing_dst, memory_a, memory_b)
        }).collect()
    }; // conn lock released here

    for (existing_dst, memory_a, memory_b) in edge_data {
        let conflict_type = classify_conflict(rel_type);

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
pub fn scan_conflicts(db: &YantrikDB) -> Result<Vec<Conflict>> {
    let mut conflicts = Vec::new();

    // Phase 1: Collect edge-based conflict candidates while holding conn lock.
    // Each candidate: (src, rel_type, dst_i, dst_j, mem_a, mem_b)
    let edge_candidates: Vec<(String, String, String, String, Option<String>, Option<String>)>;
    let entity_groups: std::collections::HashMap<String, Vec<(String, String, Vec<u8>)>>;
    let cm_rows: Vec<(String, String, String)>;

    {
        let conn = db.conn();

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

        let mut candidates = Vec::new();
        for (src, rel_type, dsts_csv) in rows {
            let is_identity = IDENTITY_REL_TYPES.contains(&rel_type.as_str());
            let is_preference = PREFERENCE_REL_TYPES.contains(&rel_type.as_str());
            if !is_identity && !is_preference {
                continue;
            }

            let dsts: Vec<String> = dsts_csv.split(',').map(|s| s.trim().to_string()).collect();
            if dsts.len() < 2 {
                continue;
            }

            for i in 0..dsts.len() {
                for j in (i + 1)..dsts.len() {
                    let mem_a = find_memory_for_edge(&conn, &src, &dsts[i], &rel_type).ok().flatten();
                    let mem_b = find_memory_for_edge(&conn, &src, &dsts[j], &rel_type).ok().flatten();
                    candidates.push((src.clone(), rel_type.clone(), dsts[i].clone(), dsts[j].clone(), mem_a, mem_b));
                }
            }
        }
        edge_candidates = candidates;

        // Scan for entity-based semantic conflicts
        let mut entity_mem_stmt = conn.prepare(
            "SELECT me.entity_name, m.rid, m.text, m.embedding
             FROM memory_entities me
             JOIN memories m ON m.rid = me.memory_rid
             WHERE m.consolidation_status = 'active'
             AND m.embedding IS NOT NULL
             ORDER BY me.entity_name",
        )?;

        let em_rows: Vec<(String, String, String, Vec<u8>)> = entity_mem_stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut groups: std::collections::HashMap<String, Vec<(String, String, Vec<u8>)>> =
            std::collections::HashMap::new();
        for (entity, rid, text, emb) in em_rows {
            let text = db.decrypt_text(&text).unwrap_or(text);
            let emb = db.decrypt_embedding(&emb).unwrap_or(emb);
            groups.entry(entity).or_default().push((rid, text, emb));
        }
        entity_groups = groups;

        // Scan for concurrent consolidation conflicts
        let mut cm_stmt = conn.prepare(
            "SELECT cm1.consolidation_rid, cm2.consolidation_rid, cm1.source_rid
             FROM consolidation_members cm1
             JOIN consolidation_members cm2
               ON cm1.source_rid = cm2.source_rid
              AND cm1.consolidation_rid < cm2.consolidation_rid",
        )?;

        cm_rows = cm_stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
    } // conn lock released here

    // Phase 2: Create conflicts (these functions acquire conn lock internally).

    // Edge-based conflicts
    for (src, rel_type, dst_i, dst_j, mem_a, mem_b) in &edge_candidates {
        if let (Some(a), Some(b)) = (mem_a, mem_b) {
            if !conflict_exists(db, a, b)? {
                let conflict_type = classify_conflict(rel_type);
                let conflict = create_conflict(
                    db,
                    &conflict_type,
                    a,
                    b,
                    Some(src),
                    Some(rel_type),
                    &format!(
                        "Entity '{}' has conflicting {} values: '{}' vs '{}'",
                        src, rel_type, dst_i, dst_j
                    ),
                )?;
                conflicts.push(conflict);
            }
        }
    }

    // Entity-based semantic conflicts
    {
        let mut seen_pairs: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();

        for (entity, memories) in &entity_groups {
            if memories.len() < 2 {
                continue;
            }
            for i in 0..memories.len() {
                for j in (i + 1)..memories.len() {
                    let (ref rid_a, ref text_a, ref emb_a) = memories[i];
                    let (ref rid_b, ref text_b, ref emb_b) = memories[j];

                    let pair = if rid_a < rid_b {
                        (rid_a.clone(), rid_b.clone())
                    } else {
                        (rid_b.clone(), rid_a.clone())
                    };
                    if seen_pairs.contains(&pair) {
                        continue;
                    }

                    let emb_a_f32 = crate::serde_helpers::deserialize_f32(emb_a);
                    let emb_b_f32 = crate::serde_helpers::deserialize_f32(emb_b);
                    let sim = crate::consolidate::cosine_similarity(&emb_a_f32, &emb_b_f32);

                    // Similar topic (>0.5) but not exact duplicate (<0.98)
                    if sim > 0.5 && sim < 0.98 {
                        // Compute word-level Jaccard to detect different content
                        let words_a: std::collections::HashSet<&str> =
                            text_a.split_whitespace().map(|w| w.trim_matches(|c: char| !c.is_alphanumeric())).filter(|w| !w.is_empty()).collect();
                        let words_b: std::collections::HashSet<&str> =
                            text_b.split_whitespace().map(|w| w.trim_matches(|c: char| !c.is_alphanumeric())).filter(|w| !w.is_empty()).collect();

                        let intersection = words_a.intersection(&words_b).count();
                        let union = words_a.union(&words_b).count();
                        let jaccard = if union > 0 { intersection as f64 / union as f64 } else { 1.0 };

                        // High semantic similarity + low word overlap = likely contradiction
                        // (they're about the same topic but say different things)
                        if jaccard < 0.7 {
                            seen_pairs.insert(pair);
                            if !conflict_exists(db, rid_a, rid_b)? {
                                // Run entity substitution classifier to determine
                                // conflict type and generate a specific reason
                                let (conflict_type, substitution_desc) =
                                    classify_entity_substitution(&*db.conn(), text_a, text_b);

                                let reason = match substitution_desc {
                                    Some(ref desc) => format!(
                                        "Memories sharing entity '{}' contradict via {}: \
                                         similarity={:.0}%, word_overlap={:.0}%",
                                        entity, desc, sim * 100.0, jaccard * 100.0
                                    ),
                                    None => format!(
                                        "Memories sharing entity '{}' may contradict: \
                                         similarity={:.0}%, word_overlap={:.0}%",
                                        entity, sim * 100.0, jaccard * 100.0
                                    ),
                                };

                                let conflict = create_conflict(
                                    db,
                                    &conflict_type,
                                    rid_a,
                                    rid_b,
                                    Some(entity),
                                    None,
                                    &reason,
                                )?;
                                conflicts.push(conflict);
                            }
                        }
                    }
                }
            }
        }
    }

    // Consolidation conflicts
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
        let db = YantrikDB::new(":memory:", 8).unwrap();
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
                0.8,
                "general",
                "user",
                None,
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
                0.8,
                "general",
                "user",
                None,
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
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rid_a = db
            .record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None)
            .unwrap();
        let rid_b = db
            .record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8), "default", 0.8, "general", "user", None)
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
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.relate("User", "Google", "works_at", 1.0).unwrap();
        db.relate("User", "Meta", "works_at", 1.0).unwrap();

        let conflicts = scan_conflicts(&db).unwrap();
        assert!(!conflicts.is_empty());
        assert_eq!(conflicts[0].conflict_type, "identity_fact");
        assert_eq!(conflicts[0].entity.as_deref(), Some("User"));
    }

    #[test]
    fn test_scan_no_conflict_for_non_identity_edges() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
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
