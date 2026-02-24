use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::params;

use crate::engine::AIDB;
use crate::error::Result;
use crate::scoring;
use crate::types::{PersistedTrigger, Trigger, TriggerType};

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

// ── Existing trigger checks ──

/// Find important memories that are decaying significantly.
pub fn check_decay_triggers(
    db: &AIDB,
    importance_threshold: f64,
    decay_threshold: f64,
    max_triggers: usize,
) -> Result<Vec<Trigger>> {
    let ts = now();
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT rid, text, type, importance, half_life, last_access, valence \
         FROM memories \
         WHERE consolidation_status = 'active' \
         AND importance >= ?1",
    )?;

    let mut triggers = Vec::new();

    let rows = stmt.query_map(rusqlite::params![importance_threshold], |row| {
        Ok((
            row.get::<_, String>("rid")?,
            row.get::<_, String>("text")?,
            row.get::<_, String>("type")?,
            row.get::<_, f64>("importance")?,
            row.get::<_, f64>("half_life")?,
            row.get::<_, f64>("last_access")?,
            row.get::<_, f64>("valence")?,
        ))
    })?;

    for row in rows {
        let (rid, text, mem_type, importance, half_life, last_access, valence) = row?;
        let elapsed = ts - last_access;
        let current_score = scoring::decay_score(importance, half_life, elapsed);

        if current_score < decay_threshold {
            let days_since = elapsed / 86400.0;
            let decay_ratio = if importance > 0.0 {
                current_score / importance
            } else {
                0.0
            };

            let urgency = importance * (1.0 - decay_ratio);

            let mut context = HashMap::new();
            context.insert("text".to_string(), serde_json::json!(text));
            context.insert("type".to_string(), serde_json::json!(mem_type));
            context.insert("original_importance".to_string(), serde_json::json!(importance));
            context.insert("current_score".to_string(), serde_json::json!(current_score));
            context.insert("days_since_access".to_string(), serde_json::json!(days_since));
            context.insert("valence".to_string(), serde_json::json!(valence));

            triggers.push(Trigger {
                trigger_type: "decay_review".to_string(),
                reason: format!(
                    "Important memory (importance={importance:.1}) \
                     has decayed to {current_score:.3} after {days_since:.0} days"
                ),
                urgency,
                source_rids: vec![rid],
                suggested_action: "ask_user_to_confirm_or_forget".to_string(),
                context,
            });
        }
    }

    triggers.sort_by(|a, b| {
        b.urgency
            .partial_cmp(&a.urgency)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    triggers.truncate(max_triggers);
    Ok(triggers)
}

/// Trigger when there are enough active memories that consolidation might help.
pub fn check_consolidation_triggers(
    db: &AIDB,
    min_active_memories: i64,
) -> Result<Vec<Trigger>> {
    let stats = db.stats(None)?;
    let mut triggers = Vec::new();

    if stats.active_memories >= min_active_memories {
        let conn = db.conn();
        let unconsolidated: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories \
             WHERE consolidation_status = 'active' \
             AND type = 'episodic'",
            [],
            |row| row.get(0),
        )?;

        if unconsolidated >= min_active_memories {
            let mut context = HashMap::new();
            context.insert(
                "episodic_count".to_string(),
                serde_json::json!(unconsolidated),
            );
            context.insert(
                "total_active".to_string(),
                serde_json::json!(stats.active_memories),
            );

            triggers.push(Trigger {
                trigger_type: "consolidation_ready".to_string(),
                reason: format!("{unconsolidated} episodic memories could be consolidated"),
                urgency: (unconsolidated as f64 / 50.0).min(1.0),
                source_rids: vec![],
                suggested_action: "run_consolidation".to_string(),
                context,
            });
        }
    }

    Ok(triggers)
}

// ── New V3 trigger checks ──

/// Trigger when too many conflicts are open or critical ones are aging.
pub fn check_conflict_escalation(db: &AIDB) -> Result<Vec<Trigger>> {
    let conn = db.conn();
    let open_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM conflicts WHERE status = 'open'",
        [],
        |row| row.get(0),
    )?;

    let ts = now();
    let critical_aging: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM conflicts \
         WHERE status = 'open' AND priority = 'critical' \
         AND detected_at < ?1",
        params![ts - 86400.0 * 3.0],
        |row| row.get(0),
    )?;

    let mut triggers = Vec::new();
    if open_count > 5 || critical_aging {
        let mut urgency = (open_count as f64 / 10.0).min(1.0);
        if critical_aging {
            urgency = (urgency + 0.3).min(1.0);
        }

        let mut context = HashMap::new();
        context.insert("open_count".to_string(), serde_json::json!(open_count));
        context.insert("critical_aging".to_string(), serde_json::json!(critical_aging));

        triggers.push(Trigger {
            trigger_type: "conflict_escalation".to_string(),
            reason: format!("{open_count} open conflicts need attention"),
            urgency,
            source_rids: vec![],
            suggested_action: "review_conflicts".to_string(),
            context,
        });
    }

    Ok(triggers)
}

/// Trigger for old semantic memories that may be stale.
pub fn check_temporal_drift(db: &AIDB) -> Result<Vec<Trigger>> {
    let ts = now();
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT rid, text, created_at, last_access \
         FROM memories \
         WHERE type = 'semantic' \
         AND consolidation_status = 'active' \
         AND created_at < ?1 \
         AND last_access < ?2 \
         LIMIT 10",
    )?;

    let cutoff_created = ts - 86400.0 * 90.0; // 90 days old
    let cutoff_access = ts - 86400.0 * 30.0; // not accessed in 30 days
    let mut triggers = Vec::new();

    let rows = stmt.query_map(params![cutoff_created, cutoff_access], |row| {
        Ok((
            row.get::<_, String>("rid")?,
            row.get::<_, String>("text")?,
            row.get::<_, f64>("created_at")?,
            row.get::<_, f64>("last_access")?,
        ))
    })?;

    for row in rows {
        let (rid, text, created_at, _last_access) = row?;
        let age_days = (ts - created_at) / 86400.0;
        let urgency = (age_days / 365.0).min(1.0);

        let mut context = HashMap::new();
        context.insert("text".to_string(), serde_json::json!(text));
        context.insert("age_days".to_string(), serde_json::json!(age_days));

        triggers.push(Trigger {
            trigger_type: "temporal_drift".to_string(),
            reason: format!("Semantic memory is {age_days:.0} days old and may be outdated"),
            urgency,
            source_rids: vec![rid],
            suggested_action: "verify_or_update".to_string(),
            context,
        });
    }

    Ok(triggers)
}

/// Trigger for near-duplicate active memories (cosine similarity > 0.85).
pub fn check_redundancy(db: &AIDB, _sim_threshold: f64) -> Result<Vec<Trigger>> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT rid, text, embedding \
         FROM memories \
         WHERE consolidation_status = 'active' \
         AND embedding IS NOT NULL \
         ORDER BY created_at DESC \
         LIMIT 100",
    )?;

    let rows: Vec<(String, String, Vec<u8>)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>("rid")?,
                row.get::<_, String>("text")?,
                row.get::<_, Vec<u8>>("embedding")?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let mut triggers = Vec::new();
    let threshold = 0.85;

    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            let emb_a = crate::serde_helpers::deserialize_f32(&rows[i].2);
            let emb_b = crate::serde_helpers::deserialize_f32(&rows[j].2);
            let sim = crate::consolidate::cosine_similarity(&emb_a, &emb_b);

            if sim > threshold {
                let mut context = HashMap::new();
                context.insert("text_a".to_string(), serde_json::json!(rows[i].1));
                context.insert("text_b".to_string(), serde_json::json!(rows[j].1));
                context.insert("similarity".to_string(), serde_json::json!(sim));

                triggers.push(Trigger {
                    trigger_type: "redundancy".to_string(),
                    reason: format!(
                        "Two memories are {:.0}% similar and may be redundant",
                        sim * 100.0
                    ),
                    urgency: sim,
                    source_rids: vec![rows[i].0.clone(), rows[j].0.clone()],
                    suggested_action: "consolidate_or_forget".to_string(),
                    context,
                });
            }
        }
    }

    triggers.truncate(5);
    Ok(triggers)
}

/// Trigger for high-degree entities (relationship hubs).
pub fn check_relationship_insight(db: &AIDB) -> Result<Vec<Trigger>> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT src, COUNT(*) as degree \
         FROM edges WHERE tombstoned = 0 \
         GROUP BY src HAVING degree >= 5 \
         ORDER BY degree DESC \
         LIMIT 10",
    )?;

    let mut triggers = Vec::new();
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;

    for row in rows {
        let (entity, degree) = row?;
        let urgency = (degree as f64 / 20.0).min(1.0);

        let mut context = HashMap::new();
        context.insert("entity".to_string(), serde_json::json!(entity));
        context.insert("degree".to_string(), serde_json::json!(degree));

        triggers.push(Trigger {
            trigger_type: "relationship_insight".to_string(),
            reason: format!("Entity '{entity}' is a hub with {degree} connections"),
            urgency,
            source_rids: vec![],
            suggested_action: "explore_entity".to_string(),
            context,
        });
    }

    Ok(triggers)
}

/// Trigger when emotional valence shifts significantly over time.
pub fn check_valence_trend(db: &AIDB) -> Result<Vec<Trigger>> {
    let ts = now();
    let conn = db.conn();

    // Recent 7 days
    let recent_stats: Option<(f64, i64)> = conn
        .query_row(
            "SELECT AVG(valence), COUNT(*) FROM memories \
             WHERE consolidation_status = 'active' \
             AND created_at > ?1",
            params![ts - 86400.0 * 7.0],
            |row| Ok((row.get::<_, f64>(0)?, row.get::<_, i64>(1)?)),
        )
        .ok();

    // Preceding 30 days (day 7 to day 37)
    let baseline_stats: Option<(f64, i64)> = conn
        .query_row(
            "SELECT AVG(valence), COUNT(*) FROM memories \
             WHERE consolidation_status = 'active' \
             AND created_at BETWEEN ?1 AND ?2",
            params![ts - 86400.0 * 37.0, ts - 86400.0 * 7.0],
            |row| Ok((row.get::<_, f64>(0)?, row.get::<_, i64>(1)?)),
        )
        .ok();

    let mut triggers = Vec::new();

    if let (Some((recent_avg, recent_count)), Some((baseline_avg, baseline_count))) =
        (recent_stats, baseline_stats)
    {
        if recent_count >= 3 && baseline_count >= 3 {
            let delta = recent_avg - baseline_avg;
            if delta.abs() > 0.3 {
                let direction = if delta > 0.0 { "positive" } else { "negative" };
                let urgency = delta.abs().min(1.0);

                let mut context = HashMap::new();
                context.insert("recent_avg".to_string(), serde_json::json!(recent_avg));
                context.insert("baseline_avg".to_string(), serde_json::json!(baseline_avg));
                context.insert("delta".to_string(), serde_json::json!(delta));
                context.insert("direction".to_string(), serde_json::json!(direction));

                triggers.push(Trigger {
                    trigger_type: "valence_trend".to_string(),
                    reason: format!(
                        "Emotional tone has shifted {direction} by {:.2} over the past week",
                        delta.abs()
                    ),
                    urgency,
                    source_rids: vec![],
                    suggested_action: "acknowledge_trend".to_string(),
                    context,
                });
            }
        }
    }

    Ok(triggers)
}

/// Trigger for entities with contradictory edges (same src+rel_type, different dst).
pub fn check_entity_anomaly(db: &AIDB) -> Result<Vec<Trigger>> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT src, rel_type, COUNT(DISTINCT dst) as dst_count \
         FROM edges WHERE tombstoned = 0 \
         GROUP BY src, rel_type \
         HAVING dst_count >= 3 \
         ORDER BY dst_count DESC \
         LIMIT 10",
    )?;

    let identity_types: &[&str] = &[
        "birthday", "age", "lives_in", "works_at", "email", "phone",
        "full_name", "spouse", "hometown",
    ];
    let preference_types: &[&str] = &["prefers", "favorite", "likes", "dislikes"];

    let mut triggers = Vec::new();
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;

    for row in rows {
        let (entity, rel_type, dst_count) = row?;
        let is_identity = identity_types.contains(&rel_type.as_str());
        let is_preference = preference_types.contains(&rel_type.as_str());

        if is_identity || is_preference {
            let urgency = (dst_count as f64 / 5.0).min(1.0);

            let mut context = HashMap::new();
            context.insert("entity".to_string(), serde_json::json!(entity));
            context.insert("rel_type".to_string(), serde_json::json!(rel_type));
            context.insert("distinct_values".to_string(), serde_json::json!(dst_count));

            triggers.push(Trigger {
                trigger_type: "entity_anomaly".to_string(),
                reason: format!(
                    "Entity '{entity}' has {dst_count} different values for '{rel_type}'"
                ),
                urgency,
                source_rids: vec![],
                suggested_action: "review_entity".to_string(),
                context,
            });
        }
    }

    Ok(triggers)
}

// ── Unified trigger check ──

/// Run all trigger checks and return a unified, priority-sorted list.
pub fn check_all_triggers(
    db: &AIDB,
    importance_threshold: f64,
    decay_threshold: f64,
    max_triggers: usize,
) -> Result<Vec<Trigger>> {
    let mut triggers = Vec::new();
    triggers.extend(check_decay_triggers(db, importance_threshold, decay_threshold, max_triggers)?);
    triggers.extend(check_consolidation_triggers(db, 10)?);
    triggers.extend(check_conflict_escalation(db)?);
    triggers.extend(check_temporal_drift(db)?);
    triggers.extend(check_redundancy(db, 0.85)?);
    triggers.extend(check_relationship_insight(db)?);
    triggers.extend(check_valence_trend(db)?);
    triggers.extend(check_entity_anomaly(db)?);

    triggers.sort_by(|a, b| {
        b.urgency
            .partial_cmp(&a.urgency)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    triggers.truncate(max_triggers);
    Ok(triggers)
}

// ── Trigger persistence ──

/// Build a cooldown key for deduplication.
pub fn build_cooldown_key(trigger: &Trigger) -> String {
    if trigger.source_rids.is_empty() {
        trigger.trigger_type.clone()
    } else {
        let mut rids = trigger.source_rids.clone();
        rids.sort();
        format!("{}:{}", trigger.trigger_type, rids.join(","))
    }
}

/// Persist a trigger to trigger_log with cooldown checking.
/// Returns the trigger_id if persisted, None if suppressed by cooldown.
pub fn persist_trigger(db: &AIDB, trigger: &Trigger, ts: f64) -> Result<Option<String>> {
    let trigger_type = TriggerType::from_str(&trigger.trigger_type);
    let cooldown_key = build_cooldown_key(trigger);
    let cooldown_secs = trigger_type.default_cooldown_secs();

    let conn = db.conn();
    let active_exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM trigger_log \
         WHERE cooldown_key = ?1 \
         AND status IN ('pending', 'delivered') \
         AND created_at > ?2",
        params![cooldown_key, ts - cooldown_secs],
        |row| row.get(0),
    )?;

    if active_exists {
        return Ok(None);
    }

    let trigger_id = uuid7::uuid7().to_string();
    let hlc_ts = db.tick_hlc();
    let hlc_bytes = hlc_ts.to_bytes().to_vec();
    let actor_id = db.actor_id().to_string();
    let expires_at = ts + trigger_type.default_expiry_secs();
    let source_rids_json = serde_json::to_string(&trigger.source_rids)?;
    let context_json = serde_json::to_string(&trigger.context)?;

    conn.execute(
        "INSERT INTO trigger_log \
         (trigger_id, trigger_type, urgency, status, reason, suggested_action, \
          source_rids, context, created_at, expires_at, cooldown_key, hlc, origin_actor) \
         VALUES (?1, ?2, ?3, 'pending', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            trigger_id,
            trigger.trigger_type,
            trigger.urgency,
            trigger.reason,
            trigger.suggested_action,
            source_rids_json,
            context_json,
            ts,
            expires_at,
            cooldown_key,
            hlc_bytes,
            actor_id,
        ],
    )?;

    // Dual-write to join table
    for rid in &trigger.source_rids {
        conn.execute(
            "INSERT OR IGNORE INTO trigger_source_rids (trigger_id, rid) VALUES (?1, ?2)",
            params![trigger_id, rid],
        )?;
    }

    db.log_op(
        "trigger_fire",
        Some(&trigger_id),
        &serde_json::json!({
            "trigger_id": trigger_id,
            "trigger_type": trigger.trigger_type,
            "urgency": trigger.urgency,
            "reason": trigger.reason,
            "suggested_action": trigger.suggested_action,
            "source_rids": trigger.source_rids,
            "context": trigger.context,
            "cooldown_key": cooldown_key,
            "expires_at": expires_at,
        }),
        None,
    )?;

    Ok(Some(trigger_id))
}

/// Expire old triggers past their expires_at.
pub fn expire_triggers(db: &AIDB, ts: f64) -> Result<usize> {
    let conn = db.conn();
    let changes = conn.execute(
        "UPDATE trigger_log SET status = 'expired' \
         WHERE status = 'pending' AND expires_at IS NOT NULL AND expires_at < ?1",
        params![ts],
    )?;
    Ok(changes)
}

/// Filter triggers by cooldown and persist. Returns only non-suppressed triggers.
pub fn filter_and_persist_triggers(
    db: &AIDB,
    triggers: Vec<Trigger>,
    ts: f64,
) -> Result<Vec<Trigger>> {
    let mut persisted = Vec::new();
    for t in triggers {
        if persist_trigger(db, &t, ts)?.is_some() {
            persisted.push(t);
        }
    }
    Ok(persisted)
}

/// Query persisted triggers from trigger_log.
pub fn get_pending_triggers(db: &AIDB, limit: usize) -> Result<Vec<PersistedTrigger>> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT trigger_id, trigger_type, urgency, status, reason, suggested_action, \
         source_rids, context, created_at, delivered_at, acknowledged_at, acted_at, expires_at \
         FROM trigger_log \
         WHERE status = 'pending' \
         ORDER BY urgency DESC \
         LIMIT ?1",
    )?;

    let rows = stmt
        .query_map(params![limit as i64], |row| {
            let source_rids_str: String = row.get("source_rids")?;
            let context_str: String = row.get("context")?;
            Ok(PersistedTrigger {
                trigger_id: row.get("trigger_id")?,
                trigger_type: row.get("trigger_type")?,
                urgency: row.get("urgency")?,
                status: row.get("status")?,
                reason: row.get("reason")?,
                suggested_action: row.get("suggested_action")?,
                source_rids: serde_json::from_str(&source_rids_str).unwrap_or_default(),
                context: serde_json::from_str(&context_str)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
                created_at: row.get("created_at")?,
                delivered_at: row.get("delivered_at")?,
                acknowledged_at: row.get("acknowledged_at")?,
                acted_at: row.get("acted_at")?,
                expires_at: row.get("expires_at")?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(rows)
}

/// Query trigger history with optional type filter.
pub fn get_trigger_history(
    db: &AIDB,
    trigger_type: Option<&str>,
    limit: usize,
) -> Result<Vec<PersistedTrigger>> {
    let conn = db.conn();
    let (sql, limit_val) = if let Some(tt) = trigger_type {
        let mut stmt = conn.prepare(
            "SELECT trigger_id, trigger_type, urgency, status, reason, suggested_action, \
             source_rids, context, created_at, delivered_at, acknowledged_at, acted_at, expires_at \
             FROM trigger_log \
             WHERE trigger_type = ?1 \
             ORDER BY created_at DESC \
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![tt, limit as i64], parse_persisted_trigger)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        return Ok(rows);
    } else {
        (
            "SELECT trigger_id, trigger_type, urgency, status, reason, suggested_action, \
             source_rids, context, created_at, delivered_at, acknowledged_at, acted_at, expires_at \
             FROM trigger_log \
             ORDER BY created_at DESC \
             LIMIT ?1",
            limit,
        )
    };

    let mut stmt = conn.prepare(sql)?;
    let rows = stmt
        .query_map(params![limit_val as i64], parse_persisted_trigger)?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn parse_persisted_trigger(row: &rusqlite::Row<'_>) -> rusqlite::Result<PersistedTrigger> {
    let source_rids_str: String = row.get("source_rids")?;
    let context_str: String = row.get("context")?;
    Ok(PersistedTrigger {
        trigger_id: row.get("trigger_id")?,
        trigger_type: row.get("trigger_type")?,
        urgency: row.get("urgency")?,
        status: row.get("status")?,
        reason: row.get("reason")?,
        suggested_action: row.get("suggested_action")?,
        source_rids: serde_json::from_str(&source_rids_str).unwrap_or_default(),
        context: serde_json::from_str(&context_str)
            .unwrap_or(serde_json::Value::Object(Default::default())),
        created_at: row.get("created_at")?,
        delivered_at: row.get("delivered_at")?,
        acknowledged_at: row.get("acknowledged_at")?,
        acted_at: row.get("acted_at")?,
        expires_at: row.get("expires_at")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim)
            .map(|i| (seed * (i as f32 + 1.0) * 1.7).sin() + (seed * (i as f32 + 2.0) * 0.3).cos())
            .collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_no_trigger_for_fresh() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.record("fresh", "episodic", 0.9, 0.0, 604800.0, &serde_json::json!({}), &vec_seed(1.0, 8), "default").unwrap();
        let triggers = check_decay_triggers(&db, 0.5, 0.1, 5).unwrap();
        assert!(triggers.is_empty());
    }

    #[test]
    fn test_decay_trigger_fires() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid = db.record("important deadline", "episodic", 0.9, 0.0, 100.0, &serde_json::json!({}), &vec_seed(1.0, 8), "default").unwrap();

        db.conn().execute(
            "UPDATE memories SET last_access = ?1 WHERE rid = ?2",
            rusqlite::params![now() - 10000.0, rid],
        ).unwrap();

        let triggers = check_decay_triggers(&db, 0.5, 0.1, 5).unwrap();
        assert!(!triggers.is_empty());
        assert_eq!(triggers[0].trigger_type, "decay_review");
        assert_eq!(triggers[0].source_rids, vec![rid]);
    }

    #[test]
    fn test_consolidation_trigger() {
        let db = AIDB::new(":memory:", 8).unwrap();
        for i in 0..15 {
            db.record(
                &format!("episodic memory {i}"),
                "episodic", 0.5, 0.0, 604800.0,
                &serde_json::json!({}),
                &vec_seed(i as f32, 8),
                "default",
            ).unwrap();
        }

        let triggers = check_consolidation_triggers(&db, 10).unwrap();
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers[0].trigger_type, "consolidation_ready");
    }

    #[test]
    fn test_conflict_escalation_fires() {
        let db = AIDB::new(":memory:", 8).unwrap();
        // Create 6 open conflicts manually
        let ts = now();
        for i in 0..6 {
            let id = format!("conflict-{i}");
            let hlc = db.tick_hlc();
            db.conn().execute(
                "INSERT INTO conflicts (conflict_id, conflict_type, priority, status, \
                 memory_a, memory_b, detected_at, detected_by, detection_reason, hlc, origin_actor) \
                 VALUES (?1, 'minor', 'low', 'open', 'a', 'b', ?2, 'test', 'test', ?3, 'test')",
                params![id, ts, hlc.to_bytes().to_vec()],
            ).unwrap();
        }

        let triggers = check_conflict_escalation(&db).unwrap();
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers[0].trigger_type, "conflict_escalation");
    }

    #[test]
    fn test_conflict_escalation_no_fire() {
        let db = AIDB::new(":memory:", 8).unwrap();
        // Only 2 conflicts -> should not fire
        let ts = now();
        for i in 0..2 {
            let id = format!("conflict-{i}");
            let hlc = db.tick_hlc();
            db.conn().execute(
                "INSERT INTO conflicts (conflict_id, conflict_type, priority, status, \
                 memory_a, memory_b, detected_at, detected_by, detection_reason, hlc, origin_actor) \
                 VALUES (?1, 'minor', 'low', 'open', 'a', 'b', ?2, 'test', 'test', ?3, 'test')",
                params![id, ts, hlc.to_bytes().to_vec()],
            ).unwrap();
        }

        let triggers = check_conflict_escalation(&db).unwrap();
        assert!(triggers.is_empty());
    }

    #[test]
    fn test_temporal_drift_fires() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid = db.record("works at Google", "semantic", 0.8, 0.0, 604800.0, &serde_json::json!({}), &vec_seed(1.0, 8), "default").unwrap();

        // Backdate to 120 days ago
        let old_ts = now() - 86400.0 * 120.0;
        db.conn().execute(
            "UPDATE memories SET created_at = ?1, last_access = ?1 WHERE rid = ?2",
            params![old_ts, rid],
        ).unwrap();

        let triggers = check_temporal_drift(&db).unwrap();
        assert!(!triggers.is_empty());
        assert_eq!(triggers[0].trigger_type, "temporal_drift");
    }

    #[test]
    fn test_temporal_drift_skips_recent() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.record("works at Google", "semantic", 0.8, 0.0, 604800.0, &serde_json::json!({}), &vec_seed(1.0, 8), "default").unwrap();

        let triggers = check_temporal_drift(&db).unwrap();
        assert!(triggers.is_empty());
    }

    #[test]
    fn test_relationship_insight_fires_for_hub() {
        let db = AIDB::new(":memory:", 8).unwrap();
        // Create a hub entity with 6 edges
        for i in 0..6 {
            db.relate("Alice", &format!("entity_{i}"), &format!("knows_{i}"), 1.0).unwrap();
        }

        let triggers = check_relationship_insight(&db).unwrap();
        assert!(!triggers.is_empty());
        assert_eq!(triggers[0].trigger_type, "relationship_insight");
    }

    #[test]
    fn test_cooldown_prevents_refire() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let trigger = Trigger {
            trigger_type: "decay_review".to_string(),
            reason: "test".to_string(),
            urgency: 0.8,
            source_rids: vec!["rid-1".to_string()],
            suggested_action: "test".to_string(),
            context: HashMap::new(),
        };

        let ts = now();
        let first = persist_trigger(&db, &trigger, ts).unwrap();
        assert!(first.is_some());

        // Same trigger should be suppressed by cooldown
        let second = persist_trigger(&db, &trigger, ts).unwrap();
        assert!(second.is_none());
    }

    #[test]
    fn test_expiry_clears_old_triggers() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let trigger = Trigger {
            trigger_type: "decay_review".to_string(),
            reason: "test".to_string(),
            urgency: 0.8,
            source_rids: vec!["rid-1".to_string()],
            suggested_action: "test".to_string(),
            context: HashMap::new(),
        };

        // Persist with a past timestamp so it expires immediately
        let past = now() - 86400.0 * 30.0;
        persist_trigger(&db, &trigger, past).unwrap();

        let expired = expire_triggers(&db, now()).unwrap();
        assert_eq!(expired, 1);

        let pending = get_pending_triggers(&db, 10).unwrap();
        assert!(pending.is_empty());
    }

    #[test]
    fn test_filter_and_persist_deduplicates() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let t1 = Trigger {
            trigger_type: "decay_review".to_string(),
            reason: "test".to_string(),
            urgency: 0.8,
            source_rids: vec!["rid-1".to_string()],
            suggested_action: "test".to_string(),
            context: HashMap::new(),
        };
        let t2 = t1.clone();

        let ts = now();
        let persisted = filter_and_persist_triggers(&db, vec![t1, t2], ts).unwrap();
        assert_eq!(persisted.len(), 1); // second is suppressed by cooldown
    }
}
