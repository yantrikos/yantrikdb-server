use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::params;

use crate::engine::YantrikDB;
use crate::error::Result;
use crate::scoring;
use crate::types::{PersistedTrigger, Trigger, TriggerType};

fn now() -> f64 {
    crate::time::now_secs()
}

// ── Existing trigger checks ──

/// Find important memories that are decaying significantly.
pub fn check_decay_triggers(
    db: &YantrikDB,
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
        let (rid, stored_text, mem_type, importance, half_life, last_access, valence) = row?;
        let elapsed = ts - last_access;
        let current_score = scoring::decay_score(importance, half_life, elapsed);

        if current_score < decay_threshold {
            let text = db.decrypt_text(&stored_text)?;
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
    db: &YantrikDB,
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
pub fn check_conflict_escalation(db: &YantrikDB) -> Result<Vec<Trigger>> {
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
pub fn check_temporal_drift(db: &YantrikDB) -> Result<Vec<Trigger>> {
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
        let (rid, stored_text, created_at, _last_access) = row?;
        let text = db.decrypt_text(&stored_text)?;
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
pub fn check_redundancy(db: &YantrikDB, _sim_threshold: f64) -> Result<Vec<Trigger>> {
    // Phase 1: Collect memory data while holding the conn lock.
    let rows: Vec<(String, String, Vec<u8>)> = {
        let conn = db.conn();
        let mut stmt = conn.prepare(
            "SELECT rid, text, embedding \
             FROM memories \
             WHERE consolidation_status = 'active' \
             AND embedding IS NOT NULL \
             ORDER BY created_at DESC \
             LIMIT 30",
        )?;

        let raw_rows: Vec<(String, String, Vec<u8>)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>("rid")?,
                    row.get::<_, String>("text")?,
                    row.get::<_, Vec<u8>>("embedding")?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Decrypt text and embeddings if encrypted
        raw_rows.into_iter()
            .map(|(rid, stored_text, stored_emb)| {
                let text = db.decrypt_text(&stored_text)?;
                let emb = db.decrypt_embedding(&stored_emb)?;
                Ok((rid, text, emb))
            })
            .collect::<Result<Vec<_>>>()?
    }; // conn lock released here

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

                // Check if the pair shares entities — if so, this is likely a
                // contradiction (same topic, different facts) not a simple duplicate.
                let (entities_a, entities_b) = {
                    let conn = db.conn();
                    let ea: Vec<String> = conn
                        .prepare("SELECT entity_name FROM memory_entities WHERE memory_rid = ?1")?
                        .query_map(rusqlite::params![rows[i].0], |r| r.get(0))?
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    let eb: Vec<String> = conn
                        .prepare("SELECT entity_name FROM memory_entities WHERE memory_rid = ?1")?
                        .query_map(rusqlite::params![rows[j].0], |r| r.get(0))?
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    (ea, eb)
                };

                let shared: Vec<&String> = entities_a.iter().filter(|e| entities_b.contains(e)).collect();
                let is_potential_conflict = !shared.is_empty() && sim < 0.98;

                if is_potential_conflict {
                    context.insert("shared_entities".to_string(),
                        serde_json::json!(shared.iter().map(|s| s.as_str()).collect::<Vec<_>>()));
                    triggers.push(Trigger {
                        trigger_type: "potential_conflict".to_string(),
                        reason: format!(
                            "Two memories about '{}' are {:.0}% similar but may contradict each other",
                            shared.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
                            sim * 100.0
                        ),
                        urgency: sim,
                        source_rids: vec![rows[i].0.clone(), rows[j].0.clone()],
                        suggested_action: "review_conflict".to_string(),
                        context,
                    });
                } else if let Some((cat_name, token_a, token_b)) = {
                    // CRITICAL: bind the conn guard inside this block so it
                    // is dropped before conflict_exists() below tries to take
                    // the same connection mutex. Without this scoping, the
                    // if-let scrutinee's temporary MutexGuard lives through
                    // the body and self-deadlocks the calling thread (the
                    // engine's connection mutex is std::sync::Mutex, which
                    // is non-reentrant).
                    let conn = db.conn();
                    check_substitution_category_pair(&*conn, &rows[i].1, &rows[j].1)
                } {
                    // Substitution category match -> create actual conflict record
                    let reason = format!(
                        "{} substitution: '{}' vs '{}' (similarity={:.0}%)",
                        cat_name, token_a, token_b, sim * 100.0
                    );
                    if !crate::conflict::conflict_exists(db, &rows[i].0, &rows[j].0).unwrap_or(true) {
                        let conflict_type = crate::distributed::conflict::category_to_conflict_type(&cat_name);
                        let _ = crate::conflict::create_conflict(
                            db,
                            &conflict_type,
                            &rows[i].0,
                            &rows[j].0,
                            None,
                            None,
                            &reason,
                        );
                    }
                    context.insert("category".to_string(), serde_json::json!(cat_name));
                    context.insert("token_a".to_string(), serde_json::json!(token_a));
                    context.insert("token_b".to_string(), serde_json::json!(token_b));
                    triggers.push(Trigger {
                        trigger_type: "potential_conflict".to_string(),
                        reason,
                        urgency: sim,
                        source_rids: vec![rows[i].0.clone(), rows[j].0.clone()],
                        suggested_action: "review_conflict".to_string(),
                        context,
                    });
                } else {
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
    }

    // Second pass: lower threshold for substitution category conflicts.
    // Substituting "PostgreSQL" for "MySQL" drops cosine similarity to ~0.80,
    // below the 0.85 redundancy threshold. Check pairs in [0.65, 0.85] range
    // specifically for category-based substitution.
    let cat_threshold = 0.65;
    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            let emb_a = crate::serde_helpers::deserialize_f32(&rows[i].2);
            let emb_b = crate::serde_helpers::deserialize_f32(&rows[j].2);
            let sim = crate::consolidate::cosine_similarity(&emb_a, &emb_b);

            if sim > cat_threshold && sim <= threshold {
                // Same self-deadlock fix as the first pass above: bind the
                // conn guard inside a scrutinee block so it drops before
                // conflict_exists() reacquires the connection mutex.
                if let Some((cat_name, token_a, token_b)) = {
                    let conn = db.conn();
                    check_substitution_category_pair(&*conn, &rows[i].1, &rows[j].1)
                } {
                    let reason = format!(
                        "{} substitution: '{}' vs '{}' (similarity={:.0}%)",
                        cat_name, token_a, token_b, sim * 100.0
                    );
                    if !crate::conflict::conflict_exists(db, &rows[i].0, &rows[j].0).unwrap_or(true) {
                        let conflict_type = crate::distributed::conflict::category_to_conflict_type(&cat_name);
                        let _ = crate::conflict::create_conflict(
                            db,
                            &conflict_type,
                            &rows[i].0,
                            &rows[j].0,
                            None,
                            None,
                            &reason,
                        );
                    }
                    let mut context = HashMap::new();
                    context.insert("text_a".to_string(), serde_json::json!(rows[i].1));
                    context.insert("text_b".to_string(), serde_json::json!(rows[j].1));
                    context.insert("similarity".to_string(), serde_json::json!(sim));
                    context.insert("category".to_string(), serde_json::json!(cat_name));
                    context.insert("token_a".to_string(), serde_json::json!(token_a));
                    context.insert("token_b".to_string(), serde_json::json!(token_b));
                    triggers.push(Trigger {
                        trigger_type: "potential_conflict".to_string(),
                        reason,
                        urgency: sim,
                        source_rids: vec![rows[i].0.clone(), rows[j].0.clone()],
                        suggested_action: "review_conflict".to_string(),
                        context,
                    });
                }
            }
        }
    }

    triggers.truncate(5);
    Ok(triggers)
}

/// Trigger for high-degree entities (relationship hubs).
pub fn check_relationship_insight(db: &YantrikDB) -> Result<Vec<Trigger>> {
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
pub fn check_valence_trend(db: &YantrikDB) -> Result<Vec<Trigger>> {
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
pub fn check_entity_anomaly(db: &YantrikDB) -> Result<Vec<Trigger>> {
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
    db: &YantrikDB,
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
pub fn persist_trigger(db: &YantrikDB, trigger: &Trigger, ts: f64) -> Result<Option<String>> {
    let trigger_type = TriggerType::from_str(&trigger.trigger_type);
    let cooldown_key = build_cooldown_key(trigger);
    let cooldown_secs = trigger_type.default_cooldown_secs();

    let active_exists: bool = db.conn().query_row(
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

    let trigger_id = crate::id::new_id();
    let hlc_ts = db.tick_hlc();
    let hlc_bytes = hlc_ts.to_bytes().to_vec();
    let actor_id = db.actor_id().to_string();
    let expires_at = ts + trigger_type.default_expiry_secs();
    let source_rids_json = serde_json::to_string(&trigger.source_rids)?;
    let context_json = serde_json::to_string(&trigger.context)?;

    {
        let conn = db.conn();
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
    } // conn lock released before log_op

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
pub fn expire_triggers(db: &YantrikDB, ts: f64) -> Result<usize> {
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
    db: &YantrikDB,
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
pub fn get_pending_triggers(db: &YantrikDB, limit: usize) -> Result<Vec<PersistedTrigger>> {
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
    db: &YantrikDB,
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

/// Check if two memory texts differ by tokens in the same substitution category.
/// Returns (category_name, token_a, token_b) if a match is found.
fn check_substitution_category_pair(
    conn: &rusqlite::Connection,
    text_a: &str,
    text_b: &str,
) -> Option<(String, String, String)> {
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

    // Check each diff token pair against substitution_members
    for ta in &diff_a {
        for tb in &diff_b {
            let result: std::result::Result<(String,), _> = conn.query_row(
                "SELECT c.name FROM substitution_members m1
                 JOIN substitution_members m2 ON m1.category_id = m2.category_id
                 JOIN substitution_categories c ON c.id = m1.category_id
                 WHERE m1.token_normalized = ?1 AND m2.token_normalized = ?2
                   AND m1.status = 'active' AND m2.status = 'active'
                   AND m1.confidence >= 0.6 AND m2.confidence >= 0.6
                   AND c.status = 'active' AND c.conflict_mode = 'exclusive'
                 LIMIT 1",
                params![ta.as_str(), tb.as_str()],
                |row| Ok((row.get::<_, String>(0)?,)),
            );
            if let Ok((cat_name,)) = result {
                return Some((cat_name, ta.to_string(), tb.to_string()));
            }
        }
    }

    None
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
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.record("fresh", "episodic", 0.9, 0.0, 604800.0, &serde_json::json!({}), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        let triggers = check_decay_triggers(&db, 0.5, 0.1, 5).unwrap();
        assert!(triggers.is_empty());
    }

    #[test]
    fn test_decay_trigger_fires() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rid = db.record("important deadline", "episodic", 0.9, 0.0, 100.0, &serde_json::json!({}), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

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
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for i in 0..15 {
            db.record(
                &format!("episodic memory {i}"),
                "episodic", 0.5, 0.0, 604800.0,
                &serde_json::json!({}),
                &vec_seed(i as f32, 8),
                "default", 0.8, "general", "user", None,
            ).unwrap();
        }

        let triggers = check_consolidation_triggers(&db, 10).unwrap();
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers[0].trigger_type, "consolidation_ready");
    }

    #[test]
    fn test_conflict_escalation_fires() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
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
        let db = YantrikDB::new(":memory:", 8).unwrap();
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
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rid = db.record("works at Google", "semantic", 0.8, 0.0, 604800.0, &serde_json::json!({}), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

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
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.record("works at Google", "semantic", 0.8, 0.0, 604800.0, &serde_json::json!({}), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

        let triggers = check_temporal_drift(&db).unwrap();
        assert!(triggers.is_empty());
    }

    #[test]
    fn test_relationship_insight_fires_for_hub() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
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
        let db = YantrikDB::new(":memory:", 8).unwrap();
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
        let db = YantrikDB::new(":memory:", 8).unwrap();
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
        let db = YantrikDB::new(":memory:", 8).unwrap();
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

    /// Regression test for the v0.5.8 self-deadlock bug in check_redundancy.
    ///
    /// Before the fix (commit c4c2d9d), an `if let Some(...) =
    /// check_substitution_category_pair(&*db.conn(), ...)` in the high-similarity
    /// pass extended a `MutexGuard<Connection>` lifetime through the if-let body,
    /// and the body called `conflict_exists(db, ...)` which tried to take
    /// `db.conn()` again on the same thread. std::sync::Mutex is non-reentrant,
    /// so the consolidation worker self-deadlocked while holding the outer
    /// engine mutex, wedging every other worker on the next engine.lock().
    ///
    /// This test reproduces the exact trigger conditions:
    ///   1. Two memories with cosine similarity > 0.85 (we use identical
    ///      embeddings for determinism: similarity = 1.0)
    ///   2. No shared entities (memories share no relate edges)
    ///   3. Substitution-category membership for two of the differing tokens
    ///      (we seed a `databases` category with `mysql` and `postgres`)
    ///
    /// Before v0.5.8 this test hangs forever on std::sync::Mutex.
    /// After v0.5.8 + parking_lot (v0.5.9) it completes within milliseconds
    /// AND a conflict record is written.
    ///
    /// A 5-second background timeout would be nicer but Rust's stdlib does
    /// not expose cheap per-test timeouts. Tokio-style harnesses are
    /// inappropriate here because this is a pure sync core test. If the
    /// bug regresses, the entire test binary will hang and CI will catch
    /// it via the 60-second default cargo test timeout.
    #[test]
    fn test_check_redundancy_no_self_deadlock_on_substitution_category() {
        use rusqlite::params;

        let db = YantrikDB::new(":memory:", 8).unwrap();

        // Seed a substitution category: {test_deadlock_regression} with
        // exclusive mode so a match triggers conflict creation (not just
        // redundancy). We use a test-only name to avoid colliding with
        // any default categories the schema may seed.
        let hlc_bytes = db.tick_hlc().to_bytes().to_vec();
        let ts = now();
        db.conn()
            .execute(
                "INSERT INTO substitution_categories
                 (id, name, conflict_mode, status, created_at, updated_at, hlc, origin_actor)
                 VALUES ('cat-test-deadlock', 'test_deadlock_regression',
                         'exclusive', 'active', ?1, ?1, ?2, 'test')",
                params![ts, hlc_bytes],
            )
            .unwrap();

        // Use fabricated tokens that cannot collide with any seeded members.
        for (tok, suffix) in [("zyxqvtoken1", "a"), ("zyxqvtoken2", "b")] {
            let hlc_bytes = db.tick_hlc().to_bytes().to_vec();
            db.conn()
                .execute(
                    "INSERT INTO substitution_members
                     (id, category_id, token_normalized, token_display,
                      confidence, source, status, created_at, updated_at,
                      hlc, origin_actor)
                     VALUES (?1, 'cat-test-deadlock', ?2, ?2, 0.9, 'test', 'active',
                             ?3, ?3, ?4, 'test')",
                    params![format!("mem-{suffix}"), tok, ts, hlc_bytes],
                )
                .unwrap();
        }

        // Record two memories with identical embeddings (cosine sim = 1.0,
        // well above the 0.85 redundancy threshold) and texts that share
        // most words but differ on the substitution-category tokens.
        let emb = vec_seed(1.0, 8);
        let rid_a = db
            .record(
                "we store user profiles in zyxqvtoken1 for the auth service",
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
        let rid_b = db
            .record(
                "we store user profiles in zyxqvtoken2 for the auth service",
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

        // If the v0.5.8 self-deadlock regresses, this call hangs forever and
        // the test times out. parking_lot (v0.5.9) would additionally trip
        // the runtime deadlock detector.
        let triggers = check_redundancy(&db, 0.85).unwrap();

        // The pair should have been flagged — either as a redundancy/substitution
        // trigger, or consumed into an actual conflict record by the body.
        // At minimum, check_redundancy must have returned at all.
        assert!(
            !triggers.is_empty(),
            "expected at least one trigger for mysql/postgres substitution pair"
        );

        // Verify conflict_exists path was reachable AND completed: a conflict
        // row should have been written by create_conflict() inside the body.
        // This confirms both memories and the substitution category were
        // resolved through the previously-deadlocking code path.
        let conflict_count: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM conflicts
                 WHERE (memory_a = ?1 AND memory_b = ?2)
                    OR (memory_a = ?2 AND memory_b = ?1)",
                params![rid_a, rid_b],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            conflict_count, 1,
            "expected exactly one conflict record between the substitution pair"
        );
    }
}
