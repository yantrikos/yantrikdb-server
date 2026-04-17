use rusqlite::params;

use crate::error::Result;
use crate::types::*;

use super::{now, YantrikDB};

impl YantrikDB {
    // ── Cognition loop (V3) ──

    /// Run the full cognition loop: trigger detection, conflict scanning,
    /// consolidation, and pattern mining. Returns a prioritized list of triggers
    /// and summary of actions taken.
    #[tracing::instrument(skip(self, config))]
    pub fn think(&self, config: &ThinkConfig) -> Result<ThinkResult> {
        let start = crate::time::Instant::now();
        let ts = now();

        // Phase 0: Expire old triggers
        let expired = crate::triggers::expire_triggers(self, ts)?;

        // Phase 1: Run all trigger checks
        let mut all_triggers = Vec::new();
        all_triggers.extend(crate::triggers::check_decay_triggers(
            self,
            config.importance_threshold,
            config.decay_threshold,
            config.max_triggers,
        )?);
        all_triggers.extend(crate::triggers::check_consolidation_triggers(
            self,
            config.min_active_memories,
        )?);
        all_triggers.extend(crate::triggers::check_conflict_escalation(self)?);
        all_triggers.extend(crate::triggers::check_temporal_drift(self)?);
        all_triggers.extend(crate::triggers::check_redundancy(self, config.consolidation_sim_threshold)?);
        all_triggers.extend(crate::triggers::check_relationship_insight(self)?);
        all_triggers.extend(crate::triggers::check_valence_trend(self)?);
        all_triggers.extend(crate::triggers::check_entity_anomaly(self)?);

        // Phase 2: Scan for conflicts BEFORE consolidation
        // (so contradictions are flagged before similar memories get merged)
        // Uses consolidation_limit to cap work per call (incremental processing).
        let conflicts_found = if config.run_conflict_scan {
            let entity_conflicts = crate::conflict::scan_conflicts_limited(self, config.consolidation_limit)?.len();
            // RFC 006 Phase 1: also scan claim-based conflicts (scoped, with severity + reason codes)
            let claim_conflicts = crate::conflict::scan_claim_conflicts(self, config.consolidation_limit)?.len();
            entity_conflicts + claim_conflicts
        } else {
            0
        };

        // Phase 2.5: Check if any substitution categories are ready for gossip expansion
        all_triggers.extend(self.check_gossip_triggers()?);

        // Phase 3: Run consolidation if configured
        let consolidation_count = if config.run_consolidation {
            let stats = self.stats(None)?;
            if stats.active_memories >= config.min_active_memories {
                let results = crate::consolidate::consolidate(
                    self,
                    config.consolidation_sim_threshold,
                    config.consolidation_time_window_days,
                    config.consolidation_min_cluster,
                    config.consolidation_limit,
                    config.consolidation_require_entity_overlap,
                    false,
                )?;
                results.len()
            } else {
                0
            }
        } else {
            0
        };

        // Phase 4: Mine patterns
        let pattern_result = if config.run_pattern_mining {
            crate::patterns::mine_patterns(self, &PatternConfig::default())?
        } else {
            PatternMiningResult {
                new_patterns: 0,
                updated_patterns: 0,
                stale_patterns: 0,
            }
        };

        // Phase 4.5: Adaptive learning — optimize scoring weights from feedback
        let weights_updated = self.run_learning().unwrap_or(false);
        if weights_updated {
            all_triggers.push(Trigger {
                trigger_type: "weights_updated".to_string(),
                reason: "Scoring weights optimized from recall feedback".to_string(),
                urgency: 0.3,
                source_rids: vec![],
                suggested_action: "acknowledge".to_string(),
                context: std::collections::HashMap::new(),
            });
        }

        // Phase 5: Generate pattern_discovered triggers for new patterns
        // Cross-domain and entity_bridge patterns get higher urgency ("surprise" triggers)
        if pattern_result.new_patterns > 0 {
            let new_patterns = crate::patterns::get_patterns(self, None, Some("active"), 10)?;
            for p in new_patterns {
                let mut context = std::collections::HashMap::new();
                context.insert("pattern_type".to_string(), serde_json::json!(p.pattern_type));
                context.insert("confidence".to_string(), serde_json::json!(p.confidence));
                context.insert("description".to_string(), serde_json::json!(p.description));

                // Cross-domain patterns are "surprise" discoveries — boost urgency
                let (urgency, trigger_type, action) = match p.pattern_type.as_str() {
                    "cross_domain" => {
                        // Extract domain info from context for richer trigger
                        if let Some(ctx) = p.context.as_object() {
                            if let (Some(da), Some(db)) = (
                                ctx.get("domain_a").and_then(|v| v.as_str()),
                                ctx.get("domain_b").and_then(|v| v.as_str()),
                            ) {
                                context.insert("domain_a".to_string(), serde_json::json!(da));
                                context.insert("domain_b".to_string(), serde_json::json!(db));
                            }
                        }
                        (
                            (p.confidence * 0.8).max(0.5),
                            "surprise_connection",
                            "explore_cross_domain",
                        )
                    }
                    "entity_bridge" => {
                        // Entity bridges reveal hidden connectors between domains
                        (
                            (p.confidence * 0.7).max(0.4),
                            "entity_bridge_discovered",
                            "explore_entity_bridge",
                        )
                    }
                    _ => (
                        p.confidence * 0.5,
                        "pattern_discovered",
                        "explore_pattern",
                    ),
                };

                // Add session context if available
                let active_sessions = self.active_sessions.read();
                if !active_sessions.is_empty() {
                    let session_ids: Vec<&String> = active_sessions.values().collect();
                    context.insert(
                        "active_sessions".to_string(),
                        serde_json::json!(session_ids),
                    );
                }

                all_triggers.push(Trigger {
                    trigger_type: trigger_type.to_string(),
                    reason: format!("New pattern discovered: {}", p.description),
                    urgency,
                    source_rids: p.evidence_rids,
                    suggested_action: action.to_string(),
                    context,
                });
            }
        }

        // Phase 5.5: Session awareness triggers
        // Check for gaps between sessions and surface context from last session
        {
            let conn = self.conn();
            let mut session_stmt = conn.prepare(
                "SELECT session_id, client_id, ended_at, summary, avg_valence, memory_count, topics \
                 FROM sessions WHERE status = 'ended' \
                 ORDER BY ended_at DESC LIMIT 1",
            )?;
            if let Ok((session_id, client_id, ended_at, summary, avg_valence, memory_count, topics_json)) =
                session_stmt.query_row([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<f64>>(4)?,
                        row.get::<_, i64>(5)?,
                        row.get::<_, String>(6)?,
                    ))
                })
            {
                let gap_hours = (ts - ended_at) / 3600.0;
                let topics: Vec<String> = serde_json::from_str(&topics_json).unwrap_or_default();

                // Generate session awareness trigger if there's a meaningful gap
                if gap_hours > 4.0 {
                    let mut context = std::collections::HashMap::new();
                    context.insert("last_session_id".to_string(), serde_json::json!(session_id));
                    context.insert("client_id".to_string(), serde_json::json!(client_id));
                    context.insert("gap_hours".to_string(), serde_json::json!(gap_hours));
                    context.insert("last_session_memory_count".to_string(), serde_json::json!(memory_count));
                    context.insert("last_session_topics".to_string(), serde_json::json!(topics));
                    if let Some(ref s) = summary {
                        context.insert("last_session_summary".to_string(), serde_json::json!(s));
                    }
                    if let Some(v) = avg_valence {
                        context.insert("last_session_valence".to_string(), serde_json::json!(v));
                    }

                    let urgency = if gap_hours > 72.0 { 0.7 } else if gap_hours > 24.0 { 0.5 } else { 0.3 };
                    let reason = if gap_hours > 24.0 {
                        format!(
                            "It's been {:.0} hours since your last session. Last time: {} memories stored{}.",
                            gap_hours,
                            memory_count,
                            summary.as_ref().map(|s| format!(" — {}", s)).unwrap_or_default(),
                        )
                    } else {
                        format!(
                            "Welcome back ({:.0}h gap). Last session: {} memories{}.",
                            gap_hours,
                            memory_count,
                            if !topics.is_empty() {
                                format!(" about {}", topics.iter().take(3).cloned().collect::<Vec<_>>().join(", "))
                            } else {
                                String::new()
                            },
                        )
                    };

                    all_triggers.push(Trigger {
                        trigger_type: "session_awareness".to_string(),
                        reason,
                        urgency,
                        source_rids: vec![],
                        suggested_action: "acknowledge".to_string(),
                        context,
                    });
                }
            }
        }

        // Phase 6: Deduplicate, cooldown-filter, persist triggers
        let filtered = crate::triggers::filter_and_persist_triggers(self, all_triggers, ts)?;

        // Phase 7: Sort by urgency, truncate
        let mut final_triggers = filtered;
        final_triggers.sort_by(|a, b| {
            b.urgency
                .partial_cmp(&a.urgency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        final_triggers.truncate(config.max_triggers);

        // Phase 7: Derive personality traits
        let personality_updated = if config.run_personality {
            crate::personality::derive_personality(self).is_ok()
        } else {
            false
        };

        // Record last_think_at
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_think_at', ?1)",
            params![ts.to_string()],
        )?;

        // Log think op (informational, not materialized on remote)
        self.log_op(
            "think",
            None,
            &serde_json::json!({
                "triggers_count": final_triggers.len(),
                "consolidation_count": consolidation_count,
                "conflicts_found": conflicts_found,
                "new_patterns": pattern_result.new_patterns,
            }),
            None,
        )?;

        let duration_ms = start.elapsed_ms();

        Ok(ThinkResult {
            triggers: final_triggers,
            consolidation_count,
            conflicts_found,
            patterns_new: pattern_result.new_patterns,
            patterns_updated: pattern_result.updated_patterns,
            expired_triggers: expired,
            personality_updated,
            duration_ms,
        })
    }

    // ── Gossip expansion triggers (V14) ──

    /// Check substitution categories that are ready for LLM gossip expansion.
    ///
    /// Conditions per category:
    /// - At least 3 confirmed members (source in seed/user_confirmed)
    /// - No gossip trigger for this category in last 7 days
    fn check_gossip_triggers(&self) -> Result<Vec<Trigger>> {
        let mut triggers = Vec::new();
        let conn = self.conn();

        // Find categories with enough confirmed members AND at least one non-seed member
        // (seed-only categories don't need gossip expansion until the user has interacted)
        let mut stmt = conn.prepare(
            "SELECT c.id, c.name,
                    COUNT(*) as confirmed_count,
                    GROUP_CONCAT(m.token_display, ', ') as member_list
             FROM substitution_categories c
             JOIN substitution_members m ON m.category_id = c.id
             WHERE c.status IN ('active', 'provisional')
               AND m.status = 'active'
               AND m.source IN ('seed', 'user_confirmed')
             GROUP BY c.id
             HAVING confirmed_count >= 3
               AND SUM(CASE WHEN m.source != 'seed' THEN 1 ELSE 0 END) >= 1"
        )?;

        let candidates: Vec<(String, String, i64, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let cooldown_secs = 7.0 * 86400.0; // 7 days

        for (cat_id, cat_name, _count, member_list) in candidates {
            let cooldown_key = format!("gossip_{}", cat_name);

            // Check cooldown
            let recent: bool = conn.query_row(
                "SELECT COUNT(*) > 0 FROM trigger_log
                 WHERE cooldown_key = ?1 AND created_at > ?2",
                params![cooldown_key, crate::time::now_secs() - cooldown_secs],
                |row| row.get(0),
            ).unwrap_or(false);

            if recent {
                continue;
            }

            let members: Vec<String> = member_list
                .split(", ")
                .map(|s| s.to_string())
                .collect();

            let mut context = std::collections::HashMap::new();
            context.insert("category_id".to_string(), serde_json::json!(cat_id));
            context.insert("category_name".to_string(), serde_json::json!(cat_name));
            context.insert("confirmed_members".to_string(), serde_json::json!(members));

            triggers.push(Trigger {
                trigger_type: "gossip_expand_category".to_string(),
                urgency: 0.3,
                reason: format!(
                    "Category '{}' has {} confirmed members and is ready for vocabulary expansion",
                    cat_name, members.len()
                ),
                suggested_action: format!(
                    "Ask LLM to suggest additional members for the '{}' category",
                    cat_name
                ),
                source_rids: vec![],
                context,
            });
        }

        Ok(triggers)
    }

    // ── Trigger lifecycle API ──

    /// Mark a trigger as delivered (surfaced to host app).
    pub fn deliver_trigger(&self, trigger_id: &str) -> Result<bool> {
        let ts = now();
        let changes = self.conn().execute(
            "UPDATE trigger_log SET status = 'delivered', delivered_at = ?1 \
             WHERE trigger_id = ?2 AND status = 'pending'",
            params![ts, trigger_id],
        )?;
        if changes > 0 {
            self.log_op(
                "trigger_deliver",
                Some(trigger_id),
                &serde_json::json!({"trigger_id": trigger_id, "delivered_at": ts}),
                None,
            )?;
        }
        Ok(changes > 0)
    }

    /// Mark a trigger as acknowledged (user saw it).
    pub fn acknowledge_trigger(&self, trigger_id: &str) -> Result<bool> {
        let ts = now();
        let changes = self.conn().execute(
            "UPDATE trigger_log SET status = 'acknowledged', acknowledged_at = ?1 \
             WHERE trigger_id = ?2 AND status = 'delivered'",
            params![ts, trigger_id],
        )?;
        if changes > 0 {
            self.log_op(
                "trigger_ack",
                Some(trigger_id),
                &serde_json::json!({"trigger_id": trigger_id, "acknowledged_at": ts}),
                None,
            )?;
        }
        Ok(changes > 0)
    }

    /// Mark a trigger as acted upon.
    pub fn act_on_trigger(&self, trigger_id: &str) -> Result<bool> {
        let ts = now();
        let changes = self.conn().execute(
            "UPDATE trigger_log SET status = 'acted', acted_at = ?1 \
             WHERE trigger_id = ?2 AND status IN ('delivered', 'acknowledged')",
            params![ts, trigger_id],
        )?;
        if changes > 0 {
            self.log_op(
                "trigger_act",
                Some(trigger_id),
                &serde_json::json!({"trigger_id": trigger_id, "acted_at": ts}),
                None,
            )?;
        }
        Ok(changes > 0)
    }

    /// Dismiss a trigger (user doesn't want to act on it).
    pub fn dismiss_trigger(&self, trigger_id: &str) -> Result<bool> {
        let ts = now();
        let changes = self.conn().execute(
            "UPDATE trigger_log SET status = 'dismissed', acted_at = ?1 \
             WHERE trigger_id = ?2 AND status IN ('pending', 'delivered', 'acknowledged')",
            params![ts, trigger_id],
        )?;
        if changes > 0 {
            self.log_op(
                "trigger_dismiss",
                Some(trigger_id),
                &serde_json::json!({"trigger_id": trigger_id, "dismissed_at": ts}),
                None,
            )?;
        }
        Ok(changes > 0)
    }

    /// Get pending triggers sorted by urgency.
    pub fn get_pending_triggers(&self, limit: usize) -> Result<Vec<PersistedTrigger>> {
        crate::triggers::get_pending_triggers(self, limit)
    }

    /// Get trigger history with optional type filter.
    pub fn get_trigger_history(
        &self,
        trigger_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<PersistedTrigger>> {
        crate::triggers::get_trigger_history(self, trigger_type, limit)
    }

    // ── Personality API (V11) ──

    /// Get the current personality profile without recomputing.
    pub fn get_personality(&self) -> Result<crate::types::PersonalityProfile> {
        crate::personality::get_personality(self)
    }

    /// Force recompute personality traits from memory signals.
    pub fn derive_personality(&self) -> Result<crate::types::PersonalityProfile> {
        crate::personality::derive_personality(self)
    }

    /// Manually override a personality trait score (for testing).
    pub fn set_personality_trait(&self, name: &str, score: f64) -> Result<bool> {
        crate::personality::set_personality_trait(self, name, score)
    }

    /// Get detected patterns.
    pub fn get_patterns(
        &self,
        pattern_type: Option<&str>,
        status: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Pattern>> {
        crate::patterns::get_patterns(self, pattern_type, status, limit)
    }
}
