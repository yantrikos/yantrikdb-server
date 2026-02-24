use rusqlite::params;

use crate::error::Result;
use crate::types::*;

use super::{now, AIDB};

impl AIDB {
    // ── Cognition loop (V3) ──

    /// Run the full cognition loop: trigger detection, consolidation, conflict
    /// scanning, and pattern mining. Returns a prioritized list of triggers
    /// and summary of actions taken.
    pub fn think(&self, config: &ThinkConfig) -> Result<ThinkResult> {
        let start = std::time::Instant::now();
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

        // Phase 2: Run consolidation if configured
        let consolidation_count = if config.run_consolidation {
            let stats = self.stats(None)?;
            if stats.active_memories >= config.min_active_memories {
                let results = crate::consolidate::consolidate(
                    self,
                    config.consolidation_sim_threshold,
                    config.consolidation_time_window_days,
                    config.consolidation_min_cluster,
                    false,
                )?;
                results.len()
            } else {
                0
            }
        } else {
            0
        };

        // Phase 3: Scan for conflicts
        let conflicts_found = if config.run_conflict_scan {
            crate::conflict::scan_conflicts(self)?.len()
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

        // Phase 5: Generate pattern_discovered triggers for new patterns
        if pattern_result.new_patterns > 0 {
            let new_patterns = crate::patterns::get_patterns(self, None, Some("active"), 5)?;
            for p in new_patterns {
                let mut context = std::collections::HashMap::new();
                context.insert("pattern_type".to_string(), serde_json::json!(p.pattern_type));
                context.insert("confidence".to_string(), serde_json::json!(p.confidence));
                context.insert("description".to_string(), serde_json::json!(p.description));

                all_triggers.push(Trigger {
                    trigger_type: "pattern_discovered".to_string(),
                    reason: format!("New pattern discovered: {}", p.description),
                    urgency: p.confidence * 0.5,
                    source_rids: p.evidence_rids,
                    suggested_action: "explore_pattern".to_string(),
                    context,
                });
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

        // Record last_think_at
        self.conn.execute(
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

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(ThinkResult {
            triggers: final_triggers,
            consolidation_count,
            conflicts_found,
            patterns_new: pattern_result.new_patterns,
            patterns_updated: pattern_result.updated_patterns,
            expired_triggers: expired,
            duration_ms,
        })
    }

    // ── Trigger lifecycle API ──

    /// Mark a trigger as delivered (surfaced to host app).
    pub fn deliver_trigger(&self, trigger_id: &str) -> Result<bool> {
        let ts = now();
        let changes = self.conn.execute(
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
        let changes = self.conn.execute(
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
        let changes = self.conn.execute(
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
        let changes = self.conn.execute(
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
