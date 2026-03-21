//! Self-improving procedural memory.
//!
//! Procedural memories store learned behaviors and strategies — "what worked"
//! in specific contexts. This module provides:
//!
//! 1. Context-matched surfacing: at task start, retrieve procedural memories
//!    matching the current task context (domain, entities, patterns).
//! 2. Reinforcement tracking: procedural memories that get recalled and used
//!    are automatically reinforced; unused ones decay naturally.
//! 3. Record with context: store a procedural memory with task context metadata
//!    so it can be matched to similar future tasks.

use rusqlite::params;

use crate::error::Result;
use crate::types::RecallResult;

use super::{now, YantrikDB};

impl YantrikDB {
    /// Surface procedural memories relevant to the current task context.
    ///
    /// Combines HNSW similarity search (filtered to memory_type="procedural")
    /// with entity-based boosting: procedural memories linked to the same
    /// entities as the query get a boost.
    ///
    /// This is the "what worked before for tasks like this?" query.
    pub fn surface_procedural(
        &self,
        query_embedding: &[f32],
        query_text: Option<&str>,
        domain: Option<&str>,
        top_k: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<RecallResult>> {
        // Use the full recall pipeline but filtered to procedural memories
        self.recall(
            query_embedding,
            top_k,
            None,                     // no time window
            Some("procedural"),       // filter to procedural type
            false,                    // don't include consolidated
            query_text.is_some(),     // expand entities if we have query text
            query_text,
            true,                     // skip_reinforce — we'll reinforce manually below
            namespace,
            domain,
            None,                     // no source filter
        )
    }

    /// Record a procedural memory with task context.
    ///
    /// This is a convenience wrapper around record() that:
    /// - Sets memory_type="procedural"
    /// - Stores task context in metadata for future matching
    /// - Sets appropriate defaults (high half_life since procedures are long-lived)
    pub fn record_procedural(
        &self,
        text: &str,
        embedding: &[f32],
        domain: &str,
        task_context: &str,
        effectiveness: f64,
        namespace: &str,
    ) -> Result<String> {
        let metadata = serde_json::json!({
            "task_context": task_context,
            "effectiveness": effectiveness,
            "source_type": "procedural",
        });

        self.record(
            text,
            "procedural",
            effectiveness.clamp(0.0, 1.0),  // importance = effectiveness
            0.0,                             // neutral valence
            604800.0 * 4.0,                  // 4-week half life (procedures are long-lived)
            &metadata,
            embedding,
            namespace,
            effectiveness.clamp(0.0, 1.0),   // certainty = effectiveness
            domain,
            "inference",                      // source = inference (learned by the system)
            None,
        )
    }

    /// Update a procedural memory's effectiveness based on outcome.
    ///
    /// If a procedural memory was surfaced and the agent used it successfully,
    /// increase its importance (reinforcement). If it was surfaced but not useful,
    /// decrease importance (weakening).
    pub fn reinforce_procedural(
        &self,
        rid: &str,
        outcome: f64,   // 1.0 = fully successful, 0.0 = not useful
    ) -> Result<bool> {
        let ts = now();
        let conn = &self.conn;

        // Get current importance
        let current_importance: f64 = match conn.query_row(
            "SELECT importance FROM memories WHERE rid = ?1 AND type = 'procedural' \
             AND consolidation_status = 'active'",
            params![rid],
            |row| row.get(0),
        ) {
            Ok(imp) => imp,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(false),
            Err(e) => return Err(e.into()),
        };

        // Exponential moving average: importance moves toward outcome
        // Alpha = 0.3 means recent outcomes have moderate influence
        let alpha = 0.3;
        let new_importance = ((1.0 - alpha) * current_importance + alpha * outcome).clamp(0.0, 1.0);

        // Also update certainty to reflect confidence in this procedure
        let new_certainty = ((1.0 - alpha) * current_importance + alpha * outcome).clamp(0.0, 1.0);

        conn.execute(
            "UPDATE memories SET importance = ?1, certainty = ?2, last_access = ?3 \
             WHERE rid = ?4",
            params![new_importance, new_certainty, ts, rid],
        )?;

        // Update scoring cache
        if let Some(cached) = self.scoring_cache.borrow_mut().get_mut(rid) {
            cached.importance = new_importance;
            cached.certainty = new_certainty;
            cached.last_access = ts;
        }

        // Log reinforcement
        self.log_op(
            "reinforce_procedural",
            Some(rid),
            &serde_json::json!({
                "rid": rid,
                "outcome": outcome,
                "old_importance": current_importance,
                "new_importance": new_importance,
            }),
            None,
        )?;

        Ok(true)
    }

    /// Get procedural memory statistics: count by domain, average effectiveness.
    pub fn procedural_stats(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, i64, f64)>> {
        let sql = if namespace.is_some() {
            "SELECT domain, COUNT(*), AVG(importance) FROM memories \
             WHERE type = 'procedural' AND consolidation_status = 'active' \
             AND namespace = ?1 \
             GROUP BY domain ORDER BY COUNT(*) DESC"
        } else {
            "SELECT domain, COUNT(*), AVG(importance) FROM memories \
             WHERE type = 'procedural' AND consolidation_status = 'active' \
             GROUP BY domain ORDER BY COUNT(*) DESC"
        };

        let mut stmt = self.conn.prepare(sql)?;
        let rows = if let Some(ns) = namespace {
            stmt.query_map(params![ns], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, f64>(2)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, f64>(2)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
        };

        Ok(rows)
    }
}
