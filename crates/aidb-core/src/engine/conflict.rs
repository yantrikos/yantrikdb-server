use rusqlite::params;

use crate::error::{AidbError, Result};
use crate::types::*;

use super::{now, AIDB};

impl AIDB {
    // ── Conflict resolution API (V2) ──

    /// Get all conflicts, optionally filtered.
    pub fn get_conflicts(
        &self,
        status: Option<&str>,
        conflict_type: Option<&str>,
        entity: Option<&str>,
        priority: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Conflict>> {
        let mut sql = String::from("SELECT * FROM conflicts WHERE 1=1");
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut idx = 1;

        if let Some(s) = status {
            sql.push_str(&format!(" AND status = ?{idx}"));
            param_values.push(Box::new(s.to_string()));
            idx += 1;
        }
        if let Some(ct) = conflict_type {
            sql.push_str(&format!(" AND conflict_type = ?{idx}"));
            param_values.push(Box::new(ct.to_string()));
            idx += 1;
        }
        if let Some(e) = entity {
            sql.push_str(&format!(" AND entity = ?{idx}"));
            param_values.push(Box::new(e.to_string()));
            idx += 1;
        }
        if let Some(p) = priority {
            sql.push_str(&format!(" AND priority = ?{idx}"));
            param_values.push(Box::new(p.to_string()));
            let _ = idx;
        }

        sql.push_str(
            " ORDER BY
            CASE priority
                WHEN 'critical' THEN 0
                WHEN 'high' THEN 1
                WHEN 'medium' THEN 2
                WHEN 'low' THEN 3
            END,
            detected_at DESC",
        );
        sql.push_str(&format!(" LIMIT {limit}"));

        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = self.conn.prepare(&sql)?;
        let conflicts = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok(Conflict {
                    conflict_id: row.get("conflict_id")?,
                    conflict_type: row.get("conflict_type")?,
                    priority: row.get("priority")?,
                    status: row.get("status")?,
                    memory_a: row.get("memory_a")?,
                    memory_b: row.get("memory_b")?,
                    entity: row.get("entity")?,
                    rel_type: row.get("rel_type")?,
                    detected_at: row.get("detected_at")?,
                    detected_by: row.get("detected_by")?,
                    detection_reason: row.get("detection_reason")?,
                    resolved_at: row.get("resolved_at")?,
                    resolved_by: row.get("resolved_by")?,
                    strategy: row.get("strategy")?,
                    winner_rid: row.get("winner_rid")?,
                    resolution_note: row.get("resolution_note")?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(conflicts)
    }

    /// Get a single conflict by ID.
    pub fn get_conflict(&self, conflict_id: &str) -> Result<Option<Conflict>> {
        let result = self.conn.query_row(
            "SELECT * FROM conflicts WHERE conflict_id = ?1",
            params![conflict_id],
            |row| {
                Ok(Conflict {
                    conflict_id: row.get("conflict_id")?,
                    conflict_type: row.get("conflict_type")?,
                    priority: row.get("priority")?,
                    status: row.get("status")?,
                    memory_a: row.get("memory_a")?,
                    memory_b: row.get("memory_b")?,
                    entity: row.get("entity")?,
                    rel_type: row.get("rel_type")?,
                    detected_at: row.get("detected_at")?,
                    detected_by: row.get("detected_by")?,
                    detection_reason: row.get("detection_reason")?,
                    resolved_at: row.get("resolved_at")?,
                    resolved_by: row.get("resolved_by")?,
                    strategy: row.get("strategy")?,
                    winner_rid: row.get("winner_rid")?,
                    resolution_note: row.get("resolution_note")?,
                })
            },
        );

        match result {
            Ok(c) => Ok(Some(c)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Resolve a conflict with a chosen strategy.
    ///
    /// Strategies:
    ///   - keep_a: tombstone memory_b, keep memory_a
    ///   - keep_b: tombstone memory_a, keep memory_b
    ///   - keep_both: mark resolved, keep both memories
    ///   - merge: create new memory (new_text required), tombstone both
    pub fn resolve_conflict(
        &self,
        conflict_id: &str,
        strategy: &str,
        winner_rid: Option<&str>,
        new_text: Option<&str>,
        resolution_note: Option<&str>,
    ) -> Result<ConflictResolutionResult> {
        let conflict = self
            .get_conflict(conflict_id)?
            .ok_or_else(|| AidbError::NotFound(format!("conflict: {}", conflict_id)))?;

        if conflict.status != "open" {
            return Err(AidbError::SyncError(format!(
                "conflict {} is already {}",
                conflict_id, conflict.status
            )));
        }

        let ts = now();
        let actor_id = self.actor_id.clone();
        let mut loser_tombstoned = false;
        let mut new_memory_rid = None;

        let (effective_winner, loser_rid) = match strategy {
            "keep_a" => {
                let winner = winner_rid.unwrap_or(&conflict.memory_a);
                let loser = if winner == conflict.memory_a {
                    &conflict.memory_b
                } else {
                    &conflict.memory_a
                };
                self.forget(loser)?;
                loser_tombstoned = true;
                (Some(winner.to_string()), Some(loser.to_string()))
            }
            "keep_b" => {
                let winner = winner_rid.unwrap_or(&conflict.memory_b);
                let loser = if winner == conflict.memory_b {
                    &conflict.memory_a
                } else {
                    &conflict.memory_b
                };
                self.forget(loser)?;
                loser_tombstoned = true;
                (Some(winner.to_string()), Some(loser.to_string()))
            }
            "keep_both" => (None, None),
            "merge" => {
                let text = new_text.ok_or_else(|| {
                    AidbError::SyncError("merge strategy requires new_text".to_string())
                })?;
                let mem_a = self.get(&conflict.memory_a)?;
                let mem_b = self.get(&conflict.memory_b)?;
                let imp_a = mem_a.as_ref().map(|m| m.importance).unwrap_or(0.5);
                let imp_b = mem_b.as_ref().map(|m| m.importance).unwrap_or(0.5);
                let merged_importance = imp_a.max(imp_b);

                let zero_emb = vec![0.0f32; self.embedding_dim];
                let meta = serde_json::json!({
                    "merged_from": [conflict.memory_a, conflict.memory_b],
                    "conflict_id": conflict_id,
                });
                let merge_ns = mem_a.as_ref().map(|m| m.namespace.as_str()).unwrap_or("default");
                let rid = self.record(
                    text,
                    "semantic",
                    merged_importance,
                    0.0,
                    604800.0,
                    &meta,
                    &zero_emb,
                    merge_ns,
                )?;
                new_memory_rid = Some(rid.clone());

                self.forget(&conflict.memory_a)?;
                self.forget(&conflict.memory_b)?;
                loser_tombstoned = true;

                (Some(rid), None)
            }
            _ => {
                return Err(AidbError::SyncError(format!(
                    "unknown resolution strategy: {}",
                    strategy
                )));
            }
        };

        // Update the conflict record
        self.conn.execute(
            "UPDATE conflicts SET
             status = 'resolved',
             resolved_at = ?1,
             resolved_by = ?2,
             strategy = ?3,
             winner_rid = ?4,
             resolution_note = ?5
             WHERE conflict_id = ?6",
            params![ts, actor_id, strategy, effective_winner, resolution_note, conflict_id],
        )?;

        // Log to oplog for replication
        self.log_op(
            "conflict_resolve",
            Some(conflict_id),
            &serde_json::json!({
                "conflict_id": conflict_id,
                "strategy": strategy,
                "winner_rid": effective_winner,
                "loser_rid": loser_rid,
                "new_text": new_text,
                "resolution_note": resolution_note,
                "resolved_at": ts,
                "resolved_by": actor_id,
            }),
            None,
        )?;

        Ok(ConflictResolutionResult {
            conflict_id: conflict_id.to_string(),
            strategy: strategy.to_string(),
            winner_rid: effective_winner,
            loser_tombstoned,
            new_memory_rid,
        })
    }

    /// Dismiss a conflict (mark as not-a-conflict).
    pub fn dismiss_conflict(&self, conflict_id: &str, note: Option<&str>) -> Result<()> {
        let ts = now();
        let actor_id = self.actor_id.clone();

        self.conn.execute(
            "UPDATE conflicts SET
             status = 'dismissed',
             resolved_at = ?1,
             resolved_by = ?2,
             strategy = 'keep_both',
             resolution_note = ?3
             WHERE conflict_id = ?4 AND status = 'open'",
            params![ts, actor_id, note, conflict_id],
        )?;

        self.log_op(
            "conflict_resolve",
            Some(conflict_id),
            &serde_json::json!({
                "conflict_id": conflict_id,
                "strategy": "keep_both",
                "resolution_note": note,
                "resolved_at": ts,
                "resolved_by": actor_id,
                "dismissed": true,
            }),
            None,
        )?;

        Ok(())
    }
}
