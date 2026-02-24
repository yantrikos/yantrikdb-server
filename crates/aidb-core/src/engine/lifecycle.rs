use rusqlite::params;

use crate::error::{AidbError, Result};
use crate::scoring;
use crate::types::*;

use super::{now, embedding_hash, AIDB};

impl AIDB {
    /// Get a single memory by RID.
    #[tracing::instrument(skip(self))]
    pub fn get(&self, rid: &str) -> Result<Option<Memory>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM memories WHERE rid = ?1",
        )?;

        let result = stmt.query_row(params![rid], |row| {
            Ok((
                row.get::<_, String>("rid")?,
                row.get::<_, String>("type")?,
                row.get::<_, String>("text")?,
                row.get::<_, f64>("created_at")?,
                row.get::<_, f64>("importance")?,
                row.get::<_, f64>("valence")?,
                row.get::<_, f64>("half_life")?,
                row.get::<_, f64>("last_access")?,
                row.get::<_, i64>("access_count")?,
                row.get::<_, String>("consolidation_status")?,
                row.get::<_, String>("storage_tier")?,
                row.get::<_, Option<String>>("consolidated_into")?,
                row.get::<_, String>("metadata")?,
                row.get::<_, String>("namespace")?,
            ))
        });

        match result {
            Ok(row) => {
                let text = self.decrypt_text(&row.2)?;
                let meta_str = self.decrypt_text(&row.12)?;
                let metadata: serde_json::Value = serde_json::from_str(&meta_str)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                Ok(Some(Memory {
                    rid: row.0,
                    memory_type: row.1,
                    text,
                    created_at: row.3,
                    importance: row.4,
                    valence: row.5,
                    half_life: row.6,
                    last_access: row.7,
                    access_count: row.8 as u32,
                    consolidation_status: row.9,
                    storage_tier: row.10,
                    consolidated_into: row.11,
                    metadata,
                    namespace: row.13,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Find memories that have decayed below a threshold.
    #[tracing::instrument(skip(self))]
    pub fn decay(&self, threshold: f64) -> Result<Vec<DecayedMemory>> {
        let ts = now();
        let mut stmt = self.conn.prepare(
            "SELECT rid, text, importance, half_life, last_access, type FROM memories \
             WHERE consolidation_status = 'active'",
        )?;

        let mut decayed = Vec::new();
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>("rid")?,
                row.get::<_, String>("text")?,
                row.get::<_, f64>("importance")?,
                row.get::<_, f64>("half_life")?,
                row.get::<_, f64>("last_access")?,
                row.get::<_, String>("type")?,
            ))
        })?;

        for row in rows {
            let (rid, stored_text, importance, half_life, last_access, mem_type) = row?;
            let elapsed = ts - last_access;
            let score = scoring::decay_score(importance, half_life, elapsed);
            if score < threshold {
                let text = self.decrypt_text(&stored_text)?;
                decayed.push(DecayedMemory {
                    rid,
                    text,
                    memory_type: mem_type,
                    original_importance: importance,
                    current_score: score,
                    days_since_access: elapsed / 86400.0,
                });
            }
        }

        Ok(decayed)
    }

    /// Tombstone a memory. Returns true if the memory was found and tombstoned.
    #[tracing::instrument(skip(self))]
    pub fn forget(&self, rid: &str) -> Result<bool> {
        let ts = now();
        let changes = self.conn.execute(
            "UPDATE memories SET consolidation_status = 'tombstoned', updated_at = ?1 WHERE rid = ?2",
            params![ts, rid],
        )?;

        if changes > 0 {
            self.vec_index.borrow_mut().remove(rid);
            self.graph_index.borrow_mut().unlink_memory(rid);
            // Remove from scoring cache (tombstoned memories excluded)
            self.cache_remove(rid);
            self.log_op(
                "forget",
                Some(rid),
                &serde_json::json!({
                    "rid": rid,
                    "updated_at": ts,
                }),
                None,
            )?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// User-initiated memory correction.
    ///
    /// Creates a new corrected memory and tombstones the original.
    #[tracing::instrument(skip(self, new_embedding))]
    pub fn correct(
        &self,
        rid: &str,
        new_text: &str,
        new_importance: Option<f64>,
        new_valence: Option<f64>,
        new_embedding: &[f32],
        correction_note: Option<&str>,
    ) -> Result<CorrectionResult> {
        let original = self
            .get(rid)?
            .ok_or_else(|| AidbError::NotFound(format!("memory: {}", rid)))?;

        let ts = now();
        let importance = new_importance.unwrap_or(original.importance);
        let valence = new_valence.unwrap_or(original.valence);
        let meta = serde_json::json!({
            "corrected_from": rid,
            "correction_note": correction_note,
            "original_text": original.text,
        });

        // Create the corrected memory (logs a "record" op)
        let new_rid = self.record(
            new_text,
            &original.memory_type,
            importance,
            valence,
            original.half_life,
            &meta,
            new_embedding,
            &original.namespace,
        )?;

        // Tombstone the original (logs a "forget" op)
        self.forget(rid)?;

        // Transfer edges from original to corrected memory
        let edges = self.get_edges(rid)?;
        for edge in &edges {
            if edge.src == rid {
                self.relate(&new_rid, &edge.dst, &edge.rel_type, edge.weight)?;
            } else if edge.dst == rid {
                self.relate(&edge.src, &new_rid, &edge.rel_type, edge.weight)?;
            }
        }

        // Log a "correct" op that bundles the correction semantics
        let emb_hash = embedding_hash(new_embedding);
        self.log_op(
            "correct",
            Some(&new_rid),
            &serde_json::json!({
                "original_rid": rid,
                "new_rid": new_rid,
                "text": new_text,
                "type": original.memory_type,
                "importance": importance,
                "valence": valence,
                "half_life": original.half_life,
                "created_at": ts,
                "metadata": meta,
                "correction_note": correction_note,
            }),
            Some(&emb_hash),
        )?;

        // Auto-resolve any open conflicts involving the original rid
        let related_conflicts: Vec<String> = self
            .conn
            .prepare(
                "SELECT conflict_id FROM conflicts
                 WHERE status = 'open' AND (memory_a = ?1 OR memory_b = ?1)",
            )?
            .query_map(params![rid], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        for cid in related_conflicts {
            let _ = self.resolve_conflict(
                &cid,
                "keep_both",
                Some(&new_rid),
                None,
                Some(&format!("Auto-resolved: original memory corrected to '{}'", new_rid)),
            );
        }

        Ok(CorrectionResult {
            original_rid: rid.to_string(),
            corrected_rid: new_rid,
            original_tombstoned: true,
        })
    }
}
