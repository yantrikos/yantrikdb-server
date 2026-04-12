use rusqlite::params;

use crate::error::Result;
use crate::types::LearnedWeights;

use super::{now, YantrikDB};

impl YantrikDB {
    /// Record feedback on a recall result.
    ///
    /// The AI agent calls this when it knows a result was relevant or irrelevant.
    /// Accumulated feedback powers the adaptive learning loop.
    pub fn recall_feedback(
        &self,
        query_text: Option<&str>,
        query_embedding: Option<&[f32]>,
        rid: &str,
        feedback: &str, // "relevant" | "irrelevant"
        score_at_retrieval: Option<f64>,
        rank_at_retrieval: Option<i32>,
    ) -> Result<()> {
        let ts = now();
        let emb_blob = query_embedding.map(|e| crate::serde_helpers::serialize_f32(e));

        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO recall_feedback (query_text, query_embedding, rid, feedback, \
             score_at_retrieval, rank_at_retrieval, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                query_text,
                emb_blob,
                rid,
                feedback,
                score_at_retrieval,
                rank_at_retrieval,
                ts,
            ],
        )?;

        // Update feedback count in learned_weights
        conn.execute(
            "UPDATE learned_weights SET feedback_count = feedback_count + 1 WHERE id = 1",
            [],
        )?;

        Ok(())
    }

    /// Load the current learned weights from the database.
    pub fn load_learned_weights(&self) -> Result<LearnedWeights> {
        let conn = self.conn.lock();
        let result = conn.query_row(
            "SELECT w_sim, w_decay, w_recency, gate_tau, alpha_imp, keyword_boost, generation \
             FROM learned_weights WHERE id = 1",
            [],
            |row| {
                Ok(LearnedWeights {
                    w_sim: row.get(0)?,
                    w_decay: row.get(1)?,
                    w_recency: row.get(2)?,
                    gate_tau: row.get(3)?,
                    alpha_imp: row.get(4)?,
                    keyword_boost: row.get(5)?,
                    generation: row.get(6)?,
                })
            },
        );

        match result {
            Ok(w) => Ok(w),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(LearnedWeights::default()),
            Err(e) => Err(e.into()),
        }
    }

    /// Get the current feedback count.
    pub fn feedback_count(&self) -> Result<i64> {
        let conn = self.conn.lock();
        let count: i64 = conn.query_row(
            "SELECT COALESCE(feedback_count, 0) FROM learned_weights WHERE id = 1",
            [],
            |row| row.get(0),
        ).unwrap_or(0);
        Ok(count)
    }
}
