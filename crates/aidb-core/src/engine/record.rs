use rusqlite::params;

use crate::error::Result;
use crate::serde_helpers::serialize_f32;
use crate::types::*;

use super::{now, embedding_hash, AIDB};

impl AIDB {
    /// Store a new memory and return its RID.
    #[tracing::instrument(skip(self, metadata, embedding), fields(memory_type, namespace))]
    pub fn record(
        &self,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        embedding: &[f32],
        namespace: &str,
    ) -> Result<String> {
        let rid = uuid7::uuid7().to_string();
        let ts = now();
        let emb_blob = serialize_f32(embedding);
        let meta_str = serde_json::to_string(metadata)?;

        // Encrypt fields if encryption is enabled
        let stored_text = self.encrypt_text(text)?;
        let stored_meta = self.encrypt_text(&meta_str)?;
        let stored_emb = self.encrypt_embedding(&emb_blob)?;

        self.conn.execute(
            "INSERT INTO memories \
             (rid, type, text, embedding, created_at, updated_at, importance, \
              half_life, last_access, valence, metadata, namespace) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![rid, memory_type, stored_text, stored_emb, ts, ts, importance, half_life, ts, valence, stored_meta, namespace],
        )?;

        // Insert into vector index
        self.vec_index.borrow_mut().insert(&rid, embedding)?;

        // Insert into scoring cache
        self.cache_insert(rid.clone(), ScoringRow {
            created_at: ts,
            importance,
            half_life,
            last_access: ts,
            access_count: 0,
            valence,
            consolidation_status: "active".to_string(),
            memory_type: memory_type.to_string(),
            namespace: namespace.to_string(),
        });

        let emb_hash = embedding_hash(embedding);
        self.log_op(
            "record",
            Some(&rid),
            &serde_json::json!({
                "rid": rid,
                "type": memory_type,
                "text": text,
                "importance": importance,
                "valence": valence,
                "half_life": half_life,
                "metadata": metadata,
                "created_at": ts,
                "updated_at": ts,
                "namespace": namespace,
            }),
            Some(&emb_hash),
        )?;

        Ok(rid)
    }

    /// Record multiple memories in a single transaction.
    /// Uses SAVEPOINT for atomicity while keeping `&self` (no `&mut self`).
    #[tracing::instrument(skip(self, inputs), fields(batch_size = inputs.len()))]
    pub fn record_batch(&self, inputs: &[RecordInput]) -> Result<Vec<String>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }

        self.conn.execute_batch("SAVEPOINT batch_record")?;

        let mut rids = Vec::with_capacity(inputs.len());
        for input in inputs {
            let rid = uuid7::uuid7().to_string();
            let ts = now();
            let emb_blob = serialize_f32(&input.embedding);
            let meta_str = serde_json::to_string(&input.metadata)?;

            // Encrypt fields if encryption is enabled
            let stored_text = self.encrypt_text(&input.text)?;
            let stored_meta = self.encrypt_text(&meta_str)?;
            let stored_emb = self.encrypt_embedding(&emb_blob)?;

            let result = self.conn.execute(
                "INSERT INTO memories \
                 (rid, type, text, embedding, created_at, updated_at, importance, \
                  half_life, last_access, valence, metadata, namespace) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![rid, input.memory_type, stored_text, stored_emb, ts, ts,
                        input.importance, input.half_life, ts, input.valence, stored_meta,
                        input.namespace],
            );

            if let Err(e) = result {
                self.conn.execute_batch("ROLLBACK TO batch_record")?;
                return Err(e.into());
            }

            rids.push(rid);
        }

        self.conn.execute_batch("RELEASE batch_record")?;

        // Insert into HNSW vec index + scoring cache after SQL commit
        {
            let mut vi = self.vec_index.borrow_mut();
            let mut cache = self.scoring_cache.borrow_mut();
            for (rid, input) in rids.iter().zip(inputs.iter()) {
                let ts = now();
                vi.insert(rid, &input.embedding)?;
                cache.insert(rid.clone(), ScoringRow {
                    created_at: ts,
                    importance: input.importance,
                    half_life: input.half_life,
                    last_access: ts,
                    access_count: 0,
                    valence: input.valence,
                    consolidation_status: "active".to_string(),
                    memory_type: input.memory_type.clone(),
                    namespace: input.namespace.clone(),
                });
            }
        }

        // Log a single batch op
        self.log_op(
            "record_batch",
            None,
            &serde_json::json!({
                "count": rids.len(),
                "rids": rids,
            }),
            None,
        )?;

        Ok(rids)
    }
}
