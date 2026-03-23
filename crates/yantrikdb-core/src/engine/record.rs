use rusqlite::params;

use crate::error::Result;
use crate::serde_helpers::serialize_f32;
use crate::types::*;

use super::{now, embedding_hash, YantrikDB};

impl YantrikDB {
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
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
    ) -> Result<String> {
        let rid = crate::id::new_id();
        let ts = now();
        let emb_blob = serialize_f32(embedding);
        let meta_str = serde_json::to_string(metadata)?;

        // Encrypt fields if encryption is enabled
        let stored_text = self.encrypt_text(text)?;
        let stored_meta = self.encrypt_text(&meta_str)?;
        let stored_emb = self.encrypt_embedding(&emb_blob)?;

        // Read active session for this namespace into a local before acquiring conn
        let session_id = self.active_sessions.read().unwrap().get(namespace).cloned();

        // Acquire conn, do all SQL, then drop before other locks
        {
            let conn = self.conn();
            conn.execute(
                "INSERT INTO memories \
                 (rid, type, text, embedding, created_at, updated_at, importance, \
                  half_life, last_access, valence, metadata, namespace, \
                  certainty, domain, source, emotional_state) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
                params![rid, memory_type, stored_text, stored_emb, ts, ts, importance, half_life, ts, valence, stored_meta, namespace,
                        certainty, domain, source, emotional_state],
            )?;

            // Auto-link to active session for this namespace
            if let Some(session_id) = &session_id {
                conn.execute(
                    "UPDATE memories SET session_id = ?1 WHERE rid = ?2",
                    params![session_id, rid],
                )?;
                conn.execute(
                    "UPDATE sessions SET memory_count = memory_count + 1 WHERE session_id = ?1",
                    params![session_id],
                )?;
            }
        }
        // conn dropped here

        // Insert into vector index (lock ordering: conn already dropped)
        self.vec_index.write().unwrap().insert(&rid, embedding)?;

        // Insert into scoring cache (conn and vec_index dropped)
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
            certainty,
            domain: domain.to_string(),
            source: source.to_string(),
            emotional_state: emotional_state.map(|s| s.to_string()),
        });

        // Auto-link memory to known entities (populates memory_entities for graph recall)
        {
            let text_tokens = crate::graph::tokenize(text);
            // Read graph_index to find matching entities, then drop read lock
            let all_entities = self.graph_index.read().unwrap().all_entity_names();
            let matching: Vec<String> = all_entities
                .into_iter()
                .filter(|entity| crate::graph::entity_matches_text(entity, &text_tokens))
                .collect();

            if !matching.is_empty() {
                // Acquire conn for entity inserts
                {
                    let conn = self.conn();
                    for entity in &matching {
                        conn.execute(
                            "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                            params![rid, entity],
                        )?;
                    }
                }
                // conn dropped, now acquire graph_index write lock
                let mut gi = self.graph_index.write().unwrap();
                for entity in &matching {
                    gi.link_memory(&rid, entity);
                }
            }
        }

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
                "certainty": certainty,
                "domain": domain,
                "source": source,
                "emotional_state": emotional_state,
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

        // Clone active sessions map before acquiring conn
        let sessions = self.active_sessions.read().unwrap().clone();

        let mut rids = Vec::with_capacity(inputs.len());

        // Lock conn once for the entire batch SQL work
        {
            let conn = self.conn();
            conn.execute_batch("SAVEPOINT batch_record")?;

            for input in inputs {
                let rid = crate::id::new_id();
                let ts = now();
                let emb_blob = serialize_f32(&input.embedding);
                let meta_str = serde_json::to_string(&input.metadata)?;

                // Encrypt fields if encryption is enabled
                let stored_text = self.encrypt_text(&input.text)?;
                let stored_meta = self.encrypt_text(&meta_str)?;
                let stored_emb = self.encrypt_embedding(&emb_blob)?;

                let result = conn.execute(
                    "INSERT INTO memories \
                     (rid, type, text, embedding, created_at, updated_at, importance, \
                      half_life, last_access, valence, metadata, namespace, \
                      certainty, domain, source, emotional_state) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
                    params![rid, input.memory_type, stored_text, stored_emb, ts, ts,
                            input.importance, input.half_life, ts, input.valence, stored_meta,
                            input.namespace, input.certainty, input.domain, input.source,
                            input.emotional_state],
                );

                if let Err(e) = result {
                    conn.execute_batch("ROLLBACK TO batch_record")?;
                    return Err(e.into());
                }

                rids.push(rid);
            }

            // Auto-link batch to active sessions
            for (rid, input) in rids.iter().zip(inputs.iter()) {
                if let Some(session_id) = sessions.get(&input.namespace) {
                    conn.execute(
                        "UPDATE memories SET session_id = ?1 WHERE rid = ?2",
                        params![session_id, rid],
                    )?;
                    conn.execute(
                        "UPDATE sessions SET memory_count = memory_count + 1 WHERE session_id = ?1",
                        params![session_id],
                    )?;
                }
            }

            conn.execute_batch("RELEASE batch_record")?;
        }
        // conn dropped here

        // Insert into HNSW vec index + scoring cache after SQL commit
        {
            let mut vi = self.vec_index.write().unwrap();
            for (rid, input) in rids.iter().zip(inputs.iter()) {
                vi.insert(rid, &input.embedding)?;
            }
        }
        // vec_index dropped, now scoring_cache
        {
            let mut cache = self.scoring_cache.write().unwrap();
            for (rid, input) in rids.iter().zip(inputs.iter()) {
                let ts = now();
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
                    certainty: input.certainty,
                    domain: input.domain.clone(),
                    source: input.source.clone(),
                    emotional_state: input.emotional_state.clone(),
                });
            }
        }

        // Log a single batch op (log_op locks conn internally)
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
