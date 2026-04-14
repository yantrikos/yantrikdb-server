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
        let session_id = self.active_sessions.read().get(namespace).cloned();

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
        self.vec_index.write().insert(&rid, embedding)?;

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

        // Auto-link memory to entities. Two passes:
        //   1. Heuristic extraction from text — seeds new proper-noun entities
        //      into `entities` + `graph_index` so conflict detection has data
        //      to scan even when the user never calls `/v1/relate`.
        //   2. Match against all known entities in `graph_index` (catches
        //      entities relate()d earlier whose names don't follow the
        //      capitalization heuristic, e.g. lowercase product names).
        {
            let text_tokens = crate::graph::tokenize(text);
            let heuristic_entities = crate::graph::extract_heuristic_entities(text);

            // Seed heuristic entities into the `entities` table (idempotent).
            if !heuristic_entities.is_empty() {
                let conn = self.conn();
                for entity in &heuristic_entities {
                    let entity_type = crate::graph::classify_entity_type(entity);
                    conn.execute(
                        "INSERT INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
                         VALUES (?1, ?2, ?3, ?3, 1) \
                         ON CONFLICT(name) DO UPDATE SET \
                            last_seen = ?3, \
                            mention_count = mention_count + 1, \
                            entity_type = CASE \
                                WHEN entity_type = 'unknown' AND ?2 != 'unknown' THEN ?2 \
                                ELSE entity_type END",
                        params![entity, entity_type, ts],
                    )?;
                }
            }

            // Compose the candidate set: heuristic + already-known entities.
            let mut candidates: std::collections::HashSet<String> =
                heuristic_entities.iter().cloned().collect();
            for known in self.graph_index.read().all_entity_names() {
                if crate::graph::entity_matches_text(&known, &text_tokens) {
                    candidates.insert(known);
                }
            }

            if !candidates.is_empty() {
                {
                    let conn = self.conn();
                    for entity in &candidates {
                        conn.execute(
                            "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                            params![rid, entity],
                        )?;
                    }
                }
                let mut gi = self.graph_index.write();
                for entity in &candidates {
                    let entity_type = crate::graph::classify_entity_type(entity);
                    gi.add_entity(entity, entity_type);
                    gi.link_memory(&rid, entity);
                }
            }

            // RFC 006 Phase 0: emit extraction audit telemetry. Log-only, no
            // behavior change. Consumed by external observability to calibrate
            // the v0.6.0 extraction whitelist against real agent-write data.
            let heuristic_vec: Vec<String> = heuristic_entities.iter().cloned().collect();
            let features = crate::graph::analyze_text_features(text, &heuristic_vec);
            tracing::info!(
                target: "yantrikdb::audit::extraction",
                namespace = %namespace,
                memory_rid = %rid,
                domain = %domain,
                source = %source,
                extractor_version = "heuristic_v1",
                char_length = features.char_length,
                sentence_count = features.sentence_count,
                entity_count = features.entity_count,
                entities_matched_in_graph = candidates.len().saturating_sub(heuristic_entities.len()),
                negation_cue_count = features.negation_cue_count,
                temporal_cue_count = features.temporal_cue_count,
                modality_cue_count = features.modality_cue_count,
                has_compound_markers = features.has_compound_markers,
                likely_assertion = features.likely_assertion,
                "extraction audit"
            );
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
        let sessions = self.active_sessions.read().clone();

        // Precompute entity candidates per memory before touching conn/graph_index.
        // Two sources:
        //   (a) heuristic extraction from text (capitalized proper-nouns)
        //   (b) match against already-known entities in graph_index
        let known_entities = self.graph_index.read().all_entity_names();
        let per_memory_linkage: Vec<(Vec<String>, std::collections::HashSet<String>)> = inputs
            .iter()
            .map(|input| {
                let text_tokens = crate::graph::tokenize(&input.text);
                let heuristic = crate::graph::extract_heuristic_entities(&input.text);
                let mut candidates: std::collections::HashSet<String> =
                    heuristic.iter().cloned().collect();
                for known in &known_entities {
                    if crate::graph::entity_matches_text(known, &text_tokens) {
                        candidates.insert(known.clone());
                    }
                }
                (heuristic, candidates)
            })
            .collect();

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

            // Persist entity linkage (SQL side). graph_index in-memory update
            // happens after conn is dropped to avoid holding two write locks.
            let batch_ts = now();
            for (rid, (heuristic, candidates)) in rids.iter().zip(per_memory_linkage.iter()) {
                for entity in heuristic {
                    let entity_type = crate::graph::classify_entity_type(entity);
                    conn.execute(
                        "INSERT INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
                         VALUES (?1, ?2, ?3, ?3, 1) \
                         ON CONFLICT(name) DO UPDATE SET \
                            last_seen = ?3, \
                            mention_count = mention_count + 1, \
                            entity_type = CASE \
                                WHEN entity_type = 'unknown' AND ?2 != 'unknown' THEN ?2 \
                                ELSE entity_type END",
                        params![entity, entity_type, batch_ts],
                    )?;
                }
                for entity in candidates {
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                        params![rid, entity],
                    )?;
                }
            }

            conn.execute_batch("RELEASE batch_record")?;
        }
        // conn dropped; now update graph_index in-memory.
        {
            let mut gi = self.graph_index.write();
            for (rid, (_, candidates)) in rids.iter().zip(per_memory_linkage.iter()) {
                for entity in candidates {
                    let entity_type = crate::graph::classify_entity_type(entity);
                    gi.add_entity(entity, entity_type);
                    gi.link_memory(rid, entity);
                }
            }
        }

        // RFC 006 Phase 0: emit one audit event per memory in the batch.
        for (rid, (input, (heuristic_entities, candidates))) in
            rids.iter().zip(inputs.iter().zip(per_memory_linkage.iter()))
        {
            let heuristic_vec: Vec<String> = heuristic_entities.iter().cloned().collect();
            let features = crate::graph::analyze_text_features(&input.text, &heuristic_vec);
            tracing::info!(
                target: "yantrikdb::audit::extraction",
                namespace = %input.namespace,
                memory_rid = %rid,
                domain = %input.domain,
                source = %input.source,
                extractor_version = "heuristic_v1",
                batch = true,
                char_length = features.char_length,
                sentence_count = features.sentence_count,
                entity_count = features.entity_count,
                entities_matched_in_graph = candidates.len().saturating_sub(heuristic_entities.len()),
                negation_cue_count = features.negation_cue_count,
                temporal_cue_count = features.temporal_cue_count,
                modality_cue_count = features.modality_cue_count,
                has_compound_markers = features.has_compound_markers,
                likely_assertion = features.likely_assertion,
                "extraction audit"
            );
        }

        // Insert into HNSW vec index + scoring cache after SQL commit
        {
            let mut vi = self.vec_index.write();
            for (rid, input) in rids.iter().zip(inputs.iter()) {
                vi.insert(rid, &input.embedding)?;
            }
        }
        // vec_index dropped, now scoring_cache
        {
            let mut cache = self.scoring_cache.write();
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
