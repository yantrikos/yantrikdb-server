use std::collections::HashMap;

use rusqlite::params;

use crate::error::Result;
use crate::scoring;
use crate::types::*;

use super::{now, TextMetadataRow, AIDB};

impl AIDB {
    /// Retrieve memories using multi-signal fusion scoring.
    /// When `expand_entities` is true, graph edges are followed to pull in
    /// entity-connected memories that pure vector search would miss.
    pub fn recall(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
    ) -> Result<Vec<RecallResult>> {
        let ts = now();

        // Step 1: Vector candidate generation via HNSW
        let fetch_k = (top_k * 5).min(200);
        let vec_results = self.vec_index.borrow().search(query_embedding, fetch_k)?;

        if vec_results.is_empty() {
            return Ok(vec![]);
        }

        // Step 2: Score from in-memory cache (replaces fetch_memories_by_rids)
        let mut scored: Vec<RecallResult> = Vec::new();
        {
            let cache = self.scoring_cache.borrow();
            for (rid, distance) in &vec_results {
                let Some(row) = cache.get(rid) else { continue };

                // Filter: consolidation_status
                let status_ok = if include_consolidated {
                    row.consolidation_status == "active" || row.consolidation_status == "consolidated"
                } else {
                    row.consolidation_status == "active"
                };
                if !status_ok { continue; }

                // Filter: memory_type
                if let Some(mt) = memory_type {
                    if row.memory_type != mt { continue; }
                }

                // Filter: time_window
                if let Some((start, end)) = time_window {
                    if row.created_at < start || row.created_at > end { continue; }
                }

                // Filter: namespace
                if let Some(ns) = namespace {
                    if row.namespace != ns { continue; }
                }

                let sim_score = 1.0 - distance;
                let elapsed = ts - row.last_access;
                let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                let age = ts - row.created_at;
                let recency = scoring::recency_score(age);
                let composite = scoring::composite_score(sim_score, decay, recency, row.importance, row.valence);
                let why = scoring::build_why(sim_score, recency, decay, row.valence);

                scored.push(RecallResult {
                    rid: rid.clone(),
                    memory_type: row.memory_type.clone(),
                    text: String::new(),  // hydrated after top_k selection
                    created_at: row.created_at,
                    importance: row.importance,
                    valence: row.valence,
                    score: composite,
                    scores: ScoreBreakdown {
                        similarity: sim_score,
                        decay,
                        recency,
                        importance: row.importance,
                        graph_proximity: 0.0,
                    },
                    why_retrieved: why,
                    metadata: serde_json::Value::Null,  // hydrated after top_k selection
                    namespace: row.namespace.clone(),
                });
            }
        } // drop cache borrow

        // Step 3: Graph expansion (when enabled)
        if expand_entities {
            let gi = self.graph_index.borrow();
            let query_entities: Vec<(String, String, u32)> = if let Some(qt) = query_text {
                let query_tokens = crate::graph::tokenize(qt);
                gi.entity_matches_query(&query_tokens)
            } else {
                vec![]
            };

            let (base_boost, seed_entities, entity_idfs) = if !query_entities.is_empty() {
                let has_person = query_entities.iter().any(|(_, etype, _)| etype == "person");
                let factor = if has_person {
                    0.20
                } else if query_entities.len() >= 2 {
                    0.15
                } else {
                    0.12
                };
                let idfs: std::collections::HashMap<String, f64> = query_entities
                    .iter()
                    .map(|(name, _, mc)| {
                        let idf = 1.0 / (1.0 + (*mc as f64).max(1.0).ln());
                        (name.to_lowercase(), idf)
                    })
                    .collect();
                let names: Vec<String> = query_entities.into_iter().map(|(n, _, _)| n).collect();
                (factor, names, idfs)
            } else if query_text.is_none() {
                let mut seed_sorted = scored.clone();
                seed_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                let seed_count = 3.min(seed_sorted.len());
                let seed_rids: Vec<&str> = seed_sorted[..seed_count].iter().map(|r| r.rid.as_str()).collect();
                let seeds = gi.entities_for_memories(&seed_rids);
                (0.05, seeds, std::collections::HashMap::new())
            } else {
                (0.0, vec![], std::collections::HashMap::new())
            };

            const MAX_BOOST_PER_MEMORY: f64 = 0.25;
            const MAX_GRAPH_FRACTION: f64 = 0.40;

            if !seed_entities.is_empty() && base_boost > 0.0 {
                let seed_refs: Vec<&str> = seed_entities.iter().map(|s| s.as_str()).collect();
                let expanded = gi.expand_bfs(&seed_refs, 1, 20);

                let expanded_map: std::collections::HashMap<String, (u8, f64)> = expanded
                    .iter()
                    .map(|(name, hops, weight)| (name.clone(), (*hops, *weight)))
                    .collect();

                // (a) IDF-weighted additive boost for existing results
                for result in &mut scored {
                    let prox = gi.graph_proximity(&result.rid, &expanded_map);
                    if prox > 0.0 {
                        let mem_entities: Vec<String> = gi.entities_for_memory(&result.rid).into_iter().map(|s| s.to_string()).collect();

                        let mut best_idf = 1.0f64;
                        let mut connecting_entity = String::new();
                        for entity in &mem_entities {
                            if expanded_map.contains_key(entity) {
                                let idf = entity_idfs.get(&entity.to_lowercase()).copied().unwrap_or(1.0);
                                if connecting_entity.is_empty() || idf > best_idf {
                                    best_idf = idf;
                                    connecting_entity = entity.clone();
                                }
                            }
                        }

                        // Consolidation penalty: use consolidation_status as proxy
                        let cache = self.scoring_cache.borrow();
                        let consolidation_factor = cache.get(&result.rid)
                            .map(|r| if r.consolidation_status == "consolidated" { 0.5 } else { 1.0 })
                            .unwrap_or(1.0);
                        drop(cache);

                        let boost = (base_boost * prox * best_idf * consolidation_factor)
                            .min(MAX_BOOST_PER_MEMORY);
                        result.scores.graph_proximity = prox;
                        result.score += boost;
                        if !connecting_entity.is_empty() {
                            result.why_retrieved.push(format!("graph-connected via {connecting_entity}"));
                        }
                    }
                }

                // (b) Graph-only candidates: score from cache + batch embedding fetch
                let max_graph_only = ((MAX_GRAPH_FRACTION * top_k as f64).ceil() as usize).max(1);
                let all_entity_names: Vec<&str> = expanded.iter().map(|(n, _, _)| n.as_str()).collect();
                let graph_rids = gi.memories_for_entities(&all_entity_names);

                let existing_rids: std::collections::HashSet<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
                let new_rids: Vec<String> = graph_rids
                    .into_iter()
                    .filter(|r| !existing_rids.contains(r.as_str()))
                    .collect();

                // Filter graph-only candidates using cache
                let filtered_rids: Vec<String> = {
                    let cache = self.scoring_cache.borrow();
                    new_rids.into_iter()
                        .filter(|rid| {
                            let Some(row) = cache.get(rid) else { return false };
                            let status_ok = if include_consolidated {
                                row.consolidation_status == "active" || row.consolidation_status == "consolidated"
                            } else {
                                row.consolidation_status == "active"
                            };
                            if !status_ok { return false; }
                            if let Some(mt) = memory_type {
                                if row.memory_type != mt { return false; }
                            }
                            if let Some((start, end)) = time_window {
                                if row.created_at < start || row.created_at > end { return false; }
                            }
                            if let Some(ns) = namespace {
                                if row.namespace != ns { return false; }
                            }
                            true
                        })
                        .take(max_graph_only)
                        .collect()
                };

                if !filtered_rids.is_empty() {
                    // Batch fetch embeddings for cosine similarity
                    let rid_refs: Vec<&str> = filtered_rids.iter().map(|r| r.as_str()).collect();
                    let embeddings = self.fetch_embeddings_by_rids(&rid_refs)?;

                    let cache = self.scoring_cache.borrow();
                    for rid in &filtered_rids {
                        let Some(row) = cache.get(rid) else { continue };
                        let Some(emb_blob_row) = embeddings.get(rid) else { continue };

                        let mem_embedding = crate::serde_helpers::deserialize_f32(emb_blob_row);
                        let sim_score = crate::consolidate::cosine_similarity(query_embedding, &mem_embedding) as f64;

                        let elapsed = ts - row.last_access;
                        let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                        let age = ts - row.created_at;
                        let recency = scoring::recency_score(age);

                        let prox = gi.graph_proximity(rid, &expanded_map);
                        let composite = scoring::graph_composite_score(
                            sim_score, decay, recency, row.importance, row.valence, prox,
                        );

                        let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
                        let mem_entities: Vec<String> = gi.entities_for_memory(rid).into_iter().map(|s| s.to_string()).collect();
                        for entity in &mem_entities {
                            if expanded_map.contains_key(entity) {
                                why.push(format!("graph-connected via {entity}"));
                                break;
                            }
                        }

                        scored.push(RecallResult {
                            rid: rid.clone(),
                            memory_type: row.memory_type.clone(),
                            text: String::new(),
                            created_at: row.created_at,
                            importance: row.importance,
                            valence: row.valence,
                            score: composite,
                            scores: ScoreBreakdown {
                                similarity: sim_score,
                                decay,
                                recency,
                                importance: row.importance,
                                graph_proximity: prox,
                            },
                            why_retrieved: why,
                            metadata: serde_json::Value::Null,
                            namespace: row.namespace.clone(),
                        });
                    }
                    drop(cache);
                }
            }
        }

        // Step 4: Sort and truncate to top_k
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        // Step 5: Hydrate final top_k with text + metadata from SQLite
        let final_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
        let text_meta = self.fetch_text_metadata_by_rids(&final_rids)?;
        for result in &mut scored {
            if let Some(tm) = text_meta.get(&result.rid) {
                result.text = tm.text.clone();
                result.metadata = serde_json::from_str(&tm.metadata)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
            }
        }

        // Reinforce accessed memories (spaced repetition)
        if !skip_reinforce {
            for r in &scored {
                self.reinforce(&r.rid)?;
            }
        }

        Ok(scored)
    }

    /// Profiled version of recall() that returns per-phase timing breakdown.
    #[cfg(feature = "profiling")]
    pub fn recall_profiled(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
    ) -> Result<RecallProfiledResult> {
        use std::time::Instant;
        let t_start = Instant::now();

        // ── Phase 1: Vector search (HNSW) ──
        let t_vec = Instant::now();
        let ts = now();
        let fetch_k = (top_k * 5).min(200);
        let vec_results = self.vec_index.borrow().search(query_embedding, fetch_k)?;
        let vec_search_ms = t_vec.elapsed().as_secs_f64() * 1000.0;
        let candidate_count = vec_results.len();

        if vec_results.is_empty() {
            return Ok(RecallProfiledResult {
                results: vec![],
                timings: RecallTimings {
                    vec_search_ms,
                    cache_score_ms: 0.0,
                    fetch_ms: 0.0,
                    scoring_ms: 0.0,
                    graph_ms: 0.0,
                    reinforce_ms: 0.0,
                    sort_truncate_ms: 0.0,
                    total_ms: t_start.elapsed().as_secs_f64() * 1000.0,
                    candidate_count: 0,
                    graph_expansion_count: 0,
                },
            });
        }

        // ── Phase 2: Score from in-memory cache ──
        let t_cache_score = Instant::now();
        let mut scored: Vec<RecallResult> = Vec::new();
        {
            let cache = self.scoring_cache.borrow();
            for (rid, distance) in &vec_results {
                let sim_score = 1.0 - distance;
                if let Some(row) = cache.get(rid.as_str()) {
                    // Filter by consolidation_status
                    if row.consolidation_status == "tombstoned" {
                        continue;
                    }
                    if !include_consolidated && row.consolidation_status == "consolidated" {
                        continue;
                    }
                    // Filter by memory_type
                    if let Some(mt) = memory_type {
                        if row.memory_type != mt {
                            continue;
                        }
                    }
                    // Filter by time_window
                    if let Some((start, end)) = time_window {
                        if row.created_at < start || row.created_at > end {
                            continue;
                        }
                    }

                    let elapsed = ts - row.last_access;
                    let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                    let age = ts - row.created_at;
                    let recency = scoring::recency_score(age);
                    let composite = scoring::composite_score(sim_score, decay, recency, row.importance, row.valence);
                    let why = scoring::build_why(sim_score, recency, decay, row.valence);

                    scored.push(RecallResult {
                        rid: rid.clone(),
                        memory_type: row.memory_type.clone(),
                        text: String::new(), // hydrated later
                        created_at: row.created_at,
                        importance: row.importance,
                        valence: row.valence,
                        score: composite,
                        scores: ScoreBreakdown {
                            similarity: sim_score,
                            decay,
                            recency,
                            importance: row.importance,
                            graph_proximity: 0.0,
                        },
                        why_retrieved: why,
                        metadata: serde_json::Value::Object(Default::default()),
                    });
                }
            }
        }
        let cache_score_ms = t_cache_score.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 3: Graph expansion ──
        let t_graph = Instant::now();
        let mut graph_expansion_count = 0usize;
        if expand_entities {
            let gi = self.graph_index.borrow();
            let query_entities: Vec<(String, String, u32)> = if let Some(qt) = query_text {
                let query_tokens = crate::graph::tokenize(qt);
                gi.entity_matches_query(&query_tokens)
            } else {
                vec![]
            };

            let (base_boost, seed_entities, entity_idfs) = if !query_entities.is_empty() {
                let has_person = query_entities.iter().any(|(_, etype, _)| etype == "person");
                let factor = if has_person { 0.20 } else if query_entities.len() >= 2 { 0.15 } else { 0.12 };
                let idfs: std::collections::HashMap<String, f64> = query_entities
                    .iter()
                    .map(|(name, _, mc)| {
                        let idf = 1.0 / (1.0 + (*mc as f64).max(1.0).ln());
                        (name.to_lowercase(), idf)
                    })
                    .collect();
                let names: Vec<String> = query_entities.into_iter().map(|(n, _, _)| n).collect();
                (factor, names, idfs)
            } else if query_text.is_none() {
                let mut seed_sorted = scored.clone();
                seed_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                let seed_count = 3.min(seed_sorted.len());
                let seed_rids: Vec<&str> = seed_sorted[..seed_count].iter().map(|r| r.rid.as_str()).collect();
                let seeds = gi.entities_for_memories(&seed_rids);
                (0.05, seeds, std::collections::HashMap::new())
            } else {
                (0.0, vec![], std::collections::HashMap::new())
            };

            const MAX_BOOST_PER_MEMORY: f64 = 0.25;
            const MAX_GRAPH_FRACTION: f64 = 0.40;

            if !seed_entities.is_empty() && base_boost > 0.0 {
                let seed_refs: Vec<&str> = seed_entities.iter().map(|s| s.as_str()).collect();
                let expanded = gi.expand_bfs(&seed_refs, 1, 20);
                let expanded_map: std::collections::HashMap<String, (u8, f64)> = expanded
                    .iter()
                    .map(|(name, hops, weight)| (name.clone(), (*hops, *weight)))
                    .collect();

                // Phase (a): Boost existing scored results
                for result in &mut scored {
                    let prox = gi.graph_proximity(&result.rid, &expanded_map);
                    if prox > 0.0 {
                        let mem_entities: Vec<String> = gi.entities_for_memory(&result.rid).into_iter().map(|s| s.to_string()).collect();
                        let mut best_idf = 1.0f64;
                        let mut connecting_entity = String::new();
                        for entity in &mem_entities {
                            if expanded_map.contains_key(entity) {
                                let idf = entity_idfs.get(&entity.to_lowercase()).copied().unwrap_or(1.0);
                                if connecting_entity.is_empty() || idf > best_idf {
                                    best_idf = idf;
                                    connecting_entity = entity.clone();
                                }
                            }
                        }
                        let consolidation_factor = {
                            let cache = self.scoring_cache.borrow();
                            if let Some(row) = cache.get(&result.rid) {
                                if row.consolidation_status == "consolidated" { 0.5 } else { 1.0 }
                            } else { 1.0 }
                        };
                        let boost = (base_boost * prox * best_idf * consolidation_factor).min(MAX_BOOST_PER_MEMORY);
                        result.scores.graph_proximity = prox;
                        result.score += boost;
                        if !connecting_entity.is_empty() {
                            result.why_retrieved.push(format!("graph-connected via {connecting_entity}"));
                        }
                    }
                }

                // Phase (b): Graph-only candidates
                let max_graph_only = ((MAX_GRAPH_FRACTION * top_k as f64).ceil() as usize).max(1);
                let all_entity_names: Vec<&str> = expanded.iter().map(|(n, _, _)| n.as_str()).collect();
                let graph_rids = gi.memories_for_entities(&all_entity_names);
                let existing_rids: std::collections::HashSet<&str> = scored.iter().map(|r| r.rid.as_str()).collect();

                // Filter from cache, collect eligible rids
                let new_rids: Vec<String> = {
                    let cache = self.scoring_cache.borrow();
                    graph_rids
                        .into_iter()
                        .filter(|r| {
                            if existing_rids.contains(r.as_str()) { return false; }
                            if let Some(row) = cache.get(r.as_str()) {
                                if row.consolidation_status == "tombstoned" { return false; }
                                if !include_consolidated && row.consolidation_status == "consolidated" { return false; }
                                if let Some(mt) = memory_type {
                                    if row.memory_type != mt { return false; }
                                }
                                if let Some((start, end)) = time_window {
                                    if row.created_at < start || row.created_at > end { return false; }
                                }
                                true
                            } else { false }
                        })
                        .take(max_graph_only)
                        .collect()
                };
                graph_expansion_count = new_rids.len();

                if !new_rids.is_empty() {
                    // Batch-fetch embeddings for cosine similarity
                    let new_rid_refs: Vec<&str> = new_rids.iter().map(|r| r.as_str()).collect();
                    let emb_map = self.fetch_embeddings_by_rids(&new_rid_refs)?;

                    let cache = self.scoring_cache.borrow();
                    for rid in &new_rids {
                        if let (Some(row), Some(emb_blob)) = (cache.get(rid.as_str()), emb_map.get(rid.as_str())) {
                            let mem_embedding = crate::serde_helpers::deserialize_f32(emb_blob);
                            let sim_score = crate::consolidate::cosine_similarity(query_embedding, &mem_embedding) as f64;
                            let elapsed = ts - row.last_access;
                            let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                            let age = ts - row.created_at;
                            let recency = scoring::recency_score(age);
                            let prox = gi.graph_proximity(rid, &expanded_map);
                            let composite = scoring::graph_composite_score(sim_score, decay, recency, row.importance, row.valence, prox);
                            let mut why = scoring::build_why(sim_score, recency, decay, row.valence);

                            let mem_entities: Vec<String> = gi.entities_for_memory(rid).into_iter().map(|s| s.to_string()).collect();
                            for entity in &mem_entities {
                                if expanded_map.contains_key(entity) {
                                    why.push(format!("graph-connected via {entity}"));
                                    break;
                                }
                            }

                            scored.push(RecallResult {
                                rid: rid.clone(),
                                memory_type: row.memory_type.clone(),
                                text: String::new(),
                                created_at: row.created_at,
                                importance: row.importance,
                                valence: row.valence,
                                score: composite,
                                scores: ScoreBreakdown {
                                    similarity: sim_score,
                                    decay,
                                    recency,
                                    importance: row.importance,
                                    graph_proximity: prox,
                                },
                                why_retrieved: why,
                                metadata: serde_json::Value::Object(Default::default()),
                            });
                        }
                    }
                }
            }
        }
        let graph_ms = t_graph.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 4: Sort + truncate ──
        let t_sort = Instant::now();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        let sort_truncate_ms = t_sort.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 5: Hydrate final top_k with text + metadata ──
        let t_fetch = Instant::now();
        {
            let final_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
            let text_map = self.fetch_text_metadata_by_rids(&final_rids)?;
            for result in &mut scored {
                if let Some(tm) = text_map.get(result.rid.as_str()) {
                    result.text = tm.text.clone();
                    result.metadata = serde_json::from_str(&tm.metadata)
                        .unwrap_or(serde_json::Value::Object(Default::default()));
                }
            }
        }
        let fetch_ms = t_fetch.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 6: Reinforce ──
        let t_reinforce = Instant::now();
        if !skip_reinforce {
            for r in &scored {
                self.reinforce(&r.rid)?;
            }
        }
        let reinforce_ms = t_reinforce.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        Ok(RecallProfiledResult {
            results: scored,
            timings: RecallTimings {
                vec_search_ms,
                cache_score_ms,
                fetch_ms,
                scoring_ms: 0.0, // Kept for backward compat — scoring now in cache_score_ms
                graph_ms,
                reinforce_ms,
                sort_truncate_ms,
                total_ms,
                candidate_count,
                graph_expansion_count,
            },
        })
    }

    /// Fetch only text and metadata for a set of RIDs (post-scoring hydration).
    pub(crate) fn fetch_text_metadata_by_rids(
        &self,
        rids: &[&str],
    ) -> Result<HashMap<String, TextMetadataRow>> {
        if rids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders: String = (0..rids.len())
            .map(|i| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT rid, type, text, metadata FROM memories WHERE rid IN ({placeholders})"
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for r in rids {
            param_values.push(Box::new(r.to_string()));
        }
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok(TextMetadataRow {
                    rid: row.get("rid")?,
                    text: row.get("text")?,
                    metadata: row.get("metadata")?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut map = HashMap::new();
        for row in rows {
            map.insert(row.rid.clone(), row);
        }
        Ok(map)
    }

    /// Batch-fetch embeddings for a set of RIDs (for graph-only candidate scoring).
    pub(crate) fn fetch_embeddings_by_rids(
        &self,
        rids: &[&str],
    ) -> Result<HashMap<String, Vec<u8>>> {
        if rids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders: String = (0..rids.len())
            .map(|i| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT rid, embedding FROM memories WHERE rid IN ({placeholders})"
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for r in rids {
            param_values.push(Box::new(r.to_string()));
        }
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut map = HashMap::new();
        for (rid, emb) in rows {
            map.insert(rid, emb);
        }
        Ok(map)
    }

    /// Reinforce a memory on access — increase half_life and update last_access.
    fn reinforce(&self, rid: &str) -> Result<()> {
        let ts = now();

        // Read half_life from cache (eliminates SELECT query)
        let current_half_life = {
            let cache = self.scoring_cache.borrow();
            cache.get(rid).map(|r| r.half_life)
        };
        let new_half_life = match current_half_life {
            Some(hl) => (hl * 1.2_f64).min(31536000.0),
            None => 604800.0, // fallback if not in cache
        };

        self.conn.execute(
            "UPDATE memories SET last_access = ?1, half_life = ?2 WHERE rid = ?3",
            params![ts, new_half_life, rid],
        )?;

        // Update cache with new values
        {
            let mut cache = self.scoring_cache.borrow_mut();
            if let Some(row) = cache.get_mut(rid) {
                row.last_access = ts;
                row.half_life = new_half_life;
            }
        }

        self.log_op(
            "reinforce",
            Some(rid),
            &serde_json::json!({
                "rid": rid,
                "last_access": ts,
                "half_life": new_half_life,
                "local_only": true,
            }),
            None,
        )?;

        Ok(())
    }
}
