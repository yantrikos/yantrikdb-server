use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::Rng;
use rusqlite::{params, Connection};

use crate::error::{AidbError, Result};
use crate::graph_index::GraphIndex;
use crate::hlc::{HLCTimestamp, HLC};
use crate::hnsw::HnswIndex;
use crate::schema::{MIGRATE_V1_TO_V2, MIGRATE_V2_TO_V3, MIGRATE_V3_TO_V4, MIGRATE_V4_TO_V5, MIGRATE_V5_TO_V6, MIGRATE_V6_TO_V7, SCHEMA_SQL, SCHEMA_VERSION};
use crate::scoring;
use crate::serde_helpers::{deserialize_f32, serialize_f32};
use crate::types::*;

/// The AIDB cognitive memory engine.
pub struct AIDB {
    conn: Connection,
    embedding_dim: usize,
    hlc: RefCell<HLC>,
    actor_id: String,
    scoring_cache: RefCell<HashMap<String, ScoringRow>>,
    pub(crate) vec_index: RefCell<HnswIndex>,
    pub(crate) graph_index: RefCell<GraphIndex>,
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Compute BLAKE3 hash of an embedding blob.
fn embedding_hash(embedding: &[f32]) -> Vec<u8> {
    let blob = serialize_f32(embedding);
    blake3::hash(&blob).as_bytes().to_vec()
}

impl AIDB {
    /// Create a new AIDB instance with auto-generated actor_id.
    pub fn new(db_path: &str, embedding_dim: usize) -> Result<Self> {
        Self::open(db_path, embedding_dim, None)
    }

    /// Create a new AIDB instance with an explicit actor_id (for sync tests).
    pub fn new_with_actor(db_path: &str, embedding_dim: usize, actor_id: &str) -> Result<Self> {
        Self::open(db_path, embedding_dim, Some(actor_id.to_string()))
    }

    fn open(db_path: &str, embedding_dim: usize, actor_id: Option<String>) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        // Check existing schema version for migration
        let existing_version = Self::get_schema_version(&conn);

        if existing_version == Some(1) {
            conn.execute_batch(MIGRATE_V1_TO_V2)?;
            conn.execute_batch(MIGRATE_V2_TO_V3)?;
            conn.execute_batch(MIGRATE_V3_TO_V4)?;
            conn.execute_batch(MIGRATE_V4_TO_V5)?;
            conn.execute_batch(MIGRATE_V5_TO_V6)?;
            conn.execute_batch(MIGRATE_V6_TO_V7)?;
        } else if existing_version == Some(2) {
            conn.execute_batch(MIGRATE_V2_TO_V3)?;
            conn.execute_batch(MIGRATE_V3_TO_V4)?;
            conn.execute_batch(MIGRATE_V4_TO_V5)?;
            conn.execute_batch(MIGRATE_V5_TO_V6)?;
            conn.execute_batch(MIGRATE_V6_TO_V7)?;
        } else if existing_version == Some(3) {
            conn.execute_batch(MIGRATE_V3_TO_V4)?;
            conn.execute_batch(MIGRATE_V4_TO_V5)?;
            conn.execute_batch(MIGRATE_V5_TO_V6)?;
            conn.execute_batch(MIGRATE_V6_TO_V7)?;
        } else if existing_version == Some(4) {
            conn.execute_batch(MIGRATE_V4_TO_V5)?;
            conn.execute_batch(MIGRATE_V5_TO_V6)?;
            conn.execute_batch(MIGRATE_V6_TO_V7)?;
        } else if existing_version == Some(5) {
            conn.execute_batch(MIGRATE_V5_TO_V6)?;
            conn.execute_batch(MIGRATE_V6_TO_V7)?;
        } else if existing_version == Some(6) {
            conn.execute_batch(MIGRATE_V6_TO_V7)?;
        }

        conn.execute_batch(SCHEMA_SQL)?;

        // Set schema version
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?1)",
            params![SCHEMA_VERSION.to_string()],
        )?;

        // Resolve actor_id: explicit > stored in meta > generate new
        let actor_id = if let Some(id) = actor_id {
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('actor_id', ?1)",
                params![id],
            )?;
            id
        } else {
            match Self::get_meta(&conn, "actor_id")? {
                Some(id) => id,
                None => {
                    let id = uuid7::uuid7().to_string();
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES ('actor_id', ?1)",
                        params![id],
                    )?;
                    id
                }
            }
        };

        // Resolve node_id: stored in meta > generate random
        let node_id: u32 = match Self::get_meta(&conn, "node_id")? {
            Some(s) => s.parse().unwrap_or_else(|_| {
                let id: u32 = rand::thread_rng().gen();
                id
            }),
            None => {
                let id: u32 = rand::thread_rng().gen();
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('node_id', ?1)",
                    params![id.to_string()],
                )?;
                id
            }
        };

        let scoring_cache = Self::load_scoring_cache(&conn)?;
        let vec_index = Self::build_vec_index(&conn, embedding_dim)?;
        let graph_index = GraphIndex::build_from_db(&conn)?;

        Ok(Self {
            conn,
            embedding_dim,
            hlc: RefCell::new(HLC::new(node_id)),
            actor_id,
            scoring_cache: RefCell::new(scoring_cache),
            vec_index: RefCell::new(vec_index),
            graph_index: RefCell::new(graph_index),
        })
    }

    /// Load scoring-relevant fields for all non-tombstoned memories into a HashMap.
    fn load_scoring_cache(conn: &Connection) -> Result<HashMap<String, ScoringRow>> {
        let mut stmt = conn.prepare(
            "SELECT rid, created_at, importance, half_life, last_access, \
             valence, consolidation_status, type \
             FROM memories \
             WHERE consolidation_status != 'tombstoned'",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                ScoringRow {
                    created_at: row.get(1)?,
                    importance: row.get(2)?,
                    half_life: row.get(3)?,
                    last_access: row.get(4)?,
                    valence: row.get(5)?,
                    consolidation_status: row.get(6)?,
                    memory_type: row.get(7)?,
                },
            ))
        })?;

        let mut cache = HashMap::new();
        for row in rows {
            let (rid, scoring_row) = row?;
            cache.insert(rid, scoring_row);
        }
        Ok(cache)
    }

    /// Build the HNSW vector index from active hot-tier embeddings in SQLite.
    fn build_vec_index(conn: &Connection, embedding_dim: usize) -> Result<HnswIndex> {
        let mut index = HnswIndex::new(embedding_dim);
        let mut stmt = conn.prepare(
            "SELECT rid, embedding FROM memories \
             WHERE consolidation_status IN ('active', 'consolidated') \
             AND storage_tier = 'hot' \
             AND embedding IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |row| {
            let rid: String = row.get(0)?;
            let emb_blob: Vec<u8> = row.get(1)?;
            Ok((rid, emb_blob))
        })?;
        for row in rows {
            let (rid, emb_blob) = row?;
            let embedding = deserialize_f32(&emb_blob);
            if embedding.len() == embedding_dim {
                index.insert(&rid, &embedding)?;
            }
        }
        Ok(index)
    }

    /// Rebuild the HNSW vector index from scratch. Called after replication.
    pub fn rebuild_vec_index(&self) -> Result<usize> {
        let new_index = Self::build_vec_index(&self.conn, self.embedding_dim)?;
        let count = new_index.len();
        *self.vec_index.borrow_mut() = new_index;
        Ok(count)
    }

    pub fn rebuild_graph_index(&self) -> Result<usize> {
        let new_index = crate::graph_index::GraphIndex::build_from_db(&self.conn)?;
        let count = new_index.entity_count();
        *self.graph_index.borrow_mut() = new_index;
        Ok(count)
    }

    fn get_schema_version(conn: &Connection) -> Option<i32> {
        // meta table might not exist yet
        conn.query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| {
                let v: String = row.get(0)?;
                Ok(v.parse::<i32>().unwrap_or(0))
            },
        )
        .ok()
    }

    fn get_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
        match conn.query_row(
            "SELECT value FROM meta WHERE key = ?1",
            params![key],
            |row| row.get(0),
        ) {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get a new HLC timestamp (ticks the clock forward).
    pub fn tick_hlc(&self) -> HLCTimestamp {
        self.hlc.borrow_mut().now()
    }

    /// Merge a remote HLC timestamp into the local clock.
    pub fn merge_hlc(&self, remote: HLCTimestamp) -> HLCTimestamp {
        self.hlc.borrow_mut().recv(remote)
    }

    /// Get the actor_id of this instance.
    pub fn actor_id(&self) -> &str {
        &self.actor_id
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get a reference to the underlying connection (for test compatibility).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Get a mutable reference to the underlying connection.
    pub fn conn_mut(&mut self) -> &mut Connection {
        &mut self.conn
    }

    // ── Scoring cache helpers ──

    /// Insert a scoring row into the in-memory cache.
    pub fn cache_insert(&self, rid: String, row: ScoringRow) {
        self.scoring_cache.borrow_mut().insert(rid, row);
    }

    /// Remove a scoring row from the in-memory cache.
    pub fn cache_remove(&self, rid: &str) {
        self.scoring_cache.borrow_mut().remove(rid);
    }

    /// Mark a memory as consolidated in the cache and reduce its importance.
    pub fn cache_mark_consolidated(&self, rid: &str, importance_factor: f64) {
        let mut cache = self.scoring_cache.borrow_mut();
        if let Some(row) = cache.get_mut(rid) {
            row.consolidation_status = "consolidated".to_string();
            row.importance *= importance_factor;
        }
    }

    // ── record() — store a memory ──

    /// Store a new memory and return its RID.
    pub fn record(
        &self,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        embedding: &[f32],
    ) -> Result<String> {
        let rid = uuid7::uuid7().to_string();
        let ts = now();
        let emb_blob = serialize_f32(embedding);
        let meta_str = serde_json::to_string(metadata)?;

        self.conn.execute(
            "INSERT INTO memories \
             (rid, type, text, embedding, created_at, updated_at, importance, \
              half_life, last_access, valence, metadata) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![rid, memory_type, text, emb_blob, ts, ts, importance, half_life, ts, valence, meta_str],
        )?;

        // Insert into vector index
        self.vec_index.borrow_mut().insert(&rid, embedding)?;

        // Insert into scoring cache
        self.cache_insert(rid.clone(), ScoringRow {
            created_at: ts,
            importance,
            half_life,
            last_access: ts,
            valence,
            consolidation_status: "active".to_string(),
            memory_type: memory_type.to_string(),
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
            }),
            Some(&emb_hash),
        )?;

        Ok(rid)
    }

    // ── recall() — multi-signal retrieval with optional graph expansion ──

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
    /// Only available when the `profiling` feature is enabled.
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
    fn fetch_text_metadata_by_rids(
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
    fn fetch_embeddings_by_rids(
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

    // ── relate() — create entity links ──

    /// Create or update a relationship between entities.
    pub fn relate(
        &self,
        src: &str,
        dst: &str,
        rel_type: &str,
        weight: f64,
    ) -> Result<String> {
        let edge_id = uuid7::uuid7().to_string();
        let ts = now();

        self.conn.execute(
            "INSERT INTO edges (edge_id, src, dst, rel_type, weight, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(src, dst, rel_type) DO UPDATE SET weight = ?5, created_at = ?6",
            params![edge_id, src, dst, rel_type, weight, ts],
        )?;

        // Ensure entities exist with classified entity_type
        for entity in [src, dst] {
            let etype = crate::graph::classify_entity_type(entity);
            self.conn.execute(
                "INSERT INTO entities (name, entity_type, first_seen, last_seen) \
                 VALUES (?1, ?2, ?3, ?4) \
                 ON CONFLICT(name) DO UPDATE SET last_seen = ?4, mention_count = mention_count + 1, \
                 entity_type = CASE WHEN entities.entity_type = 'unknown' THEN ?2 ELSE entities.entity_type END",
                params![entity, etype, ts, ts],
            )?;
        }

        // Update in-memory graph index
        {
            let mut gi = self.graph_index.borrow_mut();
            let src_type = crate::graph::classify_entity_type(src);
            let dst_type = crate::graph::classify_entity_type(dst);
            gi.add_entity(src, src_type);
            gi.add_entity(dst, dst_type);
            gi.add_edge(src, dst, weight as f32);
        }

        self.log_op(
            "relate",
            Some(&edge_id),
            &serde_json::json!({
                "edge_id": edge_id,
                "src": src,
                "dst": dst,
                "rel_type": rel_type,
                "weight": weight,
                "created_at": ts,
            }),
            None,
        )?;

        Ok(edge_id)
    }

    // ── decay() — compute current importance scores ──

    /// Find memories that have decayed below a threshold.
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
            let (rid, text, importance, half_life, last_access, mem_type) = row?;
            let elapsed = ts - last_access;
            let score = scoring::decay_score(importance, half_life, elapsed);
            if score < threshold {
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

    // ── forget() — tombstone a memory ──

    /// Tombstone a memory. Returns true if the memory was found and tombstoned.
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

    // ── Utility methods ──

    /// Get a single memory by RID.
    pub fn get(&self, rid: &str) -> Result<Option<Memory>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM memories WHERE rid = ?1",
        )?;

        let result = stmt.query_row(params![rid], |row| {
            let meta_str: String = row.get("metadata")?;
            let metadata: serde_json::Value =
                serde_json::from_str(&meta_str).unwrap_or(serde_json::Value::Object(Default::default()));

            Ok(Memory {
                rid: row.get("rid")?,
                memory_type: row.get("type")?,
                text: row.get("text")?,
                created_at: row.get("created_at")?,
                importance: row.get("importance")?,
                valence: row.get("valence")?,
                half_life: row.get("half_life")?,
                last_access: row.get("last_access")?,
                consolidation_status: row.get("consolidation_status")?,
                storage_tier: row.get("storage_tier")?,
                consolidated_into: row.get("consolidated_into")?,
                metadata,
            })
        });

        match result {
            Ok(mem) => Ok(Some(mem)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get all edges connected to an entity.
    pub fn get_edges(&self, entity: &str) -> Result<Vec<Edge>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM edges WHERE (src = ?1 OR dst = ?1) AND tombstoned = 0",
        )?;

        let edges = stmt
            .query_map(params![entity], |row| {
                Ok(Edge {
                    edge_id: row.get("edge_id")?,
                    src: row.get("src")?,
                    dst: row.get("dst")?,
                    rel_type: row.get("rel_type")?,
                    weight: row.get("weight")?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(edges)
    }

    // ── Memory-Entity Linkage ──

    /// Link a memory to an entity for graph-augmented recall.
    pub fn link_memory_entity(&self, memory_rid: &str, entity_name: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
            params![memory_rid, entity_name],
        )?;
        self.graph_index.borrow_mut().link_memory(memory_rid, entity_name);
        Ok(())
    }

    /// Backfill the memory_entities table by scanning memory text for known entity names.
    /// Uses word-boundary matching to avoid false positives.
    /// Returns the number of links created. Idempotent (uses INSERT OR IGNORE).
    pub fn backfill_memory_entities(&self) -> Result<usize> {
        let entities: Vec<String> = self.conn.prepare(
            "SELECT name FROM entities",
        )?.query_map([], |row| row.get(0))?.collect::<std::result::Result<Vec<_>, _>>()?;

        if entities.is_empty() {
            return Ok(0);
        }

        let memories: Vec<(String, String)> = self.conn.prepare(
            "SELECT rid, text FROM memories WHERE consolidation_status = 'active'",
        )?.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?.collect::<std::result::Result<Vec<_>, _>>()?;

        let mut count = 0usize;
        let mut gi = self.graph_index.borrow_mut();
        for (rid, text) in &memories {
            let text_tokens = crate::graph::tokenize(text);
            for entity in &entities {
                if crate::graph::entity_matches_text(entity, &text_tokens) {
                    self.conn.execute(
                        "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                        params![rid, entity],
                    )?;
                    gi.link_memory(rid, entity);
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    // ── Storage tier operations ──

    /// Archive a hot memory to cold storage (compress embedding, remove from vec index).
    /// Returns true if the memory was archived, false if not found or already cold.
    pub fn archive(&self, rid: &str) -> Result<bool> {
        let row = self.conn.query_row(
            "SELECT embedding FROM memories WHERE rid = ?1 AND storage_tier = 'hot' AND consolidation_status = 'active'",
            params![rid],
            |row| row.get::<_, Vec<u8>>(0),
        );

        let raw_blob = match row {
            Ok(blob) => blob,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(false),
            Err(e) => return Err(e.into()),
        };

        let embedding = crate::serde_helpers::deserialize_f32(&raw_blob);
        let compressed = crate::compression::compress_embedding(&embedding);
        let ts = now();

        self.conn.execute(
            "UPDATE memories SET storage_tier = 'cold', embedding = ?1, updated_at = ?2 WHERE rid = ?3",
            params![compressed, ts, rid],
        )?;

        self.vec_index.borrow_mut().remove(rid);

        self.log_op(
            "archive",
            Some(rid),
            &serde_json::json!({
                "rid": rid,
                "updated_at": ts,
            }),
            None,
        )?;

        Ok(true)
    }

    /// Hydrate a cold memory back to hot storage (decompress embedding, re-insert into vec index).
    /// Returns true if the memory was hydrated, false if not found or already hot.
    pub fn hydrate(&self, rid: &str) -> Result<bool> {
        let row = self.conn.query_row(
            "SELECT embedding FROM memories WHERE rid = ?1 AND storage_tier = 'cold'",
            params![rid],
            |row| row.get::<_, Vec<u8>>(0),
        );

        let compressed_blob = match row {
            Ok(blob) => blob,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(false),
            Err(e) => return Err(e.into()),
        };

        let embedding = crate::compression::decompress_embedding(&compressed_blob);
        let raw_blob = serialize_f32(&embedding);
        let ts = now();

        self.conn.execute(
            "UPDATE memories SET storage_tier = 'hot', embedding = ?1, updated_at = ?2 WHERE rid = ?3",
            params![raw_blob, ts, rid],
        )?;

        self.vec_index.borrow_mut().insert(rid, &embedding)?;

        self.log_op(
            "hydrate",
            Some(rid),
            &serde_json::json!({
                "rid": rid,
                "updated_at": ts,
            }),
            None,
        )?;

        Ok(true)
    }

    // ── Batch operations ──

    /// Record multiple memories in a single transaction.
    /// Uses SAVEPOINT for atomicity while keeping `&self` (no `&mut self`).
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

            let result = self.conn.execute(
                "INSERT INTO memories \
                 (rid, type, text, embedding, created_at, updated_at, importance, \
                  half_life, last_access, valence, metadata) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![rid, input.memory_type, input.text, emb_blob, ts, ts,
                        input.importance, input.half_life, ts, input.valence, meta_str],
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
                    valence: input.valence,
                    consolidation_status: "active".to_string(),
                    memory_type: input.memory_type.clone(),
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

    // ── Eviction ──

    /// Evict memories to cold storage based on decay scores.
    /// Archives the lowest-scoring memories until at most `max_active` hot memories remain.
    /// Returns the list of archived RIDs.
    pub fn evict(&self, max_active: usize) -> Result<Vec<String>> {
        let hot_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'active' AND storage_tier = 'hot'",
            [],
            |row| row.get(0),
        )?;

        if hot_count as usize <= max_active {
            return Ok(vec![]);
        }

        let to_evict = hot_count as usize - max_active;
        let ts = now();

        let mut stmt = self.conn.prepare(
            "SELECT rid, importance, half_life, last_access, created_at FROM memories \
             WHERE consolidation_status = 'active' AND storage_tier = 'hot'",
        )?;

        let mut scored: Vec<(String, f64)> = stmt
            .query_map([], |row| {
                let rid: String = row.get("rid")?;
                let importance: f64 = row.get("importance")?;
                let half_life: f64 = row.get("half_life")?;
                let last_access: f64 = row.get("last_access")?;
                let created_at: f64 = row.get("created_at")?;
                let elapsed = ts - last_access;
                let decay = crate::scoring::decay_score(importance, half_life, elapsed);
                let age = ts - created_at;
                let recency = crate::scoring::recency_score(age);
                let score = crate::scoring::eviction_score(decay, recency);
                Ok((rid, score))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Sort ascending — lowest score = most evictable
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut archived_rids = Vec::new();
        for (rid, _) in scored.into_iter().take(to_evict) {
            if self.archive(&rid)? {
                archived_rids.push(rid);
            }
        }

        Ok(archived_rids)
    }

    /// Get engine statistics.
    pub fn stats(&self) -> Result<Stats> {
        let active = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'active'",
            [], |row| row.get(0),
        )?;
        let consolidated = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'consolidated'",
            [], |row| row.get(0),
        )?;
        let tombstoned = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'tombstoned'",
            [], |row| row.get(0),
        )?;
        let archived = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE storage_tier = 'cold'",
            [], |row| row.get(0),
        )?;
        let edges = self.conn.query_row(
            "SELECT COUNT(*) FROM edges WHERE tombstoned = 0",
            [], |row| row.get(0),
        )?;
        let entities = self.conn.query_row(
            "SELECT COUNT(*) FROM entities",
            [], |row| row.get(0),
        )?;
        let operations = self.conn.query_row(
            "SELECT COUNT(*) FROM oplog",
            [], |row| row.get(0),
        )?;
        let open_conflicts = self.conn.query_row(
            "SELECT COUNT(*) FROM conflicts WHERE status = 'open'",
            [], |row| row.get(0),
        )?;
        let resolved_conflicts = self.conn.query_row(
            "SELECT COUNT(*) FROM conflicts WHERE status IN ('resolved', 'dismissed')",
            [], |row| row.get(0),
        )?;
        let pending_triggers = self.conn.query_row(
            "SELECT COUNT(*) FROM trigger_log WHERE status = 'pending'",
            [], |row| row.get(0),
        )?;
        let active_patterns = self.conn.query_row(
            "SELECT COUNT(*) FROM patterns WHERE status = 'active'",
            [], |row| row.get(0),
        )?;

        Ok(Stats {
            active_memories: active,
            consolidated_memories: consolidated,
            tombstoned_memories: tombstoned,
            archived_memories: archived,
            edges,
            entities,
            operations,
            open_conflicts,
            resolved_conflicts,
            pending_triggers,
            active_patterns,
            scoring_cache_entries: self.scoring_cache.borrow().len(),
            vec_index_entries: self.vec_index.borrow().len(),
            graph_index_entities: self.graph_index.borrow().entity_count(),
            graph_index_edges: self.graph_index.borrow().edge_count(),
        })
    }

    /// Append an operation to the oplog with HLC and optional embedding hash.
    pub fn log_op(
        &self,
        op_type: &str,
        target_rid: Option<&str>,
        payload: &serde_json::Value,
        emb_hash: Option<&[u8]>,
    ) -> Result<String> {
        let op_id = uuid7::uuid7().to_string();
        let hlc_ts = self.tick_hlc();
        let hlc_bytes = hlc_ts.to_bytes().to_vec();
        let payload_str = serde_json::to_string(payload)?;

        self.conn.execute(
            "INSERT INTO oplog (op_id, op_type, timestamp, target_rid, payload, \
             actor_id, hlc, embedding_hash, origin_actor, applied) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1)",
            params![
                op_id,
                op_type,
                now(),
                target_rid,
                payload_str,
                self.actor_id,
                hlc_bytes,
                emb_hash,
                self.actor_id,
            ],
        )?;
        Ok(op_id)
    }

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
                let rid = self.record(
                    text,
                    "semantic",
                    merged_importance,
                    0.0,
                    604800.0,
                    &meta,
                    &zero_emb,
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

    /// User-initiated memory correction.
    ///
    /// Creates a new corrected memory and tombstones the original.
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
            let stats = self.stats()?;
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

    /// Close the database connection. After this, the engine cannot be used.
    pub fn close(self) -> Result<()> {
        self.conn.close().map_err(|(_, e)| AidbError::Database(e))
    }
}

/// Lightweight struct for fetching only text and metadata (post-scoring hydration).
struct TextMetadataRow {
    rid: String,
    text: String,
    metadata: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    fn empty_meta() -> serde_json::Value {
        serde_json::json!({})
    }

    #[test]
    fn test_new_and_stats() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let s = db.stats().unwrap();
        assert_eq!(s.active_memories, 0);
        assert_eq!(s.edges, 0);
    }

    #[test]
    fn test_actor_id_auto_generated() {
        let db = AIDB::new(":memory:", 8).unwrap();
        assert_eq!(db.actor_id().len(), 36); // UUIDv7
    }

    #[test]
    fn test_actor_id_explicit() {
        let db = AIDB::new_with_actor(":memory:", 8, "device-A").unwrap();
        assert_eq!(db.actor_id(), "device-A");
    }

    #[test]
    fn test_record_and_get() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let emb = vec_seed(1.0, 8);
        let rid = db.record("hello world", "episodic", 0.8, 0.0, 604800.0, &empty_meta(), &emb).unwrap();
        assert_eq!(rid.len(), 36);

        let mem = db.get(&rid).unwrap().unwrap();
        assert_eq!(mem.text, "hello world");
        assert_eq!(mem.memory_type, "episodic");
        assert_eq!(mem.importance, 0.8);
        assert_eq!(mem.consolidation_status, "active");
    }

    #[test]
    fn test_record_updates_stats() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.record("one", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        db.record("two", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8)).unwrap();
        assert_eq!(db.stats().unwrap().active_memories, 2);
    }

    #[test]
    fn test_recall_basic() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.record("the cat sat on the mat", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        db.record("dogs are loyal friends", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(5.0, 8)).unwrap();
        db.record("cats love warm places", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.1, 8)).unwrap();

        let results = db.recall(&vec_seed(1.0, 8), 2, None, None, false, false, None, false).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_recall_empty() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let results = db.recall(&vec_seed(1.0, 8), 5, None, None, false, false, None, false).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_relate_and_get_edges() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let eid = db.relate("Alice", "Bob", "knows", 1.0).unwrap();
        assert_eq!(eid.len(), 36);

        let edges = db.get_edges("Alice").unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].src, "Alice");
        assert_eq!(edges[0].dst, "Bob");
    }

    #[test]
    fn test_forget() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid = db.record("forget me", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        assert!(db.forget(&rid).unwrap());
        let mem = db.get(&rid).unwrap().unwrap();
        assert_eq!(mem.consolidation_status, "tombstoned");
    }

    #[test]
    fn test_forget_nonexistent() {
        let db = AIDB::new(":memory:", 8).unwrap();
        assert!(!db.forget("nonexistent").unwrap());
    }

    #[test]
    fn test_decay_fresh() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.record("fresh", "episodic", 0.9, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        let decayed = db.decay(0.01).unwrap();
        assert!(decayed.is_empty());
    }

    #[test]
    fn test_oplog_has_hlc() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.record("test", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();

        let hlc_bytes: Vec<u8> = db.conn.query_row(
            "SELECT hlc FROM oplog ORDER BY rowid DESC LIMIT 1",
            [],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(hlc_bytes.len(), 16);

        let ts = HLCTimestamp::from_bytes(&hlc_bytes).unwrap();
        assert!(ts.millis > 0);
    }

    #[test]
    fn test_oplog_has_embedding_hash() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.record("test", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();

        // The record op should have an embedding_hash
        let hash: Vec<u8> = db.conn.query_row(
            "SELECT embedding_hash FROM oplog WHERE op_type = 'record' LIMIT 1",
            [],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(hash.len(), 32); // BLAKE3 output is 32 bytes
    }

    #[test]
    fn test_oplog_enriched_payload() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.record("test payload", "semantic", 0.7, 0.3, 1000.0, &serde_json::json!({"key": "val"}), &vec_seed(1.0, 8)).unwrap();

        let payload_str: String = db.conn.query_row(
            "SELECT payload FROM oplog WHERE op_type = 'record' LIMIT 1",
            [],
            |row| row.get(0),
        ).unwrap();
        let payload: serde_json::Value = serde_json::from_str(&payload_str).unwrap();

        assert_eq!(payload["type"], "semantic");
        assert_eq!(payload["text"], "test payload");
        assert_eq!(payload["importance"], 0.7);
        assert_eq!(payload["valence"], 0.3);
        assert_eq!(payload["half_life"], 1000.0);
        assert!(payload["rid"].is_string());
        assert!(payload["created_at"].is_number());
        assert!(payload["metadata"]["key"] == "val");
    }

    #[test]
    fn test_schema_v3_has_conflicts_table() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let count: i64 = db.conn().query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='conflicts'",
            [],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_resolve_keep_a() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid_a = db.record("birthday March 5", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        let rid_b = db.record("birthday March 15", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8)).unwrap();

        let conflict = crate::conflict::create_conflict(
            &db, &crate::types::ConflictType::IdentityFact, &rid_a, &rid_b,
            Some("User"), Some("birthday"), "conflicting birthdays",
        ).unwrap();

        let result = db.resolve_conflict(&conflict.conflict_id, "keep_a", Some(&rid_a), None, Some("User confirmed March 5")).unwrap();
        assert!(result.loser_tombstoned);

        let mem_b = db.get(&rid_b).unwrap().unwrap();
        assert_eq!(mem_b.consolidation_status, "tombstoned");

        let resolved = db.get_conflict(&conflict.conflict_id).unwrap().unwrap();
        assert_eq!(resolved.status, "resolved");
        assert_eq!(resolved.strategy.as_deref(), Some("keep_a"));
    }

    #[test]
    fn test_resolve_keep_both() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid_a = db.record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        let rid_b = db.record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8)).unwrap();

        let conflict = crate::conflict::create_conflict(
            &db, &crate::types::ConflictType::Minor, &rid_a, &rid_b, None, None, "test",
        ).unwrap();
        let result = db.resolve_conflict(&conflict.conflict_id, "keep_both", None, None, None).unwrap();
        assert!(!result.loser_tombstoned);

        let mem_a = db.get(&rid_a).unwrap().unwrap();
        let mem_b = db.get(&rid_b).unwrap().unwrap();
        assert_eq!(mem_a.consolidation_status, "active");
        assert_eq!(mem_b.consolidation_status, "active");
    }

    #[test]
    fn test_correct_memory() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid = db.record("favorite color is green", "episodic", 0.7, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();

        let result = db.correct(
            &rid, "favorite color is blue", Some(0.9), None, &vec_seed(2.0, 8),
            Some("User corrected their favorite color"),
        ).unwrap();

        assert!(result.original_tombstoned);

        let original = db.get(&rid).unwrap().unwrap();
        assert_eq!(original.consolidation_status, "tombstoned");

        let corrected = db.get(&result.corrected_rid).unwrap().unwrap();
        assert_eq!(corrected.text, "favorite color is blue");
        assert_eq!(corrected.importance, 0.9);
    }

    #[test]
    fn test_get_conflicts_filtered() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid_a = db.record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        let rid_b = db.record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8)).unwrap();
        let rid_c = db.record("c", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(3.0, 8)).unwrap();

        crate::conflict::create_conflict(
            &db, &crate::types::ConflictType::IdentityFact, &rid_a, &rid_b,
            Some("User"), Some("birthday"), "test 1",
        ).unwrap();
        crate::conflict::create_conflict(
            &db, &crate::types::ConflictType::Preference, &rid_b, &rid_c,
            Some("User"), Some("prefers"), "test 2",
        ).unwrap();

        let all = db.get_conflicts(None, None, None, None, 50).unwrap();
        assert_eq!(all.len(), 2);

        let identity_only = db.get_conflicts(None, Some("identity_fact"), None, None, 50).unwrap();
        assert_eq!(identity_only.len(), 1);

        let critical = db.get_conflicts(None, None, None, Some("critical"), 50).unwrap();
        assert_eq!(critical.len(), 1);
    }

    #[test]
    fn test_dismiss_conflict() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid_a = db.record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        let rid_b = db.record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8)).unwrap();

        let conflict = crate::conflict::create_conflict(
            &db, &crate::types::ConflictType::Minor, &rid_a, &rid_b, None, None, "test",
        ).unwrap();

        db.dismiss_conflict(&conflict.conflict_id, Some("Not really a conflict")).unwrap();

        let c = db.get_conflict(&conflict.conflict_id).unwrap().unwrap();
        assert_eq!(c.status, "dismissed");
    }

    #[test]
    fn test_stats_include_conflicts() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let s = db.stats().unwrap();
        assert_eq!(s.open_conflicts, 0);
        assert_eq!(s.resolved_conflicts, 0);

        let rid_a = db.record("a", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        let rid_b = db.record("b", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8)).unwrap();
        crate::conflict::create_conflict(
            &db, &crate::types::ConflictType::Minor, &rid_a, &rid_b, None, None, "test",
        ).unwrap();

        let s = db.stats().unwrap();
        assert_eq!(s.open_conflicts, 1);
        assert_eq!(s.resolved_conflicts, 0);
    }

    // ── V3 Cognition tests ──

    #[test]
    fn test_schema_v4_has_trigger_log_and_patterns() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let count: i64 = db.conn().query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('trigger_log', 'patterns')",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_think_empty_db() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let config = ThinkConfig {
            run_consolidation: false,
            run_conflict_scan: false,
            run_pattern_mining: false,
            ..Default::default()
        };
        let result = db.think(&config).unwrap();
        assert!(result.triggers.is_empty());
        assert_eq!(result.consolidation_count, 0);
        assert_eq!(result.conflicts_found, 0);
        assert!(result.duration_ms < 5000);
    }

    #[test]
    fn test_think_with_decayed_memories() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid = db.record("important deadline", "episodic", 0.9, 0.0, 100.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();

        // Backdate last_access
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
        db.conn().execute(
            "UPDATE memories SET last_access = ?1 WHERE rid = ?2",
            rusqlite::params![ts - 10000.0, rid],
        ).unwrap();

        let config = ThinkConfig {
            run_consolidation: false,
            run_conflict_scan: false,
            run_pattern_mining: false,
            ..Default::default()
        };
        let result = db.think(&config).unwrap();
        assert!(!result.triggers.is_empty());
        assert_eq!(result.triggers[0].trigger_type, "decay_review");
    }

    #[test]
    fn test_think_records_last_think_at() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let config = ThinkConfig {
            run_consolidation: false,
            run_conflict_scan: false,
            run_pattern_mining: false,
            ..Default::default()
        };
        db.think(&config).unwrap();

        let val: String = db.conn().query_row(
            "SELECT value FROM meta WHERE key = 'last_think_at'",
            [], |row| row.get(0),
        ).unwrap();
        let ts: f64 = val.parse().unwrap();
        assert!(ts > 0.0);
    }

    #[test]
    fn test_trigger_lifecycle() {
        let db = AIDB::new(":memory:", 8).unwrap();

        // Create a trigger via persistence
        let trigger = crate::types::Trigger {
            trigger_type: "decay_review".to_string(),
            reason: "test".to_string(),
            urgency: 0.8,
            source_rids: vec!["rid-1".to_string()],
            suggested_action: "test".to_string(),
            context: std::collections::HashMap::new(),
        };
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
        let tid = crate::triggers::persist_trigger(&db, &trigger, ts).unwrap().unwrap();

        // Verify pending
        let pending = db.get_pending_triggers(10).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].status, "pending");

        // Deliver
        assert!(db.deliver_trigger(&tid).unwrap());
        let history = db.get_trigger_history(None, 10).unwrap();
        assert_eq!(history[0].status, "delivered");

        // Acknowledge
        assert!(db.acknowledge_trigger(&tid).unwrap());

        // Act
        assert!(db.act_on_trigger(&tid).unwrap());
        let history = db.get_trigger_history(None, 10).unwrap();
        assert_eq!(history[0].status, "acted");
    }

    #[test]
    fn test_stats_include_triggers_and_patterns() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let s = db.stats().unwrap();
        assert_eq!(s.pending_triggers, 0);
        assert_eq!(s.active_patterns, 0);
    }

    // ── Graph-augmented recall: invariant & regression tests ──

    #[test]
    fn test_entity_type_stored_on_relate() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.relate("Sarah", "data pipeline", "leads", 1.0).unwrap();
        db.relate("FAISS", "recommendation engine", "used_in", 1.0).unwrap();
        db.relate("Mike", "ONNX", "built_with", 1.0).unwrap();

        // Sarah → person, FAISS → tech, data pipeline → unknown, Mike → person, ONNX → tech
        let etype: String = db.conn.query_row(
            "SELECT entity_type FROM entities WHERE name = 'Sarah'", [], |r| r.get(0),
        ).unwrap();
        assert_eq!(etype, "person");

        let etype: String = db.conn.query_row(
            "SELECT entity_type FROM entities WHERE name = 'FAISS'", [], |r| r.get(0),
        ).unwrap();
        assert_eq!(etype, "tech");

        let etype: String = db.conn.query_row(
            "SELECT entity_type FROM entities WHERE name = 'data pipeline'", [], |r| r.get(0),
        ).unwrap();
        assert_eq!(etype, "unknown");

        let etype: String = db.conn.query_row(
            "SELECT entity_type FROM entities WHERE name = 'Mike'", [], |r| r.get(0),
        ).unwrap();
        assert_eq!(etype, "person");
    }

    #[test]
    fn test_recall_deterministic_with_skip_reinforce() {
        let db = AIDB::new(":memory:", 8).unwrap();
        for i in 0..10 {
            db.record(&format!("memory {i}"), "episodic", 0.5, 0.0, 604800.0,
                &empty_meta(), &vec_seed(i as f32, 8)).unwrap();
        }
        let query = vec_seed(3.0, 8);

        let r1 = db.recall(&query, 5, None, None, false, false, None, true).unwrap();
        let r2 = db.recall(&query, 5, None, None, false, false, None, true).unwrap();
        let r3 = db.recall(&query, 5, None, None, false, false, None, true).unwrap();

        // Same RIDs in same order every time
        let rids1: Vec<&str> = r1.iter().map(|r| r.rid.as_str()).collect();
        let rids2: Vec<&str> = r2.iter().map(|r| r.rid.as_str()).collect();
        let rids3: Vec<&str> = r3.iter().map(|r| r.rid.as_str()).collect();
        assert_eq!(rids1, rids2);
        assert_eq!(rids2, rids3);

        // Scores very close (tiny drift from wall-clock recency between calls)
        for i in 0..5 {
            assert!((r1[i].score - r2[i].score).abs() < 1e-4,
                "score drift too large between calls: {} vs {}", r1[i].score, r2[i].score);
        }
    }

    #[test]
    fn test_reinforce_mutates_but_skip_does_not() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid = db.record("test", "episodic", 0.5, 0.0, 1000.0,
            &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        let original_hl = db.get(&rid).unwrap().unwrap().half_life;

        // skip_reinforce=true should NOT change half_life
        db.recall(&vec_seed(1.0, 8), 1, None, None, false, false, None, true).unwrap();
        let after_skip = db.get(&rid).unwrap().unwrap().half_life;
        assert!((original_hl - after_skip).abs() < 1e-10);

        // skip_reinforce=false SHOULD change half_life
        db.recall(&vec_seed(1.0, 8), 1, None, None, false, false, None, false).unwrap();
        let after_reinforce = db.get(&rid).unwrap().unwrap().half_life;
        assert!(after_reinforce > original_hl);
    }

    #[test]
    fn test_graph_expansion_off_no_graph_results() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let r1 = db.record("Alice discussed plan", "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        let r2 = db.record("Bob reviewed code", "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(5.0, 8)).unwrap();
        db.relate("Alice", "Bob", "knows", 1.0).unwrap();
        db.link_memory_entity(&r1, "Alice").unwrap();
        db.link_memory_entity(&r2, "Bob").unwrap();

        // expand_entities=false: no graph_proximity should be set
        let results = db.recall(&vec_seed(1.0, 8), 10, None, None, false, false,
            Some("Alice"), false).unwrap();
        for r in &results {
            assert!((r.scores.graph_proximity - 0.0).abs() < 1e-10,
                "graph_proximity should be 0.0 when expansion is disabled");
        }
    }

    #[test]
    fn test_graph_expansion_on_boosts_connected_memory() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let r1 = db.record("Alice discussed the project plan", "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        let r2 = db.record("Bob reviewed the code", "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(5.0, 8)).unwrap();
        db.relate("Alice", "Bob", "knows", 1.0).unwrap();
        db.link_memory_entity(&r1, "Alice").unwrap();
        db.link_memory_entity(&r2, "Bob").unwrap();

        // expand_entities=true with query mentioning "Alice"
        let results = db.recall(&vec_seed(1.0, 8), 10, None, None, false, true,
            Some("What is Alice working on?"), true).unwrap();

        // The Alice memory should have graph_proximity > 0
        let alice_result = results.iter().find(|r| r.rid == r1).unwrap();
        assert!(alice_result.scores.graph_proximity > 0.0,
            "Alice memory should have graph proximity when expansion is on");
    }

    #[test]
    fn test_backfill_uses_word_boundaries() {
        let db = AIDB::new(":memory:", 8).unwrap();
        // Create an entity "data"
        db.relate("data", "pipeline", "part_of", 1.0).unwrap();

        // Create memories: one with "data" as a word, one with "database" (contains "data")
        let r1 = db.record("the data is clean", "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        let r2 = db.record("the database is fast", "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(2.0, 8)).unwrap();

        let count = db.backfill_memory_entities().unwrap();

        // Check: r1 should be linked to "data", r2 should NOT
        let linked_to_data: Vec<String> = db.conn.prepare(
            "SELECT memory_rid FROM memory_entities WHERE entity_name = 'data'"
        ).unwrap().query_map([], |row| row.get(0)).unwrap()
            .collect::<std::result::Result<Vec<_>, _>>().unwrap();

        assert!(linked_to_data.contains(&r1), "memory with 'data' as word should be linked");
        assert!(!linked_to_data.contains(&r2), "memory with 'database' should NOT be linked (word boundary)");
    }

    #[test]
    fn test_recall_scores_bounded() {
        // All recall scores should be non-negative and reasonably bounded
        let db = AIDB::new(":memory:", 8).unwrap();
        for i in 0..10 {
            db.record(&format!("memory {i}"), "episodic",
                (i as f64) * 0.1, // importance 0..0.9
                ((i as f64) - 5.0) * 0.2, // valence -1.0..0.8
                604800.0, &empty_meta(), &vec_seed(i as f32, 8)).unwrap();
        }

        let results = db.recall(&vec_seed(5.0, 8), 10, None, None, false, false, None, true).unwrap();
        for r in &results {
            assert!(r.score >= 0.0, "score should be non-negative, got {}", r.score);
            assert!(r.score < 5.0, "score should be reasonably bounded, got {}", r.score);
            assert!(r.scores.similarity >= -1.0 && r.scores.similarity <= 1.0);
            assert!(r.scores.decay >= 0.0 && r.scores.decay <= 1.0);
            assert!(r.scores.recency >= 0.0 && r.scores.recency <= 1.0);
        }
    }

    #[test]
    fn test_link_memory_entity_idempotent() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid = db.record("test", "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        db.relate("Alice", "Bob", "knows", 1.0).unwrap();

        // Link twice — should not error or create duplicates
        db.link_memory_entity(&rid, "Alice").unwrap();
        db.link_memory_entity(&rid, "Alice").unwrap();

        let count: i64 = db.conn.query_row(
            "SELECT COUNT(*) FROM memory_entities WHERE memory_rid = ?1 AND entity_name = 'Alice'",
            params![rid], |r| r.get(0),
        ).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_schema_v5_has_memory_entities() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let count: i64 = db.conn().query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_entities'",
            [], |r| r.get(0),
        ).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_recall_top_k_respected_with_graph_expansion() {
        let db = AIDB::new(":memory:", 8).unwrap();
        // Create a web of interconnected memories
        for i in 0..20 {
            let rid = db.record(&format!("memory about topic {i}"), "episodic",
                0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(i as f32, 8)).unwrap();
            let entity = format!("Entity{i}");
            db.relate(&entity, &format!("Entity{}", (i + 1) % 20), "related_to", 1.0).unwrap();
            db.link_memory_entity(&rid, &entity).unwrap();
        }

        let results = db.recall(&vec_seed(0.0, 8), 5, None, None, false, true,
            Some("Entity0 topic"), true).unwrap();

        // top_k=5 must be respected even with graph expansion
        assert!(results.len() <= 5, "results should not exceed top_k=5, got {}", results.len());
    }

    // ── V4: Storage & Performance tests ──

    #[test]
    fn test_schema_v6_has_storage_tier() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid = db.record("tier test", "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        let mem = db.get(&rid).unwrap().unwrap();
        assert_eq!(mem.storage_tier, "hot");
    }

    #[test]
    fn test_schema_v7_has_fts5_and_join_tables() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let conn = db.conn();

        // FTS5 virtual table exists — insert then search
        let _rid = db.record("The quick brown fox jumps over the lazy dog", "episodic",
            0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();

        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'quick brown'",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(count, 1, "FTS5 should index inserted memory");

        // Join tables exist
        let _: i64 = conn.query_row(
            "SELECT COUNT(*) FROM trigger_source_rids", [], |row| row.get(0),
        ).unwrap();
        let _: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pattern_evidence", [], |row| row.get(0),
        ).unwrap();
        let _: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pattern_entities", [], |row| row.get(0),
        ).unwrap();

        // Schema version is 7
        let ver: String = conn.query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'", [], |row| row.get(0),
        ).unwrap();
        assert_eq!(ver, "7");
    }

    #[test]
    fn test_fts5_search_multiple_memories() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let conn = db.conn();

        db.record("Alice loves Rust programming", "semantic",
            0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        db.record("Bob prefers Python scripting", "semantic",
            0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(0.5, 8)).unwrap();
        db.record("Alice and Bob work on Rust projects", "episodic",
            0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(0.3, 8)).unwrap();

        // Search for "Rust" should match 2 memories
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'rust'",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(count, 2, "FTS5 should find 2 memories containing 'rust'");

        // Search for "Alice" should match 2 memories
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'alice'",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(count, 2, "FTS5 should find 2 memories containing 'alice'");

        // Search for "Python" should match 1
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'python'",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_archive_memory() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid = db.record("to archive", "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(1.0, 8)).unwrap();

        // Archive
        assert!(db.archive(&rid).unwrap());
        let mem = db.get(&rid).unwrap().unwrap();
        assert_eq!(mem.storage_tier, "cold");

        // Verify removed from vec_memories (recall should not find it)
        let results = db.recall(&vec_seed(1.0, 8), 10, None, None, false, false, None, true).unwrap();
        assert!(results.iter().all(|r| r.rid != rid), "archived memory should not appear in recall");

        // Stats should show archived
        assert_eq!(db.stats().unwrap().archived_memories, 1);
    }

    #[test]
    fn test_hydrate_memory() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let emb = vec_seed(2.0, 8);
        let rid = db.record("to hydrate", "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &emb).unwrap();

        // Archive then hydrate
        db.archive(&rid).unwrap();
        assert!(db.hydrate(&rid).unwrap());
        let mem = db.get(&rid).unwrap().unwrap();
        assert_eq!(mem.storage_tier, "hot");

        // Should be back in recall
        let results = db.recall(&emb, 10, None, None, false, false, None, true).unwrap();
        assert!(results.iter().any(|r| r.rid == rid), "hydrated memory should appear in recall");

        // Stats
        assert_eq!(db.stats().unwrap().archived_memories, 0);
    }

    #[test]
    fn test_archive_idempotent() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid = db.record("idempotent", "episodic", 0.5, 0.0, 604800.0,
            &empty_meta(), &vec_seed(1.0, 8)).unwrap();

        assert!(db.archive(&rid).unwrap());
        assert!(!db.archive(&rid).unwrap()); // Already cold
    }

    #[test]
    fn test_record_batch() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let inputs: Vec<RecordInput> = (0..10).map(|i| RecordInput {
            text: format!("batch memory {i}"),
            memory_type: "episodic".to_string(),
            importance: 0.5,
            valence: 0.0,
            half_life: 604800.0,
            metadata: serde_json::json!({}),
            embedding: vec_seed(i as f32, 8),
        }).collect();

        let rids = db.record_batch(&inputs).unwrap();
        assert_eq!(rids.len(), 10);

        // All retrievable
        for rid in &rids {
            assert!(db.get(rid).unwrap().is_some());
        }
        assert_eq!(db.stats().unwrap().active_memories, 10);
    }

    #[test]
    fn test_record_batch_empty() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rids = db.record_batch(&[]).unwrap();
        assert!(rids.is_empty());
    }

    #[test]
    fn test_evict() {
        let db = AIDB::new(":memory:", 8).unwrap();
        // Seed 20 memories
        for i in 0..20 {
            db.record(&format!("evict mem {i}"), "episodic", 0.5, 0.0, 604800.0,
                &empty_meta(), &vec_seed(i as f32, 8)).unwrap();
        }
        assert_eq!(db.stats().unwrap().active_memories, 20);

        // Evict to keep 10
        let archived = db.evict(10).unwrap();
        assert_eq!(archived.len(), 10);

        let stats = db.stats().unwrap();
        assert_eq!(stats.archived_memories, 10);

        // Recall should only find hot memories
        let results = db.recall(&vec_seed(0.0, 8), 20, None, None, false, false, None, true).unwrap();
        for r in &results {
            assert!(!archived.contains(&r.rid), "evicted memory should not be in recall");
        }
    }

    #[test]
    fn test_evict_no_action_when_under_limit() {
        let db = AIDB::new(":memory:", 8).unwrap();
        for i in 0..5 {
            db.record(&format!("small db {i}"), "episodic", 0.5, 0.0, 604800.0,
                &empty_meta(), &vec_seed(i as f32, 8)).unwrap();
        }
        let archived = db.evict(10).unwrap();
        assert!(archived.is_empty());
    }
}
