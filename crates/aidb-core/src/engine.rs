use std::cell::RefCell;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::Rng;
use rusqlite::{params, Connection};

use crate::error::{AidbError, Result};
use crate::hlc::{HLCTimestamp, HLC};
use crate::schema::{MIGRATE_V1_TO_V2, MIGRATE_V2_TO_V3, SCHEMA_SQL, SCHEMA_VERSION};
use crate::scoring;
use crate::serde_helpers::serialize_f32;
use crate::types::*;

/// The AIDB cognitive memory engine.
pub struct AIDB {
    conn: Connection,
    embedding_dim: usize,
    hlc: RefCell<HLC>,
    actor_id: String,
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
        // Register sqlite-vec as auto-extension before opening any connection
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }

        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        // Check existing schema version for migration
        let existing_version = Self::get_schema_version(&conn);

        if existing_version == Some(1) {
            // Migrate V1 -> V2 -> V3
            conn.execute_batch(MIGRATE_V1_TO_V2)?;
            conn.execute_batch(MIGRATE_V2_TO_V3)?;
        } else if existing_version == Some(2) {
            // Migrate V2 -> V3
            conn.execute_batch(MIGRATE_V2_TO_V3)?;
        }

        conn.execute_batch(SCHEMA_SQL)?;

        // Create virtual table for vector search
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories \
             USING vec0(rid TEXT PRIMARY KEY, embedding float[{embedding_dim}])"
        ))?;

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

        Ok(Self {
            conn,
            embedding_dim,
            hlc: RefCell::new(HLC::new(node_id)),
            actor_id,
        })
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
        self.conn.execute(
            "INSERT INTO vec_memories (rid, embedding) VALUES (?1, ?2)",
            params![rid, emb_blob],
        )?;

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

    // ── recall() — multi-signal retrieval ──

    /// Retrieve memories using multi-signal fusion scoring.
    pub fn recall(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
    ) -> Result<Vec<RecallResult>> {
        let ts = now();
        let emb_blob = serialize_f32(query_embedding);

        // Step 1: Vector candidate generation
        let fetch_k = (top_k * 5).min(200);
        let mut stmt = self.conn.prepare(
            "SELECT rid, distance FROM vec_memories \
             WHERE embedding MATCH ?1 ORDER BY distance LIMIT ?2",
        )?;

        let vec_results: Vec<(String, f64)> = stmt
            .query_map(params![emb_blob, fetch_k as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        if vec_results.is_empty() {
            return Ok(vec![]);
        }

        let rids: Vec<&str> = vec_results.iter().map(|(r, _)| r.as_str()).collect();
        let vec_scores: std::collections::HashMap<&str, f64> = vec_results
            .iter()
            .map(|(r, d)| (r.as_str(), 1.0 - d))
            .collect();

        // Step 2: Fetch full memory records with filtering
        let statuses: Vec<&str> = if include_consolidated {
            vec!["active", "consolidated"]
        } else {
            vec!["active"]
        };

        let rid_placeholders: String = (0..rids.len()).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
        let status_offset = rids.len() + 1;
        let status_placeholders: String = (0..statuses.len())
            .map(|i| format!("?{}", status_offset + i))
            .collect::<Vec<_>>()
            .join(",");

        let mut sql = format!(
            "SELECT * FROM memories WHERE rid IN ({rid_placeholders}) \
             AND consolidation_status IN ({status_placeholders})"
        );

        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for r in &rids {
            param_values.push(Box::new(r.to_string()));
        }
        for s in &statuses {
            param_values.push(Box::new(s.to_string()));
        }

        if let Some((start, end)) = time_window {
            let n = param_values.len();
            sql.push_str(&format!(" AND created_at BETWEEN ?{} AND ?{}", n + 1, n + 2));
            param_values.push(Box::new(start));
            param_values.push(Box::new(end));
        }

        if let Some(mt) = memory_type {
            let n = param_values.len();
            sql.push_str(&format!(" AND type = ?{}", n + 1));
            param_values.push(Box::new(mt.to_string()));
        }

        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = self.conn.prepare(&sql)?;
        let memories: Vec<_> = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok(MemoryRow {
                    rid: row.get("rid")?,
                    memory_type: row.get("type")?,
                    text: row.get("text")?,
                    created_at: row.get("created_at")?,
                    importance: row.get("importance")?,
                    valence: row.get("valence")?,
                    half_life: row.get("half_life")?,
                    last_access: row.get("last_access")?,
                    metadata: row.get("metadata")?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Step 3: Score with multi-signal fusion
        let mut scored: Vec<RecallResult> = Vec::new();
        for mem in &memories {
            let sim_score = *vec_scores.get(mem.rid.as_str()).unwrap_or(&0.0);

            let elapsed = ts - mem.last_access;
            let decay = scoring::decay_score(mem.importance, mem.half_life, elapsed);

            let age = ts - mem.created_at;
            let recency = scoring::recency_score(age);

            let composite = scoring::composite_score(
                sim_score,
                decay,
                recency,
                mem.importance,
                mem.valence,
            );

            let why = scoring::build_why(sim_score, recency, decay, mem.valence);

            let metadata: serde_json::Value =
                serde_json::from_str(&mem.metadata).unwrap_or(serde_json::Value::Object(Default::default()));

            scored.push(RecallResult {
                rid: mem.rid.clone(),
                memory_type: mem.memory_type.clone(),
                text: mem.text.clone(),
                created_at: mem.created_at,
                importance: mem.importance,
                valence: mem.valence,
                score: composite,
                scores: ScoreBreakdown {
                    similarity: sim_score,
                    decay,
                    recency,
                    importance: mem.importance,
                },
                why_retrieved: why,
                metadata,
            });
        }

        // Step 4: Sort and return top_k
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        // Reinforce accessed memories (spaced repetition)
        for r in &scored {
            self.reinforce(&r.rid)?;
        }

        Ok(scored)
    }

    /// Reinforce a memory on access — increase half_life and update last_access.
    fn reinforce(&self, rid: &str) -> Result<()> {
        let ts = now();
        let new_half_life: f64 = self.conn.query_row(
            "SELECT MIN(half_life * 1.2, 31536000.0) FROM memories WHERE rid = ?1",
            params![rid],
            |row| row.get(0),
        ).unwrap_or(604800.0);

        self.conn.execute(
            "UPDATE memories SET last_access = ?1, half_life = MIN(half_life * 1.2, 31536000.0) WHERE rid = ?2",
            params![ts, rid],
        )?;

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

        // Ensure entities exist
        for entity in [src, dst] {
            self.conn.execute(
                "INSERT INTO entities (name, first_seen, last_seen) \
                 VALUES (?1, ?2, ?3) \
                 ON CONFLICT(name) DO UPDATE SET last_seen = ?3, mention_count = mention_count + 1",
                params![entity, ts, ts],
            )?;
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
            self.conn.execute(
                "DELETE FROM vec_memories WHERE rid = ?1",
                params![rid],
            )?;
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

        Ok(Stats {
            active_memories: active,
            consolidated_memories: consolidated,
            tombstoned_memories: tombstoned,
            edges,
            entities,
            operations,
            open_conflicts,
            resolved_conflicts,
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

    /// Close the database connection. After this, the engine cannot be used.
    pub fn close(self) -> Result<()> {
        self.conn.close().map_err(|(_, e)| AidbError::Database(e))
    }
}

/// Internal row struct for recall query results.
struct MemoryRow {
    rid: String,
    memory_type: String,
    text: String,
    created_at: f64,
    importance: f64,
    valence: f64,
    half_life: f64,
    last_access: f64,
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

        let results = db.recall(&vec_seed(1.0, 8), 2, None, None, false).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_recall_empty() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let results = db.recall(&vec_seed(1.0, 8), 5, None, None, false).unwrap();
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
}
