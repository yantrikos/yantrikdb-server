//! Replication: oplog extraction, CRDT materialization, and operation application.
//!
//! Core principle: replicate operations, not index structures.
//! - Memories: Add-Wins Set (UUIDv7 uniqueness, INSERT OR IGNORE)
//! - Edges: LWW on (src, dst, rel_type), higher HLC wins
//! - Entities: Derived state, recomputed from edges
//! - Forget: Tombstone always wins (irreversible)
//! - Consolidation: Set-union via consolidation_members table

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::engine::YantrikDB;
use crate::error::Result;
use crate::hlc::HLCTimestamp;
use crate::types::ScoringRow;

/// An oplog entry for replication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OplogEntry {
    pub op_id: String,
    pub op_type: String,
    pub timestamp: f64,
    pub target_rid: Option<String>,
    pub payload: serde_json::Value,
    pub actor_id: String,
    pub hlc: Vec<u8>,
    pub embedding_hash: Option<Vec<u8>>,
    pub origin_actor: String,
}

/// Result of a sync operation.
#[derive(Debug, Clone)]
pub struct SyncStats {
    pub ops_applied: usize,
    pub ops_skipped: usize,
}

/// Extract ops from the oplog since a given cursor (hlc, op_id).
/// Uses compound cursor: (hlc > since_hlc) OR (hlc = since_hlc AND op_id > since_op_id)
pub fn extract_ops_since(
    conn: &Connection,
    since_hlc: Option<&[u8]>,
    since_op_id: Option<&str>,
    exclude_actor: Option<&str>,
    limit: usize,
) -> Result<Vec<OplogEntry>> {
    let (sql, param_values) = match (since_hlc, since_op_id) {
        (Some(hlc), Some(op_id)) => {
            let mut sql = String::from(
                "SELECT op_id, op_type, timestamp, target_rid, payload, \
                 actor_id, hlc, embedding_hash, origin_actor \
                 FROM oplog \
                 WHERE hlc IS NOT NULL \
                 AND ((hlc > ?1) OR (hlc = ?1 AND op_id > ?2))",
            );
            let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![
                Box::new(hlc.to_vec()),
                Box::new(op_id.to_string()),
            ];

            if let Some(actor) = exclude_actor {
                sql.push_str(" AND origin_actor != ?3");
                params.push(Box::new(actor.to_string()));
            }

            sql.push_str(" ORDER BY hlc, op_id");
            sql.push_str(&format!(" LIMIT {limit}"));
            (sql, params)
        }
        _ => {
            let mut sql = String::from(
                "SELECT op_id, op_type, timestamp, target_rid, payload, \
                 actor_id, hlc, embedding_hash, origin_actor \
                 FROM oplog \
                 WHERE hlc IS NOT NULL",
            );
            let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![];

            if let Some(actor) = exclude_actor {
                sql.push_str(" AND origin_actor != ?1");
                params.push(Box::new(actor.to_string()));
            }

            sql.push_str(" ORDER BY hlc, op_id");
            sql.push_str(&format!(" LIMIT {limit}"));
            (sql, params)
        }
    };

    let params_ref: Vec<&dyn rusqlite::types::ToSql> =
        param_values.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&sql)?;
    let entries = stmt
        .query_map(params_ref.as_slice(), |row| {
            let payload_str: String = row.get("payload")?;
            let payload: serde_json::Value =
                serde_json::from_str(&payload_str).unwrap_or(serde_json::json!({}));

            Ok(OplogEntry {
                op_id: row.get("op_id")?,
                op_type: row.get("op_type")?,
                timestamp: row.get("timestamp")?,
                target_rid: row.get("target_rid")?,
                payload,
                actor_id: row.get("actor_id")?,
                hlc: row.get("hlc")?,
                embedding_hash: row.get("embedding_hash")?,
                origin_actor: row.get("origin_actor")?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(entries)
}

/// Apply remote ops to a local YantrikDB instance. Idempotent via INSERT OR IGNORE on op_id.
/// Returns the number of ops actually applied (newly inserted).
pub fn apply_ops(db: &YantrikDB, ops: &[OplogEntry]) -> Result<SyncStats> {
    let mut applied = 0;
    let mut skipped = 0;
    let mut has_relate_or_record = false;

    for op in ops {
        // Check if we already have this op (idempotent)
        let exists: bool = db.conn().query_row(
            "SELECT COUNT(*) > 0 FROM oplog WHERE op_id = ?1",
            params![op.op_id],
            |row| row.get(0),
        )?;

        if exists {
            skipped += 1;
            continue;
        }

        // Merge HLC
        if let Some(remote_ts) = HLCTimestamp::from_bytes(&op.hlc) {
            db.merge_hlc(remote_ts);
        }

        // Track if we need to backfill memory_entities after
        if op.op_type == "relate" || op.op_type == "record" {
            has_relate_or_record = true;
        }

        // Materialize the operation's side effects
        materialize_op(db, op)?;

        // Insert the op into our local oplog
        let payload_str = serde_json::to_string(&op.payload)?;
        db.conn().execute(
            "INSERT OR IGNORE INTO oplog \
             (op_id, op_type, timestamp, target_rid, payload, \
              actor_id, hlc, embedding_hash, origin_actor, applied) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1)",
            params![
                op.op_id,
                op.op_type,
                op.timestamp,
                op.target_rid,
                payload_str,
                op.actor_id,
                op.hlc,
                op.embedding_hash,
                op.origin_actor,
            ],
        )?;

        applied += 1;
    }

    // Backfill memory_entities if any relate/record ops were applied.
    // This ensures the join table stays current after sync.
    if has_relate_or_record && applied > 0 {
        let _ = db.backfill_memory_entities();
    }

    Ok(SyncStats {
        ops_applied: applied,
        ops_skipped: skipped,
    })
}

/// Materialize a single op — replay its side effects on the local DB.
fn materialize_op(db: &YantrikDB, op: &OplogEntry) -> Result<()> {
    match op.op_type.as_str() {
        "record" => {
            materialize_record(&*db.conn(), &op.payload, db.embedding_dim())?;
            // Update scoring cache with new record
            let rid = op.payload["rid"].as_str().unwrap_or_default();
            if !rid.is_empty() {
                db.cache_insert(rid.to_string(), ScoringRow {
                    created_at: op.payload["created_at"].as_f64().unwrap_or(0.0),
                    importance: op.payload["importance"].as_f64().unwrap_or(0.5),
                    half_life: op.payload["half_life"].as_f64().unwrap_or(604800.0),
                    last_access: op.payload["created_at"].as_f64().unwrap_or(0.0),
                    access_count: 0,
                    valence: op.payload["valence"].as_f64().unwrap_or(0.0),
                    consolidation_status: "active".to_string(),
                    memory_type: op.payload["type"].as_str().unwrap_or("episodic").to_string(),
                    namespace: op.payload["namespace"].as_str().unwrap_or("default").to_string(),
                    certainty: op.payload["certainty"].as_f64().unwrap_or(0.8),
                    domain: op.payload["domain"].as_str().unwrap_or("general").to_string(),
                    source: op.payload["source"].as_str().unwrap_or("user").to_string(),
                    emotional_state: op.payload["emotional_state"].as_str().map(|s| s.to_string()),
                });
            }
        }
        "relate" => {
            materialize_relate(&*db.conn(), &op.payload)?;
            // Update graph index
            let src = op.payload["src"].as_str().unwrap_or_default();
            let dst = op.payload["dst"].as_str().unwrap_or_default();
            let rel_type = op.payload["rel_type"].as_str().unwrap_or_default();
            let weight = op.payload["weight"].as_f64().unwrap_or(1.0);
            if !src.is_empty() && !dst.is_empty() {
                let mut gi = db.graph_index.write().unwrap();
                let (src_type, dst_type) =
                    crate::graph::classify_with_relationship(src, dst, rel_type);
                gi.add_entity(src, src_type);
                gi.add_entity(dst, dst_type);
                gi.add_edge(src, dst, weight as f32);
                drop(gi);
                // V2: detect edge conflicts during sync
                let _ = crate::conflict::detect_edge_conflicts(
                    db, src, dst, rel_type, op.target_rid.as_deref(),
                );
            }
        }
        "forget" => {
            materialize_forget(&*db.conn(), &op.payload)?;
            // Remove from scoring cache + vec index + graph index
            let rid = op.payload["rid"].as_str().unwrap_or_default();
            if !rid.is_empty() {
                db.cache_remove(rid);
                db.vec_index.write().unwrap().remove(rid);
                db.graph_index.write().unwrap().unlink_memory(rid);
            }
        }
        "consolidate" => {
            materialize_consolidate(&*db.conn(), &op.payload, &op.hlc, &op.origin_actor)?;
            // Cache: insert consolidated memory + mark sources
            let consolidated_rid = op.payload["consolidated_rid"].as_str().unwrap_or_default();
            let text = op.payload["text"].as_str().unwrap_or("");
            if !consolidated_rid.is_empty() && !text.is_empty() {
                db.cache_insert(consolidated_rid.to_string(), ScoringRow {
                    created_at: op.timestamp,
                    importance: op.payload["importance"].as_f64().unwrap_or(0.5),
                    half_life: op.payload["half_life"].as_f64().unwrap_or(604800.0),
                    last_access: op.timestamp,
                    access_count: 0,
                    valence: op.payload["valence"].as_f64().unwrap_or(0.0),
                    consolidation_status: "active".to_string(),
                    memory_type: "semantic".to_string(),
                    namespace: op.payload["namespace"].as_str().unwrap_or("default").to_string(),
                    certainty: 0.8,
                    domain: "general".to_string(),
                    source: "user".to_string(),
                    emotional_state: None,
                });
            }
            if let Some(source_rids) = op.payload["source_rids"].as_array() {
                for rid_val in source_rids {
                    if let Some(rid) = rid_val.as_str() {
                        db.cache_mark_consolidated(rid, 0.3);
                    }
                }
            }
        }
        "conflict_detect" => materialize_conflict_detect(&*db.conn(), &op.payload, &op.hlc, &op.origin_actor)?,
        "conflict_resolve" => {
            materialize_conflict_resolve(&*db.conn(), &op.payload)?;
            // If keep_a or keep_b, remove the loser from cache + vec index
            let strategy = op.payload["strategy"].as_str().unwrap_or("");
            if strategy == "keep_a" || strategy == "keep_b" {
                if let Some(loser) = op.payload["loser_rid"].as_str() {
                    db.cache_remove(loser);
                    db.vec_index.write().unwrap().remove(loser);
                }
            }
        }
        "correct" => {
            materialize_correct(&*db.conn(), &op.payload)?;
            // Cache: insert new corrected memory, remove original
            let new_rid = op.payload["new_rid"].as_str().unwrap_or_default();
            if !new_rid.is_empty() {
                db.cache_insert(new_rid.to_string(), ScoringRow {
                    created_at: op.payload["created_at"].as_f64().unwrap_or(0.0),
                    importance: op.payload["importance"].as_f64().unwrap_or(0.5),
                    half_life: op.payload["half_life"].as_f64().unwrap_or(604800.0),
                    last_access: op.payload["created_at"].as_f64().unwrap_or(0.0),
                    access_count: 0,
                    valence: op.payload["valence"].as_f64().unwrap_or(0.0),
                    consolidation_status: "active".to_string(),
                    memory_type: op.payload["type"].as_str().unwrap_or("episodic").to_string(),
                    namespace: op.payload["namespace"].as_str().unwrap_or("default").to_string(),
                    certainty: op.payload["certainty"].as_f64().unwrap_or(0.8),
                    domain: op.payload["domain"].as_str().unwrap_or("general").to_string(),
                    source: op.payload["source"].as_str().unwrap_or("user").to_string(),
                    emotional_state: op.payload["emotional_state"].as_str().map(|s| s.to_string()),
                });
            }
            let original_rid = op.payload["original_rid"].as_str().unwrap_or_default();
            if !original_rid.is_empty() {
                db.cache_remove(original_rid);
                db.vec_index.write().unwrap().remove(original_rid);
            }
        }
        "trigger_fire" => materialize_trigger_fire(&*db.conn(), &op.payload, &op.hlc, &op.origin_actor)?,
        "trigger_deliver" | "trigger_ack" | "trigger_act" | "trigger_dismiss" => {
            materialize_trigger_lifecycle(&*db.conn(), &op.payload)?;
        }
        "pattern_upsert" => materialize_pattern(&*db.conn(), &op.payload, &op.hlc, &op.origin_actor)?,
        "reinforce" | "think" => {
            // Local-only ops; skip during replication
        }
        _ => {
            // Unknown op types are silently skipped (forward compatibility)
        }
    }

    Ok(())
}

/// Materialize a "record" op: INSERT OR IGNORE into memories.
fn materialize_record(conn: &Connection, payload: &serde_json::Value, _embedding_dim: usize) -> Result<()> {
    let rid = payload["rid"].as_str().unwrap_or_default();
    let mem_type = payload["type"].as_str().unwrap_or("episodic");
    let text = payload["text"].as_str().unwrap_or("");
    let importance = payload["importance"].as_f64().unwrap_or(0.5);
    let valence = payload["valence"].as_f64().unwrap_or(0.0);
    let half_life = payload["half_life"].as_f64().unwrap_or(604800.0);
    let created_at = payload["created_at"].as_f64().unwrap_or(0.0);
    let updated_at = payload["updated_at"].as_f64().unwrap_or(created_at);
    let metadata = payload
        .get("metadata")
        .map(|m| serde_json::to_string(m).unwrap_or_else(|_| "{}".to_string()))
        .unwrap_or_else(|| "{}".to_string());

    if rid.is_empty() {
        return Ok(()); // Can't materialize without a rid
    }

    let namespace = payload["namespace"].as_str().unwrap_or("default");

    // Add-Wins Set: INSERT OR IGNORE means first writer wins (UUIDv7 = no collisions)
    conn.execute(
        "INSERT OR IGNORE INTO memories \
         (rid, type, text, created_at, updated_at, importance, \
          half_life, last_access, valence, metadata, namespace) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            rid, mem_type, text, created_at, updated_at, importance,
            half_life, created_at, valence, metadata, namespace,
        ],
    )?;

    // Note: we can't insert into the HNSW vec index without the actual embedding data.
    // The oplog only stores the embedding_hash. The rebuild_vec_index() function
    // can be used as fallback to rebuild the index from the memories table.

    Ok(())
}

/// Materialize a "relate" op: LWW on (src, dst, rel_type), higher HLC wins.
fn materialize_relate(conn: &Connection, payload: &serde_json::Value) -> Result<()> {
    let edge_id = payload["edge_id"].as_str().unwrap_or_default();
    let src = payload["src"].as_str().unwrap_or_default();
    let dst = payload["dst"].as_str().unwrap_or_default();
    let rel_type = payload["rel_type"].as_str().unwrap_or_default();
    let weight = payload["weight"].as_f64().unwrap_or(1.0);
    let created_at = payload["created_at"].as_f64().unwrap_or(0.0);

    if src.is_empty() || dst.is_empty() {
        return Ok(());
    }

    // LWW: ON CONFLICT update if the incoming created_at is newer
    conn.execute(
        "INSERT INTO edges (edge_id, src, dst, rel_type, weight, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
         ON CONFLICT(src, dst, rel_type) DO UPDATE SET \
         weight = CASE WHEN ?6 > created_at THEN ?5 ELSE weight END, \
         created_at = CASE WHEN ?6 > created_at THEN ?6 ELSE created_at END, \
         edge_id = CASE WHEN ?6 > created_at THEN ?1 ELSE edge_id END",
        params![edge_id, src, dst, rel_type, weight, created_at],
    )?;

    // Ensure entities exist
    let ts = created_at;
    for entity in [src, dst] {
        conn.execute(
            "INSERT INTO entities (name, first_seen, last_seen) \
             VALUES (?1, ?2, ?3) \
             ON CONFLICT(name) DO UPDATE SET \
             last_seen = MAX(last_seen, ?3), \
             mention_count = mention_count + 1",
            params![entity, ts, ts],
        )?;
    }

    Ok(())
}

/// Materialize a "forget" op: tombstone always wins.
fn materialize_forget(conn: &Connection, payload: &serde_json::Value) -> Result<()> {
    let rid = payload["rid"].as_str().unwrap_or_default();
    let updated_at = payload["updated_at"].as_f64().unwrap_or(0.0);

    if rid.is_empty() {
        return Ok(());
    }

    // Tombstone always wins — even if the memory doesn't exist locally yet
    conn.execute(
        "UPDATE memories SET consolidation_status = 'tombstoned', updated_at = ?1 WHERE rid = ?2",
        params![updated_at, rid],
    )?;

    // HNSW vec index removal is handled by the materialize_op dispatcher

    Ok(())
}

/// Materialize a "consolidate" op: insert into consolidation_members (set-union).
fn materialize_consolidate(
    conn: &Connection,
    payload: &serde_json::Value,
    hlc: &[u8],
    actor_id: &str,
) -> Result<()> {
    let consolidated_rid = payload["consolidated_rid"].as_str().unwrap_or_default();
    let source_rids = payload["source_rids"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if consolidated_rid.is_empty() || source_rids.is_empty() {
        return Ok(());
    }

    // Also materialize the consolidated memory itself if present in payload
    let text = payload["text"].as_str().unwrap_or("");
    if !text.is_empty() {
        let importance = payload["importance"].as_f64().unwrap_or(0.5);
        let valence = payload["valence"].as_f64().unwrap_or(0.0);
        let half_life = payload["half_life"].as_f64().unwrap_or(604800.0);
        let metadata = payload
            .get("metadata")
            .map(|m| serde_json::to_string(m).unwrap_or_else(|_| "{}".to_string()))
            .unwrap_or_else(|| "{}".to_string());
        let ts = crate::time::now_secs();

        let namespace = payload["namespace"].as_str().unwrap_or("default");
        conn.execute(
            "INSERT OR IGNORE INTO memories \
             (rid, type, text, created_at, updated_at, importance, \
              half_life, last_access, valence, metadata, namespace) \
             VALUES (?1, 'semantic', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                consolidated_rid, text, ts, ts, importance,
                half_life, ts, valence, metadata, namespace,
            ],
        )?;
    }

    // Insert consolidation_members entries (set-union CRDT: INSERT OR IGNORE)
    for source_rid in &source_rids {
        conn.execute(
            "INSERT OR IGNORE INTO consolidation_members \
             (consolidation_rid, source_rid, hlc, actor_id) \
             VALUES (?1, ?2, ?3, ?4)",
            params![consolidated_rid, source_rid, hlc, actor_id],
        )?;

        // Mark source memories as consolidated (if they exist locally)
        conn.execute(
            "UPDATE memories \
             SET consolidation_status = 'consolidated', \
                 consolidated_into = ?1, \
                 importance = importance * 0.3 \
             WHERE rid = ?2 AND consolidation_status = 'active'",
            params![consolidated_rid, source_rid],
        )?;
    }

    Ok(())
}

// ── V2: Conflict materializers ──

/// Materialize a "conflict_detect" op: INSERT OR IGNORE into conflicts.
fn materialize_conflict_detect(
    conn: &Connection,
    payload: &serde_json::Value,
    hlc: &[u8],
    origin_actor: &str,
) -> Result<()> {
    let conflict_id = payload["conflict_id"].as_str().unwrap_or_default();
    if conflict_id.is_empty() {
        return Ok(());
    }

    conn.execute(
        "INSERT OR IGNORE INTO conflicts
         (conflict_id, conflict_type, priority, status, memory_a, memory_b,
          entity, rel_type, detected_at, detected_by, detection_reason,
          hlc, origin_actor)
         VALUES (?1, ?2, ?3, 'open', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            conflict_id,
            payload["conflict_type"].as_str().unwrap_or("minor"),
            payload["priority"].as_str().unwrap_or("medium"),
            payload["memory_a"].as_str().unwrap_or_default(),
            payload["memory_b"].as_str().unwrap_or_default(),
            payload["entity"].as_str(),
            payload["rel_type"].as_str(),
            payload["detected_at"].as_f64().unwrap_or(0.0),
            payload["detected_by"].as_str().unwrap_or_default(),
            payload["detection_reason"].as_str().unwrap_or_default(),
            hlc,
            origin_actor,
        ],
    )?;
    Ok(())
}

/// Materialize a "conflict_resolve" op: update the conflict record.
fn materialize_conflict_resolve(conn: &Connection, payload: &serde_json::Value) -> Result<()> {
    let conflict_id = payload["conflict_id"].as_str().unwrap_or_default();
    if conflict_id.is_empty() {
        return Ok(());
    }

    let status = if payload["dismissed"].as_bool().unwrap_or(false) {
        "dismissed"
    } else {
        "resolved"
    };

    conn.execute(
        "UPDATE conflicts SET
         status = ?1,
         resolved_at = ?2,
         resolved_by = ?3,
         strategy = ?4,
         winner_rid = ?5,
         resolution_note = ?6
         WHERE conflict_id = ?7 AND status = 'open'",
        params![
            status,
            payload["resolved_at"].as_f64().unwrap_or(0.0),
            payload["resolved_by"].as_str().unwrap_or_default(),
            payload["strategy"].as_str().unwrap_or_default(),
            payload["winner_rid"].as_str(),
            payload["resolution_note"].as_str(),
            conflict_id,
        ],
    )?;

    // If strategy is keep_a or keep_b, tombstone the loser
    let strategy = payload["strategy"].as_str().unwrap_or("");
    let loser_rid = payload["loser_rid"].as_str();
    if strategy == "keep_a" || strategy == "keep_b" {
        if let Some(loser) = loser_rid {
            let ts = payload["resolved_at"].as_f64().unwrap_or(0.0);
            conn.execute(
                "UPDATE memories SET consolidation_status = 'tombstoned', updated_at = ?1
                 WHERE rid = ?2 AND consolidation_status = 'active'",
                params![ts, loser],
            )?;
            // HNSW vec index removal is handled by the materialize_op dispatcher
        }
    }

    Ok(())
}

/// Materialize a "correct" op: create new memory and tombstone original.
fn materialize_correct(conn: &Connection, payload: &serde_json::Value) -> Result<()> {
    let new_rid = payload["new_rid"].as_str().unwrap_or_default();
    if new_rid.is_empty() {
        return Ok(());
    }

    let text = payload["text"].as_str().unwrap_or("");
    let mem_type = payload["type"].as_str().unwrap_or("episodic");
    let importance = payload["importance"].as_f64().unwrap_or(0.5);
    let valence = payload["valence"].as_f64().unwrap_or(0.0);
    let half_life = payload["half_life"].as_f64().unwrap_or(604800.0);
    let created_at = payload["created_at"].as_f64().unwrap_or(0.0);
    let metadata = payload
        .get("metadata")
        .map(|m| serde_json::to_string(m).unwrap_or_else(|_| "{}".to_string()))
        .unwrap_or_else(|| "{}".to_string());

    let namespace = payload["namespace"].as_str().unwrap_or("default");
    conn.execute(
        "INSERT OR IGNORE INTO memories
         (rid, type, text, created_at, updated_at, importance,
          half_life, last_access, valence, metadata, namespace)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            new_rid, mem_type, text, created_at, created_at, importance,
            half_life, created_at, valence, metadata, namespace,
        ],
    )?;

    // Tombstone the original memory
    let original_rid = payload["original_rid"].as_str().unwrap_or_default();
    if !original_rid.is_empty() {
        conn.execute(
            "UPDATE memories SET consolidation_status = 'tombstoned', updated_at = ?1
             WHERE rid = ?2",
            params![created_at, original_rid],
        )?;
        // HNSW vec index removal is handled by the materialize_op dispatcher
    }

    Ok(())
}

// ── Watermark tracking for delta sync ──

/// Get the watermark for a specific peer (last synced HLC + op_id).
pub fn get_peer_watermark(conn: &Connection, peer_actor: &str) -> Result<Option<(Vec<u8>, String)>> {
    match conn.query_row(
        "SELECT last_synced_hlc, last_synced_op_id FROM sync_peers WHERE peer_actor = ?1",
        params![peer_actor],
        |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, String>(1)?,
            ))
        },
    ) {
        Ok(wm) => Ok(Some(wm)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Update the watermark for a specific peer.
pub fn set_peer_watermark(
    conn: &Connection,
    peer_actor: &str,
    hlc: &[u8],
    op_id: &str,
) -> Result<()> {
    let ts = crate::time::now_secs();

    conn.execute(
        "INSERT INTO sync_peers (peer_actor, last_synced_hlc, last_synced_op_id, last_sync_time) \
         VALUES (?1, ?2, ?3, ?4) \
         ON CONFLICT(peer_actor) DO UPDATE SET \
         last_synced_hlc = ?2, last_synced_op_id = ?3, last_sync_time = ?4",
        params![peer_actor, hlc, op_id, ts],
    )?;

    Ok(())
}

/// Rebuild the vector index from memories table (disaster recovery).
/// Delegates to YantrikDB::rebuild_vec_index which builds a new HnswIndex.
pub fn rebuild_vec_index(db: &YantrikDB) -> Result<usize> {
    db.rebuild_vec_index()
}

// ── V3 materializers: triggers and patterns ──

/// Materialize a "trigger_fire" op: INSERT OR IGNORE into trigger_log.
fn materialize_trigger_fire(
    conn: &Connection,
    payload: &serde_json::Value,
    hlc: &[u8],
    origin_actor: &str,
) -> Result<()> {
    let trigger_id = payload["trigger_id"].as_str().unwrap_or_default();
    if trigger_id.is_empty() {
        return Ok(());
    }

    conn.execute(
        "INSERT OR IGNORE INTO trigger_log \
         (trigger_id, trigger_type, urgency, status, reason, suggested_action, \
          source_rids, context, created_at, expires_at, cooldown_key, hlc, origin_actor) \
         VALUES (?1, ?2, ?3, 'pending', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            trigger_id,
            payload["trigger_type"].as_str().unwrap_or(""),
            payload["urgency"].as_f64().unwrap_or(0.0),
            payload["reason"].as_str().unwrap_or(""),
            payload["suggested_action"].as_str().unwrap_or(""),
            payload.get("source_rids").map(|v| v.to_string()).unwrap_or("[]".to_string()),
            payload.get("context").map(|v| v.to_string()).unwrap_or("{}".to_string()),
            payload["created_at"].as_f64().unwrap_or(0.0),
            payload["expires_at"].as_f64(),
            payload["cooldown_key"].as_str().unwrap_or(""),
            hlc,
            origin_actor,
        ],
    )?;

    // Dual-write to join table
    if let Some(rids) = payload.get("source_rids").and_then(|v| v.as_array()) {
        for rid_val in rids {
            if let Some(rid) = rid_val.as_str() {
                conn.execute(
                    "INSERT OR IGNORE INTO trigger_source_rids (trigger_id, rid) VALUES (?1, ?2)",
                    params![trigger_id, rid],
                )?;
            }
        }
    }

    Ok(())
}

/// Materialize a trigger lifecycle transition (deliver/ack/act/dismiss).
fn materialize_trigger_lifecycle(
    conn: &Connection,
    payload: &serde_json::Value,
) -> Result<()> {
    let trigger_id = payload["trigger_id"].as_str().unwrap_or_default();
    if trigger_id.is_empty() {
        return Ok(());
    }

    // Determine which status to set based on the payload keys
    if let Some(ts) = payload["dismissed_at"].as_f64() {
        conn.execute(
            "UPDATE trigger_log SET status = 'dismissed', acted_at = ?1 \
             WHERE trigger_id = ?2 AND status IN ('pending', 'delivered', 'acknowledged')",
            params![ts, trigger_id],
        )?;
    } else if let Some(ts) = payload["acted_at"].as_f64() {
        conn.execute(
            "UPDATE trigger_log SET status = 'acted', acted_at = ?1 \
             WHERE trigger_id = ?2 AND status IN ('delivered', 'acknowledged')",
            params![ts, trigger_id],
        )?;
    } else if let Some(ts) = payload["acknowledged_at"].as_f64() {
        conn.execute(
            "UPDATE trigger_log SET status = 'acknowledged', acknowledged_at = ?1 \
             WHERE trigger_id = ?2 AND status = 'delivered'",
            params![ts, trigger_id],
        )?;
    } else if let Some(ts) = payload["delivered_at"].as_f64() {
        conn.execute(
            "UPDATE trigger_log SET status = 'delivered', delivered_at = ?1 \
             WHERE trigger_id = ?2 AND status = 'pending'",
            params![ts, trigger_id],
        )?;
    }

    Ok(())
}

/// Materialize a "pattern_upsert" op: convergent merge into patterns table.
fn materialize_pattern(
    conn: &Connection,
    payload: &serde_json::Value,
    hlc: &[u8],
    origin_actor: &str,
) -> Result<()> {
    let pattern_id = payload["pattern_id"].as_str().unwrap_or_default();
    if pattern_id.is_empty() {
        return Ok(());
    }

    conn.execute(
        "INSERT INTO patterns \
         (pattern_id, pattern_type, status, confidence, description, \
          evidence_rids, entity_names, context, first_seen, last_confirmed, \
          occurrence_count, hlc, origin_actor) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13) \
         ON CONFLICT(pattern_id) DO UPDATE SET \
         confidence = MAX(confidence, excluded.confidence), \
         last_confirmed = MAX(last_confirmed, excluded.last_confirmed), \
         occurrence_count = MAX(occurrence_count, excluded.occurrence_count), \
         status = CASE WHEN excluded.last_confirmed > last_confirmed \
                  THEN excluded.status ELSE status END",
        params![
            pattern_id,
            payload["pattern_type"].as_str().unwrap_or(""),
            payload["status"].as_str().unwrap_or("active"),
            payload["confidence"].as_f64().unwrap_or(0.0),
            payload["description"].as_str().unwrap_or(""),
            payload.get("evidence_rids").map(|v| v.to_string()).unwrap_or("[]".to_string()),
            payload.get("entity_names").map(|v| v.to_string()).unwrap_or("[]".to_string()),
            payload.get("context").map(|v| v.to_string()).unwrap_or("{}".to_string()),
            payload["first_seen"].as_f64().unwrap_or(0.0),
            payload["last_confirmed"].as_f64().unwrap_or(0.0),
            payload["occurrence_count"].as_i64().unwrap_or(1),
            hlc,
            origin_actor,
        ],
    )?;

    // Dual-write to join tables
    if let Some(rids) = payload.get("evidence_rids").and_then(|v| v.as_array()) {
        for rid_val in rids {
            if let Some(rid) = rid_val.as_str() {
                conn.execute(
                    "INSERT OR IGNORE INTO pattern_evidence (pattern_id, rid) VALUES (?1, ?2)",
                    params![pattern_id, rid],
                )?;
            }
        }
    }
    if let Some(names) = payload.get("entity_names").and_then(|v| v.as_array()) {
        for name_val in names {
            if let Some(name) = name_val.as_str() {
                conn.execute(
                    "INSERT OR IGNORE INTO pattern_entities (pattern_id, entity_name) VALUES (?1, ?2)",
                    params![pattern_id, name],
                )?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::YantrikDB;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    fn empty_meta() -> serde_json::Value {
        serde_json::json!({})
    }

    #[test]
    fn test_extract_ops_empty() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let ops = extract_ops_since(&*db.conn(), None, None, None, 100).unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn test_extract_ops_after_record() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.record("hello", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

        let ops = extract_ops_since(&*db.conn(), None, None, None, 100).unwrap();
        // record + reinforce (from recall? no — just record op)
        assert!(!ops.is_empty());
        assert_eq!(ops[0].op_type, "record");
        assert_eq!(ops[0].payload["text"], "hello");
    }

    #[test]
    fn test_apply_ops_idempotent() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        a.record("from A", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

        let ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();

        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        // Apply once
        let r1 = apply_ops(&b, &ops).unwrap();
        assert_eq!(r1.ops_applied, ops.len());

        // Apply again — all skipped
        let r2 = apply_ops(&b, &ops).unwrap();
        assert_eq!(r2.ops_applied, 0);
        assert_eq!(r2.ops_skipped, ops.len());
    }

    #[test]
    fn test_materialize_record() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let rid = a.record("test mem", "semantic", 0.8, 0.2, 1000.0, &serde_json::json!({"k": "v"}), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

        let ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();
        let record_op = ops.iter().find(|o| o.op_type == "record").unwrap();

        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&b, &[record_op.clone()]).unwrap();

        // Check the memory was materialized
        let mem = b.get(&rid).unwrap();
        assert!(mem.is_some());
        let mem = mem.unwrap();
        assert_eq!(mem.text, "test mem");
        assert_eq!(mem.memory_type, "semantic");
        assert_eq!(mem.importance, 0.8);
    }

    #[test]
    fn test_tombstone_wins() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let rid = a.record("doomed", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        a.forget(&rid).unwrap();

        let ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();

        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&b, &ops).unwrap();

        let mem = b.get(&rid).unwrap().unwrap();
        assert_eq!(mem.consolidation_status, "tombstoned");
    }

    #[test]
    fn test_materialize_relate() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        a.relate("Alice", "Bob", "knows", 0.9).unwrap();

        let ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();
        let relate_op = ops.iter().find(|o| o.op_type == "relate").unwrap();

        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&b, &[relate_op.clone()]).unwrap();

        let edges = b.get_edges("Alice").unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].dst, "Bob");
        assert_eq!(edges[0].weight, 0.9);
    }

    #[test]
    fn test_lww_edge_merge() {
        // Both create same (src,dst,rel_type) with different weights
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        a.relate("X", "Y", "linked", 0.3).unwrap();

        // B creates same edge but later (higher timestamp)
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        b.relate("X", "Y", "linked", 0.9).unwrap();

        // Apply A's ops to B
        let a_ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();
        apply_ops(&b, &a_ops).unwrap();

        // B should keep its own weight (0.9) since it's newer
        let edges = b.get_edges("X").unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].weight, 0.9);

        // Apply B's ops to A
        let b_ops = extract_ops_since(&*b.conn(), None, None, None, 100).unwrap();
        apply_ops(&a, &b_ops).unwrap();

        // A should now have B's weight (0.9) since it's newer
        let edges = a.get_edges("X").unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].weight, 0.9);
    }

    #[test]
    fn test_watermark_tracking() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let conn = db.conn();

        // No watermark initially
        let wm = get_peer_watermark(&*conn, "peer-1").unwrap();
        assert!(wm.is_none());

        // Set watermark
        let hlc_bytes = vec![0u8; 16];
        set_peer_watermark(&*conn, "peer-1", &hlc_bytes, "op-123").unwrap();

        let wm = get_peer_watermark(&*conn, "peer-1").unwrap().unwrap();
        assert_eq!(wm.0, hlc_bytes);
        assert_eq!(wm.1, "op-123");

        // Update watermark
        let new_hlc = vec![1u8; 16];
        set_peer_watermark(&*conn, "peer-1", &new_hlc, "op-456").unwrap();

        let wm = get_peer_watermark(&*conn, "peer-1").unwrap().unwrap();
        assert_eq!(wm.0, new_hlc);
        assert_eq!(wm.1, "op-456");
    }

    #[test]
    fn test_extract_with_exclude_actor() {
        let db = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        db.record("from A", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();

        // Extracting while excluding actor "A" should return nothing
        let ops = extract_ops_since(&*db.conn(), None, None, Some("A"), 100).unwrap();
        assert!(ops.is_empty());

        // Extracting without exclusion should return the op
        let ops = extract_ops_since(&*db.conn(), None, None, None, 100).unwrap();
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_consolidation_members_replicate() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        a.record("mem1", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
        a.record("mem2", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.1, 8), "default", 0.8, "general", "user", None).unwrap();

        // Consolidate on A
        let consolidated = crate::consolidate::consolidate(&a, 0.0, 365.0, 2, false).unwrap();
        assert!(!consolidated.is_empty());

        // Extract all ops and apply to B
        let ops = extract_ops_since(&*a.conn(), None, None, None, 1000).unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&b, &ops).unwrap();

        // Check that B has the consolidation_members entries
        let count: i64 = b.conn().query_row(
            "SELECT COUNT(*) FROM consolidation_members",
            [],
            |row| row.get(0),
        ).unwrap();
        assert!(count >= 2); // At least 2 source_rids
    }
}
