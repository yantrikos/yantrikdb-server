//! Cognitive State Graph persistence layer.
//!
//! Bridges the in-memory cognitive types (`CognitiveNode`, `CognitiveEdge`,
//! `WorkingSet`, `NodeIdAllocator`) with the SQLite tables introduced in
//! schema V12 (`cognitive_nodes`, `cognitive_edges`, `cognitive_node_hwm`).
//!
//! Design goals:
//! - **Full round-trip fidelity**: every field survives persist → load with zero loss
//! - **Batch operations**: insert/update N nodes in a single transaction for performance
//! - **Incremental save**: only persist dirty nodes, not the entire graph every time
//! - **Tombstone-aware**: soft-delete support for CRDT-friendly replication
//! - **WorkingSet hydration**: load the K most relevant persistent nodes into a fresh
//!   WorkingSet, restoring the allocator high-water marks so new IDs never collide

use std::collections::HashMap;

use rusqlite::params;

use crate::attention::{AttentionConfig, WorkingSet};
use crate::error::Result;
use crate::state::{
    deserialize_payload, now_secs, serialize_payload, CognitiveAttrs, CognitiveEdge,
    CognitiveEdgeKind, CognitiveNode, NodeId, NodeIdAllocator, NodeKind, Provenance,
};

use super::YantrikDB;

/// Summary of a cognitive graph save operation.
#[derive(Debug, Clone)]
pub struct CognitiveGraphSaveResult {
    pub nodes_inserted: usize,
    pub nodes_updated: usize,
    pub edges_inserted: usize,
    pub edges_updated: usize,
    pub hwm_updated: usize,
}

/// Filter criteria for loading cognitive nodes.
#[derive(Debug, Clone, Default)]
pub struct CognitiveNodeFilter {
    /// Only load nodes of these kinds (empty = all).
    pub kinds: Vec<NodeKind>,
    /// Minimum activation threshold (0.0 = no filter).
    pub min_activation: f64,
    /// Minimum salience threshold (0.0 = no filter).
    pub min_salience: f64,
    /// Minimum urgency threshold (0.0 = no filter).
    pub min_urgency: f64,
    /// Maximum number of nodes to return.
    pub limit: usize,
    /// Whether to include tombstoned nodes.
    pub include_tombstoned: bool,
    /// Sort order: "activation", "salience", "urgency", "created_at" (desc).
    pub order_by: CognitiveNodeOrder,
}

/// Sort order for cognitive node queries.
#[derive(Debug, Clone, Default)]
pub enum CognitiveNodeOrder {
    #[default]
    Activation,
    Salience,
    Urgency,
    CreatedAt,
    Relevance,
}

impl CognitiveNodeOrder {
    fn sql_clause(&self) -> &'static str {
        match self {
            Self::Activation => "activation DESC",
            Self::Salience => "salience DESC",
            Self::Urgency => "urgency DESC",
            Self::CreatedAt => "created_at DESC",
            // Composite: activation * salience * (1 + urgency)
            Self::Relevance => "(activation * salience * (1.0 + urgency)) DESC",
        }
    }
}

impl YantrikDB {
    // ══════════════════════════════════════════════════════════════════════
    // ── Single Node CRUD ──
    // ══════════════════════════════════════════════════════════════════════

    /// Persist a single cognitive node (insert or update).
    ///
    /// Uses INSERT OR REPLACE — the node_id is the primary key, so this
    /// is an upsert. All attributes, payload, and metadata are written.
    /// Also updates the high-water mark for this node's kind.
    pub fn persist_cognitive_node(&self, node: &CognitiveNode) -> Result<()> {
        let payload_json = serialize_payload(&node.payload).to_string();
        let metadata_json = serde_json::to_string(&node.metadata).unwrap_or_else(|_| "{}".into());
        let ts = now_secs();
        let hlc_bytes = self.tick_hlc().to_bytes();

        {
            let conn = self.conn();
            conn.execute(
                "INSERT OR REPLACE INTO cognitive_nodes (
                    node_id, kind, label,
                    confidence, activation, salience, persistence, valence,
                    urgency, novelty, volatility, provenance, evidence_count,
                    last_updated_ms, payload, metadata,
                    created_at, tombstoned, hlc, origin_actor
                ) VALUES (
                    ?1, ?2, ?3,
                    ?4, ?5, ?6, ?7, ?8,
                    ?9, ?10, ?11, ?12, ?13,
                    ?14, ?15, ?16,
                    ?17, 0, ?18, ?19
                )",
                params![
                    node.id.to_raw() as i64,
                    node.kind().as_str(),
                    node.label,
                    node.attrs.confidence,
                    node.attrs.activation,
                    node.attrs.salience,
                    node.attrs.persistence,
                    node.attrs.valence,
                    node.attrs.urgency,
                    node.attrs.novelty,
                    node.attrs.volatility,
                    node.attrs.provenance.as_str(),
                    node.attrs.evidence_count,
                    node.attrs.last_updated_ms as i64,
                    payload_json,
                    metadata_json,
                    ts,
                    hlc_bytes,
                    self.actor_id,
                ],
            )?;

            // Update high-water mark
            let seq = node.id.seq();
            conn.execute(
                "INSERT INTO cognitive_node_hwm (kind, high_water_mark)
                 VALUES (?1, ?2)
                 ON CONFLICT(kind) DO UPDATE SET
                    high_water_mark = MAX(high_water_mark, excluded.high_water_mark)",
                params![node.kind().as_str(), seq],
            )?;
        }

        self.log_op(
            "cognitive_node_upsert",
            None,
            &serde_json::json!({
                "node_id": node.id.to_raw(),
                "kind": node.kind().as_str(),
                "label": node.label,
            }),
            None,
        )?;

        Ok(())
    }

    /// Load a single cognitive node by its ID. Returns None if not found or tombstoned.
    pub fn load_cognitive_node(&self, id: NodeId) -> Result<Option<CognitiveNode>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "SELECT node_id, kind, label,
                    confidence, activation, salience, persistence, valence,
                    urgency, novelty, volatility, provenance, evidence_count,
                    last_updated_ms, payload, metadata
             FROM cognitive_nodes
             WHERE node_id = ?1 AND tombstoned = 0",
        )?;

        let result = stmt.query_row(params![id.to_raw() as i64], |row| {
            Ok(row_to_cognitive_node(row))
        });

        match result {
            Ok(node) => Ok(node),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Soft-delete a cognitive node (tombstone it).
    pub fn tombstone_cognitive_node(&self, id: NodeId) -> Result<bool> {
        let hlc_bytes = self.tick_hlc().to_bytes();
        let changes = {
            let conn = self.conn();
            let changes = conn.execute(
                "UPDATE cognitive_nodes SET tombstoned = 1, hlc = ?1, origin_actor = ?2
                 WHERE node_id = ?3 AND tombstoned = 0",
                params![hlc_bytes, self.actor_id, id.to_raw() as i64],
            )?;

            if changes > 0 {
                // Also tombstone all edges involving this node
                conn.execute(
                    "UPDATE cognitive_edges SET tombstoned = 1, hlc = ?1, origin_actor = ?2
                     WHERE (src_id = ?3 OR dst_id = ?3) AND tombstoned = 0",
                    params![hlc_bytes, self.actor_id, id.to_raw() as i64],
                )?;
            }
            changes
        };

        if changes > 0 {
            self.log_op(
                "cognitive_node_tombstone",
                None,
                &serde_json::json!({
                    "node_id": id.to_raw(),
                    "kind": id.kind().as_str(),
                }),
                None,
            )?;
        }

        Ok(changes > 0)
    }

    // ══════════════════════════════════════════════════════════════════════
    // ── Batch Node Operations ──
    // ══════════════════════════════════════════════════════════════════════

    /// Persist multiple cognitive nodes in a single transaction.
    /// Returns the count of nodes written.
    pub fn persist_cognitive_nodes(&self, nodes: &[CognitiveNode]) -> Result<usize> {
        if nodes.is_empty() {
            return Ok(0);
        }

        let conn = self.conn();
        let tx = conn.unchecked_transaction()?;
        let ts = now_secs();
        let hlc_bytes = self.tick_hlc().to_bytes();

        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO cognitive_nodes (
                    node_id, kind, label,
                    confidence, activation, salience, persistence, valence,
                    urgency, novelty, volatility, provenance, evidence_count,
                    last_updated_ms, payload, metadata,
                    created_at, tombstoned, hlc, origin_actor
                ) VALUES (
                    ?1, ?2, ?3,
                    ?4, ?5, ?6, ?7, ?8,
                    ?9, ?10, ?11, ?12, ?13,
                    ?14, ?15, ?16,
                    ?17, 0, ?18, ?19
                )",
            )?;

            let mut hwm_stmt = tx.prepare_cached(
                "INSERT INTO cognitive_node_hwm (kind, high_water_mark)
                 VALUES (?1, ?2)
                 ON CONFLICT(kind) DO UPDATE SET
                    high_water_mark = MAX(high_water_mark, excluded.high_water_mark)",
            )?;

            for node in nodes {
                let payload_json = serialize_payload(&node.payload).to_string();
                let metadata_json =
                    serde_json::to_string(&node.metadata).unwrap_or_else(|_| "{}".into());

                stmt.execute(params![
                    node.id.to_raw() as i64,
                    node.kind().as_str(),
                    node.label,
                    node.attrs.confidence,
                    node.attrs.activation,
                    node.attrs.salience,
                    node.attrs.persistence,
                    node.attrs.valence,
                    node.attrs.urgency,
                    node.attrs.novelty,
                    node.attrs.volatility,
                    node.attrs.provenance.as_str(),
                    node.attrs.evidence_count,
                    node.attrs.last_updated_ms as i64,
                    payload_json,
                    metadata_json,
                    ts,
                    hlc_bytes,
                    self.actor_id,
                ])?;

                hwm_stmt.execute(params![node.kind().as_str(), node.id.seq()])?;
            }
        }

        tx.commit()?;
        drop(conn);

        self.log_op(
            "cognitive_nodes_batch_upsert",
            None,
            &serde_json::json!({ "count": nodes.len() }),
            None,
        )?;

        Ok(nodes.len())
    }

    // ══════════════════════════════════════════════════════════════════════
    // ── Node Queries ──
    // ══════════════════════════════════════════════════════════════════════

    /// Load cognitive nodes matching filter criteria.
    pub fn query_cognitive_nodes(
        &self,
        filter: &CognitiveNodeFilter,
    ) -> Result<Vec<CognitiveNode>> {
        let mut sql = String::from(
            "SELECT node_id, kind, label,
                    confidence, activation, salience, persistence, valence,
                    urgency, novelty, volatility, provenance, evidence_count,
                    last_updated_ms, payload, metadata
             FROM cognitive_nodes WHERE 1=1",
        );

        if !filter.include_tombstoned {
            sql.push_str(" AND tombstoned = 0");
        }

        if !filter.kinds.is_empty() {
            let kinds_csv: Vec<String> = filter
                .kinds
                .iter()
                .map(|k| format!("'{}'", k.as_str()))
                .collect();
            sql.push_str(&format!(" AND kind IN ({})", kinds_csv.join(",")));
        }

        if filter.min_activation > 0.0 {
            sql.push_str(&format!(" AND activation >= {}", filter.min_activation));
        }
        if filter.min_salience > 0.0 {
            sql.push_str(&format!(" AND salience >= {}", filter.min_salience));
        }
        if filter.min_urgency > 0.0 {
            sql.push_str(&format!(" AND urgency >= {}", filter.min_urgency));
        }

        sql.push_str(&format!(" ORDER BY {}", filter.order_by.sql_clause()));

        let limit = if filter.limit > 0 { filter.limit } else { 1000 };
        sql.push_str(&format!(" LIMIT {limit}"));

        let conn = self.conn();
        let mut stmt = conn.prepare(&sql)?;
        let nodes = stmt
            .query_map([], |row| Ok(row_to_cognitive_node(row)))?
            .filter_map(|r| r.ok().flatten())
            .collect();

        Ok(nodes)
    }

    /// Count cognitive nodes by kind.
    pub fn count_cognitive_nodes(&self, kind: Option<NodeKind>) -> Result<usize> {
        let conn = self.conn();
        let count: i64 = if let Some(k) = kind {
            conn.query_row(
                "SELECT COUNT(*) FROM cognitive_nodes WHERE kind = ?1 AND tombstoned = 0",
                params![k.as_str()],
                |row| row.get(0),
            )?
        } else {
            conn.query_row(
                "SELECT COUNT(*) FROM cognitive_nodes WHERE tombstoned = 0",
                [],
                |row| row.get(0),
            )?
        };
        Ok(count as usize)
    }

    /// Load all cognitive nodes of a specific kind (convenience).
    pub fn load_cognitive_nodes_by_kind(&self, kind: NodeKind) -> Result<Vec<CognitiveNode>> {
        self.query_cognitive_nodes(&CognitiveNodeFilter {
            kinds: vec![kind],
            limit: 1000,
            ..Default::default()
        })
    }

    // ══════════════════════════════════════════════════════════════════════
    // ── Edge CRUD ──
    // ══════════════════════════════════════════════════════════════════════

    /// Persist a single cognitive edge (insert or update).
    ///
    /// The primary key is (src_id, dst_id, kind). If an edge with the same
    /// triple exists, it is updated (weight, confidence, observation_count).
    pub fn persist_cognitive_edge(&self, edge: &CognitiveEdge) -> Result<()> {
        let hlc_bytes = self.tick_hlc().to_bytes();

        self.conn().execute(
            "INSERT INTO cognitive_edges (
                src_id, dst_id, kind, weight, confidence,
                observation_count, created_at_ms, last_confirmed_ms,
                tombstoned, hlc, origin_actor
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9, ?10)
            ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                weight = excluded.weight,
                confidence = excluded.confidence,
                observation_count = excluded.observation_count,
                last_confirmed_ms = excluded.last_confirmed_ms,
                hlc = excluded.hlc,
                origin_actor = excluded.origin_actor",
            params![
                edge.src.to_raw() as i64,
                edge.dst.to_raw() as i64,
                edge.kind.as_str(),
                edge.weight,
                edge.confidence,
                edge.observation_count,
                edge.created_at_ms as i64,
                edge.last_confirmed_ms as i64,
                hlc_bytes,
                self.actor_id,
            ],
        )?;

        Ok(())
    }

    /// Persist multiple cognitive edges in a single transaction.
    pub fn persist_cognitive_edges(&self, edges: &[CognitiveEdge]) -> Result<usize> {
        if edges.is_empty() {
            return Ok(0);
        }

        let conn = self.conn();
        let tx = conn.unchecked_transaction()?;
        let hlc_bytes = self.tick_hlc().to_bytes();

        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO cognitive_edges (
                    src_id, dst_id, kind, weight, confidence,
                    observation_count, created_at_ms, last_confirmed_ms,
                    tombstoned, hlc, origin_actor
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9, ?10)
                ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                    weight = excluded.weight,
                    confidence = excluded.confidence,
                    observation_count = excluded.observation_count,
                    last_confirmed_ms = excluded.last_confirmed_ms,
                    hlc = excluded.hlc,
                    origin_actor = excluded.origin_actor",
            )?;

            for edge in edges {
                stmt.execute(params![
                    edge.src.to_raw() as i64,
                    edge.dst.to_raw() as i64,
                    edge.kind.as_str(),
                    edge.weight,
                    edge.confidence,
                    edge.observation_count,
                    edge.created_at_ms as i64,
                    edge.last_confirmed_ms as i64,
                    hlc_bytes,
                    self.actor_id,
                ])?;
            }
        }

        tx.commit()?;
        Ok(edges.len())
    }

    /// Load all edges originating from a node.
    pub fn load_cognitive_edges_from(&self, src: NodeId) -> Result<Vec<CognitiveEdge>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "SELECT src_id, dst_id, kind, weight, confidence,
                    observation_count, created_at_ms, last_confirmed_ms
             FROM cognitive_edges
             WHERE src_id = ?1 AND tombstoned = 0",
        )?;

        let edges = stmt
            .query_map(params![src.to_raw() as i64], |row| {
                Ok(row_to_cognitive_edge(row))
            })?
            .filter_map(|r| r.ok().flatten())
            .collect();

        Ok(edges)
    }

    /// Load all edges pointing to a node.
    pub fn load_cognitive_edges_to(&self, dst: NodeId) -> Result<Vec<CognitiveEdge>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "SELECT src_id, dst_id, kind, weight, confidence,
                    observation_count, created_at_ms, last_confirmed_ms
             FROM cognitive_edges
             WHERE dst_id = ?1 AND tombstoned = 0",
        )?;

        let edges = stmt
            .query_map(params![dst.to_raw() as i64], |row| {
                Ok(row_to_cognitive_edge(row))
            })?
            .filter_map(|r| r.ok().flatten())
            .collect();

        Ok(edges)
    }

    /// Load all edges involving a node (both directions).
    pub fn load_cognitive_edges_for(&self, node_id: NodeId) -> Result<Vec<CognitiveEdge>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "SELECT src_id, dst_id, kind, weight, confidence,
                    observation_count, created_at_ms, last_confirmed_ms
             FROM cognitive_edges
             WHERE (src_id = ?1 OR dst_id = ?1) AND tombstoned = 0",
        )?;

        let edges = stmt
            .query_map(params![node_id.to_raw() as i64], |row| {
                Ok(row_to_cognitive_edge(row))
            })?
            .filter_map(|r| r.ok().flatten())
            .collect();

        Ok(edges)
    }

    /// Load edges of a specific kind.
    pub fn load_cognitive_edges_by_kind(
        &self,
        kind: CognitiveEdgeKind,
    ) -> Result<Vec<CognitiveEdge>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "SELECT src_id, dst_id, kind, weight, confidence,
                    observation_count, created_at_ms, last_confirmed_ms
             FROM cognitive_edges
             WHERE kind = ?1 AND tombstoned = 0",
        )?;

        let edges = stmt
            .query_map(params![kind.as_str()], |row| {
                Ok(row_to_cognitive_edge(row))
            })?
            .filter_map(|r| r.ok().flatten())
            .collect();

        Ok(edges)
    }

    /// Tombstone a specific edge.
    pub fn tombstone_cognitive_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        kind: CognitiveEdgeKind,
    ) -> Result<bool> {
        let hlc_bytes = self.tick_hlc().to_bytes();
        let changes = self.conn().execute(
            "UPDATE cognitive_edges SET tombstoned = 1, hlc = ?1, origin_actor = ?2
             WHERE src_id = ?3 AND dst_id = ?4 AND kind = ?5 AND tombstoned = 0",
            params![
                hlc_bytes,
                self.actor_id,
                src.to_raw() as i64,
                dst.to_raw() as i64,
                kind.as_str(),
            ],
        )?;
        Ok(changes > 0)
    }

    /// Count cognitive edges (optionally filtered by kind).
    pub fn count_cognitive_edges(&self, kind: Option<CognitiveEdgeKind>) -> Result<usize> {
        let conn = self.conn();
        let count: i64 = if let Some(k) = kind {
            conn.query_row(
                "SELECT COUNT(*) FROM cognitive_edges WHERE kind = ?1 AND tombstoned = 0",
                params![k.as_str()],
                |row| row.get(0),
            )?
        } else {
            conn.query_row(
                "SELECT COUNT(*) FROM cognitive_edges WHERE tombstoned = 0",
                [],
                |row| row.get(0),
            )?
        };
        Ok(count as usize)
    }

    // ══════════════════════════════════════════════════════════════════════
    // ── NodeIdAllocator Persistence ──
    // ══════════════════════════════════════════════════════════════════════

    /// Load the NodeIdAllocator from persisted high-water marks.
    ///
    /// Scans the `cognitive_node_hwm` table and reconstructs the allocator
    /// so that newly allocated IDs never collide with persisted ones.
    pub fn load_node_id_allocator(&self) -> Result<NodeIdAllocator> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "SELECT kind, high_water_mark FROM cognitive_node_hwm",
        )?;

        let marks: Vec<(NodeKind, u32)> = stmt
            .query_map([], |row| {
                let kind_str: String = row.get(0)?;
                let hwm: i64 = row.get(1)?;
                Ok((kind_str, hwm as u32))
            })?
            .filter_map(|r| {
                r.ok().and_then(|(k, h)| {
                    NodeKind::from_str(&k).map(|kind| (kind, h))
                })
            })
            .collect();

        Ok(NodeIdAllocator::from_high_water_marks(&marks))
    }

    /// Persist the allocator's high-water marks to SQLite.
    pub fn persist_node_id_allocator(&self, allocator: &NodeIdAllocator) -> Result<()> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "INSERT INTO cognitive_node_hwm (kind, high_water_mark)
             VALUES (?1, ?2)
             ON CONFLICT(kind) DO UPDATE SET
                high_water_mark = MAX(high_water_mark, excluded.high_water_mark)",
        )?;

        for kind in NodeKind::ALL {
            let hwm = allocator.high_water_mark(kind);
            if hwm > 0 {
                stmt.execute(params![kind.as_str(), hwm])?;
            }
        }

        Ok(())
    }

    // ══════════════════════════════════════════════════════════════════════
    // ── WorkingSet Integration ──
    // ══════════════════════════════════════════════════════════════════════

    /// Hydrate a WorkingSet from persisted state.
    ///
    /// Loads the top-N most relevant nodes (by activation * salience * urgency),
    /// all edges between them, and restores the NodeIdAllocator. The resulting
    /// WorkingSet is ready for immediate reasoning.
    ///
    /// This is the primary entry point for restoring cognitive state on startup.
    pub fn hydrate_working_set(&self, config: AttentionConfig) -> Result<WorkingSet> {
        let allocator = self.load_node_id_allocator()?;
        let mut ws = WorkingSet::with_allocator(config.clone(), allocator);

        // Load top-N nodes by composite relevance score
        let nodes = self.query_cognitive_nodes(&CognitiveNodeFilter {
            limit: config.capacity,
            order_by: CognitiveNodeOrder::Relevance,
            ..Default::default()
        })?;

        if nodes.is_empty() {
            return Ok(ws);
        }

        // Build set of loaded node IDs for edge filtering
        let node_ids: std::collections::HashSet<u32> =
            nodes.iter().map(|n| n.id.to_raw()).collect();

        // Insert nodes into working set (without triggering eviction logic —
        // we're below capacity since we limited the query)
        for node in nodes {
            ws.insert(node);
        }

        // Load edges between the loaded nodes
        // We query all non-tombstoned edges and filter in-memory to those
        // where both endpoints are in the working set
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT src_id, dst_id, kind, weight, confidence,
                    observation_count, created_at_ms, last_confirmed_ms
             FROM cognitive_edges WHERE tombstoned = 0",
        )?;

        let edges: Vec<CognitiveEdge> = stmt
            .query_map([], |row| Ok(row_to_cognitive_edge(row)))?
            .filter_map(|r| r.ok().flatten())
            .filter(|e| {
                node_ids.contains(&e.src.to_raw()) && node_ids.contains(&e.dst.to_raw())
            })
            .collect();

        for edge in edges {
            ws.add_edge(edge);
        }

        Ok(ws)
    }

    /// Save a WorkingSet's persistent nodes and edges back to SQLite.
    ///
    /// Only saves nodes whose `is_persistent()` returns true (i.e., not
    /// IntentHypothesis or ConversationThread). Also persists edges between
    /// persistent nodes and the allocator high-water marks.
    ///
    /// This is the primary entry point for saving cognitive state on shutdown
    /// or at periodic checkpoints.
    pub fn save_working_set(&self, ws: &WorkingSet) -> Result<CognitiveGraphSaveResult> {
        // Collect persistent nodes
        let persistent_nodes: Vec<&CognitiveNode> =
            ws.iter().filter(|n| n.is_persistent()).collect();

        let persistent_ids: std::collections::HashSet<u32> =
            persistent_nodes.iter().map(|n| n.id.to_raw()).collect();

        // Collect edges between persistent nodes
        let mut persistent_edges: Vec<CognitiveEdge> = Vec::new();
        for node in &persistent_nodes {
            for (_, edge) in ws.edges_from(node.id) {
                if persistent_ids.contains(&edge.dst.to_raw()) {
                    persistent_edges.push(edge.clone());
                }
            }
        }

        // Save everything in a transaction
        let conn = self.conn();
        let tx = conn.unchecked_transaction()?;
        let ts = now_secs();
        let hlc_bytes = self.tick_hlc().to_bytes();

        // Persist nodes
        let mut node_count = 0;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO cognitive_nodes (
                    node_id, kind, label,
                    confidence, activation, salience, persistence, valence,
                    urgency, novelty, volatility, provenance, evidence_count,
                    last_updated_ms, payload, metadata,
                    created_at, tombstoned, hlc, origin_actor
                ) VALUES (
                    ?1, ?2, ?3,
                    ?4, ?5, ?6, ?7, ?8,
                    ?9, ?10, ?11, ?12, ?13,
                    ?14, ?15, ?16,
                    ?17, 0, ?18, ?19
                )",
            )?;

            let mut hwm_stmt = tx.prepare_cached(
                "INSERT INTO cognitive_node_hwm (kind, high_water_mark)
                 VALUES (?1, ?2)
                 ON CONFLICT(kind) DO UPDATE SET
                    high_water_mark = MAX(high_water_mark, excluded.high_water_mark)",
            )?;

            for node in &persistent_nodes {
                let payload_json = serialize_payload(&node.payload).to_string();
                let metadata_json =
                    serde_json::to_string(&node.metadata).unwrap_or_else(|_| "{}".into());

                stmt.execute(params![
                    node.id.to_raw() as i64,
                    node.kind().as_str(),
                    node.label,
                    node.attrs.confidence,
                    node.attrs.activation,
                    node.attrs.salience,
                    node.attrs.persistence,
                    node.attrs.valence,
                    node.attrs.urgency,
                    node.attrs.novelty,
                    node.attrs.volatility,
                    node.attrs.provenance.as_str(),
                    node.attrs.evidence_count,
                    node.attrs.last_updated_ms as i64,
                    payload_json,
                    metadata_json,
                    ts,
                    hlc_bytes,
                    self.actor_id,
                ])?;

                hwm_stmt.execute(params![node.kind().as_str(), node.id.seq()])?;
                node_count += 1;
            }
        }

        // Persist edges
        let mut edge_count = 0;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO cognitive_edges (
                    src_id, dst_id, kind, weight, confidence,
                    observation_count, created_at_ms, last_confirmed_ms,
                    tombstoned, hlc, origin_actor
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9, ?10)
                ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                    weight = excluded.weight,
                    confidence = excluded.confidence,
                    observation_count = excluded.observation_count,
                    last_confirmed_ms = excluded.last_confirmed_ms,
                    hlc = excluded.hlc,
                    origin_actor = excluded.origin_actor",
            )?;

            for edge in &persistent_edges {
                stmt.execute(params![
                    edge.src.to_raw() as i64,
                    edge.dst.to_raw() as i64,
                    edge.kind.as_str(),
                    edge.weight,
                    edge.confidence,
                    edge.observation_count,
                    edge.created_at_ms as i64,
                    edge.last_confirmed_ms as i64,
                    hlc_bytes,
                    self.actor_id,
                ])?;
                edge_count += 1;
            }
        }

        // Persist allocator HWM
        let mut hwm_count = 0;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO cognitive_node_hwm (kind, high_water_mark)
                 VALUES (?1, ?2)
                 ON CONFLICT(kind) DO UPDATE SET
                    high_water_mark = MAX(high_water_mark, excluded.high_water_mark)",
            )?;

            for kind in NodeKind::ALL {
                let hwm = ws.allocator().high_water_mark(kind);
                if hwm > 0 {
                    stmt.execute(params![kind.as_str(), hwm])?;
                    hwm_count += 1;
                }
            }
        }

        tx.commit()?;

        self.log_op(
            "cognitive_graph_save",
            None,
            &serde_json::json!({
                "nodes": node_count,
                "edges": edge_count,
                "hwm_kinds": hwm_count,
            }),
            None,
        )?;

        Ok(CognitiveGraphSaveResult {
            nodes_inserted: node_count,
            nodes_updated: 0, // We can't distinguish insert vs update with REPLACE
            edges_inserted: edge_count,
            edges_updated: 0,
            hwm_updated: hwm_count,
        })
    }

    // ══════════════════════════════════════════════════════════════════════
    // ── Graph Traversal Queries ──
    // ══════════════════════════════════════════════════════════════════════

    /// Get the immediate neighbors of a node (nodes connected by outgoing edges).
    pub fn cognitive_neighbors(&self, id: NodeId) -> Result<Vec<(CognitiveNode, CognitiveEdge)>> {
        let edges = self.load_cognitive_edges_from(id)?;
        let mut results = Vec::with_capacity(edges.len());
        for edge in edges {
            if let Some(node) = self.load_cognitive_node(edge.dst)? {
                results.push((node, edge));
            }
        }
        Ok(results)
    }

    /// Get all nodes that have edges pointing to this node (reverse neighbors).
    pub fn cognitive_predecessors(
        &self,
        id: NodeId,
    ) -> Result<Vec<(CognitiveNode, CognitiveEdge)>> {
        let edges = self.load_cognitive_edges_to(id)?;
        let mut results = Vec::with_capacity(edges.len());
        for edge in edges {
            if let Some(node) = self.load_cognitive_node(edge.src)? {
                results.push((node, edge));
            }
        }
        Ok(results)
    }

    /// Find nodes connected by a specific edge kind from a source.
    pub fn cognitive_neighbors_by_edge_kind(
        &self,
        id: NodeId,
        edge_kind: CognitiveEdgeKind,
    ) -> Result<Vec<(CognitiveNode, CognitiveEdge)>> {
        let edges = self.load_cognitive_edges_from(id)?;
        let mut results = Vec::new();
        for edge in edges {
            if edge.kind == edge_kind {
                if let Some(node) = self.load_cognitive_node(edge.dst)? {
                    results.push((node, edge));
                }
            }
        }
        Ok(results)
    }

    /// Find all beliefs that support or contradict a given node.
    pub fn cognitive_evidence_for(
        &self,
        id: NodeId,
    ) -> Result<(Vec<(CognitiveNode, CognitiveEdge)>, Vec<(CognitiveNode, CognitiveEdge)>)> {
        let incoming = self.load_cognitive_edges_to(id)?;
        let mut supporting = Vec::new();
        let mut contradicting = Vec::new();

        for edge in incoming {
            if let Some(node) = self.load_cognitive_node(edge.src)? {
                match edge.kind {
                    CognitiveEdgeKind::Supports => supporting.push((node, edge)),
                    CognitiveEdgeKind::Contradicts => contradicting.push((node, edge)),
                    _ => {}
                }
            }
        }

        Ok((supporting, contradicting))
    }

    /// Get the subgraph rooted at a goal: all tasks, subtasks, and blockers.
    pub fn cognitive_goal_subgraph(
        &self,
        goal_id: NodeId,
    ) -> Result<(Vec<CognitiveNode>, Vec<CognitiveEdge>)> {
        let mut visited = std::collections::HashSet::new();
        let mut edge_set: std::collections::HashSet<(u32, u32, &str)> = std::collections::HashSet::new();
        let mut queue = vec![goal_id];
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        while let Some(current) = queue.pop() {
            if !visited.insert(current.to_raw()) {
                continue;
            }

            if let Some(node) = self.load_cognitive_node(current)? {
                nodes.push(node);
            }

            let outgoing = self.load_cognitive_edges_from(current)?;
            for edge in outgoing {
                match edge.kind {
                    CognitiveEdgeKind::SubtaskOf
                    | CognitiveEdgeKind::Requires
                    | CognitiveEdgeKind::AdvancesGoal
                    | CognitiveEdgeKind::BlocksGoal => {
                        queue.push(edge.dst);
                        let key = (edge.src.to_raw(), edge.dst.to_raw(), edge.kind.as_str());
                        if edge_set.insert(key) {
                            edges.push(edge);
                        }
                    }
                    _ => {}
                }
            }

            // Also follow reverse SubtaskOf edges (tasks that are subtasks OF this)
            let incoming = self.load_cognitive_edges_to(current)?;
            for edge in incoming {
                if edge.kind == CognitiveEdgeKind::SubtaskOf {
                    queue.push(edge.src);
                    let key = (edge.src.to_raw(), edge.dst.to_raw(), edge.kind.as_str());
                    if edge_set.insert(key) {
                        edges.push(edge);
                    }
                }
            }
        }

        Ok((nodes, edges))
    }

    // ══════════════════════════════════════════════════════════════════════
    // ── Cognitive Graph Statistics ──
    // ══════════════════════════════════════════════════════════════════════

    /// Get summary statistics of the cognitive state graph.
    pub fn cognitive_graph_stats(&self) -> Result<CognitiveGraphStats> {
        let total_nodes = self.count_cognitive_nodes(None)?;
        let total_edges = self.count_cognitive_edges(None)?;

        let mut kind_counts = HashMap::new();
        for kind in NodeKind::ALL {
            let count = self.count_cognitive_nodes(Some(kind))?;
            if count > 0 {
                kind_counts.insert(kind.as_str().to_string(), count);
            }
        }

        let mut edge_kind_counts = HashMap::new();
        for kind in CognitiveEdgeKind::ALL {
            let count = self.count_cognitive_edges(Some(kind))?;
            if count > 0 {
                edge_kind_counts.insert(kind.as_str().to_string(), count);
            }
        }

        // Average activation across all nodes
        let conn = self.conn();
        let avg_activation: f64 = conn
            .query_row(
                "SELECT COALESCE(AVG(activation), 0.0) FROM cognitive_nodes WHERE tombstoned = 0",
                [],
                |row| row.get(0),
            )?;

        // Highest urgency node
        let max_urgency: f64 = conn
            .query_row(
                "SELECT COALESCE(MAX(urgency), 0.0) FROM cognitive_nodes WHERE tombstoned = 0",
                [],
                |row| row.get(0),
            )?;

        Ok(CognitiveGraphStats {
            total_nodes,
            total_edges,
            node_kind_counts: kind_counts,
            edge_kind_counts,
            avg_activation,
            max_urgency,
        })
    }
}

/// Summary statistics for the cognitive state graph.
#[derive(Debug, Clone)]
pub struct CognitiveGraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub node_kind_counts: HashMap<String, usize>,
    pub edge_kind_counts: HashMap<String, usize>,
    pub avg_activation: f64,
    pub max_urgency: f64,
}

// ══════════════════════════════════════════════════════════════════════
// ── Row Mapping Helpers ──
// ══════════════════════════════════════════════════════════════════════

/// Convert a SQLite row into a CognitiveNode.
/// Returns None if the payload cannot be deserialized (corrupt data).
fn row_to_cognitive_node(row: &rusqlite::Row<'_>) -> Option<CognitiveNode> {
    let raw_id: i64 = row.get(0).ok()?;
    let kind_str: String = row.get(1).ok()?;
    let label: String = row.get(2).ok()?;

    let kind = NodeKind::from_str(&kind_str)?;
    let id = NodeId::from_raw(raw_id as u32);

    let attrs = CognitiveAttrs {
        confidence: row.get(3).unwrap_or(0.5),
        activation: row.get(4).unwrap_or(0.0),
        salience: row.get(5).unwrap_or(0.5),
        persistence: row.get(6).unwrap_or(0.5),
        valence: row.get(7).unwrap_or(0.0),
        urgency: row.get(8).unwrap_or(0.0),
        novelty: row.get(9).unwrap_or(1.0),
        volatility: row.get(10).unwrap_or(0.1),
        provenance: {
            let p: String = row.get(11).unwrap_or_else(|_| "observed".to_string());
            Provenance::from_str(&p)
        },
        evidence_count: row.get::<_, i64>(12).unwrap_or(1) as u32,
        last_updated_ms: row.get::<_, i64>(13).unwrap_or(0) as u64,
    };

    let payload_str: String = row.get(14).unwrap_or_else(|_| "{}".to_string());
    let payload_json: serde_json::Value =
        serde_json::from_str(&payload_str).unwrap_or(serde_json::json!({}));
    let payload = deserialize_payload(kind, &payload_json)?;

    let metadata_str: String = row.get(15).unwrap_or_else(|_| "{}".to_string());
    let metadata: HashMap<String, serde_json::Value> =
        serde_json::from_str(&metadata_str).unwrap_or_default();

    Some(CognitiveNode {
        id,
        attrs,
        payload,
        label,
        metadata,
    })
}

/// Convert a SQLite row into a CognitiveEdge.
/// Returns None if the edge kind is unrecognized.
fn row_to_cognitive_edge(row: &rusqlite::Row<'_>) -> Option<CognitiveEdge> {
    let src_raw: i64 = row.get(0).ok()?;
    let dst_raw: i64 = row.get(1).ok()?;
    let kind_str: String = row.get(2).ok()?;

    let kind = CognitiveEdgeKind::from_str(&kind_str)?;

    Some(CognitiveEdge {
        src: NodeId::from_raw(src_raw as u32),
        dst: NodeId::from_raw(dst_raw as u32),
        kind,
        weight: row.get(3).unwrap_or(0.5),
        confidence: row.get(4).unwrap_or(0.5),
        observation_count: row.get::<_, i64>(5).unwrap_or(1) as u32,
        created_at_ms: row.get::<_, i64>(6).unwrap_or(0) as u64,
        last_confirmed_ms: row.get::<_, i64>(7).unwrap_or(0) as u64,
    })
}

// ══════════════════════════════════════════════════════════════════════
// ── Tests ──
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{
        EntityPayload, BeliefPayload, GoalPayload, GoalStatus, TaskPayload, TaskStatus,
        Priority, IntentPayload, NodePayload,
    };

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 4).unwrap()
    }

    fn make_entity(name: &str) -> NodePayload {
        NodePayload::Entity(EntityPayload {
            name: name.into(),
            entity_type: "person".into(),
            memory_rids: vec![],
        })
    }

    fn make_belief(prop: &str) -> NodePayload {
        NodePayload::Belief(BeliefPayload {
            proposition: prop.into(),
            log_odds: 1.0,
            domain: "test".into(),
            evidence_trail: vec![],
            user_confirmed: false,
        })
    }

    fn make_goal(desc: &str) -> NodePayload {
        NodePayload::Goal(GoalPayload {
            description: desc.into(),
            status: GoalStatus::Active,
            progress: 0.0,
            deadline: None,
            priority: Priority::Medium,
            parent_goal: None,
            completion_criteria: String::new(),
        })
    }

    fn make_task(desc: &str) -> NodePayload {
        NodePayload::Task(TaskPayload {
            description: desc.into(),
            status: TaskStatus::Pending,
            goal_id: None,
            deadline: None,
            priority: Priority::Medium,
            estimated_minutes: Some(60),
            prerequisites: vec![],
        })
    }

    #[test]
    fn test_persist_and_load_node() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let id = alloc.alloc(NodeKind::Entity);
        let node = CognitiveNode::new(
            id,
            "Alice".into(),
            NodePayload::Entity(EntityPayload {
                name: "alice_smith".into(),
                entity_type: "person".into(),
                memory_rids: vec!["rid1".into(), "rid2".into()],
            }),
        );

        db.persist_cognitive_node(&node).unwrap();

        let loaded = db.load_cognitive_node(id).unwrap().expect("node not found");
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.label, "Alice");
        assert_eq!(loaded.kind(), NodeKind::Entity);

        if let NodePayload::Entity(p) = &loaded.payload {
            assert_eq!(p.entity_type, "person");
            assert_eq!(p.name, "alice_smith");
            assert_eq!(p.memory_rids.len(), 2);
        } else {
            panic!("wrong payload type");
        }
    }

    #[test]
    fn test_persist_and_load_edge() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let n1 = alloc.alloc(NodeKind::Belief);
        let n2 = alloc.alloc(NodeKind::Belief);

        db.persist_cognitive_node(&CognitiveNode::new(n1, "Earth is round".into(), make_belief("Earth is round"))).unwrap();
        db.persist_cognitive_node(&CognitiveNode::new(n2, "Gravity exists".into(), make_belief("Gravity exists"))).unwrap();

        let edge = CognitiveEdge::new(n1, n2, CognitiveEdgeKind::Supports, 0.8);
        db.persist_cognitive_edge(&edge).unwrap();

        let edges_from = db.load_cognitive_edges_from(n1).unwrap();
        assert_eq!(edges_from.len(), 1);
        assert_eq!(edges_from[0].dst, n2);
        assert_eq!(edges_from[0].kind, CognitiveEdgeKind::Supports);
        assert!((edges_from[0].weight - 0.8).abs() < 0.001);

        let edges_to = db.load_cognitive_edges_to(n2).unwrap();
        assert_eq!(edges_to.len(), 1);
        assert_eq!(edges_to[0].src, n1);
    }

    #[test]
    fn test_tombstone_node_cascades_edges() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let n1 = alloc.alloc(NodeKind::Entity);
        let n2 = alloc.alloc(NodeKind::Entity);

        db.persist_cognitive_node(&CognitiveNode::new(n1, "Alice".into(), make_entity("alice"))).unwrap();
        db.persist_cognitive_node(&CognitiveNode::new(n2, "Bob".into(), make_entity("bob"))).unwrap();

        let edge = CognitiveEdge::new(n1, n2, CognitiveEdgeKind::AssociatedWith, 0.5);
        db.persist_cognitive_edge(&edge).unwrap();

        assert!(db.tombstone_cognitive_node(n1).unwrap());
        assert!(db.load_cognitive_node(n1).unwrap().is_none());
        assert_eq!(db.load_cognitive_edges_from(n1).unwrap().len(), 0);
        assert_eq!(db.load_cognitive_edges_to(n2).unwrap().len(), 0);
    }

    #[test]
    fn test_batch_persist_nodes() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let nodes: Vec<CognitiveNode> = (0..10)
            .map(|i| {
                let id = alloc.alloc(NodeKind::Entity);
                CognitiveNode::new(id, format!("Entity_{i}"), make_entity(&format!("entity_{i}")))
            })
            .collect();

        let count = db.persist_cognitive_nodes(&nodes).unwrap();
        assert_eq!(count, 10);
        assert_eq!(db.count_cognitive_nodes(None).unwrap(), 10);
        assert_eq!(db.count_cognitive_nodes(Some(NodeKind::Entity)).unwrap(), 10);
    }

    #[test]
    fn test_node_id_allocator_persistence() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        alloc.alloc(NodeKind::Entity);
        alloc.alloc(NodeKind::Entity);
        alloc.alloc(NodeKind::Belief);
        alloc.alloc(NodeKind::Goal);
        alloc.alloc(NodeKind::Goal);
        alloc.alloc(NodeKind::Goal);

        db.persist_node_id_allocator(&alloc).unwrap();

        let restored = db.load_node_id_allocator().unwrap();
        assert_eq!(restored.high_water_mark(NodeKind::Entity), 2);
        assert_eq!(restored.high_water_mark(NodeKind::Belief), 1);
        assert_eq!(restored.high_water_mark(NodeKind::Goal), 3);

        let mut restored = restored;
        assert_eq!(restored.alloc(NodeKind::Entity).seq(), 3);
        assert_eq!(restored.alloc(NodeKind::Goal).seq(), 4);
    }

    #[test]
    fn test_query_with_filters() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let mut nodes = Vec::new();
        for i in 0..5u32 {
            let id = alloc.alloc(NodeKind::Goal);
            let mut node = CognitiveNode::new(id, format!("Goal_{i}"), make_goal(&format!("Do thing {i}")));
            node.attrs.activation = i as f64 * 0.2;
            node.attrs.urgency = (4 - i) as f64 * 0.2;
            nodes.push(node);
        }
        db.persist_cognitive_nodes(&nodes).unwrap();

        // activation >= 0.3 → i=2(0.4), i=3(0.6), i=4(0.8)
        let high_activation = db.query_cognitive_nodes(&CognitiveNodeFilter {
            min_activation: 0.3,
            order_by: CognitiveNodeOrder::Activation,
            limit: 10,
            ..Default::default()
        }).unwrap();
        assert_eq!(high_activation.len(), 3);

        // urgency >= 0.5 → i=0(0.8), i=1(0.6)
        let urgent = db.query_cognitive_nodes(&CognitiveNodeFilter {
            min_urgency: 0.5,
            order_by: CognitiveNodeOrder::Urgency,
            limit: 10,
            ..Default::default()
        }).unwrap();
        assert_eq!(urgent.len(), 2);
    }

    #[test]
    fn test_hydrate_and_save_working_set() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let entity_id = alloc.alloc(NodeKind::Entity);
        let belief_id = alloc.alloc(NodeKind::Belief);
        let goal_id = alloc.alloc(NodeKind::Goal);

        let entity = CognitiveNode::new(entity_id, "Coffee".into(), make_entity("coffee"));
        let belief = CognitiveNode::new(belief_id, "Coffee helps focus".into(), make_belief("Coffee helps focus"));
        let goal = CognitiveNode::new(goal_id, "Improve focus".into(), make_goal("Improve focus during work hours"));

        db.persist_cognitive_nodes(&[entity, belief, goal]).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let edges = vec![
            CognitiveEdge::new(entity_id, belief_id, CognitiveEdgeKind::Supports, 0.7),
            CognitiveEdge::new(belief_id, goal_id, CognitiveEdgeKind::AdvancesGoal, 0.6),
        ];
        db.persist_cognitive_edges(&edges).unwrap();

        // Hydrate
        let ws = db.hydrate_working_set(AttentionConfig::default()).unwrap();
        assert_eq!(ws.len(), 3);

        // Verify allocator was restored
        let mut ws = ws;
        let new_id = ws.allocator_mut().alloc(NodeKind::Entity);
        assert_eq!(new_id.seq(), 2);

        ws.insert(CognitiveNode::new(new_id, "Tea".into(), make_entity("tea")));
        assert_eq!(ws.len(), 4);

        // Save back
        let result = db.save_working_set(&ws).unwrap();
        assert_eq!(result.nodes_inserted, 4);
        assert_eq!(result.edges_inserted, 2);

        let loaded = db.load_cognitive_node(new_id).unwrap().expect("new node not found");
        assert_eq!(loaded.label, "Tea");
    }

    #[test]
    fn test_cognitive_graph_stats() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let n1 = alloc.alloc(NodeKind::Entity);
        let n2 = alloc.alloc(NodeKind::Belief);
        let n3 = alloc.alloc(NodeKind::Goal);

        let mut node1 = CognitiveNode::new(n1, "A".into(), make_entity("a"));
        node1.attrs.activation = 0.6;
        node1.attrs.urgency = 0.1;

        let mut node2 = CognitiveNode::new(n2, "B".into(), make_belief("b"));
        node2.attrs.activation = 0.4;
        node2.attrs.urgency = 0.8;

        let mut node3 = CognitiveNode::new(n3, "C".into(), make_goal("c"));
        node3.attrs.activation = 0.2;
        node3.attrs.urgency = 0.5;

        db.persist_cognitive_nodes(&[node1, node2, node3]).unwrap();
        db.persist_cognitive_edge(&CognitiveEdge::new(n1, n2, CognitiveEdgeKind::Supports, 0.7)).unwrap();

        let stats = db.cognitive_graph_stats().unwrap();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.total_edges, 1);
        assert!((stats.avg_activation - 0.4).abs() < 0.01);
        assert!((stats.max_urgency - 0.8).abs() < 0.01);
        assert_eq!(*stats.node_kind_counts.get("entity").unwrap(), 1);
        assert_eq!(*stats.node_kind_counts.get("belief").unwrap(), 1);
        assert_eq!(*stats.node_kind_counts.get("goal").unwrap(), 1);
    }

    #[test]
    fn test_cognitive_neighbors_and_predecessors() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let n1 = alloc.alloc(NodeKind::Entity);
        let n2 = alloc.alloc(NodeKind::Entity);
        let n3 = alloc.alloc(NodeKind::Entity);

        for (id, name) in [(n1, "A"), (n2, "B"), (n3, "C")] {
            db.persist_cognitive_node(&CognitiveNode::new(id, name.into(), make_entity(&name.to_lowercase()))).unwrap();
        }

        db.persist_cognitive_edge(&CognitiveEdge::new(n1, n2, CognitiveEdgeKind::Causes, 0.8)).unwrap();
        db.persist_cognitive_edge(&CognitiveEdge::new(n1, n3, CognitiveEdgeKind::Supports, 0.6)).unwrap();

        assert_eq!(db.cognitive_neighbors(n1).unwrap().len(), 2);

        let preds = db.cognitive_predecessors(n2).unwrap();
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].0.label, "A");

        let causes_only = db.cognitive_neighbors_by_edge_kind(n1, CognitiveEdgeKind::Causes).unwrap();
        assert_eq!(causes_only.len(), 1);
        assert_eq!(causes_only[0].0.label, "B");
    }

    #[test]
    fn test_evidence_for_belief() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let target = alloc.alloc(NodeKind::Belief);
        let supporter = alloc.alloc(NodeKind::Belief);
        let contraditor = alloc.alloc(NodeKind::Belief);

        for (id, label, prop) in [
            (target, "Target belief", "X is true"),
            (supporter, "Supporting evidence", "Y implies X"),
            (contraditor, "Counter evidence", "Z contradicts X"),
        ] {
            db.persist_cognitive_node(&CognitiveNode::new(id, label.into(), make_belief(prop))).unwrap();
        }

        db.persist_cognitive_edge(&CognitiveEdge::new(supporter, target, CognitiveEdgeKind::Supports, 0.8)).unwrap();
        db.persist_cognitive_edge(&CognitiveEdge::new(contraditor, target, CognitiveEdgeKind::Contradicts, -0.7)).unwrap();

        let (supporting, contradicting) = db.cognitive_evidence_for(target).unwrap();
        assert_eq!(supporting.len(), 1);
        assert_eq!(contradicting.len(), 1);
        assert_eq!(supporting[0].0.label, "Supporting evidence");
        assert_eq!(contradicting[0].0.label, "Counter evidence");
    }

    #[test]
    fn test_goal_subgraph() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let goal = alloc.alloc(NodeKind::Goal);
        let task1 = alloc.alloc(NodeKind::Task);
        let task2 = alloc.alloc(NodeKind::Task);

        db.persist_cognitive_node(&CognitiveNode::new(goal, "Ship v1".into(), make_goal("Ship v1.0 release"))).unwrap();

        for (id, label) in [(task1, "Write tests"), (task2, "Deploy")] {
            db.persist_cognitive_node(&CognitiveNode::new(id, label.into(), make_task(label))).unwrap();
        }

        db.persist_cognitive_edge(&CognitiveEdge::new(task1, goal, CognitiveEdgeKind::SubtaskOf, 0.9)).unwrap();
        db.persist_cognitive_edge(&CognitiveEdge::new(task2, goal, CognitiveEdgeKind::SubtaskOf, 0.9)).unwrap();

        let (nodes, edges) = db.cognitive_goal_subgraph(goal).unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_transient_nodes_not_persisted_via_working_set() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let entity_id = alloc.alloc(NodeKind::Entity);
        let intent_id = alloc.alloc(NodeKind::IntentHypothesis);

        let entity = CognitiveNode::new(entity_id, "Persistent".into(), make_entity("persistent"));

        let intent = CognitiveNode::new(
            intent_id,
            "Transient intent".into(),
            NodePayload::IntentHypothesis(IntentPayload {
                description: "query the time".into(),
                features: vec![0.5, 0.3],
                posterior: 0.9,
                candidate_actions: vec![],
                source_context: "what time is it".into(),
            }),
        );

        let config = AttentionConfig::default();
        let mut ws = WorkingSet::with_allocator(config, alloc);
        ws.insert(entity);
        ws.insert(intent);

        let result = db.save_working_set(&ws).unwrap();
        assert_eq!(result.nodes_inserted, 1);

        assert!(db.load_cognitive_node(entity_id).unwrap().is_some());
        assert!(db.load_cognitive_node(intent_id).unwrap().is_none());
    }
}
