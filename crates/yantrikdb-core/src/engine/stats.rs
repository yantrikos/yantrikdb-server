use rusqlite::params;

use crate::error::Result;
use crate::types::Stats;

use super::{now, YantrikDB};

impl YantrikDB {
    /// Get engine statistics. Optionally filter memory counts by namespace.
    pub fn stats(&self, namespace: Option<&str>) -> Result<Stats> {
        let ns_filter = namespace.map(|ns| format!(" AND namespace = '{}'", ns.replace('\'', "''"))).unwrap_or_default();
        let active = self.conn.query_row(
            &format!("SELECT COUNT(*) FROM memories WHERE consolidation_status = 'active'{}", ns_filter),
            [], |row| row.get(0),
        )?;
        let consolidated = self.conn.query_row(
            &format!("SELECT COUNT(*) FROM memories WHERE consolidation_status = 'consolidated'{}", ns_filter),
            [], |row| row.get(0),
        )?;
        let tombstoned = self.conn.query_row(
            &format!("SELECT COUNT(*) FROM memories WHERE consolidation_status = 'tombstoned'{}", ns_filter),
            [], |row| row.get(0),
        )?;
        let archived = self.conn.query_row(
            &format!("SELECT COUNT(*) FROM memories WHERE storage_tier = 'cold'{}", ns_filter),
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
        let op_id = crate::id::new_id();
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
}
