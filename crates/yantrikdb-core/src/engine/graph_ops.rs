use rusqlite::params;

use crate::error::Result;
use crate::types::{Edge, Entity};

use super::{now, YantrikDB};

impl YantrikDB {
    /// Create or update a relationship between entities.
    #[tracing::instrument(skip(self))]
    pub fn relate(
        &self,
        src: &str,
        dst: &str,
        rel_type: &str,
        weight: f64,
    ) -> Result<String> {
        let edge_id = crate::id::new_id();
        let ts = now();

        self.conn.execute(
            "INSERT INTO edges (edge_id, src, dst, rel_type, weight, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(src, dst, rel_type) DO UPDATE SET weight = ?5, created_at = ?6",
            params![edge_id, src, dst, rel_type, weight, ts],
        )?;

        // Classify entity types using relationship semantics
        let (src_type, dst_type) =
            crate::graph::classify_with_relationship(src, dst, rel_type);

        // Ensure entities exist with classified entity_type
        for (entity, etype) in [(src, src_type), (dst, dst_type)] {
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

    /// Search entities by name pattern. If pattern is None, returns all entities
    /// ordered by most recently seen. Pattern uses SQL LIKE syntax (% for wildcard).
    pub fn search_entities(
        &self,
        pattern: Option<&str>,
        entity_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Entity>> {
        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match (pattern, entity_type) {
            (Some(p), Some(t)) => (
                "SELECT name, entity_type, first_seen, last_seen, mention_count \
                 FROM entities WHERE name LIKE ?1 AND entity_type = ?2 \
                 ORDER BY last_seen DESC LIMIT ?3".to_string(),
                vec![
                    Box::new(format!("%{}%", p)) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(t.to_string()),
                    Box::new(limit as i64),
                ],
            ),
            (Some(p), None) => (
                "SELECT name, entity_type, first_seen, last_seen, mention_count \
                 FROM entities WHERE name LIKE ?1 \
                 ORDER BY last_seen DESC LIMIT ?2".to_string(),
                vec![
                    Box::new(format!("%{}%", p)) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(limit as i64),
                ],
            ),
            (None, Some(t)) => (
                "SELECT name, entity_type, first_seen, last_seen, mention_count \
                 FROM entities WHERE entity_type = ?1 \
                 ORDER BY last_seen DESC LIMIT ?2".to_string(),
                vec![
                    Box::new(t.to_string()) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(limit as i64),
                ],
            ),
            (None, None) => (
                "SELECT name, entity_type, first_seen, last_seen, mention_count \
                 FROM entities ORDER BY last_seen DESC LIMIT ?1".to_string(),
                vec![Box::new(limit as i64) as Box<dyn rusqlite::types::ToSql>],
            ),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
        let entities = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok(Entity {
                    name: row.get("name")?,
                    entity_type: row.get("entity_type")?,
                    first_seen: row.get("first_seen")?,
                    last_seen: row.get("last_seen")?,
                    mention_count: row.get("mention_count")?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(entities)
    }

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

        let raw_memories: Vec<(String, String)> = self.conn.prepare(
            "SELECT rid, text FROM memories WHERE consolidation_status = 'active'",
        )?.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?.collect::<std::result::Result<Vec<_>, _>>()?;

        // Decrypt text if encrypted
        let memories: Vec<(String, String)> = raw_memories.into_iter()
            .map(|(rid, stored_text)| {
                let text = self.decrypt_text(&stored_text)?;
                Ok((rid, text))
            })
            .collect::<crate::error::Result<Vec<_>>>()?;

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
}
