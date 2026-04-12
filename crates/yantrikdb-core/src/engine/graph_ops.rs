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

        // Classify entity types using relationship semantics
        let (src_type, dst_type) =
            crate::graph::classify_with_relationship(src, dst, rel_type);

        // Phase 1: Lock conn for all SQL operations, then drop
        {
            let conn = self.conn.lock();
            conn.execute(
                "INSERT INTO edges (edge_id, src, dst, rel_type, weight, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
                 ON CONFLICT(src, dst, rel_type) DO UPDATE SET weight = ?5, created_at = ?6",
                params![edge_id, src, dst, rel_type, weight, ts],
            )?;

            // Ensure entities exist with classified entity_type
            for (entity, etype) in [(src, src_type), (dst, dst_type)] {
                conn.execute(
                    "INSERT INTO entities (name, entity_type, first_seen, last_seen) \
                     VALUES (?1, ?2, ?3, ?4) \
                     ON CONFLICT(name) DO UPDATE SET last_seen = ?4, mention_count = mention_count + 1, \
                     entity_type = CASE WHEN entities.entity_type = 'unknown' THEN ?2 ELSE entities.entity_type END",
                    params![entity, etype, ts, ts],
                )?;
            }
        } // conn dropped

        // Phase 2: Lock graph_index write for in-memory updates, then drop
        {
            let mut gi = self.graph_index.write();
            gi.add_entity(src, src_type);
            gi.add_entity(dst, dst_type);
            gi.add_edge(src, dst, weight as f32);
        } // graph_index dropped

        // Backfill memory_entities for newly-created entities.
        // When remember() runs BEFORE relate(), the memory doesn't get linked
        // because the entity doesn't exist yet. Fix: scan active memories for
        // mentions of the src/dst entities and create links retroactively.
        self.backfill_memory_entities_for(&[src, dst])?;

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
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
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

        let conn = self.conn.lock();
        let mut stmt = conn.prepare(&sql)?;
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
        // Phase 1: Lock conn for SQL INSERT, then drop
        {
            let conn = self.conn.lock();
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                params![memory_rid, entity_name],
            )?;
        } // conn dropped

        // Phase 2: Lock graph_index write for in-memory update
        self.graph_index.write().link_memory(memory_rid, entity_name);
        Ok(())
    }

    /// Backfill memory_entities for a specific set of entity names.
    /// Used by relate() to retroactively link memories to newly-created entities.
    fn backfill_memory_entities_for(&self, entity_names: &[&str]) -> Result<()> {
        // Phase 1: Lock conn, query candidate memories for each entity, drop conn
        struct LinkCandidate {
            rid: String,
            entity: String,
        }
        let mut candidates = Vec::new();

        {
            let conn = self.conn.lock();
            let mut stmt = conn.prepare_cached(
                "SELECT rid, text FROM memories \
                 WHERE consolidation_status = 'active' \
                 AND rid NOT IN (SELECT memory_rid FROM memory_entities WHERE entity_name = ?1)"
            )?;
            for &entity in entity_names {
                let entity_tokens = crate::graph::tokenize(entity);
                if entity_tokens.is_empty() {
                    continue;
                }
                let rows: Vec<(String, String)> = stmt
                    .query_map(params![entity], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                // Phase 2: Compute matches (decrypt_text doesn't need conn)
                for (rid, stored_text) in &rows {
                    let text = self.decrypt_text(stored_text).unwrap_or_else(|_| stored_text.clone());
                    let text_tokens = crate::graph::tokenize(&text);
                    if crate::graph::entity_matches_text(entity, &text_tokens) {
                        candidates.push(LinkCandidate {
                            rid: rid.clone(),
                            entity: entity.to_string(),
                        });
                    }
                }
            }
        } // conn dropped

        if candidates.is_empty() {
            return Ok(());
        }

        // Phase 3: Lock conn, do INSERT OR IGNORE for each link, drop conn
        {
            let conn = self.conn.lock();
            for c in &candidates {
                conn.execute(
                    "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                    params![c.rid, c.entity],
                )?;
            }
        } // conn dropped

        // Phase 4: Lock graph_index write, do link_memory for each, drop
        {
            let mut gi = self.graph_index.write();
            for c in &candidates {
                gi.link_memory(&c.rid, &c.entity);
            }
        } // graph_index dropped

        Ok(())
    }

    /// Backfill the memory_entities table by scanning memory text for known entity names.
    /// Uses word-boundary matching to avoid false positives.
    /// Returns the number of links created. Idempotent (uses INSERT OR IGNORE).
    pub fn backfill_memory_entities(&self) -> Result<usize> {
        // Phase 1: Lock conn, query entities and memories, drop conn
        let entities: Vec<String>;
        let raw_memories: Vec<(String, String)>;

        {
            let conn = self.conn.lock();
            entities = conn.prepare(
                "SELECT name FROM entities",
            )?.query_map([], |row| row.get(0))?.collect::<std::result::Result<Vec<_>, _>>()?;

            if entities.is_empty() {
                return Ok(0);
            }

            raw_memories = conn.prepare(
                "SELECT rid, text FROM memories WHERE consolidation_status = 'active'",
            )?.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?.collect::<std::result::Result<Vec<_>, _>>()?;
        } // conn dropped

        // Phase 2: Compute matches (decrypt_text doesn't need conn)
        let memories: Vec<(String, String)> = raw_memories.into_iter()
            .map(|(rid, stored_text)| {
                let text = self.decrypt_text(&stored_text)?;
                Ok((rid, text))
            })
            .collect::<crate::error::Result<Vec<_>>>()?;

        struct LinkCandidate {
            rid: String,
            entity: String,
        }
        let mut candidates = Vec::new();

        for (rid, text) in &memories {
            let text_tokens = crate::graph::tokenize(text);
            for entity in &entities {
                if crate::graph::entity_matches_text(entity, &text_tokens) {
                    candidates.push(LinkCandidate {
                        rid: rid.clone(),
                        entity: entity.clone(),
                    });
                }
            }
        }

        let count = candidates.len();

        if count == 0 {
            return Ok(0);
        }

        // Phase 3: Lock conn, do INSERT OR IGNORE for each link, drop conn
        {
            let conn = self.conn.lock();
            for c in &candidates {
                conn.execute(
                    "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                    params![c.rid, c.entity],
                )?;
            }
        } // conn dropped

        // Phase 4: Lock graph_index write, do link_memory for each, drop
        {
            let mut gi = self.graph_index.write();
            for c in &candidates {
                gi.link_memory(&c.rid, &c.entity);
            }
        } // graph_index dropped

        Ok(count)
    }
}
