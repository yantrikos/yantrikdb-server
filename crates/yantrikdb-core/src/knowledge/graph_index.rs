//! In-memory graph adjacency index for fast entity-augmented recall.
//!
//! Replaces per-query SQL lookups with O(1) adjacency list lookups.
//! Built from SQLite on engine init, maintained incrementally on mutations.

use std::collections::{HashMap, HashSet, VecDeque};

use rusqlite::Connection;

use crate::error::Result;
use crate::graph;

/// In-memory graph index using adjacency lists.
pub struct GraphIndex {
    // Entity name <-> integer ID mapping
    entity_to_id: HashMap<String, u32>,
    id_to_entity: Vec<String>,

    // Adjacency list: entity_id -> [(neighbor_id, weight)]
    adjacency: Vec<Vec<(u32, f32)>>,

    // Bidirectional memory-entity linkage
    memory_to_entities: HashMap<String, Vec<u32>>,
    entity_to_memories: Vec<Vec<String>>,

    // Cached entity metadata
    entity_types: Vec<String>,
    mention_counts: Vec<u32>,
}

impl GraphIndex {
    /// Create an empty graph index.
    pub fn new() -> Self {
        Self {
            entity_to_id: HashMap::new(),
            id_to_entity: Vec::new(),
            adjacency: Vec::new(),
            memory_to_entities: HashMap::new(),
            entity_to_memories: Vec::new(),
            entity_types: Vec::new(),
            mention_counts: Vec::new(),
        }
    }

    /// Build the graph index from SQLite tables (entities, edges, memory_entities).
    pub fn build_from_db(conn: &Connection) -> Result<Self> {
        let mut idx = Self::new();

        // Load entities
        let mut stmt = conn.prepare(
            "SELECT name, entity_type, mention_count FROM entities"
        )?;
        let entities: Vec<(String, String, u32)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get::<_, i64>(2)? as u32)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        for (name, etype, mc) in &entities {
            idx.ensure_entity(name, etype, *mc);
        }

        // Load non-tombstoned edges
        let mut stmt = conn.prepare(
            "SELECT src, dst, weight FROM edges WHERE tombstoned = 0"
        )?;
        let edges: Vec<(String, String, f32)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get::<_, f64>(2)? as f32)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        for (src, dst, weight) in &edges {
            // Ensure both entities exist (edges may reference entities not yet in entities table)
            idx.ensure_entity(src, "unknown", 0);
            idx.ensure_entity(dst, "unknown", 0);
            let src_id = idx.entity_to_id[src];
            let dst_id = idx.entity_to_id[dst];
            // Bidirectional
            idx.adjacency[src_id as usize].push((dst_id, *weight));
            idx.adjacency[dst_id as usize].push((src_id, *weight));
        }

        // Load memory-entity links
        let mut stmt = conn.prepare(
            "SELECT memory_rid, entity_name FROM memory_entities"
        )?;
        let links: Vec<(String, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        for (rid, entity_name) in &links {
            if let Some(&eid) = idx.entity_to_id.get(entity_name) {
                idx.memory_to_entities.entry(rid.clone()).or_default().push(eid);
                idx.entity_to_memories[eid as usize].push(rid.clone());
            }
        }

        Ok(idx)
    }

    // ── Entity management ──

    /// Ensure an entity exists in the index, returning its ID.
    fn ensure_entity(&mut self, name: &str, entity_type: &str, mention_count: u32) -> u32 {
        if let Some(&id) = self.entity_to_id.get(name) {
            return id;
        }
        let id = self.id_to_entity.len() as u32;
        self.entity_to_id.insert(name.to_string(), id);
        self.id_to_entity.push(name.to_string());
        self.adjacency.push(Vec::new());
        self.entity_to_memories.push(Vec::new());
        self.entity_types.push(entity_type.to_string());
        self.mention_counts.push(mention_count);
        id
    }

    /// Return all known entity names.
    pub fn all_entity_names(&self) -> Vec<String> {
        self.id_to_entity.clone()
    }

    /// Add or update an entity (called from relate()).
    pub fn add_entity(&mut self, name: &str, entity_type: &str) {
        if let Some(&id) = self.entity_to_id.get(name) {
            // Update mention count
            self.mention_counts[id as usize] += 1;
            // Upgrade type if currently unknown
            if self.entity_types[id as usize] == "unknown" && entity_type != "unknown" {
                self.entity_types[id as usize] = entity_type.to_string();
            }
        } else {
            self.ensure_entity(name, entity_type, 1);
        }
    }

    /// Add an edge (called from relate()). Bidirectional.
    pub fn add_edge(&mut self, src: &str, dst: &str, weight: f32) {
        let src_id = self.ensure_entity(src, "unknown", 0);
        let dst_id = self.ensure_entity(dst, "unknown", 0);

        // Remove existing edge if any (upsert semantics)
        self.adjacency[src_id as usize].retain(|&(n, _)| n != dst_id);
        self.adjacency[dst_id as usize].retain(|&(n, _)| n != src_id);

        // Add new edge
        self.adjacency[src_id as usize].push((dst_id, weight));
        self.adjacency[dst_id as usize].push((src_id, weight));
    }

    /// Link a memory to an entity (called from link_memory_entity()).
    pub fn link_memory(&mut self, rid: &str, entity_name: &str) {
        let eid = self.ensure_entity(entity_name, "unknown", 0);

        let entities = self.memory_to_entities.entry(rid.to_string()).or_default();
        if !entities.contains(&eid) {
            entities.push(eid);
        }

        let memories = &mut self.entity_to_memories[eid as usize];
        if !memories.contains(&rid.to_string()) {
            memories.push(rid.to_string());
        }
    }

    /// Unlink all entities from a memory (called from forget()).
    pub fn unlink_memory(&mut self, rid: &str) {
        if let Some(entity_ids) = self.memory_to_entities.remove(rid) {
            for eid in entity_ids {
                self.entity_to_memories[eid as usize].retain(|r| r != rid);
            }
        }
    }

    // ── Query methods (replace SQL in recall()) ──

    /// Get neighbors of an entity by ID. O(1).
    pub fn neighbors(&self, entity_id: u32) -> &[(u32, f32)] {
        &self.adjacency[entity_id as usize]
    }

    /// BFS expansion from seed entity names. Pure in-memory.
    /// Returns (entity_name, hops_from_seed, cumulative_edge_weight).
    pub fn expand_bfs(
        &self,
        seeds: &[&str],
        max_hops: u8,
        max_entities: usize,
    ) -> Vec<(String, u8, f64)> {
        let mut result: Vec<(String, u8, f64)> = Vec::new();
        let mut visited: HashMap<u32, (u8, f64)> = HashMap::new();
        let mut frontier: VecDeque<(u32, u8, f64)> = VecDeque::new();

        for seed in seeds {
            if let Some(&id) = self.entity_to_id.get(*seed) {
                if !visited.contains_key(&id) {
                    visited.insert(id, (0, 1.0));
                    result.push((seed.to_string(), 0, 1.0));
                    frontier.push_back((id, 0, 1.0));
                }
            }
        }

        while let Some((entity_id, hops, weight)) = frontier.pop_front() {
            if hops >= max_hops || result.len() >= max_entities {
                break;
            }

            for &(neighbor_id, edge_weight) in self.neighbors(entity_id) {
                if visited.contains_key(&neighbor_id) {
                    continue;
                }
                if result.len() >= max_entities {
                    break;
                }
                let cumulative = weight * edge_weight as f64;
                let next_hops = hops + 1;
                visited.insert(neighbor_id, (next_hops, cumulative));
                result.push((
                    self.id_to_entity[neighbor_id as usize].clone(),
                    next_hops,
                    cumulative,
                ));
                if next_hops < max_hops {
                    frontier.push_back((neighbor_id, next_hops, cumulative));
                }
            }
        }

        result
    }

    /// Compute graph proximity for a memory given expanded entity set.
    /// Returns max(cumulative_weight / 2^hops) across linked entities.
    pub fn graph_proximity(
        &self,
        rid: &str,
        expanded_entities: &HashMap<String, (u8, f64)>,
    ) -> f64 {
        let Some(entity_ids) = self.memory_to_entities.get(rid) else {
            return 0.0;
        };
        let mut max_prox = 0.0f64;
        for &eid in entity_ids {
            let name = &self.id_to_entity[eid as usize];
            if let Some(&(hops, weight)) = expanded_entities.get(name) {
                // Sharper decay: 4^hops so 1-hop connections contribute much less
                // than direct (0-hop) connections. Prevents graph over-expansion
                // where high-importance memories leak in through distant neighbors.
                let prox = weight / f64::powf(4.0, hops as f64);
                if prox > max_prox {
                    max_prox = prox;
                }
            }
        }
        max_prox
    }

    /// Get entity names linked to a memory.
    pub fn entities_for_memory(&self, rid: &str) -> Vec<&str> {
        match self.memory_to_entities.get(rid) {
            Some(ids) => ids.iter().map(|&id| self.id_to_entity[id as usize].as_str()).collect(),
            None => vec![],
        }
    }

    /// Get entity names for multiple memories (batch).
    pub fn entities_for_memories(&self, rids: &[&str]) -> Vec<String> {
        let mut result: HashSet<String> = HashSet::new();
        for rid in rids {
            if let Some(ids) = self.memory_to_entities.get(*rid) {
                for &id in ids {
                    result.insert(self.id_to_entity[id as usize].clone());
                }
            }
        }
        result.into_iter().collect()
    }

    /// Get all memory RIDs linked to any of the given entities.
    pub fn memories_for_entities(&self, entity_names: &[&str]) -> HashSet<String> {
        let mut result: HashSet<String> = HashSet::new();
        for name in entity_names {
            if let Some(&eid) = self.entity_to_id.get(*name) {
                for rid in &self.entity_to_memories[eid as usize] {
                    result.insert(rid.clone());
                }
            }
        }
        result
    }

    /// Find entities matching query text tokens (replaces full entity table scan).
    /// Returns (name, entity_type, mention_count).
    pub fn entity_matches_query(&self, tokens: &[String]) -> Vec<(String, String, u32)> {
        let mut matches = Vec::new();
        for (i, name) in self.id_to_entity.iter().enumerate() {
            if graph::entity_matches_text(name, tokens) {
                matches.push((
                    name.clone(),
                    self.entity_types[i].clone(),
                    self.mention_counts[i],
                ));
            }
        }
        matches
    }

    /// Get the type of a single entity by name. O(1).
    pub fn entity_type(&self, name: &str) -> Option<&str> {
        self.entity_to_id
            .get(name)
            .map(|&id| self.entity_types[id as usize].as_str())
    }

    /// Get all entity names of a given type (e.g., "person", "tech").
    pub fn entities_by_type(&self, entity_type: &str) -> Vec<String> {
        self.entity_types
            .iter()
            .enumerate()
            .filter(|(_, t)| t.as_str() == entity_type)
            .map(|(i, _)| self.id_to_entity[i].clone())
            .collect()
    }

    /// Number of entities in the index.
    pub fn entity_count(&self) -> usize {
        self.id_to_entity.len()
    }

    /// Number of directed edge entries (each undirected edge counted twice).
    pub fn edge_count(&self) -> usize {
        self.adjacency.iter().map(|adj| adj.len()).sum::<usize>() / 2
    }

    /// Number of memory-entity links.
    pub fn link_count(&self) -> usize {
        self.memory_to_entities.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::YantrikDB;

    fn setup_db() -> YantrikDB {
        let db = YantrikDB::new(":memory:", 4).unwrap();
        db.relate("Alice", "Bob", "knows", 1.0).unwrap();
        db.relate("Bob", "Charlie", "knows", 0.8).unwrap();
        db.relate("Alice", "ProjectX", "works_on", 1.0).unwrap();
        db.relate("Dave", "ProjectX", "works_on", 0.9).unwrap();

        let emb = vec![1.0f32, 0.0, 0.0, 0.0];
        let r1 = db.record("Alice discussed the plan", "episodic", 0.5, 0.0, 604800.0, &serde_json::json!({}), &emb, "default", 0.8, "general", "user", None).unwrap();
        let r2 = db.record("Bob reviewed the code", "episodic", 0.5, 0.0, 604800.0, &serde_json::json!({}), &emb, "default", 0.8, "general", "user", None).unwrap();
        let r3 = db.record("Charlie deployed to production", "episodic", 0.5, 0.0, 604800.0, &serde_json::json!({}), &emb, "default", 0.8, "general", "user", None).unwrap();

        db.link_memory_entity(&r1, "Alice").unwrap();
        db.link_memory_entity(&r1, "ProjectX").unwrap();
        db.link_memory_entity(&r2, "Bob").unwrap();
        db.link_memory_entity(&r3, "Charlie").unwrap();

        db
    }

    #[test]
    fn test_build_from_empty_db() {
        let db = YantrikDB::new(":memory:", 4).unwrap();
        let idx = GraphIndex::build_from_db(&*db.conn()).unwrap();
        assert_eq!(idx.entity_count(), 0);
        assert_eq!(idx.edge_count(), 0);
        assert_eq!(idx.link_count(), 0);
    }

    #[test]
    fn test_build_from_populated_db() {
        let db = setup_db();
        let idx = GraphIndex::build_from_db(&*db.conn()).unwrap();
        // 4 edges: Alice-Bob, Bob-Charlie, Alice-ProjectX, Dave-ProjectX
        assert_eq!(idx.edge_count(), 4);
        // 5 entities: Alice, Bob, Charlie, ProjectX, Dave
        assert_eq!(idx.entity_count(), 5);
        // 4 memory-entity links: Alice+ProjectX for r1, Bob for r2, Charlie for r3
        assert_eq!(idx.link_count(), 4);
    }

    #[test]
    fn test_expand_bfs_1hop() {
        let db = setup_db();
        let idx = GraphIndex::build_from_db(&*db.conn()).unwrap();
        let expanded = idx.expand_bfs(&["Alice"], 1, 30);
        let names: HashSet<String> = expanded.iter().map(|(n, _, _)| n.clone()).collect();
        assert!(names.contains("Alice"));
        assert!(names.contains("Bob"));
        assert!(names.contains("ProjectX"));
        assert!(!names.contains("Charlie")); // 2 hops away
    }

    #[test]
    fn test_expand_bfs_2hop() {
        let db = setup_db();
        let idx = GraphIndex::build_from_db(&*db.conn()).unwrap();
        let expanded = idx.expand_bfs(&["Alice"], 2, 30);
        let names: HashSet<String> = expanded.iter().map(|(n, _, _)| n.clone()).collect();
        assert!(names.contains("Charlie"));
        assert!(names.contains("Dave"));
    }

    #[test]
    fn test_expand_bfs_budget_limit() {
        let db = setup_db();
        let idx = GraphIndex::build_from_db(&*db.conn()).unwrap();
        let expanded = idx.expand_bfs(&["Alice"], 2, 3);
        assert!(expanded.len() <= 3);
    }

    #[test]
    fn test_expand_bfs_matches_sql() {
        let db = setup_db();
        let idx = GraphIndex::build_from_db(&*db.conn()).unwrap();

        let sql_result = graph::expand_entities_nhop(&*db.conn(), &["Alice"], 1, 20).unwrap();
        let idx_result = idx.expand_bfs(&["Alice"], 1, 20);

        let sql_names: HashSet<String> = sql_result.iter().map(|(n, _, _)| n.clone()).collect();
        let idx_names: HashSet<String> = idx_result.iter().map(|(n, _, _)| n.clone()).collect();
        assert_eq!(sql_names, idx_names);
    }

    #[test]
    fn test_graph_proximity() {
        let db = setup_db();
        let idx = GraphIndex::build_from_db(&*db.conn()).unwrap();

        let rid: String = db.conn().query_row(
            "SELECT rid FROM memories ORDER BY created_at LIMIT 1", [], |row| row.get(0),
        ).unwrap();

        let mut expanded = HashMap::new();
        expanded.insert("Alice".to_string(), (0u8, 1.0f64));
        expanded.insert("ProjectX".to_string(), (1u8, 1.0f64));

        let prox = idx.graph_proximity(&rid, &expanded);
        assert!((prox - 1.0).abs() < 1e-10); // Alice is seed (hops=0)
    }

    #[test]
    fn test_graph_proximity_matches_sql() {
        let db = setup_db();
        let idx = GraphIndex::build_from_db(&*db.conn()).unwrap();

        let rid: String = db.conn().query_row(
            "SELECT rid FROM memories ORDER BY created_at LIMIT 1", [], |row| row.get(0),
        ).unwrap();

        let expanded = graph::expand_entities_nhop(&*db.conn(), &["Alice"], 1, 20).unwrap();
        let expanded_map: HashMap<String, (u8, f64)> = expanded
            .iter()
            .map(|(n, h, w)| (n.clone(), (*h, *w)))
            .collect();

        let sql_prox = graph::graph_proximity(&*db.conn(), &rid, &expanded_map).unwrap();
        let idx_prox = idx.graph_proximity(&rid, &expanded_map);
        assert!((sql_prox - idx_prox).abs() < 1e-10);
    }

    #[test]
    fn test_entities_for_memories() {
        let db = setup_db();
        let idx = GraphIndex::build_from_db(&*db.conn()).unwrap();

        let rid: String = db.conn().query_row(
            "SELECT rid FROM memories ORDER BY created_at LIMIT 1", [], |row| row.get(0),
        ).unwrap();

        let entities = idx.entities_for_memories(&[&rid]);
        assert!(entities.contains(&"Alice".to_string()));
        assert!(entities.contains(&"ProjectX".to_string()));
    }

    #[test]
    fn test_memories_for_entities() {
        let db = setup_db();
        let idx = GraphIndex::build_from_db(&*db.conn()).unwrap();
        let rids = idx.memories_for_entities(&["Alice"]);
        assert_eq!(rids.len(), 1);
    }

    #[test]
    fn test_entity_matches_query() {
        let db = setup_db();
        let idx = GraphIndex::build_from_db(&*db.conn()).unwrap();

        let tokens = graph::tokenize("What did Alice say about ProjectX?");
        let matches = idx.entity_matches_query(&tokens);
        let names: HashSet<String> = matches.iter().map(|(n, _, _)| n.clone()).collect();
        assert!(names.contains("Alice"));
        assert!(names.contains("ProjectX"));
        assert!(!names.contains("Bob"));
    }

    #[test]
    fn test_incremental_add_edge() {
        let mut idx = GraphIndex::new();
        idx.add_edge("X", "Y", 0.9);
        assert_eq!(idx.entity_count(), 2);
        assert_eq!(idx.edge_count(), 1);

        // Verify bidirectional
        let x_id = idx.entity_to_id["X"];
        let y_id = idx.entity_to_id["Y"];
        assert!(idx.neighbors(x_id).iter().any(|&(n, _)| n == y_id));
        assert!(idx.neighbors(y_id).iter().any(|&(n, _)| n == x_id));
    }

    #[test]
    fn test_incremental_upsert_edge() {
        let mut idx = GraphIndex::new();
        idx.add_edge("X", "Y", 0.5);
        idx.add_edge("X", "Y", 0.9); // upsert
        assert_eq!(idx.edge_count(), 1); // still 1 edge

        let x_id = idx.entity_to_id["X"];
        let y_id = idx.entity_to_id["Y"];
        let w = idx.neighbors(x_id).iter().find(|&&(n, _)| n == y_id).unwrap().1;
        assert!((w - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_incremental_link_memory() {
        let mut idx = GraphIndex::new();
        idx.add_entity("Alice", "person");
        idx.link_memory("mem1", "Alice");
        assert_eq!(idx.link_count(), 1);

        let entities = idx.entities_for_memory("mem1");
        assert_eq!(entities, vec!["Alice"]);

        let rids = idx.memories_for_entities(&["Alice"]);
        assert!(rids.contains("mem1"));
    }

    #[test]
    fn test_unlink_memory() {
        let mut idx = GraphIndex::new();
        idx.add_entity("Alice", "person");
        idx.add_entity("Bob", "person");
        idx.link_memory("mem1", "Alice");
        idx.link_memory("mem1", "Bob");
        assert_eq!(idx.link_count(), 2);

        idx.unlink_memory("mem1");
        assert_eq!(idx.link_count(), 0);
        assert!(idx.entities_for_memory("mem1").is_empty());
        assert!(idx.memories_for_entities(&["Alice"]).is_empty());
    }

    #[test]
    fn test_entities_by_type() {
        // Test with manually typed entities (deterministic)
        let mut idx = GraphIndex::new();
        idx.add_entity("Alice", "person");
        idx.add_entity("Bob", "person");
        idx.add_entity("FAISS", "tech");
        idx.add_entity("ProjectX", "project");

        let persons = idx.entities_by_type("person");
        assert_eq!(persons.len(), 2);
        assert!(persons.contains(&"Alice".to_string()));
        assert!(persons.contains(&"Bob".to_string()));

        let techs = idx.entities_by_type("tech");
        assert_eq!(techs.len(), 1);
        assert!(techs.contains(&"FAISS".to_string()));

        let empty = idx.entities_by_type("nonexistent");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_idempotent_link() {
        let mut idx = GraphIndex::new();
        idx.add_entity("Alice", "person");
        idx.link_memory("mem1", "Alice");
        idx.link_memory("mem1", "Alice"); // duplicate
        assert_eq!(idx.link_count(), 1); // still 1
    }
}
