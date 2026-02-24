//! Graph traversal utilities for entity-augmented recall.

use std::collections::{HashMap, HashSet, VecDeque};

use rusqlite::{params, Connection};

use crate::error::Result;

// ── Word-boundary entity matching ──

/// Tokenize text into lowercase words, splitting on non-alphanumeric chars.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Check if an entity name appears as whole-word(s) in pre-tokenized text.
/// Single-word entities require exact token match.
/// Multi-word entities require contiguous token sequence match.
pub fn entity_matches_text(entity: &str, text_tokens: &[String]) -> bool {
    let entity_tokens = tokenize(entity);
    if entity_tokens.is_empty() {
        return false;
    }
    if entity_tokens.len() == 1 {
        text_tokens.iter().any(|t| t == &entity_tokens[0])
    } else {
        text_tokens
            .windows(entity_tokens.len())
            .any(|window| window.iter().zip(entity_tokens.iter()).all(|(w, e)| w == e))
    }
}

// ── Entity type classification ──

/// Tech terms that should NOT be classified as person names even if title-cased/all-caps.
const TECH_BLOCKLIST: &[&str] = &[
    "faiss", "onnx", "scann", "redis", "kafka", "docker", "kubernetes", "react",
    "python", "rust", "java", "swift", "flutter", "pytorch", "tensorflow",
    "numpy", "pandas", "spark", "hadoop", "nginx", "postgres", "mysql",
    "sqlite", "graphql", "grpc", "oauth", "jwt", "html", "css",
    "api", "sdk", "ml", "ai", "gpu", "cpu", "ram", "ssd", "aws", "gcp",
];

/// Classify an entity name into a type: "person", "tech", or "unknown".
pub fn classify_entity_type(name: &str) -> &'static str {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return "unknown";
    }
    let lower = trimmed.to_lowercase();

    // Check tech blocklist
    if TECH_BLOCKLIST.contains(&lower.as_str()) {
        return "tech";
    }

    // All-caps multi-char → tech (e.g., "FAISS", "ONNX")
    if trimmed.len() > 1 && trimmed.chars().all(|c| c.is_uppercase() || !c.is_alphabetic()) {
        return "tech";
    }

    // Single word, title-case, not in tech blocklist → person
    if !trimmed.contains(' ')
        && trimmed.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
        && trimmed.chars().skip(1).all(|c| c.is_lowercase() || !c.is_alphabetic())
    {
        return "person";
    }

    "unknown"
}

/// Given a set of memory RIDs, find all entities those memories are linked to.
pub fn entities_for_memories(conn: &Connection, rids: &[&str]) -> Result<Vec<String>> {
    if rids.is_empty() {
        return Ok(vec![]);
    }
    let placeholders: String = (0..rids.len()).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT DISTINCT entity_name FROM memory_entities WHERE memory_rid IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
        rids.iter().map(|r| Box::new(r.to_string()) as Box<dyn rusqlite::types::ToSql>).collect();
    let params_ref: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();
    let entities = stmt
        .query_map(params_ref.as_slice(), |row| row.get(0))?
        .collect::<std::result::Result<Vec<String>, _>>()?;
    Ok(entities)
}

/// Given a set of entity names, find all memory RIDs connected to those entities.
pub fn memories_for_entities(conn: &Connection, entity_names: &[&str]) -> Result<HashSet<String>> {
    if entity_names.is_empty() {
        return Ok(HashSet::new());
    }
    let placeholders: String = (0..entity_names.len()).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT DISTINCT memory_rid FROM memory_entities WHERE entity_name IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
        entity_names.iter().map(|e| Box::new(e.to_string()) as Box<dyn rusqlite::types::ToSql>).collect();
    let params_ref: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();
    let rids = stmt
        .query_map(params_ref.as_slice(), |row| row.get(0))?
        .collect::<std::result::Result<HashSet<String>, _>>()?;
    Ok(rids)
}

/// Expand entity set N hops via the edges table (BFS).
/// Returns (entity_name, hops_from_seed, cumulative_edge_weight).
/// Seeds are returned with hops=0 and weight=1.0.
pub fn expand_entities_nhop(
    conn: &Connection,
    seeds: &[&str],
    max_hops: u8,
    max_entities: usize,
) -> Result<Vec<(String, u8, f64)>> {
    let mut result: Vec<(String, u8, f64)> = Vec::new();
    let mut visited: HashMap<String, (u8, f64)> = HashMap::new();

    // Initialize with seeds
    for s in seeds {
        visited.insert(s.to_string(), (0, 1.0));
        result.push((s.to_string(), 0, 1.0));
    }

    let mut frontier: VecDeque<(String, u8, f64)> = seeds
        .iter()
        .map(|s| (s.to_string(), 0u8, 1.0f64))
        .collect();

    while let Some((entity, hops, weight)) = frontier.pop_front() {
        if hops >= max_hops || result.len() >= max_entities {
            break;
        }

        // Find neighbors via edges (both directions)
        let mut stmt = conn.prepare(
            "SELECT src, dst, weight FROM edges WHERE (src = ?1 OR dst = ?1) AND tombstoned = 0",
        )?;
        let neighbors: Vec<(String, f64)> = stmt
            .query_map(params![entity], |row| {
                let src: String = row.get(0)?;
                let dst: String = row.get(1)?;
                let w: f64 = row.get(2)?;
                let neighbor = if src == entity { dst } else { src };
                Ok((neighbor, w))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        for (neighbor, edge_weight) in neighbors {
            if visited.contains_key(&neighbor) {
                continue;
            }
            if result.len() >= max_entities {
                break;
            }
            let cumulative = weight * edge_weight;
            let next_hops = hops + 1;
            visited.insert(neighbor.clone(), (next_hops, cumulative));
            result.push((neighbor.clone(), next_hops, cumulative));
            if next_hops < max_hops {
                frontier.push_back((neighbor, next_hops, cumulative));
            }
        }
    }

    Ok(result)
}

/// Compute graph proximity score for a memory based on its entity connections.
/// Returns the maximum proximity across all entities the memory is linked to.
/// proximity = cumulative_weight / 2^hops  (steeper decay to stay discriminative)
/// Seeds (hops=0) → 1.0, 1-hop → 0.5, 2-hop → 0.25
pub fn graph_proximity(
    conn: &Connection,
    memory_rid: &str,
    expanded_entities: &HashMap<String, (u8, f64)>,
) -> Result<f64> {
    let mem_entities: Vec<String> = conn
        .prepare("SELECT entity_name FROM memory_entities WHERE memory_rid = ?1")?
        .query_map(params![memory_rid], |row| row.get(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let mut max_proximity = 0.0f64;
    for entity in &mem_entities {
        if let Some(&(hops, weight)) = expanded_entities.get(entity) {
            let prox = weight / f64::powf(2.0, hops as f64);
            if prox > max_proximity {
                max_proximity = prox;
            }
        }
    }
    Ok(max_proximity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AIDB;

    fn setup_db() -> AIDB {
        let db = AIDB::new(":memory:", 4).unwrap();
        // Create entities and edges
        db.relate("Alice", "Bob", "knows", 1.0).unwrap();
        db.relate("Bob", "Charlie", "knows", 0.8).unwrap();
        db.relate("Alice", "ProjectX", "works_on", 1.0).unwrap();
        db.relate("Dave", "ProjectX", "works_on", 0.9).unwrap();

        // Record memories and link to entities
        let emb = vec![1.0f32, 0.0, 0.0, 0.0];
        let r1 = db.record("Alice discussed the plan", "episodic", 0.5, 0.0, 604800.0, &serde_json::json!({}), &emb, "default").unwrap();
        let r2 = db.record("Bob reviewed the code", "episodic", 0.5, 0.0, 604800.0, &serde_json::json!({}), &emb, "default").unwrap();
        let r3 = db.record("Charlie deployed to production", "episodic", 0.5, 0.0, 604800.0, &serde_json::json!({}), &emb, "default").unwrap();

        db.link_memory_entity(&r1, "Alice").unwrap();
        db.link_memory_entity(&r1, "ProjectX").unwrap();
        db.link_memory_entity(&r2, "Bob").unwrap();
        db.link_memory_entity(&r3, "Charlie").unwrap();

        db
    }

    #[test]
    fn test_entities_for_memories() {
        let db = setup_db();
        // Get the first memory's rid
        let rid: String = db.conn().query_row(
            "SELECT rid FROM memories ORDER BY created_at LIMIT 1", [], |row| row.get(0),
        ).unwrap();

        let entities = entities_for_memories(db.conn(), &[&rid]).unwrap();
        assert!(entities.contains(&"Alice".to_string()));
        assert!(entities.contains(&"ProjectX".to_string()));
    }

    #[test]
    fn test_memories_for_entities() {
        let db = setup_db();
        let rids = memories_for_entities(db.conn(), &["Alice"]).unwrap();
        assert_eq!(rids.len(), 1); // Only the Alice memory is linked
    }

    #[test]
    fn test_expand_1hop() {
        let db = setup_db();
        let expanded = expand_entities_nhop(db.conn(), &["Alice"], 1, 30).unwrap();
        let names: HashSet<String> = expanded.iter().map(|(n, _, _)| n.clone()).collect();
        // Alice (seed) + Bob (knows) + ProjectX (works_on)
        assert!(names.contains("Alice"));
        assert!(names.contains("Bob"));
        assert!(names.contains("ProjectX"));
    }

    #[test]
    fn test_expand_2hop() {
        let db = setup_db();
        let expanded = expand_entities_nhop(db.conn(), &["Alice"], 2, 30).unwrap();
        let names: HashSet<String> = expanded.iter().map(|(n, _, _)| n.clone()).collect();
        // 2-hop from Alice: Alice->Bob->Charlie, Alice->ProjectX->Dave
        assert!(names.contains("Charlie"));
        assert!(names.contains("Dave"));
    }

    #[test]
    fn test_expand_budget_limit() {
        let db = setup_db();
        let expanded = expand_entities_nhop(db.conn(), &["Alice"], 2, 3).unwrap();
        assert!(expanded.len() <= 3);
    }

    #[test]
    fn test_no_tombstoned_edges() {
        let db = setup_db();
        // Tombstone the Alice->Bob edge
        db.conn().execute(
            "UPDATE edges SET tombstoned = 1 WHERE src = 'Alice' AND dst = 'Bob'",
            [],
        ).unwrap();
        let expanded = expand_entities_nhop(db.conn(), &["Alice"], 1, 30).unwrap();
        let names: HashSet<String> = expanded.iter().map(|(n, _, _)| n.clone()).collect();
        // Bob should NOT be reachable via tombstoned edge
        assert!(!names.contains("Bob"));
        // ProjectX should still be reachable
        assert!(names.contains("ProjectX"));
    }

    #[test]
    fn test_graph_proximity_score() {
        let db = setup_db();
        let rid: String = db.conn().query_row(
            "SELECT rid FROM memories ORDER BY created_at LIMIT 1", [], |row| row.get(0),
        ).unwrap();

        let mut expanded = HashMap::new();
        expanded.insert("Alice".to_string(), (0u8, 1.0f64));
        expanded.insert("ProjectX".to_string(), (1u8, 1.0f64));

        let prox = graph_proximity(db.conn(), &rid, &expanded).unwrap();
        // Alice is hops=0 → proximity = 1.0 / (0+1) = 1.0
        assert!((prox - 1.0).abs() < 1e-10);
    }

    // ── Word-boundary matching tests ──

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("What is Sarah working on?");
        assert_eq!(tokens, vec!["what", "is", "sarah", "working", "on"]);
    }

    #[test]
    fn test_tokenize_preserves_apostrophes() {
        let tokens = tokenize("daughter's school play");
        assert_eq!(tokens, vec!["daughter's", "school", "play"]);
    }

    #[test]
    fn test_entity_matches_single_word() {
        let tokens = tokenize("Sarah discussed the plan with Mike");
        assert!(entity_matches_text("Sarah", &tokens));
        assert!(entity_matches_text("Mike", &tokens));
        assert!(!entity_matches_text("Sara", &tokens)); // partial ≠ match
    }

    #[test]
    fn test_entity_matches_multi_word() {
        let tokens = tokenize("The data pipeline crashed during migration");
        assert!(entity_matches_text("data pipeline", &tokens));
        assert!(!entity_matches_text("data migration", &tokens)); // non-contiguous
    }

    #[test]
    fn test_entity_no_substring_false_positive() {
        let tokens = tokenize("The database was updated successfully");
        // "data" should NOT match inside "database"
        assert!(!entity_matches_text("data", &tokens));
    }

    #[test]
    fn test_entity_matches_case_insensitive() {
        let tokens = tokenize("We evaluated FAISS for vector search");
        assert!(entity_matches_text("FAISS", &tokens));
        assert!(entity_matches_text("faiss", &tokens));
    }

    // ── Entity type classification tests ──

    #[test]
    fn test_classify_person() {
        assert_eq!(classify_entity_type("Sarah"), "person");
        assert_eq!(classify_entity_type("Mike"), "person");
        assert_eq!(classify_entity_type("Tom"), "person");
    }

    #[test]
    fn test_classify_tech_blocklist() {
        assert_eq!(classify_entity_type("FAISS"), "tech");
        assert_eq!(classify_entity_type("ONNX"), "tech");
        assert_eq!(classify_entity_type("Redis"), "tech");
        assert_eq!(classify_entity_type("Python"), "tech");
    }

    #[test]
    fn test_classify_tech_allcaps() {
        assert_eq!(classify_entity_type("GPU"), "tech");
        assert_eq!(classify_entity_type("API"), "tech");
    }

    #[test]
    fn test_classify_unknown() {
        assert_eq!(classify_entity_type("recommendation engine"), "unknown");
        assert_eq!(classify_entity_type("data pipeline"), "unknown");
        assert_eq!(classify_entity_type("sleep patterns"), "unknown");
    }
}
