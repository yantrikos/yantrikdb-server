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

// ── Heuristic proper-noun extraction ──

/// English function/pronoun/auxiliary words that should be stripped from the
/// start or end of a capitalized chunk. A sentence-initial "The" or "Our" is
/// capitalized by position, not because it names an entity.
const ENTITY_STOPWORDS: &[&str] = &[
    "The", "A", "An", "I", "We", "You", "He", "She", "It", "They",
    "This", "That", "These", "Those", "My", "Your", "His", "Her",
    "Its", "Our", "Their", "But", "And", "Or", "So", "If", "When",
    "Where", "What", "Who", "Why", "How", "Is", "Are", "Was", "Were",
    "Be", "Been", "Being", "Have", "Has", "Had", "Do", "Does", "Did",
    "Of", "In", "On", "At", "To", "For", "With", "From", "By",
    "As", "Than", "Then", "Also", "Just", "Only", "Very", "Much",
];

/// Extract candidate proper-noun entities from free-form text using a
/// capitalized-chunk heuristic. Groups consecutive capitalized words into
/// multi-word entities ("Alice Chen", "San Francisco", "Acme Corp") and
/// strips leading/trailing English stopwords.
///
/// This is intentionally not a full NER — it captures the common case of
/// people, companies, places, and products well enough that conflict
/// detection can fire without requiring users to call `/v1/relate` for every
/// entity. Acronyms, lowercase entities, and ambiguous mentions still need
/// explicit `relate()` calls to enter the graph.
pub fn extract_heuristic_entities(text: &str) -> Vec<String> {
    let mut entities: Vec<String> = Vec::new();
    let mut chunk: Vec<String> = Vec::new();

    let flush = |chunk: &mut Vec<String>, out: &mut Vec<String>| {
        while !chunk.is_empty() && ENTITY_STOPWORDS.contains(&chunk[0].as_str()) {
            chunk.remove(0);
        }
        // Trailing-stopword strip skips single-character tokens so multi-word
        // entities like "Series A" or "Version B" keep their letter suffix
        // (A is a stopword but is also a valid version designator when trailing).
        while let Some(last) = chunk.last() {
            if ENTITY_STOPWORDS.contains(&last.as_str()) && last.chars().count() > 1 {
                chunk.pop();
            } else {
                break;
            }
        }
        if !chunk.is_empty() {
            let candidate = chunk.join(" ");
            let alpha_chars = candidate.chars().filter(|c| c.is_alphanumeric()).count();
            if alpha_chars >= 2 {
                out.push(candidate);
            }
        }
        chunk.clear();
    };

    for word in text
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
    {
        let first = word.chars().next().unwrap();
        let starts_upper = first.is_uppercase();
        let is_all_caps = word.len() > 1 && word.chars().all(|c| !c.is_alphabetic() || c.is_uppercase());

        let joins_chunk = if chunk.is_empty() {
            // Open a new chunk only on capitalized or all-caps tokens.
            starts_upper || is_all_caps
        } else {
            // Continue an existing chunk on capitalized words or short letter-suffixes
            // (e.g., "Series A", "Version B").
            starts_upper
                || is_all_caps
                || (word.len() == 1 && first.is_ascii_uppercase())
        };

        if joins_chunk {
            chunk.push(word.to_string());
        } else {
            flush(&mut chunk, &mut entities);
        }
    }
    flush(&mut chunk, &mut entities);

    // Deduplicate while preserving first-appearance order.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    entities.retain(|e| seen.insert(e.clone()));
    entities
}

// ── Entity type classification ──

/// Tech terms that should NOT be classified as person names even if title-cased/all-caps.
const TECH_BLOCKLIST: &[&str] = &[
    "faiss", "onnx", "scann", "redis", "kafka", "docker", "kubernetes", "react",
    "python", "rust", "java", "swift", "flutter", "pytorch", "tensorflow",
    "numpy", "pandas", "spark", "hadoop", "nginx", "postgres", "mysql",
    "sqlite", "graphql", "grpc", "oauth", "jwt", "html", "css",
    "api", "sdk", "ml", "ai", "gpu", "cpu", "ram", "ssd", "aws", "gcp",
    "claude", "openai", "anthropic", "gemini", "llama", "ollama",
];

/// Words that indicate the entity is NOT a person when used as first word.
const NON_PERSON_PREFIXES: &[&str] = &[
    "project", "team", "company", "group", "department", "org", "the",
    "operation", "task", "plan", "system", "service", "app", "tool",
    "code", "server", "client", "api", "db", "database", "agent",
    "model", "version", "release", "build", "deploy", "config",
];

/// Classify an entity name into a type: "person", "tech", or "unknown".
/// This is a name-only heuristic — prefer `classify_with_relationship()` when
/// relationship context is available.
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

    // Multi-word title-case (e.g., "Priya Sharma", "Sarah Chen") → likely person
    // But NOT if the first word is a non-person prefix (e.g., "Project Athena", "Claude Code")
    if trimmed.contains(' ') {
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if words.len() == 2
            && words
                .iter()
                .all(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
        {
            let first_lower = words[0].to_lowercase();
            if NON_PERSON_PREFIXES.contains(&first_lower.as_str()) {
                return "unknown";
            }
            // Also reject if any word is in tech blocklist
            if words.iter().any(|w| TECH_BLOCKLIST.contains(&w.to_lowercase().as_str())) {
                return "tech";
            }
            return "person";
        }
    }

    // Single-word classification is unreliable (Bangalore, Flipkart, Arjun all
    // look the same). Return "unknown" and let relationship context decide.
    "unknown"
}

/// Relationship types that imply both src and dst are persons.
const PERSON_PERSON_RELS: &[&str] = &[
    "married_to", "mother_of", "father_of", "daughter_of", "son_of",
    "sister_of", "brother_of", "sibling_of", "parent_of", "child_of",
    "knows", "friends_with", "met", "dating", "engaged_to",
    "mentors", "mentored_by", "reports_to", "manages",
    "colleagues", "roommate", "neighbor",
    "called", "texted", "messaged", "date_night",
];

/// Relationship types where dst is a place.
const PLACE_DST_RELS: &[&str] = &[
    "lives_in", "born_in", "grew_up_in", "located_in", "based_in",
    "visited", "moved_to", "traveled_to", "from",
];

/// Relationship types where dst is an organization / institution.
const ORG_DST_RELS: &[&str] = &[
    "works_at", "works_for", "employed_at", "employed_by",
    "studied_at", "attended", "enrolled_in", "graduated_from",
    "member_of", "belongs_to", "founded",
];

/// Relationship types where dst is tech/tool (src is project or person).
const TECH_DST_RELS: &[&str] = &[
    "built_with", "uses", "depends_on", "integrates", "requires",
    "written_in", "coded_in", "implemented_with", "powered_by",
    "runs_on", "compiled_with",
];

/// Relationship types where dst is infrastructure.
const INFRA_DST_RELS: &[&str] = &[
    "deployed_on", "hosted_on", "deployed_to", "hosted_at",
    "runs_on_infra", "served_by",
];

/// Relationship types where src is a person and dst is a project/thing.
const PERSON_PROJECT_RELS: &[&str] = &[
    "works_on", "contributes_to", "maintains", "leads", "created",
    "built", "designed", "architected", "owns",
];

/// Relationship types where src is a project and dst is a project (dependency).
const PROJECT_PROJECT_RELS: &[&str] = &[
    "depends_on_project", "extends", "forks", "replaces",
    "supersedes", "derived_from",
];

/// Relationship types where dst is an event or activity.
const EVENT_DST_RELS: &[&str] = &[
    "attended_event", "participated_in", "scheduled_for",
    "presented_at", "spoke_at",
];

/// Relationship types where dst is a concept/topic.
const CONCEPT_DST_RELS: &[&str] = &[
    "interested_in", "studies", "researches", "specializes_in",
    "expert_in", "learning", "teaches",
];

/// Classify entity types using relationship semantics.
/// Returns (src_type, dst_type) — either may be "unknown" if not inferable.
pub fn classify_with_relationship(
    src: &str,
    dst: &str,
    rel_type: &str,
) -> (&'static str, &'static str) {
    let rel_lower = rel_type.to_lowercase();
    let rel = rel_lower.as_str();

    // Person-person relationships
    if PERSON_PERSON_RELS.contains(&rel) {
        return ("person", "person");
    }

    // Person → Place relationships
    if PLACE_DST_RELS.contains(&rel) {
        return ("person", "place");
    }

    // Person → Organization relationships
    if ORG_DST_RELS.contains(&rel) {
        return ("person", "organization");
    }

    // * → Tech/Tool relationships (src type from name heuristic)
    if TECH_DST_RELS.contains(&rel) {
        let src_type = classify_entity_type(src);
        return (if src_type == "unknown" { "project" } else { src_type }, "tech");
    }

    // * → Infrastructure relationships
    if INFRA_DST_RELS.contains(&rel) {
        let src_type = classify_entity_type(src);
        return (if src_type == "unknown" { "project" } else { src_type }, "infrastructure");
    }

    // Person → Project relationships
    if PERSON_PROJECT_RELS.contains(&rel) {
        return ("person", "project");
    }

    // Project → Project relationships
    if PROJECT_PROJECT_RELS.contains(&rel) {
        return ("project", "project");
    }

    // * → Event relationships
    if EVENT_DST_RELS.contains(&rel) {
        return (classify_entity_type(src), "event");
    }

    // Person → Concept/Topic relationships
    if CONCEPT_DST_RELS.contains(&rel) {
        return ("person", "concept");
    }

    // Fall back to name-based heuristics
    (classify_entity_type(src), classify_entity_type(dst))
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
    use crate::YantrikDB;

    #[test]
    fn test_extract_heuristic_entities_basic_names() {
        let got = extract_heuristic_entities("Alice Chen is the CEO of Acme Corp");
        assert!(got.contains(&"Alice Chen".to_string()), "got: {:?}", got);
        assert!(got.contains(&"Acme Corp".to_string()), "got: {:?}", got);
        // CEO is all-caps standalone — should appear as an entity candidate.
        assert!(got.contains(&"CEO".to_string()), "got: {:?}", got);
    }

    #[test]
    fn test_extract_heuristic_entities_strips_sentence_start() {
        let got = extract_heuristic_entities("The database backend is PostgreSQL");
        assert_eq!(got, vec!["PostgreSQL".to_string()]);
    }

    #[test]
    fn test_extract_heuristic_entities_multi_word_place() {
        let got = extract_heuristic_entities("Acme is headquartered in San Francisco");
        assert!(got.contains(&"Acme".to_string()), "got: {:?}", got);
        assert!(got.contains(&"San Francisco".to_string()), "got: {:?}", got);
    }

    #[test]
    fn test_extract_heuristic_entities_single_letter_suffix() {
        let got = extract_heuristic_entities("Series A funding was 20 million dollars");
        assert!(got.contains(&"Series A".to_string()), "got: {:?}", got);
    }

    #[test]
    fn test_extract_heuristic_entities_dedupe() {
        let got = extract_heuristic_entities("Alice met Alice at the cafe");
        let alice_count = got.iter().filter(|e| *e == "Alice").count();
        assert_eq!(alice_count, 1);
    }

    #[test]
    fn test_extract_heuristic_entities_empty_on_lowercase() {
        let got = extract_heuristic_entities("the quick brown fox jumps over the lazy dog");
        assert!(got.is_empty(), "got: {:?}", got);
    }

    #[test]
    fn test_extract_heuristic_entities_distinct_people() {
        // Regression guard for the false-merge case that motivated this:
        // two sentences structurally similar but referring to different people.
        let a = extract_heuristic_entities("Alice Chen is the CEO of Acme Corp");
        let b = extract_heuristic_entities("Sarah Kim is the CTO of Acme Corp");
        let a_set: std::collections::HashSet<_> = a.iter().collect();
        let b_set: std::collections::HashSet<_> = b.iter().collect();
        // They share Acme Corp but differ on person name — disjointness on people.
        assert!(a_set.contains(&"Alice Chen".to_string()));
        assert!(b_set.contains(&"Sarah Kim".to_string()));
        assert!(!a_set.contains(&"Sarah Kim".to_string()));
        assert!(!b_set.contains(&"Alice Chen".to_string()));
    }

    fn setup_db() -> YantrikDB {
        let db = YantrikDB::new(":memory:", 4).unwrap();
        // Create entities and edges
        db.relate("Alice", "Bob", "knows", 1.0).unwrap();
        db.relate("Bob", "Charlie", "knows", 0.8).unwrap();
        db.relate("Alice", "ProjectX", "works_on", 1.0).unwrap();
        db.relate("Dave", "ProjectX", "works_on", 0.9).unwrap();

        // Record memories and link to entities
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
    fn test_entities_for_memories() {
        let db = setup_db();
        // Get the first memory's rid
        let rid: String = db.conn().query_row(
            "SELECT rid FROM memories ORDER BY created_at LIMIT 1", [], |row| row.get(0),
        ).unwrap();

        let entities = entities_for_memories(&*db.conn(), &[&rid]).unwrap();
        assert!(entities.contains(&"Alice".to_string()));
        assert!(entities.contains(&"ProjectX".to_string()));
    }

    #[test]
    fn test_memories_for_entities() {
        let db = setup_db();
        let rids = memories_for_entities(&*db.conn(), &["Alice"]).unwrap();
        assert_eq!(rids.len(), 1); // Only the Alice memory is linked
    }

    #[test]
    fn test_expand_1hop() {
        let db = setup_db();
        let expanded = expand_entities_nhop(&*db.conn(), &["Alice"], 1, 30).unwrap();
        let names: HashSet<String> = expanded.iter().map(|(n, _, _)| n.clone()).collect();
        // Alice (seed) + Bob (knows) + ProjectX (works_on)
        assert!(names.contains("Alice"));
        assert!(names.contains("Bob"));
        assert!(names.contains("ProjectX"));
    }

    #[test]
    fn test_expand_2hop() {
        let db = setup_db();
        let expanded = expand_entities_nhop(&*db.conn(), &["Alice"], 2, 30).unwrap();
        let names: HashSet<String> = expanded.iter().map(|(n, _, _)| n.clone()).collect();
        // 2-hop from Alice: Alice->Bob->Charlie, Alice->ProjectX->Dave
        assert!(names.contains("Charlie"));
        assert!(names.contains("Dave"));
    }

    #[test]
    fn test_expand_budget_limit() {
        let db = setup_db();
        let expanded = expand_entities_nhop(&*db.conn(), &["Alice"], 2, 3).unwrap();
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
        let expanded = expand_entities_nhop(&*db.conn(), &["Alice"], 1, 30).unwrap();
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

        let prox = graph_proximity(&*db.conn(), &rid, &expanded).unwrap();
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
    fn test_classify_name_only_ambiguous() {
        // Single-word title-case is now "unknown" without relationship context
        assert_eq!(classify_entity_type("Sarah"), "unknown");
        assert_eq!(classify_entity_type("Bangalore"), "unknown");
        assert_eq!(classify_entity_type("Flipkart"), "unknown");
    }

    #[test]
    fn test_classify_name_multi_word_person() {
        // Multi-word title-case full names are still "person"
        assert_eq!(classify_entity_type("Sarah Chen"), "person");
        assert_eq!(classify_entity_type("Priya Sharma"), "person");
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

    // ── Relationship-based classification tests ──

    #[test]
    fn test_classify_with_rel_person_person() {
        let (s, d) = classify_with_relationship("Arjun", "Priya", "married_to");
        assert_eq!(s, "person");
        assert_eq!(d, "person");
    }

    #[test]
    fn test_classify_with_rel_person_place() {
        let (s, d) = classify_with_relationship("Priya", "Bangalore", "lives_in");
        assert_eq!(s, "person");
        assert_eq!(d, "place");
    }

    #[test]
    fn test_classify_with_rel_person_org() {
        let (s, d) = classify_with_relationship("Priya", "Flipkart", "works_at");
        assert_eq!(s, "person");
        assert_eq!(d, "organization");
    }

    #[test]
    fn test_classify_with_rel_tech_dst() {
        // "uses" implies dst is tech; FAISS is tech by name heuristic
        let (s, d) = classify_with_relationship("FAISS", "data pipeline", "uses");
        assert_eq!(s, "tech");
        assert_eq!(d, "tech");
    }

    #[test]
    fn test_classify_with_rel_built_with() {
        // "built_with" → src defaults to "project" if unknown, dst is tech
        let (s, d) = classify_with_relationship("MyApp", "React", "built_with");
        assert_eq!(s, "project");
        assert_eq!(d, "tech");
    }

    #[test]
    fn test_classify_with_rel_deployed_on() {
        let (s, d) = classify_with_relationship("MyApp", "AWS", "deployed_on");
        assert_eq!(s, "project");
        assert_eq!(d, "infrastructure");
    }

    #[test]
    fn test_classify_with_rel_works_on() {
        let (s, d) = classify_with_relationship("Pranab", "YantrikDB", "works_on");
        assert_eq!(s, "person");
        assert_eq!(d, "project");
    }

    #[test]
    fn test_classify_with_rel_fallback() {
        // Truly unknown relationship → falls back to name heuristics
        let (s, d) = classify_with_relationship("FAISS", "data pipeline", "related_to");
        assert_eq!(s, "tech");
        assert_eq!(d, "unknown");
    }
}
