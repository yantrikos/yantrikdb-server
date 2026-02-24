//! Shared utilities for benchmarks and profiling.

use crate::{AIDB, RecordInput};

/// Generate a deterministic unit-norm embedding of given dimension.
///
/// Uses sin() for better angular spread at high dimensions (the linear ramp
/// in the original `vec_seed` collapses when dim >> 64).
pub fn vec_seed_dim(seed: f32, dim: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim)
        .map(|i| ((seed + i as f32) * 0.7123 + (i as f32) * 0.3171).sin())
        .collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vec![1.0 / (dim as f32).sqrt(); dim];
    }
    raw.iter().map(|x| x / norm).collect()
}

/// Standard query embedding for benchmarks.
pub fn query_embedding(dim: usize) -> Vec<f32> {
    vec_seed_dim(999.0, dim)
}

/// Seed a database with N memories at given dimension using record_batch.
///
/// When `with_graph` is true, creates entity relationships and memory_entities
/// links for graph expansion benchmarking.
pub fn seed_db_scaled(db: &AIDB, n: usize, dim: usize, with_graph: bool) {
    let batch_size = 500;
    let num_entities = (n / 100).max(10).min(200);
    let entity_names: Vec<String> = (0..num_entities)
        .map(|i| format!("Entity_{}", i))
        .collect();

    // Insert memories in batches
    for chunk_start in (0..n).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(n);
        let inputs: Vec<RecordInput> = (chunk_start..chunk_end)
            .map(|i| {
                let topic = i % 20;
                let entity_idx = i % entity_names.len();
                RecordInput {
                    text: format!(
                        "Memory {} about topic {} involving {} in context {}",
                        i, topic, entity_names[entity_idx], i % 50
                    ),
                    memory_type: match i % 4 {
                        0 => "episodic",
                        1 => "semantic",
                        2 => "procedural",
                        _ => "emotional",
                    }
                    .to_string(),
                    importance: 0.2 + (i % 9) as f64 * 0.1,
                    valence: (i % 5) as f64 * 0.2 - 0.4,
                    half_life: 604800.0 * (1 + i % 4) as f64,
                    metadata: serde_json::json!({
                        "topic": topic,
                        "source": format!("source_{}", i % 5),
                    }),
                    embedding: vec_seed_dim(i as f32 * 0.37, dim),
                    namespace: "default".to_string(),
                }
            })
            .collect();
        db.record_batch(&inputs).unwrap();
    }

    if with_graph {
        // Create entity graph: ring + skip connections for small-world topology
        for i in 0..entity_names.len() {
            let next = (i + 1) % entity_names.len();
            db.relate(&entity_names[i], &entity_names[next], "related_to", 0.8)
                .unwrap();
            if i % 3 == 0 && entity_names.len() > 3 {
                let skip = (i + entity_names.len() / 3) % entity_names.len();
                db.relate(&entity_names[i], &entity_names[skip], "associated_with", 0.5)
                    .unwrap();
            }
        }

        // Link memories to entities via memory_entities join table
        // Batch query RIDs and link to entities
        let conn = db.conn();
        let total: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .unwrap();
        let mut offset = 0i64;
        while offset < total {
            let batch = 1000i64.min(total - offset);
            let mut stmt = conn
                .prepare("SELECT rid FROM memories ORDER BY created_at LIMIT ?1 OFFSET ?2")
                .unwrap();
            let rids: Vec<String> = stmt
                .query_map(rusqlite::params![batch, offset], |row| row.get(0))
                .unwrap()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap();

            for (j, rid) in rids.iter().enumerate() {
                let i = offset as usize + j;
                let entity_idx = i % entity_names.len();
                db.link_memory_entity(rid, &entity_names[entity_idx])
                    .unwrap();
                // ~30% of memories get a second entity link
                if i % 3 == 0 {
                    let alt = (entity_idx + 1) % entity_names.len();
                    db.link_memory_entity(rid, &entity_names[alt]).unwrap();
                }
            }
            offset += batch;
        }
    }
}
