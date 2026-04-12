use crate::engine::YantrikDB;
use crate::error::Result;
use crate::serde_helpers::{deserialize_f32, serialize_f32};
use crate::types::*;

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let norm_a: f64 = a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Find clusters of related memories using greedy agglomerative approach.
///
/// Two memories can cluster together if:
///   - Embedding similarity >= sim_threshold
///   - Created within time_window_days of each other
pub fn find_clusters(
    memories: &[MemoryWithEmbedding],
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    max_cluster_size: usize,
) -> Vec<Vec<usize>> {
    if memories.len() < min_cluster_size {
        return vec![];
    }

    // Sort by creation time (return indices)
    let mut indices: Vec<usize> = (0..memories.len()).collect();
    indices.sort_by(|&a, &b| {
        memories[a]
            .created_at
            .partial_cmp(&memories[b].created_at)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut used = std::collections::HashSet::new();
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    for &i in &indices {
        if used.contains(&i) {
            continue;
        }

        let mut cluster = vec![i];
        used.insert(i);

        for &j in &indices {
            if j <= i || used.contains(&j) {
                continue;
            }

            // Time proximity check
            let time_diff = (memories[j].created_at - memories[i].created_at).abs();
            if time_diff > time_window_days * 86400.0 {
                continue;
            }

            // Similarity check
            let sim = cosine_similarity(&memories[i].embedding, &memories[j].embedding);
            if sim >= sim_threshold {
                cluster.push(j);
                used.insert(j);

                if cluster.len() >= max_cluster_size {
                    break;
                }
            }
        }

        if cluster.len() >= min_cluster_size {
            clusters.push(cluster);
        }
    }

    clusters
}

/// Generate an extractive summary by selecting the most important memory
/// and combining key facts from the cluster.
pub fn extractive_summary(memories: &[MemoryWithEmbedding]) -> String {
    let mut ranked: Vec<&MemoryWithEmbedding> = memories.iter().collect();
    ranked.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let lead = &ranked[0].text;
    let additional: Vec<&str> = ranked[1..]
        .iter()
        .filter_map(|m| {
            let text = m.text.trim();
            if !text.is_empty() && text != lead.as_str() {
                Some(text)
            } else {
                None
            }
        })
        .collect();

    if additional.is_empty() {
        lead.clone()
    } else {
        let mut parts = vec![lead.as_str()];
        parts.extend(additional);
        parts.join(" | ")
    }
}

/// Compute the mean embedding of a set of memories.
pub fn mean_embedding(memories: &[MemoryWithEmbedding]) -> Vec<f32> {
    let n = memories.len() as f32;
    let dim = memories[0].embedding.len();
    let mut result = vec![0.0f32; dim];
    for mem in memories {
        for (i, &v) in mem.embedding.iter().enumerate() {
            result[i] += v;
        }
    }
    result.iter_mut().for_each(|v| *v /= n);
    result
}

/// Find clusters of memories that are candidates for consolidation.
pub fn find_consolidation_candidates(
    db: &YantrikDB,
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    limit: usize,
) -> Result<Vec<Vec<MemoryWithEmbedding>>> {
    // Phase 1: query rows while holding the conn lock, then drop it.
    // Scope is explicit so the guard CANNOT live across the subsequent
    // calls to db.decrypt_text / db.decrypt_embedding in Phase 2. See
    // CONCURRENCY.md Rule 4: never hold db.conn() across a call taking
    // `&YantrikDB`. decrypt_text/decrypt_embedding don't currently take
    // db.conn(), but a future refactor could, and that silent deadlock
    // would be expensive to find.
    type RawRow = (String, String, String, Vec<u8>, f64, f64, f64, f64, f64, String, String);
    let raw_rows: Vec<RawRow> = {
        let conn = db.conn();
        let sql = format!(
            "SELECT rid, type, text, embedding, created_at, importance, valence, \
             half_life, last_access, metadata, namespace \
             FROM memories \
             WHERE consolidation_status = 'active' \
             AND storage_tier = 'hot' \
             AND type IN ('episodic', 'semantic') \
             LIMIT {}",
            limit
        );
        let mut stmt = conn.prepare(&sql)?;
        let mapped = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>("rid")?,
                row.get::<_, String>("type")?,
                row.get::<_, String>("text")?,
                row.get::<_, Vec<u8>>("embedding")?,
                row.get::<_, f64>("created_at")?,
                row.get::<_, f64>("importance")?,
                row.get::<_, f64>("valence")?,
                row.get::<_, f64>("half_life")?,
                row.get::<_, f64>("last_access")?,
                row.get::<_, String>("metadata")?,
                row.get::<_, String>("namespace")?,
            ))
        })?;
        let collected: std::result::Result<Vec<RawRow>, _> = mapped.collect();
        collected?
    }; // conn, stmt, mapped all dropped here before Phase 2

    // Phase 2: decrypt. Safe to call `db.decrypt_*` now because no conn
    // guard is held.
    let memories: Vec<MemoryWithEmbedding> = raw_rows.into_iter()
        .map(|(rid, memory_type, stored_text, stored_emb, created_at, importance, valence, half_life, last_access, stored_meta, namespace)| {
            let text = db.decrypt_text(&stored_text)?;
            let meta_str = db.decrypt_text(&stored_meta)?;
            let emb_blob = db.decrypt_embedding(&stored_emb)?;
            Ok(MemoryWithEmbedding {
                rid, memory_type, text,
                embedding: deserialize_f32(&emb_blob),
                created_at, importance, valence, half_life, last_access,
                metadata: serde_json::from_str(&meta_str)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
                namespace,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Group memories by namespace to prevent cross-namespace consolidation
    let mut by_namespace: std::collections::HashMap<String, Vec<MemoryWithEmbedding>> =
        std::collections::HashMap::new();
    for mem in memories {
        by_namespace.entry(mem.namespace.clone()).or_default().push(mem);
    }

    let mut result: Vec<Vec<MemoryWithEmbedding>> = Vec::new();
    for (_ns, ns_memories) in by_namespace {
        let cluster_indices = find_clusters(
            &ns_memories,
            sim_threshold,
            time_window_days,
            min_cluster_size,
            10,
        );
        for indices in cluster_indices {
            result.push(indices.into_iter().map(|i| ns_memories[i].clone()).collect());
        }
    }

    Ok(result)
}

/// Run the full consolidation pipeline.
pub fn consolidate(
    db: &YantrikDB,
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    limit: usize,
    dry_run: bool,
) -> Result<Vec<serde_json::Value>> {
    let clusters = find_consolidation_candidates(db, sim_threshold, time_window_days, min_cluster_size, limit)?;

    if dry_run {
        return Ok(clusters
            .iter()
            .map(|cluster| {
                serde_json::json!({
                    "cluster_size": cluster.len(),
                    "texts": cluster.iter().map(|m| m.text.clone()).collect::<Vec<_>>(),
                    "preview_summary": extractive_summary(cluster),
                    "source_rids": cluster.iter().map(|m| m.rid.clone()).collect::<Vec<_>>(),
                })
            })
            .collect());
    }

    let mut results = Vec::new();
    let ts = crate::time::now_secs();

    for cluster in &clusters {
        let source_rids: Vec<String> = cluster.iter().map(|m| m.rid.clone()).collect();

        // 1. Generate summary
        let summary_text = extractive_summary(cluster);

        // 2. Compute mean embedding
        let mean_emb = mean_embedding(cluster);

        // 3. Aggregate importance
        let max_importance = cluster
            .iter()
            .map(|m| m.importance)
            .fold(0.0f64, f64::max);
        let consolidated_importance = (max_importance * 1.1).min(1.0);

        // Mean valence
        let mean_valence: f64 =
            cluster.iter().map(|m| m.valence).sum::<f64>() / cluster.len() as f64;

        // Longer half-life for consolidated memories
        let max_half_life = cluster
            .iter()
            .map(|m| m.half_life)
            .fold(0.0f64, f64::max);
        let consolidated_half_life = max_half_life * 1.5;

        // 4. Record the new consolidated memory
        let meta = serde_json::json!({
            "consolidated_from": source_rids,
            "cluster_size": cluster.len(),
            "consolidation_time": ts,
        });

        let cluster_namespace = cluster.first().map(|m| m.namespace.as_str()).unwrap_or("default");
        let consolidated_rid = db.record(
            &summary_text,
            "semantic",
            consolidated_importance,
            mean_valence,
            consolidated_half_life,
            &meta,
            &mean_emb,
            cluster_namespace,
            0.8,
            "general",
            "user",
            None,
        )?;

        // 5. Transfer entity relationships
        let mut all_entities = std::collections::HashSet::new();
        for mem in cluster {
            let edges = db.get_edges(&mem.rid)?;
            for edge in &edges {
                all_entities.insert(edge.src.clone());
                all_entities.insert(edge.dst.clone());
                if edge.src == mem.rid {
                    db.relate(&consolidated_rid, &edge.dst, &edge.rel_type, edge.weight)?;
                } else if edge.dst == mem.rid {
                    db.relate(&edge.src, &consolidated_rid, &edge.rel_type, edge.weight)?;
                }
            }
        }

        // 6. Insert consolidation_members (set-union CRDT) and mark sources
        {
            let conn = db.conn();
            let hlc_ts = db.tick_hlc();
            let hlc_bytes = hlc_ts.to_bytes().to_vec();
            let actor_id = db.actor_id().to_string();

            for source_rid in &source_rids {
                conn.execute(
                    "INSERT OR IGNORE INTO consolidation_members \
                     (consolidation_rid, source_rid, hlc, actor_id) \
                     VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![consolidated_rid, source_rid, hlc_bytes, actor_id],
                )?;

                conn.execute(
                    "UPDATE memories \
                     SET consolidation_status = 'consolidated', \
                         consolidated_into = ?1, \
                         updated_at = ?2, \
                         importance = importance * 0.3 \
                     WHERE rid = ?3",
                    rusqlite::params![consolidated_rid, ts, source_rid],
                )?;
                // Update scoring cache: mark as consolidated, reduce importance
                db.cache_mark_consolidated(source_rid, 0.3);
            }
        } // conn lock released before log_op

        // 7. Log the operation
        let emb_hash = blake3::hash(&serialize_f32(&mean_emb)).as_bytes().to_vec();
        db.log_op(
            "consolidate",
            Some(&consolidated_rid),
            &serde_json::json!({
                "consolidated_rid": consolidated_rid,
                "source_rids": source_rids,
                "cluster_size": cluster.len(),
                "text": summary_text,
                "importance": consolidated_importance,
                "valence": mean_valence,
                "half_life": consolidated_half_life,
                "metadata": meta,
                "summary_preview": &summary_text[..summary_text.floor_char_boundary(200)],
            }),
            Some(&emb_hash),
        )?;

        results.push(serde_json::json!({
            "consolidated_rid": consolidated_rid,
            "source_rids": source_rids,
            "cluster_size": cluster.len(),
            "summary": summary_text,
            "importance": consolidated_importance,
            "entities_linked": all_entities.len(),
        }));
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mem(rid: &str, text: &str, embedding: Vec<f32>, created_at: f64, importance: f64) -> MemoryWithEmbedding {
        MemoryWithEmbedding {
            rid: rid.to_string(),
            memory_type: "episodic".to_string(),
            text: text.to_string(),
            embedding,
            created_at,
            importance,
            valence: 0.0,
            half_life: 604800.0,
            last_access: created_at,
            metadata: serde_json::json!({}),
            namespace: "default".to_string(),
        }
    }

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim)
            .map(|i| (seed * (i as f32 + 1.0) * 1.7).sin() + (seed * (i as f32 + 2.0) * 0.3).cos())
            .collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec_seed(1.0, 8);
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_find_clusters_basic() {
        let now = 1000000.0;
        let mems = vec![
            make_mem("a", "t1", vec_seed(1.0, 8), now, 0.5),
            make_mem("b", "t2", vec_seed(1.05, 8), now + 100.0, 0.5),
            make_mem("c", "t3", vec_seed(10.0, 8), now + 200.0, 0.5),
        ];

        let clusters = find_clusters(&mems, 0.9, 7.0, 2, 10);
        assert_eq!(clusters.len(), 1);
        assert!(clusters[0].contains(&0)); // "a"
        assert!(clusters[0].contains(&1)); // "b"
    }

    #[test]
    fn test_extractive_summary_single() {
        let mems = vec![make_mem("a", "The cat sat", vec_seed(1.0, 8), 0.0, 0.5)];
        assert_eq!(extractive_summary(&mems), "The cat sat");
    }

    #[test]
    fn test_extractive_summary_multi() {
        let mems = vec![
            make_mem("a", "Low importance", vec_seed(1.0, 8), 0.0, 0.1),
            make_mem("b", "High importance lead", vec_seed(2.0, 8), 0.0, 0.9),
        ];
        let summary = extractive_summary(&mems);
        assert!(summary.starts_with("High importance lead"));
    }

    #[test]
    fn test_mean_embedding() {
        let mems = vec![
            make_mem("a", "t1", vec![1.0, 2.0, 3.0], 0.0, 0.5),
            make_mem("b", "t2", vec![3.0, 4.0, 5.0], 0.0, 0.5),
        ];
        let mean = mean_embedding(&mems);
        assert_eq!(mean, vec![2.0, 3.0, 4.0]);
    }
}
