//! Rust-native HNSW (Hierarchical Navigable Small World) vector index.
//!
//! Implements the Malkov-Yashunin algorithm with cosine distance,
//! incremental insert, tombstone-based deletion, and configurable parameters.
//!
//! This is a purpose-built index for AIDB's cognitive memory engine —
//! single-threaded, derived from SQLite as source of truth, and rebuilt on startup.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::error::Result;

// ── Distance functions ──

/// Cosine distance: 1.0 - cosine_similarity.
/// For normalized vectors, this equals 1.0 - dot_product.
#[inline]
fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let norm_a: f64 = a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    // Clamp to [0.0, 2.0] to handle floating-point rounding
    (1.0 - (dot / (norm_a * norm_b))).clamp(0.0, 2.0)
}

// ── Heap helpers ──

/// An entry in the candidate/result heaps, ordered by distance.
#[derive(Clone)]
struct Candidate {
    idx: usize,
    distance: f64,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for Candidate {}

/// Min-heap ordering (smallest distance first).
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: BinaryHeap is a max-heap, so reverse for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Max-heap entry (largest distance first) for maintaining top-k.
#[derive(Clone)]
struct FarCandidate {
    idx: usize,
    distance: f64,
}

impl PartialEq for FarCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for FarCandidate {}

impl Ord for FarCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for FarCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── Node ──

struct HnswNode {
    /// The embedding vector.
    embedding: Vec<f32>,
    /// Connections per layer: neighbors[layer] = vec of neighbor indices.
    neighbors: Vec<Vec<usize>>,
    /// Whether this node has been deleted.
    tombstoned: bool,
}

// ── HnswIndex ──

/// A Rust-native HNSW vector index.
pub struct HnswIndex {
    dim: usize,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    ml: f64,
    entry_point: Option<usize>,
    max_layer: usize,
    nodes: Vec<HnswNode>,
    rid_to_idx: HashMap<String, usize>,
    idx_to_rid: Vec<String>,
    free_list: Vec<usize>,
    active_count: usize,
    rng: SmallRng,
}

impl HnswIndex {
    /// Create a new HNSW index with default parameters.
    pub fn new(dim: usize) -> Self {
        Self::with_params(dim, 16, 200, 50)
    }

    /// Create a new HNSW index with custom parameters.
    pub fn with_params(dim: usize, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            dim,
            m,
            m_max0: m * 2,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
            entry_point: None,
            max_layer: 0,
            nodes: Vec::new(),
            rid_to_idx: HashMap::new(),
            idx_to_rid: Vec::new(),
            free_list: Vec::new(),
            active_count: 0,
            rng: SmallRng::from_entropy(),
        }
    }

    /// Number of active (non-tombstoned) entries.
    pub fn len(&self) -> usize {
        self.active_count
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.active_count == 0
    }

    /// Insert a vector keyed by rid.
    pub fn insert(&mut self, rid: &str, embedding: &[f32]) -> Result<()> {
        assert_eq!(embedding.len(), self.dim, "embedding dimension mismatch");

        // If rid already exists and is tombstoned, resurrect it
        if let Some(&existing_idx) = self.rid_to_idx.get(rid) {
            let node = &mut self.nodes[existing_idx];
            if node.tombstoned {
                node.embedding = embedding.to_vec();
                node.tombstoned = false;
                self.active_count += 1;
                // Re-connect by inserting into the graph at its existing layers
                let level = node.neighbors.len().saturating_sub(1);
                self.connect_node(existing_idx, level);
                return Ok(());
            }
            // Already exists and active — update embedding in-place
            node.embedding = embedding.to_vec();
            return Ok(());
        }

        // Assign a random level
        let level = self.random_level();

        // Allocate node index
        let idx = if let Some(free_idx) = self.free_list.pop() {
            // Reuse a freed slot
            let neighbors = (0..=level).map(|_| Vec::new()).collect();
            self.nodes[free_idx] = HnswNode {
                embedding: embedding.to_vec(),
                neighbors,
                tombstoned: false,
            };
            self.idx_to_rid[free_idx] = rid.to_string();
            free_idx
        } else {
            // Append new slot
            let neighbors = (0..=level).map(|_| Vec::new()).collect();
            self.nodes.push(HnswNode {
                embedding: embedding.to_vec(),
                neighbors,
                tombstoned: false,
            });
            self.idx_to_rid.push(rid.to_string());
            self.nodes.len() - 1
        };

        self.rid_to_idx.insert(rid.to_string(), idx);
        self.active_count += 1;

        // First node: set as entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.max_layer = level;
            return Ok(());
        }

        self.connect_node(idx, level);

        // Update entry point if this node has a higher level
        if level > self.max_layer {
            self.entry_point = Some(idx);
            self.max_layer = level;
        }

        Ok(())
    }

    /// Connect a node into the graph at the given level.
    fn connect_node(&mut self, idx: usize, level: usize) {
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return,
        };

        let query = self.nodes[idx].embedding.clone();
        let mut current_ep = ep;

        // Phase 1: Greedy descent from top layer to level+1
        for lc in (level + 1..=self.max_layer).rev() {
            current_ep = self.greedy_closest(&query, current_ep, lc);
        }

        // Phase 2: Insert at layers min(level, max_layer) down to 0
        let insert_top = level.min(self.max_layer);
        let mut ep_candidates = vec![current_ep];

        for lc in (0..=insert_top).rev() {
            let max_m = if lc == 0 { self.m_max0 } else { self.m };
            let ef = self.ef_construction;

            // Search for neighbors at this layer
            let nearest = self.search_layer(&query, &ep_candidates, ef, lc, Some(idx));

            // Select top M neighbors
            let selected: Vec<usize> = nearest
                .iter()
                .take(max_m)
                .map(|c| c.idx)
                .collect();

            // Connect bidirectionally
            self.nodes[idx].neighbors[lc] = selected.clone();
            for &neighbor_idx in &selected {
                let neighbor = &mut self.nodes[neighbor_idx];
                if neighbor.neighbors.len() > lc {
                    neighbor.neighbors[lc].push(idx);
                    // Prune if over capacity
                    if neighbor.neighbors[lc].len() > max_m {
                        self.prune_neighbors(neighbor_idx, lc, max_m);
                    }
                }
            }

            // Propagate entry points for next layer
            ep_candidates = selected;
            if ep_candidates.is_empty() {
                ep_candidates = vec![current_ep];
            }
        }
    }

    /// Prune a node's neighbors at a given layer to max_m connections.
    fn prune_neighbors(&mut self, node_idx: usize, layer: usize, max_m: usize) {
        let node_emb = self.nodes[node_idx].embedding.clone();
        let mut neighbors_with_dist: Vec<(usize, f64)> = self.nodes[node_idx].neighbors[layer]
            .iter()
            .filter(|&&n| !self.nodes[n].tombstoned)
            .map(|&n| (n, cosine_distance(&node_emb, &self.nodes[n].embedding)))
            .collect();
        neighbors_with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        neighbors_with_dist.truncate(max_m);
        self.nodes[node_idx].neighbors[layer] =
            neighbors_with_dist.iter().map(|&(n, _)| n).collect();
    }

    /// Search for the k nearest neighbors of query.
    /// Returns (rid, distance) pairs sorted by distance (ascending).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f64)>> {
        if self.active_count == 0 || self.entry_point.is_none() {
            return Ok(vec![]);
        }

        let ep = self.entry_point.unwrap();
        let mut current_ep = ep;

        // Phase 1: Greedy descent from top layer to layer 1
        for lc in (1..=self.max_layer).rev() {
            current_ep = self.greedy_closest(query, current_ep, lc);
        }

        // Phase 2: Search layer 0 with ef_search candidates
        let nearest = self.search_layer(query, &[current_ep], self.ef_search, 0, None);

        // Return top-k non-tombstoned results
        let mut results: Vec<(String, f64)> = Vec::with_capacity(k);
        for c in &nearest {
            if !self.nodes[c.idx].tombstoned {
                results.push((self.idx_to_rid[c.idx].clone(), c.distance));
                if results.len() >= k {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Remove a vector by rid (tombstone-based).
    pub fn remove(&mut self, rid: &str) -> bool {
        if let Some(&idx) = self.rid_to_idx.get(rid) {
            if !self.nodes[idx].tombstoned {
                self.nodes[idx].tombstoned = true;
                self.active_count -= 1;
                self.free_list.push(idx);
                return true;
            }
        }
        false
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.rid_to_idx.clear();
        self.idx_to_rid.clear();
        self.free_list.clear();
        self.entry_point = None;
        self.max_layer = 0;
        self.active_count = 0;
    }

    // ── Internal helpers ──

    /// Assign a random level for a new node.
    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        let level = (-r.ln() * self.ml).floor() as usize;
        level.min(32) // Cap at 32 layers
    }

    /// Greedy descent: find the closest node to query at a given layer.
    fn greedy_closest(&self, query: &[f32], entry: usize, layer: usize) -> usize {
        let mut current = entry;
        let mut current_dist = cosine_distance(query, &self.nodes[current].embedding);

        loop {
            let mut changed = false;
            if layer < self.nodes[current].neighbors.len() {
                for &neighbor in &self.nodes[current].neighbors[layer] {
                    if neighbor >= self.nodes.len() || self.nodes[neighbor].tombstoned {
                        continue;
                    }
                    let dist = cosine_distance(query, &self.nodes[neighbor].embedding);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    /// Search a single layer starting from entry points.
    /// Returns candidates sorted by distance (ascending).
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        exclude_idx: Option<usize>,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        // Min-heap of candidates to explore
        let mut candidates = BinaryHeap::new();
        // Max-heap of best results so far
        let mut results = BinaryHeap::<FarCandidate>::new();

        for &ep in entry_points {
            if ep >= self.nodes.len() || visited.contains(&ep) {
                continue;
            }
            visited.insert(ep);
            let dist = cosine_distance(query, &self.nodes[ep].embedding);

            if exclude_idx != Some(ep) && !self.nodes[ep].tombstoned {
                candidates.push(Candidate {
                    idx: ep,
                    distance: dist,
                });
                results.push(FarCandidate {
                    idx: ep,
                    distance: dist,
                });
            } else {
                // Still add to candidates for traversal but not to results
                candidates.push(Candidate {
                    idx: ep,
                    distance: dist,
                });
            }
        }

        while let Some(closest) = candidates.pop() {
            // Check if the closest candidate is farther than the worst result
            let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f64::MAX);
            if closest.distance > worst_dist && results.len() >= ef {
                break;
            }

            // Expand neighbors at this layer
            let node = &self.nodes[closest.idx];
            if layer < node.neighbors.len() {
                for &neighbor in &node.neighbors[layer] {
                    if neighbor >= self.nodes.len() || visited.contains(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);
                    let dist = cosine_distance(query, &self.nodes[neighbor].embedding);
                    let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f64::MAX);

                    if dist < worst_dist || results.len() < ef {
                        candidates.push(Candidate {
                            idx: neighbor,
                            distance: dist,
                        });

                        if exclude_idx != Some(neighbor) && !self.nodes[neighbor].tombstoned {
                            results.push(FarCandidate {
                                idx: neighbor,
                                distance: dist,
                            });
                            if results.len() > ef {
                                results.pop(); // Remove farthest
                            }
                        }
                    }
                }
            }
        }

        // Convert max-heap to sorted vec (ascending distance)
        let mut sorted: Vec<Candidate> = results
            .into_iter()
            .map(|fc| Candidate {
                idx: fc.idx,
                distance: fc.distance,
            })
            .collect();
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        sorted
    }
}

// ── BruteForceIndex (correctness oracle for testing) ──

/// Brute-force vector index for testing HNSW recall quality.
pub struct BruteForceIndex {
    dim: usize,
    entries: Vec<(String, Vec<f32>, bool)>, // (rid, embedding, tombstoned)
    rid_to_idx: HashMap<String, usize>,
}

impl BruteForceIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: Vec::new(),
            rid_to_idx: HashMap::new(),
        }
    }

    pub fn insert(&mut self, rid: &str, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dim);
        if let Some(&idx) = self.rid_to_idx.get(rid) {
            self.entries[idx].1 = embedding.to_vec();
            self.entries[idx].2 = false;
        } else {
            let idx = self.entries.len();
            self.entries.push((rid.to_string(), embedding.to_vec(), false));
            self.rid_to_idx.insert(rid.to_string(), idx);
        }
    }

    pub fn remove(&mut self, rid: &str) -> bool {
        if let Some(&idx) = self.rid_to_idx.get(rid) {
            if !self.entries[idx].2 {
                self.entries[idx].2 = true;
                return true;
            }
        }
        false
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = self
            .entries
            .iter()
            .filter(|(_, _, tombstoned)| !tombstoned)
            .map(|(rid, emb, _)| (rid.clone(), cosine_distance(query, emb)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.truncate(k);
        scored
    }

    pub fn len(&self) -> usize {
        self.entries.iter().filter(|(_, _, t)| !t).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a deterministic unit-norm embedding.
    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim)
            .map(|i| ((seed + i as f32) * 0.7123 + (i as f32) * 0.3171).sin())
            .collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return vec![1.0 / (dim as f32).sqrt(); dim];
        }
        raw.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_cosine_distance_identical() {
        let v = vec_seed(1.0, 8);
        let d = cosine_distance(&v, &v);
        assert!(d.abs() < 1e-6, "distance to self should be ~0, got {d}");
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        let d = cosine_distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-6, "orthogonal distance should be ~1, got {d}");
    }

    #[test]
    fn test_empty_index() {
        let index = HnswIndex::new(8);
        let results = index.search(&vec_seed(1.0, 8), 10).unwrap();
        assert!(results.is_empty());
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_single_insert_search() {
        let mut index = HnswIndex::new(8);
        index.insert("a", &vec_seed(1.0, 8)).unwrap();
        assert_eq!(index.len(), 1);

        let results = index.search(&vec_seed(1.0, 8), 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 < 1e-6); // Distance to self should be ~0
    }

    #[test]
    fn test_insert_search_nearest() {
        let dim = 64;
        let mut index = HnswIndex::new(dim);

        // Insert 100 vectors
        for i in 0..100 {
            index.insert(&format!("v{i}"), &vec_seed(i as f32 * 0.37, dim)).unwrap();
        }
        assert_eq!(index.len(), 100);

        // Search for the nearest to vec_seed(0.0, dim) — should be "v0"
        let query = vec_seed(0.0, dim);
        let results = index.search(&query, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "v0");
    }

    #[test]
    fn test_tombstone_excludes_from_search() {
        let dim = 8;
        let mut index = HnswIndex::new(dim);

        index.insert("a", &vec_seed(1.0, dim)).unwrap();
        index.insert("b", &vec_seed(2.0, dim)).unwrap();
        assert_eq!(index.len(), 2);

        // Remove "a"
        assert!(index.remove("a"));
        assert_eq!(index.len(), 1);

        // Search should not return "a"
        let results = index.search(&vec_seed(1.0, dim), 10).unwrap();
        assert!(!results.iter().any(|(rid, _)| rid == "a"));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut index = HnswIndex::new(8);
        assert!(!index.remove("nonexistent"));
    }

    #[test]
    fn test_free_list_reuse() {
        let dim = 8;
        let mut index = HnswIndex::new(dim);

        index.insert("a", &vec_seed(1.0, dim)).unwrap();
        let initial_nodes = index.nodes.len();

        index.remove("a");
        index.insert("b", &vec_seed(2.0, dim)).unwrap();

        // Should reuse the freed slot
        assert_eq!(index.nodes.len(), initial_nodes);
        assert_eq!(index.len(), 1);

        let results = index.search(&vec_seed(2.0, dim), 10).unwrap();
        assert_eq!(results[0].0, "b");
    }

    #[test]
    fn test_clear() {
        let dim = 8;
        let mut index = HnswIndex::new(dim);
        for i in 0..50 {
            index.insert(&format!("v{i}"), &vec_seed(i as f32, dim)).unwrap();
        }
        assert_eq!(index.len(), 50);

        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!(index.search(&vec_seed(1.0, dim), 10).unwrap().is_empty());
    }

    #[test]
    fn test_duplicate_insert_updates() {
        let dim = 8;
        let mut index = HnswIndex::new(dim);

        index.insert("a", &vec_seed(1.0, dim)).unwrap();
        index.insert("a", &vec_seed(2.0, dim)).unwrap();
        assert_eq!(index.len(), 1);

        // Search should find "a" near vec_seed(2.0) not vec_seed(1.0)
        let results = index.search(&vec_seed(2.0, dim), 1).unwrap();
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_resurrect_tombstoned() {
        let dim = 8;
        let mut index = HnswIndex::new(dim);

        index.insert("a", &vec_seed(1.0, dim)).unwrap();
        index.remove("a");
        assert_eq!(index.len(), 0);

        // Re-insert same rid
        index.insert("a", &vec_seed(2.0, dim)).unwrap();
        assert_eq!(index.len(), 1);

        let results = index.search(&vec_seed(2.0, dim), 1).unwrap();
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn test_recall_quality_dim64() {
        let dim = 64;
        let n = 1000;
        let k = 10;

        let mut hnsw = HnswIndex::with_params(dim, 16, 200, 50);
        let mut brute = BruteForceIndex::new(dim);

        for i in 0..n {
            let emb = vec_seed(i as f32 * 0.37, dim);
            hnsw.insert(&format!("v{i}"), &emb).unwrap();
            brute.insert(&format!("v{i}"), &emb);
        }

        // Test recall with 20 different queries
        let mut total_recall = 0.0;
        let num_queries = 20;
        for q in 0..num_queries {
            let query = vec_seed(q as f32 * 7.13 + 100.0, dim);
            let hnsw_results: HashSet<String> = hnsw
                .search(&query, k)
                .unwrap()
                .into_iter()
                .map(|(rid, _)| rid)
                .collect();
            let brute_results: HashSet<String> = brute
                .search(&query, k)
                .into_iter()
                .map(|(rid, _)| rid)
                .collect();

            let intersection = hnsw_results.intersection(&brute_results).count();
            total_recall += intersection as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall > 0.90,
            "recall@{k} should be > 0.90, got {avg_recall:.3}"
        );
    }

    #[test]
    fn test_recall_quality_dim384() {
        let dim = 384;
        let n = 500;
        let k = 10;

        let mut hnsw = HnswIndex::with_params(dim, 16, 200, 50);
        let mut brute = BruteForceIndex::new(dim);

        for i in 0..n {
            let emb = vec_seed(i as f32 * 0.37, dim);
            hnsw.insert(&format!("v{i}"), &emb).unwrap();
            brute.insert(&format!("v{i}"), &emb);
        }

        let mut total_recall = 0.0;
        let num_queries = 10;
        for q in 0..num_queries {
            let query = vec_seed(q as f32 * 7.13 + 100.0, dim);
            let hnsw_results: HashSet<String> = hnsw
                .search(&query, k)
                .unwrap()
                .into_iter()
                .map(|(rid, _)| rid)
                .collect();
            let brute_results: HashSet<String> = brute
                .search(&query, k)
                .into_iter()
                .map(|(rid, _)| rid)
                .collect();

            let intersection = hnsw_results.intersection(&brute_results).count();
            total_recall += intersection as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall > 0.85,
            "recall@{k} at dim=384 should be > 0.85, got {avg_recall:.3}"
        );
    }

    #[test]
    fn test_search_results_sorted_by_distance() {
        let dim = 64;
        let mut index = HnswIndex::new(dim);
        for i in 0..200 {
            index.insert(&format!("v{i}"), &vec_seed(i as f32 * 0.37, dim)).unwrap();
        }

        let query = vec_seed(999.0, dim);
        let results = index.search(&query, 20).unwrap();

        for i in 1..results.len() {
            assert!(
                results[i - 1].1 <= results[i].1 + 1e-10,
                "results not sorted: {} > {}",
                results[i - 1].1,
                results[i].1
            );
        }
    }

    #[test]
    fn test_large_insert_search() {
        let dim = 64;
        let n = 5000;
        let mut index = HnswIndex::new(dim);

        for i in 0..n {
            index.insert(&format!("v{i}"), &vec_seed(i as f32 * 0.37, dim)).unwrap();
        }
        assert_eq!(index.len(), n);

        let results = index.search(&vec_seed(999.0, dim), 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_search_with_many_tombstones() {
        let dim = 32;
        let mut index = HnswIndex::new(dim);

        // Insert 100, tombstone 90
        for i in 0..100 {
            index.insert(&format!("v{i}"), &vec_seed(i as f32, dim)).unwrap();
        }
        for i in 0..90 {
            index.remove(&format!("v{i}"));
        }
        assert_eq!(index.len(), 10);

        let results = index.search(&vec_seed(95.0, dim), 5).unwrap();
        // All results should be from v90-v99
        for (rid, _) in &results {
            let num: usize = rid[1..].parse().unwrap();
            assert!(num >= 90, "got tombstoned result {rid}");
        }
    }

    #[test]
    fn test_brute_force_index() {
        let dim = 8;
        let mut bf = BruteForceIndex::new(dim);
        bf.insert("a", &vec_seed(1.0, dim));
        bf.insert("b", &vec_seed(2.0, dim));
        bf.insert("c", &vec_seed(3.0, dim));

        assert_eq!(bf.len(), 3);
        bf.remove("b");
        assert_eq!(bf.len(), 2);

        let results = bf.search(&vec_seed(1.0, dim), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a"); // Closest to seed 1.0
    }
}
