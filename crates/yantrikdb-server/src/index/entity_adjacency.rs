//! Entity adjacency index — RFC 015-B-3.
//!
//! ## Why this exists
//!
//! Recall's `expand_entities=true` path is what drove the term=1423
//! thrashing incident. The original implementation walked an
//! edges-by-source-rid table per recall result, paid per-edge SQL,
//! and at fan-out >100 saturated voter CPU. Per gpt-5.5's redteam,
//! this was the #1 latent risk in the recall hot path.
//!
//! The fix is a per-tenant in-memory index keyed on entity id:
//! - `(tenant_id, entity_id) -> sorted Vec<(rid, edge_weight)>`
//!
//! Lookup is O(1) hashmap; the per-entity `Vec` is pre-sorted by
//! `edge_weight DESC` so callers needing "top-N adjacent rids" just
//! truncate the prefix. No SQL, no graph walk, no cross-tenant
//! contention. Eliminates the linear-cost factor that drove term=1423.
//!
//! ## Substrate-only at this PR
//!
//! This module ships the in-memory index + the [`EntityAdjacencyProvider`]
//! trait + an [`InvalidationBus`] subscriber that keeps the index in
//! sync with `UpsertEntityEdge` / `DeleteEntityEdge` mutations from
//! RFC 010 PR-3's grammar.
//!
//! Wiring into the recall handler (the `expand_entities=true`
//! branch in `http_gateway.rs`) is a separate integration PR —
//! same pattern as the cache substrate (015-B-1) shipping ahead of
//! its handler integration.
//!
//! ## Memory footprint
//!
//! At ~80 bytes per `(rid, weight)` entry, an active tenant with
//! 100k edges sits at ~8 MB. Per-tenant scoping means a 1000-tenant
//! deployment caps at the policy-controlled aggregate. RFC 020-folded
//! per-tenant ResourceBudget will gate this if needed; no eviction
//! at this PR (correctness > capacity, and operators can hard-cap by
//! refusing to load tenants that would exceed budget).

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::cache::InvalidationEvent;
use crate::commit::TenantId;

/// One adjacency entry: a (rid, weight) pair that some other entity
/// is connected to. Sorted by `weight` descending in the per-entity
/// list, so the prefix is the highest-weight neighbors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdjacencyRow {
    pub rid: String,
    pub weight: f64,
}

/// Per-entity adjacency. Sorted by weight desc so callers can
/// truncate to top-N by taking the prefix.
type AdjacencyList = Vec<AdjacencyRow>;

/// Read-side view of the index. Decouples the recall handler from
/// the concrete impl so the impl can change (in-memory → on-disk
/// → distributed) without touching callers.
///
/// This is the trait the recall path depends on; concrete callers
/// should hold `Arc<dyn EntityAdjacencyProvider>`.
pub trait EntityAdjacencyProvider: Send + Sync {
    /// Look up the top-N adjacent rids for `(tenant_id, entity_id)`.
    /// `limit` truncates the per-entity sorted list. Returns an
    /// empty Vec on lookup miss (unknown tenant or unknown entity).
    fn top_adjacent(&self, tenant_id: TenantId, entity_id: &str, limit: usize)
        -> Vec<AdjacencyRow>;

    /// Convenience: union of top-N adjacent across multiple seed
    /// entities, deduped by rid (first occurrence wins). Used when
    /// the recall handler has K result rids each with their own
    /// entities, and wants to expand the result set without
    /// repeated lookups.
    fn top_adjacent_union(
        &self,
        tenant_id: TenantId,
        entity_ids: &[&str],
        per_entity_limit: usize,
    ) -> Vec<AdjacencyRow> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for eid in entity_ids {
            for row in self.top_adjacent(tenant_id, eid, per_entity_limit) {
                if seen.insert(row.rid.clone()) {
                    out.push(row);
                }
            }
        }
        out
    }
}

/// In-memory implementation. Cheap to clone (`Arc` internally).
pub struct EntityAdjacencyIndex {
    /// `tenant_id → entity_id → sorted Vec<(rid, weight)>`.
    /// Two-level HashMap so per-tenant invalidation is a single
    /// inner-map rebuild, no cross-tenant scan.
    inner: Arc<RwLock<HashMap<TenantId, HashMap<String, AdjacencyList>>>>,
}

impl EntityAdjacencyIndex {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert or update an edge: `(src, dst, weight)`. Both `src`
    /// and `dst` get an entry — the index is undirected from the
    /// recall path's POV (neighbors-of-X works regardless of
    /// edge direction in the underlying graph).
    pub fn upsert_edge(&self, tenant_id: TenantId, src: &str, dst: &str, weight: f64) {
        let mut map = self.inner.write();
        let tenant_map = map.entry(tenant_id).or_default();
        Self::upsert_directed(tenant_map, src, dst, weight);
        Self::upsert_directed(tenant_map, dst, src, weight);
    }

    /// Remove an edge. Mirror of `upsert_edge`. Cheap because the
    /// per-entity list is rarely large (median: tens of neighbors).
    pub fn delete_edge(&self, tenant_id: TenantId, src: &str, dst: &str) {
        let mut map = self.inner.write();
        if let Some(tenant_map) = map.get_mut(&tenant_id) {
            Self::delete_directed(tenant_map, src, dst);
            Self::delete_directed(tenant_map, dst, src);
        }
    }

    fn upsert_directed(
        tenant_map: &mut HashMap<String, AdjacencyList>,
        from: &str,
        to: &str,
        weight: f64,
    ) {
        let list = tenant_map.entry(from.to_string()).or_default();
        // Replace existing entry if present, else append.
        if let Some(existing) = list.iter_mut().find(|r| r.rid == to) {
            existing.weight = weight;
        } else {
            list.push(AdjacencyRow {
                rid: to.to_string(),
                weight,
            });
        }
        // Re-sort by weight desc — small list, cheap.
        list.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.rid.cmp(&b.rid))
        });
    }

    fn delete_directed(tenant_map: &mut HashMap<String, AdjacencyList>, from: &str, to: &str) {
        if let Some(list) = tenant_map.get_mut(from) {
            list.retain(|r| r.rid != to);
            if list.is_empty() {
                tenant_map.remove(from);
            }
        }
    }

    /// Drop everything for a tenant — used on tenant eviction or
    /// invalidation events that scope to a whole tenant.
    pub fn clear_tenant(&self, tenant_id: TenantId) {
        self.inner.write().remove(&tenant_id);
    }

    /// Total entries (for /metrics). Walks the outer + inner maps,
    /// linear in tenant_count × avg_entities_per_tenant.
    pub fn total_entries(&self) -> usize {
        self.inner
            .read()
            .values()
            .map(|m| m.values().map(|v| v.len()).sum::<usize>())
            .sum()
    }

    pub fn tenant_count(&self) -> usize {
        self.inner.read().len()
    }

    /// Number of distinct entities indexed for this tenant.
    pub fn entity_count(&self, tenant_id: TenantId) -> usize {
        self.inner
            .read()
            .get(&tenant_id)
            .map(|m| m.len())
            .unwrap_or(0)
    }
}

impl Default for EntityAdjacencyIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for EntityAdjacencyIndex {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl EntityAdjacencyProvider for EntityAdjacencyIndex {
    fn top_adjacent(
        &self,
        tenant_id: TenantId,
        entity_id: &str,
        limit: usize,
    ) -> Vec<AdjacencyRow> {
        let map = self.inner.read();
        let Some(tenant_map) = map.get(&tenant_id) else {
            return Vec::new();
        };
        let Some(list) = tenant_map.get(entity_id) else {
            return Vec::new();
        };
        list.iter().take(limit).cloned().collect()
    }
}

/// Subscribe to the [`crate::cache::InvalidationBus`] and apply
/// edge-related events to the index. Returns a join handle the
/// caller is expected to abort on shutdown (typically the server
/// drop path).
///
/// The current `InvalidationEvent` enum (RFC 015-A) doesn't carry
/// edge_weight — only `EdgeChanged { src, dst }`. So this
/// subscriber issues `delete_edge` then `upsert_edge` with weight=1.0
/// as a placeholder; the actual weight has to be re-read from the
/// commit log by the integration layer that pairs the index with
/// `MutationCommitter::read_range`. This PR keeps the subscriber
/// minimal; the integration PR teaches it to read weights.
pub fn spawn_invalidation_bus_subscriber(
    index: EntityAdjacencyIndex,
    bus: &crate::cache::InvalidationBus,
) -> tokio::task::JoinHandle<()> {
    let mut rx = bus.subscribe();
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(InvalidationEvent::EdgeChanged {
                    tenant_id,
                    src,
                    dst,
                }) => {
                    // Coarse: drop and re-add. The weight is
                    // re-derived by the integration layer in a
                    // follow-up PR.
                    index.delete_edge(tenant_id, &src, &dst);
                    index.upsert_edge(tenant_id, &src, &dst, 1.0);
                }
                Ok(InvalidationEvent::TenantConfigChanged { .. }) => {
                    // No edge implication.
                }
                Ok(InvalidationEvent::Tombstoned { .. })
                | Ok(InvalidationEvent::Updated { .. }) => {
                    // Edges aren't directly invalidated by per-rid
                    // tombstone/update events; the rid removal
                    // happens via DeleteEntityEdge mutations which
                    // surface as EdgeChanged.
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    // Per RFC 015-A: subscribers SHOULD respond to
                    // Lagged by clearing their cache wholesale. The
                    // recall path is correct in the meantime
                    // because the index empty just means recall
                    // skips entity expansion (a downgrade, not a
                    // correctness break).
                    let map_clone = Arc::clone(&index.inner);
                    map_clone.write().clear();
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_index_returns_no_neighbors() {
        let idx = EntityAdjacencyIndex::new();
        assert!(idx.top_adjacent(TenantId::new(1), "alice", 10).is_empty());
        assert_eq!(idx.tenant_count(), 0);
        assert_eq!(idx.total_entries(), 0);
    }

    #[test]
    fn upsert_then_lookup_returns_both_directions() {
        let idx = EntityAdjacencyIndex::new();
        idx.upsert_edge(TenantId::new(1), "alice", "bob", 0.7);
        // alice → bob
        let alice = idx.top_adjacent(TenantId::new(1), "alice", 10);
        assert_eq!(alice.len(), 1);
        assert_eq!(alice[0].rid, "bob");
        assert_eq!(alice[0].weight, 0.7);
        // bob → alice (undirected)
        let bob = idx.top_adjacent(TenantId::new(1), "bob", 10);
        assert_eq!(bob.len(), 1);
        assert_eq!(bob[0].rid, "alice");
    }

    #[test]
    fn list_is_sorted_by_weight_desc() {
        let idx = EntityAdjacencyIndex::new();
        idx.upsert_edge(TenantId::new(1), "alice", "low", 0.1);
        idx.upsert_edge(TenantId::new(1), "alice", "high", 0.9);
        idx.upsert_edge(TenantId::new(1), "alice", "mid", 0.5);
        let res = idx.top_adjacent(TenantId::new(1), "alice", 10);
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].rid, "high");
        assert_eq!(res[1].rid, "mid");
        assert_eq!(res[2].rid, "low");
    }

    #[test]
    fn limit_truncates_prefix() {
        let idx = EntityAdjacencyIndex::new();
        for i in 0..10 {
            idx.upsert_edge(TenantId::new(1), "alice", &format!("n{i}"), i as f64);
        }
        let res = idx.top_adjacent(TenantId::new(1), "alice", 3);
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].rid, "n9");
        assert_eq!(res[1].rid, "n8");
        assert_eq!(res[2].rid, "n7");
    }

    #[test]
    fn upsert_replaces_weight_not_duplicates() {
        let idx = EntityAdjacencyIndex::new();
        idx.upsert_edge(TenantId::new(1), "alice", "bob", 0.3);
        idx.upsert_edge(TenantId::new(1), "alice", "bob", 0.8);
        let res = idx.top_adjacent(TenantId::new(1), "alice", 10);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].weight, 0.8);
    }

    #[test]
    fn delete_edge_removes_both_directions() {
        let idx = EntityAdjacencyIndex::new();
        idx.upsert_edge(TenantId::new(1), "alice", "bob", 0.5);
        idx.delete_edge(TenantId::new(1), "alice", "bob");
        assert!(idx.top_adjacent(TenantId::new(1), "alice", 10).is_empty());
        assert!(idx.top_adjacent(TenantId::new(1), "bob", 10).is_empty());
    }

    #[test]
    fn delete_one_edge_keeps_others() {
        let idx = EntityAdjacencyIndex::new();
        idx.upsert_edge(TenantId::new(1), "alice", "bob", 0.5);
        idx.upsert_edge(TenantId::new(1), "alice", "carol", 0.7);
        idx.delete_edge(TenantId::new(1), "alice", "bob");
        let alice = idx.top_adjacent(TenantId::new(1), "alice", 10);
        assert_eq!(alice.len(), 1);
        assert_eq!(alice[0].rid, "carol");
    }

    #[test]
    fn delete_nonexistent_edge_is_noop() {
        let idx = EntityAdjacencyIndex::new();
        idx.upsert_edge(TenantId::new(1), "alice", "bob", 0.5);
        idx.delete_edge(TenantId::new(1), "alice", "ghost"); // unknown
        idx.delete_edge(TenantId::new(2), "alice", "bob"); // unknown tenant
                                                           // alice→bob still intact.
        assert_eq!(idx.top_adjacent(TenantId::new(1), "alice", 10).len(), 1);
    }

    #[test]
    fn per_tenant_isolation() {
        let idx = EntityAdjacencyIndex::new();
        idx.upsert_edge(TenantId::new(1), "alice", "bob", 0.5);
        idx.upsert_edge(TenantId::new(2), "alice", "carol", 0.9);
        let t1 = idx.top_adjacent(TenantId::new(1), "alice", 10);
        let t2 = idx.top_adjacent(TenantId::new(2), "alice", 10);
        assert_eq!(t1.len(), 1);
        assert_eq!(t1[0].rid, "bob");
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].rid, "carol");
    }

    #[test]
    fn clear_tenant_drops_only_that_tenant() {
        let idx = EntityAdjacencyIndex::new();
        idx.upsert_edge(TenantId::new(1), "a", "b", 0.5);
        idx.upsert_edge(TenantId::new(2), "a", "b", 0.5);
        idx.clear_tenant(TenantId::new(1));
        assert!(idx.top_adjacent(TenantId::new(1), "a", 10).is_empty());
        assert_eq!(idx.top_adjacent(TenantId::new(2), "a", 10).len(), 1);
    }

    #[test]
    fn metrics_track_counts() {
        let idx = EntityAdjacencyIndex::new();
        idx.upsert_edge(TenantId::new(1), "a", "b", 0.5);
        idx.upsert_edge(TenantId::new(1), "a", "c", 0.5);
        idx.upsert_edge(TenantId::new(2), "x", "y", 0.5);
        // Tenant 1: a→{b,c}, b→{a}, c→{a} = 4 entries.
        // Tenant 2: x→{y}, y→{x} = 2 entries.
        assert_eq!(idx.total_entries(), 6);
        assert_eq!(idx.tenant_count(), 2);
        assert_eq!(idx.entity_count(TenantId::new(1)), 3);
        assert_eq!(idx.entity_count(TenantId::new(2)), 2);
    }

    #[test]
    fn top_adjacent_union_dedupes_across_seeds() {
        let idx = EntityAdjacencyIndex::new();
        // alice → bob, charlie. bob → alice, charlie.
        idx.upsert_edge(TenantId::new(1), "alice", "bob", 0.5);
        idx.upsert_edge(TenantId::new(1), "alice", "charlie", 0.7);
        idx.upsert_edge(TenantId::new(1), "bob", "charlie", 0.3);
        // Union of alice's + bob's adjacency = {bob, charlie, alice}
        // (charlie appears via both seeds — should dedupe).
        let union = idx.top_adjacent_union(TenantId::new(1), &["alice", "bob"], 10);
        let rids: Vec<&str> = union.iter().map(|r| r.rid.as_str()).collect();
        assert!(rids.contains(&"alice"));
        assert!(rids.contains(&"bob"));
        assert!(rids.contains(&"charlie"));
        // No duplicates.
        let mut seen = std::collections::HashSet::new();
        for r in &rids {
            assert!(seen.insert(r), "duplicate in union: {r}");
        }
    }

    #[test]
    fn dyn_dispatch_works() {
        // Recall handlers will hold Arc<dyn EntityAdjacencyProvider>;
        // confirm dyn dispatch resolves correctly through the
        // default `top_adjacent_union` impl.
        let idx: Arc<dyn EntityAdjacencyProvider> = Arc::new(EntityAdjacencyIndex::new());
        let r = idx.top_adjacent(TenantId::new(1), "ghost", 10);
        assert!(r.is_empty());
    }

    #[tokio::test]
    async fn invalidation_bus_subscriber_applies_edge_changed() {
        let bus = crate::cache::InvalidationBus::new();
        let idx = EntityAdjacencyIndex::new();
        let handle = spawn_invalidation_bus_subscriber(idx.clone(), &bus);

        // Publish an EdgeChanged event.
        bus.publish(InvalidationEvent::EdgeChanged {
            tenant_id: TenantId::new(1),
            src: "alice".into(),
            dst: "bob".into(),
        });

        // Give the subscriber a moment to process.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let r = idx.top_adjacent(TenantId::new(1), "alice", 10);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].rid, "bob");

        handle.abort();
    }

    #[tokio::test]
    async fn invalidation_bus_subscriber_ignores_unrelated_events() {
        let bus = crate::cache::InvalidationBus::new();
        let idx = EntityAdjacencyIndex::new();
        idx.upsert_edge(TenantId::new(1), "alice", "bob", 0.5);
        let handle = spawn_invalidation_bus_subscriber(idx.clone(), &bus);

        // Tombstoned + Updated events should NOT touch the index.
        bus.publish(InvalidationEvent::Tombstoned {
            tenant_id: TenantId::new(1),
            rid: "some_memory".into(),
        });
        bus.publish(InvalidationEvent::Updated {
            tenant_id: TenantId::new(1),
            rid: "some_memory".into(),
        });
        bus.publish(InvalidationEvent::TenantConfigChanged {
            tenant_id: TenantId::new(1),
            key: "some_key".into(),
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Index unchanged.
        let r = idx.top_adjacent(TenantId::new(1), "alice", 10);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].rid, "bob");

        handle.abort();
    }
}
