//! Time-decay sorted index — RFC 015-B-3.
//!
//! ## Why this exists
//!
//! Many production agent queries are "what was I just doing?" — the
//! caller wants the most recent + important memories, regardless of
//! semantic similarity to a particular query string. Today every
//! such recall pays the full HNSW round trip plus an embedder
//! forward pass; both are wasted because the answer is "the
//! highest-scoring rids by recency × importance" — a sort, not a
//! similarity search.
//!
//! This index materializes that sort. Per `(tenant_id, namespace)`,
//! a sorted list of `(rid, decayed_score)` where
//!
//! ```text
//! decayed_score = importance × 2^(-Δhours / half_life_hours)
//! ```
//!
//! Recall handlers can short-circuit:
//! 1. If the request looks like "what was I just doing?" (no query
//!    text or a sentinel query like `*` or empty), return the top-K
//!    from this index directly.
//! 2. Else fall through to HNSW + embedder as today.
//!
//! Cost: one f64 sort key per memory, a `Vec` per namespace, kept
//! sorted on insert/touch. Memory is small — at 64 bytes per row,
//! 1M memories per namespace is ~64 MB.
//!
//! ## Stale scoring is acceptable
//!
//! The `decayed_score` decays over wall time, so a row in the
//! index drifts from "actually decayed" once it's stationary.
//! Two strategies:
//! 1. Lazy: callers re-derive the decayed score from
//!    `(importance, last_access_at, half_life)` at query time;
//!    the in-index score is just a heuristic for sort order.
//! 2. Eager: a periodic sweep re-scores everything.
//!
//! This PR ships the **lazy** strategy: the index stores
//! `(importance, last_access_at, half_life)` per row and computes
//! the decayed score at query time relative to a caller-supplied
//! `now`. The sort is by `last_access_at DESC` as a fast proxy that
//! tracks decay closely — anything more elaborate adds re-sort cost
//! on every touch.
//!
//! ## Substrate-only at this PR
//!
//! Same shape as RFC 015-B-1 + 015-B-3 entity_adjacency: ship the
//! index + provider trait + InvalidationBus subscriber. Wiring into
//! the recall handler's "short-circuit empty query" branch is a
//! follow-up integration PR.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::cache::InvalidationEvent;
use crate::commit::TenantId;

/// One row in the index. Stores the raw inputs to the decay function
/// rather than a pre-computed score, so the score is always fresh
/// at query time without a re-sort sweep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecayedRow {
    pub rid: String,
    /// Original `importance` from the memory. 0.0..=1.0 typically.
    pub importance: f64,
    /// When this row was last accessed/written. Drives the decay.
    pub last_access_at: SystemTime,
    /// Half-life from the memory's `half_life` field, in hours.
    /// `decay_score(now) = importance * 2^(-Δhours / half_life)`
    pub half_life_hours: f64,
}

impl DecayedRow {
    /// Score this row at a given `now`. Score = importance ×
    /// 2^(-elapsed_hours / half_life). Caller-driven so the same
    /// row can be scored repeatedly (e.g., when ranking) without
    /// touching shared state.
    pub fn decayed_score(&self, now: SystemTime) -> f64 {
        if self.half_life_hours <= 0.0 {
            return self.importance;
        }
        let elapsed_secs = now
            .duration_since(self.last_access_at)
            .unwrap_or(Duration::ZERO)
            .as_secs_f64();
        let elapsed_hours = elapsed_secs / 3600.0;
        let exponent = -elapsed_hours / self.half_life_hours;
        self.importance * exponent.exp2()
    }
}

/// Configuration for callers that want to override defaults
/// (recall handler can pass operator-tuned values).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecayConfig {
    /// Default half-life in hours when a memory doesn't specify one.
    /// 168 hours = 1 week, matches yantrikdb's existing default
    /// `half_life` field.
    pub default_half_life_hours: f64,
    /// Minimum decayed score for a row to be eligible. Below this
    /// the row is "effectively forgotten" — skipped at query time
    /// even though it's still in the index. Keeps results tight.
    pub min_score: f64,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            default_half_life_hours: 168.0,
            min_score: 0.0,
        }
    }
}

/// Provider trait so recall handlers depend on the surface, not the
/// in-memory impl. Same pattern as `EntityAdjacencyProvider`.
pub trait TimeDecayProvider: Send + Sync {
    /// Top-N rids by decayed score at `now`, namespace-scoped.
    /// Returns rows pre-scored so callers can join with full
    /// memories without re-deriving.
    fn top_decayed(
        &self,
        tenant_id: TenantId,
        namespace: &str,
        limit: usize,
        now: SystemTime,
    ) -> Vec<(DecayedRow, f64)>;

    /// Convenience: just the rids.
    fn top_rids(
        &self,
        tenant_id: TenantId,
        namespace: &str,
        limit: usize,
        now: SystemTime,
    ) -> Vec<String> {
        self.top_decayed(tenant_id, namespace, limit, now)
            .into_iter()
            .map(|(row, _)| row.rid)
            .collect()
    }
}

/// In-memory implementation. Rows kept in a `Vec` per namespace,
/// sorted by `last_access_at DESC` (proxy for "highest decayed
/// score" — exact ordering at query time is decided by recomputing
/// `decayed_score` and re-truncating, but the proxy gives a tight
/// candidate window).
pub struct TimeDecayIndex {
    inner: Arc<RwLock<HashMap<TenantId, HashMap<String, Vec<DecayedRow>>>>>,
    config: DecayConfig,
}

impl TimeDecayIndex {
    pub fn new(config: DecayConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    pub fn config(&self) -> &DecayConfig {
        &self.config
    }

    /// Insert or update a row. If `(tenant, namespace, rid)` exists,
    /// updates its `last_access_at` + importance + half_life.
    pub fn upsert(
        &self,
        tenant_id: TenantId,
        namespace: &str,
        rid: &str,
        importance: f64,
        half_life_hours: Option<f64>,
        when: SystemTime,
    ) {
        let half_life = half_life_hours.unwrap_or(self.config.default_half_life_hours);
        let mut map = self.inner.write();
        let ns_map = map.entry(tenant_id).or_default();
        let list = ns_map.entry(namespace.to_string()).or_default();
        if let Some(existing) = list.iter_mut().find(|r| r.rid == rid) {
            existing.importance = importance;
            existing.last_access_at = when;
            existing.half_life_hours = half_life;
        } else {
            list.push(DecayedRow {
                rid: rid.to_string(),
                importance,
                last_access_at: when,
                half_life_hours: half_life,
            });
        }
        // Sort by last_access_at DESC. Newer = higher proxy score.
        list.sort_by(|a, b| b.last_access_at.cmp(&a.last_access_at));
    }

    /// Touch a row — reset its `last_access_at` to `when` while
    /// keeping importance + half_life unchanged. Used by the
    /// recall path so accessed memories drift back to the top.
    pub fn touch(&self, tenant_id: TenantId, namespace: &str, rid: &str, when: SystemTime) {
        let mut map = self.inner.write();
        let Some(ns_map) = map.get_mut(&tenant_id) else {
            return;
        };
        let Some(list) = ns_map.get_mut(namespace) else {
            return;
        };
        if let Some(existing) = list.iter_mut().find(|r| r.rid == rid) {
            existing.last_access_at = when;
            list.sort_by(|a, b| b.last_access_at.cmp(&a.last_access_at));
        }
    }

    /// Remove a row (rid was tombstoned, or namespace dropped).
    pub fn remove(&self, tenant_id: TenantId, namespace: &str, rid: &str) {
        let mut map = self.inner.write();
        let Some(ns_map) = map.get_mut(&tenant_id) else {
            return;
        };
        let Some(list) = ns_map.get_mut(namespace) else {
            return;
        };
        list.retain(|r| r.rid != rid);
        if list.is_empty() {
            ns_map.remove(namespace);
        }
        if ns_map.is_empty() {
            map.remove(&tenant_id);
        }
    }

    /// Drop everything for a tenant.
    pub fn clear_tenant(&self, tenant_id: TenantId) {
        self.inner.write().remove(&tenant_id);
    }

    pub fn total_entries(&self) -> usize {
        self.inner
            .read()
            .values()
            .flat_map(|m| m.values())
            .map(|v| v.len())
            .sum()
    }

    pub fn tenant_count(&self) -> usize {
        self.inner.read().len()
    }

    pub fn namespace_count(&self, tenant_id: TenantId) -> usize {
        self.inner
            .read()
            .get(&tenant_id)
            .map(|m| m.len())
            .unwrap_or(0)
    }
}

impl Default for TimeDecayIndex {
    fn default() -> Self {
        Self::new(DecayConfig::default())
    }
}

impl Clone for TimeDecayIndex {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            config: self.config.clone(),
        }
    }
}

impl TimeDecayProvider for TimeDecayIndex {
    fn top_decayed(
        &self,
        tenant_id: TenantId,
        namespace: &str,
        limit: usize,
        now: SystemTime,
    ) -> Vec<(DecayedRow, f64)> {
        let map = self.inner.read();
        let Some(ns_map) = map.get(&tenant_id) else {
            return Vec::new();
        };
        let Some(list) = ns_map.get(namespace) else {
            return Vec::new();
        };
        // Take a candidate window slightly larger than `limit` from
        // the recency-sorted Vec, score each, filter sub-min,
        // sort by score DESC, truncate to `limit`. The candidate
        // window heuristic keeps the cost bounded in pathological
        // cases (e.g., a namespace where importance varies wildly
        // and recency doesn't track it).
        let candidate_window = (limit * 4).max(64).min(list.len());
        let mut scored: Vec<(DecayedRow, f64)> = list[..candidate_window]
            .iter()
            .map(|r| (r.clone(), r.decayed_score(now)))
            .filter(|(_, score)| *score >= self.config.min_score)
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(limit);
        scored
    }
}

/// Subscribe to [`crate::cache::InvalidationBus`] so the index stays
/// in sync with commit-log mutations. Tombstoned rids are removed
/// across all namespaces (we don't know which namespace it was in
/// without a back-index, so we walk all). Updated rids are touched
/// (no namespace info either; same caveat).
///
/// This subscriber is conservative: a more efficient version would
/// take a (tenant, namespace, rid) tuple in the event payload.
/// RFC 015-A's current event grammar doesn't carry namespace; the
/// integration PR that wires the index into the apply path can
/// supply that info directly.
pub fn spawn_invalidation_bus_subscriber(
    index: TimeDecayIndex,
    bus: &crate::cache::InvalidationBus,
) -> tokio::task::JoinHandle<()> {
    let mut rx = bus.subscribe();
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(InvalidationEvent::Tombstoned { tenant_id, rid }) => {
                    // Walk every namespace for this tenant and remove
                    // the rid. Cost is O(namespaces × rows-per-ns)
                    // but tombstones are rare and per-tenant scoping
                    // bounds the work.
                    let mut map = index.inner.write();
                    if let Some(ns_map) = map.get_mut(&tenant_id) {
                        for list in ns_map.values_mut() {
                            list.retain(|r| r.rid != rid);
                        }
                        ns_map.retain(|_, v| !v.is_empty());
                    }
                }
                Ok(InvalidationEvent::Updated { tenant_id: _, rid: _ })
                | Ok(InvalidationEvent::EdgeChanged { .. })
                | Ok(InvalidationEvent::TenantConfigChanged { .. }) => {
                    // Updated lacks namespace info to make the touch
                    // efficient. The integration PR (recall handler
                    // wiring) calls `index.touch(...)` directly with
                    // full info instead of going through the bus.
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    // Per RFC 015-A: clear wholesale on lag. Recall
                    // path falls back to HNSW for the short-circuit
                    // queries until the index re-hydrates.
                    index.inner.write().clear();
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn t0() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    #[test]
    fn config_defaults_are_one_week_half_life() {
        let cfg = DecayConfig::default();
        assert_eq!(cfg.default_half_life_hours, 168.0);
        assert_eq!(cfg.min_score, 0.0);
    }

    #[test]
    fn decay_at_zero_elapsed_equals_importance() {
        let now = t0();
        let row = DecayedRow {
            rid: "a".into(),
            importance: 0.7,
            last_access_at: now,
            half_life_hours: 24.0,
        };
        assert!((row.decayed_score(now) - 0.7).abs() < 1e-9);
    }

    #[test]
    fn decay_at_one_half_life_halves_score() {
        let now = t0();
        let row = DecayedRow {
            rid: "a".into(),
            importance: 0.8,
            last_access_at: now,
            half_life_hours: 24.0,
        };
        // 24h later — should be exactly 0.4.
        let later = now + Duration::from_secs(24 * 3600);
        let score = row.decayed_score(later);
        assert!((score - 0.4).abs() < 1e-9);
    }

    #[test]
    fn decay_at_two_half_lives_quarters_score() {
        let now = t0();
        let row = DecayedRow {
            rid: "a".into(),
            importance: 1.0,
            last_access_at: now,
            half_life_hours: 24.0,
        };
        let later = now + Duration::from_secs(48 * 3600);
        assert!((row.decayed_score(later) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn decay_zero_half_life_returns_importance_flat() {
        // half_life=0 is a degenerate case; we treat it as "no
        // decay" so the row stays at full importance.
        let row = DecayedRow {
            rid: "a".into(),
            importance: 0.5,
            last_access_at: t0(),
            half_life_hours: 0.0,
        };
        let later = t0() + Duration::from_secs(100 * 3600);
        assert_eq!(row.decayed_score(later), 0.5);
    }

    #[test]
    fn empty_index_returns_no_rows() {
        let idx = TimeDecayIndex::default();
        let r = idx.top_decayed(TenantId::new(1), "ns", 10, t0());
        assert!(r.is_empty());
    }

    #[test]
    fn upsert_then_top_returns_row() {
        let idx = TimeDecayIndex::default();
        idx.upsert(TenantId::new(1), "ns", "a", 0.7, Some(24.0), t0());
        let r = idx.top_decayed(TenantId::new(1), "ns", 10, t0());
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0.rid, "a");
        assert!((r[0].1 - 0.7).abs() < 1e-9);
    }

    #[test]
    fn upsert_replaces_does_not_duplicate() {
        let idx = TimeDecayIndex::default();
        idx.upsert(TenantId::new(1), "ns", "a", 0.3, Some(24.0), t0());
        idx.upsert(TenantId::new(1), "ns", "a", 0.9, Some(24.0), t0());
        let r = idx.top_decayed(TenantId::new(1), "ns", 10, t0());
        assert_eq!(r.len(), 1);
        assert!((r[0].1 - 0.9).abs() < 1e-9);
    }

    #[test]
    fn top_decayed_orders_by_decayed_score() {
        // Three rows with different importance × age:
        // - "old_high":  importance 1.0, accessed 24h ago, half_life 24h → 0.5
        // - "fresh_low": importance 0.4, accessed now,     half_life 24h → 0.4
        // - "fresh_mid": importance 0.6, accessed now,     half_life 24h → 0.6
        // Order: fresh_mid (0.6) > old_high (0.5) > fresh_low (0.4).
        let idx = TimeDecayIndex::default();
        let now = t0();
        let day_ago = now - Duration::from_secs(24 * 3600);
        idx.upsert(TenantId::new(1), "ns", "old_high", 1.0, Some(24.0), day_ago);
        idx.upsert(TenantId::new(1), "ns", "fresh_low", 0.4, Some(24.0), now);
        idx.upsert(TenantId::new(1), "ns", "fresh_mid", 0.6, Some(24.0), now);
        let r = idx.top_decayed(TenantId::new(1), "ns", 3, now);
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].0.rid, "fresh_mid");
        assert_eq!(r[1].0.rid, "old_high");
        assert_eq!(r[2].0.rid, "fresh_low");
    }

    #[test]
    fn limit_truncates_top_decayed() {
        let idx = TimeDecayIndex::default();
        for i in 0..10 {
            idx.upsert(
                TenantId::new(1),
                "ns",
                &format!("r{i}"),
                0.5,
                Some(24.0),
                t0() + Duration::from_secs(i),
            );
        }
        let r = idx.top_decayed(TenantId::new(1), "ns", 3, t0() + Duration::from_secs(100));
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn min_score_filters_out_decayed_below_floor() {
        let idx = TimeDecayIndex::new(DecayConfig {
            default_half_life_hours: 24.0,
            min_score: 0.3,
        });
        idx.upsert(TenantId::new(1), "ns", "high", 0.9, Some(24.0), t0());
        idx.upsert(TenantId::new(1), "ns", "low", 0.1, Some(24.0), t0());
        let r = idx.top_decayed(TenantId::new(1), "ns", 10, t0());
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0.rid, "high");
    }

    #[test]
    fn touch_updates_recency_only() {
        let idx = TimeDecayIndex::default();
        let now = t0();
        let two_days_ago = now - Duration::from_secs(48 * 3600);
        idx.upsert(TenantId::new(1), "ns", "a", 0.5, Some(24.0), two_days_ago);
        // Before touch: at `now`, score = 0.5 × 2^(-2) = 0.125.
        let r1 = idx.top_decayed(TenantId::new(1), "ns", 10, now);
        assert!((r1[0].1 - 0.125).abs() < 1e-6);
        // Touch: last_access_at moves to `now`. Score back to 0.5.
        idx.touch(TenantId::new(1), "ns", "a", now);
        let r2 = idx.top_decayed(TenantId::new(1), "ns", 10, now);
        assert!((r2[0].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn touch_unknown_rid_is_noop() {
        let idx = TimeDecayIndex::default();
        idx.upsert(TenantId::new(1), "ns", "a", 0.5, Some(24.0), t0());
        idx.touch(TenantId::new(1), "ns", "ghost", t0());
        idx.touch(TenantId::new(2), "ns", "a", t0()); // unknown tenant
        idx.touch(TenantId::new(1), "missing_ns", "a", t0()); // unknown ns
        // Original row intact.
        assert_eq!(idx.total_entries(), 1);
    }

    #[test]
    fn remove_drops_row() {
        let idx = TimeDecayIndex::default();
        idx.upsert(TenantId::new(1), "ns", "a", 0.5, Some(24.0), t0());
        idx.upsert(TenantId::new(1), "ns", "b", 0.5, Some(24.0), t0());
        idx.remove(TenantId::new(1), "ns", "a");
        let r = idx.top_decayed(TenantId::new(1), "ns", 10, t0());
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0.rid, "b");
    }

    #[test]
    fn remove_last_in_namespace_drops_namespace() {
        let idx = TimeDecayIndex::default();
        idx.upsert(TenantId::new(1), "ns", "a", 0.5, Some(24.0), t0());
        idx.remove(TenantId::new(1), "ns", "a");
        assert_eq!(idx.namespace_count(TenantId::new(1)), 0);
        // Tenant should also be dropped when its last namespace empties.
        assert_eq!(idx.tenant_count(), 0);
    }

    #[test]
    fn per_tenant_isolation() {
        let idx = TimeDecayIndex::default();
        idx.upsert(TenantId::new(1), "ns", "a", 0.5, Some(24.0), t0());
        idx.upsert(TenantId::new(2), "ns", "b", 0.5, Some(24.0), t0());
        let t1 = idx.top_decayed(TenantId::new(1), "ns", 10, t0());
        let t2 = idx.top_decayed(TenantId::new(2), "ns", 10, t0());
        assert_eq!(t1.len(), 1);
        assert_eq!(t1[0].0.rid, "a");
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].0.rid, "b");
    }

    #[test]
    fn per_namespace_isolation() {
        let idx = TimeDecayIndex::default();
        idx.upsert(TenantId::new(1), "ns_a", "a", 0.5, Some(24.0), t0());
        idx.upsert(TenantId::new(1), "ns_b", "b", 0.5, Some(24.0), t0());
        let a = idx.top_decayed(TenantId::new(1), "ns_a", 10, t0());
        let b = idx.top_decayed(TenantId::new(1), "ns_b", 10, t0());
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].0.rid, "a");
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].0.rid, "b");
    }

    #[test]
    fn top_rids_helper_returns_just_rids() {
        let idx = TimeDecayIndex::default();
        idx.upsert(TenantId::new(1), "ns", "a", 0.7, Some(24.0), t0());
        idx.upsert(TenantId::new(1), "ns", "b", 0.5, Some(24.0), t0());
        let rids = idx.top_rids(TenantId::new(1), "ns", 10, t0());
        assert_eq!(rids, vec!["a", "b"]);
    }

    #[test]
    fn dyn_dispatch_works() {
        let idx: Arc<dyn TimeDecayProvider> = Arc::new(TimeDecayIndex::default());
        let r = idx.top_decayed(TenantId::new(1), "ns", 10, t0());
        assert!(r.is_empty());
    }

    #[tokio::test]
    async fn invalidation_bus_subscriber_removes_tombstoned_rid() {
        let bus = crate::cache::InvalidationBus::new();
        let idx = TimeDecayIndex::default();
        idx.upsert(TenantId::new(1), "ns_a", "memory_x", 0.5, Some(24.0), t0());
        idx.upsert(TenantId::new(1), "ns_b", "memory_x", 0.5, Some(24.0), t0());
        idx.upsert(TenantId::new(1), "ns_a", "memory_y", 0.5, Some(24.0), t0());

        let handle = spawn_invalidation_bus_subscriber(idx.clone(), &bus);

        bus.publish(InvalidationEvent::Tombstoned {
            tenant_id: TenantId::new(1),
            rid: "memory_x".into(),
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        // memory_x is gone from both namespaces; memory_y remains.
        assert_eq!(
            idx.top_rids(TenantId::new(1), "ns_a", 10, t0()),
            vec!["memory_y"]
        );
        assert!(idx.top_rids(TenantId::new(1), "ns_b", 10, t0()).is_empty());

        handle.abort();
    }
}
