//! Query result cache — keyed by (query_hash, namespace, top_k,
//! expand_entities, filters) → cached recall result list.
//!
//! ## Why this cache
//!
//! Production agents repeat queries constantly: "what was the user's
//! last preference?", "find any skills tagged X", periodic health
//! sweeps, etc. Without a query cache, every recall does:
//! 1. Embed the query (~30 ms on CPU EP)
//! 2. HNSW search (~1-100 ms depending on top_k + index size)
//! 3. Engine post-processing (filtering, merging, sorting)
//!
//! With this cache, repeated identical queries skip every step and
//! return in microseconds. Combined with [`super::embedding`] (which
//! handles the case where two different query *strings* embed
//! identically, e.g. trivial whitespace differences after a normalizer),
//! the substrate's cold-vs-hot recall ratio is well into the
//! 100×-1000× range for production agent workloads.
//!
//! ## Tombstone awareness
//!
//! [`CachedQueryResult`] implements [`super::policy::RidKeyed`] so the
//! cache wrapped in [`super::TombstoneAwareCache`] consults the
//! tombstone index on every `get`. A cached result containing a
//! since-tombstoned rid invalidates + returns `None` (forcing a fresh
//! recall that will respect the tombstone).
//!
//! ## TTL
//!
//! Default 60 seconds. Matches the agent-memory use case: a fresh
//! commit shows up in recall within ~60 s of being written, even if a
//! caller is hammering the same query. Tunable per-deployment via
//! [`QueryResultCacheConfig`].

use async_trait::async_trait;
use blake3;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::bounded::BoundedCache;
use super::policy::{Cache, RidKeyed};
use crate::commit::TenantId;

/// Cache key. Hashing-by-fields rather than serde-hash so the key is
/// `Copy` and cheap to pass around. The `params_hash` rolls
/// expand_entities + any filter params into one digest so callers
/// can extend the parameter set without churning the key shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryCacheKey {
    /// blake3 of the raw query text. Lets two identical queries
    /// from different callers hit the same cache entry.
    pub query_hash: [u8; 32],
    /// Tenant scope. Two tenants with the same query against the
    /// same namespace must NOT share results.
    pub tenant_id: TenantId,
    /// Per-tenant scoping within the recall.
    pub namespace_hash: [u8; 32],
    /// Top-K — different K's may return overlapping but distinct
    /// result lists; cache them separately.
    pub top_k: u32,
    /// Hash of the rest of the recall params (expand_entities,
    /// optional filters, model_version, etc.). See
    /// [`QueryCacheKeyBuilder`].
    pub params_hash: [u8; 32],
}

/// Builder for `QueryCacheKey`. Hashes inputs incrementally so
/// callers don't have to construct intermediate strings.
pub struct QueryCacheKeyBuilder {
    tenant_id: TenantId,
    query_hasher: blake3::Hasher,
    namespace_hasher: blake3::Hasher,
    params_hasher: blake3::Hasher,
    top_k: u32,
}

impl QueryCacheKeyBuilder {
    pub fn new(tenant_id: TenantId) -> Self {
        Self {
            tenant_id,
            query_hasher: blake3::Hasher::new(),
            namespace_hasher: blake3::Hasher::new(),
            params_hasher: blake3::Hasher::new(),
            top_k: 0,
        }
    }

    pub fn query(mut self, text: &str) -> Self {
        self.query_hasher.update(text.as_bytes());
        self
    }

    pub fn namespace(mut self, namespace: &str) -> Self {
        self.namespace_hasher.update(namespace.as_bytes());
        self
    }

    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn expand_entities(mut self, expand: bool) -> Self {
        self.params_hasher.update(b"expand=");
        self.params_hasher.update(if expand { b"1" } else { b"0" });
        self.params_hasher.update(b";");
        self
    }

    pub fn model_version(mut self, model: &str) -> Self {
        self.params_hasher.update(b"model=");
        self.params_hasher.update(model.as_bytes());
        self.params_hasher.update(b";");
        self
    }

    /// Roll an arbitrary `(name, value)` pair into the params hash.
    /// Use this for any future filter parameter without changing the
    /// cache key shape — everything serializes through `params_hash`.
    pub fn param(mut self, name: &str, value: &str) -> Self {
        self.params_hasher.update(name.as_bytes());
        self.params_hasher.update(b"=");
        self.params_hasher.update(value.as_bytes());
        self.params_hasher.update(b";");
        self
    }

    pub fn build(self) -> QueryCacheKey {
        let mut q = [0u8; 32];
        q.copy_from_slice(self.query_hasher.finalize().as_bytes());
        let mut n = [0u8; 32];
        n.copy_from_slice(self.namespace_hasher.finalize().as_bytes());
        let mut p = [0u8; 32];
        p.copy_from_slice(self.params_hasher.finalize().as_bytes());
        QueryCacheKey {
            query_hash: q,
            tenant_id: self.tenant_id,
            namespace_hash: n,
            top_k: self.top_k,
            params_hash: p,
        }
    }
}

/// Cached recall result. Carries the rids it covers so the
/// tombstone-aware wrapper can short-circuit serving stale results.
///
/// Generic over `R` (the concrete result-row type) so this cache
/// composes with any recall response shape without binding to a
/// specific module's struct. Production callers will typically use
/// `R = serde_json::Value` (matching the HTTP response shape) or a
/// typed `RecallResult` if one is exposed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachedQueryResult<R> {
    pub tenant_id: TenantId,
    /// All rids covered by the cached results, in result order.
    /// Used by [`super::TombstoneAwareCache`] to filter on `get`.
    pub rids: Vec<String>,
    /// The actual cached result rows. Caller-defined shape.
    pub results: Vec<R>,
}

impl<R> RidKeyed for CachedQueryResult<R> {
    fn tenant_id(&self) -> TenantId {
        self.tenant_id
    }
    fn rids(&self) -> Vec<String> {
        self.rids.clone()
    }
}

/// Tunable parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryResultCacheConfig {
    /// Hard cap on cached query results.
    pub max_entries: usize,
    /// TTL on each entry. After this, recall is recomputed on next
    /// hit. Defaults to 60s — agents see fresh commits within a minute
    /// even when hammering the same query.
    pub ttl_secs: u64,
}

impl Default for QueryResultCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            ttl_secs: 60,
        }
    }
}

/// Concrete cache. Cheap to clone.
#[derive(Clone)]
pub struct QueryResultCache<R>
where
    R: Clone + Send + Sync + 'static,
{
    inner: BoundedCache<QueryCacheKey, CachedQueryResult<R>>,
}

impl<R> QueryResultCache<R>
where
    R: Clone + Send + Sync + 'static,
{
    pub fn new(config: QueryResultCacheConfig) -> Self {
        Self {
            inner: BoundedCache::new(
                config.max_entries,
                Some(Duration::from_secs(config.ttl_secs)),
            ),
        }
    }

    pub fn config_max_entries(&self) -> usize {
        self.inner.max_entries()
    }

    pub fn config_ttl(&self) -> Option<Duration> {
        self.inner.ttl()
    }

    /// Sweep expired entries (TTL pass). Caller-driven — typical
    /// pattern: a background task fires this every 30-60 s to keep
    /// memory in check between organic evictions.
    pub fn sweep_expired(&self) -> usize {
        self.inner.sweep_expired()
    }
}

#[async_trait]
impl<R> Cache<QueryCacheKey, CachedQueryResult<R>> for QueryResultCache<R>
where
    R: Clone + Send + Sync + 'static,
{
    async fn get(&self, key: &QueryCacheKey) -> Option<CachedQueryResult<R>> {
        self.inner.get(key)
    }

    async fn put(&self, key: QueryCacheKey, value: CachedQueryResult<R>) {
        self.inner.put(key, value);
    }

    async fn invalidate(&self, key: &QueryCacheKey) {
        self.inner.invalidate(key);
    }

    async fn clear(&self) {
        self.inner.clear();
    }

    async fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::policy::{NoopTombstoneProvider, TombstoneAwareCache, TombstoneProvider};
    use std::sync::Arc;
    use std::thread;

    fn key(text: &str, namespace: &str, top_k: u32) -> QueryCacheKey {
        QueryCacheKeyBuilder::new(TenantId::new(1))
            .query(text)
            .namespace(namespace)
            .top_k(top_k)
            .expand_entities(false)
            .model_version("minilm-l6-v2")
            .build()
    }

    fn cached_result(rids: Vec<&str>) -> CachedQueryResult<String> {
        CachedQueryResult {
            tenant_id: TenantId::new(1),
            rids: rids.iter().map(|s| s.to_string()).collect(),
            results: rids.iter().map(|s| format!("row[{s}]")).collect(),
        }
    }

    #[test]
    fn key_builder_is_deterministic() {
        let k1 = key("query", "ns", 10);
        let k2 = key("query", "ns", 10);
        assert_eq!(k1, k2);
    }

    #[test]
    fn key_changes_on_query_text_change() {
        let k1 = key("query a", "ns", 10);
        let k2 = key("query b", "ns", 10);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_on_namespace_change() {
        let k1 = key("query", "ns_a", 10);
        let k2 = key("query", "ns_b", 10);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_on_top_k_change() {
        let k1 = key("query", "ns", 10);
        let k2 = key("query", "ns", 20);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_on_tenant_change() {
        let k1 = QueryCacheKeyBuilder::new(TenantId::new(1))
            .query("q")
            .build();
        let k2 = QueryCacheKeyBuilder::new(TenantId::new(2))
            .query("q")
            .build();
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_on_expand_entities_flip() {
        let k1 = QueryCacheKeyBuilder::new(TenantId::new(1))
            .query("q")
            .expand_entities(false)
            .build();
        let k2 = QueryCacheKeyBuilder::new(TenantId::new(1))
            .query("q")
            .expand_entities(true)
            .build();
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_on_model_upgrade() {
        // Model upgrade must invalidate cached query results — the
        // vector space is different, so cached neighbors are wrong.
        let k1 = QueryCacheKeyBuilder::new(TenantId::new(1))
            .query("q")
            .model_version("minilm-l6-v2")
            .build();
        let k2 = QueryCacheKeyBuilder::new(TenantId::new(1))
            .query("q")
            .model_version("bge-base")
            .build();
        assert_ne!(k1, k2);
    }

    #[tokio::test]
    async fn put_then_get_returns_cached_results() {
        let c: QueryResultCache<String> = QueryResultCache::new(QueryResultCacheConfig::default());
        let k = key("q", "n", 10);
        c.put(k, cached_result(vec!["r1", "r2", "r3"])).await;
        let back = c.get(&k).await.unwrap();
        assert_eq!(back.rids, vec!["r1", "r2", "r3"]);
        assert_eq!(back.results.len(), 3);
    }

    #[tokio::test]
    async fn miss_returns_none() {
        let c: QueryResultCache<String> = QueryResultCache::new(QueryResultCacheConfig::default());
        assert!(c.get(&key("q", "n", 10)).await.is_none());
    }

    #[tokio::test]
    async fn ttl_expires_entry() {
        let c: QueryResultCache<String> = QueryResultCache::new(QueryResultCacheConfig {
            max_entries: 10,
            ttl_secs: 0, // 0-second TTL = expire instantly on get
        });
        let k = key("q", "n", 10);
        c.put(k, cached_result(vec!["r1"])).await;
        // Sleep slightly so the elapsed > 0 check definitely fires.
        thread::sleep(Duration::from_millis(20));
        assert!(c.get(&k).await.is_none(), "TTL=0 should always be expired");
    }

    #[tokio::test]
    async fn cached_query_result_implements_rid_keyed() {
        let r = cached_result(vec!["r1", "r2"]);
        assert_eq!(
            <CachedQueryResult<String> as RidKeyed>::tenant_id(&r),
            TenantId::new(1)
        );
        assert_eq!(
            <CachedQueryResult<String> as RidKeyed>::rids(&r),
            vec!["r1", "r2"]
        );
    }

    #[tokio::test]
    async fn empty_rids_means_no_tombstone_check_in_wrapper() {
        // CachedQueryResult with no rids is the "nothing to filter"
        // path the TombstoneAwareCache short-circuits on. Confirm rids()
        // returns an empty vec rather than accidentally including
        // sentinel data.
        let r: CachedQueryResult<String> = CachedQueryResult {
            tenant_id: TenantId::new(1),
            rids: vec![],
            results: vec![],
        };
        assert!(<CachedQueryResult<String> as RidKeyed>::rids(&r).is_empty());
    }

    #[tokio::test]
    async fn composes_with_tombstone_aware_wrapper() {
        // End-to-end: wrap QueryResultCache in TombstoneAwareCache
        // with the no-op provider; cached entries should serve
        // unchanged.
        let inner: QueryResultCache<String> =
            QueryResultCache::new(QueryResultCacheConfig::default());
        let wrapped: TombstoneAwareCache<QueryCacheKey, CachedQueryResult<String>, _> =
            TombstoneAwareCache::new(inner, Arc::new(NoopTombstoneProvider));

        let k = key("q", "n", 10);
        wrapped.put(k, cached_result(vec!["r1", "r2"])).await;
        let back = wrapped.get(&k).await.unwrap();
        assert_eq!(back.rids, vec!["r1", "r2"]);
    }

    /// A test-only `TombstoneProvider` that says "rid X is tombstoned"
    /// for a fixed list of rids.
    struct FakeTombstones {
        rids: std::collections::HashSet<String>,
        tenant: TenantId,
    }

    #[async_trait]
    impl TombstoneProvider for FakeTombstones {
        async fn is_tombstoned(&self, tenant_id: TenantId, rid: &str) -> bool {
            tenant_id == self.tenant && self.rids.contains(rid)
        }
    }

    #[tokio::test]
    async fn tombstone_aware_wrapper_invalidates_on_match() {
        // The whole point of RidKeyed integration: a cached result
        // referencing a tombstoned rid must NOT be served.
        let inner: QueryResultCache<String> =
            QueryResultCache::new(QueryResultCacheConfig::default());
        let mut tomb_set = std::collections::HashSet::new();
        tomb_set.insert("r2".to_string());
        let wrapped = TombstoneAwareCache::new(
            inner,
            Arc::new(FakeTombstones {
                rids: tomb_set,
                tenant: TenantId::new(1),
            }),
        );

        let k = key("q", "n", 10);
        wrapped.put(k, cached_result(vec!["r1", "r2", "r3"])).await;
        // r2 is tombstoned; the wrapper invalidates and returns None.
        assert!(wrapped.get(&k).await.is_none());
        // The wrapper also evicted the entry from the underlying cache.
        assert_eq!(wrapped.len().await, 0);
    }

    #[tokio::test]
    async fn capacity_evicts_least_recently_used() {
        let c: QueryResultCache<String> = QueryResultCache::new(QueryResultCacheConfig {
            max_entries: 2,
            ttl_secs: 60,
        });
        let k1 = key("q1", "n", 10);
        let k2 = key("q2", "n", 10);
        let k3 = key("q3", "n", 10);
        c.put(k1, cached_result(vec!["r1"])).await;
        c.put(k2, cached_result(vec!["r2"])).await;
        let _ = c.get(&k1).await; // touch k1
        c.put(k3, cached_result(vec!["r3"])).await; // overflow → evict k2
        assert!(c.get(&k1).await.is_some());
        assert!(c.get(&k2).await.is_none());
        assert!(c.get(&k3).await.is_some());
    }

    #[tokio::test]
    async fn sweep_expired_caller_driven() {
        let c: QueryResultCache<String> = QueryResultCache::new(QueryResultCacheConfig {
            max_entries: 10,
            ttl_secs: 0,
        });
        c.put(key("q1", "n", 10), cached_result(vec!["r1"])).await;
        c.put(key("q2", "n", 10), cached_result(vec!["r2"])).await;
        thread::sleep(Duration::from_millis(20));
        let removed = c.sweep_expired();
        assert_eq!(removed, 2);
        assert_eq!(c.len().await, 0);
    }

    #[test]
    fn config_defaults_match_spec() {
        let cfg = QueryResultCacheConfig::default();
        assert_eq!(cfg.max_entries, 10_000);
        assert_eq!(cfg.ttl_secs, 60);
    }
}
