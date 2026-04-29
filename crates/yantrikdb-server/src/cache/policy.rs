//! `TombstoneProvider` + `Cache` traits + `TombstoneAwareCache` wrapper.
//!
//! Generic over key/value types so any concrete cache (query result,
//! embedding, hot rid) can be wrapped uniformly.

use std::sync::Arc;

use async_trait::async_trait;

use crate::commit::TenantId;

/// What any cache asks of the forget subsystem: "is this rid
/// tombstoned for this tenant?"
///
/// **Stability**: this trait is the seam between RFC 011 (forget) and
/// RFC 015 (caching). Any change to the method shape is a breaking
/// change for both subsystems' implementations. Add new methods only;
/// never modify or remove existing ones without bumping a major.
///
/// ## Stub default
///
/// PR-1 ships [`NoopTombstoneProvider`] which always returns false.
/// Caches built today against the trait are immediately usable; they
/// just don't filter tombstones until RFC 011-A wires a real impl
/// over the commit log.
#[async_trait]
pub trait TombstoneProvider: Send + Sync {
    /// Has this `(tenant_id, rid)` been tombstoned? `true` = the rid
    /// has a TombstoneMemory entry in the log; cache MUST NOT serve any
    /// value associated with this rid.
    ///
    /// Implementations should return quickly — this is consulted on
    /// every cache read on the hot path. Real impls SHOULD use a bloom
    /// filter or in-memory set primed from the commit log.
    async fn is_tombstoned(&self, tenant_id: TenantId, rid: &str) -> bool;

    /// Does the value `V` returned by a cache contain ANY rid that's
    /// tombstoned? Used by caches whose values are aggregates (query
    /// result lists, expanded entity walks). Default impl checks each
    /// rid in turn; concrete impls can short-circuit with bloom filters.
    async fn any_tombstoned(&self, tenant_id: TenantId, rids: &[String]) -> bool {
        for rid in rids {
            if self.is_tombstoned(tenant_id, rid).await {
                return true;
            }
        }
        false
    }
}

/// Stub `TombstoneProvider` that always reports nothing tombstoned.
/// Default for builds without RFC 011-A wired in. Tests may also use
/// this directly to bypass tombstone filtering.
pub struct NoopTombstoneProvider;

#[async_trait]
impl TombstoneProvider for NoopTombstoneProvider {
    async fn is_tombstoned(&self, _tenant_id: TenantId, _rid: &str) -> bool {
        false
    }
}

/// Trait every concrete cache satisfies. Generic over key + value.
///
/// Implementations SHOULD be `Send + Sync` so caches can be shared
/// behind `Arc`. The async signatures keep the door open for caches
/// backed by remote stores (Redis, memcached) without breaking the
/// trait shape.
#[async_trait]
pub trait Cache<K, V>: Send + Sync
where
    K: Send + Sync,
    V: Send + Sync + Clone,
{
    async fn get(&self, key: &K) -> Option<V>;

    async fn put(&self, key: K, value: V);

    async fn invalidate(&self, key: &K);

    /// Wipe everything. Used on tenant drop, full cache reset.
    async fn clear(&self);

    /// Approximate count for /metrics. Allowed to be slightly stale
    /// (caches that estimate are fine).
    async fn len(&self) -> usize;
}

/// Trait for values that expose the rids they cover. The
/// [`TombstoneAwareCache`] wrapper consults this to decide whether a
/// cached value should be served despite tombstones in the log.
///
/// Implementations: a query-result `Vec<RecallResult>` returns the rids
/// of every result; an expanded entity record returns its own rid plus
/// any related rids; a single embedding doesn't have rids and skips
/// the wrapper (use a non-tombstone-aware cache for that).
pub trait RidKeyed {
    /// Tenant the value belongs to. Required because tombstone scope
    /// is per-tenant.
    fn tenant_id(&self) -> TenantId;
    /// All rids this cached value covers. Empty = no tombstone check.
    fn rids(&self) -> Vec<String>;
}

/// Wrapper that consults a [`TombstoneProvider`] before serving cached
/// values. On `get`, if any rid in the cached `V` is tombstoned, the
/// wrapper invalidates + returns `None` — same behavior as a cache miss.
///
/// `put` and `invalidate` and `clear` and `len` pass through unchanged.
///
/// ## Cost model
///
/// Every `get` does a tombstone check. With a real `TombstoneProvider`
/// (RFC 011-A) backed by a bloom filter, this is ~50-100ns per check —
/// dominated by hash computation. The stub provider is essentially free.
///
/// Caches whose values are large aggregates (Vec<RecallResult> with 50
/// entries) are dominated by the `any_tombstoned` walk. RFC 011-A's
/// real provider will short-circuit via bloom filter so the walk is
/// 50 hashes rather than 50 SQL lookups.
pub struct TombstoneAwareCache<K, V, C>
where
    K: Send + Sync,
    V: Send + Sync + Clone + RidKeyed,
    C: Cache<K, V>,
{
    inner: C,
    tombstones: Arc<dyn TombstoneProvider>,
    _phantom: std::marker::PhantomData<(K, V)>,
}

impl<K, V, C> TombstoneAwareCache<K, V, C>
where
    K: Send + Sync,
    V: Send + Sync + Clone + RidKeyed,
    C: Cache<K, V>,
{
    pub fn new(inner: C, tombstones: Arc<dyn TombstoneProvider>) -> Self {
        Self {
            inner,
            tombstones,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<K, V, C> Cache<K, V> for TombstoneAwareCache<K, V, C>
where
    K: Send + Sync,
    V: Send + Sync + Clone + RidKeyed,
    C: Cache<K, V>,
{
    async fn get(&self, key: &K) -> Option<V> {
        let value = self.inner.get(key).await?;
        let rids = value.rids();
        if rids.is_empty() {
            return Some(value);
        }
        let tenant = value.tenant_id();
        if self.tombstones.any_tombstoned(tenant, &rids).await {
            // At least one rid in the cached value has been tombstoned
            // since the value was cached. Invalidate + return None so
            // the caller re-fetches a fresh result that excludes the
            // forgotten rid.
            self.inner.invalidate(key).await;
            return None;
        }
        Some(value)
    }

    async fn put(&self, key: K, value: V) {
        self.inner.put(key, value).await;
    }

    async fn invalidate(&self, key: &K) {
        self.inner.invalidate(key).await;
    }

    async fn clear(&self) {
        self.inner.clear().await;
    }

    async fn len(&self) -> usize {
        self.inner.len().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::RwLock;
    use std::collections::HashMap;

    /// Test-only in-memory `Cache` impl. Concrete caches in 015-B will
    /// follow the same shape but with bounded capacity, TTL, etc.
    struct MemCache<K: Eq + std::hash::Hash + Clone + Send + Sync, V: Send + Sync + Clone> {
        inner: RwLock<HashMap<K, V>>,
    }

    impl<K: Eq + std::hash::Hash + Clone + Send + Sync, V: Send + Sync + Clone> MemCache<K, V> {
        fn new() -> Self {
            Self {
                inner: RwLock::new(HashMap::new()),
            }
        }
    }

    #[async_trait]
    impl<K, V> Cache<K, V> for MemCache<K, V>
    where
        K: Eq + std::hash::Hash + Clone + Send + Sync,
        V: Send + Sync + Clone,
    {
        async fn get(&self, key: &K) -> Option<V> {
            self.inner.read().get(key).cloned()
        }
        async fn put(&self, key: K, value: V) {
            self.inner.write().insert(key, value);
        }
        async fn invalidate(&self, key: &K) {
            self.inner.write().remove(key);
        }
        async fn clear(&self) {
            self.inner.write().clear();
        }
        async fn len(&self) -> usize {
            self.inner.read().len()
        }
    }

    /// Test value that knows its rids.
    #[derive(Clone)]
    struct CachedQueryResult {
        tenant: TenantId,
        rids: Vec<String>,
    }

    impl RidKeyed for CachedQueryResult {
        fn tenant_id(&self) -> TenantId {
            self.tenant
        }
        fn rids(&self) -> Vec<String> {
            self.rids.clone()
        }
    }

    /// Test `TombstoneProvider` with a hard-coded set of tombstones.
    struct StaticTombstones {
        tombstoned: Vec<(TenantId, String)>,
    }

    #[async_trait]
    impl TombstoneProvider for StaticTombstones {
        async fn is_tombstoned(&self, tenant_id: TenantId, rid: &str) -> bool {
            self.tombstoned
                .iter()
                .any(|(t, r)| *t == tenant_id && r == rid)
        }
    }

    #[tokio::test]
    async fn noop_provider_says_nothing_is_tombstoned() {
        let p = NoopTombstoneProvider;
        for rid in ["a", "b", "c"] {
            assert!(!p.is_tombstoned(TenantId::new(1), rid).await);
        }
    }

    #[tokio::test]
    async fn noop_provider_any_tombstoned_returns_false_for_any_input() {
        let p = NoopTombstoneProvider;
        assert!(
            !p.any_tombstoned(
                TenantId::new(1),
                &["a".into(), "b".into(), "c".into()]
            )
            .await
        );
        assert!(!p.any_tombstoned(TenantId::new(1), &[]).await);
    }

    #[tokio::test]
    async fn tombstone_aware_cache_serves_when_no_tombstones() {
        let inner = MemCache::<String, CachedQueryResult>::new();
        let provider: Arc<dyn TombstoneProvider> = Arc::new(NoopTombstoneProvider);
        let cache = TombstoneAwareCache::new(inner, provider);

        let key = "query_42".to_string();
        let value = CachedQueryResult {
            tenant: TenantId::new(1),
            rids: vec!["mem_a".into(), "mem_b".into()],
        };
        cache.put(key.clone(), value).await;

        let got = cache.get(&key).await;
        assert!(got.is_some());
        assert_eq!(got.unwrap().rids, vec!["mem_a", "mem_b"]);
    }

    #[tokio::test]
    async fn tombstone_aware_cache_invalidates_when_a_rid_is_tombstoned() {
        let inner = MemCache::<String, CachedQueryResult>::new();
        // Pre-tombstone mem_b for tenant 1.
        let provider: Arc<dyn TombstoneProvider> = Arc::new(StaticTombstones {
            tombstoned: vec![(TenantId::new(1), "mem_b".into())],
        });
        let cache = TombstoneAwareCache::new(inner, provider);

        let key = "query_with_b".to_string();
        let value = CachedQueryResult {
            tenant: TenantId::new(1),
            rids: vec!["mem_a".into(), "mem_b".into()],
        };
        cache.put(key.clone(), value).await;
        // First get: detects mem_b is tombstoned, invalidates, returns None.
        let got = cache.get(&key).await;
        assert!(got.is_none());
        // Second get: cache is now empty. Confirms invalidation happened.
        let again = cache.get(&key).await;
        assert!(again.is_none());
        assert_eq!(cache.len().await, 0);
    }

    #[tokio::test]
    async fn tombstone_aware_cache_serves_when_tombstone_is_for_other_tenant() {
        // Tombstone scope is per-tenant; tenant 2's tombstones must not
        // affect tenant 1's cached values.
        let inner = MemCache::<String, CachedQueryResult>::new();
        let provider: Arc<dyn TombstoneProvider> = Arc::new(StaticTombstones {
            tombstoned: vec![(TenantId::new(2), "mem_b".into())],
        });
        let cache = TombstoneAwareCache::new(inner, provider);

        let value = CachedQueryResult {
            tenant: TenantId::new(1),
            rids: vec!["mem_a".into(), "mem_b".into()],
        };
        cache.put("k".to_string(), value).await;

        // Tenant 1's value should still be served — the tombstone is
        // for tenant 2.
        assert!(cache.get(&"k".to_string()).await.is_some());
    }

    #[tokio::test]
    async fn tombstone_aware_cache_skips_check_for_empty_rids() {
        // Caches whose values don't carry rids (e.g. embedding cache)
        // skip the tombstone check.
        let inner = MemCache::<String, CachedQueryResult>::new();
        let provider: Arc<dyn TombstoneProvider> = Arc::new(StaticTombstones {
            tombstoned: vec![(TenantId::new(1), "anything".into())],
        });
        let cache = TombstoneAwareCache::new(inner, provider);

        let value = CachedQueryResult {
            tenant: TenantId::new(1),
            rids: vec![],
        };
        cache.put("k".to_string(), value).await;
        // Empty rids → no tombstone check → served unchanged.
        assert!(cache.get(&"k".to_string()).await.is_some());
    }

    #[tokio::test]
    async fn tombstone_aware_cache_passes_through_put_invalidate_clear() {
        let inner = MemCache::<String, CachedQueryResult>::new();
        let provider: Arc<dyn TombstoneProvider> = Arc::new(NoopTombstoneProvider);
        let cache = TombstoneAwareCache::new(inner, provider);

        let v = CachedQueryResult {
            tenant: TenantId::new(1),
            rids: vec!["x".into()],
        };
        cache.put("k1".to_string(), v.clone()).await;
        cache.put("k2".to_string(), v.clone()).await;
        assert_eq!(cache.len().await, 2);
        cache.invalidate(&"k1".to_string()).await;
        assert_eq!(cache.len().await, 1);
        cache.clear().await;
        assert_eq!(cache.len().await, 0);
    }

    #[tokio::test]
    async fn dyn_dispatch_works() {
        // The whole point of the trait is dyn-dispatchable so the
        // server can hold Arc<dyn TombstoneProvider> in AppState and
        // swap NoopTombstoneProvider → CommitLogTombstoneProvider
        // (RFC 011-A) at startup without rebuilding caches.
        let provider: Arc<dyn TombstoneProvider> = Arc::new(NoopTombstoneProvider);
        assert!(!provider.is_tombstoned(TenantId::new(1), "x").await);
    }
}
