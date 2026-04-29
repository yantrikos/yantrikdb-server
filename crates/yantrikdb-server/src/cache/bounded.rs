//! Shared bounded-cache primitive used by [`super::embedding`] and
//! [`super::query_result`]. Hand-written to avoid pulling an LRU
//! crate; the implementation is approximate-LRU + optional TTL with
//! O(N) eviction-scan at capacity overflow. At realistic capacities
//! (~10k entries) the scan is microseconds — well below any tenable
//! cache hit/miss latency budget.
//!
//! ## Why a single primitive for two caches
//!
//! `EmbeddingCache` (text → vector, no TTL) and `QueryResultCache`
//! (recall-key → results, 60s TTL) share the same "bounded map with
//! recency tracking" shape. One implementation, two configurations.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// Per-entry bookkeeping. `seq` is monotonically increasing — every
/// `get`/`put` bumps a global counter and stamps the entry; eviction
/// targets the entry with the lowest seq.
#[derive(Debug)]
struct Entry<V> {
    value: V,
    inserted_at: Instant,
    seq: u64,
}

/// The cache itself. Cheap to clone (`Arc<Mutex<...>>` internally).
pub struct BoundedCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    inner: Arc<Mutex<HashMap<K, Entry<V>>>>,
    max_entries: usize,
    /// `None` = no TTL. Otherwise entries older than `ttl` are evicted
    /// on `get` (lazy) or via a periodic sweep (caller-driven).
    ttl: Option<Duration>,
    seq: Arc<AtomicU64>,
}

impl<K, V> BoundedCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    pub fn new(max_entries: usize, ttl: Option<Duration>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::with_capacity(max_entries.min(1024)))),
            max_entries,
            ttl,
            seq: Arc::new(AtomicU64::new(0)),
        }
    }

    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Get a clone of the value if present and not expired. Updates
    /// the entry's recency stamp on hit.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut map = self.inner.lock();
        let entry = map.get_mut(key)?;
        if let Some(ttl) = self.ttl {
            if entry.inserted_at.elapsed() > ttl {
                // Expired — evict and report miss.
                map.remove(key);
                return None;
            }
        }
        entry.seq = self.next_seq();
        Some(entry.value.clone())
    }

    /// Insert. If at capacity, evict the entry with the smallest seq
    /// (approximate-LRU). For caches at realistic sizes (10k entries)
    /// the scan is sub-millisecond.
    pub fn put(&self, key: K, value: V) {
        let mut map = self.inner.lock();
        let now = Instant::now();
        let seq = self.next_seq();
        if map.len() >= self.max_entries && !map.contains_key(&key) {
            // Evict the lowest-seq entry.
            if let Some(victim_key) = map
                .iter()
                .min_by_key(|(_, e)| e.seq)
                .map(|(k, _)| k.clone())
            {
                map.remove(&victim_key);
            }
        }
        map.insert(
            key,
            Entry {
                value,
                inserted_at: now,
                seq,
            },
        );
    }

    pub fn invalidate(&self, key: &K) {
        self.inner.lock().remove(key);
    }

    pub fn clear(&self) {
        self.inner.lock().clear();
    }

    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn ttl(&self) -> Option<Duration> {
        self.ttl
    }

    /// Sweep all expired entries (TTL caches only). Returns the count
    /// removed. Cheap when nothing's expired (linear scan, no
    /// allocations beyond the eviction list).
    pub fn sweep_expired(&self) -> usize {
        let Some(ttl) = self.ttl else {
            return 0;
        };
        let mut map = self.inner.lock();
        let now = Instant::now();
        let expired: Vec<K> = map
            .iter()
            .filter(|(_, e)| now.duration_since(e.inserted_at) > ttl)
            .map(|(k, _)| k.clone())
            .collect();
        let n = expired.len();
        for k in expired {
            map.remove(&k);
        }
        n
    }
}

impl<K, V> Clone for BoundedCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            max_entries: self.max_entries,
            ttl: self.ttl,
            seq: Arc::clone(&self.seq),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn put_then_get_roundtrips() {
        let c: BoundedCache<String, i32> = BoundedCache::new(10, None);
        c.put("a".into(), 1);
        assert_eq!(c.get(&"a".to_string()), Some(1));
        assert_eq!(c.get(&"missing".to_string()), None);
    }

    #[test]
    fn invalidate_removes() {
        let c: BoundedCache<&str, i32> = BoundedCache::new(10, None);
        c.put("k", 1);
        c.invalidate(&"k");
        assert!(c.get(&"k").is_none());
    }

    #[test]
    fn clear_empties_cache() {
        let c: BoundedCache<&str, i32> = BoundedCache::new(10, None);
        c.put("a", 1);
        c.put("b", 2);
        assert_eq!(c.len(), 2);
        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn capacity_evicts_oldest_via_lowest_seq() {
        let c: BoundedCache<&str, i32> = BoundedCache::new(2, None);
        c.put("a", 1);
        c.put("b", 2);
        // Touch "a" to make it more recent than "b".
        let _ = c.get(&"a");
        c.put("c", 3); // overflow — evicts "b" (lower seq after our touch)
        assert_eq!(c.get(&"a"), Some(1));
        assert_eq!(c.get(&"b"), None, "b should have been evicted");
        assert_eq!(c.get(&"c"), Some(3));
    }

    #[test]
    fn replacing_existing_key_does_not_evict() {
        let c: BoundedCache<&str, i32> = BoundedCache::new(2, None);
        c.put("a", 1);
        c.put("b", 2);
        // Re-insert with same key — should NOT trigger eviction since
        // contains_key is true.
        c.put("a", 10);
        assert_eq!(c.get(&"a"), Some(10));
        assert_eq!(c.get(&"b"), Some(2));
    }

    #[test]
    fn ttl_expires_entry_on_get() {
        let c: BoundedCache<&str, i32> = BoundedCache::new(10, Some(Duration::from_millis(40)));
        c.put("a", 1);
        assert_eq!(c.get(&"a"), Some(1));
        thread::sleep(Duration::from_millis(80));
        assert_eq!(c.get(&"a"), None, "TTL'd entry should have expired");
        assert_eq!(c.len(), 0, "expired get should evict the entry");
    }

    #[test]
    fn ttl_none_means_entries_persist() {
        let c: BoundedCache<&str, i32> = BoundedCache::new(10, None);
        c.put("a", 1);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(c.get(&"a"), Some(1), "entry must persist without TTL");
    }

    #[test]
    fn sweep_expired_removes_all_stale() {
        let c: BoundedCache<&str, i32> = BoundedCache::new(10, Some(Duration::from_millis(30)));
        c.put("a", 1);
        c.put("b", 2);
        thread::sleep(Duration::from_millis(60));
        c.put("c", 3); // fresh
        let removed = c.sweep_expired();
        assert_eq!(removed, 2);
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(&"c"), Some(3));
    }

    #[test]
    fn sweep_on_no_ttl_cache_is_zero() {
        let c: BoundedCache<&str, i32> = BoundedCache::new(10, None);
        c.put("a", 1);
        assert_eq!(c.sweep_expired(), 0);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn clone_shares_state() {
        let c: BoundedCache<&str, i32> = BoundedCache::new(10, None);
        let c2 = c.clone();
        c.put("a", 1);
        // Clones share the underlying map (Arc<Mutex>).
        assert_eq!(c2.get(&"a"), Some(1));
    }

    #[test]
    fn capacity_one_replaces_on_overflow() {
        let c: BoundedCache<&str, i32> = BoundedCache::new(1, None);
        c.put("a", 1);
        c.put("b", 2);
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(&"a"), None);
        assert_eq!(c.get(&"b"), Some(2));
    }
}
