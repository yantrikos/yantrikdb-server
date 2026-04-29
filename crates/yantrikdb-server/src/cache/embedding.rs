//! Embedding cache — `(text, model_version) -> Vec<f32>`.
//!
//! ## Why this cache
//!
//! Every call to MiniLM-L6-v2 forward pass costs ~10-30 ms on CPU EP.
//! Production agents repeat the same query text constantly ("what was
//! the user's last preference?"), and every recall internally embeds
//! the query before HNSW search. Caching the
//! text → vector mapping eliminates the redundant inference.
//!
//! ## Design
//!
//! Key: blake3 digest of `text || "\0" || model_version`. The
//! null-byte separator prevents collisions where text "abc" + model
//! "def" hashes the same as "abcdef" + empty model.
//! 32-byte digest stored as `[u8; 32]` — zero-allocation key.
//!
//! Value: `Vec<f32>` — for MiniLM-L6-v2 that's 384 × 4 = 1536 bytes
//! per entry, plus the small overhead. Default capacity 10,000
//! entries → ~15 MB of cache memory. Tunable via
//! [`EmbeddingCacheConfig`].
//!
//! No TTL: the embedding for a given text under a fixed model is
//! deterministic. Entries only get evicted when the cache fills up
//! (LRU-by-recency), or on explicit `invalidate`/`clear`.
//!
//! Not tombstone-aware: embeddings reference text, not memory rids.
//! The query-result cache (next file) wraps tombstone-awareness.

use async_trait::async_trait;
use blake3;
use serde::{Deserialize, Serialize};

use super::bounded::BoundedCache;
use super::policy::Cache;

/// Stable on-disk key for the embedding cache. 32-byte blake3 digest;
/// `Copy` for cheap pass-by-value through async paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbeddingCacheKey(pub [u8; 32]);

impl EmbeddingCacheKey {
    /// Build a key from `(text, model_version)`. The model_version
    /// is included in the hash so a model upgrade automatically
    /// invalidates every old entry on lookup (no manual flush
    /// required).
    pub fn for_text(text: &str, model_version: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(text.as_bytes());
        hasher.update(b"\0");
        hasher.update(model_version.as_bytes());
        let digest = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(digest.as_bytes());
        Self(bytes)
    }
}

/// Tunable parameters. Defaults are sane for the built-in MiniLM
/// embedder; ops can dial up `max_entries` if memory permits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingCacheConfig {
    /// Hard cap on stored entries. At 384-dim f32, each entry is
    /// ~1.5 KB, so 10k entries ≈ 15 MB.
    pub max_entries: usize,
}

impl Default for EmbeddingCacheConfig {
    fn default() -> Self {
        Self { max_entries: 10_000 }
    }
}

/// Concrete cache implementation. Cheap to clone (`Arc` internally).
#[derive(Clone)]
pub struct EmbeddingCache {
    inner: BoundedCache<EmbeddingCacheKey, Vec<f32>>,
}

impl EmbeddingCache {
    pub fn new(config: EmbeddingCacheConfig) -> Self {
        Self {
            inner: BoundedCache::new(config.max_entries, None),
        }
    }

    /// Convenience: derive the key + look up in one call. Returns
    /// `Some(vector_clone)` on hit.
    pub fn get_for_text(&self, text: &str, model_version: &str) -> Option<Vec<f32>> {
        self.inner.get(&EmbeddingCacheKey::for_text(text, model_version))
    }

    /// Convenience: derive the key + insert in one call.
    pub fn put_for_text(&self, text: &str, model_version: &str, vector: Vec<f32>) {
        self.inner
            .put(EmbeddingCacheKey::for_text(text, model_version), vector);
    }

    pub fn config_max_entries(&self) -> usize {
        self.inner.max_entries()
    }
}

#[async_trait]
impl Cache<EmbeddingCacheKey, Vec<f32>> for EmbeddingCache {
    async fn get(&self, key: &EmbeddingCacheKey) -> Option<Vec<f32>> {
        self.inner.get(key)
    }

    async fn put(&self, key: EmbeddingCacheKey, value: Vec<f32>) {
        self.inner.put(key, value);
    }

    async fn invalidate(&self, key: &EmbeddingCacheKey) {
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

    fn vec384(seed: u32) -> Vec<f32> {
        (0..384).map(|i| (seed.wrapping_add(i)) as f32 / 1000.0).collect()
    }

    #[test]
    fn key_is_deterministic_for_same_text_and_model() {
        let k1 = EmbeddingCacheKey::for_text("hello world", "minilm-l6-v2");
        let k2 = EmbeddingCacheKey::for_text("hello world", "minilm-l6-v2");
        assert_eq!(k1, k2);
    }

    #[test]
    fn key_changes_on_model_upgrade() {
        // Critical: a model upgrade must NOT serve stale embeddings.
        // The model_version in the key namespace ensures every
        // post-upgrade lookup is a clean miss.
        let k1 = EmbeddingCacheKey::for_text("hello", "minilm-l6-v2");
        let k2 = EmbeddingCacheKey::for_text("hello", "bge-base");
        assert_ne!(k1, k2);
    }

    #[test]
    fn null_byte_separator_prevents_concat_collision() {
        // Without a separator, ("ab", "cd") and ("a", "bcd") would
        // share a hash. The null byte forces them apart.
        let k1 = EmbeddingCacheKey::for_text("ab", "cd");
        let k2 = EmbeddingCacheKey::for_text("a", "bcd");
        assert_ne!(k1, k2);
    }

    #[tokio::test]
    async fn put_then_get_returns_vector() {
        let c = EmbeddingCache::new(EmbeddingCacheConfig::default());
        let v = vec384(42);
        c.put_for_text("query text", "minilm-l6-v2", v.clone());
        let back = c.get_for_text("query text", "minilm-l6-v2").unwrap();
        assert_eq!(back, v);
    }

    #[tokio::test]
    async fn miss_returns_none() {
        let c = EmbeddingCache::new(EmbeddingCacheConfig::default());
        assert!(c.get_for_text("never inserted", "model").is_none());
    }

    #[tokio::test]
    async fn cache_trait_dispatch_works() {
        // Use the Cache trait surface (what TombstoneAwareCache and
        // generic plumbing will go through).
        let c: Box<dyn Cache<EmbeddingCacheKey, Vec<f32>>> =
            Box::new(EmbeddingCache::new(EmbeddingCacheConfig::default()));
        let key = EmbeddingCacheKey::for_text("hi", "m");
        c.put(key, vec384(1)).await;
        assert!(c.get(&key).await.is_some());
        c.invalidate(&key).await;
        assert!(c.get(&key).await.is_none());
    }

    #[tokio::test]
    async fn capacity_bound_evicts_least_recent() {
        let c = EmbeddingCache::new(EmbeddingCacheConfig { max_entries: 2 });
        c.put_for_text("a", "m", vec384(1));
        c.put_for_text("b", "m", vec384(2));
        // Touch "a" — bump its recency.
        let _ = c.get_for_text("a", "m");
        c.put_for_text("c", "m", vec384(3)); // overflow → evicts "b"
        assert!(c.get_for_text("a", "m").is_some());
        assert!(c.get_for_text("b", "m").is_none(), "b should be evicted");
        assert!(c.get_for_text("c", "m").is_some());
    }

    #[tokio::test]
    async fn clear_drops_everything() {
        let c = EmbeddingCache::new(EmbeddingCacheConfig::default());
        c.put_for_text("a", "m", vec384(1));
        c.put_for_text("b", "m", vec384(2));
        assert_eq!(<EmbeddingCache as Cache<_, _>>::len(&c).await, 2);
        <EmbeddingCache as Cache<_, _>>::clear(&c).await;
        assert_eq!(<EmbeddingCache as Cache<_, _>>::len(&c).await, 0);
    }

    #[test]
    fn config_default_is_10k_entries() {
        // Pin the default — operators read documented defaults.
        let cfg = EmbeddingCacheConfig::default();
        assert_eq!(cfg.max_entries, 10_000);
    }
}
