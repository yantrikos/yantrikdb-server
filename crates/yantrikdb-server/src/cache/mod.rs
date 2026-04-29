//! RFC 015-A — Cache invalidation substrate (interface-first).
//!
//! ## Why this module exists
//!
//! Per gpt-5.5 brainstorm 7c36ea8b: RFC 011 (forget) and RFC 015 (caching)
//! have a circular dependency. Caches need to know about tombstones to
//! avoid serving deleted data; tombstones need an invalidation channel
//! to push notifications to caches.
//!
//! The brainstorm-locked resolution: **015-A ships first with
//! interface-only contracts**:
//!
//! 1. [`TombstoneProvider`] trait — what any cache asks: "is this rid
//!    tombstoned for this tenant?"
//! 2. [`NoopTombstoneProvider`] — stub impl returning "no tombstones."
//!    Lets early integration land before RFC 011 ships the real impl.
//! 3. [`TombstoneAwareCache`] — generic wrapper that filters cached
//!    entries through a `TombstoneProvider` on every read.
//! 4. [`InvalidationBus`] — fan-out channel for cache invalidation
//!    events. Workers subscribe; the commit-log apply path publishes.
//!
//! After this PR ships:
//! - RFC 011-A implements `TombstoneProvider` over `memory_commit_log`.
//! - RFC 015-B-1 builds query-result-cache + embedding-cache on top of
//!   `TombstoneAwareCache`.
//! - RFC 015-B-2 builds hybrid BM25 + rerank using these primitives.
//!
//! ## What this is NOT
//!
//! - A concrete cache. Concrete caches (query result, embedding, hot rid)
//!   are RFC 015-B work.
//! - A complete forget implementation. RFC 011-A wires
//!   `TombstoneProvider` to the actual commit log; 011-B adds the
//!   HNSW delete queue and crypto-shred.
//!
//! ## Generic over `K, V` because we don't know the concrete shapes yet
//!
//! Different caches key on different things:
//! - Query result cache: `(query_embedding_hash, namespace, top_k)` →
//!   `Vec<RecallResult>`
//! - Embedding cache: `text_hash` → `Vec<f32>`
//! - Hot rid cache: `rid` → expanded memory record
//!
//! The trait + wrapper are generic so we don't bake in any concrete
//! shape. Each consumer picks K and V to match its workload.

pub mod bounded;
pub mod embedding;
pub mod invalidation;
pub mod policy;
pub mod query_result;

pub use embedding::{EmbeddingCache, EmbeddingCacheConfig, EmbeddingCacheKey};
pub use invalidation::{InvalidationBus, InvalidationEvent};
pub use policy::{Cache, NoopTombstoneProvider, RidKeyed, TombstoneAwareCache, TombstoneProvider};
pub use query_result::{
    CachedQueryResult, QueryCacheKey, QueryCacheKeyBuilder, QueryResultCache,
    QueryResultCacheConfig,
};
