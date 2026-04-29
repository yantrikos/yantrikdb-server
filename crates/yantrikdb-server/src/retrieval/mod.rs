//! RFC 015-B retrieval primitives.
//!
//! ## What lives here
//!
//! Retrieval-quality + retrieval-shortcut indexes that compose with
//! the recall handler. Two pieces ship in this PR (RFC 015-B-3):
//!
//! - [`time_decay`] — per-namespace sorted index over
//!   `current_decayed_importance × half_life` so recall can answer
//!   "what was I just doing?" without a full HNSW round trip.
//!
//! Future RFC 015-B work that lands here:
//!
//! - **015-B-2** — hybrid retrieval (BM25 sparse + dense fusion +
//!   cross-encoder rerank). Substantial; multi-session.
//!
//! ## Design discipline
//!
//! Retrieval primitives are **read paths only** at this layer.
//! Indexes that need to stay in sync with mutations subscribe to
//! [`crate::cache::InvalidationBus`] (RFC 015-A) — no direct
//! coupling to the commit-log apply path. This matches how
//! `crate::index::entity_adjacency` does it.

pub mod bm25;
pub mod hybrid;
pub mod rerank;
pub mod time_decay;

pub use bm25::{BM25Index, BM25Score, InMemoryBM25Index};
pub use hybrid::{rrf_merge, RrfConfig, ScoredDoc};
pub use rerank::{IdentityReranker, RerankInput, RerankOutput, Reranker};
pub use time_decay::{DecayConfig, DecayedRow, TimeDecayIndex, TimeDecayProvider};
