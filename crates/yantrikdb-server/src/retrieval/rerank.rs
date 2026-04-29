//! RFC 015-B-2 — cross-encoder rerank substrate.
//!
//! ## What this owns
//!
//! - [`Reranker`] trait — `rerank(query, candidates) → reranked`. The
//!   contract every rerank backend implements.
//! - [`IdentityReranker`] — pass-through; returns the input untouched
//!   (preserving order). Useful as a "rerank disabled" sentinel and
//!   as the default in tests that don't want a real model.
//! - [`RerankInput`] / [`RerankOutput`] — the wire types.
//!
//! ## What's NOT here (deferred)
//!
//! The actual ONNX-backed cross-encoder reranker (BAAI/bge-reranker-base
//! per RFC 015-B-2). That requires:
//! - ONNX runtime version pinning + `ort` crate dep.
//! - Model file shipping + per-platform binary handling.
//! - Tokenizer (BPE or WordPiece, depending on model).
//! - Per-tenant model selection (RFC 013-B shadow index extends this).
//!
//! This substrate ships the seam so the recall handler can `let
//! reranker: Arc<dyn Reranker> = ...` today and swap in the ONNX impl
//! when it lands without changing the call sites.
//!
//! ## Why "rerank" not "score"
//!
//! Cross-encoders are typically used to RE-rank a top-N from cheaper
//! retrieval (RFC 015-B-2: top 50 → reranker → top 10). The trait
//! reflects that: it consumes a ranked candidate list and produces a
//! ranked output. Single-document scoring is a degenerate case (a
//! one-element list) the trait still handles.

use std::sync::Arc;

/// One candidate document fed to the reranker. `text` is the document
/// content (or a snippet) that the cross-encoder evaluates against the
/// query. `prior_score` is the score from the previous-stage
/// retrieval — informational; the reranker computes its own score.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankInput {
    pub doc_id: String,
    pub text: String,
    pub prior_score: f32,
}

/// One output row. `score` is the reranker's score (cross-encoder
/// logit, typically; impls may sigmoid before returning).
#[derive(Debug, Clone, PartialEq)]
pub struct RerankOutput {
    pub doc_id: String,
    pub score: f32,
}

/// Pluggable reranker. Async because the ONNX impl hits a model
/// inference call that can be CPU-heavy.
#[async_trait::async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank `candidates` against `query`. Returns top `top_k` by
    /// the reranker's own scoring, descending. The output may be
    /// shorter than `top_k` if `candidates` is shorter.
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<RerankInput>,
        top_k: usize,
    ) -> Vec<RerankOutput>;
}

/// Pass-through reranker. Returns the input list (truncated to top_k)
/// without re-scoring — the prior order is preserved, prior_score is
/// promoted to score. Use as the default when reranking is disabled
/// or when a test wants a deterministic outcome.
pub struct IdentityReranker;

#[async_trait::async_trait]
impl Reranker for IdentityReranker {
    async fn rerank(
        &self,
        _query: &str,
        candidates: Vec<RerankInput>,
        top_k: usize,
    ) -> Vec<RerankOutput> {
        candidates
            .into_iter()
            .take(top_k)
            .map(|c| RerankOutput {
                doc_id: c.doc_id,
                score: c.prior_score,
            })
            .collect()
    }
}

/// Convenience type alias for `Arc<dyn Reranker>`. Saves callers from
/// repeating the dyn dispatch boilerplate.
pub type DynReranker = Arc<dyn Reranker>;

#[cfg(test)]
mod tests {
    use super::*;

    fn input(id: &str, prior: f32) -> RerankInput {
        RerankInput {
            doc_id: id.into(),
            text: format!("text for {}", id),
            prior_score: prior,
        }
    }

    #[tokio::test]
    async fn identity_preserves_order_and_truncates() {
        let r = IdentityReranker;
        let candidates = vec![input("a", 0.9), input("b", 0.7), input("c", 0.5)];
        let out = r.rerank("query", candidates, 2).await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].doc_id, "a");
        assert_eq!(out[1].doc_id, "b");
    }

    #[tokio::test]
    async fn identity_promotes_prior_score_to_score() {
        let r = IdentityReranker;
        let candidates = vec![input("a", 0.42)];
        let out = r.rerank("q", candidates, 10).await;
        assert_eq!(out[0].score, 0.42);
    }

    #[tokio::test]
    async fn identity_handles_empty_candidates() {
        let r = IdentityReranker;
        let out = r.rerank("q", Vec::new(), 10).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn identity_top_k_zero_returns_empty() {
        let r = IdentityReranker;
        let out = r.rerank("q", vec![input("a", 0.9)], 0).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn identity_top_k_larger_than_candidates_returns_all() {
        let r = IdentityReranker;
        let out = r
            .rerank("q", vec![input("a", 0.9), input("b", 0.7)], 100)
            .await;
        assert_eq!(out.len(), 2);
    }

    #[tokio::test]
    async fn dyn_dispatch_via_trait_object() {
        let r: DynReranker = Arc::new(IdentityReranker);
        let out = r.rerank("q", vec![input("a", 0.5)], 10).await;
        assert_eq!(out.len(), 1);
    }

    /// Sample reranker that inverts prior_score (smallest prior → highest
    /// rerank score). Demonstrates the trait correctly hands off
    /// reordering control to the impl.
    struct InvertingReranker;

    #[async_trait::async_trait]
    impl Reranker for InvertingReranker {
        async fn rerank(
            &self,
            _query: &str,
            candidates: Vec<RerankInput>,
            top_k: usize,
        ) -> Vec<RerankOutput> {
            let mut out: Vec<RerankOutput> = candidates
                .into_iter()
                .map(|c| RerankOutput {
                    doc_id: c.doc_id,
                    score: 1.0 - c.prior_score,
                })
                .collect();
            out.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.doc_id.cmp(&b.doc_id))
            });
            out.truncate(top_k);
            out
        }
    }

    #[tokio::test]
    async fn custom_reranker_can_reorder() {
        let r = InvertingReranker;
        let candidates = vec![input("a", 0.9), input("b", 0.5), input("c", 0.1)];
        let out = r.rerank("q", candidates, 10).await;
        // Inverted: c (1-0.1=0.9), b (0.5), a (0.1).
        assert_eq!(out[0].doc_id, "c");
        assert_eq!(out[1].doc_id, "b");
        assert_eq!(out[2].doc_id, "a");
    }
}
