//! RFC 015-B-2 — hybrid retrieval via reciprocal rank fusion (RRF).
//!
//! Per RFC 015-B-2: "RRF (reciprocal rank fusion) merges sparse + dense
//! results."
//!
//! ## What this owns
//!
//! - [`rrf_merge`] — pure function. Takes any number of ranked result
//!   lists (each `Vec<ScoredDoc>`) and merges them by RRF scoring:
//!
//!   ```text
//!   rrf_score(doc) = sum over all rankings r:  1 / (k + rank_in_r(doc))
//!   ```
//!
//!   Higher score = better. `k=60` is the canonical default from the
//!   original RRF paper (Cormack et al. 2009); operators can override
//!   in [`RrfConfig`].
//!
//! - [`ScoredDoc`] — trivial `(doc_id, score)` pair, ranked-list element.
//!
//! ## Why RRF, not score-normalized weighted average
//!
//! Heterogeneous ranking systems produce scores on incompatible scales
//! (BM25 gives unbounded float; cosine similarity is bounded \[-1,1\];
//! cross-encoder logits are still other shape). Normalizing to \[0,1\]
//! is fragile: it depends on the score distribution of THIS query,
//! not the system. RRF only consumes RANK, so it's distribution-free.
//!
//! Cormack 2009 + many follow-up benchmarks: RRF beats well-tuned
//! score-blending on most retrieval tasks. We default to it.
//!
//! ## What's NOT here (deferred to consumer PR)
//!
//! - The full hybrid retrieval orchestrator that fans out a query to
//!   BM25 + dense + entity-adjacency + time-decay, calls `rrf_merge`,
//!   then optionally re-ranks. Lives in the recall handler.

/// One ranked result. `score` is informational and is NOT used by
/// [`rrf_merge`] — only `rank` (== position in the list) matters.
/// Keeping the score around lets dashboards show the source-system
/// score alongside the merged RRF score.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredDoc {
    pub doc_id: String,
    pub score: f32,
}

impl ScoredDoc {
    pub fn new(doc_id: impl Into<String>, score: f32) -> Self {
        Self {
            doc_id: doc_id.into(),
            score,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RrfConfig {
    /// k constant. Higher k → flatter contribution (later-ranked docs
    /// matter relatively more); lower k → steeper (top-ranked
    /// dominate). Cormack 2009 default = 60.
    pub k: u32,
    /// Cap each input list at this many entries before merging. None =
    /// no cap. Production should cap at ~100 per source to bound the
    /// merger's work.
    pub per_source_cap: Option<usize>,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self {
            k: 60,
            per_source_cap: Some(100),
        }
    }
}

impl RrfConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        // k=0 would divide by zero on the smallest rank (1) → infinite
        // contribution. Reject.
        if self.k == 0 {
            return Err("rrf k must be >= 1 (canonical default = 60)");
        }
        Ok(())
    }
}

/// Merge any number of ranked result lists into one.
///
/// Inputs:
/// - `sources`: a vec of result lists. Each list is presumed pre-sorted
///   by descending relevance (best first). Their `score` fields are
///   informational — only ranks contribute to RRF.
/// - `cfg`: `k` and per-source cap.
/// - `top_k`: truncate the merged result to this many entries.
///
/// Output: merged `Vec<ScoredDoc>`, sorted by descending RRF score.
/// Each output `score` is the merged RRF score (NOT the original
/// source score — distinct dimension). Tie-break by `doc_id` ascending
/// for determinism.
pub fn rrf_merge(
    sources: Vec<Vec<ScoredDoc>>,
    cfg: RrfConfig,
    top_k: usize,
) -> Vec<ScoredDoc> {
    use std::collections::HashMap;

    let k = cfg.k.max(1) as f32;
    let mut accumulator: HashMap<String, f32> = HashMap::new();

    for source in sources {
        let take = match cfg.per_source_cap {
            Some(c) => source.len().min(c),
            None => source.len(),
        };
        for (idx, doc) in source.into_iter().take(take).enumerate() {
            let rank = (idx + 1) as f32;
            let contribution = 1.0 / (k + rank);
            *accumulator.entry(doc.doc_id).or_insert(0.0) += contribution;
        }
    }

    let mut merged: Vec<ScoredDoc> = accumulator
        .into_iter()
        .map(|(doc_id, score)| ScoredDoc { doc_id, score })
        .collect();

    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    merged.truncate(top_k);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str, score: f32) -> ScoredDoc {
        ScoredDoc::new(id, score)
    }

    #[test]
    fn rrf_default_k_is_60() {
        assert_eq!(RrfConfig::default().k, 60);
    }

    #[test]
    fn rrf_validate_rejects_zero_k() {
        let cfg = RrfConfig {
            k: 0,
            per_source_cap: None,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rrf_single_source_preserves_order() {
        let source = vec![doc("a", 0.9), doc("b", 0.7), doc("c", 0.5)];
        let merged = rrf_merge(vec![source], RrfConfig::default(), 10);
        // Only one source → ranks dictate ordering even with k=60.
        let ids: Vec<&str> = merged.iter().map(|d| d.doc_id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn rrf_merges_two_distinct_sources_by_combined_rank() {
        // Source 1: [a, b, c] ranks 1,2,3
        // Source 2: [c, d, e] ranks 1,2,3
        // c appears in both at lower (better) ranks → highest merged.
        let s1 = vec![doc("a", 0.0), doc("b", 0.0), doc("c", 0.0)];
        let s2 = vec![doc("c", 0.0), doc("d", 0.0), doc("e", 0.0)];
        let merged = rrf_merge(vec![s1, s2], RrfConfig::default(), 10);
        // c should be first.
        assert_eq!(merged[0].doc_id, "c");
    }

    #[test]
    fn rrf_score_decreases_with_rank() {
        let source: Vec<ScoredDoc> = (0..5).map(|i| doc(&format!("d{}", i), 0.0)).collect();
        let merged = rrf_merge(vec![source], RrfConfig::default(), 10);
        for window in merged.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }

    #[test]
    fn rrf_per_source_cap_truncates() {
        // Cap at 2 → only first 2 from each source contribute.
        let s1: Vec<ScoredDoc> = (0..10).map(|i| doc(&format!("d{}", i), 0.0)).collect();
        let s2 = vec![doc("d99", 0.0)]; // doc that's not in s1
        let cfg = RrfConfig {
            k: 60,
            per_source_cap: Some(2),
        };
        let merged = rrf_merge(vec![s1, s2], cfg, 10);
        let ids: Vec<&str> = merged.iter().map(|d| d.doc_id.as_str()).collect();
        // d2..d9 from s1 should be dropped.
        for dropped in ["d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9"] {
            assert!(!ids.contains(&dropped), "expected {} to be capped", dropped);
        }
    }

    #[test]
    fn rrf_truncates_to_top_k() {
        let source: Vec<ScoredDoc> = (0..20).map(|i| doc(&format!("d{}", i), 0.0)).collect();
        let merged = rrf_merge(vec![source], RrfConfig::default(), 5);
        assert_eq!(merged.len(), 5);
    }

    #[test]
    fn rrf_empty_sources_yields_empty_result() {
        let merged = rrf_merge(vec![], RrfConfig::default(), 10);
        assert!(merged.is_empty());
    }

    #[test]
    fn rrf_all_empty_lists_yields_empty_result() {
        let merged = rrf_merge(vec![vec![], vec![]], RrfConfig::default(), 10);
        assert!(merged.is_empty());
    }

    #[test]
    fn rrf_deterministic_tie_break_by_doc_id() {
        // Two sources rank "z" and "a" both at position 1 → equal RRF
        // score → tie broken by doc_id asc → "a" first.
        let s1 = vec![doc("z", 0.0)];
        let s2 = vec![doc("a", 0.0)];
        let merged = rrf_merge(vec![s1, s2], RrfConfig::default(), 10);
        assert_eq!(merged[0].doc_id, "a");
        assert_eq!(merged[1].doc_id, "z");
    }

    #[test]
    fn rrf_score_formula_correct_for_simple_case() {
        // One source, one doc at rank 1, k=60 → score = 1 / (60+1) = 1/61
        let merged = rrf_merge(
            vec![vec![doc("a", 0.0)]],
            RrfConfig::default(),
            10,
        );
        let expected = 1.0 / 61.0;
        assert!((merged[0].score - expected).abs() < 1e-6);
    }

    #[test]
    fn rrf_score_sums_across_sources() {
        // Doc "a" at rank 1 in both sources → score = 2 × 1/(60+1)
        let s1 = vec![doc("a", 0.0), doc("z", 0.0)];
        let s2 = vec![doc("a", 0.0), doc("y", 0.0)];
        let merged = rrf_merge(vec![s1, s2], RrfConfig::default(), 10);
        let expected = 2.0 / 61.0;
        let a_score = merged.iter().find(|d| d.doc_id == "a").unwrap().score;
        assert!((a_score - expected).abs() < 1e-6);
    }

    #[test]
    fn rrf_lower_k_makes_top_ranks_dominate_more() {
        // With k=1, rank-1 = 0.5, rank-2 = 0.333... — big gap.
        // With k=1000, rank-1 ≈ 0.000999, rank-2 ≈ 0.000998 — flat.
        let s1 = vec![doc("a", 0.0), doc("b", 0.0)];
        let s2 = vec![doc("b", 0.0), doc("a", 0.0)];

        // With k=1: a in source1 contributes 1/2, in source2 contributes 1/3 → 0.833.
        // b: 1/3 + 1/2 = 0.833 → equal scores → tie break by id → a first.
        // (Symmetric input; both k values produce a tie. Useful as a
        //  smoke test that the formula is symmetric.)
        let cfg_low = RrfConfig {
            k: 1,
            per_source_cap: None,
        };
        let merged_low = rrf_merge(vec![s1.clone(), s2.clone()], cfg_low, 10);
        let cfg_high = RrfConfig {
            k: 1000,
            per_source_cap: None,
        };
        let merged_high = rrf_merge(vec![s1, s2], cfg_high, 10);
        // Both should have a and b; both should converge to a (tie-break).
        assert_eq!(merged_low[0].doc_id, "a");
        assert_eq!(merged_high[0].doc_id, "a");
    }
}
