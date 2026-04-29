//! RFC 015-B-2 — BM25 sparse retrieval substrate.
//!
//! ## What this owns
//!
//! - [`BM25Index`] trait — the contract callers (recall handler) see.
//!   `index(rid, text)`, `delete(rid)`, `search(query, top_k)`.
//! - [`InMemoryBM25Index`] — reference impl in pure Rust. Implements
//!   the standard BM25 formula with k1=1.2, b=0.75 (defaults from
//!   Robertson & Zaragoza 2009). Tokenization is lowercase + ASCII
//!   alphanum split — sufficient for tests + correctness validation,
//!   not a production tokenizer.
//! - [`BM25Score`] — `(rid, score)` result row.
//!
//! ## Why ship this and not just bind tantivy
//!
//! tantivy is the real production target — it has proper tokenizers,
//! disk persistence, and reasonable index-update semantics. Adding it
//! is a multi-step PR (Cargo dep + tokenizer config + per-tenant index
//! lifecycle + on-disk persistence + tantivy-specific delete handling).
//!
//! What this module ships is the trait surface and a correct-by-formula
//! reference impl. Tests pin BM25 ranking behavior (longer documents
//! penalize, term frequency saturates, IDF rewards rare terms). When
//! the tantivy backend lands, the tests carry over verbatim — they
//! validate the contract, not the impl.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

/// One ranked search hit.
#[derive(Debug, Clone, PartialEq)]
pub struct BM25Score {
    pub rid: String,
    pub score: f32,
}

/// The contract every BM25 backend must satisfy. Both the in-memory
/// impl below and the future tantivy-backed impl conform.
pub trait BM25Index: Send + Sync {
    /// Index a (rid, text) pair. If `rid` already exists, replaces.
    fn index(&self, rid: &str, text: &str);

    /// Remove a rid from the index. No-op if not present.
    fn delete(&self, rid: &str) -> bool;

    /// Search for top-k matches. Returns descending by score.
    fn search(&self, query: &str, top_k: usize) -> Vec<BM25Score>;

    /// Number of indexed rids. For dashboards + tests.
    fn doc_count(&self) -> usize;
}

/// Tokenize a string into lowercase ASCII-alphanum words. Replaceable
/// with tantivy tokenizer chains in the production impl. Public so
/// tests + the InMemoryBM25Index can share it without re-coding.
pub fn ascii_lower_tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

/// In-memory BM25 reference implementation. Not optimized — small
/// inverted index, recomputes IDF on each search. Use this for tests
/// + small datasets; production routes through tantivy.
#[derive(Default)]
pub struct InMemoryBM25Index {
    inner: Arc<RwLock<State>>,
    /// k1 controls term-frequency saturation. Default 1.2.
    k1: f32,
    /// b controls document-length normalization. Default 0.75.
    b: f32,
}

#[derive(Default)]
struct State {
    /// rid → tokenized terms (with their per-doc count).
    docs: HashMap<String, DocStats>,
    /// term → set of (rid, tf).
    inverted: HashMap<String, HashMap<String, u32>>,
    /// Total tokens across all docs / total docs. Used for avgdl in BM25.
    total_tokens: u64,
}

struct DocStats {
    /// Per-term frequency in this doc.
    tf: HashMap<String, u32>,
    /// Document length (total tokens).
    length: u32,
}

impl InMemoryBM25Index {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(State::default())),
            k1: 1.2,
            b: 0.75,
        }
    }

    pub fn with_params(mut self, k1: f32, b: f32) -> Self {
        self.k1 = k1;
        self.b = b;
        self
    }
}

impl BM25Index for InMemoryBM25Index {
    fn index(&self, rid: &str, text: &str) {
        let tokens = ascii_lower_tokens(text);
        let length = tokens.len() as u32;
        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.clone()).or_insert(0) += 1;
        }

        let mut g = self.inner.write();

        // If replacing, decrement prior counts first.
        if let Some(prev) = g.docs.remove(rid) {
            g.total_tokens = g.total_tokens.saturating_sub(prev.length as u64);
            for (term, count) in prev.tf.iter() {
                if let Some(postings) = g.inverted.get_mut(term) {
                    postings.remove(rid);
                    if postings.is_empty() {
                        g.inverted.remove(term);
                    }
                }
                let _ = count; // already removed by rid above
            }
        }

        // Insert new.
        for (term, count) in &tf {
            g.inverted
                .entry(term.clone())
                .or_default()
                .insert(rid.to_string(), *count);
        }
        g.total_tokens = g.total_tokens.saturating_add(length as u64);
        g.docs.insert(rid.to_string(), DocStats { tf, length });
    }

    fn delete(&self, rid: &str) -> bool {
        let mut g = self.inner.write();
        let Some(prev) = g.docs.remove(rid) else {
            return false;
        };
        g.total_tokens = g.total_tokens.saturating_sub(prev.length as u64);
        for term in prev.tf.keys() {
            if let Some(postings) = g.inverted.get_mut(term) {
                postings.remove(rid);
                if postings.is_empty() {
                    g.inverted.remove(term);
                }
            }
        }
        true
    }

    fn search(&self, query: &str, top_k: usize) -> Vec<BM25Score> {
        let q_tokens = ascii_lower_tokens(query);
        if q_tokens.is_empty() {
            return Vec::new();
        }
        let g = self.inner.read();
        let n_docs = g.docs.len() as f32;
        if n_docs == 0.0 {
            return Vec::new();
        }
        let avg_dl = (g.total_tokens as f32) / n_docs;

        // Per-rid running score.
        let mut scores: HashMap<String, f32> = HashMap::new();
        for term in &q_tokens {
            let Some(postings) = g.inverted.get(term) else {
                continue;
            };
            let df = postings.len() as f32;
            // Standard BM25 IDF (with 0.5 smoothing). Robertson 2009.
            let idf = (((n_docs - df + 0.5) / (df + 0.5)) + 1.0).ln();
            for (rid, &tf_u) in postings {
                let tf = tf_u as f32;
                let dl = g.docs.get(rid).map(|d| d.length as f32).unwrap_or(0.0);
                let denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avg_dl.max(1.0)));
                let term_score = idf * ((tf * (self.k1 + 1.0)) / denom.max(1e-9));
                *scores.entry(rid.clone()).or_insert(0.0) += term_score;
            }
        }

        let mut hits: Vec<BM25Score> = scores
            .into_iter()
            .map(|(rid, score)| BM25Score { rid, score })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.rid.cmp(&b.rid))
        });
        hits.truncate(top_k);
        hits
    }

    fn doc_count(&self) -> usize {
        self.inner.read().docs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_lower_tokens_basic() {
        let toks = ascii_lower_tokens("Hello World!  FOO_bar 123");
        // `_` is non-alphanum → splits.
        assert_eq!(toks, vec!["hello", "world", "foo", "bar", "123"]);
    }

    #[test]
    fn empty_query_yields_empty_results() {
        let idx = InMemoryBM25Index::new();
        idx.index("a", "the quick brown fox");
        assert!(idx.search("", 10).is_empty());
    }

    #[test]
    fn empty_index_yields_empty_results() {
        let idx = InMemoryBM25Index::new();
        assert!(idx.search("anything", 10).is_empty());
    }

    #[test]
    fn search_finds_documents_containing_term() {
        let idx = InMemoryBM25Index::new();
        idx.index("a", "rust programming language");
        idx.index("b", "python programming language");
        idx.index("c", "fishing tackle reviews");
        let hits = idx.search("rust", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rid, "a");
    }

    #[test]
    fn rare_term_outranks_common_term() {
        // IDF: "the" appears in many docs, "rust" in fewer → "rust"
        // contributes more to the score.
        let idx = InMemoryBM25Index::new();
        for i in 0..10 {
            idx.index(&format!("common-{}", i), "the");
        }
        idx.index("rare", "rust the");
        let hits = idx.search("rust", 10);
        // The "rare" doc must be on top; if anything ranks higher we
        // miscomputed IDF.
        assert_eq!(hits[0].rid, "rare");
    }

    #[test]
    fn term_frequency_increases_score_but_saturates() {
        let idx = InMemoryBM25Index::new();
        idx.index("once", "rust");
        idx.index("twice", "rust rust");
        idx.index("ten", "rust rust rust rust rust rust rust rust rust rust");
        let hits = idx.search("rust", 10);
        let by_rid: HashMap<&str, f32> = hits.iter().map(|h| (h.rid.as_str(), h.score)).collect();
        // More TF → higher score, monotonically.
        assert!(by_rid["once"] < by_rid["twice"]);
        assert!(by_rid["twice"] < by_rid["ten"]);
        // Saturation: the gap from once→twice should be bigger than
        // twice→ten (BM25 k1 saturation).
        let gap_low = by_rid["twice"] - by_rid["once"];
        let gap_high = by_rid["ten"] - by_rid["twice"];
        assert!(
            gap_low > gap_high,
            "expected saturation: low gap {}, high gap {}",
            gap_low,
            gap_high
        );
    }

    #[test]
    fn longer_documents_penalized_via_length_norm() {
        // Two docs with same TF for "rust" but different lengths; b>0
        // means the longer one scores less.
        let idx = InMemoryBM25Index::new();
        idx.index("short", "rust");
        idx.index("long", "rust the the the the the the the the the");
        let hits = idx.search("rust", 10);
        let by_rid: HashMap<&str, f32> = hits.iter().map(|h| (h.rid.as_str(), h.score)).collect();
        assert!(by_rid["short"] > by_rid["long"]);
    }

    #[test]
    fn delete_removes_doc() {
        let idx = InMemoryBM25Index::new();
        idx.index("a", "rust programming");
        idx.index("b", "rust language");
        assert_eq!(idx.search("rust", 10).len(), 2);
        assert!(idx.delete("a"));
        assert_eq!(idx.search("rust", 10).len(), 1);
        // Second delete is no-op.
        assert!(!idx.delete("a"));
    }

    #[test]
    fn re_index_replaces_old_text() {
        let idx = InMemoryBM25Index::new();
        idx.index("a", "rust");
        // Now change the text — query for "rust" must miss.
        idx.index("a", "python");
        assert!(idx.search("rust", 10).is_empty());
        assert_eq!(idx.search("python", 10).len(), 1);
    }

    #[test]
    fn doc_count_tracks_inserts_and_deletes() {
        let idx = InMemoryBM25Index::new();
        assert_eq!(idx.doc_count(), 0);
        idx.index("a", "x");
        idx.index("b", "y");
        assert_eq!(idx.doc_count(), 2);
        idx.delete("a");
        assert_eq!(idx.doc_count(), 1);
    }

    #[test]
    fn truncates_to_top_k() {
        let idx = InMemoryBM25Index::new();
        for i in 0..10 {
            idx.index(&format!("d{}", i), "rust");
        }
        let hits = idx.search("rust", 3);
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn multi_term_query_combines_scores() {
        let idx = InMemoryBM25Index::new();
        idx.index("a", "rust");
        idx.index("b", "programming");
        idx.index("c", "rust programming");
        // c contains both terms → should outrank either single-term doc.
        let hits = idx.search("rust programming", 10);
        assert_eq!(hits[0].rid, "c");
    }

    #[test]
    fn case_insensitive_matching() {
        let idx = InMemoryBM25Index::new();
        idx.index("a", "Rust Programming Language");
        let hits = idx.search("RUST", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rid, "a");
    }

    #[test]
    fn dyn_dispatch() {
        let idx: Arc<dyn BM25Index> = Arc::new(InMemoryBM25Index::new());
        idx.index("a", "rust");
        assert_eq!(idx.search("rust", 10).len(), 1);
    }

    #[test]
    fn deterministic_tie_break_by_rid() {
        let idx = InMemoryBM25Index::new();
        // Identical text → identical scores → tie break by rid asc.
        idx.index("zzz", "rust");
        idx.index("aaa", "rust");
        idx.index("mmm", "rust");
        let hits = idx.search("rust", 10);
        let ids: Vec<&str> = hits.iter().map(|h| h.rid.as_str()).collect();
        assert_eq!(ids, vec!["aaa", "mmm", "zzz"]);
    }
}
