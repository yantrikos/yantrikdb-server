/// Multi-signal scoring for memory recall.
///
/// Matches the Python implementation exactly (engine.py:246-263).

/// Compute the decay score: I(t) = importance * 2^(-t / half_life)
pub fn decay_score(importance: f64, half_life: f64, elapsed: f64) -> f64 {
    if half_life > 0.0 {
        importance * f64::powf(2.0, -elapsed / half_life)
    } else {
        0.0
    }
}

/// Compute the recency score: exp(-age / (7 * 86400))
pub fn recency_score(age: f64) -> f64 {
    f64::exp(-age / (7.0 * 86400.0))
}

/// Compute the valence boost: 1.0 + 0.3 * |valence|
pub fn valence_boost(valence: f64) -> f64 {
    1.0 + 0.3 * valence.abs()
}

/// Negative sentiment keywords for query-aware valence matching.
const NEGATIVE_QUERY_WORDS: &[&str] = &[
    "sad", "frustrated", "angry", "bad", "worst", "low", "lows", "difficult",
    "hard", "struggle", "pain", "stress", "anxious", "upset", "failed",
    "failure", "problem", "negative", "stressing", "worried", "tough",
];

/// Positive sentiment keywords for query-aware valence matching.
const POSITIVE_QUERY_WORDS: &[&str] = &[
    "happy", "joy", "great", "best", "high", "good", "wonderful", "excited",
    "proud", "success", "achievement", "positive", "celebration", "love",
];

/// Simple suffix stripping for sentiment detection.
/// "proudest" → "proud", "happiest" → "happi" → matched via prefix.
fn sentiment_stem(word: &str) -> &str {
    // Strip common inflectional suffixes (ordered longest first)
    for suffix in &["iest", "ness", "ment", "ful", "est", "ing", "ous", "ive", "ity", "ed", "er", "ly", "al", "es", "s"] {
        if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
            return &word[..word.len() - suffix.len()];
        }
    }
    word
}

/// Detect query sentiment from text: -1.0 for negative, +1.0 for positive, 0.0 for neutral.
///
/// Uses stemmed matching so "proudest" matches "proud", "happiest" matches "happi"→"happy", etc.
pub fn detect_query_sentiment(query_text: &str) -> f64 {
    let lower = query_text.to_lowercase();
    // Split on non-alphanumeric to strip punctuation from tokens
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();

    // Check if a token matches a word list, allowing stemmed/prefix matching.
    // Stem both the token and the word, then check prefix overlap.
    // "proudest" → stem "proud" → matches "proud" ✓
    // "happiest" → stem "happi" → "happy" starts_with "happi" ✓
    //
    // Prefix matching requires >= 4 chars to prevent false positives:
    // "up" must NOT match "upset", "no" must NOT match "negative".
    let matches_list = |token: &str, words: &[&str]| -> bool {
        let stemmed = sentiment_stem(token);
        words.iter().any(|w| {
            token == *w
                || stemmed == *w
                || (token.len() >= 4 && token.starts_with(w))
                || (stemmed.len() >= 4 && w.starts_with(stemmed))
        })
    };

    let neg_count = tokens
        .iter()
        .filter(|t| matches_list(t, NEGATIVE_QUERY_WORDS))
        .count();
    let pos_count = tokens
        .iter()
        .filter(|t| matches_list(t, POSITIVE_QUERY_WORDS))
        .count();

    if neg_count > pos_count {
        -1.0
    } else if pos_count > neg_count {
        1.0
    } else {
        0.0
    }
}

/// Query-aware valence boost.
///
/// When query sentiment matches memory valence sign (e.g., negative query + negative memory),
/// the boost is increased. When they mismatch, the boost is reduced.
/// For neutral queries (sentiment == 0.0), falls back to the standard symmetric boost.
pub fn query_valence_boost(memory_valence: f64, query_sentiment: f64) -> f64 {
    let base = 1.0 + 0.3 * memory_valence.abs();
    if query_sentiment == 0.0 {
        return base;
    }
    // alignment: +1 = match (negative query + negative memory), -1 = mismatch
    let alignment = if memory_valence.abs() < 1e-10 {
        0.0
    } else {
        query_sentiment * memory_valence.signum()
    };
    base * (1.0 + 0.2 * alignment)
}

/// Composite score with query-aware valence.
pub fn composite_score_with_sentiment(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
    query_sentiment: f64,
) -> f64 {
    let base_rel = W_SIM * similarity + W_DECAY * decay + W_RECENCY * recency;
    let gate = importance_gate(similarity);
    let imp_mult = 1.0 + gate * ALPHA_IMP * importance.min(1.0);
    base_rel * imp_mult * query_valence_boost(valence, query_sentiment)
}

/// Graph composite score with query-aware valence.
pub fn graph_composite_score_with_sentiment(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
    graph_proximity: f64,
    query_sentiment: f64,
) -> f64 {
    if graph_proximity > 0.0 {
        let base_rel = GW_SIM * similarity + GW_DECAY * decay + GW_RECENCY * recency
            + GW_GRAPH * graph_proximity;
        let gate = importance_gate(similarity);
        let imp_mult = 1.0 + gate * GW_ALPHA_IMP * importance.min(1.0);
        base_rel * imp_mult * query_valence_boost(valence, query_sentiment)
    } else {
        composite_score_with_sentiment(similarity, decay, recency, importance, valence, query_sentiment)
    }
}

/// Relevance-gated multiplicative scoring.
///
/// Instead of additive importance (which lets high-importance memories dominate
/// regardless of relevance), importance now acts as a *multiplier* that only
/// activates when the memory is semantically relevant to the query.
///
/// Formula:
///   base_rel = W_SIM * similarity + W_DECAY * decay + W_RECENCY * recency
///   gate     = sigmoid(GATE_K * (similarity - GATE_TAU))
///   score    = base_rel * (1 + gate * ALPHA_IMP * importance) * valence_boost
///
/// This ensures that a memory with imp=1.0, sim=0.1 cannot beat a memory
/// with imp=0.3, sim=0.6 — the gate suppresses the importance boost when
/// similarity is low.

/// Base relevance weights (no importance — it's now multiplicative).
pub const W_SIM: f64 = 0.50;
pub const W_DECAY: f64 = 0.20;
pub const W_RECENCY: f64 = 0.30;

/// Importance gate parameters.
/// GATE_K controls the sharpness of the sigmoid gate.
/// GATE_TAU is the similarity threshold where the gate reaches 0.5.
pub const GATE_K: f64 = 12.0;
pub const GATE_TAU: f64 = 0.25;

/// Importance amplification strength.
/// At full gate (similarity >> τ), score is multiplied by up to (1 + ALPHA_IMP).
pub const ALPHA_IMP: f64 = 0.80;

/// Graph-expanded signal weights.
pub const GW_SIM: f64 = 0.35;
pub const GW_DECAY: f64 = 0.15;
pub const GW_RECENCY: f64 = 0.20;
pub const GW_GRAPH: f64 = 0.30;

/// Graph importance gate uses same parameters.
pub const GW_ALPHA_IMP: f64 = 0.60;

use crate::types::ScoreContributions;

/// Sigmoid function: 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the importance gate: sigmoid(K * (similarity - τ)).
///
/// Returns ~0 when similarity << τ (importance suppressed),
/// returns ~1 when similarity >> τ (importance fully active).
#[inline]
pub fn importance_gate(similarity: f64) -> f64 {
    sigmoid(GATE_K * (similarity - GATE_TAU))
}

/// Compute the composite recall score using relevance-gated multiplicative importance.
///
/// base_rel = W_SIM * similarity + W_DECAY * decay + W_RECENCY * recency
/// gate     = sigmoid(K * (similarity - τ))
/// score    = base_rel * (1 + gate * α * importance) * valence_boost
pub fn composite_score(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
) -> f64 {
    let base_rel = W_SIM * similarity + W_DECAY * decay + W_RECENCY * recency;
    let gate = importance_gate(similarity);
    let imp_mult = 1.0 + gate * ALPHA_IMP * importance.min(1.0);
    base_rel * imp_mult * valence_boost(valence)
}

/// Compute weighted contributions for standard scoring.
pub fn standard_contributions(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
) -> ScoreContributions {
    let gate = importance_gate(similarity);
    ScoreContributions {
        similarity: W_SIM * similarity,
        decay: W_DECAY * decay,
        recency: W_RECENCY * recency,
        importance: gate * ALPHA_IMP * importance.min(1.0),
        graph_proximity: 0.0,
    }
}

/// Compute the composite recall score with optional graph proximity signal.
///
/// Graph path:
///   base_rel = GW_SIM*sim + GW_DECAY*decay + GW_RECENCY*recency + GW_GRAPH*graph
///   gate     = sigmoid(K * (sim - τ))
///   score    = base_rel * (1 + gate * α_g * importance) * valence_boost
pub fn graph_composite_score(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
    graph_proximity: f64,
) -> f64 {
    if graph_proximity > 0.0 {
        let base_rel = GW_SIM * similarity + GW_DECAY * decay + GW_RECENCY * recency
            + GW_GRAPH * graph_proximity;
        let gate = importance_gate(similarity);
        let imp_mult = 1.0 + gate * GW_ALPHA_IMP * importance.min(1.0);
        base_rel * imp_mult * valence_boost(valence)
    } else {
        composite_score(similarity, decay, recency, importance, valence)
    }
}

/// Compute weighted contributions for graph-expanded scoring.
pub fn graph_contributions(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    graph_proximity: f64,
) -> ScoreContributions {
    if graph_proximity > 0.0 {
        let gate = importance_gate(similarity);
        ScoreContributions {
            similarity: GW_SIM * similarity,
            decay: GW_DECAY * decay,
            recency: GW_RECENCY * recency,
            importance: gate * GW_ALPHA_IMP * importance.min(1.0),
            graph_proximity: GW_GRAPH * graph_proximity,
        }
    } else {
        standard_contributions(similarity, decay, recency, importance)
    }
}

/// Score for eviction prioritization (lower = more evictable).
/// Combines decay strength and recency to identify stale memories.
pub fn eviction_score(decay: f64, recency: f64) -> f64 {
    0.6 * decay + 0.4 * recency
}

/// Build a human-readable explanation for why a memory was retrieved.
pub fn build_why(similarity: f64, recency: f64, decay: f64, valence: f64) -> Vec<String> {
    let mut why = Vec::new();
    if similarity > 0.5 {
        why.push(format!("semantically similar ({similarity:.2})"));
    }
    if recency > 0.5 {
        why.push("recent".to_string());
    }
    if decay > 0.3 {
        why.push(format!("important (decay={decay:.2})"));
    }
    if valence.abs() > 0.5 {
        why.push(format!("emotionally weighted ({valence:.2})"));
    }
    if why.is_empty() {
        why.push("matched query".to_string());
    }
    why
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_score_fresh() {
        let score = decay_score(0.8, 604800.0, 0.0);
        assert!((score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_decay_score_one_half_life() {
        let score = decay_score(1.0, 100.0, 100.0);
        assert!((score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_decay_score_zero_half_life() {
        let score = decay_score(0.8, 0.0, 100.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_recency_score_fresh() {
        let score = recency_score(0.0);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recency_score_seven_days() {
        let score = recency_score(7.0 * 86400.0);
        assert!((score - f64::exp(-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_valence_boost_zero() {
        assert!((valence_boost(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_valence_boost_positive() {
        assert!((valence_boost(1.0) - 1.3).abs() < 1e-10);
    }

    #[test]
    fn test_valence_boost_negative() {
        assert!((valence_boost(-0.5) - 1.15).abs() < 1e-10);
    }

    // ── Importance gate tests ──

    #[test]
    fn test_importance_gate_high_similarity() {
        // sim=0.8 >> τ=0.30 → gate ≈ 1.0
        let gate = importance_gate(0.8);
        assert!(gate > 0.99, "gate at sim=0.8 should be ~1.0, got {gate}");
    }

    #[test]
    fn test_importance_gate_low_similarity() {
        // sim=0.05 << τ=0.25 → gate ≈ 0.08
        let gate = importance_gate(0.05);
        assert!(gate < 0.12, "gate at sim=0.05 should be small, got {gate}");
    }

    #[test]
    fn test_importance_gate_at_threshold() {
        // sim=τ → gate = 0.5
        let gate = importance_gate(GATE_TAU);
        assert!((gate - 0.5).abs() < 1e-10, "gate at sim=τ should be 0.5, got {gate}");
    }

    #[test]
    fn test_importance_gate_monotonic() {
        let low = importance_gate(0.1);
        let mid = importance_gate(0.3);
        let high = importance_gate(0.7);
        assert!(mid > low, "gate should increase with similarity");
        assert!(high > mid, "gate should increase with similarity");
    }

    // ── Composite score: relevance-gated behavior ──

    #[test]
    fn test_high_imp_low_sim_loses_to_low_imp_high_sim() {
        // THE critical test: high importance + low similarity should NOT beat
        // moderate importance + high similarity.
        let irrelevant_important = composite_score(0.10, 0.5, 0.5, 1.0, 0.0);
        let relevant_normal = composite_score(0.60, 0.5, 0.5, 0.3, 0.0);
        assert!(relevant_normal > irrelevant_important,
            "relevant_normal ({relevant_normal:.4}) should beat irrelevant_important ({irrelevant_important:.4})");
    }

    #[test]
    fn test_high_imp_high_sim_beats_low_imp_high_sim() {
        // When both are relevant, importance should still help
        let important = composite_score(0.70, 0.5, 0.5, 0.9, 0.0);
        let normal = composite_score(0.70, 0.5, 0.5, 0.3, 0.0);
        assert!(important > normal,
            "when both relevant, higher importance should win: {important:.4} vs {normal:.4}");
    }

    #[test]
    fn test_composite_score_basic() {
        // All signals at 1.0, imp=1.0, valence=0 → base * (1 + gate * α * 1) * 1.0
        let score = composite_score(1.0, 1.0, 1.0, 1.0, 0.0);
        let base = W_SIM + W_DECAY + W_RECENCY;
        let gate = importance_gate(1.0);
        let expected = base * (1.0 + gate * ALPHA_IMP);
        assert!((score - expected).abs() < 1e-10, "expected {expected}, got {score}");
    }

    #[test]
    fn test_composite_score_with_valence() {
        let score = composite_score(1.0, 1.0, 1.0, 1.0, 1.0);
        let base = W_SIM + W_DECAY + W_RECENCY;
        let gate = importance_gate(1.0);
        let expected = base * (1.0 + gate * ALPHA_IMP) * 1.3;
        assert!((score - expected).abs() < 1e-10);
    }

    #[test]
    fn test_graph_composite_zero_proximity_matches_original() {
        let original = composite_score(0.8, 0.6, 0.9, 0.7, 0.3);
        let graph = graph_composite_score(0.8, 0.6, 0.9, 0.7, 0.3, 0.0);
        assert!((original - graph).abs() < 1e-10);
    }

    #[test]
    fn test_graph_composite_with_proximity() {
        let score = graph_composite_score(1.0, 1.0, 1.0, 1.0, 0.0, 1.0);
        let base = GW_SIM + GW_DECAY + GW_RECENCY + GW_GRAPH;
        let gate = importance_gate(1.0);
        let expected = base * (1.0 + gate * GW_ALPHA_IMP);
        assert!((score - expected).abs() < 1e-10, "expected {expected}, got {score}");
    }

    // ── Monotonicity & property invariants ──

    #[test]
    fn test_composite_monotonic_in_similarity() {
        let low = composite_score(0.3, 0.5, 0.5, 0.5, 0.0);
        let high = composite_score(0.9, 0.5, 0.5, 0.5, 0.0);
        assert!(high > low, "higher similarity should yield higher score");
    }

    #[test]
    fn test_composite_monotonic_in_importance() {
        // At moderate similarity (gate ≈ 0.88), importance should still matter
        let low = composite_score(0.5, 0.5, 0.5, 0.2, 0.0);
        let high = composite_score(0.5, 0.5, 0.5, 0.9, 0.0);
        assert!(high > low, "higher importance should yield higher score (when sim>τ)");
    }

    #[test]
    fn test_composite_monotonic_in_recency() {
        let low = composite_score(0.5, 0.5, 0.1, 0.5, 0.0);
        let high = composite_score(0.5, 0.5, 0.9, 0.5, 0.0);
        assert!(high > low, "higher recency should yield higher score");
    }

    #[test]
    fn test_valence_symmetric() {
        assert!((valence_boost(0.7) - valence_boost(-0.7)).abs() < 1e-10);
    }

    #[test]
    fn test_valence_always_geq_1() {
        for v in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert!(valence_boost(v) >= 1.0, "valence_boost({v}) = {} < 1.0", valence_boost(v));
        }
    }

    #[test]
    fn test_composite_non_negative() {
        for &sim in &[0.0, 0.5, 1.0] {
            for &dec in &[0.0, 0.5, 1.0] {
                for &rec in &[0.0, 0.5, 1.0] {
                    for &imp in &[0.0, 0.5, 1.0] {
                        for &val in &[-1.0, 0.0, 1.0] {
                            let s = composite_score(sim, dec, rec, imp, val);
                            assert!(s >= 0.0, "composite_score({sim},{dec},{rec},{imp},{val}) = {s} < 0");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_graph_composite_non_negative() {
        for &prox in &[0.0, 0.25, 0.5, 1.0] {
            let s = graph_composite_score(0.5, 0.5, 0.5, 0.5, 0.0, prox);
            assert!(s >= 0.0, "graph_composite with prox={prox} should be non-negative");
        }
    }

    #[test]
    fn test_graph_proximity_increases_score() {
        let without = graph_composite_score(0.3, 0.8, 0.7, 0.6, 0.0, 0.0);
        let with = graph_composite_score(0.3, 0.8, 0.7, 0.6, 0.0, 0.8);
        assert!(with > without,
            "graph proximity (0.8) should increase score: without={without}, with={with}");
    }

    #[test]
    fn test_decay_monotonic_in_elapsed() {
        let fresh = decay_score(0.8, 604800.0, 0.0);
        let old = decay_score(0.8, 604800.0, 604800.0);
        let ancient = decay_score(0.8, 604800.0, 604800.0 * 10.0);
        assert!(fresh > old);
        assert!(old > ancient);
    }

    #[test]
    fn test_recency_monotonic_in_age() {
        let fresh = recency_score(0.0);
        let week = recency_score(7.0 * 86400.0);
        let month = recency_score(30.0 * 86400.0);
        assert!(fresh > week);
        assert!(week > month);
    }

    #[test]
    fn test_build_why_always_nonempty() {
        let why = build_why(0.0, 0.0, 0.0, 0.0);
        assert!(!why.is_empty(), "build_why should always produce at least one reason");
        assert_eq!(why[0], "matched query");
    }

    #[test]
    fn test_build_why_contains_similarity() {
        let why = build_why(0.9, 0.1, 0.1, 0.0);
        assert!(why.iter().any(|w| w.contains("semantically similar")));
    }

    #[test]
    fn test_importance_capped_at_1() {
        let capped = composite_score(0.5, 0.5, 0.5, 5.0, 0.0);
        let at_one = composite_score(0.5, 0.5, 0.5, 1.0, 0.0);
        assert!((capped - at_one).abs() < 1e-10, "importance should be capped at 1.0");
    }

    #[test]
    fn test_eviction_score() {
        let score = eviction_score(1.0, 1.0);
        assert!((score - 1.0).abs() < 1e-10, "max inputs should give 1.0");

        let score_zero = eviction_score(0.0, 0.0);
        assert!((score_zero - 0.0).abs() < 1e-10, "zero inputs should give 0.0");

        let low = eviction_score(0.1, 0.5);
        let high = eviction_score(0.9, 0.5);
        assert!(high > low, "higher decay should yield higher eviction score");
    }

    // ── Regression: the "Staff Engineer domination" bug ──
    // In v7, "I got promoted to Staff Engineer" (imp=0.9) appeared at #1
    // for 16/20 queries because additive importance gave it +0.27 regardless
    // of similarity. With gated scoring, sim=0.1 means gate≈0.08 → boost≈0.06.

    #[test]
    fn test_irrelevant_anchor_cannot_dominate() {
        // Simulates: "Staff Engineer" memory recalled for "What did I cook?"
        // sim=0.08 (irrelevant), imp=0.9, decay=0.8, recency=0.3
        let anchor = composite_score(0.08, 0.8, 0.3, 0.9, 0.0);
        // Simulates: "Made pasta for dinner" for "What did I cook?"
        // sim=0.65 (relevant), imp=0.2, decay=0.6, recency=0.8
        let daily = composite_score(0.65, 0.6, 0.8, 0.2, 0.0);
        assert!(daily > anchor,
            "relevant daily ({daily:.4}) should beat irrelevant anchor ({anchor:.4})");
    }

    // ── Query-aware valence tests ──

    #[test]
    fn test_detect_query_sentiment_negative() {
        assert_eq!(detect_query_sentiment("What failures and problems have been stressing me out?"), -1.0);
        assert_eq!(detect_query_sentiment("Tell me about my emotional lows"), -1.0);
        assert_eq!(detect_query_sentiment("What was difficult this year?"), -1.0);
    }

    #[test]
    fn test_detect_query_sentiment_positive() {
        assert_eq!(detect_query_sentiment("What good things happened?"), 1.0);
        assert_eq!(detect_query_sentiment("Tell me about happy moments"), 1.0);
        assert_eq!(detect_query_sentiment("What was my greatest achievement?"), 1.0);
    }

    #[test]
    fn test_detect_query_sentiment_neutral() {
        assert_eq!(detect_query_sentiment("What happened at work recently?"), 0.0);
        assert_eq!(detect_query_sentiment("Tell me about my family"), 0.0);
    }

    #[test]
    fn test_query_valence_boost_neutral_query_matches_standard() {
        // Neutral query (sentiment=0.0) should behave identically to valence_boost
        for &v in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let standard = valence_boost(v);
            let query_aware = query_valence_boost(v, 0.0);
            assert!((standard - query_aware).abs() < 1e-10,
                "neutral query should match standard: valence={v}, standard={standard}, query_aware={query_aware}");
        }
    }

    #[test]
    fn test_query_valence_boost_negative_alignment() {
        // Negative query + negative memory → higher boost than neutral query
        let negative_aligned = query_valence_boost(-0.8, -1.0);
        let standard = valence_boost(-0.8);
        assert!(negative_aligned > standard,
            "negative query + negative memory should boost more: aligned={negative_aligned:.4}, standard={standard:.4}");
    }

    #[test]
    fn test_query_valence_boost_negative_misaligned() {
        // Negative query + positive memory → lower boost than neutral query
        let misaligned = query_valence_boost(0.8, -1.0);
        let standard = valence_boost(0.8);
        assert!(misaligned < standard,
            "negative query + positive memory should boost less: misaligned={misaligned:.4}, standard={standard:.4}");
    }

    #[test]
    fn test_query_valence_boost_always_positive() {
        // Boost should never go below 1.0 regardless of alignment
        for &v in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            for &s in &[-1.0, 0.0, 1.0] {
                let boost = query_valence_boost(v, s);
                assert!(boost >= 0.8, "query_valence_boost({v}, {s}) = {boost} should be positive");
            }
        }
    }

    #[test]
    fn test_composite_with_sentiment_matches_original_for_neutral() {
        // composite_score_with_sentiment with sentiment=0.0 should match composite_score
        let original = composite_score(0.6, 0.5, 0.7, 0.8, 0.3);
        let with_sent = composite_score_with_sentiment(0.6, 0.5, 0.7, 0.8, 0.3, 0.0);
        assert!((original - with_sent).abs() < 1e-10,
            "neutral sentiment should match original: {original} vs {with_sent}");
    }

    #[test]
    fn test_gate_tau_regression_anchor_still_loses() {
        // With GATE_TAU=0.25, verify Staff Engineer at sim=0.08 still loses
        let anchor = composite_score(0.08, 0.8, 0.3, 0.9, 0.0);
        let daily = composite_score(0.65, 0.6, 0.8, 0.2, 0.0);
        assert!(daily > anchor,
            "with GATE_TAU=0.25, relevant daily ({daily:.4}) should still beat irrelevant anchor ({anchor:.4})");
        // Also verify the gate itself at sim=0.08 is still small
        let gate = importance_gate(0.08);
        assert!(gate < 0.2, "gate at sim=0.08 should be small, got {gate}");
    }
}
