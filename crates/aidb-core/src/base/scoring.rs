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

/// Compute the composite recall score using multi-signal fusion.
///
/// score = (0.40 * similarity + 0.25 * decay + 0.20 * recency + 0.15 * importance) * valence_boost
pub fn composite_score(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
) -> f64 {
    let raw = 0.40 * similarity + 0.25 * decay + 0.20 * recency + 0.15 * importance.min(1.0);
    raw * valence_boost(valence)
}

/// Compute the composite recall score with optional graph proximity signal.
///
/// When graph_proximity > 0.0 (graph-expanded result):
///   score = (0.30*sim + 0.20*decay + 0.15*recency + 0.15*importance + 0.20*graph) * valence_boost
/// When graph_proximity == 0.0 (pure vector result):
///   Uses the original formula (unchanged).
pub fn graph_composite_score(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
    graph_proximity: f64,
) -> f64 {
    let raw = if graph_proximity > 0.0 {
        0.30 * similarity + 0.20 * decay + 0.15 * recency
            + 0.15 * importance.min(1.0) + 0.20 * graph_proximity
    } else {
        0.40 * similarity + 0.25 * decay + 0.20 * recency + 0.15 * importance.min(1.0)
    };
    raw * valence_boost(valence)
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
        // Just recorded: elapsed = 0
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

    #[test]
    fn test_composite_score_basic() {
        let score = composite_score(1.0, 1.0, 1.0, 1.0, 0.0);
        // (0.40 + 0.25 + 0.20 + 0.15) * 1.0 = 1.0
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_composite_score_with_valence() {
        let score = composite_score(1.0, 1.0, 1.0, 1.0, 1.0);
        // 1.0 * 1.3 = 1.3
        assert!((score - 1.3).abs() < 1e-10);
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
        // (0.30 + 0.20 + 0.15 + 0.15 + 0.20) * 1.0 = 1.0
        assert!((score - 1.0).abs() < 1e-10);
    }

    // ── Monotonicity & property invariants ──

    #[test]
    fn test_composite_monotonic_in_similarity() {
        // Higher similarity → higher score (all else equal)
        let low = composite_score(0.3, 0.5, 0.5, 0.5, 0.0);
        let high = composite_score(0.9, 0.5, 0.5, 0.5, 0.0);
        assert!(high > low, "higher similarity should yield higher score");
    }

    #[test]
    fn test_composite_monotonic_in_importance() {
        let low = composite_score(0.5, 0.5, 0.5, 0.2, 0.0);
        let high = composite_score(0.5, 0.5, 0.5, 0.9, 0.0);
        assert!(high > low, "higher importance should yield higher score");
    }

    #[test]
    fn test_composite_monotonic_in_recency() {
        let low = composite_score(0.5, 0.5, 0.1, 0.5, 0.0);
        let high = composite_score(0.5, 0.5, 0.9, 0.5, 0.0);
        assert!(high > low, "higher recency should yield higher score");
    }

    #[test]
    fn test_valence_symmetric() {
        // |+0.7| == |-0.7| → same boost
        assert!((valence_boost(0.7) - valence_boost(-0.7)).abs() < 1e-10);
    }

    #[test]
    fn test_valence_always_geq_1() {
        // Valence boost should never reduce scores
        for v in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert!(valence_boost(v) >= 1.0, "valence_boost({v}) = {} < 1.0", valence_boost(v));
        }
    }

    #[test]
    fn test_composite_non_negative() {
        // All inputs in [0,1], valence in [-1,1] → score ≥ 0
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
        // Use non-uniform inputs: when graph proximity redistributes weights
        // from similarity (0.40→0.30), the net change depends on the input mix.
        // With sim=0.8 (above avg), lowering sim weight hurts; proximity must compensate.
        let without = graph_composite_score(0.3, 0.8, 0.7, 0.6, 0.0, 0.0);
        let with = graph_composite_score(0.3, 0.8, 0.7, 0.6, 0.0, 0.8);
        assert!(with > without,
            "graph proximity (0.8) should increase score: without={without}, with={with}");
    }

    #[test]
    fn test_decay_monotonic_in_elapsed() {
        // More elapsed time → lower decay score
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
        // build_why should never return an empty vec
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
        // importance > 1.0 should be capped to 1.0 in composite
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

        // Higher decay → higher eviction score (less evictable)
        let low = eviction_score(0.1, 0.5);
        let high = eviction_score(0.9, 0.5);
        assert!(high > low, "higher decay should yield higher eviction score");
    }
}
