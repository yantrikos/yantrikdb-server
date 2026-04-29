//! RFC 007 — evidence snapshot used by Socratic operators.
//!
//! Each operator inspects an [`EvidenceSnapshot`] to decide whether
//! its trigger fires. The snapshot is deliberately a thin descriptor
//! — not the raw retrieved memories — so that operators stay pure
//! (no I/O) and the recall handler does the actual database work
//! once, ahead of operator dispatch.
//!
//! ## Field semantics
//!
//! - `context_variance` ∈ \[0.0, 1.0\] — how dispersed the supporting
//!   evidence is. High variance means contexts disagree on conditions
//!   surrounding the query. Crosses BINARY_TO_CONDITIONAL threshold
//!   at >= 0.7 (RFC 007 §triggers, configurable via [`EvidenceWindow`]).
//! - `temporal_spread_days` — span between the earliest and latest
//!   evidence's `valid_from`. Crosses GLOBAL_TO_TEMPORAL threshold
//!   at >= 90 days.
//! - `contradiction_density` ∈ \[0.0, 1.0\] — fraction of evidence
//!   pairs that contradict each other. Crosses
//!   PROPOSITION_TO_SOURCE_COMPARISON threshold at >= 0.2 (1 in 5).
//! - `entity_ambiguity_score` ∈ \[0.0, 1.0\] — how many alternate
//!   entities the query could refer to. Crosses ENTITY_DISAMBIGUATION
//!   threshold at >= 0.5.
//! - `unresolved_context_dimensions` — list of standard context
//!   dimensions (`time`, `person`, `dosage`, `location`, …) that the
//!   query lacks but evidence varies on. Triggers CONTEXT_COMPLETION
//!   when non-empty.
//! - `contradictions` — concrete (rid_a, rid_b, sources) tuples for
//!   PROPOSITION_TO_SOURCE_COMPARISON.

use serde::{Deserialize, Serialize};

/// One contradiction pair surfaced by the conflict scanner. Used by
/// `PROPOSITION_TO_SOURCE_COMPARISON`'s suggestion text so it can
/// name specific rids + sources.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContradictionEvidence {
    pub rid_a: String,
    pub rid_b: String,
    pub source_a: Option<String>,
    pub source_b: Option<String>,
}

/// Trigger thresholds. Operators read from this. Per-tenant overrides
/// (RFC 021 PR-2 substrate) can swap these without modifying operator
/// logic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvidenceWindow {
    pub context_variance_threshold: f32,
    pub temporal_spread_days_threshold: f32,
    pub contradiction_density_threshold: f32,
    pub entity_ambiguity_threshold: f32,
}

impl Default for EvidenceWindow {
    fn default() -> Self {
        Self {
            context_variance_threshold: 0.7,
            temporal_spread_days_threshold: 90.0,
            contradiction_density_threshold: 0.2,
            entity_ambiguity_threshold: 0.5,
        }
    }
}

/// Snapshot of what the recall handler observed. Constructed before
/// operator dispatch and shared (immutably) across all operators.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EvidenceSnapshot {
    pub context_variance: f32,
    pub temporal_spread_days: f32,
    pub contradiction_density: f32,
    pub entity_ambiguity_score: f32,
    pub alternate_entities: Vec<String>,
    pub unresolved_context_dimensions: Vec<String>,
    pub contradictions: Vec<ContradictionEvidence>,
    pub window: EvidenceWindow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_window_thresholds_match_rfc() {
        let w = EvidenceWindow::default();
        assert_eq!(w.context_variance_threshold, 0.7);
        assert_eq!(w.temporal_spread_days_threshold, 90.0);
        assert_eq!(w.contradiction_density_threshold, 0.2);
        assert_eq!(w.entity_ambiguity_threshold, 0.5);
    }

    #[test]
    fn default_snapshot_is_all_zero() {
        let s = EvidenceSnapshot::default();
        assert_eq!(s.context_variance, 0.0);
        assert_eq!(s.temporal_spread_days, 0.0);
        assert_eq!(s.contradiction_density, 0.0);
        assert!(s.alternate_entities.is_empty());
        assert!(s.unresolved_context_dimensions.is_empty());
        assert!(s.contradictions.is_empty());
    }
}
