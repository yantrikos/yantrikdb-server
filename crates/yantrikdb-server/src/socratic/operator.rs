//! RFC 007 — Socratic operator trait + 6 typed operators.
//!
//! The operators are unit structs (no per-operator state) so the
//! default set is `Vec<Arc<dyn SocraticOperator>>` cheaply. Per-tenant
//! configuration (enable/disable, threshold tuning) routes through
//! `EvidenceSnapshot::window` rather than per-operator state.

use super::evidence::EvidenceSnapshot;

/// Stable reason-code label for every operator. Wire format —
/// dashboards key on these (`socratic_suggestion_total{reason}` etc).
/// Pinned in tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReasonCode {
    BinaryToConditional,
    GlobalToTemporal,
    PropositionToSourceComparison,
    OutcomeToUpstreamLevers,
    EntityDisambiguation,
    ContextCompletion,
}

impl ReasonCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            ReasonCode::BinaryToConditional => "binary_to_conditional",
            ReasonCode::GlobalToTemporal => "global_to_temporal",
            ReasonCode::PropositionToSourceComparison => "proposition_to_source_comparison",
            ReasonCode::OutcomeToUpstreamLevers => "outcome_to_upstream_levers",
            ReasonCode::EntityDisambiguation => "entity_disambiguation",
            ReasonCode::ContextCompletion => "context_completion",
        }
    }
}

/// Output of one operator firing. The `rewritten_query` is the typed
/// transformation; an LLM downstream can paraphrase it for natural-
/// language delivery without changing semantic meaning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocraticSuggestion {
    pub reason_code: ReasonCode,
    pub rewritten_query: String,
    /// Free-form context the LLM-paraphrase pass can use, e.g. the
    /// specific rids that contradicted, the temporal range that was
    /// spread. Optional — operators may skip if redundant with the
    /// rewritten query.
    pub detail: Option<String>,
}

/// One Socratic operator. `fires_on` is pure; `suggest` is also pure
/// and called only after `fires_on` returned true.
pub trait SocraticOperator: Send + Sync {
    fn reason_code(&self) -> ReasonCode;
    fn fires_on(&self, query: &str, evidence: &EvidenceSnapshot) -> bool;
    fn suggest(&self, query: &str, evidence: &EvidenceSnapshot) -> SocraticSuggestion;
}

// ============================================================
// 1. BINARY_TO_CONDITIONAL
// ============================================================

/// Trigger: high context variance in supporting evidence.
/// Transformation: turn "is X true?" into "under what conditions is X
/// true?" — surfaces the contextual axis the evidence varies on.
pub struct BinaryToConditional;

impl SocraticOperator for BinaryToConditional {
    fn reason_code(&self) -> ReasonCode {
        ReasonCode::BinaryToConditional
    }

    fn fires_on(&self, _query: &str, evidence: &EvidenceSnapshot) -> bool {
        evidence.context_variance >= evidence.window.context_variance_threshold
    }

    fn suggest(&self, query: &str, evidence: &EvidenceSnapshot) -> SocraticSuggestion {
        SocraticSuggestion {
            reason_code: ReasonCode::BinaryToConditional,
            rewritten_query: format!(
                "Under what conditions is the answer to `{}` true?",
                query.trim()
            ),
            detail: Some(format!(
                "context_variance={:.2} >= {:.2}",
                evidence.context_variance, evidence.window.context_variance_threshold
            )),
        }
    }
}

// ============================================================
// 2. GLOBAL_TO_TEMPORAL
// ============================================================

/// Trigger: temporal spread in evidence (>= 90 days by default).
/// Transformation: bind the question to a time window because the
/// answer plausibly varies over time.
pub struct GlobalToTemporal;

impl SocraticOperator for GlobalToTemporal {
    fn reason_code(&self) -> ReasonCode {
        ReasonCode::GlobalToTemporal
    }

    fn fires_on(&self, _query: &str, evidence: &EvidenceSnapshot) -> bool {
        evidence.temporal_spread_days >= evidence.window.temporal_spread_days_threshold
    }

    fn suggest(&self, query: &str, evidence: &EvidenceSnapshot) -> SocraticSuggestion {
        SocraticSuggestion {
            reason_code: ReasonCode::GlobalToTemporal,
            rewritten_query: format!(
                "When was the answer to `{}` true, and how has it changed over time?",
                query.trim()
            ),
            detail: Some(format!(
                "temporal_spread_days={:.0} >= {:.0}",
                evidence.temporal_spread_days, evidence.window.temporal_spread_days_threshold
            )),
        }
    }
}

// ============================================================
// 3. PROPOSITION_TO_SOURCE_COMPARISON
// ============================================================

/// Trigger: contradiction density above threshold (default 0.2 = 1 in
/// 5 evidence pairs contradict).
/// Transformation: ask the user to compare specific contradicting
/// sources rather than treating their merged claim as fact.
pub struct PropositionToSourceComparison;

impl SocraticOperator for PropositionToSourceComparison {
    fn reason_code(&self) -> ReasonCode {
        ReasonCode::PropositionToSourceComparison
    }

    fn fires_on(&self, _query: &str, evidence: &EvidenceSnapshot) -> bool {
        evidence.contradiction_density >= evidence.window.contradiction_density_threshold
            && !evidence.contradictions.is_empty()
    }

    fn suggest(&self, query: &str, evidence: &EvidenceSnapshot) -> SocraticSuggestion {
        // Surface the first contradiction pair concretely.
        let pair = evidence.contradictions.first();
        let detail = pair.map(|p| {
            format!(
                "rid_a={}, rid_b={} (source_a={}, source_b={})",
                p.rid_a,
                p.rid_b,
                p.source_a.as_deref().unwrap_or("unknown"),
                p.source_b.as_deref().unwrap_or("unknown"),
            )
        });
        SocraticSuggestion {
            reason_code: ReasonCode::PropositionToSourceComparison,
            rewritten_query: format!(
                "Sources disagree on `{}`. Which source do you trust, and why?",
                query.trim()
            ),
            detail,
        }
    }
}

// ============================================================
// 4. OUTCOME_TO_UPSTREAM_LEVERS
// ============================================================

/// Trigger: depends on RFC 007 Phase 3 rule-edge whitelist (deferred).
/// Substrate ships a placeholder that never fires; the suggest()
/// implementation is correct for the day the trigger lands.
pub struct OutcomeToUpstreamLevers;

impl SocraticOperator for OutcomeToUpstreamLevers {
    fn reason_code(&self) -> ReasonCode {
        ReasonCode::OutcomeToUpstreamLevers
    }

    fn fires_on(&self, _query: &str, _evidence: &EvidenceSnapshot) -> bool {
        // Placeholder: rule-edge whitelist not yet implemented.
        // Returning false unconditionally is the correct stub behavior.
        false
    }

    fn suggest(&self, query: &str, _evidence: &EvidenceSnapshot) -> SocraticSuggestion {
        SocraticSuggestion {
            reason_code: ReasonCode::OutcomeToUpstreamLevers,
            rewritten_query: format!(
                "What upstream factors influence `{}`?",
                query.trim()
            ),
            detail: Some("rule_edge_whitelist_pending".into()),
        }
    }
}

// ============================================================
// 5. ENTITY_DISAMBIGUATION
// ============================================================

/// Trigger: entity ambiguity at subject/object — multiple entities
/// could match the named referent.
pub struct EntityDisambiguation;

impl SocraticOperator for EntityDisambiguation {
    fn reason_code(&self) -> ReasonCode {
        ReasonCode::EntityDisambiguation
    }

    fn fires_on(&self, _query: &str, evidence: &EvidenceSnapshot) -> bool {
        evidence.entity_ambiguity_score >= evidence.window.entity_ambiguity_threshold
            && !evidence.alternate_entities.is_empty()
    }

    fn suggest(&self, query: &str, evidence: &EvidenceSnapshot) -> SocraticSuggestion {
        let detail = if evidence.alternate_entities.is_empty() {
            None
        } else {
            Some(format!(
                "candidates: {}",
                evidence.alternate_entities.join(", ")
            ))
        };
        SocraticSuggestion {
            reason_code: ReasonCode::EntityDisambiguation,
            rewritten_query: format!(
                "Which specific entity does `{}` refer to?",
                query.trim()
            ),
            detail,
        }
    }
}

// ============================================================
// 6. CONTEXT_COMPLETION
// ============================================================

/// Trigger: query lacks context dimensions (time, person, dosage, …)
/// that the evidence varies on. Asks the user to specify the missing
/// dimension before answering.
pub struct ContextCompletion;

impl SocraticOperator for ContextCompletion {
    fn reason_code(&self) -> ReasonCode {
        ReasonCode::ContextCompletion
    }

    fn fires_on(&self, _query: &str, evidence: &EvidenceSnapshot) -> bool {
        !evidence.unresolved_context_dimensions.is_empty()
    }

    fn suggest(&self, query: &str, evidence: &EvidenceSnapshot) -> SocraticSuggestion {
        let dims = evidence.unresolved_context_dimensions.join(", ");
        SocraticSuggestion {
            reason_code: ReasonCode::ContextCompletion,
            rewritten_query: format!(
                "To answer `{}`, please specify: {}.",
                query.trim(),
                dims
            ),
            detail: Some(format!("unresolved_dimensions=[{}]", dims)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::evidence::ContradictionEvidence;
    use super::*;

    fn fresh() -> EvidenceSnapshot {
        EvidenceSnapshot::default()
    }

    #[test]
    fn reason_code_strings_pinned() {
        assert_eq!(
            ReasonCode::BinaryToConditional.as_str(),
            "binary_to_conditional"
        );
        assert_eq!(
            ReasonCode::GlobalToTemporal.as_str(),
            "global_to_temporal"
        );
        assert_eq!(
            ReasonCode::PropositionToSourceComparison.as_str(),
            "proposition_to_source_comparison"
        );
        assert_eq!(
            ReasonCode::OutcomeToUpstreamLevers.as_str(),
            "outcome_to_upstream_levers"
        );
        assert_eq!(
            ReasonCode::EntityDisambiguation.as_str(),
            "entity_disambiguation"
        );
        assert_eq!(
            ReasonCode::ContextCompletion.as_str(),
            "context_completion"
        );
    }

    // BINARY_TO_CONDITIONAL
    #[test]
    fn binary_to_conditional_does_not_fire_below_threshold() {
        let mut e = fresh();
        e.context_variance = 0.5;
        assert!(!BinaryToConditional.fires_on("q", &e));
    }

    #[test]
    fn binary_to_conditional_fires_at_or_above_threshold() {
        let mut e = fresh();
        e.context_variance = 0.7;
        assert!(BinaryToConditional.fires_on("q", &e));
    }

    #[test]
    fn binary_to_conditional_suggestion_starts_with_under_what_conditions() {
        let mut e = fresh();
        e.context_variance = 0.9;
        let s = BinaryToConditional.suggest("is X true?", &e);
        assert_eq!(s.reason_code, ReasonCode::BinaryToConditional);
        assert!(s.rewritten_query.starts_with("Under what conditions"));
    }

    // GLOBAL_TO_TEMPORAL
    #[test]
    fn global_to_temporal_does_not_fire_below_threshold() {
        let mut e = fresh();
        e.temporal_spread_days = 30.0;
        assert!(!GlobalToTemporal.fires_on("q", &e));
    }

    #[test]
    fn global_to_temporal_fires_at_threshold() {
        let mut e = fresh();
        e.temporal_spread_days = 90.0;
        assert!(GlobalToTemporal.fires_on("q", &e));
    }

    #[test]
    fn global_to_temporal_suggestion_mentions_time() {
        let mut e = fresh();
        e.temporal_spread_days = 365.0;
        let s = GlobalToTemporal.suggest("is X true?", &e);
        assert_eq!(s.reason_code, ReasonCode::GlobalToTemporal);
        assert!(s.rewritten_query.contains("time") || s.rewritten_query.contains("When"));
    }

    // PROPOSITION_TO_SOURCE_COMPARISON
    #[test]
    fn proposition_to_source_does_not_fire_without_contradictions() {
        let mut e = fresh();
        e.contradiction_density = 0.5; // high
                                       // but contradictions list empty
        assert!(!PropositionToSourceComparison.fires_on("q", &e));
    }

    #[test]
    fn proposition_to_source_fires_with_density_and_contradictions() {
        let mut e = fresh();
        e.contradiction_density = 0.3;
        e.contradictions.push(ContradictionEvidence {
            rid_a: "a".into(),
            rid_b: "b".into(),
            source_a: Some("alice".into()),
            source_b: Some("bob".into()),
        });
        assert!(PropositionToSourceComparison.fires_on("q", &e));
        let s = PropositionToSourceComparison.suggest("is X true?", &e);
        assert_eq!(s.reason_code, ReasonCode::PropositionToSourceComparison);
        let detail = s.detail.unwrap();
        assert!(detail.contains("rid_a=a"));
        assert!(detail.contains("alice"));
    }

    // OUTCOME_TO_UPSTREAM_LEVERS
    #[test]
    fn outcome_to_upstream_levers_currently_never_fires() {
        let mut e = fresh();
        e.context_variance = 1.0; // would fire other operators
        e.contradiction_density = 1.0;
        // Phase-3-blocked stub.
        assert!(!OutcomeToUpstreamLevers.fires_on("q", &e));
    }

    #[test]
    fn outcome_to_upstream_levers_suggestion_text_correct_for_when_trigger_lands() {
        let s = OutcomeToUpstreamLevers.suggest("did sales fall?", &fresh());
        assert!(s.rewritten_query.contains("upstream"));
        assert_eq!(s.reason_code, ReasonCode::OutcomeToUpstreamLevers);
    }

    // ENTITY_DISAMBIGUATION
    #[test]
    fn entity_disambiguation_requires_both_score_and_alternates() {
        let mut e = fresh();
        e.entity_ambiguity_score = 0.9;
        // No alternates → no fire.
        assert!(!EntityDisambiguation.fires_on("q", &e));
        e.alternate_entities = vec!["AcmeCo".into(), "AcmeInc".into()];
        assert!(EntityDisambiguation.fires_on("q", &e));
        let s = EntityDisambiguation.suggest("did Acme grow?", &e);
        assert_eq!(s.reason_code, ReasonCode::EntityDisambiguation);
        assert!(s.detail.unwrap().contains("AcmeCo"));
    }

    #[test]
    fn entity_disambiguation_does_not_fire_below_score_threshold() {
        let mut e = fresh();
        e.entity_ambiguity_score = 0.4;
        e.alternate_entities = vec!["AcmeCo".into()];
        assert!(!EntityDisambiguation.fires_on("q", &e));
    }

    // CONTEXT_COMPLETION
    #[test]
    fn context_completion_fires_only_when_dimensions_unresolved() {
        let mut e = fresh();
        assert!(!ContextCompletion.fires_on("q", &e));
        e.unresolved_context_dimensions = vec!["time".into(), "dosage".into()];
        assert!(ContextCompletion.fires_on("q", &e));
        let s = ContextCompletion.suggest("how much?", &e);
        assert_eq!(s.reason_code, ReasonCode::ContextCompletion);
        assert!(s.rewritten_query.contains("time"));
        assert!(s.rewritten_query.contains("dosage"));
    }

    #[test]
    fn dyn_dispatch_via_trait_object() {
        let op: std::sync::Arc<dyn SocraticOperator> =
            std::sync::Arc::new(BinaryToConditional);
        assert_eq!(op.reason_code(), ReasonCode::BinaryToConditional);
    }

    #[test]
    fn suggestion_rewritten_query_trims_input() {
        let mut e = fresh();
        e.context_variance = 0.9;
        let s = BinaryToConditional.suggest("   is X true?   ", &e);
        // The trimmed query is embedded in the rewritten form.
        assert!(s.rewritten_query.contains("`is X true?`"));
        assert!(!s.rewritten_query.contains("   is X true?"));
    }
}
