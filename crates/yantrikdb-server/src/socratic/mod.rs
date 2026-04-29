//! RFC 007 — Socratic query rewrite operators.
//!
//! ## What this owns
//!
//! Six typed [`SocraticOperator`] families. Each operator:
//! 1. Looks at the original query + retrieved evidence and decides
//!    whether its trigger condition fires (`fires_on`).
//! 2. If yes, produces a [`SocraticSuggestion`]: the rewritten query,
//!    a `reason_code`, and the *typed* transformation template.
//!
//! The LLM does NOT pick the operator. Operator selection is
//! deterministic + graph-grounded. The LLM (if used at all) only
//! phrases the suggestion in natural language.
//!
//! ## Why typed operators, not free-form LLM
//!
//! Free-form LLM rewrite is unauditable — the same input produces
//! different rewrites across model versions, and operators can't
//! reason about WHY a rewrite happened. Typed operators give:
//! - **Reproducibility**: same trigger inputs → same suggestion.
//! - **Auditability**: `reason_code` says exactly which operator fired.
//! - **Per-operator tuning**: enable/disable a single operator class.
//!
//! ## What's NOT here (deferred)
//!
//! - The /v1/socratic_rewrite axum handler.
//! - The actual LLM paraphrase pass — substrate ships the typed
//!   transformation strings; consumer optionally feeds them to a
//!   model for surface-form polish.
//! - The 30-50 query evaluation benchmark (saga consumer task).
//! - OUTCOME_TO_UPSTREAM_LEVERS — depends on rule-edge whitelist
//!   (RFC 007 Phase 3). Substrate exposes a placeholder operator that
//!   currently never fires; impl lands when the whitelist does.

pub mod evidence;
pub mod operator;

pub use evidence::{ContradictionEvidence, EvidenceSnapshot, EvidenceWindow};
pub use operator::{
    BinaryToConditional, ContextCompletion, EntityDisambiguation, GlobalToTemporal,
    OutcomeToUpstreamLevers, PropositionToSourceComparison, ReasonCode, SocraticOperator,
    SocraticSuggestion,
};

use std::sync::Arc;

/// Run a list of operators against a query + evidence snapshot. Returns
/// the suggestions from every operator that fires, in deterministic
/// (declared) order. The caller decides which subset to surface to
/// the user — we don't filter or rank here.
pub fn rewrite(
    operators: &[Arc<dyn SocraticOperator>],
    query: &str,
    evidence: &EvidenceSnapshot,
) -> Vec<SocraticSuggestion> {
    operators
        .iter()
        .filter_map(|op| {
            if op.fires_on(query, evidence) {
                Some(op.suggest(query, evidence))
            } else {
                None
            }
        })
        .collect()
}

/// Default set of operators ready for production. Order matches the
/// RFC's listing: BINARY_TO_CONDITIONAL → GLOBAL_TO_TEMPORAL →
/// PROPOSITION_TO_SOURCE_COMPARISON → OUTCOME_TO_UPSTREAM_LEVERS →
/// ENTITY_DISAMBIGUATION → CONTEXT_COMPLETION.
pub fn default_operators() -> Vec<Arc<dyn SocraticOperator>> {
    vec![
        Arc::new(BinaryToConditional),
        Arc::new(GlobalToTemporal),
        Arc::new(PropositionToSourceComparison),
        Arc::new(OutcomeToUpstreamLevers),
        Arc::new(EntityDisambiguation),
        Arc::new(ContextCompletion),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_operator_set_has_all_six_families() {
        let ops = default_operators();
        assert_eq!(ops.len(), 6);
    }

    #[test]
    fn rewrite_returns_only_suggestions_from_firing_operators() {
        // No evidence variance → BINARY_TO_CONDITIONAL won't fire.
        // No temporal spread → GLOBAL_TO_TEMPORAL won't fire.
        // No contradictions → PROPOSITION_TO_SOURCE_COMPARISON won't fire.
        // No entity ambiguity → ENTITY_DISAMBIGUATION won't fire.
        // No unresolved context → CONTEXT_COMPLETION won't fire.
        // Net: zero suggestions.
        let evidence = EvidenceSnapshot::default();
        let ops = default_operators();
        let suggestions = rewrite(&ops, "is X true?", &evidence);
        assert!(
            suggestions.is_empty(),
            "expected no fires on null evidence, got {:?}",
            suggestions
        );
    }

    #[test]
    fn rewrite_returns_suggestions_in_declared_order() {
        // Construct evidence that triggers two operators in opposite
        // declared order. Result should follow the operator slice order,
        // not whichever fired "harder."
        let mut evidence = EvidenceSnapshot::default();
        evidence.context_variance = 0.9; // triggers BINARY_TO_CONDITIONAL
        evidence.temporal_spread_days = 365.0; // triggers GLOBAL_TO_TEMPORAL

        let ops: Vec<Arc<dyn SocraticOperator>> = vec![
            Arc::new(GlobalToTemporal),
            Arc::new(BinaryToConditional),
        ];
        let suggestions = rewrite(&ops, "did Acme grow last year?", &evidence);
        assert_eq!(suggestions.len(), 2);
        // Order matches the operators slice, NOT a "rank" of the suggestions.
        assert_eq!(suggestions[0].reason_code, ReasonCode::GlobalToTemporal);
        assert_eq!(suggestions[1].reason_code, ReasonCode::BinaryToConditional);
    }
}
