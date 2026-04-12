//! Engine-level causal inference API.
//!
//! Wires the causal-lite module into `YantrikDB` for persistence
//! and integration with the cognitive tick pipeline.

use crate::causal::{
    apply_granger_evidence, causal_summary, discover_local_causality, estimate_effect,
    explain_causal_edge, predict_effects, record_intervention, what_if, CausalConfig,
    CausalExplanation, CausalNode, CausalStore, CausalSummary, DiscoveryReport,
    EffectEstimate, GrangerResult, PredictedEffect, WhatIfResult,
};
use crate::error::Result;
use crate::world_model::{ActionKind, StateFeatures};

use super::{now, YantrikDB};

/// Meta key for persisted causal store.
const CAUSAL_STORE_META_KEY: &str = "causal_store";
/// Meta key for persisted causal config.
const CAUSAL_CONFIG_META_KEY: &str = "causal_config";

impl YantrikDB {
    // ── Persistence ──

    /// Load the causal store from the database.
    pub fn load_causal_store(&self) -> Result<CausalStore> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), CAUSAL_STORE_META_KEY)?;
        match meta {
            Some(json) => {
                let mut store: CausalStore = serde_json::from_str(&json).map_err(|e| {
                    crate::error::YantrikDbError::Database(
                        rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                    )
                })?;
                // Rebuild runtime indices after deserialization.
                store.rebuild_indices();
                // Apply persisted config if available.
                if let Ok(config) = self.load_causal_config() {
                    store.config = config;
                }
                Ok(store)
            }
            None => {
                let mut store = CausalStore::new();
                if let Ok(config) = self.load_causal_config() {
                    store.config = config;
                }
                Ok(store)
            }
        }
    }

    /// Persist the causal store.
    pub fn save_causal_store(&self, store: &CausalStore) -> Result<()> {
        let json = serde_json::to_string(store).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![CAUSAL_STORE_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the causal configuration.
    pub fn load_causal_config(&self) -> Result<CausalConfig> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), CAUSAL_CONFIG_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(CausalConfig::default()),
        }
    }

    /// Persist the causal configuration.
    pub fn save_causal_config(&self, config: &CausalConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![CAUSAL_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Discovery ──

    /// Run a full causal discovery pass using recent events and graph edges.
    ///
    /// Loads causal store, event buffer, and graph edges; runs discovery;
    /// persists the updated store. Returns a report of what was found.
    pub fn discover_causality(&self) -> Result<DiscoveryReport> {
        let mut store = self.load_causal_store()?;
        let event_buffer = self.load_event_buffer()?;
        let graph_edges = self.load_cognitive_graph_edges()?;
        let ts = now();

        let report = discover_local_causality(&mut store, &event_buffer, &graph_edges, ts);
        self.save_causal_store(&store)?;

        // Update lifetime counters from store stats.
        Ok(report)
    }

    // ── Intervention tracking ──

    /// Record the outcome of a deliberate intervention.
    ///
    /// Call this after taking an action to track whether the expected
    /// effect was observed.
    pub fn record_causal_intervention(
        &self,
        action: ActionKind,
        expected_effect: CausalNode,
        effect_observed: bool,
    ) -> Result<()> {
        let mut store = self.load_causal_store()?;
        let ts = now();
        record_intervention(&mut store, action, expected_effect, effect_observed, ts);
        self.save_causal_store(&store)
    }

    // ── Query API ──

    /// Estimate the causal effect of a cause on a specific effect.
    pub fn estimate_causal_effect(
        &self,
        cause: &CausalNode,
        effect: &CausalNode,
    ) -> Result<Option<EffectEstimate>> {
        let store = self.load_causal_store()?;
        Ok(estimate_effect(&store, cause, effect))
    }

    /// Predict all downstream effects of activating a cause.
    pub fn predict_causal_effects(
        &self,
        cause: &CausalNode,
        max_depth: u32,
    ) -> Result<Vec<PredictedEffect>> {
        let store = self.load_causal_store()?;
        Ok(predict_effects(&store, cause, max_depth))
    }

    /// What-if analysis: if cause were activated, what would happen?
    pub fn causal_what_if(
        &self,
        cause: &CausalNode,
        context: &StateFeatures,
        max_depth: u32,
    ) -> Result<WhatIfResult> {
        let store = self.load_causal_store()?;
        let transition_model = self.load_transition_model()?;
        Ok(what_if(&store, &transition_model, cause, context, max_depth))
    }

    /// Explain a specific causal edge in detail.
    pub fn explain_causal_edge(
        &self,
        cause: &CausalNode,
        effect: &CausalNode,
    ) -> Result<Option<CausalExplanation>> {
        let store = self.load_causal_store()?;
        Ok(explain_causal_edge(&store, cause, effect))
    }

    /// Get a summary of all causal knowledge.
    pub fn causal_summary(&self) -> Result<CausalSummary> {
        let store = self.load_causal_store()?;
        Ok(causal_summary(&store))
    }

    /// Apply Granger evidence from external analysis.
    pub fn apply_granger_causal_evidence(
        &self,
        cause: CausalNode,
        effect: CausalNode,
        result: &GrangerResult,
    ) -> Result<()> {
        let mut store = self.load_causal_store()?;
        let config = store.config.clone();
        let ts = now();
        apply_granger_evidence(&mut store, cause, effect, result, ts, &config);
        self.save_causal_store(&store)
    }

    /// Get causal store statistics (edge counts by stage).
    pub fn causal_stats(&self) -> Result<CausalStoreStats> {
        let store = self.load_causal_store()?;
        let summary = causal_summary(&store);
        Ok(CausalStoreStats {
            total_edges: summary.total_edges,
            hypothesized: summary.hypothesized,
            candidates: summary.candidates,
            established: summary.established,
            weakening: summary.weakening,
            refuted: summary.refuted,
            lifetime_hypothesized: summary.lifetime_hypothesized,
            lifetime_established: summary.lifetime_established,
            lifetime_refuted: summary.lifetime_refuted,
        })
    }

    /// Reset the causal store (for testing or recalibration).
    pub fn reset_causal_store(&self) -> Result<()> {
        let store = CausalStore::new();
        self.save_causal_store(&store)
    }

    // ── Internal helpers ──

    /// Load cognitive graph edges with causal semantics.
    ///
    /// Fetches Causes, Predicts, and Prevents edges from the cognitive
    /// graph for causal analysis.
    fn load_cognitive_graph_edges(&self) -> Result<Vec<crate::state::CognitiveEdge>> {
        use crate::state::CognitiveEdgeKind;
        let mut edges = Vec::new();
        for kind in [
            CognitiveEdgeKind::Causes,
            CognitiveEdgeKind::Predicts,
            CognitiveEdgeKind::Prevents,
        ] {
            edges.extend(self.load_cognitive_edges_by_kind(kind)?);
        }
        Ok(edges)
    }
}

/// Compact statistics for the causal store.
#[derive(Debug, Clone)]
pub struct CausalStoreStats {
    pub total_edges: usize,
    pub hypothesized: u32,
    pub candidates: u32,
    pub established: u32,
    pub weakening: u32,
    pub refuted: u32,
    pub lifetime_hypothesized: u64,
    pub lifetime_established: u64,
    pub lifetime_refuted: u64,
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::causal::{CausalNode, CausalStage};
    use crate::engine::YantrikDB;
    use crate::observer::{EventKind, SystemEvent, SystemEventData};
    use crate::world_model::ActionKind;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_empty_causal_store() {
        let db = test_db();
        let stats = db.causal_stats().unwrap();
        assert_eq!(stats.total_edges, 0);
        assert_eq!(stats.lifetime_hypothesized, 0);
    }

    #[test]
    fn test_save_and_load_causal_store() {
        let db = test_db();

        // Record an intervention to create an edge.
        db.record_causal_intervention(
            ActionKind::SurfaceSuggestion,
            CausalNode::Event(EventKind::SuggestionAccepted),
            true,
        )
        .unwrap();

        // Load and verify.
        let store = db.load_causal_store().unwrap();
        assert_eq!(store.edge_count(), 1);

        let edge = store.find_edge(
            &CausalNode::Action(ActionKind::SurfaceSuggestion),
            &CausalNode::Event(EventKind::SuggestionAccepted),
        );
        assert!(edge.is_some());
        assert_eq!(edge.unwrap().intervention_count, 1);
    }

    #[test]
    fn test_record_multiple_interventions() {
        let db = test_db();
        let cause = CausalNode::Action(ActionKind::ExecuteTool);
        let effect = CausalNode::Event(EventKind::ToolCallCompleted);

        // 3 successes, 1 failure.
        db.record_causal_intervention(ActionKind::ExecuteTool, effect.clone(), true).unwrap();
        db.record_causal_intervention(ActionKind::ExecuteTool, effect.clone(), true).unwrap();
        db.record_causal_intervention(ActionKind::ExecuteTool, effect.clone(), true).unwrap();
        db.record_causal_intervention(ActionKind::ExecuteTool, effect.clone(), false).unwrap();

        let store = db.load_causal_store().unwrap();
        let edge = store.find_edge(&cause, &effect).unwrap();
        assert_eq!(edge.intervention_count, 3);
        assert_eq!(edge.non_occurrence_count, 1);
        assert!(edge.confidence > 0.5);
    }

    #[test]
    fn test_estimate_causal_effect() {
        let db = test_db();
        let cause = CausalNode::Action(ActionKind::SurfaceSuggestion);
        let effect = CausalNode::Event(EventKind::SuggestionAccepted);

        // No edge yet.
        assert!(db.estimate_causal_effect(&cause, &effect).unwrap().is_none());

        // Create edge via intervention.
        db.record_causal_intervention(ActionKind::SurfaceSuggestion, effect.clone(), true)
            .unwrap();

        // Now should have an estimate.
        let est = db.estimate_causal_effect(&cause, &effect).unwrap();
        assert!(est.is_some());
        assert!(est.unwrap().confidence > 0.0);
    }

    #[test]
    fn test_predict_causal_effects() {
        let db = test_db();
        let cause = CausalNode::Action(ActionKind::SurfaceSuggestion);
        let effect = CausalNode::Event(EventKind::SuggestionAccepted);

        db.record_causal_intervention(ActionKind::SurfaceSuggestion, effect.clone(), true)
            .unwrap();

        let effects = db.predict_causal_effects(&cause, 2).unwrap();
        assert!(!effects.is_empty());
        assert_eq!(effects[0].node, effect);
    }

    #[test]
    fn test_explain_causal_edge() {
        let db = test_db();
        let cause = CausalNode::Action(ActionKind::ExecuteTool);
        let effect = CausalNode::Event(EventKind::ToolCallCompleted);

        db.record_causal_intervention(ActionKind::ExecuteTool, effect.clone(), true).unwrap();

        let explanation = db.explain_causal_edge(&cause, &effect).unwrap();
        assert!(explanation.is_some());

        let exp = explanation.unwrap();
        assert!(exp.intervention_count > 0);
        assert!(!exp.evidence_summary.is_empty());
    }

    #[test]
    fn test_causal_summary() {
        let db = test_db();

        db.record_causal_intervention(
            ActionKind::SurfaceSuggestion,
            CausalNode::Event(EventKind::SuggestionAccepted),
            true,
        )
        .unwrap();
        db.record_causal_intervention(
            ActionKind::ExecuteTool,
            CausalNode::Event(EventKind::ToolCallCompleted),
            true,
        )
        .unwrap();

        let summary = db.causal_summary().unwrap();
        assert_eq!(summary.total_edges, 2);
    }

    #[test]
    fn test_reset_causal_store() {
        let db = test_db();

        db.record_causal_intervention(
            ActionKind::SurfaceSuggestion,
            CausalNode::Event(EventKind::SuggestionAccepted),
            true,
        )
        .unwrap();

        assert_eq!(db.causal_stats().unwrap().total_edges, 1);

        db.reset_causal_store().unwrap();
        assert_eq!(db.causal_stats().unwrap().total_edges, 0);
    }

    #[test]
    fn test_save_and_load_causal_config() {
        let db = test_db();

        let mut config = crate::causal::CausalConfig::default();
        config.max_lag_window_secs = 600.0;
        config.min_observations_candidate = 10;

        db.save_causal_config(&config).unwrap();

        let loaded = db.load_causal_config().unwrap();
        assert_eq!(loaded.max_lag_window_secs, 600.0);
        assert_eq!(loaded.min_observations_candidate, 10);
    }

    #[test]
    fn test_discovery_with_events() {
        let db = test_db();

        // Push some patterned events into the event buffer.
        let mut buffer = crate::observer::EventBuffer::new(1000);
        for i in 0..10 {
            let base = 1000.0 + i as f64 * 30.0;
            buffer.push(SystemEvent {
                timestamp: base,
                data: SystemEventData::AppOpened { app_id: 1 },
            });
            buffer.push(SystemEvent {
                timestamp: base + 3.0,
                data: SystemEventData::UserTyping {
                    app_id: 1,
                    duration_ms: 5000,
                    characters: 100,
                },
            });
        }
        db.save_event_buffer(&buffer).unwrap();

        let report = db.discover_causality().unwrap();
        // Should have found temporal patterns.
        assert!(report.total_edges > 0);

        // Verify persistence.
        let store = db.load_causal_store().unwrap();
        assert!(store.edge_count() > 0);
    }
}
