//! Engine-level world model API.
//!
//! Wires the transition model into `YantrikDB` for persistence
//! and integration with action evaluation.

use crate::error::Result;
use crate::world_model::{
    summarize_world_model, ActionKind, ActionOutcome, StateFeatures,
    TransitionModel, WorldModelSummary,
};

use super::{now, YantrikDB};

/// Meta key for persisted transition model.
const TRANSITION_MODEL_META_KEY: &str = "world_transition_model";

impl YantrikDB {
    // ── Persistence ──

    /// Load the transition model from the database.
    pub fn load_transition_model(&self) -> Result<TransitionModel> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), TRANSITION_MODEL_META_KEY)?;
        match meta {
            Some(json) => {
                let mut model: TransitionModel = serde_json::from_str(&json).map_err(|e| {
                    crate::error::YantrikDbError::Database(
                        rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                    )
                })?;
                model.rebuild_index();
                Ok(model)
            }
            None => Ok(TransitionModel::new()),
        }
    }

    /// Persist the transition model.
    pub fn save_transition_model(&self, model: &TransitionModel) -> Result<()> {
        let json = serde_json::to_string(model).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![TRANSITION_MODEL_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Main API ──

    /// Record an action→outcome transition in the world model.
    pub fn record_transition(
        &self,
        features: StateFeatures,
        action: ActionKind,
        outcome: ActionOutcome,
    ) -> Result<()> {
        let mut model = self.load_transition_model()?;
        model.record(features, action, outcome);
        self.save_transition_model(&model)
    }

    /// Predict the outcome of an action in a given state.
    pub fn predict_outcome(
        &self,
        features: &StateFeatures,
        action: ActionKind,
    ) -> Result<f64> {
        let model = self.load_transition_model()?;
        Ok(model.expected_success(features, action))
    }

    /// Get the best action for a given state (among candidates).
    pub fn best_action_for_state(
        &self,
        features: &StateFeatures,
        candidates: &[ActionKind],
    ) -> Result<Option<ActionKind>> {
        let model = self.load_transition_model()?;
        Ok(model.best_action(features, candidates))
    }

    /// Get a summary of the world model.
    pub fn world_model_summary(&self) -> Result<WorldModelSummary> {
        let model = self.load_transition_model()?;
        Ok(summarize_world_model(&model))
    }

    /// Reset the world model.
    pub fn reset_transition_model(&self) -> Result<()> {
        self.save_transition_model(&TransitionModel::new())
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::world_model::{ActionKind, ActionOutcome, StateFeatures};

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    fn noon() -> StateFeatures {
        StateFeatures::discretize(43200.0, 0.7, 600.0, 0.0, 1)
    }

    #[test]
    fn test_transition_model_persistence() {
        let db = test_db();

        db.record_transition(noon(), ActionKind::ExecuteTool, ActionOutcome::Succeeded)
            .unwrap();
        db.record_transition(noon(), ActionKind::ExecuteTool, ActionOutcome::Succeeded)
            .unwrap();
        db.record_transition(noon(), ActionKind::ExecuteTool, ActionOutcome::Failed)
            .unwrap();

        let success = db.predict_outcome(&noon(), ActionKind::ExecuteTool).unwrap();
        assert!(success > 0.4, "Expected success > 0.4, got {:.3}", success);
    }

    #[test]
    fn test_best_action() {
        let db = test_db();

        // Suggestions accepted
        for _ in 0..10 {
            db.record_transition(
                noon(),
                ActionKind::SurfaceSuggestion,
                ActionOutcome::Accepted,
            )
            .unwrap();
        }
        // Notifications rejected
        for _ in 0..10 {
            db.record_transition(
                noon(),
                ActionKind::SendNotification,
                ActionOutcome::Rejected,
            )
            .unwrap();
        }

        let best = db
            .best_action_for_state(
                &noon(),
                &[ActionKind::SurfaceSuggestion, ActionKind::SendNotification],
            )
            .unwrap();
        assert_eq!(best, Some(ActionKind::SurfaceSuggestion));
    }

    #[test]
    fn test_world_model_summary() {
        let db = test_db();

        for _ in 0..5 {
            db.record_transition(noon(), ActionKind::InvokeLlm, ActionOutcome::Succeeded)
                .unwrap();
        }

        let summary = db.world_model_summary().unwrap();
        assert_eq!(summary.total_transitions, 5);
        assert_eq!(summary.unique_pairs, 1);
    }

    #[test]
    fn test_reset() {
        let db = test_db();
        db.record_transition(noon(), ActionKind::ExecuteTool, ActionOutcome::Succeeded)
            .unwrap();
        db.reset_transition_model().unwrap();

        let summary = db.world_model_summary().unwrap();
        assert_eq!(summary.total_transitions, 0);
    }
}
