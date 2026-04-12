//! Engine-level receptivity/interruptibility API.
//!
//! Wires the receptivity model into `YantrikDB` for persistence
//! and integration with the cognitive graph.

use crate::error::Result;
use crate::receptivity::{
    ActivityState, ContextSnapshot, NotificationMode, ReceptivityEstimate,
    ReceptivityModel, SuggestionOutcome,
};

use super::{now, YantrikDB};

/// Meta key for persisted receptivity model.
const RECEPTIVITY_META_KEY: &str = "receptivity_model";

impl YantrikDB {
    // ── Persistence ──

    /// Load the receptivity model from the database (or create default).
    pub fn load_receptivity_model(&self) -> Result<ReceptivityModel> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), RECEPTIVITY_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(ReceptivityModel::new()),
        }
    }

    /// Persist the receptivity model to the database.
    pub fn save_receptivity_model(&self, model: &ReceptivityModel) -> Result<()> {
        let json = serde_json::to_string(model).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![RECEPTIVITY_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Estimation ──

    /// Estimate current user receptivity given a context snapshot.
    pub fn estimate_receptivity(
        &self,
        context: &ContextSnapshot,
    ) -> Result<ReceptivityEstimate> {
        let model = self.load_receptivity_model()?;
        Ok(model.estimate(context))
    }

    /// Quick check: is the user likely receptive right now?
    ///
    /// Uses default threshold of 0.5.
    pub fn is_user_receptive(
        &self,
        context: &ContextSnapshot,
    ) -> Result<bool> {
        let estimate = self.estimate_receptivity(context)?;
        Ok(estimate.is_receptive(0.5))
    }

    /// Check if current time is in quiet hours.
    pub fn is_quiet_hours(&self) -> Result<bool> {
        let model = self.load_receptivity_model()?;
        Ok(model.quiet_hours.is_quiet(now()))
    }

    /// Get remaining attention budget for a session.
    pub fn attention_budget_remaining(
        &self,
        suggestions_used: u32,
    ) -> Result<u32> {
        let model = self.load_receptivity_model()?;
        Ok(model.attention_budget.remaining(suggestions_used))
    }

    // ── Learning ──

    /// Record a suggestion outcome and update the receptivity model.
    pub fn receptivity_observe(
        &self,
        context: &ContextSnapshot,
        outcome: SuggestionOutcome,
    ) -> Result<()> {
        let mut model = self.load_receptivity_model()?;
        model.observe_outcome(context, outcome);
        self.save_receptivity_model(&model)
    }

    /// Batch-train the receptivity model from historical outcomes.
    ///
    /// Each entry is `(context, outcome)`.
    pub fn receptivity_train(
        &self,
        history: &[(ContextSnapshot, SuggestionOutcome)],
    ) -> Result<u64> {
        let mut model = self.load_receptivity_model()?;
        for (ctx, outcome) in history {
            model.observe_outcome(ctx, *outcome);
        }
        let count = model.training_count;
        self.save_receptivity_model(&model)?;
        Ok(count)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::receptivity::*;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_receptivity_persistence_roundtrip() {
        let db = test_db();

        // Save a model with modified weights
        let mut model = ReceptivityModel::new();
        model.bias = 1.5;
        model.training_count = 42;
        db.save_receptivity_model(&model).unwrap();

        // Load and verify
        let loaded = db.load_receptivity_model().unwrap();
        assert!((loaded.bias - 1.5).abs() < 1e-10);
        assert_eq!(loaded.training_count, 42);
    }

    #[test]
    fn test_estimate_receptivity() {
        let db = test_db();
        let context = ContextSnapshot {
            now: 50000.0,
            activity: ActivityState::Idle,
            recent_interactions_15min: 5,
            recent_outcomes: (3, 0, 0),
            secs_since_last_interaction: 30.0,
            session_duration_secs: 600.0,
            emotional_valence: 0.3,
            session_suggestions_accepted: 2,
            session_suggestion_budget: 20,
            notification_mode: NotificationMode::All,
        };

        let estimate = db.estimate_receptivity(&context).unwrap();
        assert!(estimate.score >= 0.0 && estimate.score <= 1.0);
        assert!(!estimate.factors.is_empty());
    }

    #[test]
    fn test_receptivity_learning() {
        let db = test_db();

        let receptive_context = ContextSnapshot {
            now: 50000.0,
            activity: ActivityState::TaskSwitching,
            recent_interactions_15min: 8,
            recent_outcomes: (5, 0, 0),
            secs_since_last_interaction: 10.0,
            session_duration_secs: 1200.0,
            emotional_valence: 0.5,
            session_suggestions_accepted: 3,
            session_suggestion_budget: 20,
            notification_mode: NotificationMode::All,
        };

        let est_before = db.estimate_receptivity(&receptive_context).unwrap().score;

        // Train: user accepted multiple times in this context
        for _ in 0..10 {
            db.receptivity_observe(&receptive_context, SuggestionOutcome::Accepted)
                .unwrap();
        }

        let est_after = db.estimate_receptivity(&receptive_context).unwrap().score;
        assert!(
            est_after >= est_before - 0.1, // Should stay same or increase
            "Score should not decrease much after accepts: {} -> {}",
            est_before, est_after
        );
    }

    #[test]
    fn test_quiet_hours_check() {
        let db = test_db();
        // Default model has quiet hours 22:00-07:00
        // This test just verifies the method works without error
        let result = db.is_quiet_hours();
        assert!(result.is_ok());
    }

    #[test]
    fn test_attention_budget() {
        let db = test_db();
        assert_eq!(db.attention_budget_remaining(0).unwrap(), 20);
        assert_eq!(db.attention_budget_remaining(15).unwrap(), 5);
        assert_eq!(db.attention_budget_remaining(25).unwrap(), 0);
    }
}
