//! Engine-level online learning & calibration API.
//!
//! Wires the calibration module into `YantrikDB` for persistence
//! and integration with the evaluator/policy pipeline.

use crate::calibration::{
    action_threshold, calibrated_confidence, learning_report, record_belief_confirmed,
    record_belief_contradicted, record_interaction, weight_snapshot, InteractionOutcome,
    InteractionRecord, LearningConfig, LearningReport, LearningState, WeightSnapshot,
};
use crate::error::Result;

use super::{now, YantrikDB};

/// Meta key for persisted learning state.
const LEARNING_STATE_META_KEY: &str = "learning_state";
/// Meta key for persisted learning config.
const LEARNING_CONFIG_META_KEY: &str = "learning_config";

impl YantrikDB {
    // ── Persistence ──

    /// Load the learning state from the database.
    pub fn load_learning_state(&self) -> Result<LearningState> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), LEARNING_STATE_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => {
                // Check for persisted config to apply to new state
                let mut state = LearningState::new();
                if let Ok(config) = self.load_learning_config() {
                    state.config = config;
                }
                Ok(state)
            }
        }
    }

    /// Persist the learning state.
    pub fn save_learning_state(&self, state: &LearningState) -> Result<()> {
        let json = serde_json::to_string(state).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![LEARNING_STATE_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the learning configuration.
    pub fn load_learning_config(&self) -> Result<LearningConfig> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), LEARNING_CONFIG_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(LearningConfig::default()),
        }
    }

    /// Persist the learning configuration.
    pub fn save_learning_config(&self, config: &LearningConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![LEARNING_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Main API ──

    /// Record an interaction outcome and update all learning components.
    ///
    /// This is the primary entry point — called after every user interaction
    /// with a suggestion/action. Updates bandits, calibration map, and
    /// buffers interactions for batch weight refits.
    pub fn record_learning_interaction(
        &self,
        action_kind: String,
        raw_confidence: f64,
        outcome: InteractionOutcome,
        features: [f64; 4],
    ) -> Result<()> {
        let mut state = self.load_learning_state()?;
        let ts = now();

        let record = InteractionRecord {
            action_kind,
            raw_confidence,
            outcome,
            features,
            timestamp: ts,
        };

        record_interaction(&mut state, record, ts);
        self.save_learning_state(&state)
    }

    /// Record that a belief from a specific source was confirmed.
    pub fn learning_belief_confirmed(&self, source: &str) -> Result<()> {
        let mut state = self.load_learning_state()?;
        record_belief_confirmed(&mut state, source);
        self.save_learning_state(&state)
    }

    /// Record that a belief from a specific source was contradicted.
    pub fn learning_belief_contradicted(&self, source: &str) -> Result<()> {
        let mut state = self.load_learning_state()?;
        record_belief_contradicted(&mut state, source);
        self.save_learning_state(&state)
    }

    /// Get the calibrated confidence for a raw confidence value.
    ///
    /// Maps raw model confidence through the isotonic calibration map
    /// to produce a value closer to the actual observed success rate.
    pub fn calibrated_confidence(&self, raw_confidence: f64) -> Result<f64> {
        let state = self.load_learning_state()?;
        Ok(calibrated_confidence(&state, raw_confidence))
    }

    /// Get the recommended confidence threshold for an action kind.
    ///
    /// Based on the Beta-Bernoulli bandit posterior for this action kind,
    /// returns a threshold below which the system should not act.
    pub fn action_threshold(&self, action_kind: &str) -> Result<f64> {
        let state = self.load_learning_state()?;
        Ok(action_threshold(&state, action_kind))
    }

    /// Get a snapshot of the current utility weights.
    pub fn weight_snapshot(&self) -> Result<WeightSnapshot> {
        let state = self.load_learning_state()?;
        Ok(weight_snapshot(&state))
    }

    /// Get a comprehensive learning report.
    pub fn learning_report(&self) -> Result<LearningReport> {
        let state = self.load_learning_state()?;
        Ok(learning_report(&state))
    }

    /// Get the source reliability for a given evidence source.
    pub fn source_reliability(&self, source: &str) -> Result<f64> {
        let state = self.load_learning_state()?;
        Ok(state.reliability.reliability(source))
    }

    /// Get learning statistics: (total_interactions, action_kinds, weight_refits).
    pub fn learning_stats(&self) -> Result<(u64, usize, u64)> {
        let state = self.load_learning_state()?;
        Ok((
            state.total_interactions,
            state.bandits.bandits.len(),
            state.weights.update_count,
        ))
    }

    /// Reset the learning state (clear all learned weights, bandits, calibration).
    /// Irreversible.
    pub fn reset_learning_state(&self) -> Result<()> {
        let state = LearningState::new();
        self.save_learning_state(&state)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::calibration::InteractionOutcome;
    use crate::engine::YantrikDB;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_learning_interaction_recording() {
        let db = test_db();

        // Record several interactions
        for i in 0..5 {
            let outcome = if i < 4 {
                InteractionOutcome::Accepted
            } else {
                InteractionOutcome::Rejected
            };
            db.record_learning_interaction(
                "remind".to_string(),
                0.7,
                outcome,
                [0.8, 0.5, 0.3, 0.6],
            )
            .unwrap();
        }

        let (total, kinds, _) = db.learning_stats().unwrap();
        assert_eq!(total, 5);
        assert_eq!(kinds, 1);
    }

    #[test]
    fn test_calibrated_confidence_after_training() {
        let db = test_db();

        // Seed: high confidence → positive, low confidence → negative
        for _ in 0..20 {
            db.record_learning_interaction(
                "suggest".to_string(),
                0.85,
                InteractionOutcome::Accepted,
                [0.8, 0.6, 0.4, 0.7],
            )
            .unwrap();
            db.record_learning_interaction(
                "suggest".to_string(),
                0.15,
                InteractionOutcome::Rejected,
                [0.2, 0.1, 0.1, 0.2],
            )
            .unwrap();
        }

        // Force calibration refit via state
        let mut state = db.load_learning_state().unwrap();
        state.calibration.refit();
        db.save_learning_state(&state).unwrap();

        let high = db.calibrated_confidence(0.85).unwrap();
        let low = db.calibrated_confidence(0.15).unwrap();
        assert!(
            high > low,
            "Calibrated high ({}) should exceed calibrated low ({})",
            high,
            low
        );
    }

    #[test]
    fn test_action_threshold() {
        let db = test_db();

        // Record many accepts for "remind"
        for _ in 0..10 {
            db.record_learning_interaction(
                "remind".to_string(),
                0.7,
                InteractionOutcome::Accepted,
                [0.5; 4],
            )
            .unwrap();
        }
        db.record_learning_interaction(
            "remind".to_string(),
            0.7,
            InteractionOutcome::Rejected,
            [0.5; 4],
        )
        .unwrap();

        let thresh = db.action_threshold("remind").unwrap();
        assert!(
            thresh > 0.3 && thresh < 0.9,
            "Threshold should be reasonable: {}",
            thresh
        );

        // Unknown action kind should return default
        let unknown_thresh = db.action_threshold("unknown_action").unwrap();
        assert!(
            (unknown_thresh - 0.5).abs() < 0.2,
            "Unknown action threshold should be near default: {}",
            unknown_thresh
        );
    }

    #[test]
    fn test_belief_reliability_tracking() {
        let db = test_db();

        db.learning_belief_confirmed("user").unwrap();
        db.learning_belief_confirmed("user").unwrap();
        db.learning_belief_confirmed("llm").unwrap();
        db.learning_belief_contradicted("llm").unwrap();

        let user_rel = db.source_reliability("user").unwrap();
        let llm_rel = db.source_reliability("llm").unwrap();
        assert!(
            user_rel > llm_rel,
            "User reliability ({}) should exceed LLM ({})",
            user_rel,
            llm_rel
        );
    }

    #[test]
    fn test_weight_snapshot() {
        let db = test_db();

        let snap = db.weight_snapshot().unwrap();
        let sum = snap.effect_weight + snap.intent_weight + snap.preference_weight + snap.simulation_weight;
        assert!((sum - 1.0).abs() < 0.01, "Weights should sum to ~1.0");
        assert_eq!(snap.update_count, 0);
    }

    #[test]
    fn test_learning_report() {
        let db = test_db();

        for _ in 0..3 {
            db.record_learning_interaction(
                "suggest".to_string(),
                0.6,
                InteractionOutcome::Accepted,
                [0.5, 0.4, 0.3, 0.2],
            )
            .unwrap();
        }
        db.learning_belief_confirmed("user").unwrap();

        let report = db.learning_report().unwrap();
        assert_eq!(report.total_interactions, 3);
        assert_eq!(report.action_kinds_tracked, 1);
        assert!(!report.source_reliabilities.is_empty());
    }

    #[test]
    fn test_learning_persistence() {
        let db = test_db();

        db.record_learning_interaction(
            "test".to_string(),
            0.5,
            InteractionOutcome::Accepted,
            [0.5; 4],
        )
        .unwrap();

        // Verify persistence round-trip
        let state = db.load_learning_state().unwrap();
        assert_eq!(state.total_interactions, 1);
        assert_eq!(state.bandits.bandits.len(), 1);
    }

    #[test]
    fn test_reset_learning_state() {
        let db = test_db();

        db.record_learning_interaction(
            "test".to_string(),
            0.5,
            InteractionOutcome::Accepted,
            [0.5; 4],
        )
        .unwrap();

        db.reset_learning_state().unwrap();

        let (total, kinds, _) = db.learning_stats().unwrap();
        assert_eq!(total, 0);
        assert_eq!(kinds, 0);
    }
}
