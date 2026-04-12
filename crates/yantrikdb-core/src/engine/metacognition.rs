//! Engine-level meta-cognition API.
//!
//! Wires the meta-cognitive assessment into `YantrikDB` for persistence
//! and integration with all subsystem states.

use crate::attention::AttentionConfig;
use crate::error::Result;
use crate::metacognition::{
    confidence_report, metacognitive_assessment, reasoning_health, should_abstain,
    AbstainDecision, ConfidenceReport, MetaActionCandidate, MetaCognitiveConfig,
    MetaCognitiveHistory, MetaCognitiveInputs, MetaCognitiveReport,
    ReasoningHealthReport,
};

use super::{now, YantrikDB};

/// Meta key for persisted meta-cognitive config.
const METACOG_CONFIG_META_KEY: &str = "metacognitive_config";
/// Meta key for persisted meta-cognitive history.
const METACOG_HISTORY_META_KEY: &str = "metacognitive_history";

/// Maximum snapshots retained in history.
const DEFAULT_MAX_SNAPSHOTS: usize = 100;

impl YantrikDB {
    // ── Persistence ──

    /// Load the meta-cognitive config.
    pub fn load_metacognitive_config(&self) -> Result<MetaCognitiveConfig> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), METACOG_CONFIG_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(MetaCognitiveConfig::default()),
        }
    }

    /// Persist the meta-cognitive config.
    pub fn save_metacognitive_config(&self, config: &MetaCognitiveConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![METACOG_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the meta-cognitive history.
    pub fn load_metacognitive_history(&self) -> Result<MetaCognitiveHistory> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), METACOG_HISTORY_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(MetaCognitiveHistory::new(DEFAULT_MAX_SNAPSHOTS)),
        }
    }

    /// Persist the meta-cognitive history.
    pub fn save_metacognitive_history(&self, history: &MetaCognitiveHistory) -> Result<()> {
        let json = serde_json::to_string(history).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![METACOG_HISTORY_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Meta-Cognition API ──

    /// Run a comprehensive meta-cognitive assessment.
    ///
    /// Loads all subsystem states and evaluates evidence sparsity,
    /// model disagreement, prediction accuracy, calibration drift,
    /// and coverage gaps.
    pub fn metacognitive_assessment(&self) -> Result<MetaCognitiveReport> {
        let config = self.load_metacognitive_config()?;
        let owned = self.load_metacognitive_inputs(&config)?;
        let inputs = owned.as_inputs(&config);

        let report = metacognitive_assessment(&inputs);

        // Record in history with a default "no candidates" decision.
        let decision = should_abstain(&report, &[], &config);
        let mut history = self.load_metacognitive_history()?;
        history.record(&report, &decision);
        self.save_metacognitive_history(&history)?;

        Ok(report)
    }

    /// Assess whether to proceed, escalate, or defer for given candidates.
    ///
    /// Performs a meta-cognitive assessment and then evaluates the
    /// abstain logic against the provided action candidates.
    pub fn should_abstain(
        &self,
        candidates: &[MetaActionCandidate],
    ) -> Result<(MetaCognitiveReport, AbstainDecision)> {
        let config = self.load_metacognitive_config()?;
        let owned = self.load_metacognitive_inputs(&config)?;
        let inputs = owned.as_inputs(&config);

        let report = metacognitive_assessment(&inputs);
        let decision = should_abstain(&report, candidates, &config);

        // Record in history.
        let mut history = self.load_metacognitive_history()?;
        history.record(&report, &decision);
        self.save_metacognitive_history(&history)?;

        Ok((report, decision))
    }

    /// Generate a detailed confidence and calibration report.
    pub fn confidence_report(&self) -> Result<ConfidenceReport> {
        let config = self.load_metacognitive_config()?;
        let owned = self.load_metacognitive_inputs(&config)?;
        let inputs = owned.as_inputs(&config);
        Ok(confidence_report(&inputs))
    }

    /// Generate a reasoning health report with grade and recommendations.
    pub fn reasoning_health(&self) -> Result<ReasoningHealthReport> {
        let config = self.load_metacognitive_config()?;
        let owned = self.load_metacognitive_inputs(&config)?;
        let inputs = owned.as_inputs(&config);
        Ok(reasoning_health(&inputs))
    }

    /// Get meta-cognitive statistics.
    pub fn metacognitive_stats(&self) -> Result<MetaCognitiveStats> {
        let history = self.load_metacognitive_history()?;
        let latest = history.latest().cloned();
        Ok(MetaCognitiveStats {
            total_assessments: history.total_assessments,
            total_escalations: history.total_escalations,
            total_deferrals: history.total_deferrals,
            total_proceeds: history.total_proceeds,
            snapshot_count: history.snapshot_count(),
            latest_confidence: latest.as_ref().map(|s| s.overall_confidence),
            confidence_trend_10: history.confidence_trend(10),
            escalation_rate_10: history.escalation_rate(10),
        })
    }

    /// Reset the meta-cognitive history.
    pub fn reset_metacognitive_history(&self) -> Result<()> {
        self.save_metacognitive_history(&MetaCognitiveHistory::new(DEFAULT_MAX_SNAPSHOTS))
    }

    // ── Internal ──

    /// Load all subsystem states needed for meta-cognitive assessment.
    fn load_metacognitive_inputs<'a>(
        &'a self,
        config: &'a MetaCognitiveConfig,
    ) -> Result<MetaCogInputsOwned> {
        let learning_state = self.load_learning_state()?;
        let belief_store = self.load_belief_store()?;
        let event_buffer = self.load_event_buffer()?;
        let skill_registry = self.load_skill_registry()?;
        let experiment_registry = self.load_experiment_registry()?;
        let transition_model = self.load_transition_model()?;

        Ok(MetaCogInputsOwned {
            learning_state,
            belief_store,
            event_buffer,
            skill_registry,
            experiment_registry,
            transition_model,
        })
    }
}

/// Owned version of MetaCognitiveInputs for engine-level usage.
///
/// We load all subsystem states into owned values, then create
/// the borrowed `MetaCognitiveInputs` from them.
struct MetaCogInputsOwned {
    learning_state: crate::calibration::LearningState,
    belief_store: crate::flywheel::BeliefStore,
    event_buffer: crate::observer::EventBuffer,
    skill_registry: crate::skills::SkillRegistry,
    experiment_registry: crate::experimenter::ExperimentRegistry,
    transition_model: crate::world_model::TransitionModel,
}

impl MetaCogInputsOwned {
    fn as_inputs<'a>(&'a self, config: &'a MetaCognitiveConfig) -> MetaCognitiveInputs<'a> {
        MetaCognitiveInputs {
            learning_state: &self.learning_state,
            belief_store: &self.belief_store,
            event_buffer: &self.event_buffer,
            skill_registry: &self.skill_registry,
            experiment_registry: &self.experiment_registry,
            transition_model: &self.transition_model,
            config,
            now: super::now(),
        }
    }
}

/// Compact meta-cognitive statistics.
#[derive(Debug, Clone)]
pub struct MetaCognitiveStats {
    pub total_assessments: u64,
    pub total_escalations: u64,
    pub total_deferrals: u64,
    pub total_proceeds: u64,
    pub snapshot_count: usize,
    pub latest_confidence: Option<f64>,
    pub confidence_trend_10: f64,
    pub escalation_rate_10: f64,
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::metacognition::{
        AbstainAction, MetaActionCandidate, MetaCognitiveConfig, MetaCognitiveHistory,
    };

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_save_load_metacognitive_config() {
        let db = test_db();

        let mut config = MetaCognitiveConfig::default();
        config.sparsity_escalate = 0.9;
        config.defer_threshold = 0.15;
        db.save_metacognitive_config(&config).unwrap();

        let loaded = db.load_metacognitive_config().unwrap();
        assert!((loaded.sparsity_escalate - 0.9).abs() < f64::EPSILON);
        assert!((loaded.defer_threshold - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn test_save_load_metacognitive_history() {
        let db = test_db();

        let mut history = MetaCognitiveHistory::new(50);
        history.total_assessments = 20;
        history.total_escalations = 5;
        history.total_deferrals = 2;
        history.total_proceeds = 13;
        db.save_metacognitive_history(&history).unwrap();

        let loaded = db.load_metacognitive_history().unwrap();
        assert_eq!(loaded.total_assessments, 20);
        assert_eq!(loaded.total_escalations, 5);
        assert_eq!(loaded.total_proceeds, 13);
    }

    #[test]
    fn test_metacognitive_assessment_empty() {
        let db = test_db();
        let report = db.metacognitive_assessment().unwrap();

        // Empty DB → high sparsity, low maturity.
        assert!(report.evidence_sparsity > 0.5);
        assert_eq!(report.belief_maturity, 0.0);
        assert!(!report.signal_details.is_empty());
    }

    #[test]
    fn test_should_abstain_api() {
        let db = test_db();

        // With high-confidence candidates → should proceed.
        let candidates = vec![
            MetaActionCandidate {
                description: "Send reminder".to_string(),
                confidence: 0.85,
            },
        ];
        let (report, decision) = db.should_abstain(&candidates).unwrap();
        // May or may not proceed depending on system state, but should not panic.
        assert!(report.overall_confidence >= 0.0);
        assert!(report.overall_confidence <= 1.0);

        // With very low candidates.
        let low_candidates = vec![
            MetaActionCandidate {
                description: "Risky".to_string(),
                confidence: 0.1,
            },
        ];
        let (_, decision2) = db.should_abstain(&low_candidates).unwrap();
        // Should at least warn about low candidate confidence.
        assert_ne!(decision2.action, AbstainAction::Proceed);
    }

    #[test]
    fn test_confidence_report() {
        let db = test_db();
        let report = db.confidence_report().unwrap();

        assert_eq!(report.bin_details.len(), 10);
        assert!(report.calibration_error >= 0.0);
    }

    #[test]
    fn test_reasoning_health() {
        let db = test_db();
        let health = db.reasoning_health().unwrap();

        assert!(!health.subsystem_health.is_empty());
        assert!(health.health_score >= 0.0 && health.health_score <= 1.0);
        assert!(['A', 'B', 'C', 'D', 'F'].contains(&health.grade));
    }

    #[test]
    fn test_metacognitive_stats() {
        let db = test_db();

        // Before any assessments.
        let stats = db.metacognitive_stats().unwrap();
        assert_eq!(stats.total_assessments, 0);
        assert!(stats.latest_confidence.is_none());

        // After assessment.
        db.metacognitive_assessment().unwrap();
        let stats = db.metacognitive_stats().unwrap();
        assert_eq!(stats.total_assessments, 1);
        assert!(stats.latest_confidence.is_some());
    }

    #[test]
    fn test_reset_metacognitive_history() {
        let db = test_db();

        db.metacognitive_assessment().unwrap();
        assert_eq!(db.metacognitive_stats().unwrap().total_assessments, 1);

        db.reset_metacognitive_history().unwrap();
        assert_eq!(db.metacognitive_stats().unwrap().total_assessments, 0);
    }
}
