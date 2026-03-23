//! Engine-level self-experimentation API.
//!
//! Wires the experimenter into `YantrikDB` for persistence.

use crate::error::Result;
use crate::experimenter::{
    assign_variant_round_robin, check_experiments, conclude_experiment,
    design_experiment, record_trial, ExperimentId, ExperimentRegistry,
    ExperimentVariable, SafetyBound, TrialOutcome, VariantValue,
};

use super::{now, YantrikDB};

/// Meta key for persisted experiment registry.
const EXPERIMENT_REGISTRY_META_KEY: &str = "experiment_registry";

impl YantrikDB {
    // ── Persistence ──

    /// Load the experiment registry.
    pub fn load_experiment_registry(&self) -> Result<ExperimentRegistry> {
        match Self::get_meta(&self.conn(), EXPERIMENT_REGISTRY_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(ExperimentRegistry::new()),
        }
    }

    /// Persist the experiment registry.
    pub fn save_experiment_registry(&self, registry: &ExperimentRegistry) -> Result<()> {
        let json = serde_json::to_string(registry).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![EXPERIMENT_REGISTRY_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Main API ──

    /// Design and start a new experiment.
    pub fn start_experiment(
        &self,
        hypothesis: String,
        variable: ExperimentVariable,
        variants: Vec<VariantValue>,
        sample_size_target: u32,
        safety_bounds: Vec<SafetyBound>,
    ) -> Result<Option<ExperimentId>> {
        let mut registry = self.load_experiment_registry()?;
        let ts = now();
        let id = design_experiment(
            &mut registry,
            hypothesis,
            variable,
            variants,
            sample_size_target,
            safety_bounds,
            ts,
        );
        self.save_experiment_registry(&registry)?;
        Ok(id)
    }

    /// Get the next variant assignment for an experiment (round-robin).
    pub fn experiment_next_variant(
        &self,
        experiment_id: ExperimentId,
    ) -> Result<Option<(usize, VariantValue)>> {
        let mut registry = self.load_experiment_registry()?;
        let result = if let Some(exp) = registry.find_mut(experiment_id) {
            assign_variant_round_robin(exp).map(|(idx, v)| (idx, v.clone()))
        } else {
            None
        };
        self.save_experiment_registry(&registry)?;
        Ok(result)
    }

    /// Record a trial outcome for an experiment.
    ///
    /// Returns `true` if the experiment should continue.
    pub fn record_experiment_trial(
        &self,
        experiment_id: ExperimentId,
        variant_idx: usize,
        outcome: TrialOutcome,
    ) -> Result<bool> {
        let mut registry = self.load_experiment_registry()?;
        let ts = now();
        let should_continue = if let Some(exp) = registry.find_mut(experiment_id) {
            record_trial(exp, variant_idx, outcome, ts)
        } else {
            false
        };
        self.save_experiment_registry(&registry)?;
        Ok(should_continue)
    }

    /// Conclude an experiment manually.
    pub fn conclude_experiment(
        &self,
        experiment_id: ExperimentId,
    ) -> Result<Option<(usize, VariantValue)>> {
        let mut registry = self.load_experiment_registry()?;
        let ts = now();
        let result = if let Some(exp) = registry.find_mut(experiment_id) {
            conclude_experiment(exp, ts)
        } else {
            None
        };
        if result.is_some() {
            registry.total_concluded += 1;
        }
        self.save_experiment_registry(&registry)?;
        Ok(result)
    }

    /// Check all experiments for auto-conclusion or safety abort.
    pub fn check_experiments(&self) -> Result<Vec<ExperimentId>> {
        let mut registry = self.load_experiment_registry()?;
        let ts = now();
        let concluded = check_experiments(&mut registry, ts);
        self.save_experiment_registry(&registry)?;
        Ok(concluded)
    }

    /// Get experiment statistics.
    pub fn experiment_stats(&self) -> Result<(usize, u64, u64)> {
        let registry = self.load_experiment_registry()?;
        Ok((
            registry.active_count(),
            registry.total_concluded,
            registry.total_aborted,
        ))
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::experimenter::{
        ExperimentVariable, SafetyBound, TrialOutcome, VariantValue,
    };

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_experiment_lifecycle() {
        let db = test_db();

        let id = db
            .start_experiment(
                "Test lead time".to_string(),
                ExperimentVariable::ReminderLeadMinutes,
                vec![VariantValue::Number(15.0), VariantValue::Number(30.0)],
                3,
                vec![SafetyBound::MaxConsecutiveRejections(5)],
            )
            .unwrap()
            .unwrap();

        // Get assignments and record trials
        for _ in 0..3 {
            let (idx, _) = db.experiment_next_variant(id).unwrap().unwrap();
            db.record_experiment_trial(id, idx, TrialOutcome::Positive)
                .unwrap();
        }
        for _ in 0..3 {
            let (idx, _) = db.experiment_next_variant(id).unwrap().unwrap();
            db.record_experiment_trial(id, idx, TrialOutcome::Negative)
                .unwrap();
        }

        // Conclude
        let result = db.conclude_experiment(id).unwrap();
        assert!(result.is_some());

        let (_, concluded, _) = db.experiment_stats().unwrap();
        assert_eq!(concluded, 1);
    }

    #[test]
    fn test_experiment_safety_abort() {
        let db = test_db();

        let id = db
            .start_experiment(
                "Risky test".to_string(),
                ExperimentVariable::SurfacingFrequency,
                vec![VariantValue::Number(1.0), VariantValue::Number(5.0)],
                100,
                vec![SafetyBound::MaxConsecutiveRejections(3)],
            )
            .unwrap()
            .unwrap();

        // 3 consecutive rejections → abort
        db.record_experiment_trial(id, 0, TrialOutcome::Negative).unwrap();
        db.record_experiment_trial(id, 0, TrialOutcome::Negative).unwrap();
        let cont = db.record_experiment_trial(id, 0, TrialOutcome::Negative).unwrap();
        assert!(!cont, "Should signal stop after 3 rejections");

        let (active, _, aborted) = db.experiment_stats().unwrap();
        assert_eq!(active, 0);
        assert_eq!(aborted, 0); // Aborted inline, not via check_experiments
    }

    #[test]
    fn test_experiment_persistence() {
        let db = test_db();

        db.start_experiment(
            "Persist test".to_string(),
            ExperimentVariable::SurfacingMode,
            vec![
                VariantValue::Label("whisper".to_string()),
                VariantValue::Label("nudge".to_string()),
            ],
            5,
            vec![],
        )
        .unwrap();

        // Verify persistence
        let registry = db.load_experiment_registry().unwrap();
        assert_eq!(registry.experiments.len(), 1);
        assert_eq!(registry.total_created, 1);
    }
}
