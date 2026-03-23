//! Engine-level counterfactual simulation API.
//!
//! Wires the counterfactual module into `YantrikDB`, loading
//! the causal store for simulation and persisting regret history.

use crate::counterfactual::{
    CounterfactualConfig, CounterfactualQuery, CounterfactualResult, DecisionRecord,
    RegretReport, SensitivityEntry,
    compare_alternatives, detect_regret_opportunities, sensitivity_analysis,
    simulate_counterfactual, why_not,
};
use crate::causal::CausalNode;
use crate::error::Result;

use super::YantrikDB;

const COUNTERFACTUAL_CONFIG_META_KEY: &str = "counterfactual_config";
const REGRET_HISTORY_META_KEY: &str = "regret_history";

impl YantrikDB {
    // ── Persistence ──

    /// Load counterfactual configuration.
    pub fn load_counterfactual_config(&self) -> Result<CounterfactualConfig> {
        match Self::get_meta(&self.conn(), COUNTERFACTUAL_CONFIG_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(CounterfactualConfig::default()),
        }
    }

    /// Persist counterfactual configuration.
    pub fn save_counterfactual_config(&self, config: &CounterfactualConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![COUNTERFACTUAL_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load regret history.
    pub fn load_regret_history(&self) -> Result<Vec<RegretReport>> {
        match Self::get_meta(&self.conn(), REGRET_HISTORY_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(Vec::new()),
        }
    }

    /// Persist regret history.
    pub fn save_regret_history(&self, history: &[RegretReport]) -> Result<()> {
        let json = serde_json::to_string(history).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![REGRET_HISTORY_META_KEY, json],
        )?;
        Ok(())
    }

    // ── API ──

    /// Run a counterfactual simulation using the persisted causal store.
    pub fn run_counterfactual(
        &self,
        query: &CounterfactualQuery,
    ) -> Result<CounterfactualResult> {
        let store = self.load_causal_store()?;
        let config = self.load_counterfactual_config()?;
        Ok(simulate_counterfactual(query, &store, &config))
    }

    /// Detect regret opportunities from recent decisions.
    pub fn detect_regrets(
        &self,
        decisions: &[DecisionRecord],
    ) -> Result<RegretReport> {
        let store = self.load_causal_store()?;
        let config = self.load_counterfactual_config()?;
        let report = detect_regret_opportunities(decisions, &store, &config);

        // Append to history.
        let mut history = self.load_regret_history()?;
        history.push(report.clone());
        // Keep last 50 reports.
        if history.len() > 50 {
            history.drain(0..history.len() - 50);
        }
        self.save_regret_history(&history)?;

        Ok(report)
    }

    /// Run why-not analysis.
    pub fn run_why_not(
        &self,
        desired_outcome: &CausalNode,
        actual_actions: &[CausalNode],
    ) -> Result<Vec<CounterfactualResult>> {
        let store = self.load_causal_store()?;
        let config = self.load_counterfactual_config()?;
        Ok(why_not(desired_outcome, actual_actions, &store, &config))
    }

    /// Run sensitivity analysis on a counterfactual query.
    pub fn run_sensitivity_analysis(
        &self,
        query: &CounterfactualQuery,
    ) -> Result<Vec<SensitivityEntry>> {
        let store = self.load_causal_store()?;
        let config = self.load_counterfactual_config()?;
        Ok(sensitivity_analysis(query, &store, &config))
    }

    /// Compare two alternative actions.
    pub fn compare_action_alternatives(
        &self,
        action_a: &CausalNode,
        action_b: &CausalNode,
    ) -> Result<f64> {
        let store = self.load_causal_store()?;
        let config = self.load_counterfactual_config()?;
        Ok(compare_alternatives(action_a, action_b, &store, &config))
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::counterfactual::CounterfactualConfig;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_counterfactual_config_save_load() {
        let db = test_db();
        let config = db.load_counterfactual_config().unwrap();
        assert_eq!(config.max_horizon, 5);
        db.save_counterfactual_config(&config).unwrap();
        let loaded = db.load_counterfactual_config().unwrap();
        assert_eq!(loaded.max_horizon, 5);
    }

    #[test]
    fn test_regret_history_empty() {
        let db = test_db();
        let history = db.load_regret_history().unwrap();
        assert!(history.is_empty());
    }

    #[test]
    fn test_regret_history_save_load() {
        let db = test_db();
        let history = vec![];
        db.save_regret_history(&history).unwrap();
        let loaded = db.load_regret_history().unwrap();
        assert!(loaded.is_empty());
    }
}
