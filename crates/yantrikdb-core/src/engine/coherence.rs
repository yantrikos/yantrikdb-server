//! Engine-level coherence monitoring API.
//!
//! Wires the coherence monitor into `YantrikDB` for persistence
//! and integration with the cognitive graph, working set, and
//! contradiction detection subsystems.

use crate::attention::AttentionConfig;
use crate::coherence::{
    check_coherence, plan_enforcement, CoherenceConfig, CoherenceHistory,
    CoherenceInputs, CoherenceReport, EnforcementReport,
};
use crate::contradiction::ContradictionConfig;
use crate::error::Result;
use crate::state::{CognitiveEdge, CognitiveEdgeKind};

use super::{now, YantrikDB};

/// Meta key for persisted coherence config.
const COHERENCE_CONFIG_META_KEY: &str = "coherence_config";
/// Meta key for persisted coherence history.
const COHERENCE_HISTORY_META_KEY: &str = "coherence_history";

/// Maximum snapshots retained in history.
const DEFAULT_MAX_SNAPSHOTS: usize = 100;

impl YantrikDB {
    // ── Persistence ──

    /// Load the coherence config from the database.
    pub fn load_coherence_config(&self) -> Result<CoherenceConfig> {
        match Self::get_meta(&self.conn(), COHERENCE_CONFIG_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(CoherenceConfig::default()),
        }
    }

    /// Persist the coherence config.
    pub fn save_coherence_config(&self, config: &CoherenceConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![COHERENCE_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the coherence history from the database.
    pub fn load_coherence_history(&self) -> Result<CoherenceHistory> {
        match Self::get_meta(&self.conn(), COHERENCE_HISTORY_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(CoherenceHistory::new(DEFAULT_MAX_SNAPSHOTS)),
        }
    }

    /// Persist the coherence history.
    pub fn save_coherence_history(&self, history: &CoherenceHistory) -> Result<()> {
        let json = serde_json::to_string(history).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![COHERENCE_HISTORY_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Coherence API ──

    /// Run a full coherence check on the current cognitive state.
    ///
    /// Hydrates the working set, loads relevant edges and belief
    /// contradictions, then delegates to the pure coherence checker.
    /// The result is recorded in the coherence history.
    pub fn check_coherence(
        &self,
        attention_config: &AttentionConfig,
    ) -> Result<CoherenceReport> {
        let config = self.load_coherence_config()?;
        let ws = self.hydrate_working_set(attention_config.clone())?;
        let edges = self.load_coherence_edges()?;

        // Scan for belief contradictions using default config.
        let contradiction_config = ContradictionConfig::default();
        let scan = self.detect_belief_contradictions(&contradiction_config)?;

        let inputs = CoherenceInputs {
            working_set: &ws,
            edges: &edges,
            belief_conflicts: &scan.conflicts,
            config: &config,
            now: now(),
        };

        let report = check_coherence(&inputs);

        // Record in history.
        let mut history = self.load_coherence_history()?;
        history.record(&report);
        self.save_coherence_history(&history)?;

        Ok(report)
    }

    /// Run coherence check and generate enforcement actions.
    ///
    /// Returns both the report and the enforcement plan. The caller
    /// is responsible for applying the enforcement actions to the
    /// working set (e.g., demoting zombie nodes, pruning low-salience).
    pub fn check_and_enforce_coherence(
        &self,
        attention_config: &AttentionConfig,
    ) -> Result<(CoherenceReport, EnforcementReport)> {
        let config = self.load_coherence_config()?;
        let ws = self.hydrate_working_set(attention_config.clone())?;
        let edges = self.load_coherence_edges()?;

        let contradiction_config = ContradictionConfig::default();
        let scan = self.detect_belief_contradictions(&contradiction_config)?;

        let inputs = CoherenceInputs {
            working_set: &ws,
            edges: &edges,
            belief_conflicts: &scan.conflicts,
            config: &config,
            now: now(),
        };

        let report = check_coherence(&inputs);
        let enforcement = plan_enforcement(&report, &ws, &config);

        // Record in history.
        let mut history = self.load_coherence_history()?;
        history.record(&report);
        history.record_enforcement(enforcement.emergency_triggered);
        self.save_coherence_history(&history)?;

        Ok((report, enforcement))
    }

    /// Get the latest coherence score without running a full check.
    ///
    /// Returns `None` if no checks have been run yet.
    pub fn latest_coherence_score(&self) -> Result<Option<f64>> {
        let history = self.load_coherence_history()?;
        Ok(history.latest().map(|s| s.score))
    }

    /// Get the coherence trend over the last N checks.
    ///
    /// Positive = improving, negative = degrading, 0 = stable.
    pub fn coherence_trend(&self, lookback: usize) -> Result<f64> {
        let history = self.load_coherence_history()?;
        Ok(history.trend(lookback))
    }

    /// Get the average coherence score over the last N checks.
    pub fn coherence_average(&self, lookback: usize) -> Result<f64> {
        let history = self.load_coherence_history()?;
        Ok(history.recent_average(lookback))
    }

    /// Get coherence statistics.
    pub fn coherence_stats(&self) -> Result<CoherenceStats> {
        let history = self.load_coherence_history()?;
        let latest = history.latest().cloned();
        Ok(CoherenceStats {
            total_checks: history.total_checks,
            total_enforcements: history.total_enforcements,
            total_emergency_triggers: history.total_emergency_triggers,
            snapshot_count: history.snapshot_count(),
            latest_score: latest.as_ref().map(|s| s.score),
            latest_load: latest.as_ref().map(|s| s.cognitive_load),
            trend_10: history.trend(10),
        })
    }

    /// Reset the coherence history.
    pub fn reset_coherence_history(&self) -> Result<()> {
        self.save_coherence_history(&CoherenceHistory::new(DEFAULT_MAX_SNAPSHOTS))
    }

    // ── Internal ──

    /// Load all edge kinds relevant to coherence analysis.
    fn load_coherence_edges(&self) -> Result<Vec<CognitiveEdge>> {
        let mut edges = Vec::new();
        for kind in [
            CognitiveEdgeKind::AdvancesGoal,
            CognitiveEdgeKind::BlocksGoal,
            CognitiveEdgeKind::SubtaskOf,
            CognitiveEdgeKind::Requires,
            CognitiveEdgeKind::Causes,
            CognitiveEdgeKind::Prevents,
            CognitiveEdgeKind::Contradicts,
        ] {
            edges.extend(self.load_cognitive_edges_by_kind(kind)?);
        }
        Ok(edges)
    }
}

/// Compact coherence statistics.
#[derive(Debug, Clone)]
pub struct CoherenceStats {
    pub total_checks: u64,
    pub total_enforcements: u64,
    pub total_emergency_triggers: u64,
    pub snapshot_count: usize,
    pub latest_score: Option<f64>,
    pub latest_load: Option<f64>,
    pub trend_10: f64,
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::attention::AttentionConfig;
    use crate::coherence::{CoherenceConfig, CoherenceHistory};

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    fn default_attention() -> AttentionConfig {
        AttentionConfig {
            capacity: 20,
            max_hops: 2,
            top_k_per_hop: 5,
            hop_decay: 0.5,
            activation_threshold: 0.1,
            lateral_inhibition: 0.3,
            insertion_boost: 0.1,
        }
    }

    #[test]
    fn test_save_load_coherence_config() {
        let db = test_db();

        let mut config = CoherenceConfig::default();
        config.stale_threshold_secs = 7200.0;
        config.emergency_threshold = 0.2;
        db.save_coherence_config(&config).unwrap();

        let loaded = db.load_coherence_config().unwrap();
        assert!((loaded.stale_threshold_secs - 7200.0).abs() < f64::EPSILON);
        assert!((loaded.emergency_threshold - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_save_load_coherence_history() {
        let db = test_db();

        let mut history = CoherenceHistory::new(50);
        history.total_checks = 15;
        history.total_enforcements = 3;
        history.total_emergency_triggers = 1;
        db.save_coherence_history(&history).unwrap();

        let loaded = db.load_coherence_history().unwrap();
        assert_eq!(loaded.total_checks, 15);
        assert_eq!(loaded.total_enforcements, 3);
        assert_eq!(loaded.total_emergency_triggers, 1);
    }

    #[test]
    fn test_check_coherence_empty() {
        let db = test_db();
        let config = default_attention();
        let report = db.check_coherence(&config).unwrap();

        // Empty DB → no conflicts, high coherence.
        assert!(report.goal_conflicts.is_empty());
        assert!(report.belief_contradictions.is_empty());
        assert!(report.stale_activations.is_empty());
        assert!(report.coherence_score > 0.9);
    }

    #[test]
    fn test_check_and_enforce_empty() {
        let db = test_db();
        let config = default_attention();
        let (report, enforcement) = db.check_and_enforce_coherence(&config).unwrap();

        assert!(report.coherence_score > 0.9);
        assert_eq!(enforcement.items_affected, 0);
        assert!(!enforcement.emergency_triggered);
    }

    #[test]
    fn test_coherence_stats_empty() {
        let db = test_db();
        let stats = db.coherence_stats().unwrap();

        assert_eq!(stats.total_checks, 0);
        assert_eq!(stats.total_enforcements, 0);
        assert!(stats.latest_score.is_none());
    }

    #[test]
    fn test_coherence_stats_after_check() {
        let db = test_db();
        let config = default_attention();

        db.check_coherence(&config).unwrap();

        let stats = db.coherence_stats().unwrap();
        assert_eq!(stats.total_checks, 1);
        assert!(stats.latest_score.is_some());
        assert!(stats.latest_score.unwrap() > 0.9);
    }

    #[test]
    fn test_coherence_trend() {
        let db = test_db();
        let config = default_attention();

        // Run multiple checks.
        for _ in 0..3 {
            db.check_coherence(&config).unwrap();
        }

        let trend = db.coherence_trend(3).unwrap();
        // All checks are similar → trend ≈ 0.
        assert!(trend.abs() < 0.1);

        let avg = db.coherence_average(3).unwrap();
        assert!(avg > 0.9);
    }

    #[test]
    fn test_reset_coherence_history() {
        let db = test_db();
        let config = default_attention();

        db.check_coherence(&config).unwrap();

        let stats = db.coherence_stats().unwrap();
        assert_eq!(stats.total_checks, 1);

        db.reset_coherence_history().unwrap();

        let stats = db.coherence_stats().unwrap();
        assert_eq!(stats.total_checks, 0);
        assert!(stats.latest_score.is_none());
    }

    #[test]
    fn test_latest_coherence_score() {
        let db = test_db();
        let config = default_attention();

        assert!(db.latest_coherence_score().unwrap().is_none());

        db.check_coherence(&config).unwrap();

        let score = db.latest_coherence_score().unwrap().unwrap();
        assert!(score > 0.9);
    }
}
