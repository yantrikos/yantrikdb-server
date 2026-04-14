//! Engine-level cognitive tick API.
//!
//! Wires the cognitive tick heartbeat into `YantrikDB` for persistence
//! and integration with the agenda, Hawkes registry, and cognitive graph.

use crate::error::Result;
use crate::tick::{
    cognitive_tick, next_tick_interval_ms, Anomaly, CachedSuggestion,
    TickConfig, TickReport, TickState,
};

use super::{now, YantrikDB};

/// Meta key for persisted tick state.
const TICK_STATE_META_KEY: &str = "cognitive_tick_state";
/// Meta key for persisted tick config.
const TICK_CONFIG_META_KEY: &str = "cognitive_tick_config";

impl YantrikDB {
    // ── Persistence ──

    /// Load the tick state from the database (or create default).
    pub fn load_tick_state(&self) -> Result<TickState> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), TICK_STATE_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(TickState::new()),
        }
    }

    /// Persist the tick state to the database.
    pub fn save_tick_state(&self, state: &TickState) -> Result<()> {
        let json = serde_json::to_string(state).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![TICK_STATE_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the tick config from the database (or create default).
    pub fn load_tick_config(&self) -> Result<TickConfig> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), TICK_CONFIG_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(TickConfig::default()),
        }
    }

    /// Persist the tick config to the database.
    pub fn save_tick_config(&self, config: &TickConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![TICK_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Tick Execution ──

    /// Execute one cognitive tick.
    ///
    /// Loads all required state from the database, runs the tick,
    /// and persists the updated state. This is the main entry point
    /// for the background heartbeat.
    pub fn cognitive_tick(&self) -> Result<TickReport> {
        let mut tick_state = self.load_tick_state()?;
        let tick_config = self.load_tick_config()?;
        let mut agenda = self.load_agenda()?;
        let hawkes_registry = self.load_hawkes_registry()?;

        // Load active cognitive nodes for decay and anomaly detection
        let mut nodes = self.load_active_cognitive_nodes()?;

        let ts = now();
        let report = cognitive_tick(
            ts,
            &mut tick_state,
            &mut agenda,
            &mut nodes,
            &hawkes_registry,
            &tick_config,
        );

        // Persist updated state
        self.save_tick_state(&tick_state)?;
        self.save_agenda(&agenda)?;

        // Persist decayed nodes back to DB
        for node in &nodes {
            self.persist_cognitive_node(node)?;
        }

        // If consolidation was flagged, run it at the engine level
        if report.consolidation_ran {
            let _ = self.run_consolidation();
        }

        Ok(report)
    }

    /// Get the optimal interval until the next tick (milliseconds).
    pub fn next_tick_interval(&self) -> Result<u64> {
        let tick_state = self.load_tick_state()?;
        let agenda = self.load_agenda()?;
        let ts = now();
        Ok(next_tick_interval_ms(&tick_state, &agenda, ts))
    }

    // ── Query ──

    /// Get current active anomalies from the last tick.
    pub fn tick_anomalies(&self) -> Result<Vec<Anomaly>> {
        let state = self.load_tick_state()?;
        Ok(state.active_anomalies)
    }

    /// Get cached proactive suggestions from the last tick.
    pub fn tick_cached_suggestions(&self) -> Result<Vec<CachedSuggestion>> {
        let state = self.load_tick_state()?;
        let ts = now();
        Ok(state.cached_suggestions.into_iter().filter(|s| s.is_valid(ts)).collect())
    }

    /// Get the current tick count.
    pub fn tick_count(&self) -> Result<u64> {
        let state = self.load_tick_state()?;
        Ok(state.tick_count)
    }

    // ── Helpers ──

    /// Load cognitive nodes that are "active" (activation > threshold).
    ///
    /// Used by the tick to decay activations and detect anomalies.
    fn load_active_cognitive_nodes(&self) -> Result<Vec<crate::state::CognitiveNode>> {
        use crate::state::NodeKind;

        let mut nodes = Vec::new();
        for kind in &[
            NodeKind::Task,
            NodeKind::Goal,
            NodeKind::Routine,
            NodeKind::Belief,
            NodeKind::Episode,
        ] {
            nodes.extend(self.load_cognitive_nodes_by_kind(*kind)?);
        }
        Ok(nodes)
    }

    /// Run memory consolidation (engine-level).
    ///
    /// Wraps the consolidation module — finds candidates and consolidates.
    fn run_consolidation(&self) -> Result<()> {
        let _ = crate::consolidate::consolidate(
            self,
            0.85,  // similarity threshold
            30.0,  // time window days
            2,     // min cluster size
            100,   // consolidation limit
            true,  // require entity overlap (guards cosine-only false merges)
            false, // not dry run
        )?;
        Ok(())
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::tick::{TickConfig, TickPhase, TickState};

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_tick_state_persistence_roundtrip() {
        let db = test_db();

        let mut state = TickState::new();
        state.tick_count = 42;
        state.last_urgency_at = 1_700_000_000.0;
        db.save_tick_state(&state).unwrap();

        let loaded = db.load_tick_state().unwrap();
        assert_eq!(loaded.tick_count, 42);
        assert!((loaded.last_urgency_at - 1_700_000_000.0).abs() < 0.001);
    }

    #[test]
    fn test_tick_config_persistence_roundtrip() {
        let db = test_db();

        let mut config = TickConfig::default();
        config.budget_us = 10_000;
        config.routine_check_interval = 20;
        db.save_tick_config(&config).unwrap();

        let loaded = db.load_tick_config().unwrap();
        assert_eq!(loaded.budget_us, 10_000);
        assert_eq!(loaded.routine_check_interval, 20);
    }

    #[test]
    fn test_cognitive_tick_execution() {
        let db = test_db();

        let report = db.cognitive_tick().unwrap();
        assert_eq!(report.tick_number, 0);
        assert!(report.phases_executed.contains(&TickPhase::UrgencyRefresh));
        assert!(report.phases_executed.contains(&TickPhase::ActivationDecay));

        // Verify tick count incremented
        assert_eq!(db.tick_count().unwrap(), 1);
    }

    #[test]
    fn test_multiple_ticks() {
        let db = test_db();

        for _ in 0..5 {
            db.cognitive_tick().unwrap();
        }

        assert_eq!(db.tick_count().unwrap(), 5);
    }

    #[test]
    fn test_next_tick_interval() {
        let db = test_db();

        // Set tick state to idle (long elapsed time since last tick)
        let mut state = TickState::new();
        state.last_tick_elapsed_secs = 10.0; // Idle
        db.save_tick_state(&state).unwrap();

        let interval = db.next_tick_interval().unwrap();
        assert_eq!(interval, 5000); // 5s for idle
    }

    #[test]
    fn test_tick_anomalies_empty_initially() {
        let db = test_db();
        let anomalies = db.tick_anomalies().unwrap();
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_tick_cached_suggestions_empty_initially() {
        let db = test_db();
        let suggestions = db.tick_cached_suggestions().unwrap();
        assert!(suggestions.is_empty());
    }
}
