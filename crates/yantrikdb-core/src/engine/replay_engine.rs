//! Engine-level experience replay / dream consolidation API.
//!
//! Wires the replay module into `YantrikDB`, persisting the replay
//! engine state (buffer, stats, budget) and exposing the dream cycle.

use crate::replay::{
    ReplayEngine, ReplaySummary, DreamReport, ActionRecord, OutcomeData,
    add_to_buffer, should_replay, reprioritize_buffer,
    run_replay_cycle, buffer_maintenance, replay_summary,
};
use crate::error::Result;
use crate::state::NodeId;

use super::YantrikDB;

const REPLAY_ENGINE_META_KEY: &str = "replay_engine";

impl YantrikDB {
    // ── Persistence ──

    /// Load the replay engine state from the database.
    pub fn load_replay_engine(&self) -> Result<ReplayEngine> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), REPLAY_ENGINE_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(ReplayEngine::new()),
        }
    }

    /// Persist the replay engine state.
    pub fn save_replay_engine(&self, engine: &ReplayEngine) -> Result<()> {
        let json = serde_json::to_string(engine).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![REPLAY_ENGINE_META_KEY, json],
        )?;
        Ok(())
    }

    // ── API ──

    /// Record a new experience into the replay buffer.
    ///
    /// Computes the TD error from expected vs actual utility and
    /// prioritizes the entry accordingly.
    pub fn record_experience(
        &self,
        episode_id: NodeId,
        expected_utility: f64,
        action: ActionRecord,
        outcome: OutcomeData,
        now_ms: u64,
    ) -> Result<()> {
        let mut engine = self.load_replay_engine()?;
        add_to_buffer(&mut engine, episode_id, expected_utility, action, outcome, now_ms);
        self.save_replay_engine(&engine)?;
        Ok(())
    }

    /// Check whether a replay cycle should run now.
    pub fn should_run_replay(&self, now_ms: u64) -> Result<bool> {
        let engine = self.load_replay_engine()?;
        Ok(should_replay(&engine, now_ms))
    }

    /// Run a full dream/replay cycle.
    ///
    /// Selects high-priority episodes, re-evaluates them against
    /// current beliefs, discovers cross-associations, and returns
    /// a consolidation report.
    pub fn run_dream_cycle(
        &self,
        current_beliefs: &[(NodeId, f64)],
        now_ms: u64,
    ) -> Result<DreamReport> {
        let mut engine = self.load_replay_engine()?;
        let report = run_replay_cycle(&mut engine, current_beliefs, now_ms);
        self.save_replay_engine(&engine)?;
        Ok(report)
    }

    /// Re-prioritize the replay buffer based on recency and TD error.
    pub fn reprioritize_replay_buffer(&self, now_ms: u64) -> Result<()> {
        let mut engine = self.load_replay_engine()?;
        reprioritize_buffer(&mut engine, now_ms);
        self.save_replay_engine(&engine)?;
        Ok(())
    }

    /// Run maintenance on the replay buffer (evict expired entries).
    ///
    /// Returns the number of entries removed.
    pub fn replay_buffer_maintenance(&self, now_ms: u64) -> Result<usize> {
        let mut engine = self.load_replay_engine()?;
        let removed = buffer_maintenance(&mut engine, now_ms);
        self.save_replay_engine(&engine)?;
        Ok(removed)
    }

    /// Get a summary of replay engine state.
    pub fn replay_engine_summary(&self) -> Result<ReplaySummary> {
        let engine = self.load_replay_engine()?;
        Ok(replay_summary(&engine))
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_replay_engine_save_load_roundtrip() {
        let db = test_db();
        let engine = db.load_replay_engine().unwrap();
        db.save_replay_engine(&engine).unwrap();
        let loaded = db.load_replay_engine().unwrap();
        let summary = crate::replay::replay_summary(&loaded);
        assert_eq!(summary.buffer_size, 0);
    }

    #[test]
    fn test_replay_engine_default_on_missing() {
        let db = test_db();
        let engine = db.load_replay_engine().unwrap();
        let summary = crate::replay::replay_summary(&engine);
        assert_eq!(summary.buffer_size, 0);
        assert_eq!(summary.total_replays, 0);
    }

    #[test]
    fn test_should_not_replay_empty() {
        let db = test_db();
        // Empty buffer → should not replay.
        let should = db.should_run_replay(1_000_000).unwrap();
        assert!(!should);
    }

    #[test]
    fn test_dream_cycle_empty() {
        let db = test_db();
        let report = db.run_dream_cycle(&[], 1_000_000).unwrap();
        assert_eq!(report.replays_executed, 0);
    }
}
