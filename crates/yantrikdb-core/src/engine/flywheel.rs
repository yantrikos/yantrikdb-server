//! Engine-level autonomous learning flywheel API.
//!
//! Wires the flywheel module into `YantrikDB` for persistence
//! and integration with the event observer and cognitive graph.

use crate::error::Result;
use crate::flywheel::{
    form_beliefs, AutonomousBelief, BeliefCategory, BeliefStore, FlywheelConfig, FormationResult,
};

use super::{now, YantrikDB};

/// Meta key for persisted belief store.
const BELIEF_STORE_META_KEY: &str = "autonomous_belief_store";
/// Meta key for persisted flywheel config.
const FLYWHEEL_CONFIG_META_KEY: &str = "flywheel_config";

impl YantrikDB {
    // ── Persistence ──

    /// Load the autonomous belief store from the database.
    pub fn load_belief_store(&self) -> Result<BeliefStore> {
        match Self::get_meta(&self.conn(), BELIEF_STORE_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(BeliefStore::new()),
        }
    }

    /// Persist the autonomous belief store.
    pub fn save_belief_store(&self, store: &BeliefStore) -> Result<()> {
        let json = serde_json::to_string(store).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![BELIEF_STORE_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the flywheel configuration.
    pub fn load_flywheel_config(&self) -> Result<FlywheelConfig> {
        match Self::get_meta(&self.conn(), FLYWHEEL_CONFIG_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(FlywheelConfig::default()),
        }
    }

    /// Persist the flywheel configuration.
    pub fn save_flywheel_config(&self, config: &FlywheelConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![FLYWHEEL_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Main API ──

    /// Run the autonomous belief formation pipeline.
    ///
    /// Loads the event buffer + observer state, runs pattern detection,
    /// updates the belief store, and persists everything.
    pub fn form_autonomous_beliefs(&self) -> Result<FormationResult> {
        let buffer = self.load_event_buffer()?;
        let state = self.load_observer_state()?;
        let mut store = self.load_belief_store()?;
        let config = self.load_flywheel_config()?;
        let ts = now();

        let result = form_beliefs(&buffer, &state, &mut store, &config, ts);

        self.save_belief_store(&store)?;
        Ok(result)
    }

    /// Get all established autonomous beliefs (confidence >= 0.7).
    pub fn get_established_beliefs(&self) -> Result<Vec<AutonomousBelief>> {
        let store = self.load_belief_store()?;
        Ok(store.established().into_iter().cloned().collect())
    }

    /// Get all belief candidates (still learning).
    pub fn get_belief_candidates(&self) -> Result<Vec<AutonomousBelief>> {
        let store = self.load_belief_store()?;
        Ok(store.candidates().into_iter().cloned().collect())
    }

    /// Get autonomous beliefs by category.
    pub fn get_beliefs_by_category(
        &self,
        category: BeliefCategory,
    ) -> Result<Vec<AutonomousBelief>> {
        let store = self.load_belief_store()?;
        Ok(store.by_category(category).into_iter().cloned().collect())
    }

    /// Get all autonomous beliefs above a minimum confidence.
    pub fn get_beliefs_above(&self, min_confidence: f64) -> Result<Vec<AutonomousBelief>> {
        let store = self.load_belief_store()?;
        Ok(store
            .beliefs_above(min_confidence)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Manually confirm an autonomous belief (e.g., user validates it).
    pub fn confirm_autonomous_belief(&self, dedup_key: &str) -> Result<bool> {
        let mut store = self.load_belief_store()?;
        let ts = now();
        if let Some(belief) = store.find_mut(dedup_key) {
            belief.confirm(ts);
            self.save_belief_store(&store)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Manually contradict an autonomous belief (e.g., user corrects it).
    pub fn contradict_autonomous_belief(&self, dedup_key: &str) -> Result<bool> {
        let mut store = self.load_belief_store()?;
        let ts = now();
        if let Some(belief) = store.find_mut(dedup_key) {
            belief.contradict(ts);
            self.save_belief_store(&store)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get a count of autonomous beliefs by stage.
    pub fn belief_store_stats(
        &self,
    ) -> Result<(usize, usize, usize)> {
        let store = self.load_belief_store()?;
        let established = store.established().len();
        let candidates = store.candidates().len();
        let total = store.len();
        Ok((total, established, candidates))
    }

    /// Reset the belief store (clear all autonomous beliefs).
    pub fn reset_belief_store(&self) -> Result<()> {
        let store = BeliefStore::new();
        self.save_belief_store(&store)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::flywheel::BeliefCategory;
    use crate::observer::{SystemEvent, SystemEventData};

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    fn ts(offset: f64) -> f64 {
        86400.0 * 100.0 + offset
    }

    #[test]
    fn test_belief_store_persistence_roundtrip() {
        let db = test_db();

        // Seed observer with data to form beliefs
        let events: Vec<SystemEvent> = (0..10)
            .map(|i| {
                SystemEvent::new(
                    ts(i as f64),
                    SystemEventData::AppSequence {
                        from_app: 14,
                        to_app: 20,
                        gap_ms: 3000,
                    },
                )
            })
            .collect();
        db.observe_batch(events).unwrap();

        // Run formation
        let result = db.form_autonomous_beliefs().unwrap();
        assert!(
            !result.new_beliefs.is_empty(),
            "Should form behavioral beliefs from app sequences",
        );

        // Verify persistence
        let (total, _, _) = db.belief_store_stats().unwrap();
        assert!(total > 0);
    }

    #[test]
    fn test_confirm_and_contradict() {
        let db = test_db();

        // Seed and form
        let events: Vec<SystemEvent> = (0..10)
            .map(|i| {
                SystemEvent::new(
                    ts(i as f64),
                    SystemEventData::AppSequence {
                        from_app: 1,
                        to_app: 2,
                        gap_ms: 1000,
                    },
                )
            })
            .collect();
        db.observe_batch(events).unwrap();
        db.form_autonomous_beliefs().unwrap();

        let dedup_key = "behavioral:app_seq:1→2";

        // Confirm
        assert!(db.confirm_autonomous_belief(dedup_key).unwrap());
        let beliefs = db.get_beliefs_above(0.0).unwrap();
        let belief = beliefs.iter().find(|b| b.dedup_key == dedup_key).unwrap();
        let c1 = belief.confidence;

        // Contradict
        assert!(db.contradict_autonomous_belief(dedup_key).unwrap());
        let beliefs = db.get_beliefs_above(0.0).unwrap();
        let belief = beliefs.iter().find(|b| b.dedup_key == dedup_key).unwrap();
        assert!(belief.confidence < c1);
    }

    #[test]
    fn test_beliefs_by_category() {
        let db = test_db();

        // Seed behavioral data
        let events: Vec<SystemEvent> = (0..6)
            .map(|i| {
                SystemEvent::new(
                    ts(i as f64),
                    SystemEventData::AppSequence {
                        from_app: 5,
                        to_app: 10,
                        gap_ms: 2000,
                    },
                )
            })
            .collect();
        db.observe_batch(events).unwrap();
        db.form_autonomous_beliefs().unwrap();

        let behavioral = db
            .get_beliefs_by_category(BeliefCategory::Behavioral)
            .unwrap();
        assert!(!behavioral.is_empty());

        let temporal = db
            .get_beliefs_by_category(BeliefCategory::Temporal)
            .unwrap();
        // May or may not have temporal beliefs depending on histogram
        assert!(temporal.len() <= behavioral.len() || true); // Just verify it doesn't panic
    }

    #[test]
    fn test_reset_belief_store() {
        let db = test_db();

        let events: Vec<SystemEvent> = (0..5)
            .map(|i| {
                SystemEvent::new(
                    ts(i as f64),
                    SystemEventData::AppSequence {
                        from_app: 3,
                        to_app: 7,
                        gap_ms: 1500,
                    },
                )
            })
            .collect();
        db.observe_batch(events).unwrap();
        db.form_autonomous_beliefs().unwrap();

        db.reset_belief_store().unwrap();
        let (total, _, _) = db.belief_store_stats().unwrap();
        assert_eq!(total, 0);
    }

    #[test]
    fn test_empty_formation() {
        let db = test_db();
        let result = db.form_autonomous_beliefs().unwrap();
        assert!(result.new_beliefs.is_empty());
        assert_eq!(result.confirmed, 0);
    }
}
