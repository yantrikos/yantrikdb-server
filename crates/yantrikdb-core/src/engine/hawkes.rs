//! Engine-level Hawkes process API.
//!
//! Wires the Hawkes process routine prediction models into `YantrikDB` methods
//! that persist model state and feed from Episode/Routine nodes.

use crate::error::Result;
use crate::hawkes::{
    AnticipatedEvent, EventPrediction, HawkesRegistry, HawkesRegistryConfig,
    ModelSummary,
};
use crate::state::{NodeKind, NodePayload};

use super::{now, YantrikDB};

/// Meta key for persisted Hawkes registry.
const HAWKES_META_KEY: &str = "hawkes_registry";

impl YantrikDB {
    // ── Persistence ──

    /// Load the Hawkes registry from the database (or create a new one).
    pub fn load_hawkes_registry(&self) -> Result<HawkesRegistry> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), HAWKES_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(HawkesRegistry::new()),
        }
    }

    /// Persist the Hawkes registry to the database.
    pub fn save_hawkes_registry(&self, registry: &HawkesRegistry) -> Result<()> {
        let json = serde_json::to_string(registry).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![HAWKES_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Operations ──

    /// Record a new event observation and update the Hawkes model.
    pub fn hawkes_observe(&self, label: &str, timestamp: f64) -> Result<()> {
        let mut registry = self.load_hawkes_registry()?;
        registry.observe(label, timestamp);
        self.save_hawkes_registry(&registry)
    }

    /// Record a new event at the current time.
    pub fn hawkes_observe_now(&self, label: &str) -> Result<()> {
        self.hawkes_observe(label, now())
    }

    /// Train Hawkes models from all Episode nodes in the database.
    ///
    /// Groups episodes by summary prefix (first word) and batch-trains
    /// a model for each group. Replaces existing models.
    pub fn hawkes_train_from_episodes(&self) -> Result<HawkesRegistry> {
        let episodes = self.load_cognitive_nodes_by_kind(NodeKind::Episode)?;

        // Group episode timestamps by label
        let mut groups: std::collections::HashMap<String, Vec<f64>> =
            std::collections::HashMap::new();
        for ep_node in &episodes {
            if let NodePayload::Episode(ep) = &ep_node.payload {
                let label = ep
                    .summary
                    .split_whitespace()
                    .next()
                    .unwrap_or("unknown")
                    .to_lowercase();
                groups.entry(label).or_default().push(ep.occurred_at);
            }
        }

        let mut registry = HawkesRegistry::new();
        for (label, timestamps) in groups {
            if timestamps.len() >= 3 {
                registry.observe_batch(&label, &timestamps);
            }
        }

        self.save_hawkes_registry(&registry)?;
        Ok(registry)
    }

    /// Get predictions for all event types that should be anticipated now.
    pub fn hawkes_anticipate(&self) -> Result<Vec<AnticipatedEvent>> {
        let registry = self.load_hawkes_registry()?;
        let ts = now();
        Ok(registry.anticipate_all(ts))
    }

    /// Predict next occurrence for a specific event type.
    pub fn hawkes_predict(&self, label: &str) -> Result<Option<EventPrediction>> {
        let registry = self.load_hawkes_registry()?;
        let ts = now();
        Ok(registry.predict(label, ts))
    }

    /// Get summaries for all tracked Hawkes models.
    pub fn hawkes_summaries(&self) -> Result<Vec<ModelSummary>> {
        let registry = self.load_hawkes_registry()?;
        Ok(registry.summaries())
    }

    /// Get the number of tracked event types.
    pub fn hawkes_model_count(&self) -> Result<usize> {
        let registry = self.load_hawkes_registry()?;
        Ok(registry.model_count())
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::state::*;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    fn persist_episode(
        db: &YantrikDB,
        alloc: &mut NodeIdAllocator,
        summary: &str,
        occurred_at: f64,
    ) -> NodeId {
        let id = alloc.alloc(NodeKind::Episode);
        let node = CognitiveNode::new(
            id,
            summary.to_string(),
            NodePayload::Episode(EpisodePayload {
                memory_rid: format!("rid_{}", id.seq()),
                summary: summary.to_string(),
                occurred_at,
                participants: vec![],
            }),
        );
        db.persist_cognitive_node(&node).unwrap();
        id
    }

    #[test]
    fn test_hawkes_persistence_roundtrip() {
        let db = test_db();

        db.hawkes_observe("email_check", 1000.0).unwrap();
        db.hawkes_observe("email_check", 1600.0).unwrap();
        db.hawkes_observe("terminal_open", 1200.0).unwrap();

        let registry = db.load_hawkes_registry().unwrap();
        assert_eq!(registry.model_count(), 2);
        assert_eq!(
            registry.models.get("email_check").unwrap().total_observations,
            2
        );
    }

    #[test]
    fn test_hawkes_train_from_episodes() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        // Create regular email check episodes
        for i in 0..15 {
            let ts = 1_700_000_000.0 + i as f64 * 3600.0;
            persist_episode(&db, &mut alloc, "email check inbox", ts);
        }
        // Create some calendar events
        for i in 0..8 {
            let ts = 1_700_000_000.0 + i as f64 * 86400.0;
            persist_episode(&db, &mut alloc, "calendar review", ts);
        }
        db.persist_node_id_allocator(&alloc).unwrap();

        let registry = db.hawkes_train_from_episodes().unwrap();
        assert!(registry.model_count() >= 2);

        let summaries = db.hawkes_summaries().unwrap();
        assert!(summaries.len() >= 2);
        for s in &summaries {
            assert!(s.base_rate_per_hour > 0.0);
            assert!(s.branching_ratio < 1.0);
        }
    }

    #[test]
    fn test_hawkes_predict_after_training() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        // Regular events every 600s
        for i in 0..30 {
            let ts = 1_700_000_000.0 + i as f64 * 600.0;
            persist_episode(&db, &mut alloc, "standup meeting", ts);
        }
        db.persist_node_id_allocator(&alloc).unwrap();

        db.hawkes_train_from_episodes().unwrap();

        let registry = db.load_hawkes_registry().unwrap();
        let last_ts = 1_700_000_000.0 + 29.0 * 600.0;
        let pred = registry.predict("standup", last_ts + 60.0);
        assert!(pred.is_some(), "Should predict next standup");
        let pred = pred.unwrap();
        assert!(pred.time_until > 0.0);
    }

    #[test]
    fn test_hawkes_anticipate() {
        let db = test_db();

        // Create a burst of recent observations
        for i in 0..10 {
            db.hawkes_observe("active_task", 1_000_000.0 + i as f64 * 60.0)
                .unwrap();
        }

        let anticipated = db.hawkes_anticipate().unwrap();
        // Should not panic; results depend on intensity vs threshold
        for ae in &anticipated {
            assert!(ae.prediction.confidence >= 0.0);
        }
    }
}
