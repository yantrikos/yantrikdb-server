//! Engine-level narrative memory API.
//!
//! Wires the narrative module into `YantrikDB` for persistence
//! of autobiographical timelines, arcs, and milestones.

use crate::error::Result;
use crate::narrative::{
    ArcAlert, ArcId, AutobiographicalTimeline, NarrativeEpisode,
    NarrativeQuery, NarrativeResult,
    arc_health_check, assign_to_arc, generate_arc_summary, query_timeline,
};

use super::YantrikDB;

const TIMELINE_META_KEY: &str = "autobiographical_timeline";

impl YantrikDB {
    // ── Persistence ──

    /// Load the autobiographical timeline.
    pub fn load_timeline(&self) -> Result<AutobiographicalTimeline> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), TIMELINE_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(AutobiographicalTimeline::default()),
        }
    }

    /// Persist the autobiographical timeline.
    pub fn save_timeline(&self, timeline: &AutobiographicalTimeline) -> Result<()> {
        let json = serde_json::to_string(timeline).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![TIMELINE_META_KEY, json],
        )?;
        Ok(())
    }

    // ── API ──

    /// Assign an episode to a narrative arc (or create a new one).
    pub fn assign_episode_to_arc(
        &self,
        episode: &NarrativeEpisode,
    ) -> Result<ArcId> {
        let mut timeline = self.load_timeline()?;
        let arc_id = assign_to_arc(episode, &mut timeline);
        self.save_timeline(&timeline)?;
        Ok(arc_id)
    }

    /// Query the narrative timeline.
    pub fn query_narrative(
        &self,
        query: &NarrativeQuery,
    ) -> Result<NarrativeResult> {
        let timeline = self.load_timeline()?;
        Ok(query_timeline(&timeline, query))
    }

    /// Run arc health check.
    pub fn arc_health_report(&self, now_ms: u64) -> Result<Vec<ArcAlert>> {
        let timeline = self.load_timeline()?;
        Ok(arc_health_check(&timeline, now_ms))
    }

    /// Generate a summary for a specific arc.
    pub fn generate_narrative_summary(&self, arc_id: ArcId) -> Result<Option<String>> {
        let timeline = self.load_timeline()?;
        let arc = timeline.arcs.iter().find(|a| a.id == arc_id);
        Ok(arc.map(generate_arc_summary))
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::narrative::NarrativeQuery;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_timeline_save_load_roundtrip() {
        let db = test_db();
        let timeline = db.load_timeline().unwrap();
        assert!(timeline.arcs.is_empty());
        db.save_timeline(&timeline).unwrap();
        let loaded = db.load_timeline().unwrap();
        assert!(loaded.arcs.is_empty());
    }

    #[test]
    fn test_timeline_default_on_missing() {
        let db = test_db();
        let timeline = db.load_timeline().unwrap();
        assert!(timeline.arcs.is_empty());
    }

    #[test]
    fn test_arc_health_empty() {
        let db = test_db();
        let alerts = db.arc_health_report(1000000).unwrap();
        assert!(alerts.is_empty());
    }
}
