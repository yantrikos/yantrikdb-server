//! Engine-level autonomous skill acquisition API.
//!
//! Wires the skill acquisition module into `YantrikDB` for persistence
//! and integration with the event observer.

use crate::error::Result;
use crate::skills::{
    discover_skills, find_matching_skills, summarize_skills, DiscoveryResult, LearnedSkill,
    SkillConfig, SkillMatch, SkillOrigin, SkillRegistry, SkillSummary,
};

use super::{now, YantrikDB};

/// Meta key for persisted skill registry.
const SKILL_REGISTRY_META_KEY: &str = "skill_registry";
/// Meta key for persisted skill config.
const SKILL_CONFIG_META_KEY: &str = "skill_config";

impl YantrikDB {
    // ── Persistence ──

    /// Load the skill registry from the database.
    pub fn load_skill_registry(&self) -> Result<SkillRegistry> {
        match Self::get_meta(&self.conn(), SKILL_REGISTRY_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(SkillRegistry::new()),
        }
    }

    /// Persist the skill registry.
    pub fn save_skill_registry(&self, registry: &SkillRegistry) -> Result<()> {
        let json = serde_json::to_string(registry).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![SKILL_REGISTRY_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the skill configuration.
    pub fn load_skill_config(&self) -> Result<SkillConfig> {
        match Self::get_meta(&self.conn(), SKILL_CONFIG_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(SkillConfig::default()),
        }
    }

    /// Persist the skill configuration.
    pub fn save_skill_config(&self, config: &SkillConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![SKILL_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Main API ──

    /// Run the skill discovery pipeline.
    ///
    /// Loads the event buffer, runs sequence mining, updates the registry,
    /// and persists everything.
    pub fn discover_skills(&self) -> Result<DiscoveryResult> {
        let buffer = self.load_event_buffer()?;
        let mut registry = self.load_skill_registry()?;
        let config = self.load_skill_config()?;
        let ts = now();

        let result = discover_skills(&buffer, &mut registry, &config, ts);

        self.save_skill_registry(&registry)?;
        Ok(result)
    }

    /// Find skills relevant to the current context.
    pub fn find_relevant_skills(
        &self,
        hour: u8,
        day: u8,
        current_app: Option<u16>,
        recent_transition: Option<(u16, u16)>,
        session_duration_secs: f64,
        was_idle: bool,
        max_results: usize,
    ) -> Result<Vec<SkillMatch>> {
        let registry = self.load_skill_registry()?;
        let config = self.load_skill_config()?;
        Ok(find_matching_skills(
            &registry,
            hour,
            day,
            current_app,
            recent_transition,
            session_duration_secs,
            was_idle,
            config.min_offer_confidence,
            max_results,
        ))
    }

    /// Record that a skill was offered to the user.
    pub fn skill_offered(&self, dedup_key: &str) -> Result<bool> {
        let mut registry = self.load_skill_registry()?;
        if let Some(skill) = registry.find_mut(dedup_key) {
            skill.record_offer();
            self.save_skill_registry(&registry)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Record that the user accepted an offered skill.
    pub fn skill_accepted(&self, dedup_key: &str) -> Result<bool> {
        let mut registry = self.load_skill_registry()?;
        let ts = now();
        if let Some(skill) = registry.find_mut(dedup_key) {
            skill.record_acceptance(ts);
            self.save_skill_registry(&registry)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Record a successful skill execution.
    pub fn skill_succeeded(&self, dedup_key: &str) -> Result<bool> {
        let mut registry = self.load_skill_registry()?;
        let ts = now();
        if let Some(skill) = registry.find_mut(dedup_key) {
            skill.record_success(ts);
            self.save_skill_registry(&registry)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Record a failed skill execution.
    pub fn skill_failed(&self, dedup_key: &str) -> Result<bool> {
        let mut registry = self.load_skill_registry()?;
        let ts = now();
        if let Some(skill) = registry.find_mut(dedup_key) {
            skill.record_failure(ts);
            self.save_skill_registry(&registry)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Record that the user rejected an offered skill.
    pub fn skill_rejected(&self, dedup_key: &str) -> Result<bool> {
        let mut registry = self.load_skill_registry()?;
        let ts = now();
        if let Some(skill) = registry.find_mut(dedup_key) {
            skill.record_rejection(ts);
            self.save_skill_registry(&registry)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Manually deprecate a skill.
    pub fn deprecate_skill(&self, dedup_key: &str) -> Result<bool> {
        let mut registry = self.load_skill_registry()?;
        if let Some(skill) = registry.find_mut(dedup_key) {
            skill.deprecate();
            registry.total_deprecated += 1;
            self.save_skill_registry(&registry)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Manually teach a new skill.
    pub fn teach_skill(
        &self,
        description: String,
        dedup_key: String,
        steps: Vec<crate::skills::SkillStep>,
        trigger: crate::skills::SkillTrigger,
    ) -> Result<bool> {
        let mut registry = self.load_skill_registry()?;
        if registry.find(&dedup_key).is_some() {
            return Ok(false); // Already exists
        }

        let ts = now();
        let mut skill = LearnedSkill::new(
            description,
            SkillOrigin::UserTaught,
            dedup_key.clone(),
            steps,
            trigger,
            1,
            ts,
        );
        // User-taught skills start with higher confidence
        skill.confidence = 0.7;
        skill.stage = crate::skills::SkillStage::Reliable;

        registry.skills.insert(dedup_key, skill);
        registry.total_discovered += 1;
        self.save_skill_registry(&registry)?;
        Ok(true)
    }

    /// Get all skills by origin type.
    pub fn skills_by_origin(&self, origin: SkillOrigin) -> Result<Vec<LearnedSkill>> {
        let registry = self.load_skill_registry()?;
        Ok(registry.by_origin(origin).into_iter().cloned().collect())
    }

    /// Get a summary of the skill registry.
    pub fn skill_summary(&self) -> Result<SkillSummary> {
        let registry = self.load_skill_registry()?;
        Ok(summarize_skills(&registry))
    }

    /// Get skill registry statistics: (total, active, promoted).
    pub fn skill_stats(&self) -> Result<(usize, usize, u64)> {
        let registry = self.load_skill_registry()?;
        Ok((registry.len(), registry.active_count(), registry.total_promoted))
    }

    /// Reset the skill registry (clear all learned skills). Irreversible.
    pub fn reset_skill_registry(&self) -> Result<()> {
        let registry = SkillRegistry::new();
        self.save_skill_registry(&registry)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::observer::{SystemEvent, SystemEventData};
    use crate::skills::{SkillOrigin, SkillStep, SkillTrigger};

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    fn ts(offset: f64) -> f64 {
        86400.0 * 100.0 + offset
    }

    #[test]
    fn test_skill_discovery_from_app_sequences() {
        let db = test_db();

        // Seed observer with repeated app sequences
        let events: Vec<SystemEvent> = (0..5)
            .flat_map(|round| {
                let base = round as f64 * 100.0;
                vec![
                    SystemEvent::new(
                        ts(base + 1.0),
                        SystemEventData::AppSequence {
                            from_app: 14,
                            to_app: 20,
                            gap_ms: 2000,
                        },
                    ),
                    SystemEvent::new(
                        ts(base + 3.0),
                        SystemEventData::AppSequence {
                            from_app: 20,
                            to_app: 15,
                            gap_ms: 3000,
                        },
                    ),
                ]
            })
            .collect();
        db.observe_batch(events).unwrap();

        // Run discovery
        let result = db.discover_skills().unwrap();
        assert!(
            !result.new_skills.is_empty(),
            "Should discover app sequence skills from observer data"
        );

        // Verify persistence
        let (total, active, _) = db.skill_stats().unwrap();
        assert!(total > 0);
        assert!(active > 0);
    }

    #[test]
    fn test_skill_lifecycle_tracking() {
        let db = test_db();

        // Teach a skill manually
        let taught = db
            .teach_skill(
                "Morning workflow".to_string(),
                "manual:morning".to_string(),
                vec![SkillStep {
                    ordinal: 0,
                    action_kind: "app_open".to_string(),
                    app_id: Some(14),
                    tool_name: None,
                    expected_duration_ms: 0,
                    optional: false,
                }],
                SkillTrigger::new(),
            )
            .unwrap();
        assert!(taught);

        // Duplicate should fail
        let dup = db
            .teach_skill(
                "Same".to_string(),
                "manual:morning".to_string(),
                vec![],
                SkillTrigger::new(),
            )
            .unwrap();
        assert!(!dup);

        // Track lifecycle
        assert!(db.skill_offered("manual:morning").unwrap());
        assert!(db.skill_accepted("manual:morning").unwrap());
        assert!(db.skill_succeeded("manual:morning").unwrap());

        // Verify stats
        let (total, _, _) = db.skill_stats().unwrap();
        assert_eq!(total, 1);
    }

    #[test]
    fn test_skill_rejection_and_deprecation() {
        let db = test_db();

        db.teach_skill(
            "Bad skill".to_string(),
            "manual:bad".to_string(),
            vec![],
            SkillTrigger::new(),
        )
        .unwrap();

        // Reject it
        assert!(db.skill_rejected("manual:bad").unwrap());

        // Deprecate it
        assert!(db.deprecate_skill("manual:bad").unwrap());

        // Non-existent skill operations
        assert!(!db.skill_offered("nonexistent").unwrap());
        assert!(!db.deprecate_skill("nonexistent").unwrap());
    }

    #[test]
    fn test_skill_summary() {
        let db = test_db();

        db.teach_skill(
            "Skill A".to_string(),
            "a".to_string(),
            vec![],
            SkillTrigger::new(),
        )
        .unwrap();
        db.teach_skill(
            "Skill B".to_string(),
            "b".to_string(),
            vec![],
            SkillTrigger::new(),
        )
        .unwrap();

        let summary = db.skill_summary().unwrap();
        assert_eq!(summary.total, 2);
        assert_eq!(summary.active, 2);
        assert_eq!(summary.reliable, 2); // User-taught start at Reliable
    }

    #[test]
    fn test_skills_by_origin() {
        let db = test_db();

        db.teach_skill(
            "Taught".to_string(),
            "taught:1".to_string(),
            vec![],
            SkillTrigger::new(),
        )
        .unwrap();

        let taught = db.skills_by_origin(SkillOrigin::UserTaught).unwrap();
        assert_eq!(taught.len(), 1);

        let auto = db.skills_by_origin(SkillOrigin::AppSequence).unwrap();
        assert!(auto.is_empty());
    }

    #[test]
    fn test_reset_skill_registry() {
        let db = test_db();

        db.teach_skill(
            "Will be reset".to_string(),
            "reset:1".to_string(),
            vec![],
            SkillTrigger::new(),
        )
        .unwrap();

        db.reset_skill_registry().unwrap();
        let (total, _, _) = db.skill_stats().unwrap();
        assert_eq!(total, 0);
    }

    #[test]
    fn test_find_relevant_skills() {
        let db = test_db();

        db.teach_skill(
            "Morning email check".to_string(),
            "taught:email".to_string(),
            vec![SkillStep {
                ordinal: 0,
                action_kind: "open_email".to_string(),
                app_id: Some(17),
                tool_name: None,
                expected_duration_ms: 0,
                optional: false,
            }],
            SkillTrigger {
                time_bins: vec![8, 9, 10],
                initiating_app_id: None,
                preceding_transition: None,
                day_of_week: vec![],
                min_session_duration_secs: None,
                preceded_by_idle: None,
            },
        )
        .unwrap();

        // Query in the morning
        let matches = db
            .find_relevant_skills(9, 2, None, None, 3600.0, false, 5)
            .unwrap();
        assert!(!matches.is_empty(), "Should find morning skill");

        // Query at night — should have lower relevance
        let night_matches = db
            .find_relevant_skills(23, 2, None, None, 3600.0, false, 5)
            .unwrap();
        // Night matches may or may not be empty depending on min_offer_confidence threshold
        if !night_matches.is_empty() {
            assert!(
                night_matches[0].relevance < matches[0].relevance,
                "Night relevance should be lower"
            );
        }
    }
}
