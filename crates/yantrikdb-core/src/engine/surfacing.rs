//! Engine-level anticipatory surfacing API.
//!
//! Wires the surfacing pipeline into `YantrikDB` for persistence
//! and integration with the agenda, receptivity model, and preferences.

use crate::error::Result;
use crate::receptivity::ContextSnapshot;
use crate::surfacing::{
    run_surfacing_pipeline, run_surfacing_with_preferences,
    SurfaceMode, SurfaceOutcome, SurfaceRateLimiter,
    SurfacingConfig, SurfacingPreferences, SurfacingResult,
};

use super::{now, YantrikDB};

/// Meta key for persisted surfacing preferences.
const SURFACING_PREFS_META_KEY: &str = "surfacing_preferences";
/// Meta key for persisted rate limiter.
const SURFACING_RATE_META_KEY: &str = "surfacing_rate_limiter";
/// Meta key for persisted surfacing config.
const SURFACING_CONFIG_META_KEY: &str = "surfacing_config";

impl YantrikDB {
    // ── Persistence ──

    /// Load surfacing preferences from the database.
    pub fn load_surfacing_preferences(&self) -> Result<SurfacingPreferences> {
        match Self::get_meta(&self.conn(), SURFACING_PREFS_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(SurfacingPreferences::new()),
        }
    }

    /// Persist surfacing preferences.
    pub fn save_surfacing_preferences(&self, prefs: &SurfacingPreferences) -> Result<()> {
        let json = serde_json::to_string(prefs).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![SURFACING_PREFS_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the rate limiter state.
    pub fn load_surface_rate_limiter(&self) -> Result<SurfaceRateLimiter> {
        match Self::get_meta(&self.conn(), SURFACING_RATE_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(SurfaceRateLimiter::new()),
        }
    }

    /// Persist the rate limiter state.
    pub fn save_surface_rate_limiter(&self, limiter: &SurfaceRateLimiter) -> Result<()> {
        let json = serde_json::to_string(limiter).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![SURFACING_RATE_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load surfacing configuration.
    pub fn load_surfacing_config(&self) -> Result<SurfacingConfig> {
        match Self::get_meta(&self.conn(), SURFACING_CONFIG_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(SurfacingConfig::default()),
        }
    }

    /// Persist surfacing configuration.
    pub fn save_surfacing_config(&self, config: &SurfacingConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![SURFACING_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Main API ──

    /// Get proactive suggestions for the current moment.
    ///
    /// This is the primary API for the UI layer. It:
    /// 1. Loads agenda, receptivity model, rate limiter, and preferences
    /// 2. Estimates current receptivity
    /// 3. Runs the surfacing pipeline
    /// 4. Records surfaced items in the rate limiter
    /// 5. Marks items as surfaced in the agenda
    pub fn get_proactive_suggestions(
        &self,
        context: &ContextSnapshot,
        max: usize,
    ) -> Result<SurfacingResult> {
        let agenda = self.load_agenda()?;
        let receptivity_model = self.load_receptivity_model()?;
        let rate_limiter = self.load_surface_rate_limiter()?;
        let preferences = self.load_surfacing_preferences()?;
        let mut config = self.load_surfacing_config()?;
        config.max_suggestions = max;

        // Estimate current receptivity
        let receptivity = receptivity_model.estimate(context);

        // Get active agenda items
        let ts = now();
        let active_items = agenda.get_active(ts, 50);
        let items: Vec<_> = active_items.into_iter().cloned().collect();

        // Run pipeline with preferences
        let result = run_surfacing_with_preferences(
            &items,
            ts,
            &receptivity,
            &rate_limiter,
            context,
            &config,
            &preferences,
        );

        // Record surfaces in rate limiter and agenda
        if !result.suggestions.is_empty() {
            let mut rate_limiter = rate_limiter;
            let mut agenda = agenda;
            for suggestion in &result.suggestions {
                rate_limiter.record_surface(ts);
                agenda.mark_surfaced(suggestion.agenda_id, ts);
            }
            self.save_surface_rate_limiter(&rate_limiter)?;
            self.save_agenda(&agenda)?;
        }

        Ok(result)
    }

    /// Get proactive suggestions without learned preferences (basic pipeline).
    pub fn get_proactive_suggestions_basic(
        &self,
        context: &ContextSnapshot,
        max: usize,
    ) -> Result<SurfacingResult> {
        let agenda = self.load_agenda()?;
        let receptivity_model = self.load_receptivity_model()?;
        let rate_limiter = self.load_surface_rate_limiter()?;
        let mut config = self.load_surfacing_config()?;
        config.max_suggestions = max;

        let receptivity = receptivity_model.estimate(context);
        let ts = now();
        let active_items = agenda.get_active(ts, 50);
        let items: Vec<_> = active_items.into_iter().cloned().collect();

        Ok(run_surfacing_pipeline(
            &items, ts, &receptivity, &rate_limiter, context, &config,
        ))
    }

    // ── Feedback ──

    /// Record user feedback on a surfaced suggestion.
    ///
    /// Updates both the receptivity model and surfacing preferences.
    pub fn surfacing_feedback(
        &self,
        agenda_id: crate::agenda::AgendaId,
        mode: SurfaceMode,
        outcome: SurfaceOutcome,
        context: &ContextSnapshot,
    ) -> Result<()> {
        let mut preferences = self.load_surfacing_preferences()?;
        let mut agenda = self.load_agenda()?;

        // Find the item to get its kind
        if let Some(item) = agenda.find(agenda_id) {
            let hour = ((context.now % 86400.0) / 3600.0) as u8;
            preferences.observe(item.kind, mode, hour, outcome);
        }

        // Handle agenda state changes based on outcome
        match outcome {
            SurfaceOutcome::Acted => {
                agenda.resolve(agenda_id);
            }
            SurfaceOutcome::Dismissed | SurfaceOutcome::Annoyed => {
                agenda.dismiss(agenda_id);
            }
            SurfaceOutcome::Deferred => {
                // Snooze for default duration
                let config = self.load_surfacing_config()?;
                agenda.snooze(agenda_id, context.now, config.min_resurface_interval_secs);
            }
            SurfaceOutcome::Expired => {} // No action needed
        }

        // Update receptivity model with this outcome
        let recep_outcome = match outcome {
            SurfaceOutcome::Acted => crate::receptivity::SuggestionOutcome::Accepted,
            SurfaceOutcome::Deferred => crate::receptivity::SuggestionOutcome::Ignored,
            SurfaceOutcome::Dismissed | SurfaceOutcome::Expired => {
                crate::receptivity::SuggestionOutcome::Dismissed
            }
            SurfaceOutcome::Annoyed => crate::receptivity::SuggestionOutcome::Dismissed,
        };
        let mut receptivity_model = self.load_receptivity_model()?;
        receptivity_model.observe_outcome(context, recep_outcome);
        self.save_receptivity_model(&receptivity_model)?;

        self.save_surfacing_preferences(&preferences)?;
        self.save_agenda(&agenda)?;
        Ok(())
    }

    /// Reset the session rate limiter (e.g., on new login/session).
    pub fn reset_surfacing_session(&self) -> Result<()> {
        let mut limiter = self.load_surface_rate_limiter()?;
        limiter.reset_session();
        self.save_surface_rate_limiter(&limiter)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::agenda::{AgendaKind, UrgencyFn};
    use crate::engine::YantrikDB;
    use crate::receptivity::{ActivityState, ContextSnapshot, NotificationMode};
    use crate::state::NodeId;
    use crate::surfacing::{SurfaceOutcome, SurfacingPreferences};

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    fn test_context() -> ContextSnapshot {
        // Use a fixed daytime timestamp to avoid quiet hours
        ContextSnapshot {
            now: 86400.0 * 100.0 + 43200.0, // Noon UTC on day 100
            activity: ActivityState::TaskSwitching,
            recent_interactions_15min: 5,
            recent_outcomes: (3, 0, 0),
            secs_since_last_interaction: 30.0,
            session_duration_secs: 600.0,
            emotional_valence: 0.0,
            session_suggestions_accepted: 2,
            session_suggestion_budget: 20,
            notification_mode: NotificationMode::All,
        }
    }

    #[test]
    fn test_preferences_persistence() {
        let db = test_db();

        let mut prefs = SurfacingPreferences::new();
        prefs.observe(
            AgendaKind::FollowUpNeeded,
            crate::surfacing::SurfaceMode::Nudge,
            10,
            SurfaceOutcome::Acted,
        );
        db.save_surfacing_preferences(&prefs).unwrap();

        let loaded = db.load_surfacing_preferences().unwrap();
        assert_eq!(loaded.total_feedback, 1);
    }

    #[test]
    fn test_proactive_suggestions_empty() {
        let db = test_db();
        let ctx = test_context();

        let result = db.get_proactive_suggestions(&ctx, 5).unwrap();
        assert!(result.suggestions.is_empty());
        assert_eq!(result.items_evaluated, 0);
    }

    #[test]
    fn test_proactive_suggestions_with_items() {
        let db = test_db();

        // Add a high-urgency agenda item
        db.agenda_add(
            NodeId::NIL,
            AgendaKind::DeadlineApproaching,
            UrgencyFn::Constant { value: 0.85 },
            None,
            "Submit report by EOD".to_string(),
        ).unwrap();

        let ctx = test_context();
        let result = db.get_proactive_suggestions(&ctx, 5).unwrap();

        assert!(
            !result.suggestions.is_empty(),
            "Should surface high-urgency item"
        );
        assert_eq!(result.suggestions[0].description, "Submit report by EOD");
    }

    #[test]
    fn test_surfacing_feedback_resolves() {
        let db = test_db();

        let id = db.agenda_add(
            NodeId::NIL,
            AgendaKind::FollowUpNeeded,
            UrgencyFn::Constant { value: 0.7 },
            None,
            "Follow up".to_string(),
        ).unwrap();

        let ctx = test_context();
        db.surfacing_feedback(
            id,
            crate::surfacing::SurfaceMode::Nudge,
            SurfaceOutcome::Acted,
            &ctx,
        ).unwrap();

        // Item should be resolved
        let agenda = db.load_agenda().unwrap();
        assert_eq!(agenda.active_count(), 0);
    }

    #[test]
    fn test_surfacing_feedback_dismisses() {
        let db = test_db();

        let id = db.agenda_add(
            NodeId::NIL,
            AgendaKind::RoutineWindowOpening,
            UrgencyFn::Constant { value: 0.5 },
            None,
            "Check email".to_string(),
        ).unwrap();

        let ctx = test_context();
        db.surfacing_feedback(
            id,
            crate::surfacing::SurfaceMode::Whisper,
            SurfaceOutcome::Dismissed,
            &ctx,
        ).unwrap();

        // Item should be dismissed
        let agenda = db.load_agenda().unwrap();
        assert_eq!(agenda.active_count(), 0);

        // Preferences should learn
        let prefs = db.load_surfacing_preferences().unwrap();
        assert_eq!(prefs.total_feedback, 1);
        assert!(prefs.threshold_adjustment > 0.0);
    }

    #[test]
    fn test_reset_session() {
        let db = test_db();

        let mut limiter = db.load_surface_rate_limiter().unwrap();
        limiter.session_surface_count = 15;
        db.save_surface_rate_limiter(&limiter).unwrap();

        db.reset_surfacing_session().unwrap();

        let loaded = db.load_surface_rate_limiter().unwrap();
        assert_eq!(loaded.session_surface_count, 0);
    }
}
