//! Engine-level passive event observer API.
//!
//! Wires the observer module into `YantrikDB` for persistence
//! and integration with the cognitive tick and surfacing pipeline.

use crate::error::Result;
use crate::observer::{
    compute_derived_signals, detect_app_sequences, mark_flushed, needs_flush,
    observe_event, query_events, summarize, CircadianHistogram, DerivedSignals,
    EventBuffer, EventFilter, EventKind, ObserverConfig, ObserverState,
    ObserverSummary, SystemEvent,
};

use super::{now, YantrikDB};

/// Meta key for persisted observer state (counters, histogram, config).
const OBSERVER_STATE_META_KEY: &str = "event_observer_state";
/// Meta key for persisted event buffer.
const OBSERVER_BUFFER_META_KEY: &str = "event_observer_buffer";
/// Meta key for persisted observer config.
const OBSERVER_CONFIG_META_KEY: &str = "event_observer_config";

impl YantrikDB {
    // ── Persistence ──

    /// Load the observer state from the database.
    pub fn load_observer_state(&self) -> Result<ObserverState> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), OBSERVER_STATE_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(ObserverState::new()),
        }
    }

    /// Persist the observer state.
    pub fn save_observer_state(&self, state: &ObserverState) -> Result<()> {
        let json = serde_json::to_string(state).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![OBSERVER_STATE_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the event buffer from the database.
    pub fn load_event_buffer(&self) -> Result<EventBuffer> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), OBSERVER_BUFFER_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(EventBuffer::new(10_000)),
        }
    }

    /// Persist the event buffer.
    pub fn save_event_buffer(&self, buffer: &EventBuffer) -> Result<()> {
        let json = serde_json::to_string(buffer).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![OBSERVER_BUFFER_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the observer configuration.
    pub fn load_observer_config(&self) -> Result<ObserverConfig> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), OBSERVER_CONFIG_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(ObserverConfig::default()),
        }
    }

    /// Persist the observer configuration.
    pub fn save_observer_config(&self, config: &ObserverConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![OBSERVER_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Main API ──

    /// Observe a system event.
    ///
    /// Non-blocking from the caller's perspective. The event is validated,
    /// buffered, and tracked. Auto-flushes to disk when thresholds are met.
    ///
    /// Returns `true` if the event was accepted.
    pub fn observe(&self, event: SystemEvent) -> Result<bool> {
        let mut buffer = self.load_event_buffer()?;
        let mut state = self.load_observer_state()?;

        let accepted = observe_event(event, &mut buffer, &mut state);

        // Auto-flush if thresholds met
        let ts = now();
        if needs_flush(&state, ts) {
            mark_flushed(&mut state, ts);
            self.save_event_buffer(&buffer)?;
        }

        self.save_observer_state(&state)?;

        // Only save buffer if we didn't just flush (avoid double-write)
        if state.pending_batch_count > 0 {
            self.save_event_buffer(&buffer)?;
        }

        Ok(accepted)
    }

    /// Observe multiple events in a batch (more efficient than individual calls).
    pub fn observe_batch(&self, events: Vec<SystemEvent>) -> Result<u32> {
        let mut buffer = self.load_event_buffer()?;
        let mut state = self.load_observer_state()?;

        let mut accepted = 0u32;
        for event in events {
            if observe_event(event, &mut buffer, &mut state) {
                accepted += 1;
            }
        }

        let ts = now();
        mark_flushed(&mut state, ts);
        self.save_observer_state(&state)?;
        self.save_event_buffer(&buffer)?;

        Ok(accepted)
    }

    /// Start a new observer session (e.g., on login/boot).
    pub fn observer_start_session(&self) -> Result<()> {
        let mut state = self.load_observer_state()?;
        state.session_start = now();
        self.save_observer_state(&state)
    }

    // ── Query API ──

    /// Get recent events with optional filtering.
    pub fn observer_recent_events(
        &self,
        filter: &EventFilter,
        limit: usize,
    ) -> Result<Vec<SystemEvent>> {
        let buffer = self.load_event_buffer()?;
        Ok(query_events(&buffer, filter, limit)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Get the event rate (events/sec) for a specific kind.
    pub fn observer_event_rate(&self, kind: EventKind) -> Result<f64> {
        let state = self.load_observer_state()?;
        Ok(state.counters.rate(kind, now()))
    }

    /// Get the 24-hour circadian distribution for an event kind.
    pub fn observer_circadian_distribution(&self, kind: EventKind) -> Result<[u32; 24]> {
        let state = self.load_observer_state()?;
        Ok(state.histogram.distribution(kind))
    }

    /// Get a full observer summary.
    pub fn observer_summary(&self) -> Result<ObserverSummary> {
        let buffer = self.load_event_buffer()?;
        let state = self.load_observer_state()?;
        Ok(summarize(&buffer, &state, now()))
    }

    /// Compute derived signals from recent observations.
    pub fn observer_derived_signals(&self, window_secs: f64) -> Result<DerivedSignals> {
        let buffer = self.load_event_buffer()?;
        let ts = now();
        Ok(compute_derived_signals(&buffer, ts - window_secs, ts))
    }

    /// Detect app transition sequences from recent observations.
    pub fn observer_detect_app_sequences(&self, max_gap_ms: u64) -> Result<Vec<SystemEvent>> {
        let buffer = self.load_event_buffer()?;
        let ts = now();
        Ok(detect_app_sequences(
            &buffer,
            ts - 3600.0, // Last hour
            max_gap_ms,
        ))
    }

    /// Get the total event count for a specific kind (all-time).
    pub fn observer_event_total(&self, kind: EventKind) -> Result<u64> {
        let state = self.load_observer_state()?;
        Ok(state.counters.total(kind))
    }

    // ── Maintenance ──

    /// Prune events older than the configured retention period.
    pub fn observer_prune(&self) -> Result<usize> {
        let mut buffer = self.load_event_buffer()?;
        let state = self.load_observer_state()?;
        let cutoff = now() - state.config.retention_secs;
        let pruned = buffer.prune_before(cutoff);
        if pruned > 0 {
            self.save_event_buffer(&buffer)?;
        }
        Ok(pruned)
    }

    /// Disable an event kind (stops recording future events of this type).
    pub fn observer_disable_kind(&self, kind: EventKind) -> Result<()> {
        let mut state = self.load_observer_state()?;
        if !state.config.disabled_kinds.contains(&kind) {
            state.config.disabled_kinds.push(kind);
            self.save_observer_state(&state)?;
        }
        Ok(())
    }

    /// Enable an event kind.
    pub fn observer_enable_kind(&self, kind: EventKind) -> Result<()> {
        let mut state = self.load_observer_state()?;
        state.config.disabled_kinds.retain(|&k| k != kind);
        self.save_observer_state(&state)?;
        Ok(())
    }

    /// Reset the observer (clear buffer and counters). Irreversible.
    pub fn observer_reset(&self) -> Result<()> {
        let config = self.load_observer_config()?;
        let state = ObserverState::with_config(config.clone());
        let buffer = EventBuffer::new(config.buffer_capacity);
        self.save_observer_state(&state)?;
        self.save_event_buffer(&buffer)?;
        Ok(())
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::observer::{EventFilter, EventKind, SystemEvent, SystemEventData};

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    fn make_event(kind_data: SystemEventData) -> SystemEvent {
        SystemEvent::new(86400.0 * 100.0 + 43200.0, kind_data) // Noon on day 100
    }

    #[test]
    fn test_observer_state_persistence_roundtrip() {
        let db = test_db();

        let event = make_event(SystemEventData::AppOpened { app_id: 14 });
        assert!(db.observe(event).unwrap());

        // Verify state persisted
        let state = db.load_observer_state().unwrap();
        assert_eq!(state.counters.total(EventKind::AppOpened), 1);
    }

    #[test]
    fn test_observer_batch() {
        let db = test_db();

        let events = vec![
            make_event(SystemEventData::AppOpened { app_id: 1 }),
            make_event(SystemEventData::AppOpened { app_id: 2 }),
            make_event(SystemEventData::AppClosed {
                app_id: 1,
                duration_ms: 5000,
            }),
        ];
        let accepted = db.observe_batch(events).unwrap();
        assert_eq!(accepted, 3);

        let state = db.load_observer_state().unwrap();
        assert_eq!(state.counters.total(EventKind::AppOpened), 2);
        assert_eq!(state.counters.total(EventKind::AppClosed), 1);
    }

    #[test]
    fn test_observer_query() {
        let db = test_db();

        db.observe(make_event(SystemEventData::AppOpened { app_id: 1 }))
            .unwrap();
        db.observe(make_event(SystemEventData::SuggestionAccepted {
            suggestion_id: 1,
            action_kind: "test".to_string(),
            latency_ms: 200,
        }))
        .unwrap();
        db.observe(make_event(SystemEventData::AppOpened { app_id: 2 }))
            .unwrap();

        let filter = EventFilter::new().kind(EventKind::AppOpened);
        let results = db.observer_recent_events(&filter, 10).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_observer_summary() {
        let db = test_db();
        db.observer_start_session().unwrap();

        for i in 0..5 {
            db.observe(make_event(SystemEventData::AppOpened { app_id: i }))
                .unwrap();
        }

        let summary = db.observer_summary().unwrap();
        assert_eq!(summary.buffer_size, 5);
        assert_eq!(summary.most_active_kind, Some(EventKind::AppOpened));
    }

    #[test]
    fn test_observer_disable_kind() {
        let db = test_db();

        db.observer_disable_kind(EventKind::UserTyping).unwrap();

        let event = make_event(SystemEventData::UserTyping {
            app_id: 1,
            duration_ms: 1000,
            characters: 50,
        });
        let accepted = db.observe(event).unwrap();
        assert!(!accepted);

        // Re-enable
        db.observer_enable_kind(EventKind::UserTyping).unwrap();
        let event2 = make_event(SystemEventData::UserTyping {
            app_id: 1,
            duration_ms: 2000,
            characters: 80,
        });
        let accepted2 = db.observe(event2).unwrap();
        assert!(accepted2);
    }

    #[test]
    fn test_observer_derived_signals() {
        let db = test_db();

        db.observe(make_event(SystemEventData::SuggestionAccepted {
            suggestion_id: 1,
            action_kind: "nudge".to_string(),
            latency_ms: 200,
        }))
        .unwrap();
        db.observe(make_event(SystemEventData::SuggestionRejected {
            suggestion_id: 2,
            action_kind: "whisper".to_string(),
        }))
        .unwrap();

        let signals = db.observer_derived_signals(3600.0).unwrap();
        assert!((signals.suggestion_acceptance_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_observer_reset() {
        let db = test_db();

        db.observe(make_event(SystemEventData::AppOpened { app_id: 1 }))
            .unwrap();
        db.observer_reset().unwrap();

        let summary = db.observer_summary().unwrap();
        assert_eq!(summary.buffer_size, 0);
        assert_eq!(summary.total_events, 0);
    }

    #[test]
    fn test_observer_circadian() {
        let db = test_db();

        // Event at noon (hour 12)
        db.observe(make_event(SystemEventData::AppOpened { app_id: 1 }))
            .unwrap();

        let dist = db
            .observer_circadian_distribution(EventKind::AppOpened)
            .unwrap();
        assert_eq!(dist[12], 1);
    }
}
