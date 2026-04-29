//! RFC 013-B — embedding-model migration via shadow index.
//!
//! ## What this owns
//!
//! - [`ShadowMigrationState`] — the per-tenant migration phase machine.
//! - [`ShadowIndexConfig`] — source-model + target-model description,
//!   including the dimension change that motivates this whole feature.
//! - [`DualReadMerger`] — score-normalized merge of two query result
//!   sets during the dual-read phase. Pure function, no I/O.
//! - [`MigrationProgress`] — observable progress (watermark deltas,
//!   percent complete). Used by `/v1/admin/tenants/{id}/migration`
//!   read-only endpoint and the operator runbook.
//!
//! ## What's NOT here (deferred to consumer PR)
//!
//! - The actual re-embedding worker (RFC 019 job). It pulls from the
//!   commit log, re-embeds with the target model, writes to the shadow
//!   index. The substrate exposes the watermark interface; the worker
//!   advances it.
//! - The recall handler's "fan out to both indexes" plumbing — that's
//!   RFC 015 retrieval glue.
//! - Per-tenant CLI (`yantrikdb tenant rebuild-embeddings --tenant X
//!   --model bge-base`) — admin path.
//! - SQLite-backed store for migration state — substrate exposes the
//!   in-memory store + trait; the prod impl persists.
//!
//! ## Phase machine
//!
//! ```text
//!   Idle ──[start]──▶ Backfilling ──[caught up]──▶ DualRead ──[cutover_ok]──▶ Cutover
//!                                                                              │
//!                                                                              ▼
//!                                                                          Complete
//!
//!   Any state ──[abort]──▶ Aborted   (allows operator to bail mid-migration)
//! ```
//!
//! - **Idle**: no migration in progress; only the source index serves
//!   reads + writes.
//! - **Backfilling**: shadow index is being populated by the re-embed
//!   worker. Source index serves reads; new writes are dual-written
//!   (source + shadow). `target_log_watermark` advances toward the
//!   source watermark.
//! - **DualRead**: shadow index is caught up. Reads fan out to both;
//!   results merged via score normalization. Writes still dual-written.
//!   Validation period — operators inspect dashboards before cutover.
//! - **Cutover**: shadow becomes primary. Source index is retained
//!   for the configured rollback window.
//! - **Complete**: source index has been retired. Shadow is primary.
//! - **Aborted**: migration was aborted; shadow is discarded.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::RwLock;

/// Description of the source + target embedding models. Captured at
/// migration start; the worker validates each re-embed against this.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowIndexConfig {
    pub source_model: String,
    pub source_dim: u32,
    pub target_model: String,
    pub target_dim: u32,
}

impl ShadowIndexConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.source_model.is_empty() {
            return Err("source_model must not be empty");
        }
        if self.target_model.is_empty() {
            return Err("target_model must not be empty");
        }
        if self.source_dim == 0 || self.target_dim == 0 {
            return Err("dimensions must be > 0");
        }
        if self.source_model == self.target_model && self.source_dim == self.target_dim {
            return Err("source and target model are identical — no migration needed");
        }
        Ok(())
    }
}

/// Phase of a per-tenant migration. Strings are the wire form for
/// persistence and `migration_state{tenant}` Prometheus label —
/// pinned in tests so dashboards don't break silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShadowMigrationPhase {
    Idle,
    Backfilling,
    DualRead,
    Cutover,
    Complete,
    Aborted,
}

impl ShadowMigrationPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            ShadowMigrationPhase::Idle => "idle",
            ShadowMigrationPhase::Backfilling => "backfilling",
            ShadowMigrationPhase::DualRead => "dual_read",
            ShadowMigrationPhase::Cutover => "cutover",
            ShadowMigrationPhase::Complete => "complete",
            ShadowMigrationPhase::Aborted => "aborted",
        }
    }

    /// Per the phase machine: which phases are valid transitions from
    /// `self`?
    pub fn allowed_next(self) -> &'static [ShadowMigrationPhase] {
        use ShadowMigrationPhase::*;
        match self {
            Idle => &[Backfilling, Aborted],
            Backfilling => &[DualRead, Aborted],
            DualRead => &[Cutover, Aborted],
            Cutover => &[Complete, Aborted],
            Complete => &[],
            Aborted => &[],
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            ShadowMigrationPhase::Complete | ShadowMigrationPhase::Aborted
        )
    }
}

impl std::fmt::Display for ShadowMigrationPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One tenant's migration state. Cheap to clone (POD).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowMigrationState {
    pub tenant_id: String,
    pub phase: ShadowMigrationPhase,
    pub config: ShadowIndexConfig,
    /// Where the re-embed worker has caught up to. 0 at start; advances
    /// monotonically toward `source_log_watermark`. When equal, the
    /// shadow is "caught up" and can transition to DualRead.
    pub target_log_watermark: u64,
    /// What the source index is currently at. Advances independently
    /// (new writes still land in source). The cutover gate checks
    /// `target_log_watermark >= source_log_watermark` AT cutover time.
    pub source_log_watermark: u64,
    /// Errors-since-start for the re-embed worker. If this exceeds
    /// the operator's tolerance threshold, the migration should be
    /// aborted and investigated.
    pub error_count: u32,
    /// When the migration was started (unix micros). Helps operators
    /// answer "is this stuck?".
    pub started_at_unix_micros: i64,
    /// When the phase last changed. Used for the
    /// `migration_phase_age_seconds` dashboard panel.
    pub phase_entered_at_unix_micros: i64,
}

#[derive(Debug, thiserror::Error)]
pub enum ShadowMigrationError {
    #[error("invalid config: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid phase transition `{from}` → `{to}`")]
    InvalidTransition {
        from: ShadowMigrationPhase,
        to: ShadowMigrationPhase,
    },
    #[error("shadow watermark {target_wm} below source watermark {source_wm}; not caught up")]
    NotCaughtUp { target_wm: u64, source_wm: u64 },
    #[error("no migration in progress for tenant `{0}`")]
    NotFound(String),
    #[error("migration already in progress for tenant `{0}` (phase `{1}`)")]
    AlreadyActive(String, &'static str),
}

impl ShadowMigrationState {
    /// Create a fresh state for a brand-new migration, in `Idle`.
    /// Caller transitions to `Backfilling` via `start_backfilling`.
    pub fn new(
        tenant_id: impl Into<String>,
        config: ShadowIndexConfig,
        source_log_watermark: u64,
        now_unix_micros: i64,
    ) -> Result<Self, ShadowMigrationError> {
        config
            .validate()
            .map_err(ShadowMigrationError::InvalidConfig)?;
        Ok(Self {
            tenant_id: tenant_id.into(),
            phase: ShadowMigrationPhase::Idle,
            config,
            target_log_watermark: 0,
            source_log_watermark,
            error_count: 0,
            started_at_unix_micros: now_unix_micros,
            phase_entered_at_unix_micros: now_unix_micros,
        })
    }

    /// Transition to a new phase, validating the transition is allowed.
    pub fn transition_to(
        &mut self,
        next: ShadowMigrationPhase,
        now_unix_micros: i64,
    ) -> Result<(), ShadowMigrationError> {
        if !self.phase.allowed_next().contains(&next) {
            return Err(ShadowMigrationError::InvalidTransition {
                from: self.phase,
                to: next,
            });
        }
        // Caught-up gate: shadow must be caught up before DualRead and
        // again at Cutover. RFC 013-B: "DualRead: shadow index is caught
        // up."; "Cutover when source_log_watermark >= target_watermark."
        if matches!(
            next,
            ShadowMigrationPhase::DualRead | ShadowMigrationPhase::Cutover
        ) {
            if self.target_log_watermark < self.source_log_watermark {
                return Err(ShadowMigrationError::NotCaughtUp {
                    target_wm: self.target_log_watermark,
                    source_wm: self.source_log_watermark,
                });
            }
        }
        self.phase = next;
        self.phase_entered_at_unix_micros = now_unix_micros;
        Ok(())
    }

    /// Advance the target watermark from worker progress. Saturating —
    /// re-emit of the same boundary doesn't decrement.
    pub fn advance_target(&mut self, watermark: u64) {
        self.target_log_watermark = self.target_log_watermark.max(watermark);
    }

    /// Update where the source has reached. Called by the commit observer.
    pub fn observe_source(&mut self, watermark: u64) {
        self.source_log_watermark = self.source_log_watermark.max(watermark);
    }

    /// Increment error count from a failed re-embed.
    pub fn record_error(&mut self) {
        self.error_count = self.error_count.saturating_add(1);
    }

    /// Compute a [`MigrationProgress`] snapshot.
    pub fn progress(&self) -> MigrationProgress {
        let percent = if self.source_log_watermark == 0 {
            0.0
        } else {
            let p = self.target_log_watermark as f64 / self.source_log_watermark as f64;
            (p * 100.0).clamp(0.0, 100.0)
        };
        MigrationProgress {
            phase: self.phase,
            percent_complete: percent,
            target_log_watermark: self.target_log_watermark,
            source_log_watermark: self.source_log_watermark,
            error_count: self.error_count,
        }
    }

    /// True iff `target_log_watermark >= source_log_watermark`. Used by
    /// the orchestrator to decide whether to advance Backfilling →
    /// DualRead.
    pub fn is_caught_up(&self) -> bool {
        self.target_log_watermark >= self.source_log_watermark
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MigrationProgress {
    pub phase: ShadowMigrationPhase,
    /// 0.0 — 100.0
    pub percent_complete: f64,
    pub target_log_watermark: u64,
    pub source_log_watermark: u64,
    pub error_count: u32,
}

/// Pluggable store for per-tenant migration state. Substrate trait;
/// the SQLite-backed prod impl will live in a follow-up PR.
pub trait MigrationStateStore: Send + Sync {
    fn upsert(&self, state: ShadowMigrationState) -> Result<(), ShadowMigrationError>;
    fn get(&self, tenant_id: &str) -> Result<Option<ShadowMigrationState>, ShadowMigrationError>;
    fn list_active(&self) -> Result<Vec<ShadowMigrationState>, ShadowMigrationError>;
    fn delete(&self, tenant_id: &str) -> Result<bool, ShadowMigrationError>;
}

/// In-memory store. For tests and dev.
#[derive(Default, Clone)]
pub struct InMemoryMigrationStore {
    inner: Arc<RwLock<BTreeMap<String, ShadowMigrationState>>>,
}

impl InMemoryMigrationStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }
}

impl MigrationStateStore for InMemoryMigrationStore {
    fn upsert(&self, state: ShadowMigrationState) -> Result<(), ShadowMigrationError> {
        self.inner.write().insert(state.tenant_id.clone(), state);
        Ok(())
    }

    fn get(&self, tenant_id: &str) -> Result<Option<ShadowMigrationState>, ShadowMigrationError> {
        Ok(self.inner.read().get(tenant_id).cloned())
    }

    fn list_active(&self) -> Result<Vec<ShadowMigrationState>, ShadowMigrationError> {
        Ok(self
            .inner
            .read()
            .values()
            .filter(|s| !s.phase.is_terminal())
            .cloned()
            .collect())
    }

    fn delete(&self, tenant_id: &str) -> Result<bool, ShadowMigrationError> {
        Ok(self.inner.write().remove(tenant_id).is_some())
    }
}

/// One scored hit returned by either index. Substrate-level — the
/// recall handler converts its native rid+score representation to
/// this for merging.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredHit {
    pub rid: String,
    /// Normalized similarity score in [0.0, 1.0]. Higher is better.
    /// The merger expects callers to pre-normalize because the
    /// normalization rules differ per distance metric (cosine,
    /// inner product, L2). The merger itself is metric-agnostic.
    pub score: f32,
    /// Which index produced this hit. The merger uses it for tracing
    /// + metric labels; doesn't influence ordering.
    pub source: HitSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitSource {
    SourceIndex,
    ShadowIndex,
}

impl HitSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            HitSource::SourceIndex => "source",
            HitSource::ShadowIndex => "shadow",
        }
    }
}

/// Score-normalized merge of two result sets during DualRead. Caller
/// supplies pre-normalized scores; merger picks max-score for any rid
/// that appears in both sets and returns top-k.
pub struct DualReadMerger;

impl DualReadMerger {
    /// Merge two pre-normalized result sets.
    /// - For rids present in only one set, take that score.
    /// - For rids present in both, take the max score (arguably the
    ///   safer default than mean — if either index has a strong
    ///   semantic signal for this rid, we should respect it).
    /// - The merged result preserves the source label of the chosen
    ///   score (or `ShadowIndex` on ties — shadow is the future).
    pub fn merge(source: Vec<ScoredHit>, shadow: Vec<ScoredHit>, top_k: usize) -> Vec<ScoredHit> {
        use std::collections::HashMap;
        let mut by_rid: HashMap<String, ScoredHit> =
            HashMap::with_capacity(source.len() + shadow.len());
        for h in source {
            by_rid.insert(h.rid.clone(), h);
        }
        for h in shadow {
            by_rid
                .entry(h.rid.clone())
                .and_modify(|existing| {
                    if h.score >= existing.score {
                        // shadow wins on tie (its score "should be" the
                        // semantically newer signal under the new model).
                        *existing = h.clone();
                    }
                })
                .or_insert(h);
        }
        let mut all: Vec<ScoredHit> = by_rid.into_values().collect();
        // Descending by score; ties broken by rid for determinism.
        all.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.rid.cmp(&b.rid))
        });
        all.truncate(top_k);
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_default() -> ShadowIndexConfig {
        ShadowIndexConfig {
            source_model: "MiniLM-L6-v2".into(),
            source_dim: 384,
            target_model: "bge-base".into(),
            target_dim: 768,
        }
    }

    #[test]
    fn config_validate_accepts_real_migration() {
        assert!(cfg_default().validate().is_ok());
    }

    #[test]
    fn config_validate_rejects_identical_models() {
        let c = ShadowIndexConfig {
            source_model: "MiniLM-L6-v2".into(),
            source_dim: 384,
            target_model: "MiniLM-L6-v2".into(),
            target_dim: 384,
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_rejects_empty_model_or_zero_dim() {
        let c = ShadowIndexConfig {
            source_model: "".into(),
            source_dim: 384,
            target_model: "bge-base".into(),
            target_dim: 768,
        };
        assert!(c.validate().is_err());
        let c = ShadowIndexConfig {
            source_model: "MiniLM".into(),
            source_dim: 0,
            target_model: "bge".into(),
            target_dim: 768,
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn phase_string_pinned() {
        assert_eq!(ShadowMigrationPhase::Idle.as_str(), "idle");
        assert_eq!(ShadowMigrationPhase::Backfilling.as_str(), "backfilling");
        assert_eq!(ShadowMigrationPhase::DualRead.as_str(), "dual_read");
        assert_eq!(ShadowMigrationPhase::Cutover.as_str(), "cutover");
        assert_eq!(ShadowMigrationPhase::Complete.as_str(), "complete");
        assert_eq!(ShadowMigrationPhase::Aborted.as_str(), "aborted");
    }

    #[test]
    fn phase_terminal_only_complete_or_aborted() {
        for p in [
            ShadowMigrationPhase::Idle,
            ShadowMigrationPhase::Backfilling,
            ShadowMigrationPhase::DualRead,
            ShadowMigrationPhase::Cutover,
        ] {
            assert!(!p.is_terminal());
        }
        assert!(ShadowMigrationPhase::Complete.is_terminal());
        assert!(ShadowMigrationPhase::Aborted.is_terminal());
    }

    #[test]
    fn allowed_transitions_match_phase_machine() {
        // Idle → {Backfilling, Aborted}
        assert!(ShadowMigrationPhase::Idle
            .allowed_next()
            .contains(&ShadowMigrationPhase::Backfilling));
        assert!(ShadowMigrationPhase::Idle
            .allowed_next()
            .contains(&ShadowMigrationPhase::Aborted));
        // Idle → Cutover NOT allowed (would bypass backfill).
        assert!(!ShadowMigrationPhase::Idle
            .allowed_next()
            .contains(&ShadowMigrationPhase::Cutover));
        // Complete is terminal — no transitions out.
        assert!(ShadowMigrationPhase::Complete.allowed_next().is_empty());
    }

    #[test]
    fn new_state_starts_in_idle() {
        let s = ShadowMigrationState::new("t1", cfg_default(), 100, 1234).unwrap();
        assert_eq!(s.phase, ShadowMigrationPhase::Idle);
        assert_eq!(s.target_log_watermark, 0);
        assert_eq!(s.source_log_watermark, 100);
    }

    #[test]
    fn legal_transition_succeeds() {
        let mut s = ShadowMigrationState::new("t1", cfg_default(), 100, 1).unwrap();
        s.transition_to(ShadowMigrationPhase::Backfilling, 2)
            .unwrap();
        assert_eq!(s.phase, ShadowMigrationPhase::Backfilling);
        assert_eq!(s.phase_entered_at_unix_micros, 2);
    }

    #[test]
    fn illegal_transition_rejected() {
        let mut s = ShadowMigrationState::new("t1", cfg_default(), 100, 1).unwrap();
        let err = s
            .transition_to(ShadowMigrationPhase::Cutover, 2)
            .unwrap_err();
        assert!(matches!(
            err,
            ShadowMigrationError::InvalidTransition { .. }
        ));
    }

    #[test]
    fn cutover_blocked_until_caught_up() {
        let mut s = ShadowMigrationState::new("t1", cfg_default(), 100, 1).unwrap();
        s.transition_to(ShadowMigrationPhase::Backfilling, 2)
            .unwrap();
        s.transition_to(ShadowMigrationPhase::DualRead, 3)
            .unwrap_err(); // not caught up

        // Catch up part-way — still blocked.
        s.advance_target(50);
        s.transition_to(ShadowMigrationPhase::DualRead, 4)
            .unwrap_err();

        // Fully catch up — ok.
        s.advance_target(100);
        s.transition_to(ShadowMigrationPhase::DualRead, 5).unwrap();
        assert_eq!(s.phase, ShadowMigrationPhase::DualRead);

        // Cutover gate: shadow ≥ source.
        s.transition_to(ShadowMigrationPhase::Cutover, 6).unwrap();
        assert_eq!(s.phase, ShadowMigrationPhase::Cutover);
    }

    #[test]
    fn cutover_blocked_if_source_advances_past_target() {
        let mut s = ShadowMigrationState::new("t1", cfg_default(), 100, 1).unwrap();
        s.transition_to(ShadowMigrationPhase::Backfilling, 2)
            .unwrap();
        s.advance_target(100);
        s.transition_to(ShadowMigrationPhase::DualRead, 3).unwrap();
        // New writes land in source while we were dual-reading.
        s.observe_source(150);
        let err = s
            .transition_to(ShadowMigrationPhase::Cutover, 4)
            .unwrap_err();
        assert!(matches!(
            err,
            ShadowMigrationError::NotCaughtUp {
                target_wm: 100,
                source_wm: 150
            }
        ));
    }

    #[test]
    fn advance_target_is_monotonic() {
        let mut s = ShadowMigrationState::new("t1", cfg_default(), 100, 1).unwrap();
        s.advance_target(50);
        s.advance_target(30); // smaller — must be ignored.
        assert_eq!(s.target_log_watermark, 50);
    }

    #[test]
    fn observe_source_is_monotonic() {
        let mut s = ShadowMigrationState::new("t1", cfg_default(), 100, 1).unwrap();
        s.observe_source(50); // smaller than 100 → ignored.
        assert_eq!(s.source_log_watermark, 100);
        s.observe_source(150);
        assert_eq!(s.source_log_watermark, 150);
    }

    #[test]
    fn progress_percent_clamped_to_100() {
        let mut s = ShadowMigrationState::new("t1", cfg_default(), 100, 1).unwrap();
        s.advance_target(150); // legitimate "ahead" via test state
        let p = s.progress();
        assert!((p.percent_complete - 100.0).abs() < 1e-6);
    }

    #[test]
    fn progress_zero_at_zero_source() {
        let s = ShadowMigrationState::new("t1", cfg_default(), 0, 1).unwrap();
        let p = s.progress();
        assert_eq!(p.percent_complete, 0.0);
    }

    #[test]
    fn record_error_increments() {
        let mut s = ShadowMigrationState::new("t1", cfg_default(), 100, 1).unwrap();
        s.record_error();
        s.record_error();
        assert_eq!(s.error_count, 2);
    }

    #[test]
    fn abort_allowed_from_any_active_phase() {
        for from in [
            ShadowMigrationPhase::Idle,
            ShadowMigrationPhase::Backfilling,
            ShadowMigrationPhase::DualRead,
            ShadowMigrationPhase::Cutover,
        ] {
            assert!(from.allowed_next().contains(&ShadowMigrationPhase::Aborted));
        }
    }

    #[test]
    fn store_upsert_and_get_round_trip() {
        let store = InMemoryMigrationStore::new();
        let s = ShadowMigrationState::new("t1", cfg_default(), 100, 1).unwrap();
        store.upsert(s.clone()).unwrap();
        let got = store.get("t1").unwrap().unwrap();
        assert_eq!(got, s);
    }

    #[test]
    fn store_list_active_excludes_terminal() {
        let store = InMemoryMigrationStore::new();
        let mut a = ShadowMigrationState::new("t1", cfg_default(), 100, 1).unwrap();
        a.transition_to(ShadowMigrationPhase::Backfilling, 2)
            .unwrap();
        let mut b = ShadowMigrationState::new("t2", cfg_default(), 100, 1).unwrap();
        b.transition_to(ShadowMigrationPhase::Aborted, 2).unwrap();
        store.upsert(a).unwrap();
        store.upsert(b).unwrap();
        let active = store.list_active().unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].tenant_id, "t1");
    }

    #[test]
    fn store_delete_returns_existed_flag() {
        let store = InMemoryMigrationStore::new();
        let s = ShadowMigrationState::new("t1", cfg_default(), 100, 1).unwrap();
        store.upsert(s).unwrap();
        assert!(store.delete("t1").unwrap());
        assert!(!store.delete("t1").unwrap()); // already gone
    }

    fn h(rid: &str, score: f32, source: HitSource) -> ScoredHit {
        ScoredHit {
            rid: rid.into(),
            score,
            source,
        }
    }

    #[test]
    fn merger_unions_distinct_rids() {
        let src = vec![h("a", 0.9, HitSource::SourceIndex)];
        let sha = vec![h("b", 0.8, HitSource::ShadowIndex)];
        let merged = DualReadMerger::merge(src, sha, 10);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].rid, "a"); // higher score first
        assert_eq!(merged[1].rid, "b");
    }

    #[test]
    fn merger_prefers_higher_score_on_overlap() {
        let src = vec![h("a", 0.5, HitSource::SourceIndex)];
        let sha = vec![h("a", 0.9, HitSource::ShadowIndex)];
        let merged = DualReadMerger::merge(src, sha, 10);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].score, 0.9);
        assert_eq!(merged[0].source, HitSource::ShadowIndex);
    }

    #[test]
    fn merger_breaks_ties_with_shadow_winner() {
        // Tie: equal score → shadow wins (it's the "future" signal).
        let src = vec![h("a", 0.5, HitSource::SourceIndex)];
        let sha = vec![h("a", 0.5, HitSource::ShadowIndex)];
        let merged = DualReadMerger::merge(src, sha, 10);
        assert_eq!(merged[0].source, HitSource::ShadowIndex);
    }

    #[test]
    fn merger_truncates_to_top_k() {
        // Construct so shadow strictly dominates: every shadow hit is
        // greater than every source hit. Top 5 then must all come from
        // shadow.
        let src: Vec<ScoredHit> = (0..20)
            .map(|i| {
                h(
                    &format!("s{}", i),
                    0.5 - i as f32 * 0.01,
                    HitSource::SourceIndex,
                )
            })
            .collect();
        let sha: Vec<ScoredHit> = (0..20)
            .map(|i| {
                h(
                    &format!("h{}", i),
                    1.0 - i as f32 * 0.005,
                    HitSource::ShadowIndex,
                )
            })
            .collect();
        let merged = DualReadMerger::merge(src, sha, 5);
        assert_eq!(merged.len(), 5);
        for hit in &merged {
            assert_eq!(hit.source, HitSource::ShadowIndex);
        }
    }

    #[test]
    fn merger_deterministic_for_equal_scores_distinct_rids() {
        let src = vec![h("zzz", 0.5, HitSource::SourceIndex)];
        let sha = vec![h("aaa", 0.5, HitSource::ShadowIndex)];
        let merged = DualReadMerger::merge(src, sha, 10);
        // rid-asc tie-breaker → "aaa" first.
        assert_eq!(merged[0].rid, "aaa");
        assert_eq!(merged[1].rid, "zzz");
    }

    #[test]
    fn hit_source_strings_pinned() {
        assert_eq!(HitSource::SourceIndex.as_str(), "source");
        assert_eq!(HitSource::ShadowIndex.as_str(), "shadow");
    }

    #[test]
    fn store_is_dyn_dispatchable() {
        // Trait must remain object-safe so the data plane can hold
        // Arc<dyn MigrationStateStore>.
        let store: Arc<dyn MigrationStateStore> = Arc::new(InMemoryMigrationStore::new());
        let s = ShadowMigrationState::new("t1", cfg_default(), 100, 1).unwrap();
        store.upsert(s).unwrap();
        assert!(store.get("t1").unwrap().is_some());
    }
}
