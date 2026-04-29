//! RFC 009 PR-5 — per-stage deadline budgets.
//!
//! ## What this owns
//!
//! - [`DeadlineBudget`] — splits a total request deadline across the
//!   stages of a recall: parsing, dense vector search, expansion,
//!   merge/rerank, response assembly. Each stage gets a slice of the
//!   total. The `start` timestamp is captured at request entry; each
//!   `remaining_for_*` call returns "how much time is left for this
//!   stage given how much I've spent so far."
//! - [`run_with_deadline`] — convenience helper that races a future
//!   against `remaining_for_stage`, returning either the future's
//!   output or `Err(DeadlineExceeded)`.
//!
//! ## Why per-stage, not just total
//!
//! A 5-second total deadline doesn't help if expansion alone takes
//! 4.9 seconds — by the time merge/rerank runs, there's no time left.
//! Per-stage deadlines force each stage to be predictable individually,
//! which lets operators dashboard "which stage is the time going?".
//!
//! ## RFC 009 §rollout PR-5 numbers
//!
//! - `recall_total_deadline_ms = 5000`
//! - `expansion_deadline_ms = 2000`
//!
//! Substrate ships these as defaults; per-tenant overrides come via
//! the RFC 021 tenant-overrides substrate (#166).

use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;

/// Stages of a recall request. Used as keys in `DeadlineBudget` for
/// per-stage allocation. Pinned in tests so dashboard queries on
/// `recall_stage_duration_seconds{stage}` don't break silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecallStage {
    Parse,
    DenseSearch,
    Expansion,
    Merge,
    Response,
}

impl RecallStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            RecallStage::Parse => "parse",
            RecallStage::DenseSearch => "dense_search",
            RecallStage::Expansion => "expansion",
            RecallStage::Merge => "merge",
            RecallStage::Response => "response",
        }
    }
}

/// Configuration: total deadline + per-stage caps. Per-stage cap is
/// the *upper bound* a stage can consume; the actual remaining at run
/// time is `min(stage_cap, total_remaining)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeadlineConfig {
    pub total: Duration,
    pub parse: Duration,
    pub dense_search: Duration,
    pub expansion: Duration,
    pub merge: Duration,
    pub response: Duration,
}

impl Default for DeadlineConfig {
    fn default() -> Self {
        Self {
            total: Duration::from_millis(5000),
            parse: Duration::from_millis(50),
            dense_search: Duration::from_millis(1500),
            expansion: Duration::from_millis(2000),
            merge: Duration::from_millis(500),
            response: Duration::from_millis(200),
        }
    }
}

impl DeadlineConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.total.is_zero() {
            return Err("total deadline must be > 0");
        }
        // Sum of stage caps may exceed total — that's fine. The
        // per-stage cap is an UPPER bound; total acts as a floor.
        Ok(())
    }

    pub fn cap_for(&self, stage: RecallStage) -> Duration {
        match stage {
            RecallStage::Parse => self.parse,
            RecallStage::DenseSearch => self.dense_search,
            RecallStage::Expansion => self.expansion,
            RecallStage::Merge => self.merge,
            RecallStage::Response => self.response,
        }
    }
}

/// Per-request deadline tracker. Constructed at request entry; each
/// stage queries `remaining_for(stage)` to know how long it has.
#[derive(Debug, Clone)]
pub struct DeadlineBudget {
    cfg: DeadlineConfig,
    started_at: Instant,
}

impl DeadlineBudget {
    pub fn new(cfg: DeadlineConfig) -> Self {
        Self {
            cfg,
            started_at: Instant::now(),
        }
    }

    /// Construct with explicit start. For tests with deterministic clocks.
    pub fn started_at(cfg: DeadlineConfig, started_at: Instant) -> Self {
        Self { cfg, started_at }
    }

    pub fn config(&self) -> &DeadlineConfig {
        &self.cfg
    }

    /// Time elapsed since request start.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Remaining time before total deadline expires. Saturating: never
    /// returns negative.
    pub fn total_remaining(&self) -> Duration {
        self.cfg.total.saturating_sub(self.elapsed())
    }

    /// True iff the total budget has been consumed.
    pub fn is_expired(&self) -> bool {
        self.elapsed() >= self.cfg.total
    }

    /// How long this stage has, given the per-stage cap and the time
    /// remaining in the total. Saturating; ZERO if expired.
    pub fn remaining_for(&self, stage: RecallStage) -> Duration {
        let total_left = self.total_remaining();
        let cap = self.cfg.cap_for(stage);
        total_left.min(cap)
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DeadlineError {
    #[error("stage `{stage}` deadline exceeded after {elapsed_ms}ms")]
    Exceeded { stage: &'static str, elapsed_ms: u64 },
}

/// Race a future against the stage's remaining deadline. Returns
/// `Ok(value)` on success, `Err(Exceeded)` on timeout. Cancellation
/// tokens compose externally — the caller wires `CancellationToken`
/// for breaker-open / client-disconnect cancellation, separately from
/// this deadline race.
pub async fn run_with_deadline<F, T>(
    budget: &DeadlineBudget,
    stage: RecallStage,
    fut: F,
) -> Result<T, DeadlineError>
where
    F: std::future::Future<Output = T>,
{
    let remaining = budget.remaining_for(stage);
    if remaining.is_zero() {
        return Err(DeadlineError::Exceeded {
            stage: stage.as_str(),
            elapsed_ms: budget.elapsed().as_millis() as u64,
        });
    }
    match tokio::time::timeout(remaining, fut).await {
        Ok(v) => Ok(v),
        Err(_) => Err(DeadlineError::Exceeded {
            stage: stage.as_str(),
            elapsed_ms: budget.elapsed().as_millis() as u64,
        }),
    }
}

/// Race a future against either the deadline OR a cancellation token.
/// Returns:
/// - `Ok(Ok(value))` — the future completed.
/// - `Ok(Err(Cancelled))` — the cancellation token fired.
/// - `Err(Exceeded)` — the stage deadline elapsed.
///
/// Used by the data plane when the breaker opens and we want to drain
/// in-flight expensive recalls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

pub async fn run_with_deadline_or_cancel<F, T>(
    budget: &DeadlineBudget,
    stage: RecallStage,
    cancel: &CancellationToken,
    fut: F,
) -> Result<Result<T, Cancelled>, DeadlineError>
where
    F: std::future::Future<Output = T>,
{
    let remaining = budget.remaining_for(stage);
    if remaining.is_zero() {
        return Err(DeadlineError::Exceeded {
            stage: stage.as_str(),
            elapsed_ms: budget.elapsed().as_millis() as u64,
        });
    }
    tokio::select! {
        _ = cancel.cancelled() => Ok(Err(Cancelled)),
        v = tokio::time::timeout(remaining, fut) => match v {
            Ok(value) => Ok(Ok(value)),
            Err(_) => Err(DeadlineError::Exceeded {
                stage: stage.as_str(),
                elapsed_ms: budget.elapsed().as_millis() as u64,
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_strings_pinned() {
        assert_eq!(RecallStage::Parse.as_str(), "parse");
        assert_eq!(RecallStage::DenseSearch.as_str(), "dense_search");
        assert_eq!(RecallStage::Expansion.as_str(), "expansion");
        assert_eq!(RecallStage::Merge.as_str(), "merge");
        assert_eq!(RecallStage::Response.as_str(), "response");
    }

    #[test]
    fn default_config_matches_rfc_009_pr5() {
        let cfg = DeadlineConfig::default();
        assert_eq!(cfg.total, Duration::from_millis(5000));
        assert_eq!(cfg.expansion, Duration::from_millis(2000));
    }

    #[test]
    fn validate_rejects_zero_total() {
        let cfg = DeadlineConfig {
            total: Duration::ZERO,
            ..DeadlineConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn budget_remaining_for_returns_min_of_cap_and_total() {
        let started = Instant::now();
        let cfg = DeadlineConfig {
            total: Duration::from_millis(100),
            expansion: Duration::from_millis(2000),
            ..DeadlineConfig::default()
        };
        let b = DeadlineBudget::started_at(cfg, started);
        // total=100ms, expansion=2000ms → min = 100ms.
        let r = b.remaining_for(RecallStage::Expansion);
        // Allow some elapsed slack for test scheduling.
        assert!(r <= Duration::from_millis(100));
        assert!(r > Duration::ZERO);
    }

    #[test]
    fn budget_remaining_zero_when_total_consumed() {
        let started = Instant::now() - Duration::from_secs(10);
        let cfg = DeadlineConfig::default();
        let b = DeadlineBudget::started_at(cfg, started);
        assert!(b.is_expired());
        assert_eq!(b.total_remaining(), Duration::ZERO);
        assert_eq!(b.remaining_for(RecallStage::Expansion), Duration::ZERO);
    }

    #[tokio::test]
    async fn run_with_deadline_returns_value_when_fast() {
        let b = DeadlineBudget::new(DeadlineConfig::default());
        let v: u32 = run_with_deadline(&b, RecallStage::Parse, async { 42 })
            .await
            .unwrap();
        assert_eq!(v, 42);
    }

    #[tokio::test]
    async fn run_with_deadline_times_out_when_slow() {
        let cfg = DeadlineConfig {
            total: Duration::from_millis(50),
            parse: Duration::from_millis(50),
            ..DeadlineConfig::default()
        };
        let b = DeadlineBudget::new(cfg);
        let result: Result<u32, _> = run_with_deadline(&b, RecallStage::Parse, async {
            tokio::time::sleep(Duration::from_millis(500)).await;
            42
        })
        .await;
        let err = result.unwrap_err();
        assert!(matches!(err, DeadlineError::Exceeded { stage: "parse", .. }));
    }

    #[tokio::test]
    async fn run_with_deadline_returns_exceeded_immediately_when_already_expired() {
        let started = Instant::now() - Duration::from_secs(10);
        let cfg = DeadlineConfig::default();
        let b = DeadlineBudget::started_at(cfg, started);
        let result: Result<u32, _> = run_with_deadline(&b, RecallStage::Parse, async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            42
        })
        .await;
        assert!(matches!(result, Err(DeadlineError::Exceeded { .. })));
    }

    #[tokio::test]
    async fn run_with_deadline_or_cancel_completes_normally() {
        let b = DeadlineBudget::new(DeadlineConfig::default());
        let token = CancellationToken::new();
        let result: Result<Result<u32, _>, _> =
            run_with_deadline_or_cancel(&b, RecallStage::Parse, &token, async { 42 }).await;
        let inner = result.unwrap();
        assert_eq!(inner.unwrap(), 42);
    }

    #[tokio::test]
    async fn run_with_deadline_or_cancel_cancels_when_token_fires() {
        let b = DeadlineBudget::new(DeadlineConfig::default());
        let token = CancellationToken::new();
        let token_clone = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            token_clone.cancel();
        });
        let result: Result<Result<u32, _>, _> = run_with_deadline_or_cancel(
            &b,
            RecallStage::Expansion,
            &token,
            async {
                tokio::time::sleep(Duration::from_secs(10)).await;
                42
            },
        )
        .await;
        let inner = result.unwrap();
        assert!(inner.is_err()); // Cancelled
    }

    #[tokio::test]
    async fn run_with_deadline_or_cancel_times_out_independently_of_token() {
        let cfg = DeadlineConfig {
            total: Duration::from_millis(50),
            parse: Duration::from_millis(50),
            ..DeadlineConfig::default()
        };
        let b = DeadlineBudget::new(cfg);
        let token = CancellationToken::new();
        let result: Result<Result<u32, _>, _> = run_with_deadline_or_cancel(
            &b,
            RecallStage::Parse,
            &token,
            async {
                tokio::time::sleep(Duration::from_secs(10)).await;
                42
            },
        )
        .await;
        // Deadline fires before token does → outer Err.
        assert!(matches!(result, Err(DeadlineError::Exceeded { .. })));
    }

    #[test]
    fn cap_for_returns_per_stage_bound() {
        let cfg = DeadlineConfig::default();
        assert_eq!(cfg.cap_for(RecallStage::Parse), Duration::from_millis(50));
        assert_eq!(cfg.cap_for(RecallStage::Expansion), Duration::from_millis(2000));
    }
}
