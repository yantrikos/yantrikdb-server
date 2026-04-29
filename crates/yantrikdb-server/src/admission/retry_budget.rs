//! RFC 009 PR-5 — retry-storm controls.
//!
//! ## What this owns
//!
//! - [`RetryBudget`] — per-tenant token-bucket-shaped state that tracks
//!   how many retries a tenant is allowed in the current window. Used
//!   to populate the `X-YDB-Retry-Budget` response header so honest
//!   clients know to back off before the server has to start
//!   rejecting.
//! - [`RetryGuidance`] — `(retry_after, retry_after_jitter, budget_remaining)`
//!   bundle the response middleware emits as headers.
//! - [`compute_retry_after`] — pure function that takes a base
//!   `Retry-After` and adds jitter (full-jitter algorithm, AWS-style)
//!   to spread retry storms.
//!
//! ## Why a budget, not just rate-limits
//!
//! Rate-limit headers tell the client "you exceeded RPS." Retry
//! budgets tell the client "you're in a degraded state and have N
//! retries before you start hitting hard rejections." Cooperative
//! clients (the AWS / GCP SDK pattern) can use this to circuit-break
//! THEMSELVES, sparing the server.
//!
//! The substrate doesn't enforce — it computes the value. Enforcement
//! happens in PR-4 staged enforcement, where a tenant in `enforce_hard`
//! mode will start receiving 429s once budget hits zero.
//!
//! ## Full-jitter
//!
//! `Retry-After` with no jitter creates synchronized retry storms.
//! Full-jitter (Marc Brooker, AWS, 2015): `actual_delay = rand(0,
//! base_delay)`. Each client picks a random delay in `[0, base]`,
//! statistically spreading load over the full window.

use std::time::{Duration, Instant};

use parking_lot::Mutex;
use rand::Rng;

/// Per-tenant retry budget configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetryBudgetConfig {
    /// Maximum retries allowed in `window`.
    pub capacity: u32,
    /// Sliding window over which the capacity applies.
    pub window: Duration,
}

impl Default for RetryBudgetConfig {
    fn default() -> Self {
        Self {
            capacity: 100,
            window: Duration::from_secs(60),
        }
    }
}

impl RetryBudgetConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.capacity == 0 {
            return Err("retry budget capacity must be > 0");
        }
        if self.window.is_zero() {
            return Err("retry budget window must be > 0");
        }
        Ok(())
    }
}

/// Sliding-window retry counter. The implementation uses a simple
/// "bucket" scheme: each window has a count; on observation, if the
/// current window has elapsed, reset to zero. Slightly less precise
/// than a true sliding window (which needs O(history) memory) but
/// uses O(1) state per tenant — appropriate at the substrate level.
#[derive(Debug)]
pub struct RetryBudget {
    cfg: RetryBudgetConfig,
    inner: Mutex<RetryBudgetState>,
}

#[derive(Debug)]
struct RetryBudgetState {
    /// Remaining retries in current window. Starts at capacity, decrements.
    remaining: u32,
    /// When the current window started.
    window_started_at: Instant,
}

impl RetryBudget {
    pub fn new(cfg: RetryBudgetConfig) -> Self {
        Self {
            cfg,
            inner: Mutex::new(RetryBudgetState {
                remaining: cfg.capacity,
                window_started_at: Instant::now(),
            }),
        }
    }

    /// Test-friendly constructor with explicit start time.
    pub fn new_at(cfg: RetryBudgetConfig, started_at: Instant) -> Self {
        Self {
            cfg,
            inner: Mutex::new(RetryBudgetState {
                remaining: cfg.capacity,
                window_started_at: started_at,
            }),
        }
    }

    /// Roll the window forward if the current one has elapsed. Always
    /// safe to call — idempotent within a window.
    fn refresh_against(state: &mut RetryBudgetState, cfg: &RetryBudgetConfig, now: Instant) {
        let elapsed = now.saturating_duration_since(state.window_started_at);
        if elapsed >= cfg.window {
            state.remaining = cfg.capacity;
            state.window_started_at = now;
        }
    }

    /// Consume one retry slot. Returns the post-decrement budget. If
    /// the budget was already 0 BEFORE this call, returns 0 (no
    /// underflow).
    pub fn consume(&self) -> u32 {
        self.consume_at(Instant::now())
    }

    pub fn consume_at(&self, now: Instant) -> u32 {
        let mut g = self.inner.lock();
        Self::refresh_against(&mut g, &self.cfg, now);
        g.remaining = g.remaining.saturating_sub(1);
        g.remaining
    }

    /// Read the current remaining budget without decrementing. Used
    /// for the header on non-retry responses.
    pub fn remaining(&self) -> u32 {
        self.remaining_at(Instant::now())
    }

    pub fn remaining_at(&self, now: Instant) -> u32 {
        let mut g = self.inner.lock();
        Self::refresh_against(&mut g, &self.cfg, now);
        g.remaining
    }

    pub fn capacity(&self) -> u32 {
        self.cfg.capacity
    }

    /// True iff a cooperative client should stop retrying. Consumers
    /// (response middleware) emit `X-YDB-Retry-Budget` with this
    /// value so honest clients can self-circuit-break.
    pub fn is_exhausted(&self) -> bool {
        self.remaining() == 0
    }
}

/// Bundle of retry-related guidance emitted on a 429/503 response.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetryGuidance {
    /// Floor on `Retry-After` — what the server suggests as a minimum.
    pub retry_after: Duration,
    /// Jittered actual `Retry-After` to suggest. Per-request randomized
    /// to spread storms.
    pub retry_after_jittered: Duration,
    /// Remaining budget to advertise via `X-YDB-Retry-Budget`.
    pub budget_remaining: u32,
}

/// Compute a jittered `Retry-After` per AWS full-jitter:
/// `jittered = uniform_random(0, base_delay)`. The base value is
/// returned alongside so consumers can include both in dashboards
/// (the floor lets you see what the server thinks the right minimum
/// retry is; the jittered value is what each request actually got).
pub fn compute_retry_after(base: Duration) -> (Duration, Duration) {
    let base_ms = base.as_millis() as u64;
    if base_ms == 0 {
        return (base, Duration::ZERO);
    }
    let jittered_ms = rand::thread_rng().gen_range(0..=base_ms);
    (base, Duration::from_millis(jittered_ms))
}

/// Build a full guidance bundle for a 429/503 response.
pub fn build_guidance(base_retry_after: Duration, budget: &RetryBudget) -> RetryGuidance {
    let (base, jittered) = compute_retry_after(base_retry_after);
    RetryGuidance {
        retry_after: base,
        retry_after_jittered: jittered,
        budget_remaining: budget.remaining(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_capacity_and_window() {
        let cfg = RetryBudgetConfig::default();
        assert_eq!(cfg.capacity, 100);
        assert_eq!(cfg.window, Duration::from_secs(60));
    }

    #[test]
    fn validate_rejects_zero_capacity() {
        let cfg = RetryBudgetConfig {
            capacity: 0,
            window: Duration::from_secs(60),
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_window() {
        let cfg = RetryBudgetConfig {
            capacity: 10,
            window: Duration::ZERO,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn fresh_budget_starts_at_capacity() {
        let b = RetryBudget::new(RetryBudgetConfig::default());
        assert_eq!(b.remaining(), 100);
        assert!(!b.is_exhausted());
    }

    #[test]
    fn consume_decrements() {
        let b = RetryBudget::new(RetryBudgetConfig {
            capacity: 3,
            window: Duration::from_secs(60),
        });
        assert_eq!(b.consume(), 2);
        assert_eq!(b.consume(), 1);
        assert_eq!(b.consume(), 0);
        // Already at 0 — saturating, no underflow.
        assert_eq!(b.consume(), 0);
        assert!(b.is_exhausted());
    }

    #[test]
    fn budget_resets_after_window_elapses() {
        let started = Instant::now();
        let b = RetryBudget::new_at(
            RetryBudgetConfig {
                capacity: 3,
                window: Duration::from_secs(10),
            },
            started,
        );
        b.consume_at(started);
        b.consume_at(started);
        b.consume_at(started);
        assert_eq!(b.remaining_at(started), 0);
        // Advance past window — bucket refills.
        let later = started + Duration::from_secs(11);
        assert_eq!(b.remaining_at(later), 3);
    }

    #[test]
    fn budget_does_not_reset_within_window() {
        let started = Instant::now();
        let b = RetryBudget::new_at(
            RetryBudgetConfig {
                capacity: 3,
                window: Duration::from_secs(10),
            },
            started,
        );
        b.consume_at(started);
        let half_window_later = started + Duration::from_secs(5);
        assert_eq!(b.remaining_at(half_window_later), 2);
    }

    #[test]
    fn capacity_accessor_returns_config_value() {
        let b = RetryBudget::new(RetryBudgetConfig {
            capacity: 50,
            window: Duration::from_secs(60),
        });
        assert_eq!(b.capacity(), 50);
    }

    #[test]
    fn is_exhausted_only_when_zero() {
        let b = RetryBudget::new(RetryBudgetConfig {
            capacity: 1,
            window: Duration::from_secs(60),
        });
        assert!(!b.is_exhausted());
        b.consume();
        assert!(b.is_exhausted());
    }

    #[test]
    fn compute_retry_after_zero_base_yields_zero() {
        let (base, jittered) = compute_retry_after(Duration::ZERO);
        assert_eq!(base, Duration::ZERO);
        assert_eq!(jittered, Duration::ZERO);
    }

    #[test]
    fn compute_retry_after_jitter_in_range() {
        // Jittered must always be in [0, base].
        let base = Duration::from_secs(2);
        for _ in 0..50 {
            let (b, j) = compute_retry_after(base);
            assert_eq!(b, base);
            assert!(j <= base);
        }
    }

    #[test]
    fn compute_retry_after_actually_jitters() {
        // Across N samples, we should see at least 2 distinct values.
        // (Deterministic equality is exceedingly unlikely with 50
        // samples in a 1000-ms range.)
        let base = Duration::from_secs(1);
        let mut values: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for _ in 0..50 {
            let (_, j) = compute_retry_after(base);
            values.insert(j.as_millis() as u64);
        }
        assert!(
            values.len() > 1,
            "expected jitter to produce variance, got {} unique values",
            values.len()
        );
    }

    #[test]
    fn build_guidance_includes_budget() {
        let b = RetryBudget::new(RetryBudgetConfig {
            capacity: 3,
            window: Duration::from_secs(60),
        });
        b.consume();
        let g = build_guidance(Duration::from_secs(1), &b);
        assert_eq!(g.budget_remaining, 2);
        assert_eq!(g.retry_after, Duration::from_secs(1));
        assert!(g.retry_after_jittered <= g.retry_after);
    }

    #[test]
    fn budget_send_sync() {
        // Must be Send+Sync to be held in Arc by middleware.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RetryBudget>();
    }
}
