//! RFC 009 PR-2 — token-bucket primitive.
//!
//! Per RFC 009 §2 ("Admission control"):
//! - Per-principal RPS — req/s rate limit.
//! - Per-principal cost-budget — cost-units/s.
//!
//! Both are token buckets with the same shape but different "tokens"
//! (request count vs cost units). This module implements the bucket
//! primitive once; [`crate::admission::registry`] composes it per-scope
//! per-dimension.
//!
//! ## Why monotonic clock
//!
//! Wall-clock skew (NTP step, leap second) can move time backwards.
//! Bucket refill computed against backwards time would either over-
//! or under-credit, depending on direction. We use [`Instant`] which
//! is monotonic on all supported platforms — insulated from skew.
//! Documented in RFC 009 §6 ("Failure modes").
//!
//! ## Why hand-rolled instead of `governor`
//!
//! `governor` is a fine library, but adopting it would add:
//! - A non-trivial dep in our hot-path admission code.
//! - A tagging API (`KeyedRateLimiter`) that's harder to compose with
//!   our two-dimension (RPS + cost) scheme.
//! - GCRA semantics that don't quite match the simple refill model the
//!   RFC documents.
//!
//! A hand-rolled token bucket is ~50 lines, fully testable, and gives
//! us the exact contract the RFC committed to operators.
//!
//! ## Startup warm fraction
//!
//! Per RFC §2 ("Restart resets bucket state"), a rolling restart can
//! synchronize the fleet and create a refilled-burst surge. The bucket
//! starts at `capacity * startup_warm_fraction` (default 0.25) instead
//! of full capacity. This prevents post-restart thundering herd while
//! still permitting reasonable burst from clients that have been
//! waiting through the restart.

use std::time::{Duration, Instant};

/// Token-bucket configuration. The same struct works for both RPS
/// (refill_rate=N tokens/s where 1 token = 1 request) and cost-budget
/// (refill_rate=N tokens/s where 1 token = 1 cost unit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenBucketConfig {
    /// Maximum tokens in the bucket. This is the burst size.
    pub capacity: u64,
    /// Tokens added per second. The sustained rate.
    pub refill_per_sec: u64,
    /// Fraction of capacity to start with on construction. Defaults to
    /// 0.25 to mitigate rolling-restart thundering herd. Range [0.0, 1.0].
    /// Values outside the range are clamped.
    pub startup_warm_fraction: f64,
}

impl TokenBucketConfig {
    pub const DEFAULT_STARTUP_WARM_FRACTION: f64 = 0.25;

    pub fn new(capacity: u64, refill_per_sec: u64) -> Self {
        Self {
            capacity,
            refill_per_sec,
            startup_warm_fraction: Self::DEFAULT_STARTUP_WARM_FRACTION,
        }
    }

    /// Validate. capacity=0 means "always reject" which is almost
    /// certainly a config typo, so we surface it. refill_per_sec=0
    /// means "never refill" which is valid (one-time burst, never
    /// renewed) — kept allowed.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.capacity == 0 {
            return Err("token bucket capacity must be > 0");
        }
        if !self.startup_warm_fraction.is_finite() {
            return Err("startup_warm_fraction must be a finite number");
        }
        Ok(())
    }
}

/// Single token bucket. Not internally synchronized — wrap in `Mutex`
/// or use the `BucketRegistry` type which owns locking.
///
/// All time accounting uses [`Instant`] (monotonic). Refill is computed
/// lazily on each operation: `now - last_refill_at` × `refill_per_sec`,
/// clamped to `capacity`. No background tick.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    cfg: TokenBucketConfig,
    /// Current tokens. Starts at `capacity * startup_warm_fraction`.
    tokens: u64,
    /// Monotonic clock at last refill computation.
    last_refill_at: Instant,
}

impl TokenBucket {
    pub fn new(cfg: TokenBucketConfig) -> Self {
        let warm_frac = cfg.startup_warm_fraction.clamp(0.0, 1.0);
        let initial = ((cfg.capacity as f64) * warm_frac).floor() as u64;
        Self {
            tokens: initial.min(cfg.capacity),
            last_refill_at: Instant::now(),
            cfg,
        }
    }

    /// Construct with explicit initial token count and start instant.
    /// For tests that need a deterministic clock.
    pub fn new_at(cfg: TokenBucketConfig, initial_tokens: u64, now: Instant) -> Self {
        Self {
            tokens: initial_tokens.min(cfg.capacity),
            last_refill_at: now,
            cfg,
        }
    }

    /// Current configured capacity.
    pub fn capacity(&self) -> u64 {
        self.cfg.capacity
    }

    /// Tokens currently in the bucket. Calls [`Self::refill`] first so
    /// the answer reflects elapsed time at call time.
    pub fn tokens(&mut self) -> u64 {
        self.refill_now();
        self.tokens
    }

    /// Refill against current time. Idempotent — calling twice in a row
    /// without elapsed time changes nothing.
    pub fn refill_now(&mut self) {
        let now = Instant::now();
        self.refill_against(now);
    }

    /// Refill against an explicit `now`. Test entry point so the test
    /// can inject a deterministic clock.
    pub fn refill_against(&mut self, now: Instant) {
        // Use saturating_duration_since: monotonic guarantees `now >=
        // last_refill_at`, but a clone-then-replay pattern in tests
        // could reverse them. Saturating is the safe contract.
        let elapsed = now.saturating_duration_since(self.last_refill_at);
        let elapsed_secs = elapsed.as_secs_f64();
        let added = (elapsed_secs * self.cfg.refill_per_sec as f64).floor() as u64;
        if added > 0 {
            self.tokens = self.tokens.saturating_add(added).min(self.cfg.capacity);
            self.last_refill_at = now;
        }
        // If `added` rounds to zero (very small elapsed), DON'T advance
        // last_refill_at — otherwise we'd silently lose the fractional
        // refill across many short polls.
    }

    /// Try to consume `n` tokens. Returns `true` if the bucket had ≥ n,
    /// `false` if not enough (no partial consumption).
    pub fn try_consume(&mut self, n: u64) -> bool {
        self.refill_now();
        self.try_consume_no_refill(n)
    }

    /// Variant that uses an explicit `now` for the refill step. Test
    /// entry point.
    pub fn try_consume_at(&mut self, n: u64, now: Instant) -> bool {
        self.refill_against(now);
        self.try_consume_no_refill(n)
    }

    fn try_consume_no_refill(&mut self, n: u64) -> bool {
        if self.tokens >= n {
            self.tokens -= n;
            true
        } else {
            false
        }
    }

    /// Estimate when the bucket will next have at least `n` tokens
    /// available. Returns `Duration::ZERO` if already available.
    /// `None` if `n > capacity` (will never be satisfiable).
    pub fn time_until_n(&mut self, n: u64) -> Option<Duration> {
        if n > self.cfg.capacity {
            return None;
        }
        self.refill_now();
        if self.tokens >= n {
            return Some(Duration::ZERO);
        }
        if self.cfg.refill_per_sec == 0 {
            return None;
        }
        let needed = n - self.tokens;
        let secs = needed as f64 / self.cfg.refill_per_sec as f64;
        Some(Duration::from_secs_f64(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_cfg(capacity: u64, refill: u64) -> TokenBucketConfig {
        // Tests use full warm-fraction so initial=capacity unless they
        // explicitly construct via new_at.
        TokenBucketConfig {
            capacity,
            refill_per_sec: refill,
            startup_warm_fraction: 1.0,
        }
    }

    #[test]
    fn new_initializes_to_warm_fraction() {
        let cfg = TokenBucketConfig {
            capacity: 100,
            refill_per_sec: 10,
            startup_warm_fraction: 0.25,
        };
        let mut b = TokenBucket::new(cfg);
        assert_eq!(b.tokens(), 25);
    }

    #[test]
    fn warm_fraction_zero_starts_empty() {
        let cfg = TokenBucketConfig {
            capacity: 100,
            refill_per_sec: 0,
            startup_warm_fraction: 0.0,
        };
        let mut b = TokenBucket::new(cfg);
        assert_eq!(b.tokens(), 0);
    }

    #[test]
    fn warm_fraction_clamps_above_one() {
        let cfg = TokenBucketConfig {
            capacity: 100,
            refill_per_sec: 0,
            startup_warm_fraction: 5.0, // intentionally bogus
        };
        let mut b = TokenBucket::new(cfg);
        assert_eq!(b.tokens(), 100); // clamped to capacity
    }

    #[test]
    fn try_consume_succeeds_when_enough_tokens() {
        let now = Instant::now();
        let mut b = TokenBucket::new_at(mk_cfg(100, 0), 100, now);
        assert!(b.try_consume_at(40, now));
        assert!(b.try_consume_at(60, now));
        assert!(!b.try_consume_at(1, now)); // exhausted
    }

    #[test]
    fn try_consume_rejects_partial_when_under() {
        let now = Instant::now();
        let mut b = TokenBucket::new_at(mk_cfg(100, 0), 50, now);
        assert!(!b.try_consume_at(60, now));
        // Bucket unchanged.
        assert_eq!(b.tokens, 50);
    }

    #[test]
    fn refill_replenishes_at_configured_rate() {
        let t0 = Instant::now();
        let mut b = TokenBucket::new_at(mk_cfg(100, 10), 0, t0);
        // 5 seconds elapsed → 50 tokens.
        let t1 = t0 + Duration::from_secs(5);
        b.refill_against(t1);
        assert_eq!(b.tokens, 50);
    }

    #[test]
    fn refill_clamps_at_capacity() {
        let t0 = Instant::now();
        let mut b = TokenBucket::new_at(mk_cfg(100, 100), 90, t0);
        // 10 seconds × 100/s = 1000, but capped at capacity=100.
        let t1 = t0 + Duration::from_secs(10);
        b.refill_against(t1);
        assert_eq!(b.tokens, 100);
    }

    #[test]
    fn refill_zero_rate_never_replenishes() {
        let t0 = Instant::now();
        let mut b = TokenBucket::new_at(mk_cfg(100, 0), 30, t0);
        let t1 = t0 + Duration::from_secs(60);
        b.refill_against(t1);
        assert_eq!(b.tokens, 30);
    }

    #[test]
    fn time_until_n_zero_when_already_available() {
        let now = Instant::now();
        let mut b = TokenBucket::new_at(mk_cfg(100, 10), 100, now);
        assert_eq!(b.time_until_n(50), Some(Duration::ZERO));
    }

    #[test]
    fn time_until_n_estimates_refill_wait() {
        let now = Instant::now();
        let mut b = TokenBucket::new_at(mk_cfg(100, 10), 0, now);
        // Need 50 tokens, refill at 10/s → 5s wait.
        let wait = b.time_until_n(50).unwrap();
        assert!((wait.as_secs_f64() - 5.0).abs() < 0.01);
    }

    #[test]
    fn time_until_n_none_when_n_exceeds_capacity() {
        let now = Instant::now();
        let mut b = TokenBucket::new_at(mk_cfg(100, 10), 100, now);
        assert_eq!(b.time_until_n(200), None);
    }

    #[test]
    fn time_until_n_none_when_zero_refill_and_under() {
        let now = Instant::now();
        let mut b = TokenBucket::new_at(mk_cfg(100, 0), 10, now);
        assert_eq!(b.time_until_n(50), None);
    }

    #[test]
    fn validate_rejects_zero_capacity() {
        let cfg = TokenBucketConfig {
            capacity: 0,
            refill_per_sec: 10,
            startup_warm_fraction: 0.25,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_zero_refill() {
        // One-time burst, never renewed — valid use case for short-lived
        // promotional or one-shot windows.
        let cfg = TokenBucketConfig {
            capacity: 10,
            refill_per_sec: 0,
            startup_warm_fraction: 0.25,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn refill_idempotent_with_no_elapsed_time() {
        let now = Instant::now();
        let mut b = TokenBucket::new_at(mk_cfg(100, 10), 50, now);
        b.refill_against(now);
        b.refill_against(now);
        assert_eq!(b.tokens, 50);
    }

    #[test]
    fn fractional_refill_does_not_lose_progress_across_short_polls() {
        // refill_per_sec=10 means 1 token per 100ms. If we poll every
        // 50ms, integer floor would drop the fractional refill on each
        // poll — accumulating to zero forever. The `last_refill_at`
        // hold-back logic prevents this: refill_against only advances
        // the "last refill" timestamp once at least 1 token was added.
        let t0 = Instant::now();
        let mut b = TokenBucket::new_at(mk_cfg(100, 10), 0, t0);
        let mut t = t0;
        for _ in 0..20 {
            t += Duration::from_millis(50);
            b.refill_against(t);
        }
        // 1 second elapsed, 10 tokens/s → 10 tokens accumulated. Not 0.
        assert_eq!(b.tokens, 10);
    }
}
