//! RFC 009 PR-2 — bucket registry.
//!
//! Per-scope, per-dimension token-bucket store. The middleware (PR-2
//! consumer, deferred) calls [`BucketRegistry::consume`] for each
//! enforcement dimension on each admitted request, observes the
//! [`ConsumeOutcome`], and emits `quota_consumption{scope, dimension}`
//! metrics. In SHADOW mode the outcome is observed only; nothing is
//! actually rejected.
//!
//! ## Lazy-on-first-request backfill
//!
//! Per RFC §6 ("Tenant backfill"), policy rows are NOT pre-populated
//! at rollout. The first request from a never-before-seen principal
//! triggers creation of a row at `PROVISIONAL_DEFAULTS`. The registry
//! mirrors that pattern: the bucket for a scope is materialized on
//! first `consume` call.
//!
//! ## Concurrency
//!
//! Bucket operations are short and CPU-bound. We use a single
//! `parking_lot::Mutex` over the inner map — no lock-free trickery.
//! Per-scope buckets are stored as `Arc<Mutex<TokenBucket>>` so a
//! consume call holds only the per-scope lock during refill+consume,
//! not the registry lock.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use super::bucket::{TokenBucket, TokenBucketConfig};
use super::policy::{QuotaPolicy, QuotaScope};

/// Which dimension a bucket counts. Each scope has independent buckets
/// for each dimension because they refill at different rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BucketDimension {
    /// Requests-per-second. Each request consumes 1 token.
    Rps,
    /// Cost-units-per-second. Each request consumes [`super::cost::cost_units`].
    Cost,
}

impl BucketDimension {
    pub fn as_str(self) -> &'static str {
        match self {
            BucketDimension::Rps => "rps",
            BucketDimension::Cost => "cost",
        }
    }
}

/// Registry key: scope + dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BucketKey {
    pub scope: QuotaScope,
    pub dimension: BucketDimension,
}

/// Outcome of a [`BucketRegistry::consume`] call. SHADOW mode handlers
/// emit metrics from this outcome but never reject; ENFORCE mode (PR-4,
/// future) translates `Rejected` into HTTP 429.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsumeOutcome {
    /// Tokens were consumed. Bucket had enough.
    Allowed {
        /// Tokens remaining after this consume. For
        /// `X-YDB-RateLimit-Remaining` and dashboards.
        remaining: u64,
    },
    /// Bucket did not have enough tokens. In SHADOW mode this is
    /// observed but not enforced; in ENFORCE mode it becomes 429.
    Rejected {
        /// Estimated wait until refill provides enough tokens. Used
        /// for `Retry-After` (PR-4 future) and dashboards.
        retry_after: Option<Duration>,
        /// Tokens at time of rejection (always < requested).
        tokens_available: u64,
        /// Tokens that were requested.
        tokens_requested: u64,
    },
    /// The dimension is not configured at this scope (policy dimension
    /// is `None`). Pass-through; counts as `Allowed { remaining: u64::MAX }`
    /// for accounting.
    NotConfigured,
}

/// In-memory bucket store. Constructed once at server start, cloned
/// (cheap — Arc) into the middleware layer.
#[derive(Clone, Default)]
pub struct BucketRegistry {
    inner: Arc<Mutex<HashMap<BucketKey, Arc<Mutex<TokenBucket>>>>>,
}

impl BucketRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of materialized buckets. For `quota_buckets_active` gauge
    /// and tests.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop all buckets. Used by config-reload to flush state when
    /// global tuning changes (PR-4 future). Tests use it for isolation.
    pub fn clear(&self) {
        self.inner.lock().clear();
    }

    /// Compute the consume outcome against the policy + dimension.
    /// Materializes the bucket on first call. `tokens` is the cost of
    /// this request in the dimension's unit (1 for RPS, [`super::cost::cost_units`]
    /// for Cost).
    ///
    /// SHADOW vs ENFORCE: this method has no opinion. It returns the
    /// outcome; callers decide whether to reject.
    pub fn consume(
        &self,
        scope: QuotaScope,
        dimension: BucketDimension,
        tokens: u64,
        policy: &QuotaPolicy,
    ) -> ConsumeOutcome {
        let limit = match dimension {
            BucketDimension::Rps => policy.rps_limit,
            BucketDimension::Cost => policy.cost_budget_per_sec,
        };
        let Some(rate) = limit else {
            return ConsumeOutcome::NotConfigured;
        };

        // Capacity = burst budget. Match it to refill rate so the bucket
        // can sustain a 1-second burst at full rate. Future PR can split
        // capacity from rate if operators ask for finer-grained burst
        // control.
        let cfg = TokenBucketConfig::new(rate.max(1), rate);

        let key = BucketKey { scope, dimension };
        let bucket_arc = {
            let mut guard = self.inner.lock();
            guard
                .entry(key)
                .or_insert_with(|| Arc::new(Mutex::new(TokenBucket::new(cfg))))
                .clone()
        };

        let mut bucket = bucket_arc.lock();
        if bucket.try_consume(tokens) {
            ConsumeOutcome::Allowed {
                remaining: bucket.tokens(),
            }
        } else {
            let retry_after = bucket.time_until_n(tokens);
            ConsumeOutcome::Rejected {
                retry_after,
                tokens_available: bucket.tokens(),
                tokens_requested: tokens,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::policy::PROVISIONAL_DEFAULTS;

    fn provisional_principal(id: &str) -> QuotaPolicy {
        QuotaPolicy::provisional_for(QuotaScope::principal(id))
    }

    #[test]
    fn empty_registry_starts_with_no_buckets() {
        let r = BucketRegistry::new();
        assert!(r.is_empty());
    }

    #[test]
    fn first_consume_materializes_bucket() {
        let r = BucketRegistry::new();
        let pol = provisional_principal("alice");
        let _ = r.consume(
            QuotaScope::principal("alice"),
            BucketDimension::Rps,
            1,
            &pol,
        );
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn rps_dimension_allows_within_limit() {
        let r = BucketRegistry::new();
        let pol = provisional_principal("alice");
        // PROVISIONAL_DEFAULTS.rps = 100 → first request always passes.
        let outcome = r.consume(
            QuotaScope::principal("alice"),
            BucketDimension::Rps,
            1,
            &pol,
        );
        assert!(matches!(outcome, ConsumeOutcome::Allowed { .. }));
    }

    #[test]
    fn rps_dimension_rejects_when_burst_exhausted() {
        let r = BucketRegistry::new();
        let pol = QuotaPolicy {
            rps_limit: Some(3),
            ..provisional_principal("alice")
        };
        let scope = QuotaScope::principal("alice");
        // Bucket starts at warm fraction × capacity = 25% × 3 = 0.75 →
        // floor = 0. First consume even of 1 fails. So pre-fill by
        // simulating a bigger bucket via cost dimension instead.
        // For this test, use rps with capacity high enough to start
        // non-empty, but configured limit small enough to exhaust:
        let pol = QuotaPolicy {
            rps_limit: Some(10),
            ..provisional_principal("alice")
        };
        // 10 capacity × 0.25 warm = 2 tokens initially.
        let mut allowed = 0;
        let mut rejected = 0;
        for _ in 0..5 {
            match r.consume(scope.clone(), BucketDimension::Rps, 1, &pol) {
                ConsumeOutcome::Allowed { .. } => allowed += 1,
                ConsumeOutcome::Rejected { .. } => rejected += 1,
                ConsumeOutcome::NotConfigured => panic!("dimension is configured"),
            }
        }
        assert!(allowed >= 1, "at least the warm tokens should pass");
        assert!(rejected >= 1, "exhausting the warm tokens should reject");
    }

    #[test]
    fn cost_dimension_uses_cost_budget_field() {
        let r = BucketRegistry::new();
        let pol = QuotaPolicy {
            cost_budget_per_sec: Some(20),
            ..provisional_principal("alice")
        };
        // 20 capacity × 0.25 warm = 5 tokens initially. Cost of 5 passes,
        // next cost-of-5 fails.
        let scope = QuotaScope::principal("alice");
        let first = r.consume(scope.clone(), BucketDimension::Cost, 5, &pol);
        assert!(matches!(first, ConsumeOutcome::Allowed { .. }));
        let second = r.consume(scope.clone(), BucketDimension::Cost, 5, &pol);
        assert!(matches!(second, ConsumeOutcome::Rejected { .. }));
    }

    #[test]
    fn rps_and_cost_buckets_are_independent() {
        let r = BucketRegistry::new();
        let pol = provisional_principal("alice");
        let scope = QuotaScope::principal("alice");
        let _ = r.consume(scope.clone(), BucketDimension::Rps, 1, &pol);
        let _ = r.consume(scope.clone(), BucketDimension::Cost, 5, &pol);
        // One scope × two dimensions = two distinct buckets.
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn dimension_not_configured_is_pass_through() {
        let r = BucketRegistry::new();
        let pol = QuotaPolicy {
            rps_limit: None,
            cost_budget_per_sec: Some(100),
            ..provisional_principal("alice")
        };
        let outcome = r.consume(
            QuotaScope::principal("alice"),
            BucketDimension::Rps,
            1,
            &pol,
        );
        assert_eq!(outcome, ConsumeOutcome::NotConfigured);
        // No bucket materialized for unconfigured dimension.
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn different_scopes_have_independent_buckets() {
        let r = BucketRegistry::new();
        let pol_a = provisional_principal("alice");
        let pol_b = provisional_principal("bob");
        let _ = r.consume(
            QuotaScope::principal("alice"),
            BucketDimension::Rps,
            1,
            &pol_a,
        );
        let _ = r.consume(
            QuotaScope::principal("bob"),
            BucketDimension::Rps,
            1,
            &pol_b,
        );
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn rejected_outcome_includes_diagnostics() {
        let r = BucketRegistry::new();
        let pol = QuotaPolicy {
            rps_limit: Some(10),
            ..provisional_principal("alice")
        };
        // Drain the warm fraction first.
        for _ in 0..10 {
            let _ = r.consume(
                QuotaScope::principal("alice"),
                BucketDimension::Rps,
                1,
                &pol,
            );
        }
        let outcome = r.consume(
            QuotaScope::principal("alice"),
            BucketDimension::Rps,
            100,
            &pol,
        );
        match outcome {
            ConsumeOutcome::Rejected {
                retry_after,
                tokens_available,
                tokens_requested,
            } => {
                assert_eq!(tokens_requested, 100);
                assert!(tokens_available < 100);
                // Retry-after is None because requested > capacity (10).
                assert!(retry_after.is_none());
            }
            other => panic!("expected Rejected, got {:?}", other),
        }
    }

    #[test]
    fn clear_drops_all_buckets() {
        let r = BucketRegistry::new();
        let pol = provisional_principal("alice");
        let _ = r.consume(
            QuotaScope::principal("alice"),
            BucketDimension::Rps,
            1,
            &pol,
        );
        let _ = r.consume(QuotaScope::namespace("ns1"), BucketDimension::Cost, 1, &pol);
        assert!(r.len() >= 1);
        r.clear();
        assert!(r.is_empty());
    }

    #[test]
    fn registry_is_clone_cheap() {
        // Regression guard: registry is `Clone` only if Arc<Mutex<_>>
        // is the storage. If someone refactors to a non-shared store,
        // this test fails to compile.
        let r = BucketRegistry::new();
        let r2 = r.clone();
        let pol = provisional_principal("alice");
        let _ = r.consume(
            QuotaScope::principal("alice"),
            BucketDimension::Rps,
            1,
            &pol,
        );
        // r2 sees the same bucket because Arc<Mutex<_>> is shared.
        assert_eq!(r2.len(), 1);
    }

    #[test]
    fn provisional_defaults_yield_well_formed_buckets() {
        // Smoke: PROVISIONAL_DEFAULTS produces a usable policy + bucket
        // pair end-to-end.
        let r = BucketRegistry::new();
        let pol = QuotaPolicy::provisional_for(QuotaScope::global());
        let outcome = r.consume(QuotaScope::global(), BucketDimension::Cost, 100, &pol);
        // 1000 budget × 25% warm = 250 tokens initially → cost 100 passes.
        match outcome {
            ConsumeOutcome::Allowed { remaining } => {
                assert!(remaining < PROVISIONAL_DEFAULTS.cost_budget);
            }
            other => panic!("expected Allowed, got {:?}", other),
        }
    }
}
