//! Admission control: hard pre-admission caps + concurrency semaphores.
//!
//! Per RFC 009 §4 (hard caps layer of CPU isolation) and §2 (admission
//! control middleware). PR-1 establishes the hard caps; PR-2 wires the
//! quota / cost-budget middleware on top.
//!
//! ## Three layers of admission
//!
//! 1. **Body size cap** (HTTP middleware): rejects requests with bodies
//!    over `MAX_REQUEST_BODY_BYTES` with HTTP 413. Prevents memory blow
//!    from oversized payloads. Wired in `http_gateway::router`.
//! 2. **`top_k` clamp** (handler-level): on `recall`, requests with
//!    `top_k > HARD_TOP_K_CAP` return HTTP 400 immediately, before HNSW
//!    search runs. Prevents the term=1423 thrashing pattern at its source.
//! 3. **Concurrency semaphores** (handler-level): bounded permits gate
//!    expanded recall (`max_concurrent_expanded_recall`) and total
//!    in-flight recall (`max_in_flight_recall`). Permits acquire
//!    immediately or return HTTP 503 + `Retry-After`.
//!
//! ## Why semaphores at the handler, not at the runtime
//!
//! Runtime split + SCHED_FIFO buys us *priority isolation* (Raft tasks
//! preempt app tasks). Concurrency caps buy us *workload boundedness*
//! (no more than N expanded recalls running at once, regardless of how
//! many threads are scheduled). Both are needed: priority alone doesn't
//! prevent app threads from being CPU-bound for long stretches; caps
//! alone don't prevent priority inversion if Raft tasks have to fight
//! for cores.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;

use crate::metrics;

pub mod bucket;
pub mod circuit_breaker;
pub mod cost;
pub mod deadlines;
pub mod policy;
pub mod registry;
pub mod retry_budget;

pub use bucket::{TokenBucket, TokenBucketConfig};
pub use circuit_breaker::{
    BreakerState, BreakerVerdict, CircuitBreaker, CircuitBreakerConfig, DeploymentProfile,
    HealthObservation, OpenReason,
};
pub use cost::{cost_units, CostInputs, CostParams, DEFAULT_E_EXPAND};
pub use policy::{
    PolicyResolver, ProvisionalDefaults, QuotaPolicy, QuotaScope, ScopeKind, FALLBACK_DEFAULTS,
    PROVISIONAL_DEFAULTS,
};
pub use deadlines::{
    run_with_deadline, run_with_deadline_or_cancel, Cancelled, DeadlineBudget, DeadlineConfig,
    DeadlineError, RecallStage,
};
pub use registry::{BucketDimension, BucketKey, BucketRegistry, ConsumeOutcome};
pub use retry_budget::{
    build_guidance, compute_retry_after, RetryBudget, RetryBudgetConfig, RetryGuidance,
};

/// Hard cap on `top_k` parameter. Requests above this return 400. Prevents
/// the term=1423 thrashing pattern (clients sending top_k=200 with
/// expand_entities=true) at the source. Pollers that legitimately need
/// big-scan use cases will move to the future `/v2/recall_ids` + `/v2/expand`
/// pattern (RFC 010 Phase 2). For the hot path, this is a defensive cap.
pub const HARD_TOP_K_CAP: usize = 1000;

/// Default body size cap on HTTP requests. 64KiB handles realistic recall
/// queries (~256 bytes) and reasonable batch remember (~50 entries × 1KiB).
/// Above this is almost always a client bug or attack. Wired via
/// `tower_http::limit::RequestBodyLimitLayer`.
pub const MAX_REQUEST_BODY_BYTES: usize = 64 * 1024;

/// Default max concurrent expanded recalls per node. Sized to leave at
/// least 1 core of headroom on a typical 4-core deployment so heartbeats
/// always have somewhere to schedule. Configurable via
/// [`AdmissionConfig::max_concurrent_expanded_recall`].
pub const DEFAULT_MAX_CONCURRENT_EXPANDED_RECALL: usize = 4;

/// Default max total in-flight recalls per node (expanded + non-expanded).
/// Bounds memory blowup from queued requests under spike load.
pub const DEFAULT_MAX_IN_FLIGHT_RECALL: usize = 64;

/// How long a permit-acquire will wait before returning 503. Short — the
/// purpose of the semaphore is to shed load fast, not to queue clients.
pub const PERMIT_ACQUIRE_TIMEOUT: Duration = Duration::from_millis(100);

#[derive(Debug, Clone)]
pub struct AdmissionConfig {
    pub max_concurrent_expanded_recall: usize,
    pub max_in_flight_recall: usize,
    pub max_request_body_bytes: usize,
    pub hard_top_k_cap: usize,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            max_concurrent_expanded_recall: DEFAULT_MAX_CONCURRENT_EXPANDED_RECALL,
            max_in_flight_recall: DEFAULT_MAX_IN_FLIGHT_RECALL,
            max_request_body_bytes: MAX_REQUEST_BODY_BYTES,
            hard_top_k_cap: HARD_TOP_K_CAP,
        }
    }
}

/// Shared admission state attached to `AppState`. All semaphores are
/// `Arc<Semaphore>` so they can be cheaply cloned into per-request
/// futures.
#[derive(Clone)]
pub struct AdmissionState {
    pub cfg: Arc<AdmissionConfig>,
    /// Permit per concurrent expanded-recall request. Acquired when
    /// `expand_entities=true`, released on response.
    pub expanded_recall: Arc<Semaphore>,
    /// Permit per in-flight recall request (expanded + non-expanded).
    /// Acquired before HNSW search, released on response.
    pub in_flight_recall: Arc<Semaphore>,
}

impl AdmissionState {
    pub fn new(cfg: AdmissionConfig) -> Self {
        Self {
            expanded_recall: Arc::new(Semaphore::new(cfg.max_concurrent_expanded_recall)),
            in_flight_recall: Arc::new(Semaphore::new(cfg.max_in_flight_recall)),
            cfg: Arc::new(cfg),
        }
    }

    /// Try to acquire both an in-flight permit and (if expand=true) an
    /// expanded-recall permit, with a short timeout. Returns owned permits
    /// that release on drop.
    ///
    /// Returns `Err(reason)` if either cap is saturated. The reason is
    /// suitable for the `recall_rejected_total{reason}` metric.
    pub async fn acquire_recall_permits(
        &self,
        expand_entities: bool,
    ) -> Result<RecallPermits, RejectReason> {
        // Acquire in-flight first. It's the broader cap.
        let in_flight = match tokio::time::timeout(
            PERMIT_ACQUIRE_TIMEOUT,
            self.in_flight_recall.clone().acquire_owned(),
        )
        .await
        {
            Ok(Ok(p)) => p,
            Ok(Err(_closed)) => return Err(RejectReason::ServerShutdown),
            Err(_timeout) => {
                metrics::increment_recall_rejected("in_flight_saturated");
                return Err(RejectReason::InFlightSaturated);
            }
        };

        let expanded = if expand_entities {
            match tokio::time::timeout(
                PERMIT_ACQUIRE_TIMEOUT,
                self.expanded_recall.clone().acquire_owned(),
            )
            .await
            {
                Ok(Ok(p)) => Some(p),
                Ok(Err(_closed)) => return Err(RejectReason::ServerShutdown),
                Err(_timeout) => {
                    metrics::increment_recall_rejected("expanded_saturated");
                    // in_flight permit drops here automatically
                    return Err(RejectReason::ExpandedSaturated);
                }
            }
        } else {
            None
        };

        // Update the gauge so dashboards reflect current concurrency.
        metrics::set_recall_in_flight_gauge(
            (self.cfg.max_in_flight_recall - self.in_flight_recall.available_permits()) as i64,
        );
        if expand_entities {
            metrics::set_expansion_concurrent_gauge(
                (self.cfg.max_concurrent_expanded_recall
                    - self.expanded_recall.available_permits()) as i64,
            );
        }

        Ok(RecallPermits {
            _in_flight: in_flight,
            _expanded: expanded,
        })
    }
}

/// RAII guard for held recall permits. Drop releases.
pub struct RecallPermits {
    _in_flight: tokio::sync::OwnedSemaphorePermit,
    _expanded: Option<tokio::sync::OwnedSemaphorePermit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// `top_k` exceeded the hard cap.
    TopKTooLarge,
    /// In-flight recall semaphore saturated.
    InFlightSaturated,
    /// Expanded-recall semaphore saturated.
    ExpandedSaturated,
    /// Body size cap exceeded.
    BodyTooLarge,
    /// Server is shutting down.
    ServerShutdown,
}

impl RejectReason {
    /// Stable label for metrics — never include user data here.
    pub fn metric_label(&self) -> &'static str {
        match self {
            RejectReason::TopKTooLarge => "top_k_cap",
            RejectReason::InFlightSaturated => "in_flight_saturated",
            RejectReason::ExpandedSaturated => "expanded_saturated",
            RejectReason::BodyTooLarge => "body_too_large",
            RejectReason::ServerShutdown => "server_shutdown",
        }
    }

    /// Human-readable reason for the response payload.
    pub fn message(&self) -> &'static str {
        match self {
            RejectReason::TopKTooLarge => {
                "top_k exceeds hard cap; reduce top_k or use the v2 scan endpoint when available"
            }
            RejectReason::InFlightSaturated => {
                "server in-flight recall capacity exhausted; retry after a short backoff"
            }
            RejectReason::ExpandedSaturated => {
                "server expanded-recall capacity exhausted; retry, or set expand_entities=false for cheap recall"
            }
            RejectReason::BodyTooLarge => "request body exceeds limit",
            RejectReason::ServerShutdown => "server shutting down",
        }
    }

    /// Suggested HTTP status code for this rejection.
    pub fn http_status(&self) -> u16 {
        match self {
            RejectReason::TopKTooLarge | RejectReason::BodyTooLarge => 400,
            RejectReason::InFlightSaturated | RejectReason::ExpandedSaturated => 503,
            RejectReason::ServerShutdown => 503,
        }
    }
}

/// Validate `top_k` against the hard cap. Returns `Err(TopKTooLarge)` if
/// over. Increment the rejection counter as a side effect so metrics
/// reflect every rejected request, not just permitted ones.
pub fn check_top_k(top_k: usize, cap: usize) -> Result<(), RejectReason> {
    if top_k > cap {
        metrics::increment_recall_rejected("top_k_cap");
        return Err(RejectReason::TopKTooLarge);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_top_k_accepts_within_cap() {
        assert!(check_top_k(10, HARD_TOP_K_CAP).is_ok());
        assert!(check_top_k(HARD_TOP_K_CAP, HARD_TOP_K_CAP).is_ok());
    }

    #[test]
    fn check_top_k_rejects_above_cap() {
        let err = check_top_k(HARD_TOP_K_CAP + 1, HARD_TOP_K_CAP).unwrap_err();
        assert_eq!(err, RejectReason::TopKTooLarge);
        assert_eq!(err.http_status(), 400);
    }

    #[test]
    fn reject_reason_metric_labels_are_stable() {
        // Stability matters: dashboards key on these strings. If you change
        // them, you break grafana queries silently. This test pins them.
        assert_eq!(RejectReason::TopKTooLarge.metric_label(), "top_k_cap");
        assert_eq!(
            RejectReason::InFlightSaturated.metric_label(),
            "in_flight_saturated"
        );
        assert_eq!(
            RejectReason::ExpandedSaturated.metric_label(),
            "expanded_saturated"
        );
        assert_eq!(RejectReason::BodyTooLarge.metric_label(), "body_too_large");
        assert_eq!(RejectReason::ServerShutdown.metric_label(), "server_shutdown");
    }

    #[tokio::test]
    async fn acquire_permits_succeeds_within_cap() {
        let st = AdmissionState::new(AdmissionConfig::default());
        let p1 = st.acquire_recall_permits(true).await.unwrap();
        let p2 = st.acquire_recall_permits(false).await.unwrap();
        drop(p1);
        drop(p2);
    }

    #[tokio::test]
    async fn acquire_permits_rejects_when_expanded_saturated() {
        let cfg = AdmissionConfig {
            max_concurrent_expanded_recall: 1,
            max_in_flight_recall: 8,
            ..Default::default()
        };
        let st = AdmissionState::new(cfg);

        let _hold = st.acquire_recall_permits(true).await.unwrap();
        let result = st.acquire_recall_permits(true).await;
        assert!(matches!(result, Err(RejectReason::ExpandedSaturated)));
    }

    #[tokio::test]
    async fn acquire_permits_rejects_when_in_flight_saturated() {
        let cfg = AdmissionConfig {
            max_concurrent_expanded_recall: 8,
            max_in_flight_recall: 1,
            ..Default::default()
        };
        let st = AdmissionState::new(cfg);

        let _hold = st.acquire_recall_permits(false).await.unwrap();
        let result = st.acquire_recall_permits(false).await;
        assert!(matches!(result, Err(RejectReason::InFlightSaturated)));
    }
}
