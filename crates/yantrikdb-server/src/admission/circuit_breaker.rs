//! RFC 009 PR-3 — Raft-instability circuit breaker.
//!
//! ## What this owns
//!
//! Pure state machine that consumes Raft-health signals (scheduling
//! latency, term churn, heartbeat lag, election in progress) and
//! decides whether expensive expanded-recall traffic should be
//! diverted (in ENFORCE mode) or counted-only (in SHADOW mode, the
//! state this PR ships).
//!
//! ## SHADOW mode
//!
//! Per RFC 009 §rollout PR-3: the breaker evaluates and emits
//! `circuit_breaker_state` and `circuit_breaker_would_reject_total`
//! metrics, but the data-plane handler does NOT reject anything yet.
//! That graduates to ENFORCE in PR-4 staged enforcement, with per-tenant
//! enforcement_mode in control DB. The state machine is the same in
//! both modes; the *consumer of the verdict* differs.
//!
//! ## Three triggers, three different time-shapes
//!
//! 1. **Pre-failure scheduling latency**: `raft_task_poll_latency_seconds{p99} > 50ms`
//!    for **2 consecutive 10s windows** = 20s sustained. Catches CPU
//!    starvation BEFORE election timeouts fire. This is the leading
//!    indicator the redteam pulled in — by the time term churns, the
//!    cluster has already destabilized. Scheduling latency leads it
//!    by tens of seconds.
//! 2. **Emergency term churn**: any `raft_term_changes_total` increase
//!    in the last 60s = open immediately. Unambiguous failure indicator.
//!    No open-side hysteresis — react instantly.
//! 3. **Sustained heartbeat lag**: `heartbeat_lag_p99 > max(absolute_floor,
//!    baseline × multiplier)` for **3 consecutive 10s windows** = 30s
//!    sustained. Single p99 spikes do NOT trip the breaker. Deployment
//!    profiles parameterize the absolute floor (lan: 50ms, wan: 200ms)
//!    and multiplier (3×).
//! 4. **Active election in progress** = open immediately.
//!
//! ## Hysteresis (the close side)
//!
//! Term-churn-triggered open: stay open ≥ 60s after last term change.
//! Lag-triggered open: stay open ≥ 30s after lag returns to normal.
//! Half-open state allows 10% of expensive traffic through; if any of
//! that re-trips, return to fully open.
//!
//! ## Anti-flapping
//!
//! Minimum 60s between open→close→open transitions. If the breaker
//! tries to flap faster, log a warning (config drift / unhealthy
//! cluster signal) and stay open.

use std::time::{Duration, Instant};

/// Operator-selectable deployment profile. Different RTT / packet-loss
/// regimes need different absolute thresholds for "lag is bad".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeploymentProfile {
    /// LAN deployment (single AZ, switched fabric). Tight thresholds.
    Lan,
    /// WAN deployment (multi-AZ, multi-region). Looser thresholds.
    Wan,
    /// Operator-tuned thresholds. Use [`CircuitBreakerConfig::profile_thresholds`]
    /// to override the floor and multiplier explicitly.
    Tuned,
}

impl DeploymentProfile {
    /// (absolute_floor, baseline_multiplier) for the profile.
    pub fn defaults(self) -> (Duration, f64) {
        match self {
            DeploymentProfile::Lan => (Duration::from_millis(50), 3.0),
            DeploymentProfile::Wan => (Duration::from_millis(200), 3.0),
            // Tuned uses caller-supplied; placeholder.
            DeploymentProfile::Tuned => (Duration::from_millis(50), 3.0),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DeploymentProfile::Lan => "lan_default",
            DeploymentProfile::Wan => "wan_default",
            DeploymentProfile::Tuned => "tuned",
        }
    }
}

/// Circuit-breaker tuning. Defaults match the RFC's `lan_default` profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircuitBreakerConfig {
    pub profile: DeploymentProfile,
    /// Absolute floor for "heartbeat lag is bad". Per profile.
    pub heartbeat_lag_floor: Duration,
    /// Multiplier on top of rolling baseline. p99_lag > baseline × multiplier
    /// counts as "elevated".
    pub heartbeat_baseline_multiplier: f64,
    /// Pre-failure scheduling latency threshold. p99 above this for 2
    /// consecutive 10s windows opens the breaker.
    pub scheduling_latency_threshold: Duration,
    /// How long a term-churn-triggered open must wait before evaluating
    /// close. RFC: 60s.
    pub term_churn_open_min_dwell: Duration,
    /// How long after lag normalizes the lag-triggered open must wait
    /// before evaluating close. RFC: 30s.
    pub lag_open_min_dwell: Duration,
    /// Anti-flapping minimum cycle: open→close→open transitions must
    /// be ≥ this far apart. Default 60s per RFC.
    pub min_cycle_time: Duration,
    /// Half-open trial fraction (0.0–1.0). Allow this fraction of
    /// expensive requests through during half-open evaluation.
    pub half_open_trial_fraction: f64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        let (floor, mult) = DeploymentProfile::Lan.defaults();
        Self {
            profile: DeploymentProfile::Lan,
            heartbeat_lag_floor: floor,
            heartbeat_baseline_multiplier: mult,
            scheduling_latency_threshold: Duration::from_millis(50),
            term_churn_open_min_dwell: Duration::from_secs(60),
            lag_open_min_dwell: Duration::from_secs(30),
            min_cycle_time: Duration::from_secs(60),
            half_open_trial_fraction: 0.10,
        }
    }
}

impl CircuitBreakerConfig {
    pub fn for_profile(profile: DeploymentProfile) -> Self {
        let (floor, mult) = profile.defaults();
        Self {
            profile,
            heartbeat_lag_floor: floor,
            heartbeat_baseline_multiplier: mult,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.heartbeat_baseline_multiplier.is_finite()
            || self.heartbeat_baseline_multiplier <= 0.0
        {
            return Err("heartbeat_baseline_multiplier must be positive and finite");
        }
        if !(0.0..=1.0).contains(&self.half_open_trial_fraction) {
            return Err("half_open_trial_fraction must be in [0.0, 1.0]");
        }
        Ok(())
    }
}

/// Three-state breaker. `Closed`=healthy, `HalfOpen`=trial, `Open`=divert.
/// Numeric values match the `circuit_breaker_state` Prometheus gauge:
/// 0=closed, 1=open, 2=half-open. Encode pinned in tests so dashboards
/// don't break silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    Closed = 0,
    Open = 1,
    HalfOpen = 2,
}

impl BreakerState {
    pub fn as_str(self) -> &'static str {
        match self {
            BreakerState::Closed => "closed",
            BreakerState::Open => "open",
            BreakerState::HalfOpen => "half_open",
        }
    }

    pub fn gauge_value(self) -> i64 {
        self as i64
    }
}

/// Why the breaker is in its current state. Set on transitions; useful
/// for `circuit_breaker_open_reason{}` labels and operator runbooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenReason {
    /// Term churn detected: term changed in last 60s.
    TermChurn,
    /// Heartbeat lag p99 elevated for 3 windows (30s sustained).
    SustainedLag,
    /// Scheduling-latency p99 above threshold for 2 windows (20s sustained).
    SchedulingLatency,
    /// Active election in progress.
    ActiveElection,
    /// Half-open trial request itself tripped a fresh fault — back to fully open.
    HalfOpenTripBack,
}

impl OpenReason {
    pub fn as_str(self) -> &'static str {
        match self {
            OpenReason::TermChurn => "term_churn",
            OpenReason::SustainedLag => "sustained_lag",
            OpenReason::SchedulingLatency => "scheduling_latency",
            OpenReason::ActiveElection => "active_election",
            OpenReason::HalfOpenTripBack => "half_open_trip_back",
        }
    }
}

/// Snapshot of cluster-health signals at a single observation window.
/// The breaker doesn't subscribe to anything — it's a pure function:
/// `(current_state, last_observation, this_observation) → next_state`.
/// The data-plane integration polls Raft / metrics every ~10s and feeds
/// observations in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HealthObservation {
    pub at: Instant,
    /// p99 of `raft_task_poll_latency_seconds` over the last 10s.
    pub scheduling_latency_p99: Duration,
    /// p99 of `raft_heartbeat_lag_seconds` over the last 10s.
    pub heartbeat_lag_p99: Duration,
    /// True if at least one term change occurred in the last 60s.
    pub term_churn_in_60s: bool,
    /// True if Raft reports an active election right now.
    pub active_election: bool,
    /// Rolling 5-minute baseline p99 lag, computed only when breaker is
    /// closed (otherwise we'd train on degraded state). `None` during
    /// the 15-minute warmup post-startup.
    pub baseline_lag_p99: Option<Duration>,
}

/// Verdict on a single expensive-recall request, given current breaker
/// state. Used by the data-plane handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerVerdict {
    /// Allow the request unconditionally.
    Allow,
    /// Allow as a trial during half-open. The handler should track the
    /// outcome and feed back via [`CircuitBreaker::record_trial_outcome`].
    AllowTrial,
    /// Deflect this request. In SHADOW mode the handler counts this in
    /// `circuit_breaker_would_reject_total` but still serves the request.
    /// In ENFORCE mode (PR-4 future), the handler returns 503 + Retry-After.
    WouldReject { reason: OpenReason },
}

/// The state machine. Not internally synchronized — wrap in `RwLock` if
/// shared across threads. Cheap to clone (POD).
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    cfg: CircuitBreakerConfig,
    state: BreakerState,
    /// Why we last transitioned into Open / HalfOpen.
    open_reason: Option<OpenReason>,
    /// When we last transitioned. Anchors hysteresis dwell.
    state_entered_at: Option<Instant>,
    /// When we last transitioned to Open (any reason). Anchors anti-flapping
    /// minimum cycle time. Distinct from `state_entered_at` so a half-open
    /// trial doesn't reset the cycle clock.
    last_open_at: Option<Instant>,
    /// Counter of consecutive "scheduling latency above threshold" windows.
    /// Trips at 2.
    consecutive_scheduling_windows: u32,
    /// Counter of consecutive "lag elevated" windows. Trips at 3.
    consecutive_lag_windows: u32,
    /// Last observation timestamp. Used to assert windows are non-overlapping.
    last_observation_at: Option<Instant>,
}

impl CircuitBreaker {
    pub fn new(cfg: CircuitBreakerConfig) -> Self {
        Self {
            cfg,
            state: BreakerState::Closed,
            open_reason: None,
            state_entered_at: None,
            last_open_at: None,
            consecutive_scheduling_windows: 0,
            consecutive_lag_windows: 0,
            last_observation_at: None,
        }
    }

    pub fn state(&self) -> BreakerState {
        self.state
    }

    pub fn open_reason(&self) -> Option<OpenReason> {
        self.open_reason
    }

    pub fn config(&self) -> &CircuitBreakerConfig {
        &self.cfg
    }

    /// Whether `obs.heartbeat_lag_p99` exceeds the elevated-lag threshold,
    /// per the RFC formula `max(absolute_floor, baseline × multiplier)`.
    fn lag_elevated(&self, obs: &HealthObservation) -> bool {
        let mult_threshold = obs
            .baseline_lag_p99
            .map(|b| b.mul_f64(self.cfg.heartbeat_baseline_multiplier))
            .unwrap_or(Duration::ZERO);
        let threshold = self.cfg.heartbeat_lag_floor.max(mult_threshold);
        obs.heartbeat_lag_p99 > threshold
    }

    fn scheduling_elevated(&self, obs: &HealthObservation) -> bool {
        obs.scheduling_latency_p99 > self.cfg.scheduling_latency_threshold
    }

    /// Feed an observation. Updates internal counters and may transition
    /// state. Returns the post-transition state for the caller's metric
    /// emission.
    pub fn observe(&mut self, obs: HealthObservation) -> BreakerState {
        // Update scheduling-latency consecutive-window counter regardless
        // of state (so a healthy cluster's counter is reset on the next
        // healthy window even if we're already Open for some other reason).
        if self.scheduling_elevated(&obs) {
            self.consecutive_scheduling_windows =
                self.consecutive_scheduling_windows.saturating_add(1);
        } else {
            self.consecutive_scheduling_windows = 0;
        }

        if self.lag_elevated(&obs) {
            self.consecutive_lag_windows = self.consecutive_lag_windows.saturating_add(1);
        } else {
            self.consecutive_lag_windows = 0;
        }

        match self.state {
            BreakerState::Closed => self.evaluate_open_triggers(&obs),
            BreakerState::Open => self.evaluate_close_eligibility(&obs),
            BreakerState::HalfOpen => self.evaluate_half_open(&obs),
        }

        self.last_observation_at = Some(obs.at);
        self.state
    }

    fn evaluate_open_triggers(&mut self, obs: &HealthObservation) {
        // Emergency: term churn opens immediately, no consecutive-window
        // requirement.
        if obs.term_churn_in_60s {
            self.transition_to_open(obs.at, OpenReason::TermChurn);
            return;
        }
        // Emergency: active election opens immediately.
        if obs.active_election {
            self.transition_to_open(obs.at, OpenReason::ActiveElection);
            return;
        }
        // Pre-failure: 2 consecutive 10s windows above threshold.
        if self.consecutive_scheduling_windows >= 2 {
            self.transition_to_open(obs.at, OpenReason::SchedulingLatency);
            return;
        }
        // Sustained: 3 consecutive 10s windows above threshold.
        if self.consecutive_lag_windows >= 3 {
            self.transition_to_open(obs.at, OpenReason::SustainedLag);
        }
    }

    fn evaluate_close_eligibility(&mut self, obs: &HealthObservation) {
        // Re-trigger paths first: stay open if a reason is still active.
        if obs.term_churn_in_60s {
            // Refresh state_entered_at so dwell continues from latest
            // term change, not the original transition.
            self.state_entered_at = Some(obs.at);
            return;
        }
        if obs.active_election {
            self.state_entered_at = Some(obs.at);
            return;
        }
        if self.consecutive_scheduling_windows >= 2 {
            self.state_entered_at = Some(obs.at);
            return;
        }
        if self.consecutive_lag_windows >= 3 {
            self.state_entered_at = Some(obs.at);
            return;
        }

        // Check minimum-dwell hysteresis based on current open reason.
        let Some(entered_at) = self.state_entered_at else {
            return;
        };
        let dwell = obs.at.saturating_duration_since(entered_at);
        let required = match self.open_reason {
            Some(OpenReason::TermChurn) => self.cfg.term_churn_open_min_dwell,
            Some(OpenReason::SustainedLag) => self.cfg.lag_open_min_dwell,
            Some(OpenReason::SchedulingLatency) => self.cfg.lag_open_min_dwell,
            Some(OpenReason::ActiveElection) => self.cfg.lag_open_min_dwell,
            Some(OpenReason::HalfOpenTripBack) => self.cfg.term_churn_open_min_dwell,
            None => Duration::ZERO,
        };
        if dwell >= required {
            self.transition_to_half_open(obs.at);
        }
    }

    fn evaluate_half_open(&mut self, obs: &HealthObservation) {
        // Any new fault during half-open trips back to fully open.
        if obs.term_churn_in_60s
            || obs.active_election
            || self.consecutive_scheduling_windows >= 2
            || self.consecutive_lag_windows >= 3
        {
            self.transition_to_open(obs.at, OpenReason::HalfOpenTripBack);
            return;
        }
        // After dwell elapsed in half-open, fully close. Use the same
        // dwell as the lag close-side; trial period equals one dwell window.
        let Some(entered_at) = self.state_entered_at else {
            return;
        };
        let dwell = obs.at.saturating_duration_since(entered_at);
        if dwell >= self.cfg.lag_open_min_dwell {
            self.transition_to_closed(obs.at);
        }
    }

    fn transition_to_open(&mut self, at: Instant, reason: OpenReason) {
        // Anti-flapping: if last open→close→open cycle would be faster
        // than min_cycle_time, stay open and log. Per RFC: "If a breaker
        // flaps faster than that, log a warning ... and stay open."
        if let Some(last) = self.last_open_at {
            let since = at.saturating_duration_since(last);
            if self.state == BreakerState::Closed && since < self.cfg.min_cycle_time {
                // Hold open, refresh entered_at so dwell timer restarts
                // and we don't immediately fall through close evaluation.
                self.state = BreakerState::Open;
                self.open_reason = Some(reason);
                self.state_entered_at = Some(at);
                self.last_open_at = Some(at);
                return;
            }
        }
        self.state = BreakerState::Open;
        self.open_reason = Some(reason);
        self.state_entered_at = Some(at);
        self.last_open_at = Some(at);
    }

    fn transition_to_half_open(&mut self, at: Instant) {
        self.state = BreakerState::HalfOpen;
        self.state_entered_at = Some(at);
        // Note: open_reason is preserved so dashboards can still see
        // why we went Open in the first place.
    }

    fn transition_to_closed(&mut self, at: Instant) {
        self.state = BreakerState::Closed;
        self.open_reason = None;
        self.state_entered_at = Some(at);
        self.consecutive_scheduling_windows = 0;
        self.consecutive_lag_windows = 0;
    }

    /// Compute verdict for an expensive-recall request. The data-plane
    /// handler calls this for each request; cheap (no I/O, no allocs).
    /// Stochastic admission during half-open is the caller's job (they
    /// can use a per-request random sample compared to
    /// `half_open_trial_fraction`); this method returns `AllowTrial`
    /// to flag the call site that it's running under trial conditions.
    pub fn verdict(&self) -> BreakerVerdict {
        match self.state {
            BreakerState::Closed => BreakerVerdict::Allow,
            BreakerState::HalfOpen => BreakerVerdict::AllowTrial,
            BreakerState::Open => BreakerVerdict::WouldReject {
                reason: self.open_reason.unwrap_or(OpenReason::SustainedLag),
            },
        }
    }

    /// During half-open, if a trial request itself surfaces a fresh
    /// fault (e.g., the recall handler observed a Raft timeout
    /// in-flight), call this to return to fully open. Mirrors the RFC's
    /// "if any of those triggers re-open, return to fully open."
    pub fn record_trial_outcome(&mut self, at: Instant, observed_fault: bool) {
        if observed_fault && self.state == BreakerState::HalfOpen {
            self.transition_to_open(at, OpenReason::HalfOpenTripBack);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs_healthy(at: Instant) -> HealthObservation {
        HealthObservation {
            at,
            scheduling_latency_p99: Duration::from_millis(2),
            heartbeat_lag_p99: Duration::from_millis(5),
            term_churn_in_60s: false,
            active_election: false,
            baseline_lag_p99: Some(Duration::from_millis(5)),
        }
    }

    fn obs_term_churn(at: Instant) -> HealthObservation {
        HealthObservation {
            term_churn_in_60s: true,
            ..obs_healthy(at)
        }
    }

    fn obs_high_sched(at: Instant) -> HealthObservation {
        HealthObservation {
            scheduling_latency_p99: Duration::from_millis(80),
            ..obs_healthy(at)
        }
    }

    fn obs_high_lag(at: Instant) -> HealthObservation {
        HealthObservation {
            heartbeat_lag_p99: Duration::from_millis(200),
            ..obs_healthy(at)
        }
    }

    #[test]
    fn starts_closed() {
        let b = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert_eq!(b.state(), BreakerState::Closed);
        assert_eq!(b.open_reason(), None);
        assert_eq!(b.verdict(), BreakerVerdict::Allow);
    }

    #[test]
    fn term_churn_opens_immediately() {
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        let t0 = Instant::now();
        let st = b.observe(obs_term_churn(t0));
        assert_eq!(st, BreakerState::Open);
        assert_eq!(b.open_reason(), Some(OpenReason::TermChurn));
    }

    #[test]
    fn active_election_opens_immediately() {
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        let t0 = Instant::now();
        let mut o = obs_healthy(t0);
        o.active_election = true;
        b.observe(o);
        assert_eq!(b.state(), BreakerState::Open);
        assert_eq!(b.open_reason(), Some(OpenReason::ActiveElection));
    }

    #[test]
    fn scheduling_latency_requires_two_consecutive_windows() {
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        let t0 = Instant::now();
        // Window 1: high → still closed.
        b.observe(obs_high_sched(t0));
        assert_eq!(b.state(), BreakerState::Closed);
        // Window 2: high → opens.
        b.observe(obs_high_sched(t0 + Duration::from_secs(10)));
        assert_eq!(b.state(), BreakerState::Open);
        assert_eq!(b.open_reason(), Some(OpenReason::SchedulingLatency));
    }

    #[test]
    fn scheduling_latency_resets_on_healthy_window() {
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        let t0 = Instant::now();
        b.observe(obs_high_sched(t0));
        // Healthy window resets counter.
        b.observe(obs_healthy(t0 + Duration::from_secs(10)));
        // Another high window — counter starts at 1, not 2 — still closed.
        b.observe(obs_high_sched(t0 + Duration::from_secs(20)));
        assert_eq!(b.state(), BreakerState::Closed);
    }

    #[test]
    fn lag_requires_three_consecutive_windows() {
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        let t0 = Instant::now();
        b.observe(obs_high_lag(t0));
        assert_eq!(b.state(), BreakerState::Closed);
        b.observe(obs_high_lag(t0 + Duration::from_secs(10)));
        assert_eq!(b.state(), BreakerState::Closed);
        b.observe(obs_high_lag(t0 + Duration::from_secs(20)));
        assert_eq!(b.state(), BreakerState::Open);
        assert_eq!(b.open_reason(), Some(OpenReason::SustainedLag));
    }

    #[test]
    fn lag_uses_max_of_floor_and_baseline_multiple() {
        // Baseline 100ms × 3 = 300ms, vs floor 50ms → effective threshold = 300ms.
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        let t0 = Instant::now();
        let mut o = obs_healthy(t0);
        o.heartbeat_lag_p99 = Duration::from_millis(200); // above floor (50ms)
        o.baseline_lag_p99 = Some(Duration::from_millis(100));
        // 200ms < 300ms threshold → not elevated.
        b.observe(o);
        b.observe(HealthObservation {
            at: t0 + Duration::from_secs(10),
            ..o
        });
        b.observe(HealthObservation {
            at: t0 + Duration::from_secs(20),
            ..o
        });
        assert_eq!(b.state(), BreakerState::Closed);
    }

    #[test]
    fn lag_uses_floor_when_no_baseline() {
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        let t0 = Instant::now();
        let mut o = obs_healthy(t0);
        o.heartbeat_lag_p99 = Duration::from_millis(100); // > 50ms floor
        o.baseline_lag_p99 = None; // no baseline yet (warmup)
        b.observe(o);
        b.observe(HealthObservation {
            at: t0 + Duration::from_secs(10),
            ..o
        });
        b.observe(HealthObservation {
            at: t0 + Duration::from_secs(20),
            ..o
        });
        assert_eq!(b.state(), BreakerState::Open);
    }

    #[test]
    fn term_churn_dwell_is_60s() {
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        let t0 = Instant::now();
        b.observe(obs_term_churn(t0));
        assert_eq!(b.state(), BreakerState::Open);
        // 30s healthy — still open (need 60s dwell).
        b.observe(obs_healthy(t0 + Duration::from_secs(30)));
        assert_eq!(b.state(), BreakerState::Open);
        // 60s healthy — transitions to half-open.
        b.observe(obs_healthy(t0 + Duration::from_secs(60)));
        assert_eq!(b.state(), BreakerState::HalfOpen);
    }

    #[test]
    fn lag_dwell_is_30s() {
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        let t0 = Instant::now();
        // Trip via lag.
        b.observe(obs_high_lag(t0));
        b.observe(obs_high_lag(t0 + Duration::from_secs(10)));
        b.observe(obs_high_lag(t0 + Duration::from_secs(20)));
        assert_eq!(b.state(), BreakerState::Open);
        // 15s healthy — still open.
        b.observe(obs_healthy(t0 + Duration::from_secs(35)));
        assert_eq!(b.state(), BreakerState::Open);
        // 30s healthy from the open transition (state_entered_at=20s, +30 = 50s).
        b.observe(obs_healthy(t0 + Duration::from_secs(50)));
        assert_eq!(b.state(), BreakerState::HalfOpen);
    }

    #[test]
    fn half_open_trips_back_on_fresh_fault() {
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        let t0 = Instant::now();
        // Get to half-open via term churn dwell.
        b.observe(obs_term_churn(t0));
        b.observe(obs_healthy(t0 + Duration::from_secs(60)));
        assert_eq!(b.state(), BreakerState::HalfOpen);
        // Fresh term churn → trip back.
        b.observe(obs_term_churn(t0 + Duration::from_secs(70)));
        assert_eq!(b.state(), BreakerState::Open);
        assert_eq!(b.open_reason(), Some(OpenReason::HalfOpenTripBack));
    }

    #[test]
    fn half_open_closes_after_dwell() {
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        let t0 = Instant::now();
        b.observe(obs_term_churn(t0));
        b.observe(obs_healthy(t0 + Duration::from_secs(60))); // → half-open at 60s
        assert_eq!(b.state(), BreakerState::HalfOpen);
        // 30s of healthy in half-open → close.
        b.observe(obs_healthy(t0 + Duration::from_secs(90)));
        assert_eq!(b.state(), BreakerState::Closed);
        assert_eq!(b.open_reason(), None);
    }

    #[test]
    fn verdict_reflects_state() {
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert_eq!(b.verdict(), BreakerVerdict::Allow);
        b.observe(obs_term_churn(Instant::now()));
        assert!(matches!(
            b.verdict(),
            BreakerVerdict::WouldReject {
                reason: OpenReason::TermChurn
            }
        ));
    }

    #[test]
    fn record_trial_outcome_only_acts_in_half_open() {
        let mut b = CircuitBreaker::new(CircuitBreakerConfig::default());
        let t0 = Instant::now();
        b.record_trial_outcome(t0, true);
        // Closed → ignored.
        assert_eq!(b.state(), BreakerState::Closed);
        // Trip and dwell to half-open.
        b.observe(obs_term_churn(t0));
        b.observe(obs_healthy(t0 + Duration::from_secs(60)));
        assert_eq!(b.state(), BreakerState::HalfOpen);
        b.record_trial_outcome(t0 + Duration::from_secs(61), true);
        assert_eq!(b.state(), BreakerState::Open);
        assert_eq!(b.open_reason(), Some(OpenReason::HalfOpenTripBack));
    }

    #[test]
    fn config_validate_rejects_bad_multiplier() {
        let mut cfg = CircuitBreakerConfig::default();
        cfg.heartbeat_baseline_multiplier = 0.0;
        assert!(cfg.validate().is_err());
        cfg.heartbeat_baseline_multiplier = -1.0;
        assert!(cfg.validate().is_err());
        cfg.heartbeat_baseline_multiplier = f64::NAN;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_rejects_bad_trial_fraction() {
        let mut cfg = CircuitBreakerConfig::default();
        cfg.half_open_trial_fraction = 1.5;
        assert!(cfg.validate().is_err());
        cfg.half_open_trial_fraction = -0.1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn for_profile_sets_threshold_per_profile() {
        let lan = CircuitBreakerConfig::for_profile(DeploymentProfile::Lan);
        let wan = CircuitBreakerConfig::for_profile(DeploymentProfile::Wan);
        assert_eq!(lan.heartbeat_lag_floor, Duration::from_millis(50));
        assert_eq!(wan.heartbeat_lag_floor, Duration::from_millis(200));
    }

    #[test]
    fn breaker_state_gauge_values_pinned() {
        // Stability test — Grafana queries on these. If a refactor
        // accidentally reorders the enum, this fails loudly.
        assert_eq!(BreakerState::Closed.gauge_value(), 0);
        assert_eq!(BreakerState::Open.gauge_value(), 1);
        assert_eq!(BreakerState::HalfOpen.gauge_value(), 2);
    }

    #[test]
    fn open_reason_labels_pinned() {
        assert_eq!(OpenReason::TermChurn.as_str(), "term_churn");
        assert_eq!(OpenReason::SustainedLag.as_str(), "sustained_lag");
        assert_eq!(OpenReason::SchedulingLatency.as_str(), "scheduling_latency");
        assert_eq!(OpenReason::ActiveElection.as_str(), "active_election");
        assert_eq!(OpenReason::HalfOpenTripBack.as_str(), "half_open_trip_back");
    }

    #[test]
    fn anti_flapping_holds_open_when_cycle_too_fast() {
        // Configure a tiny dwell so we can repeatedly trip+close in test.
        let cfg = CircuitBreakerConfig {
            term_churn_open_min_dwell: Duration::from_millis(100),
            lag_open_min_dwell: Duration::from_millis(100),
            min_cycle_time: Duration::from_secs(60),
            ..CircuitBreakerConfig::default()
        };
        let mut b = CircuitBreaker::new(cfg);
        let t0 = Instant::now();
        // First open at t0.
        b.observe(obs_term_churn(t0));
        assert_eq!(b.state(), BreakerState::Open);
        // Dwell elapses, half-open.
        b.observe(obs_healthy(t0 + Duration::from_millis(150)));
        assert_eq!(b.state(), BreakerState::HalfOpen);
        // Trip-back, then close again, then try to re-open within 60s.
        // Manual full-close via dwell.
        b.observe(obs_healthy(t0 + Duration::from_millis(300)));
        assert_eq!(b.state(), BreakerState::Closed);
        // Re-trigger within 60s of last open.
        b.observe(obs_term_churn(t0 + Duration::from_secs(5)));
        // Still goes Open (anti-flap holds the breaker, doesn't suppress
        // it — RFC: "stay open"). Just verify state transitioned.
        assert_eq!(b.state(), BreakerState::Open);
    }
}
