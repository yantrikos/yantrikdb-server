//! Lightweight metrics collection for YantrikDB server.
//!
//! Uses the `metrics` crate facade to record histograms and counters.
//! The actual storage is a simple in-process recorder that renders to
//! Prometheus text format on demand (called by the /metrics endpoint).
//!
//! Key metrics:
//!   - `yantrikdb_handler_duration_seconds` — per-handler HTTP latency
//!   - `yantrikdb_engine_lock_wait_seconds` — time waiting to acquire the engine mutex
//!   - `yantrikdb_requests_total` — per-handler request counter

use std::collections::HashMap;
use std::time::Instant;

use parking_lot::Mutex;

/// A simple histogram bucket collector. Not a full Prometheus client —
/// just enough to emit meaningful percentile data in text format.
#[derive(Clone)]
struct HistogramData {
    /// Sum of all observed values.
    sum: f64,
    /// Count of observations.
    count: u64,
    /// Bucket boundaries and their cumulative counts.
    buckets: Vec<(f64, u64)>,
}

impl HistogramData {
    fn new() -> Self {
        // Buckets tuned for lock-wait and handler-duration use cases.
        // Range: 100μs to 60s.
        let boundaries = vec![
            0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            30.0, 60.0,
        ];
        let buckets = boundaries.into_iter().map(|b| (b, 0u64)).collect();
        Self {
            sum: 0.0,
            count: 0,
            buckets,
        }
    }

    fn observe(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        for (boundary, count) in &mut self.buckets {
            if value <= *boundary {
                *count += 1;
            }
        }
    }

    fn render(&self, name: &str, labels: &str, help: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!("# HELP {} {}\n", name, help));
        out.push_str(&format!("# TYPE {} histogram\n", name));
        for (boundary, count) in &self.buckets {
            out.push_str(&format!(
                "{}_bucket{{{},le=\"{}\"}} {}\n",
                name, labels, boundary, count
            ));
        }
        out.push_str(&format!(
            "{}_bucket{{{},le=\"+Inf\"}} {}\n",
            name, labels, self.count
        ));
        out.push_str(&format!("{}_sum{{{}}} {}\n", name, labels, self.sum));
        out.push_str(&format!("{}_count{{{}}} {}\n", name, labels, self.count));
        out
    }
}

/// Global metrics store. One instance per process.
pub struct MetricsStore {
    handler_durations: Mutex<HashMap<String, HistogramData>>,
    lock_waits: Mutex<HashMap<String, HistogramData>>,
    request_counts: Mutex<HashMap<String, u64>>,

    // RFC 009 metrics — admission control + Raft observability.
    /// Counter: rejections of recall, by reason label
    /// (top_k_cap | in_flight_saturated | expanded_saturated |
    ///  body_too_large | rate_limit | circuit_breaker).
    recall_rejected_counts: Mutex<HashMap<&'static str, u64>>,
    /// Gauge: current in-flight recalls (any kind).
    recall_in_flight_gauge: std::sync::atomic::AtomicI64,
    /// Gauge: current concurrent expanded recalls.
    expansion_concurrent_gauge: std::sync::atomic::AtomicI64,
    /// Counter: Raft term increments (new election started or stepdown).
    /// Term=1423 thrashing fingerprint signal.
    raft_term_changes: std::sync::atomic::AtomicU64,
    /// Counter: Raft elections, labelled by result (won|lost|stepped_down).
    raft_elections: Mutex<HashMap<&'static str, u64>>,
    /// Histogram: heartbeat round-trip lag in seconds.
    raft_heartbeat_lag: Mutex<HistogramData>,
    /// Histogram: control-runtime task scheduling latency in seconds.
    /// THIS IS THE ACCEPTANCE GATE for PR-1's CPU isolation: under
    /// saturation, p99 must stay < 10ms. See `tests/cpu_isolation.rs`.
    raft_task_poll_latency: Mutex<HistogramData>,
    // RFC 010 PR-4 — openraft cluster gauges. Updated by the metrics
    // recorder spawned from `raft::status::spawn_raft_metrics_recorder`,
    // which subscribes to openraft's `RaftMetrics` watch channel.
    //
    // For Optional<u64> values we use `i64` with `-1` as the sentinel
    // for `None` so Prometheus gauges can render the absence without
    // a separate "_present" gauge. Operators read `-1` as "no value
    // yet" — same idiom as `node_filesystem_files_free` reporting
    // negative values for unsupported filesystems.
    /// Gauge: current term as observed by this node.
    openraft_current_term: std::sync::atomic::AtomicU64,
    /// Gauge: 1 if this node is the leader, 0 otherwise.
    openraft_is_leader: std::sync::atomic::AtomicI64,
    /// Gauge: last log index appended on this node. `-1` = none.
    openraft_last_log_index: std::sync::atomic::AtomicI64,
    /// Gauge: last log index applied to the state machine. `-1` = none.
    openraft_last_applied_index: std::sync::atomic::AtomicI64,
    /// Gauge: last snapshot's last_log_index. `-1` = no snapshot.
    openraft_snapshot_index: std::sync::atomic::AtomicI64,
    /// Gauge: earliest log_index still in the log. `-1` = none.
    openraft_purged_index: std::sync::atomic::AtomicI64,
    /// Gauge: ms since quorum last acknowledged the leader. `-1`
    /// when this node isn't the leader. Spike here = partition or
    /// replication backpressure.
    openraft_quorum_ack_lag_ms: std::sync::atomic::AtomicI64,
    /// Gauge: 1 if openraft `running_state` is `Ok`, 0 after fatal.
    openraft_running_state_healthy: std::sync::atomic::AtomicI64,
    /// Gauge: number of voter members.
    openraft_voters: std::sync::atomic::AtomicU64,
    /// Gauge: number of learner members.
    openraft_learners: std::sync::atomic::AtomicU64,

    /// Counter: recall requests, labelled by api_version + expand.
    recall_request_counts: Mutex<HashMap<(&'static str, bool), u64>>,
    /// Histogram: requested top_k values, labelled by api_version.
    recall_request_top_k: Mutex<HashMap<&'static str, HistogramData>>,

    /// Counter: embedder failures during write paths, labelled by handler
    /// (`remember` | `remember_batch`). Issue #19 — surfaces the
    /// previously-silent NULL-embedding writes that poisoned recall on
    /// the namespace.
    embedder_failures: Mutex<HashMap<&'static str, u64>>,

    /// Gauge: per-tenant count of rows with `embedding IS NULL`. Issue
    /// #20 — should be 0 in steady state (issue #19 closes the writer
    /// side). Hourly background healthcheck updates this. Non-zero
    /// values indicate pre-v0.8.1 stale data or a regression.
    null_embedding_counts: Mutex<HashMap<i64, i64>>,

    /// RFC 010 PR-6.4 — counter: enrichment ticks paused due to engine
    /// pressure (`count_pending_ops > threshold`). Keyed by db name so
    /// per-tenant pressure is visible. Without this, operators can't
    /// tell if the rule fires too often or too rarely.
    enrichment_paused: Mutex<HashMap<String, u64>>,
    /// Counter: enrichment ticks that ran (engine pressure under
    /// threshold). Sibling to `enrichment_paused` so the operator can
    /// read "we paused N times, we resumed N times" as a sanity check.
    enrichment_resumed: Mutex<HashMap<String, u64>>,
    /// Histogram: pending-op count observed at pause time. Buckets per
    /// yantrikdb-core's spec: 100, 250, 500, 1000, 2500. Diagnostic
    /// "is the threshold tuned right" — without this the operator is
    /// guessing.
    enrichment_pending_at_pause: Mutex<HistogramData>,
}

impl MetricsStore {
    pub fn new() -> Self {
        Self {
            handler_durations: Mutex::new(HashMap::new()),
            lock_waits: Mutex::new(HashMap::new()),
            request_counts: Mutex::new(HashMap::new()),
            recall_rejected_counts: Mutex::new(HashMap::new()),
            recall_in_flight_gauge: std::sync::atomic::AtomicI64::new(0),
            expansion_concurrent_gauge: std::sync::atomic::AtomicI64::new(0),
            raft_term_changes: std::sync::atomic::AtomicU64::new(0),
            raft_elections: Mutex::new(HashMap::new()),
            raft_heartbeat_lag: Mutex::new(HistogramData::new()),
            raft_task_poll_latency: Mutex::new(HistogramData::new()),
            openraft_current_term: std::sync::atomic::AtomicU64::new(0),
            openraft_is_leader: std::sync::atomic::AtomicI64::new(0),
            openraft_last_log_index: std::sync::atomic::AtomicI64::new(-1),
            openraft_last_applied_index: std::sync::atomic::AtomicI64::new(-1),
            openraft_snapshot_index: std::sync::atomic::AtomicI64::new(-1),
            openraft_purged_index: std::sync::atomic::AtomicI64::new(-1),
            openraft_quorum_ack_lag_ms: std::sync::atomic::AtomicI64::new(-1),
            openraft_running_state_healthy: std::sync::atomic::AtomicI64::new(0),
            openraft_voters: std::sync::atomic::AtomicU64::new(0),
            openraft_learners: std::sync::atomic::AtomicU64::new(0),
            recall_request_counts: Mutex::new(HashMap::new()),
            recall_request_top_k: Mutex::new(HashMap::new()),
            embedder_failures: Mutex::new(HashMap::new()),
            null_embedding_counts: Mutex::new(HashMap::new()),
            enrichment_paused: Mutex::new(HashMap::new()),
            enrichment_resumed: Mutex::new(HashMap::new()),
            enrichment_pending_at_pause: Mutex::new(HistogramData::new()),
        }
    }

    /// Record an HTTP handler's duration.
    pub fn record_handler_duration(&self, handler: &str, duration_secs: f64) {
        let mut map = self.handler_durations.lock();
        map.entry(handler.to_string())
            .or_insert_with(HistogramData::new)
            .observe(duration_secs);
    }

    /// Record time spent waiting for the engine mutex.
    pub fn record_lock_wait(&self, lock_name: &str, duration_secs: f64) {
        let mut map = self.lock_waits.lock();
        map.entry(lock_name.to_string())
            .or_insert_with(HistogramData::new)
            .observe(duration_secs);
    }

    /// Increment the per-handler request counter.
    pub fn increment_request(&self, handler: &str) {
        let mut map = self.request_counts.lock();
        *map.entry(handler.to_string()).or_insert(0) += 1;
    }

    /// Render all metrics in Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(4096);

        // Handler durations
        {
            let map = self.handler_durations.lock();
            for (handler, hist) in map.iter() {
                out.push_str(&hist.render(
                    "yantrikdb_handler_duration_seconds",
                    &format!("handler=\"{}\"", handler),
                    "Duration of HTTP handler execution in seconds",
                ));
            }
        }

        // Lock waits
        {
            let map = self.lock_waits.lock();
            for (lock_name, hist) in map.iter() {
                out.push_str(&hist.render(
                    "yantrikdb_lock_wait_seconds",
                    &format!("lock=\"{}\"", lock_name),
                    "Time spent waiting to acquire a lock in seconds",
                ));
            }
        }

        // Request counts
        {
            let map = self.request_counts.lock();
            if !map.is_empty() {
                out.push_str("# HELP yantrikdb_requests_total Total HTTP requests per handler\n");
                out.push_str("# TYPE yantrikdb_requests_total counter\n");
                for (handler, count) in map.iter() {
                    out.push_str(&format!(
                        "yantrikdb_requests_total{{handler=\"{}\"}} {}\n",
                        handler, count,
                    ));
                }
            }
        }

        // ── RFC 009 metrics ─────────────────────────────────────────

        // Recall rejection counter, labelled by reason.
        {
            let map = self.recall_rejected_counts.lock();
            if !map.is_empty() {
                out.push_str(
                    "# HELP yantrikdb_recall_rejected_total Recall requests rejected by admission control, by reason\n",
                );
                out.push_str("# TYPE yantrikdb_recall_rejected_total counter\n");
                for (reason, count) in map.iter() {
                    out.push_str(&format!(
                        "yantrikdb_recall_rejected_total{{reason=\"{}\"}} {}\n",
                        reason, count
                    ));
                }
            }
        }

        // In-flight gauges.
        out.push_str("# HELP yantrikdb_recall_in_flight Current in-flight recalls (any kind)\n");
        out.push_str("# TYPE yantrikdb_recall_in_flight gauge\n");
        out.push_str(&format!(
            "yantrikdb_recall_in_flight {}\n",
            self.recall_in_flight_gauge
                .load(std::sync::atomic::Ordering::Relaxed)
        ));

        out.push_str("# HELP yantrikdb_expansion_concurrent Current concurrent expanded recalls\n");
        out.push_str("# TYPE yantrikdb_expansion_concurrent gauge\n");
        out.push_str(&format!(
            "yantrikdb_expansion_concurrent {}\n",
            self.expansion_concurrent_gauge
                .load(std::sync::atomic::Ordering::Relaxed)
        ));

        // Raft term changes — the term=1423 thrashing fingerprint.
        out.push_str(
            "# HELP yantrikdb_raft_term_changes_total Raft term increments (new election or stepdown)\n",
        );
        out.push_str("# TYPE yantrikdb_raft_term_changes_total counter\n");
        out.push_str(&format!(
            "yantrikdb_raft_term_changes_total {}\n",
            self.raft_term_changes
                .load(std::sync::atomic::Ordering::Relaxed)
        ));

        // Raft elections by result.
        {
            let map = self.raft_elections.lock();
            if !map.is_empty() {
                out.push_str("# HELP yantrikdb_raft_elections_total Raft elections by outcome\n");
                out.push_str("# TYPE yantrikdb_raft_elections_total counter\n");
                for (result, count) in map.iter() {
                    out.push_str(&format!(
                        "yantrikdb_raft_elections_total{{result=\"{}\"}} {}\n",
                        result, count
                    ));
                }
            }
        }

        // Raft heartbeat lag histogram.
        {
            let hist = self.raft_heartbeat_lag.lock();
            if hist.count > 0 {
                out.push_str(&hist.render(
                    "yantrikdb_raft_heartbeat_lag_seconds",
                    "",
                    "Heartbeat round-trip lag in seconds",
                ));
            }
        }

        // Raft task poll latency — the PR-1 acceptance gate.
        {
            let hist = self.raft_task_poll_latency.lock();
            if hist.count > 0 {
                out.push_str(&hist.render(
                    "yantrikdb_raft_task_poll_latency_seconds",
                    "",
                    "Control-runtime task scheduling latency in seconds (acceptance gate signal)",
                ));
            }
        }

        // RFC 010 PR-4 — openraft cluster gauges. Always rendered (even
        // at default values) so dashboards using these metrics don't
        // disappear when the cluster is freshly bootstrapped.
        let load_u64 =
            |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
        let load_i64 =
            |a: &std::sync::atomic::AtomicI64| a.load(std::sync::atomic::Ordering::Relaxed);
        out.push_str(
            "# HELP yantrikdb_openraft_current_term Current Raft term observed by this node\n",
        );
        out.push_str("# TYPE yantrikdb_openraft_current_term gauge\n");
        out.push_str(&format!(
            "yantrikdb_openraft_current_term {}\n",
            load_u64(&self.openraft_current_term)
        ));
        out.push_str("# HELP yantrikdb_openraft_is_leader 1 if this node is the cluster leader, 0 otherwise\n");
        out.push_str("# TYPE yantrikdb_openraft_is_leader gauge\n");
        out.push_str(&format!(
            "yantrikdb_openraft_is_leader {}\n",
            load_i64(&self.openraft_is_leader)
        ));
        out.push_str("# HELP yantrikdb_openraft_last_log_index Last log index appended on this node (-1 = none)\n");
        out.push_str("# TYPE yantrikdb_openraft_last_log_index gauge\n");
        out.push_str(&format!(
            "yantrikdb_openraft_last_log_index {}\n",
            load_i64(&self.openraft_last_log_index)
        ));
        out.push_str("# HELP yantrikdb_openraft_last_applied_index Last log index applied to state machine (-1 = none)\n");
        out.push_str("# TYPE yantrikdb_openraft_last_applied_index gauge\n");
        out.push_str(&format!(
            "yantrikdb_openraft_last_applied_index {}\n",
            load_i64(&self.openraft_last_applied_index)
        ));
        out.push_str("# HELP yantrikdb_openraft_snapshot_index Last log index included in the most recent snapshot (-1 = none)\n");
        out.push_str("# TYPE yantrikdb_openraft_snapshot_index gauge\n");
        out.push_str(&format!(
            "yantrikdb_openraft_snapshot_index {}\n",
            load_i64(&self.openraft_snapshot_index)
        ));
        out.push_str("# HELP yantrikdb_openraft_purged_index Largest log index purged from storage (-1 = none)\n");
        out.push_str("# TYPE yantrikdb_openraft_purged_index gauge\n");
        out.push_str(&format!(
            "yantrikdb_openraft_purged_index {}\n",
            load_i64(&self.openraft_purged_index)
        ));
        out.push_str("# HELP yantrikdb_openraft_quorum_ack_lag_ms Ms since quorum last acknowledged the leader (-1 = not leader)\n");
        out.push_str("# TYPE yantrikdb_openraft_quorum_ack_lag_ms gauge\n");
        out.push_str(&format!(
            "yantrikdb_openraft_quorum_ack_lag_ms {}\n",
            load_i64(&self.openraft_quorum_ack_lag_ms)
        ));
        out.push_str("# HELP yantrikdb_openraft_running_state_healthy 1 if openraft running_state is Ok, 0 after fatal\n");
        out.push_str("# TYPE yantrikdb_openraft_running_state_healthy gauge\n");
        out.push_str(&format!(
            "yantrikdb_openraft_running_state_healthy {}\n",
            load_i64(&self.openraft_running_state_healthy)
        ));
        out.push_str("# HELP yantrikdb_openraft_voters Number of voter members in cluster\n");
        out.push_str("# TYPE yantrikdb_openraft_voters gauge\n");
        out.push_str(&format!(
            "yantrikdb_openraft_voters {}\n",
            load_u64(&self.openraft_voters)
        ));
        out.push_str("# HELP yantrikdb_openraft_learners Number of learner members in cluster\n");
        out.push_str("# TYPE yantrikdb_openraft_learners gauge\n");
        out.push_str(&format!(
            "yantrikdb_openraft_learners {}\n",
            load_u64(&self.openraft_learners)
        ));

        // Recall request counter, labelled by api_version and expand.
        {
            let map = self.recall_request_counts.lock();
            if !map.is_empty() {
                out.push_str("# HELP yantrikdb_recall_requests_total Recall requests received\n");
                out.push_str("# TYPE yantrikdb_recall_requests_total counter\n");
                for ((version, expand), count) in map.iter() {
                    out.push_str(&format!(
                        "yantrikdb_recall_requests_total{{api_version=\"{}\",expand=\"{}\"}} {}\n",
                        version, expand, count
                    ));
                }
            }
        }

        // Recall top_k histogram, labelled by api_version.
        {
            let map = self.recall_request_top_k.lock();
            for (version, hist) in map.iter() {
                if hist.count > 0 {
                    out.push_str(&hist.render(
                        "yantrikdb_recall_request_top_k",
                        &format!("api_version=\"{}\"", version),
                        "Distribution of requested top_k values",
                    ));
                }
            }
        }

        // RFC 017-A version gauges. Always emitted (build constants).
        render_version_gauges_if_set(&mut out);

        out
    }
}

// ── RFC 009 metric helpers ──────────────────────────────────────────
//
// These wrap the global() store. Keeping them as free functions matches
// the existing `record_engine_lock_wait` style and makes call sites read
// cleanly: `metrics::increment_recall_rejected("top_k_cap")` not
// `metrics::global().increment_recall_rejected("top_k_cap")`.

/// Increment the recall-rejection counter. `reason` MUST be a stable
/// string used in dashboards — see [`crate::admission::RejectReason::metric_label`].
pub fn increment_recall_rejected(reason: &'static str) {
    let mut map = global().recall_rejected_counts.lock();
    *map.entry(reason).or_insert(0) += 1;
}

/// Issue #19: increment the embedder-failure counter for write paths.
/// `handler` MUST be `"remember"` or `"remember_batch"` — pinned for
/// dashboard query stability. Surfaces previously-silent failures that
/// would have stored NULL-embedding rows and poisoned recall.
pub fn increment_embedder_failure(handler: &'static str) {
    let mut map = global().embedder_failures.lock();
    *map.entry(handler).or_insert(0) += 1;
}

/// Read a snapshot of embedder-failure counts. For `/metrics` rendering
/// and ops dashboards. Returns `(handler, count)` pairs.
pub fn embedder_failure_counts() -> Vec<(&'static str, u64)> {
    let map = global().embedder_failures.lock();
    map.iter().map(|(k, v)| (*k, *v)).collect()
}

/// Issue #20: set the per-tenant NULL-embedding gauge. Called by the
/// hourly background healthcheck loop. Steady-state value should be 0
/// after v0.8.1 is deployed (issue #19 closes the writer-side hole).
pub fn set_null_embedding_count(tenant_id: i64, count: i64) {
    let mut map = global().null_embedding_counts.lock();
    map.insert(tenant_id, count);
}

/// Read snapshot of NULL-embedding counts per tenant. For `/metrics`
/// rendering. Returns `(tenant_id, count)` pairs.
pub fn null_embedding_counts_snapshot() -> Vec<(i64, i64)> {
    let map = global().null_embedding_counts.lock();
    map.iter().map(|(k, v)| (*k, *v)).collect()
}

/// RFC 010 PR-6.4: record an enrichment-tick pause due to engine pressure.
/// `db_name` is the per-database label so operators see which tenants
/// are under pressure. `pending` is observed `count_pending_ops()` for
/// the histogram dimension that tells the operator if the threshold
/// is tuned right.
pub fn record_enrichment_paused(db_name: &str, pending: u64) {
    {
        let mut map = global().enrichment_paused.lock();
        *map.entry(db_name.to_string()).or_insert(0) += 1;
    }
    global()
        .enrichment_pending_at_pause
        .lock()
        .observe(pending as f64);
}

/// RFC 010 PR-6.4: record an enrichment-tick that ran (engine pressure
/// under threshold). Sibling counter to [`record_enrichment_paused`] so
/// the operator can read paused/resumed ratio as a sanity check.
pub fn record_enrichment_resumed(db_name: &str) {
    let mut map = global().enrichment_resumed.lock();
    *map.entry(db_name.to_string()).or_insert(0) += 1;
}

/// Snapshot of enrichment pause counts per db. For `/metrics` rendering.
pub fn enrichment_paused_snapshot() -> Vec<(String, u64)> {
    let map = global().enrichment_paused.lock();
    map.iter().map(|(k, v)| (k.clone(), *v)).collect()
}

/// Snapshot of enrichment resume counts per db. For `/metrics` rendering.
pub fn enrichment_resumed_snapshot() -> Vec<(String, u64)> {
    let map = global().enrichment_resumed.lock();
    map.iter().map(|(k, v)| (k.clone(), *v)).collect()
}

/// Snapshot of pending-at-pause histogram totals (count + sum). For
/// `/metrics` rendering. Returning `(count, sum)` rather than the full
/// `HistogramData` keeps the type private to this module.
pub fn enrichment_pending_at_pause_totals() -> (u64, f64) {
    let h = global().enrichment_pending_at_pause.lock();
    (h.count, h.sum)
}

/// Set the in-flight recall gauge.
pub fn set_recall_in_flight_gauge(value: i64) {
    global()
        .recall_in_flight_gauge
        .store(value, std::sync::atomic::Ordering::Relaxed);
}

/// Set the concurrent expanded-recall gauge.
pub fn set_expansion_concurrent_gauge(value: i64) {
    global()
        .expansion_concurrent_gauge
        .store(value, std::sync::atomic::Ordering::Relaxed);
}

/// Record a Raft term increment. Term=1423 thrashing fingerprint.
pub fn increment_raft_term_changes() {
    global()
        .raft_term_changes
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Record a Raft election outcome. `result` MUST be one of
/// "won" | "lost" | "stepped_down" — pinned for dashboard stability.
pub fn record_raft_election(result: &'static str) {
    debug_assert!(
        matches!(result, "won" | "lost" | "stepped_down"),
        "raft election result must be one of won|lost|stepped_down"
    );
    let mut map = global().raft_elections.lock();
    *map.entry(result).or_insert(0) += 1;
}

/// Record a heartbeat round-trip duration.
pub fn record_raft_heartbeat_lag(duration: std::time::Duration) {
    global()
        .raft_heartbeat_lag
        .lock()
        .observe(duration.as_secs_f64());
}

/// Record a control-runtime task scheduling latency. THIS IS THE PR-1
/// ACCEPTANCE GATE: under app saturation, p99 must stay < 10ms or
/// `tests/cpu_isolation.rs` fails the merge.
pub fn record_raft_task_poll_latency(duration: std::time::Duration) {
    global()
        .raft_task_poll_latency
        .lock()
        .observe(duration.as_secs_f64());
}

/// Primitive-typed setter for the openraft cluster gauges. Kept on
/// primitive args (not `&RaftStatus`) so this module stays a leaf —
/// the integration tests for unrelated modules (cpu_isolation, etc.)
/// re-include `metrics.rs` via `#[path]` and shouldn't be forced to
/// pull in `raft/`. The `raft::status` module wraps this in a
/// status-aware helper.
#[allow(clippy::too_many_arguments)]
pub fn record_openraft_gauges(
    current_term: u64,
    is_leader: bool,
    last_log_index: Option<u64>,
    last_applied_index: Option<u64>,
    snapshot_index: Option<u64>,
    purged_index: Option<u64>,
    millis_since_quorum_ack: Option<u64>,
    healthy: bool,
    voters: u64,
    learners: u64,
) {
    use std::sync::atomic::Ordering::Relaxed;
    let g = global();
    g.openraft_current_term.store(current_term, Relaxed);
    g.openraft_is_leader
        .store(if is_leader { 1 } else { 0 }, Relaxed);
    g.openraft_last_log_index
        .store(last_log_index.map(|n| n as i64).unwrap_or(-1), Relaxed);
    g.openraft_last_applied_index
        .store(last_applied_index.map(|n| n as i64).unwrap_or(-1), Relaxed);
    g.openraft_snapshot_index
        .store(snapshot_index.map(|n| n as i64).unwrap_or(-1), Relaxed);
    g.openraft_purged_index
        .store(purged_index.map(|n| n as i64).unwrap_or(-1), Relaxed);
    g.openraft_quorum_ack_lag_ms.store(
        millis_since_quorum_ack.map(|n| n as i64).unwrap_or(-1),
        Relaxed,
    );
    g.openraft_running_state_healthy
        .store(if healthy { 1 } else { 0 }, Relaxed);
    g.openraft_voters.store(voters, Relaxed);
    g.openraft_learners.store(learners, Relaxed);
}

/// Record an incoming recall request. `api_version` is "v1" or "v2";
/// `expand` is the `expand_entities` flag.
pub fn record_recall_request(api_version: &'static str, expand: bool) {
    let mut map = global().recall_request_counts.lock();
    *map.entry((api_version, expand)).or_insert(0) += 1;
}

/// Record the `top_k` value of an incoming recall.
pub fn record_recall_top_k(api_version: &'static str, top_k: usize) {
    let mut map = global().recall_request_top_k.lock();
    map.entry(api_version)
        .or_insert_with(HistogramData::new)
        .observe(top_k as f64);
}

// ── RFC 017-A version metrics ──────────────────────────────────────

/// Counter: rejections caused by version mismatches.
/// `reason` MUST be a stable [`crate::version::VersionError::metric_label`]
/// — see version/error.rs for the closed set.
pub fn increment_version_rejection(reason: &'static str) {
    let mut map = global().recall_rejected_counts.lock();
    // Reuse the same counter map but with a distinct prefix so dashboards
    // can split version errors from recall caps. Stable label.
    *map.entry(reason).or_insert(0) += 1;
}

/// Caller-provided opaque hook for rendering version gauges. Lets us
/// keep `metrics.rs` independent of `crate::version` so the test that
/// re-includes `metrics.rs` standalone (`tests/cpu_isolation.rs` via
/// `#[path]`) compiles without dragging in the whole module tree.
///
/// The real /metrics handler in `http_gateway.rs` registers a real hook;
/// tests pass a no-op or omit it.
pub type VersionGaugeRenderer = fn(&mut String);

static VERSION_GAUGE_RENDERER: std::sync::OnceLock<VersionGaugeRenderer> =
    std::sync::OnceLock::new();

/// Install the version gauge renderer (call once at server startup).
pub fn set_version_gauge_renderer(f: VersionGaugeRenderer) {
    let _ = VERSION_GAUGE_RENDERER.set(f);
}

/// Render version gauges if a renderer is installed; no-op otherwise.
fn render_version_gauges_if_set(out: &mut String) {
    if let Some(renderer) = VERSION_GAUGE_RENDERER.get() {
        renderer(out);
    }
}

/// Snapshot of the p99 of `raft_task_poll_latency_seconds` for tests
/// that need to assert on the acceptance gate without parsing the full
/// Prometheus output.
#[cfg(test)]
pub fn raft_task_poll_latency_p99() -> f64 {
    let hist = global().raft_task_poll_latency.lock();
    if hist.count == 0 {
        return 0.0;
    }
    let target = (hist.count as f64 * 0.99) as u64;
    let mut acc = 0u64;
    let mut last_boundary = 0.0;
    for (boundary, bucket_count) in &hist.buckets {
        // bucket_count is cumulative ≤ boundary
        acc = *bucket_count;
        if acc >= target {
            return *boundary;
        }
        last_boundary = *boundary;
    }
    // All observations sit above the highest bucket boundary.
    last_boundary.max(hist.sum / hist.count.max(1) as f64)
}

/// Lazy global metrics store. Initialized once on first access.
static METRICS: std::sync::OnceLock<MetricsStore> = std::sync::OnceLock::new();

/// Get the global metrics store.
pub fn global() -> &'static MetricsStore {
    METRICS.get_or_init(MetricsStore::new)
}

/// RAII timer that records handler duration on drop.
pub struct HandlerTimer {
    handler: &'static str,
    start: Instant,
}

impl HandlerTimer {
    pub fn new(handler: &'static str) -> Self {
        global().increment_request(handler);
        Self {
            handler,
            start: Instant::now(),
        }
    }
}

impl Drop for HandlerTimer {
    fn drop(&mut self) {
        global().record_handler_duration(self.handler, self.start.elapsed().as_secs_f64());
    }
}

/// Record engine lock wait time. Call before and after lock acquisition.
pub fn record_engine_lock_wait(duration: std::time::Duration) {
    global().record_lock_wait("engine", duration.as_secs_f64());
}

/// RAII timer that records engine-lock-hold metric on drop. Use right
/// after `engine.lock()` to measure how long the lock is held by the
/// caller's scope. Dropping the timer fires `record_engine_lock_hold`,
/// which logs a warn-level slow-holder line if hold exceeds threshold.
pub struct LockHoldTimer {
    op: &'static str,
    start: std::time::Instant,
}

impl LockHoldTimer {
    pub fn start(op: &'static str) -> Self {
        Self {
            op,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for LockHoldTimer {
    fn drop(&mut self) {
        record_engine_lock_hold(self.op, self.start.elapsed());
    }
}

/// Record engine lock hold time tagged by operation name. Logs a warning
/// if the hold exceeds the configured slow-holder threshold (default 50ms),
/// which is the operator-visible signal that the engine mutex is being
/// held across expensive work and starving concurrent requests.
pub fn record_engine_lock_hold(op: &str, duration: std::time::Duration) {
    let secs = duration.as_secs_f64();
    global().record_lock_wait(&format!("engine_hold_{op}"), secs);
    let slow_ms: u128 = std::env::var("YANTRIKDB_SLOW_LOCK_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let elapsed_ms = duration.as_millis();
    if elapsed_ms > slow_ms {
        tracing::warn!(
            op = %op,
            hold_ms = %elapsed_ms,
            threshold_ms = %slow_ms,
            "engine lock held longer than threshold (slow-holder)"
        );
    }
}

/// Record control lock wait time. Not currently instrumented — reserved for
/// future control-path metrics once resolve_engine is instrumented.
#[allow(dead_code)]
pub fn record_control_lock_wait(duration: std::time::Duration) {
    global().record_lock_wait("control", duration.as_secs_f64());
}

// ── Lock-Order Checker (debug builds only) ──────────────────────────
//
// See CONCURRENCY.md Rule 3 for the ordering invariant:
//   control(0) > tenant_pool(1) > engine(2) > conn(3) > vec_index(4)
//   > graph_index(5) > scoring_cache(6) > active_sessions(7) > hlc(8)
//
// In debug builds, every lock acquisition site calls `check_lock_order`
// with its rank. If the current thread already holds a lock with a HIGHER
// rank, we panic — that's an ordering violation which could deadlock in
// production.

/// Lock rank constants. Lower number = acquired first in the global order.
/// Not yet wired into all lock sites — will be instrumented as part of
/// the InstrumentedMutex wrapper in a future commit. Present now so
/// the constants and checker functions are available for manual use
/// in new code and tests.
#[allow(dead_code)]
#[cfg(debug_assertions)]
pub mod lock_rank {
    pub const CONTROL: u8 = 0;
    pub const TENANT_POOL: u8 = 1;
    pub const ENGINE: u8 = 2;
    pub const CONN: u8 = 3;
    pub const VEC_INDEX: u8 = 4;
    pub const GRAPH_INDEX: u8 = 5;
    pub const SCORING_CACHE: u8 = 6;
    pub const ACTIVE_SESSIONS: u8 = 7;
    pub const HLC: u8 = 8;
}

/// Check that acquiring a lock at `rank` doesn't violate the ordering
/// invariant. Panics in debug builds if a higher-rank lock is already held.
#[allow(dead_code)]
#[cfg(debug_assertions)]
pub fn check_lock_order(rank: u8, lock_name: &str) {
    thread_local! {
        static HELD_RANKS: std::cell::RefCell<Vec<(u8, &'static str)>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    HELD_RANKS.with(|held| {
        let held = held.borrow();
        for &(held_rank, held_name) in held.iter() {
            if held_rank > rank {
                panic!(
                    "LOCK ORDER VIOLATION: trying to acquire '{}' (rank {}) \
                     while holding '{}' (rank {}). See CONCURRENCY.md Rule 3.",
                    lock_name, rank, held_name, held_rank,
                );
            }
        }
    });
}

/// Record that a lock at `rank` is now held by this thread.
#[allow(dead_code)]
#[cfg(debug_assertions)]
pub fn push_lock(rank: u8, lock_name: &'static str) {
    thread_local! {
        static HELD_RANKS: std::cell::RefCell<Vec<(u8, &'static str)>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    HELD_RANKS.with(|held| {
        held.borrow_mut().push((rank, lock_name));
    });
}

/// Record that a lock at `rank` has been released by this thread.
#[allow(dead_code)]
#[cfg(debug_assertions)]
pub fn pop_lock(rank: u8) {
    thread_local! {
        static HELD_RANKS: std::cell::RefCell<Vec<(u8, &'static str)>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    HELD_RANKS.with(|held| {
        let mut held = held.borrow_mut();
        if let Some(pos) = held.iter().rposition(|(r, _)| *r == rank) {
            held.remove(pos);
        }
    });
}

// In release builds, these are no-ops.
#[cfg(not(debug_assertions))]
pub fn check_lock_order(_rank: u8, _lock_name: &str) {}
#[cfg(not(debug_assertions))]
pub fn push_lock(_rank: u8, _lock_name: &'static str) {}
#[cfg(not(debug_assertions))]
pub fn pop_lock(_rank: u8) {}
