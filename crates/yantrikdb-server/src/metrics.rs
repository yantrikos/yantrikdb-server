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
}

impl MetricsStore {
    pub fn new() -> Self {
        Self {
            handler_durations: Mutex::new(HashMap::new()),
            lock_waits: Mutex::new(HashMap::new()),
            request_counts: Mutex::new(HashMap::new()),
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

        out
    }
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
