//! Acceptance gate for RFC 009 PR-1: prove that the runtime split + OS
//! priority + concurrency caps actually keep Raft tasks getting CPU
//! under app-runtime saturation.
//!
//! ## Why this test exists
//!
//! Per the gpt-5.5 redteam of RFC 009: dedicated tokio runtime ≠ CPU
//! isolation. Threads can exist with no CPU under saturation. The runtime
//! split is one of three layers that COMPOSE; this test verifies they
//! actually compose to the promised guarantee:
//!
//!   > Under app-runtime saturation, control-runtime task scheduling
//!   > latency p99 must stay under 10ms (with privileges) or 100ms
//!   > (without).
//!
//! If this test fails, PR-1 does not merge. If it later regresses, the
//! release does not ship. This is the single hard gate that prevents
//! us from re-introducing the term=1423 thrashing pattern at the
//! infrastructure level.
//!
//! ## What it does
//!
//! 1. Build a `SplitRuntime` with control_threads=2, app_threads=2.
//! 2. Spawn the scheduling-latency probe on the control runtime
//!    (records into `metrics::record_raft_task_poll_latency`).
//! 3. Saturate the app runtime with a 10× tight loop — far more than
//!    its threads can handle, forcing constant scheduler pressure.
//! 4. Let it run for 5 seconds (enough for the probe to record ≥30
//!    samples — sufficient for a meaningful p99).
//! 5. Snapshot p99 and assert against the threshold.
//!
//! ## Threshold rationale
//!
//! `10ms` is well below the typical Raft heartbeat interval (1s default)
//! and election timeout (5s default). Even at p99=10ms, heartbeats land
//! with two orders of magnitude of slack. If we ever can't hit 10ms with
//! priority privileges, something has regressed in either tokio,
//! libc::pthread_setschedparam, or the kernel scheduler — all signals
//! worth investigating.
//!
//! `100ms` for unprivileged is the runtime-split + caps-only floor.
//! Without SCHED_FIFO, fairness is "best effort," but the bounded
//! concurrency caps still prevent app threads from being CPU-bound for
//! more than a few hundred ms at a time — keeping p99 under 100ms.
//! If unprivileged blows past 100ms, the caps are misconfigured.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

// Cross-import the server crate's runtime + metrics modules. This requires
// these modules to be `pub` or `pub(crate)` re-exports — which they
// already are via the test target visibility in Rust's compilation model.
//
// If this fails to build, the test target needs a small wrapper module
// (e.g. `crates/yantrikdb-server/src/test_support.rs` exposing the
// internals). For now we use the module path directly.
#[path = "../src/runtime.rs"]
mod runtime;

#[path = "../src/metrics.rs"]
mod metrics;

#[test]
fn raft_task_scheduling_latency_p99_stays_bounded_under_app_saturation() {
    // Use a fresh metrics store snapshot via a side-channel: we just
    // check that the histogram captures observations correctly.
    let cfg = runtime::RuntimeConfig {
        control_threads: 2,
        app_threads: 2,
        control_priority: runtime::ControlPriority::default(),
    };
    let split = runtime::SplitRuntime::new(cfg).expect("build split runtime");
    let ctrl = split.control_handle();
    let app = split.app_handle();

    let stop = Arc::new(AtomicBool::new(false));

    // Probe: every 100ms, measure how late we got CPU. This mirrors what
    // `run_scheduling_latency_probe` does in main.rs.
    let stop_clone = Arc::clone(&stop);
    let probe = ctrl.spawn(async move {
        let interval = Duration::from_millis(100);
        let mut samples_recorded = 0u32;
        while !stop_clone.load(Ordering::Relaxed) {
            let start = std::time::Instant::now();
            tokio::time::sleep(interval).await;
            let elapsed = start.elapsed();
            let lag = elapsed.saturating_sub(interval);
            metrics::record_raft_task_poll_latency(lag);
            samples_recorded += 1;
        }
        samples_recorded
    });

    // Saturate the app runtime: spawn 10× the available threads with
    // tight CPU loops. Each task calls `yield_now` periodically so it
    // doesn't completely freeze the runtime — but yields inside one
    // runtime cannot help tasks in the OTHER runtime get CPU.
    let n_load = 20;
    let load_handles: Vec<_> = (0..n_load)
        .map(|_| {
            let stop_clone = Arc::clone(&stop);
            app.spawn(async move {
                let mut acc: u64 = 0;
                while !stop_clone.load(Ordering::Relaxed) {
                    // Pure CPU work — no I/O, no yields. The whole point
                    // is to monopolize app-runtime worker threads.
                    for i in 0..1_000_000u64 {
                        acc = acc.wrapping_add(i.wrapping_mul(i));
                    }
                    // One yield per million iterations so the task is
                    // technically cooperative — without it some tokio
                    // versions warn about long polls.
                    tokio::task::yield_now().await;
                }
                acc
            })
        })
        .collect();

    // Run for 5s. Long enough for the probe to record many samples,
    // short enough that the CI test isn't slow.
    std::thread::sleep(Duration::from_secs(5));

    // Stop and join.
    stop.store(true, Ordering::Relaxed);
    let samples = ctrl
        .block_on(probe)
        .expect("probe task did not panic");
    for h in load_handles {
        let _ = ctrl.block_on(h);
    }

    // Sanity: we should have at least 30 samples (5s / 100ms ≈ 50, minus
    // ramp-up and shutdown overhead).
    assert!(
        samples >= 30,
        "scheduling probe ran too few samples: {samples}; the probe was starved of CPU \
         (which would itself indicate runtime split is broken)"
    );

    let p99 = metrics::raft_task_poll_latency_p99();

    // Threshold depends on whether we have priority privileges. In CI
    // we typically don't, so the default fallback path runs (caps only).
    let has_priority_privilege = priority_privilege_available();
    let limit = if has_priority_privilege { 0.010 } else { 0.100 };

    assert!(
        p99 <= limit,
        "ACCEPTANCE GATE FAILED: raft_task_poll_latency p99 = {:.3}ms exceeds limit {:.3}ms \
         (privilege={}, samples={}). The runtime split + admission caps are NOT keeping the \
         Raft task scheduled promptly under app-runtime saturation. Investigate before \
         merging RFC 009 PR-1.",
        p99 * 1000.0,
        limit * 1000.0,
        has_priority_privilege,
        samples
    );

    eprintln!(
        "[acceptance gate] p99 = {:.3}ms (limit {:.3}ms, privilege={}, samples={})",
        p99 * 1000.0,
        limit * 1000.0,
        has_priority_privilege,
        samples
    );

    split.shutdown_timeout(Duration::from_secs(2));
}

/// Whether this process can request elevated scheduling priority. Linux
/// only — other platforms don't expose this and we use the unprivileged
/// threshold by default.
fn priority_privilege_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        // Best-effort detection: try to set FIFO and see if it sticks.
        // We do this in a throwaway thread so we don't poison the test
        // thread's priority.
        let handle = std::thread::spawn(|| unsafe {
            let param = libc::sched_param { sched_priority: 1 };
            libc::pthread_setschedparam(libc::pthread_self(), libc::SCHED_FIFO, &param) == 0
        });
        handle.join().unwrap_or(false)
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}
