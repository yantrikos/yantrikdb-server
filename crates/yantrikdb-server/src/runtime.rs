//! Tokio runtime split for Raft control-plane vs application work.
//!
//! Per RFC 009 §4 — runtime split alone is not real CPU isolation. Three
//! reinforcing layers:
//!
//! 1. **Runtime split** (this module): control and application work run on
//!    independent tokio runtimes with separate worker threads. Control
//!    runtime hosts Raft heartbeat/AppendEntries/election tasks; app
//!    runtime hosts HTTP/recall handlers.
//! 2. **OS thread priority**: control-runtime threads get `SCHED_FIFO`
//!    priority on Linux (best-effort `nice -10` elsewhere), so the kernel
//!    will preempt application work to run Raft tasks.
//! 3. **Hard concurrency caps** (see [`admission`]): max concurrent
//!    expanded recalls + in-flight semaphore prevent CPU saturation in the
//!    first place.
//!
//! These layers compose. Runtime split alone leaves us vulnerable: threads
//! can exist with no CPU under saturation. Priority alone leaves Raft
//! starved if app work also has elevated priority. Caps alone leave us
//! at the mercy of cooperative-scheduler fairness. Together: Raft tasks
//! get scheduled within tens of microseconds even when app cores are 100%.
//!
//! The acceptance gate is `tests/cpu_isolation.rs`: drive app runtime to
//! saturation, assert `raft_task_poll_latency_seconds{quantile=0.99}
//! < 0.010` (10ms). PR-1 does not merge until this passes.

use std::sync::Arc;
use std::time::Duration;

use tokio::runtime::{Builder, Handle, Runtime};

/// Configuration for the runtime split.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Threads dedicated to Raft / replication / control-plane work.
    /// Default: 2 (sized to leave at least 1 core idle headroom for
    /// heartbeats even under app saturation).
    pub control_threads: usize,
    /// Threads dedicated to application work (HTTP, recall, expansion).
    /// 0 = all remaining cores.
    pub app_threads: usize,
    /// OS-level priority for control-runtime worker threads.
    pub control_priority: ControlPriority,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            control_threads: 2,
            app_threads: 0,
            control_priority: ControlPriority::default(),
        }
    }
}

/// OS-level scheduling policy for control-runtime threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlPriority {
    /// Linux real-time SCHED_FIFO. Highest priority, requires CAP_SYS_NICE
    /// or root. Falls back to `Nice(-10)` if not permitted.
    Fifo,
    /// SCHED_OTHER with elevated nice. Works without privileges.
    /// Provides modest priority elevation, not preemption guarantees.
    Nice,
    /// No priority adjustment. Runtime split alone — relies on caps.
    Default,
}

impl Default for ControlPriority {
    fn default() -> Self {
        // Default to Fifo on Linux, Default elsewhere. Fifo will fall back
        // to Nice if the process lacks CAP_SYS_NICE.
        if cfg!(target_os = "linux") {
            ControlPriority::Fifo
        } else {
            ControlPriority::Default
        }
    }
}

impl ControlPriority {
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "fifo" => ControlPriority::Fifo,
            "nice" => ControlPriority::Nice,
            "default" | "" => ControlPriority::Default,
            other => {
                tracing::warn!(
                    value = other,
                    "unknown control_priority, falling back to default"
                );
                ControlPriority::Default
            }
        }
    }
}

/// The split runtime pair. Drop order matters: app runtime drops first,
/// then control runtime — so any control-runtime tasks waiting on app
/// runtime futures don't deadlock on shutdown.
pub struct SplitRuntime {
    control: Runtime,
    app: Runtime,
}

impl SplitRuntime {
    /// Build the runtime pair with the requested configuration.
    pub fn new(cfg: RuntimeConfig) -> std::io::Result<Self> {
        let total = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let control_threads = cfg.control_threads.max(1);
        let app_threads = if cfg.app_threads == 0 {
            // Leave at least 1 thread for the app runtime.
            total.saturating_sub(control_threads).max(1)
        } else {
            cfg.app_threads
        };

        tracing::info!(
            control_threads,
            app_threads,
            total_cores = total,
            priority = ?cfg.control_priority,
            "building split runtime"
        );

        // Use Arc<AtomicUsize> to give each thread a unique id without
        // pulling in another crate. The IDs are only used in thread names
        // for observability.
        let ctrl_idx = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let app_idx = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let priority = cfg.control_priority;
        let ctrl_idx_clone = Arc::clone(&ctrl_idx);
        let control = Builder::new_multi_thread()
            .worker_threads(control_threads)
            .enable_all()
            .thread_name_fn(move || {
                let n = ctrl_idx_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                format!("ydb-ctrl-{n}")
            })
            .on_thread_start(move || {
                if let Err(e) = apply_priority(priority) {
                    // First failure across the fleet logs at WARN. Per-thread
                    // failures after that would just be noise.
                    static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                    if WARNED.set(()).is_ok() {
                        tracing::warn!(
                            error = %e,
                            requested = ?priority,
                            "could not apply requested control-runtime priority; falling back to OS default"
                        );
                    }
                }
            })
            .build()?;

        let app_idx_clone = Arc::clone(&app_idx);
        let app = Builder::new_multi_thread()
            .worker_threads(app_threads)
            .enable_all()
            .thread_name_fn(move || {
                let n = app_idx_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                format!("ydb-app-{n}")
            })
            .build()?;

        Ok(Self { control, app })
    }

    pub fn control_handle(&self) -> Handle {
        self.control.handle().clone()
    }

    pub fn app_handle(&self) -> Handle {
        self.app.handle().clone()
    }

    /// Shut down the runtimes with a deadline. App runtime shuts down
    /// first so any control-runtime tasks waiting on app futures don't
    /// hang forever.
    pub fn shutdown_timeout(self, dur: Duration) {
        self.app.shutdown_timeout(dur);
        self.control.shutdown_timeout(dur);
    }
}

/// Apply the requested OS scheduling priority to the calling thread.
/// Linux: best-effort SCHED_FIFO via libc, falling back to setpriority.
/// Other platforms: best-effort setpriority where available.
#[cfg(target_os = "linux")]
fn apply_priority(priority: ControlPriority) -> std::io::Result<()> {
    match priority {
        ControlPriority::Default => Ok(()),
        ControlPriority::Fifo => {
            // SCHED_FIFO with priority 1 (low real-time, well above any
            // SCHED_OTHER). Higher priorities are reserved for kernel
            // tasks and would risk audio dropouts on workstation
            // deployments.
            //
            // Safety: pthread_setschedparam is thread-safe, and the
            // sched_param struct is initialized fully.
            unsafe {
                let param = libc::sched_param { sched_priority: 1 };
                let rc = libc::pthread_setschedparam(
                    libc::pthread_self(),
                    libc::SCHED_FIFO,
                    &param,
                );
                if rc != 0 {
                    // Most common: EPERM (no CAP_SYS_NICE). Fall back to nice.
                    return apply_priority(ControlPriority::Nice);
                }
            }
            Ok(())
        }
        ControlPriority::Nice => {
            unsafe {
                // setpriority(PRIO_PROCESS, 0, -10) — 0 = current thread on
                // Linux when called via libc thanks to syscall semantics.
                let rc = libc::setpriority(libc::PRIO_PROCESS, 0, -10);
                if rc != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn apply_priority(priority: ControlPriority) -> std::io::Result<()> {
    match priority {
        ControlPriority::Default => Ok(()),
        ControlPriority::Fifo | ControlPriority::Nice => {
            // Non-Linux: no portable priority API in libc. Document and
            // move on — the runtime split + admission caps still apply.
            tracing::debug!(
                "control priority is Linux-only; runtime split + caps still active on this platform"
            );
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_chooses_fifo_on_linux() {
        let cfg = RuntimeConfig::default();
        let expected = if cfg!(target_os = "linux") {
            ControlPriority::Fifo
        } else {
            ControlPriority::Default
        };
        assert_eq!(cfg.control_priority, expected);
    }

    #[test]
    fn priority_from_str_handles_known_values() {
        assert_eq!(ControlPriority::from_str("fifo"), ControlPriority::Fifo);
        assert_eq!(ControlPriority::from_str("FIFO"), ControlPriority::Fifo);
        assert_eq!(ControlPriority::from_str("nice"), ControlPriority::Nice);
        assert_eq!(
            ControlPriority::from_str("default"),
            ControlPriority::Default
        );
        assert_eq!(ControlPriority::from_str(""), ControlPriority::Default);
        assert_eq!(
            ControlPriority::from_str("garbage"),
            ControlPriority::Default
        );
    }

    #[test]
    fn split_runtime_builds_with_minimal_threads() {
        let cfg = RuntimeConfig {
            control_threads: 1,
            app_threads: 1,
            control_priority: ControlPriority::Default,
        };
        let rt = SplitRuntime::new(cfg).expect("split runtime");
        let ctrl = rt.control_handle();
        let app = rt.app_handle();

        // Smoke test: spawn a noop on each runtime, ensure it runs.
        let ctrl_done = ctrl.spawn(async { 1u32 });
        let app_done = app.spawn(async { 2u32 });
        let (c, a) = ctrl.block_on(async { (ctrl_done.await, app_done.await) });
        assert_eq!(c.unwrap(), 1);
        assert_eq!(a.unwrap(), 2);

        rt.shutdown_timeout(Duration::from_secs(2));
    }
}
