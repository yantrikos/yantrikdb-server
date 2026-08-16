//! Background workers — the server thinks for itself.
//!
//! Spawns per-database tokio tasks that run consolidation, decay sweeps,
//! and stale session cleanup on configurable intervals.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use yantrikdb::types::ThinkConfig;
use yantrikdb::{MaintenanceCycleConfig, YantrikDB};

use crate::config::{BackgroundSection, MaintenanceSection};

/// Live check of whether this node should run state-mutating maintenance.
///
/// RFC 027 cluster-safety invariant: `run_maintenance_cycle()` mutates state
/// (resolves conflicts, prunes triggers, recalibrates importance, upserts
/// edges). Running it independently on a follower would fork the state
/// machine — the v0.8.18 class of bug. So maintenance runs **only where writes
/// are accepted** (standalone, or the current leader); the mutations then
/// propagate through the same replication path as every other write.
///
/// The gate is polled every tick because the role changes on failover. It
/// mirrors the exact write-acceptance logic the wire/HTTP write path uses
/// (openraft leader → raft-lite `accepts_writes()` → standalone-always).
#[derive(Clone)]
pub struct WriteAcceptanceGate {
    inner: Arc<dyn Fn() -> bool + Send + Sync>,
}

impl WriteAcceptanceGate {
    /// Standalone / single-node: always accepts writes.
    pub fn standalone() -> Self {
        Self {
            inner: Arc::new(|| true),
        }
    }

    /// Build from a live predicate (closure over the raft/cluster handles).
    pub fn from_fn(f: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        Self { inner: Arc::new(f) }
    }

    /// Whether this node currently accepts writes (leader or standalone).
    pub fn accepts_writes(&self) -> bool {
        (self.inner)()
    }
}

impl std::fmt::Debug for WriteAcceptanceGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WriteAcceptanceGate(accepts={})", self.accepts_writes())
    }
}

/// Manages background worker tasks per database.
pub struct WorkerRegistry {
    workers: Mutex<HashMap<i64, DatabaseWorkers>>,
    config: BackgroundSection,
    /// RFC 027 / v0.8.24: autonomous-maintenance cadence config.
    maintenance: MaintenanceSection,
    /// Cluster write-acceptance gate for the maintenance loop.
    write_gate: WriteAcceptanceGate,
}

struct DatabaseWorkers {
    cancel: CancellationToken,
    /// Kept alive so the task handles drop when this struct drops, which
    /// cancels the associated tasks via Drop. Never read explicitly — the
    /// Drop behaviour is the whole point.
    #[allow(dead_code)]
    handles: Vec<JoinHandle<()>>,
}

impl WorkerRegistry {
    pub fn new(
        config: &BackgroundSection,
        maintenance: &MaintenanceSection,
        write_gate: WriteAcceptanceGate,
    ) -> Self {
        Self {
            workers: Mutex::new(HashMap::new()),
            config: config.clone(),
            maintenance: maintenance.clone(),
            write_gate,
        }
    }

    /// Start background workers for a database engine.
    /// Call this when an engine is first loaded into the pool.
    pub fn start_for_database(&self, db_id: i64, db_name: String, engine: Arc<YantrikDB>) {
        let mut workers = self.workers.lock();
        if workers.contains_key(&db_id) {
            return; // Already running
        }

        let cancel = CancellationToken::new();
        let mut handles = Vec::new();

        // Consolidation worker
        if self.config.consolidation_interval_minutes > 0 {
            let interval = Duration::from_secs(self.config.consolidation_interval_minutes * 60);
            let engine = Arc::clone(&engine);
            let token = cancel.clone();
            let name = db_name.clone();
            let pause_threshold = self.config.enrichment_pause_threshold;
            handles.push(tokio::spawn(async move {
                consolidation_loop(engine, interval, token, name, pause_threshold).await;
            }));
        }

        // Decay sweep worker
        if self.config.decay_sweep_interval_minutes > 0 {
            let interval = Duration::from_secs(self.config.decay_sweep_interval_minutes * 60);
            let engine = Arc::clone(&engine);
            let token = cancel.clone();
            let name = db_name.clone();
            handles.push(tokio::spawn(async move {
                decay_loop(engine, interval, token, name).await;
            }));
        }

        // Stale session cleanup
        {
            let interval = Duration::from_secs(60 * 60); // every hour
            let engine = Arc::clone(&engine);
            let token = cancel.clone();
            let name = db_name.clone();
            handles.push(tokio::spawn(async move {
                session_cleanup_loop(engine, interval, token, name).await;
            }));
        }

        // WAL checkpoint — prevent unbounded WAL growth under steady writes.
        // PRAGMA wal_autocheckpoint handles normal usage but can fall behind
        // under sustained load. This explicit TRUNCATE checkpoint reclaims
        // the WAL file space entirely.
        {
            let interval = Duration::from_secs(5 * 60); // every 5 minutes
            let engine = Arc::clone(&engine);
            let token = cancel.clone();
            let name = db_name.clone();
            handles.push(tokio::spawn(async move {
                wal_checkpoint_loop(engine, interval, token, name).await;
            }));
        }

        // Oplog GC — keep oplog bounded for long-running clusters
        {
            let interval = Duration::from_secs(60 * 60); // every hour
            let keep_recent = 100_000;
            let engine = Arc::clone(&engine);
            let token = cancel.clone();
            let name = db_name.clone();
            handles.push(tokio::spawn(async move {
                run_oplog_gc_loop(engine, interval, keep_recent, token, name).await;
            }));
        }

        // NULL-embedding healthcheck (issue #20). Surfaces rows whose
        // embedder failed silently — see issue #19 for the writer-side
        // root cause that was fixed in v0.8.1. This check is the
        // observability defense layer: if a regression re-introduces
        // silent NULL writes, operators see it via the
        // `yantrikdb_null_embedding_count` Prometheus gauge instead of
        // discovering it when /v1/recall starts 500-ing.
        {
            let interval = Duration::from_secs(60 * 60); // every hour
            let engine = Arc::clone(&engine);
            let token = cancel.clone();
            let name = db_name.clone();
            let id = db_id;
            handles.push(tokio::spawn(async move {
                null_embedding_check_loop(engine, interval, token, name, id).await;
            }));
        }

        // RFC 027 / v0.8.24: the sleep cycle. Drive the engine's
        // run_maintenance_cycle() on a per-tenant timer so closing loops
        // (conflict burn-down, trigger prune, importance recalibration, entity
        // backfill, auto-relate) is structural, not voluntary. Cluster-safe:
        // runs only where writes are accepted (see WriteAcceptanceGate).
        if self.maintenance.enabled && self.maintenance.interval_secs > 0 {
            let interval = Duration::from_secs(self.maintenance.interval_secs);
            // Per-tenant jitter on the initial delay so N tenants don't fire
            // their first (and subsequent aligned) cycles in lockstep. Derive
            // it deterministically from db_id — no RNG needed, and stable
            // across restarts for reproducible scheduling.
            let jitter = Duration::from_secs((db_id.unsigned_abs() % 60) + 1);
            let initial_delay = Duration::from_secs(self.maintenance.initial_delay_secs) + jitter;
            let engine = Arc::clone(&engine);
            let token = cancel.clone();
            let name = db_name.clone();
            let mcfg = self.maintenance.clone();
            let gate = self.write_gate.clone();
            handles.push(tokio::spawn(async move {
                maintenance_loop(engine, interval, initial_delay, token, name, mcfg, gate).await;
            }));
        }

        tracing::info!(
            db_id,
            db_name = %db_name,
            worker_count = handles.len(),
            "background workers started"
        );

        workers.insert(db_id, DatabaseWorkers { cancel, handles });
    }

    /// Stop background workers for a database.
    ///
    /// Not currently called from anywhere — present as public API for
    /// graceful tenant eviction once that feature lands.
    #[allow(dead_code)]
    pub fn stop_for_database(&self, db_id: i64) {
        let mut workers = self.workers.lock();
        if let Some(db_workers) = workers.remove(&db_id) {
            db_workers.cancel.cancel();
            // Handles will be dropped — tasks will see cancellation and exit
            tracing::info!(db_id, "background workers stopped");
        }
    }

    /// Stop all workers (server shutdown).
    pub fn stop_all(&self) {
        let mut workers = self.workers.lock();
        for (db_id, db_workers) in workers.drain() {
            db_workers.cancel.cancel();
            tracing::debug!(db_id, "background workers cancelled");
        }
    }

    /// Number of databases with active workers.
    ///
    /// Not currently called — reserved for the /metrics and /health
    /// endpoints once they surface tenant-level worker state.
    #[allow(dead_code)]
    pub fn active_count(&self) -> usize {
        self.workers.lock().len()
    }
}

// ── Worker loops ────────────────────────────────────────────────

/// Floor for the auto-scaled enrichment-pressure threshold. Cap of 50
/// is "we're definitely not in healthy load territory" — any meaningful
/// production deployment has more capacity than this. Named so debug
/// logs reading "threshold=50" are immediately self-explanatory.
pub const ENRICHMENT_PAUSE_THRESHOLD_FLOOR: u64 = 50;

/// Pure-arithmetic helper: compute the effective threshold from
/// `delta_max` + `config_override`. Extracted from
/// [`effective_enrichment_threshold`] so tests can hit the math
/// without spinning up a `YantrikDB`.
fn enrichment_threshold_from(delta_max: u64, config_override: Option<u64>) -> u64 {
    if let Some(t) = config_override {
        return t;
    }
    (delta_max * 75 / 100).max(ENRICHMENT_PAUSE_THRESHOLD_FLOOR)
}

/// Compute the effective enrichment-pause threshold for a given engine.
///
/// Returns `config_override` if the operator pinned a value in
/// `[background] enrichment_pause_threshold`. Otherwise auto-scales:
/// 75% of `engine.delta_max()`, floored at
/// [`ENRICHMENT_PAUSE_THRESHOLD_FLOOR`]. The 75% mark catches "the
/// compactor is clearly behind" without firing under healthy load.
pub fn effective_enrichment_threshold(engine: &YantrikDB, config_override: Option<u64>) -> u64 {
    enrichment_threshold_from(engine.delta_max() as u64, config_override)
}

async fn consolidation_loop(
    engine: Arc<YantrikDB>,
    interval: Duration,
    cancel: CancellationToken,
    db_name: String,
    pause_threshold: Option<u64>,
) {
    // Initial delay — don't run immediately on startup
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_secs(30)) => {}
        _ = cancel.cancelled() => return,
    }

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = cancel.cancelled() => {
                tracing::debug!(db = %db_name, "consolidation worker shutting down");
                return;
            }
        }

        let result = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let db_name = db_name.clone();
            move || {
                let db = engine.as_ref();
                let _hold_timer = crate::metrics::LockHoldTimer::start("worker_consolidation");

                // RFC 010 PR-6.4 enrichment-pressure rule: under
                // sustained ingest load, enrichment work compounds the
                // pressure (extra SQL, extra recall, extra compactor
                // invalidation). yantrikdb-core's audit (commits 84318c0+,
                // CONCURRENCY.md) recommends pausing enrichment when
                // `count_pending_ops > 75% of delta_max`. Decay loop
                // does NOT participate — memory aging is wall-clock-
                // bound, not load-bound.
                let pending = db.count_pending_ops().unwrap_or(0).max(0) as u64;
                let threshold = effective_enrichment_threshold(db, pause_threshold);
                if pending > threshold {
                    crate::metrics::record_enrichment_paused(&db_name, pending);
                    tracing::debug!(
                        db = %db_name,
                        pending,
                        threshold,
                        "engine pressure: skipping consolidation tick"
                    );
                    return None;
                }
                crate::metrics::record_enrichment_resumed(&db_name);

                // Skip if too few memories
                let stats = db.stats(None);
                if let Ok(s) = &stats {
                    if s.active_memories < 10 {
                        return None;
                    }
                }

                let config = ThinkConfig {
                    run_consolidation: true,
                    run_conflict_scan: true,
                    run_pattern_mining: false,
                    run_personality: false,
                    consolidation_limit: 50,
                    ..ThinkConfig::default()
                };

                match db.think(&config) {
                    Ok(result) => Some(result),
                    Err(e) => {
                        tracing::error!(db = %db_name, error = %e, "consolidation failed");
                        None
                    }
                }
            }
        })
        .await;

        if let Ok(Some(result)) = result {
            if result.consolidation_count > 0 || result.conflicts_found > 0 {
                tracing::info!(
                    db = %db_name,
                    consolidated = result.consolidation_count,
                    conflicts = result.conflicts_found,
                    duration_ms = result.duration_ms,
                    "consolidation complete"
                );
            }
        }
    }
}

async fn decay_loop(
    engine: Arc<YantrikDB>,
    interval: Duration,
    cancel: CancellationToken,
    db_name: String,
) {
    // Initial delay
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_secs(60)) => {}
        _ = cancel.cancelled() => return,
    }

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = cancel.cancelled() => {
                tracing::debug!(db = %db_name, "decay worker shutting down");
                return;
            }
        }

        let result = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let db_name = db_name.clone();
            move || {
                let db = engine.as_ref();
                let _hold_timer = crate::metrics::LockHoldTimer::start("worker_decay");
                match db.decay(0.01) {
                    Ok(decayed) => Some(decayed.len()),
                    Err(e) => {
                        tracing::error!(db = %db_name, error = %e, "decay sweep failed");
                        None
                    }
                }
            }
        })
        .await;

        if let Ok(Some(count)) = result {
            if count > 0 {
                tracing::info!(db = %db_name, expired = count, "decay sweep complete");
            }
        }
    }
}

async fn session_cleanup_loop(
    engine: Arc<YantrikDB>,
    interval: Duration,
    cancel: CancellationToken,
    db_name: String,
) {
    // Initial delay
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_secs(120)) => {}
        _ = cancel.cancelled() => return,
    }

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = cancel.cancelled() => {
                tracing::debug!(db = %db_name, "session cleanup worker shutting down");
                return;
            }
        }

        let result = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let db_name = db_name.clone();
            move || {
                let db = engine.as_ref();
                let _hold_timer = crate::metrics::LockHoldTimer::start("worker_session_cleanup");
                match db.session_abandon_stale(24.0) {
                    Ok(count) => Some(count),
                    Err(e) => {
                        tracing::error!(db = %db_name, error = %e, "session cleanup failed");
                        None
                    }
                }
            }
        })
        .await;

        if let Ok(Some(count)) = result {
            if count > 0 {
                tracing::info!(db = %db_name, abandoned = count, "stale sessions cleaned up");
            }
        }
    }
}

/// Oplog garbage collection — prune old applied entries to bound storage growth.
///
/// Keeps the most recent N entries per database (default 100k), only deleting
/// entries that have been marked applied=1.
pub async fn run_oplog_gc_loop(
    engine: Arc<YantrikDB>,
    interval: Duration,
    keep_recent: usize,
    cancel: CancellationToken,
    db_name: String,
) {
    // Initial delay
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_secs(300)) => {}
        _ = cancel.cancelled() => return,
    }

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = cancel.cancelled() => {
                tracing::debug!(db = %db_name, "oplog GC worker shutting down");
                return;
            }
        }

        let result = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let db_name = db_name.clone();
            move || {
                let db = engine.as_ref();
                let _hold_timer = crate::metrics::LockHoldTimer::start("worker_oplog_gc");
                let conn = db.conn();

                // Count current oplog
                let total: i64 = conn
                    .query_row("SELECT COUNT(*) FROM oplog WHERE applied = 1", [], |r| {
                        r.get(0)
                    })
                    .unwrap_or(0);

                if (total as usize) <= keep_recent {
                    return Some(0);
                }

                // Delete oldest applied entries beyond keep_recent
                // Use HLC ordering since op_ids are time-sortable UUIDv7
                let to_delete = total as usize - keep_recent;
                let result = conn.execute(
                    "DELETE FROM oplog WHERE op_id IN (
                        SELECT op_id FROM oplog
                        WHERE applied = 1
                        ORDER BY hlc ASC, op_id ASC
                        LIMIT ?1
                    )",
                    rusqlite::params![to_delete as i64],
                );

                match result {
                    Ok(deleted) => Some(deleted),
                    Err(e) => {
                        tracing::error!(db = %db_name, error = %e, "oplog GC failed");
                        None
                    }
                }
            }
        })
        .await;

        if let Ok(Some(count)) = result {
            if count > 0 {
                tracing::info!(db = %db_name, pruned = count, "oplog GC complete");
            }
        }
    }
}

/// NULL-embedding healthcheck (issue #20).
///
/// Periodically counts rows in `memories` with `embedding IS NULL` per
/// tenant, emits the count to a Prometheus gauge, and logs a warning
/// when count > 0. The writer-side fix in v0.8.1 (issue #19) makes
/// this scenario impossible going forward; this loop catches:
/// - Pre-v0.8.1 data already on disk
/// - Future regressions that re-introduce the silent-NULL bug
/// - Operator-induced state (manual SQL inserts, bad imports)
///
/// Cheap query: a covering index on `embedding` could be added if the
/// COUNT(*) becomes expensive, but at typical tenant sizes the scan
/// is well under 100ms and runs hourly.
async fn null_embedding_check_loop(
    engine: Arc<YantrikDB>,
    interval: Duration,
    cancel: CancellationToken,
    db_name: String,
    db_id: i64,
) {
    // Initial delay — give the engine 5 minutes to settle after startup.
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_secs(5 * 60)) => {}
        _ = cancel.cancelled() => return,
    }

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = cancel.cancelled() => {
                tracing::debug!(db = %db_name, "null-embedding healthcheck shutting down");
                return;
            }
        }

        let result = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let db_name = db_name.clone();
            move || -> Option<i64> {
                let db = engine.as_ref();
                let _hold_timer =
                    crate::metrics::LockHoldTimer::start("worker_null_embedding_check");
                let conn = db.conn();
                match conn.query_row(
                    "SELECT COUNT(*) FROM memories WHERE embedding IS NULL",
                    [],
                    |r| r.get::<_, i64>(0),
                ) {
                    Ok(n) => Some(n),
                    Err(e) => {
                        tracing::warn!(
                            db = %db_name,
                            error = %e,
                            "null-embedding healthcheck query failed"
                        );
                        None
                    }
                }
            }
        })
        .await;

        if let Ok(Some(count)) = result {
            crate::metrics::set_null_embedding_count(db_id, count);
            if count > 0 {
                tracing::warn!(
                    db = %db_name,
                    null_embedding_count = count,
                    "null-embedding rows detected — these poison /v1/recall on the namespace; \
                     run `DELETE FROM memories WHERE embedding IS NULL` to remediate \
                     (issue #20)"
                );
            }
        }
    }
}

/// WAL checkpoint — truncate the write-ahead log to reclaim disk space.
///
/// PRAGMA wal_autocheckpoint handles normal cases, but under sustained write
/// load the WAL can grow faster than auto-checkpointing reclaims. This
/// explicit TRUNCATE checkpoint resets the WAL file to zero size.
async fn wal_checkpoint_loop(
    engine: Arc<YantrikDB>,
    interval: Duration,
    cancel: CancellationToken,
    db_name: String,
) {
    // Initial delay — let the engine stabilize before first checkpoint
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_secs(60)) => {}
        _ = cancel.cancelled() => return,
    }

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = cancel.cancelled() => {
                tracing::debug!(db = %db_name, "WAL checkpoint worker shutting down");
                return;
            }
        }

        let result = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let db_name = db_name.clone();
            move || {
                let db = engine.as_ref();
                let _hold_timer = crate::metrics::LockHoldTimer::start("worker_wal_checkpoint");
                let conn = db.conn();

                // Query WAL size before checkpoint for metrics
                let wal_pages: i64 = conn
                    .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(1))
                    .unwrap_or(0);

                if wal_pages > 0 {
                    tracing::debug!(
                        db = %db_name,
                        wal_pages,
                        "WAL checkpoint: truncated"
                    );
                }
                Some(wal_pages)
            }
        })
        .await;

        if let Ok(Some(pages)) = result {
            if pages > 100 {
                tracing::info!(
                    db = %db_name,
                    wal_pages = pages,
                    "WAL checkpoint: large WAL truncated"
                );
            }
        }
    }
}

/// Translate the server-side [`MaintenanceSection`] into the engine's
/// [`MaintenanceCycleConfig`]. The light idempotent passes are always on; the
/// heavy corpus-rewriting passes (split, repair) are gated by config.
pub fn maintenance_cycle_config(m: &MaintenanceSection) -> MaintenanceCycleConfig {
    MaintenanceCycleConfig {
        // Engine 0.15 added dry_run (a "preview" once ran wet and tombstoned
        // live records). This worker IS the real scheduled cycle, so false —
        // stated explicitly because the compiler now forces every Rust
        // caller to decide, which is the field working as intended.
        dry_run: false,
        run_think: true,
        burn_down_conflicts: true,
        prune_triggers: true,
        max_pending_triggers: m.max_pending_triggers,
        recalibrate_importance: true,
        backfill_entities: true,
        auto_relate: true,
        max_auto_relate_edges: m.max_auto_relate_edges,
        split_oversized: m.run_split_oversized,
        split_min_chars: m.split_min_chars,
        repair_artifacts: m.run_repair_artifacts,
    }
}

/// The sleep cycle (RFC 027 / v0.8.24). Drives `run_maintenance_cycle()` on a
/// per-tenant timer, gated for cluster safety (write-acceptance) and load
/// (backpressure). Idempotent: a missed or double-run tick converges.
#[allow(clippy::too_many_arguments)]
async fn maintenance_loop(
    engine: Arc<YantrikDB>,
    interval: Duration,
    initial_delay: Duration,
    cancel: CancellationToken,
    db_name: String,
    cfg: MaintenanceSection,
    write_gate: WriteAcceptanceGate,
) {
    // Initial (jittered) delay — let the engine settle and avoid a startup
    // stampede across tenants.
    tokio::select! {
        _ = tokio::time::sleep(initial_delay) => {}
        _ = cancel.cancelled() => return,
    }

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = cancel.cancelled() => {
                tracing::debug!(db = %db_name, "maintenance worker shutting down");
                return;
            }
        }

        // Cluster-safety gate: only mutate state where writes are accepted
        // (standalone or leader). On a follower this would fork the state
        // machine. `pause_during_replication_catchup = false` is an operator
        // escape hatch that disables the gate (unsafe on clusters).
        if cfg.pause_during_replication_catchup && !write_gate.accepts_writes() {
            crate::metrics::record_maintenance_skipped(&db_name, "not_write_accepting");
            tracing::debug!(
                db = %db_name,
                "maintenance: node does not accept writes (follower/learner) — skipping tick"
            );
            continue;
        }

        // Backpressure gate: hygiene is load-bound work. Reuse the existing
        // enrichment-pressure rule — if the engine is behind on its delta
        // tier, skip this tick rather than compound the pressure.
        let cfg_for_blocking = cfg.clone();
        let result = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let db_name = db_name.clone();
            move || {
                let db = engine.as_ref();
                let pending = db.count_pending_ops().unwrap_or(0).max(0) as u64;
                let threshold = effective_enrichment_threshold(db, None);
                if pending > threshold {
                    crate::metrics::record_maintenance_skipped(&db_name, "backpressure");
                    tracing::debug!(
                        db = %db_name,
                        pending,
                        threshold,
                        "maintenance: engine under pressure — skipping tick"
                    );
                    return None;
                }

                let _hold_timer = crate::metrics::LockHoldTimer::start("worker_maintenance");
                let start = std::time::Instant::now();
                let engine_cfg = maintenance_cycle_config(&cfg_for_blocking);
                match db.run_maintenance_cycle(&engine_cfg) {
                    Ok(report) => Some((report, start.elapsed().as_secs_f64() * 1000.0)),
                    Err(e) => {
                        crate::metrics::record_maintenance_failed(&db_name);
                        tracing::error!(db = %db_name, error = %e, "maintenance cycle failed");
                        None
                    }
                }
            }
        })
        .await;

        if let Ok(Some((report, duration_ms))) = result {
            let conflicts_resolved = report
                .conflicts
                .as_ref()
                .map(|c| c.auto_resolved)
                .unwrap_or(0);
            let triggers_pruned = report
                .triggers
                .as_ref()
                .map(|t| t.expired_overdue + t.expired_over_cap)
                .unwrap_or(0);
            crate::metrics::record_maintenance_cycle(
                &db_name,
                duration_ms,
                report.think_consolidations.unwrap_or(0) as u64,
                conflicts_resolved as u64,
                triggers_pruned as u64,
                report.entities_linked.unwrap_or(0) as u64,
                report.relations_upserted.unwrap_or(0) as u64,
                report.errors.len() as u64,
            );
            // Only log when the cycle did real work or hit a pass error.
            let did_work = report.think_consolidations.unwrap_or(0) > 0
                || conflicts_resolved > 0
                || triggers_pruned > 0
                || report.relations_upserted.unwrap_or(0) > 0
                || report.entities_linked.unwrap_or(0) > 0
                || !report.errors.is_empty();
            if did_work {
                tracing::info!(
                    db = %db_name,
                    duration_ms = duration_ms as u64,
                    consolidations = report.think_consolidations.unwrap_or(0),
                    conflicts_resolved,
                    triggers_pruned,
                    entities_linked = report.entities_linked.unwrap_or(0),
                    relations_upserted = report.relations_upserted.unwrap_or(0),
                    pass_errors = report.errors.len(),
                    "maintenance cycle complete"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pr_6_4_threshold_uses_config_override_when_set() {
        // Operator pin wins over auto-scale. Default delta_max=256 would
        // auto-scale to 192 (256*75/100); the override 50 wins.
        assert_eq!(enrichment_threshold_from(256, Some(50)), 50);
        assert_eq!(enrichment_threshold_from(1024, Some(2000)), 2000);
        assert_eq!(enrichment_threshold_from(0, Some(1)), 1);
    }

    #[test]
    fn pr_6_4_threshold_auto_scales_to_75_percent_of_delta_max() {
        // delta_max=256 -> 192. delta_max=1024 -> 768.
        assert_eq!(enrichment_threshold_from(256, None), 192);
        assert_eq!(enrichment_threshold_from(1024, None), 768);
        assert_eq!(enrichment_threshold_from(2048, None), 1536);
    }

    #[test]
    fn pr_6_4_threshold_floors_at_minimum_for_tiny_delta_max() {
        // delta_max=10 would auto-scale to 7, which is meaningless under
        // any real load. Floor at ENRICHMENT_PAUSE_THRESHOLD_FLOOR (50)
        // so the rule never fires under healthy small-deployment shape.
        assert_eq!(
            enrichment_threshold_from(10, None),
            ENRICHMENT_PAUSE_THRESHOLD_FLOOR
        );
        // Boundary: delta_max=66 -> 49 -> floored to 50.
        assert_eq!(enrichment_threshold_from(66, None), 50);
        // delta_max=68 -> 51 -> not floored.
        assert_eq!(enrichment_threshold_from(68, None), 51);
    }

    #[test]
    fn pr_6_4_threshold_zero_delta_max_floors() {
        // Defensive: delta_max=0 (hypothetical, before engine init)
        // shouldn't produce 0-threshold which would mean "pause always".
        assert_eq!(
            enrichment_threshold_from(0, None),
            ENRICHMENT_PAUSE_THRESHOLD_FLOOR
        );
    }

    #[test]
    fn pr_6_4_threshold_floor_constant_is_50() {
        // Pin the constant. Future-debugging readers benefit from
        // the explicit value showing up in test output.
        assert_eq!(ENRICHMENT_PAUSE_THRESHOLD_FLOOR, 50);
    }

    // ── RFC 027 / v0.8.24 maintenance ───────────────────────────────

    #[test]
    fn maintenance_config_maps_section_defaults() {
        // Default section → light passes on, heavy passes off, engine caps.
        let section = MaintenanceSection::default();
        let cfg = maintenance_cycle_config(&section);
        assert!(cfg.run_think);
        assert!(cfg.burn_down_conflicts);
        assert!(cfg.prune_triggers);
        assert!(cfg.recalibrate_importance);
        assert!(cfg.backfill_entities);
        assert!(cfg.auto_relate);
        // Heavy passes opt-in — off by default so a routine cycle stays cheap.
        assert!(!cfg.split_oversized);
        assert!(!cfg.repair_artifacts);
        assert_eq!(cfg.max_pending_triggers, 64);
        assert_eq!(cfg.max_auto_relate_edges, 500);
        assert_eq!(cfg.split_min_chars, 1500);
    }

    #[test]
    fn maintenance_config_honors_heavy_pass_optin() {
        let section = MaintenanceSection {
            run_split_oversized: true,
            run_repair_artifacts: true,
            max_pending_triggers: 16,
            max_auto_relate_edges: 100,
            split_min_chars: 2000,
            ..MaintenanceSection::default()
        };
        let cfg = maintenance_cycle_config(&section);
        assert!(cfg.split_oversized);
        assert!(cfg.repair_artifacts);
        assert_eq!(cfg.max_pending_triggers, 16);
        assert_eq!(cfg.max_auto_relate_edges, 100);
        assert_eq!(cfg.split_min_chars, 2000);
    }

    #[test]
    fn write_gate_standalone_always_accepts() {
        // Single-node: maintenance always runs.
        assert!(WriteAcceptanceGate::standalone().accepts_writes());
    }

    #[test]
    fn write_gate_reflects_live_predicate() {
        // The cluster-safety invariant: the gate is polled live, so a
        // follower→leader transition flips it. Simulate with a toggle.
        let leader = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let gate = {
            let leader = Arc::clone(&leader);
            WriteAcceptanceGate::from_fn(move || leader.load(std::sync::atomic::Ordering::Relaxed))
        };
        // Follower: maintenance must NOT run (would fork the state machine).
        assert!(!gate.accepts_writes());
        // Promoted to leader: now it runs.
        leader.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(gate.accepts_writes());
    }
}
