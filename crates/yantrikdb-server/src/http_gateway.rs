//! HTTP/JSON gateway on port 7438.
//!
//! Thin translation layer: JSON → Command → handler → JSON response.

use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, Query, State},
    http::{HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde_json::{json, Value};

use crate::auth;
use crate::command::Command;
use crate::handler::{self, CommandResult};
use crate::server::AppState;

type AppResult = Result<Json<Value>, (StatusCode, Json<Value>)>;

/// v0.8.7 (issue #28 follow-up): canonical cluster-state view that
/// prefers openraft when active, falls back to legacy raft-lite.
///
/// Without this helper, every endpoint that exposed cluster state
/// independently chose between the two layers — leading to split-state
/// bugs where /v1/health reports openraft truth while /v1/remember
/// rejects writes citing legacy raft-lite (which has no quorum once
/// openraft is the real write path).
#[derive(Debug, Clone)]
struct ClusterStateView {
    node_id: u64,
    role: String,
    term: u64,
    leader: Option<u64>,
    leader_addr: Option<String>,
    accepts_writes: bool,
    healthy: bool,
    raft_mode: &'static str,

    // PR 6.9 — replication-state visibility.
    //
    // These fields surface the openraft state machine's progress so
    // operators don't have to hit the separate /v1/cluster/raft endpoint
    // to reason about whether a follower is keeping up. All four are
    // additive — clients that don't know about them ignore them.
    //
    // **Honest values today:** through v0.8.13, handlers bypass
    // `MutationCommitter`, so the only entries openraft sees are
    // cluster bookkeeping (membership, init). `last_log_index` will be
    // small and constant; `replication_lag_log_entries` will read 0
    // even on a structurally broken cluster. The fields become
    // load-bearing once PR 6.4 (handler migration) ships at v0.8.13.
    /// Highest log index this node knows about, from openraft metrics.
    /// `None` for raft-lite or single-node deployments.
    last_log_index: Option<u64>,
    /// Highest log index this node's state machine has applied. `None`
    /// when no entries have been applied yet (or raft-lite / single-node).
    last_applied_index: Option<u64>,
    /// `last_log_index.saturating_sub(last_applied_index)` — entries
    /// received but not yet applied. On a healthy follower this stays
    /// near 0; growing means the apply path is stuck. On the leader
    /// this is also 0 (leader applies before commit). `None` only when
    /// neither index is known.
    replication_lag_log_entries: Option<u64>,
    /// Stable label for the local node's role-within-cluster, used
    /// by metric exporters that need a low-cardinality dimension.
    /// `Some("leader" | "follower" | "candidate" | "learner" | "shutdown")`
    /// in openraft mode; `None` otherwise.
    role_label: Option<&'static str>,
}

fn cluster_state_view(state: &AppState) -> Option<ClusterStateView> {
    if let Some(ref yrp) = state.yrp {
        let quarantined = yrp.quarantine_reasons().is_some();
        let s = *yrp.status.borrow();
        let (leader, leader_addr) = yrp.leader_hint();
        let role_label: &'static str = if quarantined {
            "quarantined"
        } else {
            match s.role {
                crate::yrp::replica::Role::Leader => "leader",
                crate::yrp::replica::Role::Follower => "follower",
                crate::yrp::replica::Role::Candidate | crate::yrp::replica::Role::PreCandidate => {
                    "candidate"
                }
            }
        };
        return Some(ClusterStateView {
            node_id: yrp.node_id.0,
            role: role_label.to_string(),
            term: s.term,
            leader,
            leader_addr,
            accepts_writes: !quarantined && s.role == crate::yrp::replica::Role::Leader,
            healthy: !quarantined && leader.is_some(),
            raft_mode: "yrp",
            last_log_index: Some(s.commit),
            last_applied_index: Some(s.applied),
            replication_lag_log_entries: Some(s.commit.saturating_sub(s.applied)),
            role_label: Some(role_label),
        });
    }
    if let Some(ref cluster) = state.cluster {
        return Some(ClusterStateView {
            node_id: cluster.node_id() as u64,
            role: format!("{:?}", cluster.state.leader_role()),
            term: cluster.state.current_term(),
            leader: cluster.state.current_leader().map(|id| id as u64),
            leader_addr: None,
            accepts_writes: cluster.state.accepts_writes(),
            healthy: cluster.is_healthy(),
            raft_mode: "raft-lite",
            last_log_index: None,
            last_applied_index: None,
            replication_lag_log_entries: None,
            role_label: None,
        });
    }
    None
}

/// Shared engine handle. Type alias keeps the complex nested generic out
/// of function signatures and avoids clippy::type_complexity.
type EngineHandle = Arc<yantrikdb::YantrikDB>;

/// Error tuple returned by auth-checking helpers.
type AppError = (StatusCode, Json<Value>);

/// Issue #39: every error response across `/v1/*` now emits the
/// canonical structured envelope from [`crate::api::errors`]:
///
/// ```json
/// {"error": {"code": "stable_id", "message": "human", "hint": "optional"}}
/// ```
///
/// This helper is a migration shim for ~125 pre-existing call sites
/// that emitted `{"error": "string"}` (Option A). Those sites get the
/// new envelope shape immediately with `code: "generic"`. Each call
/// site individually migrates to a specific code via
/// [`crate::api::errors::api_error`] over time.
///
/// **New code MUST NOT use `app_error()`.** Call `api_error(status,
/// ApiErrorCode::SomeSpecificCode, message)` directly.
fn app_error(status: StatusCode, message: impl Into<String>) -> AppError {
    crate::api::errors::api_error(status, crate::api::errors::ApiErrorCode::Generic, message)
}

/// Translate a [`crate::commit::CommitError`] into an HTTP response per
/// RFC 010 PR-6 §9. Centralized so every handler that consumes a
/// committer surfaces consistent status codes + body shapes.
///
/// PR 6.6 ships the translator. PR 6.4 wires handlers to call it as
/// they migrate from `engine.record()` to `submitter.submit()`.
///
/// Status code rationale:
///
/// | Variant | Status | Why |
/// |---|---|---|
/// | `NotLeader` | **307** Temporary Redirect | Standard HTTP clients follow redirects. The body carries `leader_id`/`leader_addr` for clients that don't. PR 6.4 will populate the `Location` response header at the call site (we can't here without restructuring `AppError`). |
/// | `OpIdCollision` | **409** Conflict | Client bug: the same op_id was used with a different mutation. Don't retry. |
/// | `UnexpectedLogIndex` | **409** Conflict | Concurrent write race. Re-read state and retry. |
/// | `Version` | **426** Upgrade Required | Wire-version mismatch in a rolling upgrade. Operator runbook: bring the rest of the cluster to the new version. |
/// | `NotYetImplemented` | **501** Not Implemented | Variant exists in the grammar but the apply path isn't ready (e.g. `PurgeMemory` until RFC 011 PR-3). |
/// | `StorageFailure` | **503** Service Unavailable | Transient SQLite / disk error. Retry. |
/// | `Shutdown` | **503** Service Unavailable | Don't retry on this node. Hit a peer. |
/// | `CommitTimeout` | **503** Service Unavailable | Retry — but reuse the op_id (in the body) so the retry is idempotent. |
fn commit_error_to_app_error(err: crate::commit::CommitError) -> AppError {
    use crate::commit::CommitError as C;
    match err {
        C::NotLeader {
            leader_id,
            leader_addr,
        } => (
            StatusCode::TEMPORARY_REDIRECT,
            Json(json!({
                "error": "not_leader",
                "leader_id": leader_id,
                "leader_addr": leader_addr,
            })),
        ),
        C::OpIdCollision {
            op_id,
            tenant_id,
            existing_index,
        } => (
            StatusCode::CONFLICT,
            Json(json!({
                "error": "op_id_collision",
                "op_id": op_id.to_string(),
                "tenant_id": tenant_id.0,
                "existing_index": existing_index,
            })),
        ),
        C::UnexpectedLogIndex {
            tenant_id,
            expected,
            actual,
        } => (
            StatusCode::CONFLICT,
            Json(json!({
                "error": "unexpected_log_index",
                "tenant_id": tenant_id.0,
                "expected": expected,
                "actual": actual,
            })),
        ),
        C::Version(verr) => (
            StatusCode::UPGRADE_REQUIRED,
            Json(json!({
                "error": "wire_version_mismatch",
                "detail": verr.to_string(),
            })),
        ),
        C::NotYetImplemented {
            variant,
            planned_rfc,
        } => (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": "not_implemented",
                "variant": variant,
                "planned_rfc": planned_rfc,
            })),
        ),
        C::StorageFailure { message } => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "storage_failure",
                "detail": message,
                "retry_after_ms": 1000,
            })),
        ),
        C::Shutdown => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "shutting_down",
                "retry_after_ms": 5000,
            })),
        ),
        C::CommitTimeout { op_id } => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "commit_timeout",
                "op_id": op_id.to_string(),
                "retry_after_ms": 1000,
            })),
        ),
    }
}

/// Extract database engine from Bearer token.
fn resolve_engine(
    state: &AppState,
    auth_header: Option<&str>,
) -> Result<(i64, EngineHandle), AppError> {
    let token = auth_header
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or_else(|| app_error(StatusCode::UNAUTHORIZED, "missing Bearer token"))?;

    // Cluster master token check
    if let Some(ref cluster) = state.cluster {
        if let Some(ref secret) = cluster.config.cluster_secret {
            if token == secret.as_str() {
                let control = state.control.lock();
                let db_record = control
                    .get_database("default")
                    .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
                    .ok_or_else(|| {
                        app_error(StatusCode::NOT_FOUND, "default database not found")
                    })?;
                drop(control);
                let engine = state
                    .pool
                    .get_engine(&db_record)
                    .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                state.workers.start_for_database(
                    db_record.id,
                    db_record.name.clone(),
                    std::sync::Arc::clone(&engine),
                );
                return Ok((db_record.id, engine));
            }
        }
    }

    let token_hash = auth::hash_token(token);
    let control = state.control.lock();
    let db_id = control
        .validate_token(&token_hash)
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or_else(|| app_error(StatusCode::UNAUTHORIZED, "invalid or revoked token"))?;

    let db_record = control
        .get_database_by_id(db_id)
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or_else(|| app_error(StatusCode::NOT_FOUND, "database not found"))?;
    drop(control);

    let engine = state
        .pool
        .get_engine(&db_record)
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Start background workers if not already running
    state.workers.start_for_database(
        db_id,
        db_record.name.clone(),
        std::sync::Arc::clone(&engine),
    );

    Ok((db_id, engine))
}

/// Execute a command on a blocking thread so a slow engine call (think,
/// consolidate, embed) cannot park a tokio worker. The engine and control
/// mutexes are `parking_lot::Mutex`, which must NEVER be held across an await
/// — running the whole call inside `spawn_blocking` makes that structurally
/// impossible.
///
/// Load shedding: if the inflight count exceeds MAX_INFLIGHT, reject with
/// 503 immediately instead of queuing. Better to fail fast than pile up.
async fn execute_cmd(
    engine: Arc<yantrikdb::YantrikDB>,
    cmd: Command,
    control: Arc<parking_lot::Mutex<crate::control::ControlDb>>,
    inflight: &std::sync::atomic::AtomicU32,
) -> AppResult {
    use std::sync::atomic::Ordering;

    // Load shed: reject if too many ops in flight.
    //
    // The decrement MUST run on every exit path including cancellation.
    // Earlier versions decremented inline after the await, which leaks
    // the counter when axum drops the future on client disconnect /
    // timeout — over hours of traffic the counter saturates at
    // MAX_INFLIGHT and the gate stays permanently closed (v0.8.9 field
    // observation: 256/256 leaked after 12h uptime, container restart
    // was the only recovery). RAII guard fixes it: Drop fires regardless
    // of how the future exits.
    struct InflightGuard<'a>(&'a std::sync::atomic::AtomicU32);
    impl Drop for InflightGuard<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    let current = inflight.fetch_add(1, Ordering::Relaxed);
    if current >= crate::server::MAX_INFLIGHT {
        inflight.fetch_sub(1, Ordering::Relaxed);
        return Err(app_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "server overloaded: {} inflight ops (max {}). Retry later.",
                current,
                crate::server::MAX_INFLIGHT,
            ),
        ));
    }
    let _inflight_guard = InflightGuard(inflight);

    // Extract a static op name from the command for lock-hold telemetry.
    // Strings are matched against `Command` variants; new variants need a
    // new arm here or they appear as "unknown" in metrics.
    let op_name: &'static str = match &cmd {
        Command::Remember { .. } => "remember",
        Command::RememberBatch { .. } => "remember_batch",
        Command::Recall { .. } => "recall",
        Command::Forget { .. } => "forget",
        Command::Stats => "stats",
        Command::Ping => "ping",
        _ => "other",
    };
    let result = tokio::task::spawn_blocking(move || {
        // Measure engine lock acquisition time for /metrics histograms
        let lock_start = std::time::Instant::now();
        let db = engine.as_ref();
        crate::metrics::record_engine_lock_wait(lock_start.elapsed());
        // Measure how long the engine mutex is HELD during the operation.
        // If hold > slow threshold (default 50ms), a warn-level log fires
        // with op name + duration so operators can identify which command
        // is starving concurrent requests.
        let hold_start = std::time::Instant::now();
        let result = handler::execute_with_guard(db, cmd, Some(control.as_ref()));
        crate::metrics::record_engine_lock_hold(op_name, hold_start.elapsed());
        result
    })
    .await
    .map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("join error: {e}"),
        )
    });

    // _inflight_guard drops here on every path (success, error, panic
    // unwind from spawn_blocking) — replaces the previous inline
    // fetch_sub that leaked under future cancellation.
    let result = result?;
    match result {
        Ok(CommandResult::Json(v)) => Ok(Json(v)),
        Ok(CommandResult::RecallResults { results, total }) => {
            Ok(Json(json!({ "results": results, "total": total })))
        }
        Ok(CommandResult::Pong) => Ok(Json(json!({ "status": "ok" }))),
        Err(e) => Err(app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

// ── Route handlers ──────────────────────────────────────────────

/// Shallow health check — always returns 200. Use for TCP-level LB probes.
async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut payload = json!({
        "status": "ok",
        "engines_loaded": state.pool.loaded_count(),
        // Issue #58 (agreement #6): probeable capabilities so clients can
        // feature-probe instead of discovering support by silent field
        // drops. mcp/hermes flip their client-side key refusals to
        // forwards when they see "idempotency_key" here.
        "capabilities": ["idempotency_key"],
    });
    if let Some(view) = cluster_state_view(&state) {
        // PR 6.9: payload gains last_log_index, last_applied_index,
        // replication_lag_log_entries, role_label. All additive — clients
        // that don't know about them ignore them. Values today are
        // honest-zero (handlers bypass commit log, so openraft sees only
        // cluster bookkeeping); they become operationally load-bearing
        // once PR 6.4 lands.
        payload["cluster"] = json!({
            "node_id": view.node_id,
            "role": view.role,
            "term": view.term,
            "leader": view.leader,
            "accepts_writes": view.accepts_writes,
            "healthy": view.healthy,
            "raft_mode": view.raft_mode,
            "last_log_index": view.last_log_index,
            "last_applied_index": view.last_applied_index,
            "replication_lag_log_entries": view.replication_lag_log_entries,
            "role_label": view.role_label,
        });
    }
    // RFC 028 §5: the honest quarantine surface. "Process up" and "data
    // servable" are different health dimensions — a quarantined node
    // answers this endpoint (that is the whole point) but says so loudly.
    if let Some(yrp) = &state.yrp {
        if let Some(reasons) = yrp.quarantine_reasons() {
            payload["status"] = json!("quarantined");
            payload["yrp_quarantine_reasons"] = json!(reasons);
        } else if yrp.engine_incomplete() {
            // RFC 028 Phase C: protocol-current but engine still
            // backfilling a compacted range — not read/lead eligible.
            let s = *yrp.status.borrow();
            payload["status"] = json!("engine_backfilling");
            payload["yrp_engine_incomplete"] = json!(true);
            payload["yrp_backfill_applied"] = json!(s.applied);
            payload["yrp_backfill_target"] = json!(s.backfill_target);
        }
    }
    Json(payload)
}

/// Deep health check — actively probes subsystems. Returns 200 if all
/// checks pass, 503 if any fail. Use for K8s readiness / smart LB probes.
///
/// Checks:
///   1. engine mutex acquirable within 100ms (via try_lock_for)
///   2. control.db responsive to a trivial SELECT
///   3. cluster quorum present (if clustered)
///
/// Each check reports pass/fail + latency in the response body.
async fn health_deep(State(state): State<Arc<AppState>>) -> (StatusCode, Json<Value>) {
    let mut checks = Vec::new();
    let mut all_pass = true;

    // 1. Engine mutex — can we acquire the default engine's lock within 100ms?
    //    A wedged engine would fail this check.
    {
        let engine_check = tokio::task::spawn_blocking({
            let control = state.control.clone();
            let pool = state.pool.clone();
            move || {
                let start = std::time::Instant::now();
                let db_record = {
                    let ctrl = control.lock();
                    ctrl.get_database("default").ok().flatten()
                };
                if let Some(rec) = db_record {
                    if let Ok(engine) = pool.get_engine(&rec) {
                        let timeout = std::time::Duration::from_millis(100);
                        if true
                        /* arc-shared engine always available */
                        {
                            let elapsed = start.elapsed();
                            return json!({
                                "check": "engine_lock",
                                "pass": true,
                                "latency_ms": elapsed.as_secs_f64() * 1000.0,
                            });
                        }
                    }
                }
                let elapsed = start.elapsed();
                json!({
                    "check": "engine_lock",
                    "pass": false,
                    "latency_ms": elapsed.as_secs_f64() * 1000.0,
                    "error": "could not acquire engine lock within 100ms",
                })
            }
        })
        .await
        .unwrap_or_else(|e| json!({"check": "engine_lock", "pass": false, "error": e.to_string()}));

        if !engine_check["pass"].as_bool().unwrap_or(false) {
            all_pass = false;
        }
        checks.push(engine_check);
    }

    // 2. Control DB — trivial SELECT to verify SQLite is responsive
    {
        let control_check = tokio::task::spawn_blocking({
            let control = state.control.clone();
            move || {
                let start = std::time::Instant::now();
                let ctrl = control.lock();
                match ctrl.list_databases() {
                    Ok(dbs) => {
                        let elapsed = start.elapsed();
                        json!({
                            "check": "control_db",
                            "pass": true,
                            "latency_ms": elapsed.as_secs_f64() * 1000.0,
                            "databases": dbs.len(),
                        })
                    }
                    Err(e) => {
                        let elapsed = start.elapsed();
                        json!({
                            "check": "control_db",
                            "pass": false,
                            "latency_ms": elapsed.as_secs_f64() * 1000.0,
                            "error": e.to_string(),
                        })
                    }
                }
            }
        })
        .await
        .unwrap_or_else(|e| json!({"check": "control_db", "pass": false, "error": e.to_string()}));

        if !control_check["pass"].as_bool().unwrap_or(false) {
            all_pass = false;
        }
        checks.push(control_check);
    }

    // 3. Cluster quorum (if clustered) — uses cluster_state_view so
    //    openraft is the source of truth when active.
    if let Some(view) = cluster_state_view(&state) {
        if !view.healthy {
            all_pass = false;
        }
        checks.push(json!({
            "check": "cluster_quorum",
            "pass": view.healthy,
            "node_id": view.node_id,
            "role": view.role,
            "term": view.term,
            "leader": view.leader,
            "raft_mode": view.raft_mode,
        }));
    }

    let status = if all_pass {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    // Surface at-rest encryption status. Issue #3 (yantrikos/yantrikdb)
    // reported there was no way to verify encryption was active without
    // running `strings` on the SQLite file. The TenantPool already knows
    // whether a master key was configured at startup.
    let encryption_enabled = state.pool.is_encrypted();
    let encryption_status = if encryption_enabled {
        json!({"enabled": true, "algorithm": "AES-256-GCM"})
    } else {
        json!({"enabled": false, "algorithm": null})
    };

    // RFC 009 admission/observability snapshot — exposed in /health/deep
    // so operators can inspect admission state without scraping /metrics.
    // The CLI (`yantrikdb cluster status`) and yql (`\admission`) both
    // surface these fields.
    let in_flight_used = state.admission.cfg.max_in_flight_recall
        - state.admission.in_flight_recall.available_permits();
    let expanded_used = state.admission.cfg.max_concurrent_expanded_recall
        - state.admission.expanded_recall.available_permits();
    let admission_state = json!({
        "hard_top_k_cap": state.admission.cfg.hard_top_k_cap,
        "max_request_body_bytes": state.admission.cfg.max_request_body_bytes,
        "in_flight_recall": {
            "max": state.admission.cfg.max_in_flight_recall,
            "in_use": in_flight_used,
        },
        "expanded_recall": {
            "max": state.admission.cfg.max_concurrent_expanded_recall,
            "in_use": expanded_used,
        },
    });

    let runtime_state = json!({
        "control_runtime_isolated": state.control_runtime.is_some(),
    });

    // RFC 017-A version snapshot — rolling-upgrade visibility for operators.
    // Cluster gate state is wired when RFC 010 PR-1 attaches a VersionGate
    // to AppState; until then this block surfaces the local build's
    // version primitives only. The shape is forward-compatible: operators
    // can rely on `version.wire`, `version.min_supported_wire`, and
    // `version.table_schema_versions` from RFC 017-A onward.
    let local_snap = crate::version::VersionSnapshot::local();
    let version_block = json!({
        "build_id": local_snap.build_id,
        "wire": local_snap.wire,
        "min_supported_wire": local_snap.min_supported_wire,
        "table_schema_versions": local_snap.table_schema_versions,
    });

    (
        status,
        Json(json!({
            "status": if all_pass { "healthy" } else { "degraded" },
            "encryption": encryption_status,
            "checks": checks,
            "admission": admission_state,
            "runtime": runtime_state,
            "version": version_block,
        })),
    )
}

/// Reject if the tenant would exceed their max_memories quota after adding
/// `count` new memories. Reads quota from control.db and current memory
/// count from the engine's stats.
fn check_memory_quota(
    state: &AppState,
    db_id: i64,
    engine: &EngineHandle,
    count: usize,
) -> Result<(), (StatusCode, Json<Value>)> {
    let quota = {
        let ctrl = state.control.lock();
        ctrl.get_quota(db_id).unwrap_or_default()
    };

    // Quick check via engine stats. v0.8.9: no outer mutex anymore;
    // call directly. stats() may briefly hold internal SQLite read
    // connection but doesn't block recalls (separate connection).
    let current = engine.stats(None).map(|s| s.active_memories).unwrap_or(0);

    if current + count as i64 > quota.max_memories {
        return Err(app_error(
            StatusCode::TOO_MANY_REQUESTS,
            format!(
                "would exceed memory quota: current={}, adding={}, max={}",
                current, count, quota.max_memories,
            ),
        ));
    }
    Ok(())
}

/// Reject if cluster is enabled and this node doesn't accept writes.
/// Uses [`cluster_state_view`] so openraft is the source of truth when active.
fn check_writable(state: &AppState) -> Result<(), (StatusCode, Json<Value>)> {
    let Some(view) = cluster_state_view(state) else {
        return Ok(()); // No cluster — single-node, accepts writes.
    };
    if view.accepts_writes {
        return Ok(());
    }
    let msg = match view.leader {
        Some(id) => format!("read-only: not the leader (current leader: node {})", id),
        None => "read-only: no leader elected".into(),
    };
    Err((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": msg,
            "leader_node_id": view.leader,
            "leader_addr": view.leader_addr,
            "raft_mode": view.raft_mode,
        })),
    ))
}

async fn remember(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("remember");
    check_writable(&state)?;
    let (db_id, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;

    // Quota check: max_memories
    check_memory_quota(&state, db_id, &engine, 1)?;

    let text: String = body["text"]
        .as_str()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'text'"))?
        .into();

    // Issue #19: pre-embed in the server before delegating to the
    // engine. If the embedder hiccups, we return 5xx synchronously so
    // the caller can retry — instead of silently storing a row with
    // `embedding=NULL` that poisons subsequent /v1/recall.
    let client_supplied: Option<Vec<f32>> = body.get("embedding").and_then(|v| {
        v.as_array().map(|a| {
            a.iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect()
        })
    });
    let embedding = resolve_embedding(state.as_ref(), &text, client_supplied).await?;

    // Issue #58: keyed writes route through the engine's atomic
    // claim-coupled path instead of the commit_log mutation. Inside
    // `record_with_idempotency` the claim and the row commit in ONE engine
    // transaction (v0.10 T07) — which is exactly the "claim is
    // origin-ingress AND commit-coupled" contract RFC 028 §7 requires. The
    // RFC-010 mutation path cannot carry that coupling yet (the applier's
    // `record_with_rid` has no claim parameter), so in cluster mode the key
    // is REFUSED loudly (agreement #6: an ignored key converts
    // "exactly-once" into "maybe-twice" invisibly — the worst failure mode
    // this feature exists to kill). YRP Phase B moves the claim into the
    // replicated log per RFC 028 §7.
    if let Some(key) = body
        .get("idempotency_key")
        .and_then(|v| v.as_str())
        .map(String::from)
    {
        // RFC 028 §7: in yrp mode the claim rides IN the replicated log
        // (checked at origin ingress, committed with its op, truncated
        // with its op) — the coupling the engine-atomic path provides on
        // single-node. This replaces the historical cluster-mode 501.
        if let Some(yrp) = state.yrp.clone() {
            return remember_with_idempotency_yrp(&state, db_id, yrp, body, text, embedding, key)
                .await;
        }
        return remember_with_idempotency(&state, engine, body, text, embedding, key).await;
    }

    // RFC 010 PR-6.4: route through commit_log instead of calling
    // engine.record() directly. Single-node mode: LocalSqliteSubmitter
    // applies inline. Cluster mode: RaftCommitter routes through
    // openraft → state machine apply → engine.record_with_rid on every
    // node. RID is allocated server-side (deterministic per-mutation,
    // not per-replica) and carried in the mutation body.
    let rid = uuid7::uuid7().to_string();
    let mutation = upsert_mutation_from_body(&body, text, embedding, &rid);

    let receipt = state
        .commit_log
        .commit(
            crate::commit::TenantId::new(db_id),
            mutation,
            crate::commit::CommitOptions::default(),
        )
        .await
        .map_err(commit_error_to_app_error)?;

    let _ = engine; // engine handle is held only for quota check
    Ok(Json(json!({
        "rid": rid,
        "log_index": receipt.log_index,
    })))
}

/// Build the deterministic `UpsertMemory` mutation for a `/v1/remember`
/// body. The rid is allocated by the CALLER before this (server-side,
/// per-mutation — never per-replica) and the server timestamp is
/// materialized here, so every node applies identical bytes.
fn upsert_mutation_from_body(
    body: &Value,
    text: String,
    embedding: Option<Vec<f32>>,
    rid: &str,
) -> crate::commit::MemoryMutation {
    let now_micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    crate::commit::MemoryMutation::UpsertMemory {
        rid: rid.to_string(),
        text,
        memory_type: body
            .get("memory_type")
            .and_then(|v| v.as_str())
            .unwrap_or("semantic")
            .into(),
        importance: body
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5),
        valence: body.get("valence").and_then(|v| v.as_f64()).unwrap_or(0.0),
        half_life: body
            .get("half_life")
            .and_then(|v| v.as_f64())
            .unwrap_or(168.0),
        metadata: body.get("metadata").cloned().unwrap_or(json!({})),
        namespace: body
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .into(),
        certainty: body
            .get("certainty")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0),
        domain: body
            .get("domain")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .into(),
        source: body
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("user")
            .into(),
        emotional_state: body
            .get("emotional_state")
            .and_then(|v| v.as_str())
            .map(String::from),
        embedding,
        extracted_entities: vec![],
        created_at_unix_micros: Some(now_micros),
        embedding_model: Some("default".into()),
    }
}

/// Issue #58: the keyed `/v1/remember` path — engine-atomic claim + row.
///
/// Response contract (converged with yantrikdb-mcp + yantrikdb-hermes-plugin
/// in issue #58; hermes's byte-identical-across-backends argument won the
/// 200-vs-4xx question):
/// - fresh write            → 200 `{rid}`
/// - same key + same text   → 200 `{rid}` (the ORIGINAL rid, zero writes —
///   the silent-HIT principle; indistinguishable from fresh by design)
/// - same key + diff text   → 200 `{stored:false, idempotency_conflict:true,
///   rid:<existing>}` — a claim-resolution RESULT, not a protocol error
/// - invalid key            → 400 (empty/whitespace/over-long — a caller bug)
/// - cluster mode           → 501 (refused until YRP Phase B couples the
///   claim into the replicated log; agreement #6 — never silently drop)
async fn remember_with_idempotency(
    state: &Arc<AppState>,
    engine: EngineHandle,
    body: Value,
    text: String,
    embedding: Option<Vec<f32>>,
    key: String,
) -> AppResult {
    if cluster_state_view(state).is_some() {
        return Err(app_error(
            StatusCode::NOT_IMPLEMENTED,
            "idempotency_key is not yet supported in cluster mode: the \
             replicated apply path cannot couple the claim to the commit \
             (issue #58 / RFC 028 §7). Retry without the key, or run \
             single-node.",
        ));
    }
    let memory_type: String = body
        .get("memory_type")
        .and_then(|v| v.as_str())
        .unwrap_or("semantic")
        .into();
    let importance = body
        .get("importance")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5);
    let valence = body.get("valence").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let half_life = body
        .get("half_life")
        .and_then(|v| v.as_f64())
        .unwrap_or(168.0);
    let metadata = body.get("metadata").cloned().unwrap_or(json!({}));
    let namespace: String = body
        .get("namespace")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .into();
    let certainty = body
        .get("certainty")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let domain: String = body
        .get("domain")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .into();
    let source: String = body
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("user")
        .into();
    let emotional_state: Option<String> = body
        .get("emotional_state")
        .and_then(|v| v.as_str())
        .map(String::from);

    let outcome = tokio::task::spawn_blocking(move || match embedding {
        Some(emb) => engine.record_with_idempotency(
            &text,
            &memory_type,
            importance,
            valence,
            half_life,
            &metadata,
            &emb,
            &namespace,
            certainty,
            &domain,
            &source,
            emotional_state.as_deref(),
            Some(&key),
        ),
        // No embedder + no client vector: the engine's text path applies
        // its own embedding policy, identical to the unkeyed route.
        None => engine.record_text_with_idempotency(
            &text,
            &memory_type,
            importance,
            valence,
            half_life,
            &metadata,
            &namespace,
            certainty,
            &domain,
            &source,
            emotional_state.as_deref(),
            Some(&key),
        ),
    })
    .await
    .map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("join error: {e}"),
        )
    })?;

    match outcome {
        Ok(rid) => Ok(Json(json!({ "rid": rid }))),
        Err(yantrikdb::YantrikDbError::IdempotencyConflict { existing_rid, .. }) => {
            Ok(Json(json!({
                "stored": false,
                "idempotency_conflict": true,
                "rid": existing_rid,
            })))
        }
        Err(yantrikdb::YantrikDbError::InvalidIdempotencyKey { reason }) => Err(app_error(
            StatusCode::BAD_REQUEST,
            format!("invalid idempotency_key: {reason}"),
        )),
        Err(e) => Err(app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("engine error: {e}"),
        )),
    }
}

/// RFC 028 §7: the keyed `/v1/remember` path in yrp mode — claim-in-log.
///
/// The claim is checked at the leader's origin ingress (the driver's
/// `propose_keyed`), carried inside the committed entry, and never
/// re-gated at apply. The response contract is byte-identical to the
/// single-node engine path (issue #58 convergence with yantrikdb-mcp +
/// hermes): fresh → `{rid}`; same key + same text → `{rid}` (original,
/// silent HIT); same key + different text → 200 conflict shape; invalid
/// key → 400. Retries during ANY replication state resolve through the
/// claims table + durable outcome store — never a double-write.
async fn remember_with_idempotency_yrp(
    state: &Arc<AppState>,
    db_id: i64,
    yrp: Arc<crate::yrp::runtime::YrpHandle>,
    body: Value,
    text: String,
    embedding: Option<Vec<f32>>,
    key: String,
) -> AppResult {
    if key.trim().is_empty() || key.len() > 512 {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "invalid idempotency_key: must be non-blank and at most 512 bytes",
        ));
    }
    // Replicated mutations must carry a materialized embedding — a None
    // here would fail-stop the apply worker on every node. Refuse loudly.
    let Some(embedding) = embedding else {
        return Err(app_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "keyed writes in yrp mode require an embedding (server embedder \
             unavailable and no client vector supplied)",
        ));
    };

    let tenant = crate::commit::TenantId::new(db_id);
    let rid = uuid7::uuid7().to_string();
    let op = crate::yrp::op::YrpOp {
        tenant_id: tenant,
        op_id: crate::commit::OpId::new_random(),
        mutation: upsert_mutation_from_body(&body, text.clone(), Some(embedding), &rid),
        idempotency_key: Some(key.clone()),
    };
    let claim = crate::yrp::op::claim_key_for_idempotency(tenant, &key);

    let outcome = match yrp.propose_and_wait(claim, &op).await {
        Ok(o) => o,
        Err(e) => {
            return Err(commit_error_to_app_error(
                crate::yrp::runtime::propose_err_to_commit(e, op.op_id),
            ))
        }
    };

    // Digest-collision guard: the durable outcome stores the FULL key
    // string; a mismatch means two distinct keys share a 64-bit digest.
    // Refuse rather than mis-dedupe (fail closed; astronomically rare).
    if outcome.key_str.as_deref() != Some(key.as_str()) {
        return Err(app_error(
            StatusCode::CONFLICT,
            "idempotency_key digest collision with a different stored key; \
             use a different key",
        ));
    }
    let original_rid = outcome.rid.clone().unwrap_or_default();
    if original_rid == rid {
        // Our entry won the claim: a fresh write.
        return Ok(Json(json!({ "rid": rid })));
    }

    // Deduped against an earlier committed entry. Distinguish silent-HIT
    // (same text) from conflict (different text) against the ORIGINAL
    // mutation in the local commit log (materialized at apply).
    let original_text = state
        .commit_log
        .read_range(tenant, outcome.tenant_log_index, 1)
        .await
        .ok()
        .and_then(|entries| entries.into_iter().next())
        .and_then(|e| match e.mutation {
            crate::commit::MemoryMutation::UpsertMemory { text, .. } => Some(text),
            _ => None,
        });
    match original_text {
        Some(t) if t == text => Ok(Json(json!({ "rid": original_rid }))),
        _ => Ok(Json(json!({
            "stored": false,
            "idempotency_conflict": true,
            "rid": original_rid,
        }))),
    }
}

/// RFC 028: peer wire route — YRP messages ride the HTTP plane as a
/// bincode envelope (see `yrp::transport`). Authenticated by the shared
/// cluster secret when configured; malformed envelopes are a 400.
async fn yrp_msg(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> AppResult {
    let Some(yrp) = &state.yrp else {
        return Err(app_error(StatusCode::NOT_FOUND, "yrp mode not enabled"));
    };
    if let Some(secret) = &yrp.cluster_secret {
        let presented = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        if presented != Some(secret.as_str()) {
            return Err(app_error(
                StatusCode::UNAUTHORIZED,
                "invalid cluster secret",
            ));
        }
    }
    let (from, msg) = crate::yrp::transport::decode_envelope(&body)
        .map_err(|e| app_error(StatusCode::BAD_REQUEST, e))?;
    // Chaos gate (RFC 010 PR-5 registry, codex D1): faults are evaluated
    // at the receive seam, BEFORE protocol handling. Drop answers 200 —
    // to the protocol, a dropped delivery is indistinguishable from
    // network loss, and its timers re-drive. Delays defer the DELIVERY,
    // not the HTTP response.
    match state
        .fault_registry
        .verdict(from as u32, yrp.node_id.0 as u32)
    {
        crate::debug::FaultVerdict::Drop => {
            return Ok(Json(json!({ "ok": true, "faulted": "dropped" })));
        }
        crate::debug::FaultVerdict::Delay(d) => {
            let yrp = yrp.clone();
            tokio::spawn(async move {
                tokio::time::sleep(d).await;
                let _ = yrp.deliver(from, msg);
            });
            return Ok(Json(json!({ "ok": true, "faulted": "delayed" })));
        }
        crate::debug::FaultVerdict::Deliver => {}
    }
    yrp.deliver(from, msg)
        .map_err(|e| app_error(StatusCode::BAD_REQUEST, e))?;
    Ok(Json(json!({ "ok": true })))
}

/// RFC 028 Phase C: serve an engine-backfill range to a beyond-GC
/// straggler. Cluster-secret gated; body is JSON `BackfillRequest`;
/// response is a bincode `Vec<(u64, LogEntry)>` the requester feeds to
/// its apply worker. A node that cannot fully cover the range refuses
/// (503) rather than serving a hole.
async fn yrp_backfill(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<crate::yrp::runtime::BackfillRequest>,
) -> Result<Vec<u8>, AppError> {
    let Some(yrp) = &state.yrp else {
        return Err(app_error(StatusCode::NOT_FOUND, "yrp mode not enabled"));
    };
    if let Some(secret) = &yrp.cluster_secret {
        let presented = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        if presented != Some(secret.as_str()) {
            return Err(app_error(
                StatusCode::UNAUTHORIZED,
                "invalid cluster secret",
            ));
        }
    }
    if req.cluster_id != yrp.cluster_id {
        return Err(app_error(StatusCode::BAD_REQUEST, "cluster_id mismatch"));
    }
    let rows = yrp
        .serve_backfill(req.from_index, req.to_index)
        .await
        .map_err(|e| app_error(StatusCode::SERVICE_UNAVAILABLE, e))?;
    bincode::serialize(&rows).map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("encode backfill: {e}"),
        )
    })
}

/// Admin studio: aggregated cluster topology. Fans out to every YRP
/// member's `/v1/health` (public, like this endpoint) and returns one
/// array the dashboard renders. Own-node state is read directly; peers
/// are fetched concurrently with a short timeout (unreachable peers are
/// marked `reachable: false`). Read-only cluster metadata — no auth, same
/// posture as `/v1/health`.
async fn yrp_topology(State(state): State<Arc<AppState>>) -> Json<Value> {
    let Some(yrp) = &state.yrp else {
        return Json(json!({ "raft_mode": cluster_state_view(&state)
            .map(|v| v.raft_mode).unwrap_or("disabled"), "nodes": [] }));
    };
    let self_id = yrp.node_id.0;
    let peers = yrp.peer_urls();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();

    let fetches = peers.into_iter().map(|(id, url)| {
        let client = client.clone();
        let own = id == self_id;
        // Own node: read the local view directly (no self-HTTP).
        let local = if own {
            cluster_state_view(&state)
        } else {
            None
        };
        async move {
            if own {
                let v = local;
                json!({
                    "node_id": id, "addr": url, "reachable": true, "self": true,
                    "role": v.as_ref().map(|c| c.role_label).unwrap_or(Some("unknown")),
                    "term": v.as_ref().map(|c| c.term),
                    "leader": v.as_ref().and_then(|c| c.leader),
                    "healthy": v.as_ref().map(|c| c.healthy),
                    "last_applied_index": v.as_ref().and_then(|c| c.last_applied_index),
                    "replication_lag": v.as_ref().and_then(|c| c.replication_lag_log_entries),
                })
            } else {
                match client
                    .get(format!("{}/v1/health", url.trim_end_matches('/')))
                    .send()
                    .await
                {
                    Ok(resp) => match resp.json::<Value>().await {
                        Ok(h) => {
                            let c = h.get("cluster").cloned().unwrap_or(json!({}));
                            json!({
                                "node_id": id, "addr": url, "reachable": true, "self": false,
                                "role": c.get("role_label").or(c.get("role")),
                                "term": c.get("term"),
                                "leader": c.get("leader"),
                                "healthy": c.get("healthy"),
                                "last_applied_index": c.get("last_applied_index"),
                                "replication_lag": c.get("replication_lag_log_entries"),
                                "status": h.get("status"),
                            })
                        }
                        Err(_) => {
                            json!({ "node_id": id, "addr": url, "reachable": false, "self": false })
                        }
                    },
                    Err(_) => {
                        json!({ "node_id": id, "addr": url, "reachable": false, "self": false })
                    }
                }
            }
        }
    });
    let nodes: Vec<Value> = futures::future::join_all(fetches).await;
    Json(json!({
        "raft_mode": "yrp",
        "cluster_id": yrp.cluster_id,
        "viewed_from": self_id,
        "nodes": nodes,
    }))
}

/// Admin studio: a single self-contained page served from the binary at
/// `/admin` — no build step, no external assets (works air-gapped). Polls
/// `/v1/cluster/topology` and renders the live cluster. Theme-aware.
async fn admin_studio() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("admin/studio.html"))
}

/// Issue #58: keyed `/v1/remember/batch` — the engine's atomic batch path.
/// Claims + rows commit in one transaction; a key conflict fails the WHOLE
/// batch (all-or-nothing), returning the same 200-conflict shape as the
/// single-item path. Cluster mode refuses (same reasoning as
/// [`remember_with_idempotency`]).
async fn remember_batch_with_idempotency(
    state: &Arc<AppState>,
    engine: EngineHandle,
    memories: Vec<crate::command::RememberInput>,
    item_keys: Vec<Option<String>>,
) -> AppResult {
    if cluster_state_view(state).is_some() {
        return Err(app_error(
            StatusCode::NOT_IMPLEMENTED,
            "idempotency_key is not yet supported in cluster mode: the \
             replicated apply path cannot couple the claim to the commit \
             (issue #58 / RFC 028 §7). Retry without keys, or run \
             single-node.",
        ));
    }
    // The engine batch API requires concrete vectors; by this point every
    // item either shipped one or was pre-embedded. A remaining None means
    // the server has no embedder — refuse honestly rather than land a
    // NULL-embedding row behind a dedup claim.
    let mut inputs = Vec::with_capacity(memories.len());
    for (i, (m, key)) in memories.into_iter().zip(item_keys).enumerate() {
        let Some(embedding) = m.embedding else {
            return Err(app_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "memories[{i}]: keyed batches require embeddings \
                     (configure a server embedder or supply 'embedding')"
                ),
            ));
        };
        inputs.push(yantrikdb::types::RecordInput {
            text: m.text,
            memory_type: m.memory_type,
            importance: m.importance,
            valence: m.valence,
            half_life: m.half_life,
            metadata: m.metadata,
            embedding,
            namespace: m.namespace,
            certainty: m.certainty,
            domain: m.domain,
            source: m.source,
            emotional_state: m.emotional_state,
            idempotency_key: key,
        });
    }
    let outcome = tokio::task::spawn_blocking(move || engine.record_batch(&inputs))
        .await
        .map_err(|e| {
            app_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("join error: {e}"),
            )
        })?;
    match outcome {
        Ok(rids) => {
            let count = rids.len();
            Ok(Json(json!({ "rids": rids, "count": count })))
        }
        Err(yantrikdb::YantrikDbError::IdempotencyConflict { existing_rid, .. }) => {
            Ok(Json(json!({
                "stored": false,
                "idempotency_conflict": true,
                "rid": existing_rid,
            })))
        }
        Err(yantrikdb::YantrikDbError::InvalidIdempotencyKey { reason }) => Err(app_error(
            StatusCode::BAD_REQUEST,
            format!("invalid idempotency_key: {reason}"),
        )),
        Err(e) => Err(app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("engine error: {e}"),
        )),
    }
}

/// Issue #19 helper: resolve the embedding for a `/v1/remember`-style
/// payload, failing fast if the server's built-in embedder hiccups.
///
/// Decision tree:
/// 1. Caller supplied `embedding` in the request body — use it as-is.
///    The client takes responsibility for its own embedding pipeline.
/// 2. Server has a configured embedder — call it via `spawn_blocking`
///    (the underlying ONNX model holds a `parking_lot::Mutex` that we
///    must not hold across an `await`). Cache hits return in
///    microseconds; misses pay the model cost.
/// 3. No embedder configured — return `None` and let the engine path
///    decide what to do (typically rejects the write or stores NULL,
///    depending on engine config; out of this fix's scope).
///
/// Failure mode: if the server embedder returns an error, return
/// `Err((500, ...))` immediately. The caller sees a clear failure
/// signal instead of a deceptive `200 {rid: ...}` for a row that's
/// actually broken.
async fn resolve_embedding(
    state: &AppState,
    text: &str,
    client_supplied: Option<Vec<f32>>,
) -> Result<Option<Vec<f32>>, AppError> {
    if client_supplied.is_some() {
        return Ok(client_supplied);
    }
    let Some(embedder) = state.pool.embedder().cloned() else {
        return Ok(None);
    };
    let owned_text = text.to_string();
    let result = tokio::task::spawn_blocking(move || {
        use yantrikdb::types::Embedder;
        embedder.embed(&owned_text)
    })
    .await
    .map_err(|join_err| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("embed task panicked: {}", join_err),
        )
    })?;
    match result {
        Ok(v) => Ok(Some(v)),
        Err(e) => {
            crate::metrics::increment_embedder_failure("remember");
            tracing::error!(
                error = %e,
                text_len = text.len(),
                "embedder failed during /v1/remember; refusing to write a row with NULL embedding"
            );
            Err(app_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "embedder failed: {} (issue #19 — write refused to prevent NULL-embedding row that would poison recall on this namespace; please retry)",
                    e
                ),
            ))
        }
    }
}

async fn remember_batch(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("remember_batch");
    check_writable(&state)?;
    let (db_id, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;

    let memories_arr = body
        .get("memories")
        .and_then(|v| v.as_array())
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'memories' array"))?;

    if memories_arr.is_empty() {
        return Ok(Json(json!({"rids": [], "count": 0})));
    }

    // Quota checks: batch size + total memory count
    let quota = {
        let ctrl = state.control.lock();
        ctrl.get_quota(db_id).unwrap_or_default()
    };

    if memories_arr.len() > quota.max_batch_size as usize {
        return Err(app_error(
            StatusCode::TOO_MANY_REQUESTS,
            format!(
                "batch size {} exceeds quota {} for this database",
                memories_arr.len(),
                quota.max_batch_size
            ),
        ));
    }

    check_memory_quota(&state, db_id, &engine, memories_arr.len())?;

    if memories_arr.len() > 10_000 {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "batch size exceeds 10000",
        ));
    }

    let mut memories = Vec::with_capacity(memories_arr.len());
    // Issue #58: per-item idempotency keys, positionally aligned with
    // `memories`. A batch-level key derives per-item as "{key}:{index}" —
    // the convention yantrikdb-mcp and yantrikdb-hermes-plugin already
    // shipped; mirroring it means retries dedupe identically across all
    // three surfaces. An explicit per-item key wins over the derivation.
    let mut item_keys: Vec<Option<String>> = Vec::with_capacity(memories_arr.len());
    let batch_key: Option<String> = body
        .get("idempotency_key")
        .and_then(|v| v.as_str())
        .map(String::from);
    for (i, m) in memories_arr.iter().enumerate() {
        item_keys.push(
            m.get("idempotency_key")
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| batch_key.as_ref().map(|k| format!("{k}:{i}"))),
        );
        let text = m
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                app_error(
                    StatusCode::BAD_REQUEST,
                    format!("memories[{}]: missing 'text'", i),
                )
            })?
            .to_string();
        memories.push(crate::command::RememberInput {
            text,
            memory_type: m
                .get("memory_type")
                .and_then(|v| v.as_str())
                .unwrap_or("semantic")
                .into(),
            importance: m.get("importance").and_then(|v| v.as_f64()).unwrap_or(0.5),
            valence: m.get("valence").and_then(|v| v.as_f64()).unwrap_or(0.0),
            half_life: m.get("half_life").and_then(|v| v.as_f64()).unwrap_or(168.0),
            metadata: m.get("metadata").cloned().unwrap_or(json!({})),
            namespace: m
                .get("namespace")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .into(),
            certainty: m.get("certainty").and_then(|v| v.as_f64()).unwrap_or(1.0),
            domain: m
                .get("domain")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .into(),
            source: m
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("user")
                .into(),
            emotional_state: m
                .get("emotional_state")
                .and_then(|v| v.as_str())
                .map(String::from),
            embedding: m.get("embedding").and_then(|v| {
                v.as_array().map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect()
                })
            }),
        });
    }

    // Issue #19: pre-embed any memory that didn't ship with a
    // client-supplied embedding. Batch the misses through one
    // `embed_batch` call so concurrent ONNX-mutex acquisitions are
    // coalesced (the embedder cache + batch path was wired in
    // commit `e52228e`). On embedder failure we return 5xx and the
    // entire batch is rejected — partial success would land
    // NULL-embedding rows in the engine and re-create the bug.
    if let Some(embedder) = state.pool.embedder().cloned() {
        let needs_embed: Vec<usize> = memories
            .iter()
            .enumerate()
            .filter(|(_, m)| m.embedding.is_none())
            .map(|(i, _)| i)
            .collect();
        if !needs_embed.is_empty() {
            let texts: Vec<String> = needs_embed
                .iter()
                .map(|&i| memories[i].text.clone())
                .collect();
            let result = tokio::task::spawn_blocking(move || {
                use yantrikdb::types::Embedder;
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                embedder.embed_batch(&refs)
            })
            .await
            .map_err(|e| {
                app_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("embed batch task panicked: {}", e),
                )
            })?;
            match result {
                Ok(embeddings) => {
                    for (idx, vec) in needs_embed.iter().zip(embeddings.into_iter()) {
                        memories[*idx].embedding = Some(vec);
                    }
                }
                Err(e) => {
                    crate::metrics::increment_embedder_failure("remember_batch");
                    tracing::error!(
                        error = %e,
                        miss_count = needs_embed.len(),
                        batch_size = memories.len(),
                        "embedder failed during /v1/remember/batch; refusing partial-NULL write"
                    );
                    return Err(app_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(
                            "embedder failed: {} (issue #19 — batch refused to prevent NULL-embedding rows; please retry)",
                            e
                        ),
                    ));
                }
            }
        }
    }

    // Issue #58: any key present routes the WHOLE batch through the
    // engine's atomic batch path (claims + rows in one transaction,
    // all-or-nothing on conflict — the embedded semantics mcp/hermes
    // locked). Mixed keyed/unkeyed items ride together; unkeyed items
    // simply carry no claim.
    if item_keys.iter().any(|k| k.is_some()) {
        return remember_batch_with_idempotency(&state, engine, memories, item_keys).await;
    }

    // RFC 010 PR-6.4: route every batch entry through commit_log. Each
    // entry gets its own (rid, op_id, log_index). On cluster mode this
    // is N round-trips through openraft; on single-node it's N inline
    // applier dispatches. Both paths preserve byte-determinism across
    // replicas.
    let now_micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let mut rids = Vec::with_capacity(memories.len());
    let mut last_log_index: u64 = 0;
    for m in memories {
        let rid = uuid7::uuid7().to_string();
        let mutation = crate::commit::MemoryMutation::UpsertMemory {
            rid: rid.clone(),
            text: m.text,
            memory_type: m.memory_type,
            importance: m.importance,
            valence: m.valence,
            half_life: m.half_life,
            metadata: m.metadata,
            namespace: m.namespace,
            certainty: m.certainty,
            domain: m.domain,
            source: m.source,
            emotional_state: m.emotional_state,
            embedding: m.embedding,
            extracted_entities: vec![],
            created_at_unix_micros: Some(now_micros),
            embedding_model: Some("default".into()),
        };
        let receipt = state
            .commit_log
            .commit(
                crate::commit::TenantId::new(db_id),
                mutation,
                crate::commit::CommitOptions::default(),
            )
            .await
            .map_err(commit_error_to_app_error)?;
        rids.push(rid);
        last_log_index = receipt.log_index;
    }
    let _ = engine; // engine handle held only for the quota check above
    let count = rids.len();
    Ok(Json(json!({
        "rids": rids,
        "count": count,
        "log_index": last_log_index,
    })))
}

async fn recall(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("recall");

    // Parse the two admission-relevant fields up front so we can reject
    // BEFORE auth or HNSW search runs. Order matters: bad requests should
    // burn the smallest possible amount of CPU.
    let top_k = body.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    // v1 default for `expand_entities` stays `true` (backwards-compat —
    // see RFC 009 §"Backwards-compat contract"). The /v2 endpoints
    // shipped in PR-6 will flip the default to `false`.
    let expand_entities = body
        .get("expand_entities")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // Observability: count + histogram every recall request, regardless
    // of fate. Dashboards key on these for traffic shape analysis.
    crate::metrics::record_recall_request("v1", expand_entities);
    crate::metrics::record_recall_top_k("v1", top_k);

    // Hard cap — RFC 009 §4 Layer 3. Reject before HNSW search so a
    // misconfigured client requesting top_k=10000 gets a 400 in
    // microseconds instead of saturating a voter for seconds.
    if let Err(reason) = crate::admission::check_top_k(top_k, state.admission.cfg.hard_top_k_cap) {
        let status = StatusCode::from_u16(reason.http_status()).unwrap_or(StatusCode::BAD_REQUEST);
        return Err((
            status,
            Json(json!({
                "error": reason.message(),
                "reason": reason.metric_label(),
                "hard_top_k_cap": state.admission.cfg.hard_top_k_cap,
            })),
        ));
    }

    // Acquire admission permits BEFORE auth/engine resolution. Rejecting
    // on capacity is cheaper than resolving the tenant for a request
    // we're going to reject anyway. RAII: permits drop on function exit.
    let _permits = match state
        .admission
        .acquire_recall_permits(expand_entities)
        .await
    {
        Ok(p) => p,
        Err(reason) => {
            let status = StatusCode::from_u16(reason.http_status())
                .unwrap_or(StatusCode::SERVICE_UNAVAILABLE);
            return Err((
                status,
                Json(json!({
                    "error": reason.message(),
                    "reason": reason.metric_label(),
                    "retry_after_ms": 200,
                })),
            ));
        }
    };

    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::Recall {
        query: body["query"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'query'"))?
            .into(),
        top_k,
        memory_type: body
            .get("memory_type")
            .and_then(|v| v.as_str())
            .map(String::from),
        include_consolidated: body
            .get("include_consolidated")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        expand_entities,
        namespace: body
            .get("namespace")
            .and_then(|v| v.as_str())
            .map(String::from),
        domain: body
            .get("domain")
            .and_then(|v| v.as_str())
            .map(String::from),
        source: body
            .get("source")
            .and_then(|v| v.as_str())
            .map(String::from),
        query_embedding: body.get("query_embedding").and_then(|v| {
            v.as_array().map(|a| {
                a.iter()
                    .filter_map(|x| x.as_f64().map(|f| f as f32))
                    .collect()
            })
        }),
        // v0.10: default false = current-value-by-default (superseded records
        // excluded — the supersession-aware behavior we benchmarked). Opt in
        // with `"include_superseded": true` for history/archaeology.
        include_superseded: body
            .get("include_superseded")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    };
    let mut resp = execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await?;
    annotate_fts5_fallback(&mut resp);
    Ok(resp)
}

/// Issue #39 task 199: dashboard reads a top-level `fallback` field on
/// /v1/recall responses to surface "semantic search returned nothing
/// useful; we fell back to FTS5 keyword matching". The engine itself
/// doesn't expose a single response-level marker, but the per-result
/// `why_retrieved` arrays gain `"keyword_match"` entries when FTS5
/// contributed the row. We use that as the signal:
///
/// - If any result was retrieved via `keyword_match` → emit
///   `"fallback": "fts5_keyword"`.
/// - Else emit `"fallback": null` so dashboards can branch on presence
///   of the field without first checking the engine version.
///
/// Modifies `resp` in place. No-op if the body isn't an object (e.g.,
/// an error envelope already on the way out — those don't reach here
/// because `execute_cmd` returns Err for those).
fn annotate_fts5_fallback(resp: &mut Json<Value>) {
    let body = match resp.0.as_object_mut() {
        Some(o) => o,
        None => return,
    };
    let has_keyword_match = body
        .get("results")
        .and_then(|v| v.as_array())
        .map(|results| {
            results.iter().any(|r| {
                r.get("why_retrieved")
                    .and_then(|w| w.as_array())
                    .map(|arr| arr.iter().any(|x| x.as_str() == Some("keyword_match")))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    let fallback = if has_keyword_match {
        Value::String("fts5_keyword".into())
    } else {
        Value::Null
    };
    body.insert("fallback".into(), fallback);
}

async fn forget(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("forget");
    check_writable(&state)?;
    let (db_id, _engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let rid: String = body["rid"]
        .as_str()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'rid'"))?
        .into();
    let namespace: String = body
        .get("namespace")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .into();
    let reason = body
        .get("reason")
        .and_then(|v| v.as_str())
        .map(String::from);
    let now_micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);

    // RFC 010 PR-6.4: route TombstoneMemory through commit_log so the
    // mutation replicates to followers via openraft (cluster) or applies
    // inline via LocalSqliteSubmitter (single-node). Engine state is
    // updated by the applier dispatch (engine.tombstone_with_rid).
    let mutation = crate::commit::MemoryMutation::TombstoneMemory {
        rid: rid.clone(),
        reason,
        requested_at_unix_micros: now_micros,
        namespace,
    };

    let receipt = state
        .commit_log
        .commit(
            crate::commit::TenantId::new(db_id),
            mutation,
            crate::commit::CommitOptions::default(),
        )
        .await
        .map_err(commit_error_to_app_error)?;

    Ok(Json(json!({
        "rid": rid,
        "found": true,
        "log_index": receipt.log_index,
    })))
}

async fn relate(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Result<impl IntoResponse, AppError> {
    let _timer = crate::metrics::HandlerTimer::new("relate");
    check_writable(&state)?;
    let (db_id, _engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let entity: String = body["entity"]
        .as_str()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'entity'"))?
        .into();
    let target: String = body["target"]
        .as_str()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'target'"))?
        .into();
    let rel_type: String = body["relationship"]
        .as_str()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'relationship'"))?
        .into();
    let weight = body.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let namespace: String = body
        .get("namespace")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .into();

    // RFC 010 PR-6.4: route UpsertEntityEdge through commit_log. Edge id
    // allocated server-side as UUIDv7, carried in the mutation so every
    // replica produces byte-identical edge state.
    let edge_id = uuid7::uuid7().to_string();
    let mutation = crate::commit::MemoryMutation::UpsertEntityEdge {
        edge_id: edge_id.clone(),
        src: entity,
        dst: target,
        rel_type,
        weight,
        namespace,
    };

    let receipt = state
        .commit_log
        .commit(
            crate::commit::TenantId::new(db_id),
            mutation,
            crate::commit::CommitOptions::default(),
        )
        .await
        .map_err(commit_error_to_app_error)?;

    let mut response = Json(json!({
        "edge_id": edge_id,
        "log_index": receipt.log_index,
    }))
    .into_response();
    response
        .headers_mut()
        .insert("deprecation", HeaderValue::from_static("true"));
    response.headers_mut().insert(
        "link",
        HeaderValue::from_static(r#"</v1/claim>; rel="successor-version""#),
    );
    Ok(response)
}

async fn ingest_claim(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("ingest_claim");
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::IngestClaim {
        src: body["src"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'src'"))?
            .into(),
        rel_type: body["rel_type"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'rel_type'"))?
            .into(),
        dst: body["dst"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'dst'"))?
            .into(),
        namespace: body
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .into(),
        polarity: body.get("polarity").and_then(|v| v.as_i64()).unwrap_or(1) as i32,
        modality: body
            .get("modality")
            .and_then(|v| v.as_str())
            .unwrap_or("asserted")
            .into(),
        valid_from: body.get("valid_from").and_then(|v| v.as_f64()),
        valid_to: body.get("valid_to").and_then(|v| v.as_f64()),
        extractor: body
            .get("extractor")
            .and_then(|v| v.as_str())
            .unwrap_or("manual")
            .into(),
        extractor_version: body
            .get("extractor_version")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        confidence_band: body
            .get("confidence_band")
            .and_then(|v| v.as_str())
            .unwrap_or("medium")
            .into(),
        source_memory_rid: body
            .get("source_memory_rid")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        span_start: body
            .get("span_start")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32),
        span_end: body
            .get("span_end")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32),
        weight: body.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0),
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

async fn add_alias(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("add_alias");
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::AddAlias {
        alias: body["alias"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'alias'"))?
            .into(),
        canonical_name: body["canonical_name"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'canonical_name'"))?
            .into(),
        namespace: body
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .into(),
        source: body
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("explicit")
            .into(),
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

async fn get_claims(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("get_claims");
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let entity = params
        .get("entity")
        .cloned()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'entity' query parameter"))?;
    let namespace = params.get("namespace").cloned();
    execute_cmd(
        engine,
        Command::GetClaims { entity, namespace },
        state.control.clone(),
        &state.inflight,
    )
    .await
}

async fn think(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("think");
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::Think {
        run_consolidation: body
            .get("run_consolidation")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        run_conflict_scan: body
            .get("run_conflict_scan")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        run_pattern_mining: body
            .get("run_pattern_mining")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        run_personality: body
            .get("run_personality")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        consolidation_limit: body
            .get("consolidation_limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize,
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

async fn conflicts(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("conflicts");
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::Conflicts {
        status: params.get("status").cloned(),
        conflict_type: params.get("conflict_type").cloned(),
        entity: params.get("entity").cloned(),
        namespace: params.get("namespace").cloned(),
        limit: params
            .get("limit")
            .and_then(|v| v.parse().ok())
            .unwrap_or(50),
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

async fn resolve_conflict(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    AxumPath(conflict_id): AxumPath<String>,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("resolve_conflict");
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::Resolve {
        conflict_id,
        strategy: body["strategy"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'strategy'"))?
            .into(),
        winner_rid: body
            .get("winner_rid")
            .and_then(|v| v.as_str())
            .map(String::from),
        new_text: body
            .get("new_text")
            .and_then(|v| v.as_str())
            .map(String::from),
        resolution_note: body
            .get("resolution_note")
            .and_then(|v| v.as_str())
            .map(String::from),
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

async fn session_start(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::SessionStart {
        namespace: body
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .into(),
        client_id: body
            .get("client_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .into(),
        metadata: body.get("metadata").cloned().unwrap_or(json!({})),
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

async fn session_end(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    AxumPath(session_id): AxumPath<String>,
    body: Option<Json<Value>>,
) -> AppResult {
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let summary =
        body.and_then(|Json(b)| b.get("summary").and_then(|v| v.as_str()).map(String::from));
    let cmd = Command::SessionEnd {
        session_id,
        summary,
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

async fn personality(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("personality");
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    execute_cmd(
        engine,
        Command::Personality,
        state.control.clone(),
        &state.inflight,
    )
    .await
}

async fn stats(State(state): State<Arc<AppState>>, headers: axum::http::HeaderMap) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("stats");
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    execute_cmd(
        engine,
        Command::Stats,
        state.control.clone(),
        &state.inflight,
    )
    .await
}

async fn create_database(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    check_writable(&state)?;
    // For now, any valid token can create databases
    let _ = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let name: String = body["name"]
        .as_str()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'name'"))?
        .to_string();

    // Create directly via control (no engine needed)
    let control = state.control.lock();
    if control
        .database_exists(&name)
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    {
        return Err(app_error(
            StatusCode::CONFLICT,
            format!("database '{}' already exists", name),
        ));
    }
    let id = control
        .create_database(&name, &name)
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    drop(control);

    // Create the data directory
    let db_dir = state.pool.data_dir().join(&name);
    std::fs::create_dir_all(&db_dir)
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(json!({
        "name": name,
        "id": id,
        "message": format!("database '{}' created", name),
    })))
}

/// POST /v1/cluster/promote — manually trigger an election from this node.
/// Useful for forced failover during ops. Requires the node to be a voter.
async fn cluster_promote(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    // Auth check (any valid token works)
    let _ = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;

    let Some(ref ctx) = state.cluster else {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "single-node mode — nothing to promote",
        ));
    };

    if !matches!(ctx.state.configured_role, crate::config::NodeRole::Voter) {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "this node is not a voter — cannot become leader",
        ));
    }

    if ctx.state.is_leader() {
        return Ok(Json(json!({
            "status": "already_leader",
            "node_id": ctx.node_id(),
            "term": ctx.state.current_term(),
        })));
    }

    let ctx_clone = std::sync::Arc::clone(ctx);
    tokio::spawn(async move {
        if let Err(e) = crate::cluster::election::start_election(ctx_clone).await {
            tracing::error!(error = %e, "manual promotion failed");
        }
    });

    Ok(Json(json!({
        "status": "election_started",
        "node_id": ctx.node_id(),
        "current_term": ctx.state.current_term(),
        "message": "check /v1/cluster in a few seconds to see the new leader"
    })))
}

/// GET /metrics — Prometheus-format metrics for monitoring.
async fn metrics(State(state): State<Arc<AppState>>) -> String {
    let mut out = String::new();

    out.push_str("# HELP yantrikdb_engines_loaded Number of engine instances currently loaded\n");
    out.push_str("# TYPE yantrikdb_engines_loaded gauge\n");
    out.push_str(&format!(
        "yantrikdb_engines_loaded {}\n",
        state.pool.loaded_count()
    ));

    // v0.8.7: Prometheus cluster gauges read from cluster_state_view so
    // openraft is the source of truth when active. Without this, on a
    // healthy openraft cluster, `yantrikdb_cluster_is_leader` shows 0 on
    // the actual leader (because legacy raft-lite has no quorum).
    if let Some(view) = cluster_state_view(&state) {
        out.push_str("# HELP yantrikdb_cluster_term Current Raft term\n");
        out.push_str("# TYPE yantrikdb_cluster_term gauge\n");
        out.push_str(&format!(
            "yantrikdb_cluster_term {{node_id=\"{}\",raft_mode=\"{}\"}} {}\n",
            view.node_id, view.raft_mode, view.term
        ));

        out.push_str("# HELP yantrikdb_cluster_is_leader Whether this node is currently the leader (1) or not (0)\n");
        out.push_str("# TYPE yantrikdb_cluster_is_leader gauge\n");
        out.push_str(&format!(
            "yantrikdb_cluster_is_leader {{node_id=\"{}\",raft_mode=\"{}\"}} {}\n",
            view.node_id,
            view.raft_mode,
            if view.accepts_writes { 1 } else { 0 }
        ));

        out.push_str(
            "# HELP yantrikdb_cluster_healthy Whether this node has quorum (1) or not (0)\n",
        );
        out.push_str("# TYPE yantrikdb_cluster_healthy gauge\n");
        out.push_str(&format!(
            "yantrikdb_cluster_healthy {{node_id=\"{}\",raft_mode=\"{}\"}} {}\n",
            view.node_id,
            view.raft_mode,
            if view.healthy { 1 } else { 0 }
        ));
    }
    // Peer reachability is raft-lite-specific.
    {
        if let Some(ref cluster) = state.cluster {
            out.push_str(
                "# HELP yantrikdb_cluster_peer_reachable Whether each peer is reachable\n",
            );
            out.push_str("# TYPE yantrikdb_cluster_peer_reachable gauge\n");
            for peer in cluster.peers.snapshot() {
                out.push_str(&format!(
                    "yantrikdb_cluster_peer_reachable {{addr=\"{}\",role=\"{:?}\"}} {}\n",
                    peer.addr,
                    peer.configured_role,
                    if peer.reachable { 1 } else { 0 }
                ));
            }
        }
    }

    // Per-database stats (default DB only for now).
    // IMPORTANT: do NOT hold control.lock() across engine.lock() — that
    // serializes /metrics behind any long-running engine call AND blocks all
    // auth (which needs control). Scope the control lock tightly, then drop
    // it before touching the engine.
    let default_db = {
        let control = state.control.lock();
        control.get_database("default").ok().flatten()
    };
    if let Some(rec) = default_db {
        if let Ok(engine) = state.pool.get_engine(&rec) {
            let stats_opt = {
                // v0.8.9: Arc<YantrikDB> direct (no outer mutex); stats()
                // uses engine's internal read connection pool.
                engine.stats(None).ok()
            };
            if let Some(stats) = stats_opt {
                {
                    out.push_str("# HELP yantrikdb_active_memories Number of active memories\n");
                    out.push_str("# TYPE yantrikdb_active_memories gauge\n");
                    out.push_str(&format!(
                        "yantrikdb_active_memories {{db=\"default\"}} {}\n",
                        stats.active_memories
                    ));

                    out.push_str(
                        "# HELP yantrikdb_consolidated_memories Number of consolidated memories\n",
                    );
                    out.push_str("# TYPE yantrikdb_consolidated_memories gauge\n");
                    out.push_str(&format!(
                        "yantrikdb_consolidated_memories {{db=\"default\"}} {}\n",
                        stats.consolidated_memories
                    ));

                    out.push_str("# HELP yantrikdb_edges Number of knowledge graph edges\n");
                    out.push_str("# TYPE yantrikdb_edges gauge\n");
                    out.push_str(&format!(
                        "yantrikdb_edges {{db=\"default\"}} {}\n",
                        stats.edges
                    ));

                    out.push_str(
                        "# HELP yantrikdb_open_conflicts Number of unresolved conflicts\n",
                    );
                    out.push_str("# TYPE yantrikdb_open_conflicts gauge\n");
                    out.push_str(&format!(
                        "yantrikdb_open_conflicts {{db=\"default\"}} {}\n",
                        stats.open_conflicts
                    ));

                    out.push_str("# HELP yantrikdb_operations_total Total operations\n");
                    out.push_str("# TYPE yantrikdb_operations_total counter\n");
                    out.push_str(&format!(
                        "yantrikdb_operations_total {{db=\"default\"}} {}\n",
                        stats.operations
                    ));
                }
            }
        }
    }

    // RFC 027 / v0.8.24 — autonomous-maintenance metrics. The write-rich/
    // close-poor dashboard: how often hygiene ran per tenant and what it
    // closed, plus skip/failure counters and the cycle-duration histogram.
    {
        let aggs = crate::metrics::maintenance_aggs_snapshot();
        if !aggs.is_empty() {
            out.push_str("# HELP yantrikdb_maintenance_runs_total Maintenance cycles completed\n");
            out.push_str("# TYPE yantrikdb_maintenance_runs_total counter\n");
            for (db, a) in &aggs {
                out.push_str(&format!(
                    "yantrikdb_maintenance_runs_total{{db=\"{db}\"}} {}\n",
                    a.runs
                ));
            }
            out.push_str(
                "# HELP yantrikdb_maintenance_conflicts_resolved_total Conflicts auto-resolved by maintenance\n",
            );
            out.push_str("# TYPE yantrikdb_maintenance_conflicts_resolved_total counter\n");
            for (db, a) in &aggs {
                out.push_str(&format!(
                    "yantrikdb_maintenance_conflicts_resolved_total{{db=\"{db}\"}} {}\n",
                    a.conflicts_resolved
                ));
            }
            out.push_str(
                "# HELP yantrikdb_maintenance_triggers_pruned_total Pending triggers expired by maintenance\n",
            );
            out.push_str("# TYPE yantrikdb_maintenance_triggers_pruned_total counter\n");
            for (db, a) in &aggs {
                out.push_str(&format!(
                    "yantrikdb_maintenance_triggers_pruned_total{{db=\"{db}\"}} {}\n",
                    a.triggers_pruned
                ));
            }
            out.push_str(
                "# HELP yantrikdb_maintenance_consolidations_total Consolidations performed by maintenance\n",
            );
            out.push_str("# TYPE yantrikdb_maintenance_consolidations_total counter\n");
            for (db, a) in &aggs {
                out.push_str(&format!(
                    "yantrikdb_maintenance_consolidations_total{{db=\"{db}\"}} {}\n",
                    a.consolidations
                ));
            }
            out.push_str(
                "# HELP yantrikdb_maintenance_entities_linked_total Memory-entity links backfilled by maintenance\n",
            );
            out.push_str("# TYPE yantrikdb_maintenance_entities_linked_total counter\n");
            for (db, a) in &aggs {
                out.push_str(&format!(
                    "yantrikdb_maintenance_entities_linked_total{{db=\"{db}\"}} {}\n",
                    a.entities_linked
                ));
            }
            out.push_str(
                "# HELP yantrikdb_maintenance_relations_upserted_total Co-occurrence edges upserted by maintenance\n",
            );
            out.push_str("# TYPE yantrikdb_maintenance_relations_upserted_total counter\n");
            for (db, a) in &aggs {
                out.push_str(&format!(
                    "yantrikdb_maintenance_relations_upserted_total{{db=\"{db}\"}} {}\n",
                    a.relations_upserted
                ));
            }
            out.push_str(
                "# HELP yantrikdb_maintenance_failures_total Maintenance cycles that returned an error\n",
            );
            out.push_str("# TYPE yantrikdb_maintenance_failures_total counter\n");
            for (db, a) in &aggs {
                out.push_str(&format!(
                    "yantrikdb_maintenance_failures_total{{db=\"{db}\"}} {}\n",
                    a.failures
                ));
            }
            out.push_str(
                "# HELP yantrikdb_maintenance_pass_errors_total Per-pass errors recorded across maintenance cycles\n",
            );
            out.push_str("# TYPE yantrikdb_maintenance_pass_errors_total counter\n");
            for (db, a) in &aggs {
                out.push_str(&format!(
                    "yantrikdb_maintenance_pass_errors_total{{db=\"{db}\"}} {}\n",
                    a.pass_errors
                ));
            }
        }

        let skipped = crate::metrics::maintenance_skipped_snapshot();
        if !skipped.is_empty() {
            out.push_str(
                "# HELP yantrikdb_maintenance_runs_skipped_total Maintenance ticks skipped, by reason\n",
            );
            out.push_str("# TYPE yantrikdb_maintenance_runs_skipped_total counter\n");
            for (db, reason, count) in &skipped {
                out.push_str(&format!(
                    "yantrikdb_maintenance_runs_skipped_total{{db=\"{db}\",reason=\"{reason}\"}} {count}\n"
                ));
            }
        }

        let (mcount, msum) = crate::metrics::maintenance_duration_totals();
        if mcount > 0 {
            out.push_str(
                "# HELP yantrikdb_maintenance_duration_ms Maintenance-cycle wall-clock duration (ms)\n",
            );
            out.push_str("# TYPE yantrikdb_maintenance_duration_ms summary\n");
            out.push_str(&format!("yantrikdb_maintenance_duration_ms_sum {msum}\n"));
            out.push_str(&format!(
                "yantrikdb_maintenance_duration_ms_count {mcount}\n"
            ));
        }
    }

    // Append per-handler histograms, lock-wait histograms, request counters
    out.push_str(&crate::metrics::global().render_prometheus());

    out
}

async fn cluster_status(State(state): State<Arc<AppState>>) -> Json<Value> {
    let Some(ref ctx) = state.cluster else {
        return Json(json!({
            "clustered": false,
            "message": "single-node mode (no replication)"
        }));
    };

    let peers: Vec<Value> = ctx
        .peers
        .snapshot()
        .into_iter()
        .map(|p| {
            json!({
                "node_id": p.node_id,
                "addr": p.addr,
                "role": format!("{:?}", p.configured_role).to_lowercase(),
                "reachable": p.reachable,
                "current_term": p.current_term,
                "last_seen_secs_ago": p.last_seen.map(|t| t.elapsed().as_secs_f64()),
            })
        })
        .collect();

    Json(json!({
        "clustered": true,
        "node_id": ctx.node_id(),
        "role": format!("{:?}", ctx.state.leader_role()),
        "configured_role": format!("{:?}", ctx.state.configured_role).to_lowercase(),
        "current_term": ctx.state.current_term(),
        "leader_id": ctx.state.current_leader(),
        "voted_for": ctx.state.voted_for(),
        "accepts_writes": ctx.state.accepts_writes(),
        "healthy": ctx.is_healthy(),
        "quorum_size": ctx.quorum_size(),
        "peers": peers,
    }))
}

async fn list_databases(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    let _ = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let databases = state
        .control
        .lock()
        .list_databases()
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let list: Vec<Value> = databases
        .iter()
        .map(|d| json!({ "id": d.id, "name": d.name, "created_at": d.created_at }))
        .collect();
    Ok(Json(json!({ "databases": list })))
}

/// Map a control-plane propose failure (RFC 029) to an HTTP response. A
/// follower answers 503 with the leader's address so the caller (studio or
/// operator CLI) redirects the admin write to the leader — the same posture
/// as a data-plane write against a follower.
fn control_propose_err(e: crate::yrp::runtime::YrpProposeError) -> AppError {
    use crate::yrp::runtime::YrpProposeError as E;
    match e {
        E::NotLeader { leader_addr, .. } => app_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "not the leader; retry this admin write against the leader{}",
                leader_addr.map(|a| format!(" at {a}")).unwrap_or_default()
            ),
        ),
        E::Timeout => app_error(
            StatusCode::GATEWAY_TIMEOUT,
            "control op timed out awaiting quorum",
        ),
        E::Unavailable(m) => app_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("control plane unavailable: {m}"),
        ),
    }
}

/// POST /v1/admin/databases — create a database as a replicated control op
/// (RFC 029). Body: `{"name": "...", "path"?: "...", "config"?: {...}}`.
/// Master-token gated. Returns `{"id", "name", "replicated"}`. In yrp mode
/// the create is committed through YRP and applied on every node; in
/// single-node mode it writes `control.db` directly.
async fn admin_create_database(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    require_master_token(&state, &headers)?;
    let name = body
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'name'"))?
        .to_string();
    let path = body
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(&name)
        .to_string();
    let config = body
        .get("config")
        .map(|c| c.to_string())
        .unwrap_or_else(|| "{}".to_string());

    if let Some(yrp) = &state.yrp {
        let created_at = chrono::Utc::now().to_rfc3339();
        let id = yrp
            .create_database_replicated(&name, &path, &config, created_at)
            .await
            .map_err(control_propose_err)?;
        Ok(Json(json!({ "id": id, "name": name, "replicated": true })))
    } else {
        let id = tokio::task::spawn_blocking({
            let control = state.control.clone();
            let (name, path) = (name.clone(), path.clone());
            move || control.lock().create_database(&name, &path)
        })
        .await
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(Json(json!({ "id": id, "name": name, "replicated": false })))
    }
}

/// POST /v1/admin/tokens — mint a token for a database, replicated (RFC
/// 029). Body: `{"database_id": N}` or `{"database": "name"}`, optional
/// `"label"`. Master-token gated. Returns `{"token"}` — the plaintext is
/// shown ONCE and never stored; only its SHA-256 hash is replicated
/// (Invariant 3).
async fn admin_create_token(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    require_master_token(&state, &headers)?;
    let db_id = if let Some(id) = body.get("database_id").and_then(|v| v.as_i64()) {
        id
    } else if let Some(dbname) = body.get("database").and_then(|v| v.as_str()) {
        let rec = tokio::task::spawn_blocking({
            let control = state.control.clone();
            let dbname = dbname.to_string();
            move || control.lock().get_database(&dbname)
        })
        .await
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        rec.ok_or_else(|| app_error(StatusCode::NOT_FOUND, format!("no database '{dbname}'")))?
            .id
    } else {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "provide 'database_id' or 'database'",
        ));
    };
    let label = body
        .get("label")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let raw = crate::auth::generate_token();
    let hash = crate::auth::hash_token(&raw);

    if let Some(yrp) = &state.yrp {
        let op = crate::yrp::control_op::ControlOp::CreateToken {
            db_id,
            token_hash: hash,
            label,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        yrp.propose_control(&op)
            .await
            .map_err(control_propose_err)?;
        Ok(Json(
            json!({ "token": raw, "database_id": db_id, "replicated": true }),
        ))
    } else {
        tokio::task::spawn_blocking({
            let control = state.control.clone();
            move || control.lock().create_token(&hash, db_id, &label)
        })
        .await
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(Json(
            json!({ "token": raw, "database_id": db_id, "replicated": false }),
        ))
    }
}

/// POST /v1/admin/tokens/revoke — revoke a token, replicated (RFC 029).
/// Body: `{"token": "ydb_..."}` or `{"hash": "..."}`. Master-token gated.
/// Because revocation replicates, a token revoked on the leader is refused
/// cluster-wide (the auth-read barrier that makes this instantaneous on
/// every node lands in RFC 029 increment 2).
async fn admin_revoke_token(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    require_master_token(&state, &headers)?;
    let hash = if let Some(t) = body.get("token").and_then(|v| v.as_str()) {
        crate::auth::hash_token(t)
    } else if let Some(h) = body.get("hash").and_then(|v| v.as_str()) {
        h.to_string()
    } else {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "provide 'token' or 'hash'",
        ));
    };

    if let Some(yrp) = &state.yrp {
        let op = crate::yrp::control_op::ControlOp::RevokeToken {
            token_hash: hash,
            revoked_at: chrono::Utc::now().to_rfc3339(),
        };
        yrp.propose_control(&op)
            .await
            .map_err(control_propose_err)?;
        Ok(Json(json!({ "revoked": true, "replicated": true })))
    } else {
        let revoked = tokio::task::spawn_blocking({
            let control = state.control.clone();
            move || control.lock().revoke_token(&hash)
        })
        .await
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(Json(json!({ "revoked": revoked, "replicated": false })))
    }
}

/// GET /v1/admin/control-snapshot — returns a full snapshot of the control
/// plane (databases + active tokens) for replication to followers.
///
/// Authenticated by cluster master token only. Called by the follower's
/// control-sync loop, not by end users.
async fn control_snapshot(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    // Require cluster master token
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or_else(|| app_error(StatusCode::UNAUTHORIZED, "missing Bearer token"))?;

    let is_master = state
        .cluster
        .as_ref()
        .and_then(|c| c.config.cluster_secret.as_ref())
        .map(|s| token == s.as_str())
        .unwrap_or(false);

    if !is_master {
        return Err(app_error(
            StatusCode::FORBIDDEN,
            "control-snapshot requires cluster master token",
        ));
    }

    let snapshot = tokio::task::spawn_blocking({
        let control = state.control.clone();
        move || control.lock().export_snapshot()
    })
    .await
    .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::to_value(snapshot).unwrap_or_default()))
}

/// POST /v1/admin/snapshot — create an online backup of a tenant database.
///
/// Takes a consistent snapshot by WAL-checkpointing then copying the SQLite
/// file while holding the engine lock. Returns the backup path + BLAKE3
/// checksum.
///
/// ## Authentication
///
/// Two acceptable tokens (issue #7 fix):
/// 1. **Cluster master token** — accepted always. Existing cluster-mode
///    behavior. Allows snapshotting any database.
/// 2. **Per-database token for the target database** — accepted in any
///    mode (single-node OR cluster). The token must authenticate against
///    the SAME database named in the request body. Single-node operators
///    have no cluster master token; this lets them snapshot their own
///    database with the token they already have.
///
/// Body: `{"database": "default", "output_dir": "/tmp/backups"}` (optional
/// output_dir, defaults to data_dir/snapshots/).
async fn admin_snapshot(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or_else(|| app_error(StatusCode::UNAUTHORIZED, "missing Bearer token"))?;

    let db_name = body
        .get("database")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();

    // Try cluster master first (preserves cluster-mode behavior).
    let is_master = state
        .cluster
        .as_ref()
        .and_then(|c| c.config.cluster_secret.as_ref())
        .map(|s| token == s.as_str())
        .unwrap_or(false);

    if !is_master {
        // Fall back to per-database token. The token must authenticate
        // against the SAME database being snapshotted — operators can't
        // use a token for db A to snapshot db B.
        let token_hash = auth::hash_token(token);
        let control = state.control.lock();
        let token_db_id = control
            .validate_token(&token_hash)
            .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .ok_or_else(|| app_error(StatusCode::UNAUTHORIZED, "invalid or revoked token"))?;
        let target_db = control
            .get_database(&db_name)
            .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .ok_or_else(|| {
                app_error(
                    StatusCode::NOT_FOUND,
                    format!("database '{}' not found", db_name),
                )
            })?;
        drop(control);
        if token_db_id != target_db.id {
            return Err(app_error(
                StatusCode::FORBIDDEN,
                format!(
                    "token does not authenticate database '{}' — provide cluster master token \
                     or a token for that specific database",
                    db_name
                ),
            ));
        }
    }

    let output_dir = body
        .get("output_dir")
        .and_then(|v| v.as_str())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| state.pool.data_dir().join("snapshots"));

    let control = state.control.clone();
    let pool = state.pool.clone();

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let db_record = {
            let ctrl = control.lock();
            ctrl.get_database(&db_name)?
                .ok_or_else(|| anyhow::anyhow!("database '{}' not found", db_name))?
        };

        let engine = pool.get_engine(&db_record)?;
        let db = engine.as_ref();

        // WAL checkpoint before snapshot for consistency
        let conn = db.conn();
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
        drop(conn);

        // Source path
        let src_dir = pool.data_dir().join(&db_record.path);
        let src_db = src_dir.join("yantrik.db");

        if !src_db.exists() {
            anyhow::bail!("database file not found: {:?}", src_db);
        }

        // Destination
        std::fs::create_dir_all(&output_dir)?;
        let ts = chrono_ts();
        let dest_name = format!("{}-{}.db", db_name, ts);
        let dest_path = output_dir.join(&dest_name);

        // Copy the database file
        std::fs::copy(&src_db, &dest_path)?;

        // Compute checksum
        let data = std::fs::read(&dest_path)?;
        let hash = blake3::hash(&data);
        let size = data.len();

        Ok(serde_json::json!({
            "database": db_name,
            "path": dest_path.to_str().unwrap_or(""),
            "size_bytes": size,
            "checksum_blake3": hash.to_hex().to_string(),
            "timestamp": ts,
        }))
    })
    .await
    .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(result))
}

// ──────────────────────────────────────────────────────────────────
// RFC 008 substrate surface — exposes the Warrant Flow primitives as
// HTTP endpoints so an agent (MCP client, local LLM, etc.) can ingest
// claims with source_lineage, read mobility/contest state, record
// cognitive moves, and audit flagged propositions. See the tool-
// discovery doc for the full surface.
// ──────────────────────────────────────────────────────────────────

async fn ingest_claim_with_lineage(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("ingest_claim_with_lineage");
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let source_lineage: Vec<String> = body
        .get("source_lineage")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let cmd = Command::IngestClaimWithLineage {
        src: body["src"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'src'"))?
            .into(),
        rel_type: body["rel_type"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'rel_type'"))?
            .into(),
        dst: body["dst"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'dst'"))?
            .into(),
        namespace: body
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .into(),
        polarity: body.get("polarity").and_then(|v| v.as_i64()).unwrap_or(1) as i32,
        modality: body
            .get("modality")
            .and_then(|v| v.as_str())
            .unwrap_or("asserted")
            .into(),
        valid_from: body.get("valid_from").and_then(|v| v.as_f64()),
        valid_to: body.get("valid_to").and_then(|v| v.as_f64()),
        extractor: body
            .get("extractor")
            .and_then(|v| v.as_str())
            .unwrap_or("manual")
            .into(),
        extractor_version: body
            .get("extractor_version")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        confidence_band: body
            .get("confidence_band")
            .and_then(|v| v.as_str())
            .unwrap_or("medium")
            .into(),
        source_memory_rid: body
            .get("source_memory_rid")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        weight: body.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0),
        source_lineage,
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

async fn get_mobility(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("get_mobility");
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::GetMobility {
        src: params
            .get("src")
            .cloned()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'src'"))?,
        rel_type: params
            .get("rel_type")
            .cloned()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'rel_type'"))?,
        dst: params
            .get("dst")
            .cloned()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'dst'"))?,
        namespace: params
            .get("namespace")
            .cloned()
            .unwrap_or_else(|| "default".to_string()),
        regime: params
            .get("regime")
            .cloned()
            .unwrap_or_else(|| "default".to_string()),
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

async fn get_contest(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("get_contest");
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::GetContest {
        src: params
            .get("src")
            .cloned()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'src'"))?,
        rel_type: params
            .get("rel_type")
            .cloned()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'rel_type'"))?,
        dst: params
            .get("dst")
            .cloned()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'dst'"))?,
        namespace: params
            .get("namespace")
            .cloned()
            .unwrap_or_else(|| "default".to_string()),
        regime: params
            .get("regime")
            .cloned()
            .unwrap_or_else(|| "default".to_string()),
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

async fn record_move_event(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("record_move_event");
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let string_array = |key: &str| -> Vec<String> {
        body.get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    };
    let cmd = Command::RecordMoveEvent {
        move_type: body["move_type"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'move_type'"))?
            .into(),
        operator_version: body
            .get("operator_version")
            .and_then(|v| v.as_str())
            .unwrap_or("v1")
            .into(),
        context_regime: body
            .get("context_regime")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        observability: body
            .get("observability")
            .and_then(|v| v.as_str())
            .unwrap_or("observed")
            .into(),
        inference_confidence: body.get("inference_confidence").and_then(|v| v.as_f64()),
        inference_basis: body
            .get("inference_basis")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            }),
        input_claim_ids: string_array("input_claim_ids"),
        output_claim_ids: string_array("output_claim_ids"),
        side_effect_claim_ids: string_array("side_effect_claim_ids"),
        dependencies: string_array("dependencies"),
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

async fn list_flagged(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("list_flagged");
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let flag_mask = params
        .get("flag_mask")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let limit = params
        .get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50);
    let cmd = Command::ListFlaggedPropositions { flag_mask, limit };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

// ── RFC 010 PR-5: Jepsen / debug surface ───────────────────────────

/// Verify the caller holds the cluster master token. Debug endpoints
/// are operator-only — they're destructive when used wrong (fault
/// injection drops cluster traffic), so PR-5 keeps the gate strict.
/// RFC 014-B will replace this with an RBAC scope check.
fn require_master_token(state: &AppState, headers: &axum::http::HeaderMap) -> Result<(), AppError> {
    let token = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or_else(|| app_error(StatusCode::UNAUTHORIZED, "missing Bearer token"))?;
    if let Some(ref cluster) = state.cluster {
        if let Some(ref secret) = cluster.config.cluster_secret {
            if token == secret.as_str() {
                return Ok(());
            }
        }
    }
    // In yrp mode the master/bootstrap secret lives on the YRP handle
    // (`state.cluster` may be `None`) — accept it there too. This is the
    // RFC 029 bootstrap-admin credential.
    if let Some(ref yrp) = state.yrp {
        if let Some(ref secret) = yrp.cluster_secret {
            if token == secret.as_str() {
                return Ok(());
            }
        }
    }
    // Single-node mode without a configured cluster secret: deny outright.
    // Safer than auto-allowing any valid bearer; debug endpoints SHOULD
    // require explicit operator opt-in via cluster_secret.
    Err(app_error(
        StatusCode::FORBIDDEN,
        "control-plane admin requires the cluster master token",
    ))
}

/// Whether this node currently accepts writes (leader or standalone).
/// Mirrors the wire/HTTP write-path gate so state-mutating admin actions
/// (like a manual maintenance run) don't fork the cluster state machine.
fn node_accepts_writes(state: &AppState) -> bool {
    if let Some(ref yrp) = state.yrp {
        return yrp.is_leader();
    }
    if let Some(ref cluster) = state.cluster {
        return cluster.state.accepts_writes();
    }
    true
}

#[derive(serde::Deserialize)]
struct MaintenanceTenantQuery {
    /// Restrict the action to a single named tenant (database). When absent,
    /// the action covers every database.
    tenant: Option<String>,
}

#[derive(serde::Deserialize, Default)]
struct MaintenanceRunBody {
    /// Run the heavy split-oversized-episodes pass (default false).
    #[serde(default)]
    split_oversized: bool,
    /// Run the heavy tool-call-artifact repair pass (default false).
    #[serde(default)]
    repair_artifacts: bool,
}

/// Resolve the set of (db_id, name, engine) the maintenance action targets.
/// `?tenant=<name>` selects one; absent selects all databases. Loads each
/// engine (and starts its workers) as a side effect.
fn resolve_maintenance_targets(
    state: &AppState,
    tenant: Option<&str>,
) -> Result<Vec<(i64, String, EngineHandle)>, AppError> {
    let records = {
        let control = state.control.lock();
        if let Some(name) = tenant {
            let rec = control
                .get_database(name)
                .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
                .ok_or_else(|| {
                    app_error(
                        StatusCode::NOT_FOUND,
                        format!("database '{name}' not found"),
                    )
                })?;
            vec![rec]
        } else {
            control
                .list_databases()
                .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        }
    };
    let mut out = Vec::with_capacity(records.len());
    for rec in records {
        let engine = state
            .pool
            .get_engine(&rec)
            .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        state
            .workers
            .start_for_database(rec.id, rec.name.clone(), std::sync::Arc::clone(&engine));
        out.push((rec.id, rec.name, engine));
    }
    Ok(out)
}

/// POST /v1/admin/maintenance/run — RFC 027 / v0.8.24. Operator-triggered
/// maintenance cycle. Master-token gated. `?tenant=<name>` runs one tenant;
/// absent runs all. Optional body enables the heavy opt-in passes.
///
/// Cluster-safe: refuses to run on a node that does not accept writes
/// (follower/learner), because `run_maintenance_cycle` mutates state and would
/// fork the state machine. Hit the leader instead.
async fn admin_maintenance_run(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<MaintenanceTenantQuery>,
    body: Option<Json<MaintenanceRunBody>>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("admin_maintenance_run");
    require_master_token(&state, &headers)?;

    if !node_accepts_writes(&state) {
        return Err(app_error(
            StatusCode::CONFLICT,
            "this node does not accept writes (follower/learner); \
             run maintenance on the leader",
        ));
    }

    let opts = body.map(|b| b.0).unwrap_or_default();
    let targets = resolve_maintenance_targets(&state, params.tenant.as_deref())?;

    let mut results = Vec::with_capacity(targets.len());
    for (_db_id, name, engine) in targets {
        let cfg = yantrikdb::MaintenanceCycleConfig {
            split_oversized: opts.split_oversized,
            repair_artifacts: opts.repair_artifacts,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let outcome = tokio::task::spawn_blocking(move || engine.run_maintenance_cycle(&cfg))
            .await
            .map_err(|e| {
                app_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("join error: {e}"),
                )
            })?;
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        match outcome {
            Ok(report) => {
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
                    &name,
                    duration_ms,
                    report.think_consolidations.unwrap_or(0) as u64,
                    conflicts_resolved as u64,
                    triggers_pruned as u64,
                    report.entities_linked.unwrap_or(0) as u64,
                    report.relations_upserted.unwrap_or(0) as u64,
                    report.errors.len() as u64,
                );
                results.push(json!({
                    "tenant": name,
                    "duration_ms": duration_ms,
                    "report": serde_json::to_value(&report).unwrap_or(Value::Null),
                }));
            }
            Err(e) => {
                crate::metrics::record_maintenance_failed(&name);
                results.push(json!({ "tenant": name, "error": e.to_string() }));
            }
        }
    }

    Ok(Json(json!({ "ran": results.len(), "results": results })))
}

/// GET /v1/admin/maintenance/status — RFC 027 / v0.8.24. Master-token gated.
/// Returns the last persisted maintenance-cycle summary per tenant plus worker
/// liveness and this node's write-acceptance state.
async fn admin_maintenance_status(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<MaintenanceTenantQuery>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("admin_maintenance_status");
    require_master_token(&state, &headers)?;

    let targets = resolve_maintenance_targets(&state, params.tenant.as_deref())?;
    let mut tenants = Vec::with_capacity(targets.len());
    for (_db_id, name, engine) in targets {
        let last = engine.last_maintenance_cycle().ok().flatten();
        let parsed = last
            .as_deref()
            .and_then(|s| serde_json::from_str::<Value>(s).ok());
        tenants.push(json!({
            "tenant": name,
            "last_maintenance_cycle": parsed,
        }));
    }

    Ok(Json(json!({
        "active_workers": state.workers.active_count(),
        "accepts_writes": node_accepts_writes(&state),
        "tenants": tenants,
    })))
}

#[derive(serde::Deserialize)]
struct CurrentParams {
    /// The append-only chain namespace whose head (current value) to read.
    namespace: String,
}

/// GET /v1/current — RFC 027 / v0.8.27 (pillar 3: trust on the wire).
///
/// **The structural current-value read.** Returns the head of an append-only
/// chain namespace — the record that is *current*, not the record that is most
/// similar. This is the one query class similarity search cannot answer at any
/// retrieval budget: stale records are often *more* similar than the current
/// one, so recall-with-bigger-k never converges on "what is true now" (measured
/// 0.00 for RAG at k=8/20/50, vs 0.78–1.00 for the substrate's revision chain).
/// `chain_head` resolves it in one exact read instead of a probabilistic guess.
///
/// 200 with the head record, or 404 when the chain is empty/unknown.
/// Tenant-scoped (engine resolved from the bearer token).
async fn current_value(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<CurrentParams>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("current_value");
    let (_db_id, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    if params.namespace.trim().is_empty() {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "namespace must not be empty",
        ));
    }
    let ns = params.namespace.clone();
    let head = tokio::task::spawn_blocking(move || engine.chain_head(&ns))
        .await
        .map_err(|e| {
            app_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("join error: {e}"),
            )
        })?
        .map_err(|e| {
            app_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("chain_head failed: {e}"),
            )
        })?;
    match head {
        Some(m) => Ok(Json(json!({
            "found": true,
            "namespace": params.namespace,
            "record": memory_to_record_item(&m),
        }))),
        None => Err(crate::api::errors::api_error(
            StatusCode::NOT_FOUND,
            crate::api::errors::ApiErrorCode::Generic,
            format!("no chain head for namespace '{}'", params.namespace),
        )),
    }
}

#[derive(serde::Deserialize)]
struct GapsParams {
    /// Restrict to one namespace. Omit for the whole tenant DB.
    namespace: Option<String>,
    /// Only surface queries asked at least this many times (default 2 — a
    /// repeated ask is a pattern, a one-off is noise).
    min_count: Option<u64>,
    /// Only surface queries whose mean best-hit score is at or below this
    /// (default 0.5 — i.e. we answered it poorly).
    max_avg_top_score: Option<f64>,
    limit: Option<usize>,
}

/// GET /v1/insights/gaps — RFC 027 / v0.8.27. **The substrate's known unknowns.**
///
/// Every recall logs query demand; this surfaces the queries that are asked
/// *often* and answered *badly*. Most memory systems can only tell an agent
/// what they know — this tells it what it keeps failing to answer, which is the
/// signal an agent can actually act on (go learn X). Pairs with
/// `/v1/session/digest?include_gaps=true` to close the loop at boot.
///
/// Tenant-scoped. Read-only (safe on any node, no write gate).
async fn insights_gaps(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<GapsParams>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("insights_gaps");
    let (_db_id, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let gaps = fetch_knowledge_gaps(
        engine,
        params.namespace.clone(),
        params.min_count.unwrap_or(2),
        params.max_avg_top_score.unwrap_or(0.5),
        params.limit.unwrap_or(10),
    )
    .await?;
    Ok(Json(json!({ "count": gaps.len(), "gaps": gaps })))
}

/// Shared by `/v1/insights/gaps` and the digest's `?include_gaps=true`.
/// Returns the gaps as JSON values so both call sites emit one shape.
async fn fetch_knowledge_gaps(
    engine: EngineHandle,
    namespace: Option<String>,
    min_count: u64,
    max_avg_top_score: f64,
    limit: usize,
) -> Result<Vec<Value>, AppError> {
    let gaps = tokio::task::spawn_blocking(move || {
        engine.knowledge_gaps(namespace.as_deref(), min_count, max_avg_top_score, limit)
    })
    .await
    .map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("join error: {e}"),
        )
    })?
    .map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("knowledge_gaps failed: {e}"),
        )
    })?;
    Ok(gaps
        .into_iter()
        .map(|g| {
            json!({
                "query": g.query,
                "count": g.count,
                "avg_top_score": g.avg_top_score,
                "avg_results": g.avg_results,
                "last_seen": g.last_seen,
                "last_seen_iso": unix_to_iso(g.last_seen),
            })
        })
        .collect())
}

#[derive(serde::Deserialize)]
struct SessionDigestParams {
    /// Append-only identity/narrative chain to read the head of (optional).
    namespace: Option<String>,
    /// v0.9.3 isolation scope (engine v0.9.4): filter the digest's content
    /// aggregates (top decisions, open conflicts + count) to this namespace,
    /// so a host composing one digest per tenant never mixes another tenant's
    /// memories in. Omit for whole-DB (single-tenant) behavior. Pending
    /// triggers remain global regardless (engine limitation).
    scope: Option<String>,
    max_decisions: Option<usize>,
    max_conflicts: Option<usize>,
    max_triggers: Option<usize>,
    snippet_chars: Option<usize>,
    /// v0.8.27: fold the substrate's known-unknowns into the boot briefing.
    /// Off by default so the digest stays the cheap call it was designed to be.
    include_gaps: Option<bool>,
    /// Cap for the folded-in gaps (default 5 — the digest is token-budgeted).
    max_gaps: Option<usize>,
    /// v0.10 / v0.8.28: re-admit superseded records into the digest's content
    /// aggregates. Default `false` — a boot briefing must report only CURRENT
    /// decisions; quoting a superseded decision back at a waking agent would
    /// mislead it. `true` is for history/archaeology packets.
    include_superseded: Option<bool>,
}

/// GET /v1/session/digest — RFC 027 / v0.8.25 (pillar 2: lifecycle).
///
/// One-call boot briefing the host injects at session start: narrative chain
/// head + live high-importance decisions + open conflicts + pending triggers +
/// the last maintenance summary. Structurally fixes substrate-underuse drift —
/// one cheap call replaces N recalls a fresh agent may not think to make.
/// Tenant-scoped (engine resolved from the bearer token).
///
/// v0.8.27: `?include_gaps=true` folds the substrate's known-unknowns into the
/// briefing, which turns the digest from *informative* into *actionable* — the
/// agent wakes up knowing not just what it knows, but what it keeps failing to
/// answer. Opt-in so the default digest stays cheap.
async fn session_digest(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<SessionDigestParams>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("session_digest");
    let (_db_id, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    // Gap scope follows the digest's content-isolation scope so a per-tenant
    // digest reports that tenant's gaps, not the whole DB's.
    let gap_scope = params.scope.clone();
    let include_gaps = params.include_gaps.unwrap_or(false);
    let max_gaps = params.max_gaps.unwrap_or(5);
    let cfg = yantrikdb::SessionDigestConfig {
        narrative_namespace: params.namespace,
        namespace: params.scope,
        max_decisions: params.max_decisions.unwrap_or(8),
        max_conflicts: params.max_conflicts.unwrap_or(5),
        max_triggers: params.max_triggers.unwrap_or(5),
        snippet_chars: params.snippet_chars.unwrap_or(240),
        // v0.10: default false = the boot briefing reports only CURRENT
        // records. Opt in with ?include_superseded=true for history packets.
        include_superseded: params.include_superseded.unwrap_or(false),
    };
    let engine_for_gaps = std::sync::Arc::clone(&engine);
    let digest = tokio::task::spawn_blocking(move || engine.session_digest(&cfg))
        .await
        .map_err(|e| {
            app_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("join error: {e}"),
            )
        })?
        .map_err(|e| {
            app_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("session_digest failed: {e}"),
            )
        })?;
    let mut body = serde_json::to_value(digest).unwrap_or(Value::Null);
    if include_gaps {
        let gaps = fetch_knowledge_gaps(engine_for_gaps, gap_scope, 2, 0.5, max_gaps).await?;
        if let Some(obj) = body.as_object_mut() {
            obj.insert("knowledge_gaps".into(), Value::Array(gaps));
        }
    }
    Ok(Json(body))
}

#[derive(serde::Deserialize)]
struct SessionEndBody {
    /// Free-text session summary to segment into atomic candidate facts.
    summary: String,
    /// Row-tag namespace to file the drafted facts under (default "default").
    namespace: Option<String>,
    /// Domain for the drafted facts (default "general").
    domain: Option<String>,
}

/// POST /v1/session/end — RFC 027 / v0.8.25 (pillar 2: lifecycle).
///
/// End-of-session capture: segments a session summary into atomic, provisional
/// candidate facts via the engine's `draft_memories_from_summary`, so sessions
/// stop leaving no trace. Tenant-scoped. Cluster-safe: this is a direct engine
/// write, so it refuses (409) on a node that does not accept writes — run it on
/// the leader.
async fn session_end_capture(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<SessionEndBody>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("session_end_capture");
    let (_db_id, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    if !node_accepts_writes(&state) {
        return Err(app_error(
            StatusCode::CONFLICT,
            "this node does not accept writes (follower/learner); \
             session capture must run on the leader",
        ));
    }
    if body.summary.trim().is_empty() {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "summary must not be empty",
        ));
    }
    let namespace = body.namespace.unwrap_or_else(|| "default".to_string());
    let domain = body.domain.unwrap_or_else(|| "general".to_string());
    let summary = body.summary;
    let drafted = tokio::task::spawn_blocking(move || {
        engine.draft_memories_from_summary(&summary, &namespace, &domain)
    })
    .await
    .map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("join error: {e}"),
        )
    })?
    .map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("draft_memories_from_summary failed: {e}"),
        )
    })?;
    let count = drafted.len();
    Ok(Json(json!({ "drafted": drafted, "count": count })))
}

async fn debug_history(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    AxumPath(tenant_id): AxumPath<i64>,
    axum::extract::Query(params): axum::extract::Query<crate::debug::history::HistoryParams>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("debug_history");
    require_master_token(&state, &headers)?;
    let resp = crate::debug::history::read_history(
        &state.commit_log,
        crate::commit::TenantId::new(tenant_id),
        &params,
    )
    .await
    .map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("read_history failed: {e}"),
        )
    })?;
    Ok(Json(serde_json::to_value(resp).map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("history serialize failed: {e}"),
        )
    })?))
}

#[derive(serde::Deserialize)]
struct DebugFaultInjectBody {
    fault: crate::debug::FaultKind,
    /// Auto-clear after this many seconds. Useful for self-cleaning
    /// chaos tests that don't want to leave a leaked fault on a node
    /// after a CI run dies.
    ttl_secs: Option<u64>,
}

async fn debug_fault_inject(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<DebugFaultInjectBody>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("debug_fault_inject");
    require_master_token(&state, &headers)?;
    let id = state.fault_registry.inject(body.fault, body.ttl_secs);
    Ok(Json(json!({
        "fault_id": id.to_string(),
    })))
}

async fn debug_fault_list(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("debug_fault_list");
    require_master_token(&state, &headers)?;
    let faults = state.fault_registry.list();
    Ok(Json(serde_json::to_value(faults).map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("fault list serialize failed: {e}"),
        )
    })?))
}

async fn debug_fault_clear(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("debug_fault_clear");
    require_master_token(&state, &headers)?;
    let n = state.fault_registry.clear();
    Ok(Json(json!({ "cleared": n })))
}

async fn debug_fault_remove(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    AxumPath(fault_id): AxumPath<u64>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("debug_fault_remove");
    require_master_token(&state, &headers)?;
    let removed = state
        .fault_registry
        .remove(crate::debug::FaultId::new(fault_id));
    if removed {
        Ok(Json(json!({ "removed": true })))
    } else {
        Err(app_error(
            StatusCode::NOT_FOUND,
            format!("no fault with id fault_{fault_id}"),
        ))
    }
}

// ── Phase 1 polish: jobs + migrations admin surface ────────────────

#[derive(serde::Deserialize)]
struct JobsListParams {
    /// Filter by tenant id (optional).
    tenant: Option<i64>,
    /// Filter by state ("Pending" | "Leased" | "Succeeded" | "Failed" | "Cancelled").
    state: Option<String>,
    /// Maximum entries to return. Capped at 500 server-side.
    limit: Option<usize>,
}

const MAX_JOBS_LIST_LIMIT: usize = 500;

async fn jobs_list(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<JobsListParams>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("jobs_list");
    require_master_token(&state, &headers)?;
    let tenant_filter = params.tenant.map(crate::commit::TenantId::new);
    let state_filter = params
        .state
        .as_deref()
        .and_then(crate::jobs::JobState::from_str);
    let limit = params.limit.unwrap_or(100).min(MAX_JOBS_LIST_LIMIT);
    let records = state
        .jobs
        .list(tenant_filter, state_filter, limit)
        .await
        .map_err(|e| {
            app_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("jobs list failed: {e}"),
            )
        })?;
    Ok(Json(serde_json::to_value(records).map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("jobs list serialize failed: {e}"),
        )
    })?))
}

async fn jobs_get(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    AxumPath(job_id_str): AxumPath<String>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("jobs_get");
    require_master_token(&state, &headers)?;
    let uuid = job_id_str.parse::<uuid7::Uuid>().map_err(|e| {
        app_error(
            StatusCode::BAD_REQUEST,
            format!("invalid job id `{job_id_str}`: {e}"),
        )
    })?;
    let record = state
        .jobs
        .get(crate::jobs::JobId(uuid))
        .await
        .map_err(|e| match e {
            crate::jobs::JobError::NotFound { .. } => {
                app_error(StatusCode::NOT_FOUND, e.to_string())
            }
            other => app_error(StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
        })?;
    Ok(Json(serde_json::to_value(record).map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("job serialize failed: {e}"),
        )
    })?))
}

async fn jobs_cancel(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    AxumPath(job_id_str): AxumPath<String>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("jobs_cancel");
    require_master_token(&state, &headers)?;
    let uuid = job_id_str.parse::<uuid7::Uuid>().map_err(|e| {
        app_error(
            StatusCode::BAD_REQUEST,
            format!("invalid job id `{job_id_str}`: {e}"),
        )
    })?;
    state
        .jobs
        .cancel(crate::jobs::JobId(uuid))
        .await
        .map_err(|e| match e {
            crate::jobs::JobError::NotFound { .. } => {
                app_error(StatusCode::NOT_FOUND, e.to_string())
            }
            crate::jobs::JobError::TerminalState { .. } => {
                app_error(StatusCode::CONFLICT, e.to_string())
            }
            other => app_error(StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
        })?;
    Ok(Json(json!({ "cancelled": job_id_str })))
}

async fn admin_migrations_list(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("admin_migrations");
    require_master_token(&state, &headers)?;
    // Migrations are tracked separately per SQLite file. Surface the
    // commit_log + jobs DBs (the two we know about). Future RFCs adding
    // per-tenant DBs will register them here.
    let mut all = serde_json::json!({});
    // Re-open each DB read-only via a fresh connection so we don't
    // compete with the live committer's mutex. SQLite read-only doesn't
    // lock-conflict with WAL writes.
    let commit_log_path = state.data_dir.join("commit_log.sqlite");
    let jobs_path = state.data_dir.join("jobs.sqlite");
    for (label, path) in [("commit_log", &commit_log_path), ("jobs", &jobs_path)] {
        let summary = match rusqlite::Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        ) {
            Ok(conn) => match crate::migrations::MigrationRunner::applied_summary(&conn) {
                Ok(rows) => serde_json::json!(rows
                    .iter()
                    .map(|(id, name)| serde_json::json!({"id": id, "name": name}))
                    .collect::<Vec<_>>()),
                Err(e) => serde_json::json!({"error": e.to_string()}),
            },
            Err(e) => serde_json::json!({"error": format!("open failed: {e}")}),
        };
        all[label] = summary;
    }
    Ok(Json(all))
}

/// Simple timestamp for backup filenames.
fn chrono_ts() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}", secs)
}

// ─────────────────────────────────────────────────────────────────────────
// RFC 022 §1 — Skill Substrate API (v0.8.11)
//
// Five thin endpoints over the existing memory primitives, hardcoded to
// `namespace=skill_substrate, metadata.record_type=skill, memory_type=
// procedural`. Schema-validated on define (skill_id format, applies_to
// entry regex, skill_type enum), no semantic ontology (no validation
// gates, no outcome rollups). The point is to standardise the shape so
// every program stops reinventing skill_define / skill_get / skill_recall
// in agent code with subtle bugs (the cross-lane bug pattern that drove
// this RFC).
//
// In v0.8.11 `skill_get` does scan-then-filter via `engine.list_memories`
// + client-side filter on `metadata.skill_id`. v0.8.12 replaces this with
// /v1/lookup (O(log N) via indexed metadata).
// ─────────────────────────────────────────────────────────────────────────

const SKILL_NAMESPACE: &str = "skill_substrate";
const OUTCOME_NAMESPACE: &str = "outcome_substrate";
const VALID_SKILL_TYPES: &[&str] = &["procedure", "reference", "lesson", "pattern", "rule"];

/// Validate a skill_id against `^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$` by hand.
/// Returns Err with a specific reason if invalid.
fn validate_skill_id(s: &str) -> Result<(), &'static str> {
    if s.len() < 4 || s.len() > 200 {
        return Err("skill_id length must be 4..200 characters");
    }
    let bytes = s.as_bytes();
    if !bytes[0].is_ascii_lowercase() {
        return Err("skill_id must start with a lowercase letter");
    }
    let mut has_dot = false;
    let mut last_was_dot = false;
    for &b in bytes {
        let c = b as char;
        let is_valid = c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '.';
        if !is_valid {
            return Err("skill_id contains invalid character (allowed: lowercase a-z, 0-9, _, .)");
        }
        if c == '.' {
            if last_was_dot {
                return Err("skill_id contains consecutive dots");
            }
            has_dot = true;
            last_was_dot = true;
        } else {
            last_was_dot = false;
        }
    }
    if !has_dot {
        return Err("skill_id must contain at least one '.' (dotted form, e.g. skill.foo.v1)");
    }
    if last_was_dot {
        return Err("skill_id must not end with '.'");
    }
    Ok(())
}

/// Validate an `applies_to` array entry against `^[a-z][a-z0-9_]*$`.
/// Critical for catching hyphen-vs-underscore drift bugs (cf. Brainstorm 3
/// round 2 deepseek's `meta_agent` vs `meta-agent` example).
fn validate_applies_to_entry(s: &str) -> Result<(), &'static str> {
    if s.is_empty() {
        return Err("applies_to entry must be non-empty");
    }
    let bytes = s.as_bytes();
    if !bytes[0].is_ascii_lowercase() {
        return Err("applies_to entry must start with a lowercase letter");
    }
    for &b in bytes {
        let c = b as char;
        if !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
            return Err(
                "applies_to entry contains invalid character (allowed: lowercase a-z, 0-9, _)",
            );
        }
    }
    Ok(())
}

/// `POST /v1/skills/define` — write a new skill record. Strict shape
/// validation, 409 on duplicate skill_id by default. Wraps /v1/remember
/// internally with namespace=skill_substrate, metadata.record_type=skill,
/// memory_type=procedural.
async fn skill_define(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("skill_define");
    check_writable(&state)?;

    // ── Schema validation ──────────────────────────────────
    let skill_id = body["skill_id"]
        .as_str()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'skill_id'"))?;
    validate_skill_id(skill_id)
        .map_err(|e| app_error(StatusCode::BAD_REQUEST, format!("INVALID_SKILL_ID: {}", e)))?;

    let body_text = body["body"]
        .as_str()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'body'"))?;
    if body_text.len() < 50 || body_text.len() > 5000 {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "INVALID_BODY_LENGTH: body must be 50..5000 characters",
        ));
    }

    let applies_to = body["applies_to"]
        .as_array()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing or non-array 'applies_to'"))?;
    if applies_to.is_empty() {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "EMPTY_APPLIES_TO: applies_to must be a non-empty array",
        ));
    }
    if applies_to.len() > 10 {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "TOO_MANY_APPLIES_TO: applies_to may have at most 10 entries",
        ));
    }
    let mut applies_to_strs = Vec::with_capacity(applies_to.len());
    for v in applies_to {
        let s = v.as_str().ok_or_else(|| {
            app_error(
                StatusCode::BAD_REQUEST,
                "INVALID_APPLIES_TO_ENTRY: each entry must be a string",
            )
        })?;
        validate_applies_to_entry(s).map_err(|e| {
            app_error(
                StatusCode::BAD_REQUEST,
                format!("INVALID_APPLIES_TO_ENTRY: {}", e),
            )
        })?;
        applies_to_strs.push(s.to_string());
    }

    let skill_type = body["skill_type"]
        .as_str()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'skill_type'"))?;
    if !VALID_SKILL_TYPES.contains(&skill_type) {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            format!(
                "INVALID_SKILL_TYPE: must be one of {:?}, got '{}'",
                VALID_SKILL_TYPES, skill_type
            ),
        ));
    }

    // ── Resolve engine + check duplicate ────────────────────
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;

    let on_conflict = body
        .get("on_conflict")
        .and_then(|v| v.as_str())
        .unwrap_or("reject");

    if let Some(existing_rid) = find_skill_rid(&engine, skill_id) {
        match on_conflict {
            "reject" => {
                return Err(app_error(
                    StatusCode::CONFLICT,
                    format!(
                        "SKILL_ID_CONFLICT: '{}' already exists (rid={}); pass on_conflict=update to overwrite or on_conflict=ignore to no-op",
                        skill_id, existing_rid
                    ),
                ));
            }
            "ignore" => {
                return Ok(Json(serde_json::json!({
                    "rid":         existing_rid,
                    "skill_id":    skill_id,
                    "namespace":   SKILL_NAMESPACE,
                    "memory_type": "procedural",
                    "on_conflict": "ignore"
                })));
            }
            "update" => {
                // Tombstone the existing rid; new write below proceeds.
                let _ = engine.forget(&existing_rid);
            }
            other => {
                return Err(app_error(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "INVALID_ON_CONFLICT: '{}' (allowed: reject, update, ignore)",
                        other
                    ),
                ));
            }
        }
    }

    // ── Build metadata + dispatch through Command::Remember ─
    let user_metadata = body.get("metadata").cloned().unwrap_or(Value::Null);
    let mut metadata = serde_json::json!({
        "record_type": "skill",
        "skill_id":    skill_id,
        "applies_to":  applies_to_strs,
        "skill_type":  skill_type,
    });
    // Merge user-supplied extra metadata fields without overwriting reserved keys.
    if let Value::Object(user_map) = user_metadata {
        if let Value::Object(meta_map) = &mut metadata {
            for (k, v) in user_map {
                meta_map.entry(k).or_insert(v);
            }
        }
    }

    execute_cmd(
        engine,
        Command::Remember {
            text: body_text.to_string(),
            memory_type: "procedural".to_string(),
            importance: body
                .get("importance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.7),
            valence: 0.0,
            half_life: 30.0 * 24.0 * 3600.0,
            metadata,
            namespace: SKILL_NAMESPACE.to_string(),
            certainty: 0.9,
            domain: "skill".to_string(),
            source: body
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("agent")
                .to_string(),
            emotional_state: None,
            embedding: None,
        },
        state.control.clone(),
        &state.inflight,
    )
    .await
}

/// `GET /v1/skills/{skill_id}` — exact lookup. v0.8.11 implementation:
/// scan namespace + filter on metadata.skill_id (slow path, O(N)).
/// v0.8.12 will replace with `/v1/lookup` for O(log N).
async fn skill_get(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Path(skill_id): axum::extract::Path<String>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("skill_get");
    validate_skill_id(&skill_id)
        .map_err(|e| app_error(StatusCode::BAD_REQUEST, format!("INVALID_SKILL_ID: {}", e)))?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;

    if let Some(rid) = find_skill_rid(&engine, &skill_id) {
        if let Ok(Some(mem)) = engine.get(&rid) {
            return Ok(Json(serde_json::json!({
                "rid":         mem.rid,
                "skill_id":    skill_id,
                "body":        mem.text,
                "namespace":   mem.namespace,
                "memory_type": mem.memory_type,
                "metadata":    mem.metadata,
                "created_at":  mem.created_at,
            })));
        }
    }
    Err(app_error(
        StatusCode::NOT_FOUND,
        format!("skill_id '{}' not found", skill_id),
    ))
}

/// `POST /v1/skills/search` — semantic search over skill_substrate.
/// Optional `applies_to` and `skill_type` filters are post-fetch in
/// v0.8.11; v0.8.13 makes them prefilter via the where-clause work.
async fn skill_search(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("skill_search");
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;

    let query = body["query"]
        .as_str()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'query'"))?
        .to_string();
    let top_k = body
        .get("top_k")
        .and_then(|v| v.as_u64())
        .unwrap_or(5)
        .min(50) as usize;
    let applies_to_filter: Option<String> = body
        .get("applies_to")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let skill_type_filter: Option<String> = body
        .get("skill_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Overfetch to allow post-filter to land top_k after exclusion.
    let fetch_k = (top_k * 4).max(20);
    execute_cmd_with_post_filter(
        engine,
        Command::Recall {
            query,
            top_k: fetch_k,
            memory_type: Some("procedural".to_string()),
            include_consolidated: false,
            expand_entities: false,
            namespace: Some(SKILL_NAMESPACE.to_string()),
            domain: Some("skill".to_string()),
            source: None,
            query_embedding: None,
            // Skill recall wants only CURRENT skills, never superseded ones.
            include_superseded: false,
        },
        state.control.clone(),
        &state.inflight,
        applies_to_filter,
        skill_type_filter,
        top_k,
    )
    .await
}

/// `POST /v1/skills/{skill_id}/outcome` — append-only outcome log. Engine
/// NEVER auto-rolls-up success_count on the parent skill (architectural
/// enforcement of schema-not-semantics).
async fn skill_record_outcome(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Path(skill_id): axum::extract::Path<String>,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("skill_record_outcome");
    check_writable(&state)?;
    validate_skill_id(&skill_id)
        .map_err(|e| app_error(StatusCode::BAD_REQUEST, format!("INVALID_SKILL_ID: {}", e)))?;
    let success = body["success"]
        .as_bool()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing or non-bool 'success'"))?;
    let context = body
        .get("context")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;

    // Require the skill to actually exist; otherwise outcome is dangling.
    if find_skill_rid(&engine, &skill_id).is_none() {
        return Err(app_error(
            StatusCode::NOT_FOUND,
            format!("skill_id '{}' not found", skill_id),
        ));
    }

    let metadata = serde_json::json!({
        "record_type": "skill_outcome",
        "skill_ref":   skill_id,
        "success":     success,
        "context":     context,
    });
    execute_cmd(
        engine,
        Command::Remember {
            text: format!(
                "Outcome for {}: success={} — {}",
                skill_id, success, context
            ),
            memory_type: "episodic".to_string(),
            importance: 0.5,
            valence: if success { 0.3 } else { -0.3 },
            half_life: 90.0 * 24.0 * 3600.0,
            metadata,
            namespace: OUTCOME_NAMESPACE.to_string(),
            certainty: 1.0,
            domain: "skill_outcome".to_string(),
            source: "skill_api".to_string(),
            emotional_state: None,
            embedding: None,
        },
        state.control.clone(),
        &state.inflight,
    )
    .await
}

/// `POST /v1/skills/{skill_id}/forget` — tombstone the skill. Optional
/// cascade_outcomes (default false) for explicit hard-delete of outcome
/// records too.
async fn skill_forget(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Path(skill_id): axum::extract::Path<String>,
    Json(_body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("skill_forget");
    check_writable(&state)?;
    validate_skill_id(&skill_id)
        .map_err(|e| app_error(StatusCode::BAD_REQUEST, format!("INVALID_SKILL_ID: {}", e)))?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;

    let rid = find_skill_rid(&engine, &skill_id).ok_or_else(|| {
        app_error(
            StatusCode::NOT_FOUND,
            format!("skill_id '{}' not found", skill_id),
        )
    })?;

    execute_cmd(
        engine,
        Command::Forget { rid },
        state.control.clone(),
        &state.inflight,
    )
    .await
}

/// Helper: scan skill_substrate and find the rid whose
/// `metadata.skill_id` equals the given id. v0.8.11 stopgap; v0.8.12
/// replaces with /v1/lookup. Bounded scan limit (10000) so the worst
/// case is bounded even with thousands of skills.
fn find_skill_rid(engine: &Arc<yantrikdb::YantrikDB>, skill_id: &str) -> Option<String> {
    let (memories, _total) = engine
        .list_memories(
            10000,
            0,
            Some("skill"),
            Some("procedural"),
            Some(SKILL_NAMESPACE),
            "created_at",
        )
        .ok()?;
    for mem in memories {
        if mem.metadata.get("skill_id").and_then(|v| v.as_str()) == Some(skill_id) {
            return Some(mem.rid);
        }
    }
    None
}

/// Variant of execute_cmd that performs post-fetch filtering on Recall
/// results before returning to the client. Used by /v1/skills/search to
/// apply applies_to / skill_type filters that are post-fetch in v0.8.11
/// (v0.8.13 makes them prefilter via the where-clause work).
async fn execute_cmd_with_post_filter(
    engine: Arc<yantrikdb::YantrikDB>,
    cmd: Command,
    control: Arc<parking_lot::Mutex<crate::control::ControlDb>>,
    inflight: &std::sync::atomic::AtomicU32,
    applies_to_filter: Option<String>,
    skill_type_filter: Option<String>,
    final_top_k: usize,
) -> AppResult {
    use std::sync::atomic::Ordering;
    struct InflightGuard<'a>(&'a std::sync::atomic::AtomicU32);
    impl Drop for InflightGuard<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let current = inflight.fetch_add(1, Ordering::Relaxed);
    if current >= crate::server::MAX_INFLIGHT {
        inflight.fetch_sub(1, Ordering::Relaxed);
        return Err(app_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "server overloaded: {} inflight ops (max {}). Retry later.",
                current,
                crate::server::MAX_INFLIGHT,
            ),
        ));
    }
    let _g = InflightGuard(inflight);

    let inner = tokio::task::spawn_blocking(move || {
        let db = engine.as_ref();
        handler::execute_with_guard(db, cmd, Some(control.as_ref()))
    })
    .await
    .map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("join error: {e}"),
        )
    })?;
    let result = inner.map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    match result {
        crate::handler::CommandResult::RecallResults { results, total: _ } => {
            // Filter by applies_to / skill_type, then truncate to top_k.
            // RecallResults are Vec<Value> (already JSON-encoded).
            let filtered: Vec<Value> = results
                .into_iter()
                .filter(|r| {
                    let metadata = r.get("metadata");
                    if let Some(ref needed) = applies_to_filter {
                        let ok = metadata
                            .and_then(|m| m.get("applies_to"))
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().any(|x| x.as_str() == Some(needed.as_str())))
                            .unwrap_or(false);
                        if !ok {
                            return false;
                        }
                    }
                    if let Some(ref needed) = skill_type_filter {
                        let ok = metadata
                            .and_then(|m| m.get("skill_type"))
                            .and_then(|v| v.as_str())
                            == Some(needed.as_str());
                        if !ok {
                            return false;
                        }
                    }
                    true
                })
                .take(final_top_k)
                .collect();
            let total = filtered.len();
            Ok(Json(serde_json::json!({
                "results": filtered,
                "total":   total,
            })))
        }
        other => Err(app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("unexpected command result: {:?}", other),
        )),
    }
}

// ── Issue #39 Phase 1 read endpoints ────────────────────────────────
//
// These routes use the RFC 014-B `Principal` substrate via the auth
// middleware at [`crate::auth::middleware::require_authenticated_principal`].
// They emit the structured error envelope from [`crate::api::errors`]
// directly (no `app_error` shim) and authorize via the guards in
// [`crate::api::access`].

/// Map an RFC 014-B [`Scope`] to the dashboard's `memory:*` / `admin` /
/// `tenant:manage` permission strings used in
/// `effective_scope.permissions`.
fn scope_to_permission(scope: crate::auth::Scope) -> &'static str {
    use crate::auth::Scope;
    match scope {
        Scope::Read => "memory:read",
        Scope::Write => "memory:write",
        Scope::Recall => "memory:recall",
        Scope::Forget => "memory:forget",
        Scope::Admin => "admin",
        Scope::TenantManagement => "tenant:manage",
    }
}

/// Pure payload builder for [`identity_scope`]. Extracted so the handler's
/// shape is unit-testable without spinning an `AppState` + axum runtime.
fn build_identity_scope_payload(
    principal: &crate::auth::Principal,
    visible_namespaces: &[String],
) -> Value {
    use crate::auth::Scope;

    let is_admin = principal.has_scope(Scope::Admin);

    let permissions: Vec<&'static str> = principal.scopes.iter().map(scope_to_permission).collect();

    // Phase 1 namespace_inventory: one entry per visible namespace.
    // `count` is null in this slice — populating it for cluster-admin
    // would mean opening every engine, which is too expensive for a
    // sync handler call. Phase 2 may surface counts via a cached path.
    let namespace_inventory: Vec<Value> = visible_namespaces
        .iter()
        .map(|ns| {
            json!({
                "namespace": ns,
                "count": Value::Null,
                "mapped": false,
                "mapped_scope": Value::Null,
                "mapped_to": Value::Null,
                "mapping_type": Value::Null,
                "mapping_source": Value::Null,
                "derived_by_config": false,
            })
        })
        .collect();

    json!({
        "schema_version": 1,
        "principal": {
            "kind": "token",
            "id": principal.id,
            "is_admin": is_admin,
        },
        "effective_scope": {
            "namespaces": visible_namespaces,
            "owners": [],
            "permissions": permissions,
            "admin": is_admin,
        },
        "identity_scope": {
            "identities": [],
            "actors": [],
            "spaces": [],
            "conversations": [],
        },
        "namespace_inventory": namespace_inventory,
        "summary": {
            "identities": 0,
            "actors": 0,
            "spaces": 0,
            "conversations": 0,
            "unmapped_namespaces": visible_namespaces.len(),
        },
    })
}

/// `GET /v1/identity-scope` — what does this token see?
///
/// Returns the nested envelope wysie's dashboard reads. Issue #39
/// Phase 1 populates the engine-derivable portions; plugin-side
/// concepts (identities, actors, spaces, conversations, namespace
/// mapping) emit empty arrays / default flags. The dashboard already
/// degrades gracefully on those.
///
/// Auth: requires `Scope::Read`. Cluster-admin principals enumerate
/// all databases as their visible namespaces; tenant-pinned principals
/// see exactly their `tenant_id`.
async fn identity_scope(
    State(state): State<Arc<AppState>>,
    axum::Extension(principal): axum::Extension<crate::auth::Principal>,
) -> AppResult {
    use crate::api::access;
    use crate::api::errors::{api_error, ApiErrorCode};
    use crate::auth::Scope;

    access::require_scope(&principal, Scope::Read)?;

    let namespaces: Vec<String> = match &principal.tenant_id {
        Some(ns) => vec![ns.clone()],
        None => {
            let ctrl = state.control.lock();
            let dbs = ctrl.list_databases().map_err(|e| {
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ApiErrorCode::InternalError,
                    format!("control db read failed: {e}"),
                )
            })?;
            dbs.into_iter().map(|d| d.name).collect()
        }
    };

    Ok(Json(build_identity_scope_payload(&principal, &namespaces)))
}

// ── /v1/memories — list endpoint (issue #39 Phase 1 task 196) ───────

const MEMORIES_DEFAULT_LIMIT: usize = 50;
const MEMORIES_MAX_LIMIT: usize = 200;
const MEMORIES_SUPPORTED_SORTS: &[&str] = &["created_at", "importance", "last_access"];
const MEMORIES_PHASE_2_SORTS: &[&str] = &["updated_at", "access_count", "certainty"];

#[derive(serde::Deserialize, Debug, Default)]
struct MemoriesListParams {
    namespace: Option<String>,
    status: Option<String>,
    domain: Option<String>,
    source: Option<String>,
    memory_type: Option<String>,
    q: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
    sort: Option<String>,
    // v0.8.23: structural query primitive — pushed down to engine.list_records.
    // `kind` and `drive_id` ride indexed VIRTUAL generated columns over
    // metadata JSON (engine v0.7.24 schema v32). `since_rid` is a keyset
    // cursor (UUIDv7 = lexically chronological). `order` is asc (oldest
    // first, default) or desc (newest first).
    kind: Option<String>,
    drive_id: Option<String>,
    since_rid: Option<String>,
    order: Option<String>,
}

fn unix_to_iso(epoch_seconds: f64) -> Option<String> {
    if !epoch_seconds.is_finite() {
        return None;
    }
    let secs = epoch_seconds.trunc() as i64;
    let nanos = ((epoch_seconds.fract().abs()) * 1_000_000_000.0).round() as u32;
    let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(secs, nanos.min(999_999_999))?;
    Some(dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
}

fn memory_to_dashboard_row(m: &yantrikdb::Memory) -> Value {
    json!({
        "rid": m.rid,
        "type": m.memory_type,
        "text": m.text,
        "created_at": m.created_at,
        "created_at_iso": unix_to_iso(m.created_at),
        "updated_at": Value::Null,
        "updated_at_iso": Value::Null,
        "importance": m.importance,
        "half_life": m.half_life,
        "last_access": m.last_access,
        "access_count": m.access_count,
        "valence": m.valence,
        "consolidated_into": m.consolidated_into,
        "consolidation_status": m.consolidation_status,
        "storage_tier": m.storage_tier,
        "metadata_json": m.metadata,
        "namespace": m.namespace,
        "certainty": m.certainty,
        "domain": m.domain,
        "source": m.source,
        "emotional_state": m.emotional_state,
        "session_id": m.session_id,
        "due_at": m.due_at,
        "temporal_kind": m.temporal_kind,
        "tombstone_reason": Value::Null,
        "embedding_model": Value::Null,
        "embedding_bytes": Value::Null,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MemoriesListResolved {
    /// Tenant database name — used by `get_database` to route the request
    /// to the correct engine. Comes from `principal.tenant_id` for
    /// per-tenant tokens; cluster-wide tokens must supply `?namespace`
    /// which is then both the DB selector AND (if pinned) the only
    /// possible tag filter.
    db_namespace: String,
    /// Optional row-level tag filter. Set ONLY when the client explicitly
    /// provided `?namespace`. When None, list every row in `db_namespace`
    /// regardless of tag. Per yantrikdb-core decision (swarm 8a97464e,
    /// 2026-06-09): `namespace` is a row tag, not a tenant scope.
    tag_filter: Option<String>,
    limit: usize,
    offset: usize,
    sort_by: String,
    domain: Option<String>,
    memory_type: Option<String>,
    /// v0.8.23 structural query primitive params — pushed down to
    /// engine.list_records (yantrikdb-core v0.7.24).
    kind: Option<String>,
    drive_id: Option<String>,
    since_rid: Option<String>,
    /// "asc" (oldest first, default) | "desc" (newest first). Validated
    /// here so callers see a 400 instead of a silent engine fallback.
    order: String,
}

fn validate_memories_params(
    principal: &crate::auth::Principal,
    params: &MemoriesListParams,
) -> Result<MemoriesListResolved, AppError> {
    use crate::api::access;
    use crate::api::errors::{api_error, ApiErrorCode};

    if let Some(s) = &params.status {
        if s != "active" {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                ApiErrorCode::InvalidQueryParameter,
                format!(
                    "status filter `{s}` is not implemented in Phase 1; only `active` is supported"
                ),
            ));
        }
    }
    if params.q.is_some() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            ApiErrorCode::InvalidQueryParameter,
            "text search filter `q` is not implemented on /v1/memories in Phase 1; use /v1/recall for semantic search",
        ));
    }
    if params.source.is_some() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            ApiErrorCode::InvalidQueryParameter,
            "source filter is not implemented on /v1/memories in Phase 1",
        ));
    }

    let sort_by = params.sort.as_deref().unwrap_or("created_at");
    if MEMORIES_PHASE_2_SORTS.contains(&sort_by) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            ApiErrorCode::InvalidQueryParameter,
            format!(
                "sort=`{sort_by}` is not implemented in Phase 1; engine v0.7.x supports {MEMORIES_SUPPORTED_SORTS:?}"
            ),
        ));
    }
    if !MEMORIES_SUPPORTED_SORTS.contains(&sort_by) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            ApiErrorCode::InvalidQueryParameter,
            format!(
                "sort=`{sort_by}` is not a recognized value; allowed: {MEMORIES_SUPPORTED_SORTS:?}"
            ),
        ));
    }

    let limit = params.limit.unwrap_or(MEMORIES_DEFAULT_LIMIT);
    if limit == 0 || limit > MEMORIES_MAX_LIMIT {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            ApiErrorCode::InvalidQueryParameter,
            format!("limit must be 1..={MEMORIES_MAX_LIMIT} (got {limit})"),
        ));
    }
    let offset = params.offset.unwrap_or(0);

    // Resolve db_namespace (= tenant routing) and tag_filter (= optional
    // row filter) as separate concerns. Per yantrikdb-core decision
    // (swarm 8a97464e, 2026-06-09): `namespace` is a row-level tag, not
    // a tenant scope. Per yantrikdb-agi report (swarm 77ffa517,
    // 2026-06-10): master tokens broke under v0.8.21 because the
    // (None, Some(q)) branch routed `?namespace` to `get_database()`,
    // 404'ing on every tag-as-namespace query.
    //
    // The fix: `?namespace` is ALWAYS a tag filter, never a DB selector.
    // Master tokens always route to "default" DB (matching
    // resolve_engine's hardcoded behavior at line 274-296).
    let (db_namespace, tag_filter) = match (&principal.tenant_id, params.namespace.as_deref()) {
        // Per-tenant token: db is fixed by the token. `?namespace` (if any)
        // is a tag filter; it does NOT need to match the tenant.
        (Some(tenant), tag) => (tenant.clone(), tag.map(|s| s.to_string())),
        // Master/cluster-wide token: route to "default" DB (matching
        // resolve_engine). `?namespace` is a tag filter only, never a DB
        // selector.
        (None, tag) => ("default".to_string(), tag.map(|s| s.to_string())),
    };
    let _ = access::resolve_namespace; // intentionally bypassed — see above

    // v0.8.23: validate `?order` here so callers see 400 BAD_REQUEST
    // instead of an engine error. asc (default) | desc.
    let order = params.order.as_deref().unwrap_or("asc");
    if order != "asc" && order != "desc" {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            ApiErrorCode::InvalidQueryParameter,
            format!("order=`{order}` is not recognized; allowed: asc | desc"),
        ));
    }

    Ok(MemoriesListResolved {
        db_namespace,
        tag_filter,
        limit,
        offset,
        sort_by: sort_by.to_string(),
        domain: params.domain.clone(),
        memory_type: params.memory_type.clone(),
        kind: params.kind.clone(),
        drive_id: params.drive_id.clone(),
        since_rid: params.since_rid.clone(),
        order: order.to_string(),
    })
}

async fn memories_list(
    State(state): State<Arc<AppState>>,
    axum::Extension(principal): axum::Extension<crate::auth::Principal>,
    Query(params): Query<MemoriesListParams>,
) -> AppResult {
    use crate::api::errors::{api_error, ApiErrorCode};
    use crate::auth::Scope;

    crate::api::access::require_scope(&principal, Scope::Read)?;
    let resolved = validate_memories_params(&principal, &params)?;

    let db_record = {
        let ctrl = state.control.lock();
        ctrl.get_database(&resolved.db_namespace)
            .map_err(|e| {
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ApiErrorCode::InternalError,
                    format!("control db lookup failed: {e}"),
                )
            })?
            .ok_or_else(|| {
                api_error(
                    StatusCode::FORBIDDEN,
                    ApiErrorCode::NamespaceNotFound,
                    format!("namespace not found: {}", resolved.db_namespace),
                )
            })?
    };

    let engine = state.pool.get_engine(&db_record).map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiErrorCode::InternalError,
            format!("engine load failed: {e}"),
        )
    })?;

    let domain = resolved.domain.clone();
    let memory_type = resolved.memory_type.clone();
    let tag_filter = resolved.tag_filter.clone();
    let kind = resolved.kind.clone();
    let drive_id = resolved.drive_id.clone();
    let since_rid = resolved.since_rid.clone();
    let order = resolved.order.clone();
    let limit = resolved.limit;
    let offset = resolved.offset;
    let engine_clone = engine.clone();

    // v0.8.23: when ANY new structural-query param is set (`?kind`,
    // `?drive_id`, `?since_rid`, `?order=desc`), route through engine's
    // `list_records` (yantrikdb-core v0.7.24) which pushes ALL filters
    // down to one SQL plan with the new indexed VIRTUAL columns + keyset
    // cursor on rid. Response: `{records, next_cursor, limit}` with
    // metadata parsed as JSON (mirrors /v1/recall's item shape so
    // clients like yantrikdb-agi's `query_typed()` deserialize through
    // their existing RecallHit type).
    //
    // Otherwise (no new params, asc order), keep the legacy
    // `list_memories` path with `{items, total, limit, offset}` for
    // backwards compatibility. Both are valid; the dashboard hasn't
    // been migrated yet.
    let uses_list_records =
        kind.is_some() || drive_id.is_some() || since_rid.is_some() || order == "desc";

    if uses_list_records {
        let res = tokio::task::spawn_blocking(move || {
            engine_clone.list_records(
                tag_filter.as_deref(),
                kind.as_deref(),
                drive_id.as_deref(),
                memory_type.as_deref(),
                domain.as_deref(),
                since_rid.as_deref(),
                limit,
                &order,
            )
        })
        .await
        .map_err(|e| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                ApiErrorCode::InternalError,
                format!("blocking thread join failed: {e}"),
            )
        })?;
        let (records, next_cursor) = res.map_err(|e| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                ApiErrorCode::InternalError,
                format!("list_records failed: {e}"),
            )
        })?;
        let items: Vec<Value> = records.iter().map(memory_to_record_item).collect();
        return Ok(Json(json!({
            "records": items,
            "next_cursor": next_cursor,
            "limit": resolved.limit,
        })));
    }

    let sort_by = resolved.sort_by.clone();
    let res = tokio::task::spawn_blocking(move || {
        engine_clone.list_memories(
            limit,
            offset,
            domain.as_deref(),
            memory_type.as_deref(),
            tag_filter.as_deref(),
            &sort_by,
        )
    })
    .await
    .map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiErrorCode::InternalError,
            format!("blocking thread join failed: {e}"),
        )
    })?;
    let (memories, total) = res.map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiErrorCode::InternalError,
            format!("list_memories failed: {e}"),
        )
    })?;

    let items: Vec<Value> = memories.iter().map(memory_to_dashboard_row).collect();
    Ok(Json(json!({
        "total": total,
        "limit": resolved.limit,
        "offset": resolved.offset,
        "items": items,
    })))
}

/// v0.8.23: record item shape for the `list_records`-backed
/// `/v1/memories` response. Mirrors `/v1/recall`'s item structure
/// exactly — metadata is a PARSED JSON object (not stringified). Per
/// yantrikdb-agi's request (swarm c1d810df, 2026-06-10): the goal is
/// that their existing `RecallHit` deserializer drops in unchanged.
fn memory_to_record_item(m: &yantrikdb::Memory) -> Value {
    // The engine returns `metadata` as a serde_json::Value, but some
    // tenants have legacy rows where the value is a Value::String
    // containing a JSON-encoded blob (the stringified-metadata pattern
    // yantrikdb-agi flagged in swarm c1d810df). Normalize both shapes
    // to a parsed object so the client sees `metadata: {...}` regardless
    // of how the row was originally written.
    let metadata = match &m.metadata {
        Value::String(s) if !s.trim().is_empty() => {
            serde_json::from_str::<Value>(s).unwrap_or_else(|_| m.metadata.clone())
        }
        other => other.clone(),
    };
    json!({
        "rid": m.rid,
        "text": m.text,
        "memory_type": m.memory_type,
        "importance": m.importance,
        "metadata": metadata,
        "namespace": m.namespace,
        "created_at": m.created_at,
        "created_at_iso": unix_to_iso(m.created_at),
        "certainty": m.certainty,
        "domain": m.domain,
        "source": m.source,
    })
}

// ── /v1/memory/{rid} — point read (issue #39 Phase 1 task 197) ──────

#[derive(serde::Deserialize, Debug, Default)]
struct MemoryGetParams {
    namespace: Option<String>,
    /// Read-your-writes hint. If supplied, the server checks the local
    /// node's applied seq against this value before reading. If the
    /// local replica is behind, returns 412 `replica_behind` so clients
    /// can retry with backoff or route to the leader. No wait primitive
    /// is available in engine v0.7.17, so this is reject-not-wait
    /// semantics — the dashboard contract documents 412 as the
    /// behavior, which matches.
    min_seq: Option<u64>,
}

/// Compare a caller's `min_seq` against the local node's applied seq.
/// Returns `Err(412 replica_behind)` if the replica hasn't caught up,
/// `Ok(())` otherwise.
///
/// Single-node mode (no `state.yrp`) always satisfies the check —
/// there is no replication lag to wait for. In YRP mode the check
/// compares against the node's durably-applied index.
fn check_min_seq(state: &AppState, min_seq: u64) -> Result<(), AppError> {
    use crate::api::errors::{api_error, ApiErrorCode};

    let Some(yrp) = state.yrp.as_ref() else {
        return Ok(());
    };
    let applied = yrp.status.borrow().applied;
    if applied < min_seq {
        return Err(api_error(
            StatusCode::PRECONDITION_FAILED,
            ApiErrorCode::ReplicaBehind,
            format!(
                "replica at seq {applied}, requested min_seq {min_seq}; retry with backoff or route to the leader"
            ),
        ));
    }
    Ok(())
}

/// `GET /v1/memory/{rid}` — point read with conditional includes + RYW.
///
/// Auth: requires `Scope::Read`. The `namespace` query param narrows
/// to the principal's scope; the returned memory's `namespace` column
/// must also match (otherwise 404, treating cross-namespace reads as
/// "not visible").
///
/// Phase-1 conditional arrays (`consolidation_sources`, `entities`,
/// `claims`) emit empty `[]`. Engine v0.7.x doesn't expose public
/// readers for these tables; populating them is a follow-up engine
/// extension. Dashboard tolerates empty arrays per its degradation
/// pattern.
async fn memory_get(
    State(state): State<Arc<AppState>>,
    axum::Extension(principal): axum::Extension<crate::auth::Principal>,
    AxumPath(rid): AxumPath<String>,
    Query(params): Query<MemoryGetParams>,
) -> AppResult {
    use crate::api::access;
    use crate::api::errors::{api_error, ApiErrorCode};
    use crate::auth::Scope;

    access::require_scope(&principal, Scope::Read)?;

    if let Some(min_seq) = params.min_seq {
        check_min_seq(&state, min_seq)?;
    }

    // Resolve db_namespace (tenant routing) directly from the principal.
    // Per yantrikdb-core decision (swarm 8a97464e, 2026-06-09) +
    // yantrikdb-agi report (swarm 77ffa517, 2026-06-10):
    // - per-tenant token routes to `principal.tenant_id`
    // - master/cluster-wide token routes to `"default"` (matching
    //   resolve_engine's hardcoded behavior at line 274-296)
    // - `?namespace` is irrelevant for point-read (rid uniquely
    //   identifies the row; namespace tag is row metadata, not scope)
    //
    // Prior code used access::resolve_namespace which 403'd master
    // tokens passing `?namespace=tag` and 404'd whenever the tag was
    // mistakenly routed as a DB selector. Both paths broke algo's
    // master-token workflow on CT 133.
    let db_namespace = principal
        .tenant_id
        .clone()
        .unwrap_or_else(|| "default".to_string());

    let db_record = {
        let ctrl = state.control.lock();
        ctrl.get_database(&db_namespace)
            .map_err(|e| {
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ApiErrorCode::InternalError,
                    format!("control db lookup failed: {e}"),
                )
            })?
            .ok_or_else(|| {
                api_error(
                    StatusCode::FORBIDDEN,
                    ApiErrorCode::NamespaceNotFound,
                    format!("namespace not found: {db_namespace}"),
                )
            })?
    };

    let engine = state.pool.get_engine(&db_record).map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiErrorCode::InternalError,
            format!("engine load failed: {e}"),
        )
    })?;

    let rid_for_engine = rid.clone();
    let engine_clone = engine.clone();
    let res = tokio::task::spawn_blocking(move || engine_clone.get(&rid_for_engine))
        .await
        .map_err(|e| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                ApiErrorCode::InternalError,
                format!("blocking thread join failed: {e}"),
            )
        })?;
    let memory = res.map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiErrorCode::InternalError,
            format!("engine get failed: {e}"),
        )
    })?;

    let Some(memory) = memory else {
        return Err(api_error(
            StatusCode::NOT_FOUND,
            ApiErrorCode::MemoryNotFound,
            format!("memory not found: {rid}"),
        ));
    };

    // No per-row namespace check. The database is the isolation
    // boundary (resolved via `get_database(effective_namespace)` above,
    // which is gated by the caller's token); `namespace` is an OPTIONAL
    // row-level TAG within that database, not a tenant scope. Any row
    // located in the caller's database by rid is the caller's row,
    // regardless of its namespace tag. Per yantrikdb-core decision
    // (swarm 8a97464e, 2026-06-09): row-tag is the canonical model.
    //
    // Prior code 404'd on `memory.namespace != effective_namespace`, which
    // hid the 200k+ rows that algo + lane-b store with custom tags like
    // `skill_substrate`, `comm_substrate`, `growth_lab_b_*`. That guard is
    // removed entirely; the `effective_namespace` variable is now only
    // used above for tenant DB routing.

    let mut row = memory_to_dashboard_row(&memory);
    // Phase-1 conditional includes: empty arrays. Engine extensions
    // populate these in a follow-up; dashboard reads empty as "no data
    // for this rid", which is structurally honest while empty.
    if let Some(obj) = row.as_object_mut() {
        obj.insert("consolidation_sources".into(), json!([]));
        obj.insert("entities".into(), json!([]));
        obj.insert("claims".into(), json!([]));
    }

    Ok(Json(row))
}

/// Build the Axum router.
pub fn router(state: Arc<AppState>) -> Router {
    let body_limit = state.admission.cfg.max_request_body_bytes;
    // Issue #39 Phase 1: read endpoints that use the RFC 014-B
    // `Principal` substrate. The auth middleware runs on these routes
    // only — legacy routes still authenticate inline via `resolve_engine`.
    let principal_auth_router: Router = Router::new()
        .route("/v1/identity-scope", get(identity_scope))
        .route("/v1/memories", get(memories_list))
        .route("/v1/memory/{rid}", get(memory_get))
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            crate::auth::require_authenticated_principal,
        ))
        .with_state(state.clone());

    let mut app = Router::new()
        .route("/v1/health", get(health))
        .route("/v1/health/deep", get(health_deep))
        .route("/v1/remember", post(remember))
        .route("/v1/remember/batch", post(remember_batch))
        .route("/v1/recall", post(recall))
        .route("/v1/forget", post(forget))
        // RFC 027 / v0.8.25 (pillar 2: lifecycle) — session boot + capture.
        .route("/v1/session/digest", get(session_digest))
        .route("/v1/session/end", post(session_end_capture))
        // RFC 027 / v0.8.27 — structural current-value read + known-unknowns.
        .route("/v1/current", get(current_value))
        .route("/v1/insights/gaps", get(insights_gaps))
        // RFC 022 §1 (v0.8.11): first-class skill primitives. Thin wrappers
        // over /v1/remember + /v1/recall + scan-and-filter (v0.8.11) or
        // /v1/lookup (v0.8.12+). Schema-validated, no semantic ontology.
        .route("/v1/skills/define", post(skill_define))
        .route("/v1/skills/search", post(skill_search))
        .route("/v1/skills/{skill_id}", get(skill_get))
        .route("/v1/skills/{skill_id}/outcome", post(skill_record_outcome))
        .route("/v1/skills/{skill_id}/forget", post(skill_forget))
        .route("/v1/relate", post(relate))
        .route("/v1/claim", post(ingest_claim))
        .route("/v1/claims", get(get_claims))
        .route("/v1/alias", post(add_alias))
        .route("/v1/think", post(think))
        .route("/v1/conflicts", get(conflicts))
        .route("/v1/conflicts/{id}/resolve", post(resolve_conflict))
        .route("/v1/sessions", post(session_start))
        .route("/v1/sessions/{id}", delete(session_end))
        .route("/v1/personality", get(personality))
        .route("/v1/stats", get(stats))
        .route("/v1/databases", post(create_database))
        .route("/v1/databases", get(list_databases))
        .route("/v1/cluster", get(cluster_status))
        .route("/v1/cluster/promote", post(cluster_promote))
        // RFC 028: YRP peer wire (bincode envelope; cluster-secret gated)
        .route("/v1/yrp/msg", post(yrp_msg))
        // RFC 028 Phase C: engine backfill for beyond-GC stragglers
        .route("/v1/yrp/backfill", post(yrp_backfill))
        // Admin studio: aggregated cluster topology + the embedded console
        .route("/v1/cluster/topology", get(yrp_topology))
        .route("/admin", get(admin_studio))
        .route("/v1/admin/control-snapshot", get(control_snapshot))
        // RFC 029: replicated control-plane admin (master-token gated).
        // Databases + tokens minted here commit through YRP and apply on
        // every node, so identity survives failover.
        .route("/v1/admin/databases", post(admin_create_database))
        .route("/v1/admin/tokens", post(admin_create_token))
        .route("/v1/admin/tokens/revoke", post(admin_revoke_token))
        .route("/v1/admin/snapshot", post(admin_snapshot))
        // RFC 027 / v0.8.24 — autonomous-maintenance operator surface.
        .route("/v1/admin/maintenance/run", post(admin_maintenance_run))
        .route(
            "/v1/admin/maintenance/status",
            get(admin_maintenance_status),
        )
        // RFC 008 Warrant Flow substrate
        .route("/v1/claim_with_lineage", post(ingest_claim_with_lineage))
        .route("/v1/mobility", get(get_mobility))
        .route("/v1/contest", get(get_contest))
        .route("/v1/move_events", post(record_move_event))
        .route("/v1/flagged_propositions", get(list_flagged))
        // RFC 010 PR-5 Jepsen / debug surface (cluster master token required)
        .route("/v1/debug/history/{tenant_id}", get(debug_history))
        .route("/v1/debug/fault/inject", post(debug_fault_inject))
        .route("/v1/debug/fault", get(debug_fault_list))
        .route("/v1/debug/fault/clear", post(debug_fault_clear))
        .route("/v1/debug/fault/{fault_id}", delete(debug_fault_remove))
        // Phase 1 polish: RFC 019 jobs admin surface (master-token gated)
        .route("/v1/jobs", get(jobs_list))
        .route("/v1/jobs/{job_id}", get(jobs_get))
        .route("/v1/jobs/{job_id}", delete(jobs_cancel))
        // Phase 1 polish: RFC 017-B migration visibility (master-token gated)
        .route("/v1/admin/migrations", get(admin_migrations_list))
        .route("/metrics", get(metrics));
    // Apply layer + state, then merge the openraft sub-router (which
    // already binds its own State<Arc<Raft>>).
    let mut app = app
        // RFC 009 §4 Layer 3: hard request body size cap. Bodies above
        // `admission.max_request_body_bytes` get HTTP 413 from this layer
        // before any handler runs. Defends against memory-blow attacks
        // and misconfigured clients.
        .layer(tower_http::limit::RequestBodyLimitLayer::new(body_limit))
        .with_state(state)
        .merge(principal_auth_router);
    app
}

#[cfg(test)]
mod skill_validation_tests {
    use super::{validate_applies_to_entry, validate_skill_id};

    // ── validate_skill_id ──────────────────────────────────

    #[test]
    fn skill_id_valid_minimal_form() {
        // Minimum: starts with lowercase, has one dot, length >= 4.
        assert!(validate_skill_id("a.b").is_err()); // length < 4
        assert!(validate_skill_id("ab.c").is_ok()); // length 4 = boundary
        assert!(validate_skill_id("skill.foo.v1").is_ok());
        assert!(validate_skill_id("skill.invoice.validation.v3").is_ok());
    }

    #[test]
    fn skill_id_rejects_uppercase() {
        assert!(validate_skill_id("Skill.foo").is_err());
        assert!(validate_skill_id("skill.Foo").is_err());
        assert!(validate_skill_id("SKILL.FOO").is_err());
    }

    #[test]
    fn skill_id_rejects_no_dot() {
        // Must contain at least one '.' (dotted form requirement).
        let err = validate_skill_id("skillfoo").unwrap_err();
        assert!(err.contains("at least one '.'"));
    }

    #[test]
    fn skill_id_rejects_consecutive_dots() {
        assert!(validate_skill_id("skill..foo").is_err());
    }

    #[test]
    fn skill_id_rejects_trailing_dot() {
        assert!(validate_skill_id("skill.foo.").is_err());
    }

    #[test]
    fn skill_id_rejects_starts_with_digit_or_underscore_or_dot() {
        assert!(validate_skill_id("1skill.foo").is_err());
        assert!(validate_skill_id("_skill.foo").is_err());
        assert!(validate_skill_id(".skill.foo").is_err());
    }

    #[test]
    fn skill_id_rejects_invalid_chars() {
        assert!(validate_skill_id("skill-foo.v1").is_err()); // hyphen
        assert!(validate_skill_id("skill foo.v1").is_err()); // space
        assert!(validate_skill_id("skill/foo.v1").is_err()); // slash
        assert!(validate_skill_id("skill@foo.v1").is_err()); // at-sign
    }

    #[test]
    fn skill_id_length_bounds() {
        // Lower bound: 4 chars.
        assert!(validate_skill_id("a.bc").is_ok());
        assert!(validate_skill_id("a.b").is_err());
        // Upper bound: 200 chars.
        let long_ok = format!("skill.{}", "a".repeat(193)); // 6 + 193 = 199
        assert!(validate_skill_id(&long_ok).is_ok());
        let long_err = format!("skill.{}", "a".repeat(195)); // 6 + 195 = 201
        assert!(validate_skill_id(&long_err).is_err());
    }

    #[test]
    fn skill_id_allows_underscores_and_digits_in_segments() {
        assert!(validate_skill_id("skill_42.foo_bar.v1_2").is_ok());
        assert!(validate_skill_id("a1.b2").is_ok());
    }

    // ── validate_applies_to_entry ──────────────────────────

    #[test]
    fn applies_to_entry_valid() {
        assert!(validate_applies_to_entry("invoice").is_ok());
        assert!(validate_applies_to_entry("meta_agent").is_ok());
        assert!(validate_applies_to_entry("a").is_ok());
        assert!(validate_applies_to_entry("a1").is_ok());
        assert!(validate_applies_to_entry("invoice_validation_2026").is_ok());
    }

    #[test]
    fn applies_to_entry_rejects_hyphen() {
        // The whole point of this regex: catch the hyphen-vs-underscore
        // drift bug Brainstorm 2 named. `meta-agent` (hyphen) and
        // `meta_agent` (underscore) are both valid Rust strings, but only
        // the underscore form is accepted as an applies_to entry. This
        // forces consistency at write time.
        let err = validate_applies_to_entry("meta-agent").unwrap_err();
        assert!(err.contains("invalid character"));
    }

    #[test]
    fn applies_to_entry_rejects_uppercase() {
        assert!(validate_applies_to_entry("Invoice").is_err());
        assert!(validate_applies_to_entry("INVOICE").is_err());
    }

    #[test]
    fn applies_to_entry_rejects_dot_or_slash() {
        // Unlike skill_id, applies_to entries are flat (no dots).
        assert!(validate_applies_to_entry("invoice.validation").is_err());
        assert!(validate_applies_to_entry("invoice/validation").is_err());
    }

    #[test]
    fn applies_to_entry_rejects_empty() {
        assert!(validate_applies_to_entry("").is_err());
    }

    #[test]
    fn applies_to_entry_rejects_starts_with_digit_or_underscore() {
        assert!(validate_applies_to_entry("1invoice").is_err());
        assert!(validate_applies_to_entry("_invoice").is_err());
    }
}

/// PR 6.6 — HTTP error-mapping conformance tests.
///
/// Pin every `CommitError` variant's status code + body shape so client
/// SDKs can build retry / redirect / error-classification logic against
/// a stable contract.
#[cfg(test)]
mod commit_error_mapping_tests {
    use super::commit_error_to_app_error;
    use crate::commit::{CommitError, OpId, TenantId};
    use axum::http::StatusCode;

    #[test]
    fn not_leader_maps_to_307_with_leader_info_in_body() {
        let (status, body) = commit_error_to_app_error(CommitError::NotLeader {
            leader_id: Some(4),
            leader_addr: Some("https://192.168.4.140:7438".into()),
        });
        assert_eq!(status, StatusCode::TEMPORARY_REDIRECT);
        let v = body.0;
        assert_eq!(v["error"], "not_leader");
        assert_eq!(v["leader_id"], 4);
        assert_eq!(v["leader_addr"], "https://192.168.4.140:7438");
    }

    #[test]
    fn not_leader_with_unknown_leader_emits_nulls() {
        // Mid-election: openraft reports ForwardToLeader with no known
        // leader. Client SHOULD interpret this as a 503-shape signal
        // even though the status is 307.
        let (status, body) = commit_error_to_app_error(CommitError::NotLeader {
            leader_id: None,
            leader_addr: None,
        });
        assert_eq!(status, StatusCode::TEMPORARY_REDIRECT);
        assert!(body.0["leader_id"].is_null());
        assert!(body.0["leader_addr"].is_null());
    }

    #[test]
    fn op_id_collision_maps_to_409_with_existing_index() {
        let op = OpId::new_random();
        let (status, body) = commit_error_to_app_error(CommitError::OpIdCollision {
            op_id: op,
            tenant_id: TenantId::new(7),
            existing_index: 42,
        });
        assert_eq!(status, StatusCode::CONFLICT);
        let v = body.0;
        assert_eq!(v["error"], "op_id_collision");
        assert_eq!(v["op_id"], op.to_string());
        assert_eq!(v["tenant_id"], 7);
        assert_eq!(v["existing_index"], 42);
    }

    #[test]
    fn unexpected_log_index_maps_to_409_with_expected_actual() {
        let (status, body) = commit_error_to_app_error(CommitError::UnexpectedLogIndex {
            tenant_id: TenantId::new(1),
            expected: 5,
            actual: 7,
        });
        assert_eq!(status, StatusCode::CONFLICT);
        let v = body.0;
        assert_eq!(v["error"], "unexpected_log_index");
        assert_eq!(v["expected"], 5);
        assert_eq!(v["actual"], 7);
    }

    #[test]
    fn not_yet_implemented_maps_to_501_with_planned_rfc() {
        let (status, body) = commit_error_to_app_error(CommitError::NotYetImplemented {
            variant: "PurgeMemory",
            planned_rfc: "011",
        });
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        let v = body.0;
        assert_eq!(v["error"], "not_implemented");
        assert_eq!(v["variant"], "PurgeMemory");
        assert_eq!(v["planned_rfc"], "011");
    }

    #[test]
    fn storage_failure_maps_to_503_with_retry_after() {
        let (status, body) = commit_error_to_app_error(CommitError::StorageFailure {
            message: "disk full".into(),
        });
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let v = body.0;
        assert_eq!(v["error"], "storage_failure");
        assert_eq!(v["detail"], "disk full");
        assert_eq!(v["retry_after_ms"], 1000);
    }

    #[test]
    fn shutdown_maps_to_503_with_longer_retry_after() {
        // Shutdown means "this node is going down — try a peer." Longer
        // retry_after_ms hints "don't hammer this address."
        let (status, body) = commit_error_to_app_error(CommitError::Shutdown);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body.0["error"], "shutting_down");
        assert_eq!(body.0["retry_after_ms"], 5000);
    }

    #[test]
    fn commit_timeout_maps_to_503_with_op_id_for_idempotent_retry() {
        // The load-bearing PR 6.6 invariant: timeout responses carry
        // the op_id so client retries are idempotent. Without this,
        // network-partition recovery duplicates writes.
        let op = OpId::new_random();
        let (status, body) = commit_error_to_app_error(CommitError::CommitTimeout { op_id: op });
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let v = body.0;
        assert_eq!(v["error"], "commit_timeout");
        assert_eq!(v["op_id"], op.to_string());
        assert_eq!(v["retry_after_ms"], 1000);
    }

    // ── PR 6.9 — replication-lag derivation tests ─────────────────
    //
    // The value computation lives inside `cluster_state_view`, which
    // holds an `AppState` reference and isn't easy to mock. The lag
    // arithmetic is the only piece that has correctness implications
    // for clients reading the field; the rest is field shuffling. Pin
    // it as a free function that mirrors the logic.

    fn replication_lag(
        last_log_index: Option<u64>,
        last_applied_index: Option<u64>,
    ) -> Option<u64> {
        match (last_log_index, last_applied_index) {
            (Some(log), Some(applied)) => Some(log.saturating_sub(applied)),
            (Some(log), None) => Some(log),
            (None, _) => None,
        }
    }

    #[test]
    fn pr_6_9_lag_is_zero_when_log_and_applied_are_equal() {
        // Healthy follower (or leader): everything received has been
        // applied. The number clients read should be exactly 0, not
        // null — the data was observed.
        assert_eq!(replication_lag(Some(18), Some(18)), Some(0));
    }

    #[test]
    fn pr_6_9_lag_reflects_unapplied_entries() {
        // 5 entries received but not yet applied. This is the value
        // a Grafana alert / health probe consumes to detect a stuck
        // apply path.
        assert_eq!(replication_lag(Some(100), Some(95)), Some(5));
    }

    #[test]
    fn pr_6_9_lag_clamps_at_zero_under_inversion() {
        // Race: the metric snapshot may briefly observe applied > log_index
        // (e.g. during membership change or snapshot install). Clients
        // would treat a giant negative-flipped-to-u64 as "lag of 18
        // exabytes" and page the operator. saturating_sub clamps at 0.
        assert_eq!(replication_lag(Some(10), Some(15)), Some(0));
    }

    #[test]
    fn pr_6_9_lag_is_log_index_when_nothing_applied_yet() {
        // Cold-start follower: log entries received but state machine
        // hasn't applied any yet. Lag = entire log.
        assert_eq!(replication_lag(Some(7), None), Some(7));
    }

    #[test]
    fn pr_6_9_lag_is_none_when_no_log_index_known() {
        // Pre-bootstrap or single-node: no openraft, no metrics.
        // Field reads as JSON null, distinguishable from an actual 0.
        assert_eq!(replication_lag(None, None), None);
        assert_eq!(replication_lag(None, Some(5)), None);
    }

    #[test]
    fn version_mismatch_maps_to_426_upgrade_required() {
        // 426 is the canonical HTTP status for "upgrade required" —
        // tells the client (or its operator) the cluster is rolling
        // through a wire-version transition and this peer is behind.
        let verr = crate::version::VersionError::WireMajorMismatch {
            node: crate::version::WireVersion::new(1, 0),
            event: crate::version::WireVersion::new(2, 0),
        };
        let (status, body) = commit_error_to_app_error(CommitError::Version(verr));
        assert_eq!(status, StatusCode::UPGRADE_REQUIRED);
        assert_eq!(body.0["error"], "wire_version_mismatch");
    }

    #[test]
    fn every_variant_produces_a_response() {
        // Belt-and-suspenders: a future maintainer who adds a new
        // CommitError variant must update commit_error_to_app_error.
        // The match is exhaustive at compile time, but we also assert
        // here that every existing variant produces a valid status code
        // (not zero, not panic).
        let cases = vec![
            CommitError::NotLeader {
                leader_id: None,
                leader_addr: None,
            },
            CommitError::OpIdCollision {
                op_id: OpId::new_random(),
                tenant_id: TenantId::new(1),
                existing_index: 0,
            },
            CommitError::UnexpectedLogIndex {
                tenant_id: TenantId::new(1),
                expected: 1,
                actual: 2,
            },
            CommitError::NotYetImplemented {
                variant: "X",
                planned_rfc: "Y",
            },
            CommitError::StorageFailure {
                message: "x".into(),
            },
            CommitError::Shutdown,
            CommitError::CommitTimeout {
                op_id: OpId::new_random(),
            },
        ];
        for err in cases {
            let label = err.metric_label();
            let (status, body) = commit_error_to_app_error(err);
            // 307 (NotLeader redirect) is the only 3xx case; everything
            // else is 4xx or 5xx. No CommitError should map to 1xx/2xx.
            let s = status.as_u16();
            assert!(
                s == 307 || s >= 400,
                "{label} unexpected status {s} (want 307 or 4xx/5xx)"
            );
            assert!(
                body.0.get("error").is_some(),
                "{label} body must include `error` key"
            );
        }
    }
}

#[cfg(test)]
mod identity_scope_tests {
    use super::{build_identity_scope_payload, scope_to_permission};
    use crate::auth::{Principal, Scope, ScopeSet};

    fn pinned_tenant_principal() -> Principal {
        Principal::new("tok_abcd1234")
            .with_tenant("acme")
            .with_scopes(ScopeSet::from_iter([
                Scope::Read,
                Scope::Write,
                Scope::Recall,
                Scope::Forget,
            ]))
    }

    fn cluster_admin_principal() -> Principal {
        Principal::new("cluster-admin")
            .with_scopes(ScopeSet::all())
            .with_label("cluster-master")
    }

    #[test]
    fn scope_to_permission_pinned_strings() {
        // The dashboard branches on these strings — they're part of the
        // wire contract. Stability test.
        assert_eq!(scope_to_permission(Scope::Read), "memory:read");
        assert_eq!(scope_to_permission(Scope::Write), "memory:write");
        assert_eq!(scope_to_permission(Scope::Recall), "memory:recall");
        assert_eq!(scope_to_permission(Scope::Forget), "memory:forget");
        assert_eq!(scope_to_permission(Scope::Admin), "admin");
        assert_eq!(
            scope_to_permission(Scope::TenantManagement),
            "tenant:manage"
        );
    }

    #[test]
    fn payload_top_level_keys_match_dashboard_contract() {
        // Stability: the dashboard reads these specific top-level keys.
        let p = pinned_tenant_principal();
        let v = build_identity_scope_payload(&p, &["acme".into()]);
        for key in [
            "schema_version",
            "principal",
            "effective_scope",
            "identity_scope",
            "namespace_inventory",
            "summary",
        ] {
            assert!(v.get(key).is_some(), "missing key `{key}` in payload");
        }
    }

    #[test]
    fn pinned_principal_emits_single_namespace() {
        let p = pinned_tenant_principal();
        let v = build_identity_scope_payload(&p, &["acme".into()]);
        assert_eq!(
            v["effective_scope"]["namespaces"],
            serde_json::json!(["acme"])
        );
        assert_eq!(v["effective_scope"]["admin"], false);
        assert_eq!(v["principal"]["id"], "tok_abcd1234");
        assert_eq!(v["principal"]["is_admin"], false);
        assert_eq!(v["principal"]["kind"], "token");
    }

    #[test]
    fn pinned_principal_permissions_are_data_plane_set() {
        let p = pinned_tenant_principal();
        let v = build_identity_scope_payload(&p, &["acme".into()]);
        let perms = v["effective_scope"]["permissions"].as_array().unwrap();
        let strs: Vec<&str> = perms.iter().filter_map(|x| x.as_str()).collect();
        assert!(strs.contains(&"memory:read"));
        assert!(strs.contains(&"memory:write"));
        assert!(strs.contains(&"memory:recall"));
        assert!(strs.contains(&"memory:forget"));
        assert!(!strs.contains(&"admin"));
        assert!(!strs.contains(&"tenant:manage"));
    }

    #[test]
    fn cluster_admin_principal_marks_admin_true() {
        let p = cluster_admin_principal();
        let v = build_identity_scope_payload(&p, &["acme".into(), "default".into()]);
        assert_eq!(v["principal"]["is_admin"], true);
        assert_eq!(v["effective_scope"]["admin"], true);
        let strs: Vec<&str> = v["effective_scope"]["permissions"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|x| x.as_str())
            .collect();
        assert!(strs.contains(&"admin"));
        assert!(strs.contains(&"tenant:manage"));
    }

    #[test]
    fn cluster_admin_principal_enumerates_all_namespaces() {
        let p = cluster_admin_principal();
        let v = build_identity_scope_payload(&p, &["acme".into(), "default".into()]);
        assert_eq!(
            v["effective_scope"]["namespaces"],
            serde_json::json!(["acme", "default"])
        );
        // namespace_inventory should mirror.
        let inv = v["namespace_inventory"].as_array().unwrap();
        assert_eq!(inv.len(), 2);
        assert_eq!(inv[0]["namespace"], "acme");
        assert_eq!(inv[1]["namespace"], "default");
    }

    #[test]
    fn namespace_inventory_entry_has_full_dashboard_shape() {
        let p = pinned_tenant_principal();
        let v = build_identity_scope_payload(&p, &["acme".into()]);
        let entry = &v["namespace_inventory"][0];
        // Every key the dashboard reads must be present, even if null.
        for key in [
            "namespace",
            "count",
            "mapped",
            "mapped_scope",
            "mapped_to",
            "mapping_type",
            "mapping_source",
            "derived_by_config",
        ] {
            assert!(entry.get(key).is_some(), "missing inventory key `{key}`");
        }
        // Phase 1 defaults: no plugin mapping, no count.
        assert_eq!(entry["mapped"], false);
        assert_eq!(entry["count"], serde_json::Value::Null);
        assert_eq!(entry["derived_by_config"], false);
    }

    #[test]
    fn summary_unmapped_namespaces_matches_namespace_count() {
        let p = cluster_admin_principal();
        let v = build_identity_scope_payload(&p, &["a".into(), "b".into(), "c".into()]);
        assert_eq!(v["summary"]["unmapped_namespaces"], 3);
        assert_eq!(v["summary"]["identities"], 0);
        assert_eq!(v["summary"]["actors"], 0);
        assert_eq!(v["summary"]["spaces"], 0);
        assert_eq!(v["summary"]["conversations"], 0);
    }

    #[test]
    fn identity_scope_arrays_are_empty_in_phase_1() {
        // Phase 1: plugin-side concepts not surfaced. Dashboard reads
        // these as arrays; empty is the correct Phase-1 value.
        let p = pinned_tenant_principal();
        let v = build_identity_scope_payload(&p, &["acme".into()]);
        for key in ["identities", "actors", "spaces", "conversations"] {
            assert_eq!(
                v["identity_scope"][key],
                serde_json::json!([]),
                "identity_scope.{key} must be []"
            );
        }
    }
}

// ── Issue #39 Phase 1 — e2e tests against the production router ─────
//
// Per issue #34 discipline, integration tests must exercise the real
// `crate::http_gateway::router(state)` and the real auth middleware,
// not mock handlers. tests/http_integration.rs can't reach production
// code (bin-only crate), so e2e tests for the new Principal-based
// endpoints live in src/ as `#[cfg(test)] mod` blocks.
#[cfg(test)]
pub(crate) mod e2e_test_support {
    use std::sync::Arc;

    use parking_lot::Mutex;

    use crate::auth::ControlDbAuthProvider;
    use crate::control::ControlDb;
    use crate::server::AppState;

    /// Bundles the live state + raw token + paths so tests can hit the
    /// router with a real Bearer and tear down cleanly on drop.
    pub struct E2eFixture {
        pub state: Arc<AppState>,
        pub raw_token: String,
        pub tenant_namespace: String,
        // Held to keep the data_dir alive until tests finish.
        pub _tmp: tempfile::TempDir,
    }

    /// Build a real `AppState` against a tempdir. Wires:
    /// - control DB with one database row + one issued token
    /// - `ControlDbAuthProvider` over that control DB
    /// - real `TenantPool`, `WorkerRegistry`, `AdmissionState`,
    ///   `LocalSqliteCommitter`, `LocalSqliteJobQueue`, `FaultRegistry`
    /// - no cluster, no openraft, no control runtime — Phase 1 read
    ///   endpoints don't need them and stubs avoid spawning runtimes.
    pub fn build_fixture(tenant_namespace: &str) -> E2eFixture {
        build_fixture_impl(tenant_namespace, None)
    }

    /// Build a fixture with a cluster context configured — for testing the
    /// cluster-safety gates on the maintenance admin endpoints (RFC 027).
    /// `role` drives write-acceptance (Single→Standalone accepts; Voter→
    /// Follower rejects); `secret` becomes the cluster master token.
    pub fn build_fixture_with_cluster(
        tenant_namespace: &str,
        role: crate::config::NodeRole,
        secret: &str,
    ) -> E2eFixture {
        build_fixture_impl(tenant_namespace, Some((role, secret.to_string())))
    }

    fn build_fixture_impl(
        tenant_namespace: &str,
        cluster: Option<(crate::config::NodeRole, String)>,
    ) -> E2eFixture {
        let tmp = tempfile::tempdir().expect("tempdir");
        let data_dir = tmp.path().to_path_buf();

        let mut cfg = crate::config::ServerConfig::default();
        cfg.server.data_dir = data_dir.clone();
        if let Some((role, ref secret)) = cluster {
            cfg.cluster.node_id = 1;
            cfg.cluster.role = role;
            cfg.cluster.cluster_secret = Some(secret.clone());
        }

        // control DB + one tenant + one token
        let control_path = data_dir.join("control.db");
        let control = ControlDb::open(&control_path).expect("control db open");
        let raw_token = crate::auth::generate_token();
        let token_hash = crate::auth::hash_token(&raw_token);
        let db_id = control
            .create_database(tenant_namespace, tenant_namespace)
            .expect("create_database");
        control
            .create_token(&token_hash, db_id, "e2e-test")
            .expect("create_token");
        let control = Arc::new(Mutex::new(control));

        let pool = Arc::new(crate::tenant_pool::TenantPool::new(&cfg, None, None));
        let workers = crate::background::WorkerRegistry::new(
            &cfg.background,
            &cfg.maintenance,
            crate::background::WriteAcceptanceGate::standalone(),
        );
        let admission = crate::admission::AdmissionState::new(Default::default());
        let commit_log: Arc<dyn crate::commit::MutationCommitter> =
            Arc::new(crate::commit::LocalSqliteCommitter::open_in_memory().expect("commit log"));
        let jobs: Arc<dyn crate::jobs::JobQueue> =
            Arc::new(crate::jobs::LocalSqliteJobQueue::open_in_memory().expect("jobs queue"));
        let cluster_secret = cluster.as_ref().map(|(_, s)| s.clone());
        let auth_provider: Arc<dyn crate::auth::AuthProvider> = Arc::new(
            ControlDbAuthProvider::new(Arc::clone(&control), cluster_secret),
        );

        let cluster_ctx = cluster.map(|(role, _)| {
            let node_state = Arc::new(
                crate::cluster::NodeState::new(1, role, data_dir.join("raft.json"))
                    .expect("node state"),
            );
            let peers = Arc::new(crate::cluster::PeerRegistry::new(&cfg.cluster.peers));
            Arc::new(crate::cluster::ClusterContext::new(
                cfg.cluster.clone(),
                node_state,
                peers,
                Arc::clone(&pool),
                Some(Arc::clone(&control)),
            ))
        });

        let state = Arc::new(AppState {
            control,
            pool,
            workers,
            cluster: cluster_ctx,
            inflight: std::sync::atomic::AtomicU32::new(0),
            admission,
            control_runtime: None,
            commit_log,
            yrp: None,
            fault_registry: crate::debug::FaultRegistry::new(),
            jobs,
            data_dir,
            auth_provider,
        });

        E2eFixture {
            state,
            raw_token,
            tenant_namespace: tenant_namespace.to_string(),
            _tmp: tmp,
        }
    }

    /// Helper: GET against the production router and return (status, body bytes).
    pub async fn get(
        state: Arc<AppState>,
        path: &str,
        bearer: Option<&str>,
    ) -> (axum::http::StatusCode, axum::body::Bytes) {
        use axum::body::to_bytes;
        use axum::http::Request;
        use tower::ServiceExt;

        let app = super::router(state);
        let mut builder = Request::builder().uri(path).method("GET");
        if let Some(tok) = bearer {
            builder = builder.header("authorization", format!("Bearer {tok}"));
        }
        let req = builder.body(axum::body::Body::empty()).unwrap();
        let res = app.oneshot(req).await.expect("oneshot");
        let status = res.status();
        let bytes = to_bytes(res.into_body(), 1024 * 1024)
            .await
            .expect("body bytes");
        (status, bytes)
    }

    /// Helper: POST (empty JSON body) against the production router.
    pub async fn post(
        state: Arc<AppState>,
        path: &str,
        bearer: Option<&str>,
    ) -> (axum::http::StatusCode, axum::body::Bytes) {
        use axum::body::to_bytes;
        use axum::http::Request;
        use tower::ServiceExt;

        let app = super::router(state);
        let mut builder = Request::builder()
            .uri(path)
            .method("POST")
            .header("content-type", "application/json");
        if let Some(tok) = bearer {
            builder = builder.header("authorization", format!("Bearer {tok}"));
        }
        let req = builder.body(axum::body::Body::from("{}")).unwrap();
        let res = app.oneshot(req).await.expect("oneshot");
        let status = res.status();
        let bytes = to_bytes(res.into_body(), 1024 * 1024)
            .await
            .expect("body bytes");
        (status, bytes)
    }

    /// Helper: POST a specific JSON body against the production router.
    pub async fn post_body(
        state: Arc<AppState>,
        path: &str,
        bearer: Option<&str>,
        json_body: &str,
    ) -> (axum::http::StatusCode, axum::body::Bytes) {
        use axum::body::to_bytes;
        use axum::http::Request;
        use tower::ServiceExt;

        let app = super::router(state);
        let mut builder = Request::builder()
            .uri(path)
            .method("POST")
            .header("content-type", "application/json");
        if let Some(tok) = bearer {
            builder = builder.header("authorization", format!("Bearer {tok}"));
        }
        let req = builder
            .body(axum::body::Body::from(json_body.to_string()))
            .unwrap();
        let res = app.oneshot(req).await.expect("oneshot");
        let status = res.status();
        let bytes = to_bytes(res.into_body(), 1024 * 1024)
            .await
            .expect("body bytes");
        (status, bytes)
    }
}

#[cfg(test)]
mod identity_scope_e2e {
    use super::e2e_test_support::{build_fixture, get};

    #[tokio::test]
    async fn returns_401_when_no_bearer_header() {
        let fx = build_fixture("acme");
        let (status, body) = get(fx.state.clone(), "/v1/identity-scope", None).await;
        assert_eq!(status, 401);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Structured envelope, not Option-A string.
        assert_eq!(v["error"]["code"], "unauthenticated");
        assert!(v["error"]["message"].is_string());
    }

    #[tokio::test]
    async fn returns_401_when_token_is_unknown() {
        let fx = build_fixture("acme");
        let (status, body) = get(
            fx.state.clone(),
            "/v1/identity-scope",
            Some("ydb_definitely_not_a_real_token"),
        )
        .await;
        assert_eq!(status, 401);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "unauthenticated");
    }

    #[tokio::test]
    async fn returns_200_with_dashboard_shape_for_valid_tenant_token() {
        let fx = build_fixture("acme");
        let (status, body) = get(fx.state.clone(), "/v1/identity-scope", Some(&fx.raw_token)).await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Top-level keys match the dashboard contract.
        for key in [
            "schema_version",
            "principal",
            "effective_scope",
            "identity_scope",
            "namespace_inventory",
            "summary",
        ] {
            assert!(v.get(key).is_some(), "missing top-level key `{key}`");
        }

        // Principal: tenant-pinned, not admin.
        assert_eq!(v["principal"]["kind"], "token");
        assert_eq!(v["principal"]["is_admin"], false);
        let id = v["principal"]["id"].as_str().unwrap();
        assert!(id.starts_with("tok_"), "principal.id should be hashed-form");
        assert!(
            !id.contains(&fx.raw_token),
            "raw token must not appear in principal.id"
        );

        // Tenant principal sees exactly its own namespace.
        assert_eq!(
            v["effective_scope"]["namespaces"],
            serde_json::json!([fx.tenant_namespace.as_str()])
        );
        assert_eq!(v["effective_scope"]["admin"], false);

        // Permissions: full data-plane set, no admin.
        let perms: Vec<&str> = v["effective_scope"]["permissions"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|x| x.as_str())
            .collect();
        for needed in [
            "memory:read",
            "memory:write",
            "memory:recall",
            "memory:forget",
        ] {
            assert!(
                perms.contains(&needed),
                "missing permission `{needed}` (got {perms:?})"
            );
        }
        assert!(!perms.contains(&"admin"));

        // namespace_inventory has the single visible entry with full shape.
        let inv = v["namespace_inventory"].as_array().unwrap();
        assert_eq!(inv.len(), 1);
        assert_eq!(inv[0]["namespace"], fx.tenant_namespace);
        assert_eq!(inv[0]["mapped"], false);
        assert_eq!(inv[0]["count"], serde_json::Value::Null);

        // Plugin-side concept arrays empty in Phase 1.
        for key in ["identities", "actors", "spaces", "conversations"] {
            assert_eq!(
                v["identity_scope"][key],
                serde_json::json!([]),
                "identity_scope.{key} must be []"
            );
        }
    }

    #[tokio::test]
    async fn raw_token_never_appears_in_response_body() {
        // Explicit defense-in-depth assertion: even if the handler is
        // refactored, the raw bearer must not leak into any field.
        let fx = build_fixture("acme");
        let (_status, body) =
            get(fx.state.clone(), "/v1/identity-scope", Some(&fx.raw_token)).await;
        let s = std::str::from_utf8(&body).unwrap();
        assert!(
            !s.contains(&fx.raw_token),
            "raw bearer token leaked into response body"
        );
    }
}

#[cfg(test)]
mod maintenance_e2e {
    use super::e2e_test_support::{build_fixture, build_fixture_with_cluster, get, post};
    use crate::config::NodeRole;

    #[tokio::test]
    async fn status_requires_bearer() {
        let fx = build_fixture("acme");
        let (status, _) = get(fx.state.clone(), "/v1/admin/maintenance/status", None).await;
        assert_eq!(status, 401);
    }

    #[tokio::test]
    async fn status_rejects_non_master_token() {
        // Single-node, no cluster secret → master-token gate denies any token.
        let fx = build_fixture("acme");
        let (status, _) = get(
            fx.state.clone(),
            "/v1/admin/maintenance/status",
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 403);
    }

    #[tokio::test]
    async fn run_rejects_non_master_token() {
        let fx = build_fixture("acme");
        let (status, _) = post(
            fx.state.clone(),
            "/v1/admin/maintenance/run",
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 403);
    }

    #[tokio::test]
    async fn status_ok_with_master_token() {
        let fx = build_fixture_with_cluster("acme", NodeRole::Single, "test-master");
        let (status, body) = get(
            fx.state.clone(),
            "/v1/admin/maintenance/status",
            Some("test-master"),
        )
        .await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Standalone (Single role) accepts writes.
        assert_eq!(v["accepts_writes"], true);
        assert!(v["tenants"].is_array(), "status must list tenants");
    }

    #[tokio::test]
    async fn run_ok_on_write_accepting_node() {
        // Standalone accepts writes → maintenance runs and returns a report
        // per tenant. The cycle is idempotent on an empty engine.
        let fx = build_fixture_with_cluster("acme", NodeRole::Single, "test-master");
        let (status, body) = post(
            fx.state.clone(),
            "/v1/admin/maintenance/run",
            Some("test-master"),
        )
        .await;
        assert_eq!(status, 200, "body: {}", String::from_utf8_lossy(&body));
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(v["ran"].as_u64().unwrap() >= 1, "should run >=1 tenant");
        let results = v["results"].as_array().expect("results array");
        assert!(!results.is_empty());
        assert!(results[0]["tenant"].is_string());
    }

    #[tokio::test]
    async fn run_refused_on_follower_is_cluster_safe() {
        // THE cluster-safety invariant (RFC 027): a follower must NOT run
        // maintenance — it mutates state and would fork the state machine.
        // Voter role starts as Follower → does not accept writes → 409.
        let fx = build_fixture_with_cluster("acme", NodeRole::Voter, "test-master");
        let (status, body) = post(
            fx.state.clone(),
            "/v1/admin/maintenance/run",
            Some("test-master"),
        )
        .await;
        assert_eq!(
            status,
            409,
            "follower must refuse maintenance; body: {}",
            String::from_utf8_lossy(&body)
        );
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        assert!(
            msg.contains("does not accept writes"),
            "409 must explain the write-acceptance gate; got: {msg}"
        );
    }
}

#[cfg(test)]
mod session_lifecycle_e2e {
    use super::e2e_test_support::{build_fixture, build_fixture_with_cluster, get, post_body};
    use crate::config::NodeRole;

    #[tokio::test]
    async fn digest_requires_bearer() {
        let fx = build_fixture("acme");
        let (status, _) = get(fx.state.clone(), "/v1/session/digest", None).await;
        assert_eq!(status, 401);
    }

    #[tokio::test]
    async fn digest_ok_for_tenant_token() {
        // Boot briefing on an empty corpus: 200 with the digest shape
        // (empty decisions/conflicts/triggers, zero counts).
        let fx = build_fixture("acme");
        let (status, body) = get(fx.state.clone(), "/v1/session/digest", Some(&fx.raw_token)).await;
        assert_eq!(status, 200, "body: {}", String::from_utf8_lossy(&body));
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Shape contract the host injects at boot.
        assert!(v.get("top_decisions").is_some());
        assert!(v.get("open_conflict_count").is_some());
        assert!(v.get("pending_trigger_count").is_some());
        assert_eq!(v["open_conflict_count"], 0);
    }

    #[tokio::test]
    async fn end_requires_bearer() {
        let fx = build_fixture("acme");
        let (status, _) = post_body(
            fx.state.clone(),
            "/v1/session/end",
            None,
            r#"{"summary":"x"}"#,
        )
        .await;
        assert_eq!(status, 401);
    }

    #[tokio::test]
    async fn end_rejects_empty_summary() {
        let fx = build_fixture("acme");
        let (status, _) = post_body(
            fx.state.clone(),
            "/v1/session/end",
            Some(&fx.raw_token),
            r#"{"summary":"   "}"#,
        )
        .await;
        assert_eq!(status, 400);
    }

    #[tokio::test]
    async fn end_refused_on_follower_is_cluster_safe() {
        // Cluster-safety: end-of-session capture is a direct engine write, so
        // a follower (Voter→Follower, not write-accepting) must refuse (409)
        // rather than fork the state machine.
        let fx = build_fixture_with_cluster("acme", NodeRole::Voter, "test-master");
        let (status, body) = post_body(
            fx.state.clone(),
            "/v1/session/end",
            Some(&fx.raw_token),
            r#"{"summary":"a meaningful session summary worth capturing"}"#,
        )
        .await;
        assert_eq!(
            status,
            409,
            "follower must refuse capture; body: {}",
            String::from_utf8_lossy(&body)
        );
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        assert!(
            msg.contains("does not accept writes"),
            "409 must explain the write gate; got: {msg}"
        );
    }
}

#[cfg(test)]
mod current_and_gaps_e2e {
    use super::e2e_test_support::{build_fixture, get, post_body};

    #[tokio::test]
    async fn current_requires_bearer() {
        let fx = build_fixture("acme");
        let (status, _) = get(fx.state.clone(), "/v1/current?namespace=chain", None).await;
        assert_eq!(status, 401);
    }

    #[tokio::test]
    async fn current_requires_namespace_param() {
        // namespace is required — a bare /v1/current is a client error, not a
        // whole-DB scan.
        let fx = build_fixture("acme");
        let (status, _) = get(fx.state.clone(), "/v1/current", Some(&fx.raw_token)).await;
        assert!(
            status == 400 || status == 422,
            "missing required namespace should be a 4xx, got {status}"
        );
    }

    #[tokio::test]
    async fn current_404s_on_empty_chain() {
        // Empty corpus → no chain head. 404 (not 200-with-null) so clients can
        // branch on "there is no current value" without inspecting the body.
        let fx = build_fixture("acme");
        let (status, body) = get(
            fx.state.clone(),
            "/v1/current?namespace=nonexistent_chain",
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 404, "body: {}", String::from_utf8_lossy(&body));
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(v["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("no chain head"));
    }

    #[tokio::test]
    async fn gaps_requires_bearer() {
        let fx = build_fixture("acme");
        let (status, _) = get(fx.state.clone(), "/v1/insights/gaps", None).await;
        assert_eq!(status, 401);
    }

    #[tokio::test]
    async fn gaps_ok_empty_on_fresh_corpus() {
        // No recalls logged yet → no demand → no gaps. Shape must still be the
        // stable {count, gaps:[]} envelope.
        let fx = build_fixture("acme");
        let (status, body) = get(fx.state.clone(), "/v1/insights/gaps", Some(&fx.raw_token)).await;
        assert_eq!(status, 200, "body: {}", String::from_utf8_lossy(&body));
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["count"], 0);
        assert!(v["gaps"].is_array());
    }

    #[tokio::test]
    async fn digest_omits_gaps_by_default() {
        // The default digest stays the cheap call — no gaps key unless asked.
        let fx = build_fixture("acme");
        let (status, body) = get(fx.state.clone(), "/v1/session/digest", Some(&fx.raw_token)).await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(
            v.get("knowledge_gaps").is_none(),
            "gaps must be opt-in, not in the default digest"
        );
    }

    #[tokio::test]
    async fn digest_include_superseded_param_is_accepted() {
        // v0.10 contract: the digest accepts ?include_superseded and stays 200
        // (default-false path is exercised by the other digest tests; this pins
        // that the opt-in param parses and doesn't error on an empty corpus).
        let fx = build_fixture("acme");
        let (status, body) = get(
            fx.state.clone(),
            "/v1/session/digest?include_superseded=true",
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 200, "body: {}", String::from_utf8_lossy(&body));
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(v.get("top_decisions").is_some());
    }

    #[tokio::test]
    async fn recall_include_superseded_param_is_accepted() {
        // v0.10 contract: /v1/recall accepts `include_superseded` in the body
        // and threads it through Command::Recall to db.recall. This fixture has
        // no server-side embedder, so a text query reaches the engine and 500s
        // at embed — which itself proves the param PARSED, built the Command,
        // and routed through the recall handler (an unknown/rejected param
        // never reaches the engine). Acceptance + threading is pinned here; the
        // clean 200 path is covered by digest_include_superseded and the
        // engine's own recall(include_superseded) tests at the v0.10 pin.
        let fx = build_fixture("acme");
        let (status, body) = post_body(
            fx.state.clone(),
            "/v1/recall",
            Some(&fx.raw_token),
            r#"{"query":"anything","top_k":5,"include_superseded":true}"#,
        )
        .await;
        let text = String::from_utf8_lossy(&body);
        // 200 (embedder present) OR the specific no-embedder 500 (param reached
        // the engine). A 400 param rejection or a panic fails the test.
        assert!(
            status == 200 || (status == 500 && text.contains("embedder")),
            "include_superseded must be accepted + threaded to the engine; got {status}: {text}"
        );
    }

    #[tokio::test]
    async fn digest_includes_gaps_when_requested() {
        // ?include_gaps=true folds the known-unknowns in — the active-learning
        // loop. Empty corpus → present but empty.
        let fx = build_fixture("acme");
        let (status, body) = get(
            fx.state.clone(),
            "/v1/session/digest?include_gaps=true",
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 200, "body: {}", String::from_utf8_lossy(&body));
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(
            v.get("knowledge_gaps").is_some(),
            "include_gaps=true must add the knowledge_gaps key"
        );
        assert!(v["knowledge_gaps"].is_array());
        // Digest's own shape still intact alongside the folded-in gaps.
        assert!(v.get("top_decisions").is_some());
    }
}

#[cfg(test)]
mod memories_list_tests {
    use super::*;
    use crate::auth::{Principal, Scope, ScopeSet};

    fn pinned(ns: &str) -> Principal {
        Principal::new("tok_abcd1234")
            .with_tenant(ns)
            .with_scopes(ScopeSet::from_iter([
                Scope::Read,
                Scope::Write,
                Scope::Recall,
                Scope::Forget,
            ]))
    }

    fn master() -> Principal {
        // Cluster-wide / master token. principal.tenant_id == None.
        // Mirrors what resolve_engine's cluster_secret branch produces.
        Principal::new("tok_master").with_scopes(ScopeSet::from_iter([
            Scope::Read,
            Scope::Write,
            Scope::Recall,
            Scope::Forget,
            Scope::Admin,
        ]))
    }

    // ── unix_to_iso ─────────────────────────────────────────────────

    #[test]
    fn unix_to_iso_epoch_zero_is_utc_1970() {
        let s = unix_to_iso(0.0).expect("convert");
        assert_eq!(s, "1970-01-01T00:00:00Z");
    }

    #[test]
    fn unix_to_iso_rejects_nan_and_infinity() {
        assert!(unix_to_iso(f64::NAN).is_none());
        assert!(unix_to_iso(f64::INFINITY).is_none());
        assert!(unix_to_iso(f64::NEG_INFINITY).is_none());
    }

    #[test]
    fn unix_to_iso_truncates_to_seconds() {
        // SecondsFormat::Secs strips sub-second precision so the
        // dashboard sees a stable string regardless of how the engine
        // wrote the timestamp.
        let s = unix_to_iso(1_700_000_000.123).expect("convert");
        assert!(
            !s.contains('.'),
            "iso form should not include fraction: {s}"
        );
    }

    // ── memory_to_dashboard_row ─────────────────────────────────────

    fn sample_memory() -> yantrikdb::Memory {
        yantrikdb::Memory {
            rid: "mem_abc".into(),
            memory_type: "fact".into(),
            text: "the sky is blue".into(),
            created_at: 1_700_000_000.0,
            importance: 0.7,
            valence: 0.0,
            half_life: 86_400.0,
            last_access: 1_700_000_100.0,
            access_count: 3,
            consolidation_status: "active".into(),
            storage_tier: "hot".into(),
            consolidated_into: None,
            metadata: serde_json::json!({"a": 1}),
            namespace: "acme".into(),
            certainty: 0.9,
            domain: "general".into(),
            source: "user".into(),
            emotional_state: None,
            session_id: None,
            due_at: None,
            temporal_kind: None,
        }
    }

    #[test]
    fn dashboard_row_has_all_25_required_keys() {
        // Dashboard reads 25 specific top-level keys per the fixture.
        // Phase 1 may emit null for some, but every key must be present.
        let row = memory_to_dashboard_row(&sample_memory());
        for key in [
            "rid",
            "type",
            "text",
            "created_at",
            "created_at_iso",
            "updated_at",
            "updated_at_iso",
            "importance",
            "half_life",
            "last_access",
            "access_count",
            "valence",
            "consolidated_into",
            "consolidation_status",
            "storage_tier",
            "metadata_json",
            "namespace",
            "certainty",
            "domain",
            "source",
            "emotional_state",
            "session_id",
            "due_at",
            "temporal_kind",
            "tombstone_reason",
            "embedding_model",
            "embedding_bytes",
        ] {
            assert!(row.get(key).is_some(), "missing dashboard key `{key}`");
        }
    }

    #[test]
    fn dashboard_row_renames_memory_type_to_type() {
        // Engine: `memory_type`. Dashboard: `type`. Wire contract.
        let row = memory_to_dashboard_row(&sample_memory());
        assert_eq!(row["type"], "fact");
        assert!(
            row.get("memory_type").is_none(),
            "must not surface engine field name"
        );
    }

    #[test]
    fn dashboard_row_renames_metadata_to_metadata_json() {
        let row = memory_to_dashboard_row(&sample_memory());
        assert_eq!(row["metadata_json"], serde_json::json!({"a": 1}));
        assert!(
            row.get("metadata").is_none(),
            "must not surface engine field name"
        );
    }

    #[test]
    fn dashboard_row_phase_1_nulls() {
        // Fields the engine doesn't surface today emit null. If a
        // future engine bump populates them, the assertion needs to be
        // updated alongside.
        let row = memory_to_dashboard_row(&sample_memory());
        for key in [
            "updated_at",
            "updated_at_iso",
            "tombstone_reason",
            "embedding_model",
            "embedding_bytes",
        ] {
            assert_eq!(
                row[key],
                serde_json::Value::Null,
                "Phase 1: `{key}` must be null"
            );
        }
    }

    #[test]
    fn dashboard_row_created_at_iso_matches_unix() {
        let row = memory_to_dashboard_row(&sample_memory());
        let iso = row["created_at_iso"].as_str().unwrap();
        // 1_700_000_000 UTC = 2023-11-14T22:13:20Z
        assert_eq!(iso, "2023-11-14T22:13:20Z");
    }

    // ── validate_memories_params ────────────────────────────────────

    fn body_field<'a>(err: &'a AppError, field: &str) -> &'a serde_json::Value {
        &err.1 .0["error"][field]
    }

    #[test]
    fn validate_defaults_when_params_empty() {
        let p = pinned("acme");
        let params = MemoriesListParams::default();
        let out = validate_memories_params(&p, &params).unwrap();
        assert_eq!(out.db_namespace, "acme");
        // No `?namespace` provided → no tag filter → list all rows in DB.
        assert_eq!(out.tag_filter, None);
        assert_eq!(out.limit, MEMORIES_DEFAULT_LIMIT);
        assert_eq!(out.offset, 0);
        assert_eq!(out.sort_by, "created_at");
    }

    #[test]
    fn validate_rejects_status_other_than_active() {
        let p = pinned("acme");
        let params = MemoriesListParams {
            status: Some("tombstoned".into()),
            ..Default::default()
        };
        let err = validate_memories_params(&p, &params).expect_err("must reject");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert_eq!(body_field(&err, "code"), "invalid_query_parameter");
    }

    #[test]
    fn validate_accepts_explicit_status_active() {
        let p = pinned("acme");
        let params = MemoriesListParams {
            status: Some("active".into()),
            ..Default::default()
        };
        assert!(validate_memories_params(&p, &params).is_ok());
    }

    #[test]
    fn validate_rejects_text_search_q() {
        let p = pinned("acme");
        let params = MemoriesListParams {
            q: Some("anything".into()),
            ..Default::default()
        };
        let err = validate_memories_params(&p, &params).expect_err("must reject");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn validate_rejects_source_filter() {
        let p = pinned("acme");
        let params = MemoriesListParams {
            source: Some("user".into()),
            ..Default::default()
        };
        let err = validate_memories_params(&p, &params).expect_err("must reject");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn validate_rejects_phase_2_sort_with_specific_message() {
        // `updated_at` is in the dashboard spec but engine v0.7.x
        // can't honor it. Surface the gap explicitly.
        let p = pinned("acme");
        let params = MemoriesListParams {
            sort: Some("updated_at".into()),
            ..Default::default()
        };
        let err = validate_memories_params(&p, &params).expect_err("must reject");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(body_field(&err, "message")
            .as_str()
            .unwrap()
            .contains("Phase 1"));
    }

    #[test]
    fn validate_rejects_unknown_sort() {
        let p = pinned("acme");
        let params = MemoriesListParams {
            sort: Some("bogus".into()),
            ..Default::default()
        };
        let err = validate_memories_params(&p, &params).expect_err("must reject");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn validate_accepts_each_supported_sort() {
        let p = pinned("acme");
        for sort in MEMORIES_SUPPORTED_SORTS {
            let params = MemoriesListParams {
                sort: Some((*sort).into()),
                ..Default::default()
            };
            let out = validate_memories_params(&p, &params).expect("must accept");
            assert_eq!(out.sort_by, *sort);
        }
    }

    #[test]
    fn validate_caps_limit_at_200() {
        let p = pinned("acme");
        let params = MemoriesListParams {
            limit: Some(500),
            ..Default::default()
        };
        let err = validate_memories_params(&p, &params).expect_err("must reject");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        // Message names the cap so callers know how to fix.
        assert!(body_field(&err, "message")
            .as_str()
            .unwrap()
            .contains(&MEMORIES_MAX_LIMIT.to_string()));
    }

    #[test]
    fn validate_rejects_limit_zero() {
        let p = pinned("acme");
        let params = MemoriesListParams {
            limit: Some(0),
            ..Default::default()
        };
        let err = validate_memories_params(&p, &params).expect_err("must reject");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn validate_accepts_arbitrary_namespace_as_tag_filter_for_pinned_token() {
        // Per yantrikdb-core decision (swarm 8a97464e, 2026-06-09):
        // `namespace` is a row-level TAG, not a tenant scope. A per-tenant
        // token can now pass `?namespace=skill_substrate` as a tag filter
        // — the DB is still routed by `principal.tenant_id`. Prior code
        // 403'd on mismatch, which was wrong.
        let p = pinned("acme");
        let params = MemoriesListParams {
            namespace: Some("skill_substrate".into()),
            ..Default::default()
        };
        let out = validate_memories_params(&p, &params).unwrap();
        assert_eq!(out.db_namespace, "acme"); // routed by tenant
        assert_eq!(out.tag_filter.as_deref(), Some("skill_substrate")); // tag filter
    }

    #[test]
    fn validate_accepts_matching_namespace_query_for_pinned_token() {
        let p = pinned("acme");
        let params = MemoriesListParams {
            namespace: Some("acme".into()),
            ..Default::default()
        };
        let out = validate_memories_params(&p, &params).unwrap();
        assert_eq!(out.db_namespace, "acme");
        // `?namespace=acme` is still treated as a tag filter — explicit
        // provision means the caller wants to filter by that tag (which
        // happens to coincide with the tenant name here).
        assert_eq!(out.tag_filter.as_deref(), Some("acme"));
    }

    #[test]
    fn validate_master_token_no_namespace_routes_to_default_db() {
        // v0.8.22 fix: master/cluster-wide tokens (principal.tenant_id ==
        // None) route to the "default" database, matching
        // resolve_engine's hardcoded behavior. Prior to v0.8.22 this
        // case errored with `namespace is required for cluster-wide
        // tokens`, blocking algo's master-token workflow on CT 133.
        let p = master();
        let params = MemoriesListParams::default();
        let out = validate_memories_params(&p, &params).unwrap();
        assert_eq!(out.db_namespace, "default");
        assert_eq!(out.tag_filter, None);
    }

    #[test]
    fn validate_master_token_with_namespace_uses_default_db_and_tag_filter() {
        // v0.8.22 fix: master tokens with `?namespace=fable3` route to
        // "default" DB AND apply `fable3` as a tag filter — they no
        // longer mis-route `?namespace` to `get_database("fable3")`
        // (which 404'd because `fable3` is a row tag, not a database).
        // This is the exact case yantrikdb-agi reported on CT 133
        // (swarm 77ffa517, 2026-06-10).
        let p = master();
        let params = MemoriesListParams {
            namespace: Some("yantrikos_entity_fable3".into()),
            ..Default::default()
        };
        let out = validate_memories_params(&p, &params).unwrap();
        assert_eq!(out.db_namespace, "default");
        assert_eq!(out.tag_filter.as_deref(), Some("yantrikos_entity_fable3"));
    }

    // ── v0.8.23 structural-query primitive params ───────────────────

    #[test]
    fn validate_v0823_default_order_is_asc() {
        let p = pinned("acme");
        let params = MemoriesListParams::default();
        let out = validate_memories_params(&p, &params).unwrap();
        assert_eq!(out.order, "asc");
        assert!(out.kind.is_none());
        assert!(out.drive_id.is_none());
        assert!(out.since_rid.is_none());
    }

    #[test]
    fn validate_v0823_accepts_desc_order() {
        let p = pinned("acme");
        let params = MemoriesListParams {
            order: Some("desc".into()),
            ..Default::default()
        };
        let out = validate_memories_params(&p, &params).unwrap();
        assert_eq!(out.order, "desc");
    }

    #[test]
    fn validate_v0823_rejects_unknown_order() {
        let p = pinned("acme");
        let params = MemoriesListParams {
            order: Some("backwards".into()),
            ..Default::default()
        };
        let err = validate_memories_params(&p, &params).expect_err("must reject");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(body_field(&err, "message")
            .as_str()
            .unwrap()
            .contains("asc | desc"));
    }

    #[test]
    fn validate_v0823_kind_drive_id_since_rid_round_trip() {
        // The validator passes these through unmolested; the handler
        // routes them down to engine.list_records.
        let p = pinned("acme");
        let params = MemoriesListParams {
            kind: Some("operator_reply_v1".into()),
            drive_id: Some("019ea-drive".into()),
            since_rid: Some("019eb-cursor".into()),
            order: Some("desc".into()),
            ..Default::default()
        };
        let out = validate_memories_params(&p, &params).unwrap();
        assert_eq!(out.kind.as_deref(), Some("operator_reply_v1"));
        assert_eq!(out.drive_id.as_deref(), Some("019ea-drive"));
        assert_eq!(out.since_rid.as_deref(), Some("019eb-cursor"));
        assert_eq!(out.order, "desc");
        assert_eq!(out.db_namespace, "acme"); // tenant routing unchanged
    }
}

#[cfg(test)]
mod memories_list_e2e {
    use super::e2e_test_support::{build_fixture, get};

    /// Plant a memory directly via the engine handle so the e2e test
    /// has rows to list without needing a real embedder.
    async fn plant_memory(
        state: std::sync::Arc<crate::server::AppState>,
        namespace: &str,
        text: &str,
    ) {
        // Look up the database for the namespace and grab its engine.
        let db = {
            let ctrl = state.control.lock();
            ctrl.get_database(namespace).unwrap().unwrap()
        };
        let engine = state.pool.get_engine(&db).unwrap();
        let text_owned = text.to_string();
        let namespace_owned = namespace.to_string();
        tokio::task::spawn_blocking(move || {
            // Fake fixed-dim embedding — engine doesn't validate vector
            // content for list_memories, only that the row exists.
            let embedding = vec![0.0_f32; 384];
            engine
                .record(
                    &text_owned,
                    "fact",
                    0.5,
                    0.0,
                    86_400.0,
                    &serde_json::json!({}),
                    &embedding,
                    &namespace_owned,
                    1.0,
                    "general",
                    "test",
                    None,
                )
                .unwrap();
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn returns_401_when_no_bearer_header() {
        let fx = build_fixture("acme");
        let (status, body) = get(fx.state.clone(), "/v1/memories", None).await;
        assert_eq!(status, 401);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "unauthenticated");
    }

    #[tokio::test]
    async fn returns_400_on_unsupported_q_filter() {
        let fx = build_fixture("acme");
        let (status, body) = get(
            fx.state.clone(),
            "/v1/memories?q=anything",
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 400);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "invalid_query_parameter");
    }

    #[tokio::test]
    async fn pinned_token_can_use_arbitrary_namespace_as_tag_filter() {
        // Per yantrikdb-core decision (swarm 8a97464e, 2026-06-09):
        // `namespace` is a row-level TAG, not a tenant scope. A per-tenant
        // token's `?namespace=skill_substrate` is a tag filter against the
        // caller's database (NOT a cross-tenant access attempt). Prior
        // code 403'd on mismatch with `namespace_not_found`; new behavior
        // is 200 with the rows tagged `secret` (empty here since none
        // exist). This unblocks the dashboard's tag-filter UI for the
        // 200k+ rows on trader's `default` tenant tagged `skill_substrate`,
        // `comm_substrate`, etc.
        let fx = build_fixture("acme");
        let (status, body) = get(
            fx.state.clone(),
            "/v1/memories?namespace=secret",
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(
            status,
            200,
            "tag-filter request must succeed (empty result), got {status}: {}",
            String::from_utf8_lossy(&body)
        );
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["total"], 0); // no rows tagged "secret" in this tenant
    }

    #[tokio::test]
    async fn returns_empty_page_with_envelope_when_no_memories() {
        let fx = build_fixture("acme");
        let (status, body) = get(fx.state.clone(), "/v1/memories", Some(&fx.raw_token)).await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["total"], 0);
        assert_eq!(v["limit"], 50);
        assert_eq!(v["offset"], 0);
        assert_eq!(v["items"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn returns_planted_memory_with_dashboard_row_shape() {
        let fx = build_fixture("acme");
        plant_memory(fx.state.clone(), "acme", "hello world").await;

        let (status, body) = get(fx.state.clone(), "/v1/memories", Some(&fx.raw_token)).await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["total"], 1);
        let items = v["items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["text"], "hello world");
        assert_eq!(items[0]["type"], "fact");
        assert_eq!(items[0]["namespace"], "acme");
        // Phase-1 null fields surface as null, not missing.
        assert_eq!(items[0]["updated_at"], serde_json::Value::Null);
        assert_eq!(items[0]["embedding_model"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn offset_paginates_planted_rows() {
        let fx = build_fixture("acme");
        for i in 0..3 {
            plant_memory(fx.state.clone(), "acme", &format!("memory-{i}")).await;
        }
        // First page: limit=2 offset=0
        let (status, body) = get(
            fx.state.clone(),
            "/v1/memories?limit=2&offset=0",
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["total"], 3);
        assert_eq!(v["limit"], 2);
        assert_eq!(v["offset"], 0);
        assert_eq!(v["items"].as_array().unwrap().len(), 2);

        // Second page: limit=2 offset=2 → 1 row
        let (status, body) = get(
            fx.state.clone(),
            "/v1/memories?limit=2&offset=2",
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["total"], 3);
        assert_eq!(v["items"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn limit_over_200_is_rejected() {
        let fx = build_fixture("acme");
        let (status, _body) = get(
            fx.state.clone(),
            "/v1/memories?limit=999",
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 400);
    }
}

#[cfg(test)]
mod memory_get_e2e {
    use super::e2e_test_support::{build_fixture, get};

    async fn plant_memory(
        state: std::sync::Arc<crate::server::AppState>,
        namespace: &str,
        text: &str,
    ) -> String {
        let db = {
            let ctrl = state.control.lock();
            ctrl.get_database(namespace).unwrap().unwrap()
        };
        let engine = state.pool.get_engine(&db).unwrap();
        let text_owned = text.to_string();
        let namespace_owned = namespace.to_string();
        tokio::task::spawn_blocking(move || {
            let embedding = vec![0.0_f32; 384];
            engine
                .record(
                    &text_owned,
                    "fact",
                    0.5,
                    0.0,
                    86_400.0,
                    &serde_json::json!({}),
                    &embedding,
                    &namespace_owned,
                    1.0,
                    "general",
                    "test",
                    None,
                )
                .unwrap()
        })
        .await
        .unwrap()
    }

    /// Plant a memory into `db_namespace`'s database but stamp its
    /// `namespace` *column* with `row_namespace`. Lets a test reproduce the
    /// production write default, where `/v1/remember` stores `namespace=""`
    /// when the client omits the field.
    async fn plant_memory_with_row_namespace(
        state: std::sync::Arc<crate::server::AppState>,
        db_namespace: &str,
        row_namespace: &str,
        text: &str,
    ) -> String {
        let db = {
            let ctrl = state.control.lock();
            ctrl.get_database(db_namespace).unwrap().unwrap()
        };
        let engine = state.pool.get_engine(&db).unwrap();
        let text_owned = text.to_string();
        let ns_owned = row_namespace.to_string();
        tokio::task::spawn_blocking(move || {
            let embedding = vec![0.0_f32; 384];
            engine
                .record(
                    &text_owned,
                    "fact",
                    0.5,
                    0.0,
                    86_400.0,
                    &serde_json::json!({}),
                    &embedding,
                    &ns_owned,
                    1.0,
                    "general",
                    "test",
                    None,
                )
                .unwrap()
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn returns_row_stored_with_empty_default_namespace() {
        // Regression: `/v1/remember` stores `namespace = ""` when the client
        // omits the field; `/v1/recall` returns such rows, but the point
        // read used to 404 because `"" != "acme"`. An unscoped row in the
        // caller's own database must be readable by rid.
        let fx = build_fixture("acme");
        let rid =
            plant_memory_with_row_namespace(fx.state.clone(), "acme", "", "brand color is blue")
                .await;
        let (status, body) = get(
            fx.state.clone(),
            &format!("/v1/memory/{rid}"),
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(
            status,
            200,
            "default-namespace row must be readable by rid, got {status}: {}",
            String::from_utf8_lossy(&body)
        );
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["rid"], rid);
        assert_eq!(v["text"], "brand color is blue");
    }

    #[tokio::test]
    async fn returns_row_with_arbitrary_nonempty_namespace_in_caller_database() {
        // Per yantrikdb-core decision (swarm 8a97464e, 2026-06-09):
        // `namespace` is a row-level TAG, not a tenant scope. ANY row
        // located in the caller's database by rid is theirs, regardless
        // of namespace tag. This was historically guarded by a strict
        // equality check that 404'd rows tagged with cross-cutting
        // values like `skill_substrate` (200k+ such rows on production
        // trader). The guard is gone; cross-tenant isolation comes from
        // the database boundary alone.
        let fx = build_fixture("acme");
        let rid = plant_memory_with_row_namespace(
            fx.state.clone(),
            "acme",
            "skill_substrate",
            "skill row",
        )
        .await;
        let (status, body) = get(
            fx.state.clone(),
            &format!("/v1/memory/{rid}"),
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(
            status,
            200,
            "skill_substrate row in caller's database must be readable by rid, got {status}: {}",
            String::from_utf8_lossy(&body)
        );
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["rid"], rid);
        assert_eq!(v["text"], "skill row");
    }

    #[tokio::test]
    async fn returns_401_when_no_bearer_header() {
        let fx = build_fixture("acme");
        let (status, body) = get(fx.state.clone(), "/v1/memory/anything", None).await;
        assert_eq!(status, 401);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "unauthenticated");
    }

    #[tokio::test]
    async fn returns_404_for_unknown_rid() {
        let fx = build_fixture("acme");
        let (status, body) = get(
            fx.state.clone(),
            "/v1/memory/mem_does_not_exist",
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 404);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "memory_not_found");
    }

    #[tokio::test]
    async fn returns_planted_memory_with_dashboard_shape_and_conditional_arrays() {
        let fx = build_fixture("acme");
        let rid = plant_memory(fx.state.clone(), "acme", "the sky is blue").await;
        let (status, body) = get(
            fx.state.clone(),
            &format!("/v1/memory/{rid}"),
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["rid"], rid);
        assert_eq!(v["text"], "the sky is blue");
        assert_eq!(v["type"], "fact");
        assert_eq!(v["namespace"], "acme");
        // Phase-1 conditional includes: empty arrays, not missing.
        assert_eq!(v["consolidation_sources"], serde_json::json!([]));
        assert_eq!(v["entities"], serde_json::json!([]));
        assert_eq!(v["claims"], serde_json::json!([]));
        // Phase-1 nulls from the row mapper still apply.
        assert_eq!(v["updated_at"], serde_json::Value::Null);
        assert_eq!(v["embedding_model"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn pinned_token_with_ns_param_on_nonexistent_rid_returns_404_not_403() {
        // Per yantrikdb-core decision (swarm 8a97464e, 2026-06-09) +
        // yantrikdb-agi report (swarm 77ffa517, 2026-06-10): `?namespace`
        // is irrelevant on point-read — rid uniquely identifies the row,
        // the database (gated by the token) is the isolation boundary.
        // A nonexistent rid returns 404 memory_not_found regardless of
        // the `?namespace` value the caller supplies.
        let fx = build_fixture("acme");
        let (status, body) = get(
            fx.state.clone(),
            "/v1/memory/mem_anything?namespace=secret",
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 404);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "memory_not_found");
    }

    #[tokio::test]
    async fn min_seq_zero_passes_in_single_node_mode() {
        // Single-node has no openraft assembly → no replica drift →
        // every min_seq value is satisfied trivially. This is the
        // documented Phase-1 behavior. Cluster-mode handling lives
        // behind state.raft.is_some().
        let fx = build_fixture("acme");
        let rid = plant_memory(fx.state.clone(), "acme", "hello").await;
        let (status, _body) = get(
            fx.state.clone(),
            &format!("/v1/memory/{rid}?min_seq=0"),
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 200);
    }

    #[tokio::test]
    async fn min_seq_huge_value_also_passes_in_single_node_mode() {
        // Same as above — pin the contract: single-node satisfies any
        // min_seq because there is no replication lag concept.
        let fx = build_fixture("acme");
        let rid = plant_memory(fx.state.clone(), "acme", "hello").await;
        let (status, _body) = get(
            fx.state.clone(),
            &format!("/v1/memory/{rid}?min_seq=9999999"),
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 200);
    }

    #[tokio::test]
    async fn ns_param_is_ignored_on_point_read() {
        // Per v0.8.22 row-tag canonicalization: `?namespace` is
        // irrelevant for /v1/memory/{rid}. The token determines the DB
        // (per-tenant) or routes to "default" (master); the rid
        // uniquely identifies the row within that DB. Any value of
        // `?namespace` returns the same result. This is the strict
        // generalization of v0.8.21's "drop the namespace guard" fix —
        // the query param is now plumbed through to the engine layer
        // but has no effect on routing or filtering for point-read.
        let fx = build_fixture("acme");
        let rid = plant_memory(fx.state.clone(), "acme", "hello").await;
        let (status, body) = get(
            fx.state.clone(),
            &format!("/v1/memory/{rid}?namespace=other"),
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(
            status,
            200,
            "?namespace=other must not block the read, got {status}: {}",
            String::from_utf8_lossy(&body)
        );
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["rid"], rid);
        assert_eq!(v["text"], "hello");
    }
}

#[cfg(test)]
mod fts5_fallback_tests {
    use super::annotate_fts5_fallback;
    use axum::Json;
    use serde_json::{json, Value};

    fn annotate(body: Value) -> Value {
        let mut wrapped = Json(body);
        annotate_fts5_fallback(&mut wrapped);
        wrapped.0
    }

    #[test]
    fn no_results_emits_null_fallback() {
        // Empty result set: no FTS5 fallback was triggered (nothing
        // matched anywhere). Field must still appear so dashboards
        // can rely on presence.
        let out = annotate(json!({"results": [], "total": 0}));
        assert_eq!(out["fallback"], Value::Null);
        assert_eq!(out["total"], 0);
    }

    #[test]
    fn semantic_only_results_emit_null_fallback() {
        // why_retrieved doesn't include "keyword_match" → semantic was
        // sufficient → fallback is null.
        let out = annotate(json!({
            "results": [
                {"rid": "mem_1", "why_retrieved": ["semantic"]},
                {"rid": "mem_2", "why_retrieved": ["semantic", "graph-connected via Alice"]},
            ],
            "total": 2,
        }));
        assert_eq!(out["fallback"], Value::Null);
    }

    #[test]
    fn any_keyword_match_emits_fts5_keyword() {
        // The dashboard contract: presence of any FTS5-sourced row
        // means the search degraded to keyword matching. Wire string
        // is pinned because the dashboard reads it literally.
        let out = annotate(json!({
            "results": [
                {"rid": "mem_1", "why_retrieved": ["keyword_match"]},
                {"rid": "mem_2", "why_retrieved": ["semantic"]},
            ],
            "total": 2,
        }));
        assert_eq!(out["fallback"], "fts5_keyword");
    }

    #[test]
    fn keyword_match_on_only_row_emits_fts5_keyword() {
        let out = annotate(json!({
            "results": [{"rid": "mem_1", "why_retrieved": ["keyword_match"]}],
            "total": 1,
        }));
        assert_eq!(out["fallback"], "fts5_keyword");
    }

    #[test]
    fn missing_why_retrieved_treated_as_no_keyword_match() {
        // Defensive: if a future engine version stops emitting
        // why_retrieved, default to "no fallback" rather than panic.
        let out = annotate(json!({
            "results": [{"rid": "mem_1"}],
            "total": 1,
        }));
        assert_eq!(out["fallback"], Value::Null);
    }

    #[test]
    fn non_object_body_is_left_alone() {
        // Sanity: passing an array instead of an object doesn't panic;
        // the annotator no-ops because there's no top-level slot to
        // insert into.
        let out = annotate(json!([1, 2, 3]));
        // Array unchanged, no fallback key.
        assert!(out.is_array());
    }

    #[test]
    fn existing_fallback_value_is_overwritten() {
        // The annotator is authoritative — if a caller pre-set a
        // fallback field, we replace it with our computed value so
        // there's only one source of truth.
        let out = annotate(json!({
            "results": [{"rid": "mem_1", "why_retrieved": ["keyword_match"]}],
            "total": 1,
            "fallback": "stale-marker",
        }));
        assert_eq!(out["fallback"], "fts5_keyword");
    }

    #[test]
    fn preserves_other_top_level_fields() {
        let out = annotate(json!({
            "results": [],
            "total": 0,
            "summary": {"top_similarity": 0.9},
        }));
        assert_eq!(out["total"], 0);
        assert_eq!(out["summary"]["top_similarity"], 0.9);
        assert!(out.get("fallback").is_some());
    }
}

// ── Issue #39 task 201: dashboard contract fixture-asserted tests ───
//
// Captures wysie's dashboard JSON shape as a set of fixture files
// under src/api/fixtures/. Each test queries the production router
// via the e2e helper, then asserts SHAPE compatibility with the
// fixture (key presence + JSON type discriminant), not value
// equality.
//
// Why shape-not-value: timestamps, RIDs, and IDs vary per run. But
// the dashboard reads specific KEYS at specific NESTING — those are
// the wire contract. If field names drift, this test fails loud.
//
// Null in the fixture is treated as a wildcard (matches any actual
// type) so Phase-1 null fields can be filled in by future engine
// extensions without rewriting the fixture.

#[cfg(test)]
mod contract_fixture_tests {
    use super::e2e_test_support::{build_fixture, get};
    use serde_json::Value;

    const FIXTURE_IDENTITY_SCOPE_PINNED: &str =
        include_str!("api/fixtures/v1_identity_scope_pinned.json");
    const FIXTURE_MEMORIES_WITH_ROW: &str = include_str!("api/fixtures/v1_memories_with_row.json");
    const FIXTURE_MEMORY_DETAIL: &str = include_str!("api/fixtures/v1_memory_detail.json");

    /// Assert that `actual` is shape-compatible with `expected`:
    /// - For objects: every key in `expected` MUST be present in
    ///   `actual` at the same nesting; values are matched recursively.
    /// - For arrays: if both have at least one element, the first
    ///   element shapes are matched. Empty arrays match each other.
    /// - For primitives: type discriminants must match.
    ///   `Value::Null` in `expected` is a wildcard (matches anything)
    ///   so Phase-1 null fields can be populated by future engine
    ///   revisions without breaking the fixture.
    ///
    /// `path` is the dotted JSON pointer reported in assertion
    /// failures so the offending key is easy to find.
    fn assert_shape_matches(actual: &Value, expected: &Value, path: &str) {
        // Null in fixture = wildcard for typed values (forward-compat
        // headroom). Null in actual = forward-compat with fixtures
        // that asserted a typed primitive but engine is still ramping
        // to that. Both directions intentional.
        if expected.is_null() || actual.is_null() {
            return;
        }
        match (actual, expected) {
            (Value::Object(a), Value::Object(e)) => {
                for (k, ev) in e {
                    let av = a.get(k).unwrap_or_else(|| {
                        panic!(
                            "missing key `{path}.{k}` in actual response; got keys: {:?}",
                            a.keys().collect::<Vec<_>>()
                        )
                    });
                    let next_path = if path.is_empty() {
                        k.clone()
                    } else {
                        format!("{path}.{k}")
                    };
                    assert_shape_matches(av, ev, &next_path);
                }
            }
            (Value::Array(a), Value::Array(e)) => {
                if let (Some(av), Some(ev)) = (a.first(), e.first()) {
                    assert_shape_matches(av, ev, &format!("{path}[0]"));
                }
                // Empty fixture array: actual may have any contents,
                // including nothing. Empty actual: matches any
                // fixture array (Phase-1 conditional-includes
                // surface as []).
            }
            (a, e) => {
                let same_kind = std::mem::discriminant(a) == std::mem::discriminant(e);
                // Treat all JSON numbers as one type — fixtures may
                // write `1700000000.0` while the engine emits an
                // integer for `access_count`.
                let numeric = a.is_number() && e.is_number();
                assert!(
                    same_kind || numeric,
                    "type mismatch at `{path}`: actual={a:?} fixture={e:?}",
                );
            }
        }
    }

    async fn plant_memory(
        state: std::sync::Arc<crate::server::AppState>,
        namespace: &str,
        text: &str,
    ) -> String {
        let db = {
            let ctrl = state.control.lock();
            ctrl.get_database(namespace).unwrap().unwrap()
        };
        let engine = state.pool.get_engine(&db).unwrap();
        let text_owned = text.to_string();
        let namespace_owned = namespace.to_string();
        tokio::task::spawn_blocking(move || {
            let embedding = vec![0.0_f32; 384];
            engine
                .record(
                    &text_owned,
                    "fact",
                    0.5,
                    0.0,
                    86_400.0,
                    &serde_json::json!({}),
                    &embedding,
                    &namespace_owned,
                    1.0,
                    "general",
                    "test",
                    None,
                )
                .unwrap()
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn identity_scope_response_matches_dashboard_fixture() {
        let fx = build_fixture("acme");
        let (status, body) = get(fx.state.clone(), "/v1/identity-scope", Some(&fx.raw_token)).await;
        assert_eq!(status, 200);
        let actual: Value = serde_json::from_slice(&body).expect("parse response");
        let expected: Value =
            serde_json::from_str(FIXTURE_IDENTITY_SCOPE_PINNED).expect("parse fixture");
        assert_shape_matches(&actual, &expected, "");
    }

    #[tokio::test]
    async fn memories_response_matches_dashboard_fixture() {
        let fx = build_fixture("acme");
        plant_memory(fx.state.clone(), "acme", "fixture seed").await;
        let (status, body) = get(fx.state.clone(), "/v1/memories", Some(&fx.raw_token)).await;
        assert_eq!(status, 200);
        let actual: Value = serde_json::from_slice(&body).expect("parse response");
        let expected: Value =
            serde_json::from_str(FIXTURE_MEMORIES_WITH_ROW).expect("parse fixture");
        assert_shape_matches(&actual, &expected, "");
    }

    #[tokio::test]
    async fn memory_detail_response_matches_dashboard_fixture() {
        let fx = build_fixture("acme");
        let rid = plant_memory(fx.state.clone(), "acme", "fixture seed").await;
        let (status, body) = get(
            fx.state.clone(),
            &format!("/v1/memory/{rid}"),
            Some(&fx.raw_token),
        )
        .await;
        assert_eq!(status, 200);
        let actual: Value = serde_json::from_slice(&body).expect("parse response");
        let expected: Value = serde_json::from_str(FIXTURE_MEMORY_DETAIL).expect("parse fixture");
        assert_shape_matches(&actual, &expected, "");
    }

    // ── Self-tests for assert_shape_matches (the assertion needs to
    //    be at least as careful as the contract it's enforcing) ──────

    #[test]
    fn shape_passes_on_identical_object() {
        let v = serde_json::json!({"a": 1, "b": "x"});
        assert_shape_matches(&v, &v, "");
    }

    #[test]
    #[should_panic(expected = "missing key")]
    fn shape_fails_on_missing_key_in_actual() {
        let actual = serde_json::json!({"a": 1});
        let expected = serde_json::json!({"a": 1, "b": "x"});
        assert_shape_matches(&actual, &expected, "");
    }

    #[test]
    fn shape_ignores_extra_keys_in_actual() {
        // Actual can have MORE keys than fixture — fixture is the
        // floor, not the ceiling. New fields land additively.
        let actual = serde_json::json!({"a": 1, "b": "x", "c": "extra"});
        let expected = serde_json::json!({"a": 1, "b": "x"});
        assert_shape_matches(&actual, &expected, "");
    }

    #[test]
    #[should_panic(expected = "type mismatch")]
    fn shape_fails_on_type_mismatch() {
        let actual = serde_json::json!({"a": "string"});
        let expected = serde_json::json!({"a": 1});
        assert_shape_matches(&actual, &expected, "");
    }

    #[test]
    fn shape_passes_when_actual_is_int_and_fixture_is_float() {
        // Both are JSON numbers; the engine may emit an integer where
        // the fixture wrote a float (e.g. access_count). Don't break
        // on that.
        let actual = serde_json::json!({"n": 3});
        let expected = serde_json::json!({"n": 1.5});
        assert_shape_matches(&actual, &expected, "");
    }

    #[test]
    fn shape_treats_null_fixture_as_wildcard() {
        // Fixture says "null today, anything tomorrow" — used for
        // Phase-1 nulls that future engine revs may populate.
        let actual = serde_json::json!({"x": "actual string"});
        let expected = serde_json::json!({"x": null});
        assert_shape_matches(&actual, &expected, "");
    }

    #[test]
    fn shape_treats_null_actual_as_wildcard_against_typed_fixture() {
        // Inverse: a Phase-1 handler emits null today but the fixture
        // captured a typed example. Don't break.
        let actual = serde_json::json!({"x": null});
        let expected = serde_json::json!({"x": "example"});
        assert_shape_matches(&actual, &expected, "");
    }
}
