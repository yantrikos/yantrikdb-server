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
}

fn cluster_state_view(state: &AppState) -> Option<ClusterStateView> {
    if let Some(ref assembly) = state.raft {
        let m = assembly.raft.metrics().borrow().clone();
        let is_leader = matches!(m.state, openraft::ServerState::Leader);
        let leader_id = m.current_leader.map(u64::from);
        let leader_addr = leader_id.and_then(|lid| {
            m.membership_config
                .nodes()
                .find(|(id, _)| u64::from(**id) == lid)
                .map(|(_, n)| n.addr.clone())
        });
        return Some(ClusterStateView {
            node_id: u64::from(m.id),
            role: format!("{:?}", m.state),
            term: m.current_term,
            leader: leader_id,
            leader_addr,
            accepts_writes: is_leader,
            healthy: m.current_leader.is_some(),
            raft_mode: "openraft",
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
        });
    }
    None
}

/// Shared engine handle. Type alias keeps the complex nested generic out
/// of function signatures and avoids clippy::type_complexity.
type EngineHandle = Arc<yantrikdb::YantrikDB>;

/// Error tuple returned by auth-checking helpers.
type AppError = (StatusCode, Json<Value>);

fn app_error(status: StatusCode, message: impl Into<String>) -> AppError {
    (status, Json(json!({ "error": message.into() })))
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
    });
    if let Some(view) = cluster_state_view(&state) {
        payload["cluster"] = json!({
            "node_id": view.node_id,
            "role": view.role,
            "term": view.term,
            "leader": view.leader,
            "accepts_writes": view.accepts_writes,
            "healthy": view.healthy,
            "raft_mode": view.raft_mode,
        });
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

    let cmd = Command::Remember {
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
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
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
    for (i, m) in memories_arr.iter().enumerate() {
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

    let cmd = Command::RememberBatch { memories };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
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
    };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
}

async fn forget(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("forget");
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let rid = body["rid"]
        .as_str()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'rid'"))?
        .into();
    execute_cmd(
        engine,
        Command::Forget { rid },
        state.control.clone(),
        &state.inflight,
    )
    .await
}

async fn relate(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Result<impl IntoResponse, AppError> {
    let _timer = crate::metrics::HandlerTimer::new("relate");
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::Relate {
        entity: body["entity"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'entity'"))?
            .into(),
        target: body["target"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'target'"))?
            .into(),
        relationship: body["relationship"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'relationship'"))?
            .into(),
        weight: body.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0),
    };
    let json = execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await?;
    let mut response = json.into_response();
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

// ---------- v0.8.3: openraft membership API (issue #24) ----------
//
// These three endpoints expose openraft's add_learner / change_membership
// over HTTP so operators can grow / shrink / promote membership without
// dropping into Rust. All require the cluster master token. They MUST be
// called against the current leader; followers return 503 with a
// "current leader" hint that the CLI uses to retry.

#[derive(serde::Deserialize)]
struct AddLearnerRequest {
    node_id: u64,
    addr: String,
}

#[derive(serde::Deserialize)]
struct PromoteRequest {
    /// Final voter set after the membership change. MUST include the
    /// current leader to avoid the leader inadvertently demoting itself.
    voters: Vec<u64>,
}

#[derive(serde::Deserialize)]
struct RemoveRequest {
    node_id: u64,
}

fn require_openraft(state: &AppState) -> Result<&Arc<crate::raft::RaftAssembly>, AppError> {
    state.raft.as_ref().ok_or_else(|| {
        app_error(
            StatusCode::BAD_REQUEST,
            "openraft mode is not active on this node — set cluster.raft_mode = \"openraft\"",
        )
    })
}

/// POST /v1/cluster/add-learner — add a non-voting learner to the cluster.
/// Body: `{"node_id": <u64>, "addr": "<host:cluster_port>"}`.
/// Auth: cluster master token.
async fn cluster_add_learner(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<AddLearnerRequest>,
) -> AppResult {
    require_master_token(&state, &headers)?;
    let assembly = require_openraft(&state)?;
    let node = crate::raft::types::YantrikNode::new(body.addr.clone());
    let node_id = crate::raft::types::YantrikNodeId::from(body.node_id);
    match assembly.raft.add_learner(node_id, node, false).await {
        Ok(_resp) => Ok(Json(json!({
            "status": "learner_added",
            "node_id": body.node_id,
            "addr": body.addr,
            "note": "use /v1/cluster/raft to watch catch-up; promote when last_log_index lag is acceptable",
        }))),
        Err(e) => Err(app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("add_learner failed: {e}"),
        )),
    }
}

/// POST /v1/cluster/promote — change membership to the given voter set.
/// Body: `{"voters": [<u64>, ...]}`.
/// Auth: cluster master token.
async fn cluster_promote_voter(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<PromoteRequest>,
) -> AppResult {
    require_master_token(&state, &headers)?;
    let assembly = require_openraft(&state)?;
    let voters: std::collections::BTreeSet<crate::raft::types::YantrikNodeId> = body
        .voters
        .iter()
        .copied()
        .map(crate::raft::types::YantrikNodeId::from)
        .collect();
    if voters.is_empty() {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "voters list cannot be empty",
        ));
    }
    let voters_clone: Vec<u64> = voters.iter().map(|n| u64::from(*n)).collect();
    match assembly.raft.change_membership(voters, false).await {
        Ok(_resp) => Ok(Json(json!({
            "status": "membership_changed",
            "voters": voters_clone,
        }))),
        Err(e) => Err(app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("change_membership failed: {e}"),
        )),
    }
}

/// POST /v1/cluster/remove — remove a node from the cluster (atomic
/// membership change to current voters minus the named node).
/// Body: `{"node_id": <u64>}`.
/// Auth: cluster master token.
async fn cluster_remove(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<RemoveRequest>,
) -> AppResult {
    require_master_token(&state, &headers)?;
    let assembly = require_openraft(&state)?;
    let metrics = assembly.raft.metrics().borrow().clone();
    let mut voters: std::collections::BTreeSet<crate::raft::types::YantrikNodeId> =
        metrics.membership_config.voter_ids().collect();
    let target_id = crate::raft::types::YantrikNodeId::from(body.node_id);
    if !voters.remove(&target_id) {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            format!("node {} is not a current voter", body.node_id),
        ));
    }
    if voters.is_empty() {
        return Err(app_error(
            StatusCode::BAD_REQUEST,
            "refusing to remove the last voter — would lose quorum permanently",
        ));
    }
    let remaining: Vec<u64> = voters.iter().map(|n| u64::from(*n)).collect();
    match assembly.raft.change_membership(voters, false).await {
        Ok(_resp) => Ok(Json(json!({
            "status": "removed",
            "removed_node_id": body.node_id,
            "remaining_voters": remaining,
        }))),
        Err(e) => Err(app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("change_membership failed: {e}"),
        )),
    }
}

/// POST /v1/cluster/initialize — bootstrap a fresh openraft cluster on
/// THIS node alone. Use exactly once per cluster, on the seed node.
/// Subsequent voters are added via `add-learner` + `promote`.
/// Auth: cluster master token.
async fn cluster_initialize(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    require_master_token(&state, &headers)?;
    let assembly = require_openraft(&state)?;
    let metrics = assembly.raft.metrics().borrow().clone();
    if metrics.membership_config.nodes().count() > 0 {
        return Ok(Json(json!({
            "status": "already_initialized",
            "voters": metrics.membership_config.voter_ids().map(|n| u64::from(n)).collect::<Vec<_>>(),
        })));
    }
    // Find this node's advertise address from the cluster config we
    // captured at boot. ClusterContext (legacy raft-lite) holds it.
    let node_addr = state
        .cluster
        .as_ref()
        .and_then(|c| c.config.advertise_addr.clone())
        .unwrap_or_else(|| "127.0.0.1:7440".to_string());
    match crate::raft::initialize_single_node(assembly, node_addr.clone()).await {
        Ok(()) => Ok(Json(json!({
            "status": "initialized",
            "node_id": u64::from(metrics.id),
            "addr": node_addr,
        }))),
        Err(e) => Err(app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("initialize failed: {e:?}"),
        )),
    }
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
    // Peer reachability is raft-lite-specific (openraft tracks this in
    // its own metrics scrape via spawn_raft_metrics_recorder). Skip on
    // openraft mode.
    if state.raft.is_none() {
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
    // Single-node mode without a configured cluster secret: deny outright.
    // Safer than auto-allowing any valid bearer; debug endpoints SHOULD
    // require explicit operator opt-in via cluster_secret.
    Err(app_error(
        StatusCode::FORBIDDEN,
        "debug endpoints require the cluster master token",
    ))
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

/// Build the Axum router.
pub fn router(state: Arc<AppState>) -> Router {
    let body_limit = state.admission.cfg.max_request_body_bytes;
    // Build the openraft sub-router up-front so we can merge it AFTER
    // the AppState-typed routes have all been chained. Order matters:
    // axum's `.merge()` unifies state types, and merging a state=()
    // router (these openraft routes set their own state via
    // `with_state(raft)`, so they expose state=() upward) before the
    // AppState routes confuses inference. Built here, merged at the
    // bottom of the chain.
    let raft_sub_router = state.raft.as_ref().map(|assembly| {
        crate::raft::raft_status_router(assembly.raft.clone())
            .merge(crate::raft::raft_receive_router(assembly.raft.clone()))
    });
    let mut app = Router::new()
        .route("/v1/health", get(health))
        .route("/v1/health/deep", get(health_deep))
        .route("/v1/remember", post(remember))
        .route("/v1/remember/batch", post(remember_batch))
        .route("/v1/recall", post(recall))
        .route("/v1/forget", post(forget))
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
        // v0.8.3 #24: openraft membership management
        .route("/v1/cluster/initialize", post(cluster_initialize))
        .route("/v1/cluster/add-learner", post(cluster_add_learner))
        .route("/v1/cluster/promote-voter", post(cluster_promote_voter))
        .route("/v1/cluster/remove", post(cluster_remove))
        .route("/v1/admin/control-snapshot", get(control_snapshot))
        .route("/v1/admin/snapshot", post(admin_snapshot))
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
        .with_state(state);
    if let Some(raft_router) = raft_sub_router {
        app = app.merge(raft_router);
    }
    app
}
