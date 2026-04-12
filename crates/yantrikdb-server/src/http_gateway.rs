//! HTTP/JSON gateway on port 7438.
//!
//! Thin translation layer: JSON → Command → handler → JSON response.

use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use serde_json::{json, Value};

use crate::auth;
use crate::command::Command;
use crate::handler::{self, CommandResult};
use crate::server::AppState;

type AppResult = Result<Json<Value>, (StatusCode, Json<Value>)>;

/// Shared engine handle. Type alias keeps the complex nested generic out
/// of function signatures and avoids clippy::type_complexity.
type EngineHandle = Arc<parking_lot::Mutex<yantrikdb::YantrikDB>>;

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
async fn execute_cmd(
    engine: Arc<parking_lot::Mutex<yantrikdb::YantrikDB>>,
    cmd: Command,
    control: Arc<parking_lot::Mutex<crate::control::ControlDb>>,
) -> AppResult {
    let result = tokio::task::spawn_blocking(move || {
        // Measure engine lock acquisition time for /metrics histograms
        let lock_start = std::time::Instant::now();
        let db = engine.lock();
        crate::metrics::record_engine_lock_wait(lock_start.elapsed());
        handler::execute_with_guard(db, cmd, Some(control.as_ref()))
    })
    .await
    .map_err(|e| {
        app_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("join error: {e}"),
        )
    })?;

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
    if let Some(ref cluster) = state.cluster {
        payload["cluster"] = json!({
            "node_id": cluster.node_id(),
            "role": format!("{:?}", cluster.state.leader_role()),
            "term": cluster.state.current_term(),
            "leader": cluster.state.current_leader(),
            "accepts_writes": cluster.state.accepts_writes(),
            "healthy": cluster.is_healthy(),
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
                        if engine.try_lock_for(timeout).is_some() {
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

    // 3. Cluster quorum (if clustered)
    if let Some(ref cluster) = state.cluster {
        let healthy = cluster.is_healthy();
        if !healthy {
            all_pass = false;
        }
        checks.push(json!({
            "check": "cluster_quorum",
            "pass": healthy,
            "node_id": cluster.node_id(),
            "role": format!("{:?}", cluster.state.leader_role()),
            "term": cluster.state.current_term(),
            "leader": cluster.state.current_leader(),
        }));
    }

    let status = if all_pass {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        status,
        Json(json!({
            "status": if all_pass { "healthy" } else { "degraded" },
            "checks": checks,
        })),
    )
}

/// Reject if cluster is enabled and this node doesn't accept writes.
fn check_writable(state: &AppState) -> Result<(), (StatusCode, Json<Value>)> {
    if let Some(ref cluster) = state.cluster {
        if !cluster.state.accepts_writes() {
            let leader = cluster.state.current_leader();
            let msg = match leader {
                Some(id) => format!("read-only: not the leader (current leader: node {})", id),
                None => "read-only: no leader elected".into(),
            };
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": msg, "leader": leader})),
            ));
        }
    }
    Ok(())
}

async fn remember(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("remember");
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::Remember {
        text: body["text"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'text'"))?
            .into(),
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
        embedding: body.get("embedding").and_then(|v| {
            v.as_array().map(|a| {
                a.iter()
                    .filter_map(|x| x.as_f64().map(|f| f as f32))
                    .collect()
            })
        }),
    };
    execute_cmd(engine, cmd, state.control.clone()).await
}

async fn remember_batch(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("remember_batch");
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
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

    let cmd = Command::RememberBatch { memories };
    execute_cmd(engine, cmd, state.control.clone()).await
}

async fn recall(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("recall");
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::Recall {
        query: body["query"]
            .as_str()
            .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'query'"))?
            .into(),
        top_k: body.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize,
        memory_type: body
            .get("memory_type")
            .and_then(|v| v.as_str())
            .map(String::from),
        include_consolidated: body
            .get("include_consolidated")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        expand_entities: body
            .get("expand_entities")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
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
    execute_cmd(engine, cmd, state.control.clone()).await
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
    execute_cmd(engine, Command::Forget { rid }, state.control.clone()).await
}

async fn relate(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
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
    execute_cmd(engine, cmd, state.control.clone()).await
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
            .unwrap_or(false),
        consolidation_limit: body
            .get("consolidation_limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize,
    };
    execute_cmd(engine, cmd, state.control.clone()).await
}

async fn conflicts(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("conflicts");
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let cmd = Command::Conflicts {
        status: None,
        conflict_type: None,
        entity: None,
        limit: 50,
    };
    execute_cmd(engine, cmd, state.control.clone()).await
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
    execute_cmd(engine, cmd, state.control.clone()).await
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
    execute_cmd(engine, cmd, state.control.clone()).await
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
    execute_cmd(engine, cmd, state.control.clone()).await
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
    execute_cmd(engine, Command::Personality, state.control.clone()).await
}

async fn stats(State(state): State<Arc<AppState>>, headers: axum::http::HeaderMap) -> AppResult {
    let _timer = crate::metrics::HandlerTimer::new("stats");
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    execute_cmd(engine, Command::Stats, state.control.clone()).await
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

    if let Some(ref cluster) = state.cluster {
        out.push_str("# HELP yantrikdb_cluster_term Current Raft term\n");
        out.push_str("# TYPE yantrikdb_cluster_term gauge\n");
        out.push_str(&format!(
            "yantrikdb_cluster_term {{node_id=\"{}\"}} {}\n",
            cluster.node_id(),
            cluster.state.current_term()
        ));

        out.push_str("# HELP yantrikdb_cluster_is_leader Whether this node is currently the leader (1) or not (0)\n");
        out.push_str("# TYPE yantrikdb_cluster_is_leader gauge\n");
        out.push_str(&format!(
            "yantrikdb_cluster_is_leader {{node_id=\"{}\"}} {}\n",
            cluster.node_id(),
            if cluster.state.is_leader() { 1 } else { 0 }
        ));

        out.push_str(
            "# HELP yantrikdb_cluster_healthy Whether this node has quorum (1) or not (0)\n",
        );
        out.push_str("# TYPE yantrikdb_cluster_healthy gauge\n");
        out.push_str(&format!(
            "yantrikdb_cluster_healthy {{node_id=\"{}\"}} {}\n",
            cluster.node_id(),
            if cluster.is_healthy() { 1 } else { 0 }
        ));

        out.push_str("# HELP yantrikdb_cluster_peer_reachable Whether each peer is reachable\n");
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
                // Use try_lock so a slow engine call (e.g. embedding generation)
                // can never wedge the metrics endpoint. Skip this scrape instead.
                // parking_lot::Mutex::try_lock returns Option<MutexGuard>.
                engine.try_lock().and_then(|db| db.stats(None).ok())
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

/// Build the Axum router.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/health/deep", get(health_deep))
        .route("/v1/remember", post(remember))
        .route("/v1/remember/batch", post(remember_batch))
        .route("/v1/recall", post(recall))
        .route("/v1/forget", post(forget))
        .route("/v1/relate", post(relate))
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
        .route("/v1/admin/control-snapshot", get(control_snapshot))
        .route("/metrics", get(metrics))
        .with_state(state)
}
