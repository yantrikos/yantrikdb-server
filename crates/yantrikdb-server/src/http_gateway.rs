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
///
/// Load shedding: if the inflight count exceeds MAX_INFLIGHT, reject with
/// 503 immediately instead of queuing. Better to fail fast than pile up.
async fn execute_cmd(
    engine: Arc<parking_lot::Mutex<yantrikdb::YantrikDB>>,
    cmd: Command,
    control: Arc<parking_lot::Mutex<crate::control::ControlDb>>,
    inflight: &std::sync::atomic::AtomicU32,
) -> AppResult {
    use std::sync::atomic::Ordering;

    // Load shed: reject if too many ops in flight
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
    });

    inflight.fetch_sub(1, Ordering::Relaxed);

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

    // Quick check via engine stats (no lock held across the check — we
    // take a snapshot then drop).
    let current = engine
        .try_lock()
        .and_then(|db| db.stats(None).ok())
        .map(|s| s.active_memories)
        .unwrap_or(0);

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
    let (db_id, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;

    // Quota check: max_memories
    check_memory_quota(&state, db_id, &engine, 1)?;

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
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
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

    let cmd = Command::RememberBatch { memories };
    execute_cmd(engine, cmd, state.control.clone(), &state.inflight).await
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

/// POST /v1/admin/snapshot — create an online backup of a tenant database.
///
/// Takes a consistent snapshot by WAL-checkpointing then copying the SQLite
/// file while holding the engine lock. Returns the backup path + BLAKE3
/// checksum. Authenticated by cluster master token.
///
/// Body: `{"database": "default", "output_dir": "/tmp/backups"}` (optional
/// output_dir, defaults to data_dir/snapshots/).
async fn admin_snapshot(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
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
            "snapshot requires cluster master token",
        ));
    }

    let db_name = body
        .get("database")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();

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
        let db = engine.lock();

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
    Router::new()
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
        .route("/v1/admin/control-snapshot", get(control_snapshot))
        .route("/v1/admin/snapshot", post(admin_snapshot))
        .route("/metrics", get(metrics))
        .with_state(state)
}
