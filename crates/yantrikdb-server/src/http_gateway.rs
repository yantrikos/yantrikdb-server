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

fn app_error(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<Value>) {
    (status, Json(json!({ "error": message.into() })))
}

/// Extract database engine from Bearer token.
fn resolve_engine(
    state: &AppState,
    auth_header: Option<&str>,
) -> Result<(i64, Arc<std::sync::Mutex<yantrikdb::YantrikDB>>), (StatusCode, Json<Value>)> {
    let token = auth_header
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or_else(|| app_error(StatusCode::UNAUTHORIZED, "missing Bearer token"))?;

    // Cluster master token check
    if let Some(ref cluster) = state.cluster {
        if let Some(ref secret) = cluster.config.cluster_secret {
            if token == secret.as_str() {
                let control = state.control.lock().unwrap();
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
    let control = state.control.lock().unwrap();
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

fn execute_cmd(
    engine: &Arc<std::sync::Mutex<yantrikdb::YantrikDB>>,
    cmd: Command,
    control: &std::sync::Mutex<crate::control::ControlDb>,
) -> AppResult {
    match handler::execute(engine, cmd, Some(control)) {
        Ok(CommandResult::Json(v)) => Ok(Json(v)),
        Ok(CommandResult::RecallResults { results, total }) => {
            Ok(Json(json!({ "results": results, "total": total })))
        }
        Ok(CommandResult::Pong) => Ok(Json(json!({ "status": "ok" }))),
        Err(e) => Err(app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

// ── Route handlers ──────────────────────────────────────────────

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
    execute_cmd(&engine, cmd, &state.control)
}

async fn recall(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
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
    execute_cmd(&engine, cmd, &state.control)
}

async fn forget(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    check_writable(&state)?;
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    let rid = body["rid"]
        .as_str()
        .ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'rid'"))?
        .into();
    execute_cmd(&engine, Command::Forget { rid }, &state.control)
}

async fn relate(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
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
    execute_cmd(&engine, cmd, &state.control)
}

async fn think(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
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
    execute_cmd(&engine, cmd, &state.control)
}

async fn conflicts(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
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
    execute_cmd(&engine, cmd, &state.control)
}

async fn resolve_conflict(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    AxumPath(conflict_id): AxumPath<String>,
    Json(body): Json<Value>,
) -> AppResult {
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
    execute_cmd(&engine, cmd, &state.control)
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
    execute_cmd(&engine, cmd, &state.control)
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
    execute_cmd(&engine, cmd, &state.control)
}

async fn personality(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    execute_cmd(&engine, Command::Personality, &state.control)
}

async fn stats(State(state): State<Arc<AppState>>, headers: axum::http::HeaderMap) -> AppResult {
    let (_, engine) = resolve_engine(
        &state,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )?;
    execute_cmd(&engine, Command::Stats, &state.control)
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
    let control = state.control.lock().unwrap();
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

        out.push_str("# HELP yantrikdb_cluster_healthy Whether this node has quorum (1) or not (0)\n");
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

    // Per-database stats (default DB only for now)
    if let Ok(db_record) = state.control.lock().unwrap().get_database("default") {
        if let Some(rec) = db_record {
            if let Ok(engine) = state.pool.get_engine(&rec) {
                if let Ok(stats) = engine.lock().unwrap().stats(None) {
                    out.push_str("# HELP yantrikdb_active_memories Number of active memories\n");
                    out.push_str("# TYPE yantrikdb_active_memories gauge\n");
                    out.push_str(&format!(
                        "yantrikdb_active_memories {{db=\"default\"}} {}\n",
                        stats.active_memories
                    ));

                    out.push_str("# HELP yantrikdb_consolidated_memories Number of consolidated memories\n");
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

                    out.push_str("# HELP yantrikdb_open_conflicts Number of unresolved conflicts\n");
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
        .unwrap()
        .list_databases()
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let list: Vec<Value> = databases
        .iter()
        .map(|d| json!({ "id": d.id, "name": d.name, "created_at": d.created_at }))
        .collect();
    Ok(Json(json!({ "databases": list })))
}

/// Build the Axum router.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/remember", post(remember))
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
        .route("/metrics", get(metrics))
        .with_state(state)
}
