//! HTTP/JSON gateway on port 7438.
//!
//! Thin translation layer: JSON → Command → handler → JSON response.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post, delete},
};
use serde_json::{json, Value};

use crate::command::Command;
use crate::handler::{self, CommandResult};
use crate::server::AppState;
use crate::auth;

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

    let token_hash = auth::hash_token(token);
    let control = state.control.lock().unwrap();
    let db_id = control.validate_token(&token_hash)
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or_else(|| app_error(StatusCode::UNAUTHORIZED, "invalid or revoked token"))?;

    let db_record = control.get_database_by_id(db_id)
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or_else(|| app_error(StatusCode::NOT_FOUND, "database not found"))?;
    drop(control);

    let engine = state.pool.get_engine(&db_record)
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

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
    Json(json!({
        "status": "ok",
        "engines_loaded": state.pool.loaded_count(),
    }))
}

async fn remember(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let (_, engine) = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
    let cmd = Command::Remember {
        text: body["text"].as_str().ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'text'"))?.into(),
        memory_type: body.get("memory_type").and_then(|v| v.as_str()).unwrap_or("semantic").into(),
        importance: body.get("importance").and_then(|v| v.as_f64()).unwrap_or(0.5),
        valence: body.get("valence").and_then(|v| v.as_f64()).unwrap_or(0.0),
        half_life: body.get("half_life").and_then(|v| v.as_f64()).unwrap_or(168.0),
        metadata: body.get("metadata").cloned().unwrap_or(json!({})),
        namespace: body.get("namespace").and_then(|v| v.as_str()).unwrap_or("").into(),
        certainty: body.get("certainty").and_then(|v| v.as_f64()).unwrap_or(1.0),
        domain: body.get("domain").and_then(|v| v.as_str()).unwrap_or("").into(),
        source: body.get("source").and_then(|v| v.as_str()).unwrap_or("user").into(),
        emotional_state: body.get("emotional_state").and_then(|v| v.as_str()).map(String::from),
        embedding: body.get("embedding").and_then(|v| {
            v.as_array().map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
        }),
    };
    execute_cmd(&engine, cmd, &state.control)
}

async fn recall(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let (_, engine) = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
    let cmd = Command::Recall {
        query: body["query"].as_str().ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'query'"))?.into(),
        top_k: body.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize,
        memory_type: body.get("memory_type").and_then(|v| v.as_str()).map(String::from),
        include_consolidated: body.get("include_consolidated").and_then(|v| v.as_bool()).unwrap_or(false),
        expand_entities: body.get("expand_entities").and_then(|v| v.as_bool()).unwrap_or(true),
        namespace: body.get("namespace").and_then(|v| v.as_str()).map(String::from),
        domain: body.get("domain").and_then(|v| v.as_str()).map(String::from),
        source: body.get("source").and_then(|v| v.as_str()).map(String::from),
        query_embedding: body.get("query_embedding").and_then(|v| {
            v.as_array().map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
        }),
    };
    execute_cmd(&engine, cmd, &state.control)
}

async fn forget(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let (_, engine) = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
    let rid = body["rid"].as_str().ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'rid'"))?.into();
    execute_cmd(&engine, Command::Forget { rid }, &state.control)
}

async fn relate(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let (_, engine) = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
    let cmd = Command::Relate {
        entity: body["entity"].as_str().ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'entity'"))?.into(),
        target: body["target"].as_str().ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'target'"))?.into(),
        relationship: body["relationship"].as_str().ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'relationship'"))?.into(),
        weight: body.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0),
    };
    execute_cmd(&engine, cmd, &state.control)
}

async fn think(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let (_, engine) = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
    let cmd = Command::Think {
        run_consolidation: body.get("run_consolidation").and_then(|v| v.as_bool()).unwrap_or(true),
        run_conflict_scan: body.get("run_conflict_scan").and_then(|v| v.as_bool()).unwrap_or(true),
        run_pattern_mining: body.get("run_pattern_mining").and_then(|v| v.as_bool()).unwrap_or(false),
        run_personality: body.get("run_personality").and_then(|v| v.as_bool()).unwrap_or(false),
        consolidation_limit: body.get("consolidation_limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize,
    };
    execute_cmd(&engine, cmd, &state.control)
}

async fn conflicts(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    let (_, engine) = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
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
    let (_, engine) = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
    let cmd = Command::Resolve {
        conflict_id,
        strategy: body["strategy"].as_str().ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'strategy'"))?.into(),
        winner_rid: body.get("winner_rid").and_then(|v| v.as_str()).map(String::from),
        new_text: body.get("new_text").and_then(|v| v.as_str()).map(String::from),
        resolution_note: body.get("resolution_note").and_then(|v| v.as_str()).map(String::from),
    };
    execute_cmd(&engine, cmd, &state.control)
}

async fn session_start(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    let (_, engine) = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
    let cmd = Command::SessionStart {
        namespace: body.get("namespace").and_then(|v| v.as_str()).unwrap_or("default").into(),
        client_id: body.get("client_id").and_then(|v| v.as_str()).unwrap_or("").into(),
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
    let (_, engine) = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
    let summary = body.and_then(|Json(b)| b.get("summary").and_then(|v| v.as_str()).map(String::from));
    let cmd = Command::SessionEnd { session_id, summary };
    execute_cmd(&engine, cmd, &state.control)
}

async fn personality(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    let (_, engine) = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
    execute_cmd(&engine, Command::Personality, &state.control)
}

async fn stats(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    let (_, engine) = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
    execute_cmd(&engine, Command::Stats, &state.control)
}

async fn create_database(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> AppResult {
    // For now, any valid token can create databases
    let _ = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
    let name: String = body["name"].as_str().ok_or_else(|| app_error(StatusCode::BAD_REQUEST, "missing 'name'"))?.to_string();

    // Create directly via control (no engine needed)
    let control = state.control.lock().unwrap();
    if control.database_exists(&name).map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))? {
        return Err(app_error(StatusCode::CONFLICT, format!("database '{}' already exists", name)));
    }
    let id = control.create_database(&name, &name)
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    drop(control);

    // Create the data directory
    let db_dir = state.pool.data_dir().join(&name);
    std::fs::create_dir_all(&db_dir).map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(json!({
        "name": name,
        "id": id,
        "message": format!("database '{}' created", name),
    })))
}

async fn list_databases(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> AppResult {
    let _ = resolve_engine(&state, headers.get("authorization").and_then(|v| v.to_str().ok()))?;
    let databases = state.control.lock().unwrap().list_databases()
        .map_err(|e| app_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let list: Vec<Value> = databases.iter().map(|d| {
        json!({ "id": d.id, "name": d.name, "created_at": d.created_at })
    }).collect();
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
        .with_state(state)
}
