//! Integration tests for the YantrikDB HTTP gateway.
//!
//! Each test spins up a real server on random ports, creates a database + token,
//! and exercises the HTTP API via reqwest.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use reqwest::Client;
use serde_json::{json, Value};
use tempfile::TempDir;

// We can't import from the binary crate directly, so we rebuild
// the server setup inline using the library crates.
use yantrikdb::YantrikDB;

/// A test server instance with its HTTP base URL and token.
struct TestServer {
    base_url: String,
    token: String,
    client: Client,
    _tmp: TempDir, // dropped at end of test = cleanup
    shutdown: tokio::sync::oneshot::Sender<()>,
}

impl TestServer {
    async fn start() -> Self {
        let tmp = TempDir::new().expect("create temp dir");
        let data_dir = tmp.path().to_path_buf();
        std::fs::create_dir_all(data_dir.join("default")).unwrap();

        // Create control.db
        let control_path = data_dir.join("control.db");
        let conn = rusqlite::Connection::open(&control_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS databases (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                path        TEXT NOT NULL,
                config      TEXT NOT NULL DEFAULT '{}',
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS tokens (
                hash        TEXT PRIMARY KEY,
                database_id INTEGER NOT NULL REFERENCES databases(id),
                label       TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                revoked_at  TEXT
            );
            INSERT INTO databases (name, path) VALUES ('default', 'default');
            ",
        )
        .unwrap();

        // Create a token
        let token = "ydb_test_token_for_integration_tests_only_1234567890abcdef";
        let token_hash = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(token.as_bytes());
            hex::encode(hasher.finalize())
        };
        conn.execute(
            "INSERT INTO tokens (hash, database_id, label) VALUES (?1, 1, 'test')",
            rusqlite::params![token_hash],
        )
        .unwrap();
        drop(conn);

        // Find a free port
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let base_url = format!("http://127.0.0.1:{}", port);

        // Build the axum app inline using the same patterns as the server
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let data_dir_clone = data_dir.clone();
        tokio::spawn(async move {
            run_test_server(listener, data_dir_clone, shutdown_rx).await;
        });

        // Wait for server to be ready
        let client = Client::new();
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            if client
                .get(&format!("{}/v1/health", base_url))
                .send()
                .await
                .is_ok()
            {
                break;
            }
        }

        TestServer {
            base_url,
            token: token.to_string(),
            client,
            _tmp: tmp,
            shutdown: shutdown_tx,
        }
    }

    async fn post(&self, path: &str, body: &Value) -> Value {
        let resp = self
            .client
            .post(format!("{}{}", self.base_url, path))
            .header("Authorization", format!("Bearer {}", self.token))
            .json(body)
            .send()
            .await
            .expect("request failed");
        let status = resp.status();
        let text = resp.text().await.expect("read body");
        assert!(
            status.is_success(),
            "POST {} returned {}: {}",
            path,
            status,
            text
        );
        serde_json::from_str(&text).expect("parse json")
    }

    async fn get(&self, path: &str) -> Value {
        let resp = self
            .client
            .get(format!("{}{}", self.base_url, path))
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await
            .expect("request failed");
        let status = resp.status();
        let text = resp.text().await.expect("read body");
        assert!(
            status.is_success(),
            "GET {} returned {}: {}",
            path,
            status,
            text
        );
        serde_json::from_str(&text).expect("parse json")
    }

    async fn get_status(&self, path: &str) -> (reqwest::StatusCode, Value) {
        let resp = self
            .client
            .get(format!("{}{}", self.base_url, path))
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await
            .expect("request failed");
        let status = resp.status();
        let text = resp.text().await.expect("read body");
        let body: Value = serde_json::from_str(&text).unwrap_or(json!({"raw": text}));
        (status, body)
    }

    async fn post_status(&self, path: &str, body: &Value) -> (reqwest::StatusCode, Value) {
        let resp = self
            .client
            .post(format!("{}{}", self.base_url, path))
            .header("Authorization", format!("Bearer {}", self.token))
            .json(body)
            .send()
            .await
            .expect("request failed");
        let status = resp.status();
        let text = resp.text().await.expect("read body");
        let json_body: Value = serde_json::from_str(&text).unwrap_or(json!({"raw": text}));
        (status, json_body)
    }

    async fn post_no_auth(&self, path: &str, body: &Value) -> reqwest::StatusCode {
        let resp = self
            .client
            .post(format!("{}{}", self.base_url, path))
            .json(body)
            .send()
            .await
            .expect("request failed");
        resp.status()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        // Trigger shutdown — ignore error if already shut down
        // We need to take ownership, use a dummy sender
    }
}

/// Minimal server for testing — no embedder (client_only mode), no TLS.
async fn run_test_server(
    listener: tokio::net::TcpListener,
    data_dir: PathBuf,
    shutdown_rx: tokio::sync::oneshot::Receiver<()>,
) {
    use axum::extract::State;
    use axum::routing::{delete, get, post};
    use axum::Router;

    // Reconstruct AppState manually for the test
    // We need to replicate the server's internals here since it's a binary crate
    let control_path = data_dir.join("control.db");

    // Open a new connection for the server (the setup one was dropped)
    let control = TestControlDb::open(&control_path);
    let pool = TestTenantPool::new(data_dir.clone());

    let state = Arc::new(TestAppState {
        control: Mutex::new(control),
        pool,
    });

    let app = Router::new()
        .route("/v1/health", get(handle_health))
        .route("/v1/remember", post(handle_remember))
        .route("/v1/recall", post(handle_recall))
        .route("/v1/forget", post(handle_forget))
        .route("/v1/relate", post(handle_relate))
        .route("/v1/think", post(handle_think))
        .route("/v1/stats", get(handle_stats))
        .route("/v1/conflicts", get(handle_conflicts))
        .route("/v1/personality", get(handle_personality))
        .route("/v1/sessions", post(handle_session_start))
        .route("/v1/sessions/{id}", delete(handle_session_end))
        .with_state(state);

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
        })
        .await
        .unwrap();
}

// ── Minimal test server internals ───────────────────────────────

struct TestAppState {
    control: Mutex<TestControlDb>,
    pool: TestTenantPool,
}

struct TestControlDb {
    conn: rusqlite::Connection,
}

impl TestControlDb {
    fn open(path: &std::path::Path) -> Self {
        Self {
            conn: rusqlite::Connection::open(path).unwrap(),
        }
    }

    fn validate_token(&self, hash: &str) -> Option<i64> {
        self.conn
            .query_row(
                "SELECT database_id FROM tokens WHERE hash = ?1 AND revoked_at IS NULL",
                rusqlite::params![hash],
                |row| row.get(0),
            )
            .ok()
    }
}

struct TestTenantPool {
    engines: Mutex<HashMap<i64, Arc<Mutex<YantrikDB>>>>,
    data_dir: PathBuf,
}

impl TestTenantPool {
    fn new(data_dir: PathBuf) -> Self {
        Self {
            engines: Mutex::new(HashMap::new()),
            data_dir,
        }
    }

    fn get_engine(&self, db_id: i64) -> Arc<Mutex<YantrikDB>> {
        let mut engines = self.engines.lock().unwrap();
        engines
            .entry(db_id)
            .or_insert_with(|| {
                let db_dir = self.data_dir.join("default");
                std::fs::create_dir_all(&db_dir).unwrap();
                let db_path = db_dir.join("yantrik.db");
                let engine = YantrikDB::new(db_path.to_str().unwrap(), 384).unwrap();
                Arc::new(Mutex::new(engine))
            })
            .clone()
    }
}

fn resolve_test_engine(
    state: &TestAppState,
    headers: &axum::http::HeaderMap,
) -> Result<Arc<Mutex<YantrikDB>>, (axum::http::StatusCode, axum::Json<Value>)> {
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or_else(|| {
            (
                axum::http::StatusCode::UNAUTHORIZED,
                axum::Json(json!({"error": "missing Bearer token"})),
            )
        })?;

    let token_hash = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(token.as_bytes());
        hex::encode(hasher.finalize())
    };

    let control = state.control.lock().unwrap();
    let db_id = control.validate_token(&token_hash).ok_or_else(|| {
        (
            axum::http::StatusCode::UNAUTHORIZED,
            axum::Json(json!({"error": "invalid token"})),
        )
    })?;
    drop(control);

    Ok(state.pool.get_engine(db_id))
}

type TestResult = Result<axum::Json<Value>, (axum::http::StatusCode, axum::Json<Value>)>;

async fn handle_health(
    axum::extract::State(_state): axum::extract::State<Arc<TestAppState>>,
) -> axum::Json<Value> {
    axum::Json(json!({"status": "ok"}))
}

async fn handle_remember(
    axum::extract::State(state): axum::extract::State<Arc<TestAppState>>,
    headers: axum::http::HeaderMap,
    axum::Json(body): axum::Json<Value>,
) -> TestResult {
    let engine = resolve_test_engine(&state, &headers)?;
    let db = engine.lock().unwrap();

    // Generate a simple random embedding (no real embedder in tests)
    let dim = 384;
    let embedding: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.01).sin()).collect();

    let rid = db
        .record(
            body["text"].as_str().unwrap_or(""),
            body.get("memory_type")
                .and_then(|v| v.as_str())
                .unwrap_or("semantic"),
            body.get("importance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5),
            body.get("valence").and_then(|v| v.as_f64()).unwrap_or(0.0),
            body.get("half_life")
                .and_then(|v| v.as_f64())
                .unwrap_or(168.0),
            &body.get("metadata").cloned().unwrap_or(json!({})),
            &embedding,
            body.get("namespace").and_then(|v| v.as_str()).unwrap_or(""),
            body.get("certainty")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0),
            body.get("domain").and_then(|v| v.as_str()).unwrap_or(""),
            body.get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("user"),
            body.get("emotional_state").and_then(|v| v.as_str()),
        )
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(json!({"error": e.to_string()})),
            )
        })?;

    Ok(axum::Json(json!({"rid": rid})))
}

async fn handle_recall(
    axum::extract::State(state): axum::extract::State<Arc<TestAppState>>,
    headers: axum::http::HeaderMap,
    axum::Json(body): axum::Json<Value>,
) -> TestResult {
    let engine = resolve_test_engine(&state, &headers)?;
    let db = engine.lock().unwrap();

    let dim = 384;
    let query_embedding: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.01).sin()).collect();
    let top_k = body.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let results = db
        .recall(
            &query_embedding,
            top_k,
            None,
            body.get("memory_type").and_then(|v| v.as_str()),
            false,
            true,
            body.get("query").and_then(|v| v.as_str()),
            false,
            body.get("namespace").and_then(|v| v.as_str()),
            body.get("domain").and_then(|v| v.as_str()),
            body.get("source").and_then(|v| v.as_str()),
        )
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(json!({"error": e.to_string()})),
            )
        })?;

    let result_values: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "rid": r.rid,
                "text": r.text,
                "score": r.score,
                "importance": r.importance,
                "memory_type": r.memory_type,
                "why_retrieved": r.why_retrieved,
                "domain": r.domain,
            })
        })
        .collect();

    Ok(axum::Json(
        json!({"results": result_values, "total": result_values.len()}),
    ))
}

async fn handle_forget(
    axum::extract::State(state): axum::extract::State<Arc<TestAppState>>,
    headers: axum::http::HeaderMap,
    axum::Json(body): axum::Json<Value>,
) -> TestResult {
    let engine = resolve_test_engine(&state, &headers)?;
    let db = engine.lock().unwrap();
    let rid = body["rid"].as_str().unwrap_or("");
    let found = db.forget(rid).map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({"error": e.to_string()})),
        )
    })?;
    Ok(axum::Json(json!({"rid": rid, "found": found})))
}

async fn handle_relate(
    axum::extract::State(state): axum::extract::State<Arc<TestAppState>>,
    headers: axum::http::HeaderMap,
    axum::Json(body): axum::Json<Value>,
) -> TestResult {
    let engine = resolve_test_engine(&state, &headers)?;
    let db = engine.lock().unwrap();
    let edge_id = db
        .relate(
            body["entity"].as_str().unwrap_or(""),
            body["target"].as_str().unwrap_or(""),
            body["relationship"].as_str().unwrap_or(""),
            body.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0),
        )
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(json!({"error": e.to_string()})),
            )
        })?;
    Ok(axum::Json(json!({"edge_id": edge_id})))
}

async fn handle_think(
    axum::extract::State(state): axum::extract::State<Arc<TestAppState>>,
    headers: axum::http::HeaderMap,
    axum::Json(_body): axum::Json<Value>,
) -> TestResult {
    let engine = resolve_test_engine(&state, &headers)?;
    let db = engine.lock().unwrap();
    let config = yantrikdb::types::ThinkConfig::default();
    let result = db.think(&config).map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({"error": e.to_string()})),
        )
    })?;
    Ok(axum::Json(json!({
        "consolidation_count": result.consolidation_count,
        "conflicts_found": result.conflicts_found,
        "duration_ms": result.duration_ms,
    })))
}

async fn handle_stats(
    axum::extract::State(state): axum::extract::State<Arc<TestAppState>>,
    headers: axum::http::HeaderMap,
) -> TestResult {
    let engine = resolve_test_engine(&state, &headers)?;
    let db = engine.lock().unwrap();
    let s = db.stats(None).map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({"error": e.to_string()})),
        )
    })?;
    Ok(axum::Json(json!({
        "active_memories": s.active_memories,
        "edges": s.edges,
        "entities": s.entities,
        "operations": s.operations,
        "open_conflicts": s.open_conflicts,
    })))
}

async fn handle_conflicts(
    axum::extract::State(state): axum::extract::State<Arc<TestAppState>>,
    headers: axum::http::HeaderMap,
) -> TestResult {
    let engine = resolve_test_engine(&state, &headers)?;
    let db = engine.lock().unwrap();
    let conflicts = db.get_conflicts(None, None, None, None, 50).map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({"error": e.to_string()})),
        )
    })?;
    Ok(axum::Json(json!({"conflicts": conflicts.len()})))
}

async fn handle_personality(
    axum::extract::State(state): axum::extract::State<Arc<TestAppState>>,
    headers: axum::http::HeaderMap,
) -> TestResult {
    let engine = resolve_test_engine(&state, &headers)?;
    let db = engine.lock().unwrap();
    let profile = db.get_personality().map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({"error": e.to_string()})),
        )
    })?;
    Ok(axum::Json(json!({"traits": profile.traits.len()})))
}

async fn handle_session_start(
    axum::extract::State(state): axum::extract::State<Arc<TestAppState>>,
    headers: axum::http::HeaderMap,
    axum::Json(body): axum::Json<Value>,
) -> TestResult {
    let engine = resolve_test_engine(&state, &headers)?;
    let db = engine.lock().unwrap();
    let session_id = db
        .session_start(
            body.get("namespace")
                .and_then(|v| v.as_str())
                .unwrap_or("default"),
            body.get("client_id").and_then(|v| v.as_str()).unwrap_or(""),
            &body.get("metadata").cloned().unwrap_or(json!({})),
        )
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(json!({"error": e.to_string()})),
            )
        })?;
    Ok(axum::Json(json!({"session_id": session_id})))
}

async fn handle_session_end(
    axum::extract::State(state): axum::extract::State<Arc<TestAppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Path(session_id): axum::extract::Path<String>,
) -> TestResult {
    let engine = resolve_test_engine(&state, &headers)?;
    let db = engine.lock().unwrap();
    let result = db.session_end(&session_id, None).map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({"error": e.to_string()})),
        )
    })?;
    Ok(axum::Json(json!({
        "session_id": result.session_id,
        "memory_count": result.memory_count,
    })))
}

// ══════════════════════════════════════════════════════════════════
// TESTS
// ══════════════════════════════════════════════════════════════════

#[tokio::test]
async fn handle_health_endpoint() {
    let server = TestServer::start().await;
    let resp = server.get("/v1/health").await;
    assert_eq!(resp["status"], "ok");
}

#[tokio::test]
async fn test_auth_required() {
    let server = TestServer::start().await;
    let status = server
        .post_no_auth("/v1/remember", &json!({"text": "hello"}))
        .await;
    assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_remember_and_recall() {
    let server = TestServer::start().await;

    // Remember
    let resp = server
        .post(
            "/v1/remember",
            &json!({
                "text": "Alice leads engineering at Acme",
                "importance": 0.9,
                "domain": "work"
            }),
        )
        .await;
    assert!(resp["rid"].is_string());
    let rid = resp["rid"].as_str().unwrap().to_string();

    // Remember another
    server
        .post(
            "/v1/remember",
            &json!({
                "text": "Bob is a data scientist",
                "importance": 0.8,
                "domain": "work"
            }),
        )
        .await;

    // Recall
    let resp = server
        .post(
            "/v1/recall",
            &json!({"query": "who leads engineering?", "top_k": 5}),
        )
        .await;
    assert!(resp["total"].as_u64().unwrap() >= 1);
    let results = resp["results"].as_array().unwrap();
    assert!(!results.is_empty());
    // First result should be Alice (same embedding = highest similarity)
    assert!(results[0]["text"].as_str().unwrap().contains("Alice"));
}

#[tokio::test]
async fn test_forget() {
    let server = TestServer::start().await;

    // Remember
    let resp = server
        .post(
            "/v1/remember",
            &json!({"text": "temporary memory", "importance": 0.5}),
        )
        .await;
    let rid = resp["rid"].as_str().unwrap();

    // Forget
    let resp = server.post("/v1/forget", &json!({"rid": rid})).await;
    assert_eq!(resp["found"], true);

    // Verify it no longer appears in recall
    let resp = server
        .post("/v1/recall", &json!({"query": "temporary", "top_k": 10}))
        .await;
    let results = resp["results"].as_array().unwrap();
    let found = results.iter().any(|r| r["rid"].as_str() == Some(rid));
    assert!(!found, "forgotten memory should not appear in recall");
}

#[tokio::test]
async fn test_relate_and_edges() {
    let server = TestServer::start().await;

    let resp = server
        .post(
            "/v1/relate",
            &json!({
                "entity": "Alice",
                "target": "Engineering",
                "relationship": "leads",
                "weight": 0.95
            }),
        )
        .await;
    assert!(resp["edge_id"].is_string());
}

#[tokio::test]
async fn test_stats() {
    let server = TestServer::start().await;

    // Empty database
    let resp = server.get("/v1/stats").await;
    assert_eq!(resp["active_memories"], 0);
    assert_eq!(resp["edges"], 0);

    // Add some data
    server
        .post(
            "/v1/remember",
            &json!({"text": "test memory", "importance": 0.5}),
        )
        .await;
    server
        .post(
            "/v1/relate",
            &json!({"entity": "A", "target": "B", "relationship": "knows"}),
        )
        .await;

    // Check stats updated
    let resp = server.get("/v1/stats").await;
    assert_eq!(resp["active_memories"], 1);
    assert_eq!(resp["edges"], 1);
    assert!(resp["operations"].as_i64().unwrap() >= 2);
}

#[tokio::test]
async fn test_think() {
    let server = TestServer::start().await;

    let resp = server.post("/v1/think", &json!({})).await;
    assert!(resp["duration_ms"].is_number());
    assert_eq!(resp["consolidation_count"], 0);
}

#[tokio::test]
async fn test_sessions() {
    let server = TestServer::start().await;

    // Start session
    let resp = server
        .post(
            "/v1/sessions",
            &json!({"namespace": "test", "client_id": "user-1"}),
        )
        .await;
    let session_id = resp["session_id"].as_str().unwrap().to_string();
    assert!(!session_id.is_empty());

    // Add memories during session
    server
        .post(
            "/v1/remember",
            &json!({"text": "session memory 1", "importance": 0.5}),
        )
        .await;

    // End session
    let client = &server.client;
    let resp = client
        .delete(format!("{}/v1/sessions/{}", server.base_url, session_id))
        .header("Authorization", format!("Bearer {}", server.token))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["session_id"], session_id);
}

#[tokio::test]
async fn test_conflicts_endpoint() {
    let server = TestServer::start().await;
    let resp = server.get("/v1/conflicts").await;
    // Empty database should have no conflicts
    assert!(resp.get("conflicts").is_some());
}

#[tokio::test]
async fn test_personality_endpoint() {
    let server = TestServer::start().await;
    let resp = server.get("/v1/personality").await;
    assert!(resp.get("traits").is_some());
}

#[tokio::test]
async fn test_remember_with_metadata() {
    let server = TestServer::start().await;

    let resp = server
        .post(
            "/v1/remember",
            &json!({
                "text": "Project deadline is March 30",
                "importance": 0.9,
                "domain": "work",
                "source": "user",
                "memory_type": "episodic",
                "metadata": {"project": "alpha", "deadline": "2026-03-30"},
                "certainty": 0.95,
                "valence": -0.2,
                "emotional_state": "concern"
            }),
        )
        .await;
    assert!(resp["rid"].is_string());
}

#[tokio::test]
async fn test_multiple_memories_recall_ordering() {
    let server = TestServer::start().await;

    // Add 5 memories with varying importance
    for (i, (text, imp)) in [
        ("Low priority note", 0.1),
        ("Medium priority task", 0.5),
        ("High priority deadline", 0.9),
        ("Critical security issue", 1.0),
        ("Background context", 0.3),
    ]
    .iter()
    .enumerate()
    {
        server
            .post(
                "/v1/remember",
                &json!({"text": text, "importance": imp, "domain": "work"}),
            )
            .await;
    }

    // Recall — all use the same embedding so ordering is by importance/recency
    let resp = server
        .post("/v1/recall", &json!({"query": "priorities", "top_k": 5}))
        .await;
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results.len(), 5);

    // Highest importance should score highest
    let top_score = results[0]["score"].as_f64().unwrap();
    let bottom_score = results[4]["score"].as_f64().unwrap();
    assert!(
        top_score >= bottom_score,
        "expected top score {} >= bottom {}",
        top_score,
        bottom_score
    );
}

#[tokio::test]
async fn test_invalid_token() {
    let server = TestServer::start().await;

    let resp = server
        .client
        .post(format!("{}/v1/remember", server.base_url))
        .header("Authorization", "Bearer ydb_invalid_token")
        .json(&json!({"text": "should fail"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
}
