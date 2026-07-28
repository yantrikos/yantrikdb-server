//! Test harness for multi-node YRP clusters over real localhost HTTP —
//! the chaos gate's substrate (saga 237). Nodes are KILLABLE and
//! RESTARTABLE against the same data dir, which is what makes
//! crash/recovery scenarios meaningful.

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::auth::ControlDbAuthProvider;
use crate::control::ControlDb;
use crate::server::AppState;
use crate::yrp::runtime::{spawn as spawn_yrp, YrpCommitter, YrpHandle, YrpPeer, YrpRuntimeConfig};

pub const SECRET: &str = "yrp-chaos-cluster-secret";
pub const TENANT: &str = "yrpchaos";

/// Serializes the heavy multi-node tests (chaos, 2-node cluster, the
/// 3-driver channel test). Each spawns real servers with 10-20ms tick
/// timers; running several such multi-threaded runtimes concurrently in
/// one test binary starves tickers and flakes elections. Hold this for
/// the duration of any multi-node test.
pub async fn serial_guard() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::LazyLock<tokio::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));
    LOCK.lock().await
}

/// Per-cluster tuning shared by every node.
#[derive(Debug, Clone, Copy)]
pub struct ClusterSpec {
    pub compact_after_entries: u64,
    pub leader_retain_entries: u64,
}

impl Default for ClusterSpec {
    fn default() -> Self {
        Self {
            compact_after_entries: 0,
            leader_retain_entries: 0,
        }
    }
}

/// A live node. Kill it to get a [`DeadNode`]; restart that to get a
/// live node again on the same data dir + port.
pub struct TestNode {
    pub node_id: u64,
    pub base: String,
    pub token: String,
    pub state: Arc<AppState>,
    pub handle: Arc<YrpHandle>,
    pub data_dir: PathBuf,
    pub port: u16,
    peers: Vec<YrpPeer>,
    spec: ClusterSpec,
    server: tokio::task::JoinHandle<()>,
}

/// A killed node's identity — everything needed to restart it.
pub struct DeadNode {
    pub node_id: u64,
    pub token: String,
    pub data_dir: PathBuf,
    pub port: u16,
    peers: Vec<YrpPeer>,
    spec: ClusterSpec,
}

impl TestNode {
    /// Stop the driver loop and the HTTP server; WAIT for the driver to
    /// actually exit before returning. The wait is load-bearing:
    /// Shutdown queues behind in-flight events, and a still-draining
    /// driver can persist yrp.state AFTER kill() — silently undoing any
    /// state corruption a chaos scenario applies next (the exact race
    /// the torn-state test hit on fast runners). The data dir survives.
    pub async fn kill(self) -> DeadNode {
        self.handle.shutdown();
        self.server.abort();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        while !self.handle.is_stopped() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "driver did not exit within 5s of Shutdown"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        DeadNode {
            node_id: self.node_id,
            token: self.token,
            data_dir: self.data_dir,
            port: self.port,
            peers: self.peers,
            spec: self.spec,
        }
    }
}

impl DeadNode {
    pub async fn restart(self) -> TestNode {
        spawn_node(
            self.node_id,
            self.peers.clone(),
            self.port,
            self.data_dir.clone(),
            Some(self.token.clone()),
            self.spec,
        )
        .await
    }
}

/// Spawn an n-node cluster (all data voters) on ephemeral pre-bound
/// ports. Returns the nodes plus the tempdirs keeping their data alive.
pub async fn spawn_cluster(n: usize, spec: ClusterSpec) -> (Vec<TestNode>, Vec<tempfile::TempDir>) {
    let mut ports = Vec::new();
    for _ in 0..n {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        ports.push(l.local_addr().unwrap().port());
        drop(l);
    }
    let peers: Vec<YrpPeer> = ports
        .iter()
        .enumerate()
        .map(|(i, p)| YrpPeer {
            node_id: i as u64 + 1,
            addr: format!("http://127.0.0.1:{p}"),
            witness: false,
        })
        .collect();
    let mut nodes = Vec::new();
    let mut tmps = Vec::new();
    for (i, port) in ports.iter().enumerate() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let node = spawn_node(
            i as u64 + 1,
            peers.clone(),
            *port,
            tmp.path().to_path_buf(),
            None,
            spec,
        )
        .await;
        nodes.push(node);
        tmps.push(tmp);
    }
    (nodes, tmps)
}

/// One full server on `port` with the production router, mirroring
/// main.rs's Yrp assembly. `existing_token = Some` on restart (the
/// control DB already holds the tenant + token).
pub async fn spawn_node(
    node_id: u64,
    peers: Vec<YrpPeer>,
    port: u16,
    data_dir: PathBuf,
    existing_token: Option<String>,
    spec: ClusterSpec,
) -> TestNode {
    let mut cfg = crate::config::ServerConfig::default();
    cfg.server.data_dir = data_dir.clone();

    let control = ControlDb::open(&data_dir.join("control.db")).expect("control db");
    let token = match existing_token {
        Some(t) => t,
        None => {
            let raw = crate::auth::generate_token();
            let hash = crate::auth::hash_token(&raw);
            let db_id = control.create_database(TENANT, TENANT).expect("create db");
            control
                .create_token(&hash, db_id, "yrp-chaos")
                .expect("create token");
            raw
        }
    };
    let control = Arc::new(Mutex::new(control));

    let pool = Arc::new(crate::tenant_pool::TenantPool::new(&cfg, None, None));
    let workers = crate::background::WorkerRegistry::new(
        &cfg.background,
        &cfg.maintenance,
        crate::background::WriteAcceptanceGate::standalone(),
    );
    let admission = crate::admission::AdmissionState::new(Default::default());
    let jobs: Arc<dyn crate::jobs::JobQueue> =
        Arc::new(crate::jobs::LocalSqliteJobQueue::open_in_memory().expect("jobs"));
    let auth_provider: Arc<dyn crate::auth::AuthProvider> = Arc::new(ControlDbAuthProvider::new(
        Arc::clone(&control),
        Some(SECRET.to_string()),
    ));

    let local: Arc<dyn crate::commit::MutationCommitter> = Arc::new(
        crate::commit::LocalSqliteCommitter::open(data_dir.join("commit_log.sqlite"))
            .expect("commit log"),
    );
    let resolver = Arc::new(crate::tenant_pool::TenantPoolEngineResolver::new(
        pool.clone(),
        control.clone(),
    )) as Arc<dyn crate::commit::EngineResolver>;
    let applier: Arc<dyn crate::commit::Applier> =
        Arc::new(crate::commit::EngineApplier::new(resolver));
    let handle = spawn_yrp(
        YrpRuntimeConfig {
            node_id,
            cluster_id: 7,
            peers: peers.clone(),
            data_dir: data_dir.clone(),
            cluster_secret: Some(SECRET.to_string()),
            tick_ms: 20,
            election_ticks: (5, 10),
            heartbeat_ticks: 2,
            compact_after_entries: spec.compact_after_entries,
            leader_retain_entries: spec.leader_retain_entries,
        },
        local.clone(),
        applier,
        control.clone(),
    )
    .expect("yrp spawn");
    let commit_log: Arc<dyn crate::commit::MutationCommitter> =
        Arc::new(YrpCommitter::new(handle.clone(), local));

    let state = Arc::new(AppState {
        control,
        pool,
        workers,
        cluster: None,
        inflight: std::sync::atomic::AtomicU32::new(0),
        admission,
        control_runtime: None,
        commit_log,
        yrp: Some(handle.clone()),
        fault_registry: crate::debug::FaultRegistry::new(),
        jobs,
        data_dir: data_dir.clone(),
        auth_provider,
    });

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("bind advertised port");
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = crate::http_gateway::router(state.clone());
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let node = TestNode {
        node_id,
        base,
        token,
        state,
        handle,
        data_dir,
        port,
        peers,
        spec,
        server,
    };
    // Readiness poll.
    let client = reqwest::Client::new();
    for _ in 0..100 {
        if client
            .get(format!("{}/v1/health", node.base))
            .send()
            .await
            .is_ok()
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    node
}

/// Wait until exactly one live node leads; return its index.
pub async fn wait_leader(nodes: &[&TestNode]) -> usize {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        for (i, n) in nodes.iter().enumerate() {
            if n.handle.is_leader() {
                return i;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "no leader elected in time"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

pub async fn post_json(node: &TestNode, path: &str, body: &Value) -> (reqwest::StatusCode, Value) {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .unwrap();
    let resp = client
        .post(format!("{}{}", node.base, path))
        .bearer_auth(&node.token)
        .json(body)
        .send()
        .await
        .expect("request");
    let status = resp.status();
    let text = resp.text().await.expect("body");
    let val: Value = serde_json::from_str(&text).unwrap_or(json!({ "raw": text }));
    (status, val)
}

/// 384-dim deterministic embedding (engine default dim).
pub fn embedding(seed: f32) -> Value {
    json!((0..384)
        .map(|i| (i as f32).mul_add(0.001, seed).sin())
        .collect::<Vec<f32>>())
}

/// Keyed remember against whichever node currently accepts it; retries
/// across nodes until a 200 lands. Returns (rid, node_index).
pub async fn keyed_write_until_accepted(
    nodes: &[&TestNode],
    key: &str,
    text: &str,
    emb: &Value,
) -> (String, usize) {
    let body = json!({ "text": text, "embedding": emb, "idempotency_key": key });
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        for (i, n) in nodes.iter().enumerate() {
            let (st, resp) = post_json(n, "/v1/remember", &body).await;
            if st == reqwest::StatusCode::OK {
                let rid = resp["rid"].as_str().expect("rid").to_string();
                return (rid, i);
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "no node accepted keyed write {key}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

/// Poll a node's LIVE /v1/recall until `rid` appears.
pub async fn wait_for_recall(node: &TestNode, query_embedding: &Value, rid: &str) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let (st, resp) = post_json(
            node,
            "/v1/recall",
            &json!({
                "query": "chaos replicated memory",
                "query_embedding": query_embedding,
                "top_k": 20,
            }),
        )
        .await;
        if st == reqwest::StatusCode::OK {
            let found = resp["results"]
                .as_array()
                .map(|a| a.iter().any(|r| r["rid"].as_str() == Some(rid)))
                .unwrap_or(false);
            if found {
                return;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "node {} live recall never surfaced {rid}; last: {resp}",
            node.node_id
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}
