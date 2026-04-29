//! `HttpRaftNetwork` — real openraft transport over HTTPS + reqwest +
//! axum. Replaces the [`super::network::StubRaftNetwork`] from PR-4-d-a.
//!
//! ## Wire format
//!
//! Three POST endpoints, JSON bodies + responses:
//!
//! | Path | Request | Response |
//! |---|---|---|
//! | `/v1/raft/append_entries` | `AppendEntriesRequest<YantrikRaftTypeConfig>` | `AppendEntriesResponse<YantrikNodeId>` |
//! | `/v1/raft/vote` | `VoteRequest<YantrikNodeId>` | `VoteResponse<YantrikNodeId>` |
//! | `/v1/raft/install_full_snapshot` | `(Vote, Snapshot)` (the snapshot bytes are JSON-serialized as Vec<u8>) | `SnapshotResponse<YantrikNodeId>` |
//!
//! JSON over HTTP is intentionally simple — both sides already use
//! serde_json elsewhere (m001-m004 payloads, snapshot envelope), and
//! the operational benefit of JSON-readable cluster traffic for
//! debugging is non-negligible. Bandwidth/CPU is bounded by the
//! snapshot size; for normal append_entries the payload is small.
//!
//! ## mTLS gate
//!
//! [`HttpRaftNetworkFactory::new_with_tls`] wraps a
//! [`reqwest::Client`] configured with the cluster's mTLS identity
//! (RFC 014-A). [`HttpRaftNetworkFactory::new_plaintext`] is dev-only
//! and intentionally noisy in logs to discourage production use.
//!
//! When openraft mode is configured but cluster_tls is missing, the
//! application's startup gate (wired in [`crate::server`]) refuses to
//! start. This module just provides the constructors; the gate lives
//! at the assembly site so the failure mode is "server doesn't come
//! up", not "server runs and replication mysteriously fails."

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use openraft::error::{
    Fatal, InstallSnapshotError, NetworkError, RPCError, RaftError, RemoteError, ReplicationClosed,
    StreamingError, Unreachable,
};
use openraft::network::{Backoff, RPCOption, RaftNetwork, RaftNetworkFactory};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    SnapshotResponse, VoteRequest, VoteResponse,
};
use openraft::{AnyError, Raft, Snapshot, SnapshotMeta, Vote};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::types::{YantrikNode, YantrikNodeId, YantrikRaftTypeConfig};

/// One connection to one peer. Cheap to clone — wraps reqwest's
/// internally-pooled client.
#[derive(Clone)]
pub struct HttpRaftNetwork {
    client: Client,
    target: YantrikNodeId,
    target_addr: String,
    request_timeout: Duration,
}

impl HttpRaftNetwork {
    fn url(&self, path: &str) -> String {
        let base = self.target_addr.trim_end_matches('/');
        format!("{base}{path}")
    }

    async fn json_post<Req, Resp>(&self, path: &str, body: &Req) -> Result<Resp, NetworkError>
    where
        Req: Serialize + ?Sized,
        Resp: for<'de> Deserialize<'de>,
    {
        let url = self.url(path);
        let res = self
            .client
            .post(&url)
            .timeout(self.request_timeout)
            .json(body)
            .send()
            .await
            .map_err(|e| {
                NetworkError::new(&AnyError::error(format!(
                    "raft RPC POST {url}: transport: {e}"
                )))
            })?;
        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            return Err(NetworkError::new(&AnyError::error(format!(
                "raft RPC POST {url}: HTTP {status}: {body}"
            ))));
        }
        let parsed = res.json::<Resp>().await.map_err(|e| {
            NetworkError::new(&AnyError::error(format!(
                "raft RPC POST {url}: response decode: {e}"
            )))
        })?;
        Ok(parsed)
    }
}

/// Wire envelope for `install_full_snapshot`. The snapshot's
/// `Cursor<Vec<u8>>` body is captured as a `Vec<u8>` so the whole
/// thing can be JSON-encoded in one POST.
#[derive(Debug, Serialize, Deserialize)]
struct InstallFullSnapshotWire {
    vote: Vote<YantrikNodeId>,
    meta: SnapshotMeta<YantrikNodeId, YantrikNode>,
    /// Snapshot bytes (the SnapshotEnvelope JSON from PR-4-c).
    data: Vec<u8>,
}

impl RaftNetwork<YantrikRaftTypeConfig> for HttpRaftNetwork {
    async fn append_entries(
        &mut self,
        rpc: AppendEntriesRequest<YantrikRaftTypeConfig>,
        _option: RPCOption,
    ) -> Result<
        AppendEntriesResponse<YantrikNodeId>,
        RPCError<YantrikNodeId, YantrikNode, RaftError<YantrikNodeId>>,
    > {
        match self
            .json_post::<_, Result<AppendEntriesResponse<YantrikNodeId>, RaftError<YantrikNodeId>>>(
                "/v1/raft/append_entries",
                &rpc,
            )
            .await
        {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(remote_err)) => Err(RPCError::RemoteError(RemoteError::new(
                self.target,
                remote_err,
            ))),
            Err(net_err) => Err(RPCError::Unreachable(Unreachable::new(&net_err))),
        }
    }

    async fn vote(
        &mut self,
        rpc: VoteRequest<YantrikNodeId>,
        _option: RPCOption,
    ) -> Result<
        VoteResponse<YantrikNodeId>,
        RPCError<YantrikNodeId, YantrikNode, RaftError<YantrikNodeId>>,
    > {
        match self
            .json_post::<_, Result<VoteResponse<YantrikNodeId>, RaftError<YantrikNodeId>>>(
                "/v1/raft/vote",
                &rpc,
            )
            .await
        {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(remote_err)) => Err(RPCError::RemoteError(RemoteError::new(
                self.target,
                remote_err,
            ))),
            Err(net_err) => Err(RPCError::Unreachable(Unreachable::new(&net_err))),
        }
    }

    async fn install_snapshot(
        &mut self,
        _rpc: InstallSnapshotRequest<YantrikRaftTypeConfig>,
        _option: RPCOption,
    ) -> Result<
        InstallSnapshotResponse<YantrikNodeId>,
        RPCError<YantrikNodeId, YantrikNode, RaftError<YantrikNodeId, InstallSnapshotError>>,
    > {
        // Deprecated under `generic-snapshot-data`. Always returns
        // Unreachable so openraft falls back to the `full_snapshot`
        // path below.
        Err(RPCError::Unreachable(Unreachable::new(&NetworkError::new(
            &AnyError::error(
                "install_snapshot is deprecated under generic-snapshot-data; use full_snapshot",
            ),
        ))))
    }

    async fn full_snapshot(
        &mut self,
        vote: Vote<YantrikNodeId>,
        snapshot: Snapshot<YantrikRaftTypeConfig>,
        _cancel: impl std::future::Future<Output = ReplicationClosed> + Send + 'static,
        _option: RPCOption,
    ) -> Result<
        SnapshotResponse<YantrikNodeId>,
        StreamingError<YantrikRaftTypeConfig, Fatal<YantrikNodeId>>,
    > {
        let data = snapshot.snapshot.into_inner();
        let wire = InstallFullSnapshotWire {
            vote: vote.clone(),
            meta: snapshot.meta,
            data,
        };
        match self
            .json_post::<_, Result<SnapshotResponse<YantrikNodeId>, Fatal<YantrikNodeId>>>(
                "/v1/raft/install_full_snapshot",
                &wire,
            )
            .await
        {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(fatal)) => Err(StreamingError::RemoteError(RemoteError::new(
                self.target,
                fatal,
            ))),
            Err(net_err) => Err(StreamingError::Network(net_err)),
        }
    }

    fn backoff(&self) -> Backoff {
        Backoff::new(std::iter::repeat(Duration::from_millis(500)))
    }
}

/// Factory that hands out an `HttpRaftNetwork` per peer. Holds a
/// shared [`Client`] so connection pooling works across peers.
#[derive(Clone)]
pub struct HttpRaftNetworkFactory {
    client: Client,
    request_timeout: Duration,
}

impl HttpRaftNetworkFactory {
    /// Build with a pre-configured reqwest Client. Production callers
    /// pass a client built with [`reqwest::ClientBuilder::identity`]
    /// (cluster mTLS) and [`reqwest::ClientBuilder::add_root_certificate`]
    /// (cluster CA) for RFC 014-A enforcement.
    pub fn new(client: Client, request_timeout: Duration) -> Self {
        Self {
            client,
            request_timeout,
        }
    }

    /// Plaintext HTTP — dev only. Logs a warning at construction so
    /// it's hard to accidentally ship to production.
    pub fn new_plaintext(request_timeout: Duration) -> Self {
        tracing::warn!(
            "HttpRaftNetworkFactory: plaintext HTTP — DEV ONLY. \
             Production must use new() with mTLS-configured reqwest::Client (RFC 014-A)."
        );
        let client = Client::builder()
            .timeout(request_timeout)
            .build()
            .expect("plaintext reqwest client must build");
        Self {
            client,
            request_timeout,
        }
    }
}

impl RaftNetworkFactory<YantrikRaftTypeConfig> for HttpRaftNetworkFactory {
    type Network = HttpRaftNetwork;

    async fn new_client(&mut self, target: YantrikNodeId, node: &YantrikNode) -> Self::Network {
        HttpRaftNetwork {
            client: self.client.clone(),
            target,
            target_addr: node.addr.clone(),
            request_timeout: self.request_timeout,
        }
    }
}

// ============================================================
// Receive-side: axum routes that dispatch to a local Raft instance
// ============================================================

/// Build an axum [`Router`] that exposes the three Raft RPC endpoints,
/// each dispatched to the provided [`Raft`] instance. Mount this on
/// the cluster gateway alongside the other `/v1/...` routes.
///
/// Routes:
/// - `POST /v1/raft/append_entries`
/// - `POST /v1/raft/vote`
/// - `POST /v1/raft/install_full_snapshot`
pub fn raft_receive_router(raft: Arc<Raft<YantrikRaftTypeConfig>>) -> Router {
    Router::new()
        .route("/v1/raft/append_entries", post(handle_append_entries))
        .route("/v1/raft/vote", post(handle_vote))
        .route(
            "/v1/raft/install_full_snapshot",
            post(handle_install_full_snapshot),
        )
        .with_state(raft)
}

async fn handle_append_entries(
    State(raft): State<Arc<Raft<YantrikRaftTypeConfig>>>,
    Json(rpc): Json<AppendEntriesRequest<YantrikRaftTypeConfig>>,
) -> Response {
    match raft.append_entries(rpc).await {
        Ok(resp) => Json::<Result<_, RaftError<YantrikNodeId>>>(Ok(resp)).into_response(),
        Err(e) => {
            Json::<Result<AppendEntriesResponse<YantrikNodeId>, RaftError<YantrikNodeId>>>(Err(e))
                .into_response()
        }
    }
}

async fn handle_vote(
    State(raft): State<Arc<Raft<YantrikRaftTypeConfig>>>,
    Json(rpc): Json<VoteRequest<YantrikNodeId>>,
) -> Response {
    match raft.vote(rpc).await {
        Ok(resp) => Json::<Result<_, RaftError<YantrikNodeId>>>(Ok(resp)).into_response(),
        Err(e) => Json::<Result<VoteResponse<YantrikNodeId>, RaftError<YantrikNodeId>>>(Err(e))
            .into_response(),
    }
}

async fn handle_install_full_snapshot(
    State(raft): State<Arc<Raft<YantrikRaftTypeConfig>>>,
    Json(wire): Json<InstallFullSnapshotWire>,
) -> Response {
    let snapshot = Snapshot {
        meta: wire.meta,
        snapshot: Box::new(std::io::Cursor::new(wire.data)),
    };
    match raft.install_full_snapshot(wire.vote, snapshot).await {
        Ok(resp) => Json::<Result<_, Fatal<YantrikNodeId>>>(Ok(resp)).into_response(),
        Err(e) => {
            // Surface fatal errors as HTTP 500 with the error encoded
            // so the client can distinguish them from transport errors.
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json::<Result<SnapshotResponse<YantrikNodeId>, Fatal<YantrikNodeId>>>(Err(e)),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::LocalSqliteCommitter;
    use crate::raft::log_storage::SqliteRaftLogStorage;
    use crate::raft::state_machine::YantrikStateMachine;
    use openraft::Config;
    use std::collections::BTreeMap;
    use std::net::SocketAddr;

    /// Build a single-node Raft cluster bound to a real TCP port +
    /// expose its receive routes via axum. Returns (raft, base_url).
    async fn spawn_single_node_raft() -> (Arc<Raft<YantrikRaftTypeConfig>>, String) {
        let local = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let log_store = SqliteRaftLogStorage::open_in_memory();
        let state_machine = YantrikStateMachine::new(local);
        // Use the stub network for outbound — these tests only exercise
        // the receive side. (HttpRaftNetworkFactory requires a real
        // reqwest client and we don't need outbound for receive tests.)
        let network = super::super::network::StubRaftNetworkFactory;

        let config = Arc::new(
            Config {
                cluster_name: "yantrikdb-recv-test".into(),
                heartbeat_interval: 100,
                election_timeout_min: 200,
                election_timeout_max: 400,
                ..Default::default()
            }
            .validate()
            .unwrap(),
        );

        let me = YantrikNodeId::new(1);
        let raft = Arc::new(
            Raft::<YantrikRaftTypeConfig>::new(me, config, network, log_store, state_machine)
                .await
                .unwrap(),
        );
        let mut nodes = BTreeMap::new();
        nodes.insert(me, YantrikNode::new("http://127.0.0.1:0"));
        raft.initialize(nodes).await.unwrap();
        for _ in 0..30 {
            if raft.current_leader().await == Some(me) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let router = raft_receive_router(raft.clone());
        let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .unwrap();
        let bound = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.ok();
        });
        // Give axum a moment to start.
        tokio::time::sleep(Duration::from_millis(50)).await;
        (raft, format!("http://{bound}"))
    }

    #[tokio::test]
    async fn http_factory_creates_per_peer_clients() {
        let mut f = HttpRaftNetworkFactory::new_plaintext(Duration::from_millis(500));
        let n1 = f
            .new_client(YantrikNodeId::new(1), &YantrikNode::new("http://n1"))
            .await;
        assert_eq!(n1.target, YantrikNodeId::new(1));
        assert_eq!(n1.target_addr, "http://n1");
    }

    #[tokio::test]
    async fn append_entries_roundtrip_through_http_layer() {
        let (_raft, base) = spawn_single_node_raft().await;
        let mut f = HttpRaftNetworkFactory::new_plaintext(Duration::from_secs(2));
        let mut net = f
            .new_client(YantrikNodeId::new(1), &YantrikNode::new(&base))
            .await;
        // Empty heartbeat (no entries) directed at the leader.
        let rpc = AppendEntriesRequest::<YantrikRaftTypeConfig> {
            vote: Vote::new_committed(1, YantrikNodeId::new(1)),
            prev_log_id: None,
            entries: Vec::new(),
            leader_commit: None,
        };
        let resp = net
            .append_entries(rpc, RPCOption::new(Duration::from_secs(1)))
            .await;
        // The leader rejecting another leader's heartbeat is a valid
        // remote response. We only assert the HTTP transport itself
        // worked — i.e. NOT a transport-level Unreachable error.
        match resp {
            Ok(_) => {}
            Err(RPCError::RemoteError(_)) => {}
            Err(other) => panic!("expected Ok or RemoteError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn vote_roundtrip_through_http_layer() {
        let (_raft, base) = spawn_single_node_raft().await;
        let mut f = HttpRaftNetworkFactory::new_plaintext(Duration::from_secs(2));
        let mut net = f
            .new_client(YantrikNodeId::new(1), &YantrikNode::new(&base))
            .await;
        let rpc = VoteRequest {
            vote: Vote::new(2, YantrikNodeId::new(2)),
            last_log_id: None,
        };
        let resp = net
            .vote(rpc, RPCOption::new(Duration::from_secs(1)))
            .await
            .unwrap();
        // Some response — vote may be granted or rejected depending on
        // raft state; both are legitimate "transport works" outcomes.
        let _ = resp.vote_granted;
    }

    #[tokio::test]
    async fn unreachable_address_surfaces_as_transport_error() {
        // Bind a socket and immediately drop it — connecting fails.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dead_addr = listener.local_addr().unwrap();
        drop(listener);

        let mut f = HttpRaftNetworkFactory::new_plaintext(Duration::from_millis(500));
        let mut net = f
            .new_client(
                YantrikNodeId::new(99),
                &YantrikNode::new(&format!("http://{dead_addr}")),
            )
            .await;
        let rpc = VoteRequest {
            vote: Vote::new(1, YantrikNodeId::new(1)),
            last_log_id: None,
        };
        let err = net
            .vote(rpc, RPCOption::new(Duration::from_millis(500)))
            .await
            .unwrap_err();
        assert!(matches!(err, RPCError::Unreachable(_)));
    }

    #[tokio::test]
    async fn install_snapshot_returns_unreachable_under_generic_data() {
        let (_raft, base) = spawn_single_node_raft().await;
        let mut f = HttpRaftNetworkFactory::new_plaintext(Duration::from_secs(1));
        let mut net = f
            .new_client(YantrikNodeId::new(1), &YantrikNode::new(&base))
            .await;
        let dummy_meta = SnapshotMeta::<YantrikNodeId, YantrikNode>::default();
        let rpc = InstallSnapshotRequest {
            vote: Vote::new(1, YantrikNodeId::new(1)),
            meta: dummy_meta,
            offset: 0,
            data: Vec::new(),
            done: true,
        };
        let err = net
            .install_snapshot(rpc, RPCOption::new(Duration::from_secs(1)))
            .await
            .unwrap_err();
        assert!(matches!(err, RPCError::Unreachable(_)));
    }
}
