//! Raft network transport — `RaftNetworkFactory` + `RaftNetwork`
//! implementations.
//!
//! ## Sub-PR layout
//!
//! | Sub-PR | What |
//! |---|---|
//! | **4-d-a (this)** | [`StubRaftNetwork`] — returns `Unreachable` for every RPC. Single-node Raft never invokes the network, so this lets PR-4-d-a ship the openraft assembly + [`super::committer::RaftCommitter`] before the real HTTP transport lands. |
//! | 4-d-b | `HttpRaftNetwork` — reqwest-backed transport with mTLS. Adds axum receive routes on the gateway side. |
//!
//! ## Why ship a stub network in 4-d-a
//!
//! The full HTTP transport adds ~800 LOC (reqwest client, axum receive
//! handlers, mTLS wiring, snapshot fragment-transport for
//! `generic-snapshot-data`). Bundling it with the openraft assembly +
//! `RaftCommitter` would make PR-4-d the largest sub-PR by far, with no
//! intermediate green test suite.
//!
//! Splitting lets PR-4-d-a prove end-to-end: log_storage (PR-4-b) +
//! state_machine (PR-4-c) + RaftCommitter assemble into a working
//! single-node Raft that satisfies the `MutationCommitter` contract.
//! PR-4-d-b then swaps the stub for the real network without changing
//! any other code.

use std::time::Duration;

use openraft::error::{
    InstallSnapshotError, NetworkError, RPCError, RaftError, ReplicationClosed, StreamingError,
    Unreachable,
};
use openraft::network::{Backoff, RaftNetwork, RaftNetworkFactory, RPCOption};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    SnapshotResponse, VoteRequest, VoteResponse,
};
use openraft::{AnyError, Snapshot, Vote};

use super::types::{YantrikNode, YantrikNodeId, YantrikRaftTypeConfig};

/// Single connection to a peer node. PR-4-d-a stub: returns
/// `Unreachable` for every RPC. This is correct for single-node Raft
/// (the network is never invoked) and obviously broken for cluster —
/// which is exactly the contract PR-4-d-b will fulfill.
pub struct StubRaftNetwork {
    target: YantrikNodeId,
    target_addr: String,
}

impl StubRaftNetwork {
    fn unreachable<E>(&self, op: &'static str) -> RPCError<YantrikNodeId, YantrikNode, E>
    where
        E: std::error::Error,
    {
        RPCError::Unreachable(Unreachable::new(&NetworkError::new(&AnyError::error(
            format!(
                "stub network: {op} to {} ({}) — PR-4-d-b will ship real HTTP transport",
                self.target, self.target_addr
            ),
        ))))
    }
}

impl RaftNetwork<YantrikRaftTypeConfig> for StubRaftNetwork {
    async fn append_entries(
        &mut self,
        _rpc: AppendEntriesRequest<YantrikRaftTypeConfig>,
        _option: RPCOption,
    ) -> Result<
        AppendEntriesResponse<YantrikNodeId>,
        RPCError<YantrikNodeId, YantrikNode, RaftError<YantrikNodeId>>,
    > {
        Err(self.unreachable("append_entries"))
    }

    async fn install_snapshot(
        &mut self,
        _rpc: InstallSnapshotRequest<YantrikRaftTypeConfig>,
        _option: RPCOption,
    ) -> Result<
        InstallSnapshotResponse<YantrikNodeId>,
        RPCError<YantrikNodeId, YantrikNode, RaftError<YantrikNodeId, InstallSnapshotError>>,
    > {
        Err(self.unreachable("install_snapshot"))
    }

    async fn vote(
        &mut self,
        _rpc: VoteRequest<YantrikNodeId>,
        _option: RPCOption,
    ) -> Result<
        VoteResponse<YantrikNodeId>,
        RPCError<YantrikNodeId, YantrikNode, RaftError<YantrikNodeId>>,
    > {
        Err(self.unreachable("vote"))
    }

    async fn full_snapshot(
        &mut self,
        _vote: Vote<YantrikNodeId>,
        _snapshot: Snapshot<YantrikRaftTypeConfig>,
        _cancel: impl std::future::Future<Output = ReplicationClosed> + Send + 'static,
        _option: RPCOption,
    ) -> Result<
        SnapshotResponse<YantrikNodeId>,
        StreamingError<YantrikRaftTypeConfig, openraft::error::Fatal<YantrikNodeId>>,
    > {
        // StreamingError doesn't wrap RPCError directly; surface as
        // a network error.
        Err(StreamingError::Network(NetworkError::new(&AnyError::error(
            format!(
                "stub network: full_snapshot to {} ({}) — PR-4-d-b will ship real HTTP transport",
                self.target, self.target_addr
            ),
        ))))
    }

    fn backoff(&self) -> Backoff {
        // Quick backoff so single-node tests don't wait long.
        Backoff::new(std::iter::repeat(Duration::from_millis(100)))
    }
}

/// Factory for stub network connections. Each call returns a fresh
/// `StubRaftNetwork` targeting one peer.
#[derive(Default, Clone)]
pub struct StubRaftNetworkFactory;

impl RaftNetworkFactory<YantrikRaftTypeConfig> for StubRaftNetworkFactory {
    type Network = StubRaftNetwork;

    async fn new_client(&mut self, target: YantrikNodeId, node: &YantrikNode) -> Self::Network {
        StubRaftNetwork {
            target,
            target_addr: node.addr.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openraft::raft::AppendEntriesRequest;
    use openraft::LeaderId;

    #[tokio::test]
    async fn factory_creates_clients_per_peer() {
        let mut f = StubRaftNetworkFactory;
        let n1 = f
            .new_client(YantrikNodeId::new(1), &YantrikNode::new("http://n1"))
            .await;
        let n2 = f
            .new_client(YantrikNodeId::new(2), &YantrikNode::new("http://n2"))
            .await;
        assert_eq!(n1.target, YantrikNodeId::new(1));
        assert_eq!(n2.target, YantrikNodeId::new(2));
        assert_eq!(n1.target_addr, "http://n1");
        assert_eq!(n2.target_addr, "http://n2");
    }

    #[tokio::test]
    async fn append_entries_returns_unreachable() {
        let mut net = StubRaftNetwork {
            target: YantrikNodeId::new(7),
            target_addr: "http://nowhere".into(),
        };
        let rpc = AppendEntriesRequest {
            vote: Vote::new(0, YantrikNodeId::new(0)),
            prev_log_id: None,
            entries: Vec::new(),
            leader_commit: None,
        };
        let err = net
            .append_entries(rpc, RPCOption::new(Duration::from_millis(100)))
            .await
            .unwrap_err();
        match err {
            RPCError::Unreachable(_) => {}
            other => panic!("expected Unreachable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn vote_rpc_returns_unreachable() {
        let mut net = StubRaftNetwork {
            target: YantrikNodeId::new(7),
            target_addr: "http://nowhere".into(),
        };
        let rpc = VoteRequest {
            vote: Vote::new(1, YantrikNodeId::new(7)),
            last_log_id: Some(openraft::LogId::new(
                LeaderId::new(0, YantrikNodeId::new(0)),
                0,
            )),
        };
        let err = net
            .vote(rpc, RPCOption::new(Duration::from_millis(100)))
            .await
            .unwrap_err();
        assert!(matches!(err, RPCError::Unreachable(_)));
    }

    #[test]
    fn backoff_is_tight_for_tests() {
        let net = StubRaftNetwork {
            target: YantrikNodeId::new(1),
            target_addr: "http://x".into(),
        };
        let b = net.backoff();
        // Pull the first interval — should be 100ms.
        let first = b.into_iter().next().unwrap();
        assert_eq!(first, Duration::from_millis(100));
    }
}
