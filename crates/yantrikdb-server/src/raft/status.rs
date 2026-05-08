//! Operator-facing status surface for the openraft cluster.
//!
//! - [`RaftStatus`] — JSON-friendly snapshot of the cluster state
//!   (leader, term, log indices, membership, server state).
//! - [`raft_status_router`] — axum [`Router`] exposing
//!   `GET /v1/cluster/raft` so operators (CLI, dashboards) can inspect
//!   without needing in-process access to the [`Raft`] handle.
//! - Used by `yantrikdb cluster raft-status` CLI and `\raft` yql.

use std::sync::Arc;

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use openraft::Raft;
use serde::{Deserialize, Serialize};

use super::types::{YantrikNode, YantrikNodeId, YantrikRaftTypeConfig};

/// JSON-friendly cluster status snapshot. Stable shape — operators
/// build dashboards against these field names, so renames are wire
/// breaks. Add new fields only at the end.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RaftStatus {
    /// This node's id.
    pub node_id: u64,
    /// "Leader", "Follower", "Candidate", "Learner". Lowercase.
    pub state: String,
    /// Current Raft term as observed by this node.
    pub current_term: u64,
    /// Cluster leader id, or `None` mid-election / no leader yet.
    pub current_leader: Option<u64>,
    /// Last log index appended on this node.
    pub last_log_index: Option<u64>,
    /// Last log index applied to the state machine on this node.
    pub last_applied_index: Option<u64>,
    /// Term of the last applied log id. Useful for detecting stale
    /// state across rolling restarts.
    pub last_applied_term: Option<u64>,
    /// Last snapshot's last_log_index, if any.
    pub snapshot_index: Option<u64>,
    /// Earliest log index still present in the log (purged below this).
    pub purged_index: Option<u64>,
    /// For a leader, ms since quorum last acknowledged the leader.
    /// `None` when this node isn't the leader or quorum hasn't yet.
    pub millis_since_quorum_ack: Option<u64>,
    /// Cluster membership: each entry is a node id + its address.
    pub members: Vec<RaftMember>,
    /// Whether `running_state` is `Ok`. False after a fatal error.
    pub healthy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RaftMember {
    pub node_id: u64,
    pub addr: String,
    /// `true` if this member is a voter (counts toward quorum), `false`
    /// if a learner (replicated to but doesn't vote).
    pub is_voter: bool,
}

impl RaftStatus {
    /// Build a snapshot from a live [`Raft`] handle. Cheap — reads the
    /// current value of openraft's metrics watch channel without
    /// blocking.
    pub fn from_raft(raft: &Arc<Raft<YantrikRaftTypeConfig>>) -> Self {
        let metrics = raft.metrics().borrow().clone();
        Self::from_metrics(&metrics)
    }

    /// Build from an already-captured RaftMetrics. Used by the metrics
    /// recorder that subscribes to the watch channel.
    pub fn from_metrics(m: &openraft::RaftMetrics<YantrikNodeId, YantrikNode>) -> Self {
        let voters: std::collections::BTreeSet<YantrikNodeId> =
            m.membership_config.membership().voter_ids().collect();
        let mut members: Vec<RaftMember> = m
            .membership_config
            .nodes()
            .map(|(id, node)| RaftMember {
                node_id: id.raw(),
                addr: node.addr.clone(),
                is_voter: voters.contains(id),
            })
            .collect();
        members.sort_by_key(|m| m.node_id);

        Self {
            node_id: m.id.raw(),
            state: format!("{:?}", m.state).to_lowercase(),
            current_term: m.current_term,
            current_leader: m.current_leader.map(|id| id.raw()),
            last_log_index: m.last_log_index,
            last_applied_index: m.last_applied.as_ref().map(|l| l.index),
            last_applied_term: m.last_applied.as_ref().map(|l| l.leader_id.term),
            snapshot_index: m.snapshot.as_ref().map(|l| l.index),
            purged_index: m.purged.as_ref().map(|l| l.index),
            millis_since_quorum_ack: m.millis_since_quorum_ack,
            members,
            healthy: m.running_state.is_ok(),
        }
    }

    /// True if this node currently believes it is the leader.
    pub fn is_leader(&self) -> bool {
        self.current_leader == Some(self.node_id)
    }
}

/// Update the Prometheus openraft gauges from a captured
/// [`RaftStatus`]. Status-aware wrapper around the leaf primitive
/// setter [`crate::metrics::record_openraft_gauges`].
pub fn record_openraft_status(status: &RaftStatus) {
    let voters = status.members.iter().filter(|m| m.is_voter).count() as u64;
    let learners = status.members.iter().filter(|m| !m.is_voter).count() as u64;
    crate::metrics::record_openraft_gauges(
        status.current_term,
        status.is_leader(),
        status.last_log_index,
        status.last_applied_index,
        status.snapshot_index,
        status.purged_index,
        status.millis_since_quorum_ack,
        status.healthy,
        voters,
        learners,
    );
}

/// axum route: `GET /v1/cluster/raft` returning [`RaftStatus`] as JSON.
pub fn raft_status_router(raft: Arc<Raft<YantrikRaftTypeConfig>>) -> Router {
    Router::new()
        .route("/v1/cluster/raft", get(handle_raft_status))
        .with_state(raft)
}

/// Spawn a background task that subscribes to openraft's metrics watch
/// channel and updates the Prometheus gauges in `crate::metrics`. The
/// task runs until `cancel` is signaled (typically the server's
/// shutdown token) or the watch channel is dropped.
///
/// Caller MUST keep the returned `JoinHandle` alive for the lifetime of
/// the Raft instance; dropping it before shutdown won't crash but
/// stops the recorder.
pub fn spawn_raft_metrics_recorder(
    raft: Arc<Raft<YantrikRaftTypeConfig>>,
    cancel: tokio_util::sync::CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let mut rx = raft.metrics();
    tokio::spawn(async move {
        // Push an initial snapshot so dashboards have data before the
        // first metric change arrives.
        let initial = rx.borrow().clone();
        record_openraft_status(&RaftStatus::from_metrics(&initial));

        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                changed = rx.changed() => {
                    if changed.is_err() {
                        // Watch channel sender dropped — Raft shut down.
                        break;
                    }
                    let m = rx.borrow().clone();
                    record_openraft_status(&RaftStatus::from_metrics(&m));
                }
            }
        }
    })
}

async fn handle_raft_status(
    State(raft): State<Arc<Raft<YantrikRaftTypeConfig>>>,
) -> Json<RaftStatus> {
    Json(RaftStatus::from_raft(&raft))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::LocalSqliteCommitter;
    use crate::raft::log_storage::SqliteRaftLogStorage;
    use crate::raft::network::StubRaftNetworkFactory;
    use crate::raft::state_machine::YantrikStateMachine;
    use openraft::Config;
    use std::collections::BTreeMap;
    use std::time::Duration;

    async fn build_single_node_raft() -> Arc<Raft<YantrikRaftTypeConfig>> {
        let local = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let log_store = SqliteRaftLogStorage::open_in_memory();
        let state_machine = YantrikStateMachine::new(
            local,
            std::sync::Arc::new(crate::commit::LocalApplier::new()),
        );
        let network = StubRaftNetworkFactory;
        let config = Arc::new(
            Config {
                cluster_name: "yantrikdb-status-test".into(),
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
        raft
    }

    #[tokio::test]
    async fn status_reflects_single_node_leader_state() {
        let raft = build_single_node_raft().await;
        let status = RaftStatus::from_raft(&raft);
        assert_eq!(status.node_id, 1);
        assert_eq!(status.state, "leader");
        assert!(status.is_leader());
        assert_eq!(status.current_leader, Some(1));
        assert!(status.current_term >= 1);
        assert!(status.healthy);
        assert_eq!(status.members.len(), 1);
        assert_eq!(status.members[0].node_id, 1);
        assert!(status.members[0].is_voter);
    }

    #[tokio::test]
    async fn status_serializes_to_stable_json_shape() {
        // Operators key dashboards on field names — pin a few critical ones.
        let raft = build_single_node_raft().await;
        let status = RaftStatus::from_raft(&raft);
        let json = serde_json::to_string(&status).unwrap();
        // The most operationally important fields:
        assert!(json.contains("\"node_id\""));
        assert!(json.contains("\"state\""));
        assert!(json.contains("\"current_term\""));
        assert!(json.contains("\"current_leader\""));
        assert!(json.contains("\"members\""));
        assert!(json.contains("\"healthy\""));
        // Round-trip.
        let back: RaftStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, back);
    }

    #[tokio::test]
    async fn raft_status_router_serves_json() {
        use axum::body::to_bytes;
        use axum::http::Request;
        use tower::ServiceExt;

        let raft = build_single_node_raft().await;
        let app = raft_status_router(raft);
        let req = Request::builder()
            .uri("/v1/cluster/raft")
            .body(axum::body::Body::empty())
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), 200);
        let body_bytes = to_bytes(res.into_body(), 64 * 1024).await.unwrap();
        let parsed: RaftStatus = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(parsed.node_id, 1);
        assert_eq!(parsed.state, "leader");
    }

    #[tokio::test]
    async fn record_openraft_status_updates_prometheus_gauges() {
        // End-to-end: build a single-node leader, record its status,
        // verify the rendered Prometheus text reflects the values.
        let raft = build_single_node_raft().await;
        let status = RaftStatus::from_raft(&raft);
        record_openraft_status(&status);
        let rendered = crate::metrics::global().render_prometheus();
        // Single-node leader → is_leader=1, voters=1, healthy=1.
        assert!(rendered.contains("yantrikdb_openraft_is_leader 1"));
        assert!(rendered.contains("yantrikdb_openraft_voters 1"));
        assert!(rendered.contains("yantrikdb_openraft_running_state_healthy 1"));
        // Term should be > 0 after election won.
        assert!(
            rendered.contains("yantrikdb_openraft_current_term"),
            "expected current_term gauge in rendered metrics"
        );
        // Last applied may be None → -1 sentinel; either way the line exists.
        assert!(rendered.contains("yantrikdb_openraft_last_applied_index"));
    }

    #[test]
    fn record_openraft_status_handles_none_indices_with_minus_one() {
        // Synthetic status with all indices = None, verify the Prometheus
        // text shows -1 sentinels.
        let s = RaftStatus {
            node_id: 5,
            state: "follower".into(),
            current_term: 7,
            current_leader: Some(1),
            last_log_index: None,
            last_applied_index: None,
            last_applied_term: None,
            snapshot_index: None,
            purged_index: None,
            millis_since_quorum_ack: None,
            members: vec![],
            healthy: true,
        };
        record_openraft_status(&s);
        let rendered = crate::metrics::global().render_prometheus();
        assert!(rendered.contains("yantrikdb_openraft_last_log_index -1"));
        assert!(rendered.contains("yantrikdb_openraft_quorum_ack_lag_ms -1"));
        assert!(rendered.contains("yantrikdb_openraft_is_leader 0"));
    }

    #[test]
    fn is_leader_reads_current_leader_against_self() {
        let leader = RaftStatus {
            node_id: 1,
            state: "leader".into(),
            current_term: 1,
            current_leader: Some(1),
            last_log_index: None,
            last_applied_index: None,
            last_applied_term: None,
            snapshot_index: None,
            purged_index: None,
            millis_since_quorum_ack: None,
            members: vec![],
            healthy: true,
        };
        assert!(leader.is_leader());

        let follower = RaftStatus {
            node_id: 2,
            state: "follower".into(),
            current_term: 1,
            current_leader: Some(1),
            last_log_index: None,
            last_applied_index: None,
            last_applied_term: None,
            snapshot_index: None,
            purged_index: None,
            millis_since_quorum_ack: None,
            members: vec![],
            healthy: true,
        };
        assert!(!follower.is_leader());

        let no_leader = RaftStatus {
            node_id: 1,
            state: "candidate".into(),
            current_term: 2,
            current_leader: None,
            last_log_index: None,
            last_applied_index: None,
            last_applied_term: None,
            snapshot_index: None,
            purged_index: None,
            millis_since_quorum_ack: None,
            members: vec![],
            healthy: true,
        };
        assert!(!no_leader.is_leader());
    }
}
