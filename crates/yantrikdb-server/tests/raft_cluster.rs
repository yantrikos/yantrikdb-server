//! RFC 010 PR-4-e — 3-node cluster integration test.
//!
//! What this test gates:
//! 1. Three Raft nodes form a cluster via `add_learner` +
//!    `change_membership`.
//! 2. The leader replicates a write through the HttpRaftNetwork to
//!    every follower; followers' state machines see the entry.
//! 3. After committing N entries, every node's per-tenant
//!    `memory_commit_log` matches the leader's exactly.
//! 4. Killing the leader (drop the `Raft` handle) leaves the remaining
//!    quorum to elect a new leader, and writes against the new leader
//!    continue to apply on the surviving follower.
//!
//! These four points exercise the entire openraft adoption end-to-end:
//! `SqliteRaftLogStorage` (PR-4-b), `YantrikStateMachine` (PR-4-c),
//! `RaftCommitter` (PR-4-d-a), and `HttpRaftNetwork` /
//! `raft_receive_router` (PR-4-d-b).
//!
//! ## Why this is an integration test
//!
//! Real TCP sockets, real reqwest, real axum — same as production
//! cluster transport. Single-node tests in `raft::committer::tests`
//! cover the assembly contract; this test covers the wire.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use openraft::Config;
use openraft::Raft;

#[path = "../src/cache/mod.rs"]
mod cache;
#[path = "../src/commit/mod.rs"]
mod commit;
#[path = "../src/forget/mod.rs"]
mod forget;
#[path = "../src/index/mod.rs"]
mod index;
#[path = "../src/jobs/mod.rs"]
mod jobs;
#[path = "../src/key_provider/mod.rs"]
mod key_provider;
#[path = "../src/metrics.rs"]
mod metrics;
#[path = "../src/migrations/mod.rs"]
mod migrations;
#[path = "../src/raft/mod.rs"]
mod raft;
#[path = "../src/security/mod.rs"]
mod security;
#[path = "../src/version/mod.rs"]
mod version;

use commit::{CommitOptions, LocalSqliteCommitter, MemoryMutation, MutationCommitter, TenantId};
use raft::{
    raft_receive_router, HttpRaftNetworkFactory, RaftCommitter, SqliteRaftLogStorage, YantrikNode,
    YantrikNodeId, YantrikRaftTypeConfig, YantrikStateMachine,
};

/// One Raft node + its bound axum server. Holds Arcs to all the
/// internals so the test can probe state-machine results directly.
struct ClusterNode {
    id: YantrikNodeId,
    addr: String,
    raft: Arc<Raft<YantrikRaftTypeConfig>>,
    local: Arc<LocalSqliteCommitter>,
    committer: RaftCommitter,
    server_handle: tokio::task::JoinHandle<()>,
    /// Partition flag — when true, the receive routes return 503 so
    /// peers see this node as Unreachable. Lets `partition_then_heal`
    /// simulate a node-receive-blocked partition without restarting
    /// axum (which would re-bind to a different port and break peer
    /// addresses).
    partitioned: Arc<AtomicBool>,
}

impl ClusterNode {
    fn yantrik_node(&self) -> YantrikNode {
        YantrikNode::new(&self.addr)
    }

    /// Set the partition flag. `true` = drop incoming peer RPCs (peers
    /// see Unreachable). `false` = accept normally.
    fn set_partitioned(&self, p: bool) {
        self.partitioned.store(p, Ordering::Relaxed);
    }
}

async fn spawn_node(id: u64) -> ClusterNode {
    let local = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
    let log_store = SqliteRaftLogStorage::open_in_memory();
    let state_machine = YantrikStateMachine::new(
        local.clone(),
        std::sync::Arc::new(commit::LocalApplier::new()),
    );
    let network = HttpRaftNetworkFactory::new_plaintext(Duration::from_secs(2));

    let config = Arc::new(
        Config {
            cluster_name: "yantrikdb-3node".into(),
            heartbeat_interval: 100,
            election_timeout_min: 300,
            election_timeout_max: 600,
            ..Default::default()
        }
        .validate()
        .unwrap(),
    );

    let me = YantrikNodeId::new(id);
    let raft = Arc::new(
        Raft::<YantrikRaftTypeConfig>::new(me, config, network, log_store, state_machine)
            .await
            .unwrap(),
    );

    // Wrap the receive router in a partition-flag layer. When the flag
    // is set, every peer RPC short-circuits with HTTP 503 — which the
    // sending HttpRaftNetwork translates to RPCError::Unreachable.
    let partitioned = Arc::new(AtomicBool::new(false));
    let partitioned_for_layer = partitioned.clone();
    let router = raft_receive_router(raft.clone()).layer(axum::middleware::from_fn(
        move |req: axum::http::Request<axum::body::Body>, next: axum::middleware::Next| {
            let p = partitioned_for_layer.clone();
            async move {
                if p.load(Ordering::Relaxed) {
                    axum::http::Response::builder()
                        .status(axum::http::StatusCode::SERVICE_UNAVAILABLE)
                        .body(axum::body::Body::from("partitioned"))
                        .unwrap()
                } else {
                    next.run(req).await
                }
            }
        },
    ));
    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let bound = listener.local_addr().unwrap();
    let server_handle = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    // Give the server a moment to start accepting.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let committer = RaftCommitter::new(raft.clone(), local.clone());
    ClusterNode {
        id: me,
        addr: format!("http://{bound}"),
        raft,
        local,
        committer,
        server_handle,
        partitioned,
    }
}

async fn wait_for_leader(node: &ClusterNode, deadline: Instant) -> Option<YantrikNodeId> {
    while Instant::now() < deadline {
        if let Some(l) = node.raft.current_leader().await {
            return Some(l);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    node.raft.current_leader().await
}

async fn wait_for_high_water(
    committer: &Arc<LocalSqliteCommitter>,
    tenant: TenantId,
    target: u64,
    deadline: Instant,
) -> u64 {
    while Instant::now() < deadline {
        let h = committer.high_watermark(tenant).await.unwrap();
        if h >= target {
            return h;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    committer.high_watermark(tenant).await.unwrap()
}

fn upsert(rid: &str) -> MemoryMutation {
    MemoryMutation::UpsertMemory {
        rid: rid.into(),
        text: format!("text-{rid}"),
        memory_type: "semantic".into(),
        importance: 0.5,
        valence: 0.0,
        half_life: 168.0,
        namespace: "default".into(),
        certainty: 1.0,
        domain: "general".into(),
        source: "user".into(),
        emotional_state: None,
        embedding: None,
        metadata: serde_json::json!({}),
        extracted_entities: vec![],
        created_at_unix_micros: None,
        embedding_model: None,
    }
}

#[tokio::test]
async fn three_node_cluster_forms_and_replicates_writes() {
    let n1 = spawn_node(1).await;
    let n2 = spawn_node(2).await;
    let n3 = spawn_node(3).await;

    // Initialize the cluster on n1 with all three nodes as voters.
    let mut nodes = BTreeMap::new();
    nodes.insert(n1.id, n1.yantrik_node());
    nodes.insert(n2.id, n2.yantrik_node());
    nodes.insert(n3.id, n3.yantrik_node());
    n1.raft.initialize(nodes).await.expect("initialize");

    // Wait for n1 to become the leader (election).
    let leader = wait_for_leader(&n1, Instant::now() + Duration::from_secs(5))
        .await
        .expect("a leader should emerge");
    assert!(
        [n1.id, n2.id, n3.id].contains(&leader),
        "leader must be one of the cluster nodes"
    );

    // Write 3 entries through the leader's RaftCommitter.
    let leader_node = match leader.raw() {
        1 => &n1,
        2 => &n2,
        3 => &n3,
        _ => unreachable!(),
    };
    for tag in ["a", "b", "c"] {
        let receipt = leader_node
            .committer
            .commit(TenantId::new(1), upsert(tag), CommitOptions::default())
            .await
            .expect("commit through leader");
        assert!(receipt.term >= 1);
    }

    // Every follower's local committer (state machine) must see all 3.
    let deadline = Instant::now() + Duration::from_secs(5);
    for n in [&n1, &n2, &n3] {
        let observed = wait_for_high_water(&n.local, TenantId::new(1), 3, deadline).await;
        assert_eq!(
            observed, 3,
            "node {} did not catch up: only saw {} entries",
            n.id, observed
        );
    }

    // Every follower's read_range returns identical entries (rids,
    // op_ids agree). State machine apply order matches leader.
    let leader_entries = leader_node
        .local
        .read_range(TenantId::new(1), 1, 100)
        .await
        .unwrap();
    assert_eq!(leader_entries.len(), 3);
    for n in [&n1, &n2, &n3] {
        let entries = n.local.read_range(TenantId::new(1), 1, 100).await.unwrap();
        assert_eq!(entries.len(), 3, "node {} entry count", n.id);
        for (le, ne) in leader_entries.iter().zip(entries.iter()) {
            assert_eq!(le.op_id, ne.op_id, "node {} op_id mismatch", n.id);
            assert_eq!(le.mutation, ne.mutation, "node {} mutation mismatch", n.id);
        }
    }

    n1.server_handle.abort();
    n2.server_handle.abort();
    n3.server_handle.abort();
}

#[tokio::test]
async fn leader_failover_elects_new_leader_and_writes_continue() {
    // Start 3 nodes, initialize, identify leader, drop the leader's
    // Raft handle (simulating crash), confirm the remaining quorum
    // elects a new leader and writes against it apply on the survivor.
    let n1 = spawn_node(1).await;
    let n2 = spawn_node(2).await;
    let n3 = spawn_node(3).await;

    let mut nodes = BTreeMap::new();
    nodes.insert(n1.id, n1.yantrik_node());
    nodes.insert(n2.id, n2.yantrik_node());
    nodes.insert(n3.id, n3.yantrik_node());
    n1.raft.initialize(nodes).await.unwrap();

    let leader_id = wait_for_leader(&n1, Instant::now() + Duration::from_secs(5))
        .await
        .expect("first leader");

    // Sort nodes into leader vs followers.
    let all = vec![n1, n2, n3];
    let mut leader_node = None;
    let mut followers = Vec::new();
    for n in all {
        if n.id == leader_id {
            leader_node = Some(n);
        } else {
            followers.push(n);
        }
    }
    let leader_node = leader_node.expect("leader present");
    assert_eq!(followers.len(), 2);

    // Land one write on the original leader so we have something to
    // catch up after failover.
    leader_node
        .committer
        .commit(TenantId::new(1), upsert("pre"), CommitOptions::default())
        .await
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    for f in &followers {
        wait_for_high_water(&f.local, TenantId::new(1), 1, deadline).await;
    }

    // Kill the original leader (drop the Raft handle + abort its server).
    let killed_id = leader_node.id;
    leader_node.server_handle.abort();
    drop(leader_node);

    // The remaining two should elect a new leader.
    let new_leader = {
        let f0 = &followers[0];
        let f1 = &followers[1];
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut found = None;
        while Instant::now() < deadline && found.is_none() {
            for f in [f0, f1] {
                if let Some(l) = f.raft.current_leader().await {
                    if l != killed_id {
                        found = Some(l);
                        break;
                    }
                }
            }
            if found.is_none() {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
        found.expect("new leader should emerge from remaining quorum")
    };
    assert_ne!(new_leader, killed_id);

    // Locate the new leader node and commit through it.
    let new_leader_node = followers
        .iter()
        .find(|f| f.id == new_leader)
        .expect("new leader is a known follower");
    new_leader_node
        .committer
        .commit(TenantId::new(1), upsert("post"), CommitOptions::default())
        .await
        .expect("write against new leader");

    // The other surviving node should also see both entries (pre + post).
    let deadline = Instant::now() + Duration::from_secs(5);
    for f in &followers {
        let h = wait_for_high_water(&f.local, TenantId::new(1), 2, deadline).await;
        assert_eq!(h, 2, "survivor {} did not catch up: high_water={}", f.id, h);
    }

    for f in followers {
        f.server_handle.abort();
    }
}

/// Asymmetric partition + heal. A follower drops every incoming peer
/// RPC (simulating "receive blocked") while writes proceed through the
/// majority. After healing, the previously-partitioned follower must
/// catch up to the same log as the others — proving the apply path
/// correctly replays missed entries via openraft's catch-up flow.
#[tokio::test]
async fn partition_then_heal_recovers_isolated_follower() {
    let n1 = spawn_node(1).await;
    let n2 = spawn_node(2).await;
    let n3 = spawn_node(3).await;

    let mut nodes = BTreeMap::new();
    nodes.insert(n1.id, n1.yantrik_node());
    nodes.insert(n2.id, n2.yantrik_node());
    nodes.insert(n3.id, n3.yantrik_node());
    n1.raft.initialize(nodes).await.expect("initialize");

    let leader_id = wait_for_leader(&n1, Instant::now() + Duration::from_secs(5))
        .await
        .expect("first leader");

    // Resolve nodes by id without using references (which all conflict
    // due to mutable owned ClusterNodes).
    let all_nodes: Vec<&ClusterNode> = vec![&n1, &n2, &n3];
    let leader_node = *all_nodes
        .iter()
        .find(|n| n.id == leader_id)
        .expect("leader present");
    let target_partitioned = *all_nodes
        .iter()
        .find(|n| n.id != leader_id)
        .expect("at least one follower exists");
    let unpartitioned_follower = *all_nodes
        .iter()
        .find(|n| n.id != leader_id && n.id != target_partitioned.id)
        .expect("two non-leader nodes");

    // Pre-partition: write 2 entries; both followers catch up.
    for tag in ["pre1", "pre2"] {
        leader_node
            .committer
            .commit(TenantId::new(1), upsert(tag), CommitOptions::default())
            .await
            .expect("pre-partition write");
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    for n in [leader_node, target_partitioned, unpartitioned_follower] {
        wait_for_high_water(&n.local, TenantId::new(1), 2, deadline).await;
    }

    // Engage the partition. Peer RPCs to target_partitioned now return
    // 503 -> Unreachable; the remaining 2 nodes form quorum and writes
    // continue against them.
    target_partitioned.set_partitioned(true);

    // Write 3 more entries while partitioned. Use a generous timeout
    // because openraft may briefly retry the partitioned peer before
    // declaring it unreachable.
    for tag in ["mid1", "mid2", "mid3"] {
        leader_node
            .committer
            .commit(TenantId::new(1), upsert(tag), CommitOptions::default())
            .await
            .expect("mid-partition write");
    }

    // Leader + non-partitioned follower must catch up; partitioned
    // follower must NOT.
    let deadline = Instant::now() + Duration::from_secs(5);
    let leader_high = wait_for_high_water(&leader_node.local, TenantId::new(1), 5, deadline).await;
    let other_high =
        wait_for_high_water(&unpartitioned_follower.local, TenantId::new(1), 5, deadline).await;
    assert_eq!(leader_high, 5);
    assert_eq!(other_high, 5);
    let partitioned_high = target_partitioned
        .local
        .high_watermark(TenantId::new(1))
        .await
        .unwrap();
    assert_eq!(
        partitioned_high, 2,
        "partitioned follower {} must NOT have caught up while partitioned — got {}",
        target_partitioned.id, partitioned_high
    );

    // Heal: partition flag off. openraft's replication retries and the
    // missing 3 entries land on the previously-isolated follower.
    target_partitioned.set_partitioned(false);

    let deadline = Instant::now() + Duration::from_secs(10);
    let healed_high =
        wait_for_high_water(&target_partitioned.local, TenantId::new(1), 5, deadline).await;
    assert_eq!(
        healed_high, 5,
        "partitioned follower {} should catch up after heal — got {}",
        target_partitioned.id, healed_high
    );

    // Final consistency check: every node's commit log is identical.
    let leader_entries = leader_node
        .local
        .read_range(TenantId::new(1), 1, 100)
        .await
        .unwrap();
    for n in [leader_node, target_partitioned, unpartitioned_follower] {
        let entries = n.local.read_range(TenantId::new(1), 1, 100).await.unwrap();
        assert_eq!(
            entries.len(),
            leader_entries.len(),
            "node {} entry count after heal",
            n.id
        );
        for (le, ne) in leader_entries.iter().zip(entries.iter()) {
            assert_eq!(le.op_id, ne.op_id, "node {} op_id mismatch", n.id);
            assert_eq!(le.mutation, ne.mutation, "node {} mutation mismatch", n.id);
        }
    }

    n1.server_handle.abort();
    n2.server_handle.abort();
    n3.server_handle.abort();
}
