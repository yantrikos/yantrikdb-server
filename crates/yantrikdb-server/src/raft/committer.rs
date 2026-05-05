//! `RaftCommitter` — implements [`crate::commit::MutationCommitter`] by
//! routing every write through openraft's `client_write` so the
//! application gateway code (`/v1/remember`, `/v1/forget`, etc.)
//! doesn't have to know whether it's running single-node or in a Raft
//! cluster.
//!
//! ## Design
//!
//! `RaftCommitter` holds:
//! - `raft: Arc<openraft::Raft<YantrikRaftTypeConfig>>` — the assembled
//!   Raft instance. Writes go through `raft.client_write(YantrikLogEntry)`.
//! - `local: Arc<dyn MutationCommitter>` — the local committer that
//!   backs the state-machine apply path. Reads (`read_range`,
//!   `high_watermark`) delegate here because the state machine has
//!   already committed applied entries into this exact local log.
//!
//! ## Why reads go to the local committer instead of through Raft
//!
//! In Raft, follower reads are not linearizable unless gated through
//! the leader (or via lease/read-index protocols). For Phase 2 we
//! intentionally serve reads from the local applied state — that's
//! "stale reads OK" semantics, which matches what the existing
//! `LocalSqliteCommitter` callers were already getting in single-node
//! mode. PR-4-d-b can layer linearizable reads via openraft's
//! `Raft::ensure_linearizable()` once the network exists.
//!
//! ## Sub-PR layout for openraft adoption
//!
//! - **PR-4-a**: types
//! - **PR-4-b**: `SqliteRaftLogStorage`
//! - **PR-4-c**: `YantrikStateMachine` + snapshot
//! - **PR-4-d-a (this)**: `RaftCommitter` + `StubRaftNetwork` (single-node only)
//! - **PR-4-d-b**: `HttpRaftNetwork` + axum receive routes (real cluster transport)
//! - **PR-4-e**: 3-node integration test, leader-kill failover, partition heal

use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use openraft::error::{CheckIsLeaderError, ClientWriteError, ForwardToLeader, RaftError};
use openraft::Raft;

use super::types::{YantrikLogEntry, YantrikNode, YantrikNodeId, YantrikRaftTypeConfig};
use crate::commit::{
    CommitError, CommitOptions, CommitReceipt, CommittedEntry, MemoryMutation, MutationCommitter,
    OpId, TenantId,
};

/// Committer that replicates writes through openraft. Cheap to clone —
/// the Raft instance and the local committer are both `Arc`-shared.
#[derive(Clone)]
pub struct RaftCommitter {
    raft: Arc<Raft<YantrikRaftTypeConfig>>,
    /// Local committer that backs the state machine. Reads delegate
    /// here so they reflect the state machine's applied view.
    local: Arc<dyn MutationCommitter>,
}

impl RaftCommitter {
    pub fn new(raft: Arc<Raft<YantrikRaftTypeConfig>>, local: Arc<dyn MutationCommitter>) -> Self {
        Self { raft, local }
    }

    fn map_raft_error(
        err: RaftError<YantrikNodeId, ClientWriteError<YantrikNodeId, YantrikNode>>,
    ) -> CommitError {
        // ForwardToLeader is the most common Raft error in cluster mode
        // (a client lands on a follower). Surface it as `NotLeader` so
        // the HTTP gateway / CLI can redirect rather than treating it
        // as a storage failure.
        match err {
            RaftError::APIError(ClientWriteError::ForwardToLeader(ForwardToLeader {
                leader_id,
                leader_node,
            })) => CommitError::NotLeader {
                leader_id: leader_id.map(|id| id.raw()),
                leader_addr: leader_node.map(|n| n.addr),
            },
            other => CommitError::StorageFailure {
                message: format!("raft client_write: {other}"),
            },
        }
    }

    fn map_check_is_leader_error(
        err: RaftError<YantrikNodeId, CheckIsLeaderError<YantrikNodeId, YantrikNode>>,
    ) -> CommitError {
        match err {
            RaftError::APIError(CheckIsLeaderError::ForwardToLeader(ForwardToLeader {
                leader_id,
                leader_node,
            })) => CommitError::NotLeader {
                leader_id: leader_id.map(|id| id.raw()),
                leader_addr: leader_node.map(|n| n.addr),
            },
            other => CommitError::StorageFailure {
                message: format!("raft ensure_linearizable: {other}"),
            },
        }
    }
}

#[async_trait]
impl MutationCommitter for RaftCommitter {
    async fn commit(
        &self,
        tenant_id: TenantId,
        mutation: MemoryMutation,
        opts: CommitOptions,
    ) -> Result<CommitReceipt, CommitError> {
        // Reject grammar-only variants up-front (same gate as
        // LocalSqliteCommitter — replicating an unimplemented variant
        // would just fail in apply on every follower).
        if !mutation.is_implemented() {
            return Err(CommitError::NotYetImplemented {
                variant: mutation.variant_name(),
                planned_rfc: mutation.planned_rfc(),
            });
        }

        let op_id = opts.op_id.unwrap_or_else(OpId::new_random);
        let entry = YantrikLogEntry::new(tenant_id, op_id, mutation);

        let response = self
            .raft
            .client_write(entry)
            .await
            .map_err(Self::map_raft_error)?;

        // ClientWriteResponse carries the log_id (Raft term + index)
        // and the application-defined data (YantrikRaftResponse).
        let raft_log_id = response.log_id;
        let app = response.data;
        let applied_at = SystemTime::UNIX_EPOCH
            .checked_add(std::time::Duration::from_micros(
                app.applied_at_unix_micros.max(0) as u64,
            ))
            .unwrap_or(SystemTime::UNIX_EPOCH);

        Ok(CommitReceipt {
            op_id,
            tenant_id,
            term: raft_log_id.leader_id.term,
            log_index: app.tenant_log_index,
            committed_at: applied_at,
            applied_at: Some(applied_at),
        })
    }

    async fn read_range(
        &self,
        tenant_id: TenantId,
        from_index: u64,
        limit: usize,
    ) -> Result<Vec<CommittedEntry>, CommitError> {
        // Stale-read OK: the local committer has every entry the state
        // machine has applied on this node.
        self.local.read_range(tenant_id, from_index, limit).await
    }

    async fn high_watermark(&self, tenant_id: TenantId) -> Result<u64, CommitError> {
        self.local.high_watermark(tenant_id).await
    }

    async fn list_active_tenants(&self) -> Result<Vec<TenantId>, CommitError> {
        self.local.list_active_tenants().await
    }

    async fn ensure_linearizable(&self) -> Result<(), CommitError> {
        // openraft heartbeats a quorum to confirm leadership and waits
        // for the state machine to apply up to the read-barrier index.
        // After this returns Ok, a subsequent read from the local
        // committer is linearizable across the cluster.
        self.raft
            .ensure_linearizable()
            .await
            .map_err(Self::map_check_is_leader_error)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::LocalSqliteCommitter;
    use crate::raft::log_storage::SqliteRaftLogStorage;
    use crate::raft::network::StubRaftNetworkFactory;
    use crate::raft::state_machine::YantrikStateMachine;
    use crate::raft::types::YantrikNode;
    use openraft::Config;
    use std::collections::BTreeMap;

    /// Build a single-node Raft cluster with everything wired up.
    /// Returns the assembled committer + the inner local committer
    /// (so tests can verify the apply path landed entries durably).
    async fn build_single_node_committer() -> (RaftCommitter, Arc<LocalSqliteCommitter>) {
        let local = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let log_store = SqliteRaftLogStorage::open_in_memory();
        let state_machine = YantrikStateMachine::new(local.clone());
        let network = StubRaftNetworkFactory;

        let config = Arc::new(
            Config {
                cluster_name: "yantrikdb-test".into(),
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

        // Initialize as a single-node cluster.
        let mut nodes = BTreeMap::new();
        nodes.insert(me, YantrikNode::new("http://127.0.0.1:0"));
        raft.initialize(nodes).await.unwrap();

        // Wait for leadership — single-node should win immediately.
        for _ in 0..30 {
            if raft.current_leader().await == Some(me) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert_eq!(raft.current_leader().await, Some(me), "should be leader");

        (RaftCommitter::new(raft, local.clone()), local)
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
    async fn single_node_commit_round_trips_through_raft() {
        let (cm, local) = build_single_node_committer().await;
        let receipt = cm
            .commit(TenantId::new(1), upsert("a"), CommitOptions::default())
            .await
            .unwrap();
        assert_eq!(receipt.tenant_id, TenantId::new(1));
        assert_eq!(receipt.log_index, 1);
        assert!(receipt.term >= 1, "Raft term should be >= 1");
        assert!(receipt.applied_at.is_some());

        // The local committer (state machine apply path) saw the entry.
        assert_eq!(local.high_watermark(TenantId::new(1)).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn unimplemented_variant_is_rejected_before_raft() {
        let (cm, _) = build_single_node_committer().await;
        let cfg = MemoryMutation::TenantConfigPatch {
            key: "k".into(),
            value: serde_json::Value::Null,
        };
        let err = cm
            .commit(TenantId::new(1), cfg, CommitOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, CommitError::NotYetImplemented { .. }));
    }

    #[tokio::test]
    async fn read_range_delegates_to_local_committer() {
        let (cm, _) = build_single_node_committer().await;
        let _ = cm
            .commit(TenantId::new(1), upsert("a"), CommitOptions::default())
            .await
            .unwrap();
        let _ = cm
            .commit(TenantId::new(1), upsert("b"), CommitOptions::default())
            .await
            .unwrap();
        let entries = cm.read_range(TenantId::new(1), 1, 100).await.unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[tokio::test]
    async fn high_watermark_reflects_committed_entries() {
        let (cm, _) = build_single_node_committer().await;
        for i in 0..3 {
            let _ = cm
                .commit(
                    TenantId::new(1),
                    upsert(&format!("e{i}")),
                    CommitOptions::default(),
                )
                .await
                .unwrap();
        }
        assert_eq!(cm.high_watermark(TenantId::new(1)).await.unwrap(), 3);
    }

    #[tokio::test]
    async fn idempotent_op_id_returns_same_log_index() {
        let (cm, _) = build_single_node_committer().await;
        let op_id = OpId::new_random();
        let r1 = cm
            .commit(
                TenantId::new(1),
                upsert("a"),
                CommitOptions::default().with_op_id(op_id),
            )
            .await
            .unwrap();
        // The committer's apply path uses the same op_id, so the second
        // call would short-circuit on the (tenant, op_id) UNIQUE index
        // inside LocalSqliteCommitter — both receipts agree on log_index.
        let r2 = cm
            .commit(
                TenantId::new(1),
                upsert("a"),
                CommitOptions::default().with_op_id(op_id),
            )
            .await
            .unwrap();
        assert_eq!(r1.log_index, r2.log_index);
    }

    #[tokio::test]
    async fn per_tenant_log_index_is_independent() {
        let (cm, _) = build_single_node_committer().await;
        let r1 = cm
            .commit(TenantId::new(1), upsert("a"), CommitOptions::default())
            .await
            .unwrap();
        let r2 = cm
            .commit(TenantId::new(2), upsert("b"), CommitOptions::default())
            .await
            .unwrap();
        // Per-tenant log indices both start at 1 even though Raft term
        // and global log_index are shared.
        assert_eq!(r1.log_index, 1);
        assert_eq!(r2.log_index, 1);
    }

    #[tokio::test]
    async fn list_active_tenants_returns_all_used_ids() {
        let (cm, _) = build_single_node_committer().await;
        // Tenants 1, 3, 7 — verify list_active_tenants surfaces exactly these.
        for t in [1, 3, 7] {
            let _ = cm
                .commit(TenantId::new(t), upsert("a"), CommitOptions::default())
                .await
                .unwrap();
        }
        let mut tenants = cm.list_active_tenants().await.unwrap();
        tenants.sort();
        assert_eq!(
            tenants,
            vec![TenantId::new(1), TenantId::new(3), TenantId::new(7)]
        );
    }

    #[tokio::test]
    async fn forward_to_leader_translates_to_not_leader() {
        // Map an explicit ForwardToLeader RaftError to confirm the
        // translation surfaces leader_id + leader_addr — without
        // standing up a full multi-node cluster (the 3-node cluster
        // integration test exercises the live path).
        let ftl = ForwardToLeader {
            leader_id: Some(YantrikNodeId::new(7)),
            leader_node: Some(YantrikNode::new("http://10.0.0.5:7100")),
        };
        let err = RaftError::APIError(ClientWriteError::ForwardToLeader(ftl));
        let mapped = RaftCommitter::map_raft_error(err);
        match mapped {
            CommitError::NotLeader {
                leader_id,
                leader_addr,
            } => {
                assert_eq!(leader_id, Some(7));
                assert_eq!(leader_addr.as_deref(), Some("http://10.0.0.5:7100"));
            }
            other => panic!("expected NotLeader, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn forward_to_leader_with_unknown_leader_yields_none_fields() {
        // Mid-election: openraft returns ForwardToLeader with leader_id
        // = None. Translation must preserve that None so the caller
        // surfaces "no leader yet" rather than a bogus address.
        let ftl = ForwardToLeader::<YantrikNodeId, YantrikNode>::empty();
        let err = RaftError::APIError(ClientWriteError::ForwardToLeader(ftl));
        let mapped = RaftCommitter::map_raft_error(err);
        match mapped {
            CommitError::NotLeader {
                leader_id,
                leader_addr,
            } => {
                assert_eq!(leader_id, None);
                assert_eq!(leader_addr, None);
            }
            other => panic!("expected NotLeader, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn ensure_linearizable_succeeds_on_single_node_leader() {
        // Single-node Raft is always trivially linearizable — the
        // leader is the only node, so heartbeating "a quorum" is just
        // confirming itself.
        let (cm, _) = build_single_node_committer().await;
        cm.ensure_linearizable()
            .await
            .expect("single-node leader must always be linearizable");
    }

    #[tokio::test]
    async fn ensure_linearizable_local_committer_is_no_op() {
        // The default trait impl is correct for LocalSqliteCommitter:
        // single-node has no replication, so every read is trivially
        // linearizable.
        let local = LocalSqliteCommitter::open_in_memory().unwrap();
        local
            .ensure_linearizable()
            .await
            .expect("local committer trivially linearizable");
    }

    #[tokio::test]
    async fn check_is_leader_error_translates_to_not_leader() {
        // Same pattern as ForwardToLeader → NotLeader for client_write.
        let ftl = ForwardToLeader {
            leader_id: Some(YantrikNodeId::new(3)),
            leader_node: Some(YantrikNode::new("http://10.0.0.3:7100")),
        };
        let err = RaftError::APIError(CheckIsLeaderError::ForwardToLeader(ftl));
        let mapped = RaftCommitter::map_check_is_leader_error(err);
        match mapped {
            CommitError::NotLeader {
                leader_id,
                leader_addr,
            } => {
                assert_eq!(leader_id, Some(3));
                assert_eq!(leader_addr.as_deref(), Some("http://10.0.0.3:7100"));
            }
            other => panic!("expected NotLeader, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn not_leader_is_classified_retryable_against_leader() {
        // Standing acceptance: NotLeader is retryable (against a
        // different node, the leader). Distinct from OpIdCollision
        // which is a client bug and NOT retryable.
        let err = CommitError::NotLeader {
            leader_id: Some(2),
            leader_addr: Some("http://x".into()),
        };
        assert!(err.is_retryable());
        assert_eq!(err.metric_label(), "not_leader");
    }
}
