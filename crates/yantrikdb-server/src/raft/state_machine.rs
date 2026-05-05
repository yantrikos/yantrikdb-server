//! `YantrikStateMachine` — implements openraft's `RaftStateMachine`
//! (storage-v2) by routing applied log entries through the existing
//! `MutationCommitter` so the server's per-tenant `memory_commit_log`
//! stays the source of truth even in cluster mode.
//!
//! ## Apply-path design
//!
//! Each Raft log entry has one of three payload variants:
//! - **`Normal(YantrikLogEntry)`** — application data. We extract
//!   `(tenant_id, op_id, mutation)` and call
//!   `MutationCommitter::commit` so the per-tenant commit log advances.
//!   The `op_id` carried end-to-end gives client retries the same
//!   receipt across leader failover.
//! - **`Membership(...)`** — config change. Stored in the state-machine
//!   `last_membership` field. No application-side side effect.
//! - **`Blank`** — openraft's term-promotion sentinel. No-op.
//!
//! The response we return per applied entry is a [`YantrikRaftResponse`]
//! carrying the Raft term + per-tenant log_index from the committer
//! receipt + apply timestamp. The `RaftCommitter` (PR-4-d) reconstructs
//! a full [`crate::commit::CommitReceipt`] from this.
//!
//! ## Snapshot strategy
//!
//! Snapshot data carries a JSON envelope:
//! ```text
//! { "version": 1,
//!   "tenants": { "<tenant_id>": [ <CommittedEntry>, <CommittedEntry>, ... ] },
//!   "last_applied_log_id": <LogId>,
//!   "last_membership": <StoredMembership> }
//! ```
//!
//! `build_snapshot` reads every tenant's full commit log via
//! `MutationCommitter::read_range` and serializes it. `install_snapshot`
//! atomically replaces the in-memory state machine state and replays
//! the committed entries through the committer's apply path so a fresh
//! follower comes up consistent with the leader.
//!
//! ## What's NOT in this PR
//!
//! - **Streaming snapshot reader** — the placeholder
//!   `SnapshotData = std::io::Cursor<Vec<u8>>` materializes the whole
//!   snapshot in memory. Streaming via SQLite checkpoint comes later.
//! - **Engine state in snapshots** — the engine crate (`yantrikdb`)
//!   keeps HNSW + entity edges separately. Those derive from
//!   `memory_commit_log` and will be reconciled via the RFC 013
//!   reconciler at startup once engine wiring lands. For now, snapshot
//!   round-trip rebuilds only the commit log; HNSW catches up via
//!   reconciliation.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::Arc;
use std::time::SystemTime;

use openraft::entry::{RaftEntry, RaftPayload};
use openraft::storage::{RaftSnapshotBuilder, RaftStateMachine};
use openraft::{
    AnyError, Entry, EntryPayload, LogId, Snapshot, SnapshotMeta, StorageError, StorageIOError,
    StoredMembership,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::types::{YantrikNodeId, YantrikRaftResponse, YantrikRaftTypeConfig};
use crate::commit::{
    CommitOptions, CommittedEntry, MemoryMutation, MutationCommitter, OpId, TenantId,
};

/// Internal mutable state — log id watermark + membership. Kept
/// separate from the underlying committer so cloning the state machine
/// (for snapshot building) doesn't deep-copy the committer.
#[derive(Debug, Clone, Default)]
pub struct StateMachineState {
    pub last_applied_log_id: Option<LogId<YantrikNodeId>>,
    pub last_membership: StoredMembership<YantrikNodeId, super::YantrikNode>,
}

/// State machine that routes applied entries through a
/// `MutationCommitter`. Cheap to clone — internal state is `Arc`-shared.
#[derive(Clone)]
pub struct YantrikStateMachine {
    committer: Arc<dyn MutationCommitter>,
    state: Arc<Mutex<StateMachineState>>,
}

impl YantrikStateMachine {
    pub fn new(committer: Arc<dyn MutationCommitter>) -> Self {
        Self {
            committer,
            state: Arc::new(Mutex::new(StateMachineState::default())),
        }
    }

    /// Snapshot the committer state across every active tenant. Uses
    /// `list_active_tenants()` so cost is O(active_tenant_count) SQL
    /// queries — not O(MAX_TENANTS) as the prior linear-probe version
    /// did.
    async fn drain_committer_to_snapshot(
        &self,
    ) -> Result<BTreeMap<i64, Vec<CommittedEntry>>, StorageError<YantrikNodeId>> {
        let mut out: BTreeMap<i64, Vec<CommittedEntry>> = BTreeMap::new();
        let tenants = self.committer.list_active_tenants().await.map_err(|e| {
            StorageIOError::write_snapshot(
                None,
                AnyError::error(format!("list_active_tenants: {e}")),
            )
        })?;
        for tenant in tenants {
            let high = self.committer.high_watermark(tenant).await.map_err(|e| {
                StorageIOError::write_snapshot(
                    None,
                    AnyError::error(format!("high_watermark tenant {tenant}: {e}")),
                )
            })?;
            if high == 0 {
                continue;
            }
            let entries = self
                .committer
                .read_range(tenant, 1, high as usize)
                .await
                .map_err(|e| {
                    StorageIOError::write_snapshot(
                        None,
                        AnyError::error(format!("read_range tenant {tenant}: {e}")),
                    )
                })?;
            if !entries.is_empty() {
                out.insert(tenant.0, entries);
            }
        }
        Ok(out)
    }

    fn build_snapshot_id(meta_last: &Option<LogId<YantrikNodeId>>) -> String {
        let now = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u128)
            .unwrap_or(0);
        match meta_last {
            Some(lid) => format!("snap-{}-{}-{}", lid.leader_id.term, lid.index, now),
            None => format!("snap-empty-{}", now),
        }
    }
}

/// On-wire snapshot envelope. JSON-serialized into the `SnapshotData`
/// (a `Cursor<Vec<u8>>`).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotEnvelope {
    /// Increment when the envelope schema changes; until v1 ships
    /// elsewhere this is always 1.
    version: u32,
    /// Tenant id → committed entries in log_index order.
    tenants: BTreeMap<i64, Vec<CommittedEntry>>,
    last_applied_log_id: Option<LogId<YantrikNodeId>>,
    last_membership: StoredMembership<YantrikNodeId, super::YantrikNode>,
}

#[derive(Clone)]
pub struct YantrikSnapshotBuilder {
    inner: YantrikStateMachine,
}

impl RaftSnapshotBuilder<YantrikRaftTypeConfig> for YantrikSnapshotBuilder {
    async fn build_snapshot(
        &mut self,
    ) -> Result<Snapshot<YantrikRaftTypeConfig>, StorageError<YantrikNodeId>> {
        let tenants = self.inner.drain_committer_to_snapshot().await?;
        let (last_applied, last_membership) = {
            let s = self.inner.state.lock();
            (s.last_applied_log_id.clone(), s.last_membership.clone())
        };
        let envelope = SnapshotEnvelope {
            version: 1,
            tenants,
            last_applied_log_id: last_applied.clone(),
            last_membership: last_membership.clone(),
        };
        let blob = serde_json::to_vec(&envelope).map_err(|e| {
            StorageIOError::write_snapshot(
                None,
                AnyError::error(format!("snapshot serialize: {e}")),
            )
        })?;
        let meta = SnapshotMeta {
            last_log_id: last_applied,
            last_membership,
            snapshot_id: YantrikStateMachine::build_snapshot_id(&envelope.last_applied_log_id),
        };
        Ok(Snapshot {
            meta,
            snapshot: Box::new(Cursor::new(blob)),
        })
    }
}

impl RaftStateMachine<YantrikRaftTypeConfig> for YantrikStateMachine {
    type SnapshotBuilder = YantrikSnapshotBuilder;

    async fn applied_state(
        &mut self,
    ) -> Result<
        (
            Option<LogId<YantrikNodeId>>,
            StoredMembership<YantrikNodeId, super::YantrikNode>,
        ),
        StorageError<YantrikNodeId>,
    > {
        let s = self.state.lock();
        Ok((s.last_applied_log_id.clone(), s.last_membership.clone()))
    }

    async fn apply<I>(
        &mut self,
        entries: I,
    ) -> Result<Vec<YantrikRaftResponse>, StorageError<YantrikNodeId>>
    where
        I: IntoIterator<Item = Entry<YantrikRaftTypeConfig>> + Send,
        I::IntoIter: Send,
    {
        let entries: Vec<_> = entries.into_iter().collect();
        let mut responses = Vec::with_capacity(entries.len());

        for entry in entries {
            let log_id = entry.log_id.clone();

            // Membership entries land in state but produce a no-op
            // response. openraft conventions: every entry must produce
            // exactly one response.
            if let Some(mship) = entry.get_membership() {
                let mut s = self.state.lock();
                s.last_membership = StoredMembership::new(Some(log_id.clone()), mship.clone());
                s.last_applied_log_id = Some(log_id.clone());
                drop(s);
                responses.push(YantrikRaftResponse::new(
                    log_id.leader_id.term,
                    0,
                    SystemTime::now(),
                ));
                continue;
            }

            // Otherwise dispatch on payload variant.
            let response = match entry.payload {
                EntryPayload::Normal(app) => {
                    let (tenant_id, op_id, mutation) = app.into_parts();
                    self.apply_normal(tenant_id, op_id, mutation, &log_id)
                        .await?
                }
                EntryPayload::Blank => {
                    // Term-promotion sentinel — no application effect.
                    YantrikRaftResponse::new(log_id.leader_id.term, 0, SystemTime::now())
                }
                EntryPayload::Membership(_) => {
                    // Already handled above via get_membership().
                    unreachable!("membership handled by entry.get_membership() branch")
                }
            };

            self.state.lock().last_applied_log_id = Some(log_id);
            responses.push(response);
        }

        Ok(responses)
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        YantrikSnapshotBuilder {
            inner: self.clone(),
        }
    }

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<YantrikNodeId>> {
        // openraft writes snapshot bytes into this cursor. We use a
        // fresh empty cursor; install_snapshot will deserialize from it.
        Ok(Box::new(Cursor::new(Vec::new())))
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<YantrikNodeId, super::YantrikNode>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<YantrikNodeId>> {
        let blob = snapshot.into_inner();
        let envelope: SnapshotEnvelope = serde_json::from_slice(&blob).map_err(|e| {
            StorageIOError::read_snapshot(
                Some(meta.signature()),
                AnyError::error(format!("snapshot deserialize: {e}")),
            )
        })?;
        if envelope.version != 1 {
            return Err(StorageIOError::read_snapshot(
                Some(meta.signature()),
                AnyError::error(format!(
                    "unsupported snapshot envelope version: {}",
                    envelope.version
                )),
            )
            .into());
        }

        // Replay entries through the committer's apply path so the
        // memory_commit_log on this node matches the leader's state at
        // snapshot time.
        for (tid, entries) in &envelope.tenants {
            let tenant = TenantId::new(*tid);
            for ce in entries {
                self.committer
                    .commit(
                        tenant,
                        ce.mutation.clone(),
                        CommitOptions::default().with_op_id(ce.op_id),
                    )
                    .await
                    .map_err(|e| {
                        StorageIOError::read_snapshot(
                            Some(meta.signature()),
                            AnyError::error(format!(
                                "snapshot replay tenant {tid} op_id {}: {e}",
                                ce.op_id
                            )),
                        )
                    })?;
            }
        }

        let mut s = self.state.lock();
        s.last_applied_log_id = envelope.last_applied_log_id;
        s.last_membership = envelope.last_membership;
        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<YantrikRaftTypeConfig>>, StorageError<YantrikNodeId>> {
        // Phase 1: rebuild on demand. Persistent snapshot caching is a
        // Phase 2 concern — until then, every call freshly reads the
        // committer.
        let mut builder = self.get_snapshot_builder().await;
        let snap = builder.build_snapshot().await?;
        Ok(Some(snap))
    }
}

impl YantrikStateMachine {
    async fn apply_normal(
        &self,
        tenant_id: TenantId,
        op_id: OpId,
        mutation: MemoryMutation,
        log_id: &LogId<YantrikNodeId>,
    ) -> Result<YantrikRaftResponse, StorageError<YantrikNodeId>> {
        // Forward to the committer with the op_id preserved so retries
        // are idempotent across leader failover.
        let receipt = self
            .committer
            .commit(
                tenant_id,
                mutation,
                CommitOptions::default().with_op_id(op_id),
            )
            .await
            .map_err(|e| {
                StorageIOError::apply(
                    log_id.clone(),
                    AnyError::error(format!("apply commit: {e}")),
                )
            })?;
        Ok(YantrikRaftResponse::new(
            log_id.leader_id.term,
            receipt.log_index,
            receipt.applied_at.unwrap_or_else(SystemTime::now),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::{LocalSqliteCommitter, MemoryMutation, OpId};
    use crate::raft::types::{YantrikLogEntry, YantrikNode};
    use openraft::{LeaderId, Membership};
    use std::collections::{BTreeMap as StdBTreeMap, BTreeSet};

    fn upsert_app(tenant: i64, rid: &str) -> YantrikLogEntry {
        YantrikLogEntry::new(
            TenantId::new(tenant),
            OpId::new_random(),
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
            },
        )
    }

    fn entry_normal(
        index: u64,
        term: u64,
        node: u64,
        app: YantrikLogEntry,
    ) -> Entry<YantrikRaftTypeConfig> {
        Entry {
            log_id: LogId::new(LeaderId::new(term, YantrikNodeId::new(node)), index),
            payload: EntryPayload::Normal(app),
        }
    }

    fn entry_blank(index: u64, term: u64) -> Entry<YantrikRaftTypeConfig> {
        Entry {
            log_id: LogId::new(LeaderId::new(term, YantrikNodeId::new(1)), index),
            payload: EntryPayload::Blank,
        }
    }

    fn entry_membership(index: u64, term: u64) -> Entry<YantrikRaftTypeConfig> {
        let mut nodes_set = BTreeSet::new();
        nodes_set.insert(YantrikNodeId::new(1));
        nodes_set.insert(YantrikNodeId::new(2));
        let mut nodes = StdBTreeMap::new();
        nodes.insert(YantrikNodeId::new(1), YantrikNode::new("http://n1"));
        nodes.insert(YantrikNodeId::new(2), YantrikNode::new("http://n2"));
        Entry {
            log_id: LogId::new(LeaderId::new(term, YantrikNodeId::new(1)), index),
            payload: EntryPayload::Membership(Membership::new(vec![nodes_set], nodes)),
        }
    }

    fn make_sm() -> YantrikStateMachine {
        let committer = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        YantrikStateMachine::new(committer)
    }

    #[tokio::test]
    async fn fresh_state_machine_reports_empty() {
        let mut sm = make_sm();
        let (last, mship) = sm.applied_state().await.unwrap();
        assert_eq!(last, None);
        assert!(mship.membership().get_joint_config().is_empty());
    }

    #[tokio::test]
    async fn apply_normal_advances_log_id_and_routes_to_committer() {
        let mut sm = make_sm();
        let entries = vec![
            entry_normal(1, 1, 1, upsert_app(1, "a")),
            entry_normal(2, 1, 1, upsert_app(1, "b")),
        ];
        let responses = sm.apply(entries).await.unwrap();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].term, 1);
        assert_eq!(responses[0].tenant_log_index, 1);
        assert_eq!(responses[1].tenant_log_index, 2);

        let (last, _m) = sm.applied_state().await.unwrap();
        assert_eq!(last.unwrap().index, 2);
    }

    #[tokio::test]
    async fn apply_blank_is_no_op_but_advances_watermark() {
        let mut sm = make_sm();
        let responses = sm.apply(vec![entry_blank(1, 5)]).await.unwrap();
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].term, 5);
        assert_eq!(responses[0].tenant_log_index, 0);

        let (last, _) = sm.applied_state().await.unwrap();
        assert_eq!(last.unwrap().index, 1);
    }

    #[tokio::test]
    async fn apply_membership_stores_config_and_does_not_call_committer() {
        let committer = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let mut sm = YantrikStateMachine::new(committer.clone());
        let entries = vec![entry_membership(1, 1)];
        let _ = sm.apply(entries).await.unwrap();

        let (last, mship) = sm.applied_state().await.unwrap();
        assert_eq!(last.unwrap().index, 1);
        // Membership has 2 nodes.
        let voters: Vec<_> = mship.voter_ids().collect();
        assert_eq!(voters.len(), 2);

        // Committer untouched: no entries committed for any tenant.
        assert_eq!(committer.high_watermark(TenantId::new(1)).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn apply_preserves_op_id_for_idempotent_replay() {
        // Same op_id twice → second call is idempotent at committer.
        let committer = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let mut sm = YantrikStateMachine::new(committer.clone());
        let app = upsert_app(1, "a");
        let op_id = app.op_id;
        let mutation = app.mutation.clone();

        let _ = sm.apply(vec![entry_normal(1, 1, 1, app)]).await.unwrap();

        // A "retry" of the same op_id shouldn't double-write.
        let app2 = YantrikLogEntry::new(TenantId::new(1), op_id, mutation);
        let _ = sm.apply(vec![entry_normal(2, 1, 1, app2)]).await.unwrap();

        // Committer high-watermark is 1 (idempotent re-commit returns
        // the existing receipt without inserting a new row).
        assert_eq!(committer.high_watermark(TenantId::new(1)).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn snapshot_round_trips_full_state() {
        // Apply 3 entries, build snapshot, install into fresh sm,
        // verify state matches.
        let mut sm = make_sm();
        let _ = sm
            .apply(vec![
                entry_normal(1, 1, 1, upsert_app(1, "a")),
                entry_normal(2, 1, 1, upsert_app(1, "b")),
                entry_normal(3, 1, 1, upsert_app(2, "c")),
            ])
            .await
            .unwrap();

        let mut builder = sm.get_snapshot_builder().await;
        let snap = builder.build_snapshot().await.unwrap();

        // Fresh state machine — bring it up via install_snapshot.
        let dest_committer = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let mut dest = YantrikStateMachine::new(dest_committer.clone());
        let cursor = snap.snapshot;
        dest.install_snapshot(&snap.meta, cursor).await.unwrap();

        // last_applied_log_id matches.
        let (last, _) = dest.applied_state().await.unwrap();
        assert_eq!(last.unwrap().index, 3);

        // Tenant 1 has 2 entries, tenant 2 has 1.
        assert_eq!(
            dest_committer
                .high_watermark(TenantId::new(1))
                .await
                .unwrap(),
            2
        );
        assert_eq!(
            dest_committer
                .high_watermark(TenantId::new(2))
                .await
                .unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn snapshot_includes_membership() {
        let mut sm = make_sm();
        let _ = sm.apply(vec![entry_membership(1, 1)]).await.unwrap();

        let mut builder = sm.get_snapshot_builder().await;
        let snap = builder.build_snapshot().await.unwrap();

        let dest_committer = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let mut dest = YantrikStateMachine::new(dest_committer);
        dest.install_snapshot(&snap.meta, snap.snapshot)
            .await
            .unwrap();

        let (_last, mship) = dest.applied_state().await.unwrap();
        let voters: Vec<_> = mship.voter_ids().collect();
        assert_eq!(voters.len(), 2);
    }

    #[tokio::test]
    async fn get_current_snapshot_rebuilds_on_demand() {
        let mut sm = make_sm();
        let _ = sm
            .apply(vec![entry_normal(1, 1, 1, upsert_app(1, "a"))])
            .await
            .unwrap();
        let snap = sm.get_current_snapshot().await.unwrap().unwrap();
        assert_eq!(snap.meta.last_log_id.unwrap().index, 1);
    }

    #[tokio::test]
    async fn install_snapshot_rejects_unknown_envelope_version() {
        let mut sm = make_sm();
        let bad_envelope = serde_json::json!({
            "version": 999,
            "tenants": {},
            "last_applied_log_id": null,
            "last_membership": StoredMembership::<YantrikNodeId, YantrikNode>::default(),
        });
        let blob = serde_json::to_vec(&bad_envelope).unwrap();
        let meta = SnapshotMeta::<YantrikNodeId, YantrikNode>::default();
        let err = sm
            .install_snapshot(&meta, Box::new(Cursor::new(blob)))
            .await
            .unwrap_err();
        // Should fail with a read-snapshot StorageError.
        assert!(format!("{err}").contains("envelope version"));
    }

    #[tokio::test]
    async fn empty_snapshot_round_trips_cleanly() {
        // A brand-new state machine produces a snapshot of an empty
        // state. Installing it elsewhere should not corrupt anything.
        let mut sm = make_sm();
        let mut builder = sm.get_snapshot_builder().await;
        let snap = builder.build_snapshot().await.unwrap();

        let dest_committer = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let mut dest = YantrikStateMachine::new(dest_committer.clone());
        dest.install_snapshot(&snap.meta, snap.snapshot)
            .await
            .unwrap();

        let (last, _) = dest.applied_state().await.unwrap();
        assert_eq!(last, None);
        assert_eq!(
            dest_committer
                .high_watermark(TenantId::new(1))
                .await
                .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn applied_state_reflects_each_apply_call() {
        let mut sm = make_sm();
        for i in 1..=5_u64 {
            let _ = sm.apply(vec![entry_blank(i, 1)]).await.unwrap();
            let (last, _) = sm.applied_state().await.unwrap();
            assert_eq!(last.unwrap().index, i);
        }
    }
}
