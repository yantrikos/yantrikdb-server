//! `Submitter` — the trait that durably appends a mutation and triggers apply.
//!
//! ## Why this exists (RFC 010 PR-6 §1)
//!
//! `MutationCommitter` (PR-1 through PR-5) conflates log-append and
//! state-apply. `Submitter` is the half that owns log-append; `Applier`
//! ([`super::applier::Applier`]) is the half that owns state-apply. The
//! split lets the same Applier implementation drive engine state on
//! both single-node and cluster nodes.
//!
//! ## Implementations
//!
//! - [`LocalSqliteSubmitter`] — single-node. Wraps the existing
//!   [`LocalSqliteCommitter`] for log-append and dispatches to a
//!   companion [`Applier`] for state mutation.
//! - `RaftSubmitter` (RFC 010 PR-6.4) — cluster. Calls openraft's
//!   `client_write`. The state machine's `apply_to_state_machine`
//!   callback drives Applier::apply on every node.
//!
//! ## Scope of PR 6.1 (this file)
//!
//! Trait shape + a `LocalSqliteSubmitter` that delegates the durable
//! log-append to today's `LocalSqliteCommitter` and then dispatches
//! to its companion `Applier`. Production handlers DO NOT use this
//! yet — they continue to call `engine.record()` directly until PR 6.4.
//!
//! The trait is intentionally a strict superset of `MutationCommitter`'s
//! shape so RaftSubmitter can drop into the same `Arc<dyn Submitter>`
//! slot in PR 6.4 without changing handler signatures.

use std::sync::Arc;

use async_trait::async_trait;

use super::applier::{Applier, ApplyError};
use super::local::LocalSqliteCommitter;
use super::mutation::{MemoryMutation, TenantId};
use super::trait_def::{
    CommitError, CommitOptions, CommitReceipt, CommittedEntry, MutationCommitter,
};

/// Trait every commit-substrate frontend implements.
///
/// Single-node uses [`LocalSqliteSubmitter`]; cluster uses `RaftSubmitter`
/// (PR 6.4). Handlers hold an `Arc<dyn Submitter>` and don't care which.
///
/// All methods mirror [`MutationCommitter`] so the migration in PR 6.4
/// is a type-rename for handlers, not a logic change.
#[async_trait]
pub trait Submitter: Send + Sync {
    /// Durably append the mutation to the per-tenant commit log and
    /// (if `wait_for_apply=true`, the default) trigger apply against
    /// engine state. Returns once the receipt is final.
    async fn submit(
        &self,
        tenant_id: TenantId,
        mutation: MemoryMutation,
        opts: CommitOptions,
    ) -> Result<CommitReceipt, CommitError>;

    /// Read a range of committed entries for a tenant. Used for replay,
    /// audit, debug-history, and Jepsen fault-injection paths.
    async fn read_range(
        &self,
        tenant_id: TenantId,
        from_index: u64,
        limit: usize,
    ) -> Result<Vec<CommittedEntry>, CommitError>;

    /// High watermark — max `log_index` ever assigned for this tenant.
    async fn high_watermark(&self, tenant_id: TenantId) -> Result<u64, CommitError>;

    /// List every tenant id with at least one entry in the log.
    async fn list_active_tenants(&self) -> Result<Vec<TenantId>, CommitError>;

    /// Establish a linearizability barrier for subsequent reads.
    /// Default: trivially Ok on single-node where every read is
    /// linearizable. RaftSubmitter (PR 6.4) overrides via
    /// `Raft::ensure_linearizable`.
    async fn ensure_linearizable(&self) -> Result<(), CommitError> {
        Ok(())
    }
}

/// Single-node Submitter that composes a [`LocalSqliteCommitter`] for
/// the durable log with a companion [`Applier`] for engine state.
///
/// On `submit`:
/// 1. Delegate to `committer.commit(...)` to durably append.
/// 2. If `wait_for_apply=true` (the default), call `applier.apply(...)`.
/// 3. Return the receipt unchanged.
///
/// The Applier's `AlreadyApplied` is treated as success — replay
/// during startup or snapshot-install is normal.
pub struct LocalSqliteSubmitter {
    committer: Arc<LocalSqliteCommitter>,
    applier: Arc<dyn Applier>,
}

impl LocalSqliteSubmitter {
    pub fn new(committer: Arc<LocalSqliteCommitter>, applier: Arc<dyn Applier>) -> Self {
        Self { committer, applier }
    }

    /// Access the underlying committer for tests and bootstrap paths
    /// that pre-existed Submitter (e.g. RFC 011 forget integration).
    pub fn committer(&self) -> &Arc<LocalSqliteCommitter> {
        &self.committer
    }
}

#[async_trait]
impl Submitter for LocalSqliteSubmitter {
    async fn submit(
        &self,
        tenant_id: TenantId,
        mutation: MemoryMutation,
        opts: CommitOptions,
    ) -> Result<CommitReceipt, CommitError> {
        let wait_for_apply = opts.wait_for_apply;
        let receipt = self
            .committer
            .commit(tenant_id, mutation.clone(), opts)
            .await?;

        if wait_for_apply {
            match self
                .applier
                .apply(tenant_id, receipt.log_index, &mutation)
                .await
            {
                Ok(()) => {}
                // PR 6.1 scope: LocalApplier returns NotYetWired for
                // every variant because real engine wiring is PR 6.4.
                // Tolerate that here so existing handlers can be
                // migrated to use Submitter without breaking.
                Err(ApplyError::NotYetWired { .. }) => {}
                // Replay during boot/snapshot is normal — caller MAY
                // treat as success per the Applier contract.
                Err(e) if e.is_idempotent_ok() => {}
                Err(e) => {
                    return Err(CommitError::StorageFailure {
                        message: format!("apply failed for log_index {}: {e}", receipt.log_index),
                    });
                }
            }
        }

        Ok(receipt)
    }

    async fn read_range(
        &self,
        tenant_id: TenantId,
        from_index: u64,
        limit: usize,
    ) -> Result<Vec<CommittedEntry>, CommitError> {
        self.committer
            .read_range(tenant_id, from_index, limit)
            .await
    }

    async fn high_watermark(&self, tenant_id: TenantId) -> Result<u64, CommitError> {
        self.committer.high_watermark(tenant_id).await
    }

    async fn list_active_tenants(&self) -> Result<Vec<TenantId>, CommitError> {
        self.committer.list_active_tenants().await
    }

    async fn ensure_linearizable(&self) -> Result<(), CommitError> {
        self.committer.ensure_linearizable().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::applier::LocalApplier;
    use crate::commit::mutation::OpId;
    use serde_json::json;

    fn upsert_memory(rid: &str) -> MemoryMutation {
        MemoryMutation::UpsertMemory {
            rid: rid.to_string(),
            text: "hello".into(),
            memory_type: "semantic".into(),
            importance: 0.5,
            valence: 0.0,
            half_life: 86400.0,
            namespace: "test".into(),
            certainty: 1.0,
            domain: "general".into(),
            source: "test".into(),
            emotional_state: None,
            embedding: None,
            extracted_entities: vec![],
            created_at_unix_micros: None,
            embedding_model: None,
            metadata: json!({}),
        }
    }

    fn build_submitter() -> LocalSqliteSubmitter {
        let committer = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let applier: Arc<dyn Applier> = Arc::new(LocalApplier::new());
        LocalSqliteSubmitter::new(committer, applier)
    }

    #[tokio::test]
    async fn submit_round_trips_through_log() {
        // Submitter::submit must produce a receipt observable via
        // read_range — the durable log is the source of truth.
        let s = build_submitter();
        let r = s
            .submit(
                TenantId::new(1),
                upsert_memory("rid-1"),
                CommitOptions::new(),
            )
            .await
            .unwrap();
        assert_eq!(r.log_index, 1);
        assert_eq!(r.tenant_id, TenantId::new(1));
        let entries = s.read_range(TenantId::new(1), 1, 10).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].log_index, 1);
    }

    #[tokio::test]
    async fn submit_assigns_monotonic_per_tenant_log_index() {
        let s = build_submitter();
        let r1 = s
            .submit(
                TenantId::new(1),
                upsert_memory("rid-1"),
                CommitOptions::new(),
            )
            .await
            .unwrap();
        let r2 = s
            .submit(
                TenantId::new(1),
                upsert_memory("rid-2"),
                CommitOptions::new(),
            )
            .await
            .unwrap();
        let r3 = s
            .submit(
                TenantId::new(2),
                upsert_memory("rid-3"),
                CommitOptions::new(),
            )
            .await
            .unwrap();
        // Per-tenant: tenant 1 gets 1, 2; tenant 2 starts fresh at 1.
        assert_eq!(r1.log_index, 1);
        assert_eq!(r2.log_index, 2);
        assert_eq!(r3.log_index, 1);
    }

    #[tokio::test]
    async fn submit_is_idempotent_on_op_id() {
        // Repeating the same (tenant_id, op_id, mutation) returns the
        // original receipt. This is the contract that lets HTTP retries
        // after network blips be safe.
        let s = build_submitter();
        let op_id = OpId::new_random();
        let opts = CommitOptions::new().with_op_id(op_id);

        let r1 = s
            .submit(TenantId::new(1), upsert_memory("rid-1"), opts.clone())
            .await
            .unwrap();
        let r2 = s
            .submit(TenantId::new(1), upsert_memory("rid-1"), opts)
            .await
            .unwrap();
        assert_eq!(r1.log_index, r2.log_index);
        assert_eq!(r1.op_id, r2.op_id);
        assert_eq!(r1.committed_at, r2.committed_at);

        // Only one log entry exists — duplicates dedupe.
        let entries = s.read_range(TenantId::new(1), 1, 10).await.unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[tokio::test]
    async fn high_watermark_tracks_per_tenant() {
        let s = build_submitter();
        assert_eq!(s.high_watermark(TenantId::new(1)).await.unwrap(), 0);
        let _ = s
            .submit(
                TenantId::new(1),
                upsert_memory("rid-1"),
                CommitOptions::new(),
            )
            .await
            .unwrap();
        let _ = s
            .submit(
                TenantId::new(1),
                upsert_memory("rid-2"),
                CommitOptions::new(),
            )
            .await
            .unwrap();
        assert_eq!(s.high_watermark(TenantId::new(1)).await.unwrap(), 2);
        assert_eq!(s.high_watermark(TenantId::new(2)).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn list_active_tenants_finds_every_writer() {
        let s = build_submitter();
        let _ = s
            .submit(
                TenantId::new(1),
                upsert_memory("rid-1"),
                CommitOptions::new(),
            )
            .await
            .unwrap();
        let _ = s
            .submit(
                TenantId::new(7),
                upsert_memory("rid-2"),
                CommitOptions::new(),
            )
            .await
            .unwrap();
        let _ = s
            .submit(
                TenantId::new(7),
                upsert_memory("rid-3"),
                CommitOptions::new(),
            )
            .await
            .unwrap();
        let mut tenants = s.list_active_tenants().await.unwrap();
        tenants.sort();
        assert_eq!(tenants, vec![TenantId::new(1), TenantId::new(7)]);
    }

    #[tokio::test]
    async fn ensure_linearizable_is_trivially_ok_on_single_node() {
        // Single-node: every read is trivially linearizable. PR 6.4's
        // RaftSubmitter overrides this to call openraft's quorum check.
        let s = build_submitter();
        s.ensure_linearizable().await.unwrap();
    }

    #[tokio::test]
    async fn no_wait_does_not_call_applier() {
        // wait_for_apply=false skips the applier dispatch — used by the
        // bulk-import path. The receipt's applied_at reflects that:
        // LocalSqliteCommitter sets None when wait_for_apply=false.
        let s = build_submitter();
        let r = s
            .submit(
                TenantId::new(1),
                upsert_memory("rid-1"),
                CommitOptions::new().no_wait(),
            )
            .await
            .unwrap();
        assert!(r.applied_at.is_none());
    }

    #[tokio::test]
    async fn submitter_committer_accessor_works() {
        // The committer accessor exists for bootstrap paths that
        // predate Submitter (forget integration, retention, etc.).
        // Until those migrate to Submitter, they need raw access.
        let s = build_submitter();
        let _committer = s.committer();
    }

    // Compile-time pin: Submitter trait is dyn-compatible (object safe).
    // Handler code in PR 6.4 will hold `Arc<dyn Submitter>` so this MUST
    // compile. If a future trait method takes a non-dyn-safe shape this
    // breaks — fix the trait, not this test.
    #[allow(dead_code)]
    fn _dyn_submitter_compile_check(_s: Arc<dyn Submitter>) {}
}
