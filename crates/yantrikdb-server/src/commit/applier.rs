//! `Applier` — the trait that drives mutations into engine state.
//!
//! ## Why this exists (RFC 010 PR-6 §1)
//!
//! Today's `MutationCommitter::commit` conflates two responsibilities:
//! durably appending the mutation to a log, and applying it to engine
//! state. PR-6 splits these so the same Applier implementation runs on
//! both leader and follower.
//!
//! - On a single-node deployment, `LocalSqliteSubmitter::submit` calls
//!   `Applier::apply` synchronously after the log insert.
//! - On a cluster, `RaftSubmitter::submit` (PR 6.4) returns once the
//!   entry is durable on majority quorum; the openraft state machine's
//!   `apply_to_state_machine` callback then calls `Applier::apply` on
//!   every node — including followers that received the entry via
//!   append-entries.
//!
//! This is the structural enforcement of "every node applies the same
//! mutation to the same state." Combined with PR 6.2's deterministic
//! mutations (which carry materialized embedding + extracted entities +
//! server-assigned timestamps), follower apply is byte-deterministic.
//!
//! ## Scope of PR 6.1 (this file)
//!
//! PR 6.1 ships the trait shape and a placeholder `LocalApplier` that
//! tracks applied `(tenant_id, log_index)` pairs in-memory for
//! idempotency tests. The real engine wiring — calls into
//! `yantrikdb::YantrikDB` for `UpsertMemory`, `TombstoneMemory`,
//! `UpsertEntityEdge`, `DeleteEntityEdge` — lands in PR 6.4 (handler
//! migration), once the engine API surface lands the
//! `record_with_rid`-style methods that deterministic mutations need.
//!
//! Until PR 6.4: handlers continue to call `engine.record()` directly.
//! `Applier` is unused on the production hot path. The trait shape is
//! what PR 6.1 commits to.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use thiserror::Error;

use super::mutation::{MemoryMutation, TenantId};

/// Apply errors. Distinct from `CommitError` because apply is a
/// downstream phase: by the time we're applying, the entry is already
/// durable in the log. Any non-transient error here means the state
/// machine has diverged or is about to — callers SHOULD treat it as
/// cause for shutdown rather than retry.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ApplyError {
    /// The mutation variant exists in the grammar but the apply path
    /// isn't wired yet. PR 6.1 returns this for every variant; PR 6.4
    /// wires `UpsertMemory` / `TombstoneMemory` / `UpsertEntityEdge` /
    /// `DeleteEntityEdge` to real engine calls.
    #[error("apply path for `{variant}` not yet wired (planned in {planned_pr})")]
    NotYetWired {
        variant: &'static str,
        planned_pr: &'static str,
    },

    /// Replay-detection: this `(tenant_id, log_index)` was already
    /// applied. Idempotent — callers MAY treat this as success.
    /// `LocalApplier` returns this on duplicate `apply` calls.
    #[error("(tenant {tenant_id}, log_index {log_index}) already applied")]
    AlreadyApplied { tenant_id: TenantId, log_index: u64 },

    /// The mutation is well-formed but engine-side execution failed.
    /// In production this is a hard error: the log entry is durable
    /// but the state machine couldn't apply it. Treat as divergence
    /// risk and shut down the node rather than continuing.
    #[error("engine apply failed: {message}")]
    EngineFailure { message: String },
}

impl ApplyError {
    /// Stable label for metrics. No user data — safe for Prometheus.
    pub fn metric_label(&self) -> &'static str {
        match self {
            ApplyError::NotYetWired { .. } => "not_yet_wired",
            ApplyError::AlreadyApplied { .. } => "already_applied",
            ApplyError::EngineFailure { .. } => "engine_failure",
        }
    }

    /// `true` for errors that callers MAY safely treat as success
    /// (the apply effectively happened or doesn't need to happen).
    pub fn is_idempotent_ok(&self) -> bool {
        matches!(self, ApplyError::AlreadyApplied { .. })
    }
}

/// The trait every state-machine apply backend implements.
///
/// Contract:
/// - **Deterministic.** Given identical input mutation, every node
///   produces identical engine state. PR 6.2 enforces this at the
///   mutation grammar level (materialized embedding/entities/timestamps);
///   the Applier MUST NOT introduce nondeterminism (no random ids, no
///   wall-clock reads, no embedder calls).
/// - **Idempotent on `(tenant_id, log_index)`.** Replaying the same
///   entry yields the same state. Implementations track a high
///   watermark per tenant and return `ApplyError::AlreadyApplied`
///   for duplicates. Callers (snapshot replay, append-entries with
///   already-applied entries) treat that as success.
/// - **Errors are catastrophic.** Any non-`AlreadyApplied` error
///   diverges the state machine. Implementations SHOULD log loudly
///   and the caller SHOULD raise a node-level health alarm.
#[async_trait]
pub trait Applier: Send + Sync {
    /// Apply a single committed mutation to engine state.
    async fn apply(
        &self,
        tenant_id: TenantId,
        log_index: u64,
        mutation: &MemoryMutation,
    ) -> Result<(), ApplyError>;

    /// High watermark: the largest `log_index` this Applier has applied
    /// for `tenant_id`. Returns 0 if no entries have been applied.
    /// Used by snapshot/replay code to skip already-applied entries.
    async fn applied_high_watermark(&self, tenant_id: TenantId) -> Result<u64, ApplyError>;
}

/// In-memory replay-detection LocalApplier scaffold.
///
/// PR 6.1 scope: tracks `(tenant_id, log_index)` pairs that have been
/// applied so the idempotency contract is exercised by tests. Real
/// engine integration (`yantrikdb::YantrikDB::record_with_rid`,
/// `tombstone`, `upsert_entity_edge`, `delete_entity_edge`) lands in
/// PR 6.4.
///
/// Until then, every apply call returns `ApplyError::NotYetWired` for
/// the engine-mutating variants — except on duplicate replay, which
/// returns `AlreadyApplied` (caller treats as ok).
pub struct LocalApplier {
    /// Set of `(tenant_id, log_index)` pairs already seen by `apply`.
    /// Cheap in-memory cache; no persistence in PR 6.1 because the
    /// production hot path doesn't go through Applier yet.
    seen: Arc<Mutex<HashSet<(TenantId, u64)>>>,
}

impl Default for LocalApplier {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalApplier {
    pub fn new() -> Self {
        Self {
            seen: Arc::new(Mutex::new(HashSet::new())),
        }
    }
}

#[async_trait]
impl Applier for LocalApplier {
    async fn apply(
        &self,
        tenant_id: TenantId,
        log_index: u64,
        mutation: &MemoryMutation,
    ) -> Result<(), ApplyError> {
        // Replay detection first — duplicate apply is a normal case
        // (snapshot install, log replay) and must be a fast no-op.
        {
            let mut seen = self.seen.lock();
            if !seen.insert((tenant_id, log_index)) {
                return Err(ApplyError::AlreadyApplied {
                    tenant_id,
                    log_index,
                });
            }
        }

        // Real engine wiring lands in PR 6.4. Until then, the trait
        // shape is exercised but the engine isn't touched.
        Err(ApplyError::NotYetWired {
            variant: mutation.variant_name(),
            planned_pr: "RFC 010 PR-6.4",
        })
    }

    async fn applied_high_watermark(&self, tenant_id: TenantId) -> Result<u64, ApplyError> {
        let seen = self.seen.lock();
        Ok(seen
            .iter()
            .filter(|(t, _)| *t == tenant_id)
            .map(|(_, idx)| *idx)
            .max()
            .unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn tombstone(rid: &str) -> MemoryMutation {
        MemoryMutation::TombstoneMemory {
            rid: rid.to_string(),
            reason: None,
            requested_at_unix_micros: 0,
            namespace: String::new(),
        }
    }

    #[tokio::test]
    async fn first_apply_returns_not_yet_wired() {
        // PR 6.1 scope: real engine wiring is PR 6.4. Until then every
        // first apply returns NotYetWired so callers see the gap loudly.
        let applier = LocalApplier::new();
        let err = applier
            .apply(TenantId::new(1), 1, &upsert_memory("rid-1"))
            .await
            .unwrap_err();
        assert!(matches!(err, ApplyError::NotYetWired { .. }));
        assert_eq!(err.metric_label(), "not_yet_wired");
    }

    #[tokio::test]
    async fn duplicate_apply_is_idempotent() {
        // The non-negotiable contract: replaying the same (tenant, log_index)
        // is detected and reported as AlreadyApplied (which callers treat
        // as success).
        let applier = LocalApplier::new();
        let m = upsert_memory("rid-1");
        let _ = applier.apply(TenantId::new(1), 1, &m).await;
        let err = applier.apply(TenantId::new(1), 1, &m).await.unwrap_err();
        assert!(matches!(err, ApplyError::AlreadyApplied { .. }));
        assert!(err.is_idempotent_ok());
    }

    #[tokio::test]
    async fn different_log_index_same_tenant_is_not_duplicate() {
        let applier = LocalApplier::new();
        let m1 = upsert_memory("rid-1");
        let m2 = upsert_memory("rid-2");
        let e1 = applier.apply(TenantId::new(1), 1, &m1).await.unwrap_err();
        let e2 = applier.apply(TenantId::new(1), 2, &m2).await.unwrap_err();
        // Both are first-time applies → NotYetWired, not AlreadyApplied.
        assert!(matches!(e1, ApplyError::NotYetWired { .. }));
        assert!(matches!(e2, ApplyError::NotYetWired { .. }));
    }

    #[tokio::test]
    async fn same_log_index_different_tenant_is_not_duplicate() {
        // (tenant_id, log_index) is the dedup key — log_index alone is
        // not unique across tenants because each tenant has its own log.
        let applier = LocalApplier::new();
        let m = upsert_memory("rid-1");
        let e1 = applier.apply(TenantId::new(1), 1, &m).await.unwrap_err();
        let e2 = applier.apply(TenantId::new(2), 1, &m).await.unwrap_err();
        assert!(matches!(e1, ApplyError::NotYetWired { .. }));
        assert!(matches!(e2, ApplyError::NotYetWired { .. }));
    }

    #[tokio::test]
    async fn watermark_tracks_per_tenant_max() {
        let applier = LocalApplier::new();
        assert_eq!(
            applier
                .applied_high_watermark(TenantId::new(1))
                .await
                .unwrap(),
            0
        );

        // Even though apply returns NotYetWired, the watermark is updated
        // before the engine call — replay detection is the load-bearing
        // contract, not engine success.
        let m = upsert_memory("rid-1");
        let _ = applier.apply(TenantId::new(1), 7, &m).await;
        let _ = applier.apply(TenantId::new(1), 3, &m).await;
        let _ = applier.apply(TenantId::new(2), 9, &m).await;

        assert_eq!(
            applier
                .applied_high_watermark(TenantId::new(1))
                .await
                .unwrap(),
            7
        );
        assert_eq!(
            applier
                .applied_high_watermark(TenantId::new(2))
                .await
                .unwrap(),
            9
        );
        assert_eq!(
            applier
                .applied_high_watermark(TenantId::new(99))
                .await
                .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn tombstone_variant_routes_through_apply() {
        // Variant dispatch is the trait's job — tombstones and upserts
        // both land in the same apply() but the variant_name() shows
        // up in the NotYetWired error so callers can tell which
        // engine method PR 6.4 needs to wire.
        let applier = LocalApplier::new();
        let err = applier
            .apply(TenantId::new(1), 1, &tombstone("rid-1"))
            .await
            .unwrap_err();
        match err {
            ApplyError::NotYetWired { variant, .. } => {
                assert_eq!(variant, "TombstoneMemory");
            }
            other => panic!("expected NotYetWired, got {other:?}"),
        }
    }

    #[test]
    fn apply_error_metric_labels_are_stable() {
        // Dashboards key on these strings.
        assert_eq!(
            ApplyError::NotYetWired {
                variant: "x",
                planned_pr: "y"
            }
            .metric_label(),
            "not_yet_wired"
        );
        assert_eq!(
            ApplyError::AlreadyApplied {
                tenant_id: TenantId::new(1),
                log_index: 1
            }
            .metric_label(),
            "already_applied"
        );
        assert_eq!(
            ApplyError::EngineFailure {
                message: "x".into()
            }
            .metric_label(),
            "engine_failure"
        );
    }

    #[test]
    fn is_idempotent_ok_classification() {
        // AlreadyApplied is the ONLY error a caller should silently
        // swallow as success — duplicate apply during replay/snapshot
        // is normal. Everything else is a real failure.
        assert!(ApplyError::AlreadyApplied {
            tenant_id: TenantId::new(1),
            log_index: 1
        }
        .is_idempotent_ok());
        assert!(!ApplyError::NotYetWired {
            variant: "x",
            planned_pr: "y"
        }
        .is_idempotent_ok());
        assert!(!ApplyError::EngineFailure {
            message: "x".into()
        }
        .is_idempotent_ok());
    }

    // Compile-time pin: the OpId import keeps mutation.rs and the
    // applier in lock-step. If OpId moves or its public interface
    // changes, this trips.
    #[allow(dead_code)]
    fn _op_id_compile_check() {
        let _ = OpId::new_random();
    }
}
