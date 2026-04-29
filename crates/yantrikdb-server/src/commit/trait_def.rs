//! `MutationCommitter` trait — the abstraction that lets us swap
//! `LocalSqliteCommitter` ↔ `RaftCommitter` without changing API code.
//!
//! Every API write (`/v1/remember`, `/v1/forget`, `/v1/relate`, ...) calls
//! `committer.commit(...)` instead of mutating storage directly. The
//! committer is responsible for: assigning a `(term, log_index)`,
//! persisting the mutation durably, applying it to the state machine,
//! and returning a receipt the caller can use to verify durability.

use std::time::SystemTime;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::mutation::{MemoryMutation, OpId, TenantId};
use crate::version::VersionError;

/// Receipt returned from a successful commit. Carries enough info that
/// the caller can verify durability and look up the mutation later in
/// the commit log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitReceipt {
    pub op_id: OpId,
    pub tenant_id: TenantId,
    /// Raft-style term. Always 0 for `LocalSqliteCommitter` since there's
    /// no leadership; populated by `RaftCommitter` (RFC 010 PR-4).
    pub term: u64,
    /// Monotonic per-tenant. Increments by 1 within a tenant.
    pub log_index: u64,
    pub committed_at: SystemTime,
    /// `Some` when the mutation has been applied to the state machine;
    /// `None` if `wait_for_apply: false` was passed and the apply is
    /// still in flight.
    pub applied_at: Option<SystemTime>,
}

/// Hook invoked by a committer after every successful commit, so
/// in-process consumers (e.g. [`crate::forget::TombstoneIndex`]) can
/// stay in lock-step with the durable log without the commit layer
/// taking a hard dependency on each consumer's concrete type.
///
/// Implementations MUST be cheap and non-blocking — this runs on the
/// commit hot path. Heavier work (cache invalidation broadcasts, async
/// notifications) belongs on the
/// [`crate::cache::InvalidationBus`] which fans out off-path.
pub trait CommitObserver: Send + Sync {
    /// Called immediately after a commit's SQL transaction commits and
    /// the receipt has been built. Side effects observed here MUST NOT
    /// fail in a way that affects the receipt — the commit has already
    /// happened durably.
    fn after_commit(&self, tenant_id: TenantId, mutation: &MemoryMutation);
}

/// Per-call options for `commit`.
#[derive(Debug, Clone, Default)]
pub struct CommitOptions {
    /// If `Some(n)`, fail the commit unless the next assigned `log_index`
    /// would be exactly `n`. Used for compare-and-swap semantics by the
    /// few callers that need it (e.g. cluster bootstrap). Default: `None`
    /// (no constraint).
    pub expected_log_index: Option<u64>,

    /// If `true`, wait for the mutation to be applied to the state
    /// machine before returning. If `false`, return as soon as the entry
    /// is durably appended to the log; apply happens asynchronously.
    /// Default: `true` for correctness; callers like the bulk-import
    /// path can opt out for throughput.
    pub wait_for_apply: bool,

    /// Client-provided op_id for idempotent retries. When `Some`, the
    /// committer will:
    /// - Look up `(tenant_id, op_id)` in the existing log first
    /// - If found AND mutation matches → return the existing receipt
    ///   (idempotent retry, the contract that lets HTTP clients retry
    ///   on network failures without duplicating writes)
    /// - If found AND mutation differs → return `OpIdCollision` (client bug)
    /// - If not found → assign a new log_index, persist with this op_id
    ///
    /// When `None`, the committer auto-generates a fresh UUIDv7. Use
    /// `None` for server-internal commits where retry-deduplication
    /// isn't needed.
    pub op_id: Option<super::mutation::OpId>,
}

impl CommitOptions {
    pub fn new() -> Self {
        Self {
            expected_log_index: None,
            wait_for_apply: true,
            op_id: None,
        }
    }

    pub fn no_wait(mut self) -> Self {
        self.wait_for_apply = false;
        self
    }

    pub fn expecting_index(mut self, idx: u64) -> Self {
        self.expected_log_index = Some(idx);
        self
    }

    pub fn with_op_id(mut self, op_id: super::mutation::OpId) -> Self {
        self.op_id = Some(op_id);
        self
    }
}

/// A single committed entry in the per-tenant log. Returned by
/// [`MutationCommitter::read_range`] for replay, audit, debug-history,
/// and Jepsen fault-injection paths.
///
/// Only `PartialEq` (not `Eq`) because `MemoryMutation` carries `f64`
/// fields (importance, weight, etc.) which are `PartialEq` but not `Eq`
/// (NaN inequality). Tests that compare entries use `assert_eq!` which
/// only requires `PartialEq`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommittedEntry {
    pub op_id: OpId,
    pub tenant_id: TenantId,
    pub term: u64,
    pub log_index: u64,
    pub mutation: MemoryMutation,
    pub committed_at: SystemTime,
    pub applied_at: Option<SystemTime>,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum CommitError {
    /// The same `op_id` was committed previously with a different
    /// mutation payload. Callers MUST NOT change the mutation while
    /// keeping the same op_id — that's a client bug.
    #[error(
        "op_id collision for tenant {tenant_id}: existing log_index {existing_index} \
         has different mutation payload than the one being committed. Client bug: \
         the same op_id must always carry the same mutation."
    )]
    OpIdCollision {
        op_id: OpId,
        tenant_id: TenantId,
        existing_index: u64,
    },

    /// `expected_log_index` mismatch (compare-and-swap failure).
    #[error(
        "expected log_index {expected} for tenant {tenant_id}, but next index is {actual}. \
         Concurrent write happened — caller should re-read state and retry."
    )]
    UnexpectedLogIndex {
        tenant_id: TenantId,
        expected: u64,
        actual: u64,
    },

    /// Mutation rejected by version policy (e.g. wire major mismatch
    /// during a rolling upgrade).
    #[error("version check failed: {0}")]
    Version(#[from] VersionError),

    /// Mutation variant exists in the grammar but isn't implemented yet.
    /// `PurgeMemory` returns this until RFC 011 ships.
    #[error("mutation variant `{variant}` not yet implemented (planned in RFC {planned_rfc})")]
    NotYetImplemented {
        variant: &'static str,
        planned_rfc: &'static str,
    },

    /// Underlying storage / IO failure.
    #[error("storage failure: {message}")]
    StorageFailure { message: String },

    /// Server is shutting down — caller should NOT retry on this node.
    #[error("server shutting down")]
    Shutdown,

    /// This node is not the cluster leader. Returned by `RaftCommitter`
    /// when openraft's `client_write` reports `ForwardToLeader`. The
    /// caller (HTTP gateway, CLI client) SHOULD redirect to
    /// `leader_addr` if known, or surface a 503 with `Retry-After` if
    /// the cluster is mid-election. `leader_id` / `leader_addr` are
    /// `None` when no leader is currently known.
    #[error(
        "not the cluster leader; redirect to leader id={leader_id:?} addr={leader_addr:?}"
    )]
    NotLeader {
        leader_id: Option<u64>,
        leader_addr: Option<String>,
    },
}

impl CommitError {
    /// Stable label for metrics. No user data — safe for Prometheus.
    pub fn metric_label(&self) -> &'static str {
        match self {
            CommitError::OpIdCollision { .. } => "op_id_collision",
            CommitError::UnexpectedLogIndex { .. } => "unexpected_log_index",
            CommitError::Version(_) => "version",
            CommitError::NotYetImplemented { .. } => "not_yet_implemented",
            CommitError::StorageFailure { .. } => "storage_failure",
            CommitError::Shutdown => "shutdown",
            CommitError::NotLeader { .. } => "not_leader",
        }
    }

    /// Whether the caller should retry. `OpIdCollision` is a client bug
    /// (don't retry); `UnexpectedLogIndex` and `StorageFailure` are
    /// transient (retry after backoff); `Shutdown` and
    /// `NotYetImplemented` and `Version` are terminal.
    /// `NotLeader` is retryable AGAINST THE LEADER (not the same node).
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            CommitError::UnexpectedLogIndex { .. }
                | CommitError::StorageFailure { .. }
                | CommitError::NotLeader { .. }
        )
    }
}

/// Trait every commit backend implements. `LocalSqliteCommitter` (RFC 010
/// PR-1+2) and `RaftCommitter` (RFC 010 PR-4) both satisfy this interface;
/// API handlers hold an `Arc<dyn MutationCommitter>` and don't care which
/// is in use.
#[async_trait]
pub trait MutationCommitter: Send + Sync {
    /// Commit a mutation. Idempotent on `op_id`: re-committing the same
    /// (op_id, mutation) returns the original receipt.
    async fn commit(
        &self,
        tenant_id: TenantId,
        mutation: MemoryMutation,
        opts: CommitOptions,
    ) -> Result<CommitReceipt, CommitError>;

    /// Read a range of committed entries for a tenant, starting at
    /// `from_index` (inclusive), up to `limit` entries. Used for replay,
    /// audit, debug-history, and Jepsen fault-injection paths.
    async fn read_range(
        &self,
        tenant_id: TenantId,
        from_index: u64,
        limit: usize,
    ) -> Result<Vec<CommittedEntry>, CommitError>;

    /// High watermark — max `log_index` ever assigned for this tenant.
    /// Returns 0 if the tenant has no entries yet. Used by HNSW
    /// reconciliation (RFC 013) to know how much it needs to catch up.
    async fn high_watermark(&self, tenant_id: TenantId) -> Result<u64, CommitError>;

    /// List every tenant id that has at least one entry in the commit
    /// log. Returns ids in ascending order. Used by the Raft state
    /// machine's snapshot builder (RFC 010 PR-4-c) to enumerate
    /// tenants without probing — replaces the prior O(MAX_TENANTS)
    /// linear scan.
    ///
    /// Implementations SHOULD be efficient: production callers invoke
    /// this on every snapshot build. The default `LocalSqliteCommitter`
    /// uses `SELECT DISTINCT tenant_id`.
    async fn list_active_tenants(&self) -> Result<Vec<TenantId>, CommitError>;

    /// Establish a linearizability barrier for subsequent reads from
    /// the local state machine.
    ///
    /// Contract:
    /// - On `Ok(())`, the caller MAY proceed to read from this node's
    ///   state machine and the result is linearizable across the
    ///   cluster as of the moment this call returned.
    /// - On `CommitError::NotLeader`, this node is not the leader; the
    ///   caller SHOULD redirect to the leader.
    /// - On any other error, the caller SHOULD treat reads from this
    ///   node as stale and retry / surface the error.
    ///
    /// The default implementation returns `Ok(())` — correct for
    /// single-node committers where every read is trivially
    /// linearizable. `RaftCommitter` overrides this to call
    /// `Raft::ensure_linearizable()` which heartbeats a quorum to
    /// confirm leadership and waits for the state machine to apply up
    /// to the read-barrier index.
    async fn ensure_linearizable(&self) -> Result<(), CommitError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_options_builder_pattern() {
        let opts = CommitOptions::new().no_wait().expecting_index(42);
        assert!(!opts.wait_for_apply);
        assert_eq!(opts.expected_log_index, Some(42));
    }

    #[test]
    fn commit_options_default_is_safe() {
        // Default MUST wait_for_apply=true. Anything else is a footgun
        // that causes "I committed but recall doesn't see it" reports.
        let opts = CommitOptions::new();
        assert!(opts.wait_for_apply);
        assert_eq!(opts.expected_log_index, None);
    }

    #[test]
    fn metric_labels_are_stable() {
        // Dashboards key on these strings.
        assert_eq!(
            CommitError::OpIdCollision {
                op_id: OpId::new_random(),
                tenant_id: TenantId::new(1),
                existing_index: 0,
            }
            .metric_label(),
            "op_id_collision"
        );
        assert_eq!(
            CommitError::UnexpectedLogIndex {
                tenant_id: TenantId::new(1),
                expected: 1,
                actual: 2,
            }
            .metric_label(),
            "unexpected_log_index"
        );
        assert_eq!(
            CommitError::NotYetImplemented {
                variant: "PurgeMemory",
                planned_rfc: "011",
            }
            .metric_label(),
            "not_yet_implemented"
        );
        assert_eq!(
            CommitError::StorageFailure {
                message: "disk full".into(),
            }
            .metric_label(),
            "storage_failure"
        );
        assert_eq!(CommitError::Shutdown.metric_label(), "shutdown");
    }

    #[test]
    fn retryable_classification_is_correct() {
        // Retry on transient errors only. OpIdCollision is a CLIENT BUG —
        // retrying makes it worse. Shutdown means stop, don't retry.
        assert!(!CommitError::OpIdCollision {
            op_id: OpId::new_random(),
            tenant_id: TenantId::new(1),
            existing_index: 0,
        }
        .is_retryable());
        assert!(CommitError::UnexpectedLogIndex {
            tenant_id: TenantId::new(1),
            expected: 1,
            actual: 2,
        }
        .is_retryable());
        assert!(CommitError::StorageFailure {
            message: "transient".into(),
        }
        .is_retryable());
        assert!(!CommitError::Shutdown.is_retryable());
        assert!(!CommitError::NotYetImplemented {
            variant: "x",
            planned_rfc: "y",
        }
        .is_retryable());
    }

    #[test]
    fn commit_receipt_serde_round_trip() {
        let r = CommitReceipt {
            op_id: OpId::new_random(),
            tenant_id: TenantId::new(7),
            term: 1,
            log_index: 42,
            committed_at: SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000),
            applied_at: Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_001)),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: CommitReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
