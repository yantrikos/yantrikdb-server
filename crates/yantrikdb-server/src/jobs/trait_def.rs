//! `JobQueue` trait + types.

use std::time::SystemTime;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid7::Uuid;

use crate::commit::TenantId;

/// Job identifier — UUIDv7, time-ordered + globally unique.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct JobId(pub Uuid);

impl JobId {
    pub fn new_random() -> Self {
        Self(uuid7::uuid7())
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Lease identifier. Returned to the worker when it acquires a lease;
/// used as proof-of-lease when heartbeating + completing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LeaseId(pub Uuid);

impl LeaseId {
    pub fn new_random() -> Self {
        Self(uuid7::uuid7())
    }
}

impl std::fmt::Display for LeaseId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Opaque worker identifier. Caller chooses the format; convention is
/// `<hostname>-<pid>-<uuid>` so logs trace back to a specific process.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkerId(pub String);

impl WorkerId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// Job lifecycle states. Stored as TEXT in SQLite for operator readability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobState {
    /// In queue, no worker has leased it yet.
    Pending,
    /// A worker has acquired a lease and is executing.
    Leased,
    /// Terminal — worker reported `Outcome::Succeeded`.
    Succeeded,
    /// Terminal — worker reported `Outcome::Failed`.
    Failed,
    /// Terminal — admin/operator cancelled before execution.
    Cancelled,
}

impl JobState {
    pub fn as_str(&self) -> &'static str {
        match self {
            JobState::Pending => "Pending",
            JobState::Leased => "Leased",
            JobState::Succeeded => "Succeeded",
            JobState::Failed => "Failed",
            JobState::Cancelled => "Cancelled",
        }
    }

    pub fn from_str(s: &str) -> Option<JobState> {
        Some(match s {
            "Pending" => JobState::Pending,
            "Leased" => JobState::Leased,
            "Succeeded" => JobState::Succeeded,
            "Failed" => JobState::Failed,
            "Cancelled" => JobState::Cancelled,
            _ => return None,
        })
    }

    /// Whether this state is terminal (job won't change again).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobState::Succeeded | JobState::Failed | JobState::Cancelled
        )
    }
}

/// What the worker reports when finishing a leased job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Outcome {
    Succeeded,
    Failed { error_message: String },
}

/// Inputs for `JobQueue::enqueue`. Distinct from `JobRecord` (which
/// includes server-assigned fields like id, state, timestamps).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewJob {
    pub tenant_id: TenantId,
    /// Job kind — e.g. `"hnsw_delete"`, `"snapshot_create"`. Workers
    /// filter pickup by kind. **Convention**: `<rfc>.<task>` (e.g.
    /// `"rfc011.hnsw_delete"`) so it's clear which RFC owns the kind.
    pub kind: String,
    pub payload: serde_json::Value,
    /// Higher = picked up first when multiple Pending jobs exist for
    /// a kind. Default 5; range 0-10.
    pub priority: u8,
}

/// One job + its current state. Returned by `list` / `get`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: JobId,
    pub tenant_id: TenantId,
    pub kind: String,
    pub payload: serde_json::Value,
    pub state: JobState,
    pub priority: u8,
    pub created_at: SystemTime,
    /// Worker that currently holds the lease; None if Pending or terminal.
    pub leased_by: Option<WorkerId>,
    /// When the lease was acquired; None if never leased.
    pub leased_at: Option<SystemTime>,
    /// When the lease expires (heartbeat-renewable). None if not leased.
    pub lease_expires_at: Option<SystemTime>,
    /// When the job reached terminal state. None if still in flight.
    pub completed_at: Option<SystemTime>,
    /// On terminal Failed: the error message reported by the worker.
    pub error_message: Option<String>,
}

/// Active lease — proof a worker can heartbeat or complete a job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lease {
    pub lease_id: LeaseId,
    pub job_id: JobId,
    pub worker_id: WorkerId,
    pub acquired_at: SystemTime,
    pub expires_at: SystemTime,
    /// The job's payload, copied here so the worker doesn't need a
    /// second round-trip to read it. (Job records are small; this is
    /// not a memory concern.)
    pub job: JobRecord,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum JobError {
    /// Caller tried to operate on a non-existent job.
    #[error("job {id} not found")]
    NotFound { id: JobId },

    /// Caller tried to lease a job that's already leased by another
    /// worker (or its lease hasn't expired yet).
    #[error("job {id} is already leased; expires at {expires_at:?}")]
    AlreadyLeased {
        id: JobId,
        expires_at: SystemTime,
    },

    /// Heartbeat / complete called with a stale lease (the lease has
    /// expired or another worker has stolen the job).
    #[error("lease {lease_id} is no longer valid for job {job_id}")]
    InvalidLease {
        job_id: JobId,
        lease_id: LeaseId,
    },

    /// Caller tried to transition a job from a terminal state.
    #[error(
        "job {id} is in terminal state {state:?}; cannot transition to {attempted:?}"
    )]
    TerminalState {
        id: JobId,
        state: JobState,
        attempted: JobState,
    },

    /// Underlying storage failure.
    #[error("storage failure: {message}")]
    StorageFailure { message: String },
}

impl JobError {
    /// Stable label for metrics. No user data.
    pub fn metric_label(&self) -> &'static str {
        match self {
            JobError::NotFound { .. } => "not_found",
            JobError::AlreadyLeased { .. } => "already_leased",
            JobError::InvalidLease { .. } => "invalid_lease",
            JobError::TerminalState { .. } => "terminal_state",
            JobError::StorageFailure { .. } => "storage_failure",
        }
    }
}

/// Trait every job-queue backend implements. `LocalSqliteJobQueue`
/// (RFC 019 PR-1) and a future `RaftJobQueue` would both satisfy this.
#[async_trait]
pub trait JobQueue: Send + Sync {
    /// Add a new job. Returns the assigned id.
    async fn enqueue(&self, job: NewJob) -> Result<JobId, JobError>;

    /// Acquire a lease on the highest-priority Pending job whose kind
    /// matches one of `kinds_filter` (or any kind if `None`). Returns
    /// `Ok(None)` if no eligible job is available.
    ///
    /// `lease_ttl_secs` is how long the worker has to heartbeat or
    /// complete before the lease expires.
    async fn acquire_lease(
        &self,
        worker_id: WorkerId,
        kinds_filter: Option<Vec<String>>,
        tenant_filter: Option<TenantId>,
        lease_ttl_secs: u64,
    ) -> Result<Option<Lease>, JobError>;

    /// Extend the lease by another `lease_ttl_secs`. Long-running
    /// workers heartbeat on a cadence shorter than the TTL.
    async fn heartbeat(
        &self,
        lease_id: LeaseId,
        job_id: JobId,
        lease_ttl_secs: u64,
    ) -> Result<(), JobError>;

    /// Mark the job terminal with the given outcome. Lease is consumed.
    async fn complete(
        &self,
        lease_id: LeaseId,
        job_id: JobId,
        outcome: Outcome,
    ) -> Result<(), JobError>;

    /// Cancel a Pending or Leased job. Terminal states refuse the
    /// transition with `JobError::TerminalState`.
    async fn cancel(&self, job_id: JobId) -> Result<(), JobError>;

    /// Get a single job by id.
    async fn get(&self, job_id: JobId) -> Result<JobRecord, JobError>;

    /// List jobs, optionally filtered by tenant + state. Sorted by
    /// `created_at` DESC (newest first). Bounded by `limit`.
    async fn list(
        &self,
        tenant_filter: Option<TenantId>,
        state_filter: Option<JobState>,
        limit: usize,
    ) -> Result<Vec<JobRecord>, JobError>;

    /// Return all leases past TTL to the Pending pool. Returns the
    /// number of leases reclaimed. Run on a control-plane cadence
    /// (every few seconds).
    async fn expire_stale_leases(&self) -> Result<usize, JobError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_state_strings_are_stable_and_round_trip() {
        // Stability matters because state is stored as TEXT in SQLite.
        let states = [
            JobState::Pending,
            JobState::Leased,
            JobState::Succeeded,
            JobState::Failed,
            JobState::Cancelled,
        ];
        for state in &states {
            let s = state.as_str();
            let back = JobState::from_str(s).expect("round-trip");
            assert_eq!(*state, back);
        }
        // Pin specific strings.
        assert_eq!(JobState::Pending.as_str(), "Pending");
        assert_eq!(JobState::Leased.as_str(), "Leased");
        assert_eq!(JobState::Failed.as_str(), "Failed");
    }

    #[test]
    fn unknown_state_string_returns_none() {
        assert!(JobState::from_str("UnknownState").is_none());
        assert!(JobState::from_str("").is_none());
    }

    #[test]
    fn terminal_states_are_classified_correctly() {
        assert!(!JobState::Pending.is_terminal());
        assert!(!JobState::Leased.is_terminal());
        assert!(JobState::Succeeded.is_terminal());
        assert!(JobState::Failed.is_terminal());
        assert!(JobState::Cancelled.is_terminal());
    }

    #[test]
    fn job_error_metric_labels_are_stable() {
        let job_id = JobId::new_random();
        let lease_id = LeaseId::new_random();
        assert_eq!(
            JobError::NotFound { id: job_id }.metric_label(),
            "not_found"
        );
        assert_eq!(
            JobError::AlreadyLeased {
                id: job_id,
                expires_at: SystemTime::UNIX_EPOCH,
            }
            .metric_label(),
            "already_leased"
        );
        assert_eq!(
            JobError::InvalidLease {
                job_id,
                lease_id,
            }
            .metric_label(),
            "invalid_lease"
        );
        assert_eq!(
            JobError::TerminalState {
                id: job_id,
                state: JobState::Succeeded,
                attempted: JobState::Pending,
            }
            .metric_label(),
            "terminal_state"
        );
        assert_eq!(
            JobError::StorageFailure {
                message: "x".into(),
            }
            .metric_label(),
            "storage_failure"
        );
    }
}
