//! HNSW compaction consumer — RFC 011 PR-3.
//!
//! ## What this module ships
//!
//! - [`HnswCompactor`] trait. The engine-crate side implements this;
//!   it's the seam between the server-substrate's job queue and the
//!   actual HNSW data structure (which lives in the `yantrikdb`
//!   engine crate that we don't modify here).
//! - [`NoopHnswCompactor`] — stub for tests + early integration.
//! - [`process_one_delete_job`] — async fn that pulls one
//!   `rfc011.hnsw_delete` lease from the JobQueue, dispatches to a
//!   [`HnswCompactor`], and completes the lease.
//!
//! ## Why a trait + stub instead of a direct call
//!
//! Same pattern as RFC 015-A's `TombstoneProvider` / `CommitObserver`
//! / RFC 013's `HnswManifestStore`: the server substrate ships the
//! contract; the engine crate implements it. This lets the substrate
//! round-trip its own tests without dragging in the engine, and lets
//! the engine wire in when ready without further substrate changes.
//!
//! ## Throttling + batching policy
//!
//! [`process_one_delete_job`] handles one job per call. The eventual
//! periodic worker invokes it in a loop, and any throttling
//! (max-deletes-per-second per tenant, batching N rids before
//! flushing the HNSW state, etc.) lives in that wrapper. Keeping
//! the per-job step granular here means the policy knobs land in
//! one place when we wire the worker, not scattered across the
//! consumer surface.

use async_trait::async_trait;

use crate::commit::TenantId;
use crate::forget::delete_queue::{HnswDeletePayload, HNSW_DELETE_JOB_KIND};
use crate::jobs::{JobQueue, Outcome, WorkerId};

/// Trait for the engine-side HNSW compactor. One method per logical
/// operation; today there's just `delete_rid`, but the surface is
/// kept narrow on purpose — when crypto-shred (RFC 011 PR-4) lands
/// it adds its own method here rather than overloading delete.
#[async_trait]
pub trait HnswCompactor: Send + Sync {
    /// Stable name for metrics + audit logs.
    fn name(&self) -> &'static str;

    /// Mark `rid` as deleted in the tenant's HNSW. Idempotent —
    /// deleting the same rid twice is a no-op (returns Ok). The
    /// engine MAY defer the physical-erase step (HNSW deletes are
    /// typically tombstone-on-node + lazy compaction), but the rid
    /// MUST stop being returned by recall after this call.
    async fn delete_rid(&self, tenant_id: TenantId, rid: &str) -> Result<(), CompactionError>;
}

#[derive(Debug, thiserror::Error)]
pub enum CompactionError {
    /// The engine-side delete failed in a recoverable way (e.g. the
    /// HNSW index was locked by another writer). The job's lease
    /// will expire and another worker will retry.
    #[error("transient HNSW delete failure for tenant {tenant_id} rid {rid}: {message}")]
    Transient {
        tenant_id: TenantId,
        rid: String,
        message: String,
    },

    /// The engine-side delete failed permanently (corruption, schema
    /// mismatch). The job is marked Failed; operator inspects.
    #[error("permanent HNSW delete failure for tenant {tenant_id} rid {rid}: {message}")]
    Permanent {
        tenant_id: TenantId,
        rid: String,
        message: String,
    },
}

/// No-op stub. Always returns Ok. Used in tests + early integration
/// before the engine-side implementation is wired.
#[derive(Default, Clone)]
pub struct NoopHnswCompactor;

#[async_trait]
impl HnswCompactor for NoopHnswCompactor {
    fn name(&self) -> &'static str {
        "noop_hnsw_compactor"
    }

    async fn delete_rid(&self, _tenant_id: TenantId, _rid: &str) -> Result<(), CompactionError> {
        Ok(())
    }
}

/// Outcome of one delete-job processing pass. Operator dashboards +
/// metrics consume this. `None` = no Pending job was available
/// (queue was empty), so the worker should sleep before retrying.
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessOutcome {
    /// A job was processed successfully.
    Succeeded { tenant_id: TenantId, rid: String },
    /// A job was processed and failed permanently.
    Failed {
        tenant_id: TenantId,
        rid: String,
        message: String,
    },
    /// Transient failure — lease is allowed to expire, retried later.
    /// We mark Failed too so the worker doesn't churn; production
    /// should layer a JobQueue retry policy on top (RFC 019 PR-2).
    TransientFailed {
        tenant_id: TenantId,
        rid: String,
        message: String,
    },
    /// No job was available.
    NoJob,
    /// Job payload was malformed JSON. Should never happen in
    /// practice (we control the producer); cancelling the job to
    /// drain the queue.
    MalformedPayload { error: String },
}

/// Pull one `rfc011.hnsw_delete` lease from `jobs`, dispatch to
/// `compactor`, mark the job Succeeded/Failed.
///
/// `lease_ttl_secs` controls how long the worker has to complete
/// before the JobQueue reclaims the lease for another worker. 60 s
/// is reasonable for HNSW deletes which are typically <100 ms each.
pub async fn process_one_delete_job(
    jobs: &(dyn JobQueue),
    compactor: &(dyn HnswCompactor),
    worker_id: WorkerId,
    lease_ttl_secs: u64,
) -> ProcessOutcome {
    let lease = match jobs
        .acquire_lease(
            worker_id,
            Some(vec![HNSW_DELETE_JOB_KIND.to_string()]),
            None,
            lease_ttl_secs,
        )
        .await
    {
        Ok(Some(lease)) => lease,
        Ok(None) => return ProcessOutcome::NoJob,
        Err(e) => {
            return ProcessOutcome::TransientFailed {
                tenant_id: TenantId::new(0),
                rid: String::new(),
                message: format!("lease acquire: {e}"),
            };
        }
    };

    let payload = match HnswDeletePayload::from_value(&lease.job.payload) {
        Ok(p) => p,
        Err(e) => {
            // Cancel the malformed job so it doesn't keep getting
            // re-leased forever.
            let _ = jobs.cancel(lease.job_id).await;
            return ProcessOutcome::MalformedPayload {
                error: format!("payload deserialize: {e}"),
            };
        }
    };

    match compactor.delete_rid(payload.tenant_id, &payload.rid).await {
        Ok(()) => {
            if let Err(e) = jobs
                .complete(lease.lease_id, lease.job_id, Outcome::Succeeded)
                .await
            {
                return ProcessOutcome::TransientFailed {
                    tenant_id: payload.tenant_id,
                    rid: payload.rid.clone(),
                    message: format!("complete Succeeded: {e}"),
                };
            }
            ProcessOutcome::Succeeded {
                tenant_id: payload.tenant_id,
                rid: payload.rid,
            }
        }
        Err(CompactionError::Permanent { message, .. }) => {
            let _ = jobs
                .complete(
                    lease.lease_id,
                    lease.job_id,
                    Outcome::Failed {
                        error_message: message.clone(),
                    },
                )
                .await;
            ProcessOutcome::Failed {
                tenant_id: payload.tenant_id,
                rid: payload.rid,
                message,
            }
        }
        Err(CompactionError::Transient { message, .. }) => {
            // Mark Failed; the operator can re-enqueue if needed.
            // RFC 019 PR-2 (retry policies) will graduate this to
            // automatic retry-with-backoff.
            let _ = jobs
                .complete(
                    lease.lease_id,
                    lease.job_id,
                    Outcome::Failed {
                        error_message: format!("transient: {message}"),
                    },
                )
                .await;
            ProcessOutcome::TransientFailed {
                tenant_id: payload.tenant_id,
                rid: payload.rid,
                message,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forget::delete_queue::{enqueue_one, HnswDeletePayload};
    use crate::jobs::{JobState, LocalSqliteJobQueue};
    use parking_lot::Mutex;
    use std::sync::Arc;

    /// Test compactor that records every delete_rid call.
    struct RecordingCompactor {
        deletes: Mutex<Vec<(TenantId, String)>>,
    }

    impl RecordingCompactor {
        fn new() -> Self {
            Self {
                deletes: Mutex::new(Vec::new()),
            }
        }
        fn deletes(&self) -> Vec<(TenantId, String)> {
            self.deletes.lock().clone()
        }
    }

    #[async_trait]
    impl HnswCompactor for RecordingCompactor {
        fn name(&self) -> &'static str {
            "recording"
        }
        async fn delete_rid(&self, tenant_id: TenantId, rid: &str) -> Result<(), CompactionError> {
            self.deletes.lock().push((tenant_id, rid.to_string()));
            Ok(())
        }
    }

    /// Test compactor that always fails the requested way.
    struct FailingCompactor {
        permanent: bool,
    }

    #[async_trait]
    impl HnswCompactor for FailingCompactor {
        fn name(&self) -> &'static str {
            "failing"
        }
        async fn delete_rid(&self, tenant_id: TenantId, rid: &str) -> Result<(), CompactionError> {
            if self.permanent {
                Err(CompactionError::Permanent {
                    tenant_id,
                    rid: rid.to_string(),
                    message: "synthetic permanent".into(),
                })
            } else {
                Err(CompactionError::Transient {
                    tenant_id,
                    rid: rid.to_string(),
                    message: "synthetic transient".into(),
                })
            }
        }
    }

    fn open_jobs() -> Arc<LocalSqliteJobQueue> {
        Arc::new(LocalSqliteJobQueue::open_in_memory().unwrap())
    }

    fn payload(tenant: i64, rid: &str) -> HnswDeletePayload {
        HnswDeletePayload {
            tenant_id: TenantId::new(tenant),
            rid: rid.into(),
            source_log_index: 1,
            reason: None,
        }
    }

    #[tokio::test]
    async fn noop_compactor_returns_ok() {
        let c = NoopHnswCompactor;
        c.delete_rid(TenantId::new(1), "rid").await.unwrap();
        assert_eq!(c.name(), "noop_hnsw_compactor");
    }

    #[tokio::test]
    async fn process_no_job_returns_no_job() {
        let jobs = open_jobs();
        let compactor = NoopHnswCompactor;
        let outcome = process_one_delete_job(&*jobs, &compactor, WorkerId::new("test"), 30).await;
        assert_eq!(outcome, ProcessOutcome::NoJob);
    }

    #[tokio::test]
    async fn process_succeeded_marks_job_succeeded() {
        let jobs = open_jobs();
        let id = enqueue_one(&*jobs, payload(1, "rid_a")).await.unwrap();
        let compactor = RecordingCompactor::new();
        let outcome = process_one_delete_job(&*jobs, &compactor, WorkerId::new("test"), 30).await;
        assert_eq!(
            outcome,
            ProcessOutcome::Succeeded {
                tenant_id: TenantId::new(1),
                rid: "rid_a".into(),
            }
        );
        assert_eq!(
            compactor.deletes(),
            vec![(TenantId::new(1), "rid_a".into())]
        );
        let job = jobs.get(id).await.unwrap();
        assert_eq!(job.state, JobState::Succeeded);
    }

    #[tokio::test]
    async fn process_permanent_failure_marks_job_failed() {
        let jobs = open_jobs();
        let id = enqueue_one(&*jobs, payload(1, "rid_a")).await.unwrap();
        let compactor = FailingCompactor { permanent: true };
        let outcome = process_one_delete_job(&*jobs, &compactor, WorkerId::new("test"), 30).await;
        match outcome {
            ProcessOutcome::Failed {
                tenant_id,
                rid,
                message,
            } => {
                assert_eq!(tenant_id, TenantId::new(1));
                assert_eq!(rid, "rid_a");
                assert!(message.contains("synthetic permanent"));
            }
            other => panic!("expected Failed, got {other:?}"),
        }
        let job = jobs.get(id).await.unwrap();
        assert_eq!(job.state, JobState::Failed);
    }

    #[tokio::test]
    async fn process_transient_failure_marks_failed_with_prefix() {
        let jobs = open_jobs();
        let id = enqueue_one(&*jobs, payload(1, "rid_a")).await.unwrap();
        let compactor = FailingCompactor { permanent: false };
        let outcome = process_one_delete_job(&*jobs, &compactor, WorkerId::new("test"), 30).await;
        assert!(matches!(outcome, ProcessOutcome::TransientFailed { .. }));
        let job = jobs.get(id).await.unwrap();
        assert_eq!(job.state, JobState::Failed);
        assert!(job.error_message.unwrap_or_default().contains("transient:"));
    }

    #[tokio::test]
    async fn process_skips_jobs_of_other_kinds() {
        // Only `rfc011.hnsw_delete` jobs should be picked up. Other
        // kinds in the queue must be ignored.
        let jobs = open_jobs();
        jobs.enqueue(crate::jobs::NewJob {
            tenant_id: TenantId::new(1),
            kind: "rfc012.snapshot_create".to_string(),
            payload: serde_json::json!({}),
            priority: 5,
        })
        .await
        .unwrap();
        let compactor = NoopHnswCompactor;
        let outcome = process_one_delete_job(&*jobs, &compactor, WorkerId::new("test"), 30).await;
        assert_eq!(outcome, ProcessOutcome::NoJob);
    }

    #[tokio::test]
    async fn process_malformed_payload_cancels_job() {
        // Inject a job with the right kind but a payload the
        // consumer can't parse. Should be cancelled (not retried)
        // so the queue drains.
        let jobs = open_jobs();
        let id = jobs
            .enqueue(crate::jobs::NewJob {
                tenant_id: TenantId::new(1),
                kind: HNSW_DELETE_JOB_KIND.to_string(),
                payload: serde_json::json!({"this_is_not": "a_valid_payload"}),
                priority: 5,
            })
            .await
            .unwrap();
        let compactor = NoopHnswCompactor;
        let outcome = process_one_delete_job(&*jobs, &compactor, WorkerId::new("test"), 30).await;
        assert!(matches!(outcome, ProcessOutcome::MalformedPayload { .. }));
        let job = jobs.get(id).await.unwrap();
        assert_eq!(job.state, JobState::Cancelled);
    }

    #[tokio::test]
    async fn end_to_end_producer_consumer() {
        // The full handoff: a TombstoneMemory mutation lands in the
        // commit log, the producer scans + enqueues, the consumer
        // processes the job, the compactor records the delete.
        use crate::commit::{
            CommitOptions, LocalSqliteCommitter, MemoryMutation, MutationCommitter,
        };
        use crate::forget::delete_queue::enqueue_range;

        let committer: Arc<LocalSqliteCommitter> =
            Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let jobs = open_jobs();
        let tenant = TenantId::new(1);

        // Producer side: write a tombstone, scan + enqueue.
        committer
            .commit(
                tenant,
                MemoryMutation::TombstoneMemory {
                    rid: "doomed".into(),
                    reason: Some("e2e".into()),
                    requested_at_unix_micros: 0,
                },
                CommitOptions::default(),
            )
            .await
            .unwrap();
        let result = enqueue_range(&*committer, &*jobs, tenant, 1, 100)
            .await
            .unwrap();
        assert_eq!(result.jobs_enqueued, 1);

        // Consumer side: pick up the job + dispatch.
        let compactor = RecordingCompactor::new();
        let outcome =
            process_one_delete_job(&*jobs, &compactor, WorkerId::new("e2e-worker"), 30).await;
        assert!(matches!(
            outcome,
            ProcessOutcome::Succeeded { ref rid, .. } if rid == "doomed"
        ));
        assert_eq!(compactor.deletes(), vec![(tenant, "doomed".to_string())]);

        // Queue is now empty.
        let next =
            process_one_delete_job(&*jobs, &compactor, WorkerId::new("e2e-worker"), 30).await;
        assert_eq!(next, ProcessOutcome::NoJob);
    }
}
