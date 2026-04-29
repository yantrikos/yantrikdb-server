//! HNSW delete queue producer — RFC 011 PR-3.
//!
//! ## What this module ships
//!
//! [`TombstoneScanner`] walks a range of the per-tenant commit log,
//! pulls out `TombstoneMemory` mutations, and emits one
//! [`HnswDeletePayload`] per tombstone. [`enqueue_range`] wraps that
//! with a [`crate::jobs::JobQueue`] write, producing durable
//! `rfc011.hnsw_delete` jobs that the engine-side compactor
//! ([`crate::index::hnsw::compaction::HnswCompactor`]) consumes.
//!
//! ## Design choices
//!
//! - **Scanner is a pure function over a log range.** Caller drives
//!   the loop (with persistent `last_scanned_log_index` state). This
//!   keeps the scanner trivially testable and lets the eventual
//!   periodic-runner orchestrate batching, throttling, and
//!   restart-resume on its own terms.
//! - **Job payload pins `(rid, log_index)`** so the consumer can
//!   answer "which tombstone am I draining?" deterministically and
//!   `last_scanned_log_index` can advance even if a job retries.
//! - **`PurgeMemory` mutations are NOT enqueued here.** Purge is a
//!   physical-erase event that the engine-crate side runs against
//!   already-compacted HNSW state; this module only handles the
//!   logical-tombstone → HNSW-delete handoff.
//!
//! ## What's NOT in this PR
//!
//! - The actual HNSW delete (engine-crate code; the `HnswCompactor`
//!   trait here is the seam).
//! - The periodic runner that drives `enqueue_range` on a cadence.
//!   That belongs alongside the existing background-worker
//!   infrastructure; it'll wire to the saga + per-tenant
//!   `last_scanned_log_index` state in a follow-up.
//! - `last_scanned_log_index` persistence. For now it's caller-held
//!   (an in-memory `HashMap<TenantId, u64>` works for the periodic
//!   runner). RFC 021 (config versioning + tenant-scoped state) will
//!   eventually persist this; meanwhile, restart-from-zero is safe
//!   because the consumer is idempotent (deleting the same rid
//!   twice in HNSW is a no-op).

use serde::{Deserialize, Serialize};

use crate::commit::{CommitError, MemoryMutation, MutationCommitter, TenantId};
use crate::jobs::{JobError, JobQueue, JobId, NewJob};

/// Stable kind string for HNSW-delete jobs. Pinned because
/// `JobQueue::acquire_lease` filters on this exact value;
/// renaming it would orphan in-flight jobs at upgrade time.
pub const HNSW_DELETE_JOB_KIND: &str = "rfc011.hnsw_delete";

/// Default priority for tombstone-driven deletes. Mid-band so
/// snapshot/backup jobs (priority 7+) preempt them but consumer
/// loops don't starve them either.
pub const HNSW_DELETE_DEFAULT_PRIORITY: u8 = 4;

/// Payload shape of an `rfc011.hnsw_delete` job. Serialized into
/// [`NewJob::payload`] as JSON so any operator inspecting the job
/// queue sees the rid + originating log_index without needing to
/// decode an opaque blob.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswDeletePayload {
    pub tenant_id: TenantId,
    pub rid: String,
    /// The log_index of the originating `TombstoneMemory` mutation.
    /// Lets the periodic runner advance `last_scanned_log_index` even
    /// when jobs retry. Also lets operator dashboards correlate
    /// individual jobs back to specific tombstone events.
    pub source_log_index: u64,
    /// Reason from the original tombstone, if any. Non-load-bearing;
    /// surfaces in audit dashboards.
    pub reason: Option<String>,
}

impl HnswDeletePayload {
    /// Convert to a `NewJob` ready for enqueue.
    pub fn into_new_job(self) -> NewJob {
        NewJob {
            tenant_id: self.tenant_id,
            kind: HNSW_DELETE_JOB_KIND.to_string(),
            // .expect: serializing a fixed Rust struct to JSON is
            // infallible in practice — every field is a primitive
            // or String.
            payload: serde_json::to_value(&self).expect("HnswDeletePayload serializes"),
            priority: HNSW_DELETE_DEFAULT_PRIORITY,
        }
    }

    /// Parse the payload back from a job's JSON. Used by the
    /// consumer to recover `(rid, source_log_index)` from a Lease.
    pub fn from_value(v: &serde_json::Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(v.clone())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DeleteQueueError {
    #[error("commit log read failed: {0}")]
    Commit(#[from] CommitError),
    #[error("job queue write failed: {0}")]
    Jobs(#[from] JobError),
}

/// Scanner: walks `[from_index, to_index)` of a tenant's commit log
/// and emits a payload per `TombstoneMemory` mutation found.
///
/// Pure function — does not write to the job queue. Use
/// [`enqueue_range`] when you want the full producer behavior.
pub async fn scan_tombstones_in_range(
    committer: &(dyn MutationCommitter),
    tenant_id: TenantId,
    from_log_index: u64,
    limit: usize,
) -> Result<Vec<HnswDeletePayload>, DeleteQueueError> {
    let from = from_log_index.max(1);
    let entries = committer.read_range(tenant_id, from, limit).await?;
    let mut out = Vec::new();
    for entry in entries {
        if let MemoryMutation::TombstoneMemory { rid, reason, .. } = &entry.mutation {
            out.push(HnswDeletePayload {
                tenant_id,
                rid: rid.clone(),
                source_log_index: entry.log_index,
                reason: reason.clone(),
            });
        }
    }
    Ok(out)
}

/// Result of one scan-and-enqueue pass. Operator dashboards +
/// metrics consume this; the periodic runner uses
/// `next_log_index_to_scan` to advance its persistent state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnqueueRangeResult {
    pub tenant_id: TenantId,
    /// Number of `TombstoneMemory` mutations found in the scanned
    /// range.
    pub tombstones_found: usize,
    /// Number of jobs successfully enqueued.
    pub jobs_enqueued: usize,
    /// The log_index the next scan should resume from. Always >
    /// the input `from_log_index` if the scan saw at least one
    /// entry; equal to it (no advance) if the range was empty.
    pub next_log_index_to_scan: u64,
}

/// Producer: scan + enqueue in one call. Caller advances persistent
/// `last_scanned_log_index` to `next_log_index_to_scan` only after
/// this returns Ok.
///
/// If a single enqueue fails, returns the error after enqueuing the
/// already-successful jobs — the periodic runner can re-scan the
/// same range without producing duplicates because the consumer
/// is idempotent (deleting the same rid in HNSW twice is a no-op).
pub async fn enqueue_range(
    committer: &(dyn MutationCommitter),
    jobs: &(dyn JobQueue),
    tenant_id: TenantId,
    from_log_index: u64,
    limit: usize,
) -> Result<EnqueueRangeResult, DeleteQueueError> {
    let from = from_log_index.max(1);
    let entries = committer.read_range(tenant_id, from, limit).await?;
    let scanned_through = entries.last().map(|e| e.log_index).unwrap_or(from - 1);

    let mut tombstones = 0usize;
    let mut enqueued = 0usize;
    for entry in entries {
        if let MemoryMutation::TombstoneMemory { rid, reason, .. } = &entry.mutation {
            tombstones += 1;
            let payload = HnswDeletePayload {
                tenant_id,
                rid: rid.clone(),
                source_log_index: entry.log_index,
                reason: reason.clone(),
            };
            jobs.enqueue(payload.into_new_job()).await?;
            enqueued += 1;
        }
    }

    Ok(EnqueueRangeResult {
        tenant_id,
        tombstones_found: tombstones,
        jobs_enqueued: enqueued,
        next_log_index_to_scan: scanned_through.saturating_add(1),
    })
}

/// Convenience: enqueue an [`HnswDeletePayload`] directly. Used by
/// the [`crate::commit::CommitObserver`] integration if a deployment
/// chooses to enqueue at commit time rather than via the periodic
/// scanner. Both paths terminate at the same job queue, and the
/// consumer is idempotent so duplicate enqueues are safe.
pub async fn enqueue_one(
    jobs: &(dyn JobQueue),
    payload: HnswDeletePayload,
) -> Result<JobId, DeleteQueueError> {
    Ok(jobs.enqueue(payload.into_new_job()).await?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::{CommitOptions, LocalSqliteCommitter};
    use crate::jobs::{JobQueue, JobState, LocalSqliteJobQueue, WorkerId};
    use std::sync::Arc;

    fn upsert(rid: &str) -> MemoryMutation {
        MemoryMutation::UpsertMemory {
            rid: rid.into(),
            text: "x".into(),
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
        }
    }

    fn tombstone(rid: &str, reason: Option<&str>) -> MemoryMutation {
        MemoryMutation::TombstoneMemory {
            rid: rid.into(),
            reason: reason.map(|s| s.to_string()),
            requested_at_unix_micros: 0,
        }
    }

    fn open_committer() -> Arc<LocalSqliteCommitter> {
        Arc::new(LocalSqliteCommitter::open_in_memory().unwrap())
    }

    fn open_jobs() -> Arc<LocalSqliteJobQueue> {
        Arc::new(LocalSqliteJobQueue::open_in_memory().unwrap())
    }

    #[test]
    fn job_kind_is_pinned_by_convention() {
        // The kind string is part of the operator-facing surface.
        // Renaming it orphans in-flight jobs at upgrade — pin it.
        assert_eq!(HNSW_DELETE_JOB_KIND, "rfc011.hnsw_delete");
    }

    #[test]
    fn payload_round_trips_through_job_value() {
        let p = HnswDeletePayload {
            tenant_id: TenantId::new(7),
            rid: "mem_42".into(),
            source_log_index: 1234,
            reason: Some("user requested".into()),
        };
        let job = p.clone().into_new_job();
        assert_eq!(job.kind, HNSW_DELETE_JOB_KIND);
        assert_eq!(job.priority, HNSW_DELETE_DEFAULT_PRIORITY);
        assert_eq!(job.tenant_id, p.tenant_id);
        let back = HnswDeletePayload::from_value(&job.payload).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn payload_with_no_reason_round_trips() {
        let p = HnswDeletePayload {
            tenant_id: TenantId::new(1),
            rid: "x".into(),
            source_log_index: 1,
            reason: None,
        };
        let job = p.clone().into_new_job();
        let back = HnswDeletePayload::from_value(&job.payload).unwrap();
        assert_eq!(back.reason, None);
    }

    #[tokio::test]
    async fn scanner_finds_tombstones_skips_other_kinds() {
        let committer = open_committer();
        let tenant = TenantId::new(1);
        // log_index 1: upsert (skipped)
        committer
            .commit(tenant, upsert("a"), CommitOptions::default())
            .await
            .unwrap();
        // log_index 2: tombstone (kept)
        committer
            .commit(tenant, tombstone("a", Some("expired")), CommitOptions::default())
            .await
            .unwrap();
        // log_index 3: upsert (skipped)
        committer
            .commit(tenant, upsert("b"), CommitOptions::default())
            .await
            .unwrap();
        // log_index 4: tombstone (kept)
        committer
            .commit(tenant, tombstone("b", None), CommitOptions::default())
            .await
            .unwrap();

        let payloads = scan_tombstones_in_range(&*committer, tenant, 1, 100)
            .await
            .unwrap();
        assert_eq!(payloads.len(), 2);
        assert_eq!(payloads[0].rid, "a");
        assert_eq!(payloads[0].source_log_index, 2);
        assert_eq!(payloads[0].reason.as_deref(), Some("expired"));
        assert_eq!(payloads[1].rid, "b");
        assert_eq!(payloads[1].source_log_index, 4);
        assert_eq!(payloads[1].reason, None);
    }

    #[tokio::test]
    async fn scanner_respects_from_log_index() {
        let committer = open_committer();
        let tenant = TenantId::new(1);
        committer
            .commit(tenant, tombstone("a", None), CommitOptions::default())
            .await
            .unwrap(); // log_index 1
        committer
            .commit(tenant, tombstone("b", None), CommitOptions::default())
            .await
            .unwrap(); // log_index 2

        // Scanner starting from 2 sees only the second tombstone.
        let payloads = scan_tombstones_in_range(&*committer, tenant, 2, 100)
            .await
            .unwrap();
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].rid, "b");
    }

    #[tokio::test]
    async fn scanner_with_zero_from_clamps_to_one() {
        // log_index is 1-indexed; from_log_index=0 must clamp to 1
        // rather than scanning a non-existent log_index 0.
        let committer = open_committer();
        let tenant = TenantId::new(1);
        committer
            .commit(tenant, tombstone("a", None), CommitOptions::default())
            .await
            .unwrap();
        let payloads = scan_tombstones_in_range(&*committer, tenant, 0, 100)
            .await
            .unwrap();
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].source_log_index, 1);
    }

    #[tokio::test]
    async fn scanner_returns_empty_for_no_entries() {
        let committer = open_committer();
        let tenant = TenantId::new(1);
        let payloads = scan_tombstones_in_range(&*committer, tenant, 1, 100)
            .await
            .unwrap();
        assert!(payloads.is_empty());
    }

    #[tokio::test]
    async fn scanner_returns_empty_when_no_tombstones() {
        // Tenant with only Upsert mutations — scanner should not
        // confuse them for tombstones.
        let committer = open_committer();
        let tenant = TenantId::new(1);
        for tag in ["a", "b", "c"] {
            committer
                .commit(tenant, upsert(tag), CommitOptions::default())
                .await
                .unwrap();
        }
        let payloads = scan_tombstones_in_range(&*committer, tenant, 1, 100)
            .await
            .unwrap();
        assert!(payloads.is_empty());
    }

    #[tokio::test]
    async fn scanner_per_tenant_isolation() {
        let committer = open_committer();
        committer
            .commit(TenantId::new(1), tombstone("a", None), CommitOptions::default())
            .await
            .unwrap();
        committer
            .commit(TenantId::new(2), tombstone("b", None), CommitOptions::default())
            .await
            .unwrap();
        let t1 = scan_tombstones_in_range(&*committer, TenantId::new(1), 1, 100)
            .await
            .unwrap();
        let t2 = scan_tombstones_in_range(&*committer, TenantId::new(2), 1, 100)
            .await
            .unwrap();
        assert_eq!(t1.len(), 1);
        assert_eq!(t1[0].rid, "a");
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].rid, "b");
    }

    #[tokio::test]
    async fn enqueue_range_writes_jobs_and_advances_watermark() {
        let committer = open_committer();
        let jobs = open_jobs();
        let tenant = TenantId::new(1);
        // 5 entries: tombstone, upsert, tombstone, tombstone, upsert.
        committer
            .commit(tenant, tombstone("a", None), CommitOptions::default())
            .await
            .unwrap();
        committer
            .commit(tenant, upsert("b"), CommitOptions::default())
            .await
            .unwrap();
        committer
            .commit(tenant, tombstone("c", None), CommitOptions::default())
            .await
            .unwrap();
        committer
            .commit(tenant, tombstone("d", None), CommitOptions::default())
            .await
            .unwrap();
        committer
            .commit(tenant, upsert("e"), CommitOptions::default())
            .await
            .unwrap();

        let result = enqueue_range(&*committer, &*jobs, tenant, 1, 100)
            .await
            .unwrap();
        assert_eq!(result.tenant_id, tenant);
        assert_eq!(result.tombstones_found, 3);
        assert_eq!(result.jobs_enqueued, 3);
        assert_eq!(result.next_log_index_to_scan, 6); // last seen was 5

        let listed = jobs.list(Some(tenant), Some(JobState::Pending), 100).await.unwrap();
        assert_eq!(listed.len(), 3);
        for j in &listed {
            assert_eq!(j.kind, HNSW_DELETE_JOB_KIND);
            assert_eq!(j.priority, HNSW_DELETE_DEFAULT_PRIORITY);
        }
    }

    #[tokio::test]
    async fn enqueue_range_empty_range_does_not_advance() {
        let committer = open_committer();
        let jobs = open_jobs();
        let tenant = TenantId::new(99);
        let result = enqueue_range(&*committer, &*jobs, tenant, 1, 100)
            .await
            .unwrap();
        assert_eq!(result.tombstones_found, 0);
        assert_eq!(result.jobs_enqueued, 0);
        // No entries → next_log_index_to_scan stays at the input
        // (preserving idempotent retry).
        assert_eq!(result.next_log_index_to_scan, 1);
    }

    #[tokio::test]
    async fn enqueue_range_resume_from_watermark() {
        // First pass enqueues 1 tombstone, then more tombstones land
        // and the scanner resumes from `next_log_index_to_scan`.
        let committer = open_committer();
        let jobs = open_jobs();
        let tenant = TenantId::new(1);
        committer
            .commit(tenant, tombstone("a", None), CommitOptions::default())
            .await
            .unwrap();

        let r1 = enqueue_range(&*committer, &*jobs, tenant, 1, 100).await.unwrap();
        assert_eq!(r1.jobs_enqueued, 1);
        assert_eq!(r1.next_log_index_to_scan, 2);

        committer
            .commit(tenant, tombstone("b", None), CommitOptions::default())
            .await
            .unwrap();
        committer
            .commit(tenant, tombstone("c", None), CommitOptions::default())
            .await
            .unwrap();

        let r2 = enqueue_range(&*committer, &*jobs, tenant, r1.next_log_index_to_scan, 100)
            .await
            .unwrap();
        assert_eq!(r2.jobs_enqueued, 2);
        assert_eq!(r2.next_log_index_to_scan, 4);

        // Total enqueued = 3 (a, b, c). No duplicates.
        let listed = jobs.list(Some(tenant), Some(JobState::Pending), 100).await.unwrap();
        assert_eq!(listed.len(), 3);
    }

    #[tokio::test]
    async fn enqueue_one_writes_a_single_job() {
        let jobs = open_jobs();
        let payload = HnswDeletePayload {
            tenant_id: TenantId::new(1),
            rid: "x".into(),
            source_log_index: 42,
            reason: None,
        };
        let id = enqueue_one(&*jobs, payload.clone()).await.unwrap();
        let job = jobs.get(id).await.unwrap();
        assert_eq!(job.kind, HNSW_DELETE_JOB_KIND);
        assert_eq!(job.tenant_id, TenantId::new(1));
        assert_eq!(job.state, JobState::Pending);
        let back = HnswDeletePayload::from_value(&job.payload).unwrap();
        assert_eq!(back, payload);
    }

    #[tokio::test]
    async fn job_can_be_acquired_by_kind_filter() {
        // Confirms the producer/consumer wiring: a worker filtering
        // on HNSW_DELETE_JOB_KIND picks up exactly the jobs this
        // module emits, and ignores other kinds.
        let jobs = open_jobs();
        let tenant = TenantId::new(1);
        // Enqueue one of our kind + one unrelated.
        enqueue_one(
            &*jobs,
            HnswDeletePayload {
                tenant_id: tenant,
                rid: "ours".into(),
                source_log_index: 1,
                reason: None,
            },
        )
        .await
        .unwrap();
        jobs.enqueue(NewJob {
            tenant_id: tenant,
            kind: "rfc012.snapshot_create".to_string(),
            payload: serde_json::json!({}),
            priority: 5,
        })
        .await
        .unwrap();

        let lease = jobs
            .acquire_lease(
                WorkerId::new("test-worker"),
                Some(vec![HNSW_DELETE_JOB_KIND.to_string()]),
                None,
                30,
            )
            .await
            .unwrap()
            .expect("a job should be available");
        let payload = HnswDeletePayload::from_value(&lease.job.payload).unwrap();
        assert_eq!(payload.rid, "ours");
    }
}
