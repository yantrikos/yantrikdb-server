//! RFC 019 — Durable Background Jobs / Leases (narrow scope).
//!
//! ## Why this module exists (per gpt-5.5 brainstorm 7c36ea8b)
//!
//! Five upcoming RFCs all need background work:
//! - RFC 011 — HNSW delete queue (drains tombstones into vector index)
//! - RFC 012 — snapshot create/upload/validate
//! - RFC 013 — index reconciliation (catch HNSW up to commit log)
//! - RFC 013-B — re-embedding (when embedding model changes)
//! - RFC 015 — cache warming
//!
//! Without a unified job substrate, each RFC invents its own worker
//! loop with inconsistent retry semantics, broken leader-failover
//! behavior, and duplicate execution under restart. This module is the
//! shared substrate.
//!
//! ## Narrow scope (locked)
//!
//! Per multi-voice consensus: ship `Job + JobQueue + Lease` traits only.
//! Defer to post-v1:
//! - Retry policies (each consumer expresses its own)
//! - Cron / interval scheduling (one-shot only for now)
//! - DAG execution (no inter-job dependencies)
//! - Priority preemption beyond pickup-order priority
//!
//! ## Design contract
//!
//! 1. **Durable**: jobs survive process restart (SQLite-backed in PR-1).
//! 2. **At-least-once**: leases have TTLs; if a worker crashes mid-job,
//!    the lease expires and another worker picks up. Consumers MUST
//!    write idempotent execution paths.
//! 3. **Per-tenant scoped**: every job carries `tenant_id`. Multi-tenant
//!    workers can pick from any tenant or filter to a specific one.
//! 4. **Kind-filtered pickup**: workers ask for specific job kinds (e.g.
//!    "give me a `hnsw_delete` job"); other kinds stay queued.
//! 5. **Lease heartbeating**: long-running jobs heartbeat to extend TTL.
//!    No heartbeat for `lease_ttl_secs` → job reverts to Pending.
//!
//! ## What this module does NOT do
//!
//! - Run the workers. Each consumer (RFC 011, 012, 013, ...) implements
//!   its own worker loop using this trait surface.
//! - Schedule jobs by time. Use the local commit log (`enqueue at`) and
//!   add a "ready_at" filter when scheduling becomes a real need.
//! - Provide retry policies. Workers `complete(lease, Outcome::Failed)`
//!   when they can't make progress; if retry is desired, the consumer
//!   re-enqueues with a fresh job.

pub mod local;
pub mod trait_def;

pub use local::LocalSqliteJobQueue;
pub use trait_def::{
    JobError, JobId, JobQueue, JobRecord, JobState, Lease, LeaseId, NewJob, Outcome, WorkerId,
};
