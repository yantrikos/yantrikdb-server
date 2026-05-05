//! RFC 010 — Commit substrate.
//!
//! ## Why this module exists
//!
//! Every write to YantrikDB — `remember`, `forget`, `relate`, `correct`,
//! tenant config changes — flows through this module instead of mutating
//! storage directly. The result: a single canonical mutation grammar
//! ([`MemoryMutation`]), a single durable log shape ([`memory_commit_log`]),
//! and a single trait ([`MutationCommitter`]) that the API handlers call.
//!
//! The local single-node implementation ([`LocalSqliteCommitter`]) lands
//! in PR-1 (this PR); the openraft-backed cluster implementation
//! (`RaftCommitter`) lands in PR-4. Both implement the same trait — the
//! API handlers don't change between modes.
//!
//! ## Design contract
//!
//! 1. **Idempotent on `op_id`**. Calling `commit` twice with the same
//!    op_id returns the original receipt. Clients can retry safely.
//! 2. **Monotonic `log_index` per tenant**. Each tenant has its own log;
//!    `log_index` increments by 1 within a tenant. Multi-tenant servers
//!    do NOT share log indices (avoids cross-tenant leakage and
//!    simplifies per-tenant restore).
//! 3. **Versioned every event**. Every [`MemoryMutation`] variant
//!    implements [`crate::version::VersionedEvent`]. Replay-safe across
//!    rolling upgrades within a major.
//! 4. **Tombstone-shaped grammar from day one**. `TombstoneMemory` and
//!    `PurgeMemory` variants exist now, even though RFC 011 hasn't yet
//!    implemented their full semantics. This locks the log grammar
//!    against later breaking changes.
//! 5. **Async dyn dispatch**. The trait uses `async_trait` so we can
//!    store `Arc<dyn MutationCommitter>` and swap implementations at
//!    runtime without touching handler code.
//!
//! ## What's in this module
//!
//! | File | Purpose |
//! |---|---|
//! | [`trait_def`] | `MutationCommitter` trait + `CommitReceipt` + `CommitOptions` + `CommitError` |
//! | [`mutation`] | `MemoryMutation` enum + `VersionedEvent` impl + `OpId` |
//! | [`local`] | `LocalSqliteCommitter` (in-memory backing in PR-1; SQLite persistence in PR-2) |
//!
//! ## What lands in subsequent PRs
//!
//! - **RFC 010 PR-2** (`memory_commit_log` table): replaces the in-memory
//!   `Vec` in `LocalSqliteCommitter` with SQLite persistence + replay tests.
//! - **RFC 010 PR-3** (canonical mutation grammar): expands variants
//!   beyond the initial set, locks the wire encoding for v1.0.
//! - **RFC 010 PR-4** (openraft adapter): adds `RaftCommitter` impl.
//! - **RFC 010 PR-5** (Jepsen debug hooks): `read_range` exposed via
//!   `/v1/debug/history/{tenant_id}` for fault injection tests.
//! - **RFC 011** (forget): `TombstoneMemory` and `PurgeMemory` get real
//!   semantics; until then they're accepted into the grammar but rejected
//!   at apply time with "not yet implemented".

pub mod applier;
pub mod local;
pub mod materialize;
pub mod mutation;
pub mod retention;
pub mod submitter;
pub mod trait_def;

pub use applier::{Applier, ApplyError, LocalApplier};
pub use local::LocalSqliteCommitter;
pub use materialize::{
    Embedder, EntityExtractor, LocalMaterializer, MaterializeError, Materializer, RememberRequest,
};
pub use mutation::{MemoryMutation, OpId, TenantId};
pub use retention::{
    BackupWatermarkContributor, FollowerLagContributor, HnswWatermarkContributor,
    RetentionContributor, RetentionError, RetentionPolicy, RetentionRegistry, SafePurgeWatermark,
    TombstoneRetentionContributor,
};
pub use submitter::{LocalSqliteSubmitter, Submitter};
pub use trait_def::{
    CommitError, CommitObserver, CommitOptions, CommitReceipt, CommittedEntry, MutationCommitter,
};
