//! RFC 011-B — Tombstone implementation over the commit log.
//!
//! ## What this module ships
//!
//! - [`TombstoneIndex`] — a per-process index of `(tenant_id, rid)` pairs
//!   that have been tombstoned. Implements [`crate::cache::TombstoneProvider`]
//!   so any cache wrapped in [`crate::cache::TombstoneAwareCache`] can ask
//!   "is this rid tombstoned?" and get an answer in O(1) hash lookup.
//!
//! ## How it stays in sync with the commit log
//!
//! 1. **Hydration on startup**: [`TombstoneIndex::from_commit_log`] scans
//!    `memory_commit_log WHERE op_kind = 'TombstoneMemory'` and loads every
//!    historical tombstone into memory. The index is sound on cold-start.
//! 2. **Live updates**: every `LocalSqliteCommitter::commit(TombstoneMemory)`
//!    that succeeds calls [`TombstoneIndex::record`] before returning.
//!    Cache reads issued after the receipt is observed see the tombstone.
//! 3. **Purges remove entries**: when RFC 011 PR-3 ships the
//!    `PurgeMemory` apply path, [`TombstoneIndex::record_purge`] drops the
//!    rid (the data is physically gone; no further filtering needed).
//!
//! ## Why an in-memory index instead of querying SQL on every check
//!
//! Per the trait doc: `is_tombstoned` is on the cache hot path. SQLite
//! query overhead is ~10-50µs per call; an in-memory `HashSet` lookup is
//! ~50-100ns. At 10K recalls/sec with average 20 rids each, that's the
//! difference between 5ms/sec and 5sec/sec of CPU time.
//!
//! ## Memory footprint
//!
//! `HashSet<String>` with rid strings averaging 32 bytes plus hash
//! overhead is ~80 bytes per tombstone. 1M tombstoned rids ≈ 80 MB.
//! For tenants approaching that scale, RFC 011 PR-3's purge worker
//! reclaims old tombstones once the safe-purge watermark passes.
//!
//! ## What this PR does NOT include
//!
//! - HNSW delete queue (RFC 011 PR-3 / saga task #145).
//! - Crypto-shred (RFC 011 PR-4 / saga task #146).
//! - Recall-path integration (filtering tombstoned hits at query time —
//!   that lands when the recall engine moves to the commit-log substrate).
//! - The public `forget` HTTP API moving from legacy in-place delete to
//!   commit-log mutation. Wiring that requires touching the engine crate;
//!   handled in a follow-up PR scoped to the API layer.

pub mod crypto_shred;
pub mod delete_queue;
pub mod tombstone;

pub use crypto_shred::{
    CryptoShredConfig, CryptoShredError, CryptoShredder, ShredKeyResult, ShredOutcome, ShredPlan,
};
pub use delete_queue::{
    enqueue_one, enqueue_range, scan_tombstones_in_range, EnqueueRangeResult, HnswDeletePayload,
    HNSW_DELETE_DEFAULT_PRIORITY, HNSW_DELETE_JOB_KIND,
};

pub use tombstone::TombstoneIndex;
