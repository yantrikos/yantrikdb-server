//! RFC 013 Phase 1 — index lifecycle substrate.
//!
//! ## What lives here
//!
//! Per-tenant index manifests + reconciliation contracts. The actual
//! HNSW data structure lives in the engine crate (`yantrikdb`); this
//! module owns the **manifest** that tracks index/log alignment and the
//! **reconciler** that detects drift between them.
//!
//! ## Why a manifest is required
//!
//! v0.5.2 had the bug class: a follower applied the oplog so memories
//! landed in SQLite, but its HNSW index was not rebuilt from those
//! entries — so semantic recall over the affected window returned
//! nothing. The fix has three parts:
//!
//! 1. The manifest stores [`HnswManifest::source_log_watermark`] — the
//!    high water mark of the commit log as of the last successful index
//!    update.
//! 2. On startup the [`reconcile::Reconciler`] compares the watermark
//!    against `memory_commit_log.high_watermark` and reports a gap.
//! 3. The recall path consults manifest health before serving (RFC
//!    013-A wires this; the engine crate eventually reads/writes the
//!    manifest as it updates HNSW state).
//!
//! ## Sub-PR plan
//!
//! | Sub-PR | What |
//! |---|---|
//! | **Phase 1 (this)** | Manifest struct + SqliteHnswManifestStore + Reconciler + m003 migration. No engine wiring. |
//! | Phase 2 (013-B) | Embedding model migration via shadow index, RFC 019 background re-embed job. |
//!
//! ## What this PR does NOT include
//!
//! - Engine integration: the engine crate does not yet read/write the
//!   manifest. That wiring lands when the engine moves to the commit-log
//!   substrate.
//! - HNSW rebuild logic: the reconciler reports a gap; it does not fix
//!   it. Phase 2 / 013-B add the rebuild worker via RFC 019 jobs.
//! - Recall-path consultation: the recall path doesn't yet refuse on
//!   stale manifests. That hook lands once the engine reads the manifest.

pub mod entity_adjacency;
pub mod hnsw;

pub use entity_adjacency::{
    AdjacencyRow, EntityAdjacencyIndex, EntityAdjacencyProvider,
};
