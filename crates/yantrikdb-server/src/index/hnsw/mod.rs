//! HNSW lifecycle module — manifest + reconciliation.
//!
//! See [`super`] for the design rationale and [`manifest`] for the
//! storage contract.

pub mod compaction;
pub mod manifest;
pub mod reconcile;
pub mod shadow;

pub use compaction::{
    process_one_delete_job, CompactionError, HnswCompactor, NoopHnswCompactor, ProcessOutcome,
};
pub use manifest::{
    DistanceMetric, HnswManifest, HnswManifestError, HnswManifestStore, SqliteHnswManifestStore,
};
pub use reconcile::{Reconciler, ReconciliationReport, ReconciliationStatus};
pub use shadow::{
    DualReadMerger, HitSource, InMemoryMigrationStore, MigrationProgress, MigrationStateStore,
    ScoredHit, ShadowIndexConfig, ShadowMigrationError, ShadowMigrationPhase, ShadowMigrationState,
};
