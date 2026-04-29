//! RFC 012 — Backup / Restore / Disaster Recovery substrate.
//!
//! ## What lives here
//!
//! - [`manifest::SnapshotManifest`] — the per-tenant snapshot
//!   descriptor that pins everything a restore needs: SQLite
//!   checkpoint reference, HNSW manifest watermarks, oplog
//!   watermark, schema_version, forget_floor, encryption metadata.
//! - [`backend::BackupBackend`] trait — abstract storage backend
//!   (manifests + content blobs).
//! - [`backend::LocalFsBackend`] — disk-backed implementation.
//!
//! ## What ships in PR-1 vs later PRs
//!
//! PR-1 (this): manifest format + backend trait + local fs backend.
//! PR-2 (saga #148): `yantrikdb backup restore` command + validation
//! (checksum, schema_version compat, tombstone floor refusal,
//! HNSW dim/model match).
//! PR-3 (saga #149): chaos test + restore rehearsal in CI.
//!
//! Object-store backends (S3 / GCS / Azure via the `object_store`
//! crate) are deferred to a follow-up — adding that dep is a
//! standalone change that doesn't gate the manifest contract.
//!
//! ## Restore-no-resurrect invariant
//!
//! The `forget_floor` field is the load-bearing piece. RFC 011
//! guarantees that a tombstoned memory cannot reappear after a
//! backup/restore cycle. We enforce that by:
//!
//! 1. At snapshot time: capture `forget_floor` = the lowest
//!    log_index that contains any tombstone the snapshot must
//!    preserve.
//! 2. At restore time: if the destination cluster's tombstone
//!    floor (highest applied tombstone log_index) is *higher*
//!    than the manifest's `forget_floor`, the restore would
//!    resurrect already-deleted memories. Restore refuses.
//! 3. If it's *lower*, restore proceeds; the destination's
//!    tombstones get overwritten by the snapshot's, but no
//!    deleted memory comes back.
//!
//! See `manifest::SnapshotManifest::validate_for_restore` for
//! the boundary-validation logic that PR-2's restore command
//! invokes.

pub mod backend;
pub mod manifest;

pub use backend::{BackupBackend, BackupBackendError, LocalFsBackend};
pub use manifest::{
    EncryptionMetadata, HnswSnapshotEntry, ManifestValidationError, SnapshotManifest,
    SnapshotManifestVersion,
};
