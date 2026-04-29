//! RFC 012 PR-2 — restore command + validation.
//!
//! ## What this module owns
//!
//! - [`validate::RestoreValidator`] — runs every safety check
//!   before a single byte is written to the destination data dir.
//! - [`exec::RestoreExecutor`] — pulls content blobs from the
//!   backend, verifies checksums, writes to the destination data
//!   dir according to the chosen [`exec::RestoreMode`].
//! - [`RestorePlan`] — the validated, ready-to-execute description
//!   of a restore. Operators inspect this before running exec; CI
//!   restore-rehearsal harnesses (PR-3 / saga #149) consume it
//!   directly.
//!
//! ## Restore modes
//!
//! | Mode | When |
//! |---|---|
//! | `NewCluster` | Bringing up a fresh cluster from a backup; data dir is empty. |
//! | `SingleTenant` | Restoring one tenant into an existing cluster without touching others. |
//! | `WipeAndRestore` | Replacing the entire data dir with the snapshot's contents (DR scenario). |
//!
//! ## Restore-no-resurrect, end-to-end
//!
//! Validation is the load-bearing piece. RFC 011's invariant says a
//! tombstoned memory cannot reappear after a backup/restore cycle.
//! Validation refuses the restore if the destination's
//! tombstone floor (highest applied tombstone log_index) is above
//! the snapshot's `forget_floor`. See
//! `crate::backup::manifest::SnapshotManifest::validate_for_restore`
//! — that's the function this module orchestrates.

pub mod exec;
pub mod validate;

pub use exec::{RestoreExecutor, RestoreExecutorError, RestoreMode, RestoreOutcome};
pub use validate::{
    DestinationState, RestorePlan, RestorePlanItem, RestoreValidator, RestoreValidatorError,
};
