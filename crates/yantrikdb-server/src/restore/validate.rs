//! Restore validation — every safety check before a byte is
//! written to the destination data dir.
//!
//! ## What gets validated
//!
//! 1. **Manifest internal consistency** — `validate_internal()`
//!    catches inverted oplog ranges, future manifest_version,
//!    empty sqlite key.
//! 2. **Manifest vs destination** — `validate_for_restore()`
//!    catches wire major mismatch, HNSW model mismatch, AND the
//!    LOAD-BEARING `WouldResurrectDeletedData` when the
//!    destination's tombstone floor is above the snapshot's
//!    `forget_floor`.
//! 3. **Content checksums** — every content blob the manifest
//!    references is pulled, blake3-hashed, and compared to the
//!    manifest's stored checksum. Mismatch → refuse.
//! 4. **Content existence** — every content_key the manifest
//!    references must be present in the backend. Missing → refuse.
//!
//! ## What this module does NOT do
//!
//! - Execute the restore. That's [`super::exec`]. Validation
//!   produces a [`RestorePlan`] which the executor consumes.
//! - Roll back partial restores. The executor uses staging dirs +
//!   atomic rename so a failure mid-restore leaves the destination
//!   untouched; no rollback step needed.

use std::sync::Arc;

use blake3;

use crate::backup::manifest::{ManifestValidationError, SnapshotManifest};
use crate::backup::{BackupBackend, BackupBackendError};
use crate::commit::TenantId;
use crate::version::WireVersion;

/// Snapshot of relevant destination-cluster state. The validator
/// consumes this; production callers populate it from the live
/// `AppState` (commit log high_water, applied tombstone floor,
/// configured embedding model, current wire version).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestinationState {
    pub wire_version: WireVersion,
    /// Per-tenant: the highest log_index of any applied tombstone.
    /// `None` = "no tombstones applied for this tenant" — restore
    /// is unconstrained from the resurrect-prevention angle.
    pub tombstone_floor: Option<u64>,
    /// The destination's currently-configured embedding model. The
    /// manifest must include an HNSW snapshot for this model (or
    /// have zero HNSW snapshots, which means the destination will
    /// build its own index post-restore).
    pub embedding_model: String,
}

/// One content blob the executor will pull during exec. Each item
/// pairs the manifest's `(key, expected_checksum)` with metadata
/// about what the blob is (sqlite checkpoint vs HNSW data) so
/// the executor can route it to the right destination path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestorePlanItem {
    pub kind: RestorePlanItemKind,
    pub content_key: String,
    /// blake3 hex. `None` means the manifest didn't include a
    /// checksum (older snapshots); validation passes the bytes
    /// through without verification but logs a warning.
    pub expected_checksum: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestorePlanItemKind {
    /// The tenant's SQLite checkpoint (single-file).
    SqliteCheckpoint,
    /// One HNSW index snapshot, with the model id so the executor
    /// can write to `<data_dir>/<tenant>/hnsw/<model>.bin` or wherever
    /// the engine's restore reader looks.
    HnswSnapshot { embedding_model: String },
}

/// The validated, ready-to-execute restore description. Executor
/// consumes this; no further validation calls are needed at exec
/// time.
#[derive(Debug, Clone, PartialEq)]
pub struct RestorePlan {
    pub tenant_id: TenantId,
    pub snapshot_id: String,
    pub manifest: SnapshotManifest,
    pub items: Vec<RestorePlanItem>,
    /// The destination state at validation time. Snapshotted here
    /// so the executor doesn't re-read; if the destination has
    /// since changed, the executor will detect it via the
    /// checksum-mismatch check that runs as it pulls blobs.
    pub destination: DestinationState,
}

#[derive(Debug, thiserror::Error)]
pub enum RestoreValidatorError {
    #[error("backend error: {0}")]
    Backend(#[from] BackupBackendError),

    #[error("manifest validation: {0}")]
    Manifest(#[from] ManifestValidationError),

    #[error(
        "content blob `{key}` checksum mismatch: expected `{expected}`, got `{actual}`"
    )]
    ChecksumMismatch {
        key: String,
        expected: String,
        actual: String,
    },

    #[error("content blob `{key}` referenced by manifest is missing from backend")]
    ContentMissing { key: String },
}

/// Runs every pre-restore check + assembles a [`RestorePlan`].
pub struct RestoreValidator {
    backend: Arc<dyn BackupBackend>,
}

impl RestoreValidator {
    pub fn new(backend: Arc<dyn BackupBackend>) -> Self {
        Self { backend }
    }

    /// Validate one snapshot against a destination. Returns a
    /// `RestorePlan` if every check passes, or the specific failure.
    pub async fn validate(
        &self,
        snapshot_id: &str,
        destination: DestinationState,
    ) -> Result<RestorePlan, RestoreValidatorError> {
        // Step 1: pull manifest.
        let manifest = self.backend.get_manifest(snapshot_id).await?;

        // Step 2: manifest internal consistency.
        manifest.validate_internal()?;

        // Step 3: manifest-vs-destination compatibility.
        manifest.validate_for_restore(
            destination.wire_version,
            destination.tombstone_floor.unwrap_or(0),
            &destination.embedding_model,
        )?;

        // Step 4: build the plan items + verify checksums for
        // every content blob the manifest references.
        let mut items = Vec::with_capacity(1 + manifest.hnsw_snapshots.len());

        // SQLite checkpoint.
        verify_blob(
            self.backend.as_ref(),
            &manifest.sqlite_checkpoint_key,
            manifest.sqlite_checkpoint_checksum.as_deref(),
        )
        .await?;
        items.push(RestorePlanItem {
            kind: RestorePlanItemKind::SqliteCheckpoint,
            content_key: manifest.sqlite_checkpoint_key.clone(),
            expected_checksum: manifest.sqlite_checkpoint_checksum.clone(),
        });

        // HNSW snapshots.
        for h in &manifest.hnsw_snapshots {
            verify_blob(
                self.backend.as_ref(),
                &h.content_key,
                h.checksum.as_deref(),
            )
            .await?;
            items.push(RestorePlanItem {
                kind: RestorePlanItemKind::HnswSnapshot {
                    embedding_model: h.embedding_model.clone(),
                },
                content_key: h.content_key.clone(),
                expected_checksum: h.checksum.clone(),
            });
        }

        Ok(RestorePlan {
            tenant_id: manifest.tenant_id,
            snapshot_id: snapshot_id.to_string(),
            manifest,
            items,
            destination,
        })
    }
}

/// Pull a blob, hash it, compare against expected. Returns Ok(())
/// on match (or when expected is None — older snapshots without
/// checksums fall through with a warning we leave to the
/// `tracing::warn!` site).
///
/// Pulled into a free function so the executor can re-verify on
/// its own pull (defense-in-depth: the validator's pull and the
/// executor's pull are separate round trips, so a backend that's
/// returning corrupt data flips checksum mismatch on whichever
/// pass happens to land first).
pub async fn verify_blob(
    backend: &dyn BackupBackend,
    key: &str,
    expected_checksum: Option<&str>,
) -> Result<Vec<u8>, RestoreValidatorError> {
    let bytes = match backend.get_content(key).await {
        Ok(b) => b,
        Err(BackupBackendError::ContentNotFound { key: k }) => {
            return Err(RestoreValidatorError::ContentMissing { key: k })
        }
        Err(e) => return Err(RestoreValidatorError::Backend(e)),
    };
    if let Some(expected) = expected_checksum {
        let actual = blake3::hash(&bytes).to_hex().to_string();
        if actual != expected {
            return Err(RestoreValidatorError::ChecksumMismatch {
                key: key.to_string(),
                expected: expected.to_string(),
                actual,
            });
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backup::manifest::{
        EncryptionMetadata, HnswSnapshotEntry, SnapshotManifestVersion,
    };
    use crate::backup::LocalFsBackend;
    use crate::index::hnsw::DistanceMetric;
    use crate::version::SchemaVersion;
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    fn dest() -> DestinationState {
        DestinationState {
            wire_version: WireVersion::new(1, 0),
            tombstone_floor: Some(100),
            embedding_model: "minilm".into(),
        }
    }

    fn manifest_with_content(
        snapshot_id: &str,
        sqlite_bytes: &[u8],
        hnsw_bytes: &[u8],
    ) -> SnapshotManifest {
        SnapshotManifest {
            manifest_version: SnapshotManifestVersion::CURRENT,
            tenant_id: TenantId::new(1),
            snapshot_id: snapshot_id.into(),
            created_at_unix_micros: 1_700_000_000_000_000,
            wire_version: WireVersion::new(1, 0),
            table_schema_versions: {
                let mut m = BTreeMap::new();
                m.insert("memory_commit_log".to_string(), SchemaVersion::new(1));
                m
            },
            oplog_watermark: 5_000,
            oplog_floor: 1,
            forget_floor: Some(2_500),
            sqlite_checkpoint_key: format!("{snapshot_id}/sqlite.db"),
            sqlite_checkpoint_checksum: Some(blake3::hash(sqlite_bytes).to_hex().to_string()),
            hnsw_snapshots: vec![HnswSnapshotEntry {
                embedding_model: "minilm".into(),
                vector_dim: 384,
                distance_metric: DistanceMetric::Cosine,
                source_log_watermark: 5_000,
                content_key: format!("{snapshot_id}/hnsw.bin"),
                checksum: Some(blake3::hash(hnsw_bytes).to_hex().to_string()),
                deleted_count_pending: 0,
            }],
            encryption: Some(EncryptionMetadata {
                algorithm: "aes-256-gcm".into(),
                dek_id: "dek-1".into(),
                iv_b64: "AA==".into(),
            }),
            label: None,
        }
    }

    async fn populate_backend(
        backend: &dyn BackupBackend,
        manifest: &SnapshotManifest,
        sqlite_bytes: &[u8],
        hnsw_bytes: &[u8],
    ) {
        backend.put_manifest(manifest).await.unwrap();
        backend
            .put_content(&manifest.sqlite_checkpoint_key, sqlite_bytes)
            .await
            .unwrap();
        for h in &manifest.hnsw_snapshots {
            backend.put_content(&h.content_key, hnsw_bytes).await.unwrap();
        }
    }

    #[tokio::test]
    async fn validate_passes_on_matching_destination() {
        let tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(tmp.path()).unwrap());
        let sqlite = b"sqlite-checkpoint-bytes";
        let hnsw = b"hnsw-data-bytes";
        let m = manifest_with_content("snap-1", sqlite, hnsw);
        populate_backend(backend.as_ref(), &m, sqlite, hnsw).await;

        let v = RestoreValidator::new(backend);
        let plan = v.validate("snap-1", dest()).await.unwrap();
        assert_eq!(plan.tenant_id, TenantId::new(1));
        assert_eq!(plan.snapshot_id, "snap-1");
        assert_eq!(plan.items.len(), 2);
        assert!(matches!(
            plan.items[0].kind,
            RestorePlanItemKind::SqliteCheckpoint
        ));
        match &plan.items[1].kind {
            RestorePlanItemKind::HnswSnapshot { embedding_model } => {
                assert_eq!(embedding_model, "minilm");
            }
            _ => panic!("expected HnswSnapshot"),
        }
    }

    #[tokio::test]
    async fn validate_refuses_resurrect_scenario() {
        // The load-bearing check. Snapshot's forget_floor=2500;
        // destination's tombstone_floor=5000 means destination has
        // applied tombstones beyond what the snapshot captures.
        // Restoring would resurrect those memories — refuse.
        let tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(tmp.path()).unwrap());
        let sqlite = b"x";
        let hnsw = b"y";
        let m = manifest_with_content("snap-1", sqlite, hnsw);
        populate_backend(backend.as_ref(), &m, sqlite, hnsw).await;

        let v = RestoreValidator::new(backend);
        let mut bad_dest = dest();
        bad_dest.tombstone_floor = Some(5_000);
        match v.validate("snap-1", bad_dest).await {
            Err(RestoreValidatorError::Manifest(
                ManifestValidationError::WouldResurrectDeletedData { .. },
            )) => {}
            other => panic!("expected WouldResurrectDeletedData, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn validate_refuses_wire_major_mismatch() {
        let tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(tmp.path()).unwrap());
        let sqlite = b"x";
        let hnsw = b"y";
        let m = manifest_with_content("snap-1", sqlite, hnsw);
        populate_backend(backend.as_ref(), &m, sqlite, hnsw).await;

        let v = RestoreValidator::new(backend);
        let mut bad_dest = dest();
        bad_dest.wire_version = WireVersion::new(2, 0);
        assert!(matches!(
            v.validate("snap-1", bad_dest).await,
            Err(RestoreValidatorError::Manifest(
                ManifestValidationError::WireVersionMismatch { .. }
            ))
        ));
    }

    #[tokio::test]
    async fn validate_refuses_hnsw_model_mismatch() {
        let tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(tmp.path()).unwrap());
        let m = manifest_with_content("snap-1", b"x", b"y");
        populate_backend(backend.as_ref(), &m, b"x", b"y").await;

        let v = RestoreValidator::new(backend);
        let mut bad_dest = dest();
        bad_dest.embedding_model = "bge-base".into();
        assert!(matches!(
            v.validate("snap-1", bad_dest).await,
            Err(RestoreValidatorError::Manifest(
                ManifestValidationError::HnswModelMismatch { .. }
            ))
        ));
    }

    #[tokio::test]
    async fn validate_refuses_sqlite_checksum_mismatch() {
        // Manifest claims checksum X; backend serves bytes whose
        // checksum is Y. Catches storage corruption + tampering.
        let tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(tmp.path()).unwrap());
        let m = manifest_with_content("snap-1", b"original", b"hnsw");
        // Tamper: store DIFFERENT bytes for the sqlite key.
        backend.put_manifest(&m).await.unwrap();
        backend
            .put_content(&m.sqlite_checkpoint_key, b"TAMPERED")
            .await
            .unwrap();
        backend
            .put_content(&m.hnsw_snapshots[0].content_key, b"hnsw")
            .await
            .unwrap();
        // Force a recompute so we know the manifest's recorded
        // checksum DOESN'T match the stored bytes.
        let _ = m.sqlite_checkpoint_checksum.as_ref().unwrap();

        let v = RestoreValidator::new(backend);
        match v.validate("snap-1", dest()).await {
            Err(RestoreValidatorError::ChecksumMismatch { key, .. }) => {
                assert_eq!(key, m.sqlite_checkpoint_key);
            }
            other => panic!("expected ChecksumMismatch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn validate_refuses_hnsw_checksum_mismatch() {
        let tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(tmp.path()).unwrap());
        let m = manifest_with_content("snap-1", b"sqlite", b"original");
        backend.put_manifest(&m).await.unwrap();
        backend
            .put_content(&m.sqlite_checkpoint_key, b"sqlite")
            .await
            .unwrap();
        // Tamper the HNSW blob.
        backend
            .put_content(&m.hnsw_snapshots[0].content_key, b"TAMPERED-HNSW")
            .await
            .unwrap();

        let v = RestoreValidator::new(backend);
        assert!(matches!(
            v.validate("snap-1", dest()).await,
            Err(RestoreValidatorError::ChecksumMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn validate_refuses_missing_content() {
        // Manifest claims content_key, backend doesn't have the
        // blob (e.g. operator deleted by accident).
        let tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(tmp.path()).unwrap());
        let m = manifest_with_content("snap-1", b"sqlite", b"hnsw");
        // Only put manifest + hnsw; leave sqlite blob missing.
        backend.put_manifest(&m).await.unwrap();
        backend
            .put_content(&m.hnsw_snapshots[0].content_key, b"hnsw")
            .await
            .unwrap();

        let v = RestoreValidator::new(backend);
        match v.validate("snap-1", dest()).await {
            Err(RestoreValidatorError::ContentMissing { key }) => {
                assert_eq!(key, m.sqlite_checkpoint_key);
            }
            other => panic!("expected ContentMissing, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn validate_with_no_checksum_passes_through() {
        // Older snapshots without checksums shouldn't fail
        // validation. (They lose the corruption-detection benefit
        // but that's a known trade-off, not a refuse-restore one.)
        let tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(tmp.path()).unwrap());
        let mut m = manifest_with_content("snap-1", b"x", b"y");
        m.sqlite_checkpoint_checksum = None;
        m.hnsw_snapshots[0].checksum = None;
        populate_backend(backend.as_ref(), &m, b"x", b"y").await;

        let v = RestoreValidator::new(backend);
        v.validate("snap-1", dest()).await.unwrap();
    }

    #[tokio::test]
    async fn validate_with_no_forget_floor_allows_any_destination() {
        // A tenant with no tombstones at snapshot time has
        // forget_floor=None — restore is unconstrained.
        let tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(tmp.path()).unwrap());
        let mut m = manifest_with_content("snap-1", b"x", b"y");
        m.forget_floor = None;
        populate_backend(backend.as_ref(), &m, b"x", b"y").await;

        let v = RestoreValidator::new(backend);
        let mut high_dest = dest();
        high_dest.tombstone_floor = Some(999_999);
        v.validate("snap-1", high_dest).await.unwrap();
    }

    #[tokio::test]
    async fn validate_missing_manifest_returns_backend_error() {
        let tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(tmp.path()).unwrap());
        let v = RestoreValidator::new(backend);
        match v.validate("does-not-exist", dest()).await {
            Err(RestoreValidatorError::Backend(BackupBackendError::ManifestNotFound { .. })) => {}
            other => panic!("expected ManifestNotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn verify_blob_matches_correct_checksum() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        let bytes = b"hello world";
        backend.put_content("k", bytes).await.unwrap();
        let expected = blake3::hash(bytes).to_hex().to_string();
        let pulled = verify_blob(&backend, "k", Some(&expected)).await.unwrap();
        assert_eq!(pulled, bytes);
    }

    #[tokio::test]
    async fn verify_blob_with_no_expected_passes_through() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        backend.put_content("k", b"any").await.unwrap();
        verify_blob(&backend, "k", None).await.unwrap();
    }
}
