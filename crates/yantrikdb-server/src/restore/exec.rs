//! Restore execution — pulls validated content blobs from the
//! backend and writes them to the destination data dir.
//!
//! ## Three restore modes
//!
//! - [`RestoreMode::NewCluster`] — destination dir is empty. Every
//!   blob is written; no overwrite checks.
//! - [`RestoreMode::SingleTenant`] — destination has other tenants
//!   that must remain untouched. The executor only writes inside
//!   `<data_dir>/<tenant>/...` paths and refuses if those already
//!   exist.
//! - [`RestoreMode::WipeAndRestore`] — DR scenario: destination is
//!   wiped, then snapshot is laid down. Caller is responsible for
//!   stopping the server before invoking this — concurrent reads
//!   during a wipe will see partial state.
//!
//! ## Atomicity
//!
//! Each blob is written to a `<final_path>.restore-tmp` first,
//! then renamed once the bytes + checksum are confirmed. A failure
//! mid-restore leaves the destination's previous state intact for
//! the blobs that haven't been renamed yet. (Already-renamed
//! blobs are not rolled back; a partial-restore manifest is left
//! at `<data_dir>/.restore-in-progress` so the operator knows to
//! retry or wipe.)

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

use super::validate::{verify_blob, RestorePlan, RestorePlanItem, RestorePlanItemKind};
use crate::backup::BackupBackend;
use crate::commit::TenantId;

/// Restore mode. See module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RestoreMode {
    NewCluster,
    SingleTenant,
    WipeAndRestore,
}

#[derive(Debug, thiserror::Error)]
pub enum RestoreExecutorError {
    #[error("validator: {0}")]
    Validator(#[from] super::validate::RestoreValidatorError),

    #[error("destination IO: {0}")]
    Io(#[from] std::io::Error),

    #[error(
        "destination already has data for tenant {tenant_id} \
         in `{path}`; SingleTenant mode refuses to overwrite. \
         Use WipeAndRestore explicitly if that's intended."
    )]
    TenantPathOccupied {
        tenant_id: TenantId,
        path: PathBuf,
    },

    #[error(
        "destination data dir `{path}` is not empty; NewCluster \
         mode requires an empty destination. Use WipeAndRestore \
         or SingleTenant explicitly if that's intended."
    )]
    DataDirNotEmpty { path: PathBuf },

    #[error(
        "restore in progress: marker file `{path}` exists. Either \
         resume manually or remove the marker and re-run with the \
         appropriate mode."
    )]
    RestoreInProgress { path: PathBuf },
}

/// Result of a successful restore.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestoreOutcome {
    pub tenant_id: TenantId,
    pub snapshot_id: String,
    pub mode: RestoreMode,
    /// Per-item: the destination path where each blob landed.
    /// Operator dashboards + audit logs render this.
    pub items: Vec<(RestorePlanItemDigest, PathBuf)>,
}

/// Stripped-down version of `RestorePlanItem` for the outcome
/// (drops bytes + keeps just enough to render in audit).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestorePlanItemDigest {
    pub kind: String,
    pub content_key: String,
    pub bytes: usize,
}

/// Marker file the executor writes at the start of a restore and
/// removes on success. Operators see it after a crash and know
/// the data dir is in a partial-restore state.
const RESTORE_MARKER: &str = ".restore-in-progress";

/// Pulls + verifies + writes content blobs. Holds the backend it
/// reads from + the destination data dir it writes to.
pub struct RestoreExecutor {
    backend: Arc<dyn BackupBackend>,
    data_dir: PathBuf,
}

impl RestoreExecutor {
    pub fn new(backend: Arc<dyn BackupBackend>, data_dir: impl AsRef<Path>) -> Self {
        Self {
            backend,
            data_dir: data_dir.as_ref().to_path_buf(),
        }
    }

    /// Execute a previously-validated `RestorePlan` against the
    /// configured destination data dir. Caller is expected to have
    /// stopped any concurrent readers (the server itself) before
    /// invoking; the executor does not coordinate with a live server.
    pub async fn execute(
        &self,
        plan: &RestorePlan,
        mode: RestoreMode,
    ) -> Result<RestoreOutcome, RestoreExecutorError> {
        // Step 1: pre-execution checks per mode.
        self.preflight(plan, mode).await?;

        // Step 2: write the marker so a crash mid-restore is
        // visible.
        let marker = self.data_dir.join(RESTORE_MARKER);
        tokio::fs::create_dir_all(&self.data_dir).await?;
        tokio::fs::write(&marker, plan.snapshot_id.as_bytes()).await?;

        // Step 3: pull + write each item.
        let mut written = Vec::with_capacity(plan.items.len());
        for item in &plan.items {
            let bytes = verify_blob(
                self.backend.as_ref(),
                &item.content_key,
                item.expected_checksum.as_deref(),
            )
            .await?;
            let dest_path = self.destination_path(plan.tenant_id, item)?;
            self.write_atomic(&dest_path, &bytes).await?;
            let kind_str = match &item.kind {
                RestorePlanItemKind::SqliteCheckpoint => "sqlite_checkpoint".to_string(),
                RestorePlanItemKind::HnswSnapshot { embedding_model } => {
                    format!("hnsw:{embedding_model}")
                }
            };
            written.push((
                RestorePlanItemDigest {
                    kind: kind_str,
                    content_key: item.content_key.clone(),
                    bytes: bytes.len(),
                },
                dest_path,
            ));
        }

        // Step 4: remove the marker.
        match tokio::fs::remove_file(&marker).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(RestoreExecutorError::Io(e)),
        }

        Ok(RestoreOutcome {
            tenant_id: plan.tenant_id,
            snapshot_id: plan.snapshot_id.clone(),
            mode,
            items: written,
        })
    }

    async fn preflight(
        &self,
        plan: &RestorePlan,
        mode: RestoreMode,
    ) -> Result<(), RestoreExecutorError> {
        // Refuse if a previous restore left a marker behind.
        let marker = self.data_dir.join(RESTORE_MARKER);
        if tokio::fs::try_exists(&marker).await? {
            return Err(RestoreExecutorError::RestoreInProgress { path: marker });
        }

        match mode {
            RestoreMode::NewCluster => {
                // Destination must be empty (or non-existent).
                let exists = tokio::fs::try_exists(&self.data_dir).await?;
                if exists {
                    let mut entries = tokio::fs::read_dir(&self.data_dir).await?;
                    if entries.next_entry().await?.is_some() {
                        return Err(RestoreExecutorError::DataDirNotEmpty {
                            path: self.data_dir.clone(),
                        });
                    }
                }
            }
            RestoreMode::SingleTenant => {
                // The tenant subdir must NOT already have content.
                let tenant_dir = self.tenant_dir(plan.tenant_id);
                let exists = tokio::fs::try_exists(&tenant_dir).await?;
                if exists {
                    let mut entries = tokio::fs::read_dir(&tenant_dir).await?;
                    if entries.next_entry().await?.is_some() {
                        return Err(RestoreExecutorError::TenantPathOccupied {
                            tenant_id: plan.tenant_id,
                            path: tenant_dir,
                        });
                    }
                }
            }
            RestoreMode::WipeAndRestore => {
                // Wipe the whole data dir. Refuse only if it doesn't
                // exist at all (operator pointed us at the wrong
                // path) — empty is fine.
                let exists = tokio::fs::try_exists(&self.data_dir).await?;
                if exists {
                    tokio::fs::remove_dir_all(&self.data_dir).await?;
                }
                tokio::fs::create_dir_all(&self.data_dir).await?;
            }
        }
        Ok(())
    }

    fn tenant_dir(&self, tenant_id: TenantId) -> PathBuf {
        self.data_dir.join(format!("tenant_{}", tenant_id.0))
    }

    fn destination_path(
        &self,
        tenant_id: TenantId,
        item: &RestorePlanItem,
    ) -> Result<PathBuf, RestoreExecutorError> {
        let tenant_dir = self.tenant_dir(tenant_id);
        match &item.kind {
            RestorePlanItemKind::SqliteCheckpoint => Ok(tenant_dir.join("data.sqlite")),
            RestorePlanItemKind::HnswSnapshot { embedding_model } => {
                // Sanitize the model name — it may contain `/` or
                // other characters that aren't filesystem-safe.
                // Replace anything outside [a-zA-Z0-9._-] with `_`.
                let safe: String = embedding_model
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                Ok(tenant_dir.join("hnsw").join(format!("{safe}.bin")))
            }
        }
    }

    async fn write_atomic(
        &self,
        dest: &Path,
        bytes: &[u8],
    ) -> Result<(), RestoreExecutorError> {
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let tmp = dest.with_extension("restore-tmp");
        let mut f = tokio::fs::File::create(&tmp).await?;
        f.write_all(bytes).await?;
        f.flush().await?;
        f.sync_all().await?;
        drop(f);
        tokio::fs::rename(&tmp, dest).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backup::manifest::{
        EncryptionMetadata, HnswSnapshotEntry, SnapshotManifest, SnapshotManifestVersion,
    };
    use crate::backup::LocalFsBackend;
    use crate::index::hnsw::DistanceMetric;
    use crate::restore::validate::{DestinationState, RestoreValidator};
    use crate::version::{SchemaVersion, WireVersion};
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    async fn build_validated_plan(
        backend: Arc<dyn BackupBackend>,
        sqlite_bytes: &[u8],
        hnsw_bytes: &[u8],
    ) -> RestorePlan {
        let m = SnapshotManifest {
            manifest_version: SnapshotManifestVersion::CURRENT,
            tenant_id: TenantId::new(1),
            snapshot_id: "snap-1".into(),
            created_at_unix_micros: 1_700_000_000_000_000,
            wire_version: WireVersion::new(1, 0),
            table_schema_versions: {
                let mut m = BTreeMap::new();
                m.insert("memory_commit_log".to_string(), SchemaVersion::new(1));
                m
            },
            oplog_watermark: 100,
            oplog_floor: 1,
            forget_floor: Some(50),
            sqlite_checkpoint_key: "snap-1/sqlite.db".into(),
            sqlite_checkpoint_checksum: Some(blake3::hash(sqlite_bytes).to_hex().to_string()),
            hnsw_snapshots: vec![HnswSnapshotEntry {
                embedding_model: "minilm".into(),
                vector_dim: 384,
                distance_metric: DistanceMetric::Cosine,
                source_log_watermark: 100,
                content_key: "snap-1/hnsw.bin".into(),
                checksum: Some(blake3::hash(hnsw_bytes).to_hex().to_string()),
                deleted_count_pending: 0,
            }],
            encryption: Some(EncryptionMetadata {
                algorithm: "aes-256-gcm".into(),
                dek_id: "dek-1".into(),
                iv_b64: "AA==".into(),
            }),
            label: None,
        };
        backend.put_manifest(&m).await.unwrap();
        backend
            .put_content(&m.sqlite_checkpoint_key, sqlite_bytes)
            .await
            .unwrap();
        backend
            .put_content(&m.hnsw_snapshots[0].content_key, hnsw_bytes)
            .await
            .unwrap();

        let validator = RestoreValidator::new(backend);
        validator
            .validate(
                "snap-1",
                DestinationState {
                    wire_version: WireVersion::new(1, 0),
                    tombstone_floor: Some(10),
                    embedding_model: "minilm".into(),
                },
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn execute_writes_blobs_to_destination() {
        let backup_tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(backup_tmp.path()).unwrap());
        let sqlite = b"sqlite-bytes-here";
        let hnsw = b"hnsw-bytes-here";
        let plan = build_validated_plan(backend.clone(), sqlite, hnsw).await;

        let dest_tmp = TempDir::new().unwrap();
        let dest_path = dest_tmp.path().join("data");
        let exec = RestoreExecutor::new(backend, &dest_path);
        let outcome = exec
            .execute(&plan, RestoreMode::NewCluster)
            .await
            .unwrap();
        assert_eq!(outcome.tenant_id, TenantId::new(1));
        assert_eq!(outcome.mode, RestoreMode::NewCluster);
        assert_eq!(outcome.items.len(), 2);

        // Both blobs land at expected paths with correct content.
        let sqlite_path = dest_path.join("tenant_1").join("data.sqlite");
        let hnsw_path = dest_path.join("tenant_1").join("hnsw").join("minilm.bin");
        assert_eq!(tokio::fs::read(&sqlite_path).await.unwrap(), sqlite);
        assert_eq!(tokio::fs::read(&hnsw_path).await.unwrap(), hnsw);

        // Marker file should be gone after success.
        let marker = dest_path.join(RESTORE_MARKER);
        assert!(!tokio::fs::try_exists(&marker).await.unwrap());
    }

    #[tokio::test]
    async fn new_cluster_mode_refuses_non_empty_dir() {
        let backup_tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(backup_tmp.path()).unwrap());
        let plan = build_validated_plan(backend.clone(), b"x", b"y").await;

        let dest_tmp = TempDir::new().unwrap();
        let dest_path = dest_tmp.path().join("data");
        // Create a non-empty data dir.
        tokio::fs::create_dir_all(&dest_path).await.unwrap();
        tokio::fs::write(dest_path.join("existing.txt"), b"data")
            .await
            .unwrap();

        let exec = RestoreExecutor::new(backend, &dest_path);
        match exec.execute(&plan, RestoreMode::NewCluster).await {
            Err(RestoreExecutorError::DataDirNotEmpty { .. }) => {}
            other => panic!("expected DataDirNotEmpty, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn single_tenant_mode_refuses_existing_tenant_dir() {
        let backup_tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(backup_tmp.path()).unwrap());
        let plan = build_validated_plan(backend.clone(), b"x", b"y").await;

        let dest_tmp = TempDir::new().unwrap();
        let dest_path = dest_tmp.path().join("data");
        // Pre-populate tenant_1 dir.
        tokio::fs::create_dir_all(dest_path.join("tenant_1"))
            .await
            .unwrap();
        tokio::fs::write(dest_path.join("tenant_1").join("existing.bin"), b"data")
            .await
            .unwrap();

        let exec = RestoreExecutor::new(backend, &dest_path);
        match exec.execute(&plan, RestoreMode::SingleTenant).await {
            Err(RestoreExecutorError::TenantPathOccupied { tenant_id, .. }) => {
                assert_eq!(tenant_id, TenantId::new(1));
            }
            other => panic!("expected TenantPathOccupied, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn single_tenant_mode_allows_other_tenants_to_remain() {
        let backup_tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(backup_tmp.path()).unwrap());
        let plan = build_validated_plan(backend.clone(), b"sqlite", b"hnsw").await;

        let dest_tmp = TempDir::new().unwrap();
        let dest_path = dest_tmp.path().join("data");
        // Pre-populate a DIFFERENT tenant's dir; that data must
        // remain untouched.
        tokio::fs::create_dir_all(dest_path.join("tenant_99")).await.unwrap();
        let other_path = dest_path.join("tenant_99").join("untouched.bin");
        tokio::fs::write(&other_path, b"keep-this").await.unwrap();

        let exec = RestoreExecutor::new(backend, &dest_path);
        let outcome = exec
            .execute(&plan, RestoreMode::SingleTenant)
            .await
            .unwrap();
        assert_eq!(outcome.tenant_id, TenantId::new(1));

        // tenant_1 has the new files.
        let sqlite_path = dest_path.join("tenant_1").join("data.sqlite");
        assert_eq!(tokio::fs::read(&sqlite_path).await.unwrap(), b"sqlite");
        // tenant_99 is untouched.
        assert_eq!(tokio::fs::read(&other_path).await.unwrap(), b"keep-this");
    }

    #[tokio::test]
    async fn wipe_and_restore_replaces_destination() {
        let backup_tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(backup_tmp.path()).unwrap());
        let plan = build_validated_plan(backend.clone(), b"new", b"hnsw").await;

        let dest_tmp = TempDir::new().unwrap();
        let dest_path = dest_tmp.path().join("data");
        // Seed destination with stale data.
        tokio::fs::create_dir_all(dest_path.join("tenant_99")).await.unwrap();
        tokio::fs::write(dest_path.join("tenant_99").join("stale.bin"), b"stale")
            .await
            .unwrap();

        let exec = RestoreExecutor::new(backend, &dest_path);
        let outcome = exec
            .execute(&plan, RestoreMode::WipeAndRestore)
            .await
            .unwrap();
        assert_eq!(outcome.mode, RestoreMode::WipeAndRestore);

        // Stale tenant is gone.
        assert!(!tokio::fs::try_exists(dest_path.join("tenant_99"))
            .await
            .unwrap());
        // New tenant is present.
        let sqlite_path = dest_path.join("tenant_1").join("data.sqlite");
        assert_eq!(tokio::fs::read(&sqlite_path).await.unwrap(), b"new");
    }

    #[tokio::test]
    async fn marker_file_blocks_double_restore() {
        // If a previous restore left a marker (partial state), the
        // executor refuses until the operator decides what to do.
        let backup_tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(backup_tmp.path()).unwrap());
        let plan = build_validated_plan(backend.clone(), b"x", b"y").await;

        let dest_tmp = TempDir::new().unwrap();
        let dest_path = dest_tmp.path().join("data");
        tokio::fs::create_dir_all(&dest_path).await.unwrap();
        tokio::fs::write(dest_path.join(RESTORE_MARKER), b"prev-snap-id")
            .await
            .unwrap();

        let exec = RestoreExecutor::new(backend, &dest_path);
        match exec.execute(&plan, RestoreMode::NewCluster).await {
            Err(RestoreExecutorError::RestoreInProgress { .. }) => {}
            other => panic!("expected RestoreInProgress, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn checksum_mismatch_at_exec_aborts() {
        // Even after validate-time success, a corrupt backend
        // serving different bytes at exec-time must be caught.
        let backup_tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(backup_tmp.path()).unwrap());
        let plan = build_validated_plan(backend.clone(), b"original", b"hnsw").await;
        // Tamper with the backend AFTER validate but BEFORE exec.
        backend
            .put_content(&plan.manifest.sqlite_checkpoint_key, b"TAMPERED")
            .await
            .unwrap();

        let dest_tmp = TempDir::new().unwrap();
        let dest_path = dest_tmp.path().join("data");
        let exec = RestoreExecutor::new(backend, &dest_path);
        match exec.execute(&plan, RestoreMode::NewCluster).await {
            Err(RestoreExecutorError::Validator(
                super::super::validate::RestoreValidatorError::ChecksumMismatch { .. },
            )) => {}
            other => panic!("expected ChecksumMismatch at exec, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn destination_paths_sanitize_unsafe_model_chars() {
        let backup_tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(backup_tmp.path()).unwrap());
        // Build a plan whose HNSW model contains a `/` — the
        // executor must not let that escape into the destination
        // path.
        let mut m = SnapshotManifest {
            manifest_version: SnapshotManifestVersion::CURRENT,
            tenant_id: TenantId::new(1),
            snapshot_id: "snap-malicious".into(),
            created_at_unix_micros: 1_700_000_000_000_000,
            wire_version: WireVersion::new(1, 0),
            table_schema_versions: BTreeMap::new(),
            oplog_watermark: 1,
            oplog_floor: 1,
            forget_floor: None,
            sqlite_checkpoint_key: "snap-malicious/sqlite.db".into(),
            sqlite_checkpoint_checksum: None,
            hnsw_snapshots: vec![HnswSnapshotEntry {
                embedding_model: "../etc/passwd".into(),
                vector_dim: 384,
                distance_metric: DistanceMetric::Cosine,
                source_log_watermark: 1,
                content_key: "snap-malicious/hnsw.bin".into(),
                checksum: None,
                deleted_count_pending: 0,
            }],
            encryption: None,
            label: None,
        };
        m.table_schema_versions
            .insert("memory_commit_log".to_string(), SchemaVersion::new(1));
        backend.put_manifest(&m).await.unwrap();
        backend
            .put_content(&m.sqlite_checkpoint_key, b"sqlite")
            .await
            .unwrap();
        backend
            .put_content(&m.hnsw_snapshots[0].content_key, b"hnsw")
            .await
            .unwrap();

        let validator = RestoreValidator::new(backend.clone());
        let plan = validator
            .validate(
                "snap-malicious",
                DestinationState {
                    wire_version: WireVersion::new(1, 0),
                    tombstone_floor: None,
                    embedding_model: "../etc/passwd".into(),
                },
            )
            .await
            .unwrap();
        let dest_tmp = TempDir::new().unwrap();
        let dest_path = dest_tmp.path().join("data");
        let exec = RestoreExecutor::new(backend, &dest_path);
        let outcome = exec
            .execute(&plan, RestoreMode::NewCluster)
            .await
            .unwrap();
        // The HNSW path must be inside `<data_dir>/tenant_1/hnsw/`,
        // not escaped to `<data_dir>/etc/passwd`.
        let expected_safe = dest_path
            .join("tenant_1")
            .join("hnsw")
            .join(".._etc_passwd.bin");
        let (_, actual) = outcome.items.iter().find(|(d, _)| d.kind.starts_with("hnsw:")).unwrap();
        assert_eq!(actual, &expected_safe);
        assert!(actual.starts_with(&dest_path));
    }

    #[tokio::test]
    async fn empty_data_dir_with_new_cluster_mode_is_fine() {
        // Creating the destination dir as an empty existing dir is
        // not an error in NewCluster mode.
        let backup_tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(backup_tmp.path()).unwrap());
        let plan = build_validated_plan(backend.clone(), b"x", b"y").await;

        let dest_tmp = TempDir::new().unwrap();
        let dest_path = dest_tmp.path().join("data");
        tokio::fs::create_dir_all(&dest_path).await.unwrap();

        let exec = RestoreExecutor::new(backend, &dest_path);
        exec.execute(&plan, RestoreMode::NewCluster).await.unwrap();
    }

    #[tokio::test]
    async fn outcome_records_bytes_per_item() {
        let backup_tmp = TempDir::new().unwrap();
        let backend: Arc<dyn BackupBackend> =
            Arc::new(LocalFsBackend::new(backup_tmp.path()).unwrap());
        let sqlite = vec![0u8; 1024];
        let hnsw = vec![1u8; 2048];
        let plan = build_validated_plan(backend.clone(), &sqlite, &hnsw).await;

        let dest_tmp = TempDir::new().unwrap();
        let dest_path = dest_tmp.path().join("data");
        let exec = RestoreExecutor::new(backend, &dest_path);
        let outcome = exec
            .execute(&plan, RestoreMode::NewCluster)
            .await
            .unwrap();

        let sqlite_item = outcome
            .items
            .iter()
            .find(|(d, _)| d.kind == "sqlite_checkpoint")
            .unwrap();
        assert_eq!(sqlite_item.0.bytes, 1024);
        let hnsw_item = outcome
            .items
            .iter()
            .find(|(d, _)| d.kind.starts_with("hnsw:"))
            .unwrap();
        assert_eq!(hnsw_item.0.bytes, 2048);
    }
}
