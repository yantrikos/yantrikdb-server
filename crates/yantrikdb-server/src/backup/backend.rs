//! Backup storage backend — abstract interface + LocalFs impl.
//!
//! ## What ships in PR-1
//!
//! - [`BackupBackend`] trait. Reads + writes manifests; reads +
//!   writes content blobs.
//! - [`LocalFsBackend`] — directory-rooted backend on the local
//!   filesystem. Operators self-hosting on a single node use this;
//!   so do all the unit + integration tests.
//!
//! ## What's deferred to later PRs
//!
//! - Object-store backends (S3, GCS, Azure) via the `object_store`
//!   crate. The trait shape here is intentionally what
//!   `object_store::ObjectStore` exposes (put/get/list/delete) so
//!   that adapter is a thin shim once the dep is added.
//! - Streaming reads/writes for content blobs. The trait shape
//!   currently takes/returns `Vec<u8>`; for tenant snapshots that
//!   could exceed RAM, we'll add a streaming variant in a follow-up.
//!   For RFC 012 PR-1 / PR-2 the buffered API is sufficient because
//!   tenants in the test scope are well under a GB.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use thiserror::Error;
use tokio::io::AsyncWriteExt;

use super::manifest::{ManifestValidationError, SnapshotManifest};

#[derive(Debug, Error)]
pub enum BackupBackendError {
    #[error("backend IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("backend serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("manifest validation error: {0}")]
    ManifestValidation(#[from] ManifestValidationError),

    #[error("manifest `{key}` not found")]
    ManifestNotFound { key: String },

    #[error("content `{key}` not found")]
    ContentNotFound { key: String },

    #[error("backend rejected key `{key}`: {reason}")]
    InvalidKey { key: String, reason: &'static str },

    #[error("backend error: {0}")]
    Other(String),
}

/// Abstract backup-storage backend. Mirrors the surface area of
/// `object_store::ObjectStore` so the eventual S3/GCS/Azure shim is
/// straightforward.
///
/// Manifests and content live in two logical buckets:
/// - **Manifests**: one JSON file per snapshot, named
///   `manifests/<snapshot_id>.json`.
/// - **Content**: opaque blobs at caller-supplied keys. The
///   manifest references its content blobs by key
///   (`sqlite_checkpoint_key`, `hnsw_snapshots[i].content_key`).
#[async_trait]
pub trait BackupBackend: Send + Sync {
    /// Stable name for metrics + audit logs.
    fn name(&self) -> &'static str;

    /// Persist a manifest. Caller is expected to call
    /// `manifest.validate_internal()` before invoking.
    async fn put_manifest(
        &self,
        manifest: &SnapshotManifest,
    ) -> Result<(), BackupBackendError>;

    /// Read a manifest by snapshot_id. Returns
    /// `ManifestNotFound` if the manifest doesn't exist.
    async fn get_manifest(&self, snapshot_id: &str)
        -> Result<SnapshotManifest, BackupBackendError>;

    /// List all manifests (snapshot_ids). Sorted ascending so
    /// chronological iteration is the default.
    async fn list_manifests(&self) -> Result<Vec<String>, BackupBackendError>;

    /// Delete a manifest. Idempotent — deleting a missing manifest
    /// is not an error.
    async fn delete_manifest(&self, snapshot_id: &str) -> Result<(), BackupBackendError>;

    /// Write a content blob at `key`. Overwrites if present.
    async fn put_content(&self, key: &str, bytes: &[u8])
        -> Result<(), BackupBackendError>;

    /// Read a content blob.
    async fn get_content(&self, key: &str) -> Result<Vec<u8>, BackupBackendError>;

    /// Idempotent delete.
    async fn delete_content(&self, key: &str) -> Result<(), BackupBackendError>;
}

/// Local-filesystem backend. The `root` is a directory the backend
/// owns; manifests land at `<root>/manifests/<snapshot_id>.json`,
/// content lands at `<root>/<key>` (caller controls the relative
/// key shape).
///
/// Cheap to clone; just an Arc'd PathBuf.
#[derive(Clone)]
pub struct LocalFsBackend {
    root: std::sync::Arc<PathBuf>,
}

impl LocalFsBackend {
    /// Construct rooted at `root`. The directory is created if it
    /// doesn't exist; an error here surfaces as `Io`.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, BackupBackendError> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        std::fs::create_dir_all(root.join("manifests"))?;
        Ok(Self {
            root: std::sync::Arc::new(root),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn manifest_path(&self, snapshot_id: &str) -> Result<PathBuf, BackupBackendError> {
        validate_id(snapshot_id, "snapshot_id")?;
        Ok(self.root.join("manifests").join(format!("{snapshot_id}.json")))
    }

    fn content_path(&self, key: &str) -> Result<PathBuf, BackupBackendError> {
        validate_content_key(key)?;
        Ok(self.root.join(key))
    }
}

/// Validate a snapshot_id or other simple identifier — must be
/// non-empty and not contain path traversal.
fn validate_id(id: &str, what: &'static str) -> Result<(), BackupBackendError> {
    if id.is_empty() {
        return Err(BackupBackendError::InvalidKey {
            key: id.to_string(),
            reason: "empty id",
        });
    }
    let _ = what;
    if id.contains("..") || id.contains('/') || id.contains('\\') {
        return Err(BackupBackendError::InvalidKey {
            key: id.to_string(),
            reason: "id must not contain path separators or `..`",
        });
    }
    Ok(())
}

/// Validate a content key — allows nested paths but disallows
/// traversal.
fn validate_content_key(key: &str) -> Result<(), BackupBackendError> {
    if key.is_empty() {
        return Err(BackupBackendError::InvalidKey {
            key: key.to_string(),
            reason: "empty key",
        });
    }
    // Disallow upward traversal. Forward slashes are fine —
    // operators key by `tenants/<id>/...` etc. Backslashes are
    // disallowed because they're path separators on Windows.
    for component in key.split('/') {
        if component == ".." || component == "." {
            return Err(BackupBackendError::InvalidKey {
                key: key.to_string(),
                reason: "content key must not contain `..` or `.` segments",
            });
        }
    }
    if key.contains('\\') {
        return Err(BackupBackendError::InvalidKey {
            key: key.to_string(),
            reason: "content key must use forward slashes only",
        });
    }
    if key.starts_with('/') {
        return Err(BackupBackendError::InvalidKey {
            key: key.to_string(),
            reason: "content key must be relative (no leading slash)",
        });
    }
    Ok(())
}

#[async_trait]
impl BackupBackend for LocalFsBackend {
    fn name(&self) -> &'static str {
        "local_fs"
    }

    async fn put_manifest(
        &self,
        manifest: &SnapshotManifest,
    ) -> Result<(), BackupBackendError> {
        manifest.validate_internal()?;
        let path = self.manifest_path(&manifest.snapshot_id)?;
        let json = manifest.to_json()?;
        // Write to a tmp file and rename for atomicity.
        let tmp = path.with_extension("json.tmp");
        let mut f = tokio::fs::File::create(&tmp).await?;
        f.write_all(json.as_bytes()).await?;
        f.flush().await?;
        f.sync_all().await?;
        drop(f);
        tokio::fs::rename(&tmp, &path).await?;
        Ok(())
    }

    async fn get_manifest(
        &self,
        snapshot_id: &str,
    ) -> Result<SnapshotManifest, BackupBackendError> {
        let path = self.manifest_path(snapshot_id)?;
        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(BackupBackendError::ManifestNotFound {
                    key: snapshot_id.to_string(),
                });
            }
            Err(e) => return Err(BackupBackendError::Io(e)),
        };
        let s = std::str::from_utf8(&bytes).map_err(|e| {
            BackupBackendError::Other(format!("manifest UTF-8: {e}"))
        })?;
        let m = SnapshotManifest::from_json(s)?;
        Ok(m)
    }

    async fn list_manifests(&self) -> Result<Vec<String>, BackupBackendError> {
        let dir = self.root.join("manifests");
        let mut entries = match tokio::fs::read_dir(&dir).await {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(BackupBackendError::Io(e)),
        };
        let mut out = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(stem) = name.strip_suffix(".json") {
                out.push(stem.to_string());
            }
        }
        out.sort();
        Ok(out)
    }

    async fn delete_manifest(&self, snapshot_id: &str) -> Result<(), BackupBackendError> {
        let path = self.manifest_path(snapshot_id)?;
        match tokio::fs::remove_file(&path).await {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(BackupBackendError::Io(e)),
        }
    }

    async fn put_content(
        &self,
        key: &str,
        bytes: &[u8],
    ) -> Result<(), BackupBackendError> {
        let path = self.content_path(key)?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let tmp = path.with_extension("tmp");
        let mut f = tokio::fs::File::create(&tmp).await?;
        f.write_all(bytes).await?;
        f.flush().await?;
        f.sync_all().await?;
        drop(f);
        tokio::fs::rename(&tmp, &path).await?;
        Ok(())
    }

    async fn get_content(&self, key: &str) -> Result<Vec<u8>, BackupBackendError> {
        let path = self.content_path(key)?;
        match tokio::fs::read(&path).await {
            Ok(b) => Ok(b),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(BackupBackendError::ContentNotFound {
                    key: key.to_string(),
                })
            }
            Err(e) => Err(BackupBackendError::Io(e)),
        }
    }

    async fn delete_content(&self, key: &str) -> Result<(), BackupBackendError> {
        let path = self.content_path(key)?;
        match tokio::fs::remove_file(&path).await {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(BackupBackendError::Io(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backup::manifest::{
        EncryptionMetadata, HnswSnapshotEntry, SnapshotManifestVersion,
    };
    use crate::commit::TenantId;
    use crate::index::hnsw::DistanceMetric;
    use crate::version::{SchemaVersion, WireVersion};
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    fn sample_manifest(snapshot_id: &str) -> SnapshotManifest {
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
            oplog_watermark: 100,
            oplog_floor: 1,
            forget_floor: Some(50),
            sqlite_checkpoint_key: "tenants/1/sqlite.db".into(),
            sqlite_checkpoint_checksum: Some("aaa".into()),
            hnsw_snapshots: vec![HnswSnapshotEntry {
                embedding_model: "minilm".into(),
                vector_dim: 384,
                distance_metric: DistanceMetric::Cosine,
                source_log_watermark: 100,
                content_key: "tenants/1/hnsw.bin".into(),
                checksum: None,
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

    #[tokio::test]
    async fn put_then_get_manifest_round_trips() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        let m = sample_manifest("snap-1");
        backend.put_manifest(&m).await.unwrap();
        let back = backend.get_manifest("snap-1").await.unwrap();
        assert_eq!(m, back);
    }

    #[tokio::test]
    async fn get_missing_manifest_returns_not_found() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        match backend.get_manifest("does-not-exist").await {
            Err(BackupBackendError::ManifestNotFound { key }) => {
                assert_eq!(key, "does-not-exist");
            }
            other => panic!("expected ManifestNotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn list_manifests_returns_sorted_snapshot_ids() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        for id in ["snap-c", "snap-a", "snap-b"] {
            backend.put_manifest(&sample_manifest(id)).await.unwrap();
        }
        let listed = backend.list_manifests().await.unwrap();
        assert_eq!(listed, vec!["snap-a", "snap-b", "snap-c"]);
    }

    #[tokio::test]
    async fn list_on_empty_backend_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        let listed = backend.list_manifests().await.unwrap();
        assert!(listed.is_empty());
    }

    #[tokio::test]
    async fn delete_manifest_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        backend.put_manifest(&sample_manifest("snap-1")).await.unwrap();
        backend.delete_manifest("snap-1").await.unwrap();
        // Second delete is also OK.
        backend.delete_manifest("snap-1").await.unwrap();
        // Manifest is genuinely gone.
        assert!(backend.list_manifests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn put_invalid_manifest_is_rejected() {
        // Internal validation runs at put_manifest time. An
        // inverted oplog range fails before any disk write.
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        let mut m = sample_manifest("snap-1");
        m.oplog_floor = 999;
        m.oplog_watermark = 1;
        match backend.put_manifest(&m).await {
            Err(BackupBackendError::ManifestValidation(_)) => {}
            other => panic!("expected ManifestValidation error, got {other:?}"),
        }
        // Disk shows nothing.
        assert!(backend.list_manifests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn put_then_get_content_round_trips() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        let payload = vec![1, 2, 3, 4, 5];
        backend
            .put_content("tenants/1/sqlite.db", &payload)
            .await
            .unwrap();
        let back = backend.get_content("tenants/1/sqlite.db").await.unwrap();
        assert_eq!(payload, back);
    }

    #[tokio::test]
    async fn get_missing_content_returns_not_found() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        match backend.get_content("nope").await {
            Err(BackupBackendError::ContentNotFound { .. }) => {}
            other => panic!("expected ContentNotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn delete_content_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        backend.put_content("tenants/1/x", &[1]).await.unwrap();
        backend.delete_content("tenants/1/x").await.unwrap();
        backend.delete_content("tenants/1/x").await.unwrap();
        assert!(matches!(
            backend.get_content("tenants/1/x").await,
            Err(BackupBackendError::ContentNotFound { .. })
        ));
    }

    #[tokio::test]
    async fn snapshot_id_with_path_traversal_rejected() {
        // Critical: backups MUST NOT allow `../` injection in
        // snapshot_id, else a malicious caller could overwrite
        // arbitrary files outside the backend root.
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        let mut m = sample_manifest("../etc/passwd");
        m.snapshot_id = "../etc/passwd".into();
        let result = backend.put_manifest(&m).await;
        match result {
            Err(BackupBackendError::InvalidKey { .. }) => {}
            other => panic!("expected InvalidKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn content_key_with_path_traversal_rejected() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        let result = backend.put_content("tenants/1/../../etc/passwd", &[0]).await;
        match result {
            Err(BackupBackendError::InvalidKey { .. }) => {}
            other => panic!("expected InvalidKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn content_key_with_backslash_rejected() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        let result = backend.put_content("tenants\\1\\file", &[0]).await;
        match result {
            Err(BackupBackendError::InvalidKey { .. }) => {}
            other => panic!("expected InvalidKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn content_key_with_leading_slash_rejected() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        let result = backend.put_content("/etc/passwd", &[0]).await;
        match result {
            Err(BackupBackendError::InvalidKey { .. }) => {}
            other => panic!("expected InvalidKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn nested_content_key_creates_intermediate_dirs() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        backend
            .put_content("tenants/1/snap-abc/hnsw.bin", &[0xCA, 0xFE])
            .await
            .unwrap();
        let back = backend
            .get_content("tenants/1/snap-abc/hnsw.bin")
            .await
            .unwrap();
        assert_eq!(back, vec![0xCA, 0xFE]);
    }

    #[tokio::test]
    async fn empty_content_key_rejected() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        let result = backend.put_content("", &[0]).await;
        assert!(matches!(result, Err(BackupBackendError::InvalidKey { .. })));
    }

    #[tokio::test]
    async fn dyn_dispatch_works() {
        let tmp = TempDir::new().unwrap();
        let backend: std::sync::Arc<dyn BackupBackend> =
            std::sync::Arc::new(LocalFsBackend::new(tmp.path()).unwrap());
        backend.put_manifest(&sample_manifest("dyn-1")).await.unwrap();
        let m = backend.get_manifest("dyn-1").await.unwrap();
        assert_eq!(m.snapshot_id, "dyn-1");
        assert_eq!(backend.name(), "local_fs");
    }

    #[tokio::test]
    async fn put_manifest_overwrites_existing() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(tmp.path()).unwrap();
        let mut m = sample_manifest("snap-1");
        backend.put_manifest(&m).await.unwrap();
        m.label = Some("updated".into());
        backend.put_manifest(&m).await.unwrap();
        let back = backend.get_manifest("snap-1").await.unwrap();
        assert_eq!(back.label.as_deref(), Some("updated"));
    }
}
