//! Snapshot manifest — the per-tenant descriptor pinning everything
//! a restore operation needs to come up consistent.
//!
//! ## What's in a manifest
//!
//! - **Identity**: `tenant_id`, `snapshot_id` (UUIDv7),
//!   `created_at_unix_micros`, `manifest_version` (the on-wire
//!   schema version of the manifest itself).
//! - **Source pointers**: `oplog_watermark` (max log_index in the
//!   commit log at snapshot time), `schema_version` (the cluster's
//!   wire/schema version pin).
//! - **Content references**: `sqlite_checkpoint` (path/key for the
//!   tenant's SQLite WAL checkpoint blob), `hnsw_snapshots` (per-
//!   model entry list with watermark + checksum + content key).
//! - **Erasure invariants**: `forget_floor` — the lowest log_index
//!   the snapshot covers that contains any tombstone we MUST
//!   capture. Restoring into a cluster whose applied-tombstone
//!   floor is higher than this would resurrect deleted memories.
//! - **Encryption metadata**: algorithm + key_id + IV. RFC 014-C
//!   key provider implementation lands separately; this PR ships
//!   the carrier struct.
//! - **Integrity**: `checksum` over the canonical serialization
//!   minus the checksum field itself.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::commit::TenantId;
use crate::index::hnsw::DistanceMetric;
use crate::version::{SchemaVersion, WireVersion};

/// On-wire version of the manifest format itself. Bump on
/// breaking schema changes; restore compatibility checks use this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SnapshotManifestVersion(pub u32);

impl SnapshotManifestVersion {
    pub const CURRENT: SnapshotManifestVersion = SnapshotManifestVersion(1);
}

/// One HNSW index covered by the snapshot. A tenant may have
/// multiple (primary + shadow during model migration per RFC
/// 013-B) — captured in a Vec on the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswSnapshotEntry {
    /// Embedding model id (e.g. `"all-MiniLM-L6-v2"`). Matches the
    /// HnswManifest's `embedding_model` field — restore refuses if
    /// the destination's configured model differs.
    pub embedding_model: String,
    pub vector_dim: u32,
    pub distance_metric: DistanceMetric,
    pub source_log_watermark: u64,
    /// Storage key (relative to the backend's root) for the HNSW
    /// content blob.
    pub content_key: String,
    /// Optional content checksum. blake3 hex by convention. None
    /// when the backend doesn't surface a checksum.
    pub checksum: Option<String>,
    pub deleted_count_pending: u64,
}

/// Encryption metadata. Carries the info a restore needs to
/// decrypt the content blob using the tenant's DEK (data
/// encryption key) — which the restore looks up via the RFC 014-C
/// key provider. None means content is unencrypted (operator opt-in
/// only, useful for development).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncryptionMetadata {
    /// Stable identifier of the algorithm. Operators read this in
    /// audit logs; restore validates it against the supported set.
    /// Examples: `"aes-256-gcm"`, `"chacha20-poly1305"`.
    pub algorithm: String,
    /// Tenant DEK identifier. Resolved at restore time via the
    /// configured key provider. Stored opaque here.
    pub dek_id: String,
    /// Initialization vector / nonce. Base64-encoded.
    pub iv_b64: String,
}

/// The per-tenant snapshot manifest itself. Serialized as JSON
/// alongside the content blobs in whichever backend the operator
/// configured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotManifest {
    /// Manifest format version (not the cluster wire version).
    pub manifest_version: SnapshotManifestVersion,

    /// Snapshot identity.
    pub tenant_id: TenantId,
    /// UUIDv7 string. Time-ordered + globally unique.
    pub snapshot_id: String,
    /// Unix-microseconds when the snapshot was taken.
    pub created_at_unix_micros: i64,

    /// Cluster wire/schema version at snapshot time. Restore
    /// refuses if the destination cluster's wire is incompatible.
    pub wire_version: WireVersion,
    /// Per-table schema versions, indexed by table name. Restore
    /// validates each against the destination's
    /// `crate::version::TABLE_SCHEMA_VERSIONS`.
    pub table_schema_versions: BTreeMap<String, SchemaVersion>,

    /// Max log_index in the per-tenant commit log at snapshot time.
    /// Restore replays log entries up through this index.
    pub oplog_watermark: u64,
    /// Lowest log_index covered by the snapshot. Defines the
    /// snapshot's "starts-at" boundary for restore replay.
    pub oplog_floor: u64,

    /// Restore-no-resurrect invariant guard. Lowest log_index
    /// containing a tombstone the snapshot must capture. If a
    /// destination cluster's already-applied-tombstone floor is
    /// higher than this, restore would resurrect deleted memories
    /// and MUST refuse. See [`SnapshotManifest::validate_for_restore`].
    pub forget_floor: Option<u64>,

    /// Storage key for the tenant's SQLite checkpoint blob. The
    /// blob is the result of running `PRAGMA wal_checkpoint(TRUNCATE)`
    /// on the per-tenant DB, captured in a single .sqlite file.
    pub sqlite_checkpoint_key: String,
    /// blake3 hex checksum of the SQLite checkpoint blob. Verified
    /// at restore time before the file is opened.
    pub sqlite_checkpoint_checksum: Option<String>,

    /// HNSW snapshots — one entry per index (primary + optional
    /// shadow during model migration per RFC 013-B).
    pub hnsw_snapshots: Vec<HnswSnapshotEntry>,

    /// Encryption metadata. `None` means unencrypted (dev opt-in).
    pub encryption: Option<EncryptionMetadata>,

    /// Operator-supplied label, free-form. Useful for restore
    /// targeting ("the manifest tagged 'pre-migration'").
    pub label: Option<String>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ManifestValidationError {
    #[error("oplog_floor ({floor}) must be <= oplog_watermark ({watermark})")]
    OplogRangeInverted { floor: u64, watermark: u64 },

    #[error(
        "forget_floor ({forget_floor}) is above oplog_watermark ({watermark}); \
         tombstone reference is outside the snapshot range"
    )]
    ForgetFloorAboveWatermark { forget_floor: u64, watermark: u64 },

    #[error(
        "manifest_version ({got:?}) is newer than this build's max ({current:?}); \
         upgrade the binary before reading this snapshot"
    )]
    ManifestVersionUnsupported {
        got: SnapshotManifestVersion,
        current: SnapshotManifestVersion,
    },

    #[error(
        "snapshot has no SQLite checkpoint key (empty string); \
         restore would have nothing to apply"
    )]
    EmptySqliteCheckpointKey,

    #[error(
        "RESTORE WOULD RESURRECT DELETED DATA: destination tombstone floor \
         ({dest_tombstone_floor}) is above snapshot forget_floor ({snap_forget_floor}). \
         Memories tombstoned in the destination after the snapshot would reappear. \
         Refusing per RFC 011 restore-no-resurrect invariant."
    )]
    WouldResurrectDeletedData {
        dest_tombstone_floor: u64,
        snap_forget_floor: u64,
    },

    #[error("destination wire version ({dest:?}) is incompatible with snapshot ({snap:?})")]
    WireVersionMismatch {
        dest: WireVersion,
        snap: WireVersion,
    },

    #[error(
        "destination configured embedding model `{dest_model}` does not match \
         snapshot's HNSW model `{snap_model}` (vector_dim or distance_metric \
         would differ)"
    )]
    HnswModelMismatch {
        dest_model: String,
        snap_model: String,
    },
}

impl SnapshotManifest {
    /// Validate the manifest's internal consistency. Cheap; called
    /// by both the snapshot writer (before persisting) and the
    /// restore reader (after loading).
    pub fn validate_internal(&self) -> Result<(), ManifestValidationError> {
        if self.oplog_floor > self.oplog_watermark {
            return Err(ManifestValidationError::OplogRangeInverted {
                floor: self.oplog_floor,
                watermark: self.oplog_watermark,
            });
        }
        if let Some(ff) = self.forget_floor {
            if ff > self.oplog_watermark {
                return Err(ManifestValidationError::ForgetFloorAboveWatermark {
                    forget_floor: ff,
                    watermark: self.oplog_watermark,
                });
            }
        }
        if self.manifest_version.0 > SnapshotManifestVersion::CURRENT.0 {
            return Err(ManifestValidationError::ManifestVersionUnsupported {
                got: self.manifest_version,
                current: SnapshotManifestVersion::CURRENT,
            });
        }
        if self.sqlite_checkpoint_key.is_empty() {
            return Err(ManifestValidationError::EmptySqliteCheckpointKey);
        }
        Ok(())
    }

    /// Validate that this manifest is safe to restore into a
    /// destination cluster. Called by the (PR-2) restore command
    /// before any content is touched.
    ///
    /// Returns Err — surfaces the specific violation — if any
    /// invariant fails. The most load-bearing check is
    /// `WouldResurrectDeletedData`: the manifest's `forget_floor`
    /// must be ≥ the destination's already-applied-tombstone floor,
    /// else memories tombstoned post-snapshot would come back.
    pub fn validate_for_restore(
        &self,
        dest_wire: WireVersion,
        dest_tombstone_floor: u64,
        dest_embedding_model: &str,
    ) -> Result<(), ManifestValidationError> {
        self.validate_internal()?;

        // Wire-version compatibility — defer detailed handshake to
        // crate::version, which already knows compatibility rules.
        // For PR-1 we use a simple equality check; PR-2 will plumb
        // through `VersionGate::validate_compat`.
        if dest_wire.major != self.wire_version.major {
            return Err(ManifestValidationError::WireVersionMismatch {
                dest: dest_wire,
                snap: self.wire_version,
            });
        }

        // Restore-no-resurrect: refuse if destination has applied
        // tombstones beyond what the snapshot captures.
        if let Some(ff) = self.forget_floor {
            if dest_tombstone_floor > ff {
                return Err(ManifestValidationError::WouldResurrectDeletedData {
                    dest_tombstone_floor,
                    snap_forget_floor: ff,
                });
            }
        }

        // HNSW model compatibility: if the destination has any HNSW
        // index for this tenant, the snapshot must include a
        // matching one. We don't enforce that the destination's
        // model match each snapshot entry — that's the integration
        // PR's job — but we do require the destination's primary
        // model show up somewhere in the snapshot, else the
        // post-restore index is a mismatched-dim time bomb.
        let any_match = self
            .hnsw_snapshots
            .iter()
            .any(|s| s.embedding_model == dest_embedding_model);
        if !any_match && !self.hnsw_snapshots.is_empty() {
            return Err(ManifestValidationError::HnswModelMismatch {
                dest_model: dest_embedding_model.to_string(),
                snap_model: self.hnsw_snapshots[0].embedding_model.clone(),
            });
        }

        Ok(())
    }

    /// Stable JSON serialization for backend persistence. Pretty-
    /// printed so operators reading raw manifests can grep for
    /// fields easily.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::hnsw::DistanceMetric;

    fn sample_manifest() -> SnapshotManifest {
        SnapshotManifest {
            manifest_version: SnapshotManifestVersion::CURRENT,
            tenant_id: TenantId::new(1),
            snapshot_id: "01934567-89ab-7cde-8000-000000000001".into(),
            created_at_unix_micros: 1_700_000_000_000_000,
            wire_version: WireVersion::new(1, 0),
            table_schema_versions: {
                let mut m = BTreeMap::new();
                m.insert("memory_commit_log".to_string(), SchemaVersion::new(1));
                m.insert("hnsw_manifests".to_string(), SchemaVersion::new(1));
                m
            },
            oplog_watermark: 5_000,
            oplog_floor: 1,
            forget_floor: Some(2_500),
            sqlite_checkpoint_key: "tenants/1/snap-abc/sqlite.db".into(),
            sqlite_checkpoint_checksum: Some("aabbcc".into()),
            hnsw_snapshots: vec![HnswSnapshotEntry {
                embedding_model: "all-MiniLM-L6-v2".into(),
                vector_dim: 384,
                distance_metric: DistanceMetric::Cosine,
                source_log_watermark: 5_000,
                content_key: "tenants/1/snap-abc/hnsw-MiniLM.bin".into(),
                checksum: Some("ddeeff".into()),
                deleted_count_pending: 12,
            }],
            encryption: Some(EncryptionMetadata {
                algorithm: "aes-256-gcm".into(),
                dek_id: "tenant-1-dek-v3".into(),
                iv_b64: "AAECAwQFBgcICQoLDA0ODw==".into(),
            }),
            label: Some("pre-migration".into()),
        }
    }

    #[test]
    fn current_version_is_one() {
        assert_eq!(SnapshotManifestVersion::CURRENT.0, 1);
    }

    #[test]
    fn json_round_trip_is_lossless() {
        let m = sample_manifest();
        let json = m.to_json().unwrap();
        let back = SnapshotManifest::from_json(&json).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn validate_internal_passes_on_sample() {
        sample_manifest().validate_internal().unwrap();
    }

    #[test]
    fn validate_internal_rejects_inverted_oplog_range() {
        let mut m = sample_manifest();
        m.oplog_floor = 9_000;
        m.oplog_watermark = 1_000;
        match m.validate_internal() {
            Err(ManifestValidationError::OplogRangeInverted { floor, watermark }) => {
                assert_eq!(floor, 9_000);
                assert_eq!(watermark, 1_000);
            }
            other => panic!("expected OplogRangeInverted, got {other:?}"),
        }
    }

    #[test]
    fn validate_internal_rejects_forget_floor_above_watermark() {
        let mut m = sample_manifest();
        m.forget_floor = Some(99_999);
        match m.validate_internal() {
            Err(ManifestValidationError::ForgetFloorAboveWatermark { .. }) => {}
            other => panic!("expected ForgetFloorAboveWatermark, got {other:?}"),
        }
    }

    #[test]
    fn validate_internal_rejects_future_manifest_version() {
        let mut m = sample_manifest();
        m.manifest_version = SnapshotManifestVersion(99);
        assert!(matches!(
            m.validate_internal(),
            Err(ManifestValidationError::ManifestVersionUnsupported { .. })
        ));
    }

    #[test]
    fn validate_internal_rejects_empty_sqlite_key() {
        let mut m = sample_manifest();
        m.sqlite_checkpoint_key.clear();
        assert!(matches!(
            m.validate_internal(),
            Err(ManifestValidationError::EmptySqliteCheckpointKey)
        ));
    }

    #[test]
    fn forget_floor_optional_is_allowed() {
        let mut m = sample_manifest();
        m.forget_floor = None; // tenant with no tombstones
        m.validate_internal().unwrap();
    }

    #[test]
    fn validate_for_restore_passes_on_compatible_destination() {
        let m = sample_manifest();
        // Destination tombstone floor BELOW snapshot's forget_floor —
        // safe.
        m.validate_for_restore(WireVersion::new(1, 0), 100, "all-MiniLM-L6-v2")
            .unwrap();
    }

    #[test]
    fn validate_for_restore_refuses_resurrect_scenario() {
        // The load-bearing check. Snapshot's forget_floor=2500 means
        // the snapshot captured tombstones from log_index 2500
        // onwards. If the destination has APPLIED tombstones up
        // through log_index 5000 (i.e., already deleted memories
        // beyond what the snapshot captured), restoring the snapshot
        // would resurrect those memories.
        let m = sample_manifest();
        let result = m.validate_for_restore(
            WireVersion::new(1, 0),
            5_000, // dest applied tombstones way past snapshot's forget_floor
            "all-MiniLM-L6-v2",
        );
        match result {
            Err(ManifestValidationError::WouldResurrectDeletedData {
                dest_tombstone_floor,
                snap_forget_floor,
            }) => {
                assert_eq!(dest_tombstone_floor, 5_000);
                assert_eq!(snap_forget_floor, 2_500);
            }
            other => panic!("expected WouldResurrectDeletedData, got {other:?}"),
        }
    }

    #[test]
    fn validate_for_restore_no_forget_floor_means_safe() {
        // A tenant with no tombstones at snapshot time has
        // forget_floor=None. That's "no constraint" — restore
        // proceeds regardless of destination's tombstone state.
        let mut m = sample_manifest();
        m.forget_floor = None;
        m.validate_for_restore(WireVersion::new(1, 0), 999_999, "all-MiniLM-L6-v2")
            .unwrap();
    }

    #[test]
    fn validate_for_restore_refuses_wire_major_mismatch() {
        let m = sample_manifest();
        let result = m.validate_for_restore(
            WireVersion::new(2, 0), // major bump
            100,
            "all-MiniLM-L6-v2",
        );
        assert!(matches!(
            result,
            Err(ManifestValidationError::WireVersionMismatch { .. })
        ));
    }

    #[test]
    fn validate_for_restore_refuses_hnsw_model_mismatch() {
        let m = sample_manifest();
        let result = m.validate_for_restore(
            WireVersion::new(1, 0),
            100,
            "bge-base", // destination uses a different model
        );
        assert!(matches!(
            result,
            Err(ManifestValidationError::HnswModelMismatch { .. })
        ));
    }

    #[test]
    fn validate_for_restore_allows_no_hnsw_snapshots() {
        // Edge: a tenant with zero HNSW indexes (no recall use yet)
        // can be restored to any model — the destination will build
        // its own index post-restore.
        let mut m = sample_manifest();
        m.hnsw_snapshots.clear();
        m.validate_for_restore(WireVersion::new(1, 0), 100, "anything")
            .unwrap();
    }

    #[test]
    fn validate_for_restore_matches_among_multiple_hnsw_snapshots() {
        // RFC 013-B model-migration scenario: snapshot has primary +
        // shadow. Destination's model is one of them. Should pass.
        let mut m = sample_manifest();
        m.hnsw_snapshots.push(HnswSnapshotEntry {
            embedding_model: "bge-base".into(),
            vector_dim: 768,
            distance_metric: DistanceMetric::Cosine,
            source_log_watermark: 5_000,
            content_key: "tenants/1/snap-abc/hnsw-bge.bin".into(),
            checksum: None,
            deleted_count_pending: 0,
        });
        m.validate_for_restore(WireVersion::new(1, 0), 100, "bge-base")
            .unwrap();
        m.validate_for_restore(WireVersion::new(1, 0), 100, "all-MiniLM-L6-v2")
            .unwrap();
    }

    #[test]
    fn encryption_optional_means_unencrypted() {
        let mut m = sample_manifest();
        m.encryption = None;
        m.validate_internal().unwrap();
        let json = m.to_json().unwrap();
        let back = SnapshotManifest::from_json(&json).unwrap();
        assert!(back.encryption.is_none());
    }

    #[test]
    fn label_is_optional() {
        let mut m = sample_manifest();
        m.label = None;
        m.validate_internal().unwrap();
    }
}
