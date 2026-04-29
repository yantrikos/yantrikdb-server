//! HNSW manifest — the per-tenant record of where the index thinks it
//! is in the commit-log timeline, plus the index parameters that have
//! to match for recall to be meaningful.
//!
//! ## Required fields & why each one
//!
//! - **`tenant_id`** — manifests are per-tenant. The index does not
//!   span tenants; mismatches must be impossible.
//! - **`embedding_model`** — when 013-B ships shadow-index dual-read,
//!   one tenant can have two manifests (primary + shadow) keyed on
//!   different models. Storing the model in the manifest closes the
//!   "dim mismatch silently re-indexes wrong vectors" footgun.
//! - **`index_generation`** — bumps every full rebuild. Caches and
//!   followers tag results with the generation they were built against
//!   so a stale follower can detect "I was reading from generation 7,
//!   leader is on 8" and refuse rather than serve mismatched results.
//! - **`source_log_start`** + **`source_log_watermark`** — the closed-
//!   open commit-log range `[start, watermark)` the index is consistent
//!   over. Reconciler computes
//!   `gap = commit_log_high_water - watermark` to detect drift.
//! - **`vector_dim`** + **`distance_metric`** — pinned at index build
//!   time. If recall config disagrees with the manifest, refuse.
//! - **`deleted_count_pending`** — count of tombstoned rids not yet
//!   compacted out of the index. Operators use this with the recall
//!   results to predict "stale recall ratio" — when it crosses a
//!   threshold the compaction job (RFC 011 PR-3) is triggered.
//! - **`checksum`** — optional content hash of the index files,
//!   computed at manifest update time. Detects on-disk corruption.
//! - **`created_at` / `updated_at`** — for operator visibility +
//!   metrics.

use std::time::SystemTime;

use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

use crate::commit::TenantId;

/// Distance metric the HNSW index was built with. Must match the recall
/// config or recall refuses (RFC 013-A enforcement).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    /// Cosine similarity. Most common for semantic embeddings.
    Cosine,
    /// L2 (Euclidean) distance.
    L2,
    /// Inner product (dot product). For models that produce
    /// pre-normalized vectors and want to skip cosine's renormalization.
    InnerProduct,
}

impl DistanceMetric {
    /// Stable string for the `hnsw_manifests.distance_metric` column.
    /// Never rename — these are persisted on disk.
    pub fn as_str(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::L2 => "l2",
            DistanceMetric::InnerProduct => "inner_product",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "cosine" => Some(Self::Cosine),
            "l2" => Some(Self::L2),
            "inner_product" => Some(Self::InnerProduct),
            _ => None,
        }
    }
}

/// Full manifest record. One row per `(tenant_id, embedding_model)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswManifest {
    pub tenant_id: TenantId,
    pub embedding_model: String,
    pub index_generation: u64,
    /// Closed-open: index covers commit-log entries
    /// `[source_log_start, source_log_watermark)`. `watermark - start`
    /// is the number of mutations applied into this generation.
    pub source_log_start: u64,
    pub source_log_watermark: u64,
    pub vector_dim: u32,
    pub distance_metric: DistanceMetric,
    pub deleted_count_pending: u64,
    /// Optional — index implementations that don't compute checksums
    /// can leave it `None`.
    pub checksum: Option<String>,
    pub created_at_unix_micros: i64,
    pub updated_at_unix_micros: i64,
}

impl HnswManifest {
    /// Create a fresh manifest for a brand-new index. `start` and
    /// `watermark` typically both equal 0 (no entries replayed yet);
    /// callers can pass a higher start if the index was bootstrapped
    /// from a snapshot at a known watermark.
    pub fn new(
        tenant_id: TenantId,
        embedding_model: impl Into<String>,
        vector_dim: u32,
        distance_metric: DistanceMetric,
    ) -> Self {
        let now = systime_to_micros(SystemTime::now());
        Self {
            tenant_id,
            embedding_model: embedding_model.into(),
            index_generation: 1,
            source_log_start: 0,
            source_log_watermark: 0,
            vector_dim,
            distance_metric,
            deleted_count_pending: 0,
            checksum: None,
            created_at_unix_micros: now,
            updated_at_unix_micros: now,
        }
    }

    /// Returns true if `other` and `self` agree on the immutable index
    /// parameters (tenant, model, dim, metric). The reconciler uses
    /// this to refuse when recall config drifts from the manifest.
    pub fn parameters_match(&self, other: &Self) -> bool {
        self.tenant_id == other.tenant_id
            && self.embedding_model == other.embedding_model
            && self.vector_dim == other.vector_dim
            && self.distance_metric == other.distance_metric
    }

    /// Number of mutations the index covers. Useful for metrics.
    pub fn entries_covered(&self) -> u64 {
        self.source_log_watermark
            .saturating_sub(self.source_log_start)
    }
}

#[derive(Debug, Error)]
pub enum HnswManifestError {
    #[error("manifest store SQL error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error(
        "manifest watermark must be >= source_log_start (start={start}, watermark={watermark})"
    )]
    WatermarkBeforeStart { start: u64, watermark: u64 },

    #[error("manifest watermark cannot move backward (current={current}, proposed={proposed})")]
    WatermarkRegression { current: u64, proposed: u64 },

    #[error("unknown distance_metric in stored row: {0}")]
    UnknownDistanceMetric(String),
}

/// Storage trait for manifests. Lets us swap SQLite-backed
/// (production) with in-memory (tests / dry-run reconciliation) without
/// touching callers.
pub trait HnswManifestStore: Send + Sync {
    /// Look up a manifest. Returns `None` if no row exists for
    /// `(tenant_id, embedding_model)`.
    fn get(
        &self,
        tenant_id: TenantId,
        embedding_model: &str,
    ) -> Result<Option<HnswManifest>, HnswManifestError>;

    /// Insert or update a manifest. The manifest's
    /// `updated_at_unix_micros` is set to "now" by the implementation.
    /// Watermark monotonicity is enforced — a regression returns
    /// `WatermarkRegression` rather than overwriting.
    fn upsert(&self, manifest: &HnswManifest) -> Result<(), HnswManifestError>;

    /// Advance the watermark. Cheaper than a full upsert when only the
    /// watermark + updated_at change. Enforces monotonicity.
    fn advance_watermark(
        &self,
        tenant_id: TenantId,
        embedding_model: &str,
        new_watermark: u64,
    ) -> Result<(), HnswManifestError>;

    /// List every manifest for a tenant. Used during reconciliation
    /// (one tenant can have a primary + shadow manifest during model
    /// migration).
    fn list_for_tenant(&self, tenant_id: TenantId) -> Result<Vec<HnswManifest>, HnswManifestError>;
}

/// SQLite-backed implementation — production.
pub struct SqliteHnswManifestStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteHnswManifestStore {
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    fn row_to_manifest(row: &rusqlite::Row<'_>) -> Result<HnswManifest, rusqlite::Error> {
        let metric_str: String = row.get(6)?;
        let metric = DistanceMetric::parse(&metric_str).ok_or_else(|| {
            rusqlite::Error::FromSqlConversionFailure(
                6,
                rusqlite::types::Type::Text,
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unknown distance_metric: {metric_str}"),
                )),
            )
        })?;
        Ok(HnswManifest {
            tenant_id: TenantId::new(row.get::<_, i64>(0)?),
            embedding_model: row.get(1)?,
            index_generation: row.get::<_, i64>(2)? as u64,
            source_log_start: row.get::<_, i64>(3)? as u64,
            source_log_watermark: row.get::<_, i64>(4)? as u64,
            vector_dim: row.get::<_, i64>(5)? as u32,
            distance_metric: metric,
            deleted_count_pending: row.get::<_, i64>(7)? as u64,
            checksum: row.get(8)?,
            created_at_unix_micros: row.get(9)?,
            updated_at_unix_micros: row.get(10)?,
        })
    }
}

impl HnswManifestStore for SqliteHnswManifestStore {
    fn get(
        &self,
        tenant_id: TenantId,
        embedding_model: &str,
    ) -> Result<Option<HnswManifest>, HnswManifestError> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT tenant_id, embedding_model, index_generation,
                    source_log_start, source_log_watermark,
                    vector_dim, distance_metric,
                    deleted_count_pending, checksum,
                    created_at_unix_micros, updated_at_unix_micros
             FROM hnsw_manifests
             WHERE tenant_id = ?1 AND embedding_model = ?2",
            params![tenant_id.0, embedding_model],
            Self::row_to_manifest,
        )
        .optional()
        .map_err(HnswManifestError::from)
    }

    fn upsert(&self, manifest: &HnswManifest) -> Result<(), HnswManifestError> {
        if manifest.source_log_watermark < manifest.source_log_start {
            return Err(HnswManifestError::WatermarkBeforeStart {
                start: manifest.source_log_start,
                watermark: manifest.source_log_watermark,
            });
        }

        let conn = self.conn.lock();

        // Enforce watermark monotonicity for an existing row.
        let existing: Option<u64> = conn
            .query_row(
                "SELECT source_log_watermark FROM hnsw_manifests
                 WHERE tenant_id = ?1 AND embedding_model = ?2",
                params![manifest.tenant_id.0, manifest.embedding_model],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .map(|n| n as u64);
        if let Some(current) = existing {
            if manifest.source_log_watermark < current {
                return Err(HnswManifestError::WatermarkRegression {
                    current,
                    proposed: manifest.source_log_watermark,
                });
            }
        }

        let now = systime_to_micros(SystemTime::now());
        // Preserve created_at on update; only set on insert.
        conn.execute(
            "INSERT INTO hnsw_manifests (
                tenant_id, embedding_model, index_generation,
                source_log_start, source_log_watermark,
                vector_dim, distance_metric,
                deleted_count_pending, checksum,
                created_at_unix_micros, updated_at_unix_micros
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(tenant_id, embedding_model) DO UPDATE SET
                index_generation = excluded.index_generation,
                source_log_start = excluded.source_log_start,
                source_log_watermark = excluded.source_log_watermark,
                vector_dim = excluded.vector_dim,
                distance_metric = excluded.distance_metric,
                deleted_count_pending = excluded.deleted_count_pending,
                checksum = excluded.checksum,
                updated_at_unix_micros = excluded.updated_at_unix_micros",
            params![
                manifest.tenant_id.0,
                manifest.embedding_model,
                manifest.index_generation as i64,
                manifest.source_log_start as i64,
                manifest.source_log_watermark as i64,
                manifest.vector_dim as i64,
                manifest.distance_metric.as_str(),
                manifest.deleted_count_pending as i64,
                manifest.checksum,
                manifest.created_at_unix_micros,
                now,
            ],
        )?;
        Ok(())
    }

    fn advance_watermark(
        &self,
        tenant_id: TenantId,
        embedding_model: &str,
        new_watermark: u64,
    ) -> Result<(), HnswManifestError> {
        let conn = self.conn.lock();
        let current: Option<u64> = conn
            .query_row(
                "SELECT source_log_watermark FROM hnsw_manifests
                 WHERE tenant_id = ?1 AND embedding_model = ?2",
                params![tenant_id.0, embedding_model],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .map(|n| n as u64);
        let current = match current {
            Some(c) => c,
            None => {
                return Err(HnswManifestError::Sqlite(
                    rusqlite::Error::QueryReturnedNoRows,
                ))
            }
        };
        if new_watermark < current {
            return Err(HnswManifestError::WatermarkRegression {
                current,
                proposed: new_watermark,
            });
        }
        let now = systime_to_micros(SystemTime::now());
        conn.execute(
            "UPDATE hnsw_manifests
             SET source_log_watermark = ?3, updated_at_unix_micros = ?4
             WHERE tenant_id = ?1 AND embedding_model = ?2",
            params![tenant_id.0, embedding_model, new_watermark as i64, now],
        )?;
        Ok(())
    }

    fn list_for_tenant(&self, tenant_id: TenantId) -> Result<Vec<HnswManifest>, HnswManifestError> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT tenant_id, embedding_model, index_generation,
                    source_log_start, source_log_watermark,
                    vector_dim, distance_metric,
                    deleted_count_pending, checksum,
                    created_at_unix_micros, updated_at_unix_micros
             FROM hnsw_manifests
             WHERE tenant_id = ?1
             ORDER BY embedding_model ASC",
        )?;
        let rows = stmt.query_map(params![tenant_id.0], Self::row_to_manifest)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }
}

fn systime_to_micros(t: SystemTime) -> i64 {
    t.duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migrations::MigrationRunner;

    fn open_store() -> SqliteHnswManifestStore {
        let mut conn = Connection::open_in_memory().unwrap();
        MigrationRunner::run_pending(&mut conn).unwrap();
        SqliteHnswManifestStore::new(Arc::new(Mutex::new(conn)))
    }

    fn fresh_manifest(tenant: i64, model: &str) -> HnswManifest {
        HnswManifest::new(TenantId::new(tenant), model, 384, DistanceMetric::Cosine)
    }

    #[test]
    fn distance_metric_round_trip_through_str() {
        for m in [
            DistanceMetric::Cosine,
            DistanceMetric::L2,
            DistanceMetric::InnerProduct,
        ] {
            assert_eq!(DistanceMetric::parse(m.as_str()), Some(m));
        }
        assert_eq!(DistanceMetric::parse("not_a_metric"), None);
    }

    #[test]
    fn distance_metric_strings_are_pinned() {
        // Stored on disk — never rename.
        assert_eq!(DistanceMetric::Cosine.as_str(), "cosine");
        assert_eq!(DistanceMetric::L2.as_str(), "l2");
        assert_eq!(DistanceMetric::InnerProduct.as_str(), "inner_product");
    }

    #[test]
    fn manifest_new_initializes_watermark_to_zero() {
        let m = HnswManifest::new(
            TenantId::new(1),
            "MiniLM-L6-v2",
            384,
            DistanceMetric::Cosine,
        );
        assert_eq!(m.source_log_start, 0);
        assert_eq!(m.source_log_watermark, 0);
        assert_eq!(m.entries_covered(), 0);
        assert_eq!(m.index_generation, 1);
    }

    #[test]
    fn parameters_match_compares_immutable_fields() {
        let a = HnswManifest::new(TenantId::new(1), "m1", 384, DistanceMetric::Cosine);
        let b = a.clone();
        assert!(a.parameters_match(&b));

        // Different dim → mismatch.
        let mut c = a.clone();
        c.vector_dim = 768;
        assert!(!a.parameters_match(&c));

        // Different model → mismatch.
        let mut d = a.clone();
        d.embedding_model = "m2".into();
        assert!(!a.parameters_match(&d));

        // Different metric → mismatch.
        let mut e = a.clone();
        e.distance_metric = DistanceMetric::L2;
        assert!(!a.parameters_match(&e));

        // Watermark drift is NOT a parameter mismatch — that's what
        // reconciliation handles.
        let mut f = a.clone();
        f.source_log_watermark = 100;
        assert!(a.parameters_match(&f));
    }

    #[test]
    fn upsert_then_get_round_trips() {
        let store = open_store();
        let m = fresh_manifest(1, "MiniLM-L6-v2");
        store.upsert(&m).unwrap();
        let back = store
            .get(TenantId::new(1), "MiniLM-L6-v2")
            .unwrap()
            .unwrap();
        // updated_at_unix_micros may differ — assert other fields.
        assert_eq!(back.tenant_id, m.tenant_id);
        assert_eq!(back.embedding_model, m.embedding_model);
        assert_eq!(back.index_generation, m.index_generation);
        assert_eq!(back.source_log_start, m.source_log_start);
        assert_eq!(back.source_log_watermark, m.source_log_watermark);
        assert_eq!(back.vector_dim, m.vector_dim);
        assert_eq!(back.distance_metric, m.distance_metric);
        assert_eq!(back.deleted_count_pending, m.deleted_count_pending);
        assert_eq!(back.checksum, m.checksum);
    }

    #[test]
    fn get_returns_none_for_missing_tenant() {
        let store = open_store();
        let none = store.get(TenantId::new(99), "anything").unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn upsert_updates_existing_row() {
        let store = open_store();
        let mut m = fresh_manifest(1, "MiniLM-L6-v2");
        store.upsert(&m).unwrap();
        m.source_log_watermark = 10;
        m.deleted_count_pending = 3;
        store.upsert(&m).unwrap();
        let back = store
            .get(TenantId::new(1), "MiniLM-L6-v2")
            .unwrap()
            .unwrap();
        assert_eq!(back.source_log_watermark, 10);
        assert_eq!(back.deleted_count_pending, 3);
    }

    #[test]
    fn upsert_rejects_watermark_before_start() {
        let store = open_store();
        let mut m = fresh_manifest(1, "MiniLM-L6-v2");
        m.source_log_start = 50;
        m.source_log_watermark = 10; // < start
        let err = store.upsert(&m).unwrap_err();
        assert!(matches!(
            err,
            HnswManifestError::WatermarkBeforeStart { .. }
        ));
    }

    #[test]
    fn upsert_rejects_watermark_regression() {
        let store = open_store();
        let mut m = fresh_manifest(1, "MiniLM-L6-v2");
        m.source_log_watermark = 100;
        store.upsert(&m).unwrap();

        // Try to regress.
        m.source_log_watermark = 50;
        let err = store.upsert(&m).unwrap_err();
        assert!(matches!(err, HnswManifestError::WatermarkRegression { .. }));

        // The on-disk row still says 100.
        let back = store
            .get(TenantId::new(1), "MiniLM-L6-v2")
            .unwrap()
            .unwrap();
        assert_eq!(back.source_log_watermark, 100);
    }

    #[test]
    fn advance_watermark_succeeds_monotonically() {
        let store = open_store();
        let m = fresh_manifest(1, "MiniLM-L6-v2");
        store.upsert(&m).unwrap();

        store
            .advance_watermark(TenantId::new(1), "MiniLM-L6-v2", 10)
            .unwrap();
        store
            .advance_watermark(TenantId::new(1), "MiniLM-L6-v2", 50)
            .unwrap();

        let back = store
            .get(TenantId::new(1), "MiniLM-L6-v2")
            .unwrap()
            .unwrap();
        assert_eq!(back.source_log_watermark, 50);

        // Backward move rejected.
        let err = store
            .advance_watermark(TenantId::new(1), "MiniLM-L6-v2", 30)
            .unwrap_err();
        assert!(matches!(err, HnswManifestError::WatermarkRegression { .. }));
    }

    #[test]
    fn advance_watermark_on_missing_row_is_an_error() {
        let store = open_store();
        let err = store
            .advance_watermark(TenantId::new(99), "nonexistent", 10)
            .unwrap_err();
        assert!(matches!(err, HnswManifestError::Sqlite(_)));
    }

    #[test]
    fn list_for_tenant_returns_all_models() {
        let store = open_store();
        store.upsert(&fresh_manifest(1, "MiniLM-L6-v2")).unwrap();
        store.upsert(&fresh_manifest(1, "bge-base")).unwrap();
        store.upsert(&fresh_manifest(2, "MiniLM-L6-v2")).unwrap(); // different tenant

        let t1 = store.list_for_tenant(TenantId::new(1)).unwrap();
        assert_eq!(t1.len(), 2);
        // Ordered by embedding_model ASC: bge-base before MiniLM-L6-v2.
        assert_eq!(t1[0].embedding_model, "MiniLM-L6-v2");
        assert_eq!(t1[1].embedding_model, "bge-base");
        // Wait — alphabetical: 'M' (77) > 'b' (98)? No, 'M' is 77, 'b' is 98. So 'M' sorts first.
        // Adjust expectation. Sorted ASCII-ascending puts uppercase before lowercase.

        let t2 = store.list_for_tenant(TenantId::new(2)).unwrap();
        assert_eq!(t2.len(), 1);

        let empty = store.list_for_tenant(TenantId::new(99)).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn entries_covered_handles_empty_range() {
        let m = HnswManifest::new(TenantId::new(1), "m", 384, DistanceMetric::Cosine);
        // start = watermark = 0
        assert_eq!(m.entries_covered(), 0);
    }

    #[test]
    fn entries_covered_saturates_on_inverted_range() {
        // Defensive: even if start > watermark slipped through (which
        // upsert would reject), `entries_covered` doesn't underflow.
        let mut m = HnswManifest::new(TenantId::new(1), "m", 384, DistanceMetric::Cosine);
        m.source_log_start = 100;
        m.source_log_watermark = 50;
        assert_eq!(m.entries_covered(), 0);
    }
}
