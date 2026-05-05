//! Startup reconciliation — compares each manifest's
//! `source_log_watermark` against the per-tenant commit-log
//! high-watermark and reports the gap.
//!
//! ## What this catches
//!
//! v0.5.2 bug class: a follower applied the oplog so memories landed in
//! SQLite, but its HNSW index was not rebuilt from those entries —
//! recall over the affected window returned nothing. The reconciler
//! detects this on startup before any traffic is admitted.
//!
//! ## What this does NOT do
//!
//! - **Does not rebuild the index.** Phase 1 reports the gap; Phase 2 /
//!   013-B will queue an RFC 019 background job to actually replay.
//! - **Does not modify the commit log.** Reconciliation is read-only
//!   relative to the log; it may upsert manifests but never rewrites
//!   memories.
//! - **Does not know about tombstones.** The tombstone index (RFC
//!   011-B) has its own hydration; reconciliation is purely about
//!   index/log alignment.
//!
//! ## Status semantics
//!
//! - [`ReconciliationStatus::Healthy`] — manifest watermark matches
//!   commit log high water. Recall is safe.
//! - [`ReconciliationStatus::Stale { gap }`] — manifest is behind by
//!   `gap` entries. Recall MAY return incomplete results until the
//!   rebuild worker catches up.
//! - [`ReconciliationStatus::AheadOfLog { excess }`] — manifest claims
//!   watermark beyond commit log. Indicates corruption (e.g. log was
//!   truncated and manifest wasn't, or a backup/restore cycle skipped
//!   the manifest). Recall MUST refuse.
//! - [`ReconciliationStatus::MissingManifest`] — no manifest exists for
//!   a tenant that has commit-log entries. Bootstrap path: build index
//!   from scratch and stamp a fresh manifest.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::manifest::{HnswManifest, HnswManifestError, HnswManifestStore};
use crate::commit::{CommitError, MutationCommitter, TenantId};

/// What the reconciler thinks of a single (tenant, manifest) pairing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ReconciliationStatus {
    /// Manifest is in sync with the commit log.
    Healthy,
    /// Manifest is `gap` entries behind the log. Trigger rebuild.
    Stale { gap: u64 },
    /// Manifest claims a watermark beyond what the log has. Corrupt.
    AheadOfLog { excess: u64 },
    /// No manifest exists for this tenant despite log entries existing.
    MissingManifest { commit_log_high_water: u64 },
}

impl ReconciliationStatus {
    /// Stable label for /metrics gauges.
    pub fn variant_name(&self) -> &'static str {
        match self {
            ReconciliationStatus::Healthy => "Healthy",
            ReconciliationStatus::Stale { .. } => "Stale",
            ReconciliationStatus::AheadOfLog { .. } => "AheadOfLog",
            ReconciliationStatus::MissingManifest { .. } => "MissingManifest",
        }
    }

    /// Whether recall should refuse on this status. AheadOfLog =
    /// corruption, refuse. Stale + MissingManifest are degraded but
    /// can serve partial results (with a stale-result header).
    pub fn requires_recall_refusal(&self) -> bool {
        matches!(self, ReconciliationStatus::AheadOfLog { .. })
    }
}

/// One row of the reconciliation report — the status of one manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconciliationReport {
    pub tenant_id: TenantId,
    /// `None` when the status is `MissingManifest`.
    pub embedding_model: Option<String>,
    pub status: ReconciliationStatus,
    pub manifest_watermark: u64,
    pub commit_log_high_water: u64,
}

impl ReconciliationReport {
    /// True if every covered manifest is `Healthy`.
    pub fn is_healthy(&self) -> bool {
        matches!(self.status, ReconciliationStatus::Healthy)
    }
}

/// Cross-checks manifest watermarks against the commit log and reports
/// drift. Stateless — each call queries fresh.
pub struct Reconciler<C: MutationCommitter + ?Sized> {
    manifests: Arc<dyn HnswManifestStore>,
    committer: Arc<C>,
}

impl<C: MutationCommitter + ?Sized> Reconciler<C> {
    pub fn new(manifests: Arc<dyn HnswManifestStore>, committer: Arc<C>) -> Self {
        Self {
            manifests,
            committer,
        }
    }

    /// Reconcile every manifest the store knows about for `tenant_id`.
    /// Returns one report per `(tenant, embedding_model)` pair, plus
    /// (if no manifest exists but the commit log has entries) a single
    /// `MissingManifest` report.
    pub async fn reconcile_tenant(
        &self,
        tenant_id: TenantId,
    ) -> Result<Vec<ReconciliationReport>, ReconcileError> {
        let high_water = self
            .committer
            .high_watermark(tenant_id)
            .await
            .map_err(ReconcileError::Commit)?;
        let manifests = self
            .manifests
            .list_for_tenant(tenant_id)
            .map_err(ReconcileError::Manifest)?;

        if manifests.is_empty() {
            // No manifest: bootstrap-needed if the log has entries; otherwise
            // healthy-and-empty (return empty vec — nothing to reconcile).
            if high_water > 0 {
                return Ok(vec![ReconciliationReport {
                    tenant_id,
                    embedding_model: None,
                    status: ReconciliationStatus::MissingManifest {
                        commit_log_high_water: high_water,
                    },
                    manifest_watermark: 0,
                    commit_log_high_water: high_water,
                }]);
            }
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(manifests.len());
        for m in manifests {
            out.push(Self::report_for(m, high_water));
        }
        Ok(out)
    }

    fn report_for(m: HnswManifest, high_water: u64) -> ReconciliationReport {
        let status = if m.source_log_watermark > high_water {
            ReconciliationStatus::AheadOfLog {
                excess: m.source_log_watermark - high_water,
            }
        } else if m.source_log_watermark < high_water {
            ReconciliationStatus::Stale {
                gap: high_water - m.source_log_watermark,
            }
        } else {
            ReconciliationStatus::Healthy
        };
        ReconciliationReport {
            tenant_id: m.tenant_id,
            embedding_model: Some(m.embedding_model),
            status,
            manifest_watermark: m.source_log_watermark,
            commit_log_high_water: high_water,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ReconcileError {
    #[error("commit log query failed: {0}")]
    Commit(CommitError),
    #[error("manifest store error: {0}")]
    Manifest(HnswManifestError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::{CommitOptions, LocalSqliteCommitter, MemoryMutation};
    use crate::index::hnsw::manifest::{DistanceMetric, SqliteHnswManifestStore};
    use crate::migrations::MigrationRunner;
    use parking_lot::Mutex;
    use rusqlite::Connection;
    use std::sync::Arc;

    fn upsert(rid: &str) -> MemoryMutation {
        MemoryMutation::UpsertMemory {
            rid: rid.into(),
            text: "x".into(),
            memory_type: "semantic".into(),
            importance: 0.5,
            valence: 0.0,
            half_life: 168.0,
            namespace: "default".into(),
            certainty: 1.0,
            domain: "general".into(),
            source: "user".into(),
            emotional_state: None,
            embedding: None,
            extracted_entities: vec![],
            created_at_unix_micros: None,
            embedding_model: None,
            metadata: serde_json::json!({}),
        }
    }

    fn open_committer_and_store() -> (Arc<LocalSqliteCommitter>, Arc<SqliteHnswManifestStore>) {
        // Use the same in-memory connection for both committer and store
        // so they share state (the migrations must already exist on the
        // store-side connection).
        let committer = Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());

        let mut conn = Connection::open_in_memory().unwrap();
        MigrationRunner::run_pending(&mut conn).unwrap();
        let store = Arc::new(SqliteHnswManifestStore::new(Arc::new(Mutex::new(conn))));

        (committer, store)
    }

    fn fresh_manifest(tenant: i64, model: &str, watermark: u64) -> HnswManifest {
        let mut m = HnswManifest::new(TenantId::new(tenant), model, 384, DistanceMetric::Cosine);
        m.source_log_watermark = watermark;
        m
    }

    #[test]
    fn status_variant_names_are_stable() {
        assert_eq!(ReconciliationStatus::Healthy.variant_name(), "Healthy");
        assert_eq!(
            ReconciliationStatus::Stale { gap: 1 }.variant_name(),
            "Stale"
        );
        assert_eq!(
            ReconciliationStatus::AheadOfLog { excess: 1 }.variant_name(),
            "AheadOfLog"
        );
        assert_eq!(
            ReconciliationStatus::MissingManifest {
                commit_log_high_water: 0
            }
            .variant_name(),
            "MissingManifest"
        );
    }

    #[test]
    fn requires_recall_refusal_only_for_ahead_of_log() {
        assert!(!ReconciliationStatus::Healthy.requires_recall_refusal());
        assert!(!ReconciliationStatus::Stale { gap: 5 }.requires_recall_refusal());
        assert!(ReconciliationStatus::AheadOfLog { excess: 1 }.requires_recall_refusal());
        assert!(!ReconciliationStatus::MissingManifest {
            commit_log_high_water: 5
        }
        .requires_recall_refusal());
    }

    #[tokio::test]
    async fn empty_tenant_with_no_log_or_manifest_returns_empty_report() {
        let (committer, store) = open_committer_and_store();
        let reconciler = Reconciler::new(store, committer);
        let reports = reconciler
            .reconcile_tenant(TenantId::new(99))
            .await
            .unwrap();
        assert!(reports.is_empty());
    }

    #[tokio::test]
    async fn missing_manifest_when_log_has_entries() {
        let (committer, store) = open_committer_and_store();
        // Two commits, no manifest.
        committer
            .commit(TenantId::new(1), upsert("a"), CommitOptions::default())
            .await
            .unwrap();
        committer
            .commit(TenantId::new(1), upsert("b"), CommitOptions::default())
            .await
            .unwrap();

        let reconciler = Reconciler::new(store, committer);
        let reports = reconciler.reconcile_tenant(TenantId::new(1)).await.unwrap();
        assert_eq!(reports.len(), 1);
        assert!(matches!(
            reports[0].status,
            ReconciliationStatus::MissingManifest {
                commit_log_high_water: 2
            }
        ));
        assert_eq!(reports[0].embedding_model, None);
        assert_eq!(reports[0].manifest_watermark, 0);
        assert_eq!(reports[0].commit_log_high_water, 2);
    }

    #[tokio::test]
    async fn healthy_when_manifest_matches_log() {
        let (committer, store) = open_committer_and_store();
        committer
            .commit(TenantId::new(1), upsert("a"), CommitOptions::default())
            .await
            .unwrap();
        committer
            .commit(TenantId::new(1), upsert("b"), CommitOptions::default())
            .await
            .unwrap();
        store.upsert(&fresh_manifest(1, "m", 2)).unwrap();

        let reconciler = Reconciler::new(store, committer);
        let reports = reconciler.reconcile_tenant(TenantId::new(1)).await.unwrap();
        assert_eq!(reports.len(), 1);
        assert!(matches!(reports[0].status, ReconciliationStatus::Healthy));
        assert!(reports[0].is_healthy());
    }

    #[tokio::test]
    async fn stale_reports_correct_gap() {
        let (committer, store) = open_committer_and_store();
        for tag in ["a", "b", "c", "d", "e"] {
            committer
                .commit(TenantId::new(1), upsert(tag), CommitOptions::default())
                .await
                .unwrap();
        }
        store.upsert(&fresh_manifest(1, "m", 2)).unwrap();

        let reconciler = Reconciler::new(store, committer);
        let reports = reconciler.reconcile_tenant(TenantId::new(1)).await.unwrap();
        assert_eq!(reports.len(), 1);
        match &reports[0].status {
            ReconciliationStatus::Stale { gap } => assert_eq!(*gap, 3),
            other => panic!("expected Stale, got {other:?}"),
        }
        assert!(!reports[0].is_healthy());
    }

    #[tokio::test]
    async fn ahead_of_log_when_manifest_overshoots() {
        // Simulates a corrupt restore where manifest carries forward
        // but commit log was reset.
        let (committer, store) = open_committer_and_store();
        committer
            .commit(TenantId::new(1), upsert("a"), CommitOptions::default())
            .await
            .unwrap();
        // Manifest claims watermark 10, but log only has 1 entry.
        store.upsert(&fresh_manifest(1, "m", 10)).unwrap();

        let reconciler = Reconciler::new(store, committer);
        let reports = reconciler.reconcile_tenant(TenantId::new(1)).await.unwrap();
        match &reports[0].status {
            ReconciliationStatus::AheadOfLog { excess } => assert_eq!(*excess, 9),
            other => panic!("expected AheadOfLog, got {other:?}"),
        }
        assert!(reports[0].status.requires_recall_refusal());
    }

    #[tokio::test]
    async fn multiple_models_each_get_a_report() {
        let (committer, store) = open_committer_and_store();
        committer
            .commit(TenantId::new(1), upsert("a"), CommitOptions::default())
            .await
            .unwrap();
        committer
            .commit(TenantId::new(1), upsert("b"), CommitOptions::default())
            .await
            .unwrap();
        // Primary at watermark 2 (healthy), shadow at watermark 0 (stale).
        store.upsert(&fresh_manifest(1, "MiniLM-L6-v2", 2)).unwrap();
        store.upsert(&fresh_manifest(1, "bge-base", 0)).unwrap();

        let reconciler = Reconciler::new(store, committer);
        let reports = reconciler.reconcile_tenant(TenantId::new(1)).await.unwrap();
        assert_eq!(reports.len(), 2);
        // Check: one healthy, one stale.
        let mut found_healthy = false;
        let mut found_stale = false;
        for r in &reports {
            match r.status {
                ReconciliationStatus::Healthy => found_healthy = true,
                ReconciliationStatus::Stale { gap: 2 } => found_stale = true,
                _ => panic!("unexpected status: {:?}", r.status),
            }
        }
        assert!(found_healthy && found_stale);
    }

    #[tokio::test]
    async fn per_tenant_isolation() {
        // Tenant 1 has a healthy manifest, tenant 2 has none. Reconcile
        // for tenant 2 should not see tenant 1's manifest.
        let (committer, store) = open_committer_and_store();
        committer
            .commit(TenantId::new(1), upsert("a"), CommitOptions::default())
            .await
            .unwrap();
        store.upsert(&fresh_manifest(1, "m", 1)).unwrap();

        let reconciler = Reconciler::new(store, committer);
        let t2 = reconciler.reconcile_tenant(TenantId::new(2)).await.unwrap();
        assert!(t2.is_empty());
    }
}
