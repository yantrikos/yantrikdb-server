//! RFC 010-C — commit log retention / compaction watermark contract.
//!
//! ## What this module owns
//!
//! Compacting the per-tenant commit log is destructive: any entry
//! purged below the compaction watermark cannot be recovered.
//! Multiple subsystems depend on those entries staying around for
//! their own correctness — a follower replicating from behind, an
//! HNSW index that hasn't caught up to the leader yet, a backup
//! manifest pointing at a not-yet-archived range, a tombstone that
//! must propagate before the original write can be erased.
//!
//! Compaction CANNOT just pick a number. It must take the **minimum**
//! across every constraint or it WILL violate at least one of the
//! invariants those constraints encode.
//!
//! This module ships:
//! 1. [`RetentionContributor`] trait — anyone gating compaction
//!    reports a per-tenant floor (the lowest log_index they still
//!    need).
//! 2. [`RetentionRegistry`] — collects all contributors, computes
//!    the safe lower bound as `min(floor across all contributors)`.
//! 3. Built-in contributor stubs for the four invariants identified
//!    in the gpt-5.5 redteam: follower replication lag, HNSW
//!    source-log watermark, backup manifest watermark, tombstone
//!    retention windows.
//! 4. [`SafePurgeWatermark`] result type that callers (the eventual
//!    compactor) consume.
//!
//! ## What this module does NOT own
//!
//! - The compactor itself. RFC 010-C ships the API and the safe
//!   lower-bound calculation; the actual loop that calls
//!   `MutationCommitter::purge_below(watermark)` lives in a future
//!   PR alongside the policy knobs (compaction cadence, batch size).
//! - Per-tenant override config. RFC 021 (config versioning) will
//!   wire in operator-tunable retention windows; for now the
//!   defaults are baked into [`RetentionPolicy`].
//! - The HNSW manifest itself (already shipped in RFC 013 Phase 1).
//!   The contributor here just *reads* `source_log_watermark` from a
//!   `HnswManifestStore`.
//!
//! ## Invariants this module exists to protect
//!
//! - **Restore-no-resurrect** (RFC 011): a memory that was tombstoned
//!   and whose tombstone has propagated to all replicas must not
//!   reappear after a backup/restore cycle. Implemented by ensuring
//!   tombstone entries stay in the log at least until every
//!   subscriber acknowledges them, AND ensuring backups capture the
//!   tombstone before their source-log-watermark advances past it.
//! - **Follower never sees a hole**: a follower at log_index N must
//!   be able to ask the leader for entries `[N+1, ...)`. If
//!   compaction has already purged them, the follower is permanently
//!   broken.
//! - **HNSW reconcilable from log**: per RFC 013, on startup the
//!   HNSW manifest's source_log_watermark must be reachable from the
//!   commit log so any drift can be replayed.
//! - **Backup snapshot stable**: a snapshot in flight cannot have
//!   its source range purged out from under it.
//!
//! ## How contributors compose
//!
//! Each contributor reports a floor `F_i(tenant_id)` — "I still need
//! entries with log_index ≥ F_i to do my job." The safe purge
//! watermark is `min(F_i for all i)`. If any contributor returns
//! `None`, that means it doesn't constrain compaction at this moment
//! (e.g., no tombstones to retain, no backup in flight, no follower
//! lag). If every contributor returns `None`, the safe watermark is
//! `commit_log.high_watermark` minus zero — compaction has free run.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::TenantId;

/// Default tombstone retention window. A tombstone must remain in the
/// log at least this long to give every conceivable subscriber time to
/// see it before the original entry can be physically erased.
///
/// 24h is operator-friendly: long enough that a follower offline for
/// a maintenance window doesn't miss a tombstone, short enough that
/// the log doesn't accumulate dead weight indefinitely.
pub const DEFAULT_TOMBSTONE_RETENTION: Duration = Duration::from_secs(24 * 60 * 60);

/// Minimum tombstone retention. Even an aggressively-tuned deployment
/// shouldn't go below this — tombstones MUST outlive the longest
/// expected network partition + heartbeat-loss window.
pub const MIN_TOMBSTONE_RETENTION: Duration = Duration::from_secs(60 * 60);

/// Per-deployment retention configuration. Defaults are conservative:
/// operators dial down individual windows after measuring their actual
/// replication / backup cadence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub tombstone_retention: Duration,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            tombstone_retention: DEFAULT_TOMBSTONE_RETENTION,
        }
    }
}

impl RetentionPolicy {
    /// Validate operator-supplied policy. Returns Err if any window
    /// is below its minimum.
    pub fn validate(&self) -> Result<(), RetentionError> {
        if self.tombstone_retention < MIN_TOMBSTONE_RETENTION {
            return Err(RetentionError::PolicyTooAggressive {
                field: "tombstone_retention",
                min: MIN_TOMBSTONE_RETENTION,
                got: self.tombstone_retention,
            });
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RetentionError {
    #[error(
        "retention policy field `{field}` set below safety floor: \
         got {got:?}, minimum is {min:?}"
    )]
    PolicyTooAggressive {
        field: &'static str,
        min: Duration,
        got: Duration,
    },
    #[error("contributor `{name}` failed: {message}")]
    ContributorFailed { name: &'static str, message: String },
}

/// One subsystem's gate on compaction. Each contributor answers, for
/// a given tenant, "what's the lowest log_index I still need around?"
///
/// Implementations MUST be cheap to call: this is on the compaction
/// hot path, evaluated for every tenant on every compaction tick.
#[async_trait]
pub trait RetentionContributor: Send + Sync {
    /// Stable name for metrics + the audit log. Never rename — these
    /// strings end up in operator dashboards.
    fn name(&self) -> &'static str;

    /// Report the floor (lowest log_index still needed) for the given
    /// tenant. `None` means "I don't care about this tenant right
    /// now" — compaction is free to ignore me for this evaluation.
    async fn min_required_log_index(
        &self,
        tenant_id: TenantId,
    ) -> Result<Option<u64>, RetentionError>;
}

/// Collects all contributors, takes the minimum.
///
/// `Arc<dyn ...>` so the registry can be shared across the compactor
/// loop and any inspector / dashboard that wants to render per-tenant
/// floors.
#[derive(Default, Clone)]
pub struct RetentionRegistry {
    contributors: Vec<Arc<dyn RetentionContributor>>,
}

impl RetentionRegistry {
    pub fn new() -> Self {
        Self {
            contributors: Vec::new(),
        }
    }

    pub fn with(mut self, c: Arc<dyn RetentionContributor>) -> Self {
        self.contributors.push(c);
        self
    }

    pub fn add(&mut self, c: Arc<dyn RetentionContributor>) {
        self.contributors.push(c);
    }

    pub fn contributor_names(&self) -> Vec<&'static str> {
        self.contributors.iter().map(|c| c.name()).collect()
    }

    /// Compute the safe purge watermark for a tenant.
    ///
    /// The result is the **minimum** floor across every contributor,
    /// or `None` if every contributor reported `None` (no constraints,
    /// the entire log is purgeable).
    ///
    /// If any contributor errors, the entire computation errors —
    /// callers MUST treat that as "do not compact" rather than
    /// proceeding with a partial result. Compaction is destructive;
    /// missing one constraint means violating one invariant.
    pub async fn safe_purge_watermark(
        &self,
        tenant_id: TenantId,
    ) -> Result<SafePurgeWatermark, RetentionError> {
        let mut floors: Vec<(&'static str, Option<u64>)> =
            Vec::with_capacity(self.contributors.len());
        for c in &self.contributors {
            let f = c.min_required_log_index(tenant_id).await?;
            floors.push((c.name(), f));
        }

        // The minimum of the constraining floors.
        let min_floor = floors.iter().filter_map(|(_, f)| *f).min();

        // If a contributor had a floor, it gates compaction. The
        // safe purge watermark is `min_floor - 1` (the highest index
        // we are SAFE to purge through, exclusive of the floor).
        let safe_purge_through = min_floor.map(|f| f.saturating_sub(1));

        Ok(SafePurgeWatermark {
            tenant_id,
            safe_purge_through,
            min_required: min_floor,
            per_contributor: floors,
        })
    }
}

/// Result of a watermark computation. `safe_purge_through` is the
/// **highest log_index that is safe to purge**, inclusive. Callers
/// pass this to the underlying `MutationCommitter::purge_through`
/// (or equivalent) — anything ≤ this index can be physically
/// removed; anything > this index must remain.
///
/// `None` for `safe_purge_through` means "no contributor reported a
/// floor, the entire log is purgeable" — typically only true for
/// brand-new tenants with no traffic + no replication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafePurgeWatermark {
    pub tenant_id: TenantId,
    /// Highest log_index inclusive that's safe to purge. `None` =
    /// no constraints reported.
    pub safe_purge_through: Option<u64>,
    /// The single tightest floor (the minimum among contributing
    /// floors). Surfaces which constraint is binding.
    pub min_required: Option<u64>,
    /// Per-contributor floor breakdown for operator visibility +
    /// `/v1/admin/retention` introspection.
    pub per_contributor: Vec<(&'static str, Option<u64>)>,
}

impl SafePurgeWatermark {
    /// Returns the contributor name + floor that's currently
    /// binding (i.e., the minimum). `None` if no contributor
    /// reported a floor.
    pub fn binding_contributor(&self) -> Option<(&'static str, u64)> {
        self.per_contributor
            .iter()
            .filter_map(|(n, f)| f.map(|v| (*n, v)))
            .min_by_key(|(_, v)| *v)
    }
}

// ============================================================
// Built-in contributors
// ============================================================

/// Tombstones must remain visible long enough that every replica /
/// subscriber has seen them. Returns the log_index of the oldest
/// tombstone that hasn't yet aged past `policy.tombstone_retention`.
///
/// The TombstoneIndex (RFC 011-B) tracks rids in memory but doesn't
/// keep insertion timestamps. So this contributor has two backing
/// strategies:
///
/// 1. **`Threshold`** mode (current default) — operators set a
///    coarse "hold tombstones for at least N seconds" floor. The
///    contributor reports a floor equal to "the log_index N seconds
///    ago." Conservative: may keep tombstones around longer than
///    strictly needed, but never drops one before its window.
///
/// 2. **`PerTombstoneAge`** mode (deferred) — uses commit_log's
///    `committed_at_unix_micros` per tombstone entry to compute the
///    exact youngest-still-aging tombstone. More precise but
///    requires a SQL scan over the log on every evaluation. Defer
///    until we have ops data showing the conservative mode is
///    expensive.
pub struct TombstoneRetentionContributor<C: ?Sized> {
    committer: Arc<C>,
    policy: RetentionPolicy,
    /// `now_fn` for testability — production passes
    /// `|| SystemTime::now()`.
    now_fn: Arc<dyn Fn() -> SystemTime + Send + Sync>,
}

impl<C: ?Sized> TombstoneRetentionContributor<C>
where
    C: super::MutationCommitter,
{
    pub fn new(committer: Arc<C>, policy: RetentionPolicy) -> Self {
        Self {
            committer,
            policy,
            now_fn: Arc::new(SystemTime::now),
        }
    }

    /// Test-only: inject a deterministic clock.
    #[cfg(test)]
    pub fn with_clock(
        committer: Arc<C>,
        policy: RetentionPolicy,
        now_fn: impl Fn() -> SystemTime + Send + Sync + 'static,
    ) -> Self {
        Self {
            committer,
            policy,
            now_fn: Arc::new(now_fn),
        }
    }
}

#[async_trait]
impl<C: ?Sized> RetentionContributor for TombstoneRetentionContributor<C>
where
    C: super::MutationCommitter + 'static,
{
    fn name(&self) -> &'static str {
        "tombstone_retention"
    }

    async fn min_required_log_index(
        &self,
        tenant_id: TenantId,
    ) -> Result<Option<u64>, RetentionError> {
        // Walk the log from the start; find the first entry that's
        // YOUNGER than `now - retention`. Any entry older than the
        // retention window is safe to drop; that's the floor.
        //
        // For Phase 1 implementation, scan from log_index 1 in
        // chunks until we hit an entry inside the retention window.
        // Production should add an index on committed_at_unix_micros
        // for an O(log N) lookup; this scan is acceptable until
        // commit logs grow past ~100k entries per tenant.
        let now = (self.now_fn)();
        let cutoff = now
            .checked_sub(self.policy.tombstone_retention)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let cutoff_micros = cutoff
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);

        let high = self
            .committer
            .high_watermark(tenant_id)
            .await
            .map_err(|e| RetentionError::ContributorFailed {
                name: "tombstone_retention",
                message: format!("high_watermark: {e}"),
            })?;
        if high == 0 {
            return Ok(None); // no entries
        }

        // Scan in pages to bound memory.
        const PAGE: usize = 1024;
        let mut from = 1;
        while from <= high {
            let entries = self
                .committer
                .read_range(tenant_id, from, PAGE)
                .await
                .map_err(|e| RetentionError::ContributorFailed {
                    name: "tombstone_retention",
                    message: format!("read_range from={from}: {e}"),
                })?;
            if entries.is_empty() {
                break;
            }
            for e in &entries {
                let committed_micros = e
                    .committed_at
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_micros() as i64)
                    .unwrap_or(0);
                if committed_micros >= cutoff_micros {
                    // This entry is INSIDE the retention window —
                    // it (and everything after it) must remain.
                    return Ok(Some(e.log_index));
                }
            }
            from = entries.last().map(|e| e.log_index + 1).unwrap_or(high + 1);
        }
        // Every entry is older than the retention window; nothing
        // gates compaction from the tombstone-retention angle.
        Ok(None)
    }
}

/// HNSW indexes need every entry between the manifest's
/// `source_log_watermark` and the commit log's high_watermark to
/// remain reachable, so that on startup the index can replay any
/// missing mutations. Per RFC 013 Phase 1.
pub struct HnswWatermarkContributor {
    /// `Arc<dyn ...>` so this composes with whichever manifest store
    /// is configured (SqliteHnswManifestStore in production; an
    /// in-memory stub in tests).
    manifests: Arc<dyn crate::index::hnsw::HnswManifestStore>,
}

impl HnswWatermarkContributor {
    pub fn new(manifests: Arc<dyn crate::index::hnsw::HnswManifestStore>) -> Self {
        Self { manifests }
    }
}

#[async_trait]
impl RetentionContributor for HnswWatermarkContributor {
    fn name(&self) -> &'static str {
        "hnsw_watermark"
    }

    async fn min_required_log_index(
        &self,
        tenant_id: TenantId,
    ) -> Result<Option<u64>, RetentionError> {
        // Walk every manifest (a tenant may carry primary + shadow
        // during model migration per RFC 013-B); take the lowest
        // watermark — that's the index that's most behind, and we
        // can't purge below where ANY index would need to replay.
        let manifests = self.manifests.list_for_tenant(tenant_id).map_err(|e| {
            RetentionError::ContributorFailed {
                name: "hnsw_watermark",
                message: format!("list_for_tenant: {e}"),
            }
        })?;
        if manifests.is_empty() {
            return Ok(None);
        }
        let min_watermark = manifests
            .iter()
            .map(|m| m.source_log_watermark)
            .min()
            .unwrap_or(0);
        // If watermark = 0 it means the index has consumed nothing
        // yet — but the entries still need to be there for the
        // future replay. The floor is `1`, so log_index ≥ 1 is
        // protected (i.e., no entry can be purged because index
        // hasn't replayed anything).
        // If watermark = N (>0), the index has consumed up through
        // entry N. Entries 1..=N can be purged from the
        // index-replay perspective; the floor is N+1.
        Ok(Some(min_watermark.saturating_add(1)))
    }
}

/// In cluster mode, every follower's last_applied_log_index is a
/// floor: purging entries the follower hasn't replicated is fatal
/// because the follower can never catch up. This is a stub for now
/// — the openraft assembly (RFC 010 PR-4-d) doesn't yet expose a
/// per-tenant follower-lag query through the public surface. When
/// that lands, swap this stub for one that reads from the Raft
/// metrics watch channel.
pub struct FollowerLagContributor {
    /// `Some(min_last_applied)` = a measured slowest-follower floor;
    /// `None` = no follower lag info available right now (e.g.,
    /// single-node mode or no learners), so don't constrain
    /// compaction.
    measure: Arc<dyn Fn(TenantId) -> Option<u64> + Send + Sync>,
}

impl FollowerLagContributor {
    /// `measure(tenant) -> Option<min_last_applied_log_index>`. Pass
    /// `|_| None` to disable follower-lag gating in single-node mode.
    pub fn new(measure: impl Fn(TenantId) -> Option<u64> + Send + Sync + 'static) -> Self {
        Self {
            measure: Arc::new(measure),
        }
    }

    /// Convenience: a no-op contributor for single-node deployments.
    /// Reports `None` for every tenant.
    pub fn disabled() -> Self {
        Self::new(|_| None)
    }
}

#[async_trait]
impl RetentionContributor for FollowerLagContributor {
    fn name(&self) -> &'static str {
        "follower_lag"
    }

    async fn min_required_log_index(
        &self,
        tenant_id: TenantId,
    ) -> Result<Option<u64>, RetentionError> {
        // The follower has applied through index N. Entries 1..=N
        // can be purged from the follower-lag perspective; the
        // floor is N+1.
        Ok((self.measure)(tenant_id).map(|applied| applied.saturating_add(1)))
    }
}

/// A backup snapshot in flight (or recently completed) pins every
/// log entry from its declared source-log-start through its
/// source-log-watermark. The contributor returns the minimum start
/// across all in-flight backups for the tenant.
///
/// RFC 012 will plumb this against an in-memory backup-manifest
/// registry. Until then, the contributor is constructed with an
/// arbitrary `Fn` so tests + the eventual integration both work.
pub struct BackupWatermarkContributor {
    measure: Arc<dyn Fn(TenantId) -> Option<u64> + Send + Sync>,
}

impl BackupWatermarkContributor {
    pub fn new(measure: impl Fn(TenantId) -> Option<u64> + Send + Sync + 'static) -> Self {
        Self {
            measure: Arc::new(measure),
        }
    }

    pub fn disabled() -> Self {
        Self::new(|_| None)
    }
}

#[async_trait]
impl RetentionContributor for BackupWatermarkContributor {
    fn name(&self) -> &'static str {
        "backup_watermark"
    }

    async fn min_required_log_index(
        &self,
        tenant_id: TenantId,
    ) -> Result<Option<u64>, RetentionError> {
        // Backups need entries `source_log_start..=source_log_watermark`
        // intact. The floor is the source_log_start (the lowest
        // entry the backup is reading).
        Ok((self.measure)(tenant_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::{CommitOptions, LocalSqliteCommitter, MemoryMutation};
    use crate::index::hnsw::{
        DistanceMetric, HnswManifest, HnswManifestStore, SqliteHnswManifestStore,
    };
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
            metadata: serde_json::json!({}),
        }
    }

    fn open_manifest_store() -> SqliteHnswManifestStore {
        let mut conn = Connection::open_in_memory().unwrap();
        crate::migrations::MigrationRunner::run_pending(&mut conn).unwrap();
        SqliteHnswManifestStore::new(Arc::new(Mutex::new(conn)))
    }

    #[test]
    fn policy_default_meets_minimum() {
        let p = RetentionPolicy::default();
        p.validate().unwrap();
        assert_eq!(p.tombstone_retention, DEFAULT_TOMBSTONE_RETENTION);
    }

    #[test]
    fn policy_below_minimum_rejects() {
        let p = RetentionPolicy {
            tombstone_retention: Duration::from_secs(60), // 1 minute, way below floor
        };
        match p.validate() {
            Err(RetentionError::PolicyTooAggressive { field, .. }) => {
                assert_eq!(field, "tombstone_retention");
            }
            other => panic!("expected PolicyTooAggressive, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_registry_reports_no_constraint() {
        let r = RetentionRegistry::new();
        let w = r.safe_purge_watermark(TenantId::new(1)).await.unwrap();
        assert_eq!(w.safe_purge_through, None);
        assert_eq!(w.min_required, None);
        assert!(w.per_contributor.is_empty());
    }

    /// A test contributor that returns a fixed floor.
    struct StubContributor {
        name: &'static str,
        floor: Option<u64>,
    }

    #[async_trait]
    impl RetentionContributor for StubContributor {
        fn name(&self) -> &'static str {
            self.name
        }
        async fn min_required_log_index(
            &self,
            _tenant_id: TenantId,
        ) -> Result<Option<u64>, RetentionError> {
            Ok(self.floor)
        }
    }

    #[tokio::test]
    async fn min_across_contributors_is_taken() {
        // Three contributors with floors 100, 50, 200 → min is 50,
        // safe purge through is 49.
        let r = RetentionRegistry::new()
            .with(Arc::new(StubContributor {
                name: "a",
                floor: Some(100),
            }))
            .with(Arc::new(StubContributor {
                name: "b",
                floor: Some(50),
            }))
            .with(Arc::new(StubContributor {
                name: "c",
                floor: Some(200),
            }));
        let w = r.safe_purge_watermark(TenantId::new(1)).await.unwrap();
        assert_eq!(w.min_required, Some(50));
        assert_eq!(w.safe_purge_through, Some(49));
        assert_eq!(w.binding_contributor(), Some(("b", 50)));
    }

    #[tokio::test]
    async fn none_floors_skip_contributor() {
        // Contributor A returns None (no constraint). Contributor B
        // returns 30. Min should be 30 — A doesn't contribute.
        let r = RetentionRegistry::new()
            .with(Arc::new(StubContributor {
                name: "a",
                floor: None,
            }))
            .with(Arc::new(StubContributor {
                name: "b",
                floor: Some(30),
            }));
        let w = r.safe_purge_watermark(TenantId::new(1)).await.unwrap();
        assert_eq!(w.min_required, Some(30));
        assert_eq!(w.safe_purge_through, Some(29));
        assert_eq!(w.binding_contributor(), Some(("b", 30)));
    }

    #[tokio::test]
    async fn all_none_means_no_constraint() {
        let r = RetentionRegistry::new()
            .with(Arc::new(StubContributor {
                name: "a",
                floor: None,
            }))
            .with(Arc::new(StubContributor {
                name: "b",
                floor: None,
            }));
        let w = r.safe_purge_watermark(TenantId::new(1)).await.unwrap();
        assert_eq!(w.min_required, None);
        assert_eq!(w.safe_purge_through, None);
        assert!(w.binding_contributor().is_none());
    }

    /// A contributor that always errors, used to verify the
    /// "any error → whole computation errors" rule.
    struct ErrContributor;
    #[async_trait]
    impl RetentionContributor for ErrContributor {
        fn name(&self) -> &'static str {
            "err"
        }
        async fn min_required_log_index(
            &self,
            _tenant_id: TenantId,
        ) -> Result<Option<u64>, RetentionError> {
            Err(RetentionError::ContributorFailed {
                name: "err",
                message: "synthetic".into(),
            })
        }
    }

    #[tokio::test]
    async fn any_error_aborts_safe_watermark() {
        // Critical: a single error must NOT silently exclude that
        // contributor — it must fail the computation. Compaction
        // is destructive; partial info = invariant violation.
        let r = RetentionRegistry::new()
            .with(Arc::new(StubContributor {
                name: "ok",
                floor: Some(100),
            }))
            .with(Arc::new(ErrContributor));
        let result = r.safe_purge_watermark(TenantId::new(1)).await;
        assert!(matches!(
            result,
            Err(RetentionError::ContributorFailed { .. })
        ));
    }

    #[tokio::test]
    async fn safe_purge_through_one_when_floor_is_one() {
        // Edge: floor = 1 means "I need entry 1 still" → safe_purge_through is 0
        // (i.e., nothing may be purged). saturating_sub(1) handles
        // this without underflow.
        let r = RetentionRegistry::new().with(Arc::new(StubContributor {
            name: "a",
            floor: Some(1),
        }));
        let w = r.safe_purge_watermark(TenantId::new(1)).await.unwrap();
        assert_eq!(w.safe_purge_through, Some(0));
    }

    // ── HNSW contributor ────────────────────────────────────────────

    #[tokio::test]
    async fn hnsw_contributor_floor_is_watermark_plus_one() {
        let store = Arc::new(open_manifest_store());
        let mut m = HnswManifest::new(TenantId::new(1), "minilm", 384, DistanceMetric::Cosine);
        m.source_log_watermark = 42;
        store.upsert(&m).unwrap();

        let c = HnswWatermarkContributor::new(store);
        let f = c.min_required_log_index(TenantId::new(1)).await.unwrap();
        assert_eq!(f, Some(43));
    }

    #[tokio::test]
    async fn hnsw_contributor_uses_min_across_manifests() {
        // Tenant has primary + shadow manifests during model
        // migration. The lowest watermark wins (most behind).
        let store = Arc::new(open_manifest_store());
        let tenant = TenantId::new(1);
        let mut primary = HnswManifest::new(tenant, "minilm", 384, DistanceMetric::Cosine);
        primary.source_log_watermark = 100;
        store.upsert(&primary).unwrap();
        let mut shadow = HnswManifest::new(tenant, "bge-base", 768, DistanceMetric::L2);
        shadow.source_log_watermark = 30;
        store.upsert(&shadow).unwrap();

        let c = HnswWatermarkContributor::new(store);
        let f = c.min_required_log_index(tenant).await.unwrap();
        // shadow is at 30 → floor is 31 (the entry the shadow index
        // hasn't yet consumed).
        assert_eq!(f, Some(31));
    }

    #[tokio::test]
    async fn hnsw_contributor_no_manifests_returns_none() {
        let store = Arc::new(open_manifest_store());
        let c = HnswWatermarkContributor::new(store);
        let f = c.min_required_log_index(TenantId::new(99)).await.unwrap();
        assert_eq!(f, None, "tenant with no manifests should not constrain");
    }

    // ── Follower-lag contributor ────────────────────────────────────

    #[tokio::test]
    async fn follower_lag_disabled_reports_none() {
        let c = FollowerLagContributor::disabled();
        let f = c.min_required_log_index(TenantId::new(1)).await.unwrap();
        assert_eq!(f, None);
    }

    #[tokio::test]
    async fn follower_lag_floor_is_applied_plus_one() {
        let c = FollowerLagContributor::new(|t| {
            if t == TenantId::new(1) {
                Some(50)
            } else {
                None
            }
        });
        assert_eq!(
            c.min_required_log_index(TenantId::new(1)).await.unwrap(),
            Some(51)
        );
        assert_eq!(
            c.min_required_log_index(TenantId::new(2)).await.unwrap(),
            None
        );
    }

    // ── Backup watermark contributor ────────────────────────────────

    #[tokio::test]
    async fn backup_watermark_disabled_reports_none() {
        let c = BackupWatermarkContributor::disabled();
        assert_eq!(
            c.min_required_log_index(TenantId::new(1)).await.unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn backup_watermark_floor_pins_source_log_start() {
        let c = BackupWatermarkContributor::new(|_| Some(7));
        assert_eq!(
            c.min_required_log_index(TenantId::new(1)).await.unwrap(),
            Some(7)
        );
    }

    // ── Tombstone retention contributor ─────────────────────────────

    #[tokio::test]
    async fn tombstone_retention_no_entries_returns_none() {
        let committer: Arc<dyn super::super::MutationCommitter> =
            Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let c = TombstoneRetentionContributor::new(committer, RetentionPolicy::default());
        let f = c.min_required_log_index(TenantId::new(99)).await.unwrap();
        assert_eq!(f, None);
    }

    #[tokio::test]
    async fn tombstone_retention_keeps_entries_inside_window() {
        // Write 3 entries. Use a clock that says "now" is ~30 min in
        // the future of the writes, with a 24h retention window.
        // All 3 entries are inside the window → floor is the first
        // one's log_index (= 1).
        let committer: Arc<dyn super::super::MutationCommitter> =
            Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        for tag in ["a", "b", "c"] {
            committer
                .commit(TenantId::new(1), upsert(tag), CommitOptions::default())
                .await
                .unwrap();
        }
        // "now" = real now (writes were just done; all inside 24h).
        let c = TombstoneRetentionContributor::new(committer.clone(), RetentionPolicy::default());
        let f = c.min_required_log_index(TenantId::new(1)).await.unwrap();
        assert_eq!(
            f,
            Some(1),
            "all entries inside window → floor at first entry"
        );
    }

    #[tokio::test]
    async fn tombstone_retention_drops_entries_outside_window() {
        // Write 3 entries. Use a clock that says "now" is 48h in the
        // future. All 3 entries are OLDER than the 24h window →
        // contributor returns None (no constraint).
        let committer: Arc<dyn super::super::MutationCommitter> =
            Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        for tag in ["a", "b", "c"] {
            committer
                .commit(TenantId::new(1), upsert(tag), CommitOptions::default())
                .await
                .unwrap();
        }
        let two_days_later = SystemTime::now() + Duration::from_secs(48 * 60 * 60);
        let c = TombstoneRetentionContributor::with_clock(
            committer.clone(),
            RetentionPolicy::default(),
            move || two_days_later,
        );
        let f = c.min_required_log_index(TenantId::new(1)).await.unwrap();
        assert_eq!(f, None, "all entries past window → no constraint");
    }

    // ── End-to-end registry composition ─────────────────────────────

    #[tokio::test]
    async fn registry_composes_real_contributors() {
        // Single tenant with: HNSW manifest at watermark 50, follower
        // applied through 30, backup pinned at start 100. The min
        // of (51, 31, 100) is 31 → safe_purge_through = 30, binding
        // contributor = follower_lag.
        let store = Arc::new(open_manifest_store());
        let tenant = TenantId::new(1);
        let mut m = HnswManifest::new(tenant, "minilm", 384, DistanceMetric::Cosine);
        m.source_log_watermark = 50;
        store.upsert(&m).unwrap();

        let r = RetentionRegistry::new()
            .with(Arc::new(HnswWatermarkContributor::new(store)))
            .with(Arc::new(FollowerLagContributor::new(|_| Some(30))))
            .with(Arc::new(BackupWatermarkContributor::new(|_| Some(100))));

        let w = r.safe_purge_watermark(tenant).await.unwrap();
        assert_eq!(w.min_required, Some(31));
        assert_eq!(w.safe_purge_through, Some(30));
        assert_eq!(w.binding_contributor(), Some(("follower_lag", 31)));
    }

    #[tokio::test]
    async fn registry_returns_no_constraint_when_all_disabled() {
        // Single-node deployment with no follower, no in-flight
        // backup, no HNSW manifest yet, no tombstones. Compaction
        // should be unconstrained (purge whatever).
        let store = Arc::new(open_manifest_store());
        let r = RetentionRegistry::new()
            .with(Arc::new(HnswWatermarkContributor::new(store)))
            .with(Arc::new(FollowerLagContributor::disabled()))
            .with(Arc::new(BackupWatermarkContributor::disabled()));
        let w = r.safe_purge_watermark(TenantId::new(99)).await.unwrap();
        assert_eq!(w.safe_purge_through, None);
        assert_eq!(w.binding_contributor(), None);
    }
}
