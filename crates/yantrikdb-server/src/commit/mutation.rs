//! `MemoryMutation` enum — the canonical grammar of every durable write.
//!
//! ## Why every variant exists from day one
//!
//! Per the gpt-5.5 redteam of RFC 010 (locked 2026-04-28): the log grammar
//! cannot change later without breaking replay. So `TombstoneMemory` and
//! `PurgeMemory` variants exist NOW even though RFC 011 hasn't implemented
//! their semantics yet. Until RFC 011, attempts to apply them return
//! [`super::trait_def::CommitError::NotYetImplemented`] — the grammar is
//! locked, the runtime behavior matures incrementally.
//!
//! ## Versioning
//!
//! Every variant implements [`crate::version::VersionedEvent`]:
//! - `wire_version` = 1.0 (initial release of the grammar)
//! - `schema_version` = each variant's target table from
//!   [`crate::version::TABLE_SCHEMA_VERSIONS`]
//!
//! When adding a new variant in a future minor (1.1+), bump the variant's
//! `wire_version` to 1.N and ensure receivers ignore unknown variants
//! gracefully (`#[serde(other)]` on the deserializer).
//!
//! ## Why per-variant fields are explicit instead of `serde_json::Value`
//!
//! Type-safe variants catch payload shape errors at compile time. The
//! serialization is still serde-driven, so adding a field to a variant
//! is a minor wire bump and old code reading the new payload just sees
//! the field as missing/default — no runtime parser failures.

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid7::Uuid;

use crate::version::{schema::SchemaVersion, wire::WireVersion, VersionedEvent};

/// UUIDv7 identifier for a single mutation. Time-ordered + globally
/// unique. Used for idempotency: the committer keeps a (tenant_id, op_id)
/// → log_index map and re-commits return the original receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpId(pub Uuid);

impl OpId {
    /// Generate a new UUIDv7. Use this on every fresh commit.
    pub fn new_random() -> Self {
        Self(uuid7::uuid7())
    }

    pub fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }

    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl fmt::Display for OpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Tenant identifier. Wraps `i64` (the SQLite control DB primary key)
/// for type safety — prevents accidentally swapping tenant_id with
/// log_index or any other integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TenantId(pub i64);

impl TenantId {
    pub const fn new(id: i64) -> Self {
        Self(id)
    }
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The canonical mutation grammar. Every durable write to YantrikDB
/// serializes as one of these variants and lands in the per-tenant
/// commit log.
///
/// **Forward-compatibility**: never remove or renumber variants. Add new
/// ones at the end. `#[serde(other)]` on a future deserializer-side
/// fallback will let old code skip unknown variants gracefully (the
/// receiver-side error budget).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum MemoryMutation {
    /// Insert or update a memory by rid. Used by `/v1/remember`.
    ///
    /// Wire 1.0 shape: `rid` … `metadata` (12 fields).
    ///
    /// Wire 1.1 (RFC 010 PR-6.2) added three materialized-state fields
    /// — `extracted_entities`, `created_at_unix_micros`, `embedding_model`
    /// — that let followers apply the mutation deterministically without
    /// re-running NER, the embedder, or wall-clock reads. All three carry
    /// `#[serde(default)]` so a v1.0 payload still round-trips through
    /// v1.1 code (the Applier handles the empty case by computing on the
    /// fly, exactly like the leader did at v1.0). Conversely, a v1.1
    /// payload deserialized by v1.0 code drops the unknown fields and
    /// keeps working — the variant SHAPE didn't change at the wire-major
    /// boundary, only its information content grew.
    UpsertMemory {
        rid: String,
        text: String,
        memory_type: String,
        importance: f64,
        valence: f64,
        half_life: f64,
        namespace: String,
        certainty: f64,
        domain: String,
        source: String,
        emotional_state: Option<String>,
        /// Server may compute or accept client-provided. Stored as part
        /// of the mutation so replay re-creates the same vector.
        ///
        /// Wire 1.1: when a leader produces a materialized mutation
        /// destined for the commit log, this MUST be `Some`. PR 6.4
        /// enforces that at materialization time. v1.0 payloads still
        /// permit `None` for backwards-compat with the unmigrated
        /// handler path.
        embedding: Option<Vec<f32>>,
        metadata: serde_json::Value,

        /// Entities extracted by the leader's NER pass at materialization
        /// time. Stored on the mutation so followers don't re-run NER
        /// (which would diverge if NER models drift between nodes).
        ///
        /// Wire 1.1+. Empty default for v1.0 payloads — the Applier
        /// treats empty as "compute on the fly" until the leader
        /// migrates its handlers (PR 6.4).
        #[serde(default)]
        extracted_entities: Vec<String>,

        /// Server-assigned creation timestamp, microseconds since UNIX
        /// epoch. The leader stamps this once at materialization time;
        /// followers use the stamped value rather than reading their
        /// own wall clock.
        ///
        /// Wire 1.1+. `None` for v1.0 payloads — the Applier falls back
        /// to its local clock until the leader migrates its handlers.
        #[serde(default)]
        created_at_unix_micros: Option<i64>,

        /// Identifier for the embedding model that produced
        /// [`Self::UpsertMemory::embedding`]. Used by RFC 013 model
        /// migration to know which embeddings need re-encoding when
        /// the cluster upgrades its embedder.
        ///
        /// Wire 1.1+. `None` for v1.0 payloads — RFC 013 falls back to
        /// the cluster's default model identifier.
        #[serde(default)]
        embedding_model: Option<String>,
    },

    /// Patch fields on an existing memory by rid. Used by `/v1/correct`.
    /// Fields not in the patch are left as-is.
    UpdateMemoryPatch {
        rid: String,
        text: Option<String>,
        importance: Option<f64>,
        valence: Option<f64>,
        certainty: Option<f64>,
        metadata_patch: Option<serde_json::Value>,
    },

    /// Mark a memory as forgotten. Logically not-recallable; physical
    /// purge happens via [`Self::PurgeMemory`] later (RFC 011 PR-3).
    /// Tombstones must propagate to all replicas + caches before purge.
    ///
    /// Wire 1.2 (RFC 010 PR-6.4) added `namespace`. Engine
    /// `tombstone_with_rid` requires the namespace explicitly because
    /// followers must bump `visible_seq[namespace]` even when the local
    /// rid is missing (snapshot-lag case) — without the namespace
    /// stamped on the mutation, a snapshot-lagging follower can't
    /// route the visible_seq bump to the right per-namespace counter.
    TombstoneMemory {
        rid: String,
        reason: Option<String>,
        requested_at_unix_micros: i64,
        /// Wire 1.2+. Empty string for v1.0 / v1.1 payloads — Applier
        /// emits a trace-level info log when this happens (legacy log
        /// bleed-through during snapshot install) and falls back to
        /// looking up the namespace from the existing memories row.
        #[serde(default)]
        namespace: String,
    },

    /// Physically remove a memory and its derived state (HNSW node,
    /// embedding cache entry, etc.) after a tombstone has propagated
    /// past the safe-purge watermark. Not implemented in PR-1; rejected
    /// at apply time with `CommitError::NotYetImplemented` until RFC
    /// 011 PR-3 ships the delete-queue worker.
    PurgeMemory { rid: String, purge_epoch: u64 },

    /// Add or update an entity edge. Used by `/v1/relate`.
    UpsertEntityEdge {
        edge_id: String,
        src: String,
        dst: String,
        rel_type: String,
        weight: f64,
        namespace: String,
    },

    /// Remove an entity edge.
    ///
    /// Wire 1.2 (RFC 010 PR-6.4) added `namespace` and
    /// `requested_at_unix_micros` to match
    /// `engine::delete_entity_edge_with_id` semantics. Same snapshot-lag
    /// rationale as [`Self::TombstoneMemory`] — followers need the
    /// namespace to bump the right visible_seq counter.
    DeleteEntityEdge {
        edge_id: String,
        /// Wire 1.2+. Empty string for v1.0 / v1.1 payloads.
        #[serde(default)]
        namespace: String,
        /// Wire 1.2+. Defaults to 0 for legacy payloads — engine treats
        /// 0 as "use current wall clock" during apply.
        #[serde(default)]
        requested_at_unix_micros: i64,
    },

    /// Patch tenant-level config (e.g. RFC 021 per-tenant overrides).
    /// Variant exists from day one so RFC 021 doesn't need a wire bump.
    /// Until RFC 021 ships, applying this returns NotYetImplemented.
    TenantConfigPatch {
        key: String,
        value: serde_json::Value,
    },
}

impl MemoryMutation {
    /// Stable identifier for metrics + logs. Never change these strings.
    pub fn variant_name(&self) -> &'static str {
        match self {
            MemoryMutation::UpsertMemory { .. } => "UpsertMemory",
            MemoryMutation::UpdateMemoryPatch { .. } => "UpdateMemoryPatch",
            MemoryMutation::TombstoneMemory { .. } => "TombstoneMemory",
            MemoryMutation::PurgeMemory { .. } => "PurgeMemory",
            MemoryMutation::UpsertEntityEdge { .. } => "UpsertEntityEdge",
            MemoryMutation::DeleteEntityEdge { .. } => "DeleteEntityEdge",
            MemoryMutation::TenantConfigPatch { .. } => "TenantConfigPatch",
        }
    }

    /// Whether this variant is implemented yet. Variants exist in the
    /// grammar from day one (so the log can record them safely), but
    /// the runtime apply path may not be ready. `false` here causes
    /// `CommitError::NotYetImplemented` at apply time.
    pub fn is_implemented(&self) -> bool {
        match self {
            // PR-1 implements UpsertMemory + UpdateMemoryPatch +
            // UpsertEntityEdge + DeleteEntityEdge enough to round-trip
            // through the in-memory committer. Real engine integration
            // happens in PR-2+.
            MemoryMutation::UpsertMemory { .. }
            | MemoryMutation::UpdateMemoryPatch { .. }
            | MemoryMutation::UpsertEntityEdge { .. }
            | MemoryMutation::DeleteEntityEdge { .. } => true,

            // TombstoneMemory apply path ships in RFC 011 PR-2 (this PR).
            MemoryMutation::TombstoneMemory { .. } => true,
            // PurgeMemory ships in RFC 011 PR-3.
            MemoryMutation::PurgeMemory { .. } => false,
            // TenantConfigPatch ships in RFC 021 PR-2.
            MemoryMutation::TenantConfigPatch { .. } => false,
        }
    }

    /// RFC where this variant's full apply semantics land. Used in
    /// NotYetImplemented error messages so operators know when to expect
    /// support.
    pub fn planned_rfc(&self) -> &'static str {
        match self {
            MemoryMutation::TombstoneMemory { .. } => "011",
            MemoryMutation::PurgeMemory { .. } => "011",
            MemoryMutation::TenantConfigPatch { .. } => "021",
            // Implemented variants don't have a planned_rfc — return
            // a sentinel for symmetry. Should never appear in user-
            // visible messages because is_implemented() == true.
            _ => "010",
        }
    }

    /// True if this mutation logically tombstones a memory. The committer
    /// uses this to know when to update an attached
    /// [`crate::forget::TombstoneIndex`] after a successful commit.
    pub fn tombstoned_rid(&self) -> Option<&str> {
        match self {
            MemoryMutation::TombstoneMemory { rid, .. } => Some(rid),
            _ => None,
        }
    }

    /// True if this mutation finalizes a purge (the rid is physically
    /// gone). The committer uses this to clear the rid from the attached
    /// [`crate::forget::TombstoneIndex`].
    pub fn purged_rid(&self) -> Option<&str> {
        match self {
            MemoryMutation::PurgeMemory { rid, .. } => Some(rid),
            _ => None,
        }
    }

    /// Cluster-wide feature flag name for this variant. Used by the
    /// version gate (RFC 017-A) to refuse emitting a variant when any
    /// peer is too old to receive it. Variants registered in
    /// [`crate::version::gate::FEATURE_FLOORS`] with their minimum
    /// cluster wire version.
    ///
    /// Stability: these strings are part of the operator-facing surface
    /// (logs, metrics, error messages). Do not rename.
    pub fn feature_flag(&self) -> &'static str {
        match self {
            MemoryMutation::UpsertMemory { .. } => "mutation.UpsertMemory",
            MemoryMutation::UpdateMemoryPatch { .. } => "mutation.UpdateMemoryPatch",
            MemoryMutation::TombstoneMemory { .. } => "mutation.TombstoneMemory",
            MemoryMutation::PurgeMemory { .. } => "mutation.PurgeMemory",
            MemoryMutation::UpsertEntityEdge { .. } => "mutation.UpsertEntityEdge",
            MemoryMutation::DeleteEntityEdge { .. } => "mutation.DeleteEntityEdge",
            MemoryMutation::TenantConfigPatch { .. } => "mutation.TenantConfigPatch",
        }
    }

    /// Wire version this variant was introduced at. Each variant pinned
    /// to the minor it shipped in. Future variants added in 1.1+ would
    /// declare 1.1 here so older peers know to fail-fast on receive
    /// (rather than silently mis-deserializing).
    ///
    /// **Locked at v1.0**: every variant in the initial grammar is at
    /// 1.0. Adding a new variant later means bumping CURRENT_WIRE_VERSION
    /// to (1, N) AND declaring `wire_introduced_at_constant` here for
    /// the new variant. Existing variants stay at 1.0 forever.
    pub fn wire_introduced_at(&self) -> WireVersion {
        // All initial variants ship at 1.0. Pinned for forward-compat:
        // when a v1.1 ships with new variants, this match adds new arms
        // returning WireVersion::new(1, 1) but the existing arms stay
        // at WireVersion::new(1, 0).
        WireVersion::new(1, 0)
    }
}

impl VersionedEvent for MemoryMutation {
    fn wire_version(&self) -> WireVersion {
        // Per-variant: each variant returns the wire version it was
        // introduced at. Initial grammar is all 1.0; future minor
        // additions declare their introduction version via the match
        // in `wire_introduced_at`.
        self.wire_introduced_at()
    }

    fn schema_version(&self) -> Option<(&'static str, SchemaVersion)> {
        // Each variant targets a specific table from the registry in
        // crate::version::schema::TABLE_SCHEMA_VERSIONS.
        match self {
            MemoryMutation::UpsertMemory { .. }
            | MemoryMutation::UpdateMemoryPatch { .. }
            | MemoryMutation::TombstoneMemory { .. }
            | MemoryMutation::PurgeMemory { .. } => {
                // All memory-table mutations land in memory_commit_log.
                Some(("memory_commit_log", SchemaVersion::new(1)))
            }
            MemoryMutation::UpsertEntityEdge { .. } | MemoryMutation::DeleteEntityEdge { .. } => {
                // Entity edges share the commit log; the underlying table
                // is the engine's edges/claims table, but the commit log
                // is the durable record.
                Some(("memory_commit_log", SchemaVersion::new(1)))
            }
            MemoryMutation::TenantConfigPatch { .. } => {
                Some(("tenant_config_overrides", SchemaVersion::new(1)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upsert_example() -> MemoryMutation {
        MemoryMutation::UpsertMemory {
            rid: "mem_test_1".into(),
            text: "the cat sat on the mat".into(),
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

    #[test]
    fn op_id_is_unique_each_call() {
        let a = OpId::new_random();
        let b = OpId::new_random();
        assert_ne!(a, b, "uuid7 collision is astronomically unlikely");
    }

    #[test]
    fn op_id_serde_is_transparent_string() {
        let id = OpId::new_random();
        let json = serde_json::to_string(&id).unwrap();
        // Should be a JSON string, not an object with a wrapper field.
        assert!(json.starts_with('"') && json.ends_with('"'));
        let back: OpId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn tenant_id_serde_is_transparent_integer() {
        let t = TenantId::new(42);
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, "42");
        let back: TenantId = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn variant_names_are_stable_and_unique() {
        // Stability: dashboards/metrics key on these strings. Pin them.
        let muts = vec![
            upsert_example(),
            MemoryMutation::UpdateMemoryPatch {
                rid: "x".into(),
                text: None,
                importance: None,
                valence: None,
                certainty: None,
                metadata_patch: None,
            },
            MemoryMutation::TombstoneMemory {
                rid: "x".into(),
                reason: None,
                requested_at_unix_micros: 0,
                namespace: String::new(),
            },
            MemoryMutation::PurgeMemory {
                rid: "x".into(),
                purge_epoch: 0,
            },
            MemoryMutation::UpsertEntityEdge {
                edge_id: "e".into(),
                src: "a".into(),
                dst: "b".into(),
                rel_type: "rel".into(),
                weight: 1.0,
                namespace: "default".into(),
            },
            MemoryMutation::DeleteEntityEdge {
                edge_id: "e".into(),
                namespace: String::new(),
                requested_at_unix_micros: 0,
            },
            MemoryMutation::TenantConfigPatch {
                key: "k".into(),
                value: serde_json::Value::Null,
            },
        ];
        let names: Vec<_> = muts.iter().map(|m| m.variant_name()).collect();
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            assert!(seen.insert(*name), "duplicate variant_name: {name}");
        }
        // Pin specific stable values:
        assert_eq!(muts[0].variant_name(), "UpsertMemory");
        assert_eq!(muts[2].variant_name(), "TombstoneMemory");
        assert_eq!(muts[3].variant_name(), "PurgeMemory");
    }

    #[test]
    fn implementation_status_matches_rfc_plan() {
        // Implemented through RFC 011-B (this PR): Upsert/UpdatePatch/
        // UpsertEdge/DeleteEdge/TombstoneMemory.
        assert!(upsert_example().is_implemented());
        assert!(MemoryMutation::DeleteEntityEdge {
            edge_id: "e".into(),
            namespace: String::new(),
            requested_at_unix_micros: 0
        }
        .is_implemented());
        assert!(MemoryMutation::TombstoneMemory {
            rid: "x".into(),
            reason: None,
            requested_at_unix_micros: 0,
            namespace: String::new(),
        }
        .is_implemented());

        // Grammar-only — apply path lands in later PRs:
        assert!(!MemoryMutation::PurgeMemory {
            rid: "x".into(),
            purge_epoch: 0,
        }
        .is_implemented());
        assert!(!MemoryMutation::TenantConfigPatch {
            key: "k".into(),
            value: serde_json::Value::Null,
        }
        .is_implemented());
    }

    #[test]
    fn tombstoned_rid_extracts_only_for_tombstone_variant() {
        let t = MemoryMutation::TombstoneMemory {
            rid: "mem_a".into(),
            reason: None,
            requested_at_unix_micros: 0,
            namespace: String::new(),
        };
        assert_eq!(t.tombstoned_rid(), Some("mem_a"));
        assert_eq!(t.purged_rid(), None);

        let p = MemoryMutation::PurgeMemory {
            rid: "mem_b".into(),
            purge_epoch: 1,
        };
        assert_eq!(p.purged_rid(), Some("mem_b"));
        assert_eq!(p.tombstoned_rid(), None);

        // Other variants return None for both.
        assert_eq!(upsert_example().tombstoned_rid(), None);
        assert_eq!(upsert_example().purged_rid(), None);
    }

    #[test]
    fn planned_rfc_points_to_correct_followup() {
        assert_eq!(
            MemoryMutation::TombstoneMemory {
                rid: "x".into(),
                reason: None,
                requested_at_unix_micros: 0,
                namespace: String::new(),
            }
            .planned_rfc(),
            "011"
        );
        assert_eq!(
            MemoryMutation::PurgeMemory {
                rid: "x".into(),
                purge_epoch: 0,
            }
            .planned_rfc(),
            "011"
        );
        assert_eq!(
            MemoryMutation::TenantConfigPatch {
                key: "k".into(),
                value: serde_json::Value::Null,
            }
            .planned_rfc(),
            "021"
        );
    }

    #[test]
    fn versioned_event_returns_wire_1_0() {
        let m = upsert_example();
        let wv = m.wire_version();
        assert_eq!(wv.major, 1);
        assert_eq!(wv.minor, 0);
    }

    #[test]
    fn versioned_event_schema_targets_correct_table() {
        let m = upsert_example();
        let (table, ver) = m.schema_version().unwrap();
        assert_eq!(table, "memory_commit_log");
        assert_eq!(ver, SchemaVersion::new(1));

        let cfg = MemoryMutation::TenantConfigPatch {
            key: "k".into(),
            value: serde_json::Value::Null,
        };
        let (table, ver) = cfg.schema_version().unwrap();
        assert_eq!(table, "tenant_config_overrides");
        assert_eq!(ver, SchemaVersion::new(1));
    }

    #[test]
    fn mutation_serde_round_trip_is_lossless() {
        let cases = vec![
            upsert_example(),
            MemoryMutation::TombstoneMemory {
                rid: "mem_42".into(),
                reason: Some("user requested".into()),
                requested_at_unix_micros: 1_700_000_000_000_000,
                namespace: String::new(),
            },
            MemoryMutation::UpsertEntityEdge {
                edge_id: "edge_7".into(),
                src: "alice".into(),
                dst: "bob".into(),
                rel_type: "knows".into(),
                weight: 0.9,
                namespace: "default".into(),
            },
        ];
        for m in cases {
            let json = serde_json::to_string(&m).unwrap();
            let back: MemoryMutation = serde_json::from_str(&json).unwrap();
            assert_eq!(m, back);
        }
    }

    #[test]
    fn tagged_serde_includes_kind_field() {
        let m = MemoryMutation::TombstoneMemory {
            rid: "x".into(),
            reason: None,
            requested_at_unix_micros: 0,
            namespace: String::new(),
        };
        let json = serde_json::to_string(&m).unwrap();
        // tag = "kind" means the wire format is self-describing.
        // Operators reading the commit log directly can identify variants.
        assert!(
            json.contains("\"kind\":\"TombstoneMemory\""),
            "missing tag: {json}"
        );
    }
}
