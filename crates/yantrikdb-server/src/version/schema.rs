//! `SchemaVersion` — per-table SQLite schema version tracking.
//!
//! ## Two version concepts
//!
//! `SchemaVersion` is **distinct** from `WireVersion`:
//!
//! - `WireVersion` governs **inter-node** event compatibility
//! - `SchemaVersion` governs **intra-node** SQLite migrations
//!
//! A node at WireVersion 1.5 can receive an event written at WireVersion
//! 1.0 (same major). But applying that event might require its local
//! `memory_commit_log` table to be at SchemaVersion 3 or higher. If the
//! receiver is at SchemaVersion 2, it must run the migration before
//! applying.
//!
//! ## Per-table tracking
//!
//! Every durable table has its own monotonic `SchemaVersion`. This module
//! maintains [`TABLE_SCHEMA_VERSIONS`] as the source of truth — adding a
//! migration requires bumping the entry and writing a migration runner
//! step.
//!
//! ## Standing acceptance criterion (RFC 017-B)
//!
//! Every new durable table introduced by any subsequent RFC MUST:
//! 1. Add an entry to [`TABLE_SCHEMA_VERSIONS`]
//! 2. Include a `schema_version INTEGER NOT NULL` column on every row
//!    that's replicated or backed up (so receivers can validate)
//! 3. Document the migration path from previous versions in
//!    `docs/migration/<table>.md`

use std::fmt;

use serde::{Deserialize, Serialize};

use super::error::VersionError;

/// Per-table schema version. Monotonic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SchemaVersion(pub u32);

impl SchemaVersion {
    pub const fn new(v: u32) -> Self {
        Self(v)
    }

    /// Whether this node at `node_max` can apply an event tagged with
    /// `event_required` schema version. Old events on a new node are fine
    /// (we still know how to interpret them). New events on an old node
    /// are not.
    pub fn check_can_apply(
        node_max: SchemaVersion,
        event_required: SchemaVersion,
        table: &'static str,
    ) -> Result<(), VersionError> {
        if event_required > node_max {
            return Err(VersionError::SchemaTooNew {
                table,
                node_max,
                event_required,
            });
        }
        Ok(())
    }
}

impl fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl From<u32> for SchemaVersion {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<SchemaVersion> for u32 {
    fn from(s: SchemaVersion) -> Self {
        s.0
    }
}

/// The authoritative list of every replicated/durable table this binary
/// knows about, with the schema version it currently expects.
///
/// **Bumping rules**: when you add a migration, bump the entry. When you
/// add a new table, append it. Never remove an entry — old tables that no
/// longer exist still need their version recorded so a receiver replaying
/// historical events can interpret them.
///
/// Actual SQL migrations live in `crates/yantrikdb-server/src/migrations/`
/// (TODO: RFC 010 PR-2 introduces this directory). This constant is the
/// in-code source of truth that the migration runner verifies against.
pub const TABLE_SCHEMA_VERSIONS: &[(&str, SchemaVersion)] = &[
    // RFC 010 PR-2 will introduce this table; entry registered here so
    // 017-A stays the source-of-truth list. Until then, the migration
    // runner will skip tables not present in the local DB.
    ("memory_commit_log", SchemaVersion(1)),
    // RFC 009 PR-2 will introduce this table.
    ("quota_policies", SchemaVersion(1)),
    // RFC 009 PR-2 will introduce this table.
    ("quota_audit_log", SchemaVersion(1)),
    // RFC 011 will introduce this table.
    ("memory_tombstones", SchemaVersion(1)),
    // RFC 013 PR-1 will introduce this table.
    ("hnsw_manifests", SchemaVersion(1)),
    // RFC 012 PR-1 will introduce this table.
    ("backup_manifests", SchemaVersion(1)),
    // RFC 019 will introduce this table.
    ("durable_jobs", SchemaVersion(1)),
    // RFC 021 PR-2 will introduce this table.
    ("tenant_config_overrides", SchemaVersion(1)),
];

/// Look up the schema version this binary expects for a given table.
/// Returns None if the table is unknown to this binary.
pub fn expected_schema_version(table: &str) -> Option<SchemaVersion> {
    TABLE_SCHEMA_VERSIONS
        .iter()
        .find(|(name, _)| *name == table)
        .map(|(_, ver)| *ver)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_can_apply_accepts_same_or_older() {
        let node = SchemaVersion::new(5);
        assert!(SchemaVersion::check_can_apply(node, SchemaVersion::new(1), "t").is_ok());
        assert!(SchemaVersion::check_can_apply(node, SchemaVersion::new(5), "t").is_ok());
    }

    #[test]
    fn check_can_apply_rejects_newer_with_structured_error() {
        let node = SchemaVersion::new(5);
        let event = SchemaVersion::new(7);
        let err = SchemaVersion::check_can_apply(node, event, "memory_commit_log").unwrap_err();
        match err {
            VersionError::SchemaTooNew {
                table,
                node_max,
                event_required,
            } => {
                assert_eq!(table, "memory_commit_log");
                assert_eq!(node_max, node);
                assert_eq!(event_required, event);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn ordering_is_natural() {
        assert!(SchemaVersion::new(1) < SchemaVersion::new(2));
        assert!(SchemaVersion::new(99) < SchemaVersion::new(100));
    }

    #[test]
    fn expected_schema_version_finds_known_tables() {
        // The table list is a forward-looking registry; entries must
        // resolve. If you remove or rename a table, you MUST update this
        // test alongside the migration runner.
        assert!(expected_schema_version("memory_commit_log").is_some());
        assert!(expected_schema_version("quota_policies").is_some());
        assert!(expected_schema_version("hnsw_manifests").is_some());
    }

    #[test]
    fn expected_schema_version_returns_none_for_unknown() {
        assert_eq!(expected_schema_version("does_not_exist"), None);
    }

    #[test]
    fn table_names_in_registry_are_unique() {
        // Duplicate entries would create ambiguity about which version
        // is canonical. Pin this with a test.
        let mut seen = std::collections::HashSet::new();
        for (name, _) in TABLE_SCHEMA_VERSIONS {
            assert!(
                seen.insert(*name),
                "duplicate entry in TABLE_SCHEMA_VERSIONS: {name}"
            );
        }
    }

    #[test]
    fn serde_uses_compact_integer_form() {
        let v = SchemaVersion::new(42);
        let json = serde_json::to_string(&v).unwrap();
        // Transparent serde means the integer goes on the wire directly,
        // not wrapped in an object. Important for compact replicated logs.
        assert_eq!(json, "42");
    }
}
