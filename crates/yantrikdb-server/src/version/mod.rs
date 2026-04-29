//! RFC 017-A — Wire Version Framework primitives.
//!
//! ## Why this module exists
//!
//! Every durable event in YantrikDB (memory_commit_log entries, replicated
//! Raft log entries, snapshot manifests, backup manifests, schema migration
//! events) carries a *version*. Without versioning from day one, we get
//! locked into our v0.7.x format forever — any upgrade introduces a
//! "works only if every node is upgraded atomically" foot-gun.
//!
//! The cluster-thrash incident (term=1423, 2026-04-28) was the immediate
//! trigger for RFC 009 admission control. RFC 017-A is the *prerequisite*
//! for RFC 010 (commit substrate) — once we ship a memory_commit_log,
//! every event in it must be tagged with the version of code that wrote
//! it, so future code can replay it correctly.
//!
//! ## The two version concepts
//!
//! | Concept | Type | Scope | Bumps when |
//! |---|---|---|---|
//! | [`WireVersion`] | `(major: u8, minor: u16)` | replicated event format | major: incompatible change. minor: additive |
//! | [`SchemaVersion`] | `u32` | per-table SQLite schema | every migration |
//!
//! `WireVersion` governs **inter-node** compatibility (what events can be
//! exchanged across a cluster). `SchemaVersion` governs **intra-node**
//! compatibility (what migrations have run on this node's SQLite).
//!
//! These are independent: two nodes at different SchemaVersions can still
//! exchange events if their WireVersions are major-compatible — the
//! receiver applies any migrations it needs locally.
//!
//! ## Compatibility rules
//!
//! - **Same major WireVersion**: events are forward AND backward compatible.
//!   Adding new fields/variants is allowed; receivers ignore unknowns.
//! - **Different major WireVersion**: incompatible. Sender's events are
//!   rejected by receiver with a clear error and operator hint.
//! - **SchemaVersion newer than receiver**: event is queued or rejected
//!   depending on policy (see [`policy::VersionPolicy`]).
//!
//! ## What's in this module
//!
//! - [`wire`] — `WireVersion` type, current build constants, serialization
//! - [`schema`] — `SchemaVersion` per-table tracking
//! - [`gate`] — `VersionGate` runtime state for cluster-wide negotiation
//! - [`policy`] — decision logic: can_apply / can_use_feature
//! - [`error`] — typed errors with structured causes for clear operator messaging
//! - [`event`] — `VersionedEvent` trait every replicated thing implements
//!
//! ## What lands in subsequent RFCs
//!
//! - **RFC 010 PR-3** (`MemoryMutation` grammar): every variant tagged with
//!   `WireVersion` + `SchemaVersion`. The grammar enum becomes the first
//!   user of [`event::VersionedEvent`].
//! - **Standing acceptance criterion (017-B)**: every new durable table
//!   adds a `schema_version` column and registers a [`schema::SchemaVersion`]
//!   constant.
//!
//! ## Why `(major, minor)` instead of semver string
//!
//! - Compact serialization (3 bytes vs 5-30 char string)
//! - Stable comparison without string parsing
//! - Cannot accidentally introduce ambiguous "1.0.0-alpha" semantics
//! - The trade-off (no patch level) is correct: patches don't change
//!   wire compatibility by definition; if they do, they're a minor bump.

pub mod error;
pub mod event;
pub mod gate;
pub mod policy;
pub mod schema;
pub mod wire;

pub use error::VersionError;
pub use event::VersionedEvent;
pub use gate::VersionGate;
pub use policy::VersionPolicy;
pub use schema::{SchemaVersion, TABLE_SCHEMA_VERSIONS};
pub use wire::{WireVersion, CURRENT_WIRE_VERSION, MIN_SUPPORTED_WIRE_VERSION};

/// Convenience snapshot of the local node's full version state.
/// Surfaced via `/v1/health/deep` and `yantrikdb version status`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VersionSnapshot {
    pub wire: WireVersion,
    pub min_supported_wire: WireVersion,
    pub build_id: &'static str,
    pub table_schema_versions: Vec<(&'static str, SchemaVersion)>,
}

impl VersionSnapshot {
    pub fn local() -> Self {
        Self {
            wire: CURRENT_WIRE_VERSION,
            min_supported_wire: MIN_SUPPORTED_WIRE_VERSION,
            build_id: env!("CARGO_PKG_VERSION"),
            table_schema_versions: TABLE_SCHEMA_VERSIONS
                .iter()
                .map(|(name, ver)| (*name, *ver))
                .collect(),
        }
    }
}
