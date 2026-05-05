//! `WireVersion` — protocol version of replicated/persisted events.
//!
//! ## Compatibility rule
//!
//! Two `WireVersion` values are compatible iff they share the same `major`.
//! Within a major, `minor` increments are additive (new fields, new event
//! variants). Across majors, events MUST NOT be exchanged.
//!
//! ## Build-time constants
//!
//! - [`CURRENT_WIRE_VERSION`]: what this binary writes
//! - [`MIN_SUPPORTED_WIRE_VERSION`]: oldest version this binary can replay
//!
//! When you make a backwards-incompatible change to event format, bump
//! `major` AND raise `MIN_SUPPORTED_WIRE_VERSION` to the new major. When
//! you add a field/variant, bump `minor` only.
//!
//! ## Why `u8` major and `u16` minor
//!
//! - 256 majors covers a thousand years at 1 major/year — sufficient
//! - 65k minors covers ~50 years at 1 minor/day — sufficient
//! - Small fixed encoding (3 bytes total)
//! - Forces real reasons to bump major (it's a precious resource)

use std::cmp::Ordering;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::error::VersionError;

/// Protocol version of replicated/persisted events.
///
/// See module-level docs for compatibility rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WireVersion {
    pub major: u8,
    pub minor: u16,
}

impl WireVersion {
    /// Construct a wire version from major/minor.
    pub const fn new(major: u8, minor: u16) -> Self {
        Self { major, minor }
    }

    /// Whether two versions can exchange events (same major).
    pub const fn is_compatible_with(&self, other: WireVersion) -> bool {
        self.major == other.major
    }

    /// Whether this node, at version `self`, can replay an event written
    /// at version `event`. Returns Ok if compatible, structured error otherwise.
    pub fn check_can_replay(self, event: WireVersion) -> Result<(), VersionError> {
        if self.major != event.major {
            return Err(VersionError::WireMajorMismatch { node: self, event });
        }
        // Within the same major, minor differences are forward-compatible
        // (older minor reads newer minor's events, ignoring unknown
        // additive fields) AND backward-compatible (newer minor reads
        // older). The application layer is responsible for being permissive
        // on unknown fields — see `serde(default)` and `#[serde(other)]`
        // patterns in event grammars.
        Ok(())
    }
}

/// Total order on wire versions. Used for finding cluster-min/max during
/// negotiation. Major dominates minor.
impl Ord for WireVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        self.major
            .cmp(&other.major)
            .then_with(|| self.minor.cmp(&other.minor))
    }
}

impl PartialOrd for WireVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for WireVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// What this binary writes for new events.
///
/// **Bump rules** (write a release-notes section when you change this):
/// - **Bump `minor`** when adding a field to an existing event variant,
///   adding a new event variant, adding a new column to a tracked table.
///   Old code can still process new events; new code can still process
///   old events.
/// - **Bump `major` AND `MIN_SUPPORTED_WIRE_VERSION`** when changing field
///   semantics, removing a variant, renaming a field, changing
///   serialization. Major bumps require a documented migration path:
///   nodes upgrade through an intermediate "bridge" version that knows
///   both formats, until the cluster-min reaches the new major.
pub const CURRENT_WIRE_VERSION: WireVersion = WireVersion::new(1, 1);

/// Oldest wire version this binary can replay.
///
/// Events written at versions < this will be rejected at apply time with
/// [`VersionError::WireMajorMismatch`]. For RFC 010 onwards this matters
/// because the commit log MAY contain events older than the current build.
///
/// Currently `MIN_SUPPORTED_WIRE_VERSION == CURRENT_WIRE_VERSION` because
/// no v0.x events exist with this versioning scheme yet — RFC 017-A is
/// the start. Once we hit v2.0 of the wire format, this stays at 1.0
/// for at least one minor cycle so operators have a runway to migrate.
pub const MIN_SUPPORTED_WIRE_VERSION: WireVersion = WireVersion::new(1, 0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_is_lexicographic_major_then_minor() {
        assert!(WireVersion::new(1, 0) < WireVersion::new(1, 1));
        assert!(WireVersion::new(1, 99) < WireVersion::new(2, 0));
        assert!(WireVersion::new(2, 0) > WireVersion::new(1, 999));
    }

    #[test]
    fn compatibility_is_major_only() {
        assert!(WireVersion::new(1, 0).is_compatible_with(WireVersion::new(1, 50)));
        assert!(WireVersion::new(1, 50).is_compatible_with(WireVersion::new(1, 0)));
        assert!(!WireVersion::new(1, 50).is_compatible_with(WireVersion::new(2, 0)));
        assert!(!WireVersion::new(2, 0).is_compatible_with(WireVersion::new(1, 50)));
    }

    #[test]
    fn check_can_replay_accepts_same_major() {
        let node = WireVersion::new(1, 5);
        assert!(node.check_can_replay(WireVersion::new(1, 0)).is_ok());
        assert!(node.check_can_replay(WireVersion::new(1, 99)).is_ok());
    }

    #[test]
    fn check_can_replay_rejects_different_major_with_structured_error() {
        let node = WireVersion::new(1, 5);
        let event = WireVersion::new(2, 0);
        let err = node.check_can_replay(event).unwrap_err();
        match err {
            VersionError::WireMajorMismatch { node: n, event: e } => {
                assert_eq!(n, node);
                assert_eq!(e, event);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn display_uses_dotted_form() {
        assert_eq!(format!("{}", WireVersion::new(1, 5)), "1.5");
        assert_eq!(format!("{}", WireVersion::new(0, 0)), "0.0");
    }

    #[test]
    fn current_is_at_least_min_supported() {
        // Invariant: a node must be able to replay events it has just written.
        assert!(CURRENT_WIRE_VERSION >= MIN_SUPPORTED_WIRE_VERSION);
        assert!(CURRENT_WIRE_VERSION.is_compatible_with(MIN_SUPPORTED_WIRE_VERSION));
    }

    #[test]
    fn serde_round_trip_compact() {
        let v = WireVersion::new(3, 42);
        let json = serde_json::to_string(&v).unwrap();
        let back: WireVersion = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }
}
