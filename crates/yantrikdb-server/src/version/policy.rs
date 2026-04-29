//! `VersionPolicy` — decision logic combining wire + schema + gate state.
//!
//! Most callers don't need to consult `WireVersion`, `SchemaVersion`,
//! and `VersionGate` directly. They just want one of two questions
//! answered:
//!
//! 1. **Can this node apply the event I'm about to receive?**
//!    → [`VersionPolicy::can_apply`]
//! 2. **Can my cluster safely emit this new event?**
//!    → [`VersionPolicy::can_emit`]
//!
//! The policy implements both as static methods over a [`VersionGate`]
//! plus the event's [`VersionedEvent`] data. Failures are typed
//! [`VersionError`]s with operator-actionable messages.
//!
//! ## Apply-side flow (receiver)
//!
//! ```text
//! event arrives → can_apply(event, gate, table_versions)
//!                  ├── check wire major matches local
//!                  └── check event's schema_version <= local table version
//!                 → Ok or VersionError
//! ```
//!
//! ## Emit-side flow (sender / writer)
//!
//! ```text
//! about to write → can_emit(feature_id, gate)
//!                  ├── check feature has known floor
//!                  └── check cluster_min >= floor
//!                 → Ok or VersionError::ClusterTooOld | FeatureGated
//! ```

use super::error::VersionError;
use super::event::VersionedEvent;
use super::gate::VersionGate;
use super::schema::{expected_schema_version, SchemaVersion};

pub struct VersionPolicy;

impl VersionPolicy {
    /// Whether this node can apply `event`. Both wire compatibility and
    /// schema version must check out.
    pub fn can_apply<E: VersionedEvent>(event: &E, gate: &VersionGate) -> Result<(), VersionError> {
        // Step 1: wire major must match local.
        gate.local_wire().check_can_replay(event.wire_version())?;

        // Step 2: if the event references a table, our schema must be at
        // least as new as the event needs.
        if let Some((table, required)) = event.schema_version() {
            let local_max = expected_schema_version(table).unwrap_or(SchemaVersion::new(0));
            SchemaVersion::check_can_apply(local_max, required, table)?;
        }

        Ok(())
    }

    /// Whether this cluster can safely emit a new event of `feature_id`.
    /// Used by writers BEFORE serializing — a refused emit is preferable
    /// to a replicated event that some peers can't handle.
    pub fn can_emit(feature_id: &'static str, gate: &VersionGate) -> Result<(), VersionError> {
        gate.can_use_feature(feature_id)
    }

    /// Hybrid check: an event was constructed locally with the given
    /// wire version, and we want to confirm the cluster can take it.
    /// Stricter than [`Self::can_emit`] because it inspects the event's
    /// declared version directly rather than just feature gating.
    pub fn can_emit_at_version<E: VersionedEvent>(
        event: &E,
        gate: &VersionGate,
    ) -> Result<(), VersionError> {
        let cluster_min = gate.cluster_min();
        if event.wire_version() > cluster_min {
            return Err(VersionError::ClusterTooOld {
                event: event.wire_version(),
                cluster_min,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::version::wire::WireVersion;

    /// Minimal event for tests. RFC 010 PR-3 will provide the real one.
    struct TestEvent {
        wire: WireVersion,
        schema: Option<(&'static str, SchemaVersion)>,
    }

    impl VersionedEvent for TestEvent {
        fn wire_version(&self) -> WireVersion {
            self.wire
        }
        fn schema_version(&self) -> Option<(&'static str, SchemaVersion)> {
            self.schema
        }
    }

    #[test]
    fn can_apply_accepts_compatible_event() {
        let gate = VersionGate::new(WireVersion::new(1, 5));
        let event = TestEvent {
            wire: WireVersion::new(1, 0),
            schema: Some(("memory_commit_log", SchemaVersion::new(1))),
        };
        assert!(VersionPolicy::can_apply(&event, &gate).is_ok());
    }

    #[test]
    fn can_apply_rejects_wire_major_mismatch() {
        let gate = VersionGate::new(WireVersion::new(1, 5));
        let event = TestEvent {
            wire: WireVersion::new(2, 0),
            schema: None,
        };
        let err = VersionPolicy::can_apply(&event, &gate).unwrap_err();
        assert!(matches!(err, VersionError::WireMajorMismatch { .. }));
    }

    #[test]
    fn can_apply_rejects_schema_too_new() {
        let gate = VersionGate::new(WireVersion::new(1, 0));
        // The registry has memory_commit_log at v1, but our event claims
        // it needs v999 — reject.
        let event = TestEvent {
            wire: WireVersion::new(1, 0),
            schema: Some(("memory_commit_log", SchemaVersion::new(999))),
        };
        let err = VersionPolicy::can_apply(&event, &gate).unwrap_err();
        match err {
            VersionError::SchemaTooNew {
                table,
                node_max,
                event_required,
            } => {
                assert_eq!(table, "memory_commit_log");
                assert_eq!(node_max, SchemaVersion::new(1));
                assert_eq!(event_required, SchemaVersion::new(999));
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn can_apply_event_without_schema_skips_schema_check() {
        let gate = VersionGate::new(WireVersion::new(1, 0));
        let event = TestEvent {
            wire: WireVersion::new(1, 0),
            schema: None,
        };
        assert!(VersionPolicy::can_apply(&event, &gate).is_ok());
    }

    #[test]
    fn can_emit_at_version_rejects_when_cluster_too_old() {
        let gate = VersionGate::new(WireVersion::new(1, 5));
        gate.observe_peer(2, WireVersion::new(1, 0));
        // Cluster min is 1.0; we're trying to emit a 1.5-versioned event.
        let event = TestEvent {
            wire: WireVersion::new(1, 5),
            schema: None,
        };
        let err = VersionPolicy::can_emit_at_version(&event, &gate).unwrap_err();
        match err {
            VersionError::ClusterTooOld { event: e, cluster_min } => {
                assert_eq!(e, WireVersion::new(1, 5));
                assert_eq!(cluster_min, WireVersion::new(1, 0));
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn can_emit_at_version_accepts_when_cluster_caught_up() {
        let gate = VersionGate::new(WireVersion::new(1, 5));
        gate.observe_peer(2, WireVersion::new(1, 5));
        let event = TestEvent {
            wire: WireVersion::new(1, 5),
            schema: None,
        };
        assert!(VersionPolicy::can_emit_at_version(&event, &gate).is_ok());
    }

    #[test]
    fn can_emit_feature_check_uses_gate_floor() {
        let gate = VersionGate::new(WireVersion::new(1, 5));
        // baseline_v1_0 needs 1.0; cluster min == local == 1.5, fine.
        assert!(VersionPolicy::can_emit("baseline_v1_0", &gate).is_ok());
    }
}
