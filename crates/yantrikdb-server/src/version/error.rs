//! Typed errors for version compatibility checks.
//!
//! All variants carry structured fields so callers can render operator-
//! actionable messages (HTTP 400/409 with hints, log lines that name the
//! offending node and event, metrics labels that don't leak user data).

use thiserror::Error;

use super::schema::SchemaVersion;
use super::wire::WireVersion;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VersionError {
    /// Event's wire major version doesn't match this node's. The two
    /// nodes cannot exchange events at this format.
    ///
    /// **Operator hint**: upgrade the older node to a build whose major
    /// matches, or downgrade the newer node. The cluster CANNOT operate
    /// in mixed-major mode — events are either rejected or apply
    /// incorrectly.
    #[error(
        "wire major version mismatch: this node speaks {node}, event was written at {event}. \
         Cluster cannot operate in mixed-major mode. Upgrade the older node or contact ops."
    )]
    WireMajorMismatch {
        node: WireVersion,
        event: WireVersion,
    },

    /// Event references a SchemaVersion this node doesn't yet know about.
    ///
    /// **Operator hint**: this node needs to run schema migration to
    /// `event_required` before it can apply this event. The migration
    /// runner usually does this at startup; if you see this error mid-run,
    /// it means another node has migrated past this one in a rolling
    /// upgrade — pause writes until the lagging node catches up.
    #[error(
        "schema version too new for table `{table}`: this node has v{node_max}, \
         event requires v{event_required}. Run pending migrations or wait for them."
    )]
    SchemaTooNew {
        table: &'static str,
        node_max: SchemaVersion,
        event_required: SchemaVersion,
    },

    /// A new feature was used by a writer, but the cluster's negotiated
    /// minimum WireVersion doesn't support it yet — there's at least one
    /// node in the cluster still running an older build.
    ///
    /// **Operator hint**: upgrade the lagging nodes (visible in
    /// `yantrikdb cluster status`), then retry. The `cluster_min_wire`
    /// gauge in /metrics tracks the current floor.
    #[error(
        "feature `{feature}` requires cluster min wire version {requires}, \
         but cluster currently negotiates min {cluster_min}. \
         Upgrade lagging nodes before using this feature."
    )]
    FeatureGated {
        feature: &'static str,
        requires: WireVersion,
        cluster_min: WireVersion,
    },

    /// A new event was written at a version higher than what some other
    /// nodes in the cluster support. Sender side caught this before
    /// emitting; the event was NOT replicated.
    ///
    /// **Operator hint**: this is the sender-side guard. If you see this
    /// in logs, it means the writer correctly refused to break the cluster.
    /// Upgrade the lagging nodes and retry the operation.
    #[error(
        "cluster has nodes too old: write would emit event at {event}, \
         but cluster minimum is {cluster_min}. Write rejected for safety."
    )]
    ClusterTooOld {
        event: WireVersion,
        cluster_min: WireVersion,
    },
}

impl VersionError {
    /// Stable label for metrics. Never includes user data — safe for
    /// Prometheus cardinality.
    pub fn metric_label(&self) -> &'static str {
        match self {
            VersionError::WireMajorMismatch { .. } => "wire_major_mismatch",
            VersionError::SchemaTooNew { .. } => "schema_too_new",
            VersionError::FeatureGated { .. } => "feature_gated",
            VersionError::ClusterTooOld { .. } => "cluster_too_old",
        }
    }

    /// HTTP status code an API caller should receive for this error.
    /// All version errors are 409 Conflict — the request is well-formed
    /// but conflicts with the cluster's current capability state.
    pub fn http_status(&self) -> u16 {
        409
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_labels_are_stable() {
        // Dashboards key on these strings. Pin them.
        assert_eq!(
            VersionError::WireMajorMismatch {
                node: WireVersion::new(1, 0),
                event: WireVersion::new(2, 0),
            }
            .metric_label(),
            "wire_major_mismatch"
        );
        assert_eq!(
            VersionError::SchemaTooNew {
                table: "x",
                node_max: SchemaVersion::new(1),
                event_required: SchemaVersion::new(2),
            }
            .metric_label(),
            "schema_too_new"
        );
        assert_eq!(
            VersionError::FeatureGated {
                feature: "f",
                requires: WireVersion::new(1, 1),
                cluster_min: WireVersion::new(1, 0),
            }
            .metric_label(),
            "feature_gated"
        );
        assert_eq!(
            VersionError::ClusterTooOld {
                event: WireVersion::new(1, 5),
                cluster_min: WireVersion::new(1, 0),
            }
            .metric_label(),
            "cluster_too_old"
        );
    }

    #[test]
    fn display_messages_include_offending_versions() {
        // Operator-facing messages must name what's incompatible so they
        // can act without reading code. Pin the format.
        let e = VersionError::WireMajorMismatch {
            node: WireVersion::new(1, 5),
            event: WireVersion::new(2, 0),
        };
        let s = format!("{e}");
        assert!(s.contains("1.5"));
        assert!(s.contains("2.0"));
        assert!(s.contains("mixed-major"));
    }

    #[test]
    fn all_version_errors_map_to_409() {
        // 409 Conflict is the right HTTP code for "well-formed but
        // conflicts with current capabilities" — distinct from 400 (bad
        // request) and 503 (overload). API clients can treat 409 as
        // "retry after operator intervention."
        let cases = [
            VersionError::WireMajorMismatch {
                node: WireVersion::new(1, 0),
                event: WireVersion::new(2, 0),
            },
            VersionError::SchemaTooNew {
                table: "t",
                node_max: SchemaVersion::new(1),
                event_required: SchemaVersion::new(2),
            },
            VersionError::FeatureGated {
                feature: "f",
                requires: WireVersion::new(1, 5),
                cluster_min: WireVersion::new(1, 0),
            },
            VersionError::ClusterTooOld {
                event: WireVersion::new(1, 5),
                cluster_min: WireVersion::new(1, 0),
            },
        ];
        for case in &cases {
            assert_eq!(case.http_status(), 409, "{case:?}");
        }
    }
}
