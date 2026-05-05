//! `VersionGate` — runtime cluster-wide version negotiation.
//!
//! ## What it tracks
//!
//! - **Local wire version**: what this node speaks (immutable, build-time)
//! - **Cluster min/max wire**: rolling-window observation of what every
//!   peer speaks. Updated whenever a peer's heartbeat carries its version.
//! - **Feature floors**: per-feature minimum wire version required to
//!   safely use that feature in writes
//!
//! ## How negotiation works
//!
//! Every cluster heartbeat carries the sender's [`WireVersion`]. The
//! receiver records it. The cluster minimum is `min(observed_versions)`.
//! When a peer goes offline (heartbeat timeout > N seconds), it's removed
//! from the observation set — the min may rise as a consequence.
//!
//! Writers consult `can_use_feature(feature)` before emitting a new event
//! variant. If the cluster min is below the feature's floor, the write
//! is refused with [`VersionError::ClusterTooOld`] — better to refuse
//! than to break replication.
//!
//! ## Why atomics
//!
//! Hot path readers (every recall, every replicate) call `local_wire()`
//! and `cluster_min()`. These are read on every operation, so they must
//! be lock-free. Writers (heartbeat handler observing a peer) acquire a
//! Mutex briefly to recompute min/max from the peer set, then store
//! atomically.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU16, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use super::error::VersionError;
use super::wire::{WireVersion, CURRENT_WIRE_VERSION};

/// Per-feature minimum cluster wire version requirements.
///
/// Add an entry when introducing a new event variant or capability that
/// older nodes can't handle. The writer checks this before emitting.
///
/// **Stability**: feature names are part of the operator-facing API
/// (logs, metrics, error messages). Do not rename or remove entries.
///
/// **Convention**: mutation variants register as `mutation.<VariantName>`
/// to match `MemoryMutation::feature_flag()`. Other features (admin
/// endpoints, query modes, etc.) use any namespace that doesn't collide
/// with `mutation.*`.
pub const FEATURE_FLOORS: &[(&str, WireVersion)] = &[
    // Baseline — sentinel feature confirming the gate is operating.
    ("baseline_v1_0", WireVersion::new(1, 0)),
    // RFC 010 PR-3: every MemoryMutation variant registered here at
    // wire 1.0. Future variants in 1.1+ register with their introduction
    // version. Writers consult `gate.can_use_feature(mutation.feature_flag())`
    // before emitting.
    ("mutation.UpsertMemory", WireVersion::new(1, 0)),
    ("mutation.UpdateMemoryPatch", WireVersion::new(1, 0)),
    ("mutation.TombstoneMemory", WireVersion::new(1, 0)),
    ("mutation.PurgeMemory", WireVersion::new(1, 0)),
    ("mutation.UpsertEntityEdge", WireVersion::new(1, 0)),
    ("mutation.DeleteEntityEdge", WireVersion::new(1, 0)),
    ("mutation.TenantConfigPatch", WireVersion::new(1, 0)),
    // RFC 010 PR-6.2: when a leader emits a materialized UpsertMemory
    // (carrying extracted_entities + created_at_unix_micros +
    // embedding_model populated), the cluster minimum must be at wire
    // 1.1+. Older peers still decode the variant — the new fields are
    // additive — but the leader gates emission to avoid sending
    // information older peers can't act on.
    ("mutation.UpsertMemory.materialized", WireVersion::new(1, 1)),
];

/// How long a peer can be silent before we drop it from the observation set.
/// Default 30 seconds — generous enough to survive a brief network blip,
/// short enough that an actually-failed node doesn't pin the cluster min
/// to its old version forever.
pub const PEER_OBSERVATION_TIMEOUT: Duration = Duration::from_secs(30);

/// Per-peer last-known wire version + timestamp.
struct PeerObservation {
    version: WireVersion,
    last_seen: Instant,
}

/// Cluster-wide wire-version state. Cheap to clone (Arc internally).
pub struct VersionGate {
    /// What this node writes. Immutable after construction.
    local_wire: WireVersion,
    /// Cluster minimum observed wire major. Hot-path read.
    cluster_min_major: AtomicU8,
    /// Cluster minimum observed wire minor. Hot-path read.
    cluster_min_minor: AtomicU16,
    /// Cluster maximum observed wire major. Hot-path read.
    cluster_max_major: AtomicU8,
    /// Cluster maximum observed wire minor. Hot-path read.
    cluster_max_minor: AtomicU16,
    /// Per-peer observation. Updated on heartbeat.
    peers: RwLock<HashMap<u32, PeerObservation>>,
    /// Per-feature minimum. Lookup table.
    feature_floors: HashMap<&'static str, WireVersion>,
}

impl VersionGate {
    /// Construct with the local node's wire version. Cluster min/max
    /// initialize to local; updated as peers report in.
    pub fn new(local: WireVersion) -> Self {
        let feature_floors = FEATURE_FLOORS
            .iter()
            .map(|(name, ver)| (*name, *ver))
            .collect();
        Self {
            local_wire: local,
            cluster_min_major: AtomicU8::new(local.major),
            cluster_min_minor: AtomicU16::new(local.minor),
            cluster_max_major: AtomicU8::new(local.major),
            cluster_max_minor: AtomicU16::new(local.minor),
            peers: RwLock::new(HashMap::new()),
            feature_floors,
        }
    }

    /// Construct with the build's `CURRENT_WIRE_VERSION`. Most callers
    /// want this; tests use [`Self::new`] for explicit control.
    pub fn for_local_build() -> Self {
        Self::new(CURRENT_WIRE_VERSION)
    }

    /// What this node writes.
    pub fn local_wire(&self) -> WireVersion {
        self.local_wire
    }

    /// Current cluster minimum observed wire version.
    pub fn cluster_min(&self) -> WireVersion {
        WireVersion::new(
            self.cluster_min_major.load(Ordering::Relaxed),
            self.cluster_min_minor.load(Ordering::Relaxed),
        )
    }

    /// Current cluster maximum observed wire version.
    pub fn cluster_max(&self) -> WireVersion {
        WireVersion::new(
            self.cluster_max_major.load(Ordering::Relaxed),
            self.cluster_max_minor.load(Ordering::Relaxed),
        )
    }

    /// Record that a peer is at the given version. Recomputes cluster
    /// min/max. Called by the heartbeat handler.
    pub fn observe_peer(&self, peer_id: u32, version: WireVersion) {
        {
            let mut peers = self.peers.write();
            peers.insert(
                peer_id,
                PeerObservation {
                    version,
                    last_seen: Instant::now(),
                },
            );
        }
        self.recompute_min_max();
    }

    /// Drop peers that haven't been seen recently. Returns the number
    /// dropped. Called periodically by a background loop (RFC 009 PR-1
    /// added the dedicated control runtime, so this can run there).
    pub fn evict_stale_peers(&self) -> usize {
        let now = Instant::now();
        let dropped = {
            let mut peers = self.peers.write();
            let before = peers.len();
            peers.retain(|_, obs| now.duration_since(obs.last_seen) < PEER_OBSERVATION_TIMEOUT);
            before - peers.len()
        };
        if dropped > 0 {
            self.recompute_min_max();
        }
        dropped
    }

    /// Recompute cluster min/max from the peer set + local. Called after
    /// any change to the peer set.
    fn recompute_min_max(&self) {
        let peers = self.peers.read();
        let mut min_seen = self.local_wire;
        let mut max_seen = self.local_wire;
        for obs in peers.values() {
            if obs.version < min_seen {
                min_seen = obs.version;
            }
            if obs.version > max_seen {
                max_seen = obs.version;
            }
        }
        self.cluster_min_major
            .store(min_seen.major, Ordering::Relaxed);
        self.cluster_min_minor
            .store(min_seen.minor, Ordering::Relaxed);
        self.cluster_max_major
            .store(max_seen.major, Ordering::Relaxed);
        self.cluster_max_minor
            .store(max_seen.minor, Ordering::Relaxed);
    }

    /// Whether a write that would emit a new event-variant gated by
    /// `feature` can safely be issued, given the current cluster min.
    /// Returns `Ok(())` if safe; structured error if not.
    pub fn can_use_feature(&self, feature: &'static str) -> Result<(), VersionError> {
        let floor = match self.feature_floors.get(feature).copied() {
            Some(f) => f,
            // Unknown features are treated as "always allowed" — they
            // must be a build-local concept that isn't gated. If you
            // intend to gate a feature, register it in [`FEATURE_FLOORS`].
            None => return Ok(()),
        };
        let current_min = self.cluster_min();
        if current_min < floor {
            return Err(VersionError::FeatureGated {
                feature,
                requires: floor,
                cluster_min: current_min,
            });
        }
        Ok(())
    }

    /// Number of peers currently in the observation set.
    pub fn peer_count(&self) -> usize {
        self.peers.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_gate_reports_local_as_min_and_max() {
        let g = VersionGate::new(WireVersion::new(1, 5));
        assert_eq!(g.cluster_min(), WireVersion::new(1, 5));
        assert_eq!(g.cluster_max(), WireVersion::new(1, 5));
        assert_eq!(g.local_wire(), WireVersion::new(1, 5));
    }

    #[test]
    fn observing_lower_peer_lowers_min() {
        let g = VersionGate::new(WireVersion::new(1, 5));
        g.observe_peer(2, WireVersion::new(1, 0));
        assert_eq!(g.cluster_min(), WireVersion::new(1, 0));
        // Local stays as max because it's higher than the peer.
        assert_eq!(g.cluster_max(), WireVersion::new(1, 5));
    }

    #[test]
    fn observing_higher_peer_raises_max() {
        let g = VersionGate::new(WireVersion::new(1, 5));
        g.observe_peer(2, WireVersion::new(1, 9));
        assert_eq!(g.cluster_min(), WireVersion::new(1, 5));
        assert_eq!(g.cluster_max(), WireVersion::new(1, 9));
    }

    #[test]
    fn evicting_stale_peer_recomputes_min() {
        let g = VersionGate::new(WireVersion::new(1, 5));
        // Manually inject an old observation by writing directly.
        {
            let mut peers = g.peers.write();
            peers.insert(
                42,
                PeerObservation {
                    version: WireVersion::new(1, 0),
                    last_seen: Instant::now() - PEER_OBSERVATION_TIMEOUT - Duration::from_secs(1),
                },
            );
        }
        // Without recomputing yet, min still shows 1.5 (local) because
        // we bypassed observe_peer's recompute. Force a recompute by
        // calling evict.
        g.recompute_min_max();
        assert_eq!(g.cluster_min(), WireVersion::new(1, 0));

        let dropped = g.evict_stale_peers();
        assert_eq!(dropped, 1);
        // After eviction, min reverts to local.
        assert_eq!(g.cluster_min(), WireVersion::new(1, 5));
    }

    #[test]
    fn can_use_feature_allows_known_when_floor_satisfied() {
        let g = VersionGate::new(WireVersion::new(1, 5));
        // baseline_v1_0 needs 1.0; we're at 1.5, fine.
        assert!(g.can_use_feature("baseline_v1_0").is_ok());
    }

    #[test]
    fn can_use_feature_rejects_when_cluster_too_old() {
        let g = VersionGate::new(WireVersion::new(1, 5));
        // Add a peer at 1.0 — but feature requires 1.0, so this passes.
        g.observe_peer(2, WireVersion::new(1, 0));
        assert!(g.can_use_feature("baseline_v1_0").is_ok());

        // Define a synthetic gate with a higher floor for the test.
        // We can't insert into the static FEATURE_FLOORS, so we use the
        // gate's own map directly — this is the legitimate path for tests.
        let mut g2 = VersionGate::new(WireVersion::new(1, 5));
        g2.feature_floors
            .insert("synthetic_test_feature", WireVersion::new(1, 3));
        g2.observe_peer(99, WireVersion::new(1, 0));
        let err = g2.can_use_feature("synthetic_test_feature").unwrap_err();
        match err {
            VersionError::FeatureGated {
                feature,
                requires,
                cluster_min,
            } => {
                assert_eq!(feature, "synthetic_test_feature");
                assert_eq!(requires, WireVersion::new(1, 3));
                assert_eq!(cluster_min, WireVersion::new(1, 0));
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn unknown_features_are_allowed() {
        // If a writer asks about a feature we don't gate, it's safe by
        // default — the registry is the source of truth for what needs
        // gating, not the opposite.
        let g = VersionGate::new(WireVersion::new(1, 5));
        assert!(g.can_use_feature("not_in_registry").is_ok());
    }

    #[test]
    fn rolling_upgrade_simulation() {
        // 3-node cluster: nodes 1/2/3 all start at 1.0.
        let g = VersionGate::new(WireVersion::new(1, 0));
        g.observe_peer(2, WireVersion::new(1, 0));
        g.observe_peer(3, WireVersion::new(1, 0));
        assert_eq!(g.cluster_min(), WireVersion::new(1, 0));
        assert_eq!(g.cluster_max(), WireVersion::new(1, 0));

        // Operator upgrades node 2 to 1.5.
        g.observe_peer(2, WireVersion::new(1, 5));
        // Cluster min is still 1.0 (nodes 1+3 unchanged), max is 1.5.
        assert_eq!(g.cluster_min(), WireVersion::new(1, 0));
        assert_eq!(g.cluster_max(), WireVersion::new(1, 5));

        // Operator upgrades node 3.
        g.observe_peer(3, WireVersion::new(1, 5));
        // Local (node 1) is still 1.0, so min stays 1.0.
        assert_eq!(g.cluster_min(), WireVersion::new(1, 0));

        // Node 1 is the local; in real deployment it would also be
        // upgraded last (rolling upgrade pattern). We can't simulate
        // that without rebuilding the gate, but the point is: feature
        // gates correctly refuse new-feature writes during the rolling
        // window when ANY node is still old.
    }

    #[test]
    fn peer_count_tracks_observation_set() {
        let g = VersionGate::new(WireVersion::new(1, 0));
        assert_eq!(g.peer_count(), 0);
        g.observe_peer(2, WireVersion::new(1, 0));
        assert_eq!(g.peer_count(), 1);
        g.observe_peer(3, WireVersion::new(1, 0));
        assert_eq!(g.peer_count(), 2);
        // Re-observing same peer doesn't double-count.
        g.observe_peer(2, WireVersion::new(1, 1));
        assert_eq!(g.peer_count(), 2);
    }
}
