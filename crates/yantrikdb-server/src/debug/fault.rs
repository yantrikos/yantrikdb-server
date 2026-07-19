//! Fault-injection registry + `FaultyNetwork` trait.
//!
//! ## How a fault is enforced
//!
//! 1. Operator (or Jepsen nemesis) POSTs `/v1/debug/fault/inject` with a
//!    [`FaultKind`] and an optional TTL. The registry assigns a [`FaultId`]
//!    and stores the record.
//! 2. The cluster transport (RFC 010 PR-4 openraft) wraps its `Network`
//!    impl in [`RegistryFaultyNetwork`]. On every send/recv, the wrapper
//!    consults the registry: should I drop this message? Inject latency?
//!    Corrupt this byte?
//! 3. Operator POSTs `/v1/debug/fault/clear` (or DELETE the fault by ID)
//!    when the test phase is over. The registry empties; the wrapper
//!    starts forwarding messages cleanly again.
//!
//! ## What's in PR-5
//!
//! - The trait + types
//! - The in-memory registry with admin API
//! - A `NoopFaultyNetwork` that passes everything through (for non-Jepsen
//!   builds and unit tests)
//! - A `RegistryFaultyNetwork` that consults the registry on every call
//!
//! ## What's NOT in PR-5
//!
//! - **Actual cluster transport integration**: RFC 010 PR-4 (openraft)
//!   plugs `RegistryFaultyNetwork` into the openraft `Network` adapter.
//!   PR-5 just makes sure the trait is there to plug into.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Opaque identifier for an active fault. Assigned by the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FaultId(pub u64);

impl FaultId {
    pub fn new(n: u64) -> Self {
        Self(n)
    }
}

impl std::fmt::Display for FaultId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fault_{}", self.0)
    }
}

/// The kind of fault being injected. Each variant is a Jepsen nemesis
/// primitive — the test runner combines them to express scenarios.
///
/// **Stability**: variant names are part of the operator-facing API
/// (logs, metrics, error messages). Do not rename or remove entries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind")]
pub enum FaultKind {
    /// Drop a fraction of messages between two peers (or all peers if
    /// peer is None). `probability` ∈ [0.0, 1.0].
    DropMessages {
        from_peer: Option<u32>,
        to_peer: Option<u32>,
        probability: f64,
    },
    /// Inject latency on messages between two peers. `min_ms..=max_ms`
    /// uniform-random; pick a value per message.
    InjectLatency {
        from_peer: Option<u32>,
        to_peer: Option<u32>,
        min_ms: u64,
        max_ms: u64,
    },
    /// Full network partition between two sets of peers. While active,
    /// no messages flow in either direction.
    Partition { side_a: Vec<u32>, side_b: Vec<u32> },
    /// Pause delivery on the leader's outbound. Followers don't see
    /// heartbeats — used to force an election.
    PauseLeader { node_id: u32 },
    /// Corrupt a fraction of message bytes (flip bits at random offsets).
    /// Receiver-side decoder MUST reject corruption — this is a
    /// safety-property test, not a fuzz target.
    CorruptBytes {
        from_peer: Option<u32>,
        to_peer: Option<u32>,
        probability: f64,
    },
}

impl FaultKind {
    /// Stable label for metrics + logs.
    pub fn variant_name(&self) -> &'static str {
        match self {
            FaultKind::DropMessages { .. } => "DropMessages",
            FaultKind::InjectLatency { .. } => "InjectLatency",
            FaultKind::Partition { .. } => "Partition",
            FaultKind::PauseLeader { .. } => "PauseLeader",
            FaultKind::CorruptBytes { .. } => "CorruptBytes",
        }
    }
}

/// One active fault in the registry.
#[derive(Debug, Clone, Serialize)]
pub struct FaultRecord {
    pub id: FaultId,
    pub kind: FaultKind,
    pub created_at_unix_micros: i64,
    /// Auto-clear after this duration. `None` = persistent until
    /// /clear or DELETE.
    pub ttl_secs: Option<u64>,
    #[serde(skip)]
    expires_at: Option<Instant>,
}

impl FaultRecord {
    /// Whether this fault has expired and should be cleaned up.
    pub fn is_expired(&self, now: Instant) -> bool {
        self.expires_at.map(|exp| now >= exp).unwrap_or(false)
    }
}

/// In-memory registry of active fault injections. Cheap to clone (uses
/// Arc<RwLock> internally so all clones share state).
#[derive(Clone)]
pub struct FaultRegistry {
    inner: Arc<RegistryInner>,
}

struct RegistryInner {
    next_id: AtomicU64,
    faults: RwLock<HashMap<FaultId, FaultRecord>>,
}

impl FaultRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RegistryInner {
                next_id: AtomicU64::new(1),
                faults: RwLock::new(HashMap::new()),
            }),
        }
    }

    /// Inject a new fault. Returns the assigned `FaultId`.
    pub fn inject(&self, kind: FaultKind, ttl_secs: Option<u64>) -> FaultId {
        let id = FaultId(self.inner.next_id.fetch_add(1, Ordering::Relaxed));
        let now_inst = Instant::now();
        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);
        let record = FaultRecord {
            id,
            kind,
            created_at_unix_micros: now_micros,
            ttl_secs,
            expires_at: ttl_secs.map(|s| now_inst + Duration::from_secs(s)),
        };
        self.inner.faults.write().insert(id, record);
        id
    }

    /// Remove one fault by id. Returns true if a fault was removed.
    pub fn remove(&self, id: FaultId) -> bool {
        self.inner.faults.write().remove(&id).is_some()
    }

    /// Clear all faults. Returns the number cleared.
    pub fn clear(&self) -> usize {
        let mut faults = self.inner.faults.write();
        let n = faults.len();
        faults.clear();
        n
    }

    /// Snapshot of all currently-active faults. Expired entries are
    /// pruned before returning.
    pub fn list(&self) -> Vec<FaultRecord> {
        let now = Instant::now();
        let mut faults = self.inner.faults.write();
        faults.retain(|_, r| !r.is_expired(now));
        faults.values().cloned().collect()
    }

    /// Number of currently-active (non-expired) faults.
    pub fn active_count(&self) -> usize {
        self.list().len()
    }

    /// Decide the fate of one peer message `from → to`. Consulted by the
    /// YRP receive route (`POST /v1/yrp/msg`) BEFORE any protocol
    /// handling, per the chaos-gate design (codex D1: receive-side
    /// injection is externally controllable across process restarts and
    /// exercises the real HTTP path).
    ///
    /// Precedence: any matching Drop-class fault wins over delays; among
    /// delays the LONGEST matching delay applies. `CorruptBytes` is not
    /// evaluated here — the receive route's bincode decode rejecting
    /// corrupt bytes is covered separately.
    pub fn verdict(&self, from: u32, to: u32) -> FaultVerdict {
        let now = Instant::now();
        let faults = self.inner.faults.read();
        let mut delay_ms: Option<u64> = None;
        for r in faults.values() {
            if r.is_expired(now) {
                continue;
            }
            match &r.kind {
                FaultKind::Partition { side_a, side_b } => {
                    let crosses = (side_a.contains(&from) && side_b.contains(&to))
                        || (side_b.contains(&from) && side_a.contains(&to));
                    if crosses {
                        return FaultVerdict::Drop;
                    }
                }
                FaultKind::PauseLeader { node_id } => {
                    if *node_id == from {
                        return FaultVerdict::Drop;
                    }
                }
                FaultKind::DropMessages {
                    from_peer,
                    to_peer,
                    probability,
                } => {
                    let matches = from_peer.map_or(true, |f| f == from)
                        && to_peer.map_or(true, |t| t == to);
                    if matches && rand::Rng::gen::<f64>(&mut rand::thread_rng()) < *probability {
                        return FaultVerdict::Drop;
                    }
                }
                FaultKind::InjectLatency {
                    from_peer,
                    to_peer,
                    min_ms,
                    max_ms,
                } => {
                    let matches = from_peer.map_or(true, |f| f == from)
                        && to_peer.map_or(true, |t| t == to);
                    if matches {
                        let hi = (*max_ms).max(*min_ms);
                        let picked = if hi == *min_ms {
                            *min_ms
                        } else {
                            rand::Rng::gen_range(&mut rand::thread_rng(), *min_ms..=hi)
                        };
                        delay_ms = Some(delay_ms.map_or(picked, |d| d.max(picked)));
                    }
                }
                FaultKind::CorruptBytes { .. } => {}
            }
        }
        match delay_ms {
            Some(ms) => FaultVerdict::Delay(Duration::from_millis(ms)),
            None => FaultVerdict::Deliver,
        }
    }
}

/// Outcome of [`FaultRegistry::verdict`] for one message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultVerdict {
    Deliver,
    Drop,
    /// Deliver after this delay (the DELIVERY is delayed, not the HTTP
    /// response — otherwise latency faults just manufacture client
    /// timeouts, codex D1 pitfall).
    Delay(Duration),
}

impl Default for FaultRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait every cluster-transport adapter implements. RFC 010 PR-4
/// (openraft) plugs its `Network` adapter through this so injected
/// faults actually affect message flow.
///
/// PR-5 establishes the trait + two implementations:
/// - [`NoopFaultyNetwork`]: pass-through, used in single-node + non-test builds
/// - [`RegistryFaultyNetwork`]: consults a [`FaultRegistry`] on every call
pub trait FaultyNetwork: Send + Sync {
    /// Should the message from `from` to `to` be dropped? Returns true
    /// if the transport should silently discard it (Jepsen test
    /// scenario).
    fn should_drop(&self, from: u32, to: u32) -> bool;

    /// Optional latency to inject before delivering this message.
    fn inject_latency(&self, from: u32, to: u32) -> Option<Duration>;

    /// Should the transport corrupt bytes in this message before
    /// delivery? If true, the transport's encoder MUST corrupt at
    /// least one byte; the decoder is then obligated to reject.
    fn should_corrupt(&self, from: u32, to: u32) -> bool;
}

/// Pass-through impl. The default. No fault behavior — every send goes
/// through cleanly. Used for production single-node + most tests.
pub struct NoopFaultyNetwork;

impl FaultyNetwork for NoopFaultyNetwork {
    fn should_drop(&self, _from: u32, _to: u32) -> bool {
        false
    }
    fn inject_latency(&self, _from: u32, _to: u32) -> Option<Duration> {
        None
    }
    fn should_corrupt(&self, _from: u32, _to: u32) -> bool {
        false
    }
}

/// Registry-consulting impl. Wraps a [`FaultRegistry`]; on every call,
/// scans active faults and decides drop/latency/corrupt. Used by
/// Jepsen runs (RFC 010 PR-4 wires this into the openraft Network).
pub struct RegistryFaultyNetwork {
    registry: FaultRegistry,
}

impl RegistryFaultyNetwork {
    pub fn new(registry: FaultRegistry) -> Self {
        Self { registry }
    }

    fn matches_peers(from_filter: Option<u32>, to_filter: Option<u32>, from: u32, to: u32) -> bool {
        from_filter.map(|f| f == from).unwrap_or(true) && to_filter.map(|t| t == to).unwrap_or(true)
    }
}

impl FaultyNetwork for RegistryFaultyNetwork {
    fn should_drop(&self, from: u32, to: u32) -> bool {
        // Use a deterministic-ish hash of (from, to, time-bucket) so the
        // probability check is reproducible-within-a-window. Production
        // would use rand; keeping it deterministic here makes Jepsen
        // results easier to compare across runs. (Will refine in PR-4
        // when actual openraft messages have stable IDs we can hash.)
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let pseudo = ((from as u64) ^ ((to as u64) << 8) ^ now_secs).wrapping_mul(2654435761);
        let unit = ((pseudo % 1_000_000) as f64) / 1_000_000.0;

        for fault in self.registry.list() {
            match fault.kind {
                FaultKind::DropMessages {
                    from_peer,
                    to_peer,
                    probability,
                } => {
                    if Self::matches_peers(from_peer, to_peer, from, to) && unit < probability {
                        return true;
                    }
                }
                FaultKind::Partition {
                    ref side_a,
                    ref side_b,
                } => {
                    let from_a = side_a.contains(&from);
                    let from_b = side_b.contains(&from);
                    let to_a = side_a.contains(&to);
                    let to_b = side_b.contains(&to);
                    // Partitioned if peers are on different sides.
                    if (from_a && to_b) || (from_b && to_a) {
                        return true;
                    }
                }
                FaultKind::PauseLeader { node_id } => {
                    if from == node_id {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    fn inject_latency(&self, from: u32, to: u32) -> Option<Duration> {
        for fault in self.registry.list() {
            if let FaultKind::InjectLatency {
                from_peer,
                to_peer,
                min_ms,
                max_ms,
            } = fault.kind
            {
                if Self::matches_peers(from_peer, to_peer, from, to) {
                    // PR-5 returns the midpoint; PR-4 will use a real
                    // RNG keyed by openraft message id for
                    // reproducibility.
                    let mid = (min_ms + max_ms) / 2;
                    return Some(Duration::from_millis(mid));
                }
            }
        }
        None
    }

    fn should_corrupt(&self, from: u32, to: u32) -> bool {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let pseudo =
            ((from as u64) ^ ((to as u64) << 16) ^ (now_secs << 32)).wrapping_mul(2654435761);
        let unit = ((pseudo % 1_000_000) as f64) / 1_000_000.0;

        for fault in self.registry.list() {
            if let FaultKind::CorruptBytes {
                from_peer,
                to_peer,
                probability,
            } = fault.kind
            {
                if Self::matches_peers(from_peer, to_peer, from, to) && unit < probability {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_starts_empty() {
        let r = FaultRegistry::new();
        assert_eq!(r.active_count(), 0);
        assert!(r.list().is_empty());
    }

    #[test]
    fn inject_and_list() {
        let r = FaultRegistry::new();
        let id1 = r.inject(
            FaultKind::DropMessages {
                from_peer: Some(1),
                to_peer: Some(2),
                probability: 0.5,
            },
            None,
        );
        let id2 = r.inject(FaultKind::PauseLeader { node_id: 1 }, Some(60));
        assert_ne!(id1, id2, "fault IDs must be unique");
        let list = r.list();
        assert_eq!(list.len(), 2);
        let has_drop = list
            .iter()
            .any(|f| matches!(f.kind, FaultKind::DropMessages { .. }));
        let has_pause = list
            .iter()
            .any(|f| matches!(f.kind, FaultKind::PauseLeader { .. }));
        assert!(has_drop && has_pause);
    }

    #[test]
    fn remove_one_by_id() {
        let r = FaultRegistry::new();
        let id = r.inject(FaultKind::PauseLeader { node_id: 1 }, None);
        assert!(r.remove(id));
        assert!(!r.remove(id), "second remove should be no-op");
        assert_eq!(r.active_count(), 0);
    }

    #[test]
    fn clear_returns_count() {
        let r = FaultRegistry::new();
        r.inject(FaultKind::PauseLeader { node_id: 1 }, None);
        r.inject(FaultKind::PauseLeader { node_id: 2 }, None);
        let n = r.clear();
        assert_eq!(n, 2);
        assert_eq!(r.active_count(), 0);
    }

    #[test]
    fn noop_network_passes_everything() {
        let net = NoopFaultyNetwork;
        for from in 1..=3 {
            for to in 1..=3 {
                if from == to {
                    continue;
                }
                assert!(!net.should_drop(from, to));
                assert!(net.inject_latency(from, to).is_none());
                assert!(!net.should_corrupt(from, to));
            }
        }
    }

    #[test]
    fn registry_network_drops_with_partition() {
        let registry = FaultRegistry::new();
        let net = RegistryFaultyNetwork::new(registry.clone());

        // No partition active yet.
        assert!(!net.should_drop(1, 2));

        // Inject partition: {1, 2} vs {3, 4}.
        registry.inject(
            FaultKind::Partition {
                side_a: vec![1, 2],
                side_b: vec![3, 4],
            },
            None,
        );

        // Within-side: not dropped.
        assert!(!net.should_drop(1, 2));
        assert!(!net.should_drop(3, 4));
        // Across sides: dropped.
        assert!(net.should_drop(1, 3));
        assert!(net.should_drop(2, 4));
        assert!(net.should_drop(3, 1));
        assert!(net.should_drop(4, 2));
    }

    #[test]
    fn registry_network_pauses_leader() {
        let registry = FaultRegistry::new();
        let net = RegistryFaultyNetwork::new(registry.clone());
        registry.inject(FaultKind::PauseLeader { node_id: 1 }, None);
        // Outbound from 1 to anywhere: dropped.
        assert!(net.should_drop(1, 2));
        assert!(net.should_drop(1, 3));
        // Outbound from non-leader: not dropped.
        assert!(!net.should_drop(2, 3));
    }

    #[test]
    fn fault_kind_variant_names_are_stable() {
        // Pin metric labels.
        assert_eq!(
            FaultKind::DropMessages {
                from_peer: None,
                to_peer: None,
                probability: 0.0,
            }
            .variant_name(),
            "DropMessages"
        );
        assert_eq!(
            FaultKind::InjectLatency {
                from_peer: None,
                to_peer: None,
                min_ms: 0,
                max_ms: 0,
            }
            .variant_name(),
            "InjectLatency"
        );
        assert_eq!(
            FaultKind::Partition {
                side_a: vec![],
                side_b: vec![],
            }
            .variant_name(),
            "Partition"
        );
        assert_eq!(
            FaultKind::PauseLeader { node_id: 0 }.variant_name(),
            "PauseLeader"
        );
        assert_eq!(
            FaultKind::CorruptBytes {
                from_peer: None,
                to_peer: None,
                probability: 0.0,
            }
            .variant_name(),
            "CorruptBytes"
        );
    }

    #[test]
    fn fault_id_display_format() {
        assert_eq!(FaultId(42).to_string(), "fault_42");
    }
}
