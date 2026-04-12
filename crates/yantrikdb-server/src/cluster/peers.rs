//! Peer registry — tracks all known peers and their status.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::time::Instant;

use crate::config::{NodeRole, PeerConfig};

/// Status of a single peer.
#[derive(Debug, Clone)]
pub struct PeerStatus {
    pub addr: String,
    pub configured_role: NodeRole,
    pub node_id: Option<u32>,       // learned from handshake
    pub current_term: u64,          // last known term
    pub last_seen: Option<Instant>, // last successful contact
    pub reachable: bool,
    pub last_hlc: Option<Vec<u8>>, // last known oplog position
    pub last_op_id: Option<String>,
}

impl PeerStatus {
    pub fn new(cfg: &PeerConfig) -> Self {
        Self {
            addr: cfg.addr.clone(),
            configured_role: cfg.role,
            node_id: None,
            current_term: 0,
            last_seen: None,
            reachable: false,
            last_hlc: None,
            last_op_id: None,
        }
    }

    pub fn time_since_seen(&self) -> Option<std::time::Duration> {
        self.last_seen.map(|t| t.elapsed())
    }
}

/// Thread-safe registry of all peers in the cluster.
pub struct PeerRegistry {
    peers: Mutex<HashMap<String, PeerStatus>>, // keyed by addr
}

impl PeerRegistry {
    pub fn new(peers: &[PeerConfig]) -> Self {
        let mut map = HashMap::new();
        for p in peers {
            map.insert(p.addr.clone(), PeerStatus::new(p));
        }
        Self {
            peers: Mutex::new(map),
        }
    }

    /// Number of configured peers.
    pub fn count(&self) -> usize {
        self.peers.lock().len()
    }

    /// Get a snapshot of all peer statuses.
    pub fn snapshot(&self) -> Vec<PeerStatus> {
        self.peers.lock().values().cloned().collect()
    }

    /// Mark a peer as reachable, update its state from a handshake.
    pub fn record_handshake(&self, addr: &str, node_id: u32, current_term: u64) {
        let mut peers = self.peers.lock();
        if let Some(p) = peers.get_mut(addr) {
            p.node_id = Some(node_id);
            p.current_term = current_term;
            p.last_seen = Some(Instant::now());
            p.reachable = true;
        }
    }

    /// Mark a peer as unreachable.
    pub fn mark_unreachable(&self, addr: &str) {
        let mut peers = self.peers.lock();
        if let Some(p) = peers.get_mut(addr) {
            p.reachable = false;
        }
    }

    /// Update a peer's oplog position.
    pub fn update_oplog_position(&self, addr: &str, hlc: Vec<u8>, op_id: String) {
        let mut peers = self.peers.lock();
        if let Some(p) = peers.get_mut(addr) {
            p.last_hlc = Some(hlc);
            p.last_op_id = Some(op_id);
            p.last_seen = Some(Instant::now());
            p.reachable = true;
        }
    }

    /// How many voter peers (excluding witnesses, read replicas).
    pub fn voter_count(&self) -> usize {
        self.peers
            .lock()
            .values()
            .filter(|p| p.configured_role == NodeRole::Voter)
            .count()
    }

    /// How many quorum members (voters + witnesses).
    pub fn quorum_member_count(&self) -> usize {
        self.peers
            .lock()
            .values()
            .filter(|p| matches!(p.configured_role, NodeRole::Voter | NodeRole::Witness))
            .count()
    }

    /// How many quorum members are currently reachable.
    pub fn reachable_quorum_count(&self) -> usize {
        self.peers
            .lock()
            .values()
            .filter(|p| {
                matches!(p.configured_role, NodeRole::Voter | NodeRole::Witness) && p.reachable
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{NodeRole, PeerConfig};

    fn peers() -> Vec<PeerConfig> {
        vec![
            PeerConfig {
                addr: "node-2:7437".into(),
                role: NodeRole::Voter,
            },
            PeerConfig {
                addr: "witness-1:7437".into(),
                role: NodeRole::Witness,
            },
        ]
    }

    #[test]
    fn registry_tracks_peers() {
        let reg = PeerRegistry::new(&peers());
        assert_eq!(reg.count(), 2);
        assert_eq!(reg.voter_count(), 1);
        assert_eq!(reg.quorum_member_count(), 2);
        assert_eq!(reg.reachable_quorum_count(), 0);
    }

    #[test]
    fn handshake_marks_reachable() {
        let reg = PeerRegistry::new(&peers());
        reg.record_handshake("node-2:7437", 2, 0);
        assert_eq!(reg.reachable_quorum_count(), 1);

        reg.record_handshake("witness-1:7437", 99, 0);
        assert_eq!(reg.reachable_quorum_count(), 2);
    }

    #[test]
    fn unreachable_decreases_count() {
        let reg = PeerRegistry::new(&peers());
        reg.record_handshake("node-2:7437", 2, 0);
        reg.record_handshake("witness-1:7437", 99, 0);
        assert_eq!(reg.reachable_quorum_count(), 2);

        reg.mark_unreachable("node-2:7437");
        assert_eq!(reg.reachable_quorum_count(), 1);
    }
}
