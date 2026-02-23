//! Hybrid Logical Clock (HLC) for causal ordering of operations.
//!
//! Each timestamp is a 16-byte value: `(millis: u64, logical: u32, node_id: u32)`.
//! Stored as big-endian BLOB in SQLite so lexicographic byte comparison = causal ordering.

use std::time::{SystemTime, UNIX_EPOCH};

/// A 16-byte hybrid logical timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HLCTimestamp {
    /// Wall-clock milliseconds since Unix epoch.
    pub millis: u64,
    /// Logical counter for same-millisecond ordering.
    pub logical: u32,
    /// Node identifier for tie-breaking across devices.
    pub node_id: u32,
}

impl HLCTimestamp {
    pub const ZERO: HLCTimestamp = HLCTimestamp {
        millis: 0,
        logical: 0,
        node_id: 0,
    };

    /// Serialize to 16-byte big-endian representation.
    /// Layout: [millis(8) | logical(4) | node_id(4)]
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&self.millis.to_be_bytes());
        buf[8..12].copy_from_slice(&self.logical.to_be_bytes());
        buf[12..16].copy_from_slice(&self.node_id.to_be_bytes());
        buf
    }

    /// Deserialize from 16-byte big-endian representation.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 16 {
            return None;
        }
        let millis = u64::from_be_bytes(bytes[0..8].try_into().ok()?);
        let logical = u32::from_be_bytes(bytes[8..12].try_into().ok()?);
        let node_id = u32::from_be_bytes(bytes[12..16].try_into().ok()?);
        Some(Self {
            millis,
            logical,
            node_id,
        })
    }

    /// Hex string for debug/display.
    pub fn to_hex(&self) -> String {
        hex::encode(self.to_bytes())
    }

    /// Parse from hex string.
    pub fn from_hex(s: &str) -> Option<Self> {
        let bytes = hex::decode(s).ok()?;
        Self::from_bytes(&bytes)
    }
}

impl Ord for HLCTimestamp {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.millis
            .cmp(&other.millis)
            .then(self.logical.cmp(&other.logical))
            .then(self.node_id.cmp(&other.node_id))
    }
}

impl PartialOrd for HLCTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The Hybrid Logical Clock.
pub struct HLC {
    node_id: u32,
    latest: HLCTimestamp,
}

fn wall_clock_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

impl HLC {
    /// Create a new HLC for the given node.
    pub fn new(node_id: u32) -> Self {
        Self {
            node_id,
            latest: HLCTimestamp::ZERO,
        }
    }

    /// Generate a new monotonically increasing timestamp.
    pub fn now(&mut self) -> HLCTimestamp {
        let wall = wall_clock_ms();
        let millis = wall.max(self.latest.millis);

        let logical = if millis == self.latest.millis {
            self.latest.logical + 1
        } else {
            0
        };

        let ts = HLCTimestamp {
            millis,
            logical,
            node_id: self.node_id,
        };
        self.latest = ts;
        ts
    }

    /// Receive a remote timestamp and advance the local clock.
    pub fn recv(&mut self, remote: HLCTimestamp) -> HLCTimestamp {
        let wall = wall_clock_ms();
        let millis = wall.max(self.latest.millis).max(remote.millis);

        let logical = if millis == self.latest.millis && millis == remote.millis {
            self.latest.logical.max(remote.logical) + 1
        } else if millis == self.latest.millis {
            self.latest.logical + 1
        } else if millis == remote.millis {
            remote.logical + 1
        } else {
            0
        };

        let ts = HLCTimestamp {
            millis,
            logical,
            node_id: self.node_id,
        };
        self.latest = ts;
        ts
    }

    pub fn node_id(&self) -> u32 {
        self.node_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonicity() {
        let mut hlc = HLC::new(1);
        let a = hlc.now();
        let b = hlc.now();
        let c = hlc.now();
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn test_recv_advances_clock() {
        let mut hlc = HLC::new(1);
        let local = hlc.now();

        // Simulate a remote timestamp far in the future
        let remote = HLCTimestamp {
            millis: local.millis + 100_000,
            logical: 5,
            node_id: 2,
        };
        let after_recv = hlc.recv(remote);

        // After recv, clock should be >= remote
        assert!(after_recv > remote);
        assert!(after_recv.millis >= remote.millis);

        // Next local tick should still be monotonic
        let next = hlc.now();
        assert!(next > after_recv);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let ts = HLCTimestamp {
            millis: 1700000000000,
            logical: 42,
            node_id: 12345,
        };
        let bytes = ts.to_bytes();
        let decoded = HLCTimestamp::from_bytes(&bytes).unwrap();
        assert_eq!(ts, decoded);
    }

    #[test]
    fn test_hex_roundtrip() {
        let ts = HLCTimestamp {
            millis: 1700000000000,
            logical: 42,
            node_id: 99,
        };
        let hex = ts.to_hex();
        let decoded = HLCTimestamp::from_hex(&hex).unwrap();
        assert_eq!(ts, decoded);
    }

    #[test]
    fn test_blob_ordering_matches_ord() {
        let a = HLCTimestamp {
            millis: 1000,
            logical: 0,
            node_id: 1,
        };
        let b = HLCTimestamp {
            millis: 1000,
            logical: 1,
            node_id: 1,
        };
        let c = HLCTimestamp {
            millis: 1001,
            logical: 0,
            node_id: 1,
        };

        // Ord ordering
        assert!(a < b);
        assert!(b < c);

        // BLOB ordering (lexicographic on big-endian bytes)
        assert!(a.to_bytes() < b.to_bytes());
        assert!(b.to_bytes() < c.to_bytes());
    }

    #[test]
    fn test_different_nodes_break_ties() {
        let a = HLCTimestamp {
            millis: 1000,
            logical: 0,
            node_id: 1,
        };
        let b = HLCTimestamp {
            millis: 1000,
            logical: 0,
            node_id: 2,
        };
        assert_ne!(a, b);
        assert!(a < b); // lower node_id sorts first
        assert!(a.to_bytes() < b.to_bytes());
    }

    #[test]
    fn test_from_bytes_wrong_length() {
        assert!(HLCTimestamp::from_bytes(&[0u8; 8]).is_none());
        assert!(HLCTimestamp::from_bytes(&[]).is_none());
    }

    #[test]
    fn test_zero() {
        assert_eq!(HLCTimestamp::ZERO.millis, 0);
        assert_eq!(HLCTimestamp::ZERO.logical, 0);
        assert_eq!(HLCTimestamp::ZERO.node_id, 0);
    }
}
