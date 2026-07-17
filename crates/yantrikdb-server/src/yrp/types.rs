//! YRP core identifier and state types (RFC 028 v2 §2–§3).
//!
//! Everything here is plain data: `Copy` where possible, `serde`-able for the
//! wire, and free of I/O. The safety-critical distinction is between
//! [`HardState`] (MUST be durably persisted before certain messages may be
//! sent — see `election::ElectionCore`) and everything else (soft state,
//! reconstructible).

use serde::{Deserialize, Serialize};

/// Immutable cluster identity (RFC 028 v2 §3). Ops, votes, and snapshots
/// carry it; a mismatch at boot or on the wire is a quarantine trigger —
/// state from a different cluster is alien no matter how well-formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClusterId(pub u64);

/// A node's stable identifier within one cluster.
///
/// Identity alone is NOT sufficient to vote — RFC 028 v2 §3 requires the
/// node's [`Incarnation`] to be authorized by a quorum (Proxmox clones and VM
/// rollbacks resurrect old disk state, so the *cluster*, not the disk, is the
/// authority on which copy of a node is real). Incarnation enforcement lands
/// in Phase A2; the type exists now so wire shapes are stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub u64);

/// Election term. Monotonically increasing; one leader at most per term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Term(pub u64);

impl Term {
    pub const ZERO: Term = Term(0);
    #[must_use]
    pub fn next(self) -> Term {
        Term(self.0 + 1)
    }
}

/// Index into the canonical prefix log (RFC 028 v2 §2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LogIndex(pub u64);

/// A position in the canonical prefix log: the entry's term and index.
///
/// Election freshness compares positions with **Raft's rule** (higher term
/// wins; equal terms → higher index wins) via the derived lexicographic
/// `Ord` — field order `(term, index)` is therefore load-bearing. This is
/// what protects possibly-committed suffixes (RFC 028 v2 §2.2, scenario R3):
/// a scalar watermark cannot distinguish divergent histories, and a
/// committed-frontier comparison discards quorum-accepted-but-not-yet-marked
/// entries. Do NOT reorder these fields.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct LogPosition {
    pub term: u64,
    pub index: u64,
}

impl LogPosition {
    pub const ZERO: LogPosition = LogPosition { term: 0, index: 0 };

    /// Raft's up-to-date-ness comparison, named for auditability at call
    /// sites. `self.is_at_least_as_up_to_date_as(other)` ⇔ a voter with last
    /// position `other` may grant a candidate whose last position is `self`.
    #[must_use]
    pub fn is_at_least_as_up_to_date_as(&self, other: &LogPosition) -> bool {
        self >= other
    }
}

/// The durable consensus metadata. **The safety of the whole protocol rests
/// on this struct being fsynced at the right moments** (RFC 028 v2 §2.3):
/// a voter persists `(current_term, voted_for)` BEFORE its granted-vote
/// response leaves the node. `ElectionCore` enforces that ordering
/// structurally.
///
/// Fail-closed rule (Phase A2, §5): if this state is torn, missing, or from
/// a different cluster/incarnation at boot, the node starts QUARANTINED
/// (non-voting) — it never guesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardState {
    pub current_term: Term,
    pub voted_for: Option<NodeId>,
}

impl Default for HardState {
    fn default() -> Self {
        Self {
            current_term: Term::ZERO,
            voted_for: None,
        }
    }
}

/// Quorum arithmetic for a voter set of size `n`: floor(n/2) + 1.
///
/// Witnesses count toward *election* quorum only — never toward data
/// durability (RFC 028 v2 §4). That distinction lives at the call sites that
/// assemble the voter set; the arithmetic here is deliberately dumb.
#[must_use]
pub fn quorum(n_voters: usize) -> usize {
    n_voters / 2 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R3's foundation: the freshness comparison is Raft's, exactly.
    #[test]
    fn log_position_ordering_is_raft_up_to_date_rule() {
        let old_term_long = LogPosition { term: 1, index: 7 };
        let new_term_short = LogPosition { term: 2, index: 5 };
        // Higher term wins regardless of index.
        assert!(new_term_short.is_at_least_as_up_to_date_as(&old_term_long));
        assert!(!old_term_long.is_at_least_as_up_to_date_as(&new_term_short));
        // Equal terms: higher index wins.
        let a = LogPosition { term: 2, index: 9 };
        let b = LogPosition { term: 2, index: 5 };
        assert!(a.is_at_least_as_up_to_date_as(&b));
        assert!(!b.is_at_least_as_up_to_date_as(&a));
        // Equality is at-least-as-up-to-date both ways.
        assert!(b.is_at_least_as_up_to_date_as(&LogPosition { term: 2, index: 5 }));
    }

    #[test]
    fn quorum_arithmetic() {
        assert_eq!(quorum(1), 1);
        assert_eq!(quorum(2), 2); // 2-voter cluster: both required — zero fault tolerance
        assert_eq!(quorum(3), 2);
        assert_eq!(quorum(4), 3);
        assert_eq!(quorum(5), 3);
    }
}
