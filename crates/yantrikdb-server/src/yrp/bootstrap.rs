//! YRP recovery plane — quarantine + quorum-authorized rejoin (RFC 028 v2 §5).
//!
//! **The CT 141 lesson, precisely scoped.** The outage was a *process* that
//! crash-looped for 10 days: diagnostics dead, HTTP dead, operator blind. The
//! naive cure — auto-reconstruct replication state and resume — enables
//! double-voting (review scenario R2: reset `voted_for`, vote again in the
//! same term). The correct split, which this module implements:
//!
//! - **The process always starts.** Boot inspection ([`inspect`]) never
//!   panics and never refuses to produce a node; the worst outcome is a
//!   [`QuarantinedNode`] that serves diagnostics (and, when data
//!   independently verifies, explicitly-labeled stale reads).
//! - **Consensus metadata fails closed.** Any safety-critical uncertainty —
//!   torn `(term, vote)`, alien cluster ID, log damage, a commit frontier
//!   beyond verifiable data — yields quarantine: non-voting, non-leading,
//!   non-acking. Refusing to vote is not a wedge; it is the mechanism that
//!   prevents split-brain.
//! - **Rejoin is quorum-authorized, never self-directed.** A quarantined
//!   node asks the current leader; the leader's authority (its leadership
//!   is quorum-backed, proven by a committed entry in its term) authorizes
//!   the resync. The node never picks "the most advanced reachable peer"
//!   itself — a stale partition can be internally consistent and obsolete
//!   (review R5).
//! - **Recoverable incompleteness ≠ corruption evidence.** Incompleteness
//!   auto-resyncs; corruption evidence additionally raises
//!   [`BootstrapEffect::Alarm`] and is only replaceable from a source whose
//!   snapshot verifies. Either way the old state is preserved first
//!   ([`BootstrapEffect::PreserveOldState`] — the automated version of the
//!   CT 141 manual backup).
//!
//! Like the replica core, everything here is pure logic driven by effects;
//! the deterministic simulator injects torn state, alien state, and
//! corruption, then proves the Gate A #4 invariant: quarantine fails closed.
//!
//! Phase B extends rejoin with quorum-managed incarnations (Proxmox
//! clone/rollback fencing) and true membership change; in A2's fixed-voter
//! world, a rejoining node re-enters as the same voter after resync.

use super::replica::LogEntry;
use super::types::{ClusterId, HardState, NodeId, Term};

/// What the boot path recovered from disk. The driver fills this in from
/// real storage (with real checksums); the sim injects damage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveredState {
    /// The cluster this on-disk state claims to belong to; `None` = blank.
    pub cluster_id: Option<ClusterId>,
    /// `None` = unreadable/absent. `Some` = parsed (integrity says whether
    /// its checksum verified).
    pub hard: Option<HardState>,
    pub log: Option<Vec<LogEntry>>,
    /// The durably recorded applied/commit marker, if any. A marker beyond
    /// the verifiable log is a frontier-beyond-data inconsistency.
    pub commit_marker: u64,
    pub integrity: Integrity,
}

/// Driver-computed integrity verdicts (production: checksummed dual-write
/// records + log hash chain; sim: injected).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Integrity {
    /// The hard-state record parsed AND its checksum verified.
    pub hard_state_verified: bool,
    /// The log's hash chain verified end to end.
    pub log_verified: bool,
}

impl RecoveredState {
    /// A genuinely blank disk — the fresh-node case.
    pub fn blank() -> Self {
        Self {
            cluster_id: None,
            hard: None,
            log: None,
            commit_marker: 0,
            integrity: Integrity {
                hard_state_verified: true,
                log_verified: true,
            },
        }
    }
}

/// Why a node is quarantined. Kept as data (not just a log line) because the
/// diagnostics surface reports them and the rejoin path branches on them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuarantineReason {
    /// Hard state present but failed verification (torn write, bit rot).
    TornHardState,
    /// Hard state absent while other replication state exists.
    MissingHardState,
    /// On-disk state belongs to a different cluster.
    ClusterIdMismatch,
    /// Log hash chain broken — DATA corruption evidence (alarms).
    LogCorruption,
    /// Commit marker beyond the verifiable log (frontier-beyond-data).
    CommitBeyondLog,
}

impl QuarantineReason {
    /// Corruption evidence (vs recoverable incompleteness): preserved state
    /// must be alarmed about, and replacement requires a verified source.
    pub fn is_corruption_evidence(&self) -> bool {
        matches!(
            self,
            QuarantineReason::TornHardState | QuarantineReason::LogCorruption
        )
    }
}

/// The boot decision. Never an error — the process always starts as
/// *something*.
#[derive(Debug)]
pub enum BootDecision {
    /// State is coherent (or genuinely blank): construct the replica core
    /// with exactly this state.
    Healthy { hard: HardState, log: Vec<LogEntry> },
    /// Fail closed: run as a [`QuarantinedNode`] until an authorized rejoin.
    /// `term_hint` is the current_term parsed from the damaged hard-state
    /// record when the bytes were readable — UNTRUSTED (its checksum
    /// failed), usable only to REFUSE things (a rejoin grant below the
    /// hint), never to authorize them. See `QuarantinedNode::on_grant`.
    Quarantine {
        reasons: Vec<QuarantineReason>,
        term_hint: Option<Term>,
    },
}

/// Inspect recovered state against the node's configured cluster identity.
/// Pure; total; never panics. The full RFC §5 trigger list beyond what
/// exists at A2 (incarnation/epoch regression, snapshot manifests,
/// capability support) lands with the Phase B machinery that introduces
/// those artifacts.
pub fn inspect(expected_cluster: ClusterId, recovered: &RecoveredState) -> BootDecision {
    // Fresh node: nothing on disk at all → healthy with defaults. (A blank
    // disk is not "torn" — there is nothing to be torn.)
    if recovered.cluster_id.is_none()
        && recovered.hard.is_none()
        && recovered.log.is_none()
        && recovered.commit_marker == 0
    {
        return BootDecision::Healthy {
            hard: HardState::default(),
            log: Vec::new(),
        };
    }

    let mut reasons = Vec::new();

    match recovered.cluster_id {
        Some(id) if id != expected_cluster => reasons.push(QuarantineReason::ClusterIdMismatch),
        None => {
            // Replication state without a cluster identity is alien by
            // definition — we cannot prove it is ours.
            reasons.push(QuarantineReason::ClusterIdMismatch);
        }
        Some(_) => {}
    }

    match (&recovered.hard, recovered.integrity.hard_state_verified) {
        (Some(_), true) => {}
        (Some(_), false) => reasons.push(QuarantineReason::TornHardState),
        (None, _) => reasons.push(QuarantineReason::MissingHardState),
    }

    if !recovered.integrity.log_verified {
        reasons.push(QuarantineReason::LogCorruption);
    }

    let log_len = recovered.log.as_ref().map_or(0, |l| l.len() as u64);
    if recovered.commit_marker > log_len {
        reasons.push(QuarantineReason::CommitBeyondLog);
    }

    if reasons.is_empty() {
        BootDecision::Healthy {
            hard: recovered.hard.expect("verified above"),
            log: recovered.log.clone().unwrap_or_default(),
        }
    } else {
        BootDecision::Quarantine {
            reasons,
            term_hint: recovered.hard.map(|h| h.current_term),
        }
    }
}

/// Rejoin protocol messages. Carried on the same transport as replica
/// messages; the sim wraps both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejoinMessage {
    /// Quarantined node → (believed) leader.
    Request { node: NodeId },
    /// Leader → quarantined node. Authorization is quorum-backed: only a
    /// leader that has COMMITTED an entry in its current term may grant
    /// (its leadership certificate — a lone "I think I'm leader" claim from
    /// a stale partition cannot produce one, review R5). The grant carries
    /// the full authorized snapshot; `verified` asserts the source snapshot
    /// passed its own integrity checks (required when the quarantined
    /// node's reasons include corruption evidence).
    Grant {
        cluster_id: ClusterId,
        term: Term,
        log: Vec<LogEntry>,
        commit: u64,
        verified: bool,
    },
}

/// Effects a [`QuarantinedNode`] asks of its driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootstrapEffect {
    /// Move the damaged/alien state aside, timestamped — forensics
    /// preserved BEFORE any resync (the automated CT 141 backup).
    PreserveOldState,
    /// Corruption evidence found — page the operator. Auto-resync may still
    /// proceed (from a verified source), but the signal must not be
    /// swallowed (review: "auto-resync must not silence a corruption
    /// signal").
    Alarm { reasons: Vec<QuarantineReason> },
    /// Send a rejoin message.
    Send { to: NodeId, msg: RejoinMessage },
    /// Durably persist the adopted snapshot, then construct a fresh
    /// follower `ReplicaCore` from exactly this state and resume normal
    /// operation. Quarantine ends here and only here.
    AdoptSnapshot {
        cluster_id: ClusterId,
        hard: HardState,
        log: Vec<LogEntry>,
    },
}

/// A booted-but-not-voting node. It answers diagnostics, optionally serves
/// labeled stale reads, and periodically retries rejoin. It has NO vote
/// handler and NO append handler — fail-closed is enforced by absence: the
/// type simply cannot produce a vote grant or a durable ack.
pub struct QuarantinedNode {
    id: NodeId,
    cluster: ClusterId,
    reasons: Vec<QuarantineReason>,
    /// Untrusted current_term parsed from the damaged record (checksum
    /// failed, bytes readable). Used ONLY to refuse rejoin grants below it —
    /// codex review finding 1: a grant below the node's true prior term
    /// regresses the durable term and reopens a double-vote window in the
    /// prior term. Refusing below-hint grants closes the replayed/stale-
    /// grant scenario whenever the record parsed; the residual (record
    /// unreadable or hint corrupted downward) is closed by Gate C's
    /// quorum-managed incarnations, not by this hint — never treat the
    /// hint as proof.
    term_hint: Option<Term>,
    /// Whether the preserved-state effect has been issued (once).
    preserved: bool,
    alarmed: bool,
}

impl QuarantinedNode {
    pub fn new(
        id: NodeId,
        cluster: ClusterId,
        reasons: Vec<QuarantineReason>,
        term_hint: Option<Term>,
    ) -> Self {
        debug_assert!(!reasons.is_empty(), "quarantine requires a reason");
        Self {
            id,
            cluster,
            reasons,
            term_hint,
            preserved: false,
            alarmed: false,
        }
    }

    /// The diagnostics surface ("process up" is a different health dimension
    /// from "data servable").
    pub fn reasons(&self) -> &[QuarantineReason] {
        &self.reasons
    }

    /// Stale reads are only offered when the DATA verified. This is the
    /// data axis, distinct from [`QuarantineReason::is_corruption_evidence`]
    /// (the storage-failure axis that drives alarms and the verified-source
    /// requirement): a torn HARD-STATE record is storage-failure evidence,
    /// but the log/payload bytes verified fine — serving them, labeled
    /// stale, is exactly the CT 141 availability the RFC promises. Only
    /// damage to the data itself ([`QuarantineReason::LogCorruption`])
    /// withdraws reads.
    pub fn stale_reads_allowed(&self) -> bool {
        !self
            .reasons
            .iter()
            .any(|r| matches!(r, QuarantineReason::LogCorruption))
    }

    /// Driver tick (retry timer): preserve-once, alarm-once, then ask the
    /// given peer (the driver's current leader hint) for rejoin.
    pub fn tick_rejoin(&mut self, leader_hint: NodeId) -> Vec<BootstrapEffect> {
        let mut out = Vec::new();
        if !self.preserved {
            self.preserved = true;
            out.push(BootstrapEffect::PreserveOldState);
        }
        if !self.alarmed && self.reasons.iter().any(|r| r.is_corruption_evidence()) {
            self.alarmed = true;
            out.push(BootstrapEffect::Alarm {
                reasons: self.reasons.clone(),
            });
        }
        out.push(BootstrapEffect::Send {
            to: leader_hint,
            msg: RejoinMessage::Request { node: self.id },
        });
        out
    }

    /// Handle a rejoin grant from `from`. Returns the adopt effect when
    /// acceptable; a grant that fails authorization rules is ignored (keep
    /// quarantined, retry later — fail closed).
    pub fn on_grant(&mut self, from: NodeId, grant: RejoinMessage) -> Vec<BootstrapEffect> {
        let RejoinMessage::Grant {
            cluster_id,
            term,
            log,
            commit: _,
            verified,
        } = grant
        else {
            return Vec::new();
        };
        // Wrong cluster: never adopt.
        if cluster_id != self.cluster {
            return Vec::new();
        }
        // Corruption evidence requires a verified source snapshot.
        if self.reasons.iter().any(|r| r.is_corruption_evidence()) && !verified {
            return Vec::new();
        }
        // Codex finding 1: refuse grants BELOW the untrusted term hint. A
        // grant below our (possible) prior term would regress the durable
        // term; a later VoteRequest at the prior term would then find a
        // blank vote and grant — a double vote in a term where our torn
        // record may already have granted someone. Refusal is safe in both
        // directions: a hint corrupted UPWARD only over-refuses (liveness:
        // we stay quarantined until a genuinely-current leader grants); a
        // hint corrupted DOWNWARD (or unreadable) leaves the residual that
        // Gate C incarnation fencing closes. The hint authorizes nothing.
        if let Some(hint) = self.term_hint {
            if term < hint {
                return Vec::new();
            }
        }
        // Adopt as a follower with `voted_for = Some(granting leader)` —
        // NOT `None`. The node's pre-quarantine vote in `term` is unknown
        // (its record was torn) and may have helped elect someone; a blank
        // vote would let it vote AGAIN in `term` — reopening exactly the R2
        // double-vote window that quarantine exists to close. Recording the
        // granting leader as our vote makes the node incapable of emitting
        // any NEW grant in `term`: the leader never re-campaigns in its own
        // term, and every other candidate is refused by the voted_for
        // check. (A stale-but-certificated leader granting an old-term
        // snapshot is likewise safe: the adopted term is behind, so first
        // contact with the live cluster raises it via normal term rules;
        // Phase B's quorum-managed incarnations subsume this construction.)
        vec![BootstrapEffect::AdoptSnapshot {
            cluster_id,
            hard: HardState {
                current_term: term,
                voted_for: Some(from),
            },
            log,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLUSTER: ClusterId = ClusterId(7);

    fn coherent() -> RecoveredState {
        RecoveredState {
            cluster_id: Some(CLUSTER),
            hard: Some(HardState {
                current_term: Term(3),
                voted_for: Some(NodeId(2)),
            }),
            log: Some(vec![LogEntry {
                term: Term(3),
                payload: 42,
            }]),
            commit_marker: 1,
            integrity: Integrity {
                hard_state_verified: true,
                log_verified: true,
            },
        }
    }

    #[test]
    fn blank_disk_boots_healthy_fresh() {
        match inspect(CLUSTER, &RecoveredState::blank()) {
            BootDecision::Healthy { hard, log } => {
                assert_eq!(hard, HardState::default());
                assert!(log.is_empty());
            }
            BootDecision::Quarantine { reasons, .. } => {
                panic!("fresh node quarantined: {reasons:?}")
            }
        }
    }

    #[test]
    fn coherent_state_boots_healthy_with_exact_state() {
        match inspect(CLUSTER, &coherent()) {
            BootDecision::Healthy { hard, log } => {
                assert_eq!(hard.voted_for, Some(NodeId(2)));
                assert_eq!(log.len(), 1);
            }
            BootDecision::Quarantine { reasons, .. } => panic!("quarantined: {reasons:?}"),
        }
    }

    #[test]
    fn torn_hard_state_quarantines() {
        let mut s = coherent();
        s.integrity.hard_state_verified = false;
        let BootDecision::Quarantine { reasons, .. } = inspect(CLUSTER, &s) else {
            panic!("torn state booted healthy");
        };
        assert!(reasons.contains(&QuarantineReason::TornHardState));
    }

    #[test]
    fn alien_cluster_quarantines() {
        let mut s = coherent();
        s.cluster_id = Some(ClusterId(99));
        let BootDecision::Quarantine { reasons, .. } = inspect(CLUSTER, &s) else {
            panic!("alien state booted healthy");
        };
        assert!(reasons.contains(&QuarantineReason::ClusterIdMismatch));
    }

    #[test]
    fn commit_marker_beyond_log_quarantines() {
        let mut s = coherent();
        s.commit_marker = 5; // log has 1 entry
        let BootDecision::Quarantine { reasons, .. } = inspect(CLUSTER, &s) else {
            panic!("frontier-beyond-data booted healthy");
        };
        assert!(reasons.contains(&QuarantineReason::CommitBeyondLog));
    }

    #[test]
    fn log_corruption_is_corruption_evidence_and_blocks_stale_reads() {
        let mut s = coherent();
        s.integrity.log_verified = false;
        let BootDecision::Quarantine { reasons, .. } = inspect(CLUSTER, &s) else {
            panic!("corrupt log booted healthy");
        };
        let node = QuarantinedNode::new(NodeId(3), CLUSTER, reasons, None);
        assert!(!node.stale_reads_allowed());
    }

    #[test]
    fn torn_metadata_still_allows_labeled_stale_reads() {
        // Consensus metadata torn, data verified → diagnostics + stale
        // reads OK, voting closed.
        let mut s = coherent();
        s.integrity.hard_state_verified = false;
        let BootDecision::Quarantine { reasons, .. } = inspect(CLUSTER, &s) else {
            panic!()
        };
        let node = QuarantinedNode::new(NodeId(3), CLUSTER, reasons, None);
        assert!(node.stale_reads_allowed());
    }

    #[test]
    fn corruption_requires_verified_grant() {
        let mut s = coherent();
        s.integrity.log_verified = false;
        let BootDecision::Quarantine { reasons, .. } = inspect(CLUSTER, &s) else {
            panic!()
        };
        let mut node = QuarantinedNode::new(NodeId(3), CLUSTER, reasons, None);
        let unverified = RejoinMessage::Grant {
            cluster_id: CLUSTER,
            term: Term(4),
            log: Vec::new(),
            commit: 0,
            verified: false,
        };
        assert!(
            node.on_grant(NodeId(1), unverified).is_empty(),
            "unverified grant accepted"
        );
        let verified = RejoinMessage::Grant {
            cluster_id: CLUSTER,
            term: Term(4),
            log: Vec::new(),
            commit: 0,
            verified: true,
        };
        assert!(
            !node.on_grant(NodeId(1), verified).is_empty(),
            "verified grant refused"
        );
    }

    #[test]
    fn wrong_cluster_grant_is_never_adopted() {
        let mut s = coherent();
        s.integrity.hard_state_verified = false;
        let BootDecision::Quarantine { reasons, .. } = inspect(CLUSTER, &s) else {
            panic!()
        };
        let mut node = QuarantinedNode::new(NodeId(3), CLUSTER, reasons, None);
        let alien = RejoinMessage::Grant {
            cluster_id: ClusterId(99),
            term: Term(4),
            log: Vec::new(),
            commit: 0,
            verified: true,
        };
        assert!(node.on_grant(NodeId(1), alien).is_empty());
    }

    #[test]
    fn preserve_happens_before_first_rejoin_request_and_alarm_fires_once() {
        let mut s = coherent();
        s.integrity.log_verified = false;
        let BootDecision::Quarantine { reasons, .. } = inspect(CLUSTER, &s) else {
            panic!()
        };
        let mut node = QuarantinedNode::new(NodeId(3), CLUSTER, reasons, None);
        let first = node.tick_rejoin(NodeId(1));
        assert!(matches!(first[0], BootstrapEffect::PreserveOldState));
        assert!(matches!(first[1], BootstrapEffect::Alarm { .. }));
        assert!(matches!(first[2], BootstrapEffect::Send { .. }));
        let second = node.tick_rejoin(NodeId(1));
        assert_eq!(second.len(), 1, "preserve/alarm must fire once: {second:?}");
        assert!(matches!(second[0], BootstrapEffect::Send { .. }));
    }
}
