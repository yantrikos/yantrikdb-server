//! YRP election safety core (RFC 028 v2 §2) — pure logic, no I/O.
//!
//! ## The two invariants this module owns
//!
//! **Gate A #1 — vote safety (scenario R2).** A node must never vote for two
//! candidates in one term, *including across crash/restart*. The classic bug:
//! decide to grant, send the response, crash before persisting `voted_for`,
//! restart with a blank vote, grant a different candidate in the same term →
//! two majorities. The fix is ordering: persist `(term, voted_for)` BEFORE the
//! response leaves the node. This module enforces the ordering **structurally**:
//! any decision that changes [`HardState`] returns
//! [`Effect::PersistHardState`] and *withholds* the outbound messages; they
//! are only released by [`ElectionCore::hard_state_persisted`]. There is no
//! API to obtain the withheld messages without asserting durability first.
//! A crash before persist loses the (unsent) messages — safe. A crash after
//! persist loses the response — a liveness hiccup, never a safety violation.
//!
//! **Gate A #3 — possibly-committed-suffix protection (scenario R3).**
//! Election freshness uses Raft's last-`(term, index)` comparison
//! ([`LogPosition::is_at_least_as_up_to_date_as`]), never a scalar watermark
//! and never a locally-known committed frontier. A voter refuses any candidate
//! whose log is less up to date than its own — which, combined with quorum
//! intersection, guarantees every possibly-committed entry survives elections.
//!
//! ## What is deliberately NOT here
//!
//! Log replication/commit (Phase A1b), quarantine + incarnation fencing
//! (Phase A2), membership changes (Phase B). The voter set is fixed at
//! construction. Timers are the driver's job — the core only reacts to
//! [`ElectionCore::on_election_timeout`]; randomization of timeouts is
//! driver-side. Pre-vote (§2, liveness only) never mutates state.

use std::collections::BTreeSet;

use super::types::{quorum, HardState, LogPosition, NodeId, Term};

/// Wire messages for the election layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    /// Liveness probe before a real election (never changes voter state).
    /// `term` is the term the candidate WOULD campaign at (current + 1).
    PreVoteRequest {
        term: Term,
        candidate: NodeId,
        last_log: LogPosition,
    },
    PreVoteResponse {
        term: Term,
        granted: bool,
    },
    VoteRequest {
        term: Term,
        candidate: NodeId,
        last_log: LogPosition,
    },
    VoteResponse {
        term: Term,
        granted: bool,
    },
    /// Minimal leader beacon so the sim can prove single-leader-per-term and
    /// step-down. The real data-plane append lands in Phase A1b.
    Heartbeat {
        term: Term,
        leader: NodeId,
    },
}

/// Instructions to the driver. Ordering contract: a
/// [`Effect::PersistHardState`] in a batch means the driver MUST durably
/// persist that state and then call
/// [`ElectionCore::hard_state_persisted`] to obtain the withheld sends —
/// the batch itself will contain no messages that depend on the new state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    /// Durably persist this hard state, then call `hard_state_persisted()`.
    PersistHardState(HardState),
    Send {
        to: NodeId,
        msg: Message,
    },
    /// Send to every voter except self (driver expands the fan-out).
    Broadcast {
        msg: Message,
    },
    BecameLeader {
        term: Term,
    },
    /// Left leadership (or candidacy) for `term` — driver stops leader duties.
    SteppedDown {
        term: Term,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Follower,
    PreCandidate,
    Candidate,
    Leader,
}

/// Hard state waiting to be persisted, plus the messages whose transmission
/// is gated on that durability (the R2 mechanism).
#[derive(Debug, Clone)]
struct PendingPersist {
    hard: HardState,
    held: Vec<Effect>,
}

/// The pure election state machine. See module docs for the contract.
pub struct ElectionCore {
    id: NodeId,
    voters: BTreeSet<NodeId>,
    /// Last DURABLY PERSISTED hard state — what survives a crash.
    persisted: HardState,
    /// In-flight persist + withheld messages. Lost on crash (by design:
    /// the sim models crash-restart as `restore(persisted)`).
    pending: Option<PendingPersist>,
    role: Role,
    /// Last position in the canonical prefix log (updated by the log layer;
    /// static within a single election round).
    last_log: LogPosition,
    votes: BTreeSet<NodeId>,
    pre_votes: BTreeSet<NodeId>,
    /// When false, `on_election_timeout` campaigns directly (used by sim
    /// tests that target the raw vote path). Production default: true.
    use_pre_vote: bool,
}

impl ElectionCore {
    /// `restored` is whatever hard state the boot path recovered. Phase A2's
    /// quarantine decides whether construction is allowed at all — a torn
    /// hard state must never reach this constructor (fail closed upstream).
    pub fn new(
        id: NodeId,
        voters: BTreeSet<NodeId>,
        restored: HardState,
        last_log: LogPosition,
        use_pre_vote: bool,
    ) -> Self {
        debug_assert!(voters.contains(&id), "a core's own id must be a voter");
        Self {
            id,
            voters,
            persisted: restored,
            pending: None,
            role: Role::Follower,
            last_log,
            votes: BTreeSet::new(),
            pre_votes: BTreeSet::new(),
            use_pre_vote,
        }
    }

    // ── accessors ──────────────────────────────────────────────────

    pub fn role(&self) -> Role {
        self.role
    }

    /// The effective (in-memory) term: pending-if-any, else persisted.
    pub fn current_term(&self) -> Term {
        self.effective().current_term
    }

    pub fn persisted_hard_state(&self) -> HardState {
        self.persisted
    }

    pub fn persist_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Update the last log position (called by the log layer as entries
    /// append). Static during a vote decision by construction: decisions are
    /// synchronous within `on_message`.
    pub fn set_last_log(&mut self, pos: LogPosition) {
        self.last_log = pos;
    }

    fn effective(&self) -> HardState {
        self.pending
            .as_ref()
            .map(|p| p.hard)
            .unwrap_or(self.persisted)
    }

    // ── the R2 gate ────────────────────────────────────────────────

    /// Stage a hard-state change. Messages that must not leave before the
    /// state is durable go through [`Self::hold`]. Coalescing is safe: a
    /// later, newer hard state subsumes an earlier one in the same pending
    /// window — after the (single) persist, every held message's implied
    /// state is ≤ the persisted state, and stale-term responses are ignored
    /// by receivers per the term rules.
    fn stage(&mut self, hard: HardState) -> Effect {
        match &mut self.pending {
            Some(p) => p.hard = hard,
            None => {
                self.pending = Some(PendingPersist {
                    hard,
                    held: Vec::new(),
                })
            }
        }
        Effect::PersistHardState(hard)
    }

    /// Withhold a message until the pending hard state is durable. Must only
    /// be called while a persist is pending (enforced by debug assert).
    fn hold(&mut self, eff: Effect) {
        debug_assert!(self.pending.is_some(), "hold() requires a staged persist");
        if let Some(p) = &mut self.pending {
            p.held.push(eff);
        }
    }

    /// The driver confirms the staged hard state is durable. Returns the
    /// withheld effects — the ONLY way they can be obtained (the R2 gate).
    pub fn hard_state_persisted(&mut self) -> Vec<Effect> {
        match self.pending.take() {
            Some(p) => {
                self.persisted = p.hard;
                p.held
            }
            None => Vec::new(),
        }
    }

    // ── events ─────────────────────────────────────────────────────

    /// Driver-decided election timeout (no leader contact for a randomized
    /// interval). Followers/pre-candidates start a pre-vote round (or campaign
    /// directly when pre-vote is disabled); candidates retry; leaders ignore.
    pub fn on_election_timeout(&mut self) -> Vec<Effect> {
        match self.role {
            Role::Leader => Vec::new(),
            _ if self.use_pre_vote => self.start_pre_vote(),
            _ => self.start_campaign(),
        }
    }

    fn start_pre_vote(&mut self) -> Vec<Effect> {
        self.role = Role::PreCandidate;
        self.pre_votes.clear();
        self.pre_votes.insert(self.id); // we would vote for ourselves
        let msg = Message::PreVoteRequest {
            term: self.current_term().next(),
            candidate: self.id,
            last_log: self.last_log,
        };
        if self.pre_quorum_reached() {
            // Single-voter cluster: pre-vote trivially passes.
            return self.start_campaign();
        }
        vec![Effect::Broadcast { msg }]
    }

    /// Begin a real campaign: bump term, vote for self. Both the self-vote
    /// and the outgoing VoteRequests are gated behind the persist — sending
    /// a VoteRequest at term T implies "I have voted for myself in T"; if
    /// that isn't durable, a crash could let this node vote differently in T
    /// (self-inflicted R2).
    fn start_campaign(&mut self) -> Vec<Effect> {
        self.role = Role::Candidate;
        self.votes.clear();
        self.votes.insert(self.id);
        let term = self.current_term().next();
        let persist = self.stage(HardState {
            current_term: term,
            voted_for: Some(self.id),
        });
        let req = Message::VoteRequest {
            term,
            candidate: self.id,
            last_log: self.last_log,
        };
        if self.vote_quorum_reached() {
            // Single-voter cluster: leader immediately — but only after the
            // self-vote is durable, so leadership is also held.
            self.role = Role::Leader;
            self.hold(Effect::BecameLeader { term });
            return vec![persist];
        }
        self.hold(Effect::Broadcast { msg: req });
        vec![persist]
    }

    /// Handle an incoming message. `leader_recently_seen` is driver-supplied
    /// leader-stickiness (a voter that has recent leader contact refuses
    /// pre-votes, damping disruption from flapping/partitioned nodes — the
    /// .140 lesson). It plays no role in REAL vote decisions: only terms,
    /// votes, and log freshness decide those.
    pub fn on_message(
        &mut self,
        from: NodeId,
        msg: Message,
        leader_recently_seen: bool,
    ) -> Vec<Effect> {
        match msg {
            Message::PreVoteRequest {
                term,
                candidate,
                last_log,
            } => self.on_pre_vote_request(from, term, candidate, last_log, leader_recently_seen),
            Message::PreVoteResponse { term, granted } => {
                self.on_pre_vote_response(from, term, granted)
            }
            Message::VoteRequest {
                term,
                candidate,
                last_log,
            } => self.on_vote_request(from, term, candidate, last_log),
            Message::VoteResponse { term, granted } => self.on_vote_response(from, term, granted),
            Message::Heartbeat { term, leader } => self.on_heartbeat(from, term, leader),
        }
    }

    /// Pre-vote: pure read, never persists, never changes state (that is the
    /// point — a partitioned node probing forever must not burn terms).
    fn on_pre_vote_request(
        &mut self,
        from: NodeId,
        term: Term,
        _candidate: NodeId,
        last_log: LogPosition,
        leader_recently_seen: bool,
    ) -> Vec<Effect> {
        let grant = !leader_recently_seen
            && term > self.current_term()
            && last_log.is_at_least_as_up_to_date_as(&self.last_log);
        vec![Effect::Send {
            to: from,
            msg: Message::PreVoteResponse {
                term,
                granted: grant,
            },
        }]
    }

    fn on_pre_vote_response(&mut self, from: NodeId, term: Term, granted: bool) -> Vec<Effect> {
        if self.role != Role::PreCandidate || term != self.current_term().next() || !granted {
            return Vec::new();
        }
        if self.voters.contains(&from) {
            self.pre_votes.insert(from);
        }
        if self.pre_quorum_reached() {
            return self.start_campaign();
        }
        Vec::new()
    }

    /// The real vote — where Gate A invariants 1 and 3 live.
    fn on_vote_request(
        &mut self,
        from: NodeId,
        term: Term,
        candidate: NodeId,
        last_log: LogPosition,
    ) -> Vec<Effect> {
        let mut effects = Vec::new();
        let cur = self.current_term();

        // Stale-term request: refuse immediately (no state change → no gate).
        if term < cur {
            return vec![Effect::Send {
                to: from,
                msg: Message::VoteResponse {
                    term: cur,
                    granted: false,
                },
            }];
        }

        // Newer term: step down from any candidacy/leadership. The term
        // adoption itself is folded into the SINGLE persist below — emitting
        // a separate PersistHardState for it would create an intermediate
        // durable state and split one vote decision across two persist
        // boundaries (a crash between them would adopt the term but lose the
        // vote). One decision, one persist.
        if term > cur {
            if matches!(self.role, Role::Leader | Role::Candidate) {
                effects.push(Effect::SteppedDown { term: cur });
            }
            self.role = Role::Follower;
        }

        // Vote decision under `term`. A newer term resets voted_for to None.
        let voted_for_in_term = if term > cur {
            None
        } else {
            self.effective().voted_for
        };
        let may_vote = voted_for_in_term.is_none() || voted_for_in_term == Some(candidate);
        // Gate A #3 (R3): Raft freshness — never a watermark, never a
        // committed frontier. Protects possibly-committed suffixes.
        let fresh = last_log.is_at_least_as_up_to_date_as(&self.last_log);

        let refusal = Effect::Send {
            to: from,
            msg: Message::VoteResponse {
                term,
                granted: false,
            },
        };

        if may_vote && fresh {
            // Gate A #1 (R2): the grant changes (term, voted_for) → ONE stage
            // of the FINAL state, and HOLD the response. It cannot leave this
            // node before the vote is durable.
            effects.push(self.stage(HardState {
                current_term: term,
                voted_for: Some(candidate),
            }));
            self.hold(Effect::Send {
                to: from,
                msg: Message::VoteResponse {
                    term,
                    granted: true,
                },
            });
        } else if term > cur {
            // Refusing, but we still adopt the newer term — one persist of
            // {term, None}, with the refusal held behind it so no peer sees
            // our new term before it is durable.
            effects.push(self.stage(HardState {
                current_term: term,
                voted_for: None,
            }));
            self.hold(refusal);
        } else {
            // Same term, already voted (or stale log): pure refusal, no persist.
            effects.push(refusal);
        }
        effects
    }

    fn on_vote_response(&mut self, from: NodeId, term: Term, granted: bool) -> Vec<Effect> {
        let cur = self.current_term();
        if term > cur {
            // Someone is ahead of us: adopt and step down.
            let mut effects = Vec::new();
            if matches!(self.role, Role::Leader | Role::Candidate) {
                effects.push(Effect::SteppedDown { term: cur });
            }
            self.role = Role::Follower;
            effects.push(self.stage(HardState {
                current_term: term,
                voted_for: None,
            }));
            return effects;
        }
        if self.role != Role::Candidate || term != cur || !granted {
            return Vec::new();
        }
        if self.voters.contains(&from) {
            self.votes.insert(from);
        }
        if self.vote_quorum_reached() {
            self.role = Role::Leader;
            return vec![
                Effect::BecameLeader { term: cur },
                Effect::Broadcast {
                    msg: Message::Heartbeat {
                        term: cur,
                        leader: self.id,
                    },
                },
            ];
        }
        Vec::new()
    }

    fn on_heartbeat(&mut self, _from: NodeId, term: Term, _leader: NodeId) -> Vec<Effect> {
        let cur = self.current_term();
        if term < cur {
            return Vec::new(); // stale leader; data plane will reject too
        }
        let mut effects = Vec::new();
        if matches!(self.role, Role::Leader | Role::Candidate) && term >= cur {
            effects.push(Effect::SteppedDown { term: cur });
        }
        self.role = Role::Follower;
        if term > cur {
            effects.push(self.stage(HardState {
                current_term: term,
                voted_for: None,
            }));
        }
        effects
    }

    fn pre_quorum_reached(&self) -> bool {
        self.pre_votes.len() >= quorum(self.voters.len())
    }

    fn vote_quorum_reached(&self) -> bool {
        self.votes.len() >= quorum(self.voters.len())
    }
}
