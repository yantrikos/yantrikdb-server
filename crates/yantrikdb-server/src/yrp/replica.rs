//! YRP replica safety core (RFC 028 v2 §2) — election + log replication,
//! pure logic, no I/O.
//!
//! Phase A1 delivered the election layer; Phase A1b (this revision) adds the
//! canonical prefix log, append continuity, conflict truncation, per-write
//! quorum confirmation, and the current-term commit rule. One state machine
//! owns all of it — split cores sharing term/role state is where
//! synchronization bugs live.
//!
//! ## The invariants this module owns
//!
//! **Gate A #1 — vote safety (scenario R2).** A node must never vote for two
//! candidates in one term, including across crash/restart. Enforced
//! structurally: any decision that changes durable state returns
//! [`Effect::Persist`] and *withholds* dependent messages; they are only
//! released by [`ReplicaCore::state_persisted`]. There is no API to obtain
//! the withheld messages without asserting durability first.
//!
//! **Gate A #2 — authority safety (scenario R1).** No two nodes may both
//! report durable success for conflicting writes. Mechanism: per-write quorum
//! confirmation. An entry is committed only when a quorum of voters has
//! *durably* accepted it (each acceptor's success `AppendResponse` is gated
//! behind its own persist — the same withhold mechanism), and only if the
//! entry is from the leader's **current term** (Raft §5.4.2: prior-term
//! entries commit solely by implication, via the no-op a new leader appends
//! on election). Term fencing rejects stale leaders at every acceptor; no
//! wall-clock assumption is load-bearing.
//!
//! **Gate A #3 — possibly-committed-suffix protection (scenario R3).**
//! Election freshness is Raft's last-`(term, index)` comparison over the
//! candidate's log — never a scalar watermark, never a committed frontier.
//!
//! ## Durability model
//!
//! [`Effect::Persist`] carries a full snapshot `(hard state, log)`. The pure
//! core favors an unarguable contract over an efficient encoding; the
//! production driver will persist deltas (append/truncate) with identical
//! semantics — the sim proves the semantics, the driver owns the encoding.
//! The driver persists the snapshot, then calls
//! [`ReplicaCore::state_persisted`] to obtain the withheld messages.
//! A crash before persist loses unsent messages (safe); a crash after
//! persist loses only responses (liveness, never safety).
//!
//! ## What is deliberately NOT here (yet)
//!
//! Quarantine + incarnation fencing (Phase A2); membership changes, witness
//! ack-exclusion (witnesses vote but never count toward data durability),
//! snapshots/GC, and the real oplog payload binding (Phase B — `payload` is
//! an opaque `u64` until then). The voter set is fixed at construction.
//! Timers are the driver's job.

use std::collections::{BTreeMap, BTreeSet};

use super::types::{quorum, HardState, LogPosition, NodeId, Term};

/// Payload reserved for the no-op entry a new leader appends on election —
/// the §5.4.2 mechanism that lets prior-term entries commit by implication.
pub const NOOP_PAYLOAD: u64 = 0;

/// One canonical-log entry. `payload` is an opaque identifier in Phase A1b;
/// the memory-native oplog op (embedding bytes, provenance, HLC) binds here
/// in Phase B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogEntry {
    pub term: Term,
    pub payload: u64,
}

/// Wire messages for the replica layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    /// Liveness probe before a real election (never changes voter state).
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
    /// Log replication + heartbeat (empty `entries`). `prev` is the position
    /// immediately before `entries`; `commit` is the leader's commit index.
    AppendEntries {
        term: Term,
        leader: NodeId,
        prev: LogPosition,
        entries: Vec<LogEntry>,
        commit: u64,
    },
    /// `last_index` is the acceptor's last DURABLE matching index on
    /// success — the per-write quorum-confirmation signal. A success
    /// response for newly appended entries is sent only after they are
    /// persisted (gated), which is what makes counting it safe.
    AppendResponse {
        term: Term,
        success: bool,
        last_index: u64,
    },
}

/// Instructions to the driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    /// Durably persist this `(hard, log)` snapshot, then call
    /// [`ReplicaCore::state_persisted`]. Multiple `Persist` effects in one
    /// batch coalesce: persisting the last one seen satisfies all of them.
    Persist {
        hard: HardState,
        log: Vec<LogEntry>,
    },
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
    SteppedDown {
        term: Term,
    },
    /// The commit index advanced to `to` (inclusive). The driver applies
    /// entries `(previous commit, to]` to the state machine, in order.
    CommitAdvanced {
        to: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Follower,
    PreCandidate,
    Candidate,
    Leader,
}

/// State staged for persistence, plus messages gated on that durability.
#[derive(Debug, Clone)]
struct Pending {
    hard: HardState,
    held: Vec<Effect>,
}

/// The pure replica state machine. See module docs for the contract.
pub struct ReplicaCore {
    id: NodeId,
    voters: BTreeSet<NodeId>,
    /// Last DURABLY PERSISTED hard state — survives a crash.
    persisted_hard: HardState,
    /// Durable log length (entries `1..=durable_len` survive a crash).
    durable_len: u64,
    /// The in-memory canonical log, 1-indexed (`log[0]` ↔ index 1).
    log: Vec<LogEntry>,
    /// In-flight persist + withheld messages. Lost on crash.
    pending: Option<Pending>,
    role: Role,
    /// Volatile commit index (re-learned from the leader after restart).
    commit: u64,
    /// Leader bookkeeping: highest DURABLY-acked index per voter.
    match_index: BTreeMap<NodeId, u64>,
    /// Leader bookkeeping: next index to send per voter.
    next_index: BTreeMap<NodeId, u64>,
    votes: BTreeSet<NodeId>,
    pre_votes: BTreeSet<NodeId>,
    use_pre_vote: bool,
}

impl ReplicaCore {
    /// `restored_hard` / `restored_log` are what the boot path recovered —
    /// exactly what the last completed [`Effect::Persist`] wrote. Phase A2's
    /// quarantine decides whether construction is allowed at all; torn state
    /// must never reach this constructor (fail closed upstream).
    pub fn new(
        id: NodeId,
        voters: BTreeSet<NodeId>,
        restored_hard: HardState,
        restored_log: Vec<LogEntry>,
        use_pre_vote: bool,
    ) -> Self {
        debug_assert!(voters.contains(&id), "a core's own id must be a voter");
        let durable_len = restored_log.len() as u64;
        Self {
            id,
            voters,
            persisted_hard: restored_hard,
            durable_len,
            log: restored_log,
            pending: None,
            role: Role::Follower,
            commit: 0,
            match_index: BTreeMap::new(),
            next_index: BTreeMap::new(),
            votes: BTreeSet::new(),
            pre_votes: BTreeSet::new(),
            use_pre_vote,
        }
    }

    // ── accessors ──────────────────────────────────────────────────

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn current_term(&self) -> Term {
        self.effective().current_term
    }

    pub fn persisted_hard_state(&self) -> HardState {
        self.persisted_hard
    }

    pub fn commit_index(&self) -> u64 {
        self.commit
    }

    pub fn log_len(&self) -> u64 {
        self.log.len() as u64
    }

    /// Entry at 1-based `index`, if present.
    pub fn entry(&self, index: u64) -> Option<&LogEntry> {
        if index == 0 {
            return None;
        }
        self.log.get(index as usize - 1)
    }

    /// Position of the last in-memory log entry (sentinel ZERO when empty).
    /// Used for election freshness; in-memory-ahead-of-durable only makes a
    /// voter STRICTER (it may refuse an equally-fresh candidate — liveness,
    /// never safety), and grants persist before leaving anyway.
    pub fn last_log(&self) -> LogPosition {
        match self.log.last() {
            Some(e) => LogPosition {
                term: e.term.0,
                index: self.log.len() as u64,
            },
            None => LogPosition::ZERO,
        }
    }

    fn effective(&self) -> HardState {
        self.pending
            .as_ref()
            .map(|p| p.hard)
            .unwrap_or(self.persisted_hard)
    }

    // ── the durability gate (R2 + per-write quorum confirmation) ──

    /// Stage durable state. Emits a `Persist` snapshot of the CURRENT
    /// `(hard, log)`; messages that must not leave before durability go
    /// through [`Self::hold`]. Coalescing within one pending window is safe:
    /// the persisted snapshot is always ≥ every held message's implied state.
    fn stage(&mut self, hard: HardState) -> Effect {
        match &mut self.pending {
            Some(p) => p.hard = hard,
            None => {
                self.pending = Some(Pending {
                    hard,
                    held: Vec::new(),
                })
            }
        }
        Effect::Persist {
            hard,
            log: self.log.clone(),
        }
    }

    /// Stage the current log for persistence without a hard-state change.
    fn stage_log(&mut self) -> Effect {
        let hard = self.effective();
        self.stage(hard)
    }

    fn hold(&mut self, eff: Effect) {
        debug_assert!(self.pending.is_some(), "hold() requires a staged persist");
        if let Some(p) = &mut self.pending {
            p.held.push(eff);
        }
    }

    /// If a persist is pending, hold `eff` behind it; otherwise emit it now.
    fn hold_or(&mut self, eff: Effect, out: &mut Vec<Effect>) {
        if self.pending.is_some() {
            self.hold(eff);
        } else {
            out.push(eff);
        }
    }

    /// The driver confirms the staged snapshot is durable. Returns withheld
    /// messages — the ONLY way to obtain them. On a leader, self-acceptance
    /// becomes countable here (its own log is now durable), so the commit
    /// index may advance as part of the flush.
    pub fn state_persisted(&mut self) -> Vec<Effect> {
        let mut out = match self.pending.take() {
            Some(p) => {
                self.persisted_hard = p.hard;
                self.durable_len = self.log.len() as u64;
                p.held
            }
            None => return Vec::new(),
        };
        if self.role == Role::Leader {
            self.match_index.insert(self.id, self.durable_len);
            self.try_advance_commit(&mut out);
        }
        out
    }

    // ── client-facing (leader) ─────────────────────────────────────

    /// Propose a new entry. Leader-only; returns `None` otherwise (the
    /// caller redirects to the leader — the API layer's job in Phase B).
    pub fn propose(&mut self, payload: u64) -> Option<Vec<Effect>> {
        if self.role != Role::Leader {
            return None;
        }
        let term = self.current_term();
        self.log.push(LogEntry { term, payload });
        let mut out = vec![self.stage_log()];
        // Fan out eagerly; success responses confirm durable acceptance.
        self.append_effects_for_followers(&mut out);
        Some(out)
    }

    /// Leader heartbeat tick (driver timer): send each follower its next
    /// entries (empty AppendEntries when caught up). Also how stragglers
    /// catch up after drops.
    pub fn tick_heartbeat(&mut self) -> Vec<Effect> {
        if self.role != Role::Leader {
            return Vec::new();
        }
        let mut out = Vec::new();
        self.append_effects_for_followers(&mut out);
        out
    }

    fn append_effects_for_followers(&mut self, out: &mut Vec<Effect>) {
        let peers: Vec<NodeId> = self
            .voters
            .iter()
            .copied()
            .filter(|p| *p != self.id)
            .collect();
        for peer in peers {
            self.send_append_to(peer, out);
        }
    }

    fn send_append_to(&mut self, peer: NodeId, out: &mut Vec<Effect>) {
        let term = self.current_term();
        let next = *self.next_index.get(&peer).unwrap_or(&1);
        let prev_index = next.saturating_sub(1);
        let prev = if prev_index == 0 {
            LogPosition::ZERO
        } else {
            match self.entry(prev_index) {
                Some(e) => LogPosition {
                    term: e.term.0,
                    index: prev_index,
                },
                None => LogPosition::ZERO,
            }
        };
        let entries: Vec<LogEntry> = self.log.iter().skip(prev_index as usize).copied().collect();
        let msg = Message::AppendEntries {
            term,
            leader: self.id,
            prev,
            entries,
            commit: self.commit,
        };
        // If our own persist of these entries is still pending, hold: a
        // follower must never hold entries our own disk lacks (its durable
        // ack could otherwise out-run the leader's durability).
        self.hold_or(Effect::Send { to: peer, msg }, out);
    }

    // ── events ─────────────────────────────────────────────────────

    pub fn on_election_timeout(&mut self) -> Vec<Effect> {
        match self.role {
            Role::Leader => Vec::new(),
            _ if self.use_pre_vote => self.start_pre_vote(),
            _ => self.start_campaign(),
        }
    }

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
            Message::AppendEntries {
                term,
                leader,
                prev,
                entries,
                commit,
            } => self.on_append_entries(from, term, leader, prev, entries, commit),
            Message::AppendResponse {
                term,
                success,
                last_index,
            } => self.on_append_response(from, term, success, last_index),
        }
    }

    // ── election (semantics unchanged from Phase A1) ───────────────

    fn start_pre_vote(&mut self) -> Vec<Effect> {
        self.role = Role::PreCandidate;
        self.pre_votes.clear();
        self.pre_votes.insert(self.id);
        if self.pre_quorum_reached() {
            return self.start_campaign();
        }
        vec![Effect::Broadcast {
            msg: Message::PreVoteRequest {
                term: self.current_term().next(),
                candidate: self.id,
                last_log: self.last_log(),
            },
        }]
    }

    fn start_campaign(&mut self) -> Vec<Effect> {
        self.role = Role::Candidate;
        self.votes.clear();
        self.votes.insert(self.id);
        let term = self.current_term().next();
        let persist = self.stage(HardState {
            current_term: term,
            voted_for: Some(self.id),
        });
        if self.vote_quorum_reached() {
            // Single-voter cluster: leadership (and its no-op) is held
            // behind the self-vote persist.
            self.become_leader_held(term);
            return vec![persist];
        }
        self.hold(Effect::Broadcast {
            msg: Message::VoteRequest {
                term,
                candidate: self.id,
                last_log: self.last_log(),
            },
        });
        vec![persist]
    }

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
            && last_log.is_at_least_as_up_to_date_as(&self.last_log());
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

    fn on_vote_request(
        &mut self,
        from: NodeId,
        term: Term,
        candidate: NodeId,
        last_log: LogPosition,
    ) -> Vec<Effect> {
        let mut effects = Vec::new();
        let cur = self.current_term();
        if term < cur {
            return vec![Effect::Send {
                to: from,
                msg: Message::VoteResponse {
                    term: cur,
                    granted: false,
                },
            }];
        }
        // ONE decision → ONE persist. Deciding the vote and adopting the
        // term must land in a single staged hard state: a crash between a
        // term-adoption persist and a vote persist would durably adopt the
        // term while forgetting the vote — the split-persist bug the sim
        // caught in Phase A1 (and again in A1b review). Compute the final
        // hard state first, stage exactly once.
        let newer = term > cur;
        if newer {
            if matches!(self.role, Role::Leader | Role::Candidate) {
                effects.push(Effect::SteppedDown { term: cur });
            }
            self.role = Role::Follower;
        }
        // A vote from an older term never constrains a newer term.
        let prior_vote = if newer {
            None
        } else {
            self.effective().voted_for
        };
        let may_vote = prior_vote.is_none() || prior_vote == Some(candidate);
        // Gate A #3 (R3): Raft freshness over the candidate's log.
        let fresh = last_log.is_at_least_as_up_to_date_as(&self.last_log());
        if may_vote && fresh {
            // Gate A #1 (R2): the grant changes voted_for → stage + HOLD.
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
        } else {
            let refusal = Effect::Send {
                to: from,
                msg: Message::VoteResponse {
                    term,
                    granted: false,
                },
            };
            if newer {
                // Refusal, but the term adoption still needs durability;
                // sequence the refusal behind that single persist.
                effects.push(self.stage(HardState {
                    current_term: term,
                    voted_for: None,
                }));
                self.hold(refusal);
            } else {
                self.hold_or(refusal, &mut effects);
            }
        }
        effects
    }

    fn on_vote_response(&mut self, from: NodeId, term: Term, granted: bool) -> Vec<Effect> {
        let cur = self.current_term();
        if term > cur {
            return self.step_down_to(term, cur);
        }
        if self.role != Role::Candidate || term != cur || !granted {
            return Vec::new();
        }
        if self.voters.contains(&from) {
            self.votes.insert(from);
        }
        if self.vote_quorum_reached() {
            let mut out = Vec::new();
            self.become_leader(cur, &mut out);
            return out;
        }
        Vec::new()
    }

    // ── log replication (Phase A1b) ────────────────────────────────

    fn on_append_entries(
        &mut self,
        from: NodeId,
        term: Term,
        _leader: NodeId,
        prev: LogPosition,
        entries: Vec<LogEntry>,
        leader_commit: u64,
    ) -> Vec<Effect> {
        let mut effects = Vec::new();
        let cur = self.current_term();

        // Term fencing: refuse a stale leader with our newer term — the
        // response is what fences it (it steps down on receipt).
        if term < cur {
            return vec![Effect::Send {
                to: from,
                msg: Message::AppendResponse {
                    term: cur,
                    success: false,
                    last_index: self.durable_len,
                },
            }];
        }

        // Valid leadership for `term`: recognize it. Term adoption is NOT
        // staged here — one decision → one persist: the final staged hard
        // state (below, per path) carries the adopted term, so a crash can
        // never durably adopt the term while losing the accompanying
        // log/vote decision (the A1 split-persist lesson).
        if matches!(self.role, Role::Leader | Role::Candidate) {
            effects.push(Effect::SteppedDown { term: cur });
        }
        self.role = Role::Follower;
        let newer = term > cur;
        let adopted = if newer {
            HardState {
                current_term: term,
                voted_for: None,
            }
        } else {
            self.effective()
        };

        // Continuity check at `prev` (the append-side of Gate A #2: a hole
        // or a term mismatch means these entries do not extend our history —
        // refuse; the leader backs up).
        let prev_ok = prev.index == 0
            || self
                .entry(prev.index)
                .is_some_and(|e| e.term.0 == prev.term);
        if !prev_ok {
            let refusal = Effect::Send {
                to: from,
                msg: Message::AppendResponse {
                    term,
                    success: false,
                    last_index: self.durable_len.min(prev.index.saturating_sub(1)),
                },
            };
            if newer {
                effects.push(self.stage(adopted));
                self.hold(refusal);
            } else {
                self.hold_or(refusal, &mut effects);
            }
            return effects;
        }

        // Append, truncating any conflicting (necessarily uncommitted)
        // suffix. The committed prefix is never truncated: a conflict at or
        // below `commit` is protocol corruption — fail closed rather than
        // destroy applied history (Phase A2 turns this into quarantine).
        let mut idx = prev.index;
        let mut changed = false;
        for e in &entries {
            idx += 1;
            match self.entry(idx) {
                Some(existing) if *existing == *e => {}
                Some(_) => {
                    if idx <= self.commit {
                        debug_assert!(false, "conflict at/below commit index — corruption");
                        let refusal = Effect::Send {
                            to: from,
                            msg: Message::AppendResponse {
                                term,
                                success: false,
                                last_index: self.durable_len,
                            },
                        };
                        if newer {
                            effects.push(self.stage(adopted));
                            self.hold(refusal);
                        } else {
                            self.hold_or(refusal, &mut effects);
                        }
                        return effects;
                    }
                    self.log.truncate(idx as usize - 1);
                    self.log.push(*e);
                    changed = true;
                }
                None => {
                    self.log.push(*e);
                    changed = true;
                }
            }
        }
        let covered = idx.max(prev.index);

        if changed || newer {
            // Durable acceptance: the success response — our quorum-
            // confirmation vote for these entries — is gated on OUR persist
            // of BOTH the entries and the (possibly adopted) term, staged
            // as one snapshot.
            effects.push(self.stage(adopted));
            self.hold(Effect::Send {
                to: from,
                msg: Message::AppendResponse {
                    term,
                    success: true,
                    last_index: if changed {
                        covered
                    } else {
                        self.durable_len.min(covered)
                    },
                },
            });
        } else {
            // Heartbeat / duplicate at the current term: durable state
            // already covers it.
            let ack = Effect::Send {
                to: from,
                msg: Message::AppendResponse {
                    term,
                    success: true,
                    last_index: self.durable_len.min(covered),
                },
            };
            self.hold_or(ack, &mut effects);
        }

        // Adopt the leader's commit proof, bounded by what we hold.
        let new_commit = leader_commit.min(covered);
        if new_commit > self.commit {
            self.commit = new_commit;
            // Sequence apply behind the persist: apply never outruns
            // durability.
            let advanced = Effect::CommitAdvanced { to: new_commit };
            self.hold_or(advanced, &mut effects);
        }
        effects
    }

    fn on_append_response(
        &mut self,
        from: NodeId,
        term: Term,
        success: bool,
        last_index: u64,
    ) -> Vec<Effect> {
        let cur = self.current_term();
        if term > cur {
            return self.step_down_to(term, cur);
        }
        if self.role != Role::Leader || term != cur {
            return Vec::new();
        }
        let mut out = Vec::new();
        if success {
            let m = self.match_index.entry(from).or_insert(0);
            *m = (*m).max(last_index);
            self.next_index.insert(from, last_index + 1);
            self.try_advance_commit(&mut out);
        } else {
            // Back up toward the follower's durable frontier and retry.
            let next = self
                .next_index
                .get(&from)
                .copied()
                .unwrap_or(self.log_len() + 1);
            let backed = next.saturating_sub(1).clamp(1, last_index + 1);
            self.next_index.insert(from, backed);
            self.send_append_to(from, &mut out);
        }
        out
    }

    /// The commit rule (Gate A #2): the largest `n` such that a quorum of
    /// voters has DURABLY accepted through `n` and `log[n].term` is the
    /// CURRENT term. Prior-term entries never commit by counting — only by
    /// implication under a committed current-term entry (the no-op
    /// guarantees one exists). Phase B narrows the counted set to
    /// data-bearing voters (witness exclusion).
    fn try_advance_commit(&mut self, out: &mut Vec<Effect>) {
        let cur = self.current_term();
        let mut n = self.log_len();
        while n > self.commit {
            if self.entry(n).map(|e| e.term) == Some(cur) {
                let acks = self
                    .voters
                    .iter()
                    .filter(|v| self.match_index.get(v).copied().unwrap_or(0) >= n)
                    .count();
                if acks >= quorum(self.voters.len()) {
                    self.commit = n;
                    let advanced = Effect::CommitAdvanced { to: n };
                    self.hold_or(advanced, out);
                    // Share the new commit index promptly.
                    self.append_effects_for_followers(out);
                    return;
                }
            }
            n -= 1;
        }
    }

    // ── shared transitions ─────────────────────────────────────────

    fn become_leader(&mut self, term: Term, out: &mut Vec<Effect>) {
        self.init_leader_state(term);
        out.push(Effect::BecameLeader { term });
        // §5.4.2 no-op: gives the new term an entry so the prior-term
        // suffix can commit by implication.
        self.log.push(LogEntry {
            term,
            payload: NOOP_PAYLOAD,
        });
        out.push(self.stage_log());
        self.append_effects_for_followers(out);
    }

    /// Single-voter-cluster variant: everything (leadership announcement,
    /// no-op, fan-out) rides behind the already-staged self-vote persist.
    fn become_leader_held(&mut self, term: Term) {
        debug_assert!(self.pending.is_some());
        self.init_leader_state(term);
        self.hold(Effect::BecameLeader { term });
        self.log.push(LogEntry {
            term,
            payload: NOOP_PAYLOAD,
        });
        let _ = self.stage_log(); // coalesces into the pending persist
    }

    fn init_leader_state(&mut self, _term: Term) {
        self.role = Role::Leader;
        let last = self.log_len();
        self.next_index.clear();
        self.match_index.clear();
        for v in self.voters.iter().copied() {
            self.next_index.insert(v, last + 1);
            self.match_index.insert(v, 0);
        }
        self.match_index.insert(self.id, self.durable_len);
    }

    fn step_down_to(&mut self, newer: Term, cur: Term) -> Vec<Effect> {
        let mut effects = Vec::new();
        if matches!(self.role, Role::Leader | Role::Candidate) {
            effects.push(Effect::SteppedDown { term: cur });
        }
        self.role = Role::Follower;
        effects.push(self.stage(HardState {
            current_term: newer,
            voted_for: None,
        }));
        effects
    }

    fn pre_quorum_reached(&self) -> bool {
        self.pre_votes.len() >= quorum(self.voters.len())
    }

    fn vote_quorum_reached(&self) -> bool {
        self.votes.len() >= quorum(self.voters.len())
    }
}
