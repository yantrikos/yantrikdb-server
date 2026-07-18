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

/// One canonical-log entry. `payload` is an opaque identifier until the
/// memory-native oplog op (embedding bytes, provenance, HLC) binds here.
///
/// `key` (Phase B, RFC 028 §7): the idempotency claim, carried IN the entry
/// so claim and op are one atomic replicated unit — committed together,
/// truncated together, snapshot together. This is what makes the two
/// crash-scenario halves (pre-registered by the trading workspace, who
/// converged on the identical invariant independently) hold by
/// construction: a committed-but-unacked keyed entry survives failover and
/// dedupes the retry (claim never lost after commit); a tentative keyed
/// entry truncates WITH its claim, so the retry re-executes cleanly (no
/// settled-claim-without-effect ghost).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogEntry {
    pub term: Term,
    pub payload: u64,
    pub key: Option<u64>,
}

impl LogEntry {
    /// Unkeyed entry (the common case; also every pre-Phase-B test entry).
    pub fn unkeyed(term: Term, payload: u64) -> Self {
        Self {
            term,
            payload,
            key: None,
        }
    }
}

/// Outcome of a keyed proposal at the leader's origin ingress (RFC 028 §7:
/// claims are checked at ORIGIN, carried in the log, and never re-gated at
/// apply).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyedProposal {
    /// Fresh claim: the entry was appended at `index`; effects carry the
    /// persist + fan-out. The caller acks its client only when the commit
    /// index reaches `index` (quorum-durable tier — never before).
    Appended { index: u64, effects: Vec<Effect> },
    /// The key already claims `index`, and that entry is COMMITTED: a
    /// dedupe hit. The caller answers the retry with the ORIGINAL entry's
    /// outcome immediately — this is the "committed but client never
    /// learned" failover half.
    DuplicateCommitted { index: u64 },
    /// The key claims `index` but that entry is NOT yet committed (in
    /// flight, possibly from this same client's earlier attempt). The
    /// caller waits for commit (or its loss by truncation) — it must NOT
    /// append a second entry, and must NOT report success yet.
    DuplicatePending { index: u64 },
}

/// A log-compaction snapshot (RFC 028 §6, pure-model form). The production
/// manifest binds an engine checkpoint (content hash, membership epoch,
/// schema/capability/generation state); the pure core carries what the
/// PROTOCOL needs:
/// - `last`: the exact frontier the snapshot covers (entries ≤ last.index
///   are gone from the log);
/// - `claims`: every idempotency claim at or below the frontier — sol
///   P1-9's rule made structural: compaction may never create a window
///   where a GC'd claim can be replayed, so the claims RIDE the snapshot
///   exactly as they rode the entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub last: LogPosition,
    pub claims: BTreeMap<u64, u64>,
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
    /// Snapshot install (leader → straggler whose next index fell below
    /// the leader's compaction base). Acked with a normal AppendResponse
    /// at `snapshot.last.index` once durable.
    InstallSnapshot {
        term: Term,
        leader: NodeId,
        snapshot: Snapshot,
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
        base: LogPosition,
        log: Vec<LogEntry>,
        claims: BTreeMap<u64, u64>,
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
    /// A snapshot was adopted: the driver replaces its applied state with
    /// the checkpoint at `last_index` (no per-entry replay — the entries
    /// are gone; the state arrives wholesale, like A2's rejoin adoption).
    InstallState {
        last_index: u64,
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
    /// Compaction base (RFC 028 §6): the last position covered by the
    /// adopted/created snapshot. Entries ≤ base.index are compacted away;
    /// `log[0]` holds absolute index `base.index + 1`. ZERO = uncompacted.
    base: LogPosition,
    /// The in-memory canonical log SUFFIX above `base`.
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
    /// Phase B (RFC 028 §7): key → log index of the entry claiming it.
    /// Maintained incrementally on append/truncate and rebuilt from the
    /// restored log at construction — the claims table IS the log's keyed
    /// entries, never a separate source of truth.
    claims: BTreeMap<u64, u64>,
    votes: BTreeSet<NodeId>,
    pre_votes: BTreeSet<NodeId>,
    use_pre_vote: bool,
    /// Phase B (RFC 028 §3/§4): voters that are WITNESSES — they vote in
    /// elections (control quorum) but never count toward commit durability
    /// (data quorum) and never campaign. With commits requiring a quorum of
    /// DATA voters, a witness-assisted election can only ever elect a data
    /// node that holds every committed entry: in the 2-data+witness
    /// topology, every commit is on BOTH data nodes, so "stale data node +
    /// witness elect a leader" (the design review's P1-7 scenario) can lose
    /// only tentative entries — never acknowledged-durable ones.
    witnesses: BTreeSet<NodeId>,
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
        // Rebuild the claims table from the restored log: claims live IN
        // keyed entries, so a crash can never separate a claim from its op.
        let claims = restored_log
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.key.map(|k| (k, i as u64 + 1)))
            .collect();
        Self {
            id,
            voters,
            persisted_hard: restored_hard,
            durable_len,
            base: LogPosition::ZERO,
            log: restored_log,
            pending: None,
            role: Role::Follower,
            commit: 0,
            match_index: BTreeMap::new(),
            next_index: BTreeMap::new(),
            claims,
            votes: BTreeSet::new(),
            pre_votes: BTreeSet::new(),
            use_pre_vote,
            witnesses: BTreeSet::new(),
        }
    }

    /// Declare the witness subset of the voter set (driver/config-time).
    /// Witnesses vote, never campaign, never count toward commits, and a
    /// witness id must be in `voters` (it IS a voter for elections).
    pub fn set_witnesses(&mut self, witnesses: BTreeSet<NodeId>) {
        debug_assert!(witnesses.iter().all(|w| self.voters.contains(w)));
        debug_assert!(
            !witnesses.contains(&self.id) || self.role == Role::Follower,
            "a witness cannot already hold a data role"
        );
        self.witnesses = witnesses;
    }

    /// Restore a COMPACTED node: `base` + `snapshot_claims` come from the
    /// durable snapshot; `restored_log` is the suffix above the base. The
    /// claims table = snapshot claims + suffix claims (a suffix entry may
    /// re-claim a key whose original was compacted only if the original
    /// was superseded — in practice suffix wins, matching log order).
    #[allow(clippy::too_many_arguments)]
    pub fn new_from_durable(
        id: NodeId,
        voters: BTreeSet<NodeId>,
        restored_hard: HardState,
        base: LogPosition,
        restored_log: Vec<LogEntry>,
        snapshot_claims: BTreeMap<u64, u64>,
        use_pre_vote: bool,
    ) -> Self {
        let mut core = Self::new(id, voters, restored_hard, Vec::new(), use_pre_vote);
        core.base = base;
        core.claims = snapshot_claims;
        // The state below base is adopted wholesale (checkpoint semantics).
        core.commit = base.index;
        for e in restored_log {
            core.log.push(e);
            if let Some(k) = e.key {
                core.claims.insert(k, core.last_index());
            }
        }
        core.durable_len = core.log.len() as u64;
        core
    }

    /// Absolute index of the last entry (base + suffix).
    pub fn last_index(&self) -> u64 {
        self.base.index + self.log.len() as u64
    }

    /// The compaction base (snapshot frontier).
    pub fn base(&self) -> LogPosition {
        self.base
    }

    /// Compact the log through `up_to` (absolute index). Only COMMITTED
    /// entries may compact (RFC 028 §6: every recovery path keeps log
    /// coverage or a verified snapshot — an uncommitted entry has neither).
    /// Returns the snapshot the driver persists alongside the suffix; the
    /// claims table travels in it (sol P1-9).
    pub fn compact(&mut self, up_to: u64) -> Option<Snapshot> {
        if up_to <= self.base.index || up_to > self.commit {
            return None;
        }
        let last_term = self.entry(up_to)?.term;
        let drop = (up_to - self.base.index) as usize;
        self.log.drain(..drop);
        self.base = LogPosition {
            term: last_term.0,
            index: up_to,
        };
        self.durable_len = self.durable_len.saturating_sub(drop as u64);
        Some(Snapshot {
            last: self.base,
            claims: self.claims.clone(),
        })
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

    /// Leader's next send index for `peer` (observability + tests).
    pub fn next_index_of(&self, peer: NodeId) -> Option<u64> {
        self.next_index.get(&peer).copied()
    }

    pub fn log_len(&self) -> u64 {
        self.log.len() as u64
    }

    /// Entry at 1-based `index`, if present.
    pub fn entry(&self, index: u64) -> Option<&LogEntry> {
        if index <= self.base.index {
            return None; // compacted (or the 0 sentinel)
        }
        self.log.get((index - self.base.index) as usize - 1)
    }

    /// Position of the last in-memory log entry (sentinel ZERO when empty).
    /// Used for election freshness; in-memory-ahead-of-durable only makes a
    /// voter STRICTER (it may refuse an equally-fresh candidate — liveness,
    /// never safety), and grants persist before leaving anyway.
    pub fn last_log(&self) -> LogPosition {
        match self.log.last() {
            Some(e) => LogPosition {
                term: e.term.0,
                index: self.last_index(),
            },
            None => self.base,
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
            base: self.base,
            log: self.log.clone(),
            claims: self.claims.clone(),
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
            self.match_index
                .insert(self.id, self.base.index + self.durable_len);
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
        self.log.push(LogEntry::unkeyed(term, payload));
        let mut out = vec![self.stage_log()];
        // Fan out eagerly; success responses confirm durable acceptance.
        self.append_effects_for_followers(&mut out);
        Some(out)
    }

    /// Keyed proposal (Phase B, RFC 028 §7). The claim check happens HERE —
    /// origin ingress — against the claims table rebuilt from the log:
    /// a key claimed by a committed entry dedupes; a key claimed by an
    /// in-flight entry parks the retry (no second append, no premature
    /// success); a fresh key appends an entry that CARRIES the claim.
    /// Leader-only; `None` = redirect to the leader.
    ///
    /// The caller's ack contract (the pre-registered twin properties):
    /// report success for `index` only once `commit_index() >= index`
    /// (never-success-without-durable-effect), and never append the same
    /// key twice while it is claimed (never-double-write). Both halves are
    /// proven in the simulator across failover/quarantine interleavings.
    pub fn propose_keyed(&mut self, key: u64, payload: u64) -> Option<KeyedProposal> {
        if self.role != Role::Leader {
            return None;
        }
        if let Some(&index) = self.claims.get(&key) {
            return Some(if index <= self.commit {
                KeyedProposal::DuplicateCommitted { index }
            } else {
                KeyedProposal::DuplicatePending { index }
            });
        }
        let term = self.current_term();
        self.log.push(LogEntry {
            term,
            payload,
            key: Some(key),
        });
        let index = self.last_index();
        self.claims.insert(key, index);
        let mut effects = vec![self.stage_log()];
        self.append_effects_for_followers(&mut effects);
        Some(KeyedProposal::Appended { index, effects })
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
        // Straggler below our compaction base: entries are gone — ship the
        // snapshot instead (the stale-rejoin-beyond-GC path, §6). Claims
        // ride along, so the GC'd claim can never be replay-lost (P1-9).
        if next <= self.base.index {
            let msg = Message::InstallSnapshot {
                term,
                leader: self.id,
                snapshot: Snapshot {
                    last: self.base,
                    claims: self.claims.clone(),
                },
            };
            self.hold_or(Effect::Send { to: peer, msg }, out);
            return;
        }
        let prev_index = next.saturating_sub(1);
        let prev = if prev_index == self.base.index {
            self.base // covers the ZERO sentinel when uncompacted
        } else {
            match self.entry(prev_index) {
                Some(e) => LogPosition {
                    term: e.term.0,
                    index: prev_index,
                },
                None => self.base,
            }
        };
        let entries: Vec<LogEntry> = self
            .log
            .iter()
            .skip((prev.index - self.base.index) as usize)
            .copied()
            .collect();
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
        // A witness never campaigns: it has no data to lead with. It still
        // VOTES (its on_vote_request path is untouched), which is its whole
        // job — the tiebreak in even-data-node topologies.
        if self.witnesses.contains(&self.id) {
            return Vec::new();
        }
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
            Message::InstallSnapshot {
                term,
                leader: _,
                snapshot,
            } => self.on_install_snapshot(from, term, snapshot),
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
                    last_index: self.base.index + self.durable_len,
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

        // Everything at or below our base is COMMITTED here, and committed
        // entries are globally unique (Gate A #2) — a leader probing below
        // our base can simply fast-forward to it.
        if prev.index < self.base.index {
            let ack = Effect::Send {
                to: from,
                msg: Message::AppendResponse {
                    term,
                    success: true,
                    last_index: self.base.index,
                },
            };
            if newer {
                effects.push(self.stage(adopted));
                self.hold(ack);
            } else {
                self.hold_or(ack, &mut effects);
            }
            return effects;
        }

        // Continuity check at `prev` (the append-side of Gate A #2: a hole
        // or a term mismatch means these entries do not extend our history —
        // refuse; the leader backs up).
        let prev_ok = (prev.index == self.base.index && prev.term == self.base.term)
            || self
                .entry(prev.index)
                .is_some_and(|e| e.term.0 == prev.term);
        if !prev_ok {
            let refusal = Effect::Send {
                to: from,
                msg: Message::AppendResponse {
                    term,
                    success: false,
                    last_index: (self.base.index + self.durable_len)
                        .min(prev.index.saturating_sub(1)),
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
                                last_index: self.base.index + self.durable_len,
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
                    // The claim dies WITH its truncated entry (the atomic
                    // unit, RFC 028 §7): remove keys owned by the removed
                    // suffix so a re-proposed key can claim afresh.
                    for removed in self.log.iter().skip((idx - self.base.index) as usize - 1) {
                        if let Some(k) = removed.key {
                            if self.claims.get(&k) >= Some(&idx) {
                                self.claims.remove(&k);
                            }
                        }
                    }
                    self.log.truncate((idx - self.base.index) as usize - 1);
                    self.log.push(*e);
                    if let Some(k) = e.key {
                        self.claims.insert(k, idx);
                    }
                    changed = true;
                }
                None => {
                    self.log.push(*e);
                    if let Some(k) = e.key {
                        self.claims.insert(k, idx);
                    }
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
                        (self.base.index + self.durable_len).min(covered)
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
                    last_index: (self.base.index + self.durable_len).min(covered),
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
        // Codex finding 2: delayed/duplicate responses must not regress
        // next_index below match_index + 1. match_index is monotonic (max);
        // matched entries are confirmed-durable-matching, so probing below
        // match+1 is never needed — a stale success or an out-of-date
        // failure clamped to the floor costs nothing, while an unclamped
        // one triggers spurious full-suffix retransmissions.
        if success {
            let m = self.match_index.entry(from).or_insert(0);
            *m = (*m).max(last_index);
            let floor = *m + 1;
            self.next_index.insert(from, floor);
            self.try_advance_commit(&mut out);
        } else {
            let floor = self.match_index.get(&from).copied().unwrap_or(0) + 1;
            let next = self
                .next_index
                .get(&from)
                .copied()
                .unwrap_or(self.last_index() + 1);
            let backed = next.saturating_sub(1).clamp(1, last_index + 1).max(floor);
            self.next_index.insert(from, backed);
            self.send_append_to(from, &mut out);
        }
        out
    }

    /// Adopt a leader's snapshot (we are a straggler below its compaction
    /// base). Same discipline as every other durable decision: ONE persist,
    /// ack + InstallState held behind it. A snapshot at or below our own
    /// commit is stale — ack our durable frontier instead (never regress).
    fn on_install_snapshot(&mut self, from: NodeId, term: Term, snapshot: Snapshot) -> Vec<Effect> {
        let mut effects = Vec::new();
        let cur = self.current_term();
        if term < cur {
            return vec![Effect::Send {
                to: from,
                msg: Message::AppendResponse {
                    term: cur,
                    success: false,
                    last_index: self.base.index + self.durable_len,
                },
            }];
        }
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
        if snapshot.last.index <= self.commit {
            // Stale snapshot: our committed state is already ahead.
            let ack = Effect::Send {
                to: from,
                msg: Message::AppendResponse {
                    term,
                    success: true,
                    last_index: self.base.index + self.durable_len,
                },
            };
            if newer {
                effects.push(self.stage(adopted));
                self.hold(ack);
            } else {
                self.hold_or(ack, &mut effects);
            }
            return effects;
        }
        // Adopt: checkpoint replaces log prefix AND claims wholesale.
        let last_index = snapshot.last.index;
        self.base = snapshot.last;
        self.log.clear();
        self.claims = snapshot.claims;
        self.commit = last_index;
        effects.push(self.stage(adopted));
        self.hold(Effect::InstallState { last_index });
        self.hold(Effect::Send {
            to: from,
            msg: Message::AppendResponse {
                term,
                success: true,
                last_index,
            },
        });
        effects
    }

    /// The commit rule (Gate A #2): the largest `n` such that a quorum of
    /// voters has DURABLY accepted through `n` and `log[n].term` is the
    /// CURRENT term. Prior-term entries never commit by counting — only by
    /// implication under a committed current-term entry (the no-op
    /// guarantees one exists). Phase B narrows the counted set to
    /// data-bearing voters (witness exclusion).
    fn try_advance_commit(&mut self, out: &mut Vec<Effect>) {
        let cur = self.current_term();
        let mut n = self.last_index();
        while n > self.commit {
            if self.entry(n).map(|e| e.term) == Some(cur) {
                // Data quorum ONLY: witnesses never count toward commit
                // durability (RFC 028 §4 — a witness ack is a position
                // ack, not a data copy).
                let data_voters = || self.voters.iter().filter(|v| !self.witnesses.contains(v));
                let acks = data_voters()
                    .filter(|v| self.match_index.get(v).copied().unwrap_or(0) >= n)
                    .count();
                if acks >= quorum(data_voters().count()) {
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
        self.log.push(LogEntry::unkeyed(term, NOOP_PAYLOAD));
        out.push(self.stage_log());
        self.append_effects_for_followers(out);
    }

    /// Single-voter-cluster variant: everything (leadership announcement,
    /// no-op, fan-out) rides behind the already-staged self-vote persist.
    fn become_leader_held(&mut self, term: Term) {
        debug_assert!(self.pending.is_some());
        self.init_leader_state(term);
        self.hold(Effect::BecameLeader { term });
        self.log.push(LogEntry::unkeyed(term, NOOP_PAYLOAD));
        let _ = self.stage_log(); // coalesces into the pending persist
    }

    fn init_leader_state(&mut self, _term: Term) {
        self.role = Role::Leader;
        let last = self.last_index();
        self.next_index.clear();
        self.match_index.clear();
        for v in self.voters.iter().copied() {
            self.next_index.insert(v, last + 1);
            self.match_index.insert(v, 0);
        }
        self.match_index
            .insert(self.id, self.base.index + self.durable_len);
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

    /// Rejoin authorization (RFC 028 v2 §5): produce a snapshot grant for a
    /// quarantined node — but only when this node holds a **quorum-backed
    /// leadership certificate**: it is leader AND has committed an entry in
    /// its CURRENT term (the election no-op guarantees one exists once a
    /// quorum has confirmed the reign). A stale partitioned "leader" cannot
    /// satisfy that (its current-term entries never commit — Gate A #2), so
    /// it can never authorize a resync — review scenario R5's fix.
    ///
    /// Returns `(term, log, commit)` for the driver to wrap into a
    /// `bootstrap::RejoinMessage::Grant`; `None` = not authorized (the
    /// quarantined node stays quarantined and retries — fail closed).
    /// Returns `(term, base, durable_suffix, claims, commit)` — the full
    /// durable picture, snapshot-aware: `base` + `claims` carry compacted
    /// history, `durable_suffix` the live entries above it.
    pub fn rejoin_grant(
        &self,
    ) -> Option<(Term, LogPosition, Vec<LogEntry>, BTreeMap<u64, u64>, u64)> {
        if self.role != Role::Leader {
            return None;
        }
        let cur = self.current_term();
        let committed_in_current_term =
            self.commit > 0 && self.entry(self.commit).map(|e| e.term) == Some(cur);
        if !committed_in_current_term {
            return None;
        }
        // Grant only the DURABLE suffix: a snapshot must never carry
        // entries our own disk could forget in a crash. Base + claims are
        // durable by construction (they only advance via persisted state).
        let durable: Vec<LogEntry> = self
            .log
            .iter()
            .take(self.durable_len as usize)
            .copied()
            .collect();
        let commit = self.commit.min(self.base.index + self.durable_len);
        Some((cur, self.base, durable, self.claims.clone(), commit))
    }

    fn pre_quorum_reached(&self) -> bool {
        self.pre_votes.len() >= quorum(self.voters.len())
    }

    fn vote_quorum_reached(&self) -> bool {
        self.votes.len() >= quorum(self.voters.len())
    }
}
