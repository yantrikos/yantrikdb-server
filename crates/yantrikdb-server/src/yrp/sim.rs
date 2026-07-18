//! Deterministic replica simulator — the Gate A proof harness (RFC 028 v2 §11).
//!
//! Chaos is a detector; this is closer to a proof. The simulator drives
//! [`ReplicaCore`] instances through seeded schedules with message drops,
//! reorders, partitions, and crash injection at the persist boundary, then
//! checks the Gate A invariants over the entire observable history:
//!
//! - **I1 (vote safety, R2):** across the whole run, including every
//!   crash/restart, no node's SENT grants name two candidates in one term.
//! - **I2 (authority safety, R1):** a global committed-entry ledger — once
//!   ANY node applies entry `e` at index `i`, no node may ever apply a
//!   different entry at `i`. Two leaders acking conflicting writes as
//!   durable would trip this immediately.
//! - **I3 (suffix protection, R3):** every sent grant went to a candidate
//!   whose advertised log was at least as up to date as the voter's.
//! - **Single leader per term**, ever.
//!
//! Crash modeling is exact: a crash discards the in-memory core (with any
//! pending persist and its withheld messages) and restarts from the last
//! `(hard, log)` snapshot the sim's "disk" accepted. Boundaries: 0 = before
//! persist (held messages never sent — safe), 1 = after persist / before
//! flush (durable state kept, responses lost — liveness only).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::bootstrap::{
    inspect, BootDecision, BootstrapEffect, Integrity, QuarantineReason, QuarantinedNode,
    RecoveredState, RejoinMessage,
};
use super::replica::{Effect, KeyedProposal, LogEntry, Message, ReplicaCore, Role, NOOP_PAYLOAD};
use super::types::{ClusterId, HardState, LogPosition, NodeId, Term};

/// The sim's cluster identity (all healthy nodes share it; alien-state tests
/// inject a different one).
const CLUSTER: ClusterId = ClusterId(7);

/// Both protocol planes ride one transport.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Wire {
    R(Message),
    B(RejoinMessage),
}

/// Tiny deterministic PRNG (xorshift64*) — no external deps, fully seeded.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
    fn pick(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct InFlight {
    from: NodeId,
    to: NodeId,
    msg: Wire,
}

/// One simulated node: the core (None while crashed/quarantined) + its
/// durable "disk". `torn_hard`/`corrupt_log` model integrity-check failures
/// the next boot inspection will see.
struct SimNode {
    core: Option<ReplicaCore>,
    quarantined: Option<QuarantinedNode>,
    disk_hard: HardState,
    disk_log: Vec<LogEntry>,
    torn_hard: bool,
    corrupt_log: bool,
    /// Highest index this node has applied (ledgered via CommitAdvanced).
    applied: u64,
}

struct Sim {
    nodes: BTreeMap<NodeId, SimNode>,
    voters: BTreeSet<NodeId>,
    net: VecDeque<InFlight>,
    rng: Rng,
    /// Partition: unordered pairs that cannot exchange messages.
    cut: BTreeSet<(NodeId, NodeId)>,
    // ── invariant ledgers (observable history) ────────────────────
    /// (voter, term) → candidates named by the voter's SENT grants.
    grants_sent: BTreeMap<(NodeId, Term), BTreeSet<NodeId>>,
    /// term → nodes that became leader in that term.
    leaders: BTreeMap<Term, BTreeSet<NodeId>>,
    /// I2: index → the one entry ever applied there, by anyone.
    committed_at: BTreeMap<u64, LogEntry>,
    /// I3: (candidate, term) → last_log advertised in its VoteRequests.
    advertised: BTreeMap<(NodeId, Term), LogPosition>,
    /// I3 ledger: (voter_last_log_at_send, candidate_advertised).
    freshness_at_grant: Vec<(LogPosition, LogPosition)>,
    /// Gate A #4 ledgers: preserve-before-resync + corruption alarms.
    preserves: Vec<NodeId>,
    alarms: Vec<(NodeId, Vec<QuarantineReason>)>,
    /// Phase B claim ledger: key → (index, payload) of the ONE committed
    /// entry ever allowed to hold it (never-double-write, globally).
    keyed_committed: BTreeMap<u64, (u64, u64)>,
    /// Witness subset (applied to every constructed core, incl. restarts).
    witnesses: BTreeSet<NodeId>,
}

impl Sim {
    /// `start_term` seeds every node's durable term so pre-seeded log entry
    /// terms stay coherent with the entry-term ≤ current-term invariant.
    fn new(
        node_logs: &[(u64, Vec<LogEntry>)],
        start_term: u64,
        seed: u64,
        use_pre_vote: bool,
    ) -> Self {
        let voters: BTreeSet<NodeId> = node_logs.iter().map(|(id, _)| NodeId(*id)).collect();
        let hard = HardState {
            current_term: Term(start_term),
            voted_for: None,
        };
        let mut nodes = BTreeMap::new();
        for (id, log) in node_logs {
            let nid = NodeId(*id);
            let core = ReplicaCore::new(nid, voters.clone(), hard, log.clone(), use_pre_vote);
            nodes.insert(
                nid,
                SimNode {
                    core: Some(core),
                    quarantined: None,
                    disk_hard: hard,
                    disk_log: log.clone(),
                    torn_hard: false,
                    corrupt_log: false,
                    applied: 0,
                },
            );
        }
        Sim {
            nodes,
            voters,
            net: VecDeque::new(),
            rng: Rng::new(seed),
            cut: BTreeSet::new(),
            grants_sent: BTreeMap::new(),
            leaders: BTreeMap::new(),
            committed_at: BTreeMap::new(),
            advertised: BTreeMap::new(),
            freshness_at_grant: Vec::new(),
            preserves: Vec::new(),
            alarms: Vec::new(),
            keyed_committed: BTreeMap::new(),
            witnesses: BTreeSet::new(),
        }
    }

    /// Witness-topology constructor: `witness_ids` vote but never count
    /// toward commits and never campaign. Applied to every core this sim
    /// ever constructs (initial, restart, bootstrap, rejoin-adopt).
    fn new_with_witnesses(
        node_logs: &[(u64, Vec<LogEntry>)],
        start_term: u64,
        seed: u64,
        use_pre_vote: bool,
        witness_ids: &[u64],
    ) -> Self {
        let mut sim = Sim::new(node_logs, start_term, seed, use_pre_vote);
        sim.witnesses = witness_ids.iter().map(|w| NodeId(*w)).collect();
        let w = sim.witnesses.clone();
        for node in sim.nodes.values_mut() {
            if let Some(core) = node.core.as_mut() {
                core.set_witnesses(w.clone());
            }
        }
        sim
    }

    /// Run one node's effect batch. `crash_at` injects a crash at a persist
    /// boundary: 0 = before persist, 1 = after persist / before flush.
    fn run_effects(&mut self, id: NodeId, effects: Vec<Effect>, crash_at: Option<u8>) {
        let mut queue: VecDeque<Effect> = effects.into();
        while let Some(eff) = queue.pop_front() {
            match eff {
                Effect::Persist { hard, log } => {
                    if crash_at == Some(0) {
                        self.crash(id);
                        return;
                    }
                    {
                        let node = self.nodes.get_mut(&id).unwrap();
                        node.disk_hard = hard;
                        node.disk_log = log;
                    }
                    if crash_at == Some(1) {
                        self.crash(id);
                        return;
                    }
                    let flushed = self
                        .nodes
                        .get_mut(&id)
                        .unwrap()
                        .core
                        .as_mut()
                        .unwrap()
                        .state_persisted();
                    for f in flushed {
                        queue.push_back(f);
                    }
                }
                Effect::Send { to, msg } => self.record_and_route(id, to, msg),
                Effect::Broadcast { msg } => {
                    let peers: Vec<NodeId> =
                        self.voters.iter().copied().filter(|p| *p != id).collect();
                    for to in peers {
                        self.record_and_route(id, to, msg.clone());
                    }
                }
                Effect::BecameLeader { term } => {
                    self.leaders.entry(term).or_default().insert(id);
                }
                Effect::SteppedDown { .. } => {}
                Effect::CommitAdvanced { to } => self.ledger_commit(id, to),
            }
        }
    }

    /// I2 — the authority-safety ledger. Applying is reading the node's own
    /// log over the newly committed range; the global map enforces that no
    /// index is ever applied with two different entries, by anyone.
    fn ledger_commit(&mut self, id: NodeId, to: u64) {
        let from = self.nodes[&id].applied + 1;
        for i in from..=to {
            let e = *self.nodes[&id]
                .core
                .as_ref()
                .expect("commit on live node")
                .entry(i)
                .expect("committed index must be in log");
            match self.committed_at.get(&i) {
                None => {
                    self.committed_at.insert(i, e);
                }
                Some(prev) => assert_eq!(
                    *prev, e,
                    "AUTHORITY SAFETY VIOLATED: index {i} applied with two \
                     different entries ({prev:?} vs {e:?}, second by {id:?})"
                ),
            }
            // Phase B never-double-write: one committed entry per key, ever.
            if let Some(k) = e.key {
                match self.keyed_committed.get(&k) {
                    None => {
                        self.keyed_committed.insert(k, (i, e.payload));
                    }
                    Some((pi, pp)) => assert_eq!(
                        (*pi, *pp),
                        (i, e.payload),
                        "CLAIM DOUBLE-WRITE: key {k} committed at two places"
                    ),
                }
            }
        }
        let node = self.nodes.get_mut(&id).unwrap();
        node.applied = node.applied.max(to);
    }

    /// Ledger every SENT message, then put it on the wire.
    fn record_and_route(&mut self, from: NodeId, to: NodeId, msg: Message) {
        match &msg {
            Message::VoteRequest {
                term,
                candidate,
                last_log,
            } => {
                self.advertised.insert((*candidate, *term), *last_log);
            }
            Message::VoteResponse {
                term,
                granted: true,
            } => {
                self.grants_sent
                    .entry((from, *term))
                    .or_default()
                    .insert(to);
                let voter_log = self.nodes[&from]
                    .core
                    .as_ref()
                    .map(|c| c.last_log())
                    .unwrap_or(LogPosition::ZERO);
                if let Some(cand) = self.advertised.get(&(to, *term)) {
                    self.freshness_at_grant.push((voter_log, *cand));
                }
            }
            _ => {}
        }
        self.net.push_back(InFlight {
            from,
            to,
            msg: Wire::R(msg),
        });
    }

    fn crash(&mut self, id: NodeId) {
        self.nodes.get_mut(&id).unwrap().core = None;
    }

    /// Crash AND tear the durable hard-state record (models a torn write /
    /// bit rot the next boot's checksum will catch).
    fn crash_torn(&mut self, id: NodeId) {
        let node = self.nodes.get_mut(&id).unwrap();
        node.core = None;
        node.quarantined = None;
        node.torn_hard = true;
    }

    /// Crash AND break the log hash chain (data corruption evidence).
    fn crash_corrupt(&mut self, id: NodeId) {
        let node = self.nodes.get_mut(&id).unwrap();
        node.core = None;
        node.quarantined = None;
        node.corrupt_log = true;
    }

    /// The production boot path: recover disk state, run integrity checks,
    /// inspect, and become either a live replica or a quarantined node. The
    /// process ALWAYS comes up as something — that is the point.
    fn restart_via_bootstrap(&mut self, id: NodeId) {
        let voters = self.voters.clone();
        let node = self.nodes.get_mut(&id).unwrap();
        let recovered = RecoveredState {
            cluster_id: Some(CLUSTER),
            hard: Some(node.disk_hard),
            log: Some(node.disk_log.clone()),
            commit_marker: 0, // volatile in A2's model
            integrity: Integrity {
                hard_state_verified: !node.torn_hard,
                log_verified: !node.corrupt_log,
            },
        };
        match inspect(CLUSTER, &recovered) {
            BootDecision::Healthy { hard, log } => {
                let mut core = ReplicaCore::new(id, voters, hard, log, false);
                core.set_witnesses(self.witnesses.clone());
                node.core = Some(core);
                node.quarantined = None;
                node.applied = node.applied.min(node.disk_log.len() as u64);
            }
            BootDecision::Quarantine { reasons, term_hint } => {
                node.core = None;
                node.quarantined = Some(QuarantinedNode::new(id, CLUSTER, reasons, term_hint));
            }
        }
    }

    /// Drive a quarantined node's rejoin retry toward `leader_hint`.
    fn tick_rejoin(&mut self, id: NodeId, leader_hint: NodeId) {
        let Some(q) = self.nodes.get_mut(&id).unwrap().quarantined.as_mut() else {
            return;
        };
        let effects = q.tick_rejoin(leader_hint);
        self.run_bootstrap_effects(id, effects);
    }

    fn run_bootstrap_effects(&mut self, id: NodeId, effects: Vec<BootstrapEffect>) {
        for eff in effects {
            match eff {
                BootstrapEffect::PreserveOldState => {
                    self.preserves.push(id);
                }
                BootstrapEffect::Alarm { reasons } => {
                    self.alarms.push((id, reasons));
                }
                BootstrapEffect::Send { to, msg } => {
                    self.net.push_back(InFlight {
                        from: id,
                        to,
                        msg: Wire::B(msg),
                    });
                }
                BootstrapEffect::AdoptSnapshot {
                    cluster_id: _,
                    hard,
                    log,
                } => {
                    // Persist the adopted snapshot, clear damage flags, and
                    // resume as a live follower. Quarantine ends here only.
                    assert!(
                        self.preserves.contains(&id),
                        "adopt without preserving old state first"
                    );
                    let voters = self.voters.clone();
                    let node = self.nodes.get_mut(&id).unwrap();
                    node.disk_hard = hard;
                    node.disk_log = log.clone();
                    node.torn_hard = false;
                    node.corrupt_log = false;
                    node.quarantined = None;
                    let mut core = ReplicaCore::new(id, voters, hard, log, false);
                    core.set_witnesses(self.witnesses.clone());
                    node.core = Some(core);
                    node.applied = node.applied.min(node.disk_log.len() as u64);
                }
            }
        }
    }

    fn restart(&mut self, id: NodeId, use_pre_vote: bool) {
        let voters = self.voters.clone();
        let node = self.nodes.get_mut(&id).unwrap();
        // ONLY what was durably persisted survives. `applied` is clamped to
        // the durable log (the state machine re-applies deterministically;
        // the I2 ledger verifies every re-application is identical).
        let mut core = ReplicaCore::new(
            id,
            voters,
            node.disk_hard,
            node.disk_log.clone(),
            use_pre_vote,
        );
        core.set_witnesses(self.witnesses.clone());
        node.core = Some(core);
        node.quarantined = None;
        node.applied = node.applied.min(node.disk_log.len() as u64);
    }

    fn timeout(&mut self, id: NodeId, crash_at: Option<u8>) {
        if let Some(core) = self.nodes.get_mut(&id).unwrap().core.as_mut() {
            let effects = core.on_election_timeout();
            self.run_effects(id, effects, crash_at);
        }
    }

    fn propose(&mut self, id: NodeId, payload: u64) -> bool {
        let Some(core) = self.nodes.get_mut(&id).unwrap().core.as_mut() else {
            return false;
        };
        match core.propose(payload) {
            Some(effects) => {
                self.run_effects(id, effects, None);
                true
            }
            None => false,
        }
    }

    fn heartbeat(&mut self, id: NodeId) {
        if let Some(core) = self.nodes.get_mut(&id).unwrap().core.as_mut() {
            let effects = core.tick_heartbeat();
            self.run_effects(id, effects, None);
        }
    }

    /// Deliver one in-flight message (respecting partitions/crashes),
    /// optionally injecting a crash while the receiver handles it.
    fn deliver_one(&mut self, crash_at: Option<u8>) -> bool {
        let Some(m) = self.net.pop_front() else {
            return false;
        };
        let key = (m.from.min(m.to), m.from.max(m.to));
        if self.cut.contains(&key) {
            return true;
        }
        match m.msg {
            Wire::R(msg) => {
                let Some(node) = self.nodes.get_mut(&m.to) else {
                    return true;
                };
                // Quarantined (or crashed) nodes DROP replica traffic:
                // fail-closed is enforced by absence — there is no code
                // path by which a quarantined node grants a vote or acks
                // an append.
                let Some(core) = node.core.as_mut() else {
                    return true;
                };
                let effects = core.on_message(m.from, msg, false);
                self.run_effects(m.to, effects, crash_at);
            }
            Wire::B(RejoinMessage::Request { node: asker }) => {
                // Rejoin requests are answered only by a live core holding
                // the leadership certificate (committed in current term).
                let grant = self
                    .nodes
                    .get(&m.to)
                    .and_then(|n| n.core.as_ref())
                    .and_then(|c| c.rejoin_grant());
                if let Some((term, log, commit)) = grant {
                    self.net.push_back(InFlight {
                        from: m.to,
                        to: asker,
                        msg: Wire::B(RejoinMessage::Grant {
                            cluster_id: CLUSTER,
                            term,
                            log,
                            commit,
                            // The leader's snapshot is live state: verified.
                            verified: true,
                        }),
                    });
                }
            }
            Wire::B(grant @ RejoinMessage::Grant { .. }) => {
                let effects = {
                    let Some(node) = self.nodes.get_mut(&m.to) else {
                        return true;
                    };
                    let Some(q) = node.quarantined.as_mut() else {
                        return true; // no longer quarantined: stale grant ignored
                    };
                    q.on_grant(m.from, grant)
                };
                self.run_bootstrap_effects(m.to, effects);
            }
        }
        true
    }

    fn drain(&mut self) {
        while self.deliver_one(None) {}
    }

    fn current_leader(&self) -> Option<NodeId> {
        self.nodes
            .iter()
            .find(|(_, n)| n.core.as_ref().is_some_and(|c| c.role() == Role::Leader))
            .map(|(id, _)| *id)
    }

    // ── invariant checkers ────────────────────────────────────────

    fn check_vote_safety(&self) {
        for ((voter, term), cands) in &self.grants_sent {
            assert!(
                cands.len() <= 1,
                "VOTE SAFETY VIOLATED: voter {voter:?} granted {cands:?} in {term:?}"
            );
        }
    }

    fn check_suffix_protection(&self) {
        for (voter_log, cand_log) in &self.freshness_at_grant {
            assert!(
                cand_log.is_at_least_as_up_to_date_as(voter_log),
                "SUFFIX PROTECTION VIOLATED: granted candidate at {cand_log:?} \
                 while voter was at {voter_log:?}"
            );
        }
    }

    fn check_single_leader_per_term(&self) {
        for (term, ls) in &self.leaders {
            assert!(ls.len() <= 1, "TWO LEADERS IN {term:?}: {ls:?}");
        }
    }

    /// I2 is asserted incrementally in `ledger_commit`; this re-validates
    /// that every live node's log agrees with the committed ledger over its
    /// applied prefix (a truncation below commit would surface here).
    fn check_committed_prefix_integrity(&self) {
        for (id, node) in &self.nodes {
            let Some(core) = node.core.as_ref() else {
                continue;
            };
            for i in 1..=node.applied {
                if let Some(expected) = self.committed_at.get(&i) {
                    let actual = core.entry(i);
                    assert_eq!(
                        actual,
                        Some(expected),
                        "COMMITTED PREFIX DAMAGED on {id:?} at index {i}"
                    );
                }
            }
        }
    }

    fn check_all(&self) {
        self.check_vote_safety();
        self.check_suffix_protection();
        self.check_single_leader_per_term();
        self.check_committed_prefix_integrity();
    }
}

// ── helpers ────────────────────────────────────────────────────────

fn entries(terms: &[u64]) -> Vec<LogEntry> {
    terms
        .iter()
        .enumerate()
        .map(|(i, t)| LogEntry::unkeyed(Term(*t), 1000 + i as u64))
        .collect()
}

fn empty() -> Vec<LogEntry> {
    Vec::new()
}

// ── tests: Phase A1 invariants (carried forward) ───────────────────

/// Baseline: a healthy 3-node cluster elects exactly one leader, and the
/// election no-op commits across the cluster.
#[test]
fn three_nodes_elect_exactly_one_leader() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 42, true);
    sim.timeout(NodeId(1), None);
    sim.drain();
    sim.check_all();
    assert_eq!(
        sim.leaders.values().map(|s| s.len()).sum::<usize>(),
        1,
        "exactly one leadership event expected, got {:?}",
        sim.leaders
    );
    // The winner's no-op reached commit.
    assert_eq!(
        sim.committed_at.get(&1).map(|e| e.payload),
        Some(NOOP_PAYLOAD)
    );
}

/// R2, crash BEFORE persist: the vote decision (and its held response)
/// evaporates — the original grant never left the node.
#[test]
fn r2_crash_before_persist_never_leaks_the_grant() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 7, false);
    sim.timeout(NodeId(1), None);
    let mut injected = false;
    while !sim.net.is_empty() {
        let to = sim.net.front().unwrap().to;
        let inject = if to == NodeId(3) && !injected {
            injected = true;
            Some(0u8)
        } else {
            None
        };
        sim.deliver_one(inject);
    }
    sim.restart(NodeId(3), false);
    assert_eq!(sim.nodes[&NodeId(3)].disk_hard, HardState::default());
    sim.timeout(NodeId(2), None);
    sim.drain();
    sim.check_all();
}

/// R2, crash AFTER persist but before the response flushes: the vote is
/// durable, the response lost. The restarted node must refuse a different
/// candidate in the same term.
#[test]
fn r2_crash_after_persist_binds_the_restarted_node() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 9, false);
    sim.timeout(NodeId(1), None);
    let mut injected = false;
    while !sim.net.is_empty() {
        let to = sim.net.front().unwrap().to;
        let inject = if to == NodeId(3) && !injected {
            injected = true;
            Some(1u8)
        } else {
            None
        };
        sim.deliver_one(inject);
    }
    assert_eq!(
        sim.nodes[&NodeId(3)].disk_hard,
        HardState {
            current_term: Term(1),
            voted_for: Some(NodeId(1)),
        }
    );
    sim.restart(NodeId(3), false);
    let effects = sim
        .nodes
        .get_mut(&NodeId(3))
        .unwrap()
        .core
        .as_mut()
        .unwrap()
        .on_message(
            NodeId(2),
            Message::VoteRequest {
                term: Term(1),
                candidate: NodeId(2),
                last_log: LogPosition::ZERO,
            },
            false,
        );
    sim.run_effects(NodeId(3), effects, None);
    sim.drain();
    let grants = sim
        .grants_sent
        .get(&(NodeId(3), Term(1)))
        .cloned()
        .unwrap_or_default();
    assert!(
        !grants.contains(&NodeId(2)),
        "restarted node granted a second candidate in the same term: {grants:?}"
    );
    sim.check_all();
}

/// R3: a candidate with a stale log (lower last term, longer index) must be
/// refused by voters holding fresher entries.
#[test]
fn r3_stale_log_candidate_is_refused_by_fresher_voters() {
    // Nodes 2 and 3 hold a possibly-committed (term 2) suffix; node 1 has a
    // longer but staler (all term 1) log. Everyone starts at term 2.
    let stale = entries(&[1, 1, 1, 1, 1, 1, 1, 1, 1]); // last = (1, 9)
    let fresh = entries(&[1, 1, 1, 1, 2]); // last = (2, 5)
    let mut sim = Sim::new(&[(1, stale), (2, fresh.clone()), (3, fresh)], 2, 21, false);
    sim.timeout(NodeId(1), None);
    sim.drain();
    sim.check_all();
    assert!(
        sim.leaders.values().all(|s| !s.contains(&NodeId(1))),
        "stale-log candidate won an election: {:?}",
        sim.leaders
    );
    sim.timeout(NodeId(2), None);
    sim.drain();
    sim.check_all();
    assert!(
        sim.leaders.values().any(|s| s.contains(&NodeId(2))),
        "fresh candidate failed to win: {:?}",
        sim.leaders
    );
}

/// Partition + heal: a minority candidate cannot win; heal converges.
#[test]
fn partition_minority_cannot_elect_and_heal_converges() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 63, false);
    sim.cut.insert((NodeId(1), NodeId(2)));
    sim.cut.insert((NodeId(1), NodeId(3)));
    sim.timeout(NodeId(1), None);
    sim.timeout(NodeId(2), None);
    sim.drain();
    sim.check_all();
    assert!(
        sim.leaders.values().all(|s| !s.contains(&NodeId(1))),
        "partitioned minority elected itself: {:?}",
        sim.leaders
    );
    sim.cut.clear();
    sim.drain();
    sim.check_all();
}

/// Pre-vote probing by an isolated node must not advance its durable term.
#[test]
fn pre_vote_probe_never_burns_terms() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 5, true);
    sim.cut.insert((NodeId(1), NodeId(2)));
    sim.cut.insert((NodeId(1), NodeId(3)));
    for _ in 0..25 {
        sim.timeout(NodeId(1), None);
        sim.drain();
    }
    assert_eq!(sim.nodes[&NodeId(1)].disk_hard.current_term, Term(0));
    sim.check_all();
}

// ── tests: Phase A1b — replication + authority safety ──────────────

/// Proposed entries reach commit on every node; the committed ledger holds
/// exactly the proposed payloads in order after the election no-op.
#[test]
fn replication_commits_across_cluster() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 11, false);
    sim.timeout(NodeId(1), None);
    sim.drain();
    let leader = sim.current_leader().expect("leader elected");
    for p in [101, 102, 103] {
        assert!(sim.propose(leader, p), "propose on leader");
    }
    sim.drain();
    sim.heartbeat(leader); // share final commit index
    sim.drain();
    sim.check_all();
    // Index 1 = no-op, 2..=4 = payloads, committed everywhere.
    assert_eq!(sim.committed_at.get(&2).map(|e| e.payload), Some(101));
    assert_eq!(sim.committed_at.get(&3).map(|e| e.payload), Some(102));
    assert_eq!(sim.committed_at.get(&4).map(|e| e.payload), Some(103));
    for (id, node) in &sim.nodes {
        assert!(node.applied >= 4, "{id:?} applied only to {}", node.applied);
    }
}

/// R1 — the stale-leader scenario, end to end: a partitioned leader keeps
/// proposing but can never commit (no quorum of durable acks); the majority
/// elects a new leader and commits different entries at the same indices;
/// on heal the stale leader is term-fenced, steps down, and its tentative
/// suffix is truncated in favor of canonical history. The authority ledger
/// proves the stale proposals were never applied anywhere.
#[test]
fn r1_stale_leader_cannot_commit_and_gets_fenced() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 17, false);
    sim.timeout(NodeId(1), None);
    sim.drain();
    assert_eq!(sim.current_leader(), Some(NodeId(1)));
    let applied_before = sim.nodes[&NodeId(1)].applied;

    // Partition the leader away; it keeps accepting proposals (tentative).
    sim.cut.insert((NodeId(1), NodeId(2)));
    sim.cut.insert((NodeId(1), NodeId(3)));
    assert!(sim.propose(NodeId(1), 201));
    assert!(sim.propose(NodeId(1), 202));
    sim.drain();
    // No quorum → no commit advance on the stale leader.
    assert_eq!(
        sim.nodes[&NodeId(1)].applied,
        applied_before,
        "stale leader advanced commit without a quorum"
    );

    // Majority side elects a new leader and commits different entries.
    sim.timeout(NodeId(2), None);
    sim.drain();
    assert!(sim.propose(NodeId(2), 301));
    sim.drain();

    // Heal: the stale leader gets fenced and adopts canonical history.
    sim.cut.clear();
    sim.heartbeat(NodeId(2));
    sim.drain();
    sim.heartbeat(NodeId(2));
    sim.drain();
    sim.check_all();

    // The stale proposals were never applied by anyone.
    assert!(
        sim.committed_at
            .values()
            .all(|e| e.payload != 201 && e.payload != 202),
        "stale leader's tentative writes leaked into committed history"
    );
    // Node 1 now holds the canonical entry (truncated + replaced).
    let committed_301 = sim
        .committed_at
        .iter()
        .find(|(_, e)| e.payload == 301)
        .map(|(i, _)| *i)
        .expect("301 committed");
    let n1 = sim.nodes[&NodeId(1)].core.as_ref().unwrap();
    assert_eq!(n1.entry(committed_301).map(|e| e.payload), Some(301));
    assert_eq!(sim.current_leader(), Some(NodeId(2)));
}

/// A follower that crashes before persisting appended entries loses them,
/// restarts behind, and is caught up by the leader's heartbeat protocol.
#[test]
fn follower_catchup_after_crash() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 29, false);
    sim.timeout(NodeId(1), None);
    sim.drain();
    // Propose; crash node 3 at the persist boundary (entries lost).
    assert!(sim.propose(NodeId(1), 401));
    let mut injected = false;
    while !sim.net.is_empty() {
        let to = sim.net.front().unwrap().to;
        let inject = if to == NodeId(3) && !injected {
            injected = true;
            Some(0u8)
        } else {
            None
        };
        sim.deliver_one(inject);
    }
    // Quorum still commits via nodes 1+2.
    assert!(sim.committed_at.values().any(|e| e.payload == 401));
    // Node 3 restarts behind; heartbeats catch it up.
    sim.restart(NodeId(3), false);
    sim.heartbeat(NodeId(1));
    sim.drain();
    sim.heartbeat(NodeId(1));
    sim.drain();
    sim.check_all();
    let n3 = &sim.nodes[&NodeId(3)];
    assert!(
        n3.applied >= 2,
        "restarted follower failed to catch up (applied {})",
        n3.applied
    );
}

/// Seeded soak: random elections, proposals, heartbeats, crashes at random
/// persist boundaries, restarts, partitions — every invariant holds for
/// every seed.
#[test]
fn seeded_soak_invariants_hold() {
    for seed in 1..30u64 {
        let mut sim = Sim::new(
            &[(1, empty()), (2, empty()), (3, empty())],
            0,
            seed,
            seed % 2 == 0,
        );
        let mut payload = 100;
        for _step in 0..300 {
            let ids = [NodeId(1), NodeId(2), NodeId(3)];
            match sim.rng.next() % 12 {
                0 => {
                    let id = ids[sim.rng.pick(3)];
                    if sim.nodes[&id].core.is_some() {
                        let inject = sim.rng.chance(20).then(|| (sim.rng.next() % 2) as u8);
                        sim.timeout(id, inject);
                    }
                }
                1 => {
                    let id = ids[sim.rng.pick(3)];
                    if sim.nodes[&id].core.is_some() && sim.rng.chance(15) {
                        sim.crash(id);
                    }
                }
                2 => {
                    let id = ids[sim.rng.pick(3)];
                    if sim.nodes[&id].core.is_none() {
                        let pv = sim.rng.chance(50);
                        sim.restart(id, pv);
                    }
                }
                3 => {
                    // Propose on whoever believes it is leader (may be a
                    // stale leader — exactly the point).
                    let id = ids[sim.rng.pick(3)];
                    payload += 1;
                    let _ = sim.propose(id, payload);
                }
                4 => {
                    let id = ids[sim.rng.pick(3)];
                    sim.heartbeat(id);
                }
                5 => {
                    // Toggle a partition edge.
                    let a = ids[sim.rng.pick(3)];
                    let b = ids[sim.rng.pick(3)];
                    if a != b {
                        let key = (a.min(b), a.max(b));
                        if !sim.cut.remove(&key) {
                            sim.cut.insert(key);
                        }
                    }
                }
                _ => {
                    let inject = sim.rng.chance(10).then(|| (sim.rng.next() % 2) as u8);
                    sim.deliver_one(inject);
                }
            }
        }
        sim.cut.clear();
        sim.drain();
        sim.check_all();
    }
}

// ── tests: Phase A2 — quarantine + quorum-authorized rejoin ────────

/// THE CT-141 TEST. A node with torn consensus metadata BOOTS (process up,
/// diagnostics available, stale reads offered) but fails closed as a voter;
/// its damaged state is preserved before resync; a leader with a
/// quorum-backed certificate authorizes rejoin; the node adopts the
/// snapshot, resumes as a follower, and catches up. No operator surgery,
/// no 10-day outage, no double vote.
#[test]
fn ct141_torn_node_quarantines_then_rejoins_via_leader() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 31, false);
    sim.timeout(NodeId(1), None);
    sim.drain();
    assert!(sim.propose(NodeId(1), 501));
    sim.drain();

    // Node 3's disk record is torn in a crash.
    sim.crash_torn(NodeId(3));
    sim.restart_via_bootstrap(NodeId(3));
    {
        let n3 = &sim.nodes[&NodeId(3)];
        let q = n3.quarantined.as_ref().expect("torn node must quarantine");
        assert!(q.reasons().contains(&QuarantineReason::TornHardState));
        // Metadata-only damage: data verified → labeled stale reads OK.
        assert!(q.stale_reads_allowed());
        assert!(n3.core.is_none(), "quarantined node must not run a core");
    }

    // Rejoin via the leader; adopt; catch up.
    sim.tick_rejoin(NodeId(3), NodeId(1));
    sim.drain();
    {
        let n3 = &sim.nodes[&NodeId(3)];
        assert!(n3.quarantined.is_none(), "rejoin did not complete");
        assert!(n3.core.is_some());
        assert!(!n3.torn_hard);
    }
    assert!(
        sim.preserves.contains(&NodeId(3)),
        "old state must be preserved before resync"
    );
    sim.heartbeat(NodeId(1));
    sim.drain();
    sim.check_all();
    assert!(
        sim.nodes[&NodeId(3)].applied >= 2,
        "rejoined node failed to catch up (applied {})",
        sim.nodes[&NodeId(3)].applied
    );
}

/// Fail-closed proof: a quarantined node cannot contribute to ANY quorum.
/// With one node quarantined and the other two partitioned from each other,
/// no candidate can assemble a majority — the cluster correctly stalls
/// rather than electing on damaged state.
#[test]
fn quarantined_node_cannot_vote() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 37, false);
    sim.crash_torn(NodeId(3));
    sim.restart_via_bootstrap(NodeId(3));
    sim.cut.insert((NodeId(1), NodeId(2)));
    sim.timeout(NodeId(1), None);
    sim.timeout(NodeId(2), None);
    sim.drain();
    sim.check_all();
    assert!(
        sim.leaders.is_empty(),
        "an election succeeded without a legal quorum: {:?}",
        sim.leaders
    );
    // And the quarantined node sent zero grants, ever.
    assert!(sim.grants_sent.keys().all(|(voter, _)| *voter != NodeId(3)));
}

/// Corruption evidence (broken log hash chain): alarms fire, stale reads
/// are refused, and only a VERIFIED grant is adopted.
#[test]
fn corrupted_log_alarms_and_rejoins_from_verified_source() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 41, false);
    sim.timeout(NodeId(1), None);
    sim.drain();
    sim.crash_corrupt(NodeId(3));
    sim.restart_via_bootstrap(NodeId(3));
    {
        let q = sim.nodes[&NodeId(3)].quarantined.as_ref().unwrap();
        assert!(q.reasons().contains(&QuarantineReason::LogCorruption));
        assert!(!q.stale_reads_allowed(), "corrupt data must not be served");
    }
    sim.tick_rejoin(NodeId(3), NodeId(1));
    sim.drain();
    assert!(
        sim.alarms.iter().any(|(id, _)| *id == NodeId(3)),
        "corruption evidence must alarm"
    );
    assert!(sim.nodes[&NodeId(3)].quarantined.is_none());
    sim.heartbeat(NodeId(1));
    sim.drain();
    sim.check_all();
}

/// A rejoin request sent to a NON-leader (or a leader without a committed
/// current-term entry) is simply not granted — the node stays quarantined
/// and retries. Authorization requires the certificate.
#[test]
fn rejoin_requires_leadership_certificate() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 43, false);
    sim.timeout(NodeId(1), None);
    sim.drain();
    sim.crash_torn(NodeId(3));
    sim.restart_via_bootstrap(NodeId(3));
    // Ask a FOLLOWER (node 2): no grant, still quarantined.
    sim.tick_rejoin(NodeId(3), NodeId(2));
    sim.drain();
    assert!(sim.nodes[&NodeId(3)].quarantined.is_some());
    // Ask the leader: granted.
    sim.tick_rejoin(NodeId(3), NodeId(1));
    sim.drain();
    assert!(sim.nodes[&NodeId(3)].quarantined.is_none());
    sim.check_all();
}

/// Soak with damage: random torn/corrupt crashes now join the schedule;
/// crashed-damaged nodes restart through the REAL boot path (inspect →
/// quarantine → rejoin-retry). Every invariant holds for every seed —
/// including that no quarantined node ever votes or acks.
#[test]
fn seeded_soak_with_quarantine_invariants_hold() {
    for seed in 1..20u64 {
        let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, seed, false);
        let mut payload = 5000;
        for _step in 0..300 {
            let ids = [NodeId(1), NodeId(2), NodeId(3)];
            match sim.rng.next() % 14 {
                0 => {
                    let id = ids[sim.rng.pick(3)];
                    if sim.nodes[&id].core.is_some() {
                        sim.timeout(id, None);
                    }
                }
                1 => {
                    let id = ids[sim.rng.pick(3)];
                    if sim.nodes[&id].core.is_some() && sim.rng.chance(10) {
                        if sim.rng.chance(50) {
                            sim.crash_torn(id);
                        } else {
                            sim.crash(id);
                        }
                    }
                }
                2 => {
                    let id = ids[sim.rng.pick(3)];
                    if sim.nodes[&id].core.is_none() && sim.nodes[&id].quarantined.is_none() {
                        sim.restart_via_bootstrap(id);
                    }
                }
                3 => {
                    let id = ids[sim.rng.pick(3)];
                    payload += 1;
                    let _ = sim.propose(id, payload);
                }
                4 => {
                    let id = ids[sim.rng.pick(3)];
                    sim.heartbeat(id);
                }
                5 => {
                    // Quarantined nodes retry rejoin toward a random peer.
                    let id = ids[sim.rng.pick(3)];
                    let hint = ids[sim.rng.pick(3)];
                    if id != hint {
                        sim.tick_rejoin(id, hint);
                    }
                }
                6 => {
                    let a = ids[sim.rng.pick(3)];
                    let b = ids[sim.rng.pick(3)];
                    if a != b {
                        let key = (a.min(b), a.max(b));
                        if !sim.cut.remove(&key) {
                            sim.cut.insert(key);
                        }
                    }
                }
                _ => {
                    sim.deliver_one(None);
                }
            }
        }
        sim.cut.clear();
        sim.drain();
        sim.check_all();
        // Fail-closed held throughout: nodes that were EVER quarantined in
        // a term sent no grants while quarantined — enforced structurally,
        // re-checked here via the global ledgers in check_all().
    }
}

// ── tests: codex code-review fixes ─────────────────────────────────

/// Codex finding 1 (SAFETY): a stale-but-certificated leader's grant BELOW
/// the quarantined node's (untrusted) term hint is refused — adopting it
/// would regress the durable term and reopen a double-vote window in the
/// node's true prior term. The genuinely current leader's grant is adopted,
/// and the vote ledger stays clean.
#[test]
fn codex_stale_grant_below_term_hint_is_refused() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 47, false);
    // Node 1 becomes leader of term 1 with a committed no-op (certificate).
    sim.timeout(NodeId(1), None);
    sim.drain();
    assert_eq!(sim.current_leader(), Some(NodeId(1)));
    // Partition node 1 away; it keeps its stale term-1 certificate.
    sim.cut.insert((NodeId(1), NodeId(2)));
    sim.cut.insert((NodeId(1), NodeId(3)));
    // Majority elects node 2 in term 2 — node 3 votes for node 2, so its
    // durable hard state is {term 2, voted node 2}.
    sim.timeout(NodeId(2), None);
    sim.drain();
    // Node 3's record is torn; the bytes (term 2) remain readable as a hint.
    sim.crash_torn(NodeId(3));
    sim.restart_via_bootstrap(NodeId(3));
    assert!(sim.nodes[&NodeId(3)].quarantined.is_some());
    // Ask the STALE leader (node 1, term 1 certificate). Partition does not
    // block them — but the grant term (1) is below the hint (2): refused.
    sim.cut.clear();
    sim.tick_rejoin(NodeId(3), NodeId(1));
    sim.drain();
    assert!(
        sim.nodes[&NodeId(3)].quarantined.is_some(),
        "below-hint grant was adopted — term regression"
    );
    // The real leader's grant (term 2) is adopted.
    sim.heartbeat(NodeId(2));
    sim.drain(); // node 1 gets fenced by term-2 traffic
    sim.tick_rejoin(NodeId(3), NodeId(2));
    sim.drain();
    assert!(sim.nodes[&NodeId(3)].quarantined.is_none());
    // Vote safety held throughout: node 3's term-2 grants name only node 2.
    let grants = sim
        .grants_sent
        .get(&(NodeId(3), Term(2)))
        .cloned()
        .unwrap_or_default();
    assert!(grants.len() <= 1 && !grants.contains(&NodeId(1)));
    sim.check_all();
}

/// Codex finding 2 (CORRECTNESS): a delayed duplicate success response must
/// not regress next_index below match_index + 1.
#[test]
fn codex_duplicate_response_does_not_regress_next_index() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 53, false);
    sim.timeout(NodeId(1), None);
    sim.drain();
    for p in [601, 602, 603] {
        assert!(sim.propose(NodeId(1), p));
    }
    sim.drain();
    // Follower 2 is fully matched (no-op + 3 entries = index 4).
    let next_before = sim.nodes[&NodeId(1)]
        .core
        .as_ref()
        .unwrap()
        .next_index_of(NodeId(2))
        .unwrap();
    assert_eq!(next_before, 5);
    // Deliver a DELAYED duplicate: an old success covering only index 1.
    let effects = sim
        .nodes
        .get_mut(&NodeId(1))
        .unwrap()
        .core
        .as_mut()
        .unwrap()
        .on_message(
            NodeId(2),
            Message::AppendResponse {
                term: Term(1),
                success: true,
                last_index: 1,
            },
            false,
        );
    sim.run_effects(NodeId(1), effects, None);
    let next_after = sim.nodes[&NodeId(1)]
        .core
        .as_ref()
        .unwrap()
        .next_index_of(NodeId(2))
        .unwrap();
    assert_eq!(
        next_after, 5,
        "duplicate old success regressed next_index to {next_after}"
    );
    // And a stale FAILURE cannot drag next below match+1 either.
    let effects = sim
        .nodes
        .get_mut(&NodeId(1))
        .unwrap()
        .core
        .as_mut()
        .unwrap()
        .on_message(
            NodeId(2),
            Message::AppendResponse {
                term: Term(1),
                success: false,
                last_index: 0,
            },
            false,
        );
    sim.run_effects(NodeId(1), effects, None);
    let next_final = sim.nodes[&NodeId(1)]
        .core
        .as_ref()
        .unwrap()
        .next_index_of(NodeId(2))
        .unwrap();
    assert!(
        next_final >= 5,
        "stale failure regressed next_index to {next_final}"
    );
    sim.drain();
    sim.check_all();
}

// ── tests: Phase B — claim-in-log (RFC 028 §7) ─────────────────────

impl Sim {
    /// Sim driver for a keyed proposal: runs any effects, returns the
    /// outcome. Mirrors what the production API layer will do.
    fn propose_keyed(&mut self, id: NodeId, key: u64, payload: u64) -> Option<KeyedProposal> {
        let core = self.nodes.get_mut(&id).unwrap().core.as_mut()?;
        let outcome = core.propose_keyed(key, payload)?;
        if let KeyedProposal::Appended { effects, index } = outcome {
            self.run_effects(id, effects, None);
            return Some(KeyedProposal::Appended {
                index,
                effects: Vec::new(), // consumed
            });
        }
        Some(outcome)
    }
}

/// Trading scenario (a): "fill committed but claim lost" must be
/// impossible — a committed keyed entry SURVIVES failover, and the retry
/// dedupes against it instead of re-executing. (The claim rides in the
/// entry; committing the entry commits the claim.)
#[test]
fn keyed_commit_survives_failover_and_dedupes_retry() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 71, false);
    sim.timeout(NodeId(1), None);
    sim.drain();
    // Keyed write commits cluster-wide; the CLIENT never learns (leader
    // crashes before responding).
    let out = sim.propose_keyed(NodeId(1), 77, 707).expect("leader");
    let orig_index = match out {
        KeyedProposal::Appended { index, .. } => index,
        other => panic!("expected fresh append, got {other:?}"),
    };
    sim.drain();
    assert!(
        sim.keyed_committed.contains_key(&77),
        "keyed entry committed"
    );
    sim.crash(NodeId(1));
    // Failover; the new leader commits the prior suffix by implication.
    sim.timeout(NodeId(2), None);
    sim.drain();
    // The client retries the SAME keyed request on the new leader.
    let retry = sim.propose_keyed(NodeId(2), 77, 707).expect("new leader");
    match retry {
        KeyedProposal::DuplicateCommitted { index } => {
            assert_eq!(index, orig_index, "dedupe must return the ORIGINAL entry");
        }
        other => panic!("retry after committed failover must dedupe, got {other:?}"),
    }
    sim.check_all();
    assert_eq!(
        sim.keyed_committed.get(&77).map(|(i, p)| (*i, *p)),
        Some((orig_index, 707)),
        "exactly one committed effect for the key"
    );
}

/// Trading scenario (b): "claim settled but commit rolled back" must be
/// impossible — a TENTATIVE keyed entry truncates WITH its claim, so the
/// retry re-executes cleanly on the new leader and exactly one effect
/// commits, ever.
#[test]
fn keyed_tentative_loss_reexecutes_cleanly() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 73, false);
    sim.timeout(NodeId(1), None);
    sim.drain();
    // Partition the leader; its keyed write stays tentative forever.
    sim.cut.insert((NodeId(1), NodeId(2)));
    sim.cut.insert((NodeId(1), NodeId(3)));
    let out = sim.propose_keyed(NodeId(1), 88, 808).expect("stale leader");
    assert!(matches!(out, KeyedProposal::Appended { .. }));
    sim.drain();
    assert!(
        !sim.keyed_committed.contains_key(&88),
        "tentative keyed write must not commit without a quorum"
    );
    // Majority elects a new leader; the client retries there.
    sim.timeout(NodeId(2), None);
    sim.drain();
    let retry = sim.propose_keyed(NodeId(2), 88, 808).expect("new leader");
    let new_index = match retry {
        KeyedProposal::Appended { index, .. } => index,
        other => panic!("retry after tentative loss must re-execute, got {other:?}"),
    };
    sim.drain();
    // Heal: the stale leader truncates its tentative entry AND its claim,
    // then adopts the canonical keyed entry.
    sim.cut.clear();
    sim.heartbeat(NodeId(2));
    sim.drain();
    sim.heartbeat(NodeId(2));
    sim.drain();
    sim.check_all();
    assert_eq!(
        sim.keyed_committed.get(&88).map(|(i, p)| (*i, *p)),
        Some((new_index, 808)),
        "exactly one committed effect, at the canonical index"
    );
    // The healed ex-leader holds the canonical keyed entry.
    let n1 = sim.nodes[&NodeId(1)].core.as_ref().unwrap();
    assert_eq!(
        n1.entry(new_index).and_then(|e| e.key),
        Some(88),
        "healed node holds the canonical keyed entry"
    );
}

/// Same-leader retry semantics: a pending claim parks (no second append,
/// no premature success); after commit the same retry dedupes.
#[test]
fn keyed_retry_pending_parks_then_dedupes() {
    let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, 79, false);
    sim.timeout(NodeId(1), None);
    sim.drain();
    // Propose but do NOT deliver anything yet: claim is pending.
    let core = sim
        .nodes
        .get_mut(&NodeId(1))
        .unwrap()
        .core
        .as_mut()
        .unwrap();
    let out1 = core.propose_keyed(99, 909).unwrap();
    let index = match &out1 {
        KeyedProposal::Appended { index, .. } => *index,
        other => panic!("fresh append expected, got {other:?}"),
    };
    // Immediate retry while pending: parked, same index, NO new entry.
    let out2 = core.propose_keyed(99, 909).unwrap();
    assert_eq!(out2, KeyedProposal::DuplicatePending { index });
    let log_len_before = core.log_len();
    // Run the pending effects (persist + fan-out) to completion.
    if let KeyedProposal::Appended { effects, .. } = out1 {
        sim.run_effects(NodeId(1), effects, None);
    }
    sim.drain();
    let core = sim
        .nodes
        .get_mut(&NodeId(1))
        .unwrap()
        .core
        .as_mut()
        .unwrap();
    assert_eq!(
        core.log_len(),
        log_len_before,
        "no second append for the key"
    );
    let out3 = core.propose_keyed(99, 909).unwrap();
    assert_eq!(out3, KeyedProposal::DuplicateCommitted { index });
    sim.check_all();
}

/// The mcp wire-contract property + its twin, under chaos: keyed retries
/// fired at random nodes across elections, partitions, crashes,
/// torn-quarantines and rejoins never double-commit a key (ledger-asserted
/// on every commit) — and every DuplicateCommitted answer refers to a key
/// with exactly one committed effect (success implies durable effect).
#[test]
fn seeded_soak_keyed_claims_hold_under_chaos() {
    for seed in 1..15u64 {
        let mut sim = Sim::new(&[(1, empty()), (2, empty()), (3, empty())], 0, seed, false);
        let keys = [11u64, 22, 33, 44];
        for _step in 0..300 {
            let ids = [NodeId(1), NodeId(2), NodeId(3)];
            match sim.rng.next() % 14 {
                0 => {
                    let id = ids[sim.rng.pick(3)];
                    if sim.nodes[&id].core.is_some() {
                        sim.timeout(id, None);
                    }
                }
                1 => {
                    let id = ids[sim.rng.pick(3)];
                    if sim.nodes[&id].core.is_some() && sim.rng.chance(10) {
                        if sim.rng.chance(40) {
                            sim.crash_torn(id);
                        } else {
                            sim.crash(id);
                        }
                    }
                }
                2 => {
                    let id = ids[sim.rng.pick(3)];
                    if sim.nodes[&id].core.is_none() && sim.nodes[&id].quarantined.is_none() {
                        sim.restart_via_bootstrap(id);
                    }
                }
                3 | 4 => {
                    // The property under test: keyed retries at random nodes.
                    let id = ids[sim.rng.pick(3)];
                    let k = keys[sim.rng.pick(keys.len())];
                    if let Some(KeyedProposal::DuplicateCommitted { .. }) =
                        sim.propose_keyed(id, k, k * 10)
                    {
                        // Success answer implies a durable effect exists.
                        assert!(
                            sim.keyed_committed.contains_key(&k),
                            "DuplicateCommitted for key {k} without a \
                             committed effect (ghost success)"
                        );
                    }
                }
                5 => {
                    let id = ids[sim.rng.pick(3)];
                    sim.heartbeat(id);
                }
                6 => {
                    let id = ids[sim.rng.pick(3)];
                    let hint = ids[sim.rng.pick(3)];
                    if id != hint {
                        sim.tick_rejoin(id, hint);
                    }
                }
                7 => {
                    let a = ids[sim.rng.pick(3)];
                    let b = ids[sim.rng.pick(3)];
                    if a != b {
                        let kk = (a.min(b), a.max(b));
                        if !sim.cut.remove(&kk) {
                            sim.cut.insert(kk);
                        }
                    }
                }
                _ => {
                    sim.deliver_one(None);
                }
            }
        }
        sim.cut.clear();
        sim.drain();
        sim.check_all(); // includes the per-key single-commit ledger
    }
}

// ── tests: Phase B — witness data-quorum split (RFC 028 §3/§4) ─────

/// A witness never campaigns: election timeouts on it are inert.
#[test]
fn witness_never_campaigns() {
    let mut sim = Sim::new_with_witnesses(
        &[(1, empty()), (2, empty()), (3, empty())],
        0,
        83,
        false,
        &[3],
    );
    for _ in 0..5 {
        sim.timeout(NodeId(3), None);
        sim.drain();
    }
    assert!(
        sim.leaders.is_empty(),
        "witness campaigned: {:?}",
        sim.leaders
    );
    assert_eq!(sim.nodes[&NodeId(3)].disk_hard.current_term, Term(0));
}

/// The witness's vote elects a leader (control quorum counts it), but its
/// append acks never count toward commits: with the only other DATA node
/// partitioned away, commits stall — a write acked only by leader+witness
/// is NOT durable, exactly as §4 promises.
#[test]
fn witness_votes_but_never_counts_for_commit() {
    let mut sim = Sim::new_with_witnesses(
        &[(1, empty()), (2, empty()), (3, empty())],
        0,
        89,
        false,
        &[3],
    );
    // Election succeeds with the witness's vote (leader 1 + witness 3).
    sim.cut.insert((NodeId(1), NodeId(2))); // data peer unreachable
    sim.timeout(NodeId(1), None);
    sim.drain();
    assert!(
        sim.leaders.values().any(|s| s.contains(&NodeId(1))),
        "witness vote must elect: {:?}",
        sim.leaders
    );
    // But nothing can COMMIT: data quorum is 2-of-2 data nodes and node 2
    // is unreachable. The no-op and this proposal stay tentative.
    assert!(sim.propose(NodeId(1), 901));
    sim.drain();
    sim.heartbeat(NodeId(1));
    sim.drain();
    assert_eq!(
        sim.nodes[&NodeId(1)].applied,
        0,
        "commit advanced on witness acks alone"
    );
    assert!(sim.committed_at.is_empty());
    // Heal the data peer: the suffix commits.
    sim.cut.clear();
    sim.heartbeat(NodeId(1));
    sim.drain();
    sim.check_all();
    assert!(
        sim.committed_at.values().any(|e| e.payload == 901),
        "entry must commit once the data quorum is reachable"
    );
}

/// The P1-7 answer: crash the data leader; the surviving data node +
/// witness elect a new leader (control quorum 2/3) — and NO committed
/// entry can be lost, because data-quorum commits guaranteed every
/// committed entry was already on BOTH data nodes.
#[test]
fn witness_tiebreak_preserves_all_committed_entries() {
    let mut sim = Sim::new_with_witnesses(
        &[(1, empty()), (2, empty()), (3, empty())],
        0,
        97,
        false,
        &[3],
    );
    sim.timeout(NodeId(1), None);
    sim.drain();
    for p in [911, 912, 913] {
        assert!(sim.propose(NodeId(1), p));
    }
    sim.drain();
    let committed_before: Vec<u64> = sim.committed_at.values().map(|e| e.payload).collect();
    assert!(committed_before.contains(&913), "writes committed");
    // Data leader dies. Survivors: one data node + the witness.
    sim.crash(NodeId(1));
    sim.timeout(NodeId(2), None);
    sim.drain();
    assert!(
        sim.leaders.values().any(|s| s.contains(&NodeId(2))),
        "surviving data node must win with the witness vote"
    );
    // The honest tradeoff of 2-data+witness (documented per the design
    // review's P1-7 ask): the topology survives a data-node failure for
    // ELECTIONS and committed-data safety — but NOT for write
    // availability. A new write cannot commit on witness acks alone.
    sim.propose(NodeId(2), 914);
    sim.drain();
    assert!(
        !sim.committed_at.values().any(|e| e.payload == 914),
        "write committed without a data quorum"
    );
    sim.check_all(); // committed-prefix integrity: nothing lost
    for p in [911, 912, 913] {
        assert!(
            sim.committed_at.values().any(|e| e.payload == p),
            "committed entry {p} lost across witness-assisted failover"
        );
    }
    // The crashed data node returns: write availability resumes and the
    // stalled entry commits.
    sim.restart(NodeId(1), false);
    sim.heartbeat(NodeId(2));
    sim.drain();
    sim.heartbeat(NodeId(2));
    sim.drain();
    sim.check_all();
    assert!(
        sim.committed_at.values().any(|e| e.payload == 914),
        "stalled write must commit once the data quorum returns"
    );
}

/// Keyed-claims chaos on the witness topology: same properties as the
/// 3-data soak, with the witness voting through every election and never
/// polluting the data quorum.
#[test]
fn seeded_soak_witness_topology_keyed_claims_hold() {
    for seed in 1..10u64 {
        let mut sim = Sim::new_with_witnesses(
            &[(1, empty()), (2, empty()), (3, empty())],
            0,
            seed,
            false,
            &[3],
        );
        let keys = [55u64, 66];
        for _step in 0..250 {
            let ids = [NodeId(1), NodeId(2), NodeId(3)];
            match sim.rng.next() % 12 {
                0 => {
                    let id = ids[sim.rng.pick(3)];
                    if sim.nodes[&id].core.is_some() {
                        sim.timeout(id, None);
                    }
                }
                1 => {
                    let id = ids[sim.rng.pick(3)];
                    if sim.nodes[&id].core.is_some() && sim.rng.chance(10) {
                        sim.crash(id);
                    }
                }
                2 => {
                    let id = ids[sim.rng.pick(3)];
                    if sim.nodes[&id].core.is_none() && sim.nodes[&id].quarantined.is_none() {
                        sim.restart(id, false);
                    }
                }
                3 | 4 => {
                    let id = ids[sim.rng.pick(3)];
                    let k = keys[sim.rng.pick(keys.len())];
                    if let Some(KeyedProposal::DuplicateCommitted { .. }) =
                        sim.propose_keyed(id, k, k * 10)
                    {
                        assert!(sim.keyed_committed.contains_key(&k));
                    }
                }
                5 => {
                    let id = ids[sim.rng.pick(3)];
                    sim.heartbeat(id);
                }
                6 => {
                    let a = ids[sim.rng.pick(3)];
                    let b = ids[sim.rng.pick(3)];
                    if a != b {
                        let kk = (a.min(b), a.max(b));
                        if !sim.cut.remove(&kk) {
                            sim.cut.insert(kk);
                        }
                    }
                }
                _ => {
                    sim.deliver_one(None);
                }
            }
        }
        sim.cut.clear();
        sim.drain();
        sim.check_all();
    }
}
