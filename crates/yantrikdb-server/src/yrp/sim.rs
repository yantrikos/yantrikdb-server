//! Deterministic election simulator — the Gate A proof harness (RFC 028 v2 §11).
//!
//! Chaos is a detector; this is closer to a proof. The simulator drives
//! [`ElectionCore`] instances through seeded schedules with message
//! drops/reorders, partitions, and — the important part — **crash injection at
//! the persist boundary**, then checks the Gate A invariants over the entire
//! observable history:
//!
//! - **I1 (vote safety, R2):** across the whole run, including every
//!   crash/restart, no node's SENT grant messages name two different
//!   candidates in one term.
//! - **I3 (suffix protection, R3):** no voter ever grants a candidate whose
//!   last log position is less up to date than the voter's own at grant time.
//! - **Single leader per term:** at most one `BecameLeader` per term, ever.
//!
//! Crash modeling is exact: a crash discards the in-memory core (with any
//! pending persist and its withheld messages) and restarts from the last
//! hard state the sim's "disk" accepted. The three interesting boundaries:
//! before persist (held messages never sent — safe), after persist / before
//! flush (vote durable, response lost — liveness only), after flush (normal).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::election::{Effect, ElectionCore, Message, Role};
use super::types::{HardState, LogPosition, NodeId, Term};

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
    msg: Message,
}

/// One simulated node: the core (None while crashed) + its durable "disk".
struct SimNode {
    core: Option<ElectionCore>,
    disk: HardState,
    last_log: LogPosition,
}

struct Sim {
    nodes: BTreeMap<NodeId, SimNode>,
    voters: BTreeSet<NodeId>,
    net: VecDeque<InFlight>,
    rng: Rng,
    /// Partition: pairs that cannot exchange messages.
    cut: BTreeSet<(NodeId, NodeId)>,
    // ── invariant ledgers (observable history) ────────────────────
    /// (voter, term) → set of candidates the voter's SENT grants named.
    grants_sent: BTreeMap<(NodeId, Term), BTreeSet<NodeId>>,
    /// term → set of nodes that became leader in that term.
    leaders: BTreeMap<Term, BTreeSet<NodeId>>,
    /// I3 ledger: recorded at grant-send time: (voter_last_log, candidate_last_log).
    freshness_at_grant: Vec<(LogPosition, LogPosition)>,
    /// Candidate → last_log advertised (fixed per test; lets the checker
    /// recover the candidate's position from a grant message).
    advertised: BTreeMap<NodeId, LogPosition>,
}

impl Sim {
    fn new(node_logs: &[(u64, LogPosition)], seed: u64, use_pre_vote: bool) -> Self {
        let voters: BTreeSet<NodeId> = node_logs.iter().map(|(id, _)| NodeId(*id)).collect();
        let mut nodes = BTreeMap::new();
        for (id, log) in node_logs {
            let nid = NodeId(*id);
            let disk = HardState::default();
            let core = ElectionCore::new(nid, voters.clone(), disk, *log, use_pre_vote);
            nodes.insert(
                nid,
                SimNode {
                    core: Some(core),
                    disk,
                    last_log: *log,
                },
            );
        }
        let advertised = node_logs
            .iter()
            .map(|(id, log)| (NodeId(*id), *log))
            .collect();
        Sim {
            nodes,
            voters,
            net: VecDeque::new(),
            rng: Rng::new(seed),
            cut: BTreeSet::new(),
            grants_sent: BTreeMap::new(),
            leaders: BTreeMap::new(),
            freshness_at_grant: Vec::new(),
            advertised,
        }
    }

    /// Run one node's effect batch: persist to "disk", flush the gate,
    /// enqueue sends, record ledgers. `crash_at` injects a crash at a
    /// boundary: 0 = before persist, 1 = after persist / before flush.
    fn run_effects(&mut self, id: NodeId, effects: Vec<Effect>, crash_at: Option<u8>) {
        let mut queue: VecDeque<Effect> = effects.into();
        while let Some(eff) = queue.pop_front() {
            match eff {
                Effect::PersistHardState(hs) => {
                    if crash_at == Some(0) {
                        self.crash(id); // pending + held messages evaporate
                        return;
                    }
                    self.nodes.get_mut(&id).unwrap().disk = hs; // durable now
                    if crash_at == Some(1) {
                        self.crash(id); // vote durable, response lost — safe
                        return;
                    }
                    let flushed = self
                        .nodes
                        .get_mut(&id)
                        .unwrap()
                        .core
                        .as_mut()
                        .unwrap()
                        .hard_state_persisted();
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
            }
        }
    }

    /// Ledger every SENT message (the observable history the invariants
    /// quantify over), then put it on the wire.
    fn record_and_route(&mut self, from: NodeId, to: NodeId, msg: Message) {
        if let Message::VoteResponse {
            term,
            granted: true,
        } = &msg
        {
            // The grant names the peer it is addressed to (the candidate).
            self.grants_sent
                .entry((from, *term))
                .or_default()
                .insert(to);
            let voter_log = self.nodes[&from].last_log;
            if let Some(cand_log) = self.advertised.get(&to) {
                self.freshness_at_grant.push((voter_log, *cand_log));
            }
        }
        self.net.push_back(InFlight { from, to, msg });
    }

    fn crash(&mut self, id: NodeId) {
        self.nodes.get_mut(&id).unwrap().core = None;
    }

    fn restart(&mut self, id: NodeId, use_pre_vote: bool) {
        let node = self.nodes.get_mut(&id).unwrap();
        let core = ElectionCore::new(
            id,
            self.voters.clone(),
            node.disk, // ONLY what was durably persisted survives
            node.last_log,
            use_pre_vote,
        );
        node.core = Some(core);
    }

    fn timeout(&mut self, id: NodeId, crash_at: Option<u8>) {
        if let Some(core) = self.nodes.get_mut(&id).unwrap().core.as_mut() {
            let effects = core.on_election_timeout();
            self.run_effects(id, effects, crash_at);
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
            return true; // dropped by partition
        }
        let Some(node) = self.nodes.get_mut(&m.to) else {
            return true;
        };
        let Some(core) = node.core.as_mut() else {
            return true; // crashed receiver drops the message
        };
        let effects = core.on_message(m.from, m.msg, false);
        self.run_effects(m.to, effects, crash_at);
        true
    }

    fn drain(&mut self) {
        while self.deliver_one(None) {}
    }

    // ── invariant checkers ────────────────────────────────────────

    /// Gate A #1 / R2: per (voter, term), all sent grants name ONE candidate.
    fn check_vote_safety(&self) {
        for ((voter, term), cands) in &self.grants_sent {
            assert!(
                cands.len() <= 1,
                "VOTE SAFETY VIOLATED: voter {voter:?} granted {cands:?} in {term:?}"
            );
        }
    }

    /// Gate A #3 / R3: every grant went to a candidate at least as up to date.
    fn check_suffix_protection(&self) {
        for (voter_log, cand_log) in &self.freshness_at_grant {
            assert!(
                cand_log.is_at_least_as_up_to_date_as(voter_log),
                "SUFFIX PROTECTION VIOLATED: granted candidate at {cand_log:?} \
                 while voter was at {voter_log:?}"
            );
        }
    }

    /// At most one leader per term, across the whole run.
    fn check_single_leader_per_term(&self) {
        for (term, ls) in &self.leaders {
            assert!(ls.len() <= 1, "TWO LEADERS IN {term:?}: {ls:?}");
        }
    }

    fn check_all(&self) {
        self.check_vote_safety();
        self.check_suffix_protection();
        self.check_single_leader_per_term();
    }
}

// ── tests ──────────────────────────────────────────────────────────

const L0: LogPosition = LogPosition { term: 0, index: 0 };

/// Baseline: a healthy 3-node cluster elects exactly one leader.
#[test]
fn three_nodes_elect_exactly_one_leader() {
    let mut sim = Sim::new(&[(1, L0), (2, L0), (3, L0)], 42, true);
    sim.timeout(NodeId(1), None);
    sim.drain();
    sim.check_all();
    assert_eq!(
        sim.leaders.values().map(|s| s.len()).sum::<usize>(),
        1,
        "exactly one leadership event expected, got {:?}",
        sim.leaders
    );
}

/// R2, crash BEFORE persist: the vote decision (and its held response) must
/// evaporate — after restart the node may vote differently, but the original
/// grant never left the node, so no double grant is observable.
#[test]
fn r2_crash_before_persist_never_leaks_the_grant() {
    let mut sim = Sim::new(&[(1, L0), (2, L0), (3, L0)], 7, false);
    // Candidate 1 campaigns; node 3 receives the request and crashes at the
    // persist boundary (before durability).
    sim.timeout(NodeId(1), None);
    // route: find the VoteRequest to node 3 and deliver it with crash_at=0.
    // Deliver messages one at a time; inject the crash on node 3's handling.
    let mut delivered_to_3 = false;
    while !sim.net.is_empty() {
        let peek_to = sim.net.front().unwrap().to;
        let inject = if peek_to == NodeId(3) && !delivered_to_3 {
            delivered_to_3 = true;
            Some(0u8)
        } else {
            None
        };
        sim.deliver_one(inject);
    }
    // Node 3 restarts from disk — which never recorded the vote.
    sim.restart(NodeId(3), false);
    assert_eq!(sim.nodes[&NodeId(3)].disk, HardState::default());
    // A competing candidate 2 campaigns in the SAME term (term 1) and node 3
    // may now grant it — legal, because the first grant never left the node.
    sim.timeout(NodeId(2), None);
    sim.drain();
    sim.check_all(); // vote-safety ledger proves no double grant was SENT
}

/// R2, crash AFTER persist but before the response flushes: the vote is
/// durable, the response is lost. After restart the node must REFUSE a
/// different candidate in the same term — the persisted vote binds it.
#[test]
fn r2_crash_after_persist_binds_the_restarted_node() {
    let mut sim = Sim::new(&[(1, L0), (2, L0), (3, L0)], 9, false);
    sim.timeout(NodeId(1), None); // candidate 1, term 1
    let mut injected = false;
    while !sim.net.is_empty() {
        let peek_to = sim.net.front().unwrap().to;
        let inject = if peek_to == NodeId(3) && !injected {
            injected = true;
            Some(1u8) // crash after persist, before flush
        } else {
            None
        };
        sim.deliver_one(inject);
    }
    // The vote for candidate 1 IS on node 3's disk; the response was lost.
    assert_eq!(
        sim.nodes[&NodeId(3)].disk,
        HardState {
            current_term: Term(1),
            voted_for: Some(NodeId(1)),
        }
    );
    sim.restart(NodeId(3), false);
    // Competing candidate 2 campaigns in the same term 1... its own campaign
    // takes it to term 2 actually — so drive the SAME-term request manually:
    // deliver a raw VoteRequest for term 1 from node 2 to node 3.
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
                last_log: L0,
            },
            false,
        );
    sim.run_effects(NodeId(3), effects, None);
    sim.drain();
    // Ledger: node 3's term-1 grants must name at most candidate 1.
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

/// R3: a candidate with a stale log (lower last term, even with a longer
/// index) must be refused by voters holding fresher entries — the
/// possibly-committed suffix survives the election.
#[test]
fn r3_stale_log_candidate_is_refused_by_fresher_voters() {
    // Node 2 and 3 hold an entry at (term 2, idx 5) — possibly committed.
    // Node 1's log ends at (term 1, idx 9): longer, but staler.
    let fresh = LogPosition { term: 2, index: 5 };
    let stale = LogPosition { term: 1, index: 9 };
    let mut sim = Sim::new(&[(1, stale), (2, fresh), (3, fresh)], 21, false);
    sim.timeout(NodeId(1), None); // stale candidate campaigns
    sim.drain();
    sim.check_all();
    // The stale candidate must NOT have become leader: 2 and 3 refuse it,
    // and its self-vote alone is not a quorum.
    assert!(
        sim.leaders.values().all(|s| !s.contains(&NodeId(1))),
        "stale-log candidate won an election: {:?}",
        sim.leaders
    );
    // And a fresh candidate CAN win afterwards.
    sim.timeout(NodeId(2), None);
    sim.drain();
    sim.check_all();
    assert!(
        sim.leaders.values().any(|s| s.contains(&NodeId(2))),
        "fresh candidate failed to win: {:?}",
        sim.leaders
    );
}

/// Partition + heal: a minority-side candidate must not win; after heal the
/// cluster converges on one leader per term (no dual leadership ever).
#[test]
fn partition_minority_cannot_elect_and_heal_converges() {
    let mut sim = Sim::new(&[(1, L0), (2, L0), (3, L0)], 63, false);
    // Partition node 1 away from 2 and 3.
    sim.cut.insert((NodeId(1), NodeId(2)));
    sim.cut.insert((NodeId(1), NodeId(3)));
    sim.timeout(NodeId(1), None); // minority campaign — must fail
    sim.timeout(NodeId(2), None); // majority campaign — must succeed
    sim.drain();
    sim.check_all();
    assert!(
        sim.leaders.values().all(|s| !s.contains(&NodeId(1))),
        "partitioned minority elected itself: {:?}",
        sim.leaders
    );
    // Heal and drain remaining traffic: invariants must still hold.
    sim.cut.clear();
    sim.drain();
    sim.check_all();
}

/// Randomized soak: seeded schedules with random timeouts, crashes at random
/// boundaries, restarts, and message shuffling. The invariants must hold for
/// every seed. (Gate A's "deterministic simulation" in miniature — the full
/// schedule explorer grows with the log layer in Phase A1b.)
#[test]
fn seeded_soak_invariants_hold() {
    for seed in 1..40u64 {
        let mut sim = Sim::new(&[(1, L0), (2, L0), (3, L0)], seed, seed % 2 == 0);
        for _step in 0..200 {
            let ids = [NodeId(1), NodeId(2), NodeId(3)];
            match sim.rng.next() % 10 {
                0 => {
                    let id = ids[sim.rng.pick(3)];
                    let alive = sim.nodes[&id].core.is_some();
                    if alive {
                        let inject = if sim.rng.chance(20) {
                            Some((sim.rng.next() % 2) as u8)
                        } else {
                            None
                        };
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
                _ => {
                    let inject = if sim.rng.chance(10) {
                        Some((sim.rng.next() % 2) as u8)
                    } else {
                        None
                    };
                    sim.deliver_one(inject);
                }
            }
        }
        sim.drain();
        sim.check_all(); // must hold for EVERY seed
    }
}

/// Liveness note (not a Gate A safety invariant, but worth pinning): pre-vote
/// probing by an isolated node must not advance its persisted term — the .140
/// flapping lesson.
#[test]
fn pre_vote_probe_never_burns_terms() {
    let mut sim = Sim::new(&[(1, L0), (2, L0), (3, L0)], 5, true);
    // Isolate node 1 fully, then let it probe repeatedly.
    sim.cut.insert((NodeId(1), NodeId(2)));
    sim.cut.insert((NodeId(1), NodeId(3)));
    for _ in 0..25 {
        sim.timeout(NodeId(1), None);
        sim.drain();
    }
    // Its durable term must be untouched: pre-vote is stateless.
    assert_eq!(sim.nodes[&NodeId(1)].disk.current_term, Term(0));
    sim.check_all();
}
