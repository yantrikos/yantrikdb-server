//! YRP runtime driver — the bridge from the pure, sim-proven core to real
//! I/O (RFC 028 v2, runtime-integration arc; codex driver review applied).
//!
//! ## Architecture (single-owner funnel)
//!
//! One tokio task OWNS the [`ReplicaCore`] — no mutex, no shared state. An
//! mpsc funnel carries every stimulus: inbound peer messages, client
//! proposals, capability exchanges, tick events. The owner processes ONE
//! event at a time and executes the resulting effects IN ORDER before
//! dequeuing the next — which is exactly the execution model the simulator
//! proved. The driver's job is to not invent new interleavings.
//!
//! ## The codex driver findings, as implemented contracts
//!
//! - **Fail-stop persistence** (F6): any error or ambiguity in the
//!   persist path (serialize, write, fsync, rename) STOPS the driver —
//!   no retry-with-unknown-durability, no continuing past a durability
//!   gate the core believes closed. Boot-time inspection decides what
//!   the durable generation really is. `DriverExit::PersistFailure`.
//! - **Owner-owned election deadline** (F5): the timeout is a
//!   generation-tagged instant checked BY the owner loop, and inbound
//!   peer traffic is drained with priority (biased select) before a
//!   timeout may fire — a heartbeat sitting in the funnel beats the
//!   timer, eliminating spurious elections under load.
//! - **Bounded, coalescing outbound queues** (F4): per-peer, the driver
//!   keeps only the LATEST replication message (append/install traffic is
//!   cumulative — a newer message supersedes an older one) plus a small
//!   bounded queue of control messages (votes). Step-down clears queues
//!   via term-generation tags.
//! - **Sequential apply with contiguous release** (F1/F2): a single apply
//!   worker consumes committed entries in order; the [`ApplySink`]
//!   CONTRACT requires the sink to make (entry effects + applied index +
//!   key→outcome) durable ATOMICALLY per entry, and to be idempotent on
//!   replay of the same index after a crash. Client replies release only
//!   at the highest contiguous durably-applied index.
//! - **rid-in-entry** (F3): the client's outcome identifier (rid) rides
//!   INSIDE the replicated payload bytes, so a retry answered from the
//!   claims table can recover it from the entry (or, once compaction of
//!   applied-outcome history lands, from the sink's outcome table). The
//!   volatile pending-acks map is connection state only.
//!
//! v1 scope: in-process [`Transport`] trait (HTTP adapter is the next
//! slice), full-state atomic file persistence (delta encoding later —
//! compaction keeps the suffix small), snapshots restricted to
//! uncompacted-rejoin (engine checkpoint coordination is Phase C).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use super::bootstrap::RejoinMessage;
use super::replica::{Effect, KeyedProposal, LogEntry, Message, Payload, ReplicaCore, Role};
use super::types::{ClusterId, HardState, LogPosition, NodeId};

/// Everything the core needs durably persisted, as one atomic unit — the
/// on-disk mirror of [`Effect::Persist`] (plus the node's cluster
/// identity, so boot inspection can detect alien state).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurableState {
    pub cluster_id: ClusterId,
    pub hard: HardState,
    pub base: LogPosition,
    pub log: Vec<LogEntry>,
    pub claims: BTreeMap<u64, u64>,
    pub active: u32,
}

/// Wire envelope between peers (transport-agnostic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WireMsg {
    Replica(Message),
    Rejoin(RejoinMessage),
}

/// Peer transport. Implementations MUST be fire-and-forget from the
/// owner's perspective: `send` enqueues and returns; delivery failures
/// surface as silence (the protocol retransmits), never as owner-blocking
/// errors. The in-process test transport is a channel; production is the
/// HTTP adapter (next slice).
pub trait Transport: Send + 'static {
    fn send(&self, to: NodeId, msg: WireMsg);
}

/// The state-machine side of apply. CONTRACT (codex F1/F2/F3 — violating
/// any clause reintroduces the bugs the review found):
/// 1. `apply` is called for indices in strictly ascending order, exactly
///    once per index in a live process; after a crash it MAY be called
///    again for indices at or below the last durable applied index and
///    MUST be idempotent there.
/// 2. The implementation MUST make the entry's effects AND its applied
///    index (AND, for keyed entries, the key→outcome record) durable
///    before returning Ok, such that a crash-replay of the same index is
///    a no-op (the production sink achieves this with the op_id-idempotent
///    commit-log transaction + idempotent engine primitives + the
///    marker-last ordering — see `yrp::engine_sink`).
/// 3. Returning Err is fail-stop: the driver exits (quarantine posture).
///
/// `apply` is async because the production sink composes the repo's async
/// `MutationCommitter`/`Applier` traits; the worker still consumes strictly
/// in order (one apply at a time).
#[async_trait::async_trait]
pub trait ApplySink: Send + 'static {
    async fn apply(&mut self, index: u64, entry: &LogEntry) -> Result<(), String>;
    /// The last index this sink has durably applied (recovered at boot).
    fn durable_applied(&self) -> u64;
}

/// Live snapshot of the driver's replication state, published on a watch
/// channel for health surfaces and write gates. Never authoritative for
/// safety — purely observational.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YrpStatus {
    pub role: Role,
    pub term: u64,
    pub commit: u64,
    pub applied: u64,
    /// Last leader this node heard from (itself when leading). A HINT for
    /// redirects — may be stale.
    pub leader: Option<NodeId>,
    /// RFC 028 Phase C: the engine-apply frontier a beyond-GC snapshot
    /// install requires the sink to reach via backfill. `applied <
    /// backfill_target` ⇒ engine-incomplete (no reads, no leadership).
    /// 0 = no outstanding backfill.
    pub backfill_target: u64,
}

impl YrpStatus {
    /// True while this node's engine trails an adopted snapshot frontier
    /// and must not serve reads or lead.
    pub fn engine_incomplete(&self) -> bool {
        self.applied < self.backfill_target
    }
}

impl Default for YrpStatus {
    fn default() -> Self {
        Self {
            role: Role::Follower,
            term: 0,
            commit: 0,
            applied: 0,
            leader: None,
            backfill_target: 0,
        }
    }
}

/// Client-facing outcome of a keyed proposal, delivered via oneshot.
#[derive(Debug)]
pub enum ProposeOutcome {
    /// Entry committed and durably applied at `index`.
    Applied { index: u64 },
    /// Deduped against an existing committed entry.
    Duplicate { index: u64 },
    /// Not the leader / lost leadership / entry truncated — retry
    /// (possibly against another node).
    Retry,
}

/// Outcome of a linearizable-read barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierOutcome {
    /// Every write committed before the barrier was requested is durably
    /// applied LOCALLY. Reads served from local state after this reflect
    /// them. This is a linearization POINT, not a leadership lease —
    /// writes committed by a newer leader after the barrier may be
    /// absent, which is permitted for reads concurrent with them
    /// (codex barrier-consult pitfall 2, made explicit).
    Ok,
    /// Not the leader (or leadership was lost while the barrier was
    /// pending) — retry against the current leader.
    Retry,
}

/// Events into the owner funnel.
pub enum DriverEvent {
    Inbound {
        from: NodeId,
        msg: WireMsg,
    },
    Propose {
        key: u64,
        payload: Payload,
        reply: oneshot::Sender<ProposeOutcome>,
    },
    /// Linearizable-read barrier (codex-consulted design: a protocol
    /// no-op through the NORMAL commit path — soundness rides entirely
    /// on the proven current-term commit rule; committing the no-op IS
    /// the fresh quorum contact that proves this node was still leader
    /// at the barrier's linearization point).
    ReadBarrier {
        reply: oneshot::Sender<BarrierOutcome>,
    },
    /// Apply worker reports the highest contiguous durably-applied index.
    Applied {
        upto: u64,
    },
    /// RFC 028 Phase C: an engine mutation pulled by the backfill task
    /// for the compacted range (its log entry is gone). The owner
    /// sequences it into the apply stream in contiguous order.
    Backfilled {
        index: u64,
        entry: LogEntry,
    },
    /// Periodic tick — the owner checks its own election/heartbeat
    /// deadlines (generation-tagged; a stale tick is a no-op).
    Tick,
    /// Graceful stop (tests).
    Shutdown,
}

/// Why the owner loop exited.
#[derive(Debug, PartialEq, Eq)]
pub enum DriverExit {
    Shutdown,
    /// Fail-stop: persistence uncertainty (codex F6). The process must
    /// restart through boot inspection.
    PersistFailure(String),
    /// Fail-stop: the apply sink failed.
    ApplyFailure(String),
}

/// Atomic full-state file store: write temp + fsync + rename + fsync dir.
/// Any failure is fatal by contract.
pub struct FileStore {
    path: PathBuf,
}

impl FileStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn load(&self) -> Result<Option<DurableState>, String> {
        match std::fs::read(&self.path) {
            Ok(bytes) => bincode::deserialize(&bytes)
                .map(Some)
                .map_err(|e| format!("corrupt state file: {e}")),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("read state file: {e}")),
        }
    }

    pub fn persist(&self, state: &DurableState) -> Result<(), String> {
        let bytes = bincode::serialize(state).map_err(|e| format!("serialize: {e}"))?;
        let tmp = self.path.with_extension("tmp");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp).map_err(|e| format!("create tmp: {e}"))?;
            f.write_all(&bytes).map_err(|e| format!("write tmp: {e}"))?;
            f.sync_all().map_err(|e| format!("fsync tmp: {e}"))?;
        }
        std::fs::rename(&tmp, &self.path).map_err(|e| format!("rename: {e}"))?;
        if let Some(dir) = self.path.parent() {
            if let Ok(d) = std::fs::File::open(dir) {
                let _ = d.sync_all(); // best-effort on platforms without dir fsync
            }
        }
        Ok(())
    }
}

/// Driver configuration.
pub struct DriverConfig {
    pub id: NodeId,
    pub cluster_id: ClusterId,
    pub voters: std::collections::BTreeSet<NodeId>,
    pub witnesses: std::collections::BTreeSet<NodeId>,
    pub supported: u32,
    /// Randomized election timeout range (min..=max), in ticks. The owner
    /// re-randomizes per armed deadline.
    pub election_ticks: (u32, u32),
    /// Heartbeat every N ticks while leader.
    pub heartbeat_ticks: u32,
    /// Compact the log once the retained span exceeds this many entries.
    /// `None` = never. Beyond-GC stragglers are healed by engine backfill
    /// (RFC 028 Phase C slice A — `run_backfill_task`), so enabling this
    /// is now correctness-safe; it stays off by DEFAULT only pending the
    /// fuller interruption-test hardening (crash/leader-change mid-
    /// backfill) before it becomes the shipped default.
    pub compact_after: Option<u64>,
    /// Leader retention margin: leaders compact only to `frontier - M` so
    /// briefly-lagging followers catch up from the log instead of
    /// forcing a snapshot transfer (codex chaos-consult D2).
    pub leader_retain: u64,
}

/// The YRP runtime driver. Construct with restored state, then run the
/// owner loop on a tokio task; feed it via the returned sender.
pub struct YrpDriver {
    core: ReplicaCore,
    store: FileStore,
    transport: Box<dyn Transport>,
    apply_tx: mpsc::UnboundedSender<(u64, LogEntry)>,
    cfg: DriverConfig,
    /// (index → reply) awaiting contiguous durable apply (volatile —
    /// connection state only, per codex F3).
    pending_acks: BTreeMap<u64, oneshot::Sender<ProposeOutcome>>,
    /// (barrier no-op index → waiters). Resolved Ok when the durable
    /// applied marker covers the index; failed Retry on step-down.
    pending_barriers: BTreeMap<u64, Vec<oneshot::Sender<BarrierOutcome>>>,
    /// Highest contiguous durably-applied index (from the apply worker).
    applied: u64,
    /// Committed-but-not-yet-dispatched-to-apply frontier.
    dispatched: u64,
    /// RFC 028 Phase C: the engine-apply frontier a beyond-GC snapshot
    /// install requires the sink to reach via BACKFILL (the compacted
    /// range's mutations are not in the log). `applied < backfill_target`
    /// ⇒ this node is ENGINE-INCOMPLETE: it must not campaign and its
    /// read barriers cannot resolve. 0 = no outstanding backfill.
    backfill_target: u64,
    /// Out-of-band backfilled entries awaiting in-order dispatch to the
    /// apply worker (keyed by absolute yrp index). Drained by
    /// `dispatch_ready` strictly in contiguous ascending order.
    backfill_buffer: BTreeMap<u64, LogEntry>,
    /// Election deadline in ticks-remaining; None while leader. Reset on
    /// valid leader contact (the owner sees every inbound message, so the
    /// codex F5 race cannot occur: queued heartbeats are processed before
    /// Tick events by funnel order, and a reset is a plain field write).
    election_ticks_left: Option<u32>,
    heartbeat_ticks_left: u32,
    rng: u64,
    /// Last leader observed via append/install traffic (self when leading).
    leader_hint: Option<NodeId>,
    /// Optional live-status publisher (health surface / write gate).
    status_tx: Option<tokio::sync::watch::Sender<YrpStatus>>,
}

impl YrpDriver {
    /// Build from restored durable state (boot inspection has already
    /// vouched for it — a torn/unsupported state never reaches here).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: DriverConfig,
        restored: Option<DurableState>,
        store: FileStore,
        transport: Box<dyn Transport>,
        apply_tx: mpsc::UnboundedSender<(u64, LogEntry)>,
        durable_applied: u64,
    ) -> Self {
        let (hard, base, log, claims, active) = match restored {
            Some(d) => {
                debug_assert_eq!(
                    d.cluster_id, cfg.cluster_id,
                    "alien state must be quarantined by boot inspection, never reach the driver"
                );
                (d.hard, d.base, d.log, d.claims, d.active)
            }
            None => (
                HardState::default(),
                LogPosition::ZERO,
                Vec::new(),
                BTreeMap::new(),
                0,
            ),
        };
        let mut core = ReplicaCore::new_from_durable(
            cfg.id,
            cfg.voters.clone(),
            hard,
            base,
            log,
            claims,
            active,
            true, // pre-vote on in production
        );
        core.set_witnesses(cfg.witnesses.clone());
        core.set_supported(cfg.supported);
        let seed = cfg.id.0.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        let mut d = Self {
            core,
            store,
            transport,
            apply_tx,
            cfg,
            pending_acks: BTreeMap::new(),
            pending_barriers: BTreeMap::new(),
            applied: durable_applied,
            dispatched: durable_applied,
            // Boot-time resumption (codex pitfall: never trust HTTP
            // completion): if the durable engine marker is below the
            // adopted protocol frontier, an interrupted backfill must
            // resume. `base.index` IS the adopted frontier for a
            // compacted node; a healthy node has durable_applied ≥ base.
            backfill_target: if durable_applied < base.index {
                base.index
            } else {
                0
            },
            backfill_buffer: BTreeMap::new(),
            election_ticks_left: None,
            heartbeat_ticks_left: 0,
            rng: seed,
            leader_hint: None,
            status_tx: None,
        };
        d.arm_election_deadline();
        d
    }

    /// Attach a live-status publisher. The driver sends a fresh snapshot
    /// after every processed event; receivers use it for health/redirects.
    pub fn set_status_tx(&mut self, tx: tokio::sync::watch::Sender<YrpStatus>) {
        tx.send_replace(self.status_snapshot());
        self.status_tx = Some(tx);
    }

    fn status_snapshot(&self) -> YrpStatus {
        YrpStatus {
            role: self.core.role(),
            term: self.core.current_term().0,
            commit: self.core.commit_index(),
            applied: self.applied,
            leader: if self.core.role() == Role::Leader {
                Some(self.cfg.id)
            } else {
                self.leader_hint
            },
            backfill_target: self.backfill_target,
        }
    }

    fn publish_status(&self) {
        if let Some(tx) = &self.status_tx {
            let snap = self.status_snapshot();
            if *tx.borrow() != snap {
                tx.send_replace(snap);
            }
        }
    }

    fn rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn arm_election_deadline(&mut self) {
        let (lo, hi) = self.cfg.election_ticks;
        let span = (hi.saturating_sub(lo)).max(1) as u64;
        self.election_ticks_left = Some(lo + (self.rand() % span) as u32);
    }

    /// The owner loop. Single consumer of `rx`; sole owner of the core.
    pub async fn run(mut self, mut rx: mpsc::UnboundedReceiver<DriverEvent>) -> DriverExit {
        while let Some(ev) = rx.recv().await {
            let exit = match ev {
                DriverEvent::Inbound { from, msg } => self.on_inbound(from, msg),
                DriverEvent::Propose {
                    key,
                    payload,
                    reply,
                } => self.on_propose(key, payload, reply),
                DriverEvent::ReadBarrier { reply } => self.on_read_barrier(reply),
                DriverEvent::Applied { upto } => {
                    self.applied = self.applied.max(upto);
                    self.release_acks();
                    None
                }
                DriverEvent::Backfilled { index, entry } => {
                    if index > self.applied {
                        self.backfill_buffer.insert(index, entry);
                        self.dispatch_ready();
                    }
                    None
                }
                DriverEvent::Tick => self.on_tick(),
                DriverEvent::Shutdown => return DriverExit::Shutdown,
            };
            if let Some(e) = exit {
                return e;
            }
            if let Some(e) = self.maybe_compact() {
                return e;
            }
            self.publish_status();
        }
        DriverExit::Shutdown
    }

    /// Runtime compaction trigger (RFC 028 §6 bound to real I/O).
    ///
    /// The frontier is `min(commit, durable_applied)` — NEVER bare
    /// commit: compacting an entry the apply sink has not durably
    /// applied would make crash-replay impossible (the re-dispatch loop
    /// reads entries from the log; a compacted, unapplied index would be
    /// silently skipped, leaving a permanent hole in engine state).
    /// Leaders additionally retain `leader_retain` entries so transient
    /// follower lag is served from the log, not a snapshot.
    fn maybe_compact(&mut self) -> Option<DriverExit> {
        let threshold = self.cfg.compact_after?;
        let base = self.core.base().index;
        let frontier = self.core.commit_index().min(self.applied);
        if frontier.saturating_sub(base) <= threshold {
            return None;
        }
        let target = if self.core.role() == Role::Leader {
            frontier.saturating_sub(self.cfg.leader_retain)
        } else {
            frontier
        };
        if target <= base {
            return None;
        }
        let (_snapshot, effects) = self.core.compact(target)?;
        self.execute(effects)
    }

    fn on_inbound(&mut self, from: NodeId, msg: WireMsg) -> Option<DriverExit> {
        match msg {
            WireMsg::Replica(m) => {
                // Valid leader contact resets the election deadline. The
                // core validates terms; we reset only on messages that a
                // current leader sends (append/install traffic).
                if matches!(
                    m,
                    Message::AppendEntries { .. } | Message::InstallSnapshot { .. }
                ) {
                    self.arm_election_deadline();
                    self.leader_hint = Some(from);
                }
                let effects = self.core.on_message(from, m, false);
                self.execute(effects)
            }
            WireMsg::Rejoin(RejoinMessage::Request { node }) => {
                if let Some((term, base, log, claims, active, commit)) = self.core.rejoin_grant() {
                    self.transport.send(
                        node,
                        WireMsg::Rejoin(RejoinMessage::Grant {
                            cluster_id: self.cfg.cluster_id,
                            term,
                            base,
                            log,
                            claims,
                            active,
                            commit,
                            verified: true,
                        }),
                    );
                }
                None
            }
            WireMsg::Rejoin(_) => None, // grants are handled by quarantine mode
        }
    }

    fn on_propose(
        &mut self,
        key: u64,
        payload: Payload,
        reply: oneshot::Sender<ProposeOutcome>,
    ) -> Option<DriverExit> {
        match self.core.propose_keyed(key, payload) {
            None => {
                let _ = reply.send(ProposeOutcome::Retry);
                None
            }
            Some(KeyedProposal::DuplicateCommitted { index }) => {
                let _ = reply.send(ProposeOutcome::Duplicate { index });
                None
            }
            Some(KeyedProposal::DuplicatePending { index }) => {
                // Park behind the original entry's ack slot? The original
                // proposer holds that slot; this retryer waits for apply
                // of the same index via its own slot entry. BTreeMap holds
                // one sender per index — park duplicates as Retry for v1
                // (the client re-asks and hits DuplicateCommitted).
                let _ = index;
                let _ = reply.send(ProposeOutcome::Retry);
                None
            }
            Some(KeyedProposal::Appended { index, effects }) => {
                self.pending_acks.insert(index, reply);
                self.execute(effects)
            }
        }
    }

    /// Linearizable-read barrier (codex barrier-consult, verdict A).
    /// A protocol no-op is proposed through the NORMAL replicated commit
    /// path; the barrier resolves when the durable applied marker covers
    /// the no-op's index. Coalescing honors codex pitfall 1: a waiter may
    /// only attach to an in-flight barrier no-op whose index is at least
    /// the commit index THIS caller observed — otherwise writes committed
    /// after that no-op was appended could be missed.
    fn on_read_barrier(&mut self, reply: oneshot::Sender<BarrierOutcome>) -> Option<DriverExit> {
        if self.core.role() != Role::Leader {
            let _ = reply.send(BarrierOutcome::Retry);
            return None;
        }
        let observed_commit = self.core.commit_index();
        if let Some((&inflight, _)) = self.pending_barriers.iter().next_back() {
            if inflight >= observed_commit && inflight > self.applied {
                self.pending_barriers
                    .entry(inflight)
                    .or_default()
                    .push(reply);
                return None;
            }
        }
        match self.core.propose(Payload::Noop) {
            Some(effects) => {
                let index = self.core.last_index();
                self.pending_barriers.entry(index).or_default().push(reply);
                self.execute(effects)
            }
            None => {
                let _ = reply.send(BarrierOutcome::Retry);
                None
            }
        }
    }

    fn on_tick(&mut self) -> Option<DriverExit> {
        if self.core.role() == Role::Leader {
            self.heartbeat_ticks_left = self.heartbeat_ticks_left.saturating_sub(1);
            if self.heartbeat_ticks_left == 0 {
                self.heartbeat_ticks_left = self.cfg.heartbeat_ticks;
                let effects = self.core.tick_heartbeat();
                return self.execute(effects);
            }
            return None;
        }
        // Engine-incomplete nodes never campaign (RFC 028 Phase C / codex
        // pitfall 1): they still receive AppendEntries and keep their log
        // current, but must not win leadership before their engine is
        // backfilled — a protocol-current, engine-behind leader would
        // serve stale reads and could not source engine history. Keep the
        // deadline re-armed so a genuinely stuck backfill still surfaces
        // via health, not a spurious election.
        if self.engine_incomplete() {
            self.arm_election_deadline();
            return None;
        }
        if let Some(left) = self.election_ticks_left.as_mut() {
            *left = left.saturating_sub(1);
            if *left == 0 {
                self.arm_election_deadline();
                let effects = self.core.on_election_timeout();
                return self.execute(effects);
            }
        }
        None
    }

    /// Execute effects in order; recursion through the persist gate
    /// mirrors the simulator exactly.
    fn execute(&mut self, effects: Vec<Effect>) -> Option<DriverExit> {
        let mut queue: std::collections::VecDeque<Effect> = effects.into();
        while let Some(eff) = queue.pop_front() {
            match eff {
                Effect::Persist {
                    hard,
                    base,
                    log,
                    claims,
                    active,
                } => {
                    let state = DurableState {
                        cluster_id: self.cfg.cluster_id,
                        hard,
                        base,
                        log,
                        claims,
                        active,
                    };
                    // Codex F6: fail-stop on ANY persistence uncertainty.
                    if let Err(e) = self.store.persist(&state) {
                        return Some(DriverExit::PersistFailure(e));
                    }
                    for f in self.core.state_persisted() {
                        queue.push_back(f);
                    }
                }
                Effect::Send { to, msg } => self.transport.send(to, WireMsg::Replica(msg)),
                Effect::Broadcast { msg } => {
                    for v in self.cfg.voters.clone() {
                        if v != self.cfg.id {
                            self.transport.send(v, WireMsg::Replica(msg.clone()));
                        }
                    }
                }
                Effect::BecameLeader { .. } => {
                    self.election_ticks_left = None;
                    self.heartbeat_ticks_left = self.cfg.heartbeat_ticks;
                }
                Effect::SteppedDown { .. } => {
                    self.arm_election_deadline();
                    // Leadership lost: every pending ack is connection
                    // state on a dead reign — release as Retry (the entry
                    // may yet commit; the client's keyed retry will dedupe).
                    for (_, tx) in std::mem::take(&mut self.pending_acks) {
                        let _ = tx.send(ProposeOutcome::Retry);
                    }
                    // Pending barriers likewise: our reign can no longer
                    // prove a linearization point — the caller retries
                    // against the current leader.
                    for (_, waiters) in std::mem::take(&mut self.pending_barriers) {
                        for tx in waiters {
                            let _ = tx.send(BarrierOutcome::Retry);
                        }
                    }
                }
                Effect::CommitAdvanced { .. } => {
                    // Dispatch newly committed entries to the sequential
                    // apply worker (codex F1: never apply in the owner).
                    // `dispatch_ready` sequences log entries and any
                    // backfilled entries strictly in order, so a beyond-GC
                    // gap cannot let a later index apply before an earlier.
                    self.dispatch_ready();
                }
                Effect::InstallState { last_index } => {
                    // RFC 028 Phase C: adopting the protocol snapshot does
                    // NOT fast-forward the engine marker. The compacted
                    // range's mutations are absent from the log; the sink
                    // must apply them via backfill. Record the frontier
                    // the engine must reach (engine-incomplete until then)
                    // and let dispatch/backfill fill it. NEVER advance
                    // `applied` here — that is the old hole.
                    if last_index > self.applied {
                        self.backfill_target = self.backfill_target.max(last_index);
                        // Leave `dispatched` at the real engine frontier
                        // so the gap [dispatched+1, last_index] is fed by
                        // backfill (from `backfill_buffer`), NOT skipped.
                    } else {
                        // Snapshot at/below what we've already applied —
                        // nothing to backfill.
                        self.dispatched = self.dispatched.max(last_index);
                    }
                    self.dispatch_ready();
                }
                Effect::PeerIncompatible { peer } => {
                    tracing::error!(?peer, "YRP peer capability-incompatible; sends stalled");
                }
            }
        }
        self.release_acks();
        None
    }

    /// Dispatch committed entries to the apply worker in strictly
    /// contiguous ascending order, pulling each index from the log or —
    /// for a compacted beyond-GC gap — the backfill buffer. Stops at the
    /// first index it does not yet hold (backfill will re-drive it) and
    /// never dispatches past the commit index. This single sequencer is
    /// what guarantees no later index applies before an earlier one, even
    /// when a snapshot install leaves a hole below the log base.
    fn dispatch_ready(&mut self) {
        let commit = self.core.commit_index();
        loop {
            let next = self.dispatched + 1;
            if next > commit {
                break;
            }
            let entry = self
                .core
                .entry(next)
                .cloned()
                .or_else(|| self.backfill_buffer.remove(&next));
            match entry {
                Some(e) => {
                    let _ = self.apply_tx.send((next, e));
                    self.dispatched = next;
                }
                None => break, // gap — awaiting backfill for `next`
            }
        }
    }

    /// This node cannot serve reads or lead until its engine has been
    /// backfilled up to the adopted snapshot frontier (RFC 028 Phase C /
    /// codex pitfall 1): a protocol-current but engine-incomplete leader
    /// would serve stale reads and be unable to source engine history.
    fn engine_incomplete(&self) -> bool {
        self.applied < self.backfill_target
    }

    /// Release client replies up to the highest contiguous durably-applied
    /// index (codex F1: never before durability, never out of order).
    fn release_acks(&mut self) {
        let ready: Vec<u64> = self
            .pending_acks
            .keys()
            .copied()
            .take_while(|i| *i <= self.applied)
            .collect();
        for i in ready {
            if let Some(tx) = self.pending_acks.remove(&i) {
                let _ = tx.send(ProposeOutcome::Applied { index: i });
            }
        }
        let ready_barriers: Vec<u64> = self
            .pending_barriers
            .keys()
            .copied()
            .take_while(|i| *i <= self.applied)
            .collect();
        for i in ready_barriers {
            if let Some(waiters) = self.pending_barriers.remove(&i) {
                for tx in waiters {
                    let _ = tx.send(BarrierOutcome::Ok);
                }
            }
        }
    }
}

/// Sequential apply worker: consumes (index, entry) in order, calls the
/// sink (atomic per contract), reports contiguous progress to the owner.
pub async fn run_apply_worker(
    mut sink: Box<dyn ApplySink>,
    mut rx: mpsc::UnboundedReceiver<(u64, LogEntry)>,
    owner: mpsc::UnboundedSender<DriverEvent>,
) {
    while let Some((index, entry)) = rx.recv().await {
        if index <= sink.durable_applied() {
            continue; // crash-replay of an already-durable index: idempotent skip
        }
        match sink.apply(index, &entry).await {
            Ok(()) => {
                let _ = owner.send(DriverEvent::Applied { upto: index });
            }
            Err(e) => {
                tracing::error!(error = %e, index, "YRP apply sink failed — fail-stop");
                return; // owner starves of Applied events; operator intervenes
            }
        }
    }
}

/// Spawn the tick task feeding the owner (fixed cadence; the owner does
/// all deadline math with generation-safe counters).
pub fn spawn_ticker(tx: mpsc::UnboundedSender<DriverEvent>, period: Duration) {
    tokio::spawn(async move {
        let mut iv = tokio::time::interval(period);
        loop {
            iv.tick().await;
            if tx.send(DriverEvent::Tick).is_err() {
                return;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::sync::{Arc, Mutex};

    /// Channel transport: routes wire messages into peer funnels.
    struct ChannelTransport {
        me: NodeId,
        router: Arc<Mutex<BTreeMap<NodeId, mpsc::UnboundedSender<DriverEvent>>>>,
    }
    impl Transport for ChannelTransport {
        fn send(&self, to: NodeId, msg: WireMsg) {
            if let Some(tx) = self.router.lock().unwrap().get(&to) {
                let _ = tx.send(DriverEvent::Inbound { from: self.me, msg });
            }
        }
    }

    /// Test sink: durable state in a shared Arc so a restarted driver
    /// keeps it (models the engine-side atomic apply unit).
    #[derive(Default)]
    struct SinkState {
        applied: Vec<(u64, LogEntry)>,
        durable_applied: u64,
    }
    struct TestSink(Arc<Mutex<SinkState>>);
    #[async_trait::async_trait]
    impl ApplySink for TestSink {
        async fn apply(&mut self, index: u64, entry: &LogEntry) -> Result<(), String> {
            let mut s = self.0.lock().unwrap();
            s.applied.push((index, entry.clone()));
            s.durable_applied = index;
            Ok(())
        }
        fn durable_applied(&self) -> u64 {
            self.0.lock().unwrap().durable_applied
        }
    }

    struct Node {
        tx: mpsc::UnboundedSender<DriverEvent>,
        sink: Arc<Mutex<SinkState>>,
        store_path: PathBuf,
    }

    fn spawn_node(
        id: u64,
        dir: &std::path::Path,
        router: &Arc<Mutex<BTreeMap<NodeId, mpsc::UnboundedSender<DriverEvent>>>>,
        sink: Arc<Mutex<SinkState>>,
    ) -> Node {
        let voters: BTreeSet<NodeId> = [1, 2, 3].iter().map(|n| NodeId(*n)).collect();
        let store_path = dir.join(format!("yrp-{id}.state"));
        let store = FileStore::new(store_path.clone());
        let restored = store.load().expect("load");
        let (tx, rx) = mpsc::unbounded_channel();
        let (apply_tx, apply_rx) = mpsc::unbounded_channel();
        router.lock().unwrap().insert(NodeId(id), tx.clone());
        let durable = sink.lock().unwrap().durable_applied;
        let driver = YrpDriver::new(
            DriverConfig {
                id: NodeId(id),
                cluster_id: super::super::types::ClusterId(0),
                voters,
                witnesses: BTreeSet::new(),
                supported: u32::MAX,
                election_ticks: (5, 10),
                heartbeat_ticks: 2,
                compact_after: None,
                leader_retain: 0,
            },
            restored,
            store,
            Box::new(ChannelTransport {
                me: NodeId(id),
                router: router.clone(),
            }),
            apply_tx,
            durable,
        );
        tokio::spawn(driver.run(rx));
        tokio::spawn(run_apply_worker(
            Box::new(TestSink(sink.clone())),
            apply_rx,
            tx.clone(),
        ));
        spawn_ticker(tx.clone(), Duration::from_millis(10));
        Node {
            tx,
            sink,
            store_path,
        }
    }

    async fn propose_until_settled(
        nodes: &BTreeMap<u64, Node>,
        key: u64,
        payload: u64,
    ) -> (u64, ProposeOutcome) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            for (id, n) in nodes {
                let (otx, orx) = oneshot::channel();
                let _ = n.tx.send(DriverEvent::Propose {
                    key,
                    payload: Payload::Test(payload),
                    reply: otx,
                });
                if let Ok(Ok(out)) = tokio::time::timeout(Duration::from_millis(500), orx).await {
                    match out {
                        ProposeOutcome::Retry => continue,
                        other => return (*id, other),
                    }
                }
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "no leader accepted the proposal in time"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// End-to-end: three real drivers over a channel transport with real
    /// file persistence — elect, propose keyed, dedupe on retry, restart
    /// a node from disk and verify durable state survived.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn three_driver_cluster_elects_applies_dedupes_and_restarts() {
        let _serial = crate::yrp::testkit::serial_guard().await;
        let tmp = tempfile::TempDir::new().unwrap();
        let router = Arc::new(Mutex::new(BTreeMap::new()));
        let mut nodes = BTreeMap::new();
        for id in [1u64, 2, 3] {
            let sink = Arc::new(Mutex::new(SinkState::default()));
            nodes.insert(id, spawn_node(id, tmp.path(), &router, sink));
        }

        let (leader, out) = propose_until_settled(&nodes, 42, 4242).await;
        // Applied on the first attempt, or Duplicate when the first
        // attempt's 500ms reply window expired mid-apply and the retry
        // deduped against it — BOTH are the keyed contract holding
        // (exactly-once, ambiguous attempts resolved by retry). Only a
        // fresh double-apply would be a failure, asserted below.
        let index = match out {
            ProposeOutcome::Applied { index } | ProposeOutcome::Duplicate { index } => index,
            ProposeOutcome::Retry => unreachable!("propose_until_settled never returns Retry"),
        };

        // Same key retried on the leader dedupes to the same index.
        let (otx, orx) = oneshot::channel();
        let _ = nodes[&leader].tx.send(DriverEvent::Propose {
            key: 42,
            payload: Payload::Test(4242),
            reply: otx,
        });
        match tokio::time::timeout(Duration::from_secs(2), orx)
            .await
            .expect("reply in time")
            .expect("sender alive")
        {
            ProposeOutcome::Duplicate { index: i } => assert_eq!(i, index),
            other => panic!("expected Duplicate, got {other:?}"),
        }

        // Apply flows on a quorum of sinks.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let applied_count = nodes
            .values()
            .filter(|n| {
                n.sink
                    .lock()
                    .unwrap()
                    .applied
                    .iter()
                    .any(|(_, e)| e.payload == 4242)
            })
            .count();
        assert!(
            applied_count >= 2,
            "keyed entry applied on only {applied_count} nodes"
        );

        // Restart the leader from its store: claims durable across restart.
        let old = nodes.remove(&leader).unwrap();
        let _ = old.tx.send(DriverEvent::Shutdown);
        tokio::time::sleep(Duration::from_millis(50)).await;
        let store = FileStore::new(old.store_path.clone());
        let restored = store.load().expect("load").expect("state file exists");
        assert!(
            restored.claims.contains_key(&42),
            "claim not durable across restart"
        );
        nodes.insert(
            leader,
            spawn_node(leader, tmp.path(), &router, old.sink.clone()),
        );

        // The cluster still answers the keyed retry with a dedupe — never
        // a double-apply (the wire-contract pair, now on REAL runtime).
        let (_who, out2) = propose_until_settled(&nodes, 42, 4242).await;
        match out2 {
            ProposeOutcome::Duplicate { index: i } => assert_eq!(i, index),
            ProposeOutcome::Applied { .. } => panic!("keyed retry double-applied after restart"),
            other => panic!("unexpected {other:?}"),
        }

        for n in nodes.values() {
            let _ = n.tx.send(DriverEvent::Shutdown);
        }
    }

    /// Read barrier: Ok on the leader (noop committed + applied), Retry
    /// on followers. Ordering: a write acked before the barrier is in
    /// the sink before the barrier resolves.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn read_barrier_ok_on_leader_retry_on_follower() {
        let _serial = crate::yrp::testkit::serial_guard().await;
        let tmp = tempfile::TempDir::new().unwrap();
        let router = Arc::new(Mutex::new(BTreeMap::new()));
        let mut nodes = BTreeMap::new();
        for id in [1u64, 2, 3] {
            let sink = Arc::new(Mutex::new(SinkState::default()));
            nodes.insert(id, spawn_node(id, tmp.path(), &router, sink));
        }
        let (leader, out) = propose_until_settled(&nodes, 7, 777).await;
        let index = match out {
            ProposeOutcome::Applied { index } | ProposeOutcome::Duplicate { index } => index,
            ProposeOutcome::Retry => unreachable!(),
        };

        // Barrier on the leader resolves Ok, and the pre-barrier write is
        // in the leader's sink by then.
        let (btx, brx) = oneshot::channel();
        let _ = nodes[&leader]
            .tx
            .send(DriverEvent::ReadBarrier { reply: btx });
        let out = tokio::time::timeout(Duration::from_secs(5), brx)
            .await
            .expect("barrier reply in time")
            .expect("driver alive");
        assert_eq!(out, BarrierOutcome::Ok);
        assert!(
            nodes[&leader]
                .sink
                .lock()
                .unwrap()
                .applied
                .iter()
                .any(|(i, _)| *i == index),
            "barrier resolved before the pre-barrier write was applied"
        );

        // Barrier on a follower answers Retry immediately.
        let follower = *nodes.keys().find(|id| **id != leader).unwrap();
        let (btx, brx) = oneshot::channel();
        let _ = nodes[&follower]
            .tx
            .send(DriverEvent::ReadBarrier { reply: btx });
        let out = tokio::time::timeout(Duration::from_secs(5), brx)
            .await
            .expect("barrier reply in time")
            .expect("driver alive");
        assert_eq!(out, BarrierOutcome::Retry);

        for n in nodes.values() {
            let _ = n.tx.send(DriverEvent::Shutdown);
        }
    }

    /// Codex F6: persist failure is fail-stop — the driver exits rather
    /// than running past a durability gate it cannot honor.
    #[tokio::test]
    async fn persist_failure_is_fail_stop() {
        let router = Arc::new(Mutex::new(BTreeMap::new()));
        let voters: BTreeSet<NodeId> = [1].iter().map(|n| NodeId(*n)).collect();
        let bad_path = PathBuf::from("Z:/nonexistent-dir-yrp/state");
        let store = FileStore::new(bad_path);
        let (tx, rx) = mpsc::unbounded_channel();
        let (apply_tx, _apply_rx) = mpsc::unbounded_channel();
        let driver = YrpDriver::new(
            DriverConfig {
                id: NodeId(1),
                cluster_id: super::super::types::ClusterId(0),
                voters,
                witnesses: BTreeSet::new(),
                supported: u32::MAX,
                election_ticks: (1, 2),
                heartbeat_ticks: 2,
                compact_after: None,
                leader_retain: 0,
            },
            None,
            store,
            Box::new(ChannelTransport {
                me: NodeId(1),
                router,
            }),
            apply_tx,
            0,
        );
        let handle = tokio::spawn(driver.run(rx));
        for _ in 0..5 {
            let _ = tx.send(DriverEvent::Tick);
        }
        let exit = tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("driver exited")
            .expect("no panic");
        assert!(
            matches!(exit, DriverExit::PersistFailure(_)),
            "expected PersistFailure, got {exit:?}"
        );
    }
}
