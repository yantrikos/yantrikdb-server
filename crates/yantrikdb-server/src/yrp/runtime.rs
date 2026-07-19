//! YRP runtime assembly — config → boot inspection → driver (or
//! quarantine) → the server-facing handle and committer.
//!
//! This is the `raft_mode = "yrp"` startup path (RFC 028 §5 posture):
//! - [`spawn`] loads durable state, runs [`super::bootstrap::inspect`],
//!   and starts EITHER the live driver (Healthy) or the fail-closed
//!   quarantine/rejoin loop (anything less). **The process always
//!   starts** — quarantine serves diagnostics and refuses writes; it
//!   never wedges boot.
//! - [`YrpHandle`] is what the HTTP layer holds: the owner funnel, the
//!   outcome store (dedupe answers), a live status watch, and the
//!   quarantine surface.
//! - [`YrpCommitter`] implements [`MutationCommitter`] over the driver,
//!   so the existing unkeyed write path replicates without handler
//!   changes (reads delegate to the local commit log, which every node
//!   materializes at apply time).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot, watch};

use super::bootstrap::{
    inspect, BootDecision, BootstrapEffect, Integrity, QuarantinedNode, RecoveredState,
    RejoinMessage,
};
use super::driver::{
    run_apply_worker, spawn_ticker, BarrierOutcome, DriverConfig, DriverEvent, DriverExit,
    DurableState, FileStore, ProposeOutcome, Transport, WireMsg, YrpDriver, YrpStatus,
};
use super::engine_sink::{AppliedOutcome, EngineApplySink, OutcomeStore};
use super::op::{claim_key_for_op, YrpOp};
use super::replica::{Payload, Role};
use super::transport::HttpTransport;
use super::types::{ClusterId, NodeId};
use crate::commit::{
    Applier, CommitError, CommitOptions, CommitReceipt, CommittedEntry, MemoryMutation,
    MutationCommitter, OpId, TenantId,
};

/// One cluster member from the `[yrp]` config section.
#[derive(Debug, Clone)]
pub struct YrpPeer {
    pub node_id: u64,
    /// HTTP base url other nodes reach it at (scheme://host:http_port).
    pub addr: String,
    pub witness: bool,
}

/// Everything [`spawn`] needs, resolved from `ServerConfig`.
#[derive(Debug, Clone)]
pub struct YrpRuntimeConfig {
    pub node_id: u64,
    pub cluster_id: u64,
    /// ALL cluster members, INCLUDING this node (self is identified by
    /// `node_id`; its addr entry is ignored for outbound sends).
    pub peers: Vec<YrpPeer>,
    pub data_dir: PathBuf,
    pub cluster_secret: Option<String>,
    pub tick_ms: u64,
    pub election_ticks: (u32, u32),
    pub heartbeat_ticks: u32,
    /// Compact once the applied span exceeds this (0 = disabled — the
    /// production default until Phase C ships engine-checkpoint transfer
    /// for beyond-GC stragglers; the protocol path is chaos-tested).
    pub compact_after_entries: u64,
    /// Leader retention margin (entries kept above the compaction base).
    pub leader_retain_entries: u64,
}

/// How long a proposer waits for the driver's reply / the apply marker.
const PROPOSE_TIMEOUT: Duration = Duration::from_secs(15);
const OUTCOME_POLL: Duration = Duration::from_millis(10);
/// Quarantine rejoin retry cadence.
const REJOIN_RETRY: Duration = Duration::from_secs(2);

/// Propose-path failures, mapped to [`CommitError`]/HTTP by callers.
#[derive(Debug)]
pub enum YrpProposeError {
    NotLeader {
        leader_id: Option<u64>,
        leader_addr: Option<String>,
    },
    Timeout,
    Unavailable(String),
}

/// What the HTTP layer holds for a YRP node.
pub struct YrpHandle {
    pub node_id: NodeId,
    owner_tx: mpsc::UnboundedSender<DriverEvent>,
    pub outcomes: Arc<OutcomeStore>,
    pub status: watch::Receiver<YrpStatus>,
    /// node id → HTTP base url, for leader redirects.
    peer_http: BTreeMap<u64, String>,
    pub cluster_secret: Option<String>,
    /// `Some(reasons)` while quarantined (or after a fatal driver exit);
    /// `None` when replicating normally. The health surface reports it;
    /// the write path refuses on it.
    quarantine: std::sync::RwLock<Option<Vec<String>>>,
}

impl YrpHandle {
    pub fn quarantine_reasons(&self) -> Option<Vec<String>> {
        self.quarantine.read().expect("quarantine lock").clone()
    }

    fn set_quarantine(&self, reasons: Option<Vec<String>>) {
        *self.quarantine.write().expect("quarantine lock") = reasons;
    }

    /// Forward a decoded inbound wire message to whichever loop currently
    /// owns the funnel (driver or quarantine).
    pub fn deliver(&self, from: u64, msg: WireMsg) -> Result<(), String> {
        super::transport::deliver(&self.owner_tx, from, msg)
    }

    /// Linearizable-read barrier: resolves Ok once every write committed
    /// before this call is durably applied locally, with the no-op commit
    /// itself proving leadership at the linearization point. `Err` maps
    /// exactly like propose failures (NotLeader with hint / Timeout).
    pub async fn read_barrier(&self) -> Result<(), YrpProposeError> {
        if let Some(reasons) = self.quarantine_reasons() {
            return Err(YrpProposeError::Unavailable(format!(
                "node quarantined: {reasons:?}"
            )));
        }
        let (tx, rx) = oneshot::channel();
        self.owner_tx
            .send(DriverEvent::ReadBarrier { reply: tx })
            .map_err(|_| YrpProposeError::Unavailable("YRP driver not running".into()))?;
        let out = tokio::time::timeout(PROPOSE_TIMEOUT, rx)
            .await
            .map_err(|_| YrpProposeError::Timeout)?
            .map_err(|_| YrpProposeError::Unavailable("YRP driver dropped reply".into()))?;
        match out {
            BarrierOutcome::Ok => Ok(()),
            BarrierOutcome::Retry => {
                let (leader_id, leader_addr) = self.leader_hint();
                Err(YrpProposeError::NotLeader {
                    leader_id,
                    leader_addr,
                })
            }
        }
    }

    /// Graceful stop of whichever loop owns the funnel (tests/shutdown).
    pub fn shutdown(&self) {
        let _ = self.owner_tx.send(DriverEvent::Shutdown);
    }

    /// True once the owning loop has exited (its receiver dropped). A
    /// killer that intends to mutate the node's on-disk state MUST wait
    /// for this — Shutdown is queued behind in-flight events, and a
    /// still-draining driver may persist over external modifications.
    pub fn is_stopped(&self) -> bool {
        self.owner_tx.is_closed()
    }

    /// Current leader hint as (id, http addr).
    pub fn leader_hint(&self) -> (Option<u64>, Option<String>) {
        let leader = self.status.borrow().leader.map(|n| n.0);
        let addr = leader.and_then(|id| self.peer_http.get(&id).cloned());
        (leader, addr)
    }

    pub fn is_leader(&self) -> bool {
        self.quarantine_reasons().is_none() && self.status.borrow().role == Role::Leader
    }

    /// Propose a keyed op and wait for its durable outcome. This is the
    /// single funnel both the keyed gateway path and [`YrpCommitter`]
    /// ride: claim checked at origin (RFC 028 §7), ack released only at
    /// the durable-apply marker, dedupe answered from the outcome store.
    pub async fn propose_and_wait(
        &self,
        key: u64,
        op: &YrpOp,
    ) -> Result<AppliedOutcome, YrpProposeError> {
        if let Some(reasons) = self.quarantine_reasons() {
            return Err(YrpProposeError::Unavailable(format!(
                "node quarantined: {reasons:?}"
            )));
        }
        let bytes = op.encode().map_err(YrpProposeError::Unavailable)?;
        let (tx, rx) = oneshot::channel();
        self.owner_tx
            .send(DriverEvent::Propose {
                key,
                payload: Payload::Op(bytes),
                reply: tx,
            })
            .map_err(|_| YrpProposeError::Unavailable("YRP driver not running".into()))?;
        let outcome = tokio::time::timeout(PROPOSE_TIMEOUT, rx)
            .await
            .map_err(|_| YrpProposeError::Timeout)?
            .map_err(|_| YrpProposeError::Unavailable("YRP driver dropped reply".into()))?;
        let index = match outcome {
            ProposeOutcome::Applied { index } | ProposeOutcome::Duplicate { index } => index,
            ProposeOutcome::Retry => {
                let (leader_id, leader_addr) = self.leader_hint();
                return Err(YrpProposeError::NotLeader {
                    leader_id,
                    leader_addr,
                });
            }
        };
        self.wait_outcome(index).await
    }

    /// Wait for the durable-apply marker to cover `index`, then read its
    /// outcome. (An `Applied` reply already implies coverage; `Duplicate`
    /// may race a lagging apply worker — poll briefly.)
    async fn wait_outcome(&self, index: u64) -> Result<AppliedOutcome, YrpProposeError> {
        let deadline = tokio::time::Instant::now() + PROPOSE_TIMEOUT;
        while self.outcomes.applied() < index {
            if tokio::time::Instant::now() >= deadline {
                return Err(YrpProposeError::Timeout);
            }
            tokio::time::sleep(OUTCOME_POLL).await;
        }
        self.outcomes
            .lookup_by_index(index)
            .map_err(YrpProposeError::Unavailable)?
            .ok_or_else(|| {
                YrpProposeError::Unavailable(format!("applied index {index} has no outcome record"))
            })
    }
}

/// Boot the YRP node. Never fails on damaged replication state (that is
/// quarantine's job) — only on genuinely unusable local resources
/// (unopenable outcome DB, malformed config).
pub fn spawn(
    cfg: YrpRuntimeConfig,
    local: Arc<dyn MutationCommitter>,
    applier: Arc<dyn Applier>,
) -> Result<Arc<YrpHandle>, String> {
    if cfg.cluster_id == 0 {
        return Err("[yrp] cluster_id must be non-zero".into());
    }
    if cfg.node_id == 0 {
        return Err("[cluster] node_id must be non-zero in yrp mode".into());
    }
    if !cfg.peers.iter().any(|p| p.node_id == cfg.node_id) {
        return Err("[yrp] peers must include this node's node_id".into());
    }

    if cfg.compact_after_entries > 0 {
        // Codex chaos-review P0, made loud: until Phase C ships
        // engine-checkpoint transfer, a straggler that falls below the
        // compaction base receives a PROTOCOL snapshot (claims/active)
        // but no engine backfill for the compacted range. Enabling
        // compaction is a chaos-test/operator-experiment posture, not a
        // production default.
        tracing::warn!(
            compact_after = cfg.compact_after_entries,
            "[yrp] log compaction ENABLED: beyond-GC stragglers rejoin without \
             engine backfill for the compacted range until Phase C \
             (engine-checkpoint transfer). Not recommended in production."
        );
    }

    let me = NodeId(cfg.node_id);
    let cluster = ClusterId(cfg.cluster_id);
    let state_path = cfg.data_dir.join("yrp.state");
    let outcomes = Arc::new(OutcomeStore::open(cfg.data_dir.join("yrp_apply.sqlite"))?);

    let voters: BTreeSet<NodeId> = cfg.peers.iter().map(|p| NodeId(p.node_id)).collect();
    let witnesses: BTreeSet<NodeId> = cfg
        .peers
        .iter()
        .filter(|p| p.witness)
        .map(|p| NodeId(p.node_id))
        .collect();
    let peer_http: BTreeMap<u64, String> = cfg
        .peers
        .iter()
        .map(|p| (p.node_id, p.addr.clone()))
        .collect();
    let peer_urls: BTreeMap<NodeId, String> = cfg
        .peers
        .iter()
        .filter(|p| p.node_id != cfg.node_id)
        .map(|p| (NodeId(p.node_id), p.addr.clone()))
        .collect();
    let data_peers: Vec<NodeId> = cfg
        .peers
        .iter()
        .filter(|p| p.node_id != cfg.node_id && !p.witness)
        .map(|p| NodeId(p.node_id))
        .collect();

    let transport = Arc::new(HttpTransport::new(
        me,
        peer_urls,
        cfg.cluster_secret.clone(),
    ));

    let store = FileStore::new(state_path.clone());
    // Boot inspection input. bincode round-trip success stands in for the
    // record checksum (a torn/truncated file fails deserialization); a
    // dedicated hash-chain lands with the Phase C manifest work.
    let (restored, recovered) = match store.load() {
        Ok(Some(d)) => {
            let rec = RecoveredState {
                cluster_id: Some(d.cluster_id),
                hard: Some(d.hard),
                log: Some(d.log.clone()),
                active: d.active,
                // Marker is absolute; inspect compares against the log
                // suffix — normalize by the compaction base.
                commit_marker: outcomes.applied().saturating_sub(d.base.index),
                integrity: Integrity {
                    hard_state_verified: true,
                    log_verified: true,
                },
            };
            (Some(d), rec)
        }
        Ok(None) => {
            let rec = RecoveredState {
                cluster_id: None,
                hard: None,
                log: None,
                active: 0,
                // Applied engine state with NO replication state is the
                // frontier-beyond-data inconsistency — quarantine.
                commit_marker: outcomes.applied(),
                integrity: Integrity {
                    hard_state_verified: true,
                    log_verified: true,
                },
            };
            (None, rec)
        }
        Err(e) => {
            tracing::error!(error = %e, "yrp.state unreadable — boot inspection will quarantine");
            let rec = RecoveredState {
                cluster_id: None,
                hard: None,
                log: None,
                active: 0,
                commit_marker: outcomes.applied(),
                integrity: Integrity {
                    hard_state_verified: false,
                    log_verified: false,
                },
            };
            (None, rec)
        }
    };

    let (owner_tx, owner_rx) = mpsc::unbounded_channel();
    let (apply_tx, apply_rx) = mpsc::unbounded_channel();
    let (status_tx, status_rx) = watch::channel(YrpStatus::default());

    let handle = Arc::new(YrpHandle {
        node_id: me,
        owner_tx: owner_tx.clone(),
        outcomes: outcomes.clone(),
        status: status_rx,
        peer_http,
        cluster_secret: cfg.cluster_secret.clone(),
        quarantine: std::sync::RwLock::new(None),
    });

    let sink = EngineApplySink::new(local, applier, outcomes.clone());
    tokio::spawn(run_apply_worker(Box::new(sink), apply_rx, owner_tx.clone()));
    spawn_ticker(owner_tx.clone(), Duration::from_millis(cfg.tick_ms.max(1)));

    let driver_cfg = move || DriverConfig {
        id: me,
        cluster_id: cluster,
        voters: voters.clone(),
        witnesses: witnesses.clone(),
        supported: u32::MAX,
        election_ticks: cfg.election_ticks,
        heartbeat_ticks: cfg.heartbeat_ticks,
        compact_after: (cfg.compact_after_entries > 0).then_some(cfg.compact_after_entries),
        leader_retain: cfg.leader_retain_entries,
    };

    match inspect(cluster, u32::MAX, &recovered) {
        BootDecision::Healthy { .. } => {
            tracing::info!(
                node = cfg.node_id,
                cluster = cfg.cluster_id,
                "YRP boot: healthy"
            );
            let mut driver = YrpDriver::new(
                driver_cfg(),
                restored,
                store,
                Box::new(SharedTransport(transport)),
                apply_tx,
                outcomes.applied(),
            );
            driver.set_status_tx(status_tx);
            spawn_driver(driver, owner_rx, handle.clone());
        }
        BootDecision::Quarantine { reasons, term_hint } => {
            tracing::error!(
                ?reasons,
                "YRP boot: QUARANTINED (fail closed, serving diagnostics)"
            );
            handle.set_quarantine(Some(reasons.iter().map(|r| format!("{r:?}")).collect()));
            let node = QuarantinedNode::new(me, cluster, reasons, term_hint);
            let ctx = QuarantineCtx {
                node,
                store,
                state_path,
                transport,
                apply_tx,
                status_tx,
                data_peers,
                outcomes,
                driver_cfg: driver_cfg(),
                handle: handle.clone(),
            };
            tokio::spawn(run_quarantined(ctx, owner_rx));
        }
    }
    Ok(handle)
}

/// `Transport` over a shared [`HttpTransport`] so the quarantine loop and
/// the driver can each hold one.
struct SharedTransport(Arc<HttpTransport>);
impl Transport for SharedTransport {
    fn send(&self, to: NodeId, msg: WireMsg) {
        self.0.send(to, msg)
    }
}

fn spawn_driver(
    driver: YrpDriver,
    owner_rx: mpsc::UnboundedReceiver<DriverEvent>,
    handle: Arc<YrpHandle>,
) {
    tokio::spawn(async move {
        let exit = driver.run(owner_rx).await;
        match exit {
            DriverExit::Shutdown => {
                tracing::info!("YRP driver shut down");
            }
            other => {
                // Fail-stop posture: surface it on health and refuse
                // writes; the operator restarts through boot inspection.
                tracing::error!(
                    ?other,
                    "YRP driver FAILED — node degraded to quarantine posture"
                );
                handle.set_quarantine(Some(vec![format!("driver exit: {other:?}")]));
            }
        }
    });
}

struct QuarantineCtx {
    node: QuarantinedNode,
    store: FileStore,
    state_path: PathBuf,
    transport: Arc<HttpTransport>,
    apply_tx: mpsc::UnboundedSender<(u64, super::replica::LogEntry)>,
    status_tx: watch::Sender<YrpStatus>,
    data_peers: Vec<NodeId>,
    outcomes: Arc<OutcomeStore>,
    driver_cfg: DriverConfig,
    handle: Arc<YrpHandle>,
}

/// The fail-closed loop: retry rejoin against data peers round-robin;
/// ignore everything except grants (no vote handler, no append handler —
/// fail-closed by absence); on an authorized grant, persist the adopted
/// snapshot and hand the SAME owner funnel to a fresh driver.
async fn run_quarantined(mut ctx: QuarantineCtx, mut rx: mpsc::UnboundedReceiver<DriverEvent>) {
    let mut retry = tokio::time::interval(REJOIN_RETRY);
    retry.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut peer_rr = 0usize;
    loop {
        tokio::select! {
            _ = retry.tick() => {
                if ctx.data_peers.is_empty() {
                    continue;
                }
                let target = ctx.data_peers[peer_rr % ctx.data_peers.len()];
                peer_rr += 1;
                for eff in ctx.node.tick_rejoin(target) {
                    run_bootstrap_effect(&mut ctx, eff);
                }
            }
            ev = rx.recv() => {
                let Some(ev) = ev else { return };
                match ev {
                    DriverEvent::Shutdown => return,
                    DriverEvent::Inbound { from, msg: WireMsg::Rejoin(grant @ RejoinMessage::Grant { .. }) } => {
                        for eff in ctx.node.on_grant(from, grant.clone()) {
                            if let BootstrapEffect::AdoptSnapshot { cluster_id, hard, base, log, claims, active } = eff {
                                let adopted = DurableState { cluster_id, hard, base, log, claims, active };
                                if let Err(e) = ctx.store.persist(&adopted) {
                                    tracing::error!(error = %e, "adopt-snapshot persist failed; staying quarantined");
                                    continue;
                                }
                                tracing::info!(?from, "YRP rejoin: snapshot adopted — resuming as follower");
                                ctx.handle.set_quarantine(None);
                                let mut driver = YrpDriver::new(
                                    ctx.driver_cfg,
                                    Some(adopted),
                                    ctx.store,
                                    Box::new(SharedTransport(ctx.transport)),
                                    ctx.apply_tx,
                                    ctx.outcomes.applied(),
                                );
                                driver.set_status_tx(ctx.status_tx);
                                spawn_driver(driver, rx, ctx.handle);
                                return;
                            }
                        }
                    }
                    // Votes/appends while quarantined: fail-closed by
                    // absence — no handler exists to answer them.
                    _ => {}
                }
            }
        }
    }
}

fn run_bootstrap_effect(ctx: &mut QuarantineCtx, eff: BootstrapEffect) {
    match eff {
        BootstrapEffect::PreserveOldState => {
            let ts = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let dst = ctx.state_path.with_extension(format!("preserved-{ts}"));
            match std::fs::copy(&ctx.state_path, &dst) {
                Ok(_) => tracing::warn!(dst = %dst.display(), "quarantine: old state preserved"),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => tracing::error!(error = %e, "quarantine: preserve-old-state failed"),
            }
        }
        BootstrapEffect::Alarm { reasons } => {
            tracing::error!(
                ?reasons,
                "YRP QUARANTINE ALARM: corruption evidence — operator attention required"
            );
        }
        BootstrapEffect::Send { to, msg } => {
            ctx.transport.send(to, WireMsg::Rejoin(msg));
        }
        BootstrapEffect::AdoptSnapshot { .. } => {
            unreachable!("adopt is handled inline by run_quarantined")
        }
    }
}

/// [`MutationCommitter`] over the YRP driver — the yrp-mode replacement
/// for `LocalSqliteSubmitter`/`RaftCommitter`. Every mutation becomes a
/// keyed proposal (key derived from the op_id), giving the protocol layer
/// the same `(tenant, op_id)` retry-idempotency the commit log enforces
/// at the storage layer. Reads delegate to the local commit log, which
/// the apply sink materializes identically on every node.
pub struct YrpCommitter {
    handle: Arc<YrpHandle>,
    local: Arc<dyn MutationCommitter>,
}

impl YrpCommitter {
    pub fn new(handle: Arc<YrpHandle>, local: Arc<dyn MutationCommitter>) -> Self {
        Self { handle, local }
    }
}

pub fn propose_err_to_commit(e: YrpProposeError, op_id: OpId) -> CommitError {
    match e {
        YrpProposeError::NotLeader {
            leader_id,
            leader_addr,
        } => CommitError::NotLeader {
            leader_id,
            leader_addr,
        },
        YrpProposeError::Timeout => CommitError::CommitTimeout { op_id },
        YrpProposeError::Unavailable(m) => CommitError::StorageFailure { message: m },
    }
}

#[async_trait]
impl MutationCommitter for YrpCommitter {
    async fn commit(
        &self,
        tenant_id: TenantId,
        mutation: MemoryMutation,
        opts: CommitOptions,
    ) -> Result<CommitReceipt, CommitError> {
        if let Some(expected) = opts.expected_log_index {
            return Err(CommitError::StorageFailure {
                message: format!("expected_log_index ({expected}) is not supported in yrp mode"),
            });
        }
        // Codex F2: refuse unimplemented grammar variants BEFORE they
        // enter the replicated log. Without this, an entry with no apply
        // path would still advance the marker and ack the client with no
        // engine effect (same pre-check RaftCommitter performs).
        if !mutation.is_implemented() {
            return Err(CommitError::NotYetImplemented {
                variant: mutation.variant_name(),
                planned_rfc: mutation.planned_rfc(),
            });
        }
        // Codex F3: the u64 claim digest can collide across DISTINCT
        // op_ids (birthday bound over server-generated ids). A collision
        // would dedupe a fresh write against an unrelated entry — detect
        // it by verifying the outcome's op_id, and resolve by retrying
        // under a fresh op_id (fresh id → fresh digest). Bounded: two
        // independent collisions in a row are beyond astronomical.
        let mut op_id = opts.op_id.unwrap_or_else(OpId::new_random);
        for attempt in 0..3 {
            let op = YrpOp {
                tenant_id,
                op_id,
                mutation: mutation.clone(),
                idempotency_key: None,
            };
            let key = claim_key_for_op(tenant_id, &op_id);
            let outcome = self
                .handle
                .propose_and_wait(key, &op)
                .await
                .map_err(|e| propose_err_to_commit(e, op_id))?;
            if outcome.op_id != op_id.to_string() {
                tracing::warn!(
                    attempt,
                    "yrp unkeyed claim digest collision detected; retrying with fresh op_id"
                );
                // A caller-supplied op_id cannot be silently swapped —
                // its retry contract is the whole point of supplying it.
                if opts.op_id.is_some() {
                    return Err(CommitError::StorageFailure {
                        message: "claim digest collision on caller-supplied op_id".into(),
                    });
                }
                op_id = OpId::new_random();
                continue;
            }
            let applied_at = SystemTime::UNIX_EPOCH
                + Duration::from_micros(outcome.applied_at_unix_micros.max(0) as u64);
            return Ok(CommitReceipt {
                op_id,
                tenant_id,
                term: outcome.term,
                log_index: outcome.tenant_log_index,
                committed_at: applied_at,
                applied_at: Some(applied_at),
            });
        }
        Err(CommitError::StorageFailure {
            message: "repeated claim digest collisions (unkeyed)".into(),
        })
    }

    async fn read_range(
        &self,
        tenant_id: TenantId,
        from_index: u64,
        limit: usize,
    ) -> Result<Vec<CommittedEntry>, CommitError> {
        self.local.read_range(tenant_id, from_index, limit).await
    }

    async fn high_watermark(&self, tenant_id: TenantId) -> Result<u64, CommitError> {
        self.local.high_watermark(tenant_id).await
    }

    async fn list_active_tenants(&self) -> Result<Vec<TenantId>, CommitError> {
        self.local.list_active_tenants().await
    }

    /// Real linearizable-read barrier (replaces the v1 leadership-only
    /// approximation): a protocol no-op committed through the normal
    /// replicated path + a wait on the durable applied marker. A deposed
    /// leader's no-op cannot commit in its term (Gate A #2), so the
    /// stale-read window the approximation left open is closed.
    async fn ensure_linearizable(&self) -> Result<(), CommitError> {
        self.handle
            .read_barrier()
            .await
            .map_err(|e| propose_err_to_commit(e, OpId::new_random()))
    }
}
