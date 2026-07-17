# RFC 028 — Native Replication: purpose-built clustering for cognitive memory

Status: **Draft** (2026-07-18)
Author: Pranab Sarkar + Claude (yantrikdb-server)
Decision: Pranab, 2026-07-18 — *"focus on replication, leader selection, clustering… I opt for having our own solution because others might be too generic and not suitable for our needs."*
Constraint: **no shortcuts** (RFC first, then code), **cluster-mode validation is a required release gate**, **never-wedge boot** (the CT 141 lesson), **byte-deterministic apply** (the #104/Item 3 contract).
Supersedes: the openraft evaluation track of epic 60 (saga task 232).

## 1. Why our own — the evidence, not the vibe

We have run two consensus stacks side by side for months. The scorecard:

| | openraft path (`raft/`) | raft-lite path (`cluster/`) |
|---|---|---|
| Code we own | **~4,200 lines of integration glue** around a generic library | **~2,000 lines, all ours** |
| Production record | Cold-start **wedge** on torn state (`expected index [0, 64), got [None, None)`) — took CT 141 down for 10 days, required manual raft-state surgery, and the reset **didn't hold** because the topology regenerated it | Ran the original 2-voter cluster; degraded gracefully; its failure modes were topology mistakes (ours), not protocol refusals |
| Fit to workload | Generic totally-ordered log for arbitrary state machines; fights our reality (idempotent ops, similarity reads, single-writer-mostly) | Built around the **oplog** — the engine's own replication primitive (`extract_ops_since` / `apply_ops` / watermarks / HLC) |
| Operational surface | SplitRuntime + SCHED_FIFO CPU isolation, snapshot streaming, membership-change API, mixed-version snapshot incompatibilities | HTTP pull/push + watermarks; a shell script can reason about it |
| Restart posture | **Refuses to start** when persisted state disagrees with the log — correct for a bank, wrong for a memory substrate | Restarts and re-syncs |

The generic solution's core assumption — *the log is sacred; halt rather than proceed* — is the wrong default for YantrikDB. A cognitive memory substrate's availability model is closer to a distributed cache with durable provenance than to a financial ledger: **ops are idempotent (UUIDv7 `op_id`), convergence is provable (the CRDT convergence suite), and the correct response to confusion is re-sync, not refusal.**

Meanwhile every determinism problem we actually hit this year was solved **outside** the consensus library, in our own layers: embedding bytes on the wire (#52 / engine Item 3), provenance admission (`WriteAdmission`), origin-actor preservation (core #69), capability exchange (#53). The generic library contributed the wedge; our purpose-built layers contributed the guarantees.

## 2. Goals

1. **One replication stack, ours.** Harden the raft-lite lineage into the production protocol; deprecate and remove the openraft path.
2. **Never-wedge boot.** A node always starts. Torn, stale, or alien local replication state triggers **automatic re-bootstrap from the leader** — recovery is a protocol state, not an operator runbook. (CT 141 must never happen again.)
3. **Memory-native semantics.** The unit of replication is the **oplog entry** (idempotent, HLC-ordered, embedding-bytes-carrying, provenance-preserving), applied through `apply_ops` under the same admission rules as any peer ingress.
4. **Small-fleet honesty.** Designed for 1–5 nodes on homelab/edge hardware: leader + replicas + optional witness. Election tuned for 2-core LXCs under embedding load. No pretense of 100-node federations.
5. **Deterministic apply, everywhere.** No embedder calls at apply time; vectors travel in ops (Item 3); generation cutovers are leader-coordinated (adapting #104's option (c)).
6. **Chaos-proven.** The protocol ships with a chaos suite (kill-leader, partition, torn-state, stale-rejoin, disk-full, mixed-version) wired into CI as the cluster-mode release gate.

### Non-goals
- Multi-writer / multi-leader (single-writer-mostly is the workload; origin-ingress admission assumes it).
- Byzantine tolerance, dynamic large-scale membership, geo-replication.
- Linearizable reads from followers (recall is similarity search; staleness is bounded and surfaced, not forbidden — `?min_seq` RYW covers the case that matters).

## 3. The protocol (YRP — YantrikDB Replication Protocol)

Three planes, all evolving the existing `cluster/` modules:

### 3.1 Control plane — leader election (evolves `election.rs`, `heartbeat.rs`, `state.rs`)
- Term-based election with the existing roles: **Voter, ReadReplica, Witness, Standalone**. Witness (CT 142's role) votes but stores no data — the 2-node tiebreaker we already built.
- **Pre-vote + reachability check** before starting a real election: a candidate that can't reach a majority (or is behind on watermark) doesn't bump terms — kills the NAT-asymmetry flapping the .140 Docker voter caused.
- **Leader lease + heartbeat** with timeouts sized for 2-core LXCs under embedding bursts (epic 60 finding): election_timeout defaults up, heartbeat runs on the control runtime so recall CPU can't starve it.
- Watermark-aware voting: a voter refuses to elect a candidate whose oplog watermark is behind its own — the leader is always a maximal node.

### 3.2 Data plane — oplog replication (evolves `replication.rs`, `sync_loop.rs`)
- **Pull-based catch-up + push-based hot path**: followers pull `extract_ops_since(watermark)`; the leader pushes new ops to connected followers for low lag. Both idempotent — `apply_ops` dedupes by `op_id`.
- Wire format: `OplogEntryWire` **v2** (already shipped, #52) — embedding bytes for text-changing corrects, `origin_actor`, HLC, `format_version`.
- **Capability exchange at session start** (#53): peers advertise supported format versions; v2→v1 pairings downgrade explicitly or refuse determinism-carrying ops. This lands as part of YRP, not as a someday.
- Apply path: `apply_ops` with `WriteAdmission::Admitted` semantics — admission happened once at the leader's origin ingress; replicas never re-gate (the wedge-prevention rule, now protocol law).
- **Generation cutover** (reembed): leader-coordinated per #104(c) — the leader re-embeds and streams vector-carrying correct-ops; followers never embed. The cutover marker is an O(1) oplog entry; snapshot bootstraps carry the active generation.

### 3.3 Recovery plane — the never-wedge invariant (new: `bootstrap.rs`)
State machine on boot, replacing "refuse and crash-loop":

```
boot → local state consistent?
  ├─ yes → join(term, watermark) → catch-up pull → follower
  ├─ no  (torn/corrupt/alien)
  │      → quarantine local replication state (timestamped .bak, like the CT 141 manual procedure — automated)
  │      → snapshot bootstrap from leader (per-namespace yantrik.db checkpoint stream + watermark)
  │      → follower
  └─ no leader reachable → serve reads standalone-degraded (accepts_writes=false), keep retrying join
```
- Snapshot bootstrap = the engine's own database files + a watermark, not a consensus-library-specific snapshot format — which is what made mixed-version snapshot sync fail under openraft.
- An operator command (`yantrikdb cluster rejoin --from-leader`) exposes the same path manually; the protocol just runs it itself when needed.

### 3.4 Consistency contract (what we promise, honestly)
- **Durability**: an acked write is durable on the leader; `replication_mode = sync` waits for N follower acks (existing config).
- **Convergence**: all nodes that receive the same op set reach identical state (CRDT convergence suite is the proof harness; byte-determinism from Item 3 + no-embed-at-apply).
- **Read-your-writes**: `?min_seq` watermark waits (already shipped, Phase 6 RYW).
- **Failover**: on leader loss, a maximal-watermark voter wins election; unreplicated leader-tail ops are lost only in async mode — measured and surfaced via `replication_lag` metrics, bounded by push-path lag (~heartbeat interval), and eliminated in sync mode.

## 4. Migration

| Phase | Content |
|---|---|
| **A** (this arc) | RFC review → harden control plane (pre-vote, watermark voting, lease, timeouts) + recovery plane (`bootstrap.rs`, never-wedge) with unit + sim tests |
| **B** | Data-plane completeness: push path, capability exchange (#53), generation-cutover entry; chaos suite in CI (kill-leader, partition, torn-state, stale-rejoin, mixed-version) |
| **C** | 3-node homelab deployment (the epic-60 topology: always-on LXC voters + witness), soak, failover RTO benchmark (epic 59 task) |
| **D** | Deprecate `raft_mode = "openraft"` (config warning → refuse-with-migration-hint), delete `raft/` (~4,200 lines) and the openraft dependency |

Existing deployments are standalone (`raft_mode = "disabled"`) — untouched through all phases. The `commit_log`/`MutationCommitter` write path stays; YRP replaces only the consensus/replication layer beneath it.

## 5. Risks

| Risk | Mitigation |
|---|---|
| "Writing your own consensus" is famously hard | We are **not** writing Paxos. Single-writer election + idempotent oplog shipping + automatic re-sync is a far smaller correctness surface than a general totally-ordered log, and the convergence suite + chaos gate make claims mechanical, not vibes. The generic alternative already failed us in production. |
| Election edge cases (split vote, clock skew) | Pre-vote, randomized election timeouts, HLC (already in the oplog) for ordering; witness for even-node tiebreak; chaos suite exercises partitions. |
| Async-mode tail loss on failover | Documented + measured (`replication_lag`); sync mode for tenants that need zero-loss; push path keeps the window ~heartbeat-sized. |
| Never-wedge auto-resync destroys forensic state | Quarantine-first: the torn state is preserved timestamped on disk (automated version of the CT 141 manual backup) before any re-bootstrap. |

## 6. Acceptance (phase A)
1. Pre-vote + watermark-aware election with sim tests (no live cluster needed).
2. `bootstrap.rs` never-wedge path: a node with deliberately-torn state boots, quarantines, re-syncs, serves — as a test.
3. Election timing under CPU-load sim (2-core constraint) documented defaults.
4. Full workspace suite + cluster-mode gate green; zero behavior change for standalone deployments.
