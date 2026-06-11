# RFC 027 — The Active Memory Server

Status: **Draft** (2026-06-11)
Author: Pranab Sarkar + Claude Fable 5
Triggered by: yantrikdb engine **v0.8.0** ("World's Best Memory System") shipping the autonomous-hygiene API, and Fable's substrate self-critique (memory rid `019eb344`): *every close mechanism the engine has is **voluntary** — it relies on a client remembering to call it.* Write-rich, close-poor: 36 open vs 7 resolved conflicts, 16 pending triggers, 0 archived, 44/2,532 consolidated on the live corpus.
Constraint: **additive surface** (zero v1 breakage), **enterprise grade** (no band-aids), **cluster-safe** (the v0.8.18 regression is the standing reminder), **evidence-gated** (claims cite runnable harnesses).
Part of: the active-memory-server program — saga epics #55–#60.

## Motivation

Engine v0.8.0 finished the *cognition* layer. Its `maintenance` module docstring states the boundary precisely:

> "every hygiene mechanism the engine has (consolidation, conflict resolution, trigger expiry, importance correction) exists, but closing them was voluntary — and the substrate itself documents that voluntary agent protocols don't survive drift. … The engine deliberately does NOT own a timer thread — a storage engine scheduling itself is the wrong boundary. It exposes `run_maintenance_cycle()`; **the host decides the cadence**."

**yantrikdb-server is that host.** This is the architectural seam: the engine provides idempotent, safe, fault-isolated passes; the server provides time, lifecycle, trust-on-the-wire, push, fleet coordination, and proof. Every competitor (mem0, Zep, Letta, Hermes's pluggable memory) is a *passive* store — you call it, it answers, and it forgets to tend itself. The thesis of this RFC is that the best memory server is an **active** one: it runs its own hygiene, briefs the agent at boot, marks stale beliefs on every read, and calls the agent back when something needs resolving.

The empirical anchor for "why this matters" is the cross-corpus k-sweep (memory rid `019eb47e`): supersession-aware retrieval is the one structural capability RAG-over-notes cannot buy back at *any* retrieval budget `k` (RAG scores current-value 0.00 at k=8/20/50; the substrate's revision chain scores 0.78–1.00). The server's job is to make that capability — and the hygiene that keeps it true — a property of the *deployment*, not of client discipline.

## Goals

1. **Hygiene is structural, not voluntary.** A loaded engine gets its `run_maintenance_cycle()` driven on a host schedule with zero agent in the loop.
2. **The server owns the agent's wake/sleep ritual.** One-call boot digest; end-of-session capture.
3. **Trust travels on the wire.** Age, supersession, and conflict stamps ride every recall hit.
4. **Memory calls back.** Triggers/conflicts/maintenance outcomes are pushed (SSE), not only polled.
5. **Fleet, not single-user.** Multiple agents share one substrate with attributable writes + leak audit.
6. **Proof.** Served (not engine-direct) numbers, CI-gated, README-cited.
7. **Zero v1 breakage.** Every change is additive to config, HTTP, and MCP surfaces.

## Non-goals

- **No reasoning/agent layer in the server.** Cognition stays in the engine; orchestration stays in the client (Hermes, lane-b, the MCP consumer). Building an agent loop here would compete with our own users.
- **No new retrieval primitives before the close-loop ships.** Fable's diagnosis was *close-poor*, not *query-poor*; v0.8.23 + engine v0.8.0 already cover structural query. Resist scope creep into more `recall` variants.
- **The engine does not get a timer.** We honor the engine's boundary: the server schedules, the engine executes.
- **No silent caps.** If a pass is skipped (backpressure, replication catch-up, cluster non-leader), it is logged and counted, never silently dropped.

## The six pillars and their sequencing

| Release | Pillar | Surface |
|---|---|---|
| **v0.8.24** (#55) | **Time** | `MaintenanceWorker` drives `run_maintenance_cycle()` per tenant on a schedule; `[maintenance]` config; admin run/status endpoints; metrics |
| **v0.8.25** (#56) | **Lifecycle** | `GET /v1/session/digest`, `POST /v1/session/end`, trust metadata on recall hits |
| **v0.8.26** (#57) | **Trust + Push** | `GET /v1/events` (SSE), `GET /v1/current` (chain head), optional webhooks |
| **v0.8.27** (#58) | **Fleet** | `/v1/admin/audit/leak_candidates`, write provenance, link-model surface |
| parallel (#59) | **Proof** | server-tier load + current-value + failover benchmarks, CI gate, README evidence |

Each ships through the full release gate (workspace tests, **cluster-mode validation**, changelog, PR, tag, deploy, smoke, swarm-notify).

---

## v0.8.24 — Autonomous hygiene (this release)

### Engine API consumed

```rust
// re-exported at the crate root: yantrikdb::{MaintenanceCycleConfig, MaintenanceCycleReport}
impl YantrikDB {
    pub fn run_maintenance_cycle(&self, config: &MaintenanceCycleConfig) -> Result<MaintenanceCycleReport>;
    pub fn last_maintenance_cycle(&self) -> Result<Option<String>>; // persisted JSON of last cycle
}
```

`MaintenanceCycleConfig::default()` runs the light, idempotent passes (`run_think`, `burn_down_conflicts`, `prune_triggers` @ max 64, `recalibrate_importance`, `backfill_entities`, `auto_relate` @ max 500 edges) and leaves the heavy corpus-rewriting passes (`split_oversized`, `repair_artifacts`) **off** by default. The report carries per-pass sub-reports and an `errors: Vec<String>` — **a failing pass never aborts the cycle**, and the engine persists the summary to its `meta` table under `last_maintenance_cycle`.

### Design

The existing `background::WorkerRegistry` ([background.rs](../../crates/yantrikdb-server/src/background.rs)) already owns the per-database worker lifecycle: it spawns `tokio` loops with a `CancellationToken`, runs engine work on `spawn_blocking`, records `LockHoldTimer` metrics, and is started per engine-load via `start_for_database(db_id, db_name, engine)`. The maintenance loop joins this registry — no new lifecycle machinery.

```
[maintenance]
enabled = true                          # master switch; default on
interval_secs = 600                     # cadence per tenant (10 min default)
initial_delay_secs = 120                # jitter base; stagger so tenants don't stampede
pause_during_replication_catchup = true # skip a tick if this node is catching up
run_split_oversized = false             # heavy pass — opt-in
run_repair_artifacts = false            # heavy pass — opt-in
max_pending_triggers = 64
max_auto_relate_edges = 500
```

The loop, per tenant:
1. **Wait** `interval_secs` (after an `initial_delay_secs` + per-tenant jitter so N tenants don't fire in lockstep).
2. **Backpressure gate** — reuse the existing enrichment-pressure rule: if `count_pending_ops() > effective_enrichment_threshold(...)`, **skip** this tick (record `maintenance_runs_skipped{reason="backpressure"}`). Hygiene is load-bound, exactly the class the existing rule pauses.
3. **Cluster gate** (see below).
4. Build `MaintenanceCycleConfig` from the `[maintenance]` block; call `engine.run_maintenance_cycle()` on `spawn_blocking`.
5. Record metrics from the report; emit a single `tracing::info!` line when the cycle did real work.

### Cluster safety (the load-bearing decision)

`run_maintenance_cycle()` **mutates state** (resolves conflicts, prunes triggers, rewrites importance, upserts edges). On a replicated cluster, running it independently on every node would **fork the state machine** — the precise class of bug the v0.8.18 cluster regression burned us on.

**Rule: maintenance runs only where writes are accepted.** Concretely the worker checks the cluster state and runs the cycle **iff the node accepts writes** (standalone, or the current leader). On a follower/learner it skips with `maintenance_runs_skipped{reason="not_leader"}`. The mutations then propagate through the normal replication path that every other write uses — no second write channel, no divergence.

- In **standalone** mode (`raft_mode = "disabled"`, e.g. CT 141 today) the node always accepts writes → maintenance always runs. This is the common case.
- In **openraft** mode the leader runs it; on leadership change the new leader picks up the cadence. A cycle is idempotent, so a double-run across a failover converges.

This is the single most important test target for the release: a maintenance cycle on a clustered node must not produce state the followers don't also receive through replication.

### HTTP surface (additive, master-token gated)

```
POST /v1/admin/maintenance/run        # operator-triggered cycle; ?tenant=<db> optional
GET  /v1/admin/maintenance/status     # last_maintenance_cycle() per tenant + worker liveness
```

`run` returns the `MaintenanceCycleReport` JSON. `status` parses each tenant's persisted `last_maintenance_cycle` and reports `ran_at`, the sub-report counts, and any errors. Both reuse the existing master-token gate (per v0.8.22 routing: master tokens resolve to the relevant engine; `?tenant` selects which).

### Metrics (`/metrics`, Prometheus)

- `maintenance_run_duration_ms` (histogram, label `db`)
- `maintenance_conflicts_resolved_total`, `maintenance_triggers_pruned_total`, `maintenance_consolidations_total`, `maintenance_entities_linked_total`, `maintenance_relations_upserted_total` (counters)
- `maintenance_runs_skipped_total{reason}`, `maintenance_runs_failed_total`, `maintenance_pass_errors_total`
- Health gauges — the write-rich/close-poor dashboard: `memory_open_conflicts`, `memory_pending_triggers` per tenant.

### Backward compatibility

- The `[maintenance]` block defaults `enabled = true`, so **existing deployments gain hygiene on upgrade with no config change** — but every pass it drives is idempotent and the heavy passes are opt-in, so the behavior change is "the conflicts that were already going to be resolved get resolved on a timer."
- The engine pin v0.7.24 → v0.8.0 is additive (schema via `CREATE TABLE IF NOT EXISTS`; new public API only); 1591 engine tests green, slim build green.
- No wire-protocol change. No change to `/v1/recall`, `/v1/remember`, `/v1/memories`.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Maintenance forks cluster state | Run only where writes are accepted (leader/standalone); propagate via normal replication. Cluster-mode integration test gates the release. |
| Hygiene competes with ingest for the 2-core LXC | Backpressure gate skips ticks under load; `spawn_blocking` keeps it off the async reactor; per-tenant jitter avoids stampede. |
| A bad pass wedges the cycle | Engine isolates per-pass failures into `report.errors`; the worker counts them and continues. |
| Operators surprised by auto-mutation | `status` endpoint + metrics make every cycle visible; heavy passes opt-in; `enabled=false` is one line. |

## Acceptance

1. `cargo fmt` + `cargo check --workspace --tests` + `cargo test --workspace` green.
2. Integration test: worker runs against a fixture engine, is idempotent on repeat, and does not interfere with concurrent writes.
3. **Cluster-mode test**: maintenance on a replicated node does not fork state (runs leader-only; followers receive mutations via replication).
4. Deployed to CT 133 + trader CT 168; `status` endpoint shows real cycle outcomes on the live corpora; algo notified via swarm.
