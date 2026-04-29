# RFC 009 — Recall Admission Control & API Versioning

Status: **Draft** (2026-04-28)
Triggered by: 2026-04-28 cluster leader-election thrashing (term=1423 on .140), 5+ pollers driving `top_k=200, expand_entities=true` saturated voter CPU and starved Raft heartbeats.
Constraint: **backwards compatible** (v1 contract preserved), **enterprise grade** (no band-aids, full quota / circuit-breaker / runtime isolation / observability / migration).

## Goals

1. Eliminate the priority inversion that lets application-level recall work starve Raft consensus on voters.
2. Make `expand_entities=true` (the actual hot cost) opt-in for *new* clients without breaking existing ones.
3. Give operators per-tenant SLA tools (quota, rate, concurrency, request budget) so noisy neighbors can't take the cluster down.
4. Add the observability that lets us prove the fix worked and detect the next time someone gets close to this failure mode.
5. Self-heal: cluster recovers without operator intervention when load spikes pass.

## Non-goals

- Cursor cache (rejected in brainstorm session a64e5a99 redteam).
- Substrate `session`-backed cursors (replicating ephemeral read state would compound the very problem).
- Server-side `/recall/scan` endpoint.
- Flipping `expand_entities` default on `/v1/recall` (breaks existing clients — REJECTED).

## Backwards-compat contract (read this first)

The v1 contract is **frozen**. Specifically, on every existing `/v1/*` endpoint:

- Default values for fields stay the same (`top_k=10`, `expand_entities=true`, `expand_entities` defaulting to true on omitted requests).
- Response shape stays the same.
- Authentication semantics stay the same.

What v1 *gains* (not breaks):

- Documented rate limits and admission control. Existing clients that respect documented limits see no change. Clients that consistently exceed them get HTTP 429 with `Retry-After` and `X-YDB-RateLimit-*` headers — industry-standard, not a contract break.
- HTTP 503 + `Retry-After` returned during transient cluster instability windows (term churn, heartbeat lag p99 high). Industry-standard back-pressure.
- New response metadata fields (additive, not replacing). e.g., `effective_top_k` if a request was clamped.

Migration to v2 is **opt-in**, communicated via release notes, with a 12-month minimum support window for v1.

---

## Design

### 1. API versioning (`/v2/*`)

New endpoint family at `/v2/recall`, `/v2/remember`, etc. The wire format is the same struct (`RecallRequest`) but with **new defaults** in v2:

| Field | v1 default | v2 default |
|---|---|---|
| `top_k` | 10 | 10 |
| `expand_entities` | `true` | **`false`** |
| `include_consolidated` | `false` | `false` |
| Hard cap `top_k` (no expand) | none → enforced 500 | 500 |
| Hard cap `top_k` (expand=true) | none → enforced 50 | 50 |

Implementation: same `recall` handler, dispatched by route prefix. A `version: ApiVersion` field on the resolved request controls default fill-in.

**Files:**
- `crates/yantrikdb-server/src/http_gateway.rs` — add `/v2/*` routes alongside `/v1/*`, share handler functions
- `crates/yantrikdb-protocol/src/messages.rs` — add `ApiVersion` enum, version-aware default fill-in helpers
- `crates/yantrikdb-server/src/command.rs` — `Recall` command gains `api_version: ApiVersion` for cap enforcement

### 2. Admission control (per-principal + per-namespace token buckets)

Every authenticated request flows through admission middleware **before** dispatch. The middleware enforces:

- **Per-principal RPS** (default 100 req/s, configurable per tenant)
- **Per-namespace concurrent-expanded-recall semaphore** (default 4, configurable)
- **Per-principal request-budget** for cost-weighted throttling (cost function below; default budget 1000 cost-units/s)

**Cost function (simplified per redteam):**

```
cost = top_k × E_expand
E_expand = 5 if expand_entities else 1
```

Both `E_expand` and the default budget are configurable. **`namespace_size` is intentionally NOT in the default cost function** — coupling cost to data shape creates silent drift (same query gets more expensive as data grows) and invites operator confusion. If empirical benchmarks later show namespace density correlates strongly with actual resource cost, it can be added as an opt-in `cost_function = "advanced"` config — but defaults stay shape-stable.

**Limitations (document explicitly so operators aren't surprised):**

- **Token-bucket state is node-local, not globally serialized.** A hot tenant spreading load across nodes (or via LB churn) can effectively get N× their per-node quota. True cluster-global fairness would require replicated bucket state through Raft, which compounds the very write-load problem this RFC solves. Node-local is the deliberate tradeoff. Document in admin guide.
- **Restart resets bucket state.** A rolling restart can synchronize across the fleet and create a refilled-burst surge. Mitigation: `startup_warm_fraction` config (default 0.25) — buckets start at 25% capacity on boot and refill to full over 60s. Prevents post-restart thundering herd.
- **Policy convergence is eventual.** quota_policies replicates via the cluster's normal SQLite replication path; nodes may transiently disagree on policy. Each policy row carries `policy_version` (monotonic) and `updated_at`; admin API returns the version the writer sees, observability surfaces per-node observed version so divergence is visible.
- **v1 endpoints ARE subject to admission control.** Existing clients that exceed documented rate limits will see 429s. This is QoS evolution (industry-standard rate limits), not a contract break — the v1 contract never promised unlimited throughput. Documented loudly in v0.8 release notes.

Quota config lives in the control DB (`control.rs`), with a new table `quota_policies`:

```sql
CREATE TABLE quota_policies (
    scope_type TEXT NOT NULL,    -- 'principal' | 'namespace' | 'global'
    scope_value TEXT NOT NULL,    -- the principal id, namespace name, or '*'
    rps_limit INTEGER,
    cost_budget_per_sec INTEGER,
    max_concurrent_expanded INTEGER,
    tier TEXT,                    -- 'gold' | 'silver' | 'bronze' (informational)
    PRIMARY KEY (scope_type, scope_value)
);
```

Token-bucket state is in-memory per node (won't survive restarts — that's fine, tokens replenish naturally). Use `governor` crate for the bucket impl (already idiomatic in axum middlewares).

When a request exceeds quota, return:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 1
X-YDB-RateLimit-Limit: 100
X-YDB-RateLimit-Remaining: 0
X-YDB-RateLimit-Reset: 1714327500
Content-Type: application/json

{"error": "rate limit exceeded", "scope": "principal", "limit_per_sec": 100}
```

Admin API for setting quotas (cluster-master-token only):

- `POST /v1/admin/quotas` — upsert a quota policy
- `GET /v1/admin/quotas` — list current policies
- `DELETE /v1/admin/quotas/{scope_type}/{scope_value}` — remove

Tenant-visible self-service:

- `GET /v1/usage` — returns the calling principal's current quota / consumption / remaining budget

**Files:**
- `crates/yantrikdb-server/src/admission.rs` — NEW. Middleware + token-bucket state + cost function.
- `crates/yantrikdb-server/src/control.rs` — quota_policies table, CRUD methods.
- `crates/yantrikdb-server/src/http_gateway.rs` — wire middleware into router, add `/v1/admin/quotas` and `/v1/usage` routes.

### 3. Raft-instability circuit breaker

Independent of quota: when the cluster is in a self-protective window, reject **expensive expanded recalls** with HTTP 503 + `Retry-After` so the cluster has room to re-stabilize. Cheap recalls (no-expand, top_k≤50) keep flowing.

**Trigger logic (revised twice — gpt-5.5 caught that term churn is reactive):**

- **Pre-failure signal: Raft task scheduling latency.** Open when `raft_task_poll_latency_seconds{quantile=0.99} > 0.050` (50ms) for 2 consecutive 10-second windows. This catches CPU starvation BEFORE election timeouts fire. Term churn is a post-failure signal — by the time it's elevated, the cluster has already destabilized. Scheduling latency leads it by tens of seconds.
- **Emergency signal: term churn.** `raft_term_changes_total` increased in last 60s → open immediately. Unambiguous failure indicator. No hysteresis on opening (close-side hysteresis only).
- **Sustained-lag signal: heartbeat lag, deployment-profile-aware, hysteresis-windowed.** Open when `heartbeat_lag_p99 > max(absolute_floor, baseline × multiplier)` is true for **3 consecutive 10-second windows** (i.e., 30s of sustained elevated lag). Single p99 spikes do NOT trip the breaker.
- **Active election in progress** → open immediately.

**Deployment profiles** (operator selects via config `cluster.deployment_profile`):

| Profile | absolute_floor | baseline_multiplier |
|---|---|---|
| `lan_default` | 50ms | 3× |
| `wan_default` | 200ms | 3× |
| `tuned` | operator-set | operator-set |

Baseline is computed as a 5-minute rolling p99 (only when breaker is closed — otherwise we'd train on degraded state). Establishes baseline before the breaker can fire (15 min warmup post-startup).

**Hysteresis:**
- Term-churn-triggered open: stay open ≥ 60s after last term change before evaluating close
- Lag-triggered open: stay open ≥ 30s after lag returns to normal
- Half-open state: after the close threshold, allow 10% of expensive requests through; if any of those triggers re-open, return to fully open

**Anti-flapping**: minimum cycle time of 60s between open→close→open transitions. If a breaker flaps faster than that, log a warning (config drift / unhealthy cluster signal) and stay open.

Pollers respect 503 + Retry-After natively; they back off automatically. **No client change required** — server-side self-protection only.

**Files:**
- `crates/yantrikdb-server/src/admission.rs` — circuit breaker state machine, hooked to metrics.
- `crates/yantrikdb-server/src/cluster.rs` — expose Raft state signals (term changes, heartbeat lag) as observable.

### 4. Dedicated tokio runtime for Raft control plane

The structural fix to the priority inversion. Today, application work and Raft heartbeats share the same tokio runtime, so a CPU-saturating recall handler can starve heartbeat tasks. Split into two runtimes — **but runtime split alone is not real CPU isolation** (per gpt-5.5 redteam: threads can exist with no CPU under saturation). Three reinforcing layers required:

**Layer 1 — Runtime split:**
- **Control runtime**: 2 cores reserved (configurable). Hosts Raft tasks, heartbeat sender, AppendEntries handler, replication. **No application work runs here.**
- **Application runtime**: remaining cores. Hosts HTTP gateway, recall handlers, expansion workers.

**Layer 2 — OS-level thread priority (Linux only; best-effort on others):**
Control-runtime worker threads use `SCHED_FIFO` priority via `pthread_setschedparam`. Application threads stay on `SCHED_OTHER`. The kernel scheduler will preempt application work to run Raft tasks. Falls back to `SCHED_OTHER` with `nice -10` on systems where `CAP_SYS_NICE` isn't available, with a startup warning.

**Layer 3 — Hard concurrency caps that prevent CPU saturation in the first place:**
- Max concurrent expanded recalls per node: 4 (configurable, default sized to leave 1 core idle headroom)
- Max in-flight recall requests total: 64 (bounded `Semaphore`)
- Max request body size: 64 KiB (axum middleware)
- Hard `top_k` clamp at request parse: 1000 (anything above returns 400 immediately, before HNSW search runs)

**Acceptance gate (mandatory):**
- New metric: `raft_task_poll_latency_seconds` (histogram, p99). Instrumented via `tokio::task::Builder::name("raft_*")` + custom probe.
- Test `tests/cpu_isolation.rs`: drive the application runtime to 100% CPU on a 4-core system with 32 concurrent expanded recalls, assert `raft_task_poll_latency_seconds{quantile=0.99} < 0.010` (10ms). PR-1 does not merge until this test passes.

**Files:**
- `crates/yantrikdb-server/src/main.rs` — split runtime construction.
- `crates/yantrikdb-server/src/runtime.rs` — NEW. Runtime builder, SCHED_FIFO setup, scheduling-latency probe.
- `crates/yantrikdb-server/src/cluster.rs` — Raft tasks `spawn_on(control_runtime.handle())`.
- `crates/yantrikdb-server/src/server.rs` — `AppState` carries `control_runtime: Handle` and concurrency semaphores.
- `crates/yantrikdb-server/src/http_gateway.rs` — wire body-size limit + top_k clamp + Semaphore on recall handlers.
- `crates/yantrikdb-server/tests/cpu_isolation.rs` — NEW. Acceptance gate test.

Config schema:
```toml
[runtime]
control_threads = 2          # Raft / replication / control-plane
app_threads = 0              # 0 = remaining cores
control_priority = "fifo"    # "fifo" | "nice" | "default"

[admission]
max_concurrent_expanded_recall = 4
max_in_flight_recall = 64
max_request_body_bytes = 65536
hard_top_k_cap = 1000
```

### 5. Observability

Metrics added (Prometheus, exposed at existing `/metrics`):

- `raft_term_changes_total` (counter) — increments on every term bump
- `raft_heartbeat_lag_seconds` (histogram) — p50/p99 of heartbeat round-trip
- `raft_election_total` (counter, label `result=won|lost|stepped_down`)
- `recall_requests_total{api_version, expand}` (counter)
- `recall_request_top_k{api_version}` (histogram)
- `recall_rejected_total{reason}` (counter; reasons: `cap`, `circuit_breaker`, `rate_limit`, `concurrency`, `cost_budget`)
- `expansion_concurrent_in_flight` (gauge)
- `quota_consumption{scope_type, scope_value, dimension}` (gauge; dimensions: `rps`, `cost`, `concurrent`)
- `circuit_breaker_state{component}` (gauge; 0=closed, 1=open, 2=half-open)

Tracing: every request gets an OpenTelemetry span with attributes `principal`, `namespace`, `api_version`, `top_k`, `expand_entities`, `cost_estimate`, `admission_outcome`.

Dashboards (Grafana JSON in `docs/dashboards/`):
- Cluster health (term, heartbeat lag, election rate)
- Recall throughput by version + cost
- Per-tenant quota consumption
- Circuit breaker state timeline

**Files:**
- `crates/yantrikdb-server/src/metrics.rs` — register new metrics.
- `crates/yantrikdb-server/src/admission.rs` — emit metrics on each admission decision.
- `docs/dashboards/cluster_health.json` — NEW.
- `docs/dashboards/recall_quota.json` — NEW.

### 6. Tests

- **Unit**: token-bucket math, circuit-breaker state machine, cost-function correctness.
- **Integration**: spin up a 3-node cluster (test harness), drive heavy recall, assert (a) admission control rejects appropriately, (b) circuit breaker opens on simulated term churn, (c) v1 endpoints return identical results to pre-change baseline.
- **Backwards-compat**: pin every public SDK release (yantrikdb-client 0.1.x..0.3.x), run their test suites against new server build, assert pass.
- **Load reproduction**: synthetic test that recreates the term=1423 thrashing pattern (5 pollers, top_k=200, expand_entities=true). Assert (a) cluster does NOT thrash, (b) admission control rejects excess requests, (c) heartbeat lag p99 stays < 50ms.
- **Chaos**: kill leader during heavy recall, assert (a) failover completes within 5s, (b) clients see retry-able errors (503/429), (c) cluster recovers without manual intervention.

**Files:**
- `crates/yantrikdb-server/tests/admission_control.rs` — NEW.
- `crates/yantrikdb-server/tests/circuit_breaker.rs` — NEW.
- `crates/yantrikdb-server/tests/backwards_compat_v1.rs` — NEW.
- `crates/yantrikdb-server/tests/load_repro_term_thrash.rs` — NEW.
- `crates/yantrikdb-server/tests/chaos_failover.rs` — NEW.

### 7. Migration guide

Document in `docs/migration/v0.7_to_v0.8.md`:

1. Drop-in compatible: existing clients keep working, no client changes required.
2. Recommended client config for new deployments: explicitly set `expand_entities=false` unless you read entity boost scores.
3. Optional: migrate to `/v2/*` endpoints for safer defaults.
4. Operators: review `/v1/admin/quotas`, set per-tenant policies for expected workload.
5. Observability: add the new dashboards, alert on `circuit_breaker_state == 1`.
6. Deprecation timeline: v1 supported until 2027-04-28 minimum (12 months), longer if usage justifies.

---

## Rollout

### Pre-flight (before merging this RFC's PR)
1. Operator stops pollers on the live cluster, term stabilizes (Track A from incident response — code-independent).
2. Audit pollers, identify which currently-misconfigured ones can be reconfigured client-side. Reconfigured pollers restart against the existing server.
3. Cluster confirmed stable for 1 hour.

### PR sequence (revised per gpt-5.5 redteam — every PR ships its own tests + dashboards; breaker shadow-mode earlier; staged enforcement)

1. **PR-1 — Foundation**: runtime split (3-layer CPU isolation: split + SCHED_FIFO + concurrency caps) + new metrics (term, heartbeat lag, task poll latency, recall by version, expansion in-flight) + hard pre-admission caps (max body, top_k clamp, in-flight semaphore) + **`tests/cpu_isolation.rs` acceptance gate** (must pass to merge) + `docs/dashboards/cluster_health.json`. **No behavior change for clients other than 400 on top_k > 1000 and 413 on body > 64KiB.**
2. **PR-2 — Admission infrastructure**: middleware + control-DB `quota_policies` table + `/v1/admin/quotas` admin API + token-bucket internal state + lazy-on-first-request tenant backfill (auth-gated) + tests (unit: bucket math, cost function; integration: middleware wired, no rejection) + `docs/dashboards/recall_quota.json`. **No enforcement** — middleware computes cost, emits `quota_consumption` metrics, never rejects.
3. **PR-3 — Circuit breaker SHADOW mode**: full circuit breaker state machine (3 trigger signals: pre-failure scheduling latency, emergency term churn, sustained heartbeat lag) **evaluating but not rejecting** — emits `circuit_breaker_state` and `circuit_breaker_would_reject_total` metrics. Validates signals against real traffic before enforcement. Tests + dashboards.
4. **PR-4 — Staged enforcement**: graduate quota enforcement and circuit breaker from shadow → enforce. Per-tenant feature flag in control DB: `enforcement_mode = observe | warn | soft | hard`. New tenants start at `observe`; operator promotes per-tenant after metrics confirm safe. Default global goes to `warn` first (adds `X-YDB-Quota-Warning` headers but doesn't reject), then `soft` (rejects only on egregious violations e.g. 10× quota), then `hard` (full enforcement). Tests + dashboards.
5. **PR-5 — In-flight cancellation + retry storm controls**: `CancellationToken` propagation through every recall handler + per-stage deadlines (`recall_total_deadline_ms=5000`, `expansion_deadline_ms=2000`) + breaker-open triggers cancellation of in-flight expanded recalls older than 2s + bounded `JoinSet` for expansion workers + retry semantics (`Retry-After` jitter guidance, per-tenant retry damping via `X-YDB-Retry-Budget` header). Tests + dashboards.
6. **PR-6 — `/v2/*` endpoints**: new path with new defaults (`expand_entities=false`) + tests including backwards-compat suite asserting v1 unchanged + dashboards updated for per-version metrics.
7. **PR-7 — Integration tests + migration**: load reproduction of term=1423 thrashing pattern (must NOT thrash with all PRs deployed) + chaos failover test + `docs/migration/v0.7_to_v0.8.md` + release notes + benchmark validation matrix (the gate that promotes provisional defaults to actual defaults).

Each PR independently deployable, independently verifiable, no big bang. **Each PR includes its own tests + dashboards** (per gpt-5.5 — tests/dashboards in a final PR is flying blind on intermediate states).

### Release tag
v0.8.0 once PR-1 through PR-5 ship. v2 endpoints (PR-6) ship in v0.9.0. Integration test + migration (PR-7) gates v0.8.0 GA.

### Version skew handling (rolling upgrades)
Per gpt-5.5 redteam — PRs are not actually independent under rolling deploy:
- Old nodes ignore quota policies → policy table has cluster-min-version gate; old nodes that see new policy rows log a warning but don't enforce.
- New nodes enforce, old don't → operators must complete rolling deploy before any policy enforcement is visible. Document in upgrade runbook.
- `/v2` exists only on upgraded nodes → load balancer health-check should `404` on `/v2/*` for old nodes (axum returns 404 by default).
- Circuit breaker state is per-node and doesn't propagate.
- Admin API enforces a `Cluster-Min-Version` header on writes that touch new schema features.

---

## Failure modes (added per redteam)

Every admission/breaker component has explicit fail-mode behavior. Default to **fail-degraded-conservative** rather than fail-open or fail-closed.

| Component | Failure | Behavior |
|---|---|---|
| Quota policy lookup | control DB unavailable | Use last-known-good cached policy (per-node TTL 10min). If no cache (cold start), fall back to **conservative provisional defaults** (RPS=50, cost_budget=500/s, expanded_concurrent=2 — half the normal default). Emit `quota_lookup_fallback_total` metric. NEVER fail-open to unlimited. |
| Circuit breaker signal | Raft state unobservable (e.g., metrics scrape stalled) | Treat as degraded — emit `circuit_breaker_signal_unavailable` gauge=1, apply conservative reject policy (open for expensive expanded recall, allow cheap recall). Surfaces in dashboards. NEVER assume healthy by default. |
| Admin API | control DB unavailable | Reject quota CRUD with 503 + Retry-After. Data plane reads continue using cached policy. Quota changes wait for control DB recovery. |
| Admission middleware | bucket-state internal panic | Sentry-log + bypass admission for that single request, increment `admission_internal_error_total`. Single-request fail-open is preferable to total request rejection on internal bug; bug surfaces in alerts. |
| Token bucket | clock skew / NTP issue | Bucket state uses `monotonic` clock (not wall clock); insulated from skew. Documented. |

**Quota CRUD audit:** every admin API write (`POST/DELETE /v1/admin/quotas`) appends to control DB `quota_audit_log` with `(timestamp, principal, action, scope_type, scope_value, before_policy, after_policy)`. Read access via `GET /v1/admin/quotas/audit` (admin-only). Retention: 90 days minimum.

**Tenant backfill (auto-created policies):** lazy-on-first-request. The first request from a never-before-seen principal/namespace triggers creation of a policy row at provisional defaults (NOT auto-populated at rollout). This avoids:
- Snapshot-at-rollout: would create thousands of policies for inactive tenants and pollute the table.
- Background reconciler: more code, more failure modes, no operator benefit.

Lazy creation is logged for observability. Operators can pre-populate specific tenants via admin API if they want explicit policies.

---

## Decisions resolved (post-redteam, openai:gpt-5.4 brainstorm 2026-04-28)

| # | Decision | Resolution |
|---|---|---|
| 1 | Quota persistence | **Accept** — control DB + admin API. Caveats added: node-local enforcement documented, policy_version field, audit log, startup warm fraction, fallback on control DB outage. |
| 2 | Cost function | **Override** — use `cost = top_k × E_expand` (E_expand=5 default, configurable). `namespace_size` removed from default. |
| 3 | Circuit breaker thresholds | **Override** — term churn is primary; heartbeat lag uses deployment profiles (`lan_default=50ms`, `wan_default=200ms`) with 3-window hysteresis. No universal absolute threshold. |
| 4 | API versioning style | **Accept** — path versioning. Locked rule: ALL future externally-versioned APIs use `/vN/...` style. |
| 5 | Default global quotas | **Compromise** — provisional defaults (RPS=100, cost_budget=1000/s, expanded_concurrent=4), DO NOT auto-populate pre-existing tenants. PR-3 acceptance gate: pass benchmark validation against synthetic load matrix BEFORE labeling defaults non-provisional. |
| 6 | Fail-mode behavior (added) | **Defined above** — fail-degraded-conservative everywhere; never silent fail-open. |

Implementation begins with PR-1 (runtime split + metrics). Track A (operator stops pollers, audits, restarts with reconfig) is independent and unblocks the cluster regardless of when these PRs land.
