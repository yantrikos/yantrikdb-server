# Changelog

All notable changes to `yantrikdb-server` are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.24] — 2026-06-11

**The active memory server begins — autonomous hygiene (RFC 027, pillar 1: time).** The server now drives the engine's maintenance cycle on a per-tenant schedule, so closing loops (conflict burn-down, trigger prune, importance recalibration, entity backfill, auto-relate) is **structural, not voluntary**. The engine deliberately owns no timer ("a storage engine scheduling itself is the wrong boundary"); the server is the host that decides cadence. Addresses the write-rich/close-poor diagnosis: hygiene becomes a property of the deployment, not of client discipline.

### Changed — engine pin v0.7.24 → v0.8.0

Engine v0.8.0 ("World's Best Memory System") is additive and backward-compatible (schema via `CREATE TABLE IF NOT EXISTS`; new public API only). v0.8.24 consumes its `run_maintenance_cycle(&MaintenanceCycleConfig) → MaintenanceCycleReport` and `last_maintenance_cycle()` surface. No wire-protocol change; `/v1/recall`, `/v1/remember`, `/v1/memories` unchanged.

### Added — `[maintenance]` config block

```toml
[maintenance]
enabled = true              # master switch (default on)
interval_secs = 600         # per-tenant cadence (10 min)
initial_delay_secs = 120    # base delay before first cycle (+ per-tenant jitter)
pause_during_replication_catchup = true  # skip when this node doesn't accept writes
run_split_oversized = false # heavy pass — opt-in
run_repair_artifacts = false# heavy pass — opt-in
max_pending_triggers = 64
max_auto_relate_edges = 500
split_min_chars = 1500
```

Defaults are **on + light**, so existing deployments gain hygiene on upgrade with no config change; the heavy corpus-rewriting passes stay opt-in.

### Added — `MaintenanceWorker` (per-tenant sleep cycle)

Joins the existing `WorkerRegistry`. Each tenant's loop, after a jittered initial delay:
- **Cluster-safety gate** — runs `run_maintenance_cycle()` only where writes are accepted (standalone or the current leader). On a follower/learner it skips, because the cycle mutates state and would otherwise fork the state machine (the v0.8.18 class of bug). Mutations propagate through the normal replication path.
- **Backpressure gate** — reuses the enrichment-pressure rule; skips a tick when the engine is behind on its delta tier.
- Runs on `spawn_blocking`; idempotent (a missed or double-run tick converges).

### Added — admin endpoints (master-token gated)

| Endpoint | Behavior |
|---|---|
| `POST /v1/admin/maintenance/run` | Operator-triggered cycle. `?tenant=<name>` for one, else all. Optional body `{split_oversized, repair_artifacts}` enables heavy passes. **Refuses with 409 on a node that doesn't accept writes** (hit the leader). |
| `GET /v1/admin/maintenance/status` | Per-tenant last-cycle summary (from `last_maintenance_cycle()`), worker liveness, and this node's write-acceptance state. |

### Added — maintenance metrics (`/metrics`)

Per-tenant counters: `yantrikdb_maintenance_runs_total`, `_conflicts_resolved_total`, `_triggers_pruned_total`, `_consolidations_total`, `_entities_linked_total`, `_relations_upserted_total`, `_failures_total`, `_pass_errors_total`; `_runs_skipped_total{reason}` (`not_write_accepting` / `backpressure`); and the `_duration_ms` summary. The write-rich/close-poor dashboard.

## [0.8.23] — 2026-06-10

Structural query primitive on `/v1/memories` — `?kind=`, `?drive_id=`, `?since_rid=` (keyset cursor), `?order=asc|desc` — wired to engine `list_records` (yantrikdb-core v0.7.24). Replaces the fetch-all-then-filter pattern with indexed push-down. Closes algo's perf gap (swarm `c1d810df`, 2026-06-10): "list records of kind X" now uses one SQL plan with indexed VIRTUAL columns instead of a full scan.

### Changed — engine pin v0.7.22 → v0.7.24

v0.7.24 ships:
- Schema v32: VIRTUAL generated columns `kind = json_extract(metadata,'$.kind')` and `drive_id = json_extract(metadata,'$.drive_id')` with indexes; `json_valid` guard ensures encrypted/malformed metadata resolves to NULL (no migration errors).
- New `list_records(namespace, kind, drive_id, memory_type, domain, since_rid?, limit, order)` engine method — all filters AND-composed and pushed down to one SQL plan.

Auto-migrates on first open. No backfill. No write-API change.

### Added — `/v1/memories` structural query params

| Param | Behavior |
|---|---|
| `?kind=X` | row-level metadata.kind filter (indexed) |
| `?drive_id=X` | row-level metadata.drive_id filter (indexed) |
| `?since_rid=R` | keyset cursor (UUIDv7 = lexically chronological) |
| `?order=asc\|desc` | asc (default, oldest first) or desc (newest first) |

When **any** of the new structural params (`kind`, `drive_id`, `since_rid`, or `order=desc`) is set, the handler routes through `engine.list_records` with response envelope:

```json
{
  "records": [ {rid, text, memory_type, importance, metadata: {...}, namespace, ...} ],
  "next_cursor": "<last_rid>" | null,
  "limit": N
}
```

Without any structural params (legacy/dashboard usage), the existing `engine.list_memories` path is preserved with the `{items, total, limit, offset}` envelope.

### Fixed — record item shape mirrors `/v1/recall`

Records from the new `list_records` path include `metadata` as a **parsed JSON object** (not stringified). Per `yantrikdb-agi`'s shape note (swarm `c1d810df`, 2026-06-10): `/v1/recall` returns parsed `metadata: {...}` but legacy `/v1/memories` returned `metadata_json` as a stringified blob with the parsed `metadata` field null. The new `memory_to_record_item` helper normalizes both stored shapes (Value::String containing JSON, or Value::Object) to a parsed object. Algo's `query_typed()` client can now share its `RecallHit` deserializer with no special-casing.

The legacy `/v1/memories` (no structural params) still emits `metadata_json` for back-compat.

### Tests (4 new, 26 memories_list_tests total)

- `validate::validate_v0823_default_order_is_asc`
- `validate::validate_v0823_accepts_desc_order`
- `validate::validate_v0823_rejects_unknown_order`
- `validate::validate_v0823_kind_drive_id_since_rid_round_trip`

**2,096 workspace tests pass.** No regressions.

### Compatibility

- Legacy clients (no structural params): **unchanged**.
- `yantrikdb-agi`'s `query_typed()` client: one-commit swap to the new shape.
- Dashboard: untouched.

### Deferred to v0.8.24+

- Link model MCP surface (`record_with_links_partial`, `expand_links` on recall, `link/unlink/linked_records`, `reify_supersedes_links` admin tool)
- `/v1/admin/audit/leak_candidates` HTTP endpoint
- Proposal 5 validation-gate repair UX

## [0.8.22] — 2026-06-10

Master-token routing fix for `/v1/memory/{rid}` and `/v1/memories`. v0.8.21's row-tag canonicalization worked for per-tenant tokens but mis-routed master/cluster-wide tokens — `?namespace=tag` was used as a database selector, causing `404 namespace_not_found` on every tag-shaped namespace value. Algo (yantrikdb-agi) reported this as their #1 blocker on CT 133 (swarm `77ffa517`, 2026-06-10).

### Fixed — master tokens route to `default` DB, `?namespace` always tag filter

Two handlers updated to align with `/v1/recall`'s auth path (`resolve_engine`, which hardcodes `get_database("default")` for cluster master tokens):

1. **`validate_memories_params`** — master token (`principal.tenant_id == None`) now routes `db_namespace` to `"default"`. `?namespace` is exclusively a tag filter, never a database selector. Prior `(None, None)` branch erroring with "namespace is required for cluster-wide tokens" is removed.

2. **`memory_get`** — bypassed `access::resolve_namespace` entirely. Computes `db_namespace` directly from `principal.tenant_id.unwrap_or("default")`. The `?namespace` query param is now plumbed through but has no effect on routing or filtering (rid uniquely identifies the row within the resolved database).

### Updated tests

- `validate::validate_master_token_no_namespace_routes_to_default_db` — new
- `validate::validate_master_token_with_namespace_uses_default_db_and_tag_filter` — new (pins the exact CT 133 scenario)
- `memory_get_e2e::pinned_token_with_ns_param_on_nonexistent_rid_returns_404_not_403` — replaces `returns_403_when_pinned_token_asks_for_other_namespace` (old assertion no longer correct)
- `memory_get_e2e::ns_param_is_ignored_on_point_read` — replaces `cross_namespace_request_via_namespace_param_is_403` (point-read no longer uses `?namespace` for anything)

**99 http_gateway tests pass.**

### Compatibility

- Per-tenant tokens (the bulk of production usage): **unchanged behavior** vs v0.8.21. They keep using their tenant DB and any `?namespace` value still acts as a tag filter.
- Master tokens: now match `/v1/recall`'s contract. `?namespace` is filter-only. Behavior diff vs v0.8.21:
  - Master + omit `?namespace`: was 400 "namespace is required", now 200 listing all rows in `default` DB.
  - Master + `?namespace=fable3`: was 404 `namespace_not_found`, now 200 with rows tagged `fable3` in `default` DB.

### Engine pin unchanged

Stays on v0.7.22 (introduced in v0.8.21). v0.7.24 with `list_records(namespace, kind, since_rid?, limit, order)` is being prepared by yantrikdb-core (swarm `90b7072c`, 2026-06-10); it will land in v0.8.23 wiring `?kind=` + keyset cursor on `/v1/memories`.

### Deferred to v0.8.23+

- Wire `/v1/memories` to engine's `list_records` once published: `?kind=`, `?since_rid=` (keyset cursor), `?order=asc|desc`, response envelope `{records:[...], next_cursor}`. Contract locked with core (swarm `6405b7e5`, 2026-06-10).
- Link model MCP surface (`record_with_links_partial`, `expand_links` on recall, `link/unlink/linked_records`, `reify_supersedes_links` admin tool).
- `/v1/admin/audit/leak_candidates` HTTP endpoint.
- Proposal 5 validation-gate repair UX.

## [0.8.21] — 2026-06-09

Row-tag model canonicalization on the read path. `/v1/memory/{rid}` and `/v1/memories` now treat `namespace` as an optional row-level tag rather than a tenant scope. The database is the isolation boundary; `namespace` is just an organizational filter on top of it.

### Why

Pre-merge validation against trader CT 168 surfaced a real data-model question. The dashboard read endpoints filtered with `memory.namespace == effective_namespace`, where `effective_namespace = principal.tenant_id`. That assumes `namespace` is a tenant scope. The production data says otherwise:

```
trader/default tenant namespace distribution (real data):
  skill_substrate         187,932    ← lane-b autonomous workflow
  comm_substrate            2,642
  growth_lab_b_algo           658
  growth_lab_b                440
  growth_lab_b_grok           101
  phaseb_C3_seed100_*          12+   (many phaseb_C* variants)
```

200k+ rows store `namespace` as a row-level tag — `skill_substrate` is the canonical MCP skills namespace, `comm_substrate` is for swarm coordination state, `growth_lab_b_*` are lane-b workspace partitions. None of these match the tenant database name. The strict-equality filter was hiding the bulk of the production data from dashboard reads.

Per yantrikdb-core decision (swarm message `8a97464e`, 2026-06-09): **the database is the tenant boundary; `namespace` is an optional row-level tag.** The original spec was wrong; the validation caught it before merge.

### Fixed — `/v1/memory/{rid}` no longer applies a namespace guard

The cross-namespace 404 check on `memory.namespace != effective_namespace` is removed entirely. The DB lookup (`get_database(effective_namespace)`) is the real isolation boundary — any row located in the caller's database by rid belongs to the caller, regardless of its namespace tag. Lane-b's earlier empty-string stopgap is superseded by this cleaner form.

### Fixed — `/v1/memories` treats `?namespace` as optional tag filter

Previously the endpoint always filtered `WHERE namespace = effective_namespace`, dropping the 200k+ tagged rows. Now:

- **No `?namespace` provided** → list all rows in the caller's database (no tag filter).
- **`?namespace=skill_substrate` provided** → filter rows tagged `skill_substrate` within the caller's database.

The `db_namespace` (used to route to the tenant DB) and `tag_filter` (used as the optional row filter) are now resolved separately:

| Token type | `?namespace` provided? | `db_namespace` | `tag_filter` |
|---|---|---|---|
| Per-tenant `acme` | omitted | `acme` | None |
| Per-tenant `acme` | `?namespace=skill_substrate` | `acme` | `Some("skill_substrate")` |
| Per-tenant `acme` | `?namespace=acme` | `acme` | `Some("acme")` |
| Cluster-wide | required | (= `?namespace`) | (= `?namespace`) |

Per-tenant tokens can now pass any value of `?namespace` as a tag filter without triggering the prior `namespace_not_found` 403. Cross-tenant access remains impossible because routing still happens via the token's `tenant_id`.

### Changed — engine pin v0.7.20 → v0.7.23

`normalize_namespace()` coerces blank/whitespace `namespace` writes to `"default"` at the engine layer. Fully compatible with the row-tag model — a blank tag just becomes the default tag. Also picks up the backpressure session-count fix and the opt-in attribute-value conflict bridge from v0.7.21 / v0.7.22 / v0.7.23.

### Tests

```
memory_get_e2e::returns_row_stored_with_empty_default_namespace
memory_get_e2e::returns_row_with_arbitrary_nonempty_namespace_in_caller_database
                                                ↑ replaces still_hides_row_tagged_*
memories_list_e2e::pinned_token_can_use_arbitrary_namespace_as_tag_filter
                                                ↑ replaces returns_403_when_pinned_token_asks_for_other_namespace
validate::validate_defaults_when_params_empty           (tag_filter None when no ?namespace)
validate::validate_accepts_arbitrary_namespace_as_tag_filter_for_pinned_token
validate::validate_accepts_matching_namespace_query_for_pinned_token
```

**2,090 tests pass across the workspace.** No regressions.

### Notably NOT changed

- **No write-path behavior change.** `/v1/remember` and `/v1/remember/batch` continue to accept any `namespace` value (including ""), store it as the row tag, and never reject on namespace mismatch. Algo's autonomous loop is unaffected.
- **No backfill migration.** The 200k+ tagged rows are correct as-is; they were never broken, the read path was.
- **`/v1/recall` unchanged.** It never enforced per-row namespace equality.

### Authorship note

The first iteration of this fix (read-side empty-string stopgap, commit `f4951cb`) was authored by the autonomous agent on lane-b (CT 167) as part of Pranab's agentic-experiments program. The generalization to drop the guard entirely + treat `?namespace` as optional tag filter came from yantrikdb-core's decision after pre-merge validation surfaced the data-model conflict.

### Deferred to v0.8.22+

- Link model MCP surface — extend `remember` with `links` arg → `record_with_links_partial`, expand_links on `recall`, `link`/`unlink`/`linked_records`, `reify_supersedes_links` admin tool
- `/v1/admin/audit/leak_candidates` HTTP endpoint
- Proposal 5 validation-gate repair UX

## [0.8.20] — 2026-05-30

Engine bump to v0.7.20 (in-place `correct()` with revision history, schema v30) and removal of pre-split scaffolding that confused external users.

### Changed — engine pin v0.7.19 → v0.7.20

v0.7.20 ships the `correct()` rewrite: in-place mutation with an audit-trail revision history instead of a forget+remember+supersedes pattern. The signature is breaking (requires `reason`, makes `new_text` optional, drops the inline embedding param), but the server binary doesn't call `correct()` directly — it routes through the engine's HTTP/wire interface — so the pin bump applies without source changes in this crate. Schema v30 (`record_revisions` table) lands on first start; migration is additive and forward-only.

The link-model engine work (v0.7.21 / v0.7.22) is intentionally held one release out per the don't-stack-schema-migrations discipline. v0.7.21's schema v31 + the new MCP link surface land in v0.8.21 after a trader soak week on v30.

### Removed — `crates/yantrikdb-python`

The orphaned `crates/yantrikdb-python` directory has been deleted from the workspace. It was pre-split scaffolding from when engine and server lived in one repo: nothing in the server binary linked it, CI didn't build it, releases didn't publish it, and its `pyproject.toml` still claimed the engine's `yantrikdb` PyPI namespace at the stale v0.4.0.

External users running `pip install git+https://github.com/yantrikos/yantrikdb-server` were hitting a pyo3 0.23 / Python 3.14 build failure inside the orphan (#43, reported by @donbowman). The fix is removal rather than maintenance — `yantrikdb-server` is a Rust HTTP daemon, not a Python package. For Python access to the engine, the canonical paths remain `pip install yantrikdb` (engine) and `pip install yantrikdb-mcp` (MCP server wrapper).

Install paths for the server binary, unchanged:

- `cargo install --git https://github.com/yantrikos/yantrikdb-server`
- Docker: `ghcr.io/yantrikos/yantrikdb-server`
- GitHub release binaries (linux-amd64, windows-amd64, macos-arm64, macos-amd64)

Closes #43.

## [0.8.19] — 2026-05-20

Hotfix for the cluster-mode startup regression in v0.8.18. **Cluster operators must upgrade.** Single-node deployments were unaffected by the v0.8.18 regression.

### Fixed — `Cannot drop a runtime in a context where blocking is not allowed`

`run_server` constructs an optional `SplitRuntime` (RFC 009 §4 Layer 1) inside the outer tokio runtime's `block_on` context. The contract at `main.rs:479-486` requires `SplitRuntime::shutdown_timeout` to be invoked from a thread that has no tokio runtime context — otherwise the internal `BlockingPool::shutdown` blocking-wait panics at `tokio-1.52/src/runtime/blocking/shutdown.rs:51`.

The cleanup at the bottom of `run_server` only ran on the clean-exit arm of the top-level `select!`, and even there the call happened inline from inside `async_main`. v0.8.18's engine bump to v0.7.19 exposed an Err path from `raft::assembly` that pre-v0.7.19 didn't hit. The Err propagated up through `?`, the partial-state cleanup dropped `SplitRuntime`'s internal Runtimes from async context, and the panic surfaced ~2 minutes into cluster startup on every cluster-mode deployment.

Two changes bundled:

1. **`main.rs`**: wraps the body of `run_server` between `SplitRuntime` construction and the existing shutdown in an inner async block bound to a local `Result`, so the shutdown runs unconditionally regardless of which path the inner block took.
2. **`main.rs`**: dispatches the actual `rt.shutdown_timeout(...)` call through `tokio::task::spawn_blocking(move || { ... }).await` so the synchronous blocking-pool drain happens on a worker thread that has no tokio runtime context. Without this, the explicit `shutdown_timeout` call panics for the same reason as the implicit drop did.

### Fixed — `reqwest client build failed: incompatible TLS identity type`

Surfaced once the panic above was fixed (the panic had been masking it). On cluster-mode startup, `raft::assembly` builds a `reqwest::Client` with `Identity::from_pem(...)`. With both `rustls-tls` (declared by `yantrikdb-server`) and `native-tls` (pulled in transitively post-engine-v0.7.19) enabled, the Identity is constructed via the rustls backend while the Client builder defaults to native-tls — `builder.build()` rejects the mismatch with `incompatible TLS identity type`.

Fix: explicit `.use_rustls_tls()` on the Client builder in `raft/assembly.rs` pins the backend so the Identity matches.

The transitive feature unification is worth investigating engine-side too (so the workaround can be removed in a future release), tracked separately.

### Engine pin

Unchanged at v0.7.19. The engine fixes from v0.8.18 (compactor + compensating-DELETE) stay in place.

### Process change

Cluster-mode validation is now a hard release gate. v0.8.19 was validated on the live 2-node openraft cluster (CT 141 + .140 peer) end-to-end before tagging: `raft::assembly` completes cleanly, HTTP gateway binds, cluster joined as Follower under leader=4 with `replication_lag_log_entries=0`, no Runtime panic on either the clean-exit or Err-propagation paths.

## [0.8.18] — 2026-05-20

Engine bump to v0.7.19 closes two latent reliability bugs that had
been silently affecting every deployment. **All operators should
upgrade.** The previous releases ran without the engine's background
compactor spawned, which manifested as:

- New deployments locked permanently after the 256th write
  (`503 ingest queue full (256 pending ops, max=256)`).
- Established deployments shed memories silently — production trader
  accumulated 23,043 orphaned memory rows (rows in `memories` table
  with no corresponding `oplog` entry) over 39 days.

Both root-caused via a 2-core LXC reliability bench on 2026-05-20.
Full diagnosis + repro in
[`benchmarks/throughput_2core_lxc/results_2026-05-20.md`](benchmarks/throughput_2core_lxc/results_2026-05-20.md).

### Changed — engine pin v0.7.17 → v0.7.19

The engine ships the two structural fixes; the server change is a
single call site (`tenant_pool.rs::get_engine` now invokes
`spawn_all_workers` and stores the returned guard alongside the
`Arc<YantrikDB>`).

- **`spawn_all_workers` bundle** (v0.7.18): every engine that joins
  the tenant pool now spawns both the materializer threadpool *and*
  the compactor in one call. The guards drop alongside the engine
  Arc when the tenant is evicted, signaling worker shutdown. Thread
  topology gains `yantrikdb-compa` + `yantrikdb-mater` named workers
  (visible in `ps -L`).

- **Compensating DELETE on Backpressure** (v0.7.19): when
  `vec_index.append` returns `Err(Backpressure)`, the engine now
  rolls back the just-inserted `memories` row before propagating
  the 503. No more silent orphan accumulation on transient delta
  saturation.

- **`replication_apply_log` audit table** (v0.7.19, schema v29):
  separates locally-originated writes from replication-applied
  writes so operators can run a three-population audit query to
  detect provenance drift. Existing engines auto-migrate to v29
  on first open.

### Operational guidance

- Existing memory rows that pre-date v0.7.19 stay as-is; the
  retroactive backfill SQL is documented in the engine v0.7.19
  release notes (insert `unknown_path` rows into
  `replication_apply_log` to register pre-existing memories in the
  audit query).
- The compactor is automatic; no operator configuration knob is
  exposed. Tunable env vars (`YANTRIKDB_DELTA_MAX`,
  `YANTRIKDB_MAX_DIRTY_AGE_SECS`) remain at engine defaults (256,
  60s) which suit normal deployments. On 2-core hardware,
  `DELTA_MAX=64 MAX_DIRTY_AGE_SECS=10` lifts read throughput
  ~72% at writers=32 with a 40% bonus to writes — see the bench
  results doc for the empirical sweep.

### Bench results — 2-core LXC, fresh install

| Behavior | Pre-0.8.18 | v0.8.18 |
|---|---|---|
| Sustained write throughput at c=4 (HTTP, pre-computed embedding) | locks at 256 writes | **381/s sustained** |
| Engine ceiling at writers=32 (wedge_repro direct) | n/a (no compactor spawned) | **1115/s** |
| Orphan_delta (memories - oplog `record_with_rid` applied=1) | 1203 (= 503 count) | **0** |
| `yantrikdb-compa` thread on `ps -L` | absent | present |
| `meta.schema_version` on fresh DB | 28 | 29 |

## [0.8.17] — 2026-05-18

HTTP read endpoints for [wysie](https://github.com/wysie)'s
[`yantrikdb-hermes-dashboard`](https://github.com/wysie/yantrikdb-hermes-dashboard)
unlock dashboard HTTP-mode against cluster deployments (the dashboard
was embedded-SQLite-only). [Issue #39 Phase 1](https://github.com/yantrikos/yantrikdb-server/issues/39)
ships three read endpoints plus the substrate that authorizes them.
Engine pin advances to v0.7.17 (no-op for runtime behaviour;
[PR #38](https://github.com/yantrikos/yantrikdb-server/pull/38) carried the bump).

### Added — three Phase 1 HTTP read endpoints

- **`GET /v1/identity-scope`** — returns the principal, effective
  scope, namespace inventory, and identity-scope summary in the
  nested envelope wysie's dashboard reads. Plugin-side concepts
  (`identities`, `actors`, `spaces`, `conversations`) surface as
  empty arrays in Phase 1; engine-side fields (visible namespaces,
  permissions, `namespace_inventory`) are populated.

- **`GET /v1/memories`** — paged, filtered listing of active
  memories. Query params: `namespace` (narrows to token scope),
  `status` (default `active`), `domain`, `memory_type`, `limit`
  (default 50, max 200), `offset` (default 0), `sort` (default
  `created_at`; allowed: `created_at`, `importance`, `last_access`).
  Response envelope `{total, limit, offset, items[]}` with each
  item the dashboard's 25-field row shape.

- **`GET /v1/memory/{rid}`** — point read with conditional include
  arrays (`consolidation_sources`, `entities`, `claims`; empty in
  Phase 1 pending an engine extension). Supports `?min_seq=N` for
  read-your-writes: 412 `replica_behind` if the local node's
  applied seq is below `min_seq`; single-node mode trivially
  satisfies any value.

### Added — RFC 014-B Principal substrate wired through HTTP

- New `auth::middleware::require_authenticated_principal` axum
  layer extracts the Bearer token, calls `AuthProvider`, and
  injects a typed `Principal` into request extensions, or returns
  401 `unauthenticated`. Applied via a sub-router on the three new
  routes only — legacy routes still authenticate inline through
  `resolve_engine` and are unchanged.

- New `auth::ControlDbAuthProvider` bridges the legacy
  `control.tokens` table to RFC 014-B `Principal`:
  - The cluster master secret resolves to a cluster-admin
    principal (`tenant_id=None`, all scopes).
  - Regular tokens resolve to tenant-pinned principals with the
    data-plane scope bundle (`Read | Write | Recall | Forget`).
  - Tokens are stored hashed at rest (`hash_token` via SHA-256);
    the principal `id` exposes a hash-prefix only, never the raw
    token.

- `AppState` gains `auth_provider: Arc<dyn AuthProvider>`,
  constructed in `main.rs` from `cfg.cluster.cluster_secret`.

### Added — structured error envelope across `/v1/*`

All `/v1/*` error responses now emit the canonical envelope:

```json
{ "error": { "code": "stable_id", "message": "human-readable", "hint": "optional" } }
```

The stable `code` is the part of the API surface clients should
branch on. Registry lives at
[`docs/error-codes.md`](docs/error-codes.md) and is mirrored from
`crates/yantrikdb-server/src/api/errors.rs::ApiErrorCode` (gated by
unit tests so both files stay in sync). Legacy call sites emit
`code: "generic"` via a migration shim; new call sites emit
specific codes (`unauthenticated`, `insufficient_scope`,
`namespace_not_found`, `memory_not_found`, `invalid_query_parameter`,
`replica_behind`, etc.).

### Added — namespace + scope guards

- `api::access::require_scope(&principal, scope)` — returns 403
  `insufficient_scope` if the token doesn't hold the required scope.
- `api::access::resolve_namespace(&principal, query_param)` — the
  query param can narrow but never broaden the token's authorized
  namespace. Pinned-vs-cluster-wide policy: cluster-wide tokens
  must specify a namespace explicitly (no implicit default);
  pinned tokens accept their own namespace or none.

### Added — FTS5 fallback marker on `/v1/recall`

POST `/v1/recall` responses now include a top-level `fallback`
field — `"fts5_keyword"` when at least one result was retrieved via
the engine's FTS5 keyword fallback (semantic returned nothing useful),
`null` otherwise. The field is always present so dashboards can
branch on its value without first probing engine version.

### Changed — engine pin

`yantrikdb` engine pin advances from `0.7.16` → `0.7.17`. No
runtime-behaviour change for this release; the bump carries forward
[PR #38](https://github.com/yantrikos/yantrikdb-server/pull/38)'s
engine update (db.reembed() API surface).

### Test coverage — Phase 1 e2e + contract suite

Per the [issue #34](https://github.com/yantrikos/yantrikdb-server/issues/34)
discipline ("integration tests must exercise the production
handler, not mocks"), Phase 1 added an `e2e_test_support` helper
inside `src/` so tests can build a real `AppState` against a
tempdir and drive the production router via `tower::ServiceExt::oneshot`.

Cumulative test delta across Phase 1 commits: **+98 tests**:
- 4 `api::errors` unit tests
- 8 `api::access` guard tests
- 5 `ControlDbAuthProvider` tests
- 9 `identity_scope_tests` unit tests
- 4 `identity_scope_e2e` tests against production router
- 20 `memories_list_tests` unit tests
- 7 `memories_list_e2e` tests against production router
- 7 `memory_get_e2e` tests against production router
- 8 `fts5_fallback_tests` unit tests
- 10 `contract_fixture_tests` (7 self-tests + 3 live shape-asserts
  against fixtures in `src/api/fixtures/`)
- 16 supporting auth-substrate tests (Principal, ScopeSet, audit,
  middleware)

Verification on the v0.8.17 release branch:

```
cargo fmt --check                                # clean
cargo test --package yantrikdb-server            # 899 (bin) +
                                                  # 13 (http_integration) +
                                                  # 354 + 441 + 360 + 2 + 4 + 2 + 2 = 2077 passed
                                                  # 0 failed / 2 ignored
```

### Phase 1 deliberate limitations (Phase 2/3 follow-ups)

The shape ships; the depth follows engine extensions.

- `identity_scope.{identities,actors,spaces,conversations}` — empty
  arrays; these are plugin-side concepts (hermes/MCP/Slack) the
  engine doesn't store.
- `namespace_inventory[].count` — `null`; populating per-namespace
  counts for cluster-admin tokens would mean opening every engine
  on a synchronous handler call (deferred to a follow-up cached
  path).
- 25-field memory rows — `updated_at`, `updated_at_iso`,
  `tombstone_reason`, `embedding_model`, `embedding_bytes` all
  emit `null`. Engine v0.7.x doesn't surface these on
  `list_memories` / `get`.
- `/v1/memories` filters — `q` (full-text search), `source`, and
  `sort ∈ {updated_at, access_count, certainty}` each return a
  specific 400 `invalid_query_parameter` so clients see an honest
  contract instead of silently getting `created_at`-sorted rows.
- `/v1/memory/{rid}.consolidation_sources/entities/claims` — empty
  arrays; engine's per-memory lookup APIs are currently `pub(crate)`
  and not reachable from the server crate.
- `?min_seq=N` — reject-not-wait: engine v0.7.17 doesn't expose a
  `wait_for_visible_seq` primitive yet, so the handler returns 412
  `replica_behind` immediately when the local node is behind. A
  future engine bump may upgrade this to wait-and-then-reject for
  better UX under transient lag.

### Pull requests

- [#40](https://github.com/yantrikos/yantrikdb-server/pull/40) — Phase 1 implementation
- [#38](https://github.com/yantrikos/yantrikdb-server/pull/38) — engine bump 0.7.16 → 0.7.17 (already merged at d67cd4a)

---

## [0.8.16] — 2026-05-16

Docker bug fixes from [@renothing](https://github.com/renothing)'s
[issue #35](https://github.com/yantrikos/yantrikdb-server/issues/35),
plus the engine bump to v0.7.16. Both deployment failure modes
`renothing` reported now have proper fixes baked into the official
image.

### Fixed — air-gapped Docker first-run failure (#35 part 1)

`docker run --network=none yantrikos/yantrikdb` previously failed with:

```
Error: Failed to retrieve model.onnx
  https://huggingface.co/Qdrant/all-MiniLM-L6-v2-onnx/resolve/main/model.onnx:
  Dns Failed
```

…because the default `EmbeddingStrategy::Builtin` fetched the MiniLM
ONNX model from HuggingFace at first run. The Docker image now ships
`/etc/yantrikdb/yantrikdb.toml` configured with `strategy = "bundled"`,
which uses the engine's `potion-base-2M` static embedder (~7 MB baked
into the binary via `include_bytes!`, dim=64, zero network, ~89% of
MiniLM recall@5 quality).

Users who want the original MiniLM behavior can mount their own
config:

```bash
docker run -v ./my.toml:/etc/yantrikdb/yantrikdb.toml:ro yantrikos/yantrikdb
```

The in-process default (when no config file is present) is unchanged
at `Builtin` + `dim=384` — backwards compat for existing single-binary
deployments that already have dim=384 vector stores.

### Fixed — volume mount permission denied (#35 part 2)

`docker run -v ./data:/var/lib/yantrikdb yantrikos/yantrikdb`
previously failed with:

```
Error: unable to open database file: /var/lib/yantrikdb/control.db
  Caused by: Error code 14: Unable to open the database file
```

…because the in-container `yantrikdb` user (UID assigned by
`useradd -r`, typically 100-999) couldn't open SQLite files inside a
host-mounted directory owned by a different UID. A new
`docker/entrypoint.sh` chowns `/var/lib/yantrikdb` to the `yantrikdb`
user on container start, then `exec gosu` drops privileges. Standard
Postgres/Redis Docker convention. Skips the chown if the container
was started with `-u <uid>` explicitly — that user took responsibility
for ownership themselves.

### Added — `EmbeddingStrategy::Bundled` variant

```toml
[embedding]
strategy = "bundled"
dim = 64
```

Uses `yantrikdb::embedder::BundledEmbedder` directly. dim=64. Zero
HuggingFace dependency. Logged on startup as
`embedding strategy: bundled (engine BundledEmbedder, potion-base-2M,
dim=64, zero-network)`.

### Changed — engine pin v0.7.15 → v0.7.16

Picks up the v26 schema migration foundation
([engine PR #38](https://github.com/yantrikos/yantrikdb/pull/38),
closes engine #29):

- 4 additive columns on `memories` for conflict-aware-write provenance
  (`prior_rid`, `resolution_kind`, `dismissal_reason`,
  `confidence_at_write`)
- 2 partial indexes for the resolution/supersession query patterns
- source-enum normalization

Pure foundation — columns are NULL on every existing row, not yet
populated by any write path. Replay-safe per the v0.7.3 idempotent
migration runner contract. Byte-equivalent runtime surface to v0.7.15
from the agent caller's perspective.

### Refactored — `TenantPool` holds `ServerEmbedder` enum

`ServerEmbedder { Fast(FastEmbedder), Bundled(BundledEmbedder) }`
implements `yantrikdb::types::Embedder` so call sites don't need to
match on the variant.

### Verification

- `cargo fmt --check`: clean.
- `cargo test --workspace --exclude yantrikdb-python --exclude yantrikdb-wasm`:
  **2005 tests pass / 0 fail / 2 ignored**.
- Full CI matrix on the merge PR (#36) green: format / supply-chain /
  clippy / build-and-test ubuntu+macos+windows.

### Known follow-up gaps

The Docker workflow only builds + pushes images — it does NOT run
them. The bundled-embedder default and the entrypoint script are
exercised only when downstream users pull the image. A
`docker run --rm --network=none ghcr.io/yantrikos/yantrikdb --version`
smoke step in `docker.yml` would close this gap; queued as a
post-v0.8.16 follow-up PR.

## [0.8.15] — 2026-05-16

**Critical fix for silent data-loss regression introduced in v0.8.13.**

Closes yantrikos/yantrikdb#37 (reported externally by acidport on ARM64
Oracle Cloud Docker, but root cause is platform-agnostic and affects
every single-node deployment on v0.8.13 or v0.8.14).

### Fixed

- **`/v1/remember`, `/v1/relate`, and the other commit-routed write
  handlers were durably appending to the commit log but skipping the
  applier dispatch in single-node mode.** The receipt returned a valid
  `log_index`, but the engine's `memories` table, vec_index, and
  scoring cache were never updated. `/v1/recall` therefore could not
  find the row, and `stats` memory counts did not grow.

  Root cause: `CommitOptions` was declared with `#[derive(Default)]`
  (PR-6.1, shipped in v0.8.13). `bool::default()` is `false`, so
  `CommitOptions::default()` produced `wait_for_apply: false` — the
  no-wait bulk-import shape — even though the documented default for
  that field (and the explicit `CommitOptions::new()` constructor) is
  `true`. The production HTTP handlers in `http_gateway.rs` all called
  `CommitOptions::default()`, taking the no-wait branch on every
  write.

  Cluster (Raft) mode was unaffected: openraft's
  `apply_to_state_machine` callback drives apply on every node
  independently of the `wait_for_apply` field.

  Fix: replaced `#[derive(Default)]` with an explicit
  `impl Default for CommitOptions { fn default() -> Self { Self::new() } }`.
  All four production handlers automatically heal — no handler changes
  needed.

### Added — regression tests

- **`commit_options_default_is_safe` upgraded** (commit/trait_def.rs):
  the pre-existing test was named for the invariant ("Default MUST
  wait_for_apply=true") and commented for the failure mode ("a footgun
  that causes 'I committed but recall doesn't see it' reports"), but
  the test body only asserted on `CommitOptions::new()`. It now also
  asserts on `CommitOptions::default()` so a future revert of the
  explicit `Default` impl trips at the type level.

- **`issue_37_default_options_dispatches_applier`** (commit/submitter.rs):
  end-to-end pin at the layer where the regression actually lived.
  Builds a `LocalSqliteSubmitter` + concrete `LocalApplier`, calls
  `submit` with `CommitOptions::default()`, and asserts the applier's
  high watermark advances to 1. Pre-fix this watermark would have
  stayed at 0 because the no-wait branch skips the apply call.

### Known follow-up — test-coverage gap

`tests/http_integration.rs` mounts its own **mock** `handle_remember`
handler set rather than importing the production handlers from
`src/http_gateway.rs`. The mock and production handlers can drift
silently; this is exactly how the v0.8.13 regression shipped under
green CI. The structural fix is to promote `AppState` + handlers out
of the bin-only crate into a sibling `lib.rs` so integration tests
can import them. Filed as a follow-up issue. Until then, treat
`http_integration.rs` as wire-protocol coverage only; handler logic
is locked at the submitter/committer unit-test layer.

## [0.8.14] — 2026-05-14

Engine dependency bump. No yantrikdb-server source changes.

Picks up 13 yantrikdb engine patch releases (v0.7.3 → v0.7.15) since
the v0.8.13 pin. The bulk are pyo3-binding-only improvements that the
server transitively gets for free; the operationally relevant deltas
are listed below.

### Changed — engine pin v0.7.2 → v0.7.15

- **Idempotent migration runner** (engine v0.7.3 + v0.7.8). The engine
  now splits migration batches per-statement and swallows safe replay
  errors (duplicate column on additive migration replay, ALTER-on-view
  after table-to-view conversion, RENAME-TO collisions, no-such-column
  on renamed-away artifacts). `meta.schema_version` is MAX-stamped on
  every open so an accidental old-binary-against-new-DB run can no
  longer rewind the version stamp. Defensive against the homelab
  cluster v0.8.13 upgrade incident where a brief rollback to an older
  engine binary against a v24-schema DB tripped on
  `duplicate column name: embedding` on the next forward upgrade.
- **`potion-multilingual-128M` embedder** (engine v0.7.9). New tier in
  the engine's embedder-download registry: 101 languages, dim=256,
  BGE-M3 tokenizer, ~460 MB tarball, SHA-256 pinned. dim matches
  `potion-base-8M` so swapping is no-DB-reopen on the engine side.
  Accessible via the embedded engine's `set_embedder_named` API; the
  server can advertise it on the HTTP path in a follow-up if useful.
- **Embedder tarball extractor handles both archive layouts** (engine
  v0.7.13 + v0.7.15). Closes engine issue #15 + restoring a missing
  extract call.
- **SLSA build provenance attestation** on engine release artifacts
  (engine v0.7.12). Engine release pipeline now emits SLSA attestations
  consumable by downstream verifiers.
- **sdist LICENSE path fix** in engine wheels (engine v0.7.14 + v0.7.15).

The engine pin moves to git commit `9747c609` (= engine main =
yantrikdb v0.7.15).

### Verification

- `cargo check -p yantrikdb-server`: clean, 21.44s, 433 pre-existing
  warnings, zero new ones.
- `cargo test --workspace --exclude yantrikdb-python --exclude yantrikdb-wasm`:
  **2000 tests pass / 0 fail / 2 ignored** across 14 buckets.
- Full CI matrix on the merge PR (#30) green: clippy / format /
  supply-chain / build-and-test on ubuntu / macos / windows.

## [0.8.13] — 2026-05-08

RFC 010 PR-6 — handler migration. **Cluster mode replication actually
works now.** The four hot-path HTTP write endpoints route through the
durable commit log so every committed mutation flows leader → openraft
consensus → state machine apply → engine state on every node. Single-
node mode also routes through the commit log via a unified
`LocalSqliteSubmitter` that composes log-append + applier dispatch.

The cosmetic-openraft regression that the architect surfaced on
2026-05-02 is structurally closed. Empirical RYW (write to .140 leader,
recall on .141 follower) is the next gate — runnable now with this
release on the homelab cluster.

The interim cluster-routing runbook
(`docs/operations/cluster-routing.md`) becomes obsolete once homelab
empirical RYW passes.

### Added — EngineApplier real-engine dispatch (RFC 010 PR-6.4 part 1)

- `commit::EngineApplier` impls `Applier` over an `EngineResolver`
  trait. Dispatches each `MemoryMutation` variant to the corresponding
  deterministic engine primitive: `record_with_rid`,
  `tombstone_with_rid`, `upsert_entity_edge_with_id`,
  `delete_entity_edge_with_id` — all carrying `Some(log_index)` as the
  per-tenant seq.
- Replay detection: same `(tenant_id, log_index)` returns
  `ApplyError::AlreadyApplied` without invoking the engine. Snapshot
  install + log replay overlap is the normal trigger.
- Engine API is sync; `apply` wraps dispatch in
  `tokio::task::spawn_blocking` so HNSW + SQLite work doesn't park a
  tokio worker.

### Added — TenantPoolEngineResolver (RFC 010 PR-6.4 part 2)

- `tenant_pool::TenantPoolEngineResolver`: adapter from
  `Arc<TenantPool> + Arc<Mutex<ControlDb>>` to the `EngineResolver`
  trait. On apply, looks up `tenant_id → DatabaseRecord` via
  `control.get_database_by_id()` then defers to `pool.get_engine()`
  (lazy-load if cold).
- Resolution failures (control-DB lookup error, tenant not found,
  engine open failure) surface as `ApplyError::EngineFailure` —
  catastrophic at apply time because the entry is already durable in
  the log. State machine treats as divergence risk per RFC 010 §4.

### Added — Assembly threading + AppState submitter wiring (RFC 010 PR-6.4 part 3)

- `RaftAssemblyConfig` carries `applier: Arc<dyn Applier>` instead of
  the previous hard-coded `LocalApplier` placeholder. `build_raft_cluster`
  uses what the caller supplies. Tests pass `LocalApplier` for trait-
  shape coverage; production passes `EngineApplier`.
- `main.rs` openraft branch instantiates `EngineApplier::new(
  TenantPoolEngineResolver::new(pool, control))` and threads it through
  the assembly. State machine's `apply_normal` dual-call now lands
  committed mutations into engine state on every node.
- `main.rs` single-node branch wraps the existing `LocalSqliteCommitter`
  in a `LocalSqliteSubmitter` with the same `EngineApplier`, exposed via
  the existing `AppState.commit_log: Arc<dyn MutationCommitter>` slot
  (PR 6.4 lands `impl MutationCommitter for LocalSqliteSubmitter`). No
  parallel trait machinery needed.

### Changed — HTTP handlers route through commit_log (RFC 010 PR-6.4 part 4)

The four hot-path write endpoints no longer call `engine.record()` /
`engine.forget()` / `engine.relate()` directly. They build a
`MemoryMutation` and call `state.commit_log.commit(...)` instead.
Single-node: `LocalSqliteSubmitter` writes the durable log + dispatches
to `EngineApplier` inline. Cluster: `RaftCommitter` routes through
openraft consensus → state machine apply → applier on every node.

- `/v1/remember` — allocates `rid` UUIDv7 server-side, builds
  `UpsertMemory` mutation, commits. Pre-embedding (Issue #19) and
  quota check unchanged.
- `/v1/remember/batch` — same per-entry. Pre-embedding still batched
  for ONNX-mutex coalescence; commits one mutation per entry.
  Response carries the last `log_index` in the run.
- `/v1/forget` — builds `TombstoneMemory` with `requested_at_unix_micros`
  and `namespace`, commits. Engine `tombstone_with_rid` runs on apply.
- `/v1/relate` — allocates `edge_id` UUIDv7, builds `UpsertEntityEdge`,
  commits. Engine `upsert_entity_edge_with_id` runs on apply.

### Cluster mode caveat (still relevant for empirical validation)

Structural migration is complete; **empirical RYW on the homelab .140 →
.141 cluster has not yet been run.** The interim cluster-routing
runbook stays in place until that empirical test passes. Operators
running multi-node deployments should read
`docs/operations/cluster-routing.md` and the RFC 010 PR-6 spec at
`docs/rfcs/rfc_010_pr6_write_path_migration.md` before flipping
production traffic.

### Known limitations (deferred to v0.8.14 or follow-ups)

- **PR 6.7 chunked snapshot** — current snapshot serializes whole
  tenant commit logs to JSON in memory. Works for homelab-scale
  (~40k memories) but not unbounded. Chunked streaming via openraft
  `generic-snapshot-data` ships in v0.8.14.
- **PR 6.8 backfill admin tool** — `yantrikdb admin backfill-from-engine`
  for migrating existing engine rows that predate the commit log. Ships
  in v0.8.14. Without it, a fresh cluster works but existing single-node
  data on .140 (39427 memories) needs manual migration to flow into the
  log.
- **Wire-protocol path (yql via `handler::execute_with_guard`)** still
  uses `Command::Remember` → `engine.record()`. HTTP is the production
  hot path; wire-protocol migration is a follow-up.
- **`extracted_entities` materialization**: handlers pass empty vec;
  engine's NER on `record_with_rid` does not run on empty entities.
  Entity-aware recall (`expand_entities=true`) loses signal vs prior
  behavior. The Materializer trait (PR 6.2) is in place but not wired
  into the handler. Follow-up.
- **Client-supplied op_id** for HTTP idempotency: structural support
  exists (`CommitOptions.op_id`, 409 mapping in
  `commit_error_to_app_error`). Handler doesn't read `body["op_id"]`
  yet. Follow-up.
- **Command::IngestClaim, Command::AddAlias, Command::Resolve,
  Command::IngestClaimWithLineage**: no `MemoryMutation` variants exist
  for these in the grammar yet. Follow-up RFCs.

### Test counts

- Unit tests: 809 → 814 (+5)
- Integration tests across all suites: 1175 (stable)
- Cumulative: 1989 tests green, cargo fmt clean

### Engine dependency bump

- `yantrikdb` engine: 0.6.7 → 0.7.2. Brings in:
  - **0.7.0** — decoupled write path (Phase 4.3): WAL → bounded ingest
    queue → background materializer threads → DeltaIndex (ArcSwap cold +
    `RwLock<Vec>` delta) → compactor. Wedge primitive #1 fix.
  - **0.7.1** — atomic-counter hotfix for `log_op_pending` regression
    that briefly tanked write throughput in 0.7.0.
  - **0.7.2** — event-driven compactor wake at 80% delta capacity
    (saga task #18 Option 4). Empirically validated cross-platform.
    Phase-3 recovery onset moved sec 71 → sec 56-58. Engine pressure
    67.6% → 3.1%. + bundled `potion-base-2M` static embedder
    (~7.9 MB, dim=64, pure Rust via `model2vec-rs`) — ONNX Runtime no
    longer required for the default install.

### Files touched

- `crates/yantrikdb-server/src/commit/applier.rs` (+EngineApplier, +EngineResolver, +apply_to_engine dispatch)
- `crates/yantrikdb-server/src/commit/submitter.rs` (+impl MutationCommitter for LocalSqliteSubmitter, +trait method disambiguation in tests)
- `crates/yantrikdb-server/src/raft/assembly.rs` (+applier field on RaftAssemblyConfig)
- `crates/yantrikdb-server/src/tenant_pool.rs` (+TenantPoolEngineResolver)
- `crates/yantrikdb-server/src/main.rs` (+single-node + cluster applier wiring)
- `crates/yantrikdb-server/src/http_gateway.rs` (4 handlers migrated)

## [0.8.12] — 2026-05-05

RFC 010 PR-6 substrate-batch (PR 6.1, 6.2, 6.3 of 9). Pure additive
scaffolding for cluster-mode replication actually working. **Cluster
behaviour is unchanged at v0.8.12** — handlers still call
`engine.record()` directly until PR 6.4 (target v0.8.13). Production
write path bypasses the new traits; they're in place but unwired.

The architectural commitments now in code: Submitter and Applier are
separate, mutations carry materialized state for byte-deterministic
follower apply, per-tenant commit-log files are addressable. Depth
(handler migration, boot invariants, error mapping, snapshot, backfill,
extended health surface) lands across PR 6.4–6.9 in v0.8.13 + v0.8.14.

Hard blocker for PR 6.4 is filed at yantrikos/yantrikdb#9: engine API
addition for `record_with_rid` + friends at engine 0.7.0.

Full design at `docs/rfcs/rfc_010_pr6_write_path_migration.md`. Interim
operator runbook for the cluster ghosting symptom that motivated this
work at `docs/operations/cluster-routing.md`.

### Added — Submitter / Applier trait split (RFC 010 PR-6.1)

- `commit::Submitter` trait — durable log append. Single-node
  `LocalSqliteSubmitter` delegates to existing `LocalSqliteCommitter`;
  cluster `RaftSubmitter` ships in PR 6.4.
- `commit::Applier` trait — state-machine apply. `LocalApplier` is the
  only path that mutates engine state, runs on both single-node and
  cluster nodes for byte-deterministic apply.
- `ApplyError` with `NotYetWired` (PR 6.1 placeholder for variants
  awaiting engine wiring), `AlreadyApplied` (idempotent replay
  detection), `EngineFailure`. `is_idempotent_ok()` classifies which
  errors callers MAY treat as success.
- 16 new unit tests (8 applier + 8 submitter) covering trait
  conformance, op_id idempotency, monotonic per-tenant log_index,
  watermark tracking, dyn-compatibility compile-time pin.

### Added — Mutation determinism + wire 1.1 (RFC 010 PR-6.2)

- `MemoryMutation::UpsertMemory` grows three materialized-state fields:
  - `extracted_entities: Vec<String>` — NER output stamped at the leader
  - `created_at_unix_micros: Option<i64>` — server-assigned timestamp
  - `embedding_model: Option<String>` — model id for RFC 013 migration
- All three carry `#[serde(default)]` so v1.0 payloads round-trip
  cleanly. v1.1 wire output golden-pinned by
  `upsert_memory_v1_1_wire_format`. v1.0 historical compat preserved
  by `historical_v1_0_payload_round_trips_into_current_build`.
- `CURRENT_WIRE_VERSION` bumped to (1, 1).
- `FEATURE_FLOORS` gains `mutation.UpsertMemory.materialized` at wire
  1.1 — writers gate emission of populated fields on `cluster_min`
  observation per RFC 017-A.
- `commit::materialize` module: `Embedder`, `EntityExtractor`,
  `Materializer` traits + `LocalMaterializer` impl. Converts a
  `RememberRequest` into a fully-materialized mutation. Real engine
  wiring (yantrikdb::YantrikDB::embed + engine NER) lands in PR 6.4.

### Added — TenantCommitConnectionPool (RFC 010 PR-6.3)

- `commit::tenant_pool::TenantCommitConnectionPool`: per-tenant SQLite
  connection cache for the commit-log table living inside each
  tenant's `yantrik.db` (Option D layout). LRU eviction at default
  `max_size=256`; `close_idle(threshold)` for periodic maintenance.
  WAL mode pragmas matching `LocalSqliteCommitter`.
- `PathResolver` typedef so callers wire path-resolution to whatever
  logic they already use.
- 9 new unit tests covering first-open creates file + runs migrations,
  cached Arc on repeat open, distinct connections per tenant, idempotent
  migration across reopens, LRU eviction at max_size, idle-eviction,
  WAL pragmas set, parent dir auto-created.

### Changed

- ROADMAP.md refreshed: corrects the prior load-bearing "the substrate
  is correct" claim, sequences PR-6 (v0.8.12 → v0.8.14) ahead of
  concurrency series (v0.8.15 → v0.8.19) ahead of RFC 023 epistemic
  control plane (v0.8.20+).

### Test counts

- Unit tests: 770 → 785 (+15 across the three new modules)
- Integration `commit_replay`: 338 → 344
- Integration `wire_format_v1_0`: 342 → 348
- Cumulative: 1450 → 1477 tests green, cargo fmt clean

### Cluster mode caveat

Cluster mode in v0.8.12 is structurally unchanged from v0.8.11:
writes still bypass the openraft commit log; `last_log_index` only
moves on cluster bookkeeping (membership, init), not application
data. Operators running multi-node deployments should follow
`docs/operations/cluster-routing.md` until PR 6.4 ships.

## [0.8.11] — 2026-05-01

First-class skill primitives + follower HNSW backfill fix. Closes the
"wrote a memory, can't recall it for hours" follower lag that yantrikdb-agi
reported, and gives every consumer a stable substrate-layer skill API
instead of each program reinventing `skill_define` / `skill_get` /
`skill_recall` in agent code with subtle bugs.

Designed via three multi-voice brainstorm sessions (gpt-5.5 + deepseek +
claude) on 2026-05-01. Full RFC at `docs/rfcs/rfc_022_skill_substrate_and_ryw.md`.
Story of how the design got there at `docs/blog/2026-05-01-how-v0.8.11-got-designed.md`.

### Added — Skill API (RFC 022 §1)

Five new endpoints under `/v1/skills/*`. Thin wrappers over existing
memory primitives, hardcoded to `namespace=skill_substrate`,
`metadata.record_type=skill`, `memory_type=procedural`. Strict shape
validation; **no semantic ontology** (no validation gates, no auto-rollup
of success_count, no origin immutability).

- `POST /v1/skills/define` — write a new skill record. Validates:
  - `skill_id` matches `^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$`, length 4..200,
  - `body` length 50..5000,
  - `applies_to` non-empty array (≤10) of entries matching
    `^[a-z][a-z0-9_]*$` — catches the hyphen-vs-underscore drift bug
    Brainstorm 2 named (`["meta_agent"]` vs `["meta-agent"]` — both
    valid strings, only one matches),
  - `skill_type` ∈ {procedure, reference, lesson, pattern, rule}.
  Default `on_conflict=reject` returns 409 Conflict. `?on_conflict=update`
  for upsert (tombstones existing). `?on_conflict=ignore` returns the
  existing rid as a no-op.
- `GET /v1/skills/{skill_id}` — exact lookup. v0.8.11 uses
  scan-then-filter (O(N), bounded at 10000 records); v0.8.12 will replace
  with `/v1/lookup` for O(log N) via indexed metadata.
- `POST /v1/skills/search` — semantic search over `skill_substrate`
  with optional `applies_to` and `skill_type` filters (post-fetch in
  v0.8.11; prefilter via where-clause arrives in v0.8.13).
- `POST /v1/skills/{skill_id}/outcome` — append-only event log written
  to `outcome_substrate` namespace with `metadata.skill_ref={skill_id}`.
  **Engine never auto-rolls-up `success_count` on the parent skill**
  — architectural enforcement of schema-not-semantics. Programs that
  want rollups query `outcome_substrate` themselves and aggregate.
- `POST /v1/skills/{skill_id}/forget` — tombstone the skill record.

### Fixed — Follower HNSW backfill (RFC 022 §2)

The architect of yantrikdb-agi reported on 2026-05-01: *"newly-written
memories via /v1/remember are not findable via /v1/recall for some
indeterminate window after the write returns."* Initial RFC §2 draft
proposed an in-engine pending-vector overlay (~300 LOC). A 90-second
empirical test against the live cluster showed engine `record()` already
does synchronous HNSW insert — leader-side is RYW-consistent. The
actual bug lives in **follower** replication backfill at
`crates/yantrikdb-server/src/cluster/sync_loop.rs:369` which called
`db.rebuild_vec_index()` (full HNSW rebuild) at the end of every
backfill batch. With 1k+ memories per tenant the rebuild took seconds;
with 10k+ minutes; under load the multi-hour lag the architect saw.

Two engine API additions enabled the fix:

- `pub fn YantrikDB::insert_vector(&self, rid: &str, embedding: &[f32]) -> Result<()>`
- `pub fn YantrikDB::encrypt_embedding_pub(&self, blob: &[u8]) -> Result<Vec<u8>>`

Both promoted from `pub(crate)` to `pub` in the engine library bump
`yantrikdb` 0.6.4 → 0.6.5. Then `backfill_embeddings()` was rewritten
to do per-row HNSW insert instead of batch-end full rebuild. Effect:

- **Follower recall lag**: minutes-to-hours → seconds (per-row
  O(log N) insert instead of O(N log N) rebuild on the entire
  memories table per cycle).
- **Encrypted-cluster follower recall**: starts working for the first
  time. Pre-v0.8.11 the backfill skipped encrypted writes entirely
  with a TODO comment, because `encrypt_embedding` was `pub(crate)`.
- **Recall on each backfilled memory**: visible as soon as that memory's
  insert completes, not at the end of the batch.

### Architecture commitment

**Schema, not semantics.** The substrate stores structured authority data
(skill_id format, applies_to entry regex, skill_type enum); the agent layer
decides what to *do* with it. This RFC explicitly refuses validation gates,
outcome rollups, origin immutability, and any naming that overclaims
semantic capability (`/v1/skills/learn`, `/v1/skills/master`, etc.).

The pending-vector overlay design that the empirical test killed is
preserved as RFC 022 §2.7 historical reference. If a future change moves
HNSW insert to an async/batched path on the leader (e.g., to scale write
throughput), that overlay is the spec for restoring leader-side RYW.

### Backwards-compat

Fully backwards-compatible. v0.8.10 clients see no behavior change.
- Existing `/v1/remember`, `/v1/recall`, `/v1/forget`, etc. signatures unchanged.
- Engine library bump is purely additive (two methods promoted to `pub`).
- No schema migration.

### Operator notes

- After upgrade, follower nodes start using piecewise insert immediately on
  their next backfill cycle. No manual reindex needed.
- A 0.8.10 follower running against a 0.8.11 leader keeps the old (slow)
  full-rebuild path until upgraded — rolling upgrade is safe but the lag
  benefit is only realized on upgraded followers.
- Existing skills written via `/v1/remember` directly with the
  `namespace=skill_substrate, metadata.record_type=skill, skill_id=...`
  convention continue to work via `/v1/skills/get` and `/v1/skills/search`
  (the convention matches what the API expects).

### Coming next

- **v0.8.12**: namespace_schema + `/v1/lookup` makes `/v1/skills/{id}` O(log N).
- **v0.8.13**: where-clause prefilter on `/v1/recall` + planner explain;
  yantrikdb-agi migrates 67 skills onto indexed substrate.
- **v0.8.14+**: RFC 023 epistemic-control primitives one per release
  (provenance, supersede chain, recall perimeter, action-conditioned veto,
  scoped negative evidence, memory-of-missing-memory, quarantine, latent
  contradiction awareness).
- **v1.0**: category-shift release. *"YantrikDB is not a vector database.
  It's an epistemic control plane for LLMs."*

[0.8.11]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.11

## [0.8.10] — 2026-05-01

Critical follow-up to v0.8.9. The admission-control inflight counter
in `execute_cmd` was incremented before the `await` on
`spawn_blocking` and decremented inline afterward. When axum dropped
the future on client timeout / disconnect, the decrement was skipped
and the counter leaked +1 per cancellation. Over hours of traffic
the counter saturates at `MAX_INFLIGHT` (256) and the gate stays
permanently closed — every subsequent request is rejected with
`server overloaded: 256 inflight ops (max 256). Retry later.`,
even when the actual `recall_in_flight` gauge reads 0–2 and CPU
sits below 1%.

Field observation (homelab leader, 12 h uptime, single-vCPU Docker):
all recall requests fast-rejected with the 256/256 message. Container
restart was the only recovery. Pre-existing v0.8.9 read-pool
benchmarks still passed *immediately after restart*, then the gate
re-saturated as Lane B agents accumulated 30 s timeouts.

### Fixed

- **`InflightGuard` RAII drop guard.** The counter is now wrapped in
  a stack-local guard whose `Drop` impl runs the `fetch_sub` on
  every exit path including future cancellation, panic unwind from
  `spawn_blocking`, and early `?` propagation. Replaces the inline
  `inflight.fetch_sub(1, Ordering::Relaxed)` after the `await`
  which only ran on the success path.

### Reproduced + verified

Stress test on the patched binary (single-vCPU container,
Docker on Windows): 1000 forced-cancel requests
(`curl --max-time 0.03`) followed by a 50-concurrent recall round
returned 35×200 / 15×503 — exactly the v0.8.9 acceptance numbers.
Pre-fix, the same workload would have permanently saturated the
counter and forced 0×200 / 50×503 with no recovery short of
restart.

`yantrikdb_recall_in_flight` and `yantrikdb_expansion_concurrent`
gauges (which were *not* affected by the leak) confirmed no real
backlog: both returned to 1–2 within 30 s of the cancel storm.

### Operator notes

- If you upgraded to v0.8.9 and saw ".140 returning 503 'server
  overloaded: 256 inflight ops' even though `engines_loaded:7` and
  CPU is idle" — that's this bug. v0.8.10 fixes it; no migration
  needed beyond replacing the binary / image.
- The leak existed in v0.8.0 onward but was masked by the v0.8.0–8
  Mutex contention bottleneck (concurrent recalls were so slow that
  cancellations were rare). v0.8.9's read pool made the server fast
  enough that real workloads cancelled enough to expose it.

[0.8.10]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.10

## [0.8.9] — 2026-05-01

Critical fix for concurrent recall. A single client issuing parallel
recall queries clogged a CPU core and caused admission control to
shed most requests as 503, even though SQLite WAL mode supports
concurrent readers natively. The bottleneck was two layers of
unnecessary serialisation:

1. The engine library held all SQLite work behind a single
   `Mutex<Connection>`, so all recalls queued head-of-line.
2. The server wrapped the engine in an outer `Arc<Mutex<YantrikDB>>`
   that further serialised every handler call, even though
   `YantrikDB` is already `Send + Sync` with internal locks on
   each field.

Operator-facing impact (Docker on Windows, .140):

| Test                     | v0.8.8 (pre-fix)       | v0.8.9                |
|--------------------------|------------------------|------------------------|
| 10 concurrent recall     | 7×200, 3×503, p99 341ms| 10/10, p99 143 ms     |
| 50 concurrent recall     | 0×200, 47×503, 3 t/o   | 33×200, 17×503, 0 t/o |
| Single-AGI sustained CPU | 100% on 1 core (queue) | 100% doing real work  |

LXC-on-bare-metal (.141, 2 vCPU) is similar: 50-conc lands 42×200,
8×503 admission shed, 0 timeouts.

### Fixed

- **Engine read connection pool (yantrikdb 0.6.4).** `YantrikDB`
  now opens N additional read-only SQLite connections in WAL mode
  (default `YANTRIKDB_READ_POOL=4`), each behind its own mutex.
  Recall paths in `engine::recall` acquire any free connection
  round-robin via the new `read_conn()` method instead of the
  single write `conn` mutex. Writes (record/forget/correct) and
  schema migrations continue to use the write `conn`, so SQLite's
  single-writer rule is preserved without explicit coordination.

- **Removed `Arc<Mutex<YantrikDB>>` from server.** `TenantPool`
  now holds `Arc<YantrikDB>` directly. `YantrikDB` is asserted
  `Send + Sync` in the engine library; all top-level methods take
  `&self` and use internal locks. The outer mutex was dead
  serialisation that prevented every form of concurrency.

- **Watchdog probes the engine, not a lock.** The 15 s
  built-in watchdog used to time `engine.try_lock()` (now
  meaningless) — it now times `engine.stats(None)` instead, which
  exercises the read pool and reports a real liveness signal.

### Correctness verified

- `forget`-immediately-hides invariant: 0 stale reads sequentially
  (5 polls per probe), 0 stale reads under race (5 concurrent
  recallers per probe × 5 probes after the tombstone commit). The
  write conn commits the tombstone; pooled read conns begin a new
  WAL transaction per query and see the latest committed state.
- 1315 engine tests pass (was 1314 — engine 0.6.3 added one).
- 326 server tests pass.

### Telemetry added

- `LockHoldTimer` RAII helper in `metrics.rs` — wraps any
  short-lived engine borrow and emits a `WARN` log if it exceeds
  500 ms. Background workers in `background.rs` are instrumented;
  the HTTP layer extracts an op-name in `execute_cmd` for
  `engine_lock_hold_ms` histograms grouped by command.

### Tuning notes

- `YANTRIKDB_READ_POOL=0` disables the pool; recall falls back to
  the write conn. Useful only for debugging or tests.
- For high-fanout workloads (many tenants, many concurrent
  recallers per tenant), values above 4 may help. Each connection
  is ~1 MB of resident memory.

[0.8.9]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.9

## [0.8.8] — 2026-05-01

Performance fix. Operators reported recall queries timing out
(8 s default client timeout) on first hit against any namespace,
even though steady-state recall latency is ~20 ms.

### Fixed

- **Eager engine warm-up at startup.** Previously engines were
  loaded lazily on first query — `TenantPool::get_engine()` is
  called inline from the request handler, blocking the request for
  ~10 s while HNSW reloads from disk for a 400 MB engine. Clients
  with default timeouts saw "transport: timeout" errors even though
  the server was healthy.

  Now: at startup, after `TenantPool` is created, the server
  enumerates all databases from `control.db` and calls `get_engine`
  for each (sequentially, with progress logging). HTTP starts
  accepting requests AFTER all engines are warmed, so every recall
  hits a loaded engine.

  Cost: ~10 s × N databases at startup. Acceptable for a server
  that runs for days; principle is "a database server should serve
  queries at steady-state latency, not cold-load latency."

  Discovered debugging architect-side recall timeouts in production
  on a 7-database cluster. Steady-state recall is 17–23 ms warm;
  cold-load was 11 s.

### Trade-off

- Startup is slower (linear in number of databases) but predictable.
- For very large fleets (100+ databases), consider parallel warm-up
  via `tokio::spawn_blocking` — filed for v0.9.x.

[0.8.8]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.8

## [0.8.7] — 2026-05-01

Critical bug fix. v0.8.5 fixed `/v1/health` to reflect openraft state
when active, but **the actual write/read paths still consulted the
legacy raft-lite leader detection**. On a healthy openraft cluster:

- `POST /v1/remember` returned `503 read-only: no leader elected`
- `POST /v1/recall` and `/v1/forget` failed similarly
- Wire-protocol writes (port 7437) returned `READONLY_NODE` errors
- Prometheus `yantrikdb_cluster_is_leader` gauge read 0 on the actual
  leader

Discovered by yantrikdb-agi within 10 min of v0.8.5 deploy when their
Lane B smoke test failed against the new cluster. Architect was able
to query `/v1/health` (got openraft view, "leader") and then post a
write to that same node and get back `503 no leader elected` —
exactly the kind of split-state bug that should never ship.

### Fixed

- New helper `cluster_state_view()` returns canonical
  `(node_id, role, term, leader, accepts_writes, healthy, raft_mode)`.
  Prefers openraft when `state.raft` is `Some`, falls back to legacy
  raft-lite. All client-facing endpoints now route through it:
  - `GET /v1/health` (already used openraft via direct read in v0.8.5;
    refactored to use helper)
  - `GET /v1/health/deep` (cluster_quorum check)
  - `GET /metrics` (`yantrikdb_cluster_term`, `_is_leader`, `_healthy`
    gauges; new `raft_mode` label)
  - `check_writable` gating `/v1/remember`, `/v1/forget`,
    `/v1/correct`, `/v1/admin/*` write endpoints
  - Wire-protocol server (port 7437) write-command rejection
- Error responses on follower writes now include `leader_node_id`,
  `leader_addr`, and `raft_mode` fields so clients can redirect.

### Notes for operators

If you deployed v0.8.5 or v0.8.6 in openraft mode, **upgrade
immediately**. Single-node deployments and raft-lite clusters
unaffected.

[0.8.7]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.7

## [0.8.6] — 2026-05-01

`cargo fmt` cleanup — no functional change.

### Fixed

- v0.8.5 was tagged on a commit where the cluster CLI subcommand
  handlers (added in v0.8.3) used one-line struct destructuring;
  rustfmt wants multi-line. CI flagged this on the v0.8.5 commit
  with a red badge. v0.8.6 retags on the post-fmt commit so the
  release page CI is clean.

Functionally identical to v0.8.5 (`/v1/health` reflects openraft
state when active). Use v0.8.6 instead of v0.8.5 if you care about
the release-page CI badge; binary behavior is the same.

[0.8.6]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.6

## [0.8.5] — 2026-05-01

`/v1/health` cleanup. Fixes a misleading status shown to operators
running `cluster.raft_mode = "openraft"`.

### Fixed

- `/v1/health` now reflects **openraft state** when openraft mode is
  active (`accepts_writes` true on the openraft leader, `term` and
  `leader` from `RaftMetrics`, etc.). Previously it always reported
  the legacy raft-lite view, which on a healthy openraft cluster
  shows `accepts_writes: false` (because the raft-lite layer never
  successfully forms a quorum once openraft is the real write path).

  Discovered during the issue #28 cluster reform when both nodes of
  a verified-working 2-voter openraft cluster reported
  `accepts_writes: false` to monitoring scrapes — false alarm,
  cluster was healthy.

  The legacy raft-lite view is still available in the response when
  raft-lite is the configured mode (no openraft assembly active).
  New `raft_mode` field on the cluster block disambiguates.

### Notes for operators

If you were filtering on `cluster.accepts_writes` in a Grafana panel,
no change needed — the value is now correct (true on openraft leader)
instead of always-false on healthy openraft clusters.

[0.8.5]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.5

## [0.8.4] — 2026-05-01

Critical hotfix. Without this, v0.8.2/v0.8.3 openraft clusters
become unresponsive under live multi-voter replication load.

### Fixed

- **#27** Resolved properly: `SplitRuntime` (RFC 009 §4 Layer 1
  CPU-isolated control-plane runtime) is back. The v0.8.2/v0.8.3
  workaround (cluster + HTTP sharing one tokio runtime) caused HTTP
  gateway starvation when openraft replication traffic ramped up
  on `cluster_port` 7440. Symptom: `.141` HTTP gateway became
  unresponsive within ~30 sec of a follower joining the cluster,
  cluster connections backing up to ~46 in the listen queue while
  HTTP requests timed out.

  Fix: converted `fn main()` from `#[tokio::main]` to manual sync
  main. The macro builds an outer Runtime, and any nested Runtime
  built inside async context panics on drop. Sync main owns the
  tokio runtime explicitly, runs `async_main()` via `block_on()`,
  and drops both the main runtime and `SplitRuntime` from sync
  context where blocking is permitted.

### Migration

- **From v0.8.3**: rolling restart on each node. No config changes.
  `SplitRuntime` re-engages automatically when `cluster.raft_mode =
  "openraft"` and `cluster.role = "voter"`.
- **From v0.8.2 / v0.8.1 / v0.7.x**: same as v0.8.3's migration
  procedure (see v0.8.3 changelog) — this hotfix only changes the
  runtime architecture, not the on-disk format or wire protocol.

[0.8.4]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.4

## [0.8.3] — 2026-05-01

Operator surface for openraft cluster mode. Without this, `v0.8.2`
deployments could only run *single-node* openraft because there was
no way to add additional voters without writing Rust against the
openraft API directly. **Closes the cluster mode story for v0.8.x.**

### Added

- **#24** Cluster membership HTTP API + CLI:
  - `POST /v1/cluster/initialize` — bootstrap a fresh openraft cluster
    on the seed node (one-time call per cluster).
  - `POST /v1/cluster/add-learner` — add a non-voting learner.
    Catches up via openraft snapshot transfer without participating
    in elections.
  - `POST /v1/cluster/promote-voter` — change voter set (promotes
    learners, demotes voters not in the new set).
  - `POST /v1/cluster/remove` — atomic remove. Refuses if removal
    would empty the voter set.
  - All endpoints require the cluster master token.
  - CLI subcommands mirror each: `yantrikdb cluster initialize-cluster`,
    `add-learner`, `wait-caught-up`, `promote-voter`, `remove-node`.

### Changed

- Removed the v0.8.2 auto-bootstrap heuristic ("node_id 1 or 2 =
  seed"). Replaced by explicit `cluster initialize-cluster` CLI.
  Operators on v0.8.2 with already-initialized membership are
  unaffected (openraft persists membership across restarts).

### Migration

For operators on a v0.7.x raft-lite cluster moving to v0.8.3
openraft, the procedure is now:

1. Generate cluster mTLS certs (CA + per-node cert+key).
2. Stop all nodes; install v0.8.3 binary; update each toml
   (`raft_mode = "openraft"` + `[cluster_tls]` + `dev_mode = true`
   for self-signed).
3. Wipe legacy `raft.json` (if present) — leave engine state intact.
4. Bring up the seed node. Run:
   ```
   yantrikdb cluster initialize-cluster --leader http://seed:7438 \
     --master-token "$YDB_CLUSTER_MASTER_TOKEN"
   ```
5. For each additional voter:
   - **Recommended**: pre-stage engine state via cold-tar from leader
     (~30 s for a 400 MB data dir vs ~30 min for openraft 0.9's
     `full_snapshot` over the wire).
   - Bring the new node up.
   - From the leader, run `yantrikdb cluster add-learner --node-id N
     --addr host:7440 --leader http://leader:7438`.
   - `yantrikdb cluster wait-caught-up --node-id N --leader ...`.
   - `yantrikdb cluster promote-voter --voters 2,N --leader ...`
     (include the leader's id in the final voter set).

The "cold-tar pre-stage" trick avoids openraft 0.9's slow
`full_snapshot` path. Issue #25 will replace it with chunked
`install_snapshot` in v0.9.0-beta — at which point pre-staging
becomes optional.

[0.8.3]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.3

## [0.8.2] — 2026-04-30

Patch release. Single critical fix:

### Fixed

- **#26** `cluster.raft_mode = "openraft"` panicked at startup because
  rustls 0.23+ requires a process-level `CryptoProvider` and neither
  `aws-lc-rs` nor `ring` features were enabled in `Cargo.toml`, and
  `install_default()` was never called. Single-node deployments
  (`raft_mode` unset / `disabled`) were unaffected since they skip
  rustls initialization entirely.

  Fix: enable `aws-lc-rs` feature on `rustls`, install the default
  provider at the very start of `fn main()`. Two-line change.

  Discovered 2026-04-30 during attempted cluster reform from
  raft-lite to openraft on a 2-voter homelab cluster.

### Notes for operators

- All v0.8.x releases prior to v0.8.2 are unsuitable for openraft
  mode. If you're on `raft_mode = "disabled"` (the default), you
  can upgrade at your own pace.
- If you tried to follow `docs/migration/v0.7_to_v0.8.md` Option B
  (raft-lite → openraft) on v0.8.0/v0.8.1 and hit a panic, this
  release is what unblocks it.

[0.8.2]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.2

## [0.8.1] — 2026-04-30

Operational patch. Fixes two bugs reported by `yantrikdb-agi` after
two blocking incidents on 2026-04-29 (21:00 UTC + 01:00 UTC) where
`/v1/recall` returned 500 for the entire `skill_substrate` namespace.

### Fixed

- **#19** `/v1/remember` and `/v1/remember/batch` could silently store
  rows with `embedding=NULL` when the embedder service hiccupped
  (timeout, transient ONNX runtime failure, etc.). The endpoint
  returned 200 with a normal `{rid: ...}` response while the row was
  effectively broken — and a single NULL-embedding row poisoned every
  subsequent `/v1/recall` on the namespace with
  `database error: Invalid column type Null at index: 1, name: embedding`.
  Both handlers now pre-embed in the server (using the existing
  `FastEmbedder` + cache wired in v0.8.0) before delegating to the
  engine. On embedder failure, the request returns 5xx synchronously
  so the caller can retry. The batch handler runs misses through one
  coalesced `embed_batch` call so concurrent ONNX-mutex acquisitions
  remain efficient. New Prometheus counter:
  `yantrikdb_embedder_failures_total{handler}`.

- **#20** No proactive surface for detecting `memories` rows with
  `embedding IS NULL`. Operators only discovered them when a recall
  failed. Added an hourly background healthcheck that counts
  NULL-embedding rows per tenant and emits
  `yantrikdb_null_embedding_count{tenant}` Prometheus gauge.
  Non-zero values trigger a `tracing::warn!` line with the SQL one-liner
  to remediate. Should be 0 in steady state on v0.8.1+ deployments;
  non-zero indicates pre-v0.8.1 stale data or a regression to flag.

### Deferred

- **#18** Leader affinity / preferred-leader pinning was investigated
  for v0.8.1 inclusion but blocked: openraft 0.9.24 (the version we
  pin) does not expose `transfer_leader` — that's a 0.10 feature.
  Bundling the openraft 0.10-alpha upgrade with leader-affinity is
  the right path; it lands in v0.9.0. See issue #18 comments for the
  blocker analysis. Operational workaround unchanged: manual CT
  bounce + the tuned watchdog (CHECK_INTERVAL=60, HANG_RECHECK=120,
  MAX_HANGS=5) keeps the bounce contained.

### Notes for operators

- Existing pre-v0.8.1 databases may have NULL-embedding rows from
  prior writes. After upgrading, watch `yantrikdb_null_embedding_count`
  for an hour; if non-zero, run
  `DELETE FROM memories WHERE embedding IS NULL` against the affected
  tenant DB to clean up. The fix prevents new ones; cleanup is one-shot.

[0.8.1]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.1

## [0.8.0] — 2026-04-29

Substrate-first release. Twelve RFC interface layers shipped as a single
batch, plus an embedder-cache perf win and two user-reported bug fixes.

### Highlights for operators

- **Embedder cache live** — repeat queries skip the ONNX forward pass.
  Recall p50 drops measurably on warm workloads (~10× on
  cache-hit-dominated traffic). No config required; the cache is on by
  default with a 10 000-entry capacity (~15 MB).
- **Two operator-reported bugs fixed**: encryption env var honored,
  snapshot endpoint usable in single-node mode.

### Added — substrate (interface + reference impl + tests; consumer
wiring deferred to follow-up PRs)

- **RFC 009 admission control** (saga #128, #129, #131):
  - `src/admission/cost.rs` — cost function
    (`top_k × E_expand`, `E_expand=5` default).
  - `src/admission/bucket.rs` — token bucket on monotonic clock,
    `startup_warm_fraction=0.25` default to mitigate rolling-restart
    thundering herd.
  - `src/admission/policy.rs` — `QuotaPolicy` + `QuotaScope`
    (Principal/Namespace/Global) + `PolicyResolver` trait,
    `PROVISIONAL_DEFAULTS` (rps=100, cost=1000/s, exp_cc=4) and
    `FALLBACK_DEFAULTS` (half-throughput) for control-DB outage.
  - `src/admission/registry.rs` — per-(scope, dimension) bucket
    registry, lazy-on-first-consume materialization, `ConsumeOutcome`
    in SHADOW mode.
  - `src/admission/circuit_breaker.rs` — pure state machine with 4
    triggers (term churn / active election / scheduling latency /
    sustained heartbeat lag), hysteresis, and anti-flapping.
    Pinned gauge values 0/1/2 for Closed/Open/HalfOpen.
  - `src/admission/deadlines.rs` — `RecallStage` enum + per-stage
    deadline budget. RFC defaults: total=5000ms, expansion=2000ms.
    `run_with_deadline_or_cancel` races future, deadline, and
    cancellation token.
  - `src/admission/retry_budget.rs` — per-tenant token-bucket retry
    budget + AWS full-jitter `Retry-After` computation.

- **RFC 011 forget — crypto-shred** (saga #146):
  - `src/forget/crypto_shred.rs` — `CryptoShredder` consumes
    `KeyProvider`. Destroys data + backup keys on tenant delete.
    Verified contracts: half-shred protection (all versions destroyed),
    purpose isolation (TLS / signing keys preserved), tenant
    isolation, idempotence.

- **RFC 012 backup/restore — restore command** (saga #148):
  - `src/restore/validate.rs` — `RestoreValidator`. Catches
    resurrect-tombstoned-data, wire major mismatch, HNSW model
    mismatch, checksum mismatch, missing content blobs.
  - `src/restore/exec.rs` — `RestoreExecutor` with three modes
    (`NewCluster` / `SingleTenant` / `WipeAndRestore`), atomic
    tmp+rename writes, marker file for crash recovery, sanitizes `/`
    and `\` in model names.

- **RFC 013-B shadow-index migration** (saga #152):
  - `src/index/hnsw/shadow.rs` — phase machine
    (Idle → Backfilling → DualRead → Cutover → Complete) with
    caught-up gate. `DualReadMerger` for score-normalized merge during
    DualRead. `MigrationStateStore` trait + in-memory ref impl.

- **RFC 014-B Auth/RBAC** (saga #153):
  - `src/auth/scopes.rs` — typed `Scope` bitset
    (Read/Write/Recall/Forget/Admin/TenantManagement). Pinned wire
    forms; CSV parse/format.
  - `src/auth/principal.rs` — `Principal` + `AuthOutcome`
    (Authenticated / Unauthenticated / Revoked / Expired —
    distinguished so audit operators can grep revoked-token usage).
  - `src/auth/provider.rs` — `AuthProvider` async trait + reference
    in-memory impl with hash-at-rest, revoke, expiry.
  - `src/auth/audit.rs` — `AuditEvent` + `AuditSink` trait +
    `InMemoryAuditSink` ring buffer.

- **RFC 014-C KeyProvider** (saga #154):
  - `src/key_provider/{mod,local}.rs` — `KeyProvider` async trait
    (object-safe), four `KeyPurpose`s (TenantDataEncryption /
    BackupBlobEncryption / ClusterTls / AuditSigning),
    `LocalKeyProvider` reference impl. Drop-zeroize on `KeyMaterial`
    (best-effort defense).

- **RFC 015-B-2 hybrid retrieval** (saga #157, LongMemEval lever):
  - `src/retrieval/bm25.rs` — `BM25Index` trait + `InMemoryBM25Index`
    reference impl with k1=1.2, b=0.75. TF saturation, longer-doc
    penalty, IDF rare-term reward verified by tests.
  - `src/retrieval/hybrid.rs` — `rrf_merge` pure function (Cormack
    2009 RRF, k=60 default).
  - `src/retrieval/rerank.rs` — `Reranker` async trait +
    `IdentityReranker`. Production ONNX cross-encoder backend
    deferred.

- **RFC 021 config versioning** (sagas #165, #166):
  - `src/config/versioned.rs` — `ConfigVersion`, `VersionedConfig<T>`,
    `ConfigDelta<T>`.
  - `src/config/live_reload.rs` — `Reloadable` trait with
    stale-delta detection and replay-safe ignore semantics.
  - `src/config/watch.rs` — `ConfigWatchSender` / `ConfigWatch` over
    `tokio::sync::watch` with latest-value semantics.
  - `src/config/tenant_overrides.rs` — `TenantConfigOverride` +
    `TenantConfigStore` trait with strict-monotonic upsert.

- **RFC 007 Socratic operators** (saga #102):
  - `src/socratic/{evidence,operator}.rs` — six typed operators
    (BinaryToConditional, GlobalToTemporal,
    PropositionToSourceComparison, OutcomeToUpstreamLevers stub,
    EntityDisambiguation, ContextCompletion). Operator selection is
    deterministic + graph-grounded; LLM (if used) only paraphrases.

### Added — embedder cache wiring (commit `e52228e`)

- `FastEmbedder` now consults `EmbeddingCache` before acquiring the
  ONNX `Mutex`. Cache hits skip both the lock and the forward pass.
- `embed_batch` partitions inputs into hits and misses, then runs ONE
  batched ONNX call for all misses (single mutex acquisition).
- `cache_hits()`, `cache_misses()`, `cache_hit_rate()` accessors for
  future Prometheus wiring.
- Cache key includes the model version so RFC 013-B shadow migrations
  auto-invalidate on model change.

### Fixed

- **#6** `YANTRIKDB_ENCRYPTION_KEY_HEX` env var was silently ignored;
  encryption only enabled when key was in the TOML `[encryption]
  key_hex` field. Now `EncryptionSection::resolve_key()` checks the
  env var first (priority 0), TOML second. Operators can now run with
  `-e YANTRIKDB_ENCRYPTION_KEY_HEX=<hex>` per the documented setup.
- **#7** `POST /v1/admin/snapshot` rejected requests in single-node
  mode with `snapshot requires cluster master token` because the
  cluster master token doesn't exist outside cluster mode. Now
  accepts EITHER the cluster master token (cluster mode) OR a
  per-database token authenticating the SAME database named in the
  request body (any mode).

### Tests

- 739 unit tests + 11 integration test binaries — all green.
- ~310 new tests added across the substrate. Each substrate ships
  contract pin-tests that survive backend swaps (e.g.
  `InMemoryBM25Index` tests will validate the future `tantivy`-backed
  impl verbatim).

### Migration

See `docs/migration/v0.7_to_v0.8.md`.

Most users have no client-visible breaking changes. Cluster operators
migrating from raft-lite (Phase 1) to openraft must reform the
cluster — see migration doc for procedure.

[0.8.0]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.0
