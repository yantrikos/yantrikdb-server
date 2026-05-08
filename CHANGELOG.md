# Changelog

All notable changes to `yantrikdb-server` are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
