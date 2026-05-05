# Roadmap

This document captures the public roadmap for `yantrikdb-server` — what
shipped, what's coming, and what we're explicitly *not* building.

The aim is open-source-excellence-first: be the best open source
cognitive memory database in the world. Distribution comes from
undeniable quality.

## Recent — shipped

### v0.8.11 (2026-05-02) — first-class skill primitives + follower HNSW backfill
- New `/v1/skills/{define, get, search, outcome, forget}` endpoints with
  strict shape validation (skill_id format, applies_to entry regex catches
  hyphen-vs-underscore drift, skill_type enum, duplicate-on-conflict 409).
- Follower HNSW piecewise per-row backfill replaces the prior batch-end
  rebuild. Engine 0.6.5 promotes `insert_vector` and `encrypt_embedding`
  to `pub` so encrypted clusters can replicate.
- Acceptance gate `define → immediate search returns the skill with no lag`
  passes on the leader. Follower-side gate is currently un-empirically-
  verifiable until RFC 010 PR-6 (see *Next*).

### v0.8.7–v0.8.10 (2026-04-30 → 2026-05-01) — concurrency + cluster polish
- v0.8.10: fix leaked inflight counter on future cancellation.
- v0.8.9: concurrent recall via engine read pool + Mutex removal.
- v0.8.8: eager engine warm-up at startup (removes 25–60 s
  unresponsiveness window after SIGHUP / restart).
- v0.8.7: client-facing leader detection now openraft-aware (critical fix
  for write paths hitting the wrong node).

### v0.8.2–v0.8.6 (2026-04-29 → 2026-04-30) — openraft hardening
- v0.8.5: `/v1/health` reflects openraft state when active.
- v0.8.4: proper SplitRuntime fix (#27 resolved).
- v0.8.3: openraft membership API + CLI (#24).
- v0.8.2: openraft single-node auto-bootstrap; fix startup panics
  (#26, #27, +TLS feature mismatch).

### v0.8.1 (2026-04-30) — operational patch
- Two writer-side bugs (#19 NULL-embedding silent storage, #20 no
  operator-visible NULL detection) reported by an external user and fixed
  within 24 h.
- Hourly `null_embedding_count` Prometheus gauge.
- Migration script `scripts/migrate-v0.8.1-cleanup-null-embeddings.sh`.

### v0.8.0 (2026-04-29) — substrate-first release
- 12 RFC interface layers shipped as a single batch:
  - **RFC 009** admission control (cost function, token bucket,
    circuit breaker, deadlines, retry budget).
  - **RFC 010** mutation grammar + commit log + openraft assembly +
    retention contract (Phases A/B/C).
  - **RFC 011** crypto-shred forget.
  - **RFC 012** restore validator + executor.
  - **RFC 013-A** HNSW lifecycle Phase 1.
  - **RFC 013-B** shadow-index migration phase machine.
  - **RFC 014-B** Auth/RBAC scaffolding.
  - **RFC 014-C** KeyProvider trait + LocalKeyProvider reference impl.
  - **RFC 015-B-2** hybrid retrieval (BM25, RRF, reranker).
  - **RFC 017-A** wire/version framework — wire format pinned.
  - **RFC 021** config versioning + live reload + tenant overrides.
  - **RFC 007** Socratic operators (six typed operators,
    deterministic + graph-grounded).
- Embedder cache live — repeat queries skip the ONNX forward pass.
- 739 unit tests + 11 integration test binaries — all green.

## Next — v0.8.12 through v0.8.14: RFC 010 PR-6 (replication actually working)

**Status correction (2026-05-02):** the v0.8.0 release shipped RFC 010's
trait surface, mutation grammar, openraft *assembly*, and the retention
contract — but the HTTP write path was never migrated to call
`MutationCommitter`. The handlers (`Command::Remember`, `Forget`,
`Correct`, `Relate`, `IngestClaim`, `RememberBatch`) call
`engine.record()` directly, so writes land in the leader's local SQLite
but never get logged to the commit log, never replicated via openraft,
never apply on followers. The cluster has been operating in **cosmetic
openraft mode**: assembled, healthy in `/v1/health`, doing zero useful
work because handlers never feed it.

This is the root cause of every cross-cluster persistence symptom users
have reported (memories written-but-not-recallable; skills not visible
on follower; tokens not propagating).

**RFC 010 PR-6** (`docs/rfcs/rfc_010_pr6_write_path_migration.md`,
saga Epic #53) addresses this as 9 sub-PRs across three patch releases:

| Release | PRs | Theme |
|---|---|---|
| **v0.8.12** | 6.1, 6.2, 6.3 | Submitter/Applier trait split + deterministic mutations + per-tenant commit log layout. No cluster behavior change yet. |
| **v0.8.13** | 6.4, 6.5, 6.6 | Handler migration + boot invariants + HTTP error mapping. End-to-end RYW: write to .140 leader, recall on .141 follower within 5s p99. |
| **v0.8.14** | 6.7, 6.8, 6.9 | Per-tenant chunked snapshot + backfill admin tool + extended `/v1/health`. Cosmetic mode becomes structurally impossible. |

Architecture (locked via brainstorm 0e216e8c, 3 rounds, 2026-05-02):
- `Submitter` / `Applier` split — `LocalApplier` is the only path that
  mutates engine state, runs on both single-node and cluster.
- Deterministic mutations — leader pre-computes embedding + entities +
  timestamps; followers apply byte-identically (no embedder version skew
  possible).
- Per-tenant `memory_commit_log` co-located inside each tenant's
  `yantrik.db` (Option D).
- Synchronous apply with `wait_for_apply=true` intrinsic; 30 s timeout
  maps to HTTP 503 with `op_id` for idempotent retry.
- Chunked per-tenant snapshot via openraft `generic-snapshot-data`.
- Boot invariants reject `raft_mode=openraft` + non-Raft handler at
  startup.
- HTTP error mapping: 307 NotLeader, 409 OpIdCollision, 503
  CommitTimeout-with-op_id, 426 wire mismatch, 501 NotYetImplemented.
- Maintenance-window upgrade (drain → stop → migrate → restart →
  verify, ~10–15 min wall time). No rolling upgrade across the PR-6
  boundary because write semantics changes.

## After PR-6: v0.8.15 through v0.8.19 — engine concurrency

The v0.8.x concurrency series (originally targeted at v0.8.12–v0.8.16,
shifted by 3 to make room for PR-6) closes the recall p99 spikes
observed under sustained concurrent load.

Architecture: A″/B-lite/RCU hybrid (NOT naive Mutex → RwLock); locked via
gpt-5.5 redteam brainstorm session a06cdaaa, 2026-05-01.

| Release | Theme |
|---|---|
| **v0.8.15** | Per-namespace state DashMap (was v0.8.12). |
| **v0.8.16** | ArcSwap read views WITH tombstone validation — non-negotiable correctness, forget must hide. |
| **v0.8.17** | Bounded SQLite read pool + serialized writer. |
| **v0.8.18** | Embedder semaphore + ONNX thread tuning. |
| **v0.8.19** | Beta gate benchmark CI: recall p95<100ms, p99<200ms; write p99<750ms; zero 10s timeouts under concurrency 32; forget-hides-within-100ms invariant. |

## After concurrency: v0.8.20 onward — RFC 023 epistemic control plane

Eight v0.8.x patch releases shipping the primitives surfaced in
Brainstorm 3 (session 44195988, 2026-05-01) — the category shift from
"vector database for agents" to "epistemic control plane for LLMs."

Per-release primitive: provenance, supersede chain, recall perimeter,
action-conditioned constraint recall, scoped negative evidence,
memory-of-missing-memory, quarantine, latent contradiction awareness.

See saga Epic #52 for full sequencing.

## v0.9.0-beta target

`v0.9.0-beta` is the planned **first beta** release. Originally targeted
for the cluster operator surface gaps surfaced during the 2026-04-29
voter-migration incident; rescoped after PR-6 diagnosis revealed that
the substrate itself needed completing first.

After PR-6 + concurrency lands, the four critical-path issues remaining
for `-beta`:

1. **#22 — Cluster reform tool: automated raft-lite → openraft
   migration.** One-shot CLI replacing the 7-step manual procedure.
2. **#23 — Upgrade openraft 0.9 → 0.10 + wire `transfer_leader`.**
   Unblocks #18 (preferred-leader pinning) and manual leader-selection.
3. **#24 — Cluster membership API CLI:** `add-learner`, `promote`,
   `transfer-leader`, `remove` subcommands. Replaces the "edit toml +
   sed + restart" dance. Partly shipped in v0.8.3.
4. **#25 — Chunked snapshot streaming with parallel applier per
   engine.** Now subsumed by RFC 010 PR-6.7 (per-tenant chunked
   snapshot via openraft `generic-snapshot-data`).

Operator visibility — paired but separately tracked:

5. `/v1/cluster/raft` endpoint: leader, term, last_applied, per-peer
   log_index + snapshot_progress. (Partly shipped in v0.8.3 and v0.8.5;
   PR 6.9 fills remaining gaps.)
6. `yantrikdb cluster raft-status` CLI + `\raft` yql.
7. Prometheus `raft_*` metrics + Grafana dashboard JSON.

Defended correctness — deferred from RFC 010 PR-4:

8. Linearizable reads via `Raft::ensure_linearizable()`.
9. mTLS production gate at `server.rs` assembly site (RFC 014-A).

Known sharp edges to address before tagging `-beta`:

- "Read-not-ready" mode for followers with incomplete snapshot —
  prevents partial-state recall returning wrong results.

## Beyond v0.9.0-beta — toward v1.0.0 GA

Three deliverables separate `-beta` from `1.0.0`:

- **Jepsen suite**: skeleton at M3, full suite earlier in the 6-month
  window. A clustering DB without a published Jepsen run is alpha-grade
  by industry convention; a clean Jepsen run is table-stakes for taking
  the `-beta` tag seriously. PR-6 makes Jepsen meaningful — before, the
  cluster wasn't replicating at all.
- **LongMemEval benchmark vs Zep / Memento / Mastra**: the competitive
  lever per the AGI substrate paper. RFC 015-B-2 hybrid retrieval is in
  place; the benchmark publication ties it to numbers.
- **Production deployment story** (RFC 016): Helm chart, Terraform
  module, runbooks, chaos test scripts. "Deploy this in production"
  needs more than `docker compose up`.

## In-flight RFCs

- **RFC 010 PR-6** (`docs/rfcs/rfc_010_pr6_write_path_migration.md`,
  Draft, awaiting Pranab + architect sign-off) — the work above.
- **RFC 022** (`docs/rfcs/rfc_022_skill_substrate_and_ryw.md`, Draft) —
  §1 (skill API endpoints) + §2 (RYW-on-HNSW follower piecewise insert)
  shipped in v0.8.11. §3 (namespace_schema + `/v1/lookup`) and §4
  (where-clause prefilter + query planner) **paused** until PR-6 lands.
- **RFC 023** (saga Epic #52, no spec doc yet) — epistemic control
  plane primitives for v0.8.20+.

## Explicitly deferred (post v1.0)

- **RFC 007** Meta-cognitive primitives (5-layer schema, scenario_override,
  rule-edge whitelist, suggest_levers). Saga Epic #34 — all tasks
  blocked. Reactivate post-GA if research demand surfaces.
- **RFC 008** Warrant Flow & Reflexive Epistemic Control. Saga Epic #35
  — all tasks blocked. Reactivate post-GA if research demand surfaces.

## Explicit non-goals

These are deliberately out of scope and will not ship without a
deliberate roadmap revision:

- CRDT-as-primary store. Raft-replicated state machine is the primary;
  CRDTs (where used at all) are scoped to specific conflict-resolution
  sub-paths.
- Multi-region replication. Single-region clustering only at `v1.0.0`.
  Multi-region is a separate major cycle.
- Hosted SaaS provisioning. We ship the open-source artifact; hosting
  is out of scope.
- A paid tier or enterprise edition. Single edition; everything in
  this repo is the product.

## Reporting issues / contributing

Open an issue at
<https://github.com/yantrikos/yantrikdb-server/issues>. Real-world
incidents (e.g. the 2026-04-29 cluster voter-add incident that drove
items 1–4 above; the 2026-05-02 cosmetic-openraft diagnosis that
drove RFC 010 PR-6) are especially valuable — they tell us where the
operator surface is rough.
