# Roadmap

This document captures the public roadmap for `yantrikdb-server` — what
shipped, what's coming, and what we're explicitly *not* building.

The aim is open-source-excellence-first: be the best open source
cognitive memory database in the world. Distribution comes from
undeniable quality.

## Recent — shipped

### v0.8.1 (2026-04-30)
- Operational patch. Two writer-side bugs (#19 NULL-embedding silent
  storage, #20 no operator-visible NULL detection) reported by an
  external user and fixed within 24 h.
- Hourly `null_embedding_count` Prometheus gauge for early detection.
- New migration script `scripts/migrate-v0.8.1-cleanup-null-embeddings.sh`
  for operators upgrading from v0.7.x or v0.8.0.

### v0.8.0 (2026-04-29) — substrate-first release
- 12 RFC interface layers shipped as a single batch:
  - **RFC 009** admission control (cost function, token bucket,
    circuit breaker, deadlines, retry budget).
  - **RFC 010** mutation grammar + commit log + **full openraft
    integration** (replaces raft-lite).
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
  Recall p50 drops measurably on warm workloads.
- Two operator-reported fixes (#6 encryption env var; #7 single-node
  snapshot endpoint).
- 739 unit tests + 11 integration test binaries — all green.

## Next — v0.9.0-beta target

`v0.9.0-beta` is the planned **first beta** release. The focus is
**cluster operational maturity**: the substrate is correct (RFC 010
proven via leader-kill failover tests), but the operator surface
around it has gaps that surfaced in real-world ops on 2026-04-29
during a routine voter migration.

The four critical-path issues for `v0.9.0-beta` are tracked on the
issue tracker and linked here once filed:

1. **Cluster reform tool: automated raft-lite → openraft migration.**
   One-shot CLI replacing the current 7-step manual procedure.
2. **Upgrade openraft 0.9 → 0.10-alpha + wire `transfer_leader`.**
   Unblocks issue #18 (preferred-leader pinning) and the manual
   leader-selection ask.
3. **Cluster membership API CLI:** `add-learner`, `promote`,
   `transfer-leader`, `remove` subcommands. Replaces the
   "edit toml + sed + restart" dance.
4. **Chunked snapshot streaming with parallel applier per engine.**
   Replaces the current single-threaded full-snapshot path that takes
   ~30 min for a 290 MB engine snapshot.

Operator visibility — paired but separately tracked:

5. `/v1/cluster/raft` endpoint: leader, term, last_applied,
   per-peer log_index + snapshot_progress.
6. `yantrikdb cluster raft-status` CLI + `\raft` yql.
7. Prometheus `raft_*` metrics + Grafana dashboard JSON.

Defended correctness — already on roadmap, deferred from RFC 010 PR-4:

8. Linearizable reads via `Raft::ensure_linearizable()`.
9. mTLS production gate at `server.rs` assembly site (RFC 014-A).

Known sharp edges to address before tagging `-beta`:

- Lazy engine load (don't block heartbeats during HNSW reload at
  startup) — removes the 25–60 s unresponsiveness window.
- "Read-not-ready" mode for followers with incomplete snapshot —
  prevents partial-state recall returning wrong results.

## Beyond v0.9.0-beta — toward v1.0.0 GA

Three deliverables separate `-beta` from `1.0.0`:

- **Jepsen suite**: skeleton at M3, full suite earlier in the
  6-month window. A clustering DB without a published Jepsen run
  is alpha-grade by industry convention; a clean Jepsen run is the
  table-stakes for taking the `-beta` tag seriously.
- **LongMemEval benchmark vs Zep / Memento / Mastra**: the
  competitive lever per the AGI substrate paper. RFC 015-B-2
  hybrid retrieval is in place; the benchmark publication ties
  it to numbers.
- **Production deployment story** (RFC 016): Helm chart, Terraform
  module, runbooks, chaos test scripts. "Deploy this in production"
  needs more than `docker compose up`.

## Explicit non-goals

These are deliberately out of scope and will not ship without a
deliberate roadmap revision:

- CRDT-as-primary store. Raft-replicated state machine is the
  primary; CRDTs (where used at all) are scoped to specific
  conflict-resolution sub-paths.
- Multi-region replication. Single-region clustering only at
  `v1.0.0`. Multi-region is a separate major cycle.
- Hosted SaaS provisioning. We ship the open-source artifact;
  hosting is out of scope.
- A paid tier or enterprise edition. Single edition; everything
  in this repo is the product.

## Reporting issues / contributing

Open an issue at
<https://github.com/yantrikos/yantrikdb-server/issues>. Real-world
incidents (e.g. the 2026-04-29 cluster voter-add incident that drove
items 1–4 above) are especially valuable — they tell us where the
operator surface is rough.
