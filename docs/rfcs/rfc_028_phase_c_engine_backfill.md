# RFC 028 addendum — Phase C: engine backfill for beyond-GC stragglers

Implementation design for the code-level "Phase C engine-checkpoint
transfer" milestone (referenced in `src/yrp/driver.rs`,
`src/yrp/engine_sink.rs`). Normative parent: RFC 028 §6.

## The hole

YRP log compaction (`ReplicaCore::compact`) drops **protocol-log**
entries below a base index. When a follower falls below that base, the
leader sends `InstallSnapshot` carrying only protocol state
(`claims` / `active` / frontier `last`). The driver's
`Effect::InstallState` handler fast-forwards its `applied` / `dispatched`
counters to the snapshot frontier **without applying the engine
mutations** for the compacted range — a permanent hole in the
follower's engine state (memories/embeddings/entities). This is why
compaction ships `compact_after = None` (disabled) by default.

## Decision (codex-consulted 2026-07-19): log-replay backfill first

YRP compaction GCs **only the protocol log**. The per-tenant
`commit_log.sqlite` — full materialized mutations, embeddings inside —
is written by the apply sink and is **never GC'd**; the leader retains
every committed mutation. The `OutcomeStore` (`yrp_apply.sqlite`) maps
`yrp_index → (tenant_id, op_id, tenant_log_index, rid, …)`.

So a beyond-GC straggler can be healed by **replaying the retained
commit-log mutations for the gap through the existing deterministic
apply sink** — no binary checkpoint, no new crash-consistency
machinery. This is the openraft model (snapshot = replay committed log),
made GC-safe because the commit-log is the retained source of truth.

- **Slice A (this work): log-replay backfill.** Closes the correctness
  hole; lets protocol-log compaction run default-on. Does NOT bound
  total storage (commit-log still grows).
- **Slice B (tracked follow-up): binary engine checkpoint.** WAL-
  checkpoint + copy the per-tenant sqlite at a quiesced global frontier,
  extended `SnapshotManifest` (reuse `crate::backup::SnapshotManifest`),
  out-of-band blob stream, crash-safe online swap, HNSW rebuild. This is
  what bounds storage (enables commit-log GC). Separately specified.

## Slice A mechanism

### Serve (leader / any complete node)
`POST /v1/yrp/backfill` — body `{cluster_id, from_index, to_index}`
(cluster-secret bearer, like `/v1/yrp/msg`). The server, for each
`yrp_index` in `(from, to]`, joins `OutcomeStore` → `commit_log` and
returns `[(yrp_index, YrpOp-bytes)]`, contiguous and range-complete.
**Source completeness invariant:** refuse (or truncate honestly) any
range the node cannot fully cover from its own retained history —
never serve a partial range that would leave a hole the requester
believes filled.

### Request + apply (backfilling follower)
On `Effect::InstallState { last_index }`, the driver records an **engine
gap** `[durable_applied+1, last_index]` instead of blindly fast-
forwarding the durable marker. A backfill task (runtime layer) pulls the
range from the current leader in contiguous batches and feeds each
`(yrp_index, LogEntry{Payload::Op})` to the **existing apply worker** —
which does commit-log append + engine apply + marker advance, already
atomic and idempotent per `(tenant, index)`. The follower is "engine-
caught-up" only when the durable marker reaches `last_index`. Backfill
is durably resumable: a crash mid-backfill re-derives the gap from the
persisted marker vs. protocol frontier (never trust HTTP completion).

### Eligibility gate (codex pitfall 1 — the safety crux)
A node whose durable engine marker `applied < commit` (protocol frontier
it has adopted) is **engine-incomplete** and MUST be:
- ineligible to serve reads / linearizable barriers — already true by
  construction: the read barrier waits for `applied ≥ noop_index`, so an
  incomplete node's barrier never resolves (fails closed, never serves
  stale);
- ineligible to campaign — the driver suppresses `on_election_timeout`
  while engine-incomplete (same mechanism that makes a witness never
  campaign), so an incomplete node cannot win leadership and then fail
  to serve engine history to the next straggler;
- reported honestly on `/v1/health` (`engine_incomplete: true` +
  the gap).

### Duplicate-index integrity (codex pitfall 2)
Backfilled entries carry the original `op_id`; the sink's commit-log
append is idempotent on `(tenant, op_id)` and fail-stops on an
op_id-with-divergent-payload. So a duplicate `yrp_index` cannot conceal
a divergent mutation — the existing collision guard covers it.

## Invariants to test before enabling compaction by default
Per codex: crash mid-backfill, interrupted stream, leader change during
backfill, repeated snapshots, and truncation must all establish that
**no durable marker advances past an unapplied mutation**. The chaos
scenario `straggler_beyond_gc_*` is extended to write a memory that
lands in the COMPACTED range, then assert the revived straggler can
RECALL it (not just dedupe its claim) — the direct proof the engine
hole is closed.

## Storage caveat (codex pitfall 3)
Slice A shifts the bound from protocol-log growth to commit-log growth;
admission control + disk monitoring remain mandatory until Slice B's
commit-log GC lands.
