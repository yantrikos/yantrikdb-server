# RFC 032 — HTTP route parity for the six embedded-only tool families

**Status:** Accepted (Option 1) · **Issue:** #83 (+ sibling #82 `correct`) · **Depends on:** RFC 028 (YRP replication)

## Decision

**Option 1 — reads-first + honest writes.** Ship now:
- **All read actions** for all six families (zero divergence risk).
- **Node-local writes** wired with a documented node-local semantic:
  `conversation` (record/clear), `trigger` lifecycle (acknowledge/deliver/act/
  dismiss/prune).
- **Deferred global writes** — `task` (add/update/delete), `category`
  (learn/reset), `procedure` (learn/reinforce), **and `correct` (#82)** — return
  a clear **501** (`not_yet_available_over_http`, not a bare 404), because they
  need a replicated mutation path that does not exist yet. NOTE `correct` maps to
  `MemoryMutation::UpdateMemoryPatch`, which is `NotYetWired` in the applier — so
  it is *not* a freebie; it joins the deferred set, tracked for the follow-up
  that wires the replicated grammar.
- The **parity table** ships regardless.

## The gap (issue #83)

Six MCP tool families have engine functionality but **no HTTP route** — reachable
in `embedded` (in-process) mode, invisible over the Axum gateway that
`mode:"http"` / shared-cluster clients use: `category`, `conversation`,
`procedure`, `task`, `temporal`, `trigger`. (Sibling #82: `correct` is a
seventh, a clear one-route gap.) An HTTP client that assumes MCP-surface parity
gets a bare 404.

## The finding that shapes the design: reads are free, writes are not

Reconnaissance (engine crate + gateway) established two things:

1. **Every family has a real Rust `pub fn` on `YantrikDB`** — none are
   Python-only. So all are *technically* wireable with the existing direct-call
   handler pattern (`resolve_engine` → `engine.method()` → JSON), the same shape
   as `skill_get`.
2. **But only *memory* operations replicate.** YRP's apply path
   (`yrp/engine_sink.rs`) applies `MemoryMutation` (`UpsertMemory` /
   `TombstoneMemory`) only; a follower reconstructs its engine by **replaying
   the commit log**. Replication happens because a write went through
   `execute_cmd`/propose — **not** because of which table it touched. These six
   families write to *other* tables (`tasks`, `conversation_turns`,
   `substitution_categories`, `trigger_log`, procedural rows) with **no Command
   variant and no apply path**.

**Consequence:** wiring a *read* is safe on any node. Wiring a *write* as a
direct engine call makes it **land only on the node that served the request and
silently not replicate** — the same silent-divergence class as the RFC 031 pack
mount decision and the SDK follower-write drop. Exposing that over HTTP is
arguably *worse* than a clean 404. This is why "wire for parity" is not a reflex
job — the write path needs a real decision.

## Per-family analysis

| Family | Read actions (safe to wire now) | Write actions | Write replication story |
|---|---|---|---|
| **temporal** | `stale`, `upcoming`, `as_of` | — | **All reads.** Fully safe. (`as_of` = `recall_as_of`, needs an `engine.embed()` step + engine ≥0.12 — we're on 0.13.1.) |
| **category** | `list`, `members` | `learn`, `reset` | writes to `substitution_categories/_members` — **not replicated**. Global-scope data (affects conflict detection cluster-wide) → *should* replicate. |
| **task** | `get`, `list` | `add`, `update`, `delete` | writes to `tasks` — **not replicated**. Meant to be durable+global → *should* replicate. |
| **procedure** | `surface` | `learn`, `reinforce` | procedural rows are memory-shaped (embedding+rid) but written via a **direct** engine call, not the Remember command → **not replicated** as-is. Could ride the memory path. |
| **conversation** | `recent` | `record`, `clear` | bounded per-namespace ring buffer of recent turns — **legitimately node-local** (ephemeral recent-context), so "not replicated" is a *defensible documented semantic*, not a bug. |
| **trigger** | `pending`, `history` | `acknowledge`,`deliver`,`act`,`dismiss`,`prune` | status-only updates on `trigger_log`. Triggers are **generated per-node by the maintenance/cognition tick** (generation is NOT in this tool surface), so lifecycle state is naturally node-local. `prune` is bounded-backlog eviction. Node-local is defensible. |
| **correct** (#82) | — | `correct` | in-place memory correction (`correct_with_embedding`). Memory-backed → **belongs on the replicated memory path** (like Remember/Forget), not a direct call. |

**Key**: none of the six MCP surfaces reach into consolidation *internals*
(Pranab's original worry about `temporal`/`trigger`) — trigger *generation*
stays inside maintenance and is out of scope. The real axis is not
"maintenance-touching" but **"does this write need to replicate?"** — and that
splits the families into *global* (category, task, procedure, correct → should
replicate) vs *node-local* (conversation, trigger-lifecycle → defensibly not).

## The decision (write strategy) — options

- **Option 1 — Reads-first + honest writes (incremental, recommended).**
  Ship now: **all read actions** for all six families (zero divergence risk,
  unblocks the reporter's HTTP read use-cases immediately) + **`correct`**
  routed through the replicated memory path (closes #82 correctly). Wire the
  **node-local writes** (`conversation`, `trigger` lifecycle) with an explicit
  documented "node-local, not replicated" semantic. **Defer** the *global*
  writes (`task`, `category`, `procedure.learn`) to a follow-up that adds their
  replicated mutation grammar — do NOT ship them as non-replicating direct
  calls. Result: real value fast, nothing half-baked, the hard part scoped.
- **Option 2 — Full replicated grammar now.** Add new `MemoryMutation`
  variants + follower apply paths for task/category/procedure/conversation/
  trigger writes so every action is cluster-correct. Complete, but a large
  multi-family engine+server change (grammar + apply + determinism + tests) —
  its own RFC-sized effort.
- **Option 3 — Wire everything as direct calls, document the caveat.** Fastest
  to nominal "parity," but ships non-replicating writes = silent divergence.
  **Not recommended** — violates the cluster-correctness bar.

## Parity table (the documentation half of the ask — ships regardless)

A canonical `docs/operations/http-embedded-parity.md` table: for every MCP tool
family, its HTTP status (wired / read-only-over-http / node-local /
embedded-only-by-design) so `mode:"http"` clients never discover a gap via 404.
This is produced no matter which option is chosen.

## Review hardening (adversarial code review — folded in)

- **F1 — blocking on the reactor.** The two recall-equivalent reads
  (`temporal.as_of`, `procedure.surface`) run embed + HNSW; they now run that
  work inside `spawn_blocking` so a slow call can't park a tokio worker. The
  light single-`SELECT` reads stay direct calls, matching the existing read
  handlers (`skill_get`/`stats`/`conflicts`), which are all direct.
- **F2 — admission bypass.** `as_of`/`surface` are recall-equivalent, so they
  now take the same RFC 009 admission as `/v1/recall`: `check_top_k`
  (hard-cap) + `acquire_recall_permits` before HNSW. No unbounded voter load.
- **F3 — embedding parity.** Both accept a client-supplied `query_embedding`
  and fall back client → server-pool → engine embedder, so they work in
  BYO-embedding / server-pool deployments, not only where the engine carries a
  runtime embedder.
- **F4 — node-local write safety, proven not asserted.** `conversation_turns`
  and `trigger_log` are never shipped between nodes: `serve_backfill` returns
  only committed `MemoryMutation`s and `engine_sink` applies only
  `MemoryMutation`, so neither table is in the replicated grammar. There is no
  leader-authoritative copy to diverge from. The proof + a "revisit if this
  changes" note live at the handlers.
- **F5 — unbounded reads.** Every read `limit` is hard-capped at
  `MAX_READ_LIMIT` (1000) and temporal `days` at `MAX_TEMPORAL_DAYS`, so one
  request can't force a giant scan (parallel to recall's `hard_top_k_cap`).

## Invariants

1. **No non-replicating write is exposed over HTTP without an explicit,
   documented node-local semantic.** A global-scoped write either replicates or
   is not wired yet.
2. Reads are always safe to expose (they reflect local state; they never
   diverge).
3. `check_writable` still gates every write handler (follower → 503 leader-hint).
