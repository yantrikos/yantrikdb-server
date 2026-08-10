# HTTP vs embedded — tool parity

Which MCP tool families are reachable over the **HTTP gateway** (`mode:"http"`,
shared-cluster clients) vs **embedded** (in-process, single-node) only, and why.
Tracks issue #83. Status legend:

- **wired** — full parity over HTTP.
- **read-only over HTTP** — reads are wired; writes are *deferred* (below).
- **node-local** — wired over HTTP, but the state is **per-node and not
  replicated** by design; direct your client to a consistent node for a coherent
  view.
- **deferred (501)** — the route exists and returns a clear `501
  not_yet_available_over_http` (never a bare 404); the operation is a
  cluster-global write with no replicated mutation path yet (RFC 032).

## Why writes differ from reads

Only **memory** operations replicate through YRP consensus (`MemoryMutation`,
replayed from the commit log). A write to any other engine table (`tasks`,
`conversation_turns`, `substitution_categories`, `trigger_log`, …) has no
replication path, so exposing it as a direct engine call would **silently
diverge** across the cluster. Reads never diverge, so they are always safe to
expose. See RFC 032.

## Parity table

| MCP family | HTTP status | Routes |
|---|---|---|
| remember, recall, forget, memory, graph, relate, conflict, skill, personality, stats, session, think, gaps | **wired** | (pre-existing) |
| **temporal** | **wired** (all reads) | `GET /v1/temporal/stale`, `GET /v1/temporal/upcoming`, `POST /v1/temporal/as_of` |
| **category** | **read-only over HTTP** | reads: `GET /v1/categories`, `GET /v1/categories/{name}/members` · deferred: `learn`, `reset` |
| **task** | **read-only over HTTP** | reads: `GET /v1/tasks`, `GET /v1/tasks/{id}` · deferred: `POST /v1/tasks` (add), `PATCH`/`DELETE /v1/tasks/{id}` |
| **procedure** | **read-only over HTTP** | read: `POST /v1/procedures/surface` · deferred: `POST /v1/procedures` (learn), `POST /v1/procedures/{rid}/reinforce` |
| **conversation** | **node-local** | read: `GET /v1/conversation/{ns}/recent` · node-local writes: `POST`/`DELETE /v1/conversation/{ns}` |
| **trigger** | **node-local** | reads: `GET /v1/triggers`, `GET /v1/triggers/history` · node-local: `POST /v1/triggers/{id}/{acknowledge\|deliver\|act\|dismiss}`, `POST /v1/triggers/prune` |
| **correct** (#82) | **deferred (501)** | `POST /v1/correct` — maps to `UpdateMemoryPatch`, not yet wired in the applier |

## Node-local families — the semantic

- **conversation** — a bounded ring buffer of recent turns per namespace;
  ephemeral working context. Recording on the serving node writes that node's
  buffer only.
- **trigger** — triggers are *generated per-node* by the maintenance/cognition
  tick, so lifecycle (acknowledge/deliver/act/dismiss) and pruning are naturally
  node-local: you act on a trigger on the node that holds it. Trigger
  *generation* is not part of the tool surface (it stays inside maintenance).

## Deferred writes — the path forward

`task`, `category`, `procedure.learn`, and `correct` become fully cluster-correct
over HTTP once they have replicated mutation grammar (new `MemoryMutation`
variants + follower apply paths). Until then they return `501` with
`deferred: true` rather than silently diverging. Tracked as the RFC 032
follow-up.
