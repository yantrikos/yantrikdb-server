# RFC 029 — Control-plane replication (YRP)

**Status:** Draft · **Depends on:** RFC 028 (YRP) · **Codex-consulted:** yes (verdict A)

## The gap

YRP replicates the **data plane** — memory mutations per tenant, applied
deterministically on every node. It does **not** replicate the **control
plane**: tokens, database/tenant records, and scopes live in a per-node
`control.db` (SQLite), written directly by `ControlDb` and read by
`ControlDbAuthProvider`.

Consequences, all observed on the live homelab cluster:
- A token created on the leader **does not exist** on followers.
- Control state **does not survive** a node's loss — reseeding a node
  meant cloning its `control.db`.
- On failover, a client's token may be **invalid on the new leader**.
- Adding a node requires **manually seeding** auth.

An enterprise cluster whose identity/authorization does not survive
failover is not deployable. This is the #1 enterprise-grade blocker.

## Decision (codex verdict A): control ops through the existing YRP group

Model every control mutation as a **replicated log entry in a reserved
`__system__` namespace**, committed through the **same YRP consensus**,
and applied to `control.db` on every node by a **`ControlApplySink`** that
mirrors the data-plane `EngineApplySink`. One consensus group; reuse the
proven machinery (quorum-durable commit, exactly-once keys, the
linearizable-read barrier, snapshot/backfill).

Rejected: **(B) a second consensus group** — adds elections, cross-group
ordering, and monitoring for no demonstrated throughput need (control ops
are low volume). **(C) leader-owned + follower cache** — insecure on
revocation unless every auth check synchronously reaches the leader,
which sacrifices availability.

## Control mutation grammar

A `ControlOp` enum, serde-encoded into `Payload::Op` under the
`__system__` tenant (a reserved tenant id, e.g. 0):

- `CreateDatabase { db_id, name, path, config }`
- `CreateToken { db_id, token_hash, label, created_at }` — **hash only,
  never plaintext** (see Invariant 3)
- `RevokeToken { token_hash, revoked_at }`
- `GrantScope { token_hash, scopes }` / `RevokeScope { … }`
- `DeleteDatabase { db_id }` (tombstone-shaped, per the forget model)

Each carries a client op-id for exactly-once (the keyed-write contract),
so a retried `CreateToken` dedupes rather than double-inserting.

## Apply path

`ControlApplySink` applies a `ControlOp` to `control.db` in one
transaction (idempotent on op-id, mirroring the commit-log's
`(tenant, op_id)` unique index), advancing a durable **control-applied
marker** exactly as the data sink advances its outcome marker. The
existing per-node `control.db` becomes the **materialized state of the
replicated control log** — no schema change to `control.db` itself.

## Auth-read consistency (the revocation guarantee)

A stale-valid token on a follower is a security hole. Therefore an
authorization decision on any node must reflect every committed control
op that preceded it:

- **Reads pass the control-apply barrier.** `ControlDbAuthProvider`
  resolves a token only after the local control-apply marker covers the
  cluster's committed control frontier (reuse the RFC 028 read-barrier
  mechanism, scoped to the `__system__` log). A `RevokeToken` committed
  before the request is therefore visible before the request is
  authorized — cluster-wide.

## Invariants (codex pitfalls, made structural)

1. **Control-incomplete ⇒ not auth-eligible.** A node whose control
   sink is lagging, failed, or still restoring MUST refuse authenticated
   traffic even if its **data** engine is ready. This extends the RFC 028
   Phase-C `engine_incomplete` gate with a parallel `control_incomplete`
   gate; both surface on `/v1/health`.
2. **Control apply failure fences the node.** A `control.db` apply error
   or schema mismatch is fail-stop (quarantine posture) — never "continue
   and serve possibly-stale authorization." Same discipline as the data
   sink's fail-stop.
3. **Replicate verifier material, not secrets.** Tokens are already
   stored as SHA-256 hashes; the replicated `CreateToken` carries the
   **hash**, and the plaintext never leaves the minting node's response.
   Snapshots/backups of the control log are secured as credential
   material (they contain verifier hashes + membership, not plaintext).

## Snapshot / backfill / migration

Control state travels with the node lifecycle, not just the live apply:
- The YRP snapshot's frontier covers the `__system__` log; a rejoining
  node's **control** state is backfilled the same way its engine state is
  (RFC 028 Phase C), so a fresh/replacement node gets tokens+databases
  without manual seeding — which also **unblocks issue #74's node
  replacement** for the control plane.
- `control.db` schema migrations replicate as ordinary control ops or
  are applied deterministically at the same log position on every node.

## Bootstrap (chicken-and-egg)

`cluster_secret` is the **node-local bootstrap admin** that forms the
cluster before any replicated token exists (it already authenticates peer
traffic). The first real tokens/databases are created as replicated
control ops once the cluster is up. This is why `cluster_secret` stays a
peer/bootstrap credential and is **not** promoted to a data-plane token
(consistent with the finding corrected in the docs this cycle).

## Increment 1 (shipped) vs increment 2 (follow-up)

**Increment 1** delivers the replication mechanism: the `Payload::Control`
log entry, the `ControlOp` grammar, `ControlApplySink` (idempotent,
leader-assigned ids/timestamps, verifier-hash only), `propose_control`
with verify-after-apply, and the master-token-gated admin endpoints. A
token minted on the leader authenticates on every follower and survives
failover — proven in the 2-node HTTP cluster test.

Two **safety boundaries** hold increment 1 to the correctness bar (from an
adversarial review):

- **Compaction MUST be disabled** (`compact_after_entries = 0`, the
  production default). Control ops write no outcome row and are not yet
  carried in the YRP snapshot, so a compacted range containing a control op
  cannot be backfilled — a rejoining node would be stuck engine-incomplete
  and could miss a revoke. Enabling compaction now logs a loud `error!`.
  **Increment 2** carries control state in the snapshot (and/or gives
  control ops backfillable outcome rows), lifting this restriction.
- **Upgrade every node before minting the first control op** (bootstrap
  rule). A pre-`Control` binary rejects a `Payload::Control` AppendEntries
  (fail-safe — no misapply, HTTP 400) but then cannot advance past that
  index, stalling replication to it. Increment 2 gates control-op proposal
  behind a capability bit so a half-upgraded cluster refuses to mint one.

Increment 2 also adds the **auth-read barrier** (instantaneous cluster-wide
revocation — until then revocation is bounded-staleness eventually
consistent within replication lag) and the parallel `control_incomplete`
health gate.

## Out of scope (follow-ups)

External identity (OIDC/SSO/JWT) layers *on top of* this — it maps an
external identity to a `Principal`/`Scope`, which then replicates via the
same control log. Control-plane replication is the prerequisite; SSO is
the next enterprise rung.
