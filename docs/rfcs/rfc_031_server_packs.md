# RFC 031 — Server-side packs (clustered)

**Status:** Draft · **Depends on:** RFC 029 (control-plane replication), RFC 030 (admin RBAC), engine v0.11.1 (packs) · **Follows:** RFC 028 Phase C (fetch-missing-binary-state)

## The gap

Engine v0.11.0 shipped **packs** — sealed, single-file `.ydbpack` databases a
model mounts to gain knowledge it wasn't trained on and unmounts to give back
(seal/mount/unmount/`pack_context`, trust tiers, digest-verified, recall
auto-includes mounted packs at the engine's merge seam). The **server embeds
this engine but exposes none of it.** A networked or clustered YantrikDB
cannot mount a pack, and a self-hosted operator can't manage packs over the
API.

A node-local bolt-on (mount into whichever node serves the request) is a
half-feature: mount on the leader, `recall` on a follower, and the pack isn't
there — silently inconsistent, lost on failover. Packs are a headline
differentiator; they must work **correctly in the cluster**.

## The core problem: two very different things travel together

A "mounted pack" is **two** kinds of state with opposite requirements:

1. **The mount decision** — "database D has pack with digest X mounted." Tiny,
   deterministic, must be identical on every node and survive failover. This
   is control-plane state, and it belongs in the **replicated control log**
   (RFC 029), exactly like tokens and databases.
2. **The pack file** — a ~1MB+ opaque binary. Replicating megabytes *through
   the consensus log* is an anti-pattern (bloats the log, breaks compaction
   economics). And a file being momentarily absent on a node is a **liveness**
   problem (fetch it and retry), **not** a divergence — so it must **not**
   fail-stop the apply worker the way a real control-apply error does.

**The design separates them.** The manifest replicates through consensus; the
files are content-addressed and moved out-of-band; a best-effort reconciler
makes each node's *physical* mounts match the *replicated* manifest.

## Design

### 1. Replicated mount manifest (consensus, fail-stop-safe)

A new `control.db` table, materialized from replicated control ops (RFC 029):

```sql
CREATE TABLE db_packs (
    database_id  INTEGER NOT NULL,
    pack_digest  TEXT NOT NULL,   -- blake3 content digest (the identity)
    pack_name    TEXT NOT NULL,   -- display / filename
    mounted_at   TEXT NOT NULL,
    unmounted_at TEXT,            -- soft, tombstone-shaped
    PRIMARY KEY (database_id, pack_digest)
);
```

New `ControlOp` variants (ride the RFC 029 `ControlEnvelope`, actor-audited):

- `MountPack { database_id, pack_digest, pack_name, mounted_at }`
- `UnmountPack { database_id, pack_digest, unmounted_at }`

`ControlApplySink` applies these to `db_packs` **only** — a small, deterministic
SQLite write. This is idempotent (PK) and **fail-stop-safe**: writing a
manifest row can't fail on a healthy node, so it keeps RFC 029's Invariant 2
(control-apply error ⇒ fence) honestly, *without* coupling it to the physical
mount, which can legitimately fail transiently.

### 2. Content-addressed pack file store

Pack files live at `data_dir/packs/<blake3-digest>.ydbpack` — **content
addressed**, so identity == digest and dedup is automatic. The digest is the
engine's own `seal_pack` content digest (already verified at mount).

- Upload (admin) stores the file by its digest and returns its manifest.
- A node that needs a digest it doesn't have **fetches it** (below).

### 3. Peer transfer (out-of-band, digest-verified)

`GET /v1/packs/<digest>` — cluster-secret (peer) auth, streams the file. A node
missing a pack fetches it from the **leader** (leader-hint from YRP), falling
back to any reachable peer. The receiver **re-hashes and verifies the digest
before storing** — a mismatched byte stream is refused (untrusted-artifact
safety; the engine additionally re-verifies at mount). This is the RFC 028
Phase-C pattern (a node missing durable state pulls it from a peer), applied to
pack binaries.

### 4. The reconciler (best-effort, retryable, health-surfaced)

A per-node background loop makes physical mounts match the replicated manifest:

```
for each (database_id, digest) in db_packs where unmounted_at IS NULL:
    if not already mounted in that db's engine:
        ensure file present (fetch by digest if missing; verify)
        resolve the tenant engine; engine.mount_pack(path)
for each mounted pack NOT in the active manifest:
    engine.unmount_pack(id)
```

Failures (file unreachable, engine not loaded yet) are **logged, counted, and
retried** on the next tick — never fatal. A database whose manifest packs are
not all physically mounted is **`pack_incomplete`** and says so on
`/v1/health` (parallel to `engine_incomplete`). Reads still serve host memory;
they just don't yet see that pack. The reconciler runs on the same cadence as
other background reconcilers.

**Why not mount inside the apply sink?** Because mounting needs the tenant
*engine* (from the pool) and a possibly-absent *file* — coupling that to the
consensus apply marker would let a missing 1MB file wedge the whole cluster's
data apply. The manifest is the contract; the reconciler is the effort.

### 5. Failover / rejoin

The manifest is replicated control state, so a new leader and a rejoining node
already have it (log replay / control snapshot). Their reconcilers fetch the
files and mount — pack mounts **survive failover and reconstruct on a fresh
node** with no operator action, the same guarantee RFC 029 gave tokens.

## API (RBAC-gated per RFC 030)

- `POST /v1/admin/packs` — upload a `.ydbpack` (octet-stream) → store by digest,
  return `{digest, name, manifest, trust}`. **admin**.
- `GET /v1/admin/packs` — list stored packs (digest, name, size, trust). **readonly**.
- `POST /v1/admin/databases/{id}/packs` `{digest}` — propose `MountPack`
  (replicated); returns once the manifest commits (mount then reconciles).
  **admin**.
- `DELETE /v1/admin/databases/{id}/packs/{digest}` — propose `UnmountPack`. **admin**.
- `GET /v1/admin/databases/{id}/packs` — mounted + manifest + reconcile state. **readonly**.
- `GET /v1/packs/{digest}` — peer transfer (cluster-secret). Internal.
- `GET /v1/pack-context` — the mounted packs' `pack_context()` (constitution +
  coverage) for the caller's tenant, so an agent injects it into its system
  prompt. **token-authed** (normal tenant token via `resolve_engine`).
- **`/v1/recall` is unchanged** — the engine already merges mounted-pack
  candidates (scored with the host's weights × trust tier). Once mounted,
  recall surfaces pack content for free.

## Invariants

1. **Manifest is consensus; files are out-of-band.** Never put pack bytes in
   the YRP log.
2. **Digest-verified before mount** — a fetched file whose bytes don't hash to
   the requested digest is refused; the engine re-verifies at mount. Untrusted
   marketplace artifacts stay verifiable.
3. **Manifest apply is fail-stop-safe; physical mount is best-effort.** A
   missing/slow pack file degrades one database to `pack_incomplete`, never
   fences the cluster.
4. **Freshness gate (RFC 029 inc2-A) still applies** — a control-stale node
   won't serve admin pack ops or mint the manifest.
5. **Mount is idempotent + digest-keyed** — re-applying a `MountPack` or
   reconciling twice is a no-op.

## Review hardening (adversarial pass — folded in)

- **C1 — poison-pack quarantine + fault isolation (CRITICAL).** A committed
  `MountPack` whose file crashes/OOMs the shared engine on mount must NOT
  become a cluster-wide crash loop. Defenses, layered: (a) **upload caps** —
  reject a pack over `PACK_MAX_BYTES` (default 64 MB) at upload, and the engine
  already caps rows (2M) so the HNSW build is bounded; (b) **structural
  pre-vet** (engine `mount_pack` opens read-only, requires a real `memories`
  table, no `load_extension`); (c) the reconciler wraps each mount in
  `catch_unwind` and, on panic **or** after `PACK_MOUNT_MAX_ATTEMPTS` (default
  3) failures, moves that `(db, digest)` to a **terminal per-node
  `pack_poisoned` quarantine** (a local marker file, NOT the consensus
  manifest — the manifest stays the cluster's intent), stops retrying, and
  surfaces it on health. A best-effort actuator over committed state must have
  a terminal state; retry-forever over a fatal mount is crash-loop-forever.
- **H1 — fail closed / visible, not silent-open.** `MountPack` returns **202
  "accepted, reconciling"**, never "mounted." `recall` and `pack-context`
  responses on a db with manifest packs not yet locally mounted carry
  `packs_pending: [digest…]` (and `packs_poisoned`), and `pack_context` does
  **not** advertise coverage for a locally-unmounted pack. An opt-in strict
  mode (`?require_packs=1` / config) 503s like `engine_incomplete` instead of
  degrading. The agent is never told it has knowledge this node can't serve.
- **H2 — caps everywhere + control.db disk reservation.** `PACK_MAX_BYTES`
  (upload + per-fetch, aborting a peer stream that exceeds it); per-db and
  per-node mounted-pack caps → a mount over budget stays `pack_over_budget`
  (health state), never OOMs; the pack store has a disk high-water and
  **reserves headroom for `control.db`** so pack files can never starve the
  fail-stop control writes. GC of orphaned files is still a follow-up but the
  disk cap bounds the store regardless.
- **H3 / L1 — manifest apply is an unconditional idempotent UPSERT.** No `FK`,
  no `REFERENCES`, no CHECK, no "database exists" validation in the apply sink
  — nothing that can fail on committed input and fence the cluster (RFC 029
  inv2). `MountPack` UPSERTs and **clears `unmounted_at`** (so remount-after-
  unmount works); db-existence is checked only at **propose time** (endpoint,
  TOCTOU-tolerant); the reconciler treats a missing db as a no-op. Extend the
  RFC 029 delete path so `DeleteDatabase` **cascade-tombstones** `db_packs`.
- **M1 — digest is `^[0-9a-f]{64}$`-validated at every boundary** (upload, URL
  param, manifest apply) before it touches a path (implemented in
  `pack_store`), and hex is pinned (no base64 `/`). Downloads are size-capped.
- **M2 — reconciler correctness.** Single-flight (one reconcile in flight);
  process `UnmountPack`s before mounts and **re-check `unmounted_at`
  immediately before `mount_pack`** (no serve-after-unmount window); fetch to
  `<digest>.tmp.<rand>` → verify → atomic rename, and "present" means
  **verified-present**, not name-present (a truncated file heals); exponential
  backoff and a terminal `pack_unresolvable` state after exhausting reachable
  peers. **Mount hooks the engine LOAD path** — the tenant-pool loader consults
  `db_packs` and mounts before serving, so a cold/lazy tenant doesn't force-load
  everything nor retry-forever; the reconciler handles the already-loaded set +
  deltas. Unmount must free the HNSW (verified against engine v0.11.1).
- **M3 — failover is log-replay-only, contingent on compaction-off (F1).**
  §5 corrected: a rejoining node reconstructs mounts via log replay while
  compaction stays off (the RFC 029 F1 mitigation; control-in-snapshot is the
  shared follow-up). Cold-start reconcile is **bounded-concurrency and
  prioritized** (recently-active tenants first) to avoid a fetch/HNSW storm in
  the most fragile window.
- **L2/L3 — health + confidentiality notes.** `pack_incomplete`/`_poisoned`/
  `_over_budget` are **per-db, request-visible, and do NOT flip node health**
  (a best-effort liveness state must not make a load balancer evict a healthy
  node). `pack_name` is display-only and never reaches the filesystem (only the
  digest does). The content-addressed store + `cluster_secret`-only transfer is
  cluster-global by design (consistent with `cluster_secret` = break-glass
  owner); narrowing pack access below cluster-global is a future concern.

## Out of scope (follow-ups)

- **GC of orphaned pack files** (a digest no manifest references) — a periodic
  sweep; deferred, low risk (content-addressed, so safe to keep).
- **Signed-pack trust enforcement policy** at the server tier (accept only
  `signed`/`official`) — a config knob; the engine already carries the trust
  tier into scoring.
- **Compaction of the `db_packs` manifest** rides RFC 029's control-in-snapshot
  follow-up; until then compaction stays off (unchanged).
