# RFC 010 PR-6 — Write-Path Migration through MutationCommitter

Status: **Draft** (2026-05-02)
Triggered by:
- 2026-05-02 Epic #53 root cause: leader writes (e.g. `POST /v1/remember` to .140) bypass the RFC 010 commit log entirely. `commit_log.sqlite` on .141 has 0 rows; openraft last_log_index=18 is cluster-bookkeeping (init + membership), not application data. Cluster has been operating in **cosmetic openraft** mode since RFC 010 PR-4 — assembled, healthy in `/v1/health`, doing zero useful work because handlers never feed it.
- Architect's recurring report: "wrote a memory, can't recall it for hours" — same root cause.
- Lane B markets's skill_define iter 28 persistence bug — same root cause.
- v0.8.11 §2 piecewise-insert backfill (RFC 022) is correct but un-empirically-verifiable until replication actually moves bytes.

Constraint: **enterprise grade** (no shortcuts on differentiator features per user feedback memory `feedback_no_shortcuts.md`), **RFC-first** (this document blocks code), **maintenance-window upgrade tolerated** (no live rolling upgrade across PR-6 boundary because write semantics change), **deterministic apply** on every node (no embedder execution divergence between leader and follower).

Brainstorm reference: session `0e216e8c` (gpt-5.5 + deepseek + claude, three rounds, 2026-05-02 ~22:30 UTC). Synthesizer locked Submitter/Applier split, deterministic mutations, sync apply, snapshot-includes-app-state, maintenance-window upgrade, boot invariants, error mapping. Physical layout (Option D vs separate log) decided post-synthesizer by user "go" 2026-05-02 ~23:30 UTC: **Option D — per-tenant memory_commit_log table co-located in each tenant's `yantrik.db` file**.

This RFC supersedes the optimistic interpretation of RFC 010 PR-4 and PR-5 that write paths were already wired through `MutationCommitter`. They were not. PR-6 is the work that closes the gap.

---

## Goals

1. **Replication actually replicates.** Every successful `/v1/remember`, `/v1/forget`, `/v1/correct`, `/v1/relate`, `/v1/ingest_claim`, and `/v1/remember_batch` against a cluster leader produces a Raft log entry that propagates to followers and lands in their engine state within bounded time (target: read-your-writes within 1s on a healthy 3-node cluster, p99 < 5s).
2. **Single-node mode unchanged in semantics.** Single-node deployments use `LocalSqliteCommitter` and observe identical durability + idempotency contracts they already have today. No regression in single-node throughput beyond the apply-path overhead measured in §10 acceptance gates.
3. **Determinism.** Every node applies an identical mutation to identical state. The leader pre-computes embeddings, extracts entities, assigns server-side timestamps, and serializes the materialized result into the mutation. Followers do not re-embed or re-extract.
4. **Idempotency preserved end-to-end.** `(tenant_id, op_id)` retries return the original receipt regardless of which node the retry hits. Network blips during writes are safe.
5. **Boot invariants prevent regression to cosmetic mode.** Cluster mode REFUSES to start if the handler write path is not wired through `RaftSubmitter`. Detected at startup, fail-fast, no silent operation.
6. **Operator surface honest.** `/v1/health` reflects whether write replication is actually working (non-zero recent log entries, follower applied watermark within bounded distance of leader). Cosmetic mode becomes structurally impossible.

## Non-goals

- **Live rolling upgrade across the PR-6 boundary.** The write contract changes from "engine direct" to "log-then-apply". Mixed-version clusters are unsafe. Operators upgrade with a maintenance window: drain writes, stop both nodes, upgrade binaries + run schema migrations, restart. This is a one-time cost paid in v0.8.12 ship.
- **Multi-region or geo-replication.** Same-region 3-node cluster only.
- **Linearizable reads via Raft barrier.** `MutationCommitter::ensure_linearizable` exists in the trait but PR-6 does not require recall to use it. Reads remain leader-local-state for now; cross-cluster strict-RYW recall is RFC 024.
- **Submitter/Applier as a public surface.** It's an internal split inside `crates/yantrikdb-server` to make the commit-then-apply boundary explicit. No public API change.
- **Replacing `LocalSqliteCommitter` for single-node.** Single-node mode keeps the existing committer, just gets an `Applier` wired so the same code path drives engine state on both single-node and cluster.
- **CRDT-as-primary, async apply, eventual consistency.** Out of scope; locked by `wait_for_apply=true` invariant.

## Backwards-compat contract (read this first)

The v1 HTTP contract is **frozen** through PR-6. Specifically:

- `/v1/remember`, `/v1/forget`, `/v1/relate`, `/v1/correct`, `/v1/ingest_claim`, `/v1/remember_batch`, `/v1/recall`, `/v1/think` keep their existing signatures and default behaviors.
- New error responses are additive:
  - **HTTP 307** `Location: https://<leader_addr>/<original_path>` when a write hits a follower. Body includes `{ "leader_id": N, "leader_addr": "..." }`. Clients that follow redirects (every standard HTTP client does) succeed transparently. (Maps from `CommitError::NotLeader`.)
  - **HTTP 409** with `{ "error": "op_id_collision", "existing_index": N }` when `(tenant_id, op_id)` was previously committed with a different mutation. Today this never fires because the committer isn't on the path; PR-6 turns it on.
  - **HTTP 503** with `{ "error": "commit_timeout", "op_id": "...", "retry_after_ms": 1000 }` when the apply path times out. The op_id lets clients retry idempotently against the leader without duplicating writes.
- All schema migrations (m011, m012 — see §2.3) run at engine open during the maintenance-window restart; never on running engines.
- v0.8.11 piecewise-insert backfill (RFC 022 §2) becomes useful but does not change shape. The backfill loop now sees real Raft log entries instead of zero.

What v1 *gains*:

- Cluster mode actually replicates writes. Today empirically broken; PR-6 fixes.
- `/v1/health` extended (additive fields) with `replication_lag_log_entries`, `last_applied_at_unix_micros`, `commit_log_high_watermark_per_tenant_max` — see §6.2.
- `/v1/cluster/raft` admin endpoint surfaces leader id, current term, last log index, last applied, member set with each member's state. Already half-existing per RFC 010 PR-4 P1; PR-6 fills gaps.

What is **not** preserved:

- Any user expectation that single-binary mode running with `cluster.raft_mode = openraft` but no peers will work. After PR-6, that's a misconfiguration: you either run single-node (`raft_mode = disabled`) or full cluster (3+ voters). Mixed half-configs fail fast at startup. (Today they fail silently, which is worse.)

---

## Three-release sequencing

PR-6 is large enough that it ships across three patch releases. Each is independently mergeable and individually reverted-on-broken. Downstream releases depend on upstream.

| Release | PR sequence | Theme | Acceptance gate |
|---|---|---|---|
| **v0.8.12** | 6.1, 6.2, 6.3 | Submitter/Applier split + deterministic mutations + per-tenant commit log layout | Single-node passes existing 332+ unit + 187 raft_cluster + 91 commit_replay tests, plus 30+ new tests for Applier and per-tenant layout. No cluster behavior change yet. |
| **v0.8.13** | 6.4, 6.5, 6.6 | Handler migration + boot invariants + error mapping | All write handlers route through `MutationCommitter`. Cluster mode refuses to start if any handler bypasses the committer. End-to-end RYW: write to .140 leader, recall on .141 follower within 5s on healthy network. |
| **v0.8.14** | 6.7, 6.8, 6.9 | Per-tenant chunked snapshot + backfill admin tool + extended health surface | Follower bootstrap from cold catches up to leader within bounded time. `yantrikdb admin backfill-from-engine` populates commit log for .140's existing 39427 memories. `/v1/health` honestly reflects replication state. |

PR breakdown:

- **6.1** — Introduce `Submitter` and `Applier` traits in `src/commit/mod.rs`. Refactor `LocalSqliteCommitter` so it implements `Submitter` (logging) but delegates application to a separate `LocalApplier` over the engine. No handler wiring yet. Single-node still works, just composed differently.
- **6.2** — Mutation determinism. `MemoryMutation::UpsertMemory` and friends carry materialized embedding + extracted entities + server-assigned timestamps. New helpers in `src/commit/materialize.rs` that turn an HTTP request body into a deterministic mutation. Engine `record_text` is no longer the durable-write entry point.
- **6.3** — Per-tenant commit log layout (Option D). `memory_commit_log` table created inside each tenant's `yantrik.db` (existing engine SQLite) by migration m011. New `TenantCommitConnectionPool` caches one connection per active tenant. Old global `commit_log.sqlite` remains as fallback only during migration window; deleted in 6.7.
- **6.4** — Handler migration. `Command::Remember`, `Command::RememberBatch`, `Command::Forget`, `Command::Correct`, `Command::Relate`, `Command::IngestClaim` route through `MutationCommitter::commit`. `engine.record()` becomes pub(crate)-internal-only, called by `LocalApplier::apply`.
- **6.5** — Boot invariants. `RaftAssemblyConfig::validate()` checks that `HandlerContext::write_path == HandlerWritePath::Submitter` when `cluster.raft_mode = openraft`. New unit tests assert that any handler still calling `engine.record()` directly fails compilation (private visibility) or fails the boot check (defensive runtime assertion).
- **6.6** — Error mapping. `http_gateway.rs` translates `CommitError::NotLeader` → 307, `OpIdCollision` → 409, `CommitTimeout` → 503 with op_id. Existing 503 paths preserved; new ones additive.
- **6.7** — Per-tenant chunked snapshot via openraft `generic-snapshot-data` (already in `Cargo.toml`). `YantrikStateMachine::build_snapshot` walks `Submitter::list_active_tenants`, streams each tenant's `yantrik.db` SQLite checkpoint into a chunked snapshot. `install_snapshot` reverses: receives chunks, materializes per-tenant `yantrik.db` files. Empty-snapshot bug resolved.
- **6.8** — Backfill admin tool. `yantrikdb admin backfill-from-engine --tenant-id <id> --confirm` reads existing memories from a tenant's `yantrik.db`, synthesizes `UpsertMemory` mutations with materialized state, and appends them to the commit log under fabricated op_ids (UUIDv7 derived from rid + a backfill epoch). Followers replay these via normal Raft path. Admin-only, requires explicit `--confirm`, prints expected log entry count first. Closes the data-already-on-leader-but-not-in-log gap.
- **6.9** — Extended `/v1/health` and `/v1/cluster/raft` polish. New honest fields. Grafana dashboard JSON updated. `yantrikdb cluster status` CLI surfaces replication health.

Each PR ships green tests independently. PR 6.4 is the "moment of truth" — once handlers route through the committer, replication empirically works. PRs 6.5-6.9 harden and polish.

---

## Design

### Section 1 — Submitter / Applier trait split

`MutationCommitter` (RFC 010 PR-1, today's `trait_def.rs`) conflates two responsibilities: (a) durably appending a mutation to a log, and (b) applying it to engine state. PR-6 splits this so the same Applier implementation is reused on both leader and follower.

#### 1.1 Trait shape

```rust
// src/commit/submitter.rs (NEW)

#[async_trait]
pub trait Submitter: Send + Sync {
    /// Durably append the mutation to the commit log and trigger apply.
    /// On `wait_for_apply=true` (default), returns only after the local
    /// Applier has finished applying this entry.
    async fn submit(
        &self,
        tenant_id: TenantId,
        mutation: MemoryMutation,
        opts: CommitOptions,
    ) -> Result<CommitReceipt, CommitError>;

    /// Read-back, idempotency lookup, watermark queries — same signatures
    /// as today's MutationCommitter. Re-export for callers.
    async fn read_range(...) -> Result<Vec<CommittedEntry>, CommitError>;
    async fn high_watermark(...) -> Result<u64, CommitError>;
    async fn list_active_tenants(...) -> Result<Vec<TenantId>, CommitError>;
    async fn ensure_linearizable(...) -> Result<(), CommitError>;
}
```

```rust
// src/commit/applier.rs (NEW)

#[async_trait]
pub trait Applier: Send + Sync {
    /// Apply a single committed mutation to engine state. The implementation
    /// MUST be deterministic — given identical input mutation, every node
    /// produces identical engine state. MUST be idempotent on `(tenant_id,
    /// log_index)` — replaying the same entry yields the same result.
    ///
    /// Errors here are catastrophic: a single failed apply diverges the
    /// state machine. Implementations SHOULD treat any non-transient
    /// error as cause for shutdown (caller raises `CommitError::Shutdown`
    /// to upper layers and refuses further work until restart).
    async fn apply(
        &self,
        tenant_id: TenantId,
        log_index: u64,
        mutation: &MemoryMutation,
    ) -> Result<(), ApplyError>;
}
```

#### 1.2 Implementations

```
LocalApplier            — single concrete Applier; wraps yantrikdb::YantrikDB
                          per-tenant (looked up via control DB). Used on
                          BOTH single-node and cluster nodes.

LocalSqliteSubmitter    — single-node Submitter. submit() = log + immediately
                          call LocalApplier::apply, return receipt.

RaftSubmitter           — cluster Submitter. submit() = openraft client_write.
                          The state machine's apply_to_state_machine callback
                          drives LocalApplier::apply on every node.
```

The legacy `LocalSqliteCommitter` and `RaftCommitter` are kept as thin compatibility shims (they delegate to the new types) for one release, then deleted in v0.8.14.

#### 1.3 Why the split matters

- **Determinism is now structurally enforced.** `LocalApplier` is the only path that mutates engine state. Followers running the apply on bytes they received from Raft hit the same code as leaders running it on bytes they just logged. Divergence is impossible unless the mutation itself is non-deterministic — which §3 prevents by carrying materialized state.
- **Cluster apply runs in openraft's state machine apply callback** (RFC 010 PR-4-c). That callback is single-threaded per state machine; per-tenant ordering is preserved. Cross-tenant apply is sequential through the callback — acceptable because per-tenant ordering is what matters for correctness, and tenants don't share engine state.
- **Single-node submit + apply happens on the request task** (no detour through openraft). Latency overhead vs today is one extra trait dispatch + one row insert into per-tenant commit log — measured at ~150µs in PR-6.1 prototype. Within p99<750ms acceptance budget.

### Section 2 — Per-tenant commit log layout (Option D)

#### 2.1 Layout decision

Today's `commit_log.sqlite` is a single global file shared by all tenants. PR-6 moves to per-tenant: `memory_commit_log` table is created inside each tenant's existing `yantrik.db` file (the same file holding `memories`, `entity_edges`, etc.).

Rationale for co-location (Option D) over separate-log + reconciliation:
- One file per tenant → clean tenant deletion (delete the file, the log goes too). No orphan log entries from an older incarnation of a tenant.
- Single SQLite transaction can span both `memory_commit_log INSERT` and the `memories UPSERT` performed by Applier, eliminating the "logged but not applied" intermediate state on single-node.
- Tenant-scoped backups (RFC 012) inherit the log automatically.
- Schema migrations run per-tenant on first open after upgrade — no cross-tenant migration coordination.

Trade-off accepted:
- Apply-ordering across tenants is sequential through the openraft state machine callback, not parallel. Same-tenant ordering is the only guarantee that matters; this is fine.
- Snapshot serialization spans many files. Resolved by §5 chunked snapshot via openraft `generic-snapshot-data`.

#### 2.2 Connection management

```rust
// src/commit/tenant_pool.rs (NEW)

pub struct TenantCommitConnectionPool {
    /// Cached one connection per active tenant. Connections are opened
    /// against the tenant's existing yantrik.db file.
    conns: DashMap<TenantId, Arc<Mutex<Connection>>>,
    base_dir: PathBuf,
    control: Arc<ControlDb>,
}

impl TenantCommitConnectionPool {
    pub fn for_tenant(&self, t: TenantId) -> Result<Arc<Mutex<Connection>>, CommitError> {
        // Get-or-insert. First access opens the file, runs m011 + m012
        // migrations idempotently, configures pragmas (WAL, synchronous=NORMAL,
        // foreign_keys=ON — same shape as LocalSqliteCommitter).
    }

    pub fn close_idle(&self, idle_threshold: Duration) {
        // Background task evicts connections idle longer than threshold
        // to bound RSS for clusters with many tenants. Tracked via
        // last_use AtomicU64 per entry.
    }
}
```

~50 LOC for the pool, ~20 LOC for the eviction task. Total ~70 LOC for tenant connection management.

#### 2.3 Schema migrations

```sql
-- m011: memory_commit_log table inside tenant DB
CREATE TABLE IF NOT EXISTS memory_commit_log (
    log_index    INTEGER PRIMARY KEY,    -- monotonic per-tenant (no AUTOINCREMENT; we set explicitly)
    op_id        BLOB    NOT NULL,        -- UUIDv7 bytes (16)
    term         INTEGER NOT NULL,        -- 0 for single-node, raft term in cluster
    mutation     BLOB    NOT NULL,        -- serialized MemoryMutation (msgpack or bincode v2)
    committed_at INTEGER NOT NULL,        -- unix micros
    applied_at   INTEGER,                  -- unix micros, nullable until applier finishes
    schema_version INTEGER NOT NULL DEFAULT 1
);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_op_id ON memory_commit_log (op_id);
-- log_index is PK so already unique + indexed.

-- m012: cluster bookkeeping (openraft membership, vote, last_purged_log_id)
-- Lives in a per-cluster file (control.sqlite) NOT per-tenant — it's
-- cluster-global state. Migration m012 ships with it.
CREATE TABLE IF NOT EXISTS raft_metadata (
    key   TEXT PRIMARY KEY,
    value BLOB NOT NULL
);
```

m011 is per-tenant, runs on first open after upgrade. m012 is cluster-global, runs once on the control DB at upgrade.

#### 2.4 Backwards-compat during migration

The maintenance-window upgrade procedure (§7) is:
1. Drain writes (operator stops sending traffic).
2. Stop both nodes.
3. Upgrade binaries on both nodes.
4. On each node, run `yantrikdb admin migrate-commit-log --confirm` which copies any entries from the old global `commit_log.sqlite` into the appropriate per-tenant DBs (matched by tenant_id). For .140 + .141 today, this is a no-op because the commit log is empty — but the tool exists for any deployment that did light writes.
5. Restart both nodes.
6. Run backfill admin tool (PR 6.8) for memories that exist in `yantrik.db` but not in commit log.

Old `commit_log.sqlite` is renamed to `commit_log.sqlite.pre-pr6.bak` after migration; not deleted, in case operator wants to inspect it.

### Section 3 — Deterministic mutations

#### 3.1 Materialization at the leader

Today's `MemoryMutation::UpsertMemory` already carries `embedding: Option<Vec<f32>>` (it's there in `mutation.rs`). PR-6 makes it required-when-text-is-server-embedded and adds three more materialized fields:

```rust
MemoryMutation::UpsertMemory {
    rid: String,
    text: String,
    memory_type: String,
    importance: f64,
    valence: f64,
    half_life: f64,
    namespace: String,
    certainty: f64,
    domain: String,
    source: String,
    emotional_state: Option<String>,
    embedding: Vec<f32>,                    // CHANGED: was Option, now required
    metadata: serde_json::Value,
    // NEW in PR-6:
    extracted_entities: Vec<EntityRef>,     // NER output, computed by leader
    created_at_unix_micros: i64,            // server-assigned, leader's clock
    embedding_model: EmbeddingModelId,      // RFC 013 — pinned model + version
}
```

The HTTP gateway's `Command::Remember` handler now does:

```rust
async fn handle_remember(&self, req: RememberRequest) -> Result<...> {
    let tenant_id = self.auth.tenant_id;
    let materialized = self.materializer.materialize_remember(req).await?;
    //   ^- runs embedder, runs NER, assigns timestamps, fixes embedding_model.
    //      Pure function over the request + leader's clock + embedder.
    let mutation = MemoryMutation::UpsertMemory { /* fields from materialized */ };
    let receipt = self.submitter.submit(tenant_id, mutation, opts).await?;
    Ok(remember_response_from_receipt(receipt))
}
```

Followers receive the fully-materialized mutation via Raft and call `LocalApplier::apply` which:
- INSERT/UPDATE the `memories` row using the carried embedding (no embedder call).
- INSERT entity edges from `extracted_entities` (no NER call).
- Use `created_at_unix_micros` from the mutation (not the follower's clock).
- Stamp `embedding_model` so RFC 013 model migration knows which embeddings to re-encode.

#### 3.2 Wire version bump

Adding three required fields to a v1.0 variant breaks deserialization for anyone reading old payloads. Per RFC 010 PR-3 wire-version policy, this is a v1.1 bump:

- `MemoryMutation::wire_introduced_at(UpsertMemory)` becomes (1,1).
- `extracted_entities`, `created_at_unix_micros`, `embedding_model` are tagged with `#[serde(default)]` so old payloads (from the legacy commit log file imported at upgrade time) deserialize with empty defaults; the Applier handles the empty case by computing them on the fly (one-time, only for legacy entries).
- `FEATURE_FLOORS` registers `mutation.UpsertMemory.v1_1` requiring 0.8.12+.
- Wire-format conformance tests add a v1.1 golden payload.

Other variants (`UpdateMemoryPatch`, `TombstoneMemory`, `UpsertEntityEdge`, etc.) stay at v1.0 unchanged.

#### 3.3 Why determinism is non-negotiable

Without it, two failure modes:
1. **Embedder version skew.** Leader runs ONNX `bge-base-en-v1.5`; follower upgrades to `v1.6` mid-cluster. Identical text → different vectors → divergent HNSW state → recall returns different results on different nodes.
2. **NER model drift / clock skew on entity timestamps.** Same shape: nondeterminism in apply causes silent state divergence that is undetectable until a query sees the difference.

Carrying materialized state in the mutation makes apply byte-deterministic. The leader bears the embedder cost; followers bear only the SQL + HNSW insert cost (which is 100x cheaper than embedding). This is the right asymmetry: it's also the architecture every production Raft system uses.

### Section 4 — Synchronous apply contract

`CommitOptions::wait_for_apply` defaults to `true` and that test (`trait_def.rs:296-302`) is load-bearing. PR-6 preserves it intrinsically:

- **Single-node.** `LocalSqliteSubmitter::submit` calls `LocalApplier::apply` synchronously inside the same task. `applied_at` in the receipt is always `Some`.
- **Cluster.** `RaftSubmitter::submit` calls openraft's `client_write` which already waits for the entry to commit (durable on majority quorum) and apply on the leader's state machine. By the time `submit` returns, the leader has applied and the receipt's `applied_at` is `Some`. Followers apply asynchronously after receiving the entry via append-entries; by the time a write returns to the client, **leader read-after-write is consistent**, and **follower read-after-write becomes consistent within bounded time** (target 1s healthy, p99 5s).
- **Bulk import.** `wait_for_apply=false` is permitted by the existing trait but PR-6 does not surface it to HTTP. Only the bulk-import internal path (`POST /v1/admin/bulk_import`) uses it. The receipt's `applied_at` is `None`; clients using bulk-import accept that they can't read-their-write immediately.

#### 4.1 Apply timeout

The cluster Submitter waits up to `cluster.commit_apply_timeout_ms` (default 30000) for the openraft commit + apply round-trip. On timeout: `CommitError::CommitTimeout { op_id }`, mapped to HTTP 503 with retry hint. The op_id lets the client retry idempotently — if the timeout was spurious and the entry actually committed, the retry returns the original receipt; if the entry never committed, the retry creates it.

### Section 5 — Per-tenant chunked snapshot

#### 5.1 Why empty snapshots are a hard blocker

Today `YantrikStateMachine::build_snapshot` returns essentially empty bytes per the comment at `state_machine.rs:44-49`. Followers booting from cold cannot rebuild engine state from a snapshot — they'd need to replay every Raft log entry from index 0, which:
- doesn't work because old entries are compactable per RFC 010 retention contract.
- doesn't work because entries before PR-6 are bookkeeping with no application data.

PR-6 makes snapshots include application state.

#### 5.2 Layout

openraft's `generic-snapshot-data` feature (already in `Cargo.toml`) lets us serialize snapshots as a chunked stream rather than a single in-memory blob. The snapshot is:

```
snapshot.bin
├── header: { schema_version, last_log_id, membership, tenant_count }
├── for each tenant in list_active_tenants():
│   ├── tenant_header: { tenant_id, sqlite_size_bytes }
│   └── tenant_chunks: [chunk_0, chunk_1, ...]
│       (each chunk is a 1MiB slice of the tenant's yantrik.db file
│        captured via SQLite VACUUM INTO + raw read; the SQLite
│        copy is taken with the connection in a read transaction so
│        it's a consistent snapshot of all tenant data including
│        memories, entity_edges, AND the per-tenant commit log)
```

Receivers materialize each tenant's `yantrik.db` directly — so post-install-snapshot, the follower has a fully-bootstrapped engine state ready for query traffic.

#### 5.3 Ordering invariant

The snapshot's `last_log_id` is the openraft commit point at snapshot time. Each tenant's per-tenant commit log inside its `yantrik.db` is consistent up to that point. Followers replaying entries after the snapshot cursor see only entries with log_index > snapshot's last_log_id — no double-apply.

#### 5.4 Cost

Snapshot size = sum of all tenants' yantrik.db sizes. For .140 today: ~700MB across all namespaces in the meta tenant. Chunked at 1MiB, that's ~700 chunks transferred over a single openraft install_snapshot call. At 1Gbit cluster network, ~6s wall time for full bootstrap. Acceptable for cold-start; not on the hot path of any user request.

### Section 6 — Boot invariants + extended health surface

#### 6.1 Boot invariants

```rust
// src/raft/assembly.rs (extended)

impl RaftAssemblyConfig {
    pub fn validate(&self) -> Result<(), AssemblyError> {
        // ... existing checks ...

        if self.raft_mode == RaftClusterMode::OpenRaft {
            // PR-6 invariant: handler write path MUST be RaftSubmitter.
            // Single-node committer with raft_mode=openraft is forbidden.
            if self.write_path != HandlerWritePath::RaftSubmitter {
                return Err(AssemblyError::WritePathMismatch {
                    actual: self.write_path,
                    expected: HandlerWritePath::RaftSubmitter,
                    hint: "openraft mode requires RaftSubmitter handler wiring; \
                           configure cluster.handler_write_path = \"raft\" or set \
                           cluster.raft_mode = \"disabled\" for single-node",
                });
            }

            // PR-6 invariant: cluster.peers must list >= 2 peers.
            // Single-binary openraft mode is a misconfiguration trap.
            if self.peers.len() < 2 {
                return Err(AssemblyError::InsufficientPeers {
                    have: self.peers.len(),
                    need: 2,
                });
            }
        }

        Ok(())
    }
}
```

The assembly layer wires `HandlerContext::write_path` automatically from `cluster.raft_mode`, so this check is defense-in-depth — but the test `cluster_mode_with_local_submitter_panics_at_assembly` makes regression mechanically impossible.

#### 6.2 Extended `/v1/health`

```json
{
  "status": "healthy",
  "raft_mode": "openraft",
  "cluster": {
    "node_id": 2,
    "leader_id": 4,
    "term": 5,
    "last_log_index": 1842,
    "last_applied": 1842,
    "members": [...],
    "replication_lag_log_entries": 0,        // NEW
    "last_applied_at_unix_micros": 1779983412345678,  // NEW
    "commit_log_max_high_watermark": 39427    // NEW
  },
  "single_node": null
}
```

`replication_lag_log_entries`: `leader.last_log_index - this.last_applied` (0 on the leader; small positive on healthy followers; growing means broken).

`last_applied_at_unix_micros`: when the most recently applied entry was committed. If `now - last_applied_at` is large but `last_log_index > last_applied`, the apply path is stuck.

`commit_log_max_high_watermark`: max log_index across all per-tenant commit logs on this node. Cross-checks against `last_applied` (they should match within ε on a healthy node).

`/v1/health` returns HTTP 503 when:
- Cluster mode and no leader for >30s.
- This node's `replication_lag_log_entries > cluster.health_lag_threshold` (default 1000 entries).
- `last_applied_at` is more than 60s stale while there are unapplied entries (apply is stuck).

These checks make cosmetic mode structurally impossible: today's healthy-but-doing-nothing cluster would surface 503 instead of 200.

### Section 7 — Maintenance-window upgrade sequence

#### 7.1 Operator runbook

```
1. Announce maintenance window (typical: 15 min).
2. Enable read-only mode at the load balancer (or stop sending writes).
3. Wait 30s for in-flight writes to drain.
4. On EACH node in sequence:
   a. systemctl stop yantrikdb
   b. Backup data dir: tar -czf yantrik-pre-pr6-$(date +%s).tar.gz /var/lib/yantrikdb
   c. Install new binary (yantrikdb 0.8.12+).
   d. Run schema migrations: yantrikdb admin migrate-schemas --confirm
      (idempotent; runs m011 per-tenant + m012 control-DB)
   e. Run commit log migration: yantrikdb admin migrate-commit-log --confirm
      (copies any entries from old global commit_log.sqlite into per-tenant DBs;
      no-op if old file is empty or absent)
   f. systemctl start yantrikdb
   g. Verify boot invariants pass: journalctl -u yantrikdb | grep "boot invariants OK"
5. After both nodes healthy:
   a. Verify replication: write a probe to leader, expect to see it on
      follower within 5s. CLI: yantrikdb cluster verify-replication --probe-namespace _replication_probe
   b. Run backfill if existing memories exist:
      yantrikdb admin backfill-from-engine --tenant-id <id> --confirm
6. Re-enable writes at load balancer.
7. Monitor /v1/health for 1h; replication_lag_log_entries should stay near 0.
```

Estimated total: 10-15min wall time including verification. The actual stop-stop-restart window is ~3min per node; the rest is verification.

#### 7.2 Rollback plan

If anything breaks during the upgrade, rollback is:
1. `systemctl stop yantrikdb` on both nodes.
2. Restore data dir from pre-upgrade backup.
3. Reinstall previous binary (0.8.11).
4. `systemctl start yantrikdb`.

The migration tool's contract is forward-only-on-disk-but-binary-rollbackable: m011 + m012 add new tables but don't modify existing tables. A 0.8.11 binary opening a 0.8.12-migrated DB ignores `memory_commit_log` and `raft_metadata` tables and works as before.

### Section 8 — Backfill admin tool

`yantrikdb admin backfill-from-engine` is a one-shot tool to populate the commit log with synthesized entries for memories that already exist in `yantrik.db` but predate PR-6.

#### 8.1 Why we need it

.140 today has 39427 memories in its meta tenant. Those memories are in the engine but have no log entries. Without backfill:
- Cold-start follower bootstrap (via snapshot) works because the snapshot includes the full engine state.
- But the committed log "starts" at 0 entries on .140's per-tenant log → log_index assignment for new writes starts at 1, and there's no audit trail for the existing data.

Backfill synthesizes log entries so:
- Audit / replay completeness: every memory has a log entry behind it.
- Snapshot cursors reference real log entries (not entries-from-the-future).

#### 8.2 Mechanics

```
yantrikdb admin backfill-from-engine \
    --tenant-id 1 \
    --confirm

Will synthesize ~39427 commit log entries for tenant 1.
Estimated time: 8 minutes at 80 entries/sec.
Existing log entries (will not be overwritten): 0
Continue? [yes]
```

For each existing memory:
- `op_id`: UUIDv7 derived deterministically from `(rid, backfill_epoch)` so re-running the tool is idempotent.
- `term`: 0 (synthesized).
- `mutation`: `UpsertMemory` with all fields filled from the existing row + embedding loaded from HNSW + `embedding_model` from the row's metadata if present, else assumed-default.
- `committed_at`, `applied_at`: original `created_at` from the memory row.

Backfill runs on the leader only, so the leader generates entries that propagate to followers via normal Raft replication. After backfill, both nodes' commit logs match.

The tool is admin-gated (cluster mTLS + `admin:backfill` capability) and prints the entry count and ETA before proceeding. Refuses to run if `replication_lag_log_entries > 0` (cluster must be quiet first).

### Section 9 — Error mapping

| `CommitError` variant | HTTP status | Body | Notes |
|---|---|---|---|
| `NotLeader { leader_id, leader_addr }` | **307** | `{ "leader_id": N, "leader_addr": "https://..." }` | `Location: <leader_addr>/<orig_path>`. Standard HTTP clients follow transparently. |
| `OpIdCollision { op_id, tenant_id, existing_index }` | **409** | `{ "error": "op_id_collision", "existing_index": N, "op_id": "..." }` | Client bug. Don't retry. |
| `UnexpectedLogIndex { ... }` | **409** | `{ "error": "unexpected_log_index", ... }` | Concurrent write race; very rare on user paths (only used by cluster bootstrap). |
| `NotYetImplemented { variant, planned_rfc }` | **501** | `{ "error": "not_implemented", "variant": "...", "planned_rfc": "..." }` | E.g. `PurgeMemory` until RFC 011 PR-3. |
| `Version(VersionError)` | **426** | `{ "error": "wire_version_mismatch", ... }` | Upgrade required header. |
| `StorageFailure { message }` | **503** | `{ "error": "storage_failure", "retry_after_ms": 1000 }` | Retryable, transient. |
| `Shutdown` | **503** | `{ "error": "shutting_down", "retry_after_ms": 5000 }` | Don't retry on this node. |
| `CommitTimeout { op_id }` (NEW) | **503** | `{ "error": "commit_timeout", "op_id": "...", "retry_after_ms": 1000 }` | Critical: op_id allows idempotent retry. |

#### 9.1 Receipt response shape

`POST /v1/remember` response is unchanged in shape; new fields are additive:

```json
{
  "rid": "019de580-...",
  "op_id": "019de580-...",                   // NEW (additive)
  "log_index": 1843,                          // NEW (additive)
  "term": 5,                                  // NEW (additive)
  "committed_at_unix_micros": 1779983412345,  // NEW (additive)
  "applied_at_unix_micros": 1779983412567     // NEW (additive)
}
```

Existing clients ignore the new fields. New clients use `op_id` for retry safety.

### Section 10 — Acceptance gates

#### v0.8.12 (PRs 6.1, 6.2, 6.3)

- All existing tests pass: 332+ unit + 187 raft_cluster + 91 commit_replay + 95 wire_format + 4 cpu_isolation = 709+.
- New tests: 30+ covering Submitter trait, Applier trait, LocalApplier idempotency on `(tenant_id, log_index)` replay, materialization round-trip, tenant connection pool eviction.
- Single-node `/v1/remember` p99 < 10% above v0.8.11 baseline. Acceptance: p99 < 825ms (was 750ms — 10% buffer for the trait dispatch + log insert overhead).
- m011 + m012 migrations are idempotent (run twice, second time no-op).

#### v0.8.13 (PRs 6.4, 6.5, 6.6)

- End-to-end RYW: write `POST /v1/remember` to .140 leader, query `POST /v1/recall` on .141 follower within 5s, find the memory. Repeat 1000 times across a chaos network (latency injection, occasional packet loss). p99 read-your-writes < 5s.
- Cluster mode startup test: assemble with `raft_mode=openraft` and `peers=[]` → fails at boot with `InsufficientPeers`.
- Cluster mode startup test: assemble with `raft_mode=openraft` and a deliberately misconfigured handler context (LocalSqliteSubmitter) → fails at boot with `WritePathMismatch`.
- HTTP error mapping: assert all 8 `CommitError` variants map to documented HTTP status + body shape.
- Idempotency: write same `(tenant_id, op_id, mutation)` 100 times concurrently → exactly 1 log entry, all 100 receipts identical.
- Different mutation same op_id: returns 409 OpIdCollision.

#### v0.8.14 (PRs 6.7, 6.8, 6.9)

- Snapshot install: bootstrap a third (new) follower from the current leader; full engine state (memories + entity edges + per-tenant commit logs) materializes within 30s for a 1GB-engine cluster.
- Backfill admin tool: dry-run shows correct entry count; full run produces deterministic UUIDv7 op_ids (re-running yields zero new entries).
- `/v1/health` returns 503 if a follower's `replication_lag_log_entries > 1000`, returns 200 otherwise.
- Grafana dashboard JSON: `yantrikdb_replication_lag_entries{node}` gauge populated, `yantrikdb_commit_log_high_watermark{tenant}` gauge populated.

---

## Risk table

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Embedder version skew between nodes after PR-6 | Low | High (silent state divergence) | Embedder runs only on leader; mutation carries materialized vector. Apply path never embeds. RFC 013 model migration is the planned mechanism for cluster-wide embedder upgrade. |
| Per-tenant `yantrik.db` connection exhaustion under high tenant count | Medium | Medium (slow writes to seldom-used tenants) | `TenantCommitConnectionPool::close_idle` evicts connections after 5min idle. Pool size capped at 256. |
| Schema migration corruption during maintenance window | Low | High | Migrations are idempotent + forward-only. Pre-migration backup is mandatory in runbook. Rollback restores from backup. |
| openraft `generic-snapshot-data` API churn between minor versions | Low | Medium | Pin openraft = 0.9.x in Cargo.toml. Upgrade is a separate RFC. |
| Backfill tool produces non-deterministic op_ids → re-run creates duplicates | Low | High | Op_id is UUIDv7 derived from `(rid, backfill_epoch)` via blake3 — fully deterministic. Tested in PR 6.8. |
| HTTP 307 redirect loop if `leader_addr` is wrong | Low | Medium | Clients clamp redirect chain at 3 hops (curl/reqwest default). Server-side: the leader returns 200, not 307, even when about to step down (race window <1s). |
| Apply path stalls on a single tenant blocking all cluster apply | Medium | High | Apply timeout (30s default) trips `CommitError::Shutdown` and the apply task surfaces `health=degraded`. Operator alerts via Grafana. Per-tenant apply is fast (<10ms) so stall implies a real bug worth paging on. |
| Maintenance-window upgrade scheduling for production users | High (operational concern) | Low (one-time cost) | Document the procedure with copy-paste runbook. Total wall time 10-15min. Acceptable for v0.8.x patch series since user accepts non-rolling upgrade per `feedback_no_shortcuts.md`. |

## Open questions

1. **Should `LocalApplier::apply` hold the engine write lock for the full apply duration, or release between SQLite-tx and HNSW-insert?** Today's `engine.record()` does HNSW insert under the same lock as SQLite tx. PR-6.4 preserves this for correctness. The v0.8.x concurrency RFC (separate work, A″/B-lite/RCU hybrid) will revisit lock scope; PR-6 punts and inherits whatever scope record() has at v0.8.12 ship time.
2. **Bulk-import (`/v1/admin/bulk_import`) `wait_for_apply=false` behavior on followers.** Leader returns receipt as soon as log entry is durable. Follower applies in background. Read-your-writes guaranteed only on leader. Open: should bulk-import auto-block reads on the leader until apply completes? Default: no (caller opts out of RYW by using bulk-import). Document in user docs.
3. **Backfill on a pre-existing follower.** The .141 follower has 0 memories today (new node). After v0.8.12 maintenance upgrade, leader runs backfill → entries replicate to follower → follower's engine state populates from those entries via Applier. But what if the follower already had partial memories from a prior incarnation? The snapshot install mechanism (§5) overwrites; backfill assumes a clean follower. **Decision:** backfill tool refuses to run if any non-leader replica has non-empty `yantrik.db`. Operator must `yantrikdb admin reset-replica --node <id> --confirm` first.
4. **op_id determinism for backfilled entries — is blake3(rid + backfill_epoch) enough?** For idempotent re-run, yes. For cross-cluster determinism (same tenant existing on two clusters that later merge), no — but that's not a use case PR-6 supports. Document the limitation.

## References

- RFC 010 (commit substrate + openraft + retention) — original 6 PRs shipped 2026-04-28 to 2026-04-29. This RFC is the missing PR-6 (write-path migration), distinct from the original PR-6 (retention contract). Numbering pun acknowledged; semantically this is "RFC 010 PR-7" if one prefers.
- RFC 013 (HNSW lifecycle + embedding model migration) — `embedding_model` field in `UpsertMemory` couples here.
- RFC 022 (skill substrate + RYW) — v0.8.11 piecewise-insert backfill becomes empirically meaningful only after PR-6 lands.
- Brainstorm session `0e216e8c` (2026-05-02) — gpt-5.5 + deepseek + claude three-round redteam locked Submitter/Applier split, deterministic mutations, sync apply, snapshot must include app state, maintenance-window upgrade, boot invariants, error mapping. Physical layout decided by user post-synthesizer.
- Diagnosis memory `019deac9-83c8-71e6-85cb-044924c32d59` (Epic #53 root cause).
- Architectural decision memory `019deb58-4b44-7f67-971c-4c576190cdc2` (PR-6 lock).
- User feedback `feedback_no_shortcuts.md` — fix it right, RFC first, then code. Especially for differentiator features. PR-6 is the biggest correctness fix in v0.8.x; this RFC is the gate.

## Sign-off

Status remains **Draft** until:
- Pranab approves architecture (Submitter/Applier split, Option D layout, maintenance-window upgrade).
- Architect (cross-lane) reviews the cluster behavior changes (HTTP 307 redirect, /v1/health 503 thresholds, error mapping).
- One more brainstorm pass at PR boundary to confirm 6.1-6.3 (v0.8.12) is not too thin to merge cleanly.

Engine code starts only after sign-off. RFC-first rule applies (per RFC 022 §2 lesson — empirical 90s test caught a 300-LOC misdirection; this RFC must clear the same bar).
