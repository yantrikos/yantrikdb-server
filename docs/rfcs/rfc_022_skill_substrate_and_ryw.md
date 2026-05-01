# RFC 022 — Skill Substrate + Indexed Metadata + Read-Your-Writes HNSW

Status: **Draft** (2026-05-01)
Triggered by:
- 2026-04-30 cross-lane bug: yantrikdb-agi's `meta` and `iran` Lane B agents both hit the same `skill_recall` finds it / `skill_get` can't pattern. Root cause: every program reinvents `skill_define` / `skill_get` / `skill_recall` in agent code with subtle bugs. The architect runs 67 production skills via agent-layer convention; the convention has drifted.
- 2026-05-01 architect message #2: "newly-written memories via `/v1/remember` are not findable via `/v1/recall` for some indeterminate window after the write returns." Root cause: HNSW vector add is async; SQLite row + embedding blob commit synchronously but the HNSW index lags by an unbounded window. Lane B's predict→validate→revise loop is blocked by it.
- yantrikdb-agi's `skill_get(skill_id)` scales poorly: it is implemented as `recall(top_k=100) + 2-pass + client-side filter on metadata.skill_id`, because cosine ranks the literal id LOW (the body doesn't contain `skill.foo.v1` verbatim). Fails at 100k+ records.

Constraint: **backwards compatible** (v1 contract preserved), **enterprise grade** (no band-aids, RFC-first design, full validation/observability/migration), **no breaking changes to existing clients**.

Brainstorm references: sessions `4235baed`, `af916207`, `44195988` on 2026-05-01. Architect endorsement received via swarmcode.

---

## Goals

1. Ship first-class skill primitives at the substrate layer (`/v1/skills/{define, get, search, outcome, forget}`), eliminating the cross-program convention-drift bug class.
2. Read-your-writes consistency on HNSW for the cluster: leader-side already works (empirically verified — engine does synchronous HNSW insert); follower-side is fixed by replacing replication backfill's full HNSW rebuild with piecewise per-row insert. Lag drops from minutes-to-hours → seconds.
3. Provide O(1) exact-key lookup on indexed metadata fields (closes the `skill_get` scalability gap).
4. Provide pre-filter on `/v1/recall` using indexed metadata fields (closes scaling at 1k+ records per namespace).
5. Schema validation on writes — catches drift bugs (hyphen-vs-underscore in array entries, malformed IDs) WITHOUT enforcing semantic ontology (no validation gates, no outcome rollups).

## Non-goals

- **Heavy skill ontology.** No engine-side enforcement that lessons reference `kind=validation_run`. No auto-incremented `success_count`. No origin immutability. These are agent-layer pedagogy.
- **Public meta-type primitive.** No `POST /v1/types/define` for user-declared schemas (Brainstorm 2 consensus killed this — three round-trips, schema management is scope expansion). Engine internals (`namespace_schema`) are designed for extensibility but the public API stays concrete.
- **Cross-namespace recall.** Useful but distinct concern; separate RFC if/when it arrives.
- **Time-travel HNSW / temporal vector queries.** Research track for v2.x.
- **Belief synthesis primitive (reconciliation of N memories into one belief).** Reasoning operation, not storage; agent-recipe layer.
- **`/v1/skills/learn`, `/v1/skills/master`, `/v1/skills/promote`-style endpoint names.** Names that overclaim semantic capability. Refused.

## Backwards-compat contract (read this first)

The v1 contract is **frozen**. Specifically:

- `/v1/remember`, `/v1/recall`, `/v1/forget`, `/v1/relate`, `/v1/think`, `/v1/correct` keep their existing signatures and default behaviors.
- `/v1/recall` default behavior preserved. v0.8.11 ships **no new consistency flags** — engine is already RYW-consistent on the leader, follower lag is closed by the backfill fix in §2. (Cluster-strict RYW with `min_vector_seq` is reserved for RFC 024 / cluster work, not v0.8.11.)
- `/v1/remember` response shape unchanged in v0.8.11. (No new fields; cluster-RYW additions like `vector_seq` are deferred to RFC 024.)
- All new fields (perimeter, where-clause, metadata-indexing) are additive; omit them and behavior is unchanged.
- All schema migrations (v24, v25 if applicable) are forward-only and idempotent. Existing memories without indexed metadata are unaffected.

What v1 *gains*:

- `/v1/skills/*` new endpoint family.
- `/v1/lookup` new generic exact-key endpoint.
- `/v1/recall` accepts optional `where` clause (omit = same behavior).
- `/v1/recall` returns `perimeter` if `perimeter: N` is requested (omit = no perimeter; not returned in response).
- Engine library `yantrikdb` 0.6.5: two new public methods (`insert_vector`, `encrypt_embedding_pub`) — purely additive surface for replication followers' piecewise HNSW backfill. See §2.

Every schema migration runs at engine open; never on running engines. No client-visible downtime.

---

## Three-release sequencing

This RFC covers three distinct shipping units:

| Release | Theme | Acceptance gate |
|---|---|---|
| **v0.8.11** | Skill API endpoints + follower HNSW piecewise-insert backfill | Schema-validation rejection paths covered, define→immediate-recall on leader works (already does, regression check), follower recall lag drops from minutes-to-hours → seconds, encrypted-cluster follower recall starts working |
| **v0.8.12** | namespace_schema + sidecar metadata_index_values + `/v1/lookup` generic endpoint | `/v1/lookup` p95 ≤10ms embedded / ≤30ms server at 100k records, `/v1/skills/{id}` becomes O(1) |
| **v0.8.13** | `where`-clause prefilter on `/v1/recall` + query planner with `explain` + yantrikdb-agi migration | Filtered recall p95 ≤100ms embedded / ≤200ms server at 100k records 1% selectivity, agi migration succeeds, sidecar cache deleted |

Each release is independently mergeable; downstream releases depend on upstream. RFC describes the full surface; PR-level acceptance gates appear at the end of each section.

---

## Design

### Section 1 — Skill API endpoints (v0.8.11)

#### 1.1 Wire format

```
POST /v1/skills/define
Authorization: Bearer <token>
Content-Type: application/json

{
  "skill_id":   "skill.invoice.validation.v3",
  "body":       "When validating an invoice, check that the vendor is on the approved list and the amount is within the policy threshold.",
  "applies_to": ["invoice", "approval"],
  "skill_type": "rule",
  "metadata":   { "authored_by": "agent.lane_b", "version": 3 }
}
```

Response (201 Created):

```json
{
  "rid":          "019de580-1234-7abc-...",
  "skill_id":     "skill.invoice.validation.v3",
  "namespace":    "skill_substrate",
  "memory_type":  "procedural",
  "created_at":   1762012345.123
}
```

```
GET /v1/skills/{skill_id}
Authorization: Bearer <token>
```

Response (200 OK):

```json
{
  "rid":          "019de580-1234-...",
  "skill_id":     "skill.invoice.validation.v3",
  "body":         "When validating an invoice, ...",
  "applies_to":   ["invoice", "approval"],
  "skill_type":   "rule",
  "metadata":     { ... },
  "namespace":    "skill_substrate",
  "memory_type":  "procedural",
  "created_at":   1762012345.123
}
```

404 Not Found if the skill_id is not registered (or has been forgotten).

```
POST /v1/skills/search
Authorization: Bearer <token>
Content-Type: application/json

{
  "query":       "How should I validate an invoice before approving it?",
  "top_k":       5,
  "applies_to":  ["invoice"],          // optional filter
  "skill_type":  "rule"                // optional filter
}
```

Response (200 OK):

```json
{
  "results": [
    {
      "rid":         "019de580-...",
      "skill_id":    "skill.invoice.validation.v3",
      "body":        "...",
      "applies_to":  ["invoice", "approval"],
      "skill_type":  "rule",
      "score":       0.87,
      "namespace":   "skill_substrate"
    },
    ...
  ]
}
```

```
POST /v1/skills/{skill_id}/outcome
Authorization: Bearer <token>
Content-Type: application/json

{
  "success":  true,
  "context":  "applied during invoice #4521 review, blocked the policy violation",
  "ts":       1762012600.5
}
```

Response (201 Created):

```json
{
  "outcome_rid": "019de580-9999-...",
  "skill_ref":   "skill.invoice.validation.v3"
}
```

```
POST /v1/skills/{skill_id}/forget
Authorization: Bearer <token>
Content-Type: application/json

{
  "cascade_outcomes": false   // optional, default false
}
```

Response (200 OK):

```json
{
  "found":            true,
  "skill_id":         "skill.invoice.validation.v3",
  "rid":              "019de580-...",
  "outcomes_purged":  0
}
```

#### 1.2 Validation rules (strict, schema-only)

On `/v1/skills/define`:

| Field | Rule | Error code |
|---|---|---|
| `skill_id` | matches `^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$` | `INVALID_SKILL_ID_FORMAT` |
| `skill_id` | length 4..200 chars | `INVALID_SKILL_ID_LENGTH` |
| `skill_id` | unique within namespace=skill_substrate | `SKILL_ID_CONFLICT` (409) |
| `body` | length 50..5000 chars | `INVALID_BODY_LENGTH` |
| `applies_to` | non-empty array | `EMPTY_APPLIES_TO` |
| `applies_to[*]` | each entry matches `^[a-z][a-z0-9_]*$` | `INVALID_APPLIES_TO_ENTRY` |
| `applies_to` | length 1..10 | `TOO_MANY_APPLIES_TO` |
| `skill_type` | in {`procedure`, `reference`, `lesson`, `pattern`, `rule`} | `INVALID_SKILL_TYPE` |

Critical: the `applies_to[*]` entry regex catches the hyphen-vs-underscore drift bug Brainstorm 2 named (`["meta_agent"]` vs `["meta-agent"]` — both are valid strings, but only one matches). Storing both consistently was impossible at agent layer; substrate enforces it once.

Not validated (intentionally — schema, not semantics):
- Whether the body is "actually a good skill"
- Whether `applies_to` references known agents
- Whether the same body content already exists under a different skill_id
- Whether outcomes recorded against this skill imply success/failure

#### 1.3 On-conflict behavior

`POST /v1/skills/define` with existing `skill_id`:

- Default: **409 Conflict** with body `{"error": "SKILL_ID_CONFLICT", "skill_id": "...", "existing_rid": "..."}`. Silent overwrite is the trust-killer; loud rejection is the trust-builder (Brainstorm 2).
- `?on_conflict=update`: replaces the existing skill (writes new memory + tombstones the old via supersede chain when v0.8.15 ships; for v0.8.11–14 it's tombstone+rewrite). Returns 200 OK with new rid.
- `?on_conflict=ignore`: returns 200 OK with the **existing** rid. No write.

#### 1.4 Implementation

`/v1/skills/define` is a thin wrapper over `/v1/remember`:

```rust
// pseudocode
fn define_skill(req: SkillDefineRequest) -> Result<SkillDefineResponse> {
    validate_schema(&req)?;
    if engine.lookup("skill_substrate", "skill_id", &req.skill_id)?.is_some() {
        match req.on_conflict {
            OnConflict::Reject => return Err(Conflict),
            OnConflict::Update => { /* tombstone existing, fall through */ }
            OnConflict::Ignore => return Ok(existing.into()),
        }
    }
    let metadata = json!({
        "record_type": "skill",
        "skill_id":    req.skill_id,
        "applies_to":  req.applies_to,
        "skill_type":  req.skill_type,
        ...req.metadata
    });
    let result = engine.record(RecordRequest {
        text:        req.body,
        namespace:   "skill_substrate",
        memory_type: MemoryType::Procedural,
        metadata,
        ...
    })?;
    Ok(SkillDefineResponse { rid: result.rid, ... })
}
```

`/v1/skills/{id}/outcome` writes to the `outcome_substrate` namespace:

```rust
fn record_outcome(skill_id: &str, req: OutcomeRequest) -> Result<OutcomeResponse> {
    // Schema check: skill exists. (Otherwise outcome dangles.)
    engine.lookup("skill_substrate", "skill_id", skill_id)?
        .ok_or(SkillNotFound)?;
    let metadata = json!({
        "record_type": "skill_outcome",
        "skill_ref":   skill_id,
        "success":     req.success,
        "context":     req.context,
        "ts":          req.ts.unwrap_or_else(now),
    });
    let result = engine.record(RecordRequest {
        text:        format!("Outcome for {}: {} — {}", skill_id, req.success, req.context),
        namespace:   "outcome_substrate",
        memory_type: MemoryType::Episodic,
        metadata,
        ...
    })?;
    Ok(OutcomeResponse { outcome_rid: result.rid, skill_ref: skill_id.into() })
}
```

The engine never auto-rolls-up `success_count` on the parent skill. If a program wants rollups, it queries `outcome_substrate` itself and aggregates. This is the **architectural enforcement of schema-not-semantics**: there's literally no machinery in the engine to maintain derived counters, so the line cannot be crossed accidentally.

#### 1.5 Files

| File | Change |
|---|---|
| `crates/yantrikdb-server/src/handlers/skills.rs` | new module, 5 endpoint handlers |
| `crates/yantrikdb-server/src/http_gateway.rs` | mount `/v1/skills/*` routes |
| `crates/yantrikdb-server/src/command.rs` | add `Command::DefineSkill { ... }`, `Command::GetSkill { ... }`, etc. variants |
| `crates/yantrikdb-protocol/src/messages.rs` | add `SkillDefineRequest`, `SkillDefineResponse`, ... protocol types |
| `crates/yantrikdb-server/src/handler.rs` | dispatch new Command variants |
| `crates/yantrikdb-server/tests/skill_api.rs` | new integration test file |

### Section 2 — Follower HNSW backfill: piecewise insert (v0.8.11)

> **REVISED 2026-05-01 ~21:50 UTC after empirical diagnosis.** The original draft of this section proposed an in-engine pending-vector overlay (`pending_overlay: RwLock<HashMap<NamespaceKey, Vec<PendingVector>>>` + `vector_seq: AtomicU64` + `consistency: read_after_write/indexed` modes). That design was **wrong about the failure mode**.
>
> Empirical test against the live homelab cluster (.140 leader, .141 follower, term 219):
>
> | Test | Result |
> |---|---|
> | Write to .140 leader → recall on .140 leader immediately | ✅ found, 147 ms total round-trip |
>
> Engine `record()` already does **synchronous** HNSW insert at `crates/yantrikdb-core/src/engine/record.rs:68`:
>
> ```rust
> // Insert into vector index (lock ordering: conn already dropped)
> self.vec_index.write().insert(&rid, embedding)?;
> ```
>
> Leader-side write→recall is already read-your-writes consistent. The pending-vector overlay would have been ~300 LOC of dead code on the leader path.
>
> **The actual bug** the architect reports lives in **follower replication backfill**, not in engine recall. See §2.1 below. The pending-vector overlay design has been removed; the v0.8.11 RYW work is now a much smaller, surgical fix to the follower path. The original overlay design is preserved in this RFC's revision history (§2.7) for reference if a future bug ever surfaces a real leader-side gap.

#### 2.1 Current behavior — empirically verified

**Leader path (engine `record()`):**

```
1. /v1/remember called → server spawns blocking task
2. engine.record():
   a. encrypt fields if encryption enabled
   b. SQLite INSERT INTO memories (...) — synchronous, durable
   c. session linkage UPDATEs
   d. self.vec_index.write().insert(&rid, embedding) — SYNCHRONOUS, in same tokio blocking task
   e. cache_insert + heuristic entity extraction + graph index update
3. /v1/remember returns 200 OK with {rid}
4. /v1/recall called immediately on the same node — sees the memory ✓
```

The leader is RYW-consistent. Verified by 2026-05-01 21:55 UTC live test.

**Follower path (replication, `crates/yantrikdb-server/src/cluster/sync_loop.rs:290`):**

```
1. sync_loop polls leader's commit log (cadence: 5s default)
2. Receives N replicated ops via cluster_pull endpoint
3. handle_oplog_apply() inserts memory rows into follower SQLite
   — text + metadata + embedding_hash present, embedding column NULL
4. backfill_embeddings() runs:
   a. SELECT rid, text FROM memories WHERE embedding IS NULL LIMIT 500
   b. for each row: db.embed(text) → SQLite UPDATE embedding column
   c. AT END OF BATCH: db.rebuild_vec_index() — FULL HNSW REBUILD
5. follower /v1/recall now sees the new memories
```

The slow part is step 4c. `rebuild_vec_index()` walks the entire `memories` table, deserializes every embedding, and builds a fresh HNSW from scratch. With N memories that's O(N log N) HNSW work on EVERY backfill cycle. With 1k+ memories per tenant the rebuild takes seconds; with 10k+ it can take minutes; under load contention it's the multi-hour lag the architect reported.

The comment in the code at line 365–367 admits the workaround:

> *"Now rebuild the HNSW index from the SQLite table (which has all embeddings now). This is the only way to get vectors into HNSW since the index API isn't public for piecewise insertion through YantrikDB."*

That comment is the bug.

#### 2.2 The fix — expose piecewise insert + replace rebuild with insert loop

Two engine-library API additions, then one server-side change:

**Engine PR (yantrikdb 0.6.5):**

Promote the engine's existing `pub(crate)` HNSW insert path to `pub`:

```rust
// crates/yantrikdb-core/src/engine/storage.rs — already exists, currently pub(crate)
impl YantrikDB {
    /// Insert a single (rid, embedding) into the HNSW vector index.
    /// SQLite row must already exist (this is a backfill helper for replication
    /// followers that received memory rows via oplog without their vectors).
    /// Idempotent: re-inserting an already-present rid is a no-op.
    pub fn insert_vector(&self, rid: &str, embedding: &[f32]) -> Result<()> {
        // existing pub(crate) implementation, just re-exported
    }

    /// Encrypt an embedding blob using the engine's DEK if encryption is enabled.
    /// Returns the input unchanged if encryption is disabled.
    /// Used by replication backfill to write encrypted-cluster vectors.
    pub fn encrypt_embedding_pub(&self, blob: &[u8]) -> Result<Vec<u8>> {
        self.encrypt_embedding(blob)  // existing pub(crate) method
    }
}
```

These are **additive API surface only** — existing callers unaffected, no behavior change. The methods already exist; they just gain `pub` visibility. ~10 LOC + tests.

**Server PR (yantrikdb-server 0.8.11):**

Replace `db.rebuild_vec_index()` in `backfill_embeddings()` with a per-row insert loop:

```rust
// crates/yantrikdb-server/src/cluster/sync_loop.rs (rewritten ~line 320–375)

async fn backfill_embeddings(engine: &Arc<yantrikdb::YantrikDB>) -> anyhow::Result<()> {
    let db = engine.as_ref();
    if !db.has_embedder() { return Ok(()); }

    let pending = collect_pending_rids(db)?;  // SELECT rid, text WHERE embedding IS NULL

    if pending.is_empty() { return Ok(()); }
    let count = pending.len();
    tracing::debug!(count, "backfilling embeddings via piecewise insert");

    let mut backfilled = 0;
    let mut errors = 0;

    for (rid, text) in &pending {
        // 1. Embed
        let embedding = match db.embed(text) {
            Ok(v) => v,
            Err(e) => { tracing::warn!(rid=%rid, error=%e, "embed failed"); errors += 1; continue; }
        };

        // 2. Encrypt blob (no-op if encryption disabled)
        let blob = yantrikdb::serde_helpers::serialize_f32(&embedding);
        let stored_blob = match db.encrypt_embedding_pub(&blob) {
            Ok(b) => b,
            Err(e) => { tracing::warn!(rid=%rid, error=%e, "encrypt failed"); errors += 1; continue; }
        };

        // 3. SQLite UPDATE — embedding column populated
        let conn = db.conn();
        if let Err(e) = conn.execute(
            "UPDATE memories SET embedding = ?1 WHERE rid = ?2",
            params![stored_blob, rid],
        ) {
            tracing::warn!(rid=%rid, error=%e, "embedding UPDATE failed"); errors += 1; continue;
        }
        drop(conn);

        // 4. HNSW piecewise insert — the actual fix
        if let Err(e) = db.insert_vector(rid, &embedding) {
            tracing::warn!(rid=%rid, error=%e, "HNSW insert failed during backfill");
            errors += 1;
            continue;
        }

        backfilled += 1;
    }

    tracing::info!(
        backfilled,
        errors,
        total = count,
        "follower HNSW backfill complete (piecewise)"
    );
    Ok(())
}
```

The change is local to `backfill_embeddings()`. After this change:

- Per-row work is O(log N) HNSW insert (vs O(N log N) full rebuild on the previous batch path)
- Recall on follower sees each replicated memory **as soon as that memory completes step 4**, not after the entire batch finishes step 4c
- Encrypted-cluster follower recall starts working (was previously broken — line 345 in the old code skipped encrypted writes entirely with a TODO)
- No batched coordination required; SQLite WAL handles concurrent reader visibility

**Lock ordering note**: this loop calls `db.conn()` and `db.insert_vector()` per iteration. `insert_vector` internally takes `vec_index.write()`. The existing engine lock-ordering rule (conn → hlc → scoring_cache → vec_index → graph_index → active_sessions) is preserved because we drop conn (`drop(conn)`) before calling `insert_vector`.

#### 2.3 Cluster RYW

This v0.8.11 work does NOT solve the "write to leader, immediately read from follower with strict freshness guarantee" case — that's a separate concern.

What v0.8.11 ships:
- Leader-side RYW: already worked, no change needed (verified empirically)
- Follower-side recall lag: drops from O(rebuild) per batch to O(per-row) — typically seconds → milliseconds for typical batch sizes

What v0.8.11 does NOT ship:
- A `min_vector_seq` / `consistency: "read_after_write"` parameter on `/v1/recall` — deferred. Engine doesn't need it leader-side; cluster RYW would require apply-index coordination through openraft, which is RFC 010-B / RFC 024 territory.

If a future bug surfaces a real leader-side RYW gap (e.g., we move HNSW insert to async to scale write throughput), the pending-vector overlay design from §2.7 below is the spec.

#### 2.4 Acceptance gates (v0.8.11 §2)

- **API additions**: `db.insert_vector(rid, embedding)` and `db.encrypt_embedding_pub(blob)` are public, documented, idempotent on re-insert.
- **Follower backfill latency**: write a memory to leader; on follower with no other replication load, the memory is recallable within `(sync_loop_poll_interval + per_row_embed_time)` — typically 5–10 s end-to-end at 5 s poll interval, vs minutes-to-hours pre-fix.
- **Encrypted cluster**: write a memory to encrypted-cluster leader; follower recall succeeds within the same window. (Was 100% broken pre-fix.)
- **No regression on leader**: leader-side write→recall latency unchanged (still ≤200 ms p99 per v0.8.9 acceptance gates).
- **No partial-batch HNSW divergence**: under kill-9 mid-backfill, the partial state is recoverable on restart (existing reconciliation logic via RFC 013-A HNSW manifest still works).
- **Telemetry**: new counter `yantrikdb_follower_backfill_inserted_total` (per-row insert count); existing `yantrikdb_follower_backfill_errors_total` keeps counting failures.

#### 2.5 Files

| File | Change |
|---|---|
| `crates/yantrikdb-core/src/engine/storage.rs` | Promote `insert_vector` from `pub(crate)` to `pub` (or add re-export wrapper) |
| `crates/yantrikdb-core/src/engine/record.rs` | Promote `encrypt_embedding` to `pub` via `encrypt_embedding_pub()` wrapper (preserves internal call sites) |
| `crates/yantrikdb-core/Cargo.toml` | version 0.6.4 → 0.6.5 |
| `crates/yantrikdb-core/CHANGELOG.md` | document new public API |
| `crates/yantrikdb-server/src/cluster/sync_loop.rs` | rewrite `backfill_embeddings` to use piecewise insert; remove `rebuild_vec_index()` call |
| `crates/yantrikdb-server/src/metrics.rs` | add `yantrikdb_follower_backfill_inserted_total` counter |
| `crates/yantrikdb-server/tests/replication_backfill.rs` | new integration test: replicate N memories to follower, verify recall finds them within bounded time |
| `crates/yantrikdb-server/Cargo.toml` | engine bump 0.6.4 → 0.6.5 + server version 0.8.10 → 0.8.11 |

Total LOC: ~50 engine, ~80 server (most of it is the rewrite of `backfill_embeddings`), ~120 test. Substantially smaller than the previously-proposed overlay (300+ LOC + ongoing maintenance burden).

#### 2.6 Migration plan

- Engine library bump 0.6.4 → 0.6.5: pure additive API, no schema migration, no behavior change for existing callers.
- Server bump 0.8.10 → 0.8.11: drops the `rebuild_vec_index` call from backfill. Existing follower nodes that upgrade will start using piecewise insert immediately on next backfill cycle. No data migration needed; existing HNSW state on disk (if any) is rebuilt incrementally as the embeddings table is walked.
- Rolling upgrade safe: leader can run 0.8.10 while follower runs 0.8.11; the wire format of replicated ops is unchanged.

#### 2.7 Revision history — preserved overlay design

> **Status: NOT IMPLEMENTED.** Kept here as a future-reference spec in case a real leader-side RYW gap surfaces.

The original §2 design (drafted 2026-05-01 ~21:25 UTC, replaced 2026-05-01 ~21:50 UTC after empirical diagnosis):

A pending-vector overlay (`pending_overlay: RwLock<HashMap<NamespaceKey, Vec<PendingVector>>>`) inside `YantrikDB`. Every `record()` would push the new (rid, vector, namespace, vector_seq) onto the overlay; a background HNSW inserter would drain it; recall with `consistency: read_after_write` or `min_vector_seq: u64` would union the overlay with HNSW results before scoring. Backpressure via overlay-size thresholds; restart recovery via SQLite reconciliation; cluster RYW via per-node overlays + apply-index coordination.

The empirical reason this was not needed: engine `record()` does synchronous HNSW insert. There is no "write committed but not yet in HNSW" window on the leader. The overlay would always be empty and the consistency flags would be no-ops.

If a future change moves HNSW insert to an async/batched path (e.g., to scale write throughput beyond what `vec_index.write().insert()` allows), this overlay design is the spec for restoring leader-side RYW. Until then, the simpler piecewise-insert fix in §2.2 is sufficient.

### Section 3 — namespace_schema + indexed metadata + /v1/lookup (v0.8.12)

#### 3.1 Schema declaration

```
PUT /v1/namespaces/{namespace}/schema
Authorization: Bearer <token>
Content-Type: application/json

{
  "schema_version": 1,
  "indexed_metadata": {
    "skill_id":   { "path": "$.skill_id",   "type": "string",   "index": "exact",    "unique": true,  "required": true,  "format": "^[a-z][a-z0-9_]*(\\.[a-z0-9_]+)+$" },
    "skill_type": { "path": "$.skill_type", "type": "string",   "index": "exact",    "values": ["procedure", "reference", "lesson", "pattern", "rule"] },
    "applies_to": { "path": "$.applies_to[*]", "type": "string", "index": "contains", "format": "^[a-z][a-z0-9_]*$" },
    "version":    { "path": "$.version",    "type": "integer",  "index": "range" }
  },
  "strict": false
}
```

If `strict: true`, writes that don't match the schema are rejected. If `strict: false` (default), writes succeed but unindexed fields are silently dropped from the index (they remain in the metadata blob — recall still works, just slower).

The skill API in v0.8.11 will register a default schema for `skill_substrate` automatically on first skill_define, so the engine has the index machinery ready when v0.8.12 ships.

#### 3.2 Storage

Migration v24 adds:

```sql
CREATE TABLE namespace_schema (
    namespace        TEXT    NOT NULL PRIMARY KEY,
    schema_version   INTEGER NOT NULL,
    schema_json      TEXT    NOT NULL,
    strict           INTEGER NOT NULL DEFAULT 0,
    created_at       INTEGER NOT NULL,
    updated_at       INTEGER NOT NULL
);

CREATE TABLE metadata_index_values (
    namespace      TEXT    NOT NULL,
    field          TEXT    NOT NULL,
    rid            TEXT    NOT NULL,
    value_text     TEXT,
    value_int      INTEGER,
    value_real     REAL,
    value_bool     INTEGER,
    PRIMARY KEY (namespace, field, rid)
) WITHOUT ROWID;

CREATE INDEX idx_mmi_text  ON metadata_index_values(namespace, field, value_text, rid) WHERE value_text IS NOT NULL;
CREATE INDEX idx_mmi_int   ON metadata_index_values(namespace, field, value_int,  rid) WHERE value_int  IS NOT NULL;
CREATE INDEX idx_mmi_real  ON metadata_index_values(namespace, field, value_real, rid) WHERE value_real IS NOT NULL;
CREATE INDEX idx_mmi_bool  ON metadata_index_values(namespace, field, value_bool, rid) WHERE value_bool IS NOT NULL;

-- For unique fields like skill_id, partial unique index on value_text
CREATE UNIQUE INDEX idx_mmi_unique
    ON metadata_index_values(namespace, field, value_text)
    WHERE value_text IS NOT NULL
      AND field IN (SELECT field FROM namespace_schema_unique_fields);
```

(The unique-field constraint requires a tiny supporting table `namespace_schema_unique_fields(namespace, field)` populated from schema declarations. SQLite's partial unique index over a subquery isn't directly possible; we use trigger-based enforcement instead — design detail in implementation.)

#### 3.3 Index maintenance

Index rows are written **in the same SQLite transaction** as the parent memory row:

```rust
fn record(req: RecordRequest) -> Result<RecordResponse> {
    let txn = self.conn.transaction()?;

    // 1. Insert the memory row
    txn.execute("INSERT INTO memories (...) VALUES (...)", ...)?;

    // 2. Look up the namespace schema
    let schema = self.namespace_schema_cache.get(&req.namespace);

    // 3. For each indexed field, extract value via JSON1 path and insert
    if let Some(schema) = schema {
        for (field_name, field_def) in &schema.indexed_metadata {
            if let Some(values) = extract_json_path(&req.metadata, &field_def.path) {
                for value in values {  // may be multiple if path contains [*]
                    validate_field_value(&value, field_def)?;
                    txn.execute(
                        "INSERT INTO metadata_index_values (namespace, field, rid, value_text, ...) VALUES (...)",
                        ...
                    )?;
                }
            } else if field_def.required {
                return Err(MissingRequiredField(field_name.clone()));
            }
        }
    }

    // 4. Append commit log entry
    txn.execute("INSERT INTO memory_commit_log ...", ...)?;

    txn.commit()?;
    Ok(...)
}
```

Forget cascades correctly via `ON DELETE CASCADE` on the foreign key relationship (or explicit DELETE in the same transaction).

#### 3.4 /v1/lookup endpoint

```
POST /v1/lookup
Authorization: Bearer <token>
Content-Type: application/json

{
  "namespace": "skill_substrate",
  "field":     "skill_id",
  "value":     "skill.invoice.validation.v3"
}
```

Response (200 OK):

```json
{
  "found":  true,
  "memory": {
    "rid":         "019de580-...",
    "text":        "...",
    "metadata":    { ... },
    "namespace":   "skill_substrate",
    "memory_type": "procedural",
    "created_at":  1762012345.123
  },
  "index_used": "skill_id"
}
```

Or 404 Not Found if the (namespace, field, value) tuple is not indexed.

The lookup path:

```rust
fn lookup(req: LookupRequest) -> Result<Option<Memory>> {
    let value_col = match value_type_for_field(&req.namespace, &req.field) {
        Some(VT::Text) => "value_text",
        Some(VT::Int) => "value_int",
        ...
        None => return Err(FieldNotIndexed),
    };
    let rid: Option<String> = self.conn.query_row(
        &format!("SELECT rid FROM metadata_index_values WHERE namespace = ?1 AND field = ?2 AND {} = ?3", value_col),
        params![req.namespace, req.field, req.value],
        |row| row.get(0),
    ).optional()?;
    rid.map(|r| self.read_memory(&r)).transpose()
}
```

This is **O(log n) via SQLite btree**, not O(1) — important to be honest about in marketing. p95 ≤10ms embedded / ≤30ms server is the real claim.

#### 3.5 Skill default schema (registered automatically by v0.8.11)

To avoid users needing to declare the schema before using `/v1/skills/*`, v0.8.11's skill handler registers a default schema for `skill_substrate` on first define call (idempotent). The schema is the same as the validation rules in §1.2.

This means by the time v0.8.12 ships, all production skills already have indexed metadata working — the v0.8.12 release just exposes `/v1/lookup` and the underlying machinery to other consumers.

#### 3.6 Acceptance gates (v0.8.12)

- `/v1/lookup` p95 ≤10ms embedded / ≤30ms server at 100k records (single-namespace)
- `/v1/skills/{skill_id}` switches from scan-then-filter to indexed lookup (latency goes from O(n) to O(log n))
- Schema declaration accepts well-formed schemas, rejects malformed (missing `path`, invalid `type`, etc.)
- Index rows correctly maintained on record/forget/correct (verified by property test: random insert/forget sequence + assert SQLite index matches metadata blob)
- Unique constraint enforcement: defining two skills with the same skill_id returns 409 even on direct `/v1/remember` (not just `/v1/skills/define`)
- Migration v24 idempotent (running twice is a no-op)

#### 3.7 Files

| File | Change |
|---|---|
| `crates/yantrikdb-core/migrations/v24_namespace_schema.sql` | new migration |
| `crates/yantrikdb-core/src/engine/schema.rs` | new module: schema declaration API, validation |
| `crates/yantrikdb-core/src/engine/index_maintenance.rs` | new module: write-time index updates |
| `crates/yantrikdb-core/src/engine/lookup.rs` | new module: lookup() implementation |
| `crates/yantrikdb-core/src/engine/record.rs` | call into index_maintenance on every write |
| `crates/yantrikdb-core/src/engine/forget.rs` | cascade delete on metadata_index_values |
| `crates/yantrikdb-server/src/handlers/lookup.rs` | new module |
| `crates/yantrikdb-server/src/handlers/namespaces.rs` | new module: schema CRUD |
| `crates/yantrikdb-server/src/http_gateway.rs` | mount new routes |
| `crates/yantrikdb-core/tests/indexed_metadata.rs` | new unit + property tests |

### Section 4 — where-clause prefilter on /v1/recall + planner explain (v0.8.13)

#### 4.1 Wire format

```
POST /v1/recall
Authorization: Bearer <token>
Content-Type: application/json

{
  "namespace": "skill_substrate",
  "query":     "How should I validate an invoice?",
  "top_k":     5,
  "where": {
    "all": [
      { "field": "record_type", "op": "eq",       "value": "skill" },
      { "field": "skill_type",  "op": "eq",       "value": "rule" },
      { "field": "applies_to",  "op": "contains", "value": "invoice" }
    ]
  },
  "explain": true
}
```

Response (200 OK):

```json
{
  "results": [ ... ],
  "plan": {
    "filter_mode":              "indexed_prefilter",
    "indexes_used":             ["record_type", "skill_type", "applies_to"],
    "candidate_count":          384,
    "vector_strategy":          "exact_rerank_over_candidates",
    "estimated_distance_computations": 384,
    "fallback_used":            false
  }
}
```

#### 4.2 Filter operators (v0.8.13 minimum set)

| Type | Operators |
|---|---|
| `string` | `eq`, `in` |
| `string[]` | `contains`, `contains_any`, `contains_all` |
| `integer` | `eq`, `in`, `gt`, `gte`, `lt`, `lte` |
| `real` | `eq`, `gt`, `gte`, `lt`, `lte` |
| `boolean` | `eq` |

Boolean composition: `all` (AND), `any` (OR), `not` — recursive.

NOT in v0.8.13 (deferred to later if needed): regex match, full-text search, arbitrary JSONPath predicates, custom scoring functions.

#### 4.3 Query planner

Two strategies:

**Strategy A — filter-first with exact rerank** (used when candidate count is small):

1. Apply `where` filter to `metadata_index_values`, get candidate rids.
2. Fetch embeddings for those rids from SQLite.
3. Compute exact cosine similarity for each.
4. Sort by score, take top_k.
5. Apply post-filters (decay, importance) and return.

Used when `candidate_count <= prefilter_threshold` (default 1000, configurable per-namespace).

**Strategy B — HNSW with allowlist** (used when candidate count is large):

1. Apply `where` filter, get candidate set (could be large — 100k+).
2. Pass candidate rid set to HNSW as an allowlist.
3. HNSW search restricted to that set.
4. If `hnsw_rs` doesn't support allowlist (it currently does not in upstream), fall back to:
   - HNSW global search with overfetch (top_k × 5).
   - Post-filter results against candidate set.
   - Return top_k from filtered results.
   - Mark `plan.fallback_used = true`.

The fallback is correct but degraded — if all top-overfetched results are filtered out, recall@K suffers. The `explain` output makes this visible.

We will contribute an allowlist patch to upstream `hnsw_rs` if it doesn't already exist; this is tracked separately as a v0.9.x gate (we ship the fallback in v0.8.13 because it works).

#### 4.4 Acceptance gates (v0.8.13)

- Filtered recall p95 ≤100ms embedded / ≤200ms server at 100k records, 1% selectivity (Strategy A)
- Filtered recall p99 ≤500ms embedded / ≤1000ms server at 1M records, 1% selectivity (Strategy A or B)
- recall@K matches brute-force ground truth at top_k=10, ≥90% (Strategy A) / ≥80% (Strategy B fallback)
- `explain` output is accurate: `candidate_count`, `indexes_used`, `vector_strategy` reflect actual execution
- yantrikdb-agi migration: skill-loss bug not reproducible across 1000 trial runs; sidecar SQLite cache deleted; code lines reduced ≥50

#### 4.5 Files

| File | Change |
|---|---|
| `crates/yantrikdb-core/src/engine/recall.rs` | accept `where` clause, dispatch to strategy A or B |
| `crates/yantrikdb-core/src/engine/planner.rs` | new module: candidate count estimation, strategy selection |
| `crates/yantrikdb-core/src/engine/filter.rs` | new module: where-clause AST + execution against metadata_index_values |
| `crates/yantrikdb-core/src/engine/recall.rs` | exact-rerank-over-candidates path |
| `crates/yantrikdb-core/src/types.rs` | add `WhereClause`, `Plan`, `RecallExplain` types |
| `crates/yantrikdb-server/src/http_gateway.rs` | accept `where`, `explain` on `/v1/recall` |
| `crates/yantrikdb-core/tests/where_clause.rs` | new unit + integration tests |
| `crates/yantrikdb-server/tests/recall_explain.rs` | new integration test |

---

## Migration plan

### v0.8.10 → v0.8.11

- Engine library bumps yantrikdb 0.6.4 → 0.6.5 (purely additive: two methods promoted from `pub(crate)` to `pub`).
- Migration: none. No schema change.
- Existing clients: unchanged behavior. Skill endpoints are new; nobody depends on them yet.
- Existing memories: untouched.
- Cluster: rolling upgrade safe — leader can run 0.8.10 while follower runs 0.8.11; the follower's piecewise-insert backfill works against any leader because replication wire format is unchanged. A 0.8.10 follower running against a 0.8.11 leader keeps the old (slow) full-rebuild path until upgraded.

### v0.8.11 → v0.8.12

- Engine library bumps 0.6.5 → 0.6.6.
- Migration v24 (namespace_schema + metadata_index_values) — runs at engine open, idempotent.
- Backfill: existing memories without indexed metadata are NOT backfilled automatically. Consumers can call `POST /v1/namespaces/{ns}/reindex` if they want existing data indexed (slow operation, runs in background, progress visible via `GET /v1/namespaces/{ns}/index_status`).
- Skill substrate's default schema is registered idempotently on first `/v1/skills/define` after v0.8.11; v0.8.12 doesn't change this.
- Existing clients: unchanged behavior unless they call `/v1/lookup` (new endpoint).

### v0.8.12 → v0.8.13

- Engine library bumps 0.6.6 → 0.6.7.
- No schema migration (schema from v24 is sufficient).
- Existing clients: unchanged behavior unless they pass `where` clause on `/v1/recall`.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Piecewise HNSW insert in backfill fails partway through batch | Per-row error counted in `yantrikdb_follower_backfill_errors_total`; on next sync_loop poll the row is re-attempted (still has `embedding IS NULL` if step b failed, or already-in-HNSW if step d failed → idempotent re-insert). Existing v0.8.x retry logic is preserved. |
| HNSW write lock contention slows individual inserts under heavy follower-side replication catch-up | Per-row insert holds `vec_index.write()` briefly (single insert is O(log N)); existing v0.8.9 read pool means concurrent recall still works. No regression vs current full-rebuild path which holds the same lock for the entire rebuild. |
| Encrypted-cluster follower backfill | `encrypt_embedding_pub` is exposed; sync_loop calls it to encrypt vectors before SQLite UPDATE. Was 100% broken pre-fix. |
| Schema declaration drift between cluster nodes | Schemas stored in commit log (RFC 010), replicated via openraft; eventual consistency on schema is fine because writes against unknown schema fall back to "no indexed metadata" path |
| Allowlist-HNSW unavailable upstream | Fallback path (overfetch + filter) is correct, just degraded; `explain` makes degradation visible |
| Skill API on existing memories without indexed metadata | v0.8.11's default schema registration handles this; existing skills written via `/v1/remember` directly with the convention also work because they happen to match the indexed schema |
| Cross-program convention drift returning | Engine-side validation (§1.2) makes this impossible by construction for new writes; old writes that don't match the schema continue to work but won't be recallable via `/v1/skills/search`'s indexed path |

---

## Out of scope (deferred to follow-up RFCs)

- RFC 023 (Epistemic Control Plane): provenance fields, supersede chains, recall perimeter, action-conditioned veto, scoped negative evidence, memory-of-missing-memory, quarantine, latent contradiction awareness. v0.8.14 onwards.
- RFC 024 (Cluster strict RYW): `min_vector_seq` API surface, openraft commit-and-apply-index coordination so a client writing on the leader can read with strict freshness from any follower. Different problem from v0.8.11's piecewise-insert fix (which closes the slow-rebuild gap but doesn't provide the strict per-write apply-index guarantee).
- RFC 025 (Outcome rollups as agent-recipe): if a program wants `success_count`, document the standard agent-recipe pattern for aggregating outcomes via `/v1/recall` over `outcome_substrate`.
- Public meta-type primitive `/v1/types/define` — not planned. Engine internals stay extensible; new concrete endpoints (`/v1/habits/*`, `/v1/policies/*`) ship when 2+ programs ask for them.

---

## Open questions

1. Should `/v1/skills/{id}/forget` default to `cascade_outcomes = true` or `false`? Argument for true: outcomes referencing a forgotten skill are dangling. Argument for false: outcomes are themselves audit-relevant and should outlive the skill they reference.
   - **Tentative answer**: false. Outcomes are events; events outlive their subjects. Operator can pass `cascade_outcomes=true` if they want hard delete.

2. Should `applies_to` array entries be enforced as a closed set (must match a registered list) or open (any string matching the regex)?
   - **Tentative answer**: open. Closed sets are pedagogy. Programs that want closed sets enforce at agent layer.

3. For the perimeter feature in v0.8.16, should `recall(perimeter=N)` increment the existing `top_k` HNSW cost or be free?
   - **Deferred**: this is RFC 023 territory.

4. With piecewise insert closing the multi-hour follower lag, what's the new SLA on follower freshness? After v0.8.11, follower lag should be `(sync_loop_poll_interval + per_row_embed_time × N)` where N is the batch size — typically 5–10 s for small batches. Should we tighten the poll interval default from 5 s to 1 s, given backfill is now cheap?
   - **Tentative answer**: leave the default at 5 s for v0.8.11 (don't change two things at once). Revisit in v0.8.12 after observing backfill cost telemetry.

---

## References

- Brainstorm 1 (session `4235baed`, 2026-05-01): killed `/v1/types/define` meta-primitive on prior-art grounds.
- Brainstorm 2 (session `af916207`, 2026-05-01): converged on concrete `/v1/skills/*` over typed-memory abstraction.
- Brainstorm 3 (session `44195988`, 2026-05-01): produced the v1.0 epistemic-control-plane thesis (separate RFC 023).
- yantrikdb-agi cross-lane bug `019de142` (2026-05-01 ~01:55 UTC): meta + iran skill_recall/skill_get inconsistency.
- yantrikdb-agi message `d1ae4f5a` (2026-05-01 17:29 UTC): HNSW read-after-write lag report.
- Existing RFC 009 (admission control), RFC 010 (commit substrate), RFC 013 (HNSW lifecycle) — this RFC composes with all three.

---

## Status

| Section | State |
|---|---|
| §1 Skill API | RFC drafted, ready for implementation |
| §2 Follower HNSW backfill (piecewise insert) | RFC revised after empirical diagnosis 2026-05-01 ~21:50 UTC; ready for implementation. Original overlay design preserved as §2.7 for future reference. |
| §3 namespace_schema + lookup | RFC drafted |
| §4 where-clause + planner | RFC drafted |
| Pranab approval | granted 2026-05-01 ("Go" + revise-RFC option after empirical diagnosis) |
| Architect approval | parallel review notified via swarmcode 2026-05-01 ~21:25 UTC |

Implementation status: **engine code starting now (v0.8.11 PR sequence)**.
