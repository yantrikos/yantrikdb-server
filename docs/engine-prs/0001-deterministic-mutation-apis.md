# Engine PR draft — deterministic mutation APIs (`record_with_rid` + friends)

Target repo: `github.com/yantrikos/yantrikdb` (the `yantrikdb` engine crate)
Target version: **`yantrikdb 0.7.0`** (minor bump — purely additive surface)
Driver: RFC 010 PR-6.4 in `yantrikdb-server` (handler migration through MutationCommitter)
Status: **Draft spec** — for review before engine work starts

## Why this PR exists

`yantrikdb-server` v0.8.x runs in cluster mode but every write currently
bypasses the openraft commit log. RFC 010 PR-6 fixes this by routing
writes through `MutationCommitter` → openraft consensus → state-machine
apply on every node.

For follower apply to be **byte-deterministic** (a non-negotiable
correctness requirement — diverged HNSW state is undetectable until a
query notices), the leader must materialize every nondeterministic
input *before* the mutation enters the log:

| Field | Materialized at leader | Today's API |
|---|---|---|
| `rid` | `uuid7()` once | server-side `db.record(...)` returns rid |
| `embedding` | `db.embed(text)` once | server-side `db.record_text(...)` |
| `extracted_entities` | NER once on leader | engine NER (private path) |
| `created_at_unix_micros` | `SystemTime::now()` once | server-side timestamp |

Followers consume the materialized values. They MUST NOT re-embed,
re-NER, re-stamp timestamps, or assign a fresh rid. If they did,
`bge-base-en-v1.5` on .140 vs `bge-base-en-v1.6` on .141 would silently
diverge HNSW state.

Engine support needed: a sibling API to `db.record()` that **takes a
caller-assigned `rid` and embedding** rather than generating them. This
PR ships that API plus the same shape for `tombstone`, edge upsert/
delete, and any other mutator the commit log carries.

The materializer-side work (`yantrikdb-server` PR 6.2) is already
shipped and tested against trait stubs. PR 6.4 wires `db.embed` →
`Embedder` trait and the engine NER → `EntityExtractor` trait. PR 6.4
also replaces `db.record(...)` calls in the Applier with
`db.record_with_rid(...)`. Without this engine PR, PR 6.4 cannot ship.

## Public API additions

All public, all additive. Existing methods remain unchanged.

### 1. `record_with_rid`

```rust
impl YantrikDB {
    /// Insert or upsert a memory at a caller-assigned rid.
    ///
    /// Contract:
    /// - **Idempotent on rid.** Duplicate calls with the same `rid` and
    ///   identical other fields succeed and produce identical engine
    ///   state (HNSW node + memory row). The second call is a no-op.
    /// - **Different content same rid → upsert semantics.** Replaces
    ///   text/metadata/embedding in place, updates HNSW node.
    /// - **Caller supplies the embedding.** Engine does not call its
    ///   own embedder. If `embedding.len() != self.embedding_dim()`,
    ///   returns `Error::EmbeddingDimensionMismatch`.
    /// - **Caller supplies created_at.** Engine stamps it as the row's
    ///   creation timestamp; subsequent update calls preserve the
    ///   original. (This matches the leader's "stamp once, replicate"
    ///   property.)
    /// - **No NER inside.** Engine receives `extracted_entities` from
    ///   the caller and writes the entity_edges rows accordingly.
    ///   Empty `extracted_entities` means no edges; engine does NOT
    ///   fall back to its own NER.
    ///
    /// Returns `Ok(())` — the rid is the input, not the output.
    pub fn record_with_rid(
        &self,
        rid: &str,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        embedding: &[f32],
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
        created_at_unix_micros: i64,
        extracted_entities: &[&str],
        embedding_model: &str,
    ) -> Result<(), Error>;
}
```

Same parameter pack as today's `record()` minus the rid+embedding
return path, plus the four new materialized-state inputs (`created_at_unix_micros`,
`extracted_entities`, `embedding_model`).

### 2. `tombstone_with_rid`

Today's `forget(rid)` already takes a rid. Confirm semantics:
- Idempotent: calling twice with the same rid succeeds both times.
- Carries an optional reason + requested_at_unix_micros so the apply
  matches `MemoryMutation::TombstoneMemory`.

```rust
impl YantrikDB {
    pub fn tombstone_with_rid(
        &self,
        rid: &str,
        reason: Option<&str>,
        requested_at_unix_micros: i64,
    ) -> Result<(), Error>;
}
```

If `forget()` already does this, alias it; if not, ship as a new
method and keep `forget()` calling into it for back-compat.

### 3. `upsert_entity_edge_with_id`

Today's `relate(src, dst, rel, weight)` returns a generated edge_id.
Need a sibling that accepts a caller-assigned `edge_id`:

```rust
impl YantrikDB {
    pub fn upsert_entity_edge_with_id(
        &self,
        edge_id: &str,
        src: &str,
        dst: &str,
        rel_type: &str,
        weight: f64,
        namespace: &str,
    ) -> Result<(), Error>;
}
```

Idempotent on edge_id. Duplicate write with identical fields is a no-op.

### 4. `delete_entity_edge_with_id`

```rust
impl YantrikDB {
    pub fn delete_entity_edge_with_id(&self, edge_id: &str) -> Result<(), Error>;
}
```

Idempotent: deleting a non-existent edge_id returns `Ok(())`, not an
error. (The Applier may legitimately replay a delete that already
happened — snapshot-install + log replay overlap.)

## Schema-side changes

`memories` table needs the new columns from §3 of RFC 010 PR-6.2's wire
1.1 bump:

| Column | Type | Default | Source |
|---|---|---|---|
| `embedding_model` | TEXT | `NULL` | mutation.embedding_model |
| `created_at_unix_micros` | INTEGER | (existing `created_at`?) | mutation.created_at_unix_micros |

Engine migration (yantrikdb-side `m_engine_NN.sql` or whatever the
naming is):
- ADD COLUMN `embedding_model TEXT` if missing.
- Populate `embedding_model` for existing rows with the cluster default
  model id (one-shot UPDATE during migration).

`entity_edges` table doesn't need changes — caller-assigned edge_id
fits the existing primary key shape.

## Determinism contract (test surface)

The engine MUST add tests that pin determinism. PR 6.4's
end-to-end RYW test depends on these passing:

```rust
#[test]
fn record_with_rid_is_byte_deterministic() {
    let db = open_test_db();
    let embedding = vec![0.1, 0.2, 0.3, /* ... */];
    db.record_with_rid(
        "rid_test_1",
        "the quick brown fox",
        "semantic",
        0.5, 0.0, 86400.0,
        &serde_json::json!({}),
        &embedding,
        "test", 1.0, "general", "test",
        None,
        1_700_000_000_000_000,
        &["fox"],
        "test-model.v1",
    ).unwrap();

    let row = db.get("rid_test_1").unwrap().unwrap();
    assert_eq!(row.rid, "rid_test_1");
    assert_eq!(row.created_at_unix_micros, Some(1_700_000_000_000_000));
    assert_eq!(row.embedding_model.as_deref(), Some("test-model.v1"));
    // ... pin every field
}

#[test]
fn record_with_rid_is_idempotent_on_replay() {
    let db = open_test_db();
    // First call writes the row.
    db.record_with_rid("rid_test_2", "x", /* ... */).unwrap();
    let snap1 = db.snapshot_for_test();
    // Replaying produces the same engine state.
    db.record_with_rid("rid_test_2", "x", /* ... */).unwrap();
    let snap2 = db.snapshot_for_test();
    assert_eq!(snap1, snap2, "double-apply must be byte-identical");
}

#[test]
fn record_with_rid_rejects_dimension_mismatch() {
    let db = open_test_db_with_dim(384);
    let bad = vec![0.0; 100];
    let err = db.record_with_rid("rid", "x", /* ... */, &bad, /* ... */).unwrap_err();
    assert!(matches!(err, Error::EmbeddingDimensionMismatch { .. }));
}
```

Plus equivalents for tombstone / edge upsert / edge delete.

## Non-goals (this PR)

- **Bulk import path.** `record_with_rid` is the per-row primitive.
  Bulk variants can come later; PR 6.4 doesn't need them.
- **Replacing existing `record()`.** The current API stays. v0.7.x
  keeps both APIs side-by-side. Eventually `yantrikdb-server` can
  retire its remaining `record()` callsites in favor of
  `record_with_rid`, but that's incremental — not in this engine PR.
- **Embedder model migration.** RFC 013 (HNSW lifecycle + embedder
  model migration) handles the upgrade-the-cluster-embedder problem.
  This PR just stamps `embedding_model` so RFC 013 has the data.
- **Cross-tenant logic.** Engine is per-tenant; tenant-fanout is the
  server's job.

## Acceptance gates

- All existing engine tests pass (no regressions).
- New tests for the determinism + idempotency contracts above.
- `record_with_rid` benchmarked at the same throughput as `record()`
  modulo the embedder skip — caller-supplied embedding is faster, not
  slower.
- `yantrikdb-server` 0.8.x can compile against engine 0.7.0 with no
  changes (the new APIs are additive).

## Estimated effort

~6–8h on the engine side: 4 new public methods + 1 schema migration +
~10 deterministic-tests + benchmarks. The hard part is the schema
migration (preserve `embedding_model` on UPDATE; one-shot backfill).

PR 6.4 in `yantrikdb-server` cannot start until this engine PR is
merged + tagged + the server's `Cargo.toml` bumps `yantrikdb = "0.7.0"`.

## Wire-up checklist (server-side, after engine 0.7.0 lands)

This part lives in `yantrikdb-server` PR 6.4 — recorded here so the
engine reviewer knows what's downstream:

1. Bump `Cargo.toml`: `yantrikdb = { version = "0.7.0", git = "...", branch = "main" }`.
2. Implement `Embedder` trait for `Arc<yantrikdb::YantrikDB>` (delegates
   to `db.embed`). Wire into `LocalMaterializer`.
3. Implement `EntityExtractor` trait (delegates to engine NER, or stubs
   with empty Vec if NER isn't directly exposed).
4. Implement `Applier` trait for `LocalApplier`:
   - `MemoryMutation::UpsertMemory` → `db.record_with_rid(...)`
   - `MemoryMutation::TombstoneMemory` → `db.tombstone_with_rid(...)`
   - `MemoryMutation::UpsertEntityEdge` → `db.upsert_entity_edge_with_id(...)`
   - `MemoryMutation::DeleteEntityEdge` → `db.delete_entity_edge_with_id(...)`
   - `MemoryMutation::UpdateMemoryPatch` / `PurgeMemory` /
     `TenantConfigPatch` stay as `NotYetWired` until their owning RFCs
     ship.
5. Migrate `Command::Remember` (and friends) handler arms to call
   `submitter.submit(...)` instead of `db.record(...)`.
6. End-to-end RYW test (the moment-of-truth): write to leader, recall
   on follower within 5s p99 across 1000 writes on a chaos network.

## Reviewers

- Engine repo: whoever owns `yantrikdb` (likely Pranab + architect)
- Server repo: parallel review on the PR 6.4 implementation that
  consumes this API

## Open questions

1. **`tombstone_with_rid` vs existing `forget`.** Are they actually
   the same operation? If so, this PR just documents the contract
   (idempotent, accepts reason + requested_at). If not, ship both
   and define when to use which.
2. **NER public API.** Does the engine currently expose its NER at all,
   or is it only invoked internally during `record()`? If internal-only,
   the `EntityExtractor` trait impl on the server side becomes a stub
   that returns empty Vec — and we'd need a follow-up engine PR to
   expose NER as a public method. The materialized mutation still
   works (empty extracted_entities is valid wire 1.1), but determinism
   for entity edges is contingent on this being callable from the
   leader.
3. **Embedding model identifier shape.** `String` for now; RFC 013 may
   want a richer type (`(model_family, version, dim)`). Decide the
   shape during engine PR review.

---

**Bottom line:** this is the long-pole engine PR for cluster mode
actually working. Server-side PR 6.1 + 6.2 already shipped (trait
shapes locked); PR 6.3 (per-tenant log layout) is server-only and
parallels this work; PR 6.4 (the moment-of-truth handler migration)
hard-blocks on this PR landing.

Once this engine PR is merged + the server picks up `yantrikdb 0.7.0`,
PR 6.4 is unblocked. End-to-end replication starts working at v0.8.13.
