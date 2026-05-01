# Changelog

All notable changes to `yantrikdb-server` are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.5] — 2026-05-01

`/v1/health` cleanup. Fixes a misleading status shown to operators
running `cluster.raft_mode = "openraft"`.

### Fixed

- `/v1/health` now reflects **openraft state** when openraft mode is
  active (`accepts_writes` true on the openraft leader, `term` and
  `leader` from `RaftMetrics`, etc.). Previously it always reported
  the legacy raft-lite view, which on a healthy openraft cluster
  shows `accepts_writes: false` (because the raft-lite layer never
  successfully forms a quorum once openraft is the real write path).

  Discovered during the issue #28 cluster reform when both nodes of
  a verified-working 2-voter openraft cluster reported
  `accepts_writes: false` to monitoring scrapes — false alarm,
  cluster was healthy.

  The legacy raft-lite view is still available in the response when
  raft-lite is the configured mode (no openraft assembly active).
  New `raft_mode` field on the cluster block disambiguates.

### Notes for operators

If you were filtering on `cluster.accepts_writes` in a Grafana panel,
no change needed — the value is now correct (true on openraft leader)
instead of always-false on healthy openraft clusters.

[0.8.5]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.5

## [0.8.4] — 2026-05-01

Critical hotfix. Without this, v0.8.2/v0.8.3 openraft clusters
become unresponsive under live multi-voter replication load.

### Fixed

- **#27** Resolved properly: `SplitRuntime` (RFC 009 §4 Layer 1
  CPU-isolated control-plane runtime) is back. The v0.8.2/v0.8.3
  workaround (cluster + HTTP sharing one tokio runtime) caused HTTP
  gateway starvation when openraft replication traffic ramped up
  on `cluster_port` 7440. Symptom: `.141` HTTP gateway became
  unresponsive within ~30 sec of a follower joining the cluster,
  cluster connections backing up to ~46 in the listen queue while
  HTTP requests timed out.

  Fix: converted `fn main()` from `#[tokio::main]` to manual sync
  main. The macro builds an outer Runtime, and any nested Runtime
  built inside async context panics on drop. Sync main owns the
  tokio runtime explicitly, runs `async_main()` via `block_on()`,
  and drops both the main runtime and `SplitRuntime` from sync
  context where blocking is permitted.

### Migration

- **From v0.8.3**: rolling restart on each node. No config changes.
  `SplitRuntime` re-engages automatically when `cluster.raft_mode =
  "openraft"` and `cluster.role = "voter"`.
- **From v0.8.2 / v0.8.1 / v0.7.x**: same as v0.8.3's migration
  procedure (see v0.8.3 changelog) — this hotfix only changes the
  runtime architecture, not the on-disk format or wire protocol.

[0.8.4]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.4

## [0.8.3] — 2026-05-01

Operator surface for openraft cluster mode. Without this, `v0.8.2`
deployments could only run *single-node* openraft because there was
no way to add additional voters without writing Rust against the
openraft API directly. **Closes the cluster mode story for v0.8.x.**

### Added

- **#24** Cluster membership HTTP API + CLI:
  - `POST /v1/cluster/initialize` — bootstrap a fresh openraft cluster
    on the seed node (one-time call per cluster).
  - `POST /v1/cluster/add-learner` — add a non-voting learner.
    Catches up via openraft snapshot transfer without participating
    in elections.
  - `POST /v1/cluster/promote-voter` — change voter set (promotes
    learners, demotes voters not in the new set).
  - `POST /v1/cluster/remove` — atomic remove. Refuses if removal
    would empty the voter set.
  - All endpoints require the cluster master token.
  - CLI subcommands mirror each: `yantrikdb cluster initialize-cluster`,
    `add-learner`, `wait-caught-up`, `promote-voter`, `remove-node`.

### Changed

- Removed the v0.8.2 auto-bootstrap heuristic ("node_id 1 or 2 =
  seed"). Replaced by explicit `cluster initialize-cluster` CLI.
  Operators on v0.8.2 with already-initialized membership are
  unaffected (openraft persists membership across restarts).

### Migration

For operators on a v0.7.x raft-lite cluster moving to v0.8.3
openraft, the procedure is now:

1. Generate cluster mTLS certs (CA + per-node cert+key).
2. Stop all nodes; install v0.8.3 binary; update each toml
   (`raft_mode = "openraft"` + `[cluster_tls]` + `dev_mode = true`
   for self-signed).
3. Wipe legacy `raft.json` (if present) — leave engine state intact.
4. Bring up the seed node. Run:
   ```
   yantrikdb cluster initialize-cluster --leader http://seed:7438 \
     --master-token "$YDB_CLUSTER_MASTER_TOKEN"
   ```
5. For each additional voter:
   - **Recommended**: pre-stage engine state via cold-tar from leader
     (~30 s for a 400 MB data dir vs ~30 min for openraft 0.9's
     `full_snapshot` over the wire).
   - Bring the new node up.
   - From the leader, run `yantrikdb cluster add-learner --node-id N
     --addr host:7440 --leader http://leader:7438`.
   - `yantrikdb cluster wait-caught-up --node-id N --leader ...`.
   - `yantrikdb cluster promote-voter --voters 2,N --leader ...`
     (include the leader's id in the final voter set).

The "cold-tar pre-stage" trick avoids openraft 0.9's slow
`full_snapshot` path. Issue #25 will replace it with chunked
`install_snapshot` in v0.9.0-beta — at which point pre-staging
becomes optional.

[0.8.3]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.3

## [0.8.2] — 2026-04-30

Patch release. Single critical fix:

### Fixed

- **#26** `cluster.raft_mode = "openraft"` panicked at startup because
  rustls 0.23+ requires a process-level `CryptoProvider` and neither
  `aws-lc-rs` nor `ring` features were enabled in `Cargo.toml`, and
  `install_default()` was never called. Single-node deployments
  (`raft_mode` unset / `disabled`) were unaffected since they skip
  rustls initialization entirely.

  Fix: enable `aws-lc-rs` feature on `rustls`, install the default
  provider at the very start of `fn main()`. Two-line change.

  Discovered 2026-04-30 during attempted cluster reform from
  raft-lite to openraft on a 2-voter homelab cluster.

### Notes for operators

- All v0.8.x releases prior to v0.8.2 are unsuitable for openraft
  mode. If you're on `raft_mode = "disabled"` (the default), you
  can upgrade at your own pace.
- If you tried to follow `docs/migration/v0.7_to_v0.8.md` Option B
  (raft-lite → openraft) on v0.8.0/v0.8.1 and hit a panic, this
  release is what unblocks it.

[0.8.2]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.2

## [0.8.1] — 2026-04-30

Operational patch. Fixes two bugs reported by `yantrikdb-agi` after
two blocking incidents on 2026-04-29 (21:00 UTC + 01:00 UTC) where
`/v1/recall` returned 500 for the entire `skill_substrate` namespace.

### Fixed

- **#19** `/v1/remember` and `/v1/remember/batch` could silently store
  rows with `embedding=NULL` when the embedder service hiccupped
  (timeout, transient ONNX runtime failure, etc.). The endpoint
  returned 200 with a normal `{rid: ...}` response while the row was
  effectively broken — and a single NULL-embedding row poisoned every
  subsequent `/v1/recall` on the namespace with
  `database error: Invalid column type Null at index: 1, name: embedding`.
  Both handlers now pre-embed in the server (using the existing
  `FastEmbedder` + cache wired in v0.8.0) before delegating to the
  engine. On embedder failure, the request returns 5xx synchronously
  so the caller can retry. The batch handler runs misses through one
  coalesced `embed_batch` call so concurrent ONNX-mutex acquisitions
  remain efficient. New Prometheus counter:
  `yantrikdb_embedder_failures_total{handler}`.

- **#20** No proactive surface for detecting `memories` rows with
  `embedding IS NULL`. Operators only discovered them when a recall
  failed. Added an hourly background healthcheck that counts
  NULL-embedding rows per tenant and emits
  `yantrikdb_null_embedding_count{tenant}` Prometheus gauge.
  Non-zero values trigger a `tracing::warn!` line with the SQL one-liner
  to remediate. Should be 0 in steady state on v0.8.1+ deployments;
  non-zero indicates pre-v0.8.1 stale data or a regression to flag.

### Deferred

- **#18** Leader affinity / preferred-leader pinning was investigated
  for v0.8.1 inclusion but blocked: openraft 0.9.24 (the version we
  pin) does not expose `transfer_leader` — that's a 0.10 feature.
  Bundling the openraft 0.10-alpha upgrade with leader-affinity is
  the right path; it lands in v0.9.0. See issue #18 comments for the
  blocker analysis. Operational workaround unchanged: manual CT
  bounce + the tuned watchdog (CHECK_INTERVAL=60, HANG_RECHECK=120,
  MAX_HANGS=5) keeps the bounce contained.

### Notes for operators

- Existing pre-v0.8.1 databases may have NULL-embedding rows from
  prior writes. After upgrading, watch `yantrikdb_null_embedding_count`
  for an hour; if non-zero, run
  `DELETE FROM memories WHERE embedding IS NULL` against the affected
  tenant DB to clean up. The fix prevents new ones; cleanup is one-shot.

[0.8.1]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.1

## [0.8.0] — 2026-04-29

Substrate-first release. Twelve RFC interface layers shipped as a single
batch, plus an embedder-cache perf win and two user-reported bug fixes.

### Highlights for operators

- **Embedder cache live** — repeat queries skip the ONNX forward pass.
  Recall p50 drops measurably on warm workloads (~10× on
  cache-hit-dominated traffic). No config required; the cache is on by
  default with a 10 000-entry capacity (~15 MB).
- **Two operator-reported bugs fixed**: encryption env var honored,
  snapshot endpoint usable in single-node mode.

### Added — substrate (interface + reference impl + tests; consumer
wiring deferred to follow-up PRs)

- **RFC 009 admission control** (saga #128, #129, #131):
  - `src/admission/cost.rs` — cost function
    (`top_k × E_expand`, `E_expand=5` default).
  - `src/admission/bucket.rs` — token bucket on monotonic clock,
    `startup_warm_fraction=0.25` default to mitigate rolling-restart
    thundering herd.
  - `src/admission/policy.rs` — `QuotaPolicy` + `QuotaScope`
    (Principal/Namespace/Global) + `PolicyResolver` trait,
    `PROVISIONAL_DEFAULTS` (rps=100, cost=1000/s, exp_cc=4) and
    `FALLBACK_DEFAULTS` (half-throughput) for control-DB outage.
  - `src/admission/registry.rs` — per-(scope, dimension) bucket
    registry, lazy-on-first-consume materialization, `ConsumeOutcome`
    in SHADOW mode.
  - `src/admission/circuit_breaker.rs` — pure state machine with 4
    triggers (term churn / active election / scheduling latency /
    sustained heartbeat lag), hysteresis, and anti-flapping.
    Pinned gauge values 0/1/2 for Closed/Open/HalfOpen.
  - `src/admission/deadlines.rs` — `RecallStage` enum + per-stage
    deadline budget. RFC defaults: total=5000ms, expansion=2000ms.
    `run_with_deadline_or_cancel` races future, deadline, and
    cancellation token.
  - `src/admission/retry_budget.rs` — per-tenant token-bucket retry
    budget + AWS full-jitter `Retry-After` computation.

- **RFC 011 forget — crypto-shred** (saga #146):
  - `src/forget/crypto_shred.rs` — `CryptoShredder` consumes
    `KeyProvider`. Destroys data + backup keys on tenant delete.
    Verified contracts: half-shred protection (all versions destroyed),
    purpose isolation (TLS / signing keys preserved), tenant
    isolation, idempotence.

- **RFC 012 backup/restore — restore command** (saga #148):
  - `src/restore/validate.rs` — `RestoreValidator`. Catches
    resurrect-tombstoned-data, wire major mismatch, HNSW model
    mismatch, checksum mismatch, missing content blobs.
  - `src/restore/exec.rs` — `RestoreExecutor` with three modes
    (`NewCluster` / `SingleTenant` / `WipeAndRestore`), atomic
    tmp+rename writes, marker file for crash recovery, sanitizes `/`
    and `\` in model names.

- **RFC 013-B shadow-index migration** (saga #152):
  - `src/index/hnsw/shadow.rs` — phase machine
    (Idle → Backfilling → DualRead → Cutover → Complete) with
    caught-up gate. `DualReadMerger` for score-normalized merge during
    DualRead. `MigrationStateStore` trait + in-memory ref impl.

- **RFC 014-B Auth/RBAC** (saga #153):
  - `src/auth/scopes.rs` — typed `Scope` bitset
    (Read/Write/Recall/Forget/Admin/TenantManagement). Pinned wire
    forms; CSV parse/format.
  - `src/auth/principal.rs` — `Principal` + `AuthOutcome`
    (Authenticated / Unauthenticated / Revoked / Expired —
    distinguished so audit operators can grep revoked-token usage).
  - `src/auth/provider.rs` — `AuthProvider` async trait + reference
    in-memory impl with hash-at-rest, revoke, expiry.
  - `src/auth/audit.rs` — `AuditEvent` + `AuditSink` trait +
    `InMemoryAuditSink` ring buffer.

- **RFC 014-C KeyProvider** (saga #154):
  - `src/key_provider/{mod,local}.rs` — `KeyProvider` async trait
    (object-safe), four `KeyPurpose`s (TenantDataEncryption /
    BackupBlobEncryption / ClusterTls / AuditSigning),
    `LocalKeyProvider` reference impl. Drop-zeroize on `KeyMaterial`
    (best-effort defense).

- **RFC 015-B-2 hybrid retrieval** (saga #157, LongMemEval lever):
  - `src/retrieval/bm25.rs` — `BM25Index` trait + `InMemoryBM25Index`
    reference impl with k1=1.2, b=0.75. TF saturation, longer-doc
    penalty, IDF rare-term reward verified by tests.
  - `src/retrieval/hybrid.rs` — `rrf_merge` pure function (Cormack
    2009 RRF, k=60 default).
  - `src/retrieval/rerank.rs` — `Reranker` async trait +
    `IdentityReranker`. Production ONNX cross-encoder backend
    deferred.

- **RFC 021 config versioning** (sagas #165, #166):
  - `src/config/versioned.rs` — `ConfigVersion`, `VersionedConfig<T>`,
    `ConfigDelta<T>`.
  - `src/config/live_reload.rs` — `Reloadable` trait with
    stale-delta detection and replay-safe ignore semantics.
  - `src/config/watch.rs` — `ConfigWatchSender` / `ConfigWatch` over
    `tokio::sync::watch` with latest-value semantics.
  - `src/config/tenant_overrides.rs` — `TenantConfigOverride` +
    `TenantConfigStore` trait with strict-monotonic upsert.

- **RFC 007 Socratic operators** (saga #102):
  - `src/socratic/{evidence,operator}.rs` — six typed operators
    (BinaryToConditional, GlobalToTemporal,
    PropositionToSourceComparison, OutcomeToUpstreamLevers stub,
    EntityDisambiguation, ContextCompletion). Operator selection is
    deterministic + graph-grounded; LLM (if used) only paraphrases.

### Added — embedder cache wiring (commit `e52228e`)

- `FastEmbedder` now consults `EmbeddingCache` before acquiring the
  ONNX `Mutex`. Cache hits skip both the lock and the forward pass.
- `embed_batch` partitions inputs into hits and misses, then runs ONE
  batched ONNX call for all misses (single mutex acquisition).
- `cache_hits()`, `cache_misses()`, `cache_hit_rate()` accessors for
  future Prometheus wiring.
- Cache key includes the model version so RFC 013-B shadow migrations
  auto-invalidate on model change.

### Fixed

- **#6** `YANTRIKDB_ENCRYPTION_KEY_HEX` env var was silently ignored;
  encryption only enabled when key was in the TOML `[encryption]
  key_hex` field. Now `EncryptionSection::resolve_key()` checks the
  env var first (priority 0), TOML second. Operators can now run with
  `-e YANTRIKDB_ENCRYPTION_KEY_HEX=<hex>` per the documented setup.
- **#7** `POST /v1/admin/snapshot` rejected requests in single-node
  mode with `snapshot requires cluster master token` because the
  cluster master token doesn't exist outside cluster mode. Now
  accepts EITHER the cluster master token (cluster mode) OR a
  per-database token authenticating the SAME database named in the
  request body (any mode).

### Tests

- 739 unit tests + 11 integration test binaries — all green.
- ~310 new tests added across the substrate. Each substrate ships
  contract pin-tests that survive backend swaps (e.g.
  `InMemoryBM25Index` tests will validate the future `tantivy`-backed
  impl verbatim).

### Migration

See `docs/migration/v0.7_to_v0.8.md`.

Most users have no client-visible breaking changes. Cluster operators
migrating from raft-lite (Phase 1) to openraft must reform the
cluster — see migration doc for procedure.

[0.8.0]: https://github.com/yantrikos/yantrikdb-server/releases/tag/v0.8.0
