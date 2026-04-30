# Changelog

All notable changes to `yantrikdb-server` are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
