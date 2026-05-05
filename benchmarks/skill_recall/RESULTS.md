# Reference Run — Skill-Recall Benchmark

**Date**: 2026-04-29
**YantrikDB version**: ghcr.io/yantrikos/yantrikdb:latest (single-node)
**Embedder**: builtin all-MiniLM-L6-v2 (384-dim)
**Host**: Windows 11, Docker Desktop, Linux VM backend
**Config**: `yantrikdb_single.toml` (no cluster, builtin embedder, max_databases=4)

This is the canonical run that the README headline numbers cite. The bench
is deterministic (seed=42); your numbers should match within a few percent
on comparable hardware.

## Phase 2 re-baseline (2026-04-29, commit c886e9e)

After shipping RFC 015-A (cache substrate), RFC 011-B (TombstoneIndex),
RFC 013 Phase 1 (HNSW manifest), and RFC 010 PR-4 (full openraft adoption
in 5 sub-PRs + P0 hardening + P1 operator surface + live wiring + P2
correctness depth), the same deterministic bench was re-run against an
image built from the Phase 2 source tree (`yantrikdb-phase2-local`):

| Metric                       | Pre-Phase-2 baseline | Phase 2 build | Delta             |
|------------------------------|----------------------|---------------|-------------------|
| Writes (5000 skills)         | 26.1/sec, 0 fails    | 26.1/sec, 0 fails | identical     |
| Recall p50                   | 87.4 ms              | 87.3 ms       | -0.1 ms (noise)   |
| Recall p95                   | 114.9 ms             | **106.3 ms**  | **-7.5%**         |
| Recall max                   | 122.6 ms             | 119.9 ms      | -2.2%             |
| recall@5 overall             | 0.86                 | 0.86          | identical         |
| recall@5 family (broad)      | 0.96                 | 0.96          | identical         |
| recall@5 variant (sharp)     | 0.76                 | 0.76          | identical         |

**Conclusion**: Phase 2 substrate work (commit log + state machine
observer hooks + tombstone index + HNSW manifest + openraft assembly) is
regression-neutral on both write throughput and recall quality. The p95
latency improvement (~8 ms tighter) is within run-to-run variance but
consistent with the Phase 2 work touching only commit-path and read
paths whose hot loops weren't substantially changed. Recall@K numbers
are bit-identical because no embedding model, distance metric, HNSW
parameter, or recall-path reranker changed between baselines. Numbers
that would move (BM25 hybrid, cross-encoder rerank) are RFC 015-B work
and are not yet shipped.

**Known issue surfaced during this re-baseline**: `yantrikdb token
create` CLI claims success but does not persist the token row to
control.db; auth probe returns "invalid or revoked token" against a
freshly-minted token. Direct `INSERT INTO tokens` via sqlite3 +
container restart works (the server-side auth path is correct). Issue
filed for investigation; it does not affect the benchmark numbers
above (the bench was run against a directly-injected token).

---

## Corpus

- 5000 skill records
- 50 topics × 10 actions × 10 variants
- Each record ~600 chars body + structured metadata (skill_id, applies_to,
  triggers, success/failure counters, version, status)

## Write phase

```
seeding 5000 skills (range 0..5000) at 50/sec target rate
DONE: written=5000 failed=0 elapsed=191.8s effective_rate=26.1/s
```

Server-side embedding is the bottleneck: requested 50/sec, sustained
26.1/sec. The server processed every write without backpressure failures.

## Recall phase — 100 queries, top_k=10

```jsonc
{
  "n_queries": 100,
  "n_failed": 0,
  "wall_clock_sec": 7.78,
  "latency_ms": {
    "mean":   77.8,
    "median": 87.4,
    "p95":   114.9,
    "max":   122.6
  },
  "recall_overall":         { "@1": 0.71, "@3": 0.79, "@5": 0.86, "@10": 0.91 },
  "recall_family_level":    { "@1": 0.72, "@3": 0.88, "@5": 0.96, "@10": 0.98 },
  "recall_variant_specific":{ "@1": 0.70, "@3": 0.70, "@5": 0.76, "@10": 0.84 }
}
```

### Latency distribution

| Percentile | Latency |
|---|---|
| mean | 77.8 ms |
| p50  | **87.4 ms** |
| p95  | 114.9 ms |
| max  | 122.6 ms |

100% of queries returned within 125 ms; p95 was 115 ms.

### Recall@K — three cohorts

| Cohort | recall@1 | @3 | @5 | @10 |
|---|---|---|---|---|
| Overall (mixed) | 0.71 | 0.79 | **0.86** | 0.91 |
| Family-level — broad: any of 10 sibling variants | 0.72 | 0.88 | **0.96** | 0.98 |
| Variant-specific — sharp: this exact skill out of 5000 | 0.70 | 0.70 | **0.76** | 0.84 |

The family cohort (96% @5) reflects the realistic agent use case: "I'm in
the recipe-authoring topic, surface anything relevant from that family."
The variant cohort (76% @5) is the harder ablation: "find this one specific
skill out of 5000, with 49 confusably-similar siblings sharing vocabulary."
The 76% sharp-recall is the **pure-semantic floor** — adding structured
filters (`applies_to=...`, `min_success_count=N`, trigger keyword overlap)
is expected to lift sharp-recall toward the family number. That filter
work is tracked separately.

## Cache phase — exact-id lookup via SQLite side-cache

```jsonc
{
  "n_indexed": 5001,
  "populate_secs": 1.92,
  "cache_db_size_kb": 4244.0,
  "n_lookups": 1000,
  "hits": 900,
  "misses": 100,
  "latency_us": {
    "mean":   79.1,
    "median": 79.1,
    "p95":    89.3,
    "max":   145.5
  }
}
```

Single bulk recall populates 5001 skills into a 4.2 MB SQLite side-cache
in 1.9 seconds. Subsequent `skill_id → body` lookups complete in **79
microseconds at the median**, three orders of magnitude faster than going
through the embedder. The 10% miss rate (100 of 1000 lookups for known-
nonexistent skill_ids) hits the same fast path because SQLite returns
NULL on PK miss in ~the same time as a hit.

## Implications

1. **YantrikDB is fit-for-purpose as a skill substrate at this scale.**
   Sub-200ms semantic recall, zero write failures, deterministic results.
2. **The 2-tier pattern works.** Use semantic recall when you don't know
   what you're looking for; use the SQLite side-cache when you do.
3. **At 5000 skills the disk footprint is trivial** (4.2 MB cache).
   Linear extrapolation: ~85 MB at 100k skills.
4. **Pure semantic search is not the whole story.** The 76% sharp-recall
   reveals the limit; structured metadata filters need to layer on top
   to reach production-grade discrimination. This is consistent with the
   architecture: YantrikDB stores typed records with rich metadata; the
   filter API just needs to expose them.

## Reproducing this run

```bash
cd benchmarks/skill_recall
docker compose up -d

# Mint a token (the running server auto-creates a 'default' database).
docker exec yantrikdb-skill-bench yantrikdb token \
  --data-dir /var/lib/yantrikdb create --db default --label bench
# (capture printed token)
export YDB_TOKEN=ydb_...

python generate.py 5000
python seed.py    --nodes 127.0.0.1 --port 17438 --rate-per-sec 50
python bench.py   --nodes 127.0.0.1 --port 17438
python cache_bench.py --nodes 127.0.0.1 --port 17438

docker compose down -v   # cleanup
```

Total runtime: ~3.5 minutes (mostly write phase).
