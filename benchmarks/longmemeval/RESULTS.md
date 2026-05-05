# LongMemEval Reference Run — YantrikDB

**Date**: 2026-04-29
**YantrikDB build**: `yantrikdb-phase2-local` (built from this branch's `crates/`)
**Embedder**: builtin all-MiniLM-L6-v2 (384-dim) via fastembed + ONNX Runtime CPU EP
**Host**: Windows 11, Docker Desktop, Linux VM backend
**Config**: `benchmarks/longmemeval/yantrikdb_single.toml` (no cluster, builtin embedder)
**Recall settings**: `expand_entities = false`, top_k = 20 (scored at @1/@3/@5/@10/@20)

## What this run measures

500 questions from LongMemEval **oracle** variant. The oracle variant
contains *only* the evidence sessions per question (no distractors), so
this is the **upper-bound validation** of:

- Embedder quality (does MiniLM find the right turn within known-good sessions?)
- HNSW recall at small index sizes (~36 memories per question)
- Per-question namespace isolation (no bleed across question scopes)

The **headline benchmark is `_s`** (278 MB, ~40 sessions per question
including distractors), which is what Zep / Memento / Mastra report
against. Oracle establishes the harness-correctness floor; `_s` measures
discrimination under realistic noise.

## Headline metrics

- Questions scored: **500 of 500** (0 errors)
- Total runtime: **741.5 s** (~12.4 minutes)
- Recall p50 / p95 / max: **33 ms / 69 ms / 100 ms**

### Turn-level recall@K

| Question type | n | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|---|
| single-session-assistant | 56 | 37.5% | 75.0% | **85.7%** | 100.0% | 100.0% |
| temporal-reasoning | 133 | 42.1% | 71.4% | **83.5%** | 91.7% | 95.5% |
| single-session-user | 70 | 48.6% | 78.6% | **81.4%** | 88.6% | 90.0% |
| knowledge-update | 78 | 38.5% | 70.5% | **80.8%** | 89.7% | 92.3% |
| multi-session | 133 | 39.1% | 66.2% | **76.7%** | 89.5% | 94.0% |
| single-session-preference | 30 | 50.0% | 70.0% | **73.3%** | 93.3% | 100.0% |
| **Overall** | **500** | **41.6%** | **71.2%** | **80.6%** | **91.4%** | **94.6%** |

### Session-level recall@K

100% across every K and every question type — by construction, since
oracle has no distractor sessions.

## Reading the numbers

- **80.6% turn-level recall@5 overall** is the substrate's pure-embedder
  ceiling: with no distractor noise, MiniLM + HNSW finds the
  answer-bearing turn in top 5 80.6% of the time. The 19.4% that miss
  at @5 are typically caught at @10 (91.4%) or @20 (94.6%) — meaning
  the right turn is *retrieved* but ranked outside top 5.
- **Single-session-preference is hardest** (73.3% @5) — preference
  signals like "I prefer X" are subtle for cosine similarity search.
- **Single-session-assistant is easiest** (85.7% @5, 100% @10) —
  assistant turns tend to contain the answer text near-verbatim.
- **Latency p95 = 69 ms** with no rerank, no entity expansion. The
  cross-encoder rerank stage from RFC 015-B (not yet shipped) would
  consume most of the remaining recall@K headroom by re-scoring the
  top-20 against the question text — a known-good lift for systems at
  this embedder + index quality.

## `_s` subset run (2026-04-29, 50 questions of single-session-user)

A 50-question subset of `_s` was run on a fresh container to gauge
discrimination under distractor noise. The dataset is grouped by
question type, so the first 50 questions are all
`single-session-user`. **This is one type, not a representative
sample of all 500 questions.**

| Metric (single-session-user) | oracle | `_s` subset | Δ |
|---|---|---|---|
| Questions scored | 70 | 48 of 50 | (2 transient write blips, retry now patched) |
| Turn-recall@5 | 81.4% | **75.0%** | -6.4 pp |
| Turn-recall@10 | 88.6% | 89.6% | ~ |
| Session-recall@5 | 100% (by construction) | **97.9%** | -2.1 pp |
| Recall p95 | 69 ms | 92 ms | +33% |
| Mean write phase per question | 1.4 s | 40.6 s | 29× |

**Reading these numbers:** under realistic distractor noise (~50
sessions per question, mostly off-topic), yantrikdb's substrate still
hits 75% turn-recall@5 and 98% session-recall@5 on
single-session-user. Latency stays under 100 ms p95 even with the
larger per-namespace index. The 6-point drop in turn-recall@5
between oracle and `_s` is the cost of noise discrimination — that
gap is what RFC 015-B's cross-encoder rerank stage is designed to
close.

**Full `_s` run extrapolation:** 500 × ~42 s/question = **~5.9 hours**.
Beyond a single-session "kick off and wait" budget — defer to a
fresh-context session that can dedicate uninterrupted runtime.

## What's next

1. **Run `_s` variant** (278 MB, ~40 sessions per question with
   distractors). This is the comparable benchmark vs Zep / Memento /
   Mastra. Estimated runtime: ~5.9 hours per the subset extrapolation.
   Run with the retry-patched harness so transient single-write blips
   don't lose questions.
2. **Then `_m`** (2.75 GB, ~500 sessions per question) for scale
   stress.
3. **QA-correctness scoring** via GPT-4o judge (LongMemEval's
   `evaluate_qa.py`) layered on top of retrieval. Adds an external API
   dependency; separable from substrate measurement.
4. **RFC 015-B cross-encoder rerank** is the technical lever to lift
   recall@K. Not yet shipped; current numbers reflect raw
   embedder + HNSW only.

## Reproduction

```bash
cd benchmarks/longmemeval
docker compose up -d
MSYS_NO_PATHCONV=1 docker exec yantrikdb-lme yantrikdb db --data-dir /var/lib/yantrikdb create default
TOKEN=$(MSYS_NO_PATHCONV=1 docker exec yantrikdb-lme yantrikdb token \
  --data-dir /var/lib/yantrikdb create --db default --label lme | grep '^ydb_')
python fetch.py --variant oracle
YDB_TOKEN=$TOKEN python run.py --variant oracle --port 18438
python metrics.py results_oracle.jsonl
```

## Files

- `fetch.py` — downloads LongMemEval variants from HuggingFace
- `run.py` — per-question harness (write turns, recall, score)
- `metrics.py` — aggregates results_<variant>.jsonl into per-type tables
- `docker-compose.yml` — single-node yantrikdb (ports 18437/18438)
- `yantrikdb_single.toml` — server config
- `results_<variant>.jsonl` — raw per-question scores (one row per question)
