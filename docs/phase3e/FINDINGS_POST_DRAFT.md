# DRAFT — NOT PUBLISHED YET

**Status:** draft, written 2026-04-19. Do not publish before one sleep
+ morning review. Per pre-registered gate in
[`../demand-test/decision_rule.md`](../demand-test/decision_rule.md).

---

# Correction: my "structured memory vs markdown" benchmark was measured on a simulator, not yantrikdb. The rerun tells a more interesting story.

**The short version:**

1. I spent weeks running 5 phases of benchmarks on "agent memory" for
   [yantrikdb](https://github.com/yantrikos/yantrikdb-server), the
   cognitive memory database I'm building.
2. The project maintainer (me, the user of this tool) caught a
   methodology error mid-session: the condition labeled "structured
   memory" across Phase 3C and Phase 3D was actually a 120-line Python
   simulator with NO embeddings, NO HNSW, NO `think()` loop, NO
   multi-signal scoring. It was strictly weaker than a naive vector DB.
3. Reran with real yantrikdb via HTTP. The simulator-era conclusions
   are withdrawn.
4. The surprising part isn't that real yantrikdb is better. It's that
   **which yantrikdb feature earns the gain depends on your memory
   scale** — and at small scale, some features actively hurt.

## The error

The original Phase 3C writeup concluded plain structured memory loses
to a rolling markdown file (73% vs 60% answer accuracy on a synthetic
5-session supersession-heavy scenario). I pre-registered a "null
result on RFC 006 temporal substrate" based on that.

The condition I called "structured memory" was
[`memory_sim.py`](../phase3c/memory_sim.py) — a Python list of
`(key, value, session)` tuples, Dice word-overlap retrieval at query
time. 120 lines. No yantrikdb code path was ever invoked. The
simulator was a toy proxy that omitted every feature the product
actually has.

The maintainer asked during a consultation:
> "so then Yantrik functionality, the core functionality did not run
> at all. we just ran it as a vector db [or less]."

Correct observation. No defense.

Full writeup of the error and audit trail:
[`../../CORRECTIONS.md`](../../CORRECTIONS.md).

## The rerun (Phase 3E)

I built [`yantrikdb_client.py`](yantrikdb_client.py): a thin Python
wrapper around yantrikdb's HTTP endpoints that computes sentence-
transformer embeddings (`all-MiniLM-L6-v2`, 384-dim) client-side and
passes them to `/v1/remember` and `/v1/recall`. `think()` between
sessions via `/v1/think`.

Reran the Phase 3C scenario (5 sessions, 15 probes: 5 direct, 5
supersession, 3 branch-conditional, 2 alias-disambiguation). Same
seed, same probes, same Qwen 3.6 actor + Qwen judge.

**Same-scenario comparison (n=2 per cell):**

| condition | overall | answer | supersession | stale-rate | alias | prov |
|---|---|---|---|---|---|---|
| Simulator (original Phase 3C) | 0.584 | 0.600 | 0.50 | 0.40 | 0.50 | 0.94 |
| Markdown baseline | 0.667 | 0.733 | 1.00 | 0.00 | 0.00 | 0.82 |
| Real ydb (current DB) | 0.850 | 0.867 | 0.80 | 0.20 | 0.75 | 0.96 |

Real yantrikdb cuts the stale-error rate in half (0.40 → 0.20),
beats markdown on overall score (0.85 vs 0.67), and lifts answer
accuracy 27 points over the simulator. The original "plain
structured memory loses to markdown" conclusion is withdrawn.

## Which yantrikdb feature actually earned the gain? Ablation matrix.

This is the interesting part. I ran a 2×2 ablation:

- `think(on)` vs `think(off)` (is the consolidation/conflict-scan loop load-bearing?)
- current DB vs fresh DB (was Phase 2 entity-graph residue helping or hurting?)

**Ablation results (n=2 per cell, same scenario):**

| config | overall | answer | sup | stale | alias | direct | branch | prov |
|---|---|---|---|---|---|---|---|---|
| Simulator | 0.584 | 0.600 | 0.50 | 0.40 | 0.50 | 0.60 | 0.83 | 0.94 |
| Markdown | 0.667 | 0.733 | 1.00 | 0.00 | **0.00** | 0.60 | 1.00 | 0.82 |
| current + think-on | 0.850 | 0.867 | 0.80 | 0.20 | 0.75 | 0.90 | 1.00 | 0.96 |
| current + think-off | **0.900** | **0.900** | 0.80 | 0.20 | **1.00** | 0.90 | 1.00 | **1.00** |
| fresh + think-on | **0.917** | **0.967** | **1.00** | **0.00** | 0.75 | **1.00** | 1.00 | 0.90 |

Four findings from this matrix:

1. **The main win comes from embeddings + multi-signal retrieval**, not
   from think(). think-off still beats simulator by +32 points on
   overall score. The HNSW vector index + yantrikdb's relevance-
   conditioned scoring (vector × decay × importance × graph ×
   feedback) is the lever that matters for this task.

2. **Fresh DB outperforms the residue-contaminated DB** by ~7 points.
   The Phase 2 entities that had accumulated in the default DB were
   slightly hurting, not helping. No contamination advantage.

3. **think() is not load-bearing at this scale, and in one place it's
   actively harmful.** think-off jumps alias disambiguation accuracy
   from 0.75 → 1.00 because think()'s consolidation merged Project
   Aurora and Aurora-lite memories as "91% similar / redundant" —
   destroying exactly the distinction the alias probes test.

4. **Markdown's supersession score of 1.00 is a benchmark artifact.**
   It wins that specific metric because top-truncation drops old
   sessions from the window when new sessions exceed the cap — Qwen
   never sees the stale value at all. Markdown loses on alias (0.00)
   because session-1 definitions get truncated out of the window too.

## The scale insight — when each feature matters

This is the framing I think is most useful for people building with
agent memory, and the one that only became clear after the ablation:

> yantrikdb's features are scale-dependent, not uniformly-on. The
> right configuration depends on how many memories you're storing.

| memory scale | what actually helps |
|---|---|
| < 100 | embeddings + multi-signal retrieval; `think()` off |
| 100 – 10k | + entity graph, + provenance tracking |
| 10k – 1M | `think()` consolidation earns its keep (real noise to prune) |
| 1M+ | tiered storage, forgetting, archival |

At n=75 memories (this benchmark), think()'s consolidation flagged
Aurora/Aurora-lite as "91% similar, redundant." That's a false
positive — two entities that need to stay distinct. At n=100k,
the same signal probably catches real duplicate standup-meeting
notes and is load-bearing. Same feature; different regime.

This is the scale-dependent product framing, empirically anchored
instead of asserted.

## What's unchanged, what's withdrawn

**Unchanged and still valid:**
- [Phase 3A](../phase3/) and [3B](../phase3b/) (notebook vs cold,
  with increasing constraint density) — never involved the simulator.
  Findings stand.
- [Phase 3D L4 scaling methodology](../phase3d/) — padding haystacks
  with distractors to measure recall@k vs answer-accuracy divergence
  is a valid method.

**Withdrawn:**
- "Plain structured memory loses to markdown" (Phase 3C)
- "40% stale-rate" as a yantrikdb limitation (simulator limitation,
  not yantrikdb)
- "Null result on RFC 006 temporal substrate" (Phase 3D L3)
- "Tiered memory thesis — structured memory only pays off at very
  high scale" (L4; partially — the scale axis is real but the
  boundaries are lower than 3D suggested)

**Honest caveats:**
- n=2 per cell on Phase 3C; n=5 per type (30 total) on LongMemEval L1.
  Small samples.
- Same-scenario rerun; not novel cross-benchmark validation.
- LongMemEval L1 (30 oracle instances, ~36 turns/haystack):
  real yantrikdb 73% vs simulator 77% aggregate, but with
  multi-session +20 and temporal-reasoning +20 (where yantrikdb
  helps), at the cost of single-session regressions (-20 to -40
  — expected, since word-overlap on small haystacks lands
  directly on near-exact lexical matches and embeddings normalize
  that advantage away).
- LongMemEval L3 (longmemeval_s, 550-turn haystacks):
  real yantrikdb 70% vs simulator 70% aggregate — clean tie.
  But the per-type distribution is the story:

  | type | simulator L3 | real ydb L3 | Δ |
  |---|---|---|---|
  | multi-session | 40% | 80% | **+40** |
  | temporal-reasoning | 40% | 60% | +20 |
  | single-session-preference | 40% | 60% | +20 |
  | knowledge-update | 100% | 80% | -20 |
  | single-session-user | 100% | 80% | -20 |
  | single-session-assistant | 100% | 60% | -40 |

  The +40 pts on multi-session at 550-turn scale is the strongest
  per-type signal in the entire Phase 3E data. Real yantrikdb
  helps where multi-signal retrieval matters (queries requiring
  cross-session aggregation), hurts where simple word-overlap on
  small haystacks lands directly on the right turn. At even larger
  scale (longmemeval_m, 500 sessions) the scale-framing predicts
  real ydb should pull ahead on aggregate too — untested.
- Qwen 3.6 judging Qwen 3.6 answers has ~5% upward bias vs GPT-4o
  per spot checks. A GPT-4o judge pass would tighten numbers.

## What's next

1. Rerun L3 with real yantrikdb at multiple configurations (fresh DB,
   think-on/off) — the noise-heavy regime is where the tiered
   framing predicts different winners.
2. Bump Phase 3C to n=5 per cell on the fresh-DB + think-on
   configuration to reduce variance estimate.
3. Explore at-scale testing (10k+ memories, synthetic
   supersession-heavy) where `think()` should earn its keep.
4. GPT-4o judge re-pass on Phase 3E for publication-bound numbers.

## Raw data + reproduction

All harness code + raw JSONL results:
<https://github.com/yantrikos/yantrikdb-server/tree/main/docs/phase3e>

Reproduction: clone the repo, start yantrikdb locally per
`yantrikdb_local.toml`, ensure sentence-transformers is installed,
set `YDB_TOKEN`, run the Phase 3E scripts. Ollama + Qwen 3.6 for
actor/judge.

Full methodology-error writeup:
<https://github.com/yantrikos/yantrikdb-server/blob/main/CORRECTIONS.md>

## A note on adoption friction

Meta-observation from the same session: even the agent building
yantrikdb (Claude Opus 4.7 in this case, working alongside me)
defaulted to writing notes in a local file rather than calling
yantrikdb's MCP `remember`/`recall` tools. Yantrikdb's MCP tools
are marked as "deferred" in the Anthropic MCP schema — have to be
searched and loaded before calling — while `Write` is always
loaded. The path of least resistance won.

If the memory substrate isn't reach-for-it obvious to its own
builder's agent, wider adoption will be harder than the feature set
alone suggests. That's a product insight I got from this session,
free, and worth sharing.

## Acknowledgments

Credit to the (also me) maintainer for catching the methodology
error and for the scale-dependent feature framing that unifies the
ablation results. Both corrections made this post possible.

GPT-5.4 red-team consultation shaped the decision framework
(brainstorm sessions `1fe4bee6` and `ab7e4c07` — pre-registered
publication gate, one-sleep rule, non-triumphant framing, ablation
ordering).

---

*If you're building with agent memory and any of this matches or
contradicts something in your production: please comment, file an
issue, or open a PR with your counter-data. Replication beats
debate.*
