# Phase 3D — LongMemEval head-to-head

**Goal.** Move out of custom experimental designs and onto the
field-standard benchmark. LongMemEval (Wu et al., ICLR 2025) is where
Mem0, Zep, Memento, Mastra, Supermemory, and friends all publish their
numbers. Until yantrikdb has a number on LongMemEval, its claims are
unverifiable.

Pre-registered ladder, cheapest → hardest:

| level | retrieval | haystack | purpose |
|---|---|---|---|
| **L1 (done)** | Phase 3C word-overlap simulator | oracle (evidence only) | get a baseline number; establish the LLM ceiling |
| L2 (planned) | sentence-transformer embeddings | oracle | does better retrieval help at oracle? |
| L3 (planned) | same embeddings | longmemeval_s (40 sessions, 115k tokens) | does retrieval quality matter under realistic noise? |
| L4 (planned) | yantrikdb + RFC 006/008 temporal substrate | longmemeval_s, knowledge-update & temporal subsets | does the temporal substrate close a specific gap? |

Each level is gated on the prior level showing signal — no point adding
complexity if the simpler version already works (or already fails at
the same rate for a different reason).

## L1 setup

- Dataset: `longmemeval_oracle.json` from
  [huggingface.co/datasets/xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
- Subset: 30 instances (5 per question type × 6 types), seed=42
- Memory store: `../phase3c/memory_sim.py` with `value_cap=2000` — Dice
  word-overlap retrieval, no embeddings, no temporal logic
- Answer generation: Qwen 3.6 MoE Q4 via Ollama, `think:false`,
  temperature 0.2, num_ctx 32k
- Judge: **Qwen 3.6 itself** (caveat: Qwen judging Qwen has upward
  bias; replace with GPT-4o for any publishable number)
- Run: ~3 min wall time for 30 instances

Harness: [`phase3d_lme_harness.py`](phase3d_lme_harness.py) · judge:
[`phase3d_lme_judge.py`](phase3d_lme_judge.py)

## L1 results (n=5 per type, n=30 total)

| question type | correct/n | accuracy |
|---|---|---|
| single-session-user         | 5/5 | **100%** |
| single-session-assistant    | 5/5 | **100%** |
| knowledge-update            | 4/5 | 80% |
| single-session-preference   | 4/5 | 80% |
| temporal-reasoning          | 3/5 | 60% |
| multi-session               | 2/5 | 40% |
| **overall**                 | 23/30 | **76.67%** |

For context, published numbers on longmemeval_s (harder — full 40-session
haystack): GPT-4o long-context ~60-70%; Mem0 ~49%; Zep ~65-70%;
Memento/Mastra ~90%+. Our 76.67% is on the much easier oracle subset
where evidence retrieval is trivially correct, so it's not comparable —
it establishes the **LLM ceiling** at this subset for the Qwen 3.6
actor, nothing more.

## L1 failure analysis

All 7 failures tagged manually by retrieval-vs-LLM bottleneck. Raw data
in [`scored.jsonl`](scored.jsonl).

| id | type | failure | retrieval-fixable? | substrate-relevant? |
|---|---|---|---|---|
| `gpt4_0b2f1d21` | temporal | wrong event ordering | ✗ (retrieval found both events; LLM inverted) | ✗ |
| `c9f37c46` | temporal | 4 months vs 2 months | ✗ (date math; LLM) | ✗ |
| `27016adc` | multi-session | couldn't find countryside/renovation costs | ✓ (embeddings might surface) | ✗ |
| `3fdac837` | multi-session | couldn't find Chicago days count | ✓ (aggregation across sessions) | ✗ |
| `6456829e` | multi-session | tomatoes found, cucumbers missed | ✓ (incomplete recall) | ✗ |
| `852ce960` | knowledge-update | **$350K vs $400K** | ✗ (both in oracle) | **✓ yantrikdb temporal validity** |
| `afdc33df` | preference | didn't acknowledge existing efforts | partial (rubric/style) | ✗ |

**Diagnosis:**
- 3/7 failures are **retrieval-incomplete** (multi-session aggregation where
  word-overlap missed relevant items). L2 embeddings should help these.
- 3/7 failures are **LLM-bottlenecked** (temporal arithmetic, rubric match).
  No memory substrate will fix them. Only a better actor.
- **1/7 is cleanly yantrikdb-relevant**: the $350K → $400K mortgage
  pre-approval update. Both values were retrieved by oracle; Qwen picked
  the stale one. This is exactly the RFC 006 temporal-succession use
  case.

At oracle retrieval, yantrikdb's specific advantage is visible on **≤1
out of 30 questions** — too small to make a statistical claim. L3 is
the right next step: switch to `longmemeval_s` (40 sessions), where
retrieval is noisy enough that (a) more questions become
retrieval-sensitive and (b) temporal-validity value-picking becomes a
frequent failure mode.

## What this means for yantrikdb

- **Oracle is a solved problem for any reasonable retriever + modest
  LLM.** 76.67% is where Qwen 3.6 ends up; the memory layer is not the
  bottleneck at this subset.
- **The yantrikdb-specific claim (RFC 006 temporal succession) is
  genuinely testable** — we found 1 case where it should help in 30. On
  longmemeval_s's 500-question full set with 80 instances of
  knowledge-update type, extrapolating suggests 10-20 cases where
  temporal substrate could flip the answer. That's a real (if modest)
  discriminative claim.
- **Distribution gap still unchanged.** Getting a LongMemEval number is
  table stakes for being in the research conversation. It does not by
  itself create demand.

## Caveats

- n=5 per type. Noisy; per-type accuracy should not be over-interpreted.
- Oracle subset only — ~3 evidence sessions per question. The real
  benchmark is longmemeval_s (40 sessions) or longmemeval_m (500
  sessions).
- **Judge bias**: Qwen 3.6 grading Qwen 3.6-generated answers is
  optimistic. For L2+ runs, use GPT-4o or Claude as judge (the LME
  paper's protocol is GPT-4o).
- Single model (Qwen 3.6 MoE Q4). Frontier models would likely saturate
  oracle and change the failure mix.

## Next: Level 2

L2 plan (not yet built):
- Swap word-overlap for sentence-transformer embeddings
  (`all-MiniLM-L6-v2`, cheap and fast)
- Same oracle subset, same Qwen actor, same Qwen judge
- Pre-register: L2 should improve the 3 retrieval-incomplete
  multi-session cases. If it doesn't, the issue is not retrieval
  quality, it's retrieval STRATEGY (chunking, aggregation).

L3 plan (gated on L2 signal):
- Move to longmemeval_s (~40 sessions, 115k tokens per haystack)
- Retrieval matters here; the full published leaderboard is on this
  subset.
- This is where yantrikdb needs to have a number to be in the
  conversation.

L4 plan (gated on L3 being in-range):
- Replace simulator with real yantrikdb HTTP endpoints
- For knowledge-update subset specifically, use `claim_with_lineage`
  (RFC 008) with temporal validity windows
- Pre-register: **on knowledge-update subset, yantrikdb must cut
  stale-value-picking rate by ≥50% vs L3 baseline.** If it can't, RFC
  006 does not earn its complexity and should be reconsidered.
