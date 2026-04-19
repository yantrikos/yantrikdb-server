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

## L3 — longmemeval_s (~53 sessions / ~550 turns / ~120k tokens per haystack)

Same harness, same seed, same subset logic (5/type × 6 = 30), same
Qwen actor + judge. Only difference: `longmemeval_s_cleaned.json`
(evidence sessions + ~50 distractor sessions per instance), top-k=20
(up from 10 for the bigger haystack).

Harness: [`phase3d_lme_harness_L3.py`](phase3d_lme_harness_L3.py) · judge:
[`phase3d_lme_judge_L3.py`](phase3d_lme_judge_L3.py)
Log: [`harness_L3_log.txt`](harness_L3_log.txt) · scored:
[`scored_L3.jsonl`](scored_L3.jsonl)

### L3 results (n=5 per type, n=30 total)

| question type | L1 (oracle) | L3 (_s) | Δ |
|---|---|---|---|
| single-session-user         | 5/5 (100%) | 5/5 (100%) | 0 |
| single-session-assistant    | 5/5 (100%) | 5/5 (100%) | 0 |
| knowledge-update            | 4/5 (80%)  | **5/5 (100%)** | **+20** |
| single-session-preference   | 4/5 (80%)  | 2/5 (40%)  | -40 |
| temporal-reasoning          | 3/5 (60%)  | 2/5 (40%)  | -20 |
| multi-session               | 2/5 (40%)  | 2/5 (40%)  | 0 |
| **overall**                 | **76.67%** | **70.00%** | -6.67 |

Session-level retrieval recall@20: **28/30 = 93.3%** (the two misses
were preference questions where the user asked a tangentially worded
question — the answer sessions were there but not in the top-20 by
word overlap).

**That 93% recall is striking.** Plain Dice-coefficient word overlap
over 550-turn haystacks surfaces the correct answer session in the
top-20 on 28 of 30 questions. This is far better than I expected and
is an important baseline datum: you don't need embeddings or a
structured substrate to find the right turn in LongMemEval_s. You
need the LLM to read the turn and answer correctly, which is a
different problem.

### The headline finding — question `852ce960` (knowledge-update)

The same question (mortgage pre-approval $350K → $400K revision)
appeared in both L1 and L3 subsets. At L1, Qwen confidently answered
**"$350,000"** — wrong, Qwen picked the stale value out of oracle's
context. This was the one case I flagged as "exactly where yantrikdb
RFC 006 temporal-succession substrate should win."

At L3, Qwen answered: **"The user mentioned being pre-approved for
$350,000 in one instance and $400,000 in another."** Hedged. Lists
both values. LongMemEval's knowledge-update grading rubric
**explicitly accepts this**:

> "If the response contains some previous information along with an
> updated answer, the response should be considered as correct as
> long as the updated answer is the required answer."

So L3's hedged answer is scored correct. The exact failure mode RFC
006 was pre-registered to fix is, by LongMemEval's own grading
definition, **not a failure**. There is no room on the LongMemEval
knowledge-update subset for a temporal-succession substrate to
improve the numerical score.

This is uncomfortable, but it's the honest read. A product feature
claim has to either (a) win on a benchmark's actual grading, or (b)
be justified outside benchmarks entirely (cleaner UX, fewer hedged
answers, auditability). The "fewer hedged answers" argument is
legitimate — users asking "what was I pre-approved for?" don't want
both values — but it's not what LongMemEval measures.

### L3 retrieval failures

2 / 30 questions had no answer-session overlap in top-20:

- `505af2f5` (preference): asked about "recommendations for coffee
  creamer recipes"; answer session discussed spring-themed coffee
  flavors but not creamer recipes specifically. Word overlap missed
  the semantic match.
- `6b7dfb22` (preference): asked about "finding inspiration for
  paintings"; answer session used different vocabulary. Semantic
  mismatch.

These are the textbook embedding-fixable cases. L2 (embeddings) would
plausibly fix them.

## What L1+L3 together establish

- **Retrieval is not yantrikdb's killer app.** Plain word-overlap at
  top-20 gets 93% session recall on 550-turn haystacks. Embeddings
  would push this higher but the marginal gain is small (maybe 2-3
  questions).
- **The answer generator is the bottleneck.** Multi-session (40%/40%)
  and temporal-reasoning (60%/40%) failures are Qwen arithmetic /
  aggregation errors on retrieved-but-correct context. No memory
  substrate fixes them.
- **Knowledge-update under LongMemEval's grading is NOT a scenario
  where RFC 006 can show numerical wins.** The benchmark's own
  rubric accepts "old + new" hedged answers as correct. The stale-pick
  failure I was targeting gets washed out.
- **The retrieval hit rate at L3 is high enough that "bigger haystack
  breaks retrieval" (the core assumption behind wanting a substrate)
  is empirically false for this benchmark at this scale.** Maybe
  longmemeval_m (500 sessions) breaks it. Haven't tested.

## Where this leaves yantrikdb

The retreat from Phase 3C → L1 → L3 has progressively eroded the
case for the RFC 006/008 temporal substrate as a benchmark-winning
feature, specifically:

| phase | claim | status after testing |
|---|---|---|
| 3A/3B | "notebook beats cold" | supported (+36 to +67 pts) but ceilings |
| 3C (plain struct vs markdown) | "structured memory beats markdown" | **falsified** at this scale |
| 3C sub-claim | "plain structured memory has 40% stale-rate that yantrikdb could fix" | true but narrow |
| 3D L1 | "that 40% stale-rate shows up on LongMemEval" | 1/30 occurrence — too small |
| 3D L3 | "scaling haystack makes retrieval noisy so temporal validity pays off" | **falsified**: retrieval still 93% at top-20, knowledge-update ceilings at 100% because LME grading accepts hedging |

## Remaining paths for yantrikdb to be defensible

1. **longmemeval_m** (500 sessions per haystack). Maybe the retrieval
   hit rate drops at that scale. This is cheap to test (same harness,
   different data file).
2. **Custom benchmark where the grading punishes hedging.** If we
   define "knowledge-update" answers as correct ONLY when they pick
   the current value without listing the stale one, RFC 006 has room
   to improve. But this is "we grade ourselves to look good" territory;
   it's a real concern but a weaker product story than winning on an
   established benchmark.
3. **Non-benchmark product wedge**: auditability, provenance-accuracy,
   user-UX of clean answers vs hedged ones. These are real but quieter.
   Mem0's product is the comparison — they sell auto-fact-extraction
   as their value prop, not benchmark wins.
4. **Concede the memory-layer framing and pivot.** If LongMemEval's
   grading plus a strong enough actor model (Qwen 3.6 at 70% on _s)
   erodes the structured-substrate argument this much, the claim
   yantrikdb should stand on may not be "better memory" at all. It
   may be "auditable claim graphs for regulated domains" — a
   different product, a different market, a different RFC set.

## Honest next-step ranking

1. **longmemeval_m** (1 hour of work — same harness, different data).
   If retrieval stays >90% at 500 sessions too, the structured-memory-for-benchmark thesis is probably dead.
2. **Real yantrikdb integration (L4)**: test whether wire protocol and
   claim_with_lineage works end-to-end with a baseline agent. This is
   product-engineering validation independent of benchmark numbers.
3. **Sit with the data for a day.** Three phases have progressively
   narrowed the provable claim. Might be time to pause and talk to
   potential users before writing more code.
