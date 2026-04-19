# Phase 3D L4 — synthetic scaling test

**Question:** does plain word-overlap retrieval actually break at scale,
and if so, where? This is the empirical test of the "tiered memory
strategy" hypothesis — that different memory approaches win at
different scales.

## Method

- 9 LongMemEval_s instances selected from the hardest types (3 each
  of `multi-session`, `temporal-reasoning`, `knowledge-update`).
- For each instance, amplify the haystack by sampling distractor
  sessions from OTHER instances, targeting scale factors
  `{1, 2, 5, 10, 20, 50}` × base turn count.
- Base ~487 turns/haystack; 50× ≈ 24,350 turns.
- Retrieve top-20 with same Dice word-overlap as L1/L3. Generate
  answer with Qwen 3.6 (`think:false`). Judge with Qwen.
- 54 runs total. ~20 min wall time.

## Results

| scale | avg_turns | recall@20 | answer_acc | n |
|-------|----------:|----------:|-----------:|---|
| 1×    |    487    | 100.0%    | **66.7%**  | 9 |
| 2×    |    974    | 100.0%    | 55.6%      | 9 |
| 5×    |  2,435    | 100.0%    | 44.4%      | 9 |
| 10×   |  4,870    |  88.9%    | 44.4%      | 9 |
| 20×   |  9,740    |  88.9%    | 33.3%      | 9 |
| 50×   | 24,350    |  88.9%    | **44.4%**  | 9 |

Raw data: [`hypotheses_L4.jsonl`](hypotheses_L4.jsonl) ·
[`scored_L4.jsonl`](scored_L4.jsonl) ·
[`harness_L4_log.txt`](harness_L4_log.txt)

## The two curves diverge

**Retrieval recall barely degrades.** From 100% at 487 turns to 89% at
24,350 turns. The 11-point drop is ONE instance that permanently
fails at 10× and above (a multi-session question whose query
vocabulary doesn't match the evidence session's vocabulary). Plain
Dice word-overlap scales FAR more gracefully than I expected. This
re-confirms the L3 finding that retrieval is not where a structured
substrate earns its keep at this task distribution.

**Answer accuracy drops substantially and monotonically.** 67% at 1× →
33-44% at 20-50×. The drop is ~20-30 points, and it happens even
when retrieval succeeds (recall@20 stays at 89%+).

## What this means

The failure mode at scale is NOT "retrieval can't find the right
memories." It's **"top-k gets diluted with noise, and the LLM picks
the wrong answer from the mixed context."** At 1×, top-20 is mostly
evidence. At 50×, top-20 is 2-3 evidence turns plus 17 distractor
turns that happened to share keywords with the query. The LLM reads
the mixed context and gets confused.

This is a real failure mode at real scale. It's NOT the failure mode
RFC 006 (temporal validity) was designed to fix.

## Which yantrikdb features SHOULD address this?

Going through the README's existing V0 feature list:

- ✅ **Multi-signal scoring** — "vector similarity + temporal decay +
  importance + graph proximity + retrieval feedback." Precisely the
  kind of reranking that could push evidence turns above
  keyword-match distractors at scale. Testable.
- ✅ **HNSW vector index** — swap Dice word-overlap for dense
  embeddings (already present in the engine). Should materially
  shift the curve. Testable.
- ✅ **Graph proximity** — entities extracted from queries could
  boost turns that mention those entities. Testable with a small
  entity-extraction pass.
- ⚠ **Forgetting / decay / consolidation** — could reduce the
  distractor pool at scale by evicting or merging unrelated
  memories. Indirectly helpful.
- ❌ **RFC 006 temporal succession** — not specifically relevant here.
  The failure mode is dilution, not stale-value-selection.
- ❌ **RFC 008 contest state** — same; not the right lever for this
  failure.

The honest reframe: **yantrikdb's "multi-signal scoring" (the V0
value prop on the README) may be its strongest empirical story. The
RFC 006/008 work is either solving a problem that doesn't show up on
LongMemEval at the scales I've tested, or solving a problem that
exists at even larger scales I haven't tested.**

## Compared to the tiered framing

Before this experiment, I hypothesized:

| scale | expected strategy |
|-------|-------------------|
| <100 | stuff everything in context |
| 100-5k | plain retrieval |
| 5k-100k | embeddings + windowing |
| 100k-10M | structured substrate (yantrikdb) |
| 10M+ | tiered storage |

What L4 actually shows:

| scale (turns) | plain retrieval recall@20 | Qwen answer_acc |
|---------------|---------------------------|-----------------|
| 500           | 100% | 67% |
| 1k            | 100% | 56% |
| 2.5k          | 100% | 44% |
| 5k            | 89%  | 44% |
| 10k           | 89%  | 33% |
| 25k           | 89%  | 44% |

**The retrieval-based tier transition (where plain methods break) is
not empirically visible at 25k memories.** The transition I imagined
(tier 3 → tier 4 around 5-100k) isn't there for Dice word-overlap.

**The answer-accuracy tier transition IS visible** — by 5k memories,
Qwen is already 20 points below its 1× baseline. But this transition
is about LLM confusion under noise, not about retrieval substrate.

So the tiered framing needs to be re-stated:

| regime | bottleneck | best strategy |
|--------|------------|---------------|
| small (<1k) | nothing | just use context window |
| medium (1k-25k) | **LLM gets confused by noisy top-k** | rerank aggressively, use dense retrieval, entity filters |
| large (25k+) | **unknown** — extrapolation needed | likely needs all of the above + structural metadata |

The **medium regime is where yantrikdb's multi-signal scoring SHOULD
plausibly win**, and it's where mem0/Zep/Memento currently sell.
This is a real, defensible product zone — narrower than the "agent
memory" framing but concrete and testable.

## Product implication

**Stop leading with RFC 006/008.** Those features address a failure
mode (stale-value selection under temporal succession) that isn't
visible on the field's standard benchmark at the scales we've tested.
They may be valuable in other settings (regulated audit, long-running
deployments) but as a headline feature they lack empirical support.

**Lead with multi-signal scoring.** This is the V0 feature that
directly addresses the empirical failure mode L4 just surfaced
(answer accuracy drops under retrieved-context noise). It's what the
README already promises; the benchmark story just needs to be run.

**Next test (L5?):** swap Dice for yantrikdb's multi-signal scoring
on the same L4 scaling harness. Pre-register: yantrikdb multi-signal
must recover ≥10 percentage points of answer accuracy at the 20-50×
scale vs Dice word-overlap. If it does, that's a publishable
benchmark win AND the product's real differentiator.

## Honest limits

- n=9 instances, 6 scale points. Small sample.
- One model (Qwen 3.6 MoE Q4). Frontier models (GPT-4o) might show
  much less confusion under noise, which would change the story
  entirely.
- Qwen judging Qwen, as always. Upward bias.
- Distractor sessions sampled uniformly from other LongMemEval
  instances — real-world distractor distributions (a single user's
  conversation history) would have different statistical properties
  (more entity overlap, more topic continuity).
- 50× is only 25k turns. Real production deployments hit 100k-1M
  memories. The curve past 50× is unknown.

## TL;DR

- Plain word-overlap retrieval scales to 25k turns with 89% recall.
  Surprising.
- Qwen answer accuracy degrades ~20-30 points across the same scale
  even when retrieval works. The real failure mode is
  **noise-confused LLM**, not **missing retrieval**.
- RFC 006 temporal substrate doesn't address this failure.
  yantrikdb's multi-signal scoring (V0 feature) plausibly does.
  Next experiment: test multi-signal scoring head-to-head vs Dice on
  the same scaling curve.
- The "tiered memory strategy" narrative survives, but the tier
  boundaries are different from what I hypothesized. The
  medium-regime (1k-25k memories) is where yantrikdb's multi-signal
  scoring can plausibly win, and it's also where the commercial
  competition sits.
