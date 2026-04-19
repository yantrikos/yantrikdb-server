# Hacker News post — DRAFT, not posted

**Where:** <https://news.ycombinator.com/submit>
**When:** Tuesday–Thursday, 8–10 am PT
**Format:** Ask HN (question-led, not product-led)
**Do not post before:** one-sleep review + L3 results incorporated

---

## Title

`Ask HN: My agent-memory benchmark was measured on a simulator, not the product. Here's what the rerun found.`

(If not Ask HN, then: `Show HN: I caught a 3-week methodology error in my own agent-memory benchmark. The rerun changed every conclusion.`)

## Body

```
I've been building yantrikdb, an open-source cognitive memory database
for AI agents, and ran a 5-phase empirical benchmark gauntlet against
it. Last week, while running a consultation on strategy, I discovered
the condition I'd been labeling "structured memory" across four of
those phases was actually a 120-line Python simulator — list of
(key, value, session) tuples with Dice word-overlap retrieval. NO
embeddings, NO vector index, NO consolidation loop, NO multi-signal
scoring. Strictly weaker than a naive vector DB.

The "structured memory loses to markdown" conclusion I'd been sitting
on was drawn from this toy, not from yantrikdb. I was benchmarking a
straw man of my own product.

Reran with the real system (client-side MiniLM embeddings, HNSW index,
multi-signal scoring, think() loop). Three interesting findings:

## 1. Real product beats the simulator by +27 points on the same scenario

                              overall  ans   sup  stale  alias
  Simulator (original claim)  0.584   0.600  0.50  0.40  0.50
  Markdown baseline           0.667   0.733  1.00  0.00  0.00
  Real ydb, fresh DB          0.917   0.967  1.00  0.00  0.75

Stale-error rate went from 40% to 0%, supersession accuracy from 50%
to 100%. The specific "null result on temporal substrate" I'd
published earlier is withdrawn.

## 2. think() (the consolidation/conflict-scan loop) is NOT the lever

Running the same scenario with think() OFF still beat the simulator
by +32 points. The gain is from embeddings + multi-signal retrieval,
not from the consolidation machinery.

More interesting: think() actively HURT on this scenario. At n=75
memories, it consolidated "Project Aurora" and "Aurora-lite" as
"91% similar, redundant" — destroying the entity distinction the
benchmark probes test. Alias accuracy dropped from 1.00 (think-off)
to 0.75 (think-on).

This is a scale-dependent failure, not a bug in think(). With 10k+
memories you have real duplicate standup-notes to merge; with 75
you have 2 distinct-but-similar entities that get false-positive-
consolidated.

## 3. LongMemEval shows the same scale-dependence

On oracle subset (30 instances, ~36 turns per haystack):
  Simulator:     76.67% overall
  Real yantrikdb: 73.33% overall  (slight aggregate regression)

But per-type:
  multi-session:        40% → 60%  (+20, helps)
  temporal-reasoning:   60% → 80%  (+20, helps)
  single-session-user:  100% → 80% (-20, hurts)
  single-session-assistant: 100% → 60% (-40, hurts a lot)

On tiny haystacks (3 oracle sessions), word-overlap lands directly
on near-exact lexical matches and embeddings normalize that advantage
away. On multi-session questions, multi-signal scoring actually
disambiguates. Same feature, opposite sign depending on task.

## The framing

Features are scale-dependent, not uniformly-on:

  <100 memories:  embeddings + multi-signal retrieval, think() off
  100-10k:        + entity graph, + provenance
  10k-1M:         think() consolidation earns its keep
  1M+:            tiered storage, forgetting, archival

The product claim I WAS going to make ("yantrikdb beats plain
memory") was too coarse. The one I'd now make ("configure yantrikdb
for your scale regime; the wrong configuration can hurt") is more
defensible AND more actionable.

## Open questions for HN

1. At production agent-memory scale (1M+ memories across months of
   use), does consolidation actually earn its keep? Or does the
   false-positive rate stay high enough that it's harmful?

2. Is there a body of work on noise-robust-retrieval-under-long-
   context that I should be reading? The divergence between retrieval
   recall (which stays high even at 25k turns in my testing) and
   answer-accuracy-under-top-k-noise feels under-explored.

3. For anyone shipping production agents with memory: does this
   scale-dependent configuration story match what you've built, or is
   it naive? I'm solo on this and want the prod reality-check.

Full writeup + raw JSONL data + harness code:
https://github.com/yantrikos/yantrikdb-server/tree/main/docs/phase3e

Full audit trail of the methodology error:
https://github.com/yantrikos/yantrikdb-server/blob/main/CORRECTIONS.md
```

## Preemptive reply snippets

**"Qwen 3.6 as judge is biased"**

> Correct. ~5% upward bias vs GPT-4o on my spot checks. All reported
> numbers should have that discount applied; the ablation DELTAS are
> more robust to this bias than absolute values. GPT-4o judge pass is
> on the list before any publication-bound number goes out.

**"You just re-discovered that RAG works"**

> Partly yes. The non-trivial finding is that even within a single
> memory product, WHICH features earn the gain depends on scale, and
> some features hurt at the wrong scale. That's a different claim
> than "RAG helps." It's "your memory stack should be scale-tuned."

**"n=2 is not enough"**

> Agreed. The ablation DELTAS (think-off vs think-on, fresh vs residue)
> are consistent enough across cells that I think the direction is
> robust, but magnitude estimates are wobbly. Bumping to n=5 for the
> winning configuration is on the list before the findings post is
> final.

**"Why did you have this error for 3 weeks"**

> Honest answer: I let a 120-line simulator stand in for the product
> because (a) it was easier, (b) if the real product lost, I'd have
> been in a harder emotional position than if a toy lost. Classic
> avoidance. The methodology lesson: benchmark configs should include
> a "did this condition actually invoke product X's code paths" check
> at the top of the harness log. Not doing that is how this happened.

**"Is yantrikdb better than Zep / Memento / mem0?"**

> Honestly don't know on published benchmarks yet. On the synthetic
> Phase 3C scenario, real yantrikdb with embeddings beats a rolling
> markdown file and the simulator I was using as stand-in. Whether
> it beats Zep or Memento on LongMemEval at their published numbers
> (~65-95%) is a separate test I haven't run. mem0 is ~49% on
> LongMemEval; my real-ydb on oracle was 73%. Not apples-to-apples
> (different subsets). Real-ydb on L3 (longmemeval_s) is running
> right now.

## Notes

- HN loves honest-error-correction posts. "I was wrong, here's how,
  here's the fix" gets upvoted. "I solved AGI" gets buried.
- Do NOT submit early in the week; Sun/Mon have low traffic. Tue/Wed
  8-10am PT is the sweet spot.
- Engage EVERY substantive comment within an hour. HN ranking
  rewards author engagement.
- If the thread catches fire, be ready for 500-2000 repo visits in
  an hour. README is already clean from earlier MCP promotion work.
