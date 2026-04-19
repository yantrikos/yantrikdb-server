# r/LocalLLaMA post — DRAFT, not posted

**Where:** <https://reddit.com/r/LocalLLaMA/submit>
**When:** weekday evening or early AM (r/LocalLLaMA is global)
**Flair:** "Research" or "Discussion"
**Do not post before:** one-sleep review + L3 results incorporated

r/LocalLLaMA loves Qwen + 2×3090 Ti + local ablations + tables. Lead
with data, lean into the local-model angle, keep self-promotion out
of the title.

---

## Title

`Agent memory ablation: caught my own methodology error, reran 4 configs with Qwen 3.6 on 2x3090 Ti. think() hurts at small scale.`

Alternatives:
- `I ran a 3-week benchmark measuring my own memory DB. Turns out I was benchmarking a 120-line Python simulator, not the product. Here's the actual ablation.`
- `Scale-dependent features in agent memory: ablation data showing when think()/consolidation helps vs hurts (Qwen 3.6, yantrikdb)`

Recommended: first one. "caught my own methodology error" signals
epistemic honesty; "think() hurts at small scale" is the
counterintuitive hook r/LocalLLaMA responds to.

---

## Body

```markdown
**TL;DR:** I'm solo-building an open-source agent memory DB
([yantrikdb](https://github.com/yantrikos/yantrikdb-server)) and ran a
benchmark gauntlet against it. Two weeks in I realized my "structured
memory" test condition was actually a toy simulator (list of
`(key, value, session)` tuples, no embeddings), not my product. Reran
with the real HTTP endpoints + MiniLM embeddings + the full
`think()` loop. Dropped every prior conclusion. Here's the ablation
that replaced them.

**Stack:** Qwen 3.6 MoE Q4 on 2x RTX 3090 Ti via Ollama (`think:false`
for generation, temperature 0.2, num_ctx 32k), sentence-transformers
`all-MiniLM-L6-v2` for client-side embeddings, yantrikdb v0.5.13
(Rust backend, HNSW vector index, multi-signal scoring). Qwen also
judged its own answers — upward bias ~5% per spot checks vs GPT-4o;
deltas are more robust than absolute numbers.

## The scenario

5-session synthetic scenario ("Titan Industries cloud migration
consulting"). 15 probes per run: 5 direct, 5 supersession
(mortgage-pre-approval style: value X in session 1, revised to Y in
session 3), 3 branch-conditional, 2 alias-disambiguation
(Aurora vs Aurora-lite entities).

## Ablation matrix (n=2 per cell, same scenario)

| config | overall | answer | sup | stale | alias | direct | branch | prov |
|---|---|---|---|---|---|---|---|---|
| Simulator (what I'd been testing) | 0.584 | 0.600 | 0.50 | 0.40 | 0.50 | 0.60 | 0.83 | 0.94 |
| Markdown baseline (rolling, 7500-char cap, top-truncated) | 0.667 | 0.733 | 1.00 | 0.00 | **0.00** | 0.60 | 1.00 | 0.82 |
| Real ydb, current DB, think-on | 0.850 | 0.867 | 0.80 | 0.20 | 0.75 | 0.90 | 1.00 | 0.96 |
| Real ydb, current DB, think-off | **0.900** | **0.900** | 0.80 | 0.20 | **1.00** | 0.90 | 1.00 | **1.00** |
| Real ydb, fresh DB, think-on | **0.917** | **0.967** | **1.00** | **0.00** | 0.75 | **1.00** | 1.00 | 0.90 |

## Four findings

### 1. The main win is embeddings + multi-signal retrieval, not think()

Think-off still beats simulator by +32 pts overall. The lever isn't
consolidation/conflict-scan, it's vector search + relevance-
conditioned scoring (vector × decay × importance × graph × feedback).

### 2. think() HURT on this scenario — scale mismatch

At n=75 memories, think()'s consolidation merged "Project Aurora"
($2.4M budget) and "Aurora-lite" ($120K budget) as "91% similar,
redundant." These are distinct entities the alias probes test.

Alias accuracy dropped 1.00 (think-off) → 0.75 (think-on). Same
feature that would be load-bearing at 10k+ memories (where real
duplicate standup notes exist to merge) is actively harmful at n=75
(where "similar" means "near-duplicate keywords on distinct
entities").

### 3. Fresh DB beats residue-contaminated DB

Fresh: 0.917. Residue-contaminated (Phase 2 experiments left 30
entities, 85 edges in the default DB): 0.850. The residue was
slightly HURTING, not helping. No cross-contamination inflation to
worry about.

### 4. Markdown's 1.00 supersession is a window-truncation artifact

The rolling-markdown baseline only "beats" on supersession because
top-truncation drops old sessions when new ones push the cap. Qwen
literally never sees the stale value. Same mechanism murders alias
disambiguation (0.00) because session-1 alias definitions drop out
of the window too.

## Scale-dependent feature framing

| memory scale | what actually helps |
|---|---|
| <100 | embeddings + multi-signal retrieval, think() OFF |
| 100-10k | + entity graph, + provenance |
| 10k-1M | think()/consolidation earns its keep on real noise |
| 1M+ | tiered storage, forgetting, archival |

This empirically validates something the community has been saying
for a year: memory features shouldn't be "all on, all the time." The
configuration that wins is scale-dependent.

## LongMemEval L1 (oracle, 30 instances)

Same actor/judge, real yantrikdb vs simulator:

| type | simulator | real ydb | Δ |
|---|---|---|---|
| multi-session | 40% | 60% | **+20** |
| temporal-reasoning | 60% | 80% | **+20** |
| knowledge-update | 80% | 80% | 0 |
| single-session-preference | 80% | 80% | 0 |
| single-session-user | 100% | 80% | -20 |
| single-session-assistant | 100% | 60% | **-40** |
| **overall** | **76.67%** | **73.33%** | **-3.3** |

Real yantrikdb helps where multi-signal disambiguation matters
(multi-session, temporal) and hurts where word-overlap on tiny oracle
haystacks (~36 turns) lands directly on near-exact lexical matches.
Consistent with the scale framing.

L3 (longmemeval_s, 550-turn haystacks) with real ydb is running in
the background as I write this. I'll update with results when it's
done.

## Questions for the sub

1. If you're running local agents with persistent memory: what memory
   backend, what scale, what failures do you see? The
   scale-dependent story feels right on my data but I want the prod
   reality check.

2. Cross-encoder rerankers on retrieved memories (BGE-reranker, etc):
   do these close the "answer-accuracy-under-noise" gap I'm seeing?
   The divergence between retrieval recall and answer accuracy feels
   like it might be a reranker-shaped hole.

3. Any of the bitemporal / provenance-heavy memory frameworks
   (Zep, Memento, Mastra) — if you've used them, does the
   scale-feature-match framing hold, or are some of their features
   still worth turning on at small scale?

## Reproduction

Everything is open: harness code, raw JSONL results, scoring
scripts, scenarios. ~2-3 GPUs + Ollama + Python is enough to rerun.

Full writeup: https://github.com/yantrikos/yantrikdb-server/tree/main/docs/phase3e
The methodology error writeup: https://github.com/yantrikos/yantrikdb-server/blob/main/CORRECTIONS.md

## Caveats

- n=2 for the ablation cells. Deltas are consistent across cells, so
  direction is robust; magnitude estimates wobbly.
- Qwen judging Qwen — as noted, ~5% upward bias vs GPT-4o. Deltas
  less affected than absolutes.
- Same-scenario rerun for the ablation. Cross-scenario validation
  (LongMemEval L1 above; L3 in flight; real-world prod data: none) is
  pending.
- I'm solo on this; if you see methodological holes, please point
  them out. Credit goes in the post explicitly.
```

## Preemptive reply snippets

**"Qwen 36B Q4 is weak, this doesn't transfer to frontier models"**

> True concern. Frontier models likely show similar scale-dependent
> pattern at a smaller magnitude (the retrieval quality matters more
> than model quality for this specific failure mode). Running GPT-4o
> as actor is the obvious next experiment; I don't have budget for
> 500+ LME evals on GPT-4o yet.

**"Your 'structured memory' baseline was a joke"**

> Yes. That's the whole point of the post — I was running a 3-week
> gauntlet against a joke I'd built and didn't notice. The
> methodology-lesson is honest. The ablation matrix is the actual
> result.

**"mem0/Zep/Memento already do all this"**

> Partially. Those systems publish benchmark numbers on LongMemEval
> (Mem0 ~49%, Zep ~65-70%, Mastra ~95%). I haven't run head-to-head
> yet. The unique claim here isn't "yantrikdb wins" — it's "scale-
> tune your feature set." That applies to any of them.

**"You should be using [specific technique]"**

> Probably! Please name it. This post is a "here's the data, what am
> I missing" ask, not a victory lap.
```

---

## Notes

- r/LocalLLaMA is friendly to self-posts with data but ruthless on
  product promotion. Keep the product name OUT of the title.
- Reddit hides posts that link to your own site — link to GitHub in
  the body, not the title. A text post with a single link at the
  bottom is fine.
- Tables render on new reddit; check mobile.
- Reply to every substantive comment within 2 hours.
