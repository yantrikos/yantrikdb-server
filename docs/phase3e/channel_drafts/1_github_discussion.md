# GitHub Discussion — DRAFT, not posted

**Where:** <https://github.com/yantrikos/yantrikdb-server/discussions/new?category=general>
**Category:** "Show and tell" or "Announcements"
**When to post:** tomorrow morning after one-sleep review. Do NOT pre-post.

**Title:**
`Correction: the "structured memory" benchmark was measured on a simulator, not yantrikdb. Rerun shows what actually helps.`

---

## Body

```markdown
Hi everyone. Thank you for starring this repo and for the occasional
question over the last few months. I owe you a correction and a more
interesting result.

## The error

I ran four phases of benchmarks (Phase 3A–3D in `docs/`) claiming to
compare yantrikdb against a rolling markdown file on agent-memory
tasks. The "structured memory" condition in those phases was
[`memory_sim.py`](../../phase3c/memory_sim.py) — a 120-line Python
module that stored memories as a list of `(key, value, session)`
tuples with Dice word-overlap retrieval.

It had none of yantrikdb's actual features:
- No embeddings, no HNSW vector index
- No `think()` loop (no consolidation, no conflict scan)
- No multi-signal scoring
- No knowledge graph / entity extraction

It was strictly weaker than even a naive vector DB. Treating it as a
proxy for yantrikdb was a methodology error. The specific conclusions
that followed — "plain structured memory loses to markdown," "40%
stale-rate shows RFC 006 can't handle supersession," "null result on
temporal substrate" — were all drawn from experiments against this
simulator, not yantrikdb. All withdrawn.

Full audit trail: [`CORRECTIONS.md`](../../../CORRECTIONS.md)

## The rerun

Built a proper client ([`yantrikdb_client.py`](../yantrikdb_client.py))
using the real HTTP endpoints: client-side MiniLM-L6-v2 embeddings
passed to `/v1/remember` and `/v1/recall`, `/v1/think` between
sessions. Reran Phase 3C's 5-session supersession/alias/indirect
scenario.

## Same-scenario comparison (n=2 per cell)

| condition | overall | answer | supersession | stale-rate | alias | provenance |
|---|---|---|---|---|---|---|
| Simulator (original) | 0.584 | 0.600 | 0.50 | 0.40 | 0.50 | 0.94 |
| Markdown baseline | 0.667 | 0.733 | 1.00 | 0.00 | **0.00** | 0.82 |
| Real ydb, current DB, think-on | 0.850 | 0.867 | 0.80 | 0.20 | 0.75 | 0.96 |
| Real ydb, current DB, think-off | **0.900** | **0.900** | 0.80 | 0.20 | **1.00** | **1.00** |
| Real ydb, fresh DB, think-on | **0.917** | **0.967** | **1.00** | **0.00** | 0.75 | 0.90 |

## Four findings from the ablation

### 1. The main gain is from embeddings + multi-signal retrieval, NOT `think()`

Even with `think()` OFF, real yantrikdb beat the simulator by +32
points overall. The HNSW vector index + relevance-conditioned scoring
(vector × decay × importance × graph × feedback) is the lever that
matters for this task at this scale.

### 2. `think()` can actively hurt on entity-heavy scenarios at small scale

With `think()` ON at n=75 memories, alias disambiguation dropped from
1.00 to 0.75 because the consolidation pass merged "Project Aurora"
and "Aurora-lite" memories as "91% similar, redundant" — the exact
entities the alias probes test.

This is a scale-dependent cost. At 10k+ memories you have enough true
duplicates for consolidation to earn its keep; at 75 memories the
consolidator false-positives on distinct-but-similar entities. Same
feature, different regime.

### 3. The old DB's Phase 2 residue was slightly hurting

Fresh DB: 0.917. Residue-contaminated: 0.850. The 7-point gap is in
the opposite direction from what you'd expect if residue were secretly
helping. So the simulator-era headline number is NOT inflated by
cross-run contamination.

### 4. Markdown's 1.00 supersession is a benchmark artifact

The rolling-window markdown baseline only "wins" supersession because
top-truncation drops old sessions when new ones overflow the cap.
Qwen literally never sees the stale value. It loses catastrophically
on alias (0.00) for the same reason — session-1 alias definitions get
truncated out.

## The framing that unifies this: features are scale-dependent

| memory scale | configuration |
|---|---|
| < 100 memories | embeddings + multi-signal retrieval; `think()` off |
| 100 – 10k | + entity graph, + provenance tracking |
| 10k – 1M | `think()` consolidation earns its keep on real noise |
| 1M+ | tiered storage, forgetting, archival |

At n=75, `think()` was consolidating 2 distinct entities as
"redundant" — a false positive. At n=100k, the same signal would
catch real duplicate standup notes and be load-bearing. **The
product is "right features for your scale," not "all features all
the time."**

## LongMemEval L1 — the more nuanced picture

Real yantrikdb on LongMemEval oracle (30 instances, 5 per type):
73.3% overall vs simulator's 76.67% — slight aggregate regression.
But the per-type distribution tells the story:

| type | simulator | real ydb | Δ |
|---|---|---|---|
| multi-session | 40% | 60% | +20 |
| temporal-reasoning | 60% | 80% | +20 |
| knowledge-update | 80% | 80% | 0 |
| single-session-preference | 80% | 80% | 0 |
| single-session-user | 100% | 80% | -20 |
| single-session-assistant | 100% | 60% | -40 |

Real yantrikdb helps where multi-signal disambiguation matters
(multi-session, temporal) and hurts where word-overlap on tiny
oracle haystacks (36 turns) lands directly on near-exact lexical
matches. Consistent with the scale framing.

L3 (longmemeval_s, 550-turn haystacks) with real yantrikdb is
running now; expect the story to shift toward yantrikdb as
noise increases.

## Honest caveats

- n=2 per cell on Phase 3C; n=5 per type on L1. Small samples.
- Qwen judging Qwen has ~5% upward bias per spot checks vs GPT-4o.
- Same-scenario rerun for the 3C comparison — not novel validation.
- Local Qwen 3.6 Q4 actor; frontier models may show different
  patterns.

## What I'd like from this community

1. **Critique the methodology** — where does this break, what should
   I have controlled for, what ablation am I missing?
2. **Replication** — anyone with 2+ GPUs can rerun in ~1 hour.
   Harness code + raw data under [`docs/phase3e/`](../).
3. **Production stories** — if you're running agent memory at 10k+
   scale, does the "think() earns its keep at scale" prediction hold?
   Or is the consolidation / conflict-scan approach actively harmful
   even at your scale?

## Thanks

Thanks to everyone who pushed me on methodology. Specifically, the
exact question that caught the simulator-vs-product error was:
*"so then Yantrik functionality, the core functionality did not run
at all. we just ran it as a vector db [or less]"* — a single line
of feedback that reversed 3 weeks of conclusions. The right kind
of critique.

I'm solo on this project. No funding, no team. Responses — positive,
critical, or "this matches what I see in prod" — all useful.

— Pranab
```

## Notes for posting

- **Do not post today.** Pre-committed one-sleep gate. Review tomorrow.
- If L3 results are ready by tomorrow, incorporate them before posting.
- The ending has a soft CTA (replication, production stories, critique)
  — not a sales pitch. Keep it that way.
- Stargazers get notified if they've set up notifications; otherwise
  crosspost to X/HN after this one's live so the Discussion is the
  canonical link.
