# r/LocalLLaMA post

**Where:** <https://reddit.com/r/LocalLLaMA/submit>
**Subreddit culture:** very technical, loves data and local-model experiments, friendly to open-source authors who lead with results and not pitch. Qwen 3.6 + 2×3090 Ti is exactly their demographic.

**Title options (pick one):**
- `I tested agent memory 4 ways on LongMemEval with Qwen 3.6. A rolling markdown file beat my structured memory DB.`
- `At 25k conversation turns, retrieval works. The LLM doesn't. Scaling data + open questions.`
- `Honest scaling curve for plain retrieval on LongMemEval: recall stays at 89%, answer accuracy drops 30 points. Why?`

Recommended: **the first one**. It's counterintuitive, data-backed, and invites pushback (r/LocalLLaMA loves civil disagreement).

**Flair:** "Research" or "Discussion" — NOT "Self-promotion" (will get downvoted).

---

## Body

```markdown
**TL;DR:** Built a cognitive memory DB. Tested it on LongMemEval. It lost to a rolling markdown file on answer accuracy, won on provenance. Scaled up the haystack 50×; retrieval held; LLM answer accuracy tanked anyway. Posting the data because I think the pattern matters to anyone building long-context agents, and I have questions.

---

# The setup

Four memory strategies, same tasks, same Qwen 3.6 MoE Q4 actor (2×3090 Ti), same Qwen judge (yes, biased; caveat below):

- **A — Cold:** no memory at all (baseline)
- **B — Self-note:** LLM rewrites a 1500-char note each session, hard-truncated
- **C — Structured:** plain key/value + word-overlap retrieval
- **D — Markdown:** rolling markdown file, global 7500-char cap, top-truncated when over

# Phase 3C result (5 sessions, 15 supersession/alias/indirect probes)

| cond | answer_acc | provenance | supersession_acc | stale_rate | alias_acc | ctx_chars |
|---|---|---|---|---|---|---|
| A cold    | 0%  | 0%  | 0%   | 10% | 0%  | 0 |
| B note    | 63% | 22% | 80%  | 0%  | 25% | 1500 |
| C structured | 60% | **94%** | 50% | **40%** | 50% | 4983 |
| D markdown | **73%** | 82% | **100%** | 0% | 0% | 7500 |

Structured memory lost to the rolling markdown file on answer accuracy (60% vs 73%). It won on **provenance** (94% vs markdown's 82%) and **alias disambiguation** (50% vs 0% — markdown drops session-1 alias definitions when the window rolls forward).

But here's the gut-punch: **C's 40% "stale-error rate" on supersession**. Plain key/value without temporal validity stores both the old value and the revised one. The LLM retrieves both. It picks the stale one 40% of the time. That's exactly the failure that bitemporal memory systems (Zep, Memento, Mastra) are designed to fix.

# Phase 3D L4 — scaling test (9 instances × 6 scale factors up to 50× = 24k turns)

Padded haystacks with distractor sessions from OTHER LongMemEval instances. Measured top-20 retrieval recall AND downstream answer accuracy.

| scale | avg_turns | recall@20 | answer_acc | n |
|---|---:|---:|---:|---|
| 1×  |     487  | 100.0% | **66.7%** | 9 |
| 2×  |     974  | 100.0% | 55.6% | 9 |
| 5×  |   2,435  | 100.0% | 44.4% | 9 |
| 10× |   4,870  |  88.9% | 44.4% | 9 |
| 20× |   9,740  |  88.9% | **33.3%** | 9 |
| 50× |  24,350  |  88.9% | 44.4% | 9 |

**The curves DIVERGE.** Plain Dice word-overlap retrieval holds 89% recall all the way to 24k turns (that was surprising to me). But the downstream answer accuracy drops 20-30 points. When I looked at failures, the LLM's getting the right evidence memories in top-20 AND a bunch of keyword-adjacent distractor turns, and picking the wrong answer from the mix.

This is not the failure mode the field's converging on (bitemporal / stale-value-picking). It's **LLM-confused-under-noise-in-retrieved-context**.

# Open questions

1. Is this pattern something you've seen in production? Particularly at long histories where retrieval is "OK" but the model still confuses related-but-wrong content?
2. What's the right fix — better reranking (multi-signal scoring), aggressive top-k reduction, entity-aware filtering, or something I haven't thought of?
3. At frontier model scale (Claude 3.7, GPT-4o), does this divergence flatten, or is it structural?
4. If you're building with mem0 / LangChain memory / Zep / Letta / Mastra: does this match anything in your experience or is it an artifact of my setup?

# Raw data + setup

Full writeups and jsonl data, all 4 phases: <https://github.com/yantrikos/yantrikdb-server/tree/server/docs/phase3d>

Harness code is maybe 200 lines, reproducible with any local Ollama setup.

# Caveats

- Qwen judging Qwen has ~5% upward bias. A GPT-4o judge pass would tighten numbers. Haven't done it because it costs money I'd rather spend on compute.
- n=9 for L4. Small. The question "does the pattern exist" feels answered; the magnitude is noisy.
- Oracle subset is easy (3 evidence-only sessions). longmemeval_s at 40-50 sessions is still small vs real production deployments (1M+ memories).
- I'm solo on this. No network, no funding. Posting because I want to know what other people see in prod before I build the next thing.

Happy to answer questions or run additional experiments if there's a specific thing you want tested.
```

---

## Preemptive reply snippets

**"This is just bad retrieval, any decent embedding-based retriever fixes it"**

> Plausibly. Word-overlap was the deliberate Level-1 baseline; sentence-transformer embeddings is on the roadmap. Interesting if you've seen embedding-based retrieval hold answer-accuracy at scale where word-overlap doesn't — the L4 hypothesis was exactly this, but the degradation showed up regardless of recall level, which suggests it's not purely a retrieval-quality issue.

**"Just use a bigger LLM, Qwen 36B is too small"**

> Very likely part of it. My conjecture is the divergence persists but at a smaller magnitude with frontier models. Would love to see someone replicate with GPT-4o or Sonnet.

**"Yeah we see this with long-context distractors all the time"** (best-case reply)

> [Ask follow-up] What's your mitigation — rerank? filter by entity? cap top-k? Or just accept the degradation?

**"mem0 / Zep / Letta all solve this already"**

> My read: mem0 ~49% on LongMemEval, Zep ~65-70%, Memento/Mastra 90%+. Those benchmarks measure retrieval-plus-answer end-to-end, so they don't actually isolate where the failure is. If any of those systems has published separate retrieval-recall vs answer-accuracy curves under controlled noise, I'd love to see the link.

**"Is yantrikdb better than X?"**

> On the benchmarks I've run, not reliably. It has the substrate features (temporal validity, polarity, contest state) that the paper architecture assumes would win, but the measured behavior doesn't clearly differentiate from simpler approaches yet. I'm pausing to figure out which problem to double down on before building more.

---

## Notes to self

- Reddit hates self-promotion. Put the repo link in the middle/end of the body, not the title. Never comment "check out yantrikdb!" — use comments to engage with the question.
- Title should NOT contain "yantrikdb" — the product name in the title reads as promo. The data is the hook.
- Format: markdown tables are supported on new reddit. Triple-check the table renders on mobile.
- Watch for moderator removal — most subs auto-hide posts linking to GitHub. If hidden, message mods politely with context.
- Track post URL + upvote count at T+1h, T+6h, T+24h, T+48h in `7_response_tracker.md`.
