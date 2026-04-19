# Hacker News post

**Where:** <https://news.ycombinator.com/submit>

**When:** Tuesday–Thursday, 8–10 am PT is the golden window (most eyeballs, front page chance)

**Format choice:** Ask HN — not Show HN. The Phase 3D finding IS the lede, not the product. HN punishes "show my product" framing; it rewards honest research results and direct questions.

**Title (titles on HN matter a lot — they MUST be factual and non-clickbait):**

Recommended: `Ask HN: Why does agent memory retrieval scale, but agent answers don't?`

Alternatives:
- `Ask HN: At 25k conversation turns, retrieval works fine. The LLM doesn't. Why?`
- `Ask HN: I ran LongMemEval four ways. A rolling markdown file beat my structured memory. Here's why.`

Pick the first one — the question format invites engagement in a way HN specifically responds to.

---

## Body

```
I've been building a "cognitive memory database" (yantrikdb, https://github.com/yantrikos/yantrikdb-server, open source, solo project) and ran it through a 5-phase empirical gauntlet over the last few weeks. The surprise result, posted here because I think it's broadly interesting to anyone working on LLM agents:

I constructed scaling haystacks from LongMemEval's _s subset by padding the per-question conversation history with distractors — 1x (~500 turns), 2x, 5x, 10x, 20x, 50x (~24k turns). I used plain Dice word-overlap retrieval (nothing fancy, no embeddings) at top-20. Then asked Qwen 3.6 MoE Q4 to answer from retrieved context.

Results:

  scale     recall@20    answer_acc   n
   1x        100.0%        66.7%       9
   2x        100.0%        55.6%       9
   5x        100.0%        44.4%       9
   10x        88.9%        44.4%       9
   20x        88.9%        33.3%       9
   50x        88.9%        44.4%       9

Retrieval barely degrades. The answer accuracy drops 20-30 points. The answer-quality curve and the retrieval-curve DIVERGE.

The mechanism, best I can tell: at 1x, top-20 is mostly the 20 most-relevant turns from a ~500-turn haystack (evidence-heavy). At 50x, top-20 is still pulling from the same ~500-turn evidence pool — but now intermixed with keyword-adjacent distractor turns from *other* questions. The LLM gets the right memories PLUS a bunch of lookalikes and picks the wrong answer.

This isn't the failure mode most agent-memory work is targeting. The field is converging on bitemporal modeling (Zep, Memento, Mastra, TiMem) to fix "stale value picked over current value" — a real problem, but one that didn't show up in my benchmarks because modern LLM generation hedges ("X in one instance, Y in another") and graders accept that.

What actually hurt at scale was noise-in-retrieved-context confusion.

Some honest open questions I'd love HN's take on:

1. Does anyone have production agent-memory failures that match this pattern — retrieval surfaces roughly-relevant items but the LLM still can't pick the right answer?

2. Is the right fix (a) smarter reranking with multi-signal scoring (importance × decay × graph × feedback, beyond embedding similarity), (b) smaller top-k with entity-filtering, (c) a fundamentally different retrieval architecture, or (d) this goes away once frontier models replace 36B local ones?

3. Has anyone bench-tested answer-accuracy-under-noise as a separate axis from retrieval recall? Most memory benchmarks I've seen conflate them.

Full phase-by-phase writeups + raw data: https://github.com/yantrikos/yantrikdb-server/tree/server/docs/phase3d

I don't have a product pitch. I'm trying to figure out whether the real problem here is worth building for, and whether there's a wedge I'm missing.
```

---

## Preemptive reply snippets for likely HN comments

**"Qwen 3.6 is weak, GPT-4o would fix this"**

> Fair concern. Tested on Qwen because (a) local, (b) reproducible at $0 marginal cost. My conjecture is frontier models have the same problem but at a higher absolute level — the divergence between retrieval recall and answer accuracy is a structural issue, not a model-size issue. But you're right that I haven't shown that. It's the obvious next experiment.

**"You tested a toy benchmark, real systems are different"**

> Full agree. LongMemEval is a conversational-history benchmark. A production agent might have 1M+ memories spanning years. I don't have a way to test that at home, and neither does most of the published literature. If you're running agent memory in prod at this scale, I'd love to know what fails.

**"This is what {Zep/Memento/Mastra/mem0} already fixes"**

> Zep and Memento do bitemporal indexing, which targets the "stale-value" failure mode. Mastra's Observational Memory leads LongMemEval at ~95%. None of them specifically address the noise-confusion-in-retrieved-context pattern I saw — or if they do, it's a side-effect of better reranking. Which is my question: is reranking the right lever?

**"This is just RAG, you're re-discovering that RAG is hard"**

> Kind of, yes. The interesting thing to me is WHICH part of RAG breaks at scale. Retrieval recall's fine. Re-ranking / filtering / LLM-under-noise is the real gap. The field's answer to "RAG is hard" seems to mostly be "use longer context", which is different from "fix the retriever-to-generator handoff."

**"Self-promotional, why should HN care"**

> It's a genuinely open research question with data. Not pitching a product or asking for signups. If the answer to "does this pattern match your prod experience" is "no, we don't see this", that's useful to me; if it's "yes, we hack around it with X", that's more useful.

**"Why didn't you just use GPT-4o as judge"**

> Qwen judge on Qwen answers has ~5% upward bias per my comparison with human spot-checks. Good catch — it's documented as a caveat in the writeup. GPT-4o judge would be the next pass for anything publication-bound.

---

## Notes

- HN has a soft rule: one submission per domain per day. Don't submit multiple things to your own GitHub on the same day.
- If the post gets ≥2 upvotes in the first 30 min it usually reaches /newest viewership. If dead at hour 2, it's dead — don't repost.
- Engage EVERY substantive comment within an hour. HN's ranking rewards author engagement.
- If it hits front page: be ready for 500-2000 visits to the repo in an hour. Make sure README is clean (done).
- Update `7_response_tracker.md` with post URL, scores at T+1hr, T+6hr, T+24hr, T+48hr.
