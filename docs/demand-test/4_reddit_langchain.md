# r/LangChain post

**Where:** <https://reddit.com/r/LangChain/submit>
**Subreddit culture:** smaller (~50k), conversational, practical, LangChain-flavored. People here are SHIPPING agent apps. Memory is a live pain. Leave the theory; lead with concrete failure modes and mitigations.

**Title options:**
- `My agent memory DB lost to a markdown file. Here's what I'm learning.`
- `For anyone using LangChain memory / mem0 / Zep: how do you handle noisy retrieval at scale?`
- `What does "agent memory failure" actually look like in your prod app?`

Recommended: **the third one**. r/LangChain responds better to direct questions than to "I tested X" posts. You want THEIR stories.

**Flair:** "Discussion"

---

## Body

```markdown
I've been doing empirical testing on agent memory systems — built one (yantrikdb, solo project, open source), benchmarked it 4 ways, scaled it up to ~25k conversation turns per haystack. The thing I'm most surprised by, and the thing I want to ask you all about:

**The failure mode at scale is not "retrieval finds the wrong memories." It's "retrieval finds the right memories, plus a bunch of keyword-adjacent distractors, and the LLM picks the wrong answer from the mix."**

Specifically, using plain Dice word-overlap at top-20:
- At ~500 turns: retrieval recall 100%, Qwen 3.6's answer accuracy 67%
- At ~25k turns: retrieval recall 89%, answer accuracy drops to 33-44%

The retrieval curve barely moves. The answer-accuracy curve falls off a cliff. Full data and writeup: <https://github.com/yantrikos/yantrikdb-server/tree/server/docs/phase3d>

So the question for people shipping LangChain/LangGraph agents, mem0, Zep, or custom memory:

1. **Have you seen this in production?** The agent retrieves, gets the right content, and still fumbles the answer because of noise in the retrieved context?
2. **If yes, how do you handle it?**
   - aggressive top-k reduction (top-3 or top-5 instead of top-20)?
   - re-ranking with cross-encoders or multi-signal scoring?
   - entity/session filters before retrieval?
   - LLM-side chain-of-thought that explicitly enumerates and rejects irrelevant memories?
   - just bigger models and call it a day?
3. **If you've stopped worrying about it** — what memory scale are you actually operating at? (I suspect most prod agents are in the 100–5,000 memory range where this pattern doesn't dominate.)

I'm trying to figure out where the real pain is before I build more of yantrikdb's internals. If the pattern above doesn't match what you're seeing, I want to know. If it does, I want to know how you're solving it today.
```

---

## Preemptive reply snippets

**"I just use mem0 / pgvector / Chroma, seems fine"**

> Good signal — at what memory scale? If prod agents mostly stay <5k memories, my scaling results probably don't matter for most users and the problem is much more niche than I was positioning it.

**"Use a reranker like Cohere rerank-3 / BGE-reranker"**

> Interesting — do you use it inline in the agent loop or only for eval? And does it actually improve the generation-quality-after-retrieval curve, or just the retrieval-recall curve? That's the specific distinction that's biting me.

**"LangGraph has memory stores, just use them"**

> I've looked at LangGraph's memory — is the pattern you're seeing "works great for short-lived session state, harder for cross-session long-term"? Or same performance at any scale?

**"Honestly the best memory for us is just a summary in the system prompt"**

> Perfect, that's a data point. How do you keep the summary quality up as the history grows — manual, LLM-generated, some hybrid?

**"What's yantrikdb vs mem0/Zep?"**

> Honestly right now: a bunch of architectural features (temporal validity, polarity, contest state, multi-signal scoring) whose specific benchmark wins I haven't been able to clearly demonstrate vs simpler alternatives. That's why I'm asking rather than pitching. The README has the cognitive-memory framing; the docs/phase3d writeups have the actual test results, which are more mixed.

---

## Notes

- r/LangChain is more forgiving of self-linking if the post is a genuine question. Still keep repo link in the body, not title.
- Ideal engagement is 3-5 substantive replies. This isn't r/LocalLLaMA — smaller sub, more conversational, fewer pure-drive-by comments.
- Engage within an hour of any reply. This sub rewards responsive authors.
- Track in `7_response_tracker.md`.
