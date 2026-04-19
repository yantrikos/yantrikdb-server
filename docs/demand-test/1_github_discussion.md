# GitHub Discussion — post this FIRST

**Where:** <https://github.com/yantrikos/yantrikdb-server/discussions/new?category=q-a> (or whichever category fits)

**Category:** "Q&A" or "General" or "Show and tell"

**Title options (pick one):**
- `What made you star yantrikdb? And what stopped you from trying it?`
- `100+ stars, no production deployments I know of. Help me figure out why.`
- `An honest question to the people who starred this repo`

Recommended: **the second one**. It's direct, vulnerable, and pre-commits you to non-hype. Stargazers respect that.

---

## Body

```markdown
First: thank you for starring this repo. It means more than I let on.

Second: I've spent the last weeks running four phases of empirical experiments to see whether yantrikdb's design actually delivers on its claims, and the results have been... instructive.

Short version of what I found (full writeups in [docs/phase3d](../tree/server/docs/phase3d), [docs/phase3c](../tree/server/docs/phase3c)):

- **Notebook-based memory** (LLM writing its own structured notes between sessions) beats cold recall by +35 to +67 points on multi-session constraint tasks — as expected.
- **Plain structured memory (key/value + retrieval)** actually *loses* to a rolling markdown file when the grading rubric accepts hedged "old + new" answers, because markdown top-truncation removes the stale value automatically. That was a gut punch.
- **On LongMemEval's knowledge-update subset** (where yantrikdb's RFC 006 temporal-succession substrate was supposed to uniquely shine), the benchmark's own grading explicitly accepts hedged responses, so the feature has no room to win numerically.
- **At scale** (up to 25k turns per haystack), plain word-overlap retrieval recall@20 stays at 89%+. Retrieval doesn't break. But Qwen 3.6's answer accuracy drops 20-30 points as the haystack grows, because top-k gets diluted with keyword-adjacent distractors.

Which means the failure mode I built yantrikdb's substrate to solve (stale value selection under temporal succession) isn't the one that actually hurts at real scale. The real failure at scale is **LLM getting confused by noisy retrieved context**. That's a different problem with different solutions.

Now — I'm about to make some decisions about what to build next, and I'd rather not guess. So I'm asking directly:

### If you starred this repo

1. What problem were you hoping yantrikdb would solve?
2. Did you try it? If yes, what happened. If no, what stopped you?
3. Which of these pains matches something you've actually hit:
   - **F1**: agent memory that stays accurate when the conversation history gets noisy/long
   - **F2**: memory with traceable provenance (which session/source did this fact come from?)
   - **F3**: agent memory that never returns stale facts — "latest truth only"
   - **None of the above** — your pain is something else. What is it?
4. If any of F1/F2/F3 is a real pain for you, how are you solving it today? (prompts, summaries, a specific tool, giving up?)

Any of 1-4 is useful. Even "I starred it because the README looked cool and then forgot" is useful — that's data too.

I'm solo on this. No funding, no sales team, no network. I'd rather get 3 honest answers that tell me to stop than build something no one wants.

— Pranab
```

---

## Notes to self

- **Stargazer notification**: GitHub notifies the repo subscribers/watchers by default when a Discussion is posted. Stargazers are NOT auto-notified but many set up notifications on "watched" repos. Still the warmest available audience.
- **Cross-link** once posted: add a comment to the README or open an issue that links to this Discussion, so a casual visitor lands on it.
- **Pin it** if GitHub supports that in the Discussions UI.
- After posting: update `7_response_tracker.md` with the Discussion URL and start the 48-hour timer.
