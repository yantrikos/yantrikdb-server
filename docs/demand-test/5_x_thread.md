# X / Twitter thread

**Where:** your personal account + @yantrikdb (if it exists)

**When:** evenings tend to surface better for technical audiences (6-10pm local), but on a shoestring network don't overthink timing

**Strategy:** the scaling table is the hook. Post it as an image in tweet 1 so it shows in previews. Keep threading focused; end with the repo link + direct question.

---

## 3-tweet thread

**Tweet 1** (hook — this is what people scroll past or stop on):
```
built a "cognitive memory database" for AI agents.

benchmarked it.

it lost to a rolling markdown file on answer accuracy (73% vs 60%).

then I scaled the history 50×. retrieval recall stayed at 89%. answer accuracy COLLAPSED to 33%.

[SCREENSHOT of the L4 scaling table]

the failure mode wasn't what I expected 👇
```

**Tweet 2** (the substance):
```
retrieval finds the right memories. the LLM still picks the wrong answer.

at 500 turns, top-20 is mostly evidence.
at 25k turns, top-20 is evidence + keyword-adjacent distractors.
the LLM gets confused and picks something plausible-but-wrong.

retrieval and generation-quality curves DIVERGE at scale.
```

**Tweet 3** (the CTA):
```
this is a different problem than bitemporal memory (Zep, Memento) is solving.

full writeup + raw data: [link to github.com/yantrikos/yantrikdb-server/tree/server/docs/phase3d]

open question: if you're building long-running agents at scale, do you see this failure? how are you fixing it?
```

---

## Alternate 1-tweet post (if you don't want to thread)

```
at 25k conversation turns: plain retrieval holds 89% recall.
Qwen 3.6 answer accuracy drops from 67% to 33%.

retrieval works. the LLM doesn't.

this isn't the failure mode bitemporal memory systems fix.

writeup: [link]

if you're building agents at scale — do you see this?
```

---

## Screenshot to attach to tweet 1

The L4 scaling table, rendered cleanly. You can screenshot from the
`docs/phase3d/L4_scaling_writeup.md` file's rendering on GitHub. Or
make a clean image with something like:

```
scale    turns      recall@20    answer_acc
  1×       487       100.0%        66.7%
  2×       974       100.0%        55.6%
  5×     2,435       100.0%        44.4%
 10×     4,870        88.9%        44.4%
 20×     9,740        88.9%        33.3%
 50×    24,350        88.9%        44.4%
```

Use a monospace font, ~16pt, decent contrast. Twitter renders 16:9 images best.

---

## Who to @ or reply-to

If you have capacity for 1-2 targeted mentions, these are people who
work in or adjacent to the agent-memory space and sometimes respond:

- [@sauhaard_letta](https://x.com/sauhaard_letta) — Letta / MemGPT
- [@mem0ai](https://x.com/mem0ai) — mem0 (can @ the company)
- @getzep_ai — Zep
- @mastra_ai — Mastra
- Researchers who've published memory papers in 2025-2026

Do NOT spam. Max 1-2 @s in a reply, never in the original thread.

---

## Notes

- Thread length: 3 tweets hits the sweet spot for X's algorithm (enough to invite reads, short enough to complete).
- Reply to EVERY substantive comment. X's algorithm heavily rewards author-replies in threads.
- If @yantrikdb account exists, post from there too (same content, different account). Cross-reply from personal.
- Track impressions + replies at T+1hr, T+24hr, T+48hr.
