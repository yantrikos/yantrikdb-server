# Correction Notice — to post TODAY

**Where:** GitHub Discussion on yantrikos/yantrikdb-server, pinned if possible.
**Also:** add a banner to the top of README.md pointing to CORRECTIONS.md
**Tomorrow:** full findings post with the actual corrected numbers

Goal: get the correction visible publicly before leaving the day. Narrow, honest, non-triumphal.

---

## Title

`Correction: Phase 3 benchmarks committed earlier today used a simulator, not the actual yantrikdb`

## Body

```markdown
Quick correction notice. Posting this now rather than waiting, because
the wrong numbers are public.

## What happened

I spent today writing and committing a series of benchmark writeups
(`docs/phase3a/`, `docs/phase3b/`, `docs/phase3c/`, `docs/phase3d/`)
claiming to measure yantrikdb's memory features against alternatives
on synthetic scenarios and LongMemEval.

Mid-afternoon, while consulting on next steps, the actual question
came up: **"so then Yantrik functionality, the core functionality did
not run at all. we just ran it as a vector db [or less]."**

The honest answer: yes. The "structured memory" test condition in
those writeups was [`docs/phase3c/memory_sim.py`](docs/phase3c/memory_sim.py)
— a 120-line Python module that stored memories as a list of
`(key, value, session)` tuples with Dice word-overlap retrieval.

It had none of yantrikdb's actual features:
- no embeddings, no HNSW vector index
- no `think()` loop (no consolidation, no conflict scan)
- no multi-signal scoring
- no knowledge graph / entity extraction

Strictly weaker than a naive vector DB. Any conclusion I drew from
those runs about "structured memory vs markdown" or "the temporal
substrate doesn't help" was drawn from a toy that doesn't exercise
the features. Honest mistake. Caught within hours, not weeks.

## What I did about it

Immediately built a proper HTTP client
([`docs/phase3e/yantrikdb_client.py`](docs/phase3e/yantrikdb_client.py))
using the real yantrikdb endpoints with client-side MiniLM embeddings
and `think()` between sessions. Reran the Phase 3C scenario. The
numbers change materially.

## What to read instead

- [`CORRECTIONS.md`](CORRECTIONS.md) — full audit trail of the error
- [`docs/phase3c/README.md`](docs/phase3c/README.md) and
  [`docs/phase3d/README.md`](docs/phase3d/README.md) — now carry
  prominent deprecation banners pointing to the rerun
- [`docs/phase3e/`](docs/phase3e/) — the actual yantrikdb runs

## What I'm NOT doing in this notice

- Claiming the corrected numbers are final — they're n=2 per cell,
  Qwen-judged, same-scenario; they're directional, not publication-grade.
- Claiming yantrikdb "wins." The corrected results show it materially
  outperforms the simulator, but the per-type picture is nuanced and
  there are configurations where its `think()` consolidation actively
  hurts at small scale.
- Posting a full findings writeup yet. That's tomorrow's work, after
  one night's sleep on the draft. Posting the correction tonight is
  separate — it's about not letting wrong claims sit visibly public.

## Why post this now

The buggy writeups were pushed to `main` earlier today. Even though
no one commented or built on those numbers (commit window was ~6-10
hours), they were technically public. Waiting until tomorrow to
acknowledge the error would be 12 more hours of incorrect content
carrying an implicit endorsement. Cheaper to post the correction now.

If you starred or watched this repo because the benchmarks looked
interesting: the direction of the findings is actually more
interesting once corrected, but don't cite the pre-correction
numbers.

— Pranab
```

---

## Also do today

1. **Add correction banner to README.md top** — just above the opening claim, something like:

```markdown
> ⚠ **Correction notice (2026-04-19):** Phase 3 benchmarks committed
> earlier today used a simulator (not actual yantrikdb) for the
> "structured memory" condition. Rerun with real yantrikdb is in
> [docs/phase3e/](docs/phase3e/). Full writeup of the fix:
> [CORRECTIONS.md](CORRECTIONS.md). Full findings post coming
> tomorrow.
```

2. **Pin the GitHub Discussion** (if you have that permission)

3. **Don't post to HN, Reddit, or X yet** — those are for the full findings tomorrow. Correction notice is internal-repo-scope.

4. **Single tweet OK** (only if the yantrikdb X account exists): "Heads-up: posted a correction to today's benchmark commits. The 'structured memory' test used a simulator, not the actual yantrikdb. Fix pushed, full findings post tomorrow. [link to GH Discussion]"
