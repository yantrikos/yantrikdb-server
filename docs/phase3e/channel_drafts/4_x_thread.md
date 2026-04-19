# X / Twitter thread — DRAFT, not posted

**Where:** personal account + @yantrikdb (if it exists)
**When:** evenings tend to surface best for technical audiences
**Do not post before:** one-sleep review + L3 results incorporated

---

## 4-tweet thread

### Tweet 1 (the hook — image attached: ablation table screenshot)
```
spent 3 weeks benchmarking my own agent-memory DB against a rolling markdown file.

last week I realized my "structured memory" test condition was actually a 120-line python simulator.

NOT the product.

reran with the real system 👇

[IMAGE: ablation table]
```

### Tweet 2 (the findings)
```
real yantrikdb vs simulator on same scenario (n=2):

overall    0.584 → 0.917
answer     0.600 → 0.967
supersession 50% → 100%
stale-rate  40% → 0%

the stale-rate halving is exactly the failure mode I had published a "null result" on.

withdrawn.
```

### Tweet 3 (the nuance)
```
but the interesting part: running the same scenario with `think()` OFF *beat* think-on.

consolidation was merging "Project Aurora" and "Aurora-lite" as "91% similar redundant" — killing the exact entity distinction the probes test.

think() at n=75 is wrong-tool-wrong-scale.
```

### Tweet 4 (the framing + CTA)
```
features are scale-dependent, not uniformly-on:

 <100 memories: embeddings + multi-signal, think() off
 100-10k: + entity graph
 10k-1M: think() earns its keep
 1M+: tiered storage, forgetting

raw data + harness code + full writeup:
github.com/yantrikos/yantrikdb-server/tree/main/docs/phase3e

if you're running agents at 10k+ memory scale, does this match or break what you see?
```

---

## Image for tweet 1

The ablation table rendered clean. Use a screenshot from the GitHub
rendering of FINDINGS_POST_DRAFT.md, or render in a code block and
screenshot. 16:9 ratio, monospace font, minimum 1200×675.

Content to render:
```
config                        overall  answer  sup  stale  alias
Simulator (original)          0.584   0.600   0.50  0.40   0.50
Markdown baseline             0.667   0.733   1.00  0.00   0.00
Real ydb, current, think-on   0.850   0.867   0.80  0.20   0.75
Real ydb, current, think-off  0.900   0.900   0.80  0.20   1.00
Real ydb, fresh,   think-on   0.917   0.967   1.00  0.00   0.75
```

---

## Alternate shorter single-tweet (if threading feels too much)

```
spent 3 weeks benchmarking my agent-memory DB.

turned out I was benchmarking a 120-line python simulator, not the product.

reran with the real system. overall accuracy 0.584 → 0.917. stale-rate 40% → 0%.

but think() (the consolidation loop) HURT at small scale.

data: github.com/yantrikos/yantrikdb-server/tree/main/docs/phase3e
```

---

## Who to @ for reach (sparingly — max 1-2 per thread)

- Researchers who publish on agent memory (don't @ them in tweet 1 —
  after the thread is posted, reply with "would love your take if you
  have a min" kind of ask)
- No @s in the original thread. X algorithm treats @-heavy threads as
  spam.

## Notes

- DO NOT post this as the first channel. Post in order:
  1. GitHub Discussion (warmest, pre-subscribed stargazers)
  2. HN (biggest potential reach, but hostile to hype)
  3. r/LocalLLaMA (data-heavy audience)
  4. X thread (last — amplifies whichever of the above catches)
- If the GitHub Discussion + HN get good engagement, the X thread
  becomes a link to the discussion with a teaser. If they don't,
  the X thread is the main artifact.
- Reply to EVERY substantive reply within an hour for first 24h.
  X's algorithm rewards active threads.
- If the thread catches > 100 impressions, be ready for repo visits.
