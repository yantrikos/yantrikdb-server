# 48-hour demand test — execution plan

**Decision:** after Phase 3A→3D showed the original RFC 006/008 temporal
substrate claim doesn't hold on LongMemEval, and L4 scaling surfaced a
different failure mode (LLM-confused-under-noise, not missing retrieval),
we're pausing architecture work for 48 hours and running a
focused demand test.

Full consultation: `../phase3d/README.md` and GPT-5.4 brainstorm session
`1fe4bee6-efe7-42e5-a18d-5579fcce6a61`.

## What goes in this folder

| File | Channel | When | Format |
|---|---|---|---|
| [`1_github_discussion.md`](1_github_discussion.md) | [yantrikos/yantrikdb-server/discussions](https://github.com/yantrikos/yantrikdb-server/discussions) | First — warmest audience (stargazers auto-notified) | GitHub Discussion, "Ask for feedback" category |
| [`2_hn_post.md`](2_hn_post.md) | [news.ycombinator.com](https://news.ycombinator.com/submit) | Tues–Thu 8–10am PT (highest surfacing) | Show HN or Ask HN |
| [`3_reddit_localllama.md`](3_reddit_localllama.md) | [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) | Any weekday | Self-post, data-first |
| [`4_reddit_langchain.md`](4_reddit_langchain.md) | [r/LangChain](https://reddit.com/r/LangChain) | Any day | Self-post, shorter, conversational |
| [`5_x_thread.md`](5_x_thread.md) | X / Twitter | Evenings best | 3-tweet thread with scaling-table screenshot |
| [`6_reply_snippets.md`](6_reply_snippets.md) | — | In comments as people engage | F1/F2/F3 framings, ready to paste |
| [`7_response_tracker.md`](7_response_tracker.md) | — | Populate over 48 hrs | Pre-reg-aligned counting |
| [`decision_rule.md`](decision_rule.md) | — | Look at T+48h ONLY | Pre-registered; do not re-read before then |

## Strategy

**Lead with data, not product.** The Phase 3D findings are more
interesting than any positioning for yantrikdb. The unexpected result
(retrieval scales to 25k turns but answer accuracy collapses due to
noise confusion) is counterintuitive and empirical — exactly what
technical audiences (HN, r/LocalLLaMA) engage with.

The 3 product framings (F1 noise-robust retrieval / F2 auditable
provenance / F3 truth-over-time contradiction) go in the REPLIES, not
the posts. When someone engages with a comment, probe whichever
framing fits their pain.

## Execution order

1. Post GitHub Discussion FIRST (warmest audience, pre-subscribed stargazers).
2. X thread pointing to the Discussion.
3. Reddit r/LocalLLaMA (highest-probability data audience).
4. HN Show HN (widest reach, hostile to hype).
5. Reddit r/LangChain (smaller, conversational).
6. Track responses in `7_response_tracker.md` as they come in.
7. At T+48 hours from GitHub Discussion post time, open
   `decision_rule.md` and apply. **Do not open early.**

## What counts as a response

Per pre-registration:
- **Substantive pain** ("we hit this in production", "current tools fail
  here", "hacked around this manually") → counts under threshold
- **Behavior** ("I'd try this", "send me when ready", "can I integrate
  with my stack?") → counts under threshold
- **Design-partner lead** ("we should talk", concrete use case,
  engagement commitment) → counts

**Does NOT count:** compliments, upvotes, retweets without context,
star increments, "interesting" with no follow-up question, "cool idea"
standalone.

## What this session produces

I'm drafting all 8 files. Review the drafts, edit your voice in, then
post. The posting (and the honest 48-hour counting) is your move.
