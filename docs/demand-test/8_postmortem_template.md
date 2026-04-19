# Post-mortem template — use ONLY if demand test falls below threshold

**Copy this file to `8_postmortem.md` and fill it in honestly.**

This is the template for the honest writeup if the T+48h threshold in
[`decision_rule.md`](decision_rule.md) is NOT met. The goal: a public
artifact that captures what was tried, what the evidence said, and what
the decision is — without self-rescue or rationalization.

Publish the completed post-mortem in the same channels as the demand
test. It's a legitimate research contribution (negative results +
methodology are valuable) and it anchors honesty for future projects.

---

## Structure

```markdown
# Demand test post-mortem — yantrikdb, April 2026

**TL;DR:** After a 4-phase empirical gauntlet on agent memory substrates,
I ran a 48-hour demand test with 3 pre-registered value-prop framings
across GitHub Discussions, HN, r/LocalLLaMA, r/LangChain, and X. The
pre-registered threshold for continued development (3+ concrete pain
confirmations OR 2+ behavioral requests OR 1 design-partner lead) was
NOT met. Decision: [narrow to vertical X / pivot to direct enterprise /
shelve].

## The setup

[Link to Phase 3 writeups]

[Brief recap: notebook helps; plain structured memory loses to markdown;
LongMemEval grading accepts hedging; at scale retrieval holds but answer
quality drops under noise.]

## The demand test

[Copy the 3 framings from `1_github_discussion.md`]

Posted:
- GitHub Discussion (stargazers): [URL], [X views, Y comments]
- HN: [URL], [points, comments]
- r/LocalLLaMA: [URL], [upvotes, comments]
- r/LangChain: [URL], [upvotes, comments]
- X thread: [URL], [impressions, replies]

Total unique responders across all channels: [N]

## What I heard

[Categorized responses. Be specific. No paraphrasing up. If someone said
"cool project" that's what you write.]

### Responses that matched F1 (noise-robust retrieval):
[verbatim or close-paraphrase]

### Responses that matched F2 (auditable provenance):
[verbatim]

### Responses that matched F3 (truth-over-time):
[verbatim]

### Responses that matched none of the above:
[verbatim]

### Compliments and drive-by engagement:
[count only; don't need to transcribe each]

## What I learned

### Pattern 1: [what did people consistently say they care about, if anything?]

### Pattern 2: [what did they say they DON'T care about?]

### Pattern 3: [what problem statement came from multiple responders that I hadn't framed?]

## What the threshold says

Concrete pain confirmations: [count] / 3 needed
Behavioral requests: [count] / 2 needed
Design-partner leads: [count] / 1 needed

Threshold met? [YES / NO]

## Decision

[Narrow-wedge / Channel pivot / Shelve — justified with specific response
evidence.]

## What's next

[Concrete next action, with date.]

## What I'd do differently next time

[Honest learnings about the build-before-validate pattern, the RFC
006/008 sunk cost, the benchmark-as-substitute-for-demand trap.]

## Thanks

[To specific responders who engaged substantively. Public credit for
people who gave their time.]
```

---

## Publishing guidance

- Post the completed `8_postmortem.md` to the SAME channels as the
  demand test, in the same post/thread if possible.
- Do not hide it. Do not soften it. Public post-mortems are credibility.
- The audience you already have (100+ stargazers) has now seen one
  complete honest cycle from you. That's more rare than the outcome
  of any single project.
- If you choose to shelve: that decision, done publicly and honestly,
  is itself a portfolio artifact.
