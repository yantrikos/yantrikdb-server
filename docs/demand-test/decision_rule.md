# Pre-registered decision rule — DO NOT OPEN BEFORE T+48h

**Posted at:** _(fill when you post)_
**Open this file at:** _(T+48 hours from above)_

---

## Rule

**Keep building yantrikdb** (run A = L5 multi-signal vs Dice benchmark;
write up F1-F2-F3 repositioning as Phase 4; follow up with leads) ONLY IF
at least ONE of the following holds after counting qualitative responses
in [`7_response_tracker.md`](7_response_tracker.md):

- **Concrete pain confirmations ≥ 3.** "We hit this in production" /
  "Our current tools fail at this" / "I hacked around this manually".
  Not compliments, not acknowledgements. Explicit, specific pain tied
  to one of F1 / F2 / F3 or a clearly related problem.

- **Behavioral requests ≥ 2.** "I want to try it on my stack" / "Send
  me docs when you ship next" / "Can we integrate?" / "Let's talk."
  Explicit asks for follow-up action, not abstract interest.

- **Design-partner-quality lead ≥ 1.** One person or team with:
  (a) clear painful use case articulated in their own words,
  (b) explicit willingness to engage (meeting, pilot, call, design
  partner),
  (c) role/context that suggests they could actually deploy.

---

## If threshold NOT met

**Do NOT:**
- ❌ Run more benchmarks (A / B / C are OFF the table)
- ❌ Start building a new RFC
- ❌ Rewrite the README again to "clarify positioning"
- ❌ Rationalize 1-2 "cool idea" replies as signal
- ❌ Blame the channel ("I should post on LinkedIn next")
- ❌ Extend the deadline to 72h or 1 week

**Do:**
- ✅ Copy `8_postmortem_template.md` to `8_postmortem.md` and fill it out
  honestly (see template for structure)
- ✅ Share the post-mortem publicly (same channels as the demand test)
- ✅ Decide one of:
  - **Narrow-wedge:** pick ONE vertical where the pain might still exist
    (regulated-domain-audit, multi-agent-coordination, long-horizon
    copilot) and pivot the product story to that vertical. Re-run the
    demand test in a more targeted channel (LinkedIn, industry Slack,
    direct outreach) for that specific audience.
  - **Channel pivot:** if the hypothesis is that enterprise buyers are
    the audience and they're not on Reddit/HN/X, spend the next 2 weeks
    on direct enterprise outreach (LinkedIn Sales Navigator, cold email
    to CTOs of memory-heavy agent products). Harder, but targeted.
  - **Shelve:** acknowledge yantrikdb as a portfolio project. Move on
    to the next thing with the learnings: pre-register falsification
    before building, test demand before writing the engine, lead with
    data not aspiration.

Any of these is honest. The bias to resist is "one more benchmark and
we'll know."

---

## If threshold IS met

**Do:**
- ✅ Run A (L5 multi-signal scoring vs plain Dice on the L4 scaling
  harness). This is the experiment that sharpens whichever framing
  resonated.
- ✅ Write a Phase 4 positioning brief focused on the winning framing:
  F1 / F2 / F3, based on which drew the demand signal.
- ✅ Reach out to design-partner leads within 24h of confirming
  threshold met. Do not let warm leads cool.
- ✅ Update the README to lead with the demand-validated framing.
- ✅ Continue under new positioning. RFC 006/008 are de-prioritized
  unless F3 specifically was the winning framing.

---

## Pre-commitment

I, the solo builder here, pre-commit to applying this rule without
softening the thresholds at the T+48h moment. If I find myself
arguing the threshold was "too strict" or "I'd have hit it with one
more day", that is the sunk-cost bias GPT-5.4 warned about and the
rule applies unchanged.

Signature (fill in at time of posting): _______________
Post start time: _______________
T+48h open time: _______________
