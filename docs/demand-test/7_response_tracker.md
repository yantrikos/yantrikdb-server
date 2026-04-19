# Response tracker — 48-hour window

**Post start time (fill in when you post):** YYYY-MM-DD HH:MM UTC
**T+48 hour deadline (compute from start):** YYYY-MM-DD HH:MM UTC

**Do not open [`decision_rule.md`](decision_rule.md) before T+48h.**
It contains the pre-registered threshold. Opening it early creates
motivated counting.

---

## Posts — URLs and snapshot metrics

| Channel | URL | Posted at | T+1h stat | T+24h stat | T+48h stat |
|---|---|---|---|---|---|
| GitHub Discussion | | | | | |
| X / Twitter thread | | | impressions/replies | | |
| r/LocalLLaMA | | | upvotes/comments | | |
| Hacker News | | | points/comments | | |
| r/LangChain | | | upvotes/comments | | |
| Other (Discord/DMs) | | | | | |

---

## Responses — qualitative log

Log every substantive response. Use this template per entry:

```
---
Channel:  [GH Discussion / HN / r/LocalLLaMA / r/LangChain / X / DM]
Post URL: [link]
Responder: [username or handle]
Time:     [T+Xh]

Their comment (verbatim, or paraphrase if long):
  "..."

Framing they reacted to (if any): [F1 noise / F2 provenance / F3 supersession / none / other]

What they said about their actual problem:
  "..."

Their current mitigation:
  "..."

Follow-up I asked / will ask:
  "..."

Their behavior requested (if any):
  [ ] genuine "try it" / "integrate" / "discuss" request
  [ ] design partner lead (willingness to engage + clear use case)
  [ ] concrete pain confirmation (production, explicit, specific)
  [ ] compliment only ("cool", "interesting") — DOES NOT COUNT
  [ ] drive-by / no substance

Threshold bucket (count at T+48h only):
  [ ] concrete_pain
  [ ] behavioral_request  
  [ ] design_partner
  [ ] noise (compliments, jokes, questions without substance)
```

---

## Log entries

### Entry 1

*(paste template and fill in as responses come in)*

### Entry 2

### Entry 3

*(add as needed)*

---

## Counts at T+48h (fill in ONCE at deadline)

Count only from the `Threshold bucket` fields above.

- **Concrete pain confirmations:** ___
- **Behavioral requests (try / integrate / discuss):** ___
- **Design-partner-quality leads:** ___

At this point, open [`decision_rule.md`](decision_rule.md) and compare
against the pre-registered threshold. Do not rationalize. Do not
argue the threshold in retrospect. Apply it.

---

## Metadata to track separately (for the post-mortem)

- Total unique responders across all channels: ___
- Total upvotes / reactions / shares aggregated: ___
- New GitHub stars in the 48hr window: ___
- New PyPI downloads / Docker pulls: ___ *(ecosystem metric, weak but free)*
- Did any tweet / post go "viral" (>100 impressions): y / n

These are CONTEXT, not validation. The threshold is applied only on
the qualitative response count above.

---

## After the threshold check

Per [`decision_rule.md`](decision_rule.md):

- **Above threshold:** run A (multi-signal vs Dice L5 test); write up
  the strongest resonating framing as a Phase 4 positioning brief;
  follow up with design-partner leads.

- **Below threshold:** write the post-mortem (`8_postmortem.md`
  template), publish it as-is, decide whether to narrow-wedge / pivot
  channels / shelve. No more architecture work until that decision
  is made.
