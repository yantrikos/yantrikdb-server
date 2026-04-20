# Pre-registration — Scenario 2 v3 pipeline run

**Committed BEFORE running any Scenario 2 experiment.** Timestamp in git history
is the proof of pre-registration. The recipe below is frozen; no post-hoc
additions will be made based on observed performance.

## Pipeline recipe (identical to Phase 3C v3)

For each of 2 runs:

1. Fresh namespace on the fresh_p3e_v3 database (same DB as Scenario 1 v3 —
   per-namespace isolation for recall; conflicts are per-DB but in practice
   scoped to namespace via the scan by new-RIDs-only).
2. For each of the 5 sessions in `sessions.json`:
   - Let Qwen 3.6 read the session narrative
   - Qwen calls `remember(key, value)` up to 15 times (capped)
   - After ingestion: `think()` with defaults (consolidation + conflict scan)
3. After all 5 sessions ingested (ONCE per run):
   - Call `/v1/relate` for each pair in `sessions.json:alias_pairs_for_pipeline`
     (8 calls total, pre-registered list)
   - Call `store.resolve_all_latest_wins()` — for every open conflict, use
     `keep_b` if memory_b's RID > memory_a's RID, else `keep_a`
4. Probe phase: for each of 15 probes, `recall(probe.q, top_k=10)` → format
   retrieved memories → Qwen answers with format "Answer: X\nSource: Session N"
5. Score with `phase3c_scorer` (probe-type-aware, same scorer as all prior runs).

## Alias pairs (frozen, derived from scenario spec)

| entity | relationship | target |
|---|---|---|
| Isabel Marques | distinct_from | Maria Marques |
| Maria Marques | distinct_from | Isabel Marques |
| Glucophage | distinct_from | Lantus |
| Lantus | distinct_from | Glucophage |
| Northgate Family Medicine satellite | distinct_from | Northgate main campus |
| Northgate main campus | distinct_from | Northgate Family Medicine satellite |
| Monica Fairweather | distinct_from | Gregory Tan |
| Gregory Tan | distinct_from | Monica Fairweather |

These are derived from scenario narrative text that NAMES both entities as
distinct at ingest time. They are not chosen based on any observed failure in
Scenario 2.

## Conflict resolution strategy (frozen)

`resolve_all_latest_wins()`:
- Pull all open conflicts on the namespace
- For each: compare RIDs lexically (UUIDv7, time-prefixed)
- Later RID wins → strategy = "keep_b" if b > a, else "keep_a"
- `resolution_note = "auto: latest-wins post-ingest"`

Same policy as Scenario 1 v3. No adjustment.

## Pre-registered outcome response matrix

- **Answer accuracy ≥90%:** Publish tomorrow with "stronger correction note,
  transfer confirmed" — v3 pipeline generalizes.
- **Answer accuracy 70-90%:** Publish tomorrow with "v3 pipeline shows partial
  transfer, tradeoffs remain" — the most rhetorically defensible outcome.
- **Answer accuracy <70%:** Genre shift to "case study: scenario-specific
  instrumentation matters more than expected" — still publishable, different
  thesis.

**No post-hoc tweaks to the pipeline for Scenario 2.** Whatever accuracy
Scenario 2 produces IS the evidence. If the numbers disappoint, they stay in
the writeup as-is.

## What I'm NOT doing

- Not adjusting alias pairs if they miss edge cases Scenario 2 exposes
- Not tuning resolve_conflict strategy (always `keep_b`/`keep_a` by RID)
- Not re-running Scenario 1 with new features regardless of Scenario 2 outcome
- Not adding new pipeline steps (no session_start, no categories, no
  procedural memory) — those are future-work items, not publication evidence

## Why this matters

GPT-5.4 red-team consultation (session de15ce6a) identified the strongest
critique of the Phase 3C v3 result: the 8 relate pairs were chosen AFTER
analyzing Scenario 1's failures. That's targeted remediation, not clean
scaling.

Scenario 2 with this pre-reg tests whether the v3 pipeline recipe (not
scenario-specific tuning) transfers. If it does, the capability claim is
defensible. If it doesn't, that's also honest evidence and the post becomes
a case study.

Committed by: Pranab (via Claude Opus 4.7 session)
Date: 2026-04-19 evening (before Scenario 2 run)
