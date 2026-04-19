# Phase 3B — hidden-constraint recovery, scaled up

Same design as [Phase 3A](../phase3/README.md) but with more constraints and
more prose, to test whether self-note (B), oracle (C), and raw transcript
(D) pull apart once the task crosses Qwen's single-session compression
budget.

## Setup diff vs Phase 3A

| | Phase 3A | Phase 3B |
|---|---|---|
| constraints | 7 | 15 |
| doc words   | ~1500 | ~2500 |
| doc chars   | ~9000 | ~16000 |
| context window (num_ctx) | 16k | 32k |
| new constraints added | — | Artifactory mirror, Vault secrets, Postgres queue, idempotent jobs, MinIO SSE, audit-format-v3, company-common-logging, UTC-in-storage |

The new constraints were chosen to cut against common LLM priors:
cold Qwen almost always reaches for Redis, env-var secrets, stdlib
logging, and public Docker Hub images.

## Results (n=4 per cell)

| condition | correct/15 | silent/15 | violated/15 | accuracy | halluc |
|-----------|-----------|-----------|-------------|----------|--------|
| A_cold            | 5.00 | 7.00 | 3.00 | **33.3%** | 0.00 |
| B_self_note       | 15.00 | 0.00 | 0.00 | **100.0%** | 0.00 |
| C_oracle_note     | 15.00 | 0.00 | 0.00 | **100.0%** | 0.00 |
| D_raw_transcript  | 15.00 | 0.00 | 0.00 | **100.0%** | 0.00 |

Raw data: [`results.json`](results.json) · [`scored.json`](scored.json)
Run log: [`harness_log.txt`](harness_log.txt) · [`rerun_b_log.txt`](rerun_b_log.txt)

### Per-constraint CORRECT rate (cold failures, in rank order)

These cold-failure modes didn't exist in Phase 3A — they are the new
constraints that cut against priors. Every notebook condition caught
them 4/4.

| constraint | cold | cold violations |
|---|---|---|
| 01_platforms (Win/Linux, no Mac)       | 0/4 | 0 |
| 02_python_311                          | 0/4 | 2 |
| 04_p99_200ms                           | 0/4 | 1 |
| 06_keycloak                            | 2/4 | 2 |
| 08_artifactory (no Docker Hub)         | 0/4 | 0 |
| 09_vault_secrets (no env vars)         | 1/4 | 0 |
| 10_postgres_queue (no Redis)           | 0/4 | **4** |
| 12_minio_sse (no local FS photos)      | 1/4 | 0 |
| 13_audit_format_v3                     | 0/4 | 0 |
| 14_company_logging                     | 0/4 | 0 |
| 03_postgres_15                         | 1/4 | 3 |
| 05_tls_no_http                         | 4/4 | 0 |
| 07_offline_client                      | 4/4 | 0 |
| 11_idempotent_jobs                     | 3/4 | 0 |
| 15_utc_local_display                   | 4/4 | 0 |

**Cold's most reliable failure modes:** Redis as the queue (4/4
violations), managed SSO instead of Keycloak (2/4 violations), Go/Rust
for backend (2/4 violations), PostgreSQL version drift (3/4 violations —
often just says "Postgres" without specifying 15).

## What this means

- **Notebook gap widens with task difficulty.** Phase 3A: B − A = +35.7
  pts. Phase 3B: B − A = +66.7 pts. The harder the task, the larger the
  recovery a notebook provides.
- **Self-curation is still sufficient at 15 constraints / ~2500 words.**
  Qwen's self-written note averages ~8500 chars with 21–30 captured
  items per session 1 — i.e. it paraphrases the docs rather than
  compressing them. Oracle note is 4830 chars; self-notes are nearly 2×
  that length.
- **Raw transcript also ceilings at this scale.** At ~28000 chars of
  session 1 transcript, Qwen still recovers 100% — the extra bulk is
  not noise yet.

## What this does NOT show

- **Still a ceiling.** B/C/D all hit 100%. The 3A→3B scaling broke
  cold (64% → 33%) but not the notebook floor. To distinguish B from
  C from D we need a task where self-notes have to actually compress.
- **Self-curation at this scale is not curation.** Qwen is producing
  an ~8500-char paraphrase of a ~16000-char doc set. That's
  lossless-ish enumeration, not compression. A clean self-note /
  oracle comparison needs a regime where the self-note is forced to
  drop information.

## Pre-registered falsification outcome

| criterion | result |
|---|---|
| *Falsified if B ≤ A* | NOT falsified: B (100%) − A (33.3%) = +66.7 |
| *Falsified if C ≈ B* (self = oracle) | **FALSIFIED again** at this scale: ceiling still masks the oracle-vs-self difference |
| *Falsified if D ≫ C* (raw beats curated) | NOT falsified: D = C |
| *Low hallucinated continuity in B* | OK: 0.00 |

## What went wrong operationally

- **Original B runs all captured ctx=64c** (empty notes). Qwen skipped
  the `save_session_summary` tool and replied in free text. Fixed with
  stronger system prompt (tool-only directive) plus a nudge-retry loop
  in `chat_with_tool`. Re-ran B alone (see `rerun_b.py` and
  `rerun_b_log.txt`).
- **Scorer false positives** found during review and fixed:
  - *"after the mysql 8 failover incident"* — historical reference, not
    a proposal. Added incident/failover/outage to negation markers.
  - *"sqlite 3.40+ as the local primary data store"* — client-side
    offline store, not server primary. Tightened violation pattern to
    `\bserver[-\s]?side\s+sqlite\b`.
  - *"plaintext http is blocked at the ingress"* — "blocked" now in
    negation list.
  - *"library wraps stdlib logging"* — "wraps" now in negation list;
    the library is `company-common-logging` and this is a description,
    not a proposal.

## Next step

Phase 3C / 3B² needs to actually break the self-note ceiling. Three
levers, roughly in order of experimental yield:

1. **Multi-session chain.** Session 1 reads docs → note 1. Session 2
   begins design → note 2. Session 3 finalizes → proposal. Self-note
   must survive two transitions (and re-summarize at each hop). This
   is the closest analog to real Claude-Code-style memory use.
2. **Force compression.** Cap the self-note at 1500 chars via prompt
   ("you have 1500 chars; choose what matters"). If oracle-at-1500
   beats self-at-1500, curation IS the bottleneck at the compressed
   size.
3. **Harder docs.** 30–50 constraints across 8000+ words, with
   deliberate red-herring sentences that contradict the real
   constraint. Notebook must record the resolution, not just the
   surface text.

Option 1 is the most informative for the yantrikdb product claim:
structured memory's advantage is across sessions, not within them.
A Phase 3C that compares `yantrikdb.remember / yantrikdb.recall`
against a plain-text file across a 3-session chain would be the first
experiment whose outcome differentiates this project from
`notes.md`.

## Honest limits (carried from 3A)

- n=4 per cell.
- One model (Qwen 3.6 MoE, Q4). Bigger models likely saturate A cold too.
- Auto-scored via regex with clause-level negation awareness. Four
  false positives found and fixed during review; more could exist.
- Qwen's self-notes are ~2× the length of the oracle — closer to a
  paraphrase than a summary. A real "curation" comparison requires a
  regime where self-notes have to compress.
