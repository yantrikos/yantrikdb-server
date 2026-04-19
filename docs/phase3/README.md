# Phase 3A — hidden-constraint recovery

**Question.** Does giving an LLM a notebook between sessions preserve enough
intermediate epistemic state to avoid violating hidden constraints it saw
but can't re-read?

**Why this shape.** The previous iteration (Phase 2) tested whether a
YantrikDB substrate with dependence-discount and temporal-split helped
Qwen 3.6 answer adversarial questions. Result: the substrate operator was
mathematically broken for one case, and the agent failed to populate the
substrate correctly in the others. GPT-5.4 red-team (session 8074fecb)
called the follow-up — self-referential RFC 009 design across 5 sessions —
a "demo trap." Switched to a causal design: hidden-constraint recovery
with four control conditions and a pre-registered falsification matrix.

## Setup

- **Task.** Read three spec documents (~1500 words across
  `specs/01_product_context.md`, `02_infrastructure.md`,
  `03_security_compliance.md`). They contain 7 hard constraints scattered
  across prose, some stated explicitly and some buried in incident context
  or out-of-scope callouts.
- **Hidden constraints.**
  1. Windows 11 + Ubuntu 22.04 only — NO macOS.
  2. Python 3.11+ backend.
  3. PostgreSQL 15+ — no MySQL, SQLite-as-primary, or document stores.
  4. TLS end-to-end — no plain HTTP anywhere, including service-to-service.
  5. Self-hosted Keycloak identity — no Auth0/Okta/Google Workspace/Azure AD.
  6. Offline-capable client with sync on reconnect.
  7. p99 server-side latency under 200ms (hard acceptance criterion).
- **Session 2 task.** Write an architecture proposal. Session 2 has FRESH
  context — the spec docs are not available, only whatever the condition
  allows.

## Conditions (4 runs each)

| condition | notebook carried into session 2 |
|---|---|
| **A_cold** | nothing — model writes proposal with no context |
| **B_self_note** | Qwen's own structured note from session 1 via `save_session_summary` tool |
| **C_oracle_note** | hand-crafted ideal summary ([`specs/oracle_note.md`](specs/oracle_note.md)) |
| **D_raw_transcript** | full session-1 transcript (Qwen reading + thinking out loud) |

Model: `qwen3.6:latest` (36B MoE, Q4) via Ollama, `think:false` (thinking
mode disabled to fit in 16k context), temperature 0.3, num_predict 2500.
Hardware: 2× RTX 3090 Ti, prefilled model in VRAM.

## Results

Run matrix ran 2026-04-19 at 06:06. Total wall: ~20 minutes. Raw data:
[`results.json`](results.json). Scored data: [`scored.json`](scored.json).

### Per-condition means (n=4 each)

| condition | correct/7 | silent/7 | violated/7 | accuracy | hallucinated_continuity |
|-----------|-----------|----------|------------|----------|--------------------------|
| A_cold            | 4.50 | 1.50 | 1.00 | **64.3%** | 0.00 |
| B_self_note       | 7.00 | 0.00 | 0.00 | **100.0%** | 0.00 |
| C_oracle_note     | 7.00 | 0.00 | 0.00 | **100.0%** | 0.00 |
| D_raw_transcript  | 7.00 | 0.00 | 0.00 | **100.0%** | 0.00 |

- **CORRECT**: proposal explicitly states the constraint (specific version,
  platform list, named technology). Silence doesn't count as correct.
- **SILENT**: proposal doesn't mention the constraint, neither satisfies
  nor violates it. Model may have been lucky.
- **VIOLATED**: proposal explicitly proposes something the constraint
  disallows.
- **hallucinated_continuity**: fabricated references to prior discussion
  ("as we decided", "building on our earlier work"). Zero across the
  matrix — the model doesn't confuse the notebook for a shared memory.

### Per-constraint correctness rate (CORRECT / n per cell)

| constraint | A_cold | B_self | C_oracle | D_raw |
|---|---|---|---|---|
| 1_platforms (Win/Linux, no Mac) | 0/4 | 4/4 | 4/4 | 4/4 |
| 2_python_version (3.11+)        | 1/4 (v=2) | 4/4 | 4/4 | 4/4 |
| 3_postgres_version (15+)        | 4/4 | 4/4 | 4/4 | 4/4 |
| 4_tls_no_http                   | 3/4 | 4/4 | 4/4 | 4/4 |
| 5_self_hosted_auth (Keycloak)   | 2/4 (v=2) | 4/4 | 4/4 | 4/4 |
| 6_offline_client                | 4/4 | 4/4 | 4/4 | 4/4 |
| 7_p99_200ms                     | 4/4 | 4/4 | 4/4 | 4/4 |

### Pre-registered falsification outcome

Matrix set up 2026-04-19 before running. Outcomes:

| criterion | result |
|---|---|
| *Falsified if B ≤ A on accuracy* | **NOT falsified**: B (100%) > A (64.3%), Δ = +35.7 pts |
| *Falsified if C ≈ B* (self-note hits the oracle ceiling) | **FALSIFIED**: C = B = 100% |
| *Falsified if D ≫ C* (raw transcript beats curated) | **NOT falsified**: D = C |
| *B hallucinated continuity is low* | **OK**: 0.00 across all runs |

## What this means

**Supported:** Carrying a notebook between sessions recovers constraints
that a cold model misses. The +35.7-point delta is not subtle — cold Qwen
defaults to generic priors (Go or Rust microservices instead of Python,
managed SaaS identity instead of self-hosted Keycloak, Linux without
calling out Mac exclusion) and ignores half the constraints.

**Falsified sub-hypothesis:** *At this task size,* the model's own
structured self-note is as good as a hand-crafted oracle. Note-writing is
not the bottleneck. The 7-constraint / 1500-word task fits well inside
Qwen's working memory for a single session-1 pass. The curation step did
not degrade information relative to the oracle.

**Ceiling effect caveat.** B = C = D = 100% means this task doesn't
distinguish the three notebook formats. The design cannot say whether
self-notes would still match the oracle at 30 constraints, or across a
chain of 5 sessions. A useful Phase 3B would raise the task until
self-note and oracle pull apart.

### Where cold fails vs where notebook recovers

Cold Qwen failed worst on:

- **1_platforms (0/4)**: never stated the Win+Linux-only scope or
  mentioned Mac exclusion — just proposed "cross-platform clients".
- **2_python_version (1/4, 2 explicit violations)**: proposed Go or Rust
  for backend components.
- **5_self_hosted_auth (2/4, 2 explicit violations)**: proposed OAuth,
  Auth0, or "SaaS identity provider".

These are all cases where the **constraint contradicts a strong prior**
(Mac is a common dev target; Go is a common backend; managed SSO is
common). Notebook conditions caught 100% of these — the note put the
constraint directly in front of the model.

Cold succeeded on:

- **3_postgres_version, 6_offline_client, 7_p99_200ms**: 4/4 each.
  Postgres is common; offline-first is forced by the narrative ("field
  work in remote locations"); "under 200ms" is a generic sensible target
  that cold Qwen will often state without prompting.

This is a crisp decomposition: notebook value is concentrated on
**constraints that cut against priors**, not constraints that match
priors.

## Next steps

- **Phase 3B (harder task):** pick a task where oracle-note and self-note
  should pull apart. Options:
  - 20+ constraints across ~5000 words.
  - A 3-session chain — session 1 reads docs, session 2 begins design,
    session 3 finalizes. Self-note must survive two transitions.
  - Adversarial constraints — a constraint stated in one doc but
    contradicted by a red-herring sentence in another, requiring the
    notebook to record the resolution.
- **Phase 3C (integration with yantrikdb):** replace the file-based note
  with a yantrikdb `save_memory` call + session 2 `recall` call. Compare
  against B's text-note baseline. This is what would turn a demo into a
  product claim — but only after 3B shows a gap large enough to
  discriminate better retrieval.

## Honest limits

- n=4 per cell. CIs not computed; at 100% accuracy with n=4 the upper
  bound is simply "not worse than ~68% with 95% confidence."
- One model (Qwen 3.6 MoE). Does not generalize to GPT-4-class models —
  they likely saturate A cold too.
- Auto-scoring uses clause-level negation-aware regex; spot-checks found
  three false-positive cases which were fixed. A human review pass of all
  16 proposals would be a safer second-pass — not done.
- Scoring rewards explicit statement (CORRECT) not silence (SILENT).
  Silence is the honest "we don't know" bucket. A model that happens to
  make a compliant choice without stating the constraint is scored as
  SILENT, not CORRECT. This is deliberate: the point is recall, not
  luck.
- The oracle note was drafted by the author, not blind-reviewed. Its
  framing may have nudged the model toward particular phrasing patterns
  that the scorer catches. A blind-reviewed oracle is a refinement for
  Phase 3B.
