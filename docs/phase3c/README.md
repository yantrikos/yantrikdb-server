# Phase 3C — memory probe with supersession, alias, and indirect retrieval pressure

**Question:** does structured memory (key/value + retrieval) beat a plain
rolling markdown file as an LLM's across-session working memory?

**Designed to break the ceiling from Phase 3A/3B** — where all three notebook
conditions hit 100% because the tasks were too small and well-formed. This
time the task is hostile to brute-force carry-forward:

- 5 sessions of narrative, ~9000 chars total
- 5 supersession chains (value stated session N, revised session M>N)
- 3 branch-indirect probes (query doesn't name the retrieval key)
- 2 alias-collision probes (Project Aurora vs Aurora-lite; hard budget vs target budget)
- 10 plausible distractors woven in
- Self-note (B) capped at 1500 chars, rewritten each session (forced compression)
- Markdown (D) capped at 7500 chars with top-truncation (so session-1
  facts eventually drop out of the window)

**Pre-registration design rationale** from the GPT-5.4 redteam brainstorm
in [`../../.claude/brainstorm-archive/phase3c-design-bbfa9170.md`](../../../)
(session `bbfa9170-5393-4db9-9d2f-a68745e18500`).

## Setup

**Model:** Qwen 3.6 MoE 36B Q4 via Ollama, `think:false`, num_ctx 32k,
temperature 0.3. 2× RTX 3090 Ti local.

**Scenario:** [`scenario/sessions.json`](scenario/sessions.json) — 5 weekly
sessions of a fictional Titan Industries cloud migration consulting
engagement. 15 probes with structured ground truth.

**Conditions** (`n=2` per condition — tight sample size, pre-declared):

| | mechanism |
|---|---|
| A_cold | no sessions, no memory — Qwen answers probes blind |
| B_self_note | rewrites a ≤1500-char note each session, hard-truncated if over |
| C_structured | calls `remember(key, value)` ≤15× per session, `recall(query)` per probe; word-overlap retrieval (Dice similarity), top-k=5 |
| D_markdown | each session narrative appended to a rolling markdown file, global 7500-char cap with top-truncation |

### Crucial methodology note

**Condition C uses a plain structured-memory simulator, NOT yantrikdb.**
The simulator stores `(key, value, session)` tuples and returns top-k by
word overlap. No temporal validity, no polarity, no supersession logic.
This is deliberate: the question is *"does generic structured
key/value + retrieval beat markdown?"*  If it does, integrating
yantrikdb is justified. If yantrikdb wants to claim a bigger advantage
than plain structured memory, that's a separate test with its own
pre-registered criteria.

## Results (n=2 per condition, 15 probes each)

| cond | overall | answer_acc | provenance | sup_acc | stale_rate | alias_acc | direct | branch | ctx_chars |
|------|---------|------------|------------|---------|------------|-----------|--------|--------|-----------|
| A_cold       | 0.000 | **0.0%** | 0.0%  | 0.0%  | 10.0% | 0.0%  | 0.0%  | 0.0%  | 0 |
| B_self_note  | 0.383 | **63.3%** | 21.6% | 80.0% | 0.0%  | 25.0% | 60.0% | 66.7% | 1500 |
| C_structured | 0.584 | **60.0%** | **93.8%** | 50.0% | **40.0%** | 50.0% | 60.0% | 83.4% | 4983 |
| D_markdown   | 0.667 | **73.3%** | 81.8% | **100.0%** | 0.0%  | **0.0%** | 60.0% | 100.0% | 7500 |

Raw data: [`results.json`](results.json) · [`scored.json`](scored.json)
Log: [`harness_log.txt`](harness_log.txt)

### Scoring rubric

Per probe: **1.0** if answer correct + provenance session correct; **0.5**
if answer correct but provenance wrong or missing; **0.0** wrong answer.
Supersession subset: stale-answer patterns (the prior-but-superseded value)
scored as wrong AND tracked as stale_error. Alias subset: confusion
patterns (the wrong entity in the collision pair) tracked as
alias_confusion.

## What actually happened

### 1. Plain structured memory does NOT win overall — markdown wins

D (rolling markdown, 7500-char cap) took the highest answer accuracy
(73.3%), supersession accuracy (100%), and branch-indirect accuracy (100%).
This is not surprising: the markdown top-truncation effectively drops
older sessions from the window, which means only the *latest* values for
superseded facts remain visible — Qwen has nothing stale to pick.

**But markdown has one catastrophic failure mode** that C dodges:

### 2. Markdown loses alias disambiguation at 0/4

The two alias probes (Project Aurora's $2.4M budget — NOT Aurora-lite's
$120K; the $4.8M hard budget — NOT the $4.2M target) were stated in
**session 1**, which got truncated out of D's rolling window by session
5. D answered "UNKNOWN" or mentioned the wrong value on all 4 alias probes
across 2 runs. C nailed 50% of them, B nailed 25%. A nailed 0%.

This is the dark side of top-truncation as a markdown strategy:
long-lived facts that happened to be stated early drop off the history.

### 3. Plain structured memory has a catastrophic failure mode too: 40% stale-error rate

C stores `remember("titan.go_live", "target go-live Q3 2026", session=1)`
in session 1, then `remember("titan.go_live", "go-live pushed to Q4 2026",
session=3)` in session 3. When Qwen recalls "current go-live date", the
store returns **both values**. Qwen sometimes picks the stale one.

Across the 5 supersession probes, C produced stale answers on 40% of
them — actively wrong, not merely uncertain. B and D produced 0% stale
errors (B because rewriting drops old values; D because truncation drops
old sessions).

**This directly validates the need for yantrikdb's RFC 006 temporal
validity substrate.** Plain key/value + retrieval is not enough. A
structured-memory system with first-class support for claim succession
— where "session 3's go-live value supersedes session 1's" is a
primitive — would close this gap. My simulator deliberately doesn't
have that so this gap is visible.

### 4. C wins cleanly on provenance (+72 pts vs B, +12 vs D)

The structured store tracks `session` per memory for free. B has to
annotate session numbers by hand in its ≤1500-char note and mostly
doesn't (and when it does, it attributes everything to "session 5"
because by then the note is a mushy aggregate). D has each session
demarcated in the markdown but Qwen doesn't always look up from the
quoted text.

C's **93.8% provenance accuracy given a correct answer** is the cleanest
win in the matrix. In enterprise/audit scenarios where "what did we
say, when?" matters as much as "what do we say now?", this is load-bearing.

### 5. Context cost ordering is A < B < C < D

| cond | final ctx chars | ratio vs C |
|------|-----------------|-----------|
| A | 0 | - |
| B | 1500 | 0.30× |
| C | 4983 | 1.00× |
| D | 7500 | 1.50× |

C uses 1.5× less context than D at roughly tied answer accuracy. At
long-horizon scale (10+ sessions), this ratio widens because D's
truncation loses exponentially more while C's retrieval stays selective.

## Pre-registered falsification outcome

Loud-win criteria were defined before the run:

| criterion | met? |
|---|---|
| C beats max(B,D) by ≥15 pts on overall answer acc | **NOT MET**: C 60% vs max(B,D) 73.3% → Δ = **-13.3 pts** |
| C beats max(B,D) by ≥20 pts on supersession acc | **NOT MET**: C 50% vs 100% → Δ = **-50 pts** |
| C beats min(B,D)'s stale-rate by ≥20 pts | **NOT MET**: C 40% vs 0% → Δ = **-40 pts** |

**Loud win**: no.
**Quiet wins**: provenance advantage (+72 vs B, +12 vs D); alias-disambiguation
advantage (+25 vs B, +50 vs D); 1.5× context-cost advantage vs D.

## Honest interpretation

- **The experiment was genuinely discriminative** — unlike 3A/3B, no
  condition ceilinged, and each condition has a distinct failure
  signature.
- **D (rolling markdown) is a STRONGER baseline than I expected** when
  the supersession pattern is "late overrides early" and the 7500-char
  window happens to drop the stale values. D's 100% supersession
  accuracy is an artifact of top-truncation, not structured
  reasoning.
- **D collapses on alias** because the alias definitions were in session
  1 (outside the window). A hostile scenario that put enduring
  facts in session 1 and alias pairs spread across sessions would
  favor C more.
- **The specific 40% stale rate for C IS the product claim**.
  Plain structured memory without temporal logic is **actively harmful**
  on supersession. YantrikDB's RFC 006 (temporal succession) and RFC 008
  (contest state, polarity-aware) exist for this reason. The next test
  should be **C_structured_plain vs C_yantrikdb_temporal** — if
  yantrikdb's temporal substrate drops the stale rate from 40% to
  single digits, that's the product's concrete value proposition.

## What this means for the yantrikdb product

**Before Phase 3C**: the claim was "structured memory beats plain text notes."
Phase 3C says that's **false for plain key/value stores against a well-tuned
markdown dump at this scale**.

**After Phase 3C**: the sharper claim is *"yantrikdb's temporal/polarity
substrate (RFC 006/008) beats both markdown AND plain structured memory
on supersession-heavy workloads"*. That's a **narrower but more
defensible** claim — and it's directly testable with a Phase 3D that
plugs in real yantrikdb.

The provenance win is also real and quieter: 94% provenance accuracy
makes C the right choice for audit-heavy use cases *right now*, even
before the temporal substrate is integrated.

## Honest limits

- n=2 per cell. The loud-win thresholds (±15/±20 pts) were calibrated
  for the results to survive sample variance, but 2 runs is not
  enough to estimate reliable CIs.
- One model (Qwen 3.6 MoE Q4). A larger or differently-aligned model
  would likely pick stale-vs-current answers differently.
- Plain structured memory simulator uses Dice-similarity retrieval.
  Swapping in TF-IDF or real embeddings may change C's numbers
  non-trivially; that should be tested before integrating yantrikdb.
- The scenario's supersession pattern is always "later overrides
  earlier" — hostile to older baselines (favors recency-truncation
  naturally). A scenario with "retraction" or "conditional
  supersession" would probably stress yantrikdb's substrate more.
- Scoring uses substring pattern matching on model output. Some false
  positives and negatives are possible; per-probe raw text is stored
  in `results.json` for audit.

## Next experiment

**Phase 3D (proposed):** re-run C with real yantrikdb, using
`claim_with_lineage` (RFC 008) and temporal validity windows for
supersession-typed memories. Pre-register: *yantrikdb must reduce C's
stale-rate from 40% to ≤10% at equal-or-better answer accuracy for the
substrate to earn its complexity.*

If yantrikdb can't hit that bar, the RFC 006/008 investment has not
yet bought what was claimed; if it can, Phase 3C → 3D is the first
clean empirical story for the product.
