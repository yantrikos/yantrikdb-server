# RFC 006 — Context-Aware Conflict Detection

**Status**: Draft
**Target**: v0.5.13 (Phase 0), v0.6.0 (Phase 1), v0.6.1 (Phase 2), v0.6.2 (Phase 3), v0.6.3 (Phase 4), v0.7 (Phase 5)
**Author**: spranab
**Brainstorm partner**: GPT-5.4 (3-round redteam debate, session `a18d7656`)
**Motivated by**: HN comment thread on v0.5.11 launch, issue #3 (bench-surfaced false merges), and the entity extraction fix in commit ad8e655 which exposed the deeper architectural gap.

## Problem

YantrikDB's v0.5.12 conflict detection works at the naive level: same subject + cosine-similar-with-different-claim ⇒ flag as conflict. This false-positives on four real-world cases that a bench run and an HN commenter both surfaced:

1. **Scope / different-org same-role.** "Matt Garman is CEO of AWS" vs "Andy Jassy is CEO of Amazon" — same role, different orgs, NOT a conflict. Today flags because extraction produces only entities `[Matt Garman, CEO, AWS]` and `[Andy Jassy, CEO, Amazon]` with no graph edges to carry the (subject, relation, object) triple.
2. **Different relation / same subject.** "Alice is CEO of Acme" vs "Sarah is CTO of Acme" — different roles, not a conflict. Today's extractor doesn't distinguish the relation type.
3. **Temporal succession.** "Alice CEO in 2023" vs "Bob CEO in 2026" — both true in their respective windows. No time-of-validity modeled.
4. **Multi-valued relations.** "Alice is co-CEO" + "Bob is co-CEO" — legitimate simultaneous claims. Cardinality is treated as implicit `SINGLE_VALUED` in existing `IDENTITY_REL_TYPES`.

My v0.5.12 bench seeded 6 contradictions in 59 memories. `think()` flagged 60 — high recall, low precision. ~54 were noise including all four failure modes above.

## Non-goals

- Full NER / general IE in the Rust core (defer to upstream LLM extraction where available; core provides heuristic fallback only).
- Destructive truth resolution in storage (storage remains append-only / provenance-preserving; resolution is a derived read-time view).
- Universal ontology for relations (each tenant/namespace can override policy).
- Numeric probabilistic confidence ("0.83") — defer until we have calibration data.

## Design: Claims as the primitive

The key insight from the brainstorm: **the missing abstraction is *claims*, not edges**. Today's `edges` table is being asked to do two jobs — act as a semantic graph AND act as a claim ledger. Claims and edges are different: a single memory can contain multiple claims, each with independent qualifiers (time, polarity, modality, source).

For v0.6, we **extend `edges` with claim-like qualifier columns** rather than creating a new table (migration pragmatism — live cluster, two weeks of work for a proper claims table is too heavy). For v0.7 we rename to `claims` and break multi-claim-per-memory as a first-class concept.

### Five layers

#### Layer A — Claim extraction and storage
Primary lever. Two entry paths:

1. **Structured claim ingest API** — `POST /v1/claim { src, rel_type, dst, polarity, modality, valid_from?, valid_to?, source_memory_rid }`. For agents with LLM access doing extraction upstream.
2. **Heuristic extractor fallback** — for Rust-only embeddable mode. Narrow whitelist of high-precision patterns (10-20 max). Not spaCy in Rust.

Both paths write to the same extended `edges` table (to become `claims` in v0.7).

Extended columns:
- `source_memory_rid TEXT` — provenance back to the memory text
- `polarity INTEGER NOT NULL DEFAULT 1` — `1=positive, -1=negative, 0=unknown`
- `modality TEXT NOT NULL DEFAULT 'asserted'` — `asserted | reported | hypothetical | denied | quoted`
- `valid_from REAL` — nullable; world-validity start
- `valid_to REAL` — nullable; world-validity end (null = present)
- `extractor TEXT NOT NULL DEFAULT 'manual'` — `manual | structured_ingest | heuristic_v1 | agent_llm`
- `extractor_version TEXT`
- `confidence_band TEXT NOT NULL DEFAULT 'medium'` — `low | medium | high`
- `span_start INTEGER` — byte offset in source memory (nullable for manual)
- `span_end INTEGER`

#### Layer B — Entity aliasing (v0.6.0-critical per dispute)

Without aliasing, the scoped conflict story breaks on the HN commenter's own examples: `AWS` vs `Amazon Web Services` remain different entity IDs, their `ceo_of` edges never compare, conflict scan is silent. My position: aliasing ships in v0.6.0, not v0.6.1.

New table:
```sql
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    source TEXT NOT NULL CHECK(source IN ('explicit', 'auto_suggested', 'approved')),
    created_at REAL NOT NULL,
    PRIMARY KEY (alias, namespace)
);
CREATE INDEX IF NOT EXISTS idx_alias_canonical ON entity_aliases(canonical_name, namespace);
```

v0.6.0 populates only `source='explicit'` (via new `POST /v1/alias` API). Conflict scan normalizes src/dst through alias lookup before comparison.

Automatic alias suggestion (nightly job: cosine similarity on name embeddings + shared co-mention signals) deferred to v0.6.1.

#### Layer C — Temporal and modality qualifiers

Schema columns already in Layer A. This layer adds the **semantics**: conflict scan must use `valid_from/valid_to` overlap for high-severity flagging.

Rules:
- Claim A and B have `valid_from/valid_to` AND they overlap → `overlapping_validity` evidence, severity candidate high
- Claim A and B have `valid_from/valid_to` AND they DON'T overlap → NOT a conflict; auto-derive `status=superseded` for older
- Claim A or B missing time → downgrade severity to medium; emit reason code `missing_temporal_qualifier`

`polarity=negative` and `modality IN (hypothetical, denied, quoted)` claims are **excluded from default conflict scan** but retained in DB for richer later analysis.

#### Layer D — Relation conflict policy registry

Replaces the blunt `SINGLE_VALUED | MULTI_VALUED | PREFERENCE` enum. Per-relation policy:

```sql
CREATE TABLE IF NOT EXISTS relation_policies (
    relation_type TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT '*',  -- '*' = global default
    uniqueness_scope TEXT NOT NULL,  -- JSON array, e.g., ["object_entity_id"]
    overlap_allowed INTEGER NOT NULL DEFAULT 0,  -- 1 if multiple dsts normal
    temporal_required INTEGER NOT NULL DEFAULT 0,  -- 1 if conflict needs overlap
    missing_time_severity TEXT NOT NULL DEFAULT 'medium',
    qualifier_exceptions TEXT,  -- JSON: e.g., ["qualifier=co", "qualifier=interim"]
    PRIMARY KEY (relation_type, namespace)
);
```

Seeded policies for v0.6.0 whitelist of 12 relations: `ceo_of`, `cto_of`, `founded`, `works_at`, `born_in`, `headquartered_in`, `parent_of`, `married_to`, `speaks`, `located_in`, `acquired`, `subsidiary_of`.

Namespace-scoped overrides: tenant `acme` can declare `ceo_of` allows co-CEOs; default remains single-valued.

#### Layer E — Conflict surfacing / derived resolution states

No destructive resolution. Every claim gets a computed `status` at read time:

| Status | Meaning |
|---|---|
| `active` | Current claim, no active contradictions |
| `superseded` | Has later non-overlapping claim for same (subject, rel_type, scope) |
| `historical` | `valid_to < now` and explicitly time-bounded |
| `conflicted` | Active contradictions flagged; agent should review |
| `low_confidence` | Confidence band = low OR derived from uncorroborated heuristic |
| `uncorroborated` | Single source memory; no corroborating claim |

Conflicts get:
- `severity_band` ∈ `low | medium | high`
- `reason_codes[]` — machine-readable taxonomy
- `evidence_ids[]` — claim IDs on both sides
- `status_suggestion` — one of the statuses above (not written to storage)

## Reason-code taxonomy (initial)

| Code | Trigger | Default severity |
|---|---|---|
| `same_subject_same_relation_distinct_object` | Base conflict pattern | medium |
| `overlapping_validity_windows` | When combined with above | high |
| `missing_temporal_qualifier` | One side missing time | medium |
| `possible_temporal_succession` | Non-overlapping windows | downgrade to info |
| `multi_valued_relation_policy` | Policy allows multi-dst | suppress |
| `negation_detected` | Claim polarity differs | medium (excluded from default scan) |
| `modality_mismatch` | One asserted, one reported/hypothetical | low |
| `entity_link_ambiguous` | Alias suggestion unresolved | medium |
| `low_relation_extraction_confidence` | Heuristic emitted but pattern is weak | low |
| `cross_namespace_coexistence` | Same fact in different namespaces | suppress |
| `corroborated_multi_source` | Multiple source memories agree | boost to high |
| `self_reinforcing_duplicate_skipped` | Already have this claim | suppress |

## Heuristic relation patterns (v0.6.0 whitelist)

12 patterns. Each includes: pattern template (against tokenized spans), relation emitted, negation-scope window (6 tokens back), modality cue detection.

1. `<PERSON> is the <ROLE> of <ORG>` → `role_of(PERSON, ORG)` (e.g., `ceo_of`, `cto_of`, `founder_of`)
2. `<PERSON> was the <ROLE> of <ORG>` → `role_of` with `valid_to=inferred_past`
3. `<ORG>'s <ROLE>, <PERSON>` → `role_of(PERSON, ORG)` (appositive)
4. `<PERSON> leads <ORG>` → `leads(PERSON, ORG)`
5. `<PERSON> founded <ORG>` → `founded(PERSON, ORG)`
6. `<PERSON> works at <ORG>` → `works_at(PERSON, ORG)`
7. `<PERSON> was born in <PLACE>` → `born_in(PERSON, PLACE)`
8. `<ORG> is headquartered in <PLACE>` → `headquartered_in(ORG, PLACE)`
9. `<PERSON> is married to <PERSON>` → `married_to(PERSON1, PERSON2)`
10. `<ORG> acquired <ORG>` → `acquired(ORG1, ORG2)`
11. `<ORG> is a subsidiary of <ORG>` → `subsidiary_of(ORG_CHILD, ORG_PARENT)`
12. `<PERSON> speaks <LANGUAGE>` → `speaks(PERSON, LANGUAGE)`

Negation cues (window): `not`, `no`, `never`, `denied`, `refuted`, `isn't`, `wasn't`, `disputes`, `denies`.
Modality cues (window): `may`, `might`, `allegedly`, `reportedly`, `rumor`, `plans to`, `will be`, `according to`.

When negation cue present → emit with `polarity=negative`. When modality cue present → emit with matching `modality` value. Default is `polarity=positive, modality=asserted`.

Compound sentence handling (minimal for v0.6.0): split on `;`, `, then `, `, subsequently ` before extraction fires. Each sub-sentence is a candidate claim.

## Structured claim ingest API

```
POST /v1/claim
Authorization: Bearer <token>
Content-Type: application/json

{
  "src": "Alice Chen",
  "rel_type": "ceo_of",
  "dst": "Acme Corp",
  "polarity": "positive",           // optional; default positive
  "modality": "asserted",           // optional; default asserted
  "valid_from": 1672531200,         // optional
  "valid_to": null,                 // optional
  "source_memory_rid": "019d...",   // optional but recommended
  "namespace": "default",           // optional
  "extractor": "agent_llm",         // optional; default manual
  "confidence_band": "high"         // optional; default medium
}

Response:
{
  "claim_id": "019e...",
  "created_at": 1712345678.9,
  "namespace": "default",
  "status_suggestion": "active"
}
```

Validation:
- `src` and `dst` must not be empty
- `rel_type` must match `[a-z_][a-z0-9_]*`
- `polarity` ∈ `positive|negative|unknown`
- `modality` ∈ `asserted|reported|hypothetical|denied|quoted`
- If both `valid_from` and `valid_to` set, `valid_from <= valid_to`

## Conflict scan pseudocode (v0.6.0)

```
for each new or existing claim C in namespace N:
  policy = lookup_policy(C.rel_type, N)  # falls back to '*' namespace
  if policy is None:
    continue  # relation not whitelisted for conflict detection

  if C.polarity != positive:
    continue  # negations excluded from default scan

  if C.modality not in (asserted, reported):
    continue  # hypothetical/denied/quoted excluded

  # Find candidate counter-claims
  src_canonical = resolve_alias(C.src, N)
  candidates = SELECT * FROM edges
    WHERE namespace = N
      AND rel_type = C.rel_type
      AND resolve_alias(src, N) = src_canonical
      AND rid != C.rid
      AND polarity = positive
      AND modality IN (asserted, reported)

  for D in candidates:
    dst_c = resolve_alias(C.dst, N)
    dst_d = resolve_alias(D.dst, N)

    # Check uniqueness-scope from policy
    if policy.uniqueness_scope == ["object_entity_id"]:
      if dst_c == dst_d:
        continue  # same subject+rel+object: not a conflict, corroboration

    if policy.overlap_allowed and dst_c != dst_d:
      continue  # multi-valued relation; no conflict

    # Different dsts on a uniqueness-scoped relation → potential conflict
    reason_codes = ["same_subject_same_relation_distinct_object"]
    severity = policy.missing_time_severity  # default medium

    # Apply temporal logic
    c_has_time = C.valid_from is not None or C.valid_to is not None
    d_has_time = D.valid_from is not None or D.valid_to is not None
    if c_has_time and d_has_time:
      if intervals_overlap(C, D):
        reason_codes.append("overlapping_validity_windows")
        severity = "high"
      else:
        reason_codes.append("possible_temporal_succession")
        continue  # suppress; derive status=superseded on older
    else:
      reason_codes.append("missing_temporal_qualifier")
      # severity stays at policy.missing_time_severity

    record_conflict(C, D, severity, reason_codes)
```

## Sequencing and scope per release

| Version | Scope | Est days | Ship criteria |
|---|---|---|---|
| **v0.5.13** (Phase 0) | Extraction audit oplog event. **Zero behavior change.** Log every extraction attempt: pattern match?, relation type, entities linked, conflict candidates generated, final severity. | 0.5 | Deploy to Proxmox cluster, collect 7 days of real data before v0.6.0 |
| **v0.6.0** (Phase 1) | Extended `edges` schema with claim columns. 12-pattern heuristic extractor. `POST /v1/claim` structured ingest API. `entity_aliases` table + `POST /v1/alias`. Scoped conflict scan with policy lookup. Severity bands + reason codes. Negation/modality handling at extraction. Compound sentence splitting. | 4 | Bench precision ≥ 70% on seeded contradictions, no full-suite regression |
| **v0.6.1** (Phase 2) | Temporal qualifiers in conflict logic (overlap check, auto-supersede). Derived status views. Automatic alias suggestion job (nightly). Self-duplicate prevention. | 3 | `status_suggestion` returns correct values on a temporal-succession bench |
| **v0.6.2** (Phase 3) | Relation conflict policy registry table + namespace overrides. Seed policies for 12 starter relations. Agent-facing policy CRUD API. | 2 | Tenant can override `ceo_of` to multi-valued without schema change |
| **v0.6.3** (Phase 4) | Multi-claim-per-memory (via claims table rename from edges, or physical split). Corroboration weighting. Polarity/modality fully in default scan (not just excluded). | 3 | Multi-claim memory correctly produces N distinct claims with independent qualifiers |
| **v0.7** (Phase 5) | Full `claims` table abstraction. Deprecate raw-edge writes. Entity linking improvements. Optional dependency-parsing for better pattern coverage. | 5 | `edges` table becomes a view over `claims`; all new writes go through claim API |

## Open questions

1. **Compound sentence handling in v0.6.0** — I proposed splitting on `;`, `, then `, `, subsequently `. This is lossy. Proper handling requires sentence-boundary detection. Should we ship the lossy version in v0.6.0 and tighten in v0.6.3, or defer all compound handling to v0.6.3?

2. **Reason-code stability** — once agents start filtering on reason codes, changing them is a breaking contract. Proposal: reason codes are stable from v0.6.0 onwards; new ones can be added but existing ones can't be renamed or semantic-changed.

3. **Claim immutability** — should claims be append-only (new version supersedes, old preserved) or mutable via `PATCH /v1/claim/{id}`? Append-only is safer for provenance; mutable is agent-friendlier. Lean: append-only, with `POST /v1/claim/{id}/correct` for corrections that preserve history.

4. **Cross-namespace aliasing** — can `AWS` in namespace `corpA` alias to the same canonical as `AWS` in namespace `corpB`? Probably not for privacy/isolation. Each namespace has its own alias table, but we could expose a `default` namespace for global aliases. Decide before v0.6.0.

## References

- Issue #1 — consolidation merges across different entities (fixed partial in v0.5.12)
- Issue #2 — `/v1/remember` doesn't auto-extract entities (fixed in v0.5.12)
- Issue #3 — tighten consolidation guard (folded into this RFC)
- HN comment thread: https://news.ycombinator.com/item?id=<TBD>
- Brainstorm session transcript: `docs/rfcs/006-brainstorm-transcript.md` (to be saved)
- Bench v2 results: https://gist.github.com/spranab/49c618d3625dc131308227103af5aadd

## Appendix A — Phase 0 audit event schema (v0.5.13)

```json
{
  "op_type": "extraction_audit",
  "timestamp": 1712345678.9,
  "namespace": "default",
  "memory_rid": "019d...",
  "source_memory_text_preview": "Alice Chen is the CEO...",
  "extractor_version": "heuristic_v1",
  "candidates_considered": 12,
  "patterns_matched": ["pattern_1", "pattern_3"],
  "entities_before_linking": ["Alice Chen", "CEO", "Acme Corp"],
  "entities_after_linking": ["Alice Chen", "Acme Corp"],
  "relations_emitted": [
    {"src": "Alice Chen", "rel_type": "ceo_of", "dst": "Acme Corp", "confidence_band": "medium"}
  ],
  "negation_cues_detected": [],
  "modality_cues_detected": [],
  "compound_sentence_splits": 0,
  "conflict_candidates_generated": 0,
  "final_severity": null
}
```

Stored in `oplog` table with `op_type='extraction_audit'`. Not replicated across cluster (local debug data only). Purged after 30 days by the decay worker.
