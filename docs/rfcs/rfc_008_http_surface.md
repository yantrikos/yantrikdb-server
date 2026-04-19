# RFC 008 HTTP Surface

The five HTTP endpoints that expose the Warrant Flow substrate (claims with source_lineage, mobility state, contest state, cognitive moves) to agents and MCP clients. Added in yantrikdb-server commit f245049; requires yantrikdb core at branch `main` (≥ commit cd41207, post-M10).

All endpoints use the standard server auth (`Authorization: Bearer <token>`) and tenant-pool resolution. Namespace parameter defaults to `default`; regime parameter defaults to `default`.

---

## 1. `POST /v1/claim_with_lineage`

Ingest a claim with explicit source_lineage. This is the primary differentiator from `/v1/claim` — the source_lineage field drives ⊕'s dependence discount.

### Request body
```json
{
  "src": "Philippine_trustee_accounts",
  "rel_type": "balance_equals",
  "dst": "EUR_1.9_billion",
  "namespace": "wirecard-demo",
  "polarity": 1,
  "modality": "asserted",
  "valid_from": 1546214400.0,
  "valid_to": 1592438400.0,
  "extractor": "ey.audit",
  "extractor_version": "fy2018",
  "confidence_band": "medium",
  "source_memory_rid": null,
  "weight": 1.0,
  "source_lineage": ["wirecard", "ey"]
}
```

### Response
```json
{
  "claim_id": "<uuid-v7>",
  "proposition_id": "<uuid-v7>",
  "namespace": "wirecard-demo",
  "source_lineage": ["wirecard", "ey"]
}
```

### When to use
Whenever you have provenance labels — citation chains, upstream authorship, auditor-vs-source dependence. An agent processing news should populate source_lineage with the cited-source chain; a research agent should populate with paper citation graph.

---

## 2. `GET /v1/mobility?src=X&rel_type=Y&dst=Z&namespace=N&regime=R`

Read the 13-dim mobility state for a proposition. Returns the Warrant Flow vector: σ (dependence-discounted support), α (attack), τ (temporal coherence), λ (load-bearingness), ψ_a (self-gen ancestral), χ (modality consilience), ψ_l (self-gen local), plus source_diversity / effective_independence and six other components populated at different tiers.

### Required query params
`src`, `rel_type`, `dst` — the triple. `namespace` and `regime` optional (default `default`).

### Response (populated)
```json
{
  "proposition_id": "<uuid>",
  "regime": "default",
  "mobility_state": {
    "proposition_id": "<uuid>",
    "regime": "default",
    "snapshot_ts": 1745011234.123,
    "support_mass": 1.600,
    "attack_mass": 3.578,
    "modality_consilience": 0.167,
    "self_gen_local": 0.0,
    "temporal_coherence": 0.8,
    "load_bearingness": 4.0,
    "self_gen_ancestral": 0.0,
    "tier_write_components": ["support_mass", "attack_mass", "modality_consilience", "self_gen_local"],
    "tier_bg_components": ["temporal_coherence", "load_bearingness", "self_gen_ancestral"],
    "formula_version": 1,
    "content_hash": "<blake3>",
    "live_claim_count": 6,
    "state_status": "fresh",
    "computed_at": 1745011234
  }
}
```

### Response (no proposition)
```json
{
  "mobility_state": null,
  "proposition_id": null,
  "reason": "proposition not found — no claims ingested for this triple yet"
}
```

### When to use
Before making an assertion about a contested claim. An agent that says "Wirecard's €1.9B was confirmed by EY" should first call this and see σ=1.6 (not σ=2.0) before deciding how confidently to state the conclusion.

---

## 3. `GET /v1/contest?src=X&rel_type=Y&dst=Z&namespace=N&regime=R`

Read contest state (Γ(c) — the structured contradiction signature, not a scalar). Returns 5 grounded counters + 5 heuristic flags + on-demand exemplar claim_id pairs.

### Response
```json
{
  "proposition_id": "<uuid>",
  "regime": "default",
  "contest_state": {
    "proposition_id": "<uuid>",
    "regime": "default",
    "support_mass": 1.600,
    "attack_mass": 3.578,
    "support_effective_independence": 1.600,
    "attack_effective_independence": 3.578,
    "support_distinct_source_count": 2,
    "attack_distinct_source_count": 4,
    "same_source_opposite_polarity_count": 0,
    "same_artifact_extractor_polarity_conflict_count": 0,
    "temporal_overlap_conflict_count": 4,
    "temporal_separable_opposition_count": 4,
    "referent_schema_heterogeneity_count": 0,
    "heuristic_flags": 16,
    "derivation_version": 1,
    "content_hash": "<blake3>",
    "live_claim_count": 6,
    "state_status": "fresh",
    "computed_at": 1745011234
  },
  "exemplar_pairs": {
    "proposition_id": "<uuid>",
    "regime": "default",
    "heuristic_flags": 16,
    "same_source_opposite_polarity_pairs": [],
    "same_artifact_extractor_conflict_pairs": [],
    "temporal_overlap_conflict_pairs": [
      ["claim_id_A", "claim_id_B"],
      ["claim_id_A", "claim_id_C"],
      ["claim_id_D", "claim_id_B"],
      ["claim_id_D", "claim_id_C"]
    ]
  }
}
```

### Heuristic flag bitmap
- Bit 0 (`0x01`) — DUPLICATION_RISK (support_mass > 2 AND support_effective_independence < 2)
- Bit 1 (`0x02`) — SAME_SOURCE_CONFLICT
- Bit 2 (`0x04`) — REFERENT_HETEROGENEITY_PRESENT
- Bit 3 (`0x08`) — SAME_ARTIFACT_EXTRACTOR_CONFLICT
- Bit 4 (`0x10`) — PRESENT_TENSE_CONFLICT

### When to use
Whenever σ − α alone would be misleading. "Five sources agree" vs "five sources amplifying one source" are the same scalar but different contests.

---

## 4. `POST /v1/move_events`

Record a cognitive move — analogy, decomposition, source_audit, contradiction_triage, etc. Append-only. Drives distortion profile derivation (M9) and auto-files adversarial candidates on retraction (M10).

### Request body
```json
{
  "move_type": "contradiction_triage",
  "operator_version": "v1",
  "context_regime": "default",
  "observability": "observed",
  "inference_confidence": null,
  "inference_basis": null,
  "input_claim_ids": ["claim_wirecard", "claim_ey"],
  "output_claim_ids": ["claim_ft_denial"],
  "side_effect_claim_ids": [],
  "dependencies": []
}
```

### Response
```json
{ "move_id": "<uuid-v7>" }
```

### Vocabulary (soft registry — unknown types warn but don't reject)
`analogy`, `decomposition`, `aggregate_back`, `negate_and_test`, `source_audit`, `ladder_up`, `contradiction_triage`, `source_downgrade`, `source_upgrade`, `regime_transfer`, `compression`, `hypothesis_generation`, `quarantine`.

### Observability
`observed` — agent logs what it did in real-time. `self_reported` — agent declares a move after the fact. `inferred` — system reconstructs that a move must have happened. Only `inferred` may set `inference_confidence` and `inference_basis`.

### When to use
Every time the agent transforms claims via reasoning. The move_events log is the audit trail that lets the substrate learn which move types produced retracted outputs (M9 profile) and which propositions get contested after which moves (M10 auto-candidate).

---

## 5. `GET /v1/flagged_propositions?flag_mask=N&limit=M`

List propositions whose contest_state.heuristic_flags intersects the given bitmask. Primary audit entry point.

### Query params
- `flag_mask` — u64 bitmask; `0` returns empty by design
- `limit` — default 50

### Response
```json
{
  "flagged_propositions": [
    { "proposition_id": "...", "regime": "default", "heuristic_flags": 16, ... },
    ...
  ],
  "flag_mask": 16,
  "count": 1
}
```

### When to use
Batch audit / review dashboards. "Show me everything that has a present-tense contradiction": `flag_mask=16`. "Show me everything flagged as same-source conflict OR same-artifact-extractor conflict": `flag_mask=10`. Orthogonal to per-proposition lookups via `/v1/contest`.

---

## Deployment notes

- Server rebuild: `cargo build --release -p yantrikdb-server`. Typical cold build: 3-10 min on Windows. Branch `main` of yantrikdb must be reachable via git.
- Schema migration V22→V23 fires automatically on server startup; no manual DB migration step.
- Existing `/v1/*` endpoints unchanged. Adding these five is additive.

## MCP exposure

If the MCP server (the SSE endpoint at the yantrikdb MCP config URL) auto-generates tool schemas from HTTP routes, these should appear automatically after server restart. If it requires manual schema entries, the shape above is authoritative for request/response bodies.

---

*Part of the [RFC 008](https://github.com/yantrikos/yantrikdb/blob/main/crates/yantrikdb-core/src/engine/warrant.rs) Warrant Flow substrate. Introduced in server f245049.*
