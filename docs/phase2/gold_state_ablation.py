#!/usr/bin/env python3
"""Gold-state ablation for RFC 008 Phase 2 negative result.

Per GPT-5.4 post-Phase-2 red-team: before concluding "substrate broken,"
verify whether failures are (a) operator design failure or (b) extraction
failure. This script hand-populates the IDEAL structured state for each
Phase 2 case and records what the substrate says.

Hypothesis:
- Case A (rumor amp): FAILS even under gold state. ⊕ is mathematically
  incapable of flipping 5-vs-2 dependence at the current coefficient
  ceiling (ω_min = 0.4 → σ_min = 2.0 = α).
- Case B (temporal): PASSES under gold state. Extraction was the problem.
- Case C (same-source): PASSES under gold state. Canonicalization was
  the problem.

If the hypothesis holds, RFC 009 must: (1) replace ⊕ for Case A; (2)
build an extractor/canonicalizer for B and C.
"""
from __future__ import annotations
import io, json, sys, urllib.error, urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

SERVER = "http://localhost:8420"
TOKEN = "ydb_31416c8f210b15b679172740795a295cef75b796aeb521dc3d018d98ea9fbadd"

def call(path, method="GET", body=None, params=None):
    url = SERVER + path
    if params:
        from urllib.parse import urlencode
        url += "?" + urlencode(params)
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return {"__error__": f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}"}


# ─── Case A: gold-state rumor amplification ──────────────────────────
# All 5 supports share IDENTICAL lineage [nova_pharma]. Max dependence.
# Independent attacks with disjoint lineage.
# Proposition: XJ_417 reduces_mortality_in cardiac_population_A
# Also push identical extractors and self_generated flags to maximize
# discount — the adversarial best-case for ⊕.

def case_a_gold():
    print("\n" + "="*72)
    print("CASE A — rumor amplification, GOLD STATE (max-discount encoding)")
    print("="*72)
    # Strategy: all 5 supports share extractor + lineage + self_gen, so
    # D_k = P_k = S_k = 1 max. Note: the schema UNIQUE constraint forbids
    # same (src,dst,rel,extractor,polarity,ns) — so each support gets a
    # unique extractor_version suffix but same extractor-slot.
    # Actually the Rust UNIQUE is on (extractor) not (extractor,version),
    # so we can't have 5 claims with the same extractor+polarity on the
    # same proposition. Work around: use different extractors but identical
    # source_lineage — P_k = 0 but D_k = 1.
    supports = [
        ("nova_pharma_v1", ["nova_pharma"]),
        ("reuters_medical", ["nova_pharma"]),   # stripped to gold-canonical
        ("stat_news", ["nova_pharma"]),
        ("bloomberg_health", ["nova_pharma"]),
        ("medscape", ["nova_pharma"]),
    ]
    for extr, lineage in supports:
        r = call("/v1/claim_with_lineage", "POST", body={
            "src": "XJ_417_GOLD", "rel_type": "reduces_mortality_in",
            "dst": "cardiac_population_A_GOLD", "namespace": "ablation_a",
            "polarity": 1, "extractor": extr,
            "source_lineage": lineage, "weight": 1.0,
        })
        if "__error__" in r:
            print(f"  ingest fail ({extr}): {r['__error__']}")
    # 2 independent attacks
    for extr, lineage in [("tokyo_u_rct", ["tokyo_u"]), ("mayo_clinic_rct", ["mayo_clinic"])]:
        r = call("/v1/claim_with_lineage", "POST", body={
            "src": "XJ_417_GOLD", "rel_type": "reduces_mortality_in",
            "dst": "cardiac_population_A_GOLD", "namespace": "ablation_a",
            "polarity": -1, "extractor": extr,
            "source_lineage": lineage, "weight": 1.0,
        })
        if "__error__" in r:
            print(f"  ingest fail ({extr}): {r['__error__']}")

    mob = call("/v1/mobility", "GET", params={
        "src": "XJ_417_GOLD", "rel_type": "reduces_mortality_in",
        "dst": "cardiac_population_A_GOLD", "namespace": "ablation_a",
    })
    con = call("/v1/contest", "GET", params={
        "src": "XJ_417_GOLD", "rel_type": "reduces_mortality_in",
        "dst": "cardiac_population_A_GOLD", "namespace": "ablation_a",
    })
    ms = mob.get("mobility_state", {})
    cs = con.get("contest_state", {})
    print(f"\n  σ (support_mass): {ms.get('support_mass', 0):.3f} — expected to flip to <2.0 for substrate to work")
    print(f"  α (attack_mass):  {ms.get('attack_mass', 0):.3f} — expected ~2.0")
    print(f"  support_distinct_source_count: {cs.get('support_distinct_source_count', 0)}")
    print(f"  attack_distinct_source_count:  {cs.get('attack_distinct_source_count', 0)}")
    print(f"  support_effective_independence: {cs.get('support_effective_independence', 0):.3f}")
    flags = cs.get("heuristic_flags", 0)
    flag_names = [n for b,n in [(1,"DUPLICATION_RISK"),(2,"SAME_SOURCE_CONFLICT"),
                                 (4,"REFERENT_HETEROGENEITY"),(8,"SAME_ARTIFACT_EXTRACTOR"),
                                 (16,"PRESENT_TENSE_CONFLICT")] if flags & b]
    print(f"  heuristic_flags: 0x{flags:x} ({' | '.join(flag_names) or 'none'})")
    sigma = ms.get("support_mass", 0)
    alpha = ms.get("attack_mass", 0)
    verdict = "SUBSTRATE CORRECT (σ < α)" if sigma < alpha else "SUBSTRATE WRONG (σ ≥ α despite gold state)"
    print(f"\n  VERDICT: {verdict}")
    return {"sigma": sigma, "alpha": alpha, "flags": flags, "correct": sigma < alpha}


# ─── Case B: gold-state temporal state change ────────────────────────
# All 6 claims have SAME proposition triple (alice_chen is_ceo_of acme)
# with explicit DISJOINT valid_from/valid_to. Agent previously failed to
# populate these.

def case_b_gold():
    print("\n" + "="*72)
    print("CASE B — temporal state change, GOLD STATE (explicit intervals)")
    print("="*72)
    # Alice CEO 2016-2019 (valid_to = June 2019 end)
    # Bob CEO 2023+ (valid_from = June 2023 start)
    alice_to = 1561939200.0   # 2019-07-01
    bob_from = 1688169600.0   # 2023-07-01
    # Alice's claims: polarity=1 (Alice IS CEO) with valid_from/to
    for extr, vf in [("reuters_2016", 1451606400.0), ("bloomberg_2017", 1485907200.0), ("wsj_2018", 1542240000.0)]:
        r = call("/v1/claim_with_lineage", "POST", body={
            "src": "alice_chen_GOLD", "rel_type": "is_ceo_of", "dst": "acme_GOLD",
            "namespace": "ablation_b",
            "polarity": 1, "extractor": extr,
            "source_lineage": [extr.split("_")[0]], "weight": 1.0,
            "valid_from": vf, "valid_to": alice_to,
        })
        if "__error__" in r:
            print(f"  ingest fail ({extr}): {r['__error__']}")
    # Bob's claims: polarity=-1 (Alice is NOT CEO) starting 2023
    for extr, vf in [("ft_2023", bob_from), ("reuters_2023", 1695945600.0), ("bloomberg_2024", 1715644800.0)]:
        r = call("/v1/claim_with_lineage", "POST", body={
            "src": "alice_chen_GOLD", "rel_type": "is_ceo_of", "dst": "acme_GOLD",
            "namespace": "ablation_b",
            "polarity": -1, "extractor": extr,
            "source_lineage": [extr.split("_")[0]], "weight": 1.0,
            "valid_from": vf,
        })
        if "__error__" in r:
            print(f"  ingest fail ({extr}): {r['__error__']}")

    con = call("/v1/contest", "GET", params={
        "src": "alice_chen_GOLD", "rel_type": "is_ceo_of", "dst": "acme_GOLD",
        "namespace": "ablation_b",
    })
    cs = con.get("contest_state", {})
    overlap = cs.get("temporal_overlap_conflict_count", 0)
    separable = cs.get("temporal_separable_opposition_count", 0)
    flags = cs.get("heuristic_flags", 0)
    flag_names = [n for b,n in [(1,"DUPLICATION_RISK"),(2,"SAME_SOURCE_CONFLICT"),
                                 (4,"REFERENT_HETEROGENEITY"),(8,"SAME_ARTIFACT_EXTRACTOR"),
                                 (16,"PRESENT_TENSE_CONFLICT")] if flags & b]
    print(f"\n  temporal_overlap_conflict_count: {overlap}")
    print(f"  temporal_separable_opposition_count: {separable} (expected 9)")
    print(f"  heuristic_flags: 0x{flags:x} ({' | '.join(flag_names) or 'none'})")
    # Correct: 9 separable pairs, 0 overlap, no PRESENT_TENSE_CONFLICT
    correct = (separable == 9 and overlap == 0 and (flags & 16) == 0)
    verdict = "SUBSTRATE CORRECT (9 separable, 0 overlap, no present-tense flag)" if correct \
              else f"SUBSTRATE WRONG (overlap={overlap}, separable={separable}, PRESENT_TENSE={bool(flags & 16)})"
    print(f"\n  VERDICT: {verdict}")
    return {"overlap": overlap, "separable": separable, "flags": flags, "correct": correct}


# ─── Case C: gold-state same-source retraction ───────────────────────
# Reuters original and Reuters correction have IDENTICAL lineage [reuters]
# with opposite polarity. Should fire SAME_SOURCE_CONFLICT.

def case_c_gold():
    print("\n" + "="*72)
    print("CASE C — same-source retraction, GOLD STATE (identical lineage)")
    print("="*72)
    # Reuters 2018 asserts
    r = call("/v1/claim_with_lineage", "POST", body={
        "src": "company_q_GOLD", "rel_type": "contains", "dst": "misstated_revenue_GOLD",
        "namespace": "ablation_c",
        "polarity": 1, "extractor": "reuters_2018",
        "source_lineage": ["reuters"], "weight": 1.0,
    })
    # 3 syndicators
    for extr in ["bbc_2018", "cnn_2018", "ap_2018"]:
        r = call("/v1/claim_with_lineage", "POST", body={
            "src": "company_q_GOLD", "rel_type": "contains", "dst": "misstated_revenue_GOLD",
            "namespace": "ablation_c",
            "polarity": 1, "extractor": extr,
            "source_lineage": ["reuters", extr.split("_")[0]], "weight": 1.0,
        })
    # Reuters retracts — IDENTICAL lineage [reuters]
    r = call("/v1/claim_with_lineage", "POST", body={
        "src": "company_q_GOLD", "rel_type": "contains", "dst": "misstated_revenue_GOLD",
        "namespace": "ablation_c",
        "polarity": -1, "extractor": "reuters_correction",
        "source_lineage": ["reuters"], "weight": 1.0,
    })

    con = call("/v1/contest", "GET", params={
        "src": "company_q_GOLD", "rel_type": "contains", "dst": "misstated_revenue_GOLD",
        "namespace": "ablation_c",
    })
    cs = con.get("contest_state", {})
    same_src = cs.get("same_source_opposite_polarity_count", 0)
    flags = cs.get("heuristic_flags", 0)
    flag_names = [n for b,n in [(1,"DUPLICATION_RISK"),(2,"SAME_SOURCE_CONFLICT"),
                                 (4,"REFERENT_HETEROGENEITY"),(8,"SAME_ARTIFACT_EXTRACTOR"),
                                 (16,"PRESENT_TENSE_CONFLICT")] if flags & b]
    print(f"\n  same_source_opposite_polarity_count: {same_src} (expected ≥1)")
    print(f"  heuristic_flags: 0x{flags:x} ({' | '.join(flag_names) or 'none'})")
    correct = same_src >= 1 and (flags & 2) != 0  # SAME_SOURCE_CONFLICT bit
    verdict = "SUBSTRATE CORRECT (SAME_SOURCE_CONFLICT fires under gold state)" if correct \
              else "SUBSTRATE WRONG (gate didn't fire even with identical lineage)"
    print(f"\n  VERDICT: {verdict}")
    return {"same_source_count": same_src, "flags": flags, "correct": correct}


# ─── Case A': extreme — push ⊕ to its theoretical max discount ────────
# Same extractor across all 5 supports, with self_generated=true.
# Current Rust API doesn't expose self_gen via claim_with_lineage, so
# this tests the best-case attainable through the exposed endpoint.
# The theoretical max we CAN reach via this endpoint: D=1 only.
# D=1 + all other = 0 → discount=1.5 → ω=0.667 → σ=5×0.667=3.33. Still > α=2.
# That's the operator's structural ceiling at N=5 via this endpoint.
#
# To see if replacing the operator (cluster-collapse) would work:
# effective_support = number_of_distinct_source_clusters = 1 (all share nova_pharma)
# vs attack clusters = 2 (tokyo, mayo). Cluster-collapse: 1 < 2 → attack wins.
# That's the winning alternative — NOT achievable via current ⊕.

def main():
    print("RFC 008 Phase 2 gold-state ablation")
    print("="*72)
    a = case_a_gold()
    b = case_b_gold()
    c = case_c_gold()

    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    print(f"Case A (rumor amp, gold):   {'CORRECT' if a['correct'] else 'WRONG'}  σ={a['sigma']:.3f} vs α={a['alpha']:.3f}")
    print(f"Case B (temporal, gold):    {'CORRECT' if b['correct'] else 'WRONG'}  separable={b['separable']} overlap={b['overlap']}")
    print(f"Case C (same-source, gold): {'CORRECT' if c['correct'] else 'WRONG'}  same_source_count={c['same_source_count']} flags=0x{c['flags']:x}")

    print("\nInterpretation:")
    if not a["correct"] and b["correct"] and c["correct"]:
        print("  → Case A is an OPERATOR failure (⊕ structurally can't flip at N=5).")
        print("  → Cases B, C are EXTRACTION failures (work when agent populates state right).")
        print("  → RFC 009 must (1) replace ⊕, (2) build extractor/canonicalizer.")
    elif not a["correct"] and not b["correct"]:
        print("  → More severe: operator failures beyond just ⊕.")
        print("  → RFC 009 needs a deeper rewrite.")
    elif all(x["correct"] for x in [a, b, c]):
        print("  → All pass under gold state. Phase 2 failures were purely extraction.")
        print("  → Focus RFC 009 entirely on extractor/canonicalizer; keep operators.")

    with open("docs/phase2/gold_state_results.json", "w") as f:
        json.dump({"A": a, "B": b, "C": c}, f, indent=2, default=str)
    print("\nSaved → docs/phase2/gold_state_results.json")


if __name__ == "__main__":
    main()
