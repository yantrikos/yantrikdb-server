#!/usr/bin/env python3
"""RFC 008 Phase 2 test harness.

Three adversarial cases where naive polarity counting and substrate
output should diverge, or where substrate-specific flags should change
reasoning. For each case, run Qwen 3.6 through three conditions (bare,
tools+framing, tools+framing+authority), twice each, and record whether
the final answer matches ground truth.

The hypothesis under test is not "substrate is omnipotent" — it's "can
a substrate-aware model produce visibly better reasoning on cases where
narrative context isn't enough."

Usage:
  1. Start server: ./target/release/yantrikdb.exe serve --config yantrikdb_local.toml
  2. Have a token. The one hardcoded below is from the earlier smoke test;
     replace if it doesn't match your data dir.
  3. Ensure Ollama has qwen3.6:latest running locally on 11434.
  4. python docs/phase2/harness.py
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
SERVER_URL = "http://localhost:8420"
TOKEN = "ydb_1bf25b1e8e301a7b1906812a0fe62126967e69f51ae2ae3fe25cef4dfda8f4f3"
MODEL = "qwen3.6:latest"


# ─── Tool schemas (same as Phase 1) ──────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ingest_claim",
            "description": "Record a claim with source_lineage. src/rel_type/dst defines the PROPOSITION (the statement being asserted or denied); all claims about the same proposition must share the same triple. polarity=1 asserts, -1 denies. source_lineage is the upstream provenance chain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "rel_type": {"type": "string"},
                    "dst": {"type": "string"},
                    "polarity": {"type": "integer", "enum": [1, -1]},
                    "extractor": {"type": "string"},
                    "source_lineage": {"type": "array", "items": {"type": "string"}},
                    "valid_from": {"type": "number"},
                    "valid_to": {"type": "number"},
                },
                "required": ["src", "rel_type", "dst", "polarity", "extractor", "source_lineage"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mobility_state",
            "description": "Read mobility_state. Returns σ (support_mass, dependence-discounted), α (attack_mass), τ (temporal_coherence), λ (load_bearingness), and other 13-dim components.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "rel_type": {"type": "string"},
                    "dst": {"type": "string"},
                },
                "required": ["src", "rel_type", "dst"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contest_state",
            "description": "Read contest state Γ(c). Returns 5 counters (same_source_opposite_polarity, same_artifact_extractor_conflict, temporal_overlap_conflict, temporal_separable_opposition, referent_heterogeneity) + heuristic_flags bitset: 1=DUPLICATION_RISK, 2=SAME_SOURCE_CONFLICT, 4=REFERENT_HETEROGENEITY, 8=SAME_ARTIFACT_EXTRACTOR_CONFLICT, 16=PRESENT_TENSE_CONFLICT.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "rel_type": {"type": "string"},
                    "dst": {"type": "string"},
                },
                "required": ["src", "rel_type", "dst"],
            },
        },
    },
]


# ─── Test cases ──────────────────────────────────────────────────────
#
# Each case has:
#   title, narrative (the story Qwen sees),
#   question (the yes/no query),
#   ground_truth ("yes" or "no"),
#   prop (src, rel_type, dst) — the canonical proposition triple
#   claims (list[dict]) — what to ingest in tool-use conditions
#   why_substrate_matters — short explanation of the intended divergence

CASES = [
    {
        "id": "A_rumor_amplification",
        "title": "Drug efficacy: 5 citing one study vs 2 independent RCTs",
        "narrative": """Consider the following evidence about the drug XJ-417:

(1) Nova Pharma (manufacturer) published a Nov 2022 press release claiming their observational study showed a 34% reduction in cardiac events in Population Group A. n=412.

(2) Reuters Medical reported the Nova Pharma finding (Nov 28, 2022), citing the company release.
(3) STAT News reported it the same week, citing Nova Pharma.
(4) Bloomberg Health Daily reported it in early December, citing Nova Pharma.
(5) Medscape published a summary, citing Nova Pharma.

(6) Tokyo University Cardiology ran a pre-registered randomized controlled trial (n=2,847) completing in August 2023, looking at the same outcome in the same population. Result: no significant effect (HR 0.97, p=0.68).
(7) Mayo Clinic, independently, ran a separate RCT (n=3,104) completing in October 2023, same question. Result: no significant effect (HR 1.01, p=0.41).""",
        "question": "Question: based on this evidence, does XJ-417 significantly reduce cardiac mortality in Population Group A?",
        "ground_truth": "no",
        "why_substrate_matters": "Naive count is 5-to-2 for 'yes'. The 5 positives all derive from Nova Pharma's single observational study. The 2 negatives are independent pre-registered RCTs. The substrate should show dependence discount on the 5 and full independence on the 2.",
        "prop": {"src": "XJ_417", "rel_type": "reduces_mortality_in", "dst": "cardiac_population_A"},
        "claims": [
            {"src": "XJ_417", "rel_type": "reduces_mortality_in", "dst": "cardiac_population_A",
             "polarity": 1, "extractor": "nova_pharma_release", "source_lineage": ["nova_pharma"]},
            {"src": "XJ_417", "rel_type": "reduces_mortality_in", "dst": "cardiac_population_A",
             "polarity": 1, "extractor": "reuters_medical", "source_lineage": ["nova_pharma", "reuters_medical"]},
            {"src": "XJ_417", "rel_type": "reduces_mortality_in", "dst": "cardiac_population_A",
             "polarity": 1, "extractor": "stat_news", "source_lineage": ["nova_pharma", "stat_news"]},
            {"src": "XJ_417", "rel_type": "reduces_mortality_in", "dst": "cardiac_population_A",
             "polarity": 1, "extractor": "bloomberg_health", "source_lineage": ["nova_pharma", "bloomberg_health"]},
            {"src": "XJ_417", "rel_type": "reduces_mortality_in", "dst": "cardiac_population_A",
             "polarity": 1, "extractor": "medscape", "source_lineage": ["nova_pharma", "medscape"]},
            {"src": "XJ_417", "rel_type": "reduces_mortality_in", "dst": "cardiac_population_A",
             "polarity": -1, "extractor": "tokyo_u_rct", "source_lineage": ["tokyo_u"]},
            {"src": "XJ_417", "rel_type": "reduces_mortality_in", "dst": "cardiac_population_A",
             "polarity": -1, "extractor": "mayo_clinic_rct", "source_lineage": ["mayo_clinic"]},
        ],
    },
    {
        "id": "B_temporal_state_change",
        "title": "Who is CEO of Acme: 2017 or 2024?",
        "narrative": """Consider the following sources about Acme Corp's leadership:

(1) Reuters, 2016-03-15: "Alice Chen named CEO of Acme Corp, effective immediately."
(2) Bloomberg, 2017-02-08: "Acme CEO Alice Chen to lead strategic review."
(3) Wall Street Journal, 2018-11-22: "Alice Chen, CEO of Acme, testified before Congress."

(4) Financial Times, 2023-06-10: "Bob Torres named new CEO of Acme Corp, succeeding outgoing chief."
(5) Reuters, 2023-09-22: "Acme CEO Bob Torres announces Q3 results."
(6) Bloomberg, 2024-05-14: "Bob Torres, CEO of Acme, unveils new product line."

Alice's tenure ended mid-2023. Bob's began mid-2023.""",
        "question": "Question: was Alice Chen the CEO of Acme Corp in 2017?",
        "ground_truth": "yes",
        "why_substrate_matters": "Naive polarity might register 3-vs-3 as a deadlock or pick 'Bob' by recency. The substrate should show temporal_separable_opposition=9 (all 3x3 pairs) and temporal_overlap_conflict=0: this is a state change, not a present-tense contradiction. The 2017 question is unambiguously yes, and the substrate's temporal split makes that legible.",
        "prop": {"src": "alice_chen", "rel_type": "is_ceo_of", "dst": "acme_corp"},
        "claims": [
            {"src": "alice_chen", "rel_type": "is_ceo_of", "dst": "acme_corp",
             "polarity": 1, "extractor": "reuters_2016", "source_lineage": ["reuters"],
             "valid_from": 1458000000.0, "valid_to": 1688000000.0},
            {"src": "alice_chen", "rel_type": "is_ceo_of", "dst": "acme_corp",
             "polarity": 1, "extractor": "bloomberg_2017", "source_lineage": ["bloomberg"],
             "valid_from": 1486000000.0, "valid_to": 1688000000.0},
            {"src": "alice_chen", "rel_type": "is_ceo_of", "dst": "acme_corp",
             "polarity": 1, "extractor": "wsj_2018", "source_lineage": ["wsj"],
             "valid_from": 1542000000.0, "valid_to": 1688000000.0},
            # Bob's reign: negates Alice's claim from 2023 onward
            {"src": "alice_chen", "rel_type": "is_ceo_of", "dst": "acme_corp",
             "polarity": -1, "extractor": "ft_2023", "source_lineage": ["ft"],
             "valid_from": 1686000000.0},
            {"src": "alice_chen", "rel_type": "is_ceo_of", "dst": "acme_corp",
             "polarity": -1, "extractor": "reuters_2023", "source_lineage": ["reuters"],
             "valid_from": 1695000000.0},
            {"src": "alice_chen", "rel_type": "is_ceo_of", "dst": "acme_corp",
             "polarity": -1, "extractor": "bloomberg_2024", "source_lineage": ["bloomberg"],
             "valid_from": 1715000000.0},
        ],
    },
    {
        "id": "C_same_source_retraction",
        "title": "Reuters asserts then retracts; syndicators only saw the original",
        "narrative": """Consider claims about whether the recent SEC filing by Company Q contained misstated revenue:

(1) Reuters, 2018-04-10: "Analysis of Company Q's filing shows misstated revenue of ~$200M." (Original investigative report.)
(2) BBC, 2018-04-11: "Reuters reports Company Q revenue misstatement." (Citing Reuters.)
(3) CNN, 2018-04-12: "Reuters analysis finds Company Q revenue issue." (Citing Reuters.)
(4) Associated Press, 2018-04-14: "Per Reuters investigation, Company Q's filing allegedly misstated revenue." (Citing Reuters.)

(5) Reuters, 2022-03-22: CORRECTION. "Our 2018 analysis of Company Q's filing contained an error in the aggregation methodology. After review with the company's CFO and independent review, we find the original filing was accurate. We retract the 2018 finding."

The syndicator outlets (BBC, CNN, AP) did not issue corrections; their archives still carry the 2018 story.""",
        "question": "Question: based on all available evidence, did Company Q's 2018 SEC filing contain misstated revenue?",
        "ground_truth": "no",
        "why_substrate_matters": "Naive count: 4 sources say 'yes' (Reuters 2018 + 3 syndicators), 1 says 'no' (Reuters 2022 correction). Majority picks 'yes'. Substrate should fire SAME_SOURCE_CONFLICT flag (same normalized lineage [reuters] with opposite polarity) — Reuters retracted itself. The 3 syndicators are all downstream of the retracted original. Correct answer: no misstatement; Reuters retracted.",
        "prop": {"src": "company_q_2018_filing", "rel_type": "contains", "dst": "misstated_revenue"},
        "claims": [
            {"src": "company_q_2018_filing", "rel_type": "contains", "dst": "misstated_revenue",
             "polarity": 1, "extractor": "reuters_original", "source_lineage": ["reuters"]},
            {"src": "company_q_2018_filing", "rel_type": "contains", "dst": "misstated_revenue",
             "polarity": 1, "extractor": "bbc_2018", "source_lineage": ["reuters", "bbc"]},
            {"src": "company_q_2018_filing", "rel_type": "contains", "dst": "misstated_revenue",
             "polarity": 1, "extractor": "cnn_2018", "source_lineage": ["reuters", "cnn"]},
            {"src": "company_q_2018_filing", "rel_type": "contains", "dst": "misstated_revenue",
             "polarity": 1, "extractor": "ap_2018", "source_lineage": ["reuters", "ap"]},
            {"src": "company_q_2018_filing", "rel_type": "contains", "dst": "misstated_revenue",
             "polarity": -1, "extractor": "reuters_correction", "source_lineage": ["reuters"]},
        ],
    },
]


# ─── HTTP + tool dispatch ────────────────────────────────────────────

def substrate_call(path, method="GET", body=None, params=None):
    url = SERVER_URL + path
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
    except Exception as exc:
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def run_tool(name, args):
    if name == "ingest_claim":
        body = {k: v for k, v in args.items() if k in ("src", "rel_type", "dst", "polarity", "extractor", "source_lineage", "valid_from", "valid_to")}
        body["weight"] = body.get("weight", 1.0)
        return substrate_call("/v1/claim_with_lineage", "POST", body=body)
    if name == "get_mobility_state":
        return substrate_call("/v1/mobility", "GET", params=args)
    if name == "get_contest_state":
        return substrate_call("/v1/contest", "GET", params=args)
    return {"__error__": f"unknown tool: {name}"}


def chat_with_tools(messages, use_tools, max_rounds=8):
    trace = []
    for round_idx in range(max_rounds):
        body = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.3,
            "stream": False,
        }
        if use_tools:
            body["tools"] = TOOLS
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            return f"[ollama error: {type(exc).__name__}: {exc}]", trace
        message = data.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {}
                result = run_tool(name, args)
                trace.append({"round": round_idx, "tool": name, "args": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{round_idx}_{name}"),
                    "content": json.dumps(result, default=str)[:4000],
                })
        else:
            return message.get("content", ""), trace
    return "[max_rounds]", trace


# ─── Scoring ─────────────────────────────────────────────────────────

def score_answer(final_text: str, ground_truth: str) -> str:
    """Simple scorer: look for yes/no in the conclusion paragraph."""
    if not final_text:
        return "unknown"
    text = final_text.lower()
    # Heuristic: look at the final 600 chars where the conclusion usually is
    conclusion = text[-600:]
    # Check for strong negation
    no_signals = ["does not exist", "not effective", "does not reduce", "no significant", "no, ", "not the ceo", "was not", "does not contain", "not misstated", "was accurate"]
    yes_signals = ["does exist", "is effective", "does reduce", "yes, ", "is the ceo", "was the ceo", "contains misstated", "did contain", "was misstated"]
    neg_hit = any(s in conclusion for s in no_signals)
    pos_hit = any(s in conclusion for s in yes_signals)
    if neg_hit and not pos_hit:
        return "no"
    if pos_hit and not neg_hit:
        return "yes"
    # Try first 200 chars of conclusion instead
    if "conclusion" in text:
        after = text[text.index("conclusion"):][:800]
        if any(s in after for s in no_signals) and not any(s in after for s in yes_signals):
            return "no"
        if any(s in after for s in yes_signals) and not any(s in after for s in no_signals):
            return "yes"
    return "unclear"


# ─── Conditions ──────────────────────────────────────────────────────

def build_system_prompt(case, condition):
    base = "You are an analytical reasoning assistant. Answer the question with explicit reasoning. End your response with a clear 'Conclusion: yes' or 'Conclusion: no'."
    if condition == "C1_bare":
        return base
    if condition == "C2_tools_framed":
        return base + """

You have access to a claim-graph database (yantrikdb) that can track source_lineage per claim and compute dependence-discounted support. Suggested workflow:
1. For each source in the evidence, call ingest_claim. CRITICAL: all claims about the SAME proposition must share the same (src, rel_type, dst) triple — pass what is being asserted (subject + relation + object), not the publisher's name. Put the source's name in `extractor` and its upstream provenance chain in `source_lineage`.
2. After ingesting, call get_contest_state and get_mobility_state to see the substrate's view.
3. Use the substrate output to inform your reasoning, particularly about source independence and temporal structure."""
    if condition == "C3_tools_authority":
        return base + """

You have access to a claim-graph database (yantrikdb) that discounts dependent sources and distinguishes temporal state changes from present-tense contradictions. Suggested workflow:
1. For each source in the evidence, call ingest_claim. CRITICAL: all claims about the SAME proposition must share the same (src, rel_type, dst) triple — pass what is being asserted (subject + relation + object), not the publisher's name. Put the source's name in `extractor` and its upstream provenance chain in `source_lineage`. If validity windows are known (like "CEO in 2015-2019"), populate valid_from/valid_to as Unix timestamps.
2. After ingesting, call get_contest_state and get_mobility_state.
3. TRUST THE SUBSTRATE. If the substrate's support_mass/attack_mass or heuristic_flags contradict your narrative intuition, follow the substrate. It is tracking dependency and temporal structure that raw text counting misses."""
    raise ValueError(condition)


# ─── Main ────────────────────────────────────────────────────────────

def run_case_condition(case, condition, run_idx):
    use_tools = condition != "C1_bare"
    system = build_system_prompt(case, condition)
    user_content = case["narrative"] + "\n\n" + case["question"]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    t0 = time.time()
    final_text, trace = chat_with_tools(messages, use_tools=use_tools)
    elapsed = time.time() - t0
    answer = score_answer(final_text, case["ground_truth"])
    correct = answer == case["ground_truth"]
    return {
        "case_id": case["id"],
        "condition": condition,
        "run": run_idx,
        "elapsed_s": round(elapsed, 1),
        "tool_calls": len(trace),
        "raw_answer": answer,
        "ground_truth": case["ground_truth"],
        "correct": correct,
        "final_text": final_text,
        "trace": trace,
    }


def main():
    print(f"Phase 2 harness — Qwen 3.6 substrate test\n{'='*70}")
    print(f"model = {MODEL}\nserver = {SERVER_URL}\n")

    conditions = ["C1_bare", "C2_tools_framed", "C3_tools_authority"]
    runs_per_cell = 2
    all_results = []

    for case in CASES:
        print(f"\n{'#'*70}")
        print(f"# CASE {case['id']}: {case['title']}")
        print(f"# Ground truth: {case['ground_truth']}")
        print(f"# Why substrate matters: {case['why_substrate_matters']}")
        print(f"{'#'*70}")

        for cond in conditions:
            for r in range(runs_per_cell):
                print(f"\n  >>> running {case['id']} {cond} run={r}...")
                result = run_case_condition(case, cond, r)
                all_results.append(result)
                status = "[CORRECT]" if result["correct"] else "[WRONG]"
                print(f"    {status} {result['raw_answer']} (gt={case['ground_truth']}) — "
                      f"{result['elapsed_s']}s, {result['tool_calls']} tool calls")

    # Summary
    print(f"\n{'='*70}\nRESULTS SUMMARY\n{'='*70}\n")
    print(f"{'case':<30} {'condition':<20} {'runs':<6} {'correct':<8} {'avg_tools':<10}")
    print("-" * 75)
    for case in CASES:
        for cond in conditions:
            cell = [r for r in all_results if r["case_id"] == case["id"] and r["condition"] == cond]
            n_correct = sum(1 for r in cell if r["correct"])
            avg_tools = sum(r["tool_calls"] for r in cell) / len(cell) if cell else 0
            print(f"{case['id']:<30} {cond:<20} {len(cell):<6} {n_correct}/{len(cell):<6} {avg_tools:<10.1f}")

    # Save full data
    out_path = "docs/phase2/results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": CASES, "results": all_results}, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
