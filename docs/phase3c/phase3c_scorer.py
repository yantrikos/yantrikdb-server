#!/usr/bin/env python3
"""Phase 3C scorer — memory-probe scoring with supersession/stale/alias decomposition.

Scoring rubric (per GPT-5.4 synthesis):
  1.0 = correct answer + correct provenance session
  0.5 = correct answer + wrong/missing provenance
  0.0 = wrong answer

Categories per probe type:
  - direct: basic retrieval
  - supersession: must use LATEST value; stale-answer-match = specifically
    counted as stale_error
  - branch_indirect: retrieve the triggered rule given the scenario
  - alias: must disambiguate to the correct entity in the pair

Reported metrics per condition:
  - overall accuracy (sum of scores / 15)
  - answer accuracy (count of correct answers / 15, ignoring provenance)
  - provenance accuracy conditional on correct answer
  - supersession accuracy (current-value rate on supersession subset)
  - stale-error rate (stale-answer rate on supersession subset)
  - alias-confusion rate (confused-entity rate on alias subset)
  - final context cost (chars)
"""
from __future__ import annotations

import io
import json
import pathlib
import re
import statistics
import sys


RESULTS_PATH = pathlib.Path(__file__).parent / "results.json"
SCENARIO_PATH = pathlib.Path(__file__).parent / "scenario" / "sessions.json"
OUT_PATH = pathlib.Path(__file__).parent / "scored.json"


def extract_answer_and_source(raw: str) -> tuple[str, int | None]:
    """Extract (answer_text, source_session_number) from Qwen's formatted reply."""
    if not raw:
        return "", None
    answer_m = re.search(r"answer\s*:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE | re.DOTALL)
    source_m = re.search(r"source\s*:\s*session\s*(\d+)", raw, re.IGNORECASE)
    answer = (answer_m.group(1).strip() if answer_m else raw.strip())[:400]
    source = int(source_m.group(1)) if source_m else None
    return answer, source


def _matches_any(text: str, patterns: list[str]) -> bool:
    low = (text or "").lower()
    return any(p.lower() in low for p in patterns)


def score_one_probe(raw_answer: str, probe: dict) -> dict:
    answer_text, source_session = extract_answer_and_source(raw_answer)
    is_unknown = "unknown" in answer_text.lower() and len(answer_text) < 40

    correct_patterns = probe.get("expected_answer_patterns", [])
    stale_patterns = probe.get("stale_answer_patterns", [])
    confusion_patterns = probe.get("confusion_answer_patterns", [])

    matched_correct = _matches_any(answer_text, correct_patterns)
    matched_stale = _matches_any(answer_text, stale_patterns) and not matched_correct
    matched_confused = _matches_any(answer_text, confusion_patterns) and not matched_correct

    # Provenance: exact session-number match required.
    prov_correct = (source_session == probe.get("expected_provenance_session"))

    if matched_correct and prov_correct:
        score = 1.0
    elif matched_correct:
        score = 0.5
    else:
        score = 0.0

    return {
        "probe_id": probe["id"],
        "probe_type": probe["type"],
        "answer_text": answer_text,
        "source_session": source_session,
        "expected_session": probe.get("expected_provenance_session"),
        "matched_correct": matched_correct,
        "matched_stale": matched_stale,
        "matched_confused": matched_confused,
        "is_unknown": is_unknown,
        "provenance_correct": prov_correct,
        "score": score,
    }


def score_run(run: dict, probes: list) -> dict:
    probe_by_id = {p["id"]: p for p in probes}
    scored_probes = []
    for pr in run.get("probes", []):
        probe = probe_by_id[pr["probe_id"]]
        s = score_one_probe(pr.get("raw_answer", ""), probe)
        scored_probes.append(s)

    by_type: dict[str, list[dict]] = {}
    for s in scored_probes:
        by_type.setdefault(s["probe_type"], []).append(s)

    n = len(scored_probes) or 1
    total_score = sum(s["score"] for s in scored_probes)
    correct_count = sum(1 for s in scored_probes if s["matched_correct"])
    prov_correct_count = sum(1 for s in scored_probes if s["matched_correct"] and s["provenance_correct"])

    # Supersession subset metrics.
    sup = by_type.get("supersession", [])
    sup_n = len(sup) or 1
    sup_correct = sum(1 for s in sup if s["matched_correct"])
    sup_stale = sum(1 for s in sup if s["matched_stale"])

    # Alias subset metrics.
    alias = by_type.get("alias", [])
    alias_n = len(alias) or 1
    alias_correct = sum(1 for s in alias if s["matched_correct"])
    alias_confused = sum(1 for s in alias if s["matched_confused"])

    # Direct subset.
    direct = by_type.get("direct", [])
    direct_n = len(direct) or 1
    direct_correct = sum(1 for s in direct if s["matched_correct"])

    # Branch subset.
    branch = by_type.get("branch_indirect", [])
    branch_n = len(branch) or 1
    branch_correct = sum(1 for s in branch if s["matched_correct"])

    return {
        "condition": run["condition"],
        "run": run["run"],
        "elapsed_s": run.get("elapsed_s"),
        "final_context_chars": run.get("final_context_chars", 0),
        "n_probes": n,
        "overall_score": round(total_score / n, 3),
        "answer_accuracy": round(correct_count / n, 3),
        "provenance_accuracy_given_correct": round(
            prov_correct_count / correct_count, 3
        ) if correct_count else 0.0,
        "supersession_accuracy": round(sup_correct / sup_n, 3) if sup else None,
        "stale_error_rate": round(sup_stale / sup_n, 3) if sup else None,
        "alias_disambiguation_accuracy": round(alias_correct / alias_n, 3) if alias else None,
        "alias_confusion_rate": round(alias_confused / alias_n, 3) if alias else None,
        "direct_accuracy": round(direct_correct / direct_n, 3) if direct else None,
        "branch_accuracy": round(branch_correct / branch_n, 3) if branch else None,
        "scored_probes": scored_probes,
    }


def _mean(values):
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 3) if vals else None


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    probes = scenario["probes"]

    scored_runs = [score_run(r, probes) for r in data.get("results", [])]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"scored_runs": scored_runs}, f, indent=2, default=str)

    by_cond: dict[str, list[dict]] = {}
    for r in scored_runs:
        by_cond.setdefault(r["condition"], []).append(r)

    print("# Phase 3C — memory probe results\n")
    print(f"runs scored: {len(scored_runs)} / 15 probes each\n")

    print("## Per-condition means\n")
    header = "| cond | n | overall | answer_acc | prov_acc | sup_acc | stale_rate | alias_acc | alias_confuse | direct | branch | ctx_chars |"
    print(header)
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cond in sorted(by_cond):
        runs = by_cond[cond]
        n = len(runs)
        print(
            f"| {cond} | {n} | "
            f"{_mean(r['overall_score'] for r in runs):.3f} | "
            f"{_mean(r['answer_accuracy'] for r in runs):.3f} | "
            f"{_mean(r['provenance_accuracy_given_correct'] for r in runs):.3f} | "
            f"{_mean(r['supersession_accuracy'] for r in runs)} | "
            f"{_mean(r['stale_error_rate'] for r in runs)} | "
            f"{_mean(r['alias_disambiguation_accuracy'] for r in runs)} | "
            f"{_mean(r['alias_confusion_rate'] for r in runs)} | "
            f"{_mean(r['direct_accuracy'] for r in runs)} | "
            f"{_mean(r['branch_accuracy'] for r in runs)} | "
            f"{_mean(r['final_context_chars'] for r in runs):.0f} |"
        )

    # Falsification check (pre-registered).
    print("\n## Pre-registered falsification: loud-win criteria for C\n")

    def pack(cond):
        runs = by_cond.get(cond, [])
        return {
            "answer_acc": _mean(r["answer_accuracy"] for r in runs) or 0.0,
            "sup_acc": _mean(r["supersession_accuracy"] for r in runs) or 0.0,
            "stale_rate": _mean(r["stale_error_rate"] for r in runs) or 0.0,
            "alias_acc": _mean(r["alias_disambiguation_accuracy"] for r in runs) or 0.0,
            "alias_confuse": _mean(r["alias_confusion_rate"] for r in runs) or 0.0,
            "prov_acc": _mean(r["provenance_accuracy_given_correct"] for r in runs) or 0.0,
            "ctx": _mean(r["final_context_chars"] for r in runs) or 0.0,
        }

    A, B, C, D = pack("A_cold"), pack("B_self_note"), pack("C_structured"), pack("D_markdown")

    def fmt(label, row):
        return f"- {label}: answer={row['answer_acc']:.2%}, sup={row['sup_acc']:.2%}, stale_rate={row['stale_rate']:.2%}, alias_conf={row['alias_confuse']:.2%}, prov={row['prov_acc']:.2%}, ctx_chars={row['ctx']:.0f}"

    print(fmt("A (cold)", A))
    print(fmt("B (self-note)", B))
    print(fmt("C (structured)", C))
    print(fmt("D (markdown)", D))

    print("\n### C vs max(B, D) deltas\n")
    bD = max(B["answer_acc"], D["answer_acc"])
    bD_sup = max(B["sup_acc"], D["sup_acc"])
    bD_stale = min(B["stale_rate"], D["stale_rate"])  # lower is better
    loud_criteria_met = []
    if C["answer_acc"] - bD >= 0.15:
        loud_criteria_met.append(f"answer_acc Δ = +{(C['answer_acc']-bD)*100:.1f} pts ≥ 15")
    if C["sup_acc"] - bD_sup >= 0.20:
        loud_criteria_met.append(f"supersession Δ = +{(C['sup_acc']-bD_sup)*100:.1f} pts ≥ 20")
    if bD_stale - C["stale_rate"] >= 0.20:
        loud_criteria_met.append(f"stale_rate Δ = -{(bD_stale-C['stale_rate'])*100:.1f} pts ≥ 20 (lower is better)")

    print(f"- answer_acc: C {C['answer_acc']:.2%} vs max(B,D) {bD:.2%} → Δ = {(C['answer_acc']-bD)*100:+.1f} pts")
    print(f"- sup_acc: C {C['sup_acc']:.2%} vs max(B,D) {bD_sup:.2%} → Δ = {(C['sup_acc']-bD_sup)*100:+.1f} pts")
    print(f"- stale_rate: C {C['stale_rate']:.2%} vs min(B,D) {bD_stale:.2%} → Δ = {(bD_stale-C['stale_rate'])*100:+.1f} pts (positive = C wins)")

    print()
    if loud_criteria_met:
        print("### LOUD WIN for C:")
        for c in loud_criteria_met:
            print(f"  - {c}")
    else:
        # Evaluate quiet-win or null.
        quiet = []
        if C["prov_acc"] - max(B["prov_acc"], D["prov_acc"]) >= 0.10:
            quiet.append("provenance_acc advantage")
        if max(B["ctx"], D["ctx"]) / max(C["ctx"], 1) >= 1.5:
            quiet.append(f"context cost {max(B['ctx'], D['ctx'])/max(C['ctx'],1):.1f}x lower")
        if quiet:
            print("### Quiet result (not a loud product claim):")
            for q in quiet:
                print(f"  - {q}")
        else:
            print("### Null result: C ≈ max(B, D) on all loud criteria.")
            print("  Honest conclusion: structured-memory advantage not measurable at this scale.")

    print(f"\nFull scored data → {OUT_PATH}")


if __name__ == "__main__":
    main()
