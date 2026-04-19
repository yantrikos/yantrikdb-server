#!/usr/bin/env python3
"""Analyze Phase 2 results.json: accuracy per condition, tool-use patterns,
and per-case qualitative summaries. Run after harness.py finishes."""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)


def main():
    with open("docs/phase2/results.json", encoding="utf-8") as f:
        data = json.load(f)

    cases = {c["id"]: c for c in data["cases"]}
    results = data["results"]
    conditions = ["C1_bare", "C2_tools_framed", "C3_tools_authority"]

    print("# Phase 2 — analysis\n")
    print("## Accuracy matrix\n")
    print("| case | C1 bare | C2 tools+framed | C3 tools+authority |")
    print("|------|---------|-----------------|--------------------|")
    for cid, case in cases.items():
        gt = case["ground_truth"]
        row = [cid.split("_", 1)[0]]
        for cond in conditions:
            cell = [r for r in results if r["case_id"] == cid and r["condition"] == cond]
            n = len(cell)
            correct = sum(1 for r in cell if r["correct"])
            row.append(f"{correct}/{n}")
        print(f"| **{cid}** (gt={gt}) | {row[1]} | {row[2]} | {row[3]} |")

    print("\n## Tool-use patterns\n")
    print("| case | condition | avg tool calls | ingest calls | mobility calls | contest calls |")
    print("|------|-----------|----------------|--------------|----------------|---------------|")
    for cid in cases:
        for cond in conditions:
            if cond == "C1_bare":
                continue
            cell = [r for r in results if r["case_id"] == cid and r["condition"] == cond]
            if not cell:
                continue
            total = sum(r["tool_calls"] for r in cell) / len(cell)
            ingests = sum(sum(1 for t in r["trace"] if t["tool"] == "ingest_claim") for r in cell) / len(cell)
            mob = sum(sum(1 for t in r["trace"] if t["tool"] == "get_mobility_state") for r in cell) / len(cell)
            con = sum(sum(1 for t in r["trace"] if t["tool"] == "get_contest_state") for r in cell) / len(cell)
            print(f"| {cid.split('_', 1)[0]} | {cond} | {total:.1f} | {ingests:.1f} | {mob:.1f} | {con:.1f} |")

    print("\n## Per-case narrative summaries\n")
    for cid, case in cases.items():
        print(f"### Case {cid}: {case['title']}\n")
        print(f"**Ground truth:** {case['ground_truth']}\n")
        print(f"**Why substrate should help:** {case['why_substrate_matters']}\n")

        # Pull substrate output from a C3 run that actually used tools
        substrate_snapshot = None
        for r in results:
            if r["case_id"] == cid and r["condition"] == "C3_tools_authority":
                contest_calls = [t for t in r["trace"] if t["tool"] == "get_contest_state" and "__error__" not in t.get("result", {})]
                if contest_calls:
                    substrate_snapshot = contest_calls[-1]["result"]
                    break
        if substrate_snapshot and "contest_state" in substrate_snapshot:
            cs = substrate_snapshot["contest_state"]
            flags = cs.get("heuristic_flags", 0)
            flag_names = []
            for bit, name in [(1, "DUPLICATION_RISK"), (2, "SAME_SOURCE_CONFLICT"),
                              (4, "REFERENT_HETEROGENEITY"), (8, "SAME_ARTIFACT_CONFLICT"),
                              (16, "PRESENT_TENSE_CONFLICT")]:
                if flags & bit:
                    flag_names.append(name)
            print(f"**Substrate output under C3:**")
            print(f"- σ (support_mass): {cs.get('support_mass', 0):.3f}")
            print(f"- α (attack_mass): {cs.get('attack_mass', 0):.3f}")
            print(f"- support_eff_independence: {cs.get('support_effective_independence', 0):.3f}")
            print(f"- attack_eff_independence: {cs.get('attack_effective_independence', 0):.3f}")
            print(f"- same_source_opposite_polarity_count: {cs.get('same_source_opposite_polarity_count', 0)}")
            print(f"- temporal_overlap_conflict_count: {cs.get('temporal_overlap_conflict_count', 0)}")
            print(f"- temporal_separable_opposition_count: {cs.get('temporal_separable_opposition_count', 0)}")
            print(f"- heuristic_flags: 0x{flags:x} ({' | '.join(flag_names) or 'none'})\n")

        # Qualitative: what did Qwen conclude in each condition?
        for cond in conditions:
            cell = [r for r in results if r["case_id"] == cid and r["condition"] == cond]
            answers = [r["raw_answer"] for r in cell]
            print(f"**{cond}:** {answers}")
        print()


if __name__ == "__main__":
    main()
