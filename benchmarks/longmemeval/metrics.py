#!/usr/bin/env python3
"""Aggregate LongMemEval results into per-question-type recall@K tables.

Reads results_<variant>.jsonl (one JSON row per question) produced by
run.py and emits a Markdown table to stdout (and optionally to a file
via --out).
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render(rows: list[dict]) -> str:
    out = []
    n_total = len(rows)
    n_err = sum(1 for r in rows if "error" in r)
    n_ok = n_total - n_err
    out.append(f"# LongMemEval Results")
    out.append("")
    out.append(f"- Questions scored: **{n_ok}** of {n_total} ({n_err} errors)")

    # Latency summary on successes.
    write_times = [r["write_secs"] for r in rows if "write_secs" in r and "error" not in r]
    recall_times = [r["recall_secs"] for r in rows if "recall_secs" in r and "error" not in r]
    if write_times:
        out.append(
            f"- Write phase per question: mean={statistics.mean(write_times):.1f}s, "
            f"median={statistics.median(write_times):.1f}s, "
            f"max={max(write_times):.1f}s"
        )
    if recall_times:
        out.append(
            f"- Recall latency: mean={statistics.mean(recall_times) * 1000:.1f}ms, "
            f"median={statistics.median(recall_times) * 1000:.1f}ms, "
            f"p95={sorted(recall_times)[int(len(recall_times) * 0.95)] * 1000:.1f}ms, "
            f"max={max(recall_times) * 1000:.1f}ms"
        )
    out.append("")

    if n_ok == 0:
        out.append("**No successful questions to score.**")
        return "\n".join(out)

    # Determine the K columns from the first successful row.
    sample = next(r for r in rows if "error" not in r)
    ks = sorted(int(k.lstrip("@")) for k in sample["turn_recall"].keys())

    # Aggregate per question_type, plus an "Overall" row.
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        by_type[r["question_type"]].append(r)
    by_type["__overall__"] = [r for r in rows if "error" not in r]

    def avg(rs: list[dict], section: str, k: int) -> float:
        vals = [r[section][f"@{k}"] for r in rs]
        return sum(vals) / len(vals) if vals else 0.0

    # Turn-level recall table.
    out.append("## Turn-level recall@K")
    out.append("")
    out.append("Question type | n | " + " | ".join(f"@{k}" for k in ks))
    out.append("---|---|" + "|".join("---" for _ in ks))
    types_sorted = sorted(t for t in by_type if t != "__overall__") + ["__overall__"]
    for t in types_sorted:
        rs = by_type[t]
        label = "**Overall**" if t == "__overall__" else t
        cells = [fmt_pct(avg(rs, "turn_recall", k)) for k in ks]
        out.append(f"{label} | {len(rs)} | " + " | ".join(cells))
    out.append("")

    # Session-level recall table.
    out.append("## Session-level recall@K")
    out.append("")
    out.append("Question type | n | " + " | ".join(f"@{k}" for k in ks))
    out.append("---|---|" + "|".join("---" for _ in ks))
    for t in types_sorted:
        rs = by_type[t]
        label = "**Overall**" if t == "__overall__" else t
        cells = [fmt_pct(avg(rs, "session_recall", k)) for k in ks]
        out.append(f"{label} | {len(rs)} | " + " | ".join(cells))
    out.append("")

    # Errors (if any).
    if n_err > 0:
        out.append("## Errors")
        out.append("")
        for r in rows:
            if "error" in r:
                out.append(f"- `{r.get('question_id', '?')}`: {r['error']}")
        out.append("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="path to results_<variant>.jsonl")
    ap.add_argument("--out", default="", help="also write the report here")
    args = ap.parse_args()

    rows = load(Path(args.results))
    report = render(rows)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nreport written to: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
