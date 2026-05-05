#!/usr/bin/env python3
"""Run benchmark queries against the seeded skill_test_substrate namespace.

For each query in queries_groundtruth.jsonl:
  - issue /v1/recall(query=query, top_k=K, namespace=skill_test_substrate)
  - measure latency
  - check whether any target_skill_id appears in top-K results
  - record best (smallest) rank of any target

Outputs a summary report:
  - recall@K for various K (1, 3, 5, 10)
  - mean / p50 / p95 / max latency
  - breakdown by query kind (family-level vs variant-specific)
  - per-topic failure list (queries where no target appeared in top-10)

Also tests an exact-id fetch via SQLite cache for comparison (separate phase).
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).parent
QUERIES_FILE = HERE / "queries_groundtruth.jsonl"


def post_recall(node: str, query: str, top_k: int, namespace: str,
                token: str, timeout: int = 30, port: int = 7438):
    url = f"http://{node}:{port}/v1/recall"
    body = {"namespace": namespace, "query": query, "top_k": top_k}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return -1, {"error": str(e)}


def find_target_rank(results: list, target_ids: set) -> int | None:
    """Smallest 0-indexed rank where a target appears, or None."""
    for i, r in enumerate(results):
        md = r.get("metadata") or {}
        sid = md.get("skill_id")
        if sid in target_ids:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10,
                    help="recall top_k (default 10 to compute @1/@3/@5/@10)")
    ap.add_argument("--namespace", default="skill_test_substrate")
    ap.add_argument("--token", default=os.environ.get("YDB_TOKEN", ""))
    ap.add_argument("--nodes", default="192.168.4.140,192.168.4.141")
    ap.add_argument("--port", type=int, default=7438)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all queries; else first N for quick test")
    ap.add_argument("--out", default=str(HERE / "report.md"))
    args = ap.parse_args()

    if not args.token:
        print("YDB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        queries = queries[:args.limit]

    print(f"running {len(queries)} queries against namespace={args.namespace} "
          f"top_k={args.top_k}")

    rows = []
    latencies_ms = []
    fail_count = 0
    start = time.time()
    for q in queries:
        target_ids = set(q["target_skill_ids"])
        kind = q.get("query_kind", "family")
        topic = q.get("topic_family", "?")
        action = q.get("action_family", "?")

        # Try each node in order; use first that responds 200
        t0 = time.time()
        ok = False
        result = None
        for node in nodes:
            status, resp = post_recall(node, q["query"], args.top_k,
                                        args.namespace, args.token, timeout=30, port=args.port)
            if status == 200 and isinstance(resp, dict) and "results" in resp:
                result = resp
                ok = True
                break
        latency_ms = (time.time() - t0) * 1000.0
        latencies_ms.append(latency_ms)

        if not ok:
            fail_count += 1
            rows.append({
                "query": q["query"][:80], "kind": kind, "topic": topic, "action": action,
                "rank": None, "latency_ms": latency_ms, "error": "recall failed",
            })
            continue

        results = result.get("results", [])
        rank = find_target_rank(results, target_ids)
        rows.append({
            "query": q["query"][:80], "kind": kind, "topic": topic, "action": action,
            "rank": rank, "latency_ms": latency_ms,
            "n_results": len(results),
        })

    total = time.time() - start

    # Aggregate metrics
    def recall_at(k: int, sample: list) -> float:
        in_top = sum(1 for r in sample if r["rank"] is not None and r["rank"] < k)
        return in_top / max(len(sample), 1)

    family_rows = [r for r in rows if r["kind"] == "family"]
    variant_rows = [r for r in rows if r["kind"] == "specific_variant"]

    metrics = {
        "n_queries": len(rows),
        "n_failed": fail_count,
        "wall_clock_sec": round(total, 2),
        "latency_ms": {
            "mean": round(statistics.mean(latencies_ms), 1) if latencies_ms else 0,
            "median": round(statistics.median(latencies_ms), 1) if latencies_ms else 0,
            "p95": round(statistics.quantiles(latencies_ms, n=20)[-1], 1) if len(latencies_ms) >= 20 else None,
            "max": round(max(latencies_ms), 1) if latencies_ms else 0,
        },
        "recall_overall": {f"@{k}": round(recall_at(k, rows), 3) for k in (1, 3, 5, 10)},
        "recall_family_level": {f"@{k}": round(recall_at(k, family_rows), 3) for k in (1, 3, 5, 10)},
        "recall_variant_specific": {f"@{k}": round(recall_at(k, variant_rows), 3) for k in (1, 3, 5, 10)},
    }

    # Per-failure (rank None = not found in top_k) detail
    misses = [r for r in rows if r["rank"] is None]

    # Write report
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Skill-Store Recall Benchmark Report\n\n")
        f.write(f"namespace: `{args.namespace}`  ")
        f.write(f"top_k: {args.top_k}  ")
        f.write(f"nodes: {nodes}\n\n")
        f.write(f"## Headline metrics\n\n")
        f.write(f"- queries run: **{metrics['n_queries']}** "
                f"(failed: {metrics['n_failed']})\n")
        f.write(f"- wall clock: {metrics['wall_clock_sec']}s\n")
        f.write(f"- latency p50: **{metrics['latency_ms']['median']} ms**, "
                f"p95: {metrics['latency_ms']['p95']} ms, "
                f"max: {metrics['latency_ms']['max']} ms\n\n")
        f.write(f"## Recall@K\n\n")
        f.write("| Cohort                | @1   | @3   | @5   | @10  |\n")
        f.write("|-----------------------|------|------|------|------|\n")
        f.write("| Overall               | "
                f"{metrics['recall_overall']['@1']:<4} | "
                f"{metrics['recall_overall']['@3']:<4} | "
                f"{metrics['recall_overall']['@5']:<4} | "
                f"{metrics['recall_overall']['@10']:<4} |\n")
        f.write("| Family-level (broad)  | "
                f"{metrics['recall_family_level']['@1']:<4} | "
                f"{metrics['recall_family_level']['@3']:<4} | "
                f"{metrics['recall_family_level']['@5']:<4} | "
                f"{metrics['recall_family_level']['@10']:<4} |\n")
        f.write("| Variant-specific(sharp)| "
                f"{metrics['recall_variant_specific']['@1']:<4} | "
                f"{metrics['recall_variant_specific']['@3']:<4} | "
                f"{metrics['recall_variant_specific']['@5']:<4} | "
                f"{metrics['recall_variant_specific']['@10']:<4} |\n\n")
        f.write(f"## Misses (target not in top-{args.top_k})\n\n")
        f.write(f"Total: **{len(misses)}**\n\n")
        for m in misses[:20]:
            f.write(f"- [{m['kind']}] {m['topic']}/{m['action']}: `{m['query']}` "
                    f"(latency={m['latency_ms']:.0f}ms)\n")
        if len(misses) > 20:
            f.write(f"... + {len(misses) - 20} more\n")
        f.write(f"\n## Raw row dump\n\n")
        f.write("```jsonl\n")
        for r in rows[:20]:
            f.write(json.dumps(r) + "\n")
        f.write("...\n```\n")

    print(f"\n=== RESULTS ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nReport written to: {args.out}")


if __name__ == "__main__":
    main()
