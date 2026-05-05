#!/usr/bin/env python3
"""LongMemEval harness for YantrikDB.

For each question in the LongMemEval dataset:
  1. Boot a fresh per-question namespace `lme_<question_id>`.
  2. Write every turn from every haystack session as a memory, with
     metadata `(session_id, turn_idx, role, has_answer)`.
  3. Issue the question text against /v1/recall at multiple top-K
     values (1, 5, 10, 20) with expand_entities=false.
  4. Score:
       - turn-level recall@K: did top-K include any has_answer=true turn?
       - session-level recall@K: did top-K include any answer_session_ids turn?
  5. Append a JSONL row per question to results_<variant>.jsonl.

Run aggregation via metrics.py.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent

# Recall top-K values we score at. The smallest is what production
# agents typically retrieve; the largest gives a "is the right answer
# even reachable" upper bound.
TOP_KS = [1, 3, 5, 10, 20]
# top_k for the actual recall call — pull the largest, score at all
# smaller K by truncation. One round-trip per question, not five.
RECALL_TOP_K = max(TOP_KS)


def http_post(url: str, token: str, body: dict, timeout: float = 30.0) -> tuple[int, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode("utf-8"))
        except Exception:
            payload = {"error": str(e)}
        return e.code, payload
    except Exception as e:
        return 0, {"error": f"transport: {e}"}


def write_session(
    base_url: str, token: str, namespace: str,
    session_id: str, session_turns: list, session_date: str,
) -> tuple[int, int]:
    """Write all turns from one session. Returns (written, failed)."""
    written, failed = 0, 0
    for turn_idx, turn in enumerate(session_turns):
        text = f"{turn.get('role', '?')}: {turn.get('content', '')}"
        body = {
            "namespace": namespace,
            "text": text,
            "memory_type": "episodic",
            "domain": "conversation",
            "importance": 0.5,
            "metadata": {
                "session_id": session_id,
                "session_date": session_date,
                "turn_idx": turn_idx,
                "role": turn.get("role"),
                "has_answer": bool(turn.get("has_answer", False)),
            },
        }
        status, _ = http_post(f"{base_url}/v1/remember", token, body, timeout=15)
        if status != 200:
            # One retry after a brief sleep — handles transient
            # admission-control / network hiccups without aborting the
            # whole question over a single failed write out of hundreds.
            time.sleep(0.25)
            status, _ = http_post(f"{base_url}/v1/remember", token, body, timeout=15)
        if status == 200:
            written += 1
        else:
            failed += 1
    return written, failed


def score_question(question: dict, top_results: list) -> dict:
    """Compute turn-level + session-level recall@K for each K in TOP_KS."""
    answer_session_ids = set(question.get("answer_session_ids") or [])
    out = {"turn_recall": {}, "session_recall": {}}
    for k in TOP_KS:
        topk = top_results[:k]
        turn_hit = any(
            (r.get("metadata") or {}).get("has_answer") is True for r in topk
        )
        sess_hit = any(
            (r.get("metadata") or {}).get("session_id") in answer_session_ids
            for r in topk
        )
        out["turn_recall"][f"@{k}"] = int(turn_hit)
        out["session_recall"][f"@{k}"] = int(sess_hit)
    return out


def run_one(
    base_url: str, token: str, question: dict, top_k: int = RECALL_TOP_K,
) -> dict:
    qid = question["question_id"]
    qtype = question["question_type"]
    namespace = f"lme_{qid}"

    haystack_sessions = question.get("haystack_sessions") or []
    haystack_session_ids = question.get("haystack_session_ids") or []
    haystack_dates = question.get("haystack_dates") or [""] * len(haystack_sessions)

    # 1. Write every turn.
    write_t0 = time.time()
    total_written, total_failed = 0, 0
    for sess_idx, session_turns in enumerate(haystack_sessions):
        sess_id = haystack_session_ids[sess_idx] if sess_idx < len(haystack_session_ids) else f"unknown_{sess_idx}"
        sess_date = haystack_dates[sess_idx] if sess_idx < len(haystack_dates) else ""
        w, f = write_session(base_url, token, namespace, sess_id, session_turns, sess_date)
        total_written += w
        total_failed += f
    write_secs = time.time() - write_t0

    if total_failed > 0:
        return {
            "question_id": qid,
            "question_type": qtype,
            "error": f"{total_failed}/{total_written + total_failed} writes failed",
            "write_secs": write_secs,
        }

    # 2. Recall at the largest K, score at all smaller K by truncation.
    recall_t0 = time.time()
    body = {
        "namespace": namespace,
        "query": question["question"],
        "top_k": top_k,
        "expand_entities": False,
    }
    status, resp = http_post(f"{base_url}/v1/recall", token, body, timeout=30)
    recall_secs = time.time() - recall_t0
    if status != 200 or not isinstance(resp, dict) or "results" not in resp:
        return {
            "question_id": qid,
            "question_type": qtype,
            "error": f"recall HTTP {status}: {resp}",
            "write_secs": write_secs,
            "recall_secs": recall_secs,
        }

    results = resp.get("results") or []
    score = score_question(question, results)

    return {
        "question_id": qid,
        "question_type": qtype,
        "n_haystack_sessions": len(haystack_sessions),
        "n_haystack_turns": total_written,
        "write_secs": round(write_secs, 3),
        "recall_secs": round(recall_secs, 3),
        "n_results_returned": len(results),
        **score,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="oracle", choices=["oracle", "s", "m"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18438)
    ap.add_argument("--token", default=os.environ.get("YDB_TOKEN", ""))
    ap.add_argument("--limit", type=int, default=0,
                    help="process only first N questions (0 = all)")
    ap.add_argument("--start", type=int, default=0,
                    help="skip the first N questions (resume support)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.token:
        print("YDB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    data_path = HERE / f"longmemeval_{args.variant}.json"
    if not data_path.exists():
        print(f"missing {data_path.name}; run `python fetch.py --variant {args.variant}` first",
              file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.out) if args.out else (HERE / f"results_{args.variant}.jsonl")

    print(f"loading {data_path.name}...", flush=True)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"loaded {len(data)} questions", flush=True)

    base_url = f"http://{args.host}:{args.port}"

    questions = data[args.start:]
    if args.limit > 0:
        questions = questions[:args.limit]
    print(f"running {len(questions)} questions (start={args.start} limit={args.limit or 'all'})",
          flush=True)
    print(f"output: {out_path}", flush=True)

    t_overall = time.time()
    n_done, n_err = 0, 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, q in enumerate(questions):
            row = run_one(base_url, args.token, q)
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            n_done += 1
            if "error" in row:
                n_err += 1
                print(f"[{i+1}/{len(questions)}] ERR {row['question_id']}: {row['error']}",
                      flush=True)
            else:
                # One-line status; aggregator does the heavy lifting later.
                t_at_5 = row["turn_recall"]["@5"]
                s_at_5 = row["session_recall"]["@5"]
                elapsed = time.time() - t_overall
                rate = n_done / elapsed if elapsed > 0 else 0
                print(
                    f"[{i+1}/{len(questions)}] {row['question_id']} "
                    f"({row['question_type']}): turn@5={t_at_5} sess@5={s_at_5} "
                    f"write={row['write_secs']:.1f}s recall={row['recall_secs']:.2f}s "
                    f"({rate:.2f} q/s)",
                    flush=True,
                )

    print(
        f"\nDONE: {n_done} questions in {time.time() - t_overall:.1f}s "
        f"({n_err} errors). results -> {out_path}"
    )


if __name__ == "__main__":
    main()
