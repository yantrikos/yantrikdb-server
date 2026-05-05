#!/usr/bin/env python3
"""Seed skills_corpus.jsonl into a YantrikDB instance.

Rate-limited writer for the skill-recall benchmark. Cluster-aware: tries
each node in --nodes (comma-separated) until one accepts the write.

The default rate (10/sec) is conservative; the server-side embedder is
typically the bottleneck and tops out around 25-30 writes/sec on a
single node with the builtin all-MiniLM-L6-v2 model.

Args:
  --start N         skip the first N skills (resume after a partial run)
  --limit N         stop after N writes (0 = all)
  --rate-per-sec X  client-side rate cap (default 10)
  --nodes IPs       comma-separated node IPs (default cluster)
  --port N          HTTP port (default 7438; 17438 for local docker)
  --token TOKEN     YDB_TOKEN; otherwise read from env
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


CORPUS = Path(__file__).parent / "skills_corpus.jsonl"


def post_remember(node: str, body: dict, token: str, timeout: int = 30, port: int = 7438):
    url = f"http://{node}:{port}/v1/remember"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--rate-per-sec", type=float, default=10.0)
    ap.add_argument("--token", default=os.environ.get("YDB_TOKEN", ""))
    ap.add_argument("--nodes", default="192.168.4.140,192.168.4.141",
                    help="comma-separated cluster node IPs")
    ap.add_argument("--port", type=int, default=7438,
                    help="HTTP port (use 17438 for local docker)")
    args = ap.parse_args()

    if not args.token:
        print("YDB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    sleep_per = 1.0 / max(args.rate_per_sec, 0.1)

    with open(CORPUS, "r", encoding="utf-8") as f:
        skills = [json.loads(line) for line in f if line.strip()]

    end = len(skills) if args.limit <= 0 else min(len(skills), args.start + args.limit)
    skills_to_send = skills[args.start:end]

    print(f"seeding {len(skills_to_send)} skills (range {args.start}..{end}) "
          f"to nodes={nodes} at {args.rate_per_sec}/sec")

    written = 0
    failed = 0
    last_print = time.time()
    start_time = time.time()
    for i, body in enumerate(skills_to_send):
        wrote = False
        last_resp = None
        for node in nodes:
            status, resp = post_remember(node, body, args.token, timeout=30, port=args.port)
            last_resp = resp
            if status == 200 and "rid" in resp:
                written += 1
                wrote = True
                break
        if not wrote:
            failed += 1
            print(f"  FAIL {body['metadata']['skill_id']}: {last_resp}", file=sys.stderr)

        # Progress
        if time.time() - last_print >= 5.0:
            elapsed = time.time() - start_time
            print(f"  progress: {i+1}/{len(skills_to_send)} (written={written} failed={failed}) "
                  f"elapsed={elapsed:.1f}s rate={(i+1)/elapsed:.1f}/s")
            last_print = time.time()

        time.sleep(sleep_per)

    elapsed = time.time() - start_time
    print(f"\nDONE: written={written} failed={failed} elapsed={elapsed:.1f}s "
          f"effective_rate={len(skills_to_send)/elapsed:.1f}/s")


if __name__ == "__main__":
    main()
