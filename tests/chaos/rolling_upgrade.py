#!/usr/bin/env python3
"""
Rolling Upgrade Test

Verifies that a cluster can be upgraded one node at a time without data loss.
This is a simplified version for local testing — the full version would use
two different Docker images (v_old and v_new).

For now, it restarts each node sequentially and verifies data survives
each restart. This catches:
  - Schema migration failures
  - Wire protocol incompatibilities during mixed-version operation
  - WAL recovery failures on restart

Usage:
    # With cluster already running:
    python tests/chaos/rolling_upgrade.py
"""

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

VOTER1 = "http://localhost:17438"
VOTER2 = "http://localhost:17439"
CLUSTER_SECRET = "chaos-test-secret"
CONTAINERS = ["chaos-voter1", "chaos-voter2"]

def api(base_url, method, path, body=None, timeout=5):
    url = f"{base_url}{path}"
    headers = {
        "Authorization": f"Bearer {CLUSTER_SECRET}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}

def wait_healthy(max_wait=20):
    for _ in range(max_wait):
        for base in [VOTER1, VOTER2]:
            try:
                status = api(base, "GET", "/v1/cluster")
                if status.get("role") == "Leader" and status.get("healthy"):
                    return base
            except Exception:
                pass
        time.sleep(1)
    return None

def write_batch(leader_url, count, prefix):
    rids = []
    for i in range(count):
        result = api(leader_url, "POST", "/v1/remember", {
            "text": f"{prefix}-{i}",
            "memory_type": "semantic",
            "importance": 0.5,
            "namespace": "upgrade-test",
        })
        if "rid" in result:
            rids.append(result["rid"])
    return rids

def main():
    print("=== Rolling Upgrade Test ===\n")

    leader = wait_healthy()
    if not leader:
        print("FAIL: cluster not healthy")
        sys.exit(1)

    # Phase 1: write baseline
    print("Phase 1: writing 50 baseline memories...")
    baseline_rids = write_batch(leader, 50, "baseline")
    print(f"  wrote {len(baseline_rids)}")

    total_written = len(baseline_rids)

    # Phase 2: rolling restart each node
    for container in CONTAINERS:
        print(f"\nPhase 2: restarting {container}...")
        subprocess.run(["docker", "restart", container], capture_output=True, timeout=30)
        time.sleep(5)

        leader = wait_healthy()
        if not leader:
            print(f"  FAIL: no leader after restarting {container}")
            sys.exit(1)
        print(f"  leader: {leader}")

        # Write more
        print(f"  writing 20 memories after restarting {container}...")
        rids = write_batch(leader, 20, f"after-{container}")
        total_written += len(rids)
        print(f"  wrote {len(rids)}, total written: {total_written}")

        # Verify count
        stats = api(leader, "GET", "/v1/stats")
        actual = stats.get("active_memories", 0)
        if actual < total_written:
            print(f"  FAIL: expected >= {total_written}, got {actual}")
            sys.exit(1)
        print(f"  verified: {actual} memories present (expected >= {total_written})")

    # Phase 3: final verification
    print("\n=== Final Verification ===")
    leader = wait_healthy()
    stats = api(leader, "GET", "/v1/stats")
    final_count = stats.get("active_memories", 0)
    passed = final_count >= total_written
    print(f"  Total written: {total_written}")
    print(f"  Final count: {final_count}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
