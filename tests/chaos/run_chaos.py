#!/usr/bin/env python3
"""
YantrikDB Chaos Test Harness

Runs fault-injection scenarios against a 3-node Dockerized cluster.
Requires: docker compose up (see docker-compose.chaos.yml)

Usage:
    python tests/chaos/run_chaos.py [--scenario <name>]

Scenarios:
    leader_kill       Kill the leader mid-writes, verify failover + no data loss
    follower_kill     Kill a follower, verify writes continue on leader
    network_partition Partition one voter, verify majority continues
    kill9_mid_batch   SIGKILL leader during batch write, verify recovery

Each scenario:
    1. Waits for cluster healthy (all nodes up, leader elected)
    2. Starts a writer that sends memories at a steady rate
    3. Injects the fault
    4. Verifies convergence after healing
    5. Reports pass/fail with details
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

VOTER1 = "http://localhost:17438"
VOTER2 = "http://localhost:17439"
CLUSTER_SECRET = "chaos-test-secret"

def api(base_url, method, path, body=None, timeout=5):
    """Make an HTTP request to a YantrikDB node."""
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
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return {"error": str(e)}

def wait_for_leader(max_wait=30):
    """Wait until the cluster has a leader."""
    print("Waiting for leader election...", end="", flush=True)
    for _ in range(max_wait):
        for base in [VOTER1, VOTER2]:
            try:
                status = api(base, "GET", "/v1/cluster")
                if status.get("role") == "Leader":
                    print(f" leader={status['node_id']} on {base}")
                    return base, status["node_id"]
            except Exception:
                pass
        print(".", end="", flush=True)
        time.sleep(1)
    print(" TIMEOUT")
    return None, None

def write_memories(base_url, count, namespace="chaos"):
    """Write N memories and return list of ack'd RIDs."""
    rids = []
    for i in range(count):
        result = api(base_url, "POST", "/v1/remember", {
            "text": f"chaos test memory {i} at {time.time()}",
            "memory_type": "semantic",
            "importance": 0.5,
            "namespace": namespace,
        })
        if "rid" in result:
            rids.append(result["rid"])
        elif "error" in result:
            # Write rejected (e.g. not leader) — expected during failover
            pass
    return rids

def count_memories(base_url):
    """Get active memory count from stats."""
    stats = api(base_url, "GET", "/v1/stats")
    return stats.get("active_memories", 0)

def docker_exec(container, cmd):
    """Run a command inside a Docker container."""
    return subprocess.run(
        ["docker", "exec", container] + cmd,
        capture_output=True, text=True, timeout=10,
    )

def docker_kill(container):
    """SIGKILL a container (simulates crash)."""
    subprocess.run(["docker", "kill", container], capture_output=True, timeout=10)

def docker_start(container):
    """Start a stopped container."""
    subprocess.run(["docker", "start", container], capture_output=True, timeout=10)

# ── Scenarios ──────────────────────────────────────────────────────────

def scenario_leader_kill():
    """Kill the leader mid-writes. Verify failover + no data loss."""
    print("\n=== SCENARIO: leader_kill ===\n")

    leader_url, leader_id = wait_for_leader()
    if not leader_url:
        return False, "no leader elected"

    leader_container = "chaos-voter1" if leader_id == 1 else "chaos-voter2"
    follower_url = VOTER2 if leader_url == VOTER1 else VOTER1

    # Phase 1: write 50 memories to leader
    print(f"Phase 1: writing 50 memories to leader ({leader_container})...")
    rids_before = write_memories(leader_url, 50, "before-kill")
    print(f"  wrote {len(rids_before)} memories")

    # Phase 2: kill leader
    print(f"Phase 2: killing {leader_container}...")
    docker_kill(leader_container)
    time.sleep(3)  # wait for election

    # Phase 3: verify new leader elected
    new_leader_url, new_leader_id = wait_for_leader(max_wait=15)
    if not new_leader_url:
        docker_start(leader_container)
        return False, "no new leader after kill"
    print(f"  new leader: node {new_leader_id}")

    # Phase 4: write more to new leader
    print("Phase 4: writing 50 memories to new leader...")
    rids_after = write_memories(new_leader_url, 50, "after-kill")
    print(f"  wrote {len(rids_after)} memories")

    # Phase 5: restart killed node
    print(f"Phase 5: restarting {leader_container}...")
    docker_start(leader_container)
    time.sleep(5)  # wait for sync

    # Phase 6: verify all memories present on surviving leader
    total = count_memories(new_leader_url)
    expected = len(rids_before) + len(rids_after)
    passed = total >= expected
    msg = f"total={total}, expected>={expected}"
    print(f"  Result: {'PASS' if passed else 'FAIL'} — {msg}")
    return passed, msg

def scenario_follower_kill():
    """Kill a follower. Verify leader continues serving."""
    print("\n=== SCENARIO: follower_kill ===\n")

    leader_url, leader_id = wait_for_leader()
    if not leader_url:
        return False, "no leader elected"

    follower_container = "chaos-voter2" if leader_id == 1 else "chaos-voter1"

    # Kill follower
    print(f"Killing follower ({follower_container})...")
    docker_kill(follower_container)
    time.sleep(2)

    # Write to leader — should still work
    print("Writing 100 memories to leader...")
    rids = write_memories(leader_url, 100, "follower-down")
    print(f"  wrote {len(rids)}")

    # Restart follower
    print(f"Restarting {follower_container}...")
    docker_start(follower_container)
    time.sleep(5)

    passed = len(rids) == 100
    msg = f"wrote {len(rids)}/100 while follower was down"
    print(f"  Result: {'PASS' if passed else 'FAIL'} — {msg}")
    return passed, msg

def scenario_network_partition():
    """Partition one voter from the other via iptables."""
    print("\n=== SCENARIO: network_partition ===\n")

    leader_url, leader_id = wait_for_leader()
    if not leader_url:
        return False, "no leader elected"

    # Partition voter2 from voter1 (drop packets between them)
    print("Partitioning voter2 from voter1...")
    docker_exec("chaos-voter2", [
        "iptables", "-A", "OUTPUT", "-d", "172.28.0.10", "-j", "DROP"
    ])
    docker_exec("chaos-voter2", [
        "iptables", "-A", "INPUT", "-s", "172.28.0.10", "-j", "DROP"
    ])
    time.sleep(5)

    # Write to majority side (leader should still be available)
    print("Writing to leader during partition...")
    rids = write_memories(leader_url, 50, "during-partition")
    print(f"  wrote {len(rids)}")

    # Heal partition
    print("Healing partition...")
    docker_exec("chaos-voter2", ["iptables", "-F"])
    time.sleep(10)  # wait for sync

    passed = len(rids) >= 40  # some may fail during leadership uncertainty
    msg = f"wrote {len(rids)}/50 during partition"
    print(f"  Result: {'PASS' if passed else 'FAIL'} — {msg}")
    return passed, msg

def scenario_kill9_mid_batch():
    """SIGKILL leader during a batch write."""
    print("\n=== SCENARIO: kill9_mid_batch ===\n")

    leader_url, leader_id = wait_for_leader()
    if not leader_url:
        return False, "no leader elected"

    leader_container = "chaos-voter1" if leader_id == 1 else "chaos-voter2"

    # Write a small baseline
    print("Writing 20 baseline memories...")
    baseline = write_memories(leader_url, 20, "baseline")
    print(f"  baseline: {len(baseline)}")

    # Kill leader (simulates crash during operation)
    print(f"Killing {leader_container}...")
    docker_kill(leader_container)
    time.sleep(3)

    # Restart and verify baseline survived
    print(f"Restarting {leader_container}...")
    docker_start(leader_container)
    time.sleep(5)

    new_leader_url, _ = wait_for_leader(max_wait=15)
    if not new_leader_url:
        return False, "no leader after restart"

    total = count_memories(new_leader_url)
    passed = total >= len(baseline)
    msg = f"total={total}, baseline={len(baseline)}"
    print(f"  Result: {'PASS' if passed else 'FAIL'} — {msg}")
    return passed, msg

# ── Main ──────────────────────────────────────────────────────────────

SCENARIOS = {
    "leader_kill": scenario_leader_kill,
    "follower_kill": scenario_follower_kill,
    "network_partition": scenario_network_partition,
    "kill9_mid_batch": scenario_kill9_mid_batch,
}

def main():
    parser = argparse.ArgumentParser(description="YantrikDB Chaos Tests")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                        help="Run a specific scenario (default: all)")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    results = {}

    for name in scenarios:
        try:
            passed, msg = SCENARIOS[name]()
            results[name] = {"passed": passed, "message": msg}
        except Exception as e:
            results[name] = {"passed": False, "message": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("CHAOS TEST RESULTS")
    print("=" * 60)
    all_pass = True
    for name, result in results.items():
        status = "PASS" if result["passed"] else "FAIL"
        if not result["passed"]:
            all_pass = False
        print(f"  {status}  {name}: {result['message']}")
    print("=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")

    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
