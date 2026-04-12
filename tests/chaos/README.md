# Chaos Tests

Fault-injection tests for YantrikDB cluster resilience.

## Prerequisites

Build the Docker images:

```bash
docker build -f docker/Dockerfile.yantrikdb -t yantrikdb:chaos .
docker build -f docker/Dockerfile.witness -t yantrikdb-witness:chaos .
```

## Running

```bash
# Start the 3-node cluster
docker compose -f tests/chaos/docker-compose.chaos.yml up -d

# Run all scenarios
python tests/chaos/run_chaos.py

# Run a specific scenario
python tests/chaos/run_chaos.py --scenario leader_kill

# Tear down
docker compose -f tests/chaos/docker-compose.chaos.yml down -v
```

## Scenarios

| Scenario | What it tests |
|---|---|
| `leader_kill` | SIGKILL leader mid-writes. Verify failover, no data loss. |
| `follower_kill` | Kill follower. Verify leader continues serving. |
| `network_partition` | iptables DROP between voters. Verify majority continues. |
| `kill9_mid_batch` | SIGKILL during writes. Verify SQLite WAL recovery. |

## Architecture

- `docker-compose.chaos.yml` — 2 voters + 1 witness on `172.28.0.0/24`
- `run_chaos.py` — Python harness that drives writes + injects faults
- Each container has `CAP_NET_ADMIN` for iptables/tc fault injection
- Cluster uses fast timers (500ms heartbeat, 2s election) for quick failover

## Adding New Scenarios

1. Add a `scenario_<name>()` function to `run_chaos.py`
2. Register it in the `SCENARIOS` dict
3. Follow the pattern: wait for leader → write → inject fault → verify → heal → verify
