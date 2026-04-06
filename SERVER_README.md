# YantrikDB Server

A cognitive memory database server with native wire protocol, HTTP gateway, built-in embeddings, automatic failover, and Raft-lite replication.

> **Status**: v0.3.0 — production-grade replication, ready for homelab/enterprise deployment.

## What it is

YantrikDB is the **memory subsystem for AI agents**. Not a vector store, not a key-value database — it's a cognitive memory database that:

- **Decays** — memories fade unless reinforced (exponential half-life)
- **Consolidates** — similar memories auto-merge into stronger ones
- **Conflicts** — contradictions are detected and surfaced for resolution
- **Sessions** — conversation context is first-class
- **Explains** — recall tells you *why* each result was returned
- **Personalizes** — emergent personality traits from access patterns
- **Replicates** — multi-node CRDT-based with auto failover

## Three binaries

| Binary | Purpose | Size |
|--------|---------|------|
| `yantrikdb` | Full server (data plane + cluster member) | ~22 MB |
| `yql` | Interactive client (like `psql`) | ~8 MB |
| `yantrikdb-witness` | Vote-only daemon for 2-node failover | ~3 MB |

All three available for `linux-amd64`, `windows-amd64`, `macos-amd64`, `macos-arm64`.

## Quick start (single node)

```bash
# Download
wget https://github.com/yantrikos/yantrikdb-server/releases/latest/download/yantrikdb-linux-amd64
chmod +x yantrikdb-linux-amd64
mv yantrikdb-linux-amd64 /usr/local/bin/yantrikdb

wget https://github.com/yantrikos/yantrikdb-server/releases/latest/download/yql-linux-amd64
chmod +x yql-linux-amd64
mv yql-linux-amd64 /usr/local/bin/yql

# Setup
yantrikdb db --data-dir ./data create default
TOKEN=$(yantrikdb token --data-dir ./data create --db default | grep ^ydb_)
echo "$TOKEN" > token.txt

# Run
yantrikdb serve --data-dir ./data
# Wire protocol on :7437, HTTP on :7438
```

In another terminal:

```bash
yql --host localhost -t "$(cat token.txt)"
yantrikdb> remember "Alice leads engineering at Acme" importance=0.9 domain=work
yantrikdb> recall who leads engineering
yantrikdb> \stats
yantrikdb> \q
```

## Cluster setup (3 nodes with auto failover)

For your homelab — works great with 2 full nodes + 1 tiny witness.

### Architecture

```
┌─────────────┐  heartbeats   ┌─────────────┐
│  node1      │ ◄───────────▶ │  node2      │
│  (voter)    │  oplog sync   │  (voter)    │
│  192.168.x  │               │  192.168.y  │
└──────┬──────┘               └──────┬──────┘
       │                             │
       │     ┌────────────────┐      │
       └────▶│ witness        │◄─────┘
             │ (vote-only)    │
             │ 192.168.z      │
             └────────────────┘
```

- **Voters** store data, accept reads/writes when leader, become read-only when follower
- **Witness** stores no data, just votes in elections to break ties
- **Quorum** = 2 of 3 nodes — survives any single failure
- **Auto failover** — if leader dies, surviving voter + witness elect a new leader in ~5 seconds

### Step 1: Generate configs

On **node1**:
```bash
yantrikdb cluster init \
  --node-id 1 \
  --output ./node1.toml \
  --data-dir ./data \
  --peers 192.168.4.181:7440,192.168.4.182:7440 \
  --witnesses 192.168.4.183:7440
```

This prints a `cluster_secret` like `ydb_cluster_3a8f...` — **copy this**, you'll need it on every other node.

On **node2** (use the same secret):
```bash
yantrikdb cluster init \
  --node-id 2 \
  --output ./node2.toml \
  --data-dir ./data \
  --peers 192.168.4.180:7440,192.168.4.183:7440 \
  --witnesses 192.168.4.183:7440 \
  --secret <paste-secret-from-node1>
```

### Step 2: Create the database on each voter

```bash
yantrikdb db --data-dir ./data create default
```

(The witness doesn't need a database.)

### Step 3: Start the witness

On the witness machine:
```bash
yantrikdb-witness \
  --node-id 99 \
  --port 7440 \
  --cluster-secret <paste-secret> \
  --state-file ./witness-state.json
```

### Step 4: Start the voters

On node1:
```bash
yantrikdb serve --config ./node1.toml
```

On node2:
```bash
yantrikdb serve --config ./node2.toml
```

After ~5 seconds, an election runs and one voter becomes leader.

### Step 5: Verify

From any machine with `yql`:
```bash
yql --host 192.168.4.180 -t <cluster_secret>
yantrikdb> \cluster
```

You'll see something like:
```
  node #1 — Leader
  term: 1
  leader: 1
  healthy: yes | writable: yes
  quorum: 2

+---------+--------------------+---------+-----------+------+----------+
| node_id | addr               | role    | reachable | term | last_seen|
+---------+--------------------+---------+-----------+------+----------+
| 2       | 192.168.4.182:7440 | voter   | ✓         | 1    | 0.5s ago |
| 99      | 192.168.4.183:7440 | witness | ✓         | 1    | 0.5s ago |
+---------+--------------------+---------+-----------+------+----------+
```

### Test failover

Kill the leader (`Ctrl+C` or `systemctl stop yantrikdb`). Within ~5 seconds:
- The other voter detects missed heartbeats
- Runs an election
- Witness grants its vote
- New leader is elected

Run `\cluster` from yql against the surviving voter — it should now show itself as the leader.

When the old leader rejoins, it sees the higher term and demotes itself to follower automatically.

## Configuration reference

Full `yantrikdb.toml`:

```toml
[server]
wire_port = 7437        # Client wire protocol
http_port = 7438        # HTTP gateway
data_dir = "./data"

[tls]                   # Optional — both server ports support TLS
cert_path = "/etc/yantrikdb/cert.pem"
key_path = "/etc/yantrikdb/key.pem"

[embedding]
strategy = "builtin"    # or "client_only"
dim = 384               # all-MiniLM-L6-v2 dimension

[background]
consolidation_interval_minutes = 30
decay_sweep_interval_minutes = 60

[limits]
max_databases = 100
max_connections = 1000

[cluster]
node_id = 1
role = "voter"          # voter | read_replica | witness | single
cluster_port = 7440
heartbeat_interval_ms = 1000
election_timeout_ms = 5000
cluster_secret = "ydb_cluster_..."
replication_mode = "async"  # or "sync"

[[cluster.peers]]
addr = "192.168.4.182:7440"
role = "voter"

[[cluster.peers]]
addr = "192.168.4.183:7440"
role = "witness"
```

## Authentication

Two ways to authenticate:

1. **Per-node tokens** — generated with `yantrikdb token create --db default`. Each node has its own tokens. Use this for single-node deployments.

2. **Cluster master token** — when clustering is enabled, the `cluster_secret` doubles as a master Bearer token that works on **any node** in the cluster. Recommended for clustered deployments.

Both pass via `Authorization: Bearer ydb_...` header (HTTP) or AUTH frame (wire protocol).

## HTTP API

All endpoints require `Authorization: Bearer <token>`.

| Method | Path | Description |
|--------|------|-------------|
| GET    | `/v1/health` | Health + cluster status |
| GET    | `/v1/cluster` | Detailed cluster status |
| GET    | `/v1/stats` | Engine statistics |
| POST   | `/v1/remember` | Store a memory |
| POST   | `/v1/recall` | Semantic search |
| POST   | `/v1/forget` | Tombstone a memory |
| POST   | `/v1/relate` | Create graph edge |
| POST   | `/v1/think` | Run consolidation/conflict scan |
| GET    | `/v1/conflicts` | List open conflicts |
| POST   | `/v1/conflicts/{id}/resolve` | Resolve a conflict |
| GET    | `/v1/personality` | Derived personality traits |
| POST   | `/v1/sessions` | Start a cognitive session |
| DELETE | `/v1/sessions/{id}` | End a session |
| GET    | `/v1/databases` | List databases |
| POST   | `/v1/databases` | Create a database |

### Example: store and search

```bash
TOKEN=ydb_cluster_3a8f...

curl -X POST http://192.168.4.180:7438/v1/remember \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "Alice leads engineering at Acme", "importance": 0.9, "domain": "work"}'

curl -X POST http://192.168.4.180:7438/v1/recall \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "who leads engineering", "top_k": 5}'
```

## Python SDK

```bash
pip install httpx msgpack
# Use sdk/python/yantrikdb (or vendor it into your project)
```

```python
from yantrikdb import connect

db = connect("http://192.168.4.180:7438", token="ydb_cluster_3a8f...")

db.remember("Alice leads engineering", importance=0.9, domain="work")
results = db.recall("who leads engineering?", top_k=5)
for r in results.results:
    print(f"[{r.score:.2f}] {r.text}")

with db.session("chat-1") as s:
    s.remember("User asked about pricing")
    related = s.recall("user questions")
```

## yql interactive client

```bash
yql --host 192.168.4.180 -p 7438 -t ydb_cluster_3a8f...
```

```
yantrikdb> remember "Alice leads engineering" importance=0.9 domain=work
✓ stored: 019d623a-3d70-712e-9315-e1da5ee41114

yantrikdb> recall who leads engineering top=5
+---+-------+---------------------------------+--------+
| # | score | text                            | domain |
+---+-------+---------------------------------+--------+
| 1 | 1.41  | Alice leads engineering at Acme | work   |
+---+-------+---------------------------------+--------+

yantrikdb> relate Alice -> Acme as works_at
✓ edge: 019d623a-41cf-71a2 (Alice -[works_at]-> Acme)

yantrikdb> \cluster
yantrikdb> \stats
yantrikdb> \q
```

Type `\h` for full command reference.

## Systemd unit

`/etc/systemd/system/yantrikdb.service`:

```ini
[Unit]
Description=YantrikDB Server
After=network.target

[Service]
Type=simple
User=yantrikdb
Group=yantrikdb
ExecStart=/usr/local/bin/yantrikdb serve --config /etc/yantrikdb/yantrikdb.toml
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

For the witness:

```ini
[Unit]
Description=YantrikDB Witness
After=network.target

[Service]
Type=simple
User=yantrikdb
Group=yantrikdb
Environment=YANTRIKDB_CLUSTER_SECRET=ydb_cluster_...
ExecStart=/usr/local/bin/yantrikdb-witness --node-id 99 --port 7440 --state-file /var/lib/yantrikdb-witness/state.json
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Operations

### Backup

```bash
yantrikdb export default --data-dir ./data > backup-$(date +%F).jsonl
```

### Restore

```bash
yantrikdb import default --data-dir ./data < backup-2026-04-06.jsonl
```

### Cluster status

```bash
yantrikdb cluster status --url http://192.168.4.180:7438 --token <secret>
```

Or from yql:

```
yantrikdb> \cluster
```

## Failure modes (3-node cluster: 2 voters + witness)

| Failure | Behavior |
|---------|----------|
| Leader voter dies | Other voter + witness elect new leader in <10s |
| Follower voter dies | Leader keeps writing (still has quorum with witness) |
| Witness dies | Both voters keep going, no new elections allowed |
| Witness + follower die | Leader becomes read-only (no quorum) |
| Network partition isolates a voter | Isolated voter loses quorum, becomes read-only |
| All nodes die | Restart any node — it loads persistent state, rejoins cluster |

## Development

```bash
git clone https://github.com/yantrikos/yantrikdb-server
cd yantrikdb-server

# Build everything
cargo build --release

# Run all tests
cargo test -p yantrikdb-protocol -p yantrikdb-server

# Run cluster integration tests (spawns subprocess clusters)
cargo test -p yantrikdb-server --test cluster_integration -- --ignored --nocapture
```

## License

- **Server** (`yantrikdb`, `yantrikdb-witness`): AGPL-3.0
- **Client SDK** (`yql`, Python SDK): MIT

## Credits

Built by Pranab Sarkar with a sidekick. Phase 1, 2, and 3 all shipped on the same day in April 2026.
