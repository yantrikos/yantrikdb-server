# yantrikdb-server

Multi-tenant cognitive memory database server. Wire protocol + HTTP gateway, built-in embeddings, automatic replication and failover.

> Built on top of [yantrikdb](https://crates.io/crates/yantrikdb) — the cognitive memory engine.

## What it is

YantrikDB is the **memory subsystem for AI agents**. Not a vector store, not a key-value database — a cognitive memory database that:

- **Decays** — memories fade unless reinforced (exponential half-life)
- **Consolidates** — similar memories auto-merge
- **Conflicts** — contradictions are detected and surfaced
- **Sessions** — conversation context is first-class
- **Explains** — recall tells you *why* each result was returned
- **Personalizes** — emergent personality traits from access patterns
- **Replicates** — multi-node CRDT-based with auto failover
- **Encrypts** — AES-256-GCM at rest, TLS in transit

## Install

```bash
cargo install yantrikdb-server
```

Or download a pre-built binary:

```bash
wget https://github.com/yantrikos/yantrikdb-server/releases/latest/download/yantrikdb-linux-amd64
chmod +x yantrikdb-linux-amd64
```

## Quick start

```bash
yantrikdb db --data-dir ./data create default
yantrikdb token --data-dir ./data create --db default
yantrikdb serve --data-dir ./data
# Wire protocol on :7437, HTTP on :7438
```

## Cluster setup

```bash
yantrikdb cluster init \
  --node-id 1 \
  --output ./node1.toml \
  --peers 192.168.1.2:7440 \
  --witnesses 192.168.1.3:7440
yantrikdb serve --config ./node1.toml
```

See the [main README](https://github.com/yantrikos/yantrikdb-server) for full deployment guides, HTTP API reference, and operational documentation.

## Companion crates

- [`yql`](https://crates.io/crates/yql) — interactive REPL client (like psql)
- [`yantrikdb-witness`](https://crates.io/crates/yantrikdb-witness) — vote-only daemon for 2-node failover
- [`yantrikdb-protocol`](https://crates.io/crates/yantrikdb-protocol) — wire protocol codec

## License

AGPL-3.0-only — server and core engine.
Client SDKs (yql, Python SDK) are MIT-licensed.
