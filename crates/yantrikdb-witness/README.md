# yantrikdb-witness

Vote-only daemon for [YantrikDB](https://github.com/yantrikos/yantrikdb-server) clusters.

A tiny (~280 lines) Rust process that participates in Raft elections without storing any data. Used to break ties in 2-node clusters so you don't need 3 full data nodes for safe automatic failover.

## What it does

Stores only:
- `current_term` (Raft term)
- `voted_for` (which candidate you voted for in the current term)

Persists this to a small JSON file. Does **not** store memories, embeddings, oplog, or anything else. Cannot become a leader. Just votes.

## When to use it

A safe HA cluster needs **at least 3 voting members** to tolerate any single failure (quorum = 2 of 3). If you only want to run 2 full data nodes, you need a 3rd voter that doesn't need disk space — that's the witness.

```
┌─────────────┐  heartbeats   ┌─────────────┐
│  data node  │ ◄───────────▶ │  data node  │
│  (voter)    │  oplog sync   │  (voter)    │
└──────┬──────┘               └──────┬──────┘
       │                             │
       │     ┌────────────────┐      │
       └────▶│ witness        │◄─────┘
             │ (vote-only)    │
             │  ~10 MB RAM    │
             └────────────────┘
```

Same pattern as Azure SQL, MongoDB arbiters, MariaDB Galera garbd, Redis Sentinel.

## Install

```bash
cargo install yantrikdb-witness
```

## Run

```bash
yantrikdb-witness \
  --node-id 99 \
  --port 7440 \
  --cluster-secret <shared-secret> \
  --state-file ./witness-state.json
```

## License

AGPL-3.0-only — see [LICENSE](../../LICENSE).
