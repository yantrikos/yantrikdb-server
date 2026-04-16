# YantrikDB

**A memory database that forgets, consolidates, and detects contradictions.**

Vector databases store memories. They don't manage them. After 10,000 memories, recall quality degrades because there's no consolidation, no forgetting, no conflict resolution. Your AI agent just gets noisier.

YantrikDB is different. It's a **cognitive memory engine** — embed it, run it as a server, or connect via MCP. It thinks about what it stores.

> **The bigger picture:** YantrikDB is the memory layer being built on the road to [YantrikOS](https://github.com/yantrikos) — an AI-native operating system where agents are first-class primitives, not apps on top. Memory was the bottleneck, so we're shipping it first.

[![Crates.io](https://img.shields.io/crates/v/yantrikdb-server)](https://crates.io/crates/yantrikdb-server)
[![PyPI](https://img.shields.io/pypi/v/yantrikdb)](https://pypi.org/project/yantrikdb/)
[![Docker](https://img.shields.io/badge/docker-ghcr.io%2Fyantrikos%2Fyantrikdb-blue)](https://github.com/yantrikos/yantrikdb-server/pkgs/container/yantrikdb)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

![YantrikDB demo: storing three facts, recalling them, then think() flagging a contradiction between two memories](docs/images/demo.gif)

## 99.9% token savings vs file-based memory

| Memories | File-Based (CLAUDE.md) | YantrikDB | Token Savings | Recall Precision |
|---|---|---|---|---|
| 100 | 1,770 tokens | 69 tokens | **96%** | 66% |
| 500 | 9,807 tokens | 72 tokens | **99.3%** | 77% |
| 1,000 | 19,988 tokens | 72 tokens | **99.6%** | 84% |
| 5,000 | 101,739 tokens | 53 tokens | **99.9%** | 88% |

At 500 memories, file-based memory exceeds 32K context. At 5,000, it doesn't fit in any model — not even 200K. YantrikDB stays at ~70 tokens per query. **Precision improves with more data** — the opposite of context stuffing.

Reproduce: `python benchmarks/bench_token_savings.py`

---

## Three things no other database does

### 1. It forgets

```python
db.record("read the SLA doc by Friday", importance=0.4, half_life=86400)  # 1 day
# 24 hours later, this memory's relevance score has decayed
# 7 days later, recall stops surfacing it unless explicitly queried
```

### 2. It consolidates

```python
# 20 similar memories about the same meeting
for note in meeting_notes:
    db.record(note, namespace="standup-2026-04-12")

db.think()
# → {"consolidation_count": 5}  # collapsed 20 fragments into 5 canonical memories
```

### 3. It detects contradictions

```python
db.record("CEO is Alice")
db.record("CEO is Bob")  # added later in another conversation

db.think()
# → {"conflicts_found": 1, "conflicts": [{"memory_a": "CEO is Alice",
#                                         "memory_b": "CEO is Bob",
#                                         "type": "factual_contradiction"}]}
```

Plus: temporal decay with configurable half-life, entity graph with relationship edges, personality derivation from memory patterns, session-aware context surfacing, multi-signal scoring (recency × importance × similarity × graph proximity).

---

## What makes this different

YantrikDB isn't just storage with operations. The engine has a layer that makes agents feel less reactive:

- **Proactive triggers** — the system surfaces what needs attention: pending conflicts, decaying important memories, approaching deadlines, patterns across domains. Agents don't have to ask what they should care about. The memory tells them.
- **Derived personality** — stable tendencies extracted from memory patterns over time. "This user prefers X, reacts to Y, values Z." Informs default agent behavior across sessions.
- **Procedural memory** — strategies that worked before get recorded and reinforced. Agents learn what to do, not just what they know.
- **Temporal awareness** — `stale` surfaces important memories that haven't been touched recently. `upcoming` surfaces memories with approaching deadlines.

Full cognitive architecture lives in the [standalone engine repo](https://github.com/yantrikos/yantrikdb). This server repo focuses on deployment, HTTP API, and cluster operations.

---

## Three ways to use it

### As a network server

```bash
docker run -p 7438:7438 ghcr.io/yantrikos/yantrikdb:latest
curl -X POST http://localhost:7438/v1/remember -d '{"text":"hello"}'
```

Single Rust binary. HTTP + binary wire protocol. 2-voter + 1-witness HA cluster via Docker Compose or Kubernetes. Per-tenant quotas, Prometheus metrics, AES-256-GCM at-rest encryption, runtime deadlock detection. See [docker-compose.cluster.yml](deploy/docker-compose.cluster.yml) and [k8s manifests](deploy/kubernetes/).

### As an MCP server (Claude Code, Cursor, Windsurf)

```bash
pip install yantrikdb-mcp
```

Add to your MCP client config — the agent auto-recalls context, auto-remembers decisions, auto-detects contradictions. No prompting needed. See [yantrikdb-mcp](https://github.com/yantrikos/yantrikdb-mcp).

### As an embedded library (Python or Rust)

```bash
pip install yantrikdb
# or
cargo add yantrikdb
```

```python
import yantrikdb
db = yantrikdb.YantrikDB("memory.db", embedding_dim=384)
db.set_embedder(SentenceTransformer("all-MiniLM-L6-v2"))
db.record("Alice leads engineering", importance=0.8)
db.recall("who leads the team?", top_k=3)
db.think()  # consolidate, detect conflicts, derive personality
```

---

## Performance

Live numbers from a 2-core LXC cluster with 1689 memories:

| Operation | Latency |
|---|---|
| Recall p50 | 112ms (most is query embedding ~100ms) |
| Recall p99 | 190ms |
| Batch write | 76 writes/sec |
| Engine lock acquire | <0.1ms |
| Deep health probe | <1ms |

For pre-computed embeddings (skip query-time embedding), recall p50 drops to ~5ms.

---

## Status

**v0.5.13** — hardened alpha + RFC 006 Phase 0 observability telemetry shipped. The embeddable engine has been used in production by the YantrikOS ecosystem since early 2026. The network server runs live on a 3-node Proxmox cluster with multiple tenants.

A 42-task hardening sprint just completed across 8 epics:
- `parking_lot` mutexes everywhere with runtime deadlock detection (caught a self-deadlock that would have taken hours to find with std::sync)
- Per-handler Prometheus metrics, structured JSON logging, deep health checks
- Chaos-tested failover (leader kill, network partition, kill-9 mid-write)
- Per-tenant quotas, load shedding, control plane replication
- 1178 core tests + chaos harness + cargo-fuzz + CRDT property tests
- 5 operational runbooks, watchdog with auto-restart

Read the maturity notes: https://yantrikdb.com/server/quickstart/#maturity



## The Problem

Current AI memory is:

> Store everything → Embed → Retrieve top-k → Inject into context → Hope it helps.

That's not memory. That's a search engine with extra steps.

Real memory is hierarchical, compressed, contextual, self-updating, emotionally weighted, time-aware, and predictive. YantrikDB is built for that.

## Why Not Existing Solutions?

| Solution | What it does | What it lacks |
|----------|-------------|---------------|
| **Vector DBs** (Pinecone, Weaviate) | Nearest-neighbor lookup | No decay, no causality, no self-organization |
| **Knowledge Graphs** (Neo4j) | Structured relations | Poor for fuzzy memory, not adaptive |
| **Memory Frameworks** (LangChain, Mem0) | Retrieval wrappers | Not a memory architecture — just middleware |
| **File-based** (CLAUDE.md, memory files) | Dump everything into context | O(n) token cost, no relevance filtering |

### Benchmark: Selective Recall vs. File-Based Memory

| Memories | File-Based | YantrikDB | Token Savings | Precision |
|----------|-----------|-----------|---------------|-----------|
| 100 | 1,770 tokens | 69 tokens | **96%** | 66% |
| 500 | 9,807 tokens | 72 tokens | **99.3%** | 77% |
| 1,000 | 19,988 tokens | 72 tokens | **99.6%** | 84% |
| 5,000 | 101,739 tokens | 53 tokens | **99.9%** | 88% |

At 500 memories, file-based exceeds 32K context windows. At 5,000, it doesn't fit in any context window — not even 200K. YantrikDB stays at ~70 tokens per query. Precision *improves* with more data — the opposite of context stuffing.

## Architecture

### Design Principles

- **Embedded, not client-server** — single file, no server process (like SQLite)
- **Local-first, sync-native** — works offline, syncs when connected
- **Cognitive operations, not SQL** — `record()`, `recall()`, `relate()`, not `SELECT`
- **Living system, not passive store** — does work between conversations
- **Thread-safe** — `Send + Sync` with internal Mutex/RwLock, safe for concurrent access

### Five Indexes, One Engine

```
┌──────────────────────────────────────────────────────┐
│                   YantrikDB Engine                    │
│                                                      │
│  ┌──────────┬──────────┬──────────┬──────────┐       │
│  │  Vector  │  Graph   │ Temporal │  Decay   │       │
│  │  (HNSW)  │(Entities)│ (Events) │  (Heap)  │       │
│  └──────────┴──────────┴──────────┴──────────┘       │
│  ┌──────────┐                                        │
│  │ Key-Value│  WAL + Replication Log (CRDT)          │
│  └──────────┘                                        │
└──────────────────────────────────────────────────────┘
```

1. **Vector Index (HNSW)** — semantic similarity search across memories
2. **Graph Index** — entity relationships, profile aggregation, bridge detection
3. **Temporal Index** — time-aware queries ("what happened Tuesday", "upcoming deadlines")
4. **Decay Heap** — importance scores that degrade over time, like human memory
5. **Key-Value Store** — fast facts, session state, scoring weights

### Memory Types (Tulving's Taxonomy)

| Type | What it stores | Example |
|------|---------------|---------|
| **Semantic** | Facts, knowledge | "User is a software engineer at Meta" |
| **Episodic** | Events with context | "Had a rough day at work on Feb 20" |
| **Procedural** | Strategies, what worked | "Deploy with blue-green, not rolling update" |

All memories carry **importance**, **valence** (emotional tone), **domain**, **source**, **certainty**, and **timestamps** — used in a multi-signal scoring function that goes far beyond cosine similarity.

## Key Capabilities

### Relevance-Conditioned Scoring

Not just vector similarity. Every recall combines:

- **Semantic similarity** (HNSW) — what's topically related
- **Temporal decay** — recent memories score higher
- **Importance weighting** — critical decisions beat trivia
- **Graph proximity** — entity relationships boost connected memories
- **Retrieval feedback** — learns from past recall quality

Weights are tuned automatically from usage patterns.

### Conflict Detection & Resolution

When memories contradict, YantrikDB doesn't guess — it creates a conflict segment:

```
"works at Google" (recorded Jan 15) vs. "works at Meta" (recorded Mar 1)
→ Conflict: identity_fact, priority: high, strategy: ask_user
```

Resolution is conversational: the AI asks naturally, not programmatically.

### Semantic Consolidation

After many conversations, memories pile up. `think()` runs:

1. **Consolidation** — merge similar memories, extract patterns
2. **Conflict scan** — find contradictions across the knowledge base
3. **Pattern mining** — cross-domain discovery ("work stress correlates with health entries")
4. **Trigger evaluation** — proactive insights worth surfacing

### Proactive Triggers

The engine generates triggers when it detects something worth reaching out about:

- Memory conflicts needing resolution
- Approaching deadlines (temporal awareness)
- Patterns detected across domains
- High-importance memories about to decay
- Goal tracking ("how's the marathon training?")

Every trigger is grounded in real memory data — not engagement farming.

### Multi-Device Sync (CRDT)

Local-first with append-only replication log:

- **CRDT merging** — graph edges, memories, and metadata merge without conflicts
- **Vector indexes rebuild locally** — raw memories sync, each device rebuilds HNSW
- **Forget propagation** — tombstones ensure forgotten memories stay forgotten
- **Conflict detection** — contradictions across devices are flagged for resolution

### Sessions & Temporal Awareness

```python
sid = db.session_start("default", "claude-code")
db.record("decided to use PostgreSQL")  # auto-linked to session
db.record("Alice suggested Redis for caching")
db.session_end(sid)
# → computes: memory_count, avg_valence, topics, duration

db.stale(days=14)    # high-importance memories not accessed recently
db.upcoming(days=7)  # memories with approaching deadlines
```

## Full API

| Operation | Methods |
|-----------|---------|
| **Core** | `record`, `record_batch`, `recall`, `recall_with_response`, `recall_refine`, `forget`, `correct` |
| **Knowledge Graph** | `relate`, `get_edges`, `search_entities`, `entity_profile`, `relationship_depth`, `link_memory_entity` |
| **Cognition** | `think`, `get_patterns`, `scan_conflicts`, `resolve_conflict`, `derive_personality` |
| **Triggers** | `get_pending_triggers`, `acknowledge_trigger`, `deliver_trigger`, `act_on_trigger`, `dismiss_trigger` |
| **Sessions** | `session_start`, `session_end`, `session_history`, `active_session`, `session_abandon_stale` |
| **Temporal** | `stale`, `upcoming` |
| **Procedural** | `record_procedural`, `surface_procedural`, `reinforce_procedural` |
| **Lifecycle** | `archive`, `hydrate`, `decay`, `evict`, `list_memories`, `stats` |
| **Sync** | `extract_ops_since`, `apply_ops`, `get_peer_watermark`, `set_peer_watermark` |
| **Maintenance** | `rebuild_vec_index`, `rebuild_graph_index`, `learned_weights` |

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Core language** | Rust | Memory safety, no GC, ideal for embedded engines |
| **Architecture** | Embedded (like SQLite) | No server overhead, sub-ms reads, single-tenant |
| **Bindings** | Python (PyO3), TypeScript | Agent/AI layer integration |
| **Storage** | Single file per user | Portable, backupable, no infrastructure |
| **Sync** | CRDTs + append-only log | Conflict-free for most operations, deterministic |
| **Thread safety** | Mutex/RwLock, Send+Sync | Safe concurrent access from multiple threads |
| **Query interface** | Cognitive operations API | Not SQL — designed for how agents think |

## Ecosystem

| Package | What | Install |
|---------|------|---------|
| [yantrikdb](https://crates.io/crates/yantrikdb) | Rust engine | `cargo add yantrikdb` |
| [yantrikdb](https://pypi.org/project/yantrikdb/) | Python bindings (PyO3) | `pip install yantrikdb` |
| [yantrikdb-mcp](https://pypi.org/project/yantrikdb-mcp/) | MCP server for AI agents | `pip install yantrikdb-mcp` |

## Roadmap

- [x] **V0** — Embedded engine, core memory model (record, recall, relate, consolidate, decay)
- [x] **V1** — Replication log, CRDT-based sync between devices
- [x] **V2** — Conflict resolution with human-in-the-loop
- [x] **V3** — Proactive cognition loop, pattern detection, trigger system
- [x] **V4** — Sessions, temporal awareness, cross-domain pattern mining, entity profiles
- [ ] **V5** — Multi-agent shared memory, federated learning across users

## Research & Publications

- **U.S. Patent Application 19/573,392** (March 2026): "Cognitive Memory Database System with Relevance-Conditioned Scoring and Autonomous Knowledge Management"
- **Zenodo:** [YantrikDB: A Cognitive Memory Engine for Persistent AI Systems](https://zenodo.org/records/14933693)

## Author

**Pranab Sarkar** — [ORCID](https://orcid.org/0009-0009-8683-1481) · [LinkedIn](https://www.linkedin.com/in/pranab-sarkar-b0511160/) · developer@pranab.co.in

## License

AGPL-3.0. See [LICENSE](LICENSE) for the full text.

The [MCP server](https://github.com/yantrikos/yantrikdb-mcp) is MIT-licensed — using the engine via the MCP server does not trigger AGPL obligations on your code.
