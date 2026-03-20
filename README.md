# YantrikDB — A Cognitive Memory Engine for Persistent AI Systems

> The memory engine for AI that actually knows you.

## The Problem

Current AI systems have no coherent memory architecture. They bolt together generic databases — vector stores, knowledge graphs, key-value caches — none of which were designed for how cognition works. This makes persistent, evolving AI relationships impossible at scale.

Today's AI memory is:

> Store everything → Embed → Retrieve top-k → Inject into context → Hope it helps.

That does not scale cognitively.

## The Thesis

AI needs a purpose-built memory engine with native support for:

- **Temporal decay** — memories age and fade like human memory
- **Semantic consolidation** — patterns are extracted, redundancy is compressed
- **Conflict resolution** — contradictions are detected and resolved conversationally
- **Multi-device replication** — local-first CRDT-based sync across devices
- **Proactive cognition** — background processing that gives AI genuine reasons to initiate conversation

All in a **single embedded engine** — no server, no network hops, no stitching together five databases.

## Why Not Use Existing Solutions?

| Solution | What it does | What it lacks |
|----------|-------------|---------------|
| **Vector DBs** (Pinecone, Weaviate, Milvus) | High-dimensional nearest-neighbor lookup | No time awareness, no causality, no compression, no self-organization |
| **Knowledge Graphs** (Neo4j) | Structured relations, entity linking | Hard to scale dynamically, poor for fuzzy memory, not adaptive |
| **Memory Frameworks** (LangChain, LlamaIndex) | Retrieval wrappers, context injection | Not true memory architectures — just middleware |

Human memory is hierarchical, compressed, contextual, self-updating, emotionally weighted, time-aware, and predictive. No existing system addresses this holistically.

## Architecture

### Design Principles

- **Embedded, not client-server** — single file, no server process (like SQLite)
- **Local-first, sync-native** — works offline, syncs when connected
- **Cognitive operations, not SQL** — `record()`, `recall()`, `relate()`, not `SELECT`
- **Living system, not passive store** — does work between conversations

### Unified Index Architecture

Five index types in one engine, sharing the same memory pages, WAL, and query planner:

```
┌─────────────────────────────────────────────────────┐
│                  YantrikDB Engine                         │
│                                                     │
│  ┌───────────┬───────────┬───────────┬───────────┐ │
│  │  Vector   │  Graph    │ Temporal  │   Decay   │ │
│  │  Index    │  Index    │  Index    │   Heap    │ │
│  │  (HNSW)  │ (Entities)│ (Events)  │(Priority) │ │
│  └───────────┴───────────┴───────────┴───────────┘ │
│  ┌───────────┐                                      │
│  │ Key-Value │                                      │
│  │  Store    │                                      │
│  └───────────┘                                      │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │         Write-Ahead Log (WAL)                 │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │      Replication Log (append-only)            │  │
│  │      CRDT-based conflict resolution           │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

1. **Vector Index (HNSW)** — semantic similarity search across memories
2. **Graph Index** — entity relationships ("Max is user's dog", "user works at Meta")
3. **Temporal Index** — time-series style, "what happened around Tuesday"
4. **Decay Heap** — priority queue with importance scores that degrade over time
5. **Key-Value Store** — fast facts ("user's name is Pranab")

### Memory Types

Inspired by cognitive science (Tulving's taxonomy):

| Type | What it stores | Example |
|------|---------------|---------|
| **Episodic** | Events, experiences with context | "User had a rough day at work on Feb 20" |
| **Semantic** | Facts, knowledge, abstractions | "User is a software engineer who likes AI" |
| **Procedural** | Strategies, behaviors, what worked | "User prefers concise answers with code examples" |
| **Emotional** | Valence weighting on memories | "Dog's death → high emotional weight → never forget" |

### Core Operations

```
yantrikdb.record(memory, importance=0.8, emotion="frustrated")
yantrikdb.recall("What does the user feel about their job?")
yantrikdb.relate("user.job", "user.stress", strength=0.7)
yantrikdb.consolidate(topic="user.career", since="30d")
yantrikdb.decay(threshold=0.1)       // prune low-importance memories
yantrikdb.forget(memory_id)          // explicit removal
yantrikdb.conflict(memory_a, memory_b)  // flag contradiction
yantrikdb.resolve(conflict_id, resolution)  // user-driven resolution
```

## Conflict Resolution — Human-in-the-Loop

When synced devices produce contradictory memories, YantrikDB doesn't guess. It creates a **conflict segment** — a first-class data structure:

```
┌──────────────────────────────────────────┐
│            Conflict Segment              │
│                                          │
│  conflict_id:  c_0042                    │
│  type:         identity_fact             │
│  priority:     high                      │
│  memory_a:     "works at Google" (phone) │
│  memory_b:     "works at Meta" (laptop)  │
│  status:       pending_resolution        │
│  strategy:     ask_user                  │
│  resolved_by:  null                      │
│  resolution:   null                      │
└──────────────────────────────────────────┘
```

Resolution happens **conversationally**, not programmatically:

> "Oh by the way — last month you mentioned something about Meta. Did you end up switching from Google?"

Conflicts are triaged by priority:

| Conflict Type | Action |
|---------------|--------|
| Critical identity facts | Ask immediately |
| Preferences that changed | Ask naturally in conversation |
| Minor contradictions | Keep both, resolve lazily |
| Temporal conflicts | Prefer most recent, flag if uncertain |

## Multi-Device Sync Protocol

YantrikDB is **local-first** with CRDT-based replication:

```
┌──────────────────────┐       ┌──────────────────────┐
│   Device A (Phone)   │       │  Device B (Laptop)   │
│                      │       │                      │
│  ┌────────────────┐  │ sync  │  ┌────────────────┐  │
│  │ YantrikDB Engine │◄─┼───────┼─►│ YantrikDB Engine │  │
│  └────────────────┘  │       │  └────────────────┘  │
│  ┌────────────────┐  │       │  ┌────────────────┐  │
│  │ Replication    │  │       │  │ Replication    │  │
│  │ Log            │  │       │  │ Log            │  │
│  └────────────────┘  │       │  └────────────────┘  │
└──────────────────────┘       └──────────────────────┘
         │                              │
         └──────────┬───────────────────┘
                    │
            P2P / Relay / BLE
        (encrypted, zero-knowledge)
```

- **Append-only replication log** — every write, consolidation, and decay event is logged
- **CRDT merging** — graph edges/nodes and facts merge without conflicts
- **Vector indexes rebuild locally** — raw memories sync, each device rebuilds HNSW
- **Forget propagation** — tombstones ensure forgotten memories stay forgotten
- **Optional cloud relay** — dumb encrypted pipe, not a server. Sees nothing.

### Storage Tiers

| Tier | Backing | Use case |
|------|---------|----------|
| **Hot** | In-memory | Recent/frequent memories, active conversation |
| **Warm** | SSD-backed | Medium-term, weeks to months |
| **Cold** | Compressed archival | Old memories, on-demand hydration |

## Proactive Cognition Loop

YantrikDB runs a **background processing loop** even between conversations — giving AI genuine reasons to reach out:

```
┌─────────────────────────────────────────────────┐
│           Proactive Trigger System               │
│                                                  │
│  Memory Conflicts    → "You mentioned two        │
│  (need resolution)     different moving dates"   │
│                                                  │
│  Pattern Detection   → "You seem stressed        │
│  (noticed something)   every Sunday evening"     │
│                                                  │
│  Temporal Triggers   → "Your mom's birthday      │
│  (time-based)          is tomorrow"              │
│                                                  │
│  Decay Warnings      → "I'm fuzzy on your        │
│  (about to forget)     new coworker's name"      │
│                                                  │
│  Goal Tracking       → "How's the marathon       │
│  (user set a goal)     training going?"          │
│                                                  │
│  Consolidation       → "I noticed you always     │
│  Insights              feel better after talking  │
│                        to your sister"            │
└─────────────────────────────────────────────────┘
```

Every proactive message is **grounded in real memory data** — not engagement farming.

Built-in safety constraints:

| Rule | Purpose |
|------|---------|
| Cooldown periods | No messaging every hour |
| Priority threshold | Only reach out when it matters |
| Time-of-day awareness | Don't message at 3am |
| User-controlled frequency | "Check in weekly" vs "only urgent" |
| Groundedness requirement | Every message must trace to real memories |

### Background Processing Cycle

1. **Consolidation pass** — compress, summarize, abstract
2. **Conflict detection** — find contradictions across synced devices
3. **Pattern mining** — "user tends to X when Y"
4. **Trigger evaluation** — "is anything worth reaching out about?"
5. **Decay pass** — age out low-importance memories

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Architecture** | Embedded (like SQLite) | No server overhead, sub-ms local reads, single-tenant |
| **Core language** | Rust | Memory safety without GC pauses, ideal for embedded engines |
| **Bindings** | Python, TypeScript | Agent/AI layer integration |
| **Storage format** | Single file per user | Portable, backupable, no infrastructure |
| **Sync** | CRDTs + append-only log | Conflict-free for most operations, deterministic |
| **Query interface** | Cognitive operations API | Not SQL — designed for how agents think |

## Target Use Cases

- **AI Companions** — persistent, evolving relationships across devices
- **Autonomous Agents** — long-horizon planning with memory consolidation
- **Multi-Agent Systems** — shared memory between cooperating agents
- **Personal AI Assistants** — that actually remember and grow with you

## Roadmap

- [x] **V0** — Single device, embedded engine, core memory model (record, recall, relate, consolidate, decay)
- [x] **V1** — Replication log, sync between two devices
- [x] **V2** — Conflict resolution with human-in-the-loop, production-grade sync
- [x] **V3** — Proactive cognition loop, pattern detection, trigger system
- [ ] **V4** — Multi-agent shared memory, federated learning across users

## Research & Publications

- **U.S. Patent Application 19/573,392** (March 2026): "Cognitive Memory Database System with Relevance-Conditioned Scoring and Autonomous Knowledge Management"
- **Zenodo:** [YantrikDB: A Cognitive Memory Engine for Persistent AI Systems](https://zenodo.org/records/14933693)
- **Related work by the author:** ["Convert Once, Consume Many: SDF for Cacheable, Typed Semantic Extraction from Web Pages"](https://zenodo.org/records/18559223) — solving efficient data ingestion for AI agents (the upstream problem to memory)

## Author

**Pranab Sarkar**
- ORCID: [0009-0009-8683-1481](https://orcid.org/0009-0009-8683-1481)
- LinkedIn: [pranab-sarkar-b0511160](https://www.linkedin.com/in/pranab-sarkar-b0511160/)
- Email: developer@pranab.co.in

## Patent

YantrikDB's cognitive memory methods are covered by U.S. Patent Application No. 19/573,392 (filed March 20, 2026), claiming priority to Provisional Application No. 63/991,357 (filed February 26, 2026).

## License

Copyright (c) 2026 Pranab Sarkar

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

See [LICENSE](LICENSE) for the full text.
