# YantrikDB Server — Design Spec

> Output of 3-round brainstorm between GPT-5.4, DeepSeek, and Claude (April 5, 2026)

## What This Is

A multi-tenant cognitive memory database server with a native wire protocol. One Rust binary. Zero dependencies. Built-in embeddings. Explainable recall. Session-aware. Conflict-detecting. Memory that decays, consolidates, and learns.

## What This Is NOT

- Not a vector database (it's a cognitive memory database)
- Not a generic key-value store
- Not an LLM (it doesn't generate text, it remembers and recalls)

## The One-Liner

**"YantrikDB: The memory database for AI. Remember, recall, forget — like a human mind, not a filing cabinet."**

## Positioning

Pinecone/Qdrant/Weaviate are vector indexes. YantrikDB is the memory subsystem for AI agents and personalized systems.

No other database does ALL of these:
1. Memories decay over time (exponential half-life, reinforcement on access)
2. Contradictions are detected automatically
3. Recall explains itself (not just a score — WHY this memory matters now)
4. Sessions are first-class (conversation context affects what's recalled)
5. Knowledge graph is fused into recall scoring
6. Consolidation happens autonomously (similar memories merge)
7. Personality emerges from memory patterns
8. CRDT replication preserves cognitive state (not just rows — memory events with provenance)

---

## Architecture: Wire Protocol + HTTP Gateway

```
                          ┌───────────────────────────┐
                          │     YantrikDB Server       │
                          │     (single Rust binary)   │
                          │                           │
  ┌─────────────┐        │  ┌───────────────────────┐ │
  │ Native SDK  │◄──TCP──┤  │   Wire Protocol       │ │
  │ (Python/    │  TLS   │  │   (binary frames)     │ │
  │  Rust/JS)   │        │  │   Port 7437           │ │
  └─────────────┘        │  └───────────┬───────────┘ │
                          │              │ Command     │
  ┌─────────────┐        │  ┌───────────▼───────────┐ │
  │ HTTP Client │◄─HTTP──┤  │   HTTP Gateway        │ │
  │ (curl/MCP/  │  /2    │  │   (thin translation)  │ │
  │  browser)   │        │  │   Port 7438           │ │
  └─────────────┘        │  └───────────┬───────────┘ │
                          │              │             │
                          │  ┌───────────▼───────────┐ │
                          │  │   Command Router      │ │
                          │  │   (enum Command)      │ │
                          │  └───────────┬───────────┘ │
                          │              │             │
                          │  ┌───────────▼───────────┐ │
                          │  │   Tenant Router       │ │
                          │  │   (token → database)  │ │
                          │  └───────────┬───────────┘ │
                          │              │             │
                          │  ┌───────────▼───────────┐ │
                          │  │   Engine Pool         │ │
                          │  │   (cached YantrikDB   │ │
                          │  │    instances per DB)   │ │
                          │  └───────────┬───────────┘ │
                          │              │             │
                          │  ┌───────────▼───────────┐ │
                          │  │   Background Workers  │ │
                          │  │   - Consolidation     │ │
                          │  │   - Decay sweep       │ │
                          │  │   - Trigger eval      │ │
                          │  │   - Personality       │ │
                          │  └───────────────────────┘ │
                          └───────────────────────────┘
```

---

## Wire Protocol Design

### Principles
- Binary, length-prefixed frames
- Multiplexed streams over one connection
- Bidirectional: server can push to client
- Session-aware: open session on connection, auto-link memories
- Designed for chatty agent workloads (5-20 ops per conversation turn)

### Frame Format
```
┌──────────┬──────────┬──────────┬──────────┬────────────┐
│ Length    │ Version  │ OpCode   │ StreamID │ Payload    │
│ (4 bytes)│ (1 byte) │ (1 byte) │ (4 bytes)│ (variable) │
└──────────┴──────────┴──────────┴──────────┴────────────┘
```

### OpCodes
```
0x01  AUTH            Client → Server    Authenticate with token
0x02  AUTH_OK         Server → Client    Authentication successful
0x03  AUTH_FAIL       Server → Client    Authentication failed

0x10  SELECT_DB       Client → Server    Select database
0x11  CREATE_DB       Client → Server    Create database
0x12  DB_OK           Server → Client    Database operation success

0x20  REMEMBER        Client → Server    Store a memory
0x21  REMEMBER_OK     Server → Client    Memory stored (returns RID)
0x22  REMEMBER_BATCH  Client → Server    Store multiple memories

0x30  RECALL          Client → Server    Semantic recall query
0x31  RECALL_RESULT   Server → Client    Single recall result (streamed)
0x32  RECALL_END      Server → Client    End of recall results

0x40  RELATE          Client → Server    Create graph edge
0x41  RELATE_OK       Server → Client    Edge created
0x42  EDGES           Client → Server    Get edges for entity
0x43  EDGES_RESULT    Server → Client    Edge results

0x50  FORGET          Client → Server    Tombstone a memory
0x51  FORGET_OK       Server → Client    Memory forgotten

0x60  SESSION_START   Client → Server    Open cognitive session
0x61  SESSION_END     Client → Server    Close cognitive session
0x62  SESSION_OK      Server → Client    Session operation success

0x70  THINK           Client → Server    Trigger consolidation
0x71  THINK_RESULT    Server → Client    Consolidation results

0x80  SUBSCRIBE       Client → Server    Subscribe to events
0x81  EVENT           Server → Client    Pushed event (trigger/conflict/decay)
0x82  UNSUBSCRIBE     Client → Server    Unsubscribe

0x90  CONFLICTS       Client → Server    List conflicts
0x91  RESOLVE         Client → Server    Resolve a conflict
0x92  CONFLICT_RESULT Server → Client    Conflict data

0xA0  PERSONALITY     Client → Server    Get personality traits
0xA1  STATS           Client → Server    Get engine stats
0xA2  INFO_RESULT     Server → Client    Info response

0xF0  ERROR           Server → Client    Error response
0xF1  PING            Either direction   Keepalive
0xF2  PONG            Either direction   Keepalive response
```

### Connection Lifecycle
```
Client                                    Server
  │                                         │
  ├── AUTH(token="ydb_abc123") ──────────►  │
  │                                         ├── validate token
  │  ◄──────────────── AUTH_OK(db="default")│
  │                                         │
  ├── SESSION_START(agent="bot-1") ──────►  │
  │  ◄──────── SESSION_OK(sid="sess_xyz")   │
  │                                         │
  ├── REMEMBER(text="User likes dark") ──►  │
  │  ◄──────── REMEMBER_OK(rid="mem_01")    │
  │                                         │
  ├── RECALL(query="user prefs") ────────►  │
  │  ◄──────── RECALL_RESULT(mem_01, 0.93)  │
  │  ◄──────── RECALL_RESULT(mem_07, 0.71)  │
  │  ◄──────── RECALL_END(total=2)          │
  │                                         │
  ├── SUBSCRIBE(events=["conflict"]) ────►  │
  │                                         │
  │  ... time passes ...                    │
  │                                         │
  │  ◄──────── EVENT(conflict_detected)     │  ← server push!
  │                                         │
  ├── SESSION_END(sid="sess_xyz") ───────►  │
  │  ◄──────── SESSION_OK(summary=...)      │
  │                                         │
```

### Payload Encoding
- MessagePack for compact binary (serde compatible, smaller than JSON, faster than protobuf for this use case)
- Optional: JSON payload mode for debugging (`Version` byte flag)

---

## Multi-Tenancy

### Model
```
control.db (SQLite)
├── databases (id, name, path, config, created_at)
├── tokens (hash_sha256, database_id, label, created_at, revoked_at)
└── server_config (key, value)

data/
├── default/
│   └── yantrik.db          (isolated YantrikDB instance)
├── customer-memory/
│   └── yantrik.db
├── worldstage/
│   └── yantrik.db
└── ...
```

### Auth
Phase 1: `ydb_<random_64_hex>` → maps to one database, full access.

### Database Lifecycle
```
CREATE DATABASE  → new directory + SQLite file + indexes
DROP DATABASE    → tombstone (soft delete, data retained N days)
LIST DATABASES   → from control.db
```

---

## Embedding Strategy

### Default: Built-in (zero config)
- all-MiniLM-L6-v2 (384 dim) via candle
- Ships with the binary (model weights downloaded on first run)
- No external API needed

### Optional: Remote (OpenAI-compatible)
```toml
[embedding]
strategy = "remote"
url = "https://api.openai.com/v1/embeddings"
model = "text-embedding-3-small"
api_key_env = "OPENAI_API_KEY"
```

### Optional: Client-provided vectors
Client sends `embedding: [0.1, 0.2, ...]` directly. Server skips embedding step.

---

## SDK Design

### Python (`pip install yantrikdb`)
```python
from yantrikdb import connect

db = connect("yantrik://localhost:7437", token="ydb_...")

db.remember("Alice leads engineering", importance=0.9)
results = db.recall("who leads engineering?", explain=True)

with db.session("chat-42") as s:
    s.remember("User asked about pricing")
    results = s.recall("what's the user asking about?")

db.relate("Alice", "Engineering", "leads")
report = db.think()

for event in db.subscribe(["conflict", "decay"]):
    print(event)
```

### Rust (`cargo add yantrikdb`)
```rust
use yantrikdb::Client;

let db = Client::connect("yantrik://localhost:7437", "ydb_...")?;

db.remember("Alice leads engineering").importance(0.9).send().await?;
let results = db.recall("who leads engineering?").explain(true).send().await?;

let session = db.session("chat-42").start().await?;
session.remember("User asked about pricing").send().await?;
let results = session.recall("what's the user asking about?").send().await?;
session.end().await?;
```

### Connection URL scheme
```
yantrik://localhost:7437           (wire protocol, default)
yantrik+tls://host:7437           (wire protocol + TLS)
http://localhost:7438              (HTTP gateway fallback)
```

---

## HTTP Gateway (Port 7438)

Thin translation layer: JSON ↔ wire protocol commands.

```
POST /v1/remember              → REMEMBER command
POST /v1/recall                → RECALL command
POST /v1/relate                → RELATE command
POST /v1/forget                → FORGET command
POST /v1/think                 → THINK command
GET  /v1/conflicts             → CONFLICTS command
POST /v1/conflicts/:id/resolve → RESOLVE command
GET  /v1/triggers              → reads trigger state
POST /v1/sessions              → SESSION_START
DELETE /v1/sessions/:id        → SESSION_END
POST /v1/databases             → CREATE_DB
GET  /v1/health                → PING/PONG + stats
GET  /v1/events                → SSE stream (translates EVENT pushes)
GET  /v1/personality           → PERSONALITY command
```

---

## Server Configuration

```toml
# yantrikdb.toml (optional — env vars also work)
[server]
wire_port = 7437
http_port = 7438
data_dir = "./data"

[embedding]
strategy = "builtin"  # "builtin" | "remote" | "client_only"

[background]
consolidation_interval_minutes = 30
decay_sweep_interval_minutes = 60
trigger_check_interval_minutes = 5

[limits]
max_databases = 100
max_memories_per_db = 1_000_000
max_connections = 1000
```

---

## CLI

```bash
yantrikdb serve                          # Start server
yantrikdb serve --config ./yantrikdb.toml
yantrikdb db create <name>               # Create database
yantrikdb db list                        # List databases
yantrikdb token create --db <name>       # Generate token
yantrikdb export <db> > backup.jsonl     # Export
yantrikdb import <db> < backup.jsonl     # Import
```

---

## Crate / Module Layout

```
yantrikdb-server/
├── Cargo.toml (workspace)
├── crates/
│   ├── yantrikdb-core/          # Existing: engine, recall, graph, HNSW, CRDT
│   ├── yantrikdb-protocol/      # NEW: wire protocol frames, opcodes, codec
│   │   ├── src/
│   │   │   ├── frame.rs         # Frame encoding/decoding
│   │   │   ├── opcodes.rs       # OpCode enum
│   │   │   ├── messages.rs      # Typed message structs
│   │   │   ├── codec.rs         # Tokio codec (Encoder + Decoder)
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   ├── yantrikdb-server/        # NEW: server binary
│   │   ├── src/
│   │   │   ├── main.rs          # CLI (clap)
│   │   │   ├── server.rs        # Wire protocol listener (tokio)
│   │   │   ├── http_gateway.rs  # Axum HTTP gateway
│   │   │   ├── auth.rs          # Token validation
│   │   │   ├── control.rs       # control.db management
│   │   │   ├── tenant_pool.rs   # Lazy-load engine instances
│   │   │   ├── session_mgr.rs   # Cognitive session management
│   │   │   ├── command.rs       # Internal Command enum
│   │   │   ├── handler.rs       # Command → engine execution
│   │   │   ├── embedder.rs      # Builtin + remote embedding
│   │   │   ├── background.rs    # Consolidation, decay, triggers
│   │   │   ├── push.rs          # Server → client event push
│   │   │   └── config.rs        # Config loading
│   │   └── Cargo.toml
│   ├── yantrikdb-client-rs/     # NEW: Rust client SDK
│   │   ├── src/
│   │   │   ├── client.rs        # Connection, auth, database selection
│   │   │   ├── memory.rs        # remember(), recall(), forget()
│   │   │   ├── graph.rs         # relate(), edges()
│   │   │   ├── session.rs       # session(), subscribe()
│   │   │   ├── builder.rs       # Request builders
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   └── yantrikdb-python/        # Existing: PyO3 bindings (extend for client)
├── sdk/
│   ├── python/                  # Python client SDK
│   │   ├── yantrikdb/
│   │   │   ├── __init__.py
│   │   │   ├── client.py        # connect(), Client class
│   │   │   ├── protocol.py      # Wire protocol codec (Python)
│   │   │   ├── session.py       # Session context manager
│   │   │   └── types.py         # Memory, RecallResult, etc.
│   │   └── pyproject.toml
│   └── typescript/              # Future: TS/JS SDK
└── tests/
    ├── integration/
    └── protocol/
```

---

## Phased Roadmap

### Phase 1: Working Server (target: build in one intense session)
- Wire protocol codec (frame encode/decode)
- TCP listener with tokio
- Auth + database selection
- REMEMBER / RECALL / RELATE / FORGET
- SESSION_START / SESSION_END
- SUBSCRIBE / EVENT push
- HTTP gateway (axum, translates to commands)
- control.db for databases + tokens
- Built-in embeddings
- CLI: serve, db create, token create
- Python SDK (basic)

### Phase 2: Production Polish
- TLS on wire protocol
- Background workers (consolidation, decay, triggers)
- THINK command
- CONFLICTS / RESOLVE
- PERSONALITY / STATS
- Export / Import
- Connection pooling + LRU tenant eviction
- Rust SDK
- Comprehensive tests

### Phase 3: Scale
- CRDT oplog replication between servers
- Read replicas
- Named embedding spaces per database
- Token scopes (read/write/admin)
- CQL query language
- Admin dashboard
- Cloud hosting

---

## License
- Server: AGPL-3.0
- Client SDKs: MIT / Apache-2.0
- This protects against cloud strip-mining while keeping SDKs freely embeddable.
