# yantrikdb-client

Python SDK for [YantrikDB](https://github.com/yantrikos/yantrikdb) — a
cognitive memory database with persistent typed memory, contradiction
handling, and reflection.

## Install

```bash
pip install yantrikdb-client
# with client-side auto-embedding
pip install 'yantrikdb-client[embed]'
```

## Quick start

```python
from yantrikdb import connect

client = connect("http://localhost:7438", token="ydb_...")

# Basic memory
client.remember("Alice prefers dark mode", domain="preference")
results = client.recall("what does Alice prefer?")

# Character-substrate primitives (v0.2.0+)
client.remember_self("I overtrust single-source reports under time pressure")
client.remember_rule(
    condition="single-source high-stakes claim",
    action="state uncertainty and request corroboration",
)
client.remember_constraint(
    label="truthfulness_over_pleasing",
    description="Disclose uncertainty even when unwelcome",
    priority=0.95,
)

# Reflect — compose a structured meta-state view for an LLM prompt
reflection = client.reflect(
    "How should I handle this high-stakes single-source claim?",
)
print(reflection.render())
```

## What's in 0.2.0

- **Character-substrate primitives**: `remember_self`, `remember_rule`,
  `remember_hypothesis`, `remember_constraint`, `remember_goal`,
  `remember_arc`, `record_signal`
- **Typed recall**: `recall_typed(query, memory_type)` for filtered
  retrieval
- **Reflect API**: `reflect(question)` composes parallel type-filtered
  recalls + open conflicts into a `Reflection` with `.render()` for
  LLM prompts
- **Auto-embedder**: pass `embedder="all-MiniLM-L6-v2"` to `connect()`
  (or keep default) and the client lazy-loads sentence-transformers

## API surface

- `connect(url, *, token, embedder=...)` — returns a `YantrikClient`
- `YantrikClient.remember(text, ...)` — store a memory
- `YantrikClient.recall(query, ...)` — semantic search
- `YantrikClient.relate(entity, target, relationship)` — knowledge graph edge
- `YantrikClient.think(...)` — trigger consolidation / conflict scan
- `YantrikClient.reflect(question, ...)` — structured meta-state view
- Typed helpers: `remember_self/rule/hypothesis/constraint/goal/arc`,
  `record_signal`, `recall_typed`
- `YantrikClient.session(...)` — context manager for cognitive sessions

## License

MIT
