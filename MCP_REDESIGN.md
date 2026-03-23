# YantrikDB MCP Server Redesign — Session Brief

## Goal
Build a production-quality MCP server that any AI agent (Claude, GPT, Cursor, etc.) would love to use as persistent cognitive memory. This is the #1 adoption driver — if the MCP experience is great, people will install it from the HN post on Tuesday.

## Current State

### What exists
- `src/yantrikdb/mcp/` — Python MCP server using FastMCP
  - `server.py` — lifespan context, YantrikDB + sentence-transformers init
  - `tools.py` — 10 tools: remember, recall, relate, entities, beliefs, conflicts, patterns, consolidate, forget, stats
  - `resources.py` — MCP resources
- Install: `pip install yantrikdb[mcp]` then `yantrikdb-mcp` command
- Config: env vars `YANTRIKDB_DB_PATH`, `YANTRIKDB_EMBEDDING_MODEL`, `YANTRIKDB_EMBEDDING_DIM`

### What works
- Basic remember/recall/relate flow
- Entity graph operations
- Consolidation, conflict detection, pattern mining via `think()`
- Runs as stdio MCP server

### What needs improvement

1. **Tool descriptions are too terse** — agents don't know *when* to call each tool. Need rich descriptions with examples that guide auto-pilot behavior (like the instructions block in Claude Code's yantrikdb server config).

2. **No server instructions** — MCP servers can provide instructions that get injected into the agent's system prompt. This is where we tell the agent "auto-recall at conversation start, auto-remember when you learn something important." Currently this is done via manual config in `.claude/mcp.json` — should be built into the server.

3. **Missing tools**:
   - `think` — run full cognition loop (consolidation + conflicts + patterns + triggers)
   - `correct` — user correction flow (tombstone old memory, create corrected one)
   - `search_entities` — search entities by name pattern
   - `get_memory` — get a specific memory by ID
   - `update_importance` — adjust importance of existing memory
   - `bulk_remember` — store multiple memories at once (efficient for conversation summaries)

4. **recall is too basic** — needs filters: by domain, source, memory_type, time_window. The Rust engine supports all of these via RecallQuery builder but the MCP tool doesn't expose them.

5. **No streaming** — for large recall results, should support pagination or streaming.

6. **Startup is slow** — sentence-transformers model download on first run. Need a progress indicator or pre-download step.

7. **No health check** — agents can't verify the server is working before relying on it.

## Architecture Decisions

### Embedding model
- Current: `all-MiniLM-L6-v2` (384 dims, ~80MB, good quality)
- Consider: making it configurable, supporting OpenAI embeddings as alternative
- First-run experience matters — downloading 80MB silently is bad UX

### Database location
- Current: `~/.yantrikdb/memory.db`
- Should support per-project databases (workspace-scoped memory)
- Consider: `YANTRIKDB_DB_PATH` env var already works, but add auto-detection of project root

### Server instructions format
The MCP spec allows servers to provide instructions. These should include:
- Auto-recall behavior (when to search memory)
- Auto-remember behavior (what to store)
- Auto-relate behavior (when to create entity links)
- What NOT to remember (ephemeral stuff, code, git history)

## Key Files

```
src/yantrikdb/mcp/
├── __init__.py      — main() entry point
├── server.py        — FastMCP server + lifespan
├── tools.py         — tool definitions
└── resources.py     — MCP resources

src/yantrikdb/
├── __init__.py      — YantrikDB class (Rust bindings)
├── consolidate.py   — consolidation logic
├── triggers.py      — trigger evaluation
├── cli.py           — CLI commands
└── api.py           — REST API server
```

## Rust Engine Capabilities (available but not exposed via MCP)
- `RecallQuery` builder: top_k, memory_type, namespace, time_window, domain, source, expand_entities
- `think()` with full ThinkConfig
- Conflict resolution with strategies (keep_a, keep_b, merge, ask_user)
- Pattern mining with configurable thresholds
- Personality profile extraction
- Spaced repetition reinforcement on access
- Batch record operations
- Replication/sync (CRDT-based)

## Success Criteria
1. `pip install yantrikdb[mcp]` → add 3 lines to mcp.json → agent has persistent memory
2. Agent auto-recalls relevant context at conversation start without being told
3. Agent auto-remembers decisions, preferences, corrections without being told
4. Agent detects and surfaces contradictions naturally
5. Works with Claude Code, Cursor, Windsurf, and any MCP client
6. First-run takes < 30 seconds (model download + DB init)
7. Memory persists across sessions, consolidates over time
8. The agent experience is noticeably better than without YantrikDB

## Reference: Current Server Instructions (from .claude/mcp.json)
These work well and should be the basis for built-in server instructions:

```
YantrikDB is your persistent cognitive memory. Use it AUTOMATICALLY.

## Auto-recall (before responding)
- At conversation start, call `recall` with a summary of the user's first message
- When the user references past work, decisions, people, or preferences, call `recall`

## Auto-remember (during conversation)
Proactively call `remember` whenever you encounter:
- Decisions made, user preferences, people & relationships, project context, corrections, important facts

## Auto-relate (knowledge graph)
Call `relate` when you learn about entity relationships

## What NOT to remember
- Ephemeral task details, things derivable from code/git, verbatim code

## Memory quality
- Use specific, searchable text
- Set importance: 0.8-1.0 critical, 0.5-0.7 useful, 0.3-0.5 minor
- Set domain: "work", "preference", "architecture", "people", "infrastructure"
```
