# yql

Interactive REPL client for [YantrikDB](https://github.com/yantrikos/yantrikdb-server) — like `psql` for cognitive memory.

## Install

```bash
cargo install yql
```

Or download a pre-built binary from [releases](https://github.com/yantrikos/yantrikdb-server/releases/latest).

## Usage

```bash
yql --host localhost -p 7438 -t ydb_your_token_here
```

```
yql connected to http://localhost:7438
type \h for help, \q to exit

yantrikdb> remember "Alice leads engineering at Acme" importance=0.9 domain=work
✓ stored: 019d623a-3d70-712e-9315-e1da5ee41114

yantrikdb> recall who leads engineering top=5
+---+-------+---------------------------------+--------+--------------------------------+
| # | score | text                            | domain | why                            |
+---+-------+---------------------------------+--------+--------------------------------+
| 1 | 1.41  | Alice leads engineering at Acme | work   | semantically similar (0.54)... |
+---+-------+---------------------------------+--------+--------------------------------+
(1 rows)

yantrikdb> relate Alice -> Acme as works_at
✓ edge: 019d623a-41cf-71a2 (Alice -[works_at]-> Acme)

yantrikdb> \stats
yantrikdb> \cluster
yantrikdb> \q
```

## Commands

### Memory operations (natural)
- `remember "text" [importance=0.9] [domain=work]` — store a memory
- `recall <query> [top=10] [domain=work]` — semantic search
- `forget <rid>` — tombstone a memory
- `relate <entity> -> <target> as <relationship>` — create graph edge

### Meta commands (psql-style)
- `\stats` `\s` — engine statistics
- `\dt` `\l` — list databases
- `\conflicts` `\c` — list open conflicts
- `\personality` `\p` — derived personality traits
- `\think` `\t` — run consolidation + conflict scan
- `\cluster` — cluster status (replication / failover)
- `\health` — server health
- `\json <path>` — raw GET request
- `\h` `\?` — help
- `\q` — quit

## Non-interactive mode

```bash
yql --host localhost -p 7438 -t $TOKEN -c '\stats'
yql --host localhost -p 7438 -t $TOKEN -c 'remember "Hello" importance=0.5'
```

## License

MIT — `yql` is free for any use, including commercial.

The server (`yantrikdb-server`) is AGPL-3.0.
