# yantrikdb-protocol

Wire protocol codec for [YantrikDB](https://github.com/yantrikos/yantrikdb-server) — a cognitive memory database.

Provides binary frames, opcodes, MessagePack message types, and a Tokio codec for client/server and peer-to-peer communication.

## Features

- **Binary frame format** — length-prefixed, versioned, multiplexed streams
- **50+ opcodes** — memory ops, graph ops, sessions, cluster/replication
- **Typed messages** — serde-friendly request/response structs for every command
- **Tokio codec** — drop-in `Encoder`/`Decoder` for async streams
- **Optional compression** — zstd flag bit for large payloads (oplog batches, recall results)

## Example

```rust
use yantrikdb_protocol::{Frame, OpCode, make_frame, unpack_frame};
use yantrikdb_protocol::messages::RememberRequest;

// Build a frame
let req = RememberRequest {
    text: "Alice leads engineering".into(),
    importance: 0.9,
    domain: "work".into(),
    ..Default::default()
};
let frame = make_frame(OpCode::Remember, 0, &req)?;

// Decode a frame
let parsed: RememberRequest = unpack_frame(&frame)?;
```

## License

AGPL-3.0-only — see [LICENSE](../../LICENSE).

The client SDKs (`yql`, Python SDK) are MIT-licensed.
