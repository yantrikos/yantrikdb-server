#![no_main]
//! Fuzz target for yantrikdb_protocol frame parsing.
//!
//! Feeds arbitrary bytes into the frame decoder. Must never panic —
//! all malformed input should produce a clean error.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try to parse as a length-prefixed frame
    if data.len() >= 4 {
        // unpack_frame expects: [4-byte length][opcode][stream_id][payload]
        let _ = yantrikdb_protocol::unpack_frame(data);
    }

    // Also try to deserialize as MessagePack into common message types
    let _ = yantrikdb_protocol::unpack::<yantrikdb_protocol::messages::ClusterHello>(data);
    let _ = yantrikdb_protocol::unpack::<yantrikdb_protocol::messages::OplogPullRequest>(data);
    let _ = yantrikdb_protocol::unpack::<yantrikdb_protocol::messages::HeartbeatMsg>(data);
});
