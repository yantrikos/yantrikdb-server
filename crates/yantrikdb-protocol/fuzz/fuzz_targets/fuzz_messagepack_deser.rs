#![no_main]
//! Fuzz target for MessagePack deserialization of oplog wire entries.
//!
//! Feeds arbitrary bytes into OplogEntryWire deserialization. Must never
//! panic — malformed input should return a clean error. This exercises the
//! code path that handles replicated ops from peers, where a corrupted
//! or malicious peer could send arbitrary bytes.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try to deserialize as an oplog entry (the most complex wire type)
    let _ = yantrikdb_protocol::unpack::<yantrikdb_protocol::messages::OplogEntryWire>(data);

    // Try as an oplog pull result (contains a vec of entries)
    let _ = yantrikdb_protocol::unpack::<yantrikdb_protocol::messages::OplogPullResult>(data);

    // Try as a push request
    let _ = yantrikdb_protocol::unpack::<yantrikdb_protocol::messages::OplogPushRequest>(data);
});
