//! Replication transport — oplog pull/push between peers.
//!
//! Wraps `yantrikdb::distributed::replication::extract_ops_since` and
//! `apply_ops` and exposes them over the wire protocol.

use std::sync::Arc;

use yantrikdb::replication::{
    apply_ops, extract_ops_since, get_peer_watermark, set_peer_watermark, OplogEntry,
};
use yantrikdb::YantrikDB;
#[allow(unused_imports)]
use yantrikdb_protocol::messages::*;
use yantrikdb_protocol::messages::{OplogEntryWire, OplogPullRequest, OplogPullResult};
#[allow(unused_imports)]
use yantrikdb_protocol::*;

/// Convert a core `OplogEntry` to wire-friendly form.
pub fn entry_to_wire(e: &OplogEntry) -> OplogEntryWire {
    OplogEntryWire {
        op_id: e.op_id.clone(),
        op_type: e.op_type.clone(),
        timestamp: e.timestamp,
        target_rid: e.target_rid.clone(),
        payload: e.payload.clone(),
        actor_id: e.actor_id.clone(),
        hlc: e.hlc.clone(),
        embedding_hash: e.embedding_hash.clone(),
        origin_actor: e.origin_actor.clone(),
    }
}

pub fn wire_to_entry(w: OplogEntryWire) -> OplogEntry {
    OplogEntry {
        op_id: w.op_id,
        op_type: w.op_type,
        timestamp: w.timestamp,
        target_rid: w.target_rid,
        payload: w.payload,
        actor_id: w.actor_id,
        hlc: w.hlc,
        embedding_hash: w.embedding_hash,
        origin_actor: w.origin_actor,
    }
}

/// Handle an OplogPull request — extract ops since the requested watermark.
pub fn handle_oplog_pull(
    engine: &Arc<std::sync::Mutex<YantrikDB>>,
    req: OplogPullRequest,
) -> anyhow::Result<OplogPullResult> {
    let db = engine.lock().unwrap();
    let conn = db.conn();

    let since_hlc = req.since_hlc.as_deref();
    let since_op_id = req.since_op_id.as_deref();
    let exclude_actor = req.exclude_actor.as_deref();
    let limit = req.limit.max(1).min(10_000);

    let ops = extract_ops_since(&conn, since_hlc, since_op_id, exclude_actor, limit)?;
    let has_more = ops.len() == limit;
    let wire_ops = ops.iter().map(entry_to_wire).collect();

    Ok(OplogPullResult {
        ops: wire_ops,
        has_more,
    })
}

/// Apply pulled/pushed ops to the local engine.
pub fn handle_oplog_apply(
    engine: &Arc<std::sync::Mutex<YantrikDB>>,
    ops_wire: Vec<OplogEntryWire>,
) -> anyhow::Result<ApplyResult> {
    let ops: Vec<OplogEntry> = ops_wire.into_iter().map(wire_to_entry).collect();

    let last_hlc = ops.last().map(|o| o.hlc.clone()).unwrap_or_default();
    let last_op_id = ops.last().map(|o| o.op_id.clone()).unwrap_or_default();
    let count = ops.len();

    let db = engine.lock().unwrap();
    let stats = apply_ops(&db, &ops)?;

    Ok(ApplyResult {
        applied: stats.ops_applied,
        skipped: stats.ops_skipped,
        total: count,
        last_hlc,
        last_op_id,
    })
}

#[derive(Debug, Clone)]
pub struct ApplyResult {
    pub applied: usize,
    pub skipped: usize,
    pub total: usize,
    pub last_hlc: Vec<u8>,
    pub last_op_id: String,
}

/// Read the local watermark for a peer (where we last synced from them).
pub fn get_local_watermark(
    engine: &Arc<std::sync::Mutex<YantrikDB>>,
    peer_actor: &str,
) -> anyhow::Result<Option<(Vec<u8>, String)>> {
    let db = engine.lock().unwrap();
    let conn = db.conn();
    Ok(get_peer_watermark(&conn, peer_actor)?)
}

/// Update the local watermark for a peer after a successful pull.
pub fn update_local_watermark(
    engine: &Arc<std::sync::Mutex<YantrikDB>>,
    peer_actor: &str,
    hlc: &[u8],
    op_id: &str,
) -> anyhow::Result<()> {
    let db = engine.lock().unwrap();
    let conn = db.conn();
    set_peer_watermark(&conn, peer_actor, hlc, op_id)?;
    Ok(())
}
