//! Oplog sync loop — followers continuously pull ops from the leader.
//!
//! For multi-DB clusters: on each tick, the follower
//! 1. Asks the leader for its database list
//! 2. Auto-creates any missing databases locally
//! 3. Pulls ops for each database independently with per-DB watermarks

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use tokio_util::sync::CancellationToken;

use yantrikdb_protocol::messages::*;
use yantrikdb_protocol::*;

use crate::cluster::client::{connect_and_handshake, CONNECT_TIMEOUT};
use crate::cluster::replication::{handle_oplog_apply, update_local_watermark};
use crate::cluster::ClusterContext;

const PULL_INTERVAL: Duration = Duration::from_millis(500);
const PULL_BATCH_SIZE: usize = 500;
const DB_LIST_REFRESH_INTERVAL: Duration = Duration::from_secs(10);
const CONTROL_SYNC_INTERVAL: Duration = Duration::from_secs(30);

/// Run the oplog sync loop. Followers and read replicas pull from the leader.
pub async fn run_sync_loop(ctx: Arc<ClusterContext>, cancel: CancellationToken) {
    let mut tick = tokio::time::interval(PULL_INTERVAL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let mut last_db_list_refresh = tokio::time::Instant::now();
    let mut last_control_sync = tokio::time::Instant::now();

    tracing::info!(node_id = ctx.node_id(), "oplog sync loop started");

    loop {
        tokio::select! {
            _ = tick.tick() => {}
            _ = cancel.cancelled() => {
                tracing::info!(node_id = ctx.node_id(), "sync loop stopped");
                return;
            }
        }

        // Only followers and read replicas pull
        if !matches!(
            ctx.state.leader_role(),
            crate::cluster::LeaderRole::Follower | crate::cluster::LeaderRole::ReadOnly
        ) {
            continue;
        }

        // Find current leader's address
        let leader_id = match ctx.state.current_leader() {
            Some(id) => id,
            None => continue,
        };

        let leader_addr = match ctx
            .peers
            .snapshot()
            .into_iter()
            .find(|p| p.node_id == Some(leader_id))
        {
            Some(p) => p.addr,
            None => continue,
        };

        // Periodically refresh our database list from the leader
        if last_db_list_refresh.elapsed() >= DB_LIST_REFRESH_INTERVAL {
            if let Err(e) = sync_database_list(&ctx, &leader_addr).await {
                tracing::trace!(leader = %leader_addr, error = %e, "db list sync failed");
            }
            last_db_list_refresh = tokio::time::Instant::now();
        }

        // Periodically sync the control plane (databases + tokens) from
        // the leader via HTTP snapshot. This ensures tokens minted on the
        // leader are available on followers within ~30 seconds, fixing
        // the "token only works on one node" footgun. See task #83.
        if last_control_sync.elapsed() >= CONTROL_SYNC_INTERVAL {
            if let Err(e) = sync_control_plane(&ctx, &leader_addr).await {
                tracing::trace!(leader = %leader_addr, error = %e, "control plane sync failed");
            }
            last_control_sync = tokio::time::Instant::now();
        }

        // Pull ops for each known database
        let dbs = ctx.list_databases();
        for db_name in dbs {
            if let Err(e) = pull_db_from_leader(&ctx, &leader_addr, &db_name).await {
                tracing::trace!(
                    leader = %leader_addr,
                    db = %db_name,
                    error = %e,
                    "pull failed"
                );
            }
        }
    }
}

/// Pull the leader's control plane snapshot (databases + tokens) via
/// its HTTP admin endpoint and merge into our local control.db.
///
/// This is an additive merge: new databases and tokens from the leader
/// are inserted locally; local-only records are NOT deleted. The snapshot
/// is O(N) for N databases + tokens, which is fine for realistic control
/// plane sizes (< 1000).
async fn sync_control_plane(ctx: &Arc<ClusterContext>, leader_addr: &str) -> anyhow::Result<()> {
    let control = match ctx.control.as_ref() {
        Some(c) => c,
        None => return Ok(()), // no control db → single-node, nothing to sync
    };

    let secret = ctx
        .config
        .cluster_secret
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("no cluster_secret configured"))?;

    // leader_addr is the cluster RPC address (e.g. "192.168.4.141:7440").
    // The HTTP gateway runs on a different port. Derive it from the config
    // or fall back to the standard mapping: cluster_port → http_port.
    // Convention: if cluster port is 7440, HTTP is 7438.
    let http_addr = leader_addr.replace(":7440", ":7438");

    let url = format!("http://{}/v1/admin/control-snapshot", http_addr);

    // Use a simple HTTP client (reqwest would be ideal but we avoid adding
    // a new dependency; tokio::net::TcpStream + manual HTTP is too low-level.
    // Instead, shell out to curl which is available on all deployment targets).
    let output = tokio::process::Command::new("curl")
        .args([
            "-sS",
            "--max-time",
            "10",
            "-H",
            &format!("Authorization: Bearer {}", secret),
            &url,
        ])
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("control-snapshot HTTP failed: {}", stderr);
    }

    let snapshot: crate::control::ControlSnapshot = serde_json::from_slice(&output.stdout)?;

    let control = control.clone();
    let (dbs_added, tokens_added) = tokio::task::spawn_blocking(move || {
        let ctrl = control.lock();
        ctrl.import_snapshot(&snapshot)
    })
    .await??;

    if dbs_added > 0 || tokens_added > 0 {
        tracing::info!(
            dbs_added,
            tokens_added,
            "control plane sync: imported from leader"
        );
    }

    Ok(())
}

async fn sync_database_list(ctx: &Arc<ClusterContext>, leader_addr: &str) -> anyhow::Result<()> {
    let mut conn = connect_and_handshake(leader_addr, ctx).await?;
    let req = ClusterDatabaseListRequest {};
    let frame = make_frame(OpCode::ClusterDatabaseList, 0, &req)?;
    conn.send(frame).await?;

    let resp = tokio::time::timeout(CONNECT_TIMEOUT, conn.next())
        .await?
        .ok_or_else(|| anyhow::anyhow!("no db list response"))??;

    if resp.opcode != OpCode::ClusterDatabaseListResult {
        anyhow::bail!("unexpected opcode for db list: {:?}", resp.opcode);
    }

    let result: ClusterDatabaseListResponse = unpack(&resp.payload)?;

    // Auto-create any missing databases
    let local_dbs: std::collections::HashSet<String> = ctx.list_databases().into_iter().collect();
    for db in &result.databases {
        if !local_dbs.contains(db) {
            if let Err(e) = ctx.ensure_database(db) {
                tracing::warn!(database = %db, error = %e, "failed to auto-create database");
            }
        }
    }

    Ok(())
}

async fn pull_db_from_leader(
    ctx: &Arc<ClusterContext>,
    leader_addr: &str,
    db_name: &str,
) -> anyhow::Result<()> {
    let engine = ctx.engine_for(db_name)?;

    // Find our actor_id (used for exclusion to avoid pulling our own ops)
    let our_actor_id = {
        let db = engine.as_ref();
        db.actor_id().to_string()
    };

    // Per-database watermark key: "{leader_addr}:{db_name}"
    let watermark_key = format!("{}:{}", leader_addr, db_name);

    let watermark = crate::cluster::replication::get_local_watermark(&engine, &watermark_key)?;

    let (since_hlc, since_op_id) = match watermark {
        Some((hlc, op_id)) => (Some(hlc), Some(op_id)),
        None => (None, None),
    };

    let req = OplogPullRequest {
        database: db_name.to_string(),
        since_hlc,
        since_op_id,
        limit: PULL_BATCH_SIZE,
        exclude_actor: Some(our_actor_id),
    };

    let mut conn = connect_and_handshake(leader_addr, ctx).await?;
    let frame = make_frame(OpCode::OplogPull, 0, &req)?;
    conn.send(frame).await?;

    let resp = tokio::time::timeout(CONNECT_TIMEOUT, conn.next())
        .await?
        .ok_or_else(|| anyhow::anyhow!("no pull response"))??;

    if resp.opcode != OpCode::OplogPullResult {
        anyhow::bail!("unexpected opcode: {:?}", resp.opcode);
    }

    let result: OplogPullResult = unpack_frame(&resp)?;
    if result.ops.is_empty() {
        return Ok(());
    }

    let count = result.ops.len();
    let last_hlc = result.ops.last().map(|o| o.hlc.clone()).unwrap_or_default();
    let last_op_id = result
        .ops
        .last()
        .map(|o| o.op_id.clone())
        .unwrap_or_default();

    let apply = handle_oplog_apply(&engine, result.ops)?;

    // Update watermark only if we actually advanced
    if !last_op_id.is_empty() {
        update_local_watermark(&engine, &watermark_key, &last_hlc, &last_op_id)?;
    }

    // After applying replicated ops, the memories rows exist in SQLite but
    // their embedding columns are NULL (the oplog only carries embedding_hash,
    // not the full vector). Re-embed locally and populate both the column
    // and the in-memory HNSW index so recall() works on the follower.
    if apply.applied > 0 {
        if let Err(e) = backfill_embeddings(&engine).await {
            tracing::warn!(error = %e, "embedding backfill failed");
        }
    }

    if apply.applied > 0 {
        tracing::info!(
            leader = %leader_addr,
            db = %db_name,
            pulled = count,
            applied = apply.applied,
            skipped = apply.skipped,
            "oplog pull"
        );
    }

    Ok(())
}

/// After replicated record ops are materialized, the memories rows have
/// no embedding (the oplog doesn't carry vectors). Re-embed each missing
/// row using the local embedder and update both the SQLite column and
/// the in-memory HNSW vector index.
async fn backfill_embeddings(engine: &std::sync::Arc<yantrikdb::YantrikDB>) -> anyhow::Result<()> {
    use rusqlite::params;

    // Collect rids + texts that need embedding
    let pending: Vec<(String, String)> = {
        let db = engine.as_ref();
        if !db.has_embedder() {
            return Ok(()); // no embedder, nothing we can do
        }
        let conn = db.conn();
        let mut stmt = conn.prepare(
            "SELECT rid, text FROM memories \
             WHERE embedding IS NULL \
             AND consolidation_status IN ('active', 'consolidated') \
             LIMIT 500",
        )?;
        let rows: Vec<_> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<Result<_, _>>()?;
        rows
    };

    if pending.is_empty() {
        return Ok(());
    }

    let count = pending.len();
    tracing::debug!(count, "backfilling embeddings for replicated memories");

    // Embed + write back, one at a time to keep lock duration short
    for (rid, text) in &pending {
        let embedding = {
            let db = engine.as_ref();
            match db.embed(text) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(rid = %rid, error = %e, "embed failed during backfill");
                    continue;
                }
            }
        };

        // Use the canonical f32 serialization from the core crate so the
        // bytes match exactly what record() would write.
        let blob = yantrikdb::serde_helpers::serialize_f32(&embedding);

        let db = engine.as_ref();

        // NOTE: if encryption is enabled, the engine's encrypt_embedding()
        // method is pub(crate) — we can't call it from here. For encrypted
        // clusters, the workaround is rebuild_vec_index from the (encrypted)
        // SQLite table will fail and recall on followers won't work until
        // encrypt_embedding is exposed in core. TODO: expose it.
        if db.is_encrypted() {
            tracing::warn!(
                rid = %rid,
                "skipping embedding backfill: encrypted databases need encrypt_embedding exposed in core"
            );
            continue;
        }

        let conn = db.conn();
        if let Err(e) = conn.execute(
            "UPDATE memories SET embedding = ?1 WHERE rid = ?2",
            params![blob, rid],
        ) {
            tracing::warn!(rid = %rid, error = %e, "embedding update failed");
            continue;
        }
        drop(conn);
    }

    // Now rebuild the HNSW index from the SQLite table (which has all embeddings now).
    // This is the only way to get vectors into HNSW since the index API isn't public
    // for piecewise insertion through YantrikDB.
    {
        let db = engine.as_ref();
        if let Err(e) = db.rebuild_vec_index() {
            tracing::warn!(error = %e, "rebuild_vec_index failed during backfill");
        }
    }

    tracing::info!(count, "backfilled embeddings for replicated memories");
    Ok(())
}
