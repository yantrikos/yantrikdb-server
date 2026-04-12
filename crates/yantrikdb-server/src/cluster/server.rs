//! Cluster TCP server — listens for peer-to-peer traffic on a separate port.
//!
//! Handles cluster opcodes: ClusterHello, OplogPull/Push, Heartbeat,
//! RequestVote, ClusterStatus.

use std::sync::Arc;

use futures::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::codec::Framed;

use yantrikdb_protocol::messages::*;
use yantrikdb_protocol::*;

use crate::cluster::replication::{handle_oplog_apply, handle_oplog_pull};
use crate::cluster::ClusterContext;

/// Run the cluster wire protocol server.
pub async fn run_cluster_server(
    listener: TcpListener,
    ctx: Arc<ClusterContext>,
) -> anyhow::Result<()> {
    tracing::info!(
        addr = %listener.local_addr()?,
        node_id = ctx.node_id(),
        "cluster server listening"
    );

    loop {
        let (stream, addr) = listener.accept().await?;
        let ctx = Arc::clone(&ctx);
        tokio::spawn(async move {
            tracing::debug!(%addr, "peer connection");
            if let Err(e) = handle_peer_connection(stream, ctx).await {
                tracing::warn!(%addr, error = %e, "peer connection error");
            }
        });
    }
}

async fn handle_peer_connection(stream: TcpStream, ctx: Arc<ClusterContext>) -> anyhow::Result<()> {
    let mut framed = Framed::new(stream, YantrikCodec::new());

    // Phase 1: handshake
    let frame = framed
        .next()
        .await
        .ok_or_else(|| anyhow::anyhow!("connection closed before hello"))??;

    if frame.opcode != OpCode::ClusterHello {
        let err = make_error(
            frame.stream_id,
            error_codes::INVALID_PAYLOAD,
            "expected ClusterHello",
        )?;
        framed.send(err).await?;
        return Ok(());
    }

    let hello: ClusterHello = unpack(&frame.payload)?;

    // Verify cluster secret
    if !ctx.verify_secret(&hello.cluster_secret) {
        let err = make_error(
            frame.stream_id,
            error_codes::CLUSTER_SECRET_MISMATCH,
            "cluster secret mismatch",
        )?;
        framed.send(err).await?;
        anyhow::bail!("cluster secret mismatch from peer");
    }

    // Record peer in registry
    ctx.peers
        .record_handshake(&hello.advertise_addr, hello.node_id, hello.current_term);

    // Send hello-ok
    let resp = ClusterHelloOk {
        node_id: ctx.node_id(),
        role: role_string(ctx.state.configured_role),
        current_term: ctx.state.current_term(),
        leader_id: ctx.state.current_leader(),
    };
    let resp_frame = make_frame(OpCode::ClusterHelloOk, frame.stream_id, &resp)?;
    framed.send(resp_frame).await?;

    // Phase 2: command loop
    while let Some(result) = framed.next().await {
        let frame = match result {
            Ok(f) => f,
            Err(e) => {
                tracing::debug!(error = %e, "frame decode error");
                break;
            }
        };

        let stream_id = frame.stream_id;
        match dispatch_cluster_op(&frame, &ctx).await {
            Ok(response_frame) => {
                if let Some(f) = response_frame {
                    framed.send(f).await?;
                }
            }
            Err(e) => {
                let err_frame = make_error(stream_id, error_codes::INTERNAL_ERROR, e.to_string())?;
                framed.send(err_frame).await?;
            }
        }
    }

    Ok(())
}

async fn dispatch_cluster_op(
    frame: &Frame,
    ctx: &Arc<ClusterContext>,
) -> anyhow::Result<Option<Frame>> {
    let stream_id = frame.stream_id;

    match frame.opcode {
        OpCode::OplogPull => {
            let req: OplogPullRequest = unpack_frame(frame)?;
            let engine = ctx.engine_for(&req.database)?;
            let result = handle_oplog_pull(&engine, req)?;
            // Auto-compress payloads >4KB (typical oplog batch is 50-500KB)
            let resp = make_frame_auto_compress(OpCode::OplogPullResult, stream_id, &result, 4096)?;
            Ok(Some(resp))
        }

        OpCode::OplogPush => {
            let req: OplogPushRequest = unpack_frame(frame)?;
            let engine = ctx.engine_for(&req.database)?;
            let apply = handle_oplog_apply(&engine, req.ops)?;
            let resp = OplogPushOkResponse {
                applied: apply.applied,
                last_hlc: apply.last_hlc,
                last_op_id: apply.last_op_id,
            };
            let resp_frame = make_frame(OpCode::OplogPushOk, stream_id, &resp)?;
            Ok(Some(resp_frame))
        }

        OpCode::ClusterDatabaseList => {
            let _req: ClusterDatabaseListRequest = unpack(&frame.payload)?;
            let resp = ClusterDatabaseListResponse {
                databases: ctx.list_databases(),
            };
            let resp_frame = make_frame(OpCode::ClusterDatabaseListResult, stream_id, &resp)?;
            Ok(Some(resp_frame))
        }

        OpCode::Heartbeat => {
            let hb: HeartbeatMsg = unpack(&frame.payload)?;
            ctx.state.record_heartbeat(hb.leader_id, hb.term)?;

            // Mark leader's peer entry as reachable based on node_id match.
            // (We don't have addr from heartbeat, so we look up by node_id.)
            for peer in ctx.peers.snapshot() {
                if peer.node_id == Some(hb.leader_id) {
                    ctx.peers.update_oplog_position(
                        &peer.addr,
                        hb.leader_last_hlc.clone(),
                        hb.leader_last_op_id.clone(),
                    );
                    break;
                }
            }

            let ack = HeartbeatAckMsg {
                term: ctx.state.current_term(),
                follower_id: ctx.node_id(),
                follower_role: role_string(ctx.state.configured_role),
                follower_last_hlc: ctx.last_hlc()?,
                follower_last_op_id: ctx.last_op_id()?,
                lag_seconds: 0.0, // TODO compute from oplog
            };
            let ack_frame = make_frame(OpCode::HeartbeatAck, stream_id, &ack)?;
            Ok(Some(ack_frame))
        }

        OpCode::RequestVote => {
            let req: RequestVoteMsg = unpack(&frame.payload)?;

            // Only voters and witnesses grant votes
            if !ctx.state.is_voter() {
                let resp = VoteResponseMsg {
                    term: ctx.state.current_term(),
                    voter_id: ctx.node_id(),
                    granted: false,
                    reason: Some("not a voter".into()),
                };
                let resp_frame = make_frame(OpCode::VoteDenied, stream_id, &resp)?;
                return Ok(Some(resp_frame));
            }

            // If candidate's term is higher, step down to follower regardless of vote outcome.
            // This ensures we don't keep claiming to be leader after a newer term started.
            if req.term > ctx.state.current_term() {
                ctx.state.become_follower(req.term, None)?;
            }

            // Check if candidate's log is at least as up-to-date as ours
            let our_last_hlc = ctx.last_hlc()?;
            let candidate_log_ok = req.last_log_hlc >= our_last_hlc;

            let granted = if candidate_log_ok {
                ctx.state.grant_vote(req.term, req.candidate_id)?
            } else {
                false
            };

            let opcode = if granted {
                OpCode::VoteGranted
            } else {
                OpCode::VoteDenied
            };

            let resp = VoteResponseMsg {
                term: ctx.state.current_term(),
                voter_id: ctx.node_id(),
                granted,
                reason: if granted {
                    None
                } else if !candidate_log_ok {
                    Some("candidate log too far behind".into())
                } else {
                    Some("already voted this term".into())
                },
            };
            let resp_frame = make_frame(opcode, stream_id, &resp)?;
            Ok(Some(resp_frame))
        }

        OpCode::ClusterStatus => {
            let status = build_cluster_status(ctx)?;
            let resp_frame = make_frame(OpCode::ClusterStatusResult, stream_id, &status)?;
            Ok(Some(resp_frame))
        }

        OpCode::Ping => Ok(Some(Frame::empty(OpCode::Pong, stream_id))),

        other => {
            anyhow::bail!("unsupported cluster opcode: {:?}", other);
        }
    }
}

fn build_cluster_status(ctx: &ClusterContext) -> anyhow::Result<ClusterStatusResultMsg> {
    let peers = ctx
        .peers
        .snapshot()
        .into_iter()
        .map(|p| PeerStatusMsg {
            node_id: p.node_id.unwrap_or(0),
            addr: p.addr,
            role: format!("{:?}", p.configured_role).to_lowercase(),
            reachable: p.reachable,
            current_term: p.current_term,
            last_seen_secs_ago: p
                .last_seen
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(f64::INFINITY),
            lag_seconds: 0.0,
        })
        .collect();

    Ok(ClusterStatusResultMsg {
        current_term: ctx.state.current_term(),
        leader_id: ctx.state.current_leader(),
        self_id: ctx.node_id(),
        self_role: role_string(ctx.state.configured_role),
        peers,
        quorum_size: ctx.quorum_size(),
        healthy: ctx.is_healthy(),
    })
}

pub fn role_string(role: crate::config::NodeRole) -> String {
    match role {
        crate::config::NodeRole::Single => "single".into(),
        crate::config::NodeRole::Voter => "voter".into(),
        crate::config::NodeRole::ReadReplica => "read_replica".into(),
        crate::config::NodeRole::Witness => "witness".into(),
    }
}
