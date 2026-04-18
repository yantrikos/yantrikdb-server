//! Wire protocol TCP server.
//!
//! Accepts connections on the wire port, authenticates via AUTH frame,
//! then dispatches commands to the handler.

use parking_lot::Mutex;
use std::sync::Arc;

use futures::SinkExt;
use futures::StreamExt;
use tokio::net::TcpListener;
use tokio_util::codec::Framed;

use yantrikdb_protocol::messages::*;
use yantrikdb_protocol::*;

use crate::auth;
use crate::background::WorkerRegistry;
use crate::command::{Command, RememberInput};
use crate::control::ControlDb;
use crate::handler::{self, CommandResult};
use crate::tenant_pool::TenantPool;

pub struct AppState {
    pub control: Arc<Mutex<ControlDb>>,
    pub pool: Arc<TenantPool>,
    pub workers: WorkerRegistry,
    /// Optional cluster context — None when running in single-node mode.
    pub cluster: Option<Arc<crate::cluster::ClusterContext>>,
    /// Inflight blocking operations counter for load shedding.
    /// When this exceeds the max, new requests are rejected with 503.
    pub inflight: std::sync::atomic::AtomicU32,
}

/// Maximum concurrent blocking operations before shedding load.
/// Default: 256 (tokio's default blocking pool is 512, shed at 50%).
pub const MAX_INFLIGHT: u32 = 256;

pub async fn run_wire_server(
    listener: TcpListener,
    state: Arc<AppState>,
    tls_acceptor: Option<tokio_rustls::TlsAcceptor>,
) -> anyhow::Result<()> {
    let tls_mode = tls_acceptor.is_some();
    tracing::info!(
        addr = %listener.local_addr()?,
        tls = tls_mode,
        "wire protocol server listening"
    );

    // Per-IP connection counter for rate limiting (task #78).
    // Simple HashMap<IpAddr, AtomicU32> with a max concurrent connections
    // per IP. Prevents a single client from exhausting file descriptors.
    let conn_counts: Arc<parking_lot::Mutex<std::collections::HashMap<std::net::IpAddr, u32>>> =
        Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new()));
    const MAX_CONNS_PER_IP: u32 = 100;

    loop {
        let (stream, addr) = listener.accept().await?;
        let ip = addr.ip();

        // Rate limit: reject if this IP has too many open connections
        {
            let mut counts = conn_counts.lock();
            let count = counts.entry(ip).or_insert(0);
            if *count >= MAX_CONNS_PER_IP {
                tracing::warn!(%addr, connections = *count, "connection rate limit exceeded");
                drop(stream);
                continue;
            }
            *count += 1;
        }

        let state = Arc::clone(&state);
        let tls = tls_acceptor.clone();
        let conn_counts_clone = Arc::clone(&conn_counts);
        tokio::spawn(async move {
            tracing::debug!(%addr, "new connection");
            let result = if let Some(acceptor) = tls {
                match acceptor.accept(stream).await {
                    Ok(tls_stream) => handle_connection_tls(tls_stream, state).await,
                    Err(e) => {
                        tracing::warn!(%addr, error = %e, "TLS handshake failed");
                        return;
                    }
                }
            } else {
                handle_connection(stream, state).await
            };
            if let Err(e) = result {
                tracing::error!(%addr, error = %e, "connection error");
            }
            // Decrement per-IP connection counter
            {
                let mut counts = conn_counts_clone.lock();
                if let Some(count) = counts.get_mut(&ip) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        counts.remove(&ip);
                    }
                }
            }
            tracing::debug!(%addr, "connection closed");
        });
    }
}

async fn handle_connection_tls(
    stream: tokio_rustls::server::TlsStream<tokio::net::TcpStream>,
    state: Arc<AppState>,
) -> anyhow::Result<()> {
    handle_connection_inner(stream, state).await
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    state: Arc<AppState>,
) -> anyhow::Result<()> {
    handle_connection_inner(stream, state).await
}

async fn handle_connection_inner<S>(stream: S, state: Arc<AppState>) -> anyhow::Result<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut framed = Framed::new(stream, YantrikCodec::new());

    // Phase 1: Authenticate
    let (db_id, _db_name) = match authenticate(&mut framed, &state).await {
        Ok(result) => result,
        Err(e) => {
            tracing::warn!(error = %e, "auth failed");
            return Ok(());
        }
    };

    // Get the engine for this database
    let db_record = state
        .control
        .lock()
        .get_database_by_id(db_id)?
        .ok_or_else(|| anyhow::anyhow!("database not found"))?;
    let engine = state.pool.get_engine(&db_record)?;

    // Start background workers if not already running
    state
        .workers
        .start_for_database(db_id, db_record.name.clone(), Arc::clone(&engine));

    // Phase 2: Command loop
    while let Some(result) = framed.next().await {
        let frame = match result {
            Ok(f) => f,
            Err(e) => {
                tracing::error!(error = %e, "frame decode error");
                break;
            }
        };

        let stream_id = frame.stream_id;

        match frame_to_command(&frame) {
            Ok(cmd) => {
                // Reject writes on non-leader nodes in clustered mode
                if let Some(ref cluster) = state.cluster {
                    if is_write_command(&cmd) && !cluster.state.accepts_writes() {
                        let leader = cluster.state.current_leader();
                        let msg = match leader {
                            Some(id) => format!("not the leader (current leader: node {})", id),
                            None => "no leader elected (cluster not ready)".into(),
                        };
                        let err = make_error(stream_id, error_codes::READONLY_NODE, msg)?;
                        framed.send(err).await?;
                        continue;
                    }
                }

                // Run the engine call on a blocking thread so a slow op
                // (think, consolidate, embed) cannot park the tokio worker
                // serving this connection. Holding parking_lot::Mutex across
                // an await would be a deadlock footgun; spawn_blocking makes
                // that structurally impossible.
                let engine_clone = std::sync::Arc::clone(&engine);
                let control_clone = std::sync::Arc::clone(&state.control);
                let response_frames = tokio::task::spawn_blocking(move || {
                    execute_and_respond(cmd, &engine_clone, &control_clone, stream_id)
                })
                .await
                .unwrap_or_else(|e| {
                    tracing::error!(error = %e, "execute_and_respond join error");
                    vec![]
                });
                for f in response_frames {
                    framed.send(f).await?;
                }
            }
            Err(e) => {
                let err_frame = make_error(stream_id, error_codes::INVALID_PAYLOAD, e.to_string())?;
                framed.send(err_frame).await?;
            }
        }
    }

    Ok(())
}

async fn authenticate<S>(
    framed: &mut Framed<S, YantrikCodec>,
    state: &AppState,
) -> anyhow::Result<(i64, String)>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    // Wait for AUTH frame
    let frame = framed
        .next()
        .await
        .ok_or_else(|| anyhow::anyhow!("connection closed before auth"))??;

    if frame.opcode != OpCode::Auth {
        let err = make_error(
            frame.stream_id,
            error_codes::AUTH_REQUIRED,
            "expected AUTH frame",
        )?;
        framed.send(err).await?;
        anyhow::bail!("expected AUTH frame, got {:?}", frame.opcode);
    }

    let auth_req: AuthRequest = unpack(&frame.payload)?;
    let token_hash = auth::hash_token(&auth_req.token);

    // Cluster master token: if clustering is enabled and the token matches
    // the cluster_secret, accept it as access to the default database.
    if let Some(ref cluster) = state.cluster {
        if let Some(ref secret) = cluster.config.cluster_secret {
            if &auth_req.token == secret {
                // Look up the default database
                let db_record = state
                    .control
                    .lock()
                    .get_database("default")?
                    .ok_or_else(|| anyhow::anyhow!("default database not found"))?;
                let resp = AuthOkResponse {
                    database: db_record.name.clone(),
                    database_id: db_record.id,
                };
                let resp_frame = make_frame(OpCode::AuthOk, frame.stream_id, &resp)?;
                framed.send(resp_frame).await?;
                return Ok((db_record.id, db_record.name));
            }
        }
    }

    // Scope the lock so MutexGuard is dropped before any .await
    let auth_result = {
        let control = state.control.lock();
        match control.validate_token(&token_hash)? {
            Some(db_id) => {
                let db_record = control
                    .get_database_by_id(db_id)?
                    .ok_or_else(|| anyhow::anyhow!("database not found for token"))?;
                Ok((db_record.id, db_record.name.clone()))
            }
            None => Err(anyhow::anyhow!("invalid token")),
        }
    };

    match auth_result {
        Ok((db_id, db_name)) => {
            let resp = AuthOkResponse {
                database: db_name.clone(),
                database_id: db_id,
            };
            let resp_frame = make_frame(OpCode::AuthOk, frame.stream_id, &resp)?;
            framed.send(resp_frame).await?;
            Ok((db_id, db_name))
        }
        Err(e) => {
            let resp = AuthFailResponse {
                reason: "invalid or revoked token".into(),
            };
            let resp_frame = make_frame(OpCode::AuthFail, frame.stream_id, &resp)?;
            framed.send(resp_frame).await?;
            Err(e)
        }
    }
}

/// Whether a command modifies state (must be rejected on read-only nodes).
fn is_write_command(cmd: &Command) -> bool {
    matches!(
        cmd,
        Command::Remember { .. }
            | Command::RememberBatch { .. }
            | Command::Forget { .. }
            | Command::Relate { .. }
            | Command::IngestClaim { .. }
            | Command::AddAlias { .. }
            | Command::SessionStart { .. }
            | Command::SessionEnd { .. }
            | Command::Think { .. }
            | Command::Resolve { .. }
            | Command::CreateDb { .. }
    )
}

fn frame_to_command(frame: &Frame) -> anyhow::Result<Command> {
    match frame.opcode {
        OpCode::Remember => {
            let req: RememberRequest = unpack(&frame.payload)?;
            Ok(Command::Remember {
                text: req.text,
                memory_type: req.memory_type,
                importance: req.importance,
                valence: req.valence,
                half_life: req.half_life,
                metadata: req.metadata,
                namespace: req.namespace,
                certainty: req.certainty,
                domain: req.domain,
                source: req.source,
                emotional_state: req.emotional_state,
                embedding: req.embedding,
            })
        }
        OpCode::RememberBatch => {
            let req: RememberBatchRequest = unpack(&frame.payload)?;
            let memories = req
                .memories
                .into_iter()
                .map(|m| RememberInput {
                    text: m.text,
                    memory_type: m.memory_type,
                    importance: m.importance,
                    valence: m.valence,
                    half_life: m.half_life,
                    metadata: m.metadata,
                    namespace: m.namespace,
                    certainty: m.certainty,
                    domain: m.domain,
                    source: m.source,
                    emotional_state: m.emotional_state,
                    embedding: m.embedding,
                })
                .collect();
            Ok(Command::RememberBatch { memories })
        }
        OpCode::Recall => {
            let req: RecallRequest = unpack(&frame.payload)?;
            Ok(Command::Recall {
                query: req.query,
                top_k: req.top_k,
                memory_type: req.memory_type,
                include_consolidated: req.include_consolidated,
                expand_entities: req.expand_entities,
                namespace: req.namespace,
                domain: req.domain,
                source: req.source,
                query_embedding: req.query_embedding,
            })
        }
        OpCode::Forget => {
            let req: ForgetRequest = unpack(&frame.payload)?;
            Ok(Command::Forget { rid: req.rid })
        }
        OpCode::Relate => {
            let req: RelateRequest = unpack(&frame.payload)?;
            Ok(Command::Relate {
                entity: req.entity,
                target: req.target,
                relationship: req.relationship,
                weight: req.weight,
            })
        }
        OpCode::Edges => {
            let req: EdgesRequest = unpack(&frame.payload)?;
            Ok(Command::Edges { entity: req.entity })
        }
        OpCode::SessionStart => {
            let req: SessionStartRequest = unpack(&frame.payload)?;
            Ok(Command::SessionStart {
                namespace: req.namespace,
                client_id: req.client_id,
                metadata: req.metadata,
            })
        }
        OpCode::SessionEnd => {
            let req: SessionEndRequest = unpack(&frame.payload)?;
            Ok(Command::SessionEnd {
                session_id: req.session_id,
                summary: req.summary,
            })
        }
        OpCode::Think => {
            let req: ThinkRequest = unpack(&frame.payload)?;
            Ok(Command::Think {
                run_consolidation: req.run_consolidation,
                run_conflict_scan: req.run_conflict_scan,
                run_pattern_mining: req.run_pattern_mining,
                run_personality: req.run_personality,
                consolidation_limit: req.consolidation_limit,
            })
        }
        OpCode::Conflicts => {
            let req: ConflictsRequest = unpack(&frame.payload)?;
            Ok(Command::Conflicts {
                status: req.status,
                conflict_type: req.conflict_type,
                entity: req.entity,
                limit: req.limit,
            })
        }
        OpCode::Resolve => {
            let req: ResolveRequest = unpack(&frame.payload)?;
            Ok(Command::Resolve {
                conflict_id: req.conflict_id,
                strategy: req.strategy,
                winner_rid: req.winner_rid,
                new_text: req.new_text,
                resolution_note: req.resolution_note,
            })
        }
        OpCode::Claim => {
            let req: ClaimRequest = unpack(&frame.payload)?;
            Ok(Command::IngestClaim {
                src: req.src,
                rel_type: req.rel_type,
                dst: req.dst,
                namespace: req.namespace,
                polarity: req.polarity,
                modality: req.modality,
                valid_from: req.valid_from,
                valid_to: req.valid_to,
                extractor: req.extractor,
                extractor_version: req.extractor_version,
                confidence_band: req.confidence_band,
                source_memory_rid: req.source_memory_rid,
                span_start: req.span_start,
                span_end: req.span_end,
                weight: req.weight,
            })
        }
        OpCode::Claims => {
            let req: ClaimsRequest = unpack(&frame.payload)?;
            Ok(Command::GetClaims {
                entity: req.entity,
                namespace: if req.namespace.is_empty() || req.namespace == "default" {
                    None
                } else {
                    Some(req.namespace)
                },
            })
        }
        OpCode::Alias => {
            let req: AliasRequest = unpack(&frame.payload)?;
            Ok(Command::AddAlias {
                alias: req.alias,
                canonical_name: req.canonical_name,
                namespace: req.namespace,
                source: req.source,
            })
        }
        OpCode::Personality => Ok(Command::Personality),
        OpCode::Stats => Ok(Command::Stats),
        OpCode::CreateDb => {
            let req: CreateDbRequest = unpack(&frame.payload)?;
            Ok(Command::CreateDb { name: req.name })
        }
        OpCode::ListDb => Ok(Command::ListDb),
        OpCode::Ping => Ok(Command::Ping),
        other => anyhow::bail!("unsupported opcode: {:?}", other),
    }
}

fn execute_and_respond(
    cmd: Command,
    engine: &std::sync::Arc<parking_lot::Mutex<yantrikdb::YantrikDB>>,
    control: &Mutex<ControlDb>,
    stream_id: u32,
) -> Vec<Frame> {
    match handler::execute(engine, cmd, Some(control)) {
        Ok(CommandResult::Json(value)) => {
            // Determine the right response opcode from the value
            let opcode = response_opcode_for_json(&value);
            match yantrikdb_protocol::pack(&value) {
                Ok(payload) => vec![Frame::new(opcode, stream_id, payload)],
                Err(e) => {
                    tracing::error!(error = %e, "failed to serialize response");
                    vec![]
                }
            }
        }
        Ok(CommandResult::RecallResults { results, total }) => {
            let mut frames = Vec::with_capacity(results.len() + 1);
            for result in &results {
                if let Ok(payload) = yantrikdb_protocol::pack(result) {
                    frames.push(Frame::new(OpCode::RecallResult, stream_id, payload));
                }
            }
            let end = RecallEndMsg {
                total,
                confidence: 0.0,
            };
            if let Ok(payload) = yantrikdb_protocol::pack(&end) {
                frames.push(Frame::new(OpCode::RecallEnd, stream_id, payload));
            }
            frames
        }
        Ok(CommandResult::Pong) => {
            vec![Frame::empty(OpCode::Pong, stream_id)]
        }
        Err(e) => match make_error(stream_id, error_codes::INTERNAL_ERROR, e.to_string()) {
            Ok(f) => vec![f],
            Err(_) => vec![],
        },
    }
}

fn response_opcode_for_json(value: &serde_json::Value) -> OpCode {
    // Check in order of specificity. Groups of keys that map to the same
    // opcode share a branch to satisfy clippy::if_same_then_else.
    if (value.get("rid").is_some() && value.get("found").is_none()) || value.get("rids").is_some() {
        OpCode::RememberOk
    } else if value.get("found").is_some() {
        OpCode::ForgetOk
    } else if value.get("claim_id").is_some() {
        OpCode::ClaimOk
    } else if value.get("claims").is_some() {
        OpCode::ClaimsResult
    } else if value.get("alias").is_some() {
        OpCode::AliasOk
    } else if value.get("edge_id").is_some() {
        OpCode::RelateOk
    } else if value.get("edges").is_some() {
        OpCode::EdgesResult
    } else if value.get("session_id").is_some() {
        OpCode::SessionOk
    } else if value.get("consolidation_count").is_some() {
        OpCode::ThinkResult
    } else if value.get("conflicts").is_some() || value.get("conflict_id").is_some() {
        OpCode::ConflictResult
    } else if value.get("databases").is_some() {
        OpCode::ListDbResult
    } else if value.get("message").is_some() {
        OpCode::DbOk
    } else {
        // Default catch-all for `traits`, `active_memories`, and other
        // free-form info responses.
        OpCode::InfoResult
    }
}
