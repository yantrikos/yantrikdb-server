//! Command handler — executes Commands against a YantrikDB engine instance.

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use yantrikdb::types::ThinkConfig;
use yantrikdb::YantrikDB;

use crate::command::Command;
use crate::control::ControlDb;

/// Result of executing a command.
#[derive(Debug)]
pub enum CommandResult {
    /// Single JSON value response.
    Json(Value),
    /// Streaming recall results (sent one at a time over wire protocol).
    RecallResults { results: Vec<Value>, total: usize },
    /// Pong response.
    Pong,
}

/// Execute a command against the given engine.
pub fn execute(
    engine: &Arc<Mutex<YantrikDB>>,
    cmd: Command,
    control: Option<&Mutex<ControlDb>>,
) -> anyhow::Result<CommandResult> {
    match cmd {
        Command::Remember {
            text,
            memory_type,
            importance,
            valence,
            half_life,
            metadata,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
            embedding,
        } => {
            let db = engine.lock().unwrap();
            let rid = if let Some(emb) = embedding {
                db.record(
                    &text,
                    &memory_type,
                    importance,
                    valence,
                    half_life,
                    &metadata,
                    &emb,
                    &namespace,
                    certainty,
                    &domain,
                    &source,
                    emotional_state.as_deref(),
                )?
            } else {
                db.record_text(
                    &text,
                    &memory_type,
                    importance,
                    valence,
                    half_life,
                    &metadata,
                    &namespace,
                    certainty,
                    &domain,
                    &source,
                    emotional_state.as_deref(),
                )?
            };
            Ok(CommandResult::Json(json!({ "rid": rid })))
        }

        Command::RememberBatch { memories } => {
            let db = engine.lock().unwrap();
            let mut rids = Vec::with_capacity(memories.len());
            for m in memories {
                let rid = if let Some(emb) = m.embedding {
                    db.record(
                        &m.text,
                        &m.memory_type,
                        m.importance,
                        m.valence,
                        m.half_life,
                        &m.metadata,
                        &emb,
                        &m.namespace,
                        m.certainty,
                        &m.domain,
                        &m.source,
                        m.emotional_state.as_deref(),
                    )?
                } else {
                    db.record_text(
                        &m.text,
                        &m.memory_type,
                        m.importance,
                        m.valence,
                        m.half_life,
                        &m.metadata,
                        &m.namespace,
                        m.certainty,
                        &m.domain,
                        &m.source,
                        m.emotional_state.as_deref(),
                    )?
                };
                rids.push(rid);
            }
            Ok(CommandResult::Json(json!({ "rids": rids })))
        }

        Command::Recall {
            query,
            top_k,
            memory_type,
            include_consolidated,
            expand_entities,
            namespace,
            domain,
            source,
            query_embedding,
        } => {
            let db = engine.lock().unwrap();

            let results = if let Some(emb) = query_embedding {
                db.recall(
                    &emb,
                    top_k,
                    None,
                    memory_type.as_deref(),
                    include_consolidated,
                    expand_entities,
                    Some(&query),
                    false,
                    namespace.as_deref(),
                    domain.as_deref(),
                    source.as_deref(),
                )?
            } else {
                // Use the convenience method that auto-embeds
                if domain.is_some() || source.is_some() {
                    db.recall_text_filtered(&query, top_k, domain.as_deref(), source.as_deref())?
                } else {
                    db.recall_text(&query, top_k)?
                }
            };

            let result_values: Vec<Value> = results
                .iter()
                .map(|r| {
                    json!({
                        "rid": r.rid,
                        "text": r.text,
                        "memory_type": r.memory_type,
                        "score": r.score,
                        "importance": r.importance,
                        "created_at": r.created_at,
                        "why_retrieved": r.why_retrieved,
                        "metadata": r.metadata,
                        "namespace": r.namespace,
                        "domain": r.domain,
                        "source": r.source,
                        "certainty": r.certainty,
                        "valence": r.valence,
                    })
                })
                .collect();

            let total = result_values.len();
            Ok(CommandResult::RecallResults {
                results: result_values,
                total,
            })
        }

        Command::Forget { rid } => {
            let db = engine.lock().unwrap();
            let found = db.forget(&rid)?;
            Ok(CommandResult::Json(json!({ "rid": rid, "found": found })))
        }

        Command::Relate {
            entity,
            target,
            relationship,
            weight,
        } => {
            let db = engine.lock().unwrap();
            let edge_id = db.relate(&entity, &target, &relationship, weight)?;
            Ok(CommandResult::Json(json!({ "edge_id": edge_id })))
        }

        Command::Edges { entity } => {
            let db = engine.lock().unwrap();
            let edges = db.get_edges(&entity)?;
            let edge_list: Vec<Value> = edges
                .iter()
                .map(|e| {
                    json!({
                        "edge_id": e.edge_id,
                        "src": e.src,
                        "dst": e.dst,
                        "rel_type": e.rel_type,
                        "weight": e.weight,
                    })
                })
                .collect();
            Ok(CommandResult::Json(json!({ "edges": edge_list })))
        }

        Command::SessionStart {
            namespace,
            client_id,
            metadata,
        } => {
            let db = engine.lock().unwrap();
            let session_id = db.session_start(&namespace, &client_id, &metadata)?;
            Ok(CommandResult::Json(json!({ "session_id": session_id })))
        }

        Command::SessionEnd {
            session_id,
            summary,
        } => {
            let db = engine.lock().unwrap();
            let result = db.session_end(&session_id, summary.as_deref())?;
            Ok(CommandResult::Json(json!({
                "session_id": result.session_id,
                "duration_secs": result.duration_secs,
                "memory_count": result.memory_count,
                "topics": result.topics,
            })))
        }

        Command::Think {
            run_consolidation,
            run_conflict_scan,
            run_pattern_mining,
            run_personality,
            consolidation_limit,
        } => {
            let db = engine.lock().unwrap();
            let config = ThinkConfig {
                run_consolidation,
                run_conflict_scan,
                run_pattern_mining,
                run_personality,
                consolidation_limit,
                ..ThinkConfig::default()
            };
            let result = db.think(&config)?;
            let triggers: Vec<Value> = result
                .triggers
                .iter()
                .map(|t| {
                    json!({
                        "trigger_type": t.trigger_type,
                        "reason": t.reason,
                        "urgency": t.urgency,
                        "source_rids": t.source_rids,
                        "suggested_action": t.suggested_action,
                    })
                })
                .collect();
            Ok(CommandResult::Json(json!({
                "consolidation_count": result.consolidation_count,
                "conflicts_found": result.conflicts_found,
                "patterns_new": result.patterns_new,
                "patterns_updated": result.patterns_updated,
                "personality_updated": result.personality_updated,
                "duration_ms": result.duration_ms,
                "triggers": triggers,
            })))
        }

        Command::Conflicts {
            status,
            conflict_type,
            entity,
            limit,
        } => {
            let db = engine.lock().unwrap();
            let conflicts = db.get_conflicts(
                status.as_deref(),
                conflict_type.as_deref(),
                entity.as_deref(),
                None, // priority
                limit,
            )?;
            let list: Vec<Value> = conflicts
                .iter()
                .map(|c| {
                    json!({
                        "conflict_id": c.conflict_id,
                        "conflict_type": c.conflict_type,
                        "priority": c.priority,
                        "status": c.status,
                        "memory_a": c.memory_a,
                        "memory_b": c.memory_b,
                        "entity": c.entity,
                        "detection_reason": c.detection_reason,
                        "detected_at": c.detected_at,
                    })
                })
                .collect();
            Ok(CommandResult::Json(json!({ "conflicts": list })))
        }

        Command::Resolve {
            conflict_id,
            strategy,
            winner_rid,
            new_text,
            resolution_note,
        } => {
            let db = engine.lock().unwrap();
            let _result = db.resolve_conflict(
                &conflict_id,
                &strategy,
                winner_rid.as_deref(),
                new_text.as_deref(),
                resolution_note.as_deref(),
            )?;
            Ok(CommandResult::Json(json!({
                "conflict_id": conflict_id,
                "strategy": strategy,
            })))
        }

        Command::Personality => {
            let db = engine.lock().unwrap();
            let profile = db.get_personality()?;
            let traits: Vec<Value> = profile
                .traits
                .iter()
                .map(|t| json!({ "name": t.trait_name, "score": t.score }))
                .collect();
            Ok(CommandResult::Json(json!({ "traits": traits })))
        }

        Command::Stats => {
            let db = engine.lock().unwrap();
            let s = db.stats(None)?;
            Ok(CommandResult::Json(json!({
                "active_memories": s.active_memories,
                "consolidated_memories": s.consolidated_memories,
                "tombstoned_memories": s.tombstoned_memories,
                "edges": s.edges,
                "entities": s.entities,
                "operations": s.operations,
                "open_conflicts": s.open_conflicts,
                "pending_triggers": s.pending_triggers,
            })))
        }

        Command::CreateDb { name } => {
            let ctrl = control.ok_or_else(|| anyhow::anyhow!("no control db"))?;
            let c = ctrl.lock().unwrap();
            if c.database_exists(&name)? {
                anyhow::bail!("database '{}' already exists", name);
            }
            let id = c.create_database(&name, &name)?;
            Ok(CommandResult::Json(json!({
                "name": name,
                "message": format!("database '{}' created", name),
                "id": id,
            })))
        }

        Command::ListDb => {
            let ctrl = control.ok_or_else(|| anyhow::anyhow!("no control db"))?;
            let c = ctrl.lock().unwrap();
            let databases = c.list_databases()?;
            let list: Vec<Value> = databases
                .iter()
                .map(|d| {
                    json!({
                        "id": d.id,
                        "name": d.name,
                        "created_at": d.created_at,
                    })
                })
                .collect();
            Ok(CommandResult::Json(json!({ "databases": list })))
        }

        Command::Ping => Ok(CommandResult::Pong),
    }
}
