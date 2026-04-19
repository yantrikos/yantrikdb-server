//! Command handler — executes Commands against a YantrikDB engine instance.

use parking_lot::Mutex;
use std::sync::Arc;

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

/// Execute a command against the given engine. Acquires the lock internally.
/// Used by the wire protocol path.
pub fn execute(
    engine: &Arc<Mutex<YantrikDB>>,
    cmd: Command,
    control: Option<&Mutex<ControlDb>>,
) -> anyhow::Result<CommandResult> {
    let db = engine.lock();
    execute_with_guard(db, cmd, control)
}

/// Execute a command with a pre-acquired engine guard. Used by the HTTP
/// gateway path which measures lock acquisition time separately.
pub fn execute_with_guard(
    db: parking_lot::MutexGuard<'_, YantrikDB>,
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
            } else if namespace.is_some()
                || domain.is_some()
                || source.is_some()
                || memory_type.is_some()
            {
                // Any filter set — must go through the full recall() path.
                // recall_text_filtered silently drops namespace/memory_type.
                let emb = db.embed(&query)?;
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
                db.recall_text(&query, top_k)?
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
            let found = db.forget(&rid)?;
            Ok(CommandResult::Json(json!({ "rid": rid, "found": found })))
        }

        Command::Relate {
            entity,
            target,
            relationship,
            weight,
        } => {
            let edge_id = db.relate(&entity, &target, &relationship, weight)?;
            Ok(CommandResult::Json(json!({ "edge_id": edge_id })))
        }

        Command::Edges { entity } => {
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
            let session_id = db.session_start(&namespace, &client_id, &metadata)?;
            Ok(CommandResult::Json(json!({ "session_id": session_id })))
        }

        Command::SessionEnd {
            session_id,
            summary,
        } => {
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
            namespace,
            limit,
        } => {
            let conflicts = db.get_conflicts(
                status.as_deref(),
                conflict_type.as_deref(),
                entity.as_deref(),
                None, // priority
                namespace.as_deref(),
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

        Command::GetClaims { entity, namespace } => {
            let claims = db.get_claims(&entity, namespace.as_deref())?;
            Ok(CommandResult::Json(json!({ "claims": claims })))
        }

        Command::IngestClaim {
            src,
            rel_type,
            dst,
            namespace,
            polarity,
            modality,
            valid_from,
            valid_to,
            extractor,
            extractor_version,
            confidence_band,
            source_memory_rid,
            span_start,
            span_end,
            weight,
        } => {
            let claim_id = db.ingest_claim(
                &src,
                &rel_type,
                &dst,
                &namespace,
                polarity,
                &modality,
                valid_from,
                valid_to,
                &extractor,
                extractor_version.as_deref(),
                &confidence_band,
                source_memory_rid.as_deref(),
                span_start,
                span_end,
                weight,
            )?;
            Ok(CommandResult::Json(json!({
                "claim_id": claim_id,
                "namespace": namespace,
            })))
        }

        Command::AddAlias {
            alias,
            canonical_name,
            namespace,
            source,
        } => {
            let added = db.add_entity_alias(&alias, &canonical_name, &namespace, &source)?;
            Ok(CommandResult::Json(json!({
                "alias": alias,
                "canonical_name": canonical_name,
                "namespace": namespace,
                "added": added,
            })))
        }

        Command::Personality => {
            let profile = db.get_personality()?;
            let traits: Vec<Value> = profile
                .traits
                .iter()
                .map(|t| json!({ "name": t.trait_name, "score": t.score }))
                .collect();
            Ok(CommandResult::Json(json!({ "traits": traits })))
        }

        Command::Stats => {
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
            let c = ctrl.lock();
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
            let c = ctrl.lock();
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

        // ── RFC 008 substrate surface ──────────────────────────────

        Command::IngestClaimWithLineage {
            src,
            rel_type,
            dst,
            namespace,
            polarity,
            modality,
            valid_from,
            valid_to,
            extractor,
            extractor_version,
            confidence_band,
            source_memory_rid,
            weight,
            source_lineage,
        } => {
            // Insert the claim via the standard ingest path (which fires
            // the M3 write-tier mobility hook and M4 contest hook), then
            // patch source_lineage on the just-inserted row and re-trigger
            // the substrate recompute so the leave-one-out ω_k picks up
            // the correct lineage. This two-step pattern exists because
            // the yantrikdb 0.x ingest_claim API doesn't yet take lineage
            // as a parameter; revisit when that argument lands.
            let claim_id = db.ingest_claim(
                &src,
                &rel_type,
                &dst,
                &namespace,
                polarity,
                &modality,
                valid_from,
                valid_to,
                &extractor,
                extractor_version.as_deref(),
                &confidence_band,
                source_memory_rid.as_deref(),
                None,
                None,
                weight,
            )?;
            let lineage_json = serde_json::to_string(&source_lineage)
                .unwrap_or_else(|_| "[]".into());
            db.conn().execute(
                "UPDATE claims SET source_lineage = ?1 WHERE claim_id = ?2",
                rusqlite::params![lineage_json, claim_id],
            )?;
            // Resolve proposition_id to re-fire the substrate recompute.
            let prop_id: String = db.conn().query_row(
                "SELECT proposition_id FROM claims WHERE claim_id = ?1",
                rusqlite::params![claim_id],
                |row| row.get(0),
            )?;
            db.compute_write_tier_mobility(&prop_id, "default")?;
            db.compute_contest_state(&prop_id, "default")?;
            Ok(CommandResult::Json(json!({
                "claim_id": claim_id,
                "proposition_id": prop_id,
                "namespace": namespace,
                "source_lineage": source_lineage,
            })))
        }

        Command::GetMobility {
            src,
            rel_type,
            dst,
            namespace,
            regime,
        } => {
            let prop_id: Option<String> = db
                .conn()
                .query_row(
                    "SELECT proposition_id FROM propositions \
                     WHERE src = ?1 AND rel_type = ?2 AND dst = ?3 AND namespace = ?4",
                    rusqlite::params![src, rel_type, dst, namespace],
                    |row| row.get(0),
                )
                .ok();
            let Some(prop_id) = prop_id else {
                return Ok(CommandResult::Json(json!({
                    "mobility_state": null,
                    "proposition_id": null,
                    "reason": "proposition not found — no claims ingested for this triple yet",
                })));
            };
            let state = db.get_mobility_state(&prop_id, &regime)?;
            Ok(CommandResult::Json(json!({
                "proposition_id": prop_id,
                "regime": regime,
                "mobility_state": state,
            })))
        }

        Command::GetContest {
            src,
            rel_type,
            dst,
            namespace,
            regime,
        } => {
            let prop_id: Option<String> = db
                .conn()
                .query_row(
                    "SELECT proposition_id FROM propositions \
                     WHERE src = ?1 AND rel_type = ?2 AND dst = ?3 AND namespace = ?4",
                    rusqlite::params![src, rel_type, dst, namespace],
                    |row| row.get(0),
                )
                .ok();
            let Some(prop_id) = prop_id else {
                return Ok(CommandResult::Json(json!({
                    "contest_state": null,
                    "proposition_id": null,
                    "reason": "proposition not found",
                })));
            };
            let state = db.get_contest_state(&prop_id, &regime)?;
            let report = db.inspect_contest_conflicts(&prop_id, &regime)?;
            Ok(CommandResult::Json(json!({
                "proposition_id": prop_id,
                "regime": regime,
                "contest_state": state,
                "exemplar_pairs": report,
            })))
        }

        Command::RecordMoveEvent {
            move_type,
            operator_version,
            context_regime,
            observability,
            inference_confidence,
            inference_basis,
            input_claim_ids,
            output_claim_ids,
            side_effect_claim_ids,
            dependencies,
        } => {
            use yantrikdb::engine::moves::{ClaimRef, RecordMoveEventInput, SideEffectRef};
            let input = RecordMoveEventInput {
                move_type,
                operator_version,
                context_regime,
                observability,
                inference_confidence,
                inference_basis,
                dependencies,
                inputs: input_claim_ids
                    .into_iter()
                    .enumerate()
                    .map(|(i, cid)| ClaimRef {
                        claim_id: cid,
                        role: "input".into(),
                        ordinal: i as i64,
                    })
                    .collect(),
                outputs: output_claim_ids
                    .into_iter()
                    .enumerate()
                    .map(|(i, cid)| ClaimRef {
                        claim_id: cid,
                        role: "output".into(),
                        ordinal: i as i64,
                    })
                    .collect(),
                side_effects: side_effect_claim_ids
                    .into_iter()
                    .map(|cid| SideEffectRef {
                        claim_id: cid,
                        effect_kind: "generic".into(),
                    })
                    .collect(),
                ..Default::default()
            };
            let move_id = db.record_move_event(input)?;
            Ok(CommandResult::Json(json!({ "move_id": move_id })))
        }

        Command::ListFlaggedPropositions { flag_mask, limit } => {
            let flagged = db.list_flagged_propositions(flag_mask, limit)?;
            let count = flagged.len();
            Ok(CommandResult::Json(json!({
                "flagged_propositions": flagged,
                "flag_mask": flag_mask,
                "count": count,
            })))
        }
    }
}
