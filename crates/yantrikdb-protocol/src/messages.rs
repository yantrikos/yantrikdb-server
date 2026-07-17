//! Typed message payloads for each wire protocol command.
//!
//! All messages serialize to MessagePack via serde.
//! The server and client use these to construct and parse frame payloads.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Auth ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthRequest {
    pub token: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthOkResponse {
    pub database: String,
    pub database_id: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthFailResponse {
    pub reason: String,
}

// ── Database ──────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct SelectDbRequest {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateDbRequest {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DbOkResponse {
    pub name: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListDbResponse {
    pub databases: Vec<DatabaseInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatabaseInfo {
    pub id: i64,
    pub name: String,
    pub created_at: String,
}

// ── Remember ──────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct RememberRequest {
    pub text: String,
    #[serde(default = "default_memory_type")]
    pub memory_type: String,
    #[serde(default = "default_importance")]
    pub importance: f64,
    #[serde(default)]
    pub valence: f64,
    #[serde(default = "default_half_life")]
    pub half_life: f64,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub namespace: String,
    #[serde(default = "default_certainty")]
    pub certainty: f64,
    #[serde(default)]
    pub domain: String,
    #[serde(default = "default_source")]
    pub source: String,
    #[serde(default)]
    pub emotional_state: Option<String>,
    /// Client-provided embedding vector. If None, server computes it.
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

fn default_memory_type() -> String {
    "semantic".into()
}
fn default_importance() -> f64 {
    0.5
}
fn default_half_life() -> f64 {
    168.0
} // 7 days in hours
fn default_certainty() -> f64 {
    1.0
}
fn default_source() -> String {
    "user".into()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RememberOkResponse {
    pub rid: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RememberBatchRequest {
    pub memories: Vec<RememberRequest>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RememberBatchOkResponse {
    pub rids: Vec<String>,
}

// ── Recall ────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct RecallRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub include_consolidated: bool,
    #[serde(default = "default_true")]
    pub expand_entities: bool,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    /// Client-provided query embedding. If None, server computes it.
    #[serde(default)]
    pub query_embedding: Option<Vec<f32>>,
}

fn default_top_k() -> usize {
    10
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RecallResultMsg {
    pub rid: String,
    pub text: String,
    pub memory_type: String,
    pub score: f64,
    pub importance: f64,
    pub created_at: f64,
    pub why_retrieved: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub namespace: String,
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub certainty: f64,
    #[serde(default)]
    pub valence: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RecallEndMsg {
    pub total: usize,
    pub confidence: f64,
}

// ── Graph ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct RelateRequest {
    pub entity: String,
    pub target: String,
    pub relationship: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
}

fn default_weight() -> f64 {
    1.0
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RelateOkResponse {
    pub edge_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EdgesRequest {
    pub entity: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EdgesResultMsg {
    pub edges: Vec<EdgeMsg>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EdgeMsg {
    pub edge_id: String,
    pub src: String,
    pub dst: String,
    pub rel_type: String,
    pub weight: f64,
}

// ── Claims ────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct ClaimRequest {
    pub src: String,
    pub rel_type: String,
    pub dst: String,
    #[serde(default = "default_namespace")]
    pub namespace: String,
    #[serde(default = "default_polarity")]
    pub polarity: i32,
    #[serde(default = "default_modality")]
    pub modality: String,
    #[serde(default)]
    pub valid_from: Option<f64>,
    #[serde(default)]
    pub valid_to: Option<f64>,
    #[serde(default = "default_extractor")]
    pub extractor: String,
    #[serde(default)]
    pub extractor_version: Option<String>,
    #[serde(default = "default_confidence_band")]
    pub confidence_band: String,
    #[serde(default)]
    pub source_memory_rid: Option<String>,
    #[serde(default)]
    pub span_start: Option<i32>,
    #[serde(default)]
    pub span_end: Option<i32>,
    #[serde(default = "default_weight")]
    pub weight: f64,
}

fn default_polarity() -> i32 {
    1
}
fn default_modality() -> String {
    "asserted".into()
}
fn default_extractor() -> String {
    "manual".into()
}
fn default_confidence_band() -> String {
    "medium".into()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClaimOkResponse {
    pub claim_id: String,
    pub namespace: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClaimsRequest {
    pub entity: String,
    #[serde(default = "default_namespace")]
    pub namespace: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClaimsResultMsg {
    pub claims: Vec<ClaimMsg>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClaimMsg {
    pub claim_id: String,
    pub src: String,
    pub dst: String,
    pub rel_type: String,
    pub weight: f64,
    pub polarity: i32,
    pub modality: String,
    pub namespace: String,
    pub confidence_band: String,
}

// ── Alias ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct AliasRequest {
    pub alias: String,
    pub canonical_name: String,
    #[serde(default = "default_namespace")]
    pub namespace: String,
    #[serde(default = "default_alias_source")]
    pub source: String,
}

fn default_alias_source() -> String {
    "explicit".into()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AliasOkResponse {
    pub alias: String,
    pub canonical_name: String,
    pub namespace: String,
    pub added: bool,
}

// ── Forget ────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct ForgetRequest {
    pub rid: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForgetOkResponse {
    pub rid: String,
    pub found: bool,
}

// ── Session ───────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionStartRequest {
    #[serde(default = "default_namespace")]
    pub namespace: String,
    #[serde(default)]
    pub client_id: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

fn default_namespace() -> String {
    "default".into()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionEndRequest {
    pub session_id: String,
    #[serde(default)]
    pub summary: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionOkResponse {
    pub session_id: String,
    #[serde(default)]
    pub duration_secs: Option<f64>,
    #[serde(default)]
    pub memory_count: Option<i64>,
    #[serde(default)]
    pub topics: Option<Vec<String>>,
}

// ── Think ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct ThinkRequest {
    #[serde(default = "default_true")]
    pub run_consolidation: bool,
    #[serde(default = "default_true")]
    pub run_conflict_scan: bool,
    #[serde(default)]
    pub run_pattern_mining: bool,
    #[serde(default)]
    pub run_personality: bool,
    #[serde(default = "default_consolidation_limit")]
    pub consolidation_limit: usize,
}

fn default_consolidation_limit() -> usize {
    50
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ThinkResultMsg {
    pub consolidation_count: usize,
    pub conflicts_found: usize,
    pub patterns_new: usize,
    pub patterns_updated: usize,
    pub personality_updated: bool,
    pub duration_ms: u64,
    pub triggers: Vec<TriggerMsg>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TriggerMsg {
    pub trigger_type: String,
    pub reason: String,
    pub urgency: f64,
    pub source_rids: Vec<String>,
    pub suggested_action: String,
}

// ── Subscribe / Events ────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct SubscribeRequest {
    pub events: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UnsubscribeRequest {
    pub events: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EventMsg {
    pub event_type: String,
    pub data: serde_json::Value,
}

// ── Conflicts ─────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct ConflictsRequest {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub conflict_type: Option<String>,
    #[serde(default)]
    pub entity: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    50
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResolveRequest {
    pub conflict_id: String,
    pub strategy: String,
    #[serde(default)]
    pub winner_rid: Option<String>,
    #[serde(default)]
    pub new_text: Option<String>,
    #[serde(default)]
    pub resolution_note: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConflictMsg {
    pub conflict_id: String,
    pub conflict_type: String,
    pub priority: String,
    pub status: String,
    pub memory_a: String,
    pub memory_b: String,
    pub entity: Option<String>,
    pub detection_reason: String,
    pub detected_at: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConflictResultMsg {
    pub conflicts: Vec<ConflictMsg>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResolveOkResponse {
    pub conflict_id: String,
    pub strategy: String,
}

// ── Info ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct PersonalityResultMsg {
    pub traits: Vec<PersonalityTraitMsg>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PersonalityTraitMsg {
    pub name: String,
    pub score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StatsResultMsg {
    pub active_memories: i64,
    pub consolidated_memories: i64,
    pub tombstoned_memories: i64,
    pub edges: i64,
    pub entities: i64,
    pub operations: i64,
    pub open_conflicts: i64,
    pub pending_triggers: i64,
}

// ── Error ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub code: u16,
    pub message: String,
    #[serde(default)]
    pub details: Option<HashMap<String, serde_json::Value>>,
}

// ── Cluster / Replication ─────────────────────────────────────────

/// Oplog format version. Bumped when the oplog entry payload encoding
/// changes in a backward-incompatible way. The invariant: newer code must
/// always be able to apply entries from ANY older format version, via
/// registered migration functions. The reverse is NOT required — old code
/// may reject entries it doesn't understand.
///
/// Version history:
///   1 — initial (v0.5.0 through v0.5.10). Payload is free-form JSON per
///       op_type. No format constraints beyond valid JSON.
///   2 — adds `embedding` (engine v0.10 Item 3). Carries the origin's exact
///       embedding bytes for a text-changing `correct` so a follower applies
///       the identical vector instead of re-embedding (which diverges by
///       model version / quantization). Additive + `serde(default)`, so a v1
///       entry decodes as `embedding: None`.
///
/// NOTE (honest limitation): nothing currently *validates* `format_version` —
/// it is stamped by `entry_to_wire` and read by no one, and there is no
/// migration registry. So this bump documents the change but enforces nothing:
/// a v1 peer receiving a v2 entry decodes it fine (serde ignores unknown
/// fields) and silently drops `embedding`, which re-opens the exact divergence
/// Item 3 closes. The Item 3 guarantee therefore only holds when BOTH ends run
/// v2+. A reader-side version gate is the follow-up if we ever run mixed-
/// version peer sync for real.
pub const OPLOG_FORMAT_VERSION: u32 = 2;

/// Wire protocol version. Bumped when the handshake, frame format, or oplog
/// entry encoding changes in a backward-incompatible way.
///
/// Compatibility rule: a node accepts a peer whose `protocol_version` equals
/// its own. Future versions may widen this to a range (e.g. "accept v1–v2")
/// via a negotiation table, but strict equality is the safe default until we
/// have a second version to negotiate with.
///
/// Version history:
///   1 — initial (v0.5.0 through v0.5.9)
pub const PROTOCOL_VERSION: u32 = 1;

/// Initial peer-to-peer handshake.
#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterHello {
    pub node_id: u32,
    pub role: String, // "voter" | "read_replica" | "witness"
    pub current_term: u64,
    pub cluster_secret: String,
    pub advertise_addr: String,
    /// Wire protocol version. Added in v0.5.10. Older nodes that don't send
    /// this field default to 0 (treated as v1 for backward compatibility).
    #[serde(default)]
    pub protocol_version: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterHelloOk {
    pub node_id: u32,
    pub role: String,
    pub current_term: u64,
    pub leader_id: Option<u32>,
    /// Echoed back so the initiator can verify version agreement.
    #[serde(default)]
    pub protocol_version: u32,
}

/// Request operations from a peer's oplog since a watermark.
#[derive(Debug, Serialize, Deserialize)]
pub struct OplogPullRequest {
    /// Database name to pull from. Defaults to "default" if missing.
    #[serde(default = "default_db_name")]
    pub database: String,
    pub since_hlc: Option<Vec<u8>>, // 16-byte HLC timestamp, None for "from beginning"
    pub since_op_id: Option<String>,
    pub limit: usize,                  // max ops per batch
    pub exclude_actor: Option<String>, // skip ops from this actor (avoid loops)
}

fn default_db_name() -> String {
    "default".into()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OplogPullResult {
    pub ops: Vec<OplogEntryWire>,
    pub has_more: bool,
}

/// Wire-friendly representation of an oplog entry.
#[derive(Debug, Serialize, Deserialize)]
pub struct OplogEntryWire {
    pub op_id: String,
    pub op_type: String,
    pub timestamp: f64,
    pub target_rid: Option<String>,
    pub payload: serde_json::Value,
    pub actor_id: String,
    pub hlc: Vec<u8>,
    pub embedding_hash: Option<Vec<u8>>,
    pub origin_actor: String,
    /// Oplog format version. Added in v0.5.11. Older entries without this
    /// field default to 0, treated as version 1. See OPLOG_FORMAT_VERSION.
    #[serde(default)]
    pub format_version: u32,
    /// Format v2 (engine v0.10 Item 3): the origin's exact embedding bytes for
    /// a text-changing `correct` op, so a follower applies the identical vector
    /// rather than re-embedding it — re-embedding diverges by model version /
    /// quantization, which is precisely the failure this closes.
    ///
    /// `Some` only on text-changing corrections; `None` for every other op.
    /// Encrypted under the origin DEK — usable on a same-DEK follower, which is
    /// the assumption the encrypted-replication path already makes for
    /// text/metadata. Mirrors `yantrikdb::OplogEntry::embedding`.
    ///
    /// `serde(default)` keeps v1 entries decodable (→ `None`).
    #[serde(default)]
    pub embedding: Option<Vec<u8>>,
}

/// Push ops to a peer (used by primary → secondary push).
#[derive(Debug, Serialize, Deserialize)]
pub struct OplogPushRequest {
    #[serde(default = "default_db_name")]
    pub database: String,
    pub ops: Vec<OplogEntryWire>,
}

/// Request the list of databases on a peer (so we can create matching ones).
#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterDatabaseListRequest {}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterDatabaseListResponse {
    pub databases: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OplogPushOkResponse {
    pub applied: usize,
    pub last_hlc: Vec<u8>,
    pub last_op_id: String,
}

/// Heartbeat from leader to followers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatMsg {
    pub term: u64,
    pub leader_id: u32,
    pub leader_last_hlc: Vec<u8>,
    pub leader_last_op_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HeartbeatAckMsg {
    pub term: u64,
    pub follower_id: u32,
    pub follower_role: String,
    pub follower_last_hlc: Vec<u8>,
    pub follower_last_op_id: String,
    pub lag_seconds: f64,
}

/// Vote request from a candidate during election.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteMsg {
    pub term: u64,
    pub candidate_id: u32,
    pub last_log_hlc: Vec<u8>,
    pub last_log_op_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VoteResponseMsg {
    pub term: u64,
    pub voter_id: u32,
    pub granted: bool,
    pub reason: Option<String>,
}

/// Cluster overview request.
#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterStatusResultMsg {
    pub current_term: u64,
    pub leader_id: Option<u32>,
    pub self_id: u32,
    pub self_role: String,
    pub peers: Vec<PeerStatusMsg>,
    pub quorum_size: usize,
    pub healthy: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PeerStatusMsg {
    pub node_id: u32,
    pub addr: String,
    pub role: String,
    pub reachable: bool,
    pub current_term: u64,
    pub last_seen_secs_ago: f64,
    pub lag_seconds: f64,
}

// ── Error codes ───────────────────────────────────────────────────

pub mod error_codes {
    pub const AUTH_REQUIRED: u16 = 1000;
    pub const AUTH_INVALID: u16 = 1001;
    pub const DB_NOT_FOUND: u16 = 2000;
    pub const DB_ALREADY_EXISTS: u16 = 2001;
    pub const MEMORY_NOT_FOUND: u16 = 3000;
    pub const INVALID_PAYLOAD: u16 = 4000;
    pub const INTERNAL_ERROR: u16 = 5000;
    pub const EMBEDDING_ERROR: u16 = 5001;
    // Cluster errors
    pub const READONLY_NODE: u16 = 6000; // Can't write to read replica
    pub const NOT_LEADER: u16 = 6001; // Try the current leader instead
    pub const NO_QUORUM: u16 = 6002; // Cluster lost quorum
    pub const CLUSTER_SECRET_MISMATCH: u16 = 6003;
    pub const PEER_TERM_MISMATCH: u16 = 6004;
}

#[cfg(test)]
mod oplog_wire_v2_tests {
    use super::*;

    fn v2_entry(embedding: Option<Vec<u8>>) -> OplogEntryWire {
        OplogEntryWire {
            op_id: "op-1".into(),
            op_type: "correct".into(),
            timestamp: 1.0,
            target_rid: Some("rid-1".into()),
            payload: serde_json::json!({"text": "corrected"}),
            actor_id: "actor-a".into(),
            hlc: vec![1, 2, 3],
            embedding_hash: Some(vec![9, 9]),
            origin_actor: "origin-a".into(),
            format_version: OPLOG_FORMAT_VERSION,
            embedding,
        }
    }

    /// Oplog format v2 (engine v0.10 Item 3). The guarantee: the origin's exact
    /// embedding bytes reach the follower so it applies the identical vector
    /// instead of re-embedding — re-embedding diverges by model version /
    /// quantization, which is the failure this closes.
    ///
    /// Assert the CONTENT survives the serde boundary, not merely that the
    /// struct compiles: a dropped field compiles perfectly and voids the
    /// guarantee silently, which is exactly how this gap hid before.
    #[test]
    fn embedding_bytes_survive_serde_roundtrip() {
        let bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let wire = v2_entry(Some(bytes.clone()));

        let json = serde_json::to_string(&wire).expect("serialize");
        // Must be the exact key. A bare `contains("embedding")` is satisfied by
        // the pre-existing `embedding_hash` field, so it would pass even if
        // `embedding` were serde-skipped — a false-positive guard on precisely
        // the regression this test exists to catch (caught in review by
        // yantrikdb-core).
        assert!(
            json.contains("\"embedding\":"),
            "v2 wire encoding must carry the `embedding` key itself \
             (not just `embedding_hash`); got: {json}"
        );

        let decoded: OplogEntryWire = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            decoded.embedding,
            Some(bytes),
            "embedding bytes must survive the peer-sync serde hop"
        );
    }

    /// Non-correct ops carry no embedding. `None` must stay `None` — not
    /// become an empty vec, which means something different downstream.
    #[test]
    fn absent_embedding_stays_none() {
        let json = serde_json::to_string(&v2_entry(None)).expect("serialize");
        let decoded: OplogEntryWire = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.embedding, None);
    }

    /// Backward compat: a v1 peer's entry has no `embedding` key at all. It
    /// must still decode (serde default -> None) rather than fail the whole
    /// sync batch.
    #[test]
    fn v1_entry_without_embedding_field_still_decodes() {
        let v1_json = serde_json::json!({
            "op_id": "op-1",
            "op_type": "record",
            "timestamp": 1.0,
            "target_rid": "rid-1",
            "payload": {"text": "hi"},
            "actor_id": "actor-a",
            "hlc": [1, 2, 3],
            "embedding_hash": null,
            "origin_actor": "origin-a",
            "format_version": 1
        })
        .to_string();

        let decoded: OplogEntryWire =
            serde_json::from_str(&v1_json).expect("v1 entry must decode under the v2 schema");
        assert_eq!(decoded.embedding, None);
        assert_eq!(decoded.format_version, 1);
    }

    /// Pin the version constant: v2 is the format that carries `embedding`.
    #[test]
    fn oplog_format_version_is_2() {
        assert_eq!(OPLOG_FORMAT_VERSION, 2);
    }
}
