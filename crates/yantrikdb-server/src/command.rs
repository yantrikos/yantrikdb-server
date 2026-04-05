//! Internal Command enum — the unified representation that both the wire
//! protocol handler and HTTP gateway produce. The handler module executes
//! these against the engine.

use serde_json::Value;

/// A database command ready for execution.
#[derive(Debug)]
pub enum Command {
    // ── Memory ────────────────────────────────────────────
    Remember {
        text: String,
        memory_type: String,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: Value,
        namespace: String,
        certainty: f64,
        domain: String,
        source: String,
        emotional_state: Option<String>,
        embedding: Option<Vec<f32>>,
    },
    RememberBatch {
        memories: Vec<RememberInput>,
    },
    Recall {
        query: String,
        top_k: usize,
        memory_type: Option<String>,
        include_consolidated: bool,
        expand_entities: bool,
        namespace: Option<String>,
        domain: Option<String>,
        source: Option<String>,
        query_embedding: Option<Vec<f32>>,
    },
    Forget {
        rid: String,
    },

    // ── Graph ─────────────────────────────────────────────
    Relate {
        entity: String,
        target: String,
        relationship: String,
        weight: f64,
    },
    Edges {
        entity: String,
    },

    // ── Session ───────────────────────────────────────────
    SessionStart {
        namespace: String,
        client_id: String,
        metadata: Value,
    },
    SessionEnd {
        session_id: String,
        summary: Option<String>,
    },

    // ── Cognition ─────────────────────────────────────────
    Think {
        run_consolidation: bool,
        run_conflict_scan: bool,
        run_pattern_mining: bool,
        run_personality: bool,
        consolidation_limit: usize,
    },

    // ── Conflicts ─────────────────────────────────────────
    Conflicts {
        status: Option<String>,
        conflict_type: Option<String>,
        entity: Option<String>,
        limit: usize,
    },
    Resolve {
        conflict_id: String,
        strategy: String,
        winner_rid: Option<String>,
        new_text: Option<String>,
        resolution_note: Option<String>,
    },

    // ── Info ──────────────────────────────────────────────
    Personality,
    Stats,

    // ── Database management ──────────────────────────────
    CreateDb {
        name: String,
    },
    ListDb,

    // ── Control ──────────────────────────────────────────
    Ping,
}

#[derive(Debug)]
pub struct RememberInput {
    pub text: String,
    pub memory_type: String,
    pub importance: f64,
    pub valence: f64,
    pub half_life: f64,
    pub metadata: Value,
    pub namespace: String,
    pub certainty: f64,
    pub domain: String,
    pub source: String,
    pub emotional_state: Option<String>,
    pub embedding: Option<Vec<f32>>,
}
