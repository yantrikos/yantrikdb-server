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
        /// v0.10: re-admit superseded records. Default `false` = current-value-
        /// by-default (the supersession-aware behavior). `true` is for
        /// history/archaeology — surfaces stale records stamped with their
        /// superseded status.
        include_superseded: bool,
        /// Handoff 1b fix 2: minimum certainty filter (engine `recall`
        /// `certainty_min`, issue #46). `None` = no filter (pre-existing
        /// behavior).
        certainty_min: Option<f64>,
        /// Handoff 1b fix 2: when `true` the recall does NOT reinforce
        /// (mutate `access_count` / `last_access`) the returned rows.
        /// Probe/eval recalls send this so measurement doesn't perturb
        /// the tuning signal — on followers a reinforcing recall is also
        /// an unreplicated local write.
        skip_reinforce: bool,
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
        /// Handoff 1b fix 3: the remaining `ThinkConfig` knobs HTTP clients
        /// (yantrikdb-mcp `think()`) send. `None` = engine default, so the
        /// wire-protocol path (which doesn't carry them yet) is unchanged.
        consolidation_time_window_days: Option<f64>,
        min_active_memories: Option<i64>,
        max_triggers: Option<usize>,
    },

    // ── Conflicts ─────────────────────────────────────────
    Conflicts {
        status: Option<String>,
        conflict_type: Option<String>,
        entity: Option<String>,
        namespace: Option<String>,
        limit: usize,
    },
    Resolve {
        conflict_id: String,
        strategy: String,
        winner_rid: Option<String>,
        new_text: Option<String>,
        resolution_note: Option<String>,
    },

    // ── Claims (RFC 006 Phase 5) ────────────────────────
    GetClaims {
        entity: String,
        namespace: Option<String>,
    },
    IngestClaim {
        src: String,
        rel_type: String,
        dst: String,
        namespace: String,
        polarity: i32,
        modality: String,
        valid_from: Option<f64>,
        valid_to: Option<f64>,
        extractor: String,
        extractor_version: Option<String>,
        confidence_band: String,
        source_memory_rid: Option<String>,
        span_start: Option<i32>,
        span_end: Option<i32>,
        weight: f64,
    },
    AddAlias {
        alias: String,
        canonical_name: String,
        namespace: String,
        source: String,
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

    // ── RFC 008: Warrant Flow substrate (M1-M10) ────────
    /// Ingest a claim with explicit source_lineage. Extends IngestClaim
    /// with the one field that makes ⊕'s dependence discount work on
    /// real data. Without source_lineage populated, mobility_state is
    /// just a fancy polarity counter.
    IngestClaimWithLineage {
        src: String,
        rel_type: String,
        dst: String,
        namespace: String,
        polarity: i32,
        modality: String,
        valid_from: Option<f64>,
        valid_to: Option<f64>,
        extractor: String,
        extractor_version: Option<String>,
        confidence_band: String,
        source_memory_rid: Option<String>,
        weight: f64,
        source_lineage: Vec<String>,
    },
    /// Read mobility_state (13-dim M(c|ρ)) for a proposition identified
    /// by its (src, rel_type, dst, namespace) triple in a given regime.
    GetMobility {
        src: String,
        rel_type: String,
        dst: String,
        namespace: String,
        regime: String,
    },
    /// Read contest_state (Γ(c) — grounded contest diagnostics) for the
    /// same triple + regime.
    GetContest {
        src: String,
        rel_type: String,
        dst: String,
        namespace: String,
        regime: String,
    },
    /// Record a cognitive move. The agent invokes this whenever its
    /// reasoning transforms one set of claims into another.
    RecordMoveEvent {
        move_type: String,
        operator_version: String,
        context_regime: Option<String>,
        observability: String,
        inference_confidence: Option<f64>,
        inference_basis: Option<Vec<String>>,
        input_claim_ids: Vec<String>,
        output_claim_ids: Vec<String>,
        side_effect_claim_ids: Vec<String>,
        dependencies: Vec<String>,
    },
    /// List propositions where contest_state.heuristic_flags intersects
    /// the requested mask. Primary audit entry point.
    ListFlaggedPropositions {
        flag_mask: u64,
        limit: usize,
    },
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
    /// Handoff 1b fix 4: caller-supplied EVENT time in epoch seconds
    /// (engine 0.14 historical-import precondition). `None` = the server
    /// stamps now() — pre-existing behavior.
    pub created_at: Option<f64>,
}
