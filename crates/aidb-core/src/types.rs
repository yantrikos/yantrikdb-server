use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A memory record returned by get() and recall().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub rid: String,
    pub memory_type: String,
    pub text: String,
    pub created_at: f64,
    pub importance: f64,
    pub valence: f64,
    pub half_life: f64,
    pub last_access: f64,
    pub consolidation_status: String,
    pub storage_tier: String,
    pub consolidated_into: Option<String>,
    pub metadata: serde_json::Value,
}

/// Score breakdown for a recall result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub similarity: f64,
    pub decay: f64,
    pub recency: f64,
    pub importance: f64,
    pub graph_proximity: f64,
}

/// A recall result with scoring information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub rid: String,
    pub memory_type: String,
    pub text: String,
    pub created_at: f64,
    pub importance: f64,
    pub valence: f64,
    pub score: f64,
    pub scores: ScoreBreakdown,
    pub why_retrieved: Vec<String>,
    pub metadata: serde_json::Value,
}

/// An edge in the entity graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub edge_id: String,
    pub src: String,
    pub dst: String,
    pub rel_type: String,
    pub weight: f64,
}

/// Engine statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub active_memories: i64,
    pub consolidated_memories: i64,
    pub tombstoned_memories: i64,
    pub archived_memories: i64,
    pub edges: i64,
    pub entities: i64,
    pub operations: i64,
    pub open_conflicts: i64,
    pub resolved_conflicts: i64,
    pub pending_triggers: i64,
    pub active_patterns: i64,
    pub scoring_cache_entries: usize,
    pub vec_index_entries: usize,
}

/// A proactive trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trigger {
    pub trigger_type: String,
    pub reason: String,
    pub urgency: f64,
    pub source_rids: Vec<String>,
    pub suggested_action: String,
    pub context: HashMap<String, serde_json::Value>,
}

/// Consolidation result (after consolidation runs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationResult {
    pub consolidated_rid: String,
    pub source_rids: Vec<String>,
    pub cluster_size: usize,
    pub summary: String,
    pub importance: f64,
    pub entities_linked: usize,
}

/// Dry run consolidation preview.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationPreview {
    pub cluster_size: usize,
    pub texts: Vec<String>,
    pub preview_summary: String,
    pub source_rids: Vec<String>,
}

/// Internal struct with embedding data for clustering.
#[derive(Debug, Clone)]
pub struct MemoryWithEmbedding {
    pub rid: String,
    pub memory_type: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub created_at: f64,
    pub importance: f64,
    pub valence: f64,
    pub half_life: f64,
    pub last_access: f64,
    pub metadata: serde_json::Value,
}

/// A decayed memory candidate from decay().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayedMemory {
    pub rid: String,
    pub text: String,
    pub memory_type: String,
    pub original_importance: f64,
    pub current_score: f64,
    pub days_since_access: f64,
}

/// Lightweight scoring fields cached in memory for fast recall scoring.
/// These are the only fields needed to compute composite_score() during recall.
#[derive(Debug, Clone)]
pub struct ScoringRow {
    pub created_at: f64,
    pub importance: f64,
    pub half_life: f64,
    pub last_access: f64,
    pub valence: f64,
    pub consolidation_status: String,
    pub memory_type: String,
}

/// Input for batch record operations.
#[derive(Debug, Clone)]
pub struct RecordInput {
    pub text: String,
    pub memory_type: String,
    pub importance: f64,
    pub valence: f64,
    pub half_life: f64,
    pub metadata: serde_json::Value,
    pub embedding: Vec<f32>,
}

// ── Conflict types (V2) ──

/// The type of semantic conflict between two memories.
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictType {
    IdentityFact,
    Preference,
    Temporal,
    Consolidation,
    Minor,
}

impl ConflictType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ConflictType::IdentityFact => "identity_fact",
            ConflictType::Preference => "preference",
            ConflictType::Temporal => "temporal",
            ConflictType::Consolidation => "consolidation",
            ConflictType::Minor => "minor",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "identity_fact" => ConflictType::IdentityFact,
            "preference" => ConflictType::Preference,
            "temporal" => ConflictType::Temporal,
            "consolidation" => ConflictType::Consolidation,
            _ => ConflictType::Minor,
        }
    }

    pub fn default_priority(&self) -> &'static str {
        match self {
            ConflictType::IdentityFact => "critical",
            ConflictType::Preference => "high",
            ConflictType::Temporal => "high",
            ConflictType::Consolidation => "medium",
            ConflictType::Minor => "low",
        }
    }
}

/// A conflict between two memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    pub conflict_id: String,
    pub conflict_type: String,
    pub priority: String,
    pub status: String,
    pub memory_a: String,
    pub memory_b: String,
    pub entity: Option<String>,
    pub rel_type: Option<String>,
    pub detected_at: f64,
    pub detected_by: String,
    pub detection_reason: String,
    pub resolved_at: Option<f64>,
    pub resolved_by: Option<String>,
    pub strategy: Option<String>,
    pub winner_rid: Option<String>,
    pub resolution_note: Option<String>,
}

/// Result of a conflict resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolutionResult {
    pub conflict_id: String,
    pub strategy: String,
    pub winner_rid: Option<String>,
    pub loser_tombstoned: bool,
    pub new_memory_rid: Option<String>,
}

/// Result of a user-initiated correction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionResult {
    pub original_rid: String,
    pub corrected_rid: String,
    pub original_tombstoned: bool,
}

// ── Cognition types (V3) ──

/// Trigger type classification.
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerType {
    DecayReview,
    ConsolidationReady,
    ConflictEscalation,
    TemporalDrift,
    Redundancy,
    RelationshipInsight,
    ValenceTrend,
    EntityAnomaly,
    PatternDiscovered,
}

impl TriggerType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TriggerType::DecayReview => "decay_review",
            TriggerType::ConsolidationReady => "consolidation_ready",
            TriggerType::ConflictEscalation => "conflict_escalation",
            TriggerType::TemporalDrift => "temporal_drift",
            TriggerType::Redundancy => "redundancy",
            TriggerType::RelationshipInsight => "relationship_insight",
            TriggerType::ValenceTrend => "valence_trend",
            TriggerType::EntityAnomaly => "entity_anomaly",
            TriggerType::PatternDiscovered => "pattern_discovered",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "decay_review" => TriggerType::DecayReview,
            "consolidation_ready" => TriggerType::ConsolidationReady,
            "conflict_escalation" => TriggerType::ConflictEscalation,
            "temporal_drift" => TriggerType::TemporalDrift,
            "redundancy" => TriggerType::Redundancy,
            "relationship_insight" => TriggerType::RelationshipInsight,
            "valence_trend" => TriggerType::ValenceTrend,
            "entity_anomaly" => TriggerType::EntityAnomaly,
            "pattern_discovered" => TriggerType::PatternDiscovered,
            _ => TriggerType::DecayReview,
        }
    }

    pub fn default_cooldown_secs(&self) -> f64 {
        match self {
            TriggerType::DecayReview => 86400.0 * 3.0,
            TriggerType::ConsolidationReady => 86400.0,
            TriggerType::ConflictEscalation => 86400.0 * 2.0,
            TriggerType::TemporalDrift => 86400.0 * 14.0,
            TriggerType::Redundancy => 86400.0,
            TriggerType::RelationshipInsight => 86400.0 * 7.0,
            TriggerType::ValenceTrend => 86400.0 * 7.0,
            TriggerType::EntityAnomaly => 86400.0 * 7.0,
            TriggerType::PatternDiscovered => 86400.0 * 7.0,
        }
    }

    pub fn default_expiry_secs(&self) -> f64 {
        match self {
            TriggerType::DecayReview => 86400.0 * 7.0,
            TriggerType::ConsolidationReady => 86400.0 * 3.0,
            TriggerType::ConflictEscalation => 86400.0 * 14.0,
            _ => 86400.0 * 7.0,
        }
    }
}

/// Configuration for the think() cognition loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkConfig {
    pub importance_threshold: f64,
    pub decay_threshold: f64,
    pub max_triggers: usize,
    pub run_consolidation: bool,
    pub run_conflict_scan: bool,
    pub run_pattern_mining: bool,
    pub consolidation_sim_threshold: f64,
    pub consolidation_time_window_days: f64,
    pub consolidation_min_cluster: usize,
    pub min_active_memories: i64,
}

impl Default for ThinkConfig {
    fn default() -> Self {
        Self {
            importance_threshold: 0.5,
            decay_threshold: 0.1,
            max_triggers: 10,
            run_consolidation: true,
            run_conflict_scan: true,
            run_pattern_mining: true,
            consolidation_sim_threshold: 0.6,
            consolidation_time_window_days: 7.0,
            consolidation_min_cluster: 2,
            min_active_memories: 10,
        }
    }
}

/// Result of a think() pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkResult {
    pub triggers: Vec<Trigger>,
    pub consolidation_count: usize,
    pub conflicts_found: usize,
    pub patterns_new: usize,
    pub patterns_updated: usize,
    pub expired_triggers: usize,
    pub duration_ms: u64,
}

/// A persisted trigger with lifecycle state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedTrigger {
    pub trigger_id: String,
    pub trigger_type: String,
    pub urgency: f64,
    pub status: String,
    pub reason: String,
    pub suggested_action: String,
    pub source_rids: Vec<String>,
    pub context: serde_json::Value,
    pub created_at: f64,
    pub delivered_at: Option<f64>,
    pub acknowledged_at: Option<f64>,
    pub acted_at: Option<f64>,
    pub expires_at: Option<f64>,
}

/// A detected pattern across memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub pattern_id: String,
    pub pattern_type: String,
    pub status: String,
    pub confidence: f64,
    pub description: String,
    pub evidence_rids: Vec<String>,
    pub entity_names: Vec<String>,
    pub context: serde_json::Value,
    pub first_seen: f64,
    pub last_confirmed: f64,
    pub occurrence_count: i64,
}

/// Result of pattern mining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMiningResult {
    pub new_patterns: usize,
    pub updated_patterns: usize,
    pub stale_patterns: usize,
}

/// Configuration for pattern mining.
#[derive(Debug, Clone)]
pub struct PatternConfig {
    pub co_occurrence_min_count: usize,
    pub temporal_cluster_min_events: usize,
    pub valence_trend_delta_threshold: f64,
    pub topic_cluster_sim_threshold: f64,
    pub topic_cluster_time_window_days: f64,
    pub entity_hub_min_degree: usize,
    pub max_patterns: usize,
}

// ── Profiling types (feature-gated) ──

/// Timing breakdown for a single recall() invocation.
#[cfg(feature = "profiling")]
#[derive(Debug, Clone)]
pub struct RecallTimings {
    pub vec_search_ms: f64,
    pub cache_score_ms: f64,
    pub fetch_ms: f64,
    pub scoring_ms: f64,
    pub graph_ms: f64,
    pub reinforce_ms: f64,
    pub sort_truncate_ms: f64,
    pub total_ms: f64,
    pub candidate_count: usize,
    pub graph_expansion_count: usize,
}

/// Result of recall_profiled() — recall results plus timing breakdown.
#[cfg(feature = "profiling")]
#[derive(Debug, Clone)]
pub struct RecallProfiledResult {
    pub results: Vec<RecallResult>,
    pub timings: RecallTimings,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            co_occurrence_min_count: 3,
            temporal_cluster_min_events: 3,
            valence_trend_delta_threshold: 0.3,
            topic_cluster_sim_threshold: 0.55,
            topic_cluster_time_window_days: 30.0,
            entity_hub_min_degree: 5,
            max_patterns: 50,
        }
    }
}
