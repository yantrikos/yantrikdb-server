//! Cognitive State Graph — Node types, edges, and attributes.
//!
//! The Cognitive State Graph (CSG) is YantrikDB's internal representation of
//! the user's cognitive landscape. It is NOT a neural network — it is a typed,
//! weighted, directed graph where nodes represent cognitive entities (beliefs,
//! goals, routines, etc.) and edges represent relationships between them
//! (supports, contradicts, causes, predicts, etc.).
//!
//! Design principles:
//! - **Arena-allocated**: Compact `NodeId` with embedded type tag for O(1) dispatch
//! - **Log-odds belief revision**: Beliefs track evidence as log-odds, not raw probabilities
//! - **Typed edges**: Every relationship has semantics that influence spreading activation
//! - **Temporal awareness**: All nodes carry temporal metadata for decay and recency
//! - **Provenance tracking**: Every node knows where it came from (observed, inferred, told, experimented)
//! - **Persistence**: Durable nodes survive across sessions via SQLite; transient nodes are working-set only

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ── Node Identity ──

/// Compact node identifier with embedded type tag.
///
/// Layout: upper 4 bits = NodeKind discriminant (0..15), lower 28 bits = sequence.
/// This gives us 16 node kinds and ~268M nodes per kind — vastly more than needed.
/// The type tag enables O(1) dispatch without HashMap lookups.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(u32);

impl NodeId {
    /// Maximum sequence value (2^28 - 1).
    const MAX_SEQ: u32 = 0x0FFF_FFFF;

    /// Create a new NodeId from a kind and sequence number.
    ///
    /// # Panics
    /// Panics if `seq` exceeds 2^28 - 1 or `kind` discriminant exceeds 15.
    pub fn new(kind: NodeKind, seq: u32) -> Self {
        assert!(seq <= Self::MAX_SEQ, "NodeId sequence overflow: {seq}");
        let tag = kind.discriminant() as u32;
        debug_assert!(tag < 16);
        Self((tag << 28) | seq)
    }

    /// Extract the node kind from the type tag.
    #[inline]
    pub fn kind(self) -> NodeKind {
        NodeKind::from_discriminant((self.0 >> 28) as u8)
    }

    /// Extract the sequence number.
    #[inline]
    pub fn seq(self) -> u32 {
        self.0 & Self::MAX_SEQ
    }

    /// Raw u32 representation (for SQLite storage).
    #[inline]
    pub fn to_raw(self) -> u32 {
        self.0
    }

    /// Reconstruct from raw u32 (from SQLite).
    #[inline]
    pub fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Whether this is a nil/sentinel value.
    #[inline]
    pub fn is_nil(self) -> bool {
        self.0 == 0
    }

    /// Nil sentinel (kind=Entity, seq=0). Used as "no node" marker.
    pub const NIL: NodeId = NodeId(0);
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({:?}:{})", self.kind(), self.seq())
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind().as_str(), self.seq())
    }
}

// ── Node Kinds ──

/// The 14 cognitive node types that make up the state graph.
///
/// Each kind has distinct semantics that affect how it participates in
/// spreading activation, belief revision, and action selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum NodeKind {
    /// A named entity from the knowledge graph (person, place, concept).
    /// Bridge between memory-level entities and cognitive-level reasoning.
    Entity = 0,

    /// An episodic memory reference — a specific event or interaction.
    /// Links to the underlying memory rid for full content retrieval.
    Episode = 1,

    /// A held belief about the world, tracked via log-odds for principled revision.
    /// Beliefs can be supported or contradicted by evidence edges.
    Belief = 2,

    /// A desired future state the system is trying to achieve.
    /// Goals have completion criteria and can be decomposed into sub-goals.
    Goal = 3,

    /// A concrete action item with a deadline and assignee.
    /// Tasks advance goals and have preconditions and effects.
    Task = 4,

    /// A hypothesis about the user's current intent.
    /// Multiple hypotheses compete; the winner drives action selection.
    IntentHypothesis = 5,

    /// A recurring behavioral pattern (e.g., "checks email at 9am").
    /// Routines have periodicity, reliability scores, and trigger conditions.
    Routine = 6,

    /// An inferred user need (physical, emotional, informational).
    /// Needs drive proactive suggestions when urgency exceeds threshold.
    Need = 7,

    /// A detected opportunity for helpful action.
    /// Time-bounded: opportunities expire if not acted upon.
    Opportunity = 8,

    /// A potential risk or problem the system has identified.
    /// Risks have severity and likelihood, driving preventive suggestions.
    Risk = 9,

    /// A constraint that limits what actions are acceptable.
    /// E.g., "user doesn't want notifications after 10pm".
    Constraint = 10,

    /// A learned user preference (stronger than a single observation).
    /// Preferences bias action selection and response generation.
    Preference = 11,

    /// An active conversation thread with context and state.
    /// Tracks topic, emotional arc, and unresolved questions.
    ConversationThread = 12,

    /// A template for an action the system can take.
    /// Defines preconditions, effects, and learned confidence thresholds.
    ActionSchema = 13,
}

impl NodeKind {
    /// Number of distinct node kinds.
    pub const COUNT: usize = 14;

    /// Discriminant value (0..13).
    #[inline]
    pub fn discriminant(self) -> u8 {
        self as u8
    }

    /// Reconstruct from discriminant. Returns Entity for unknown values.
    #[inline]
    pub fn from_discriminant(d: u8) -> Self {
        match d {
            0 => Self::Entity,
            1 => Self::Episode,
            2 => Self::Belief,
            3 => Self::Goal,
            4 => Self::Task,
            5 => Self::IntentHypothesis,
            6 => Self::Routine,
            7 => Self::Need,
            8 => Self::Opportunity,
            9 => Self::Risk,
            10 => Self::Constraint,
            11 => Self::Preference,
            12 => Self::ConversationThread,
            13 => Self::ActionSchema,
            _ => Self::Entity,
        }
    }

    /// String representation for serialization and display.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entity => "entity",
            Self::Episode => "episode",
            Self::Belief => "belief",
            Self::Goal => "goal",
            Self::Task => "task",
            Self::IntentHypothesis => "intent_hypothesis",
            Self::Routine => "routine",
            Self::Need => "need",
            Self::Opportunity => "opportunity",
            Self::Risk => "risk",
            Self::Constraint => "constraint",
            Self::Preference => "preference",
            Self::ConversationThread => "conversation_thread",
            Self::ActionSchema => "action_schema",
        }
    }

    /// Parse from string. Returns None for unknown values.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "entity" => Some(Self::Entity),
            "episode" => Some(Self::Episode),
            "belief" => Some(Self::Belief),
            "goal" => Some(Self::Goal),
            "task" => Some(Self::Task),
            "intent_hypothesis" => Some(Self::IntentHypothesis),
            "routine" => Some(Self::Routine),
            "need" => Some(Self::Need),
            "opportunity" => Some(Self::Opportunity),
            "risk" => Some(Self::Risk),
            "constraint" => Some(Self::Constraint),
            "preference" => Some(Self::Preference),
            "conversation_thread" => Some(Self::ConversationThread),
            "action_schema" => Some(Self::ActionSchema),
            _ => None,
        }
    }

    /// Whether this kind persists to SQLite (vs working-set-only).
    /// IntentHypothesis and ConversationThread are transient by default.
    pub fn is_persistent(self) -> bool {
        !matches!(self, Self::IntentHypothesis | Self::ConversationThread)
    }

    /// Whether this kind participates in belief revision.
    pub fn supports_belief_revision(self) -> bool {
        matches!(self, Self::Belief | Self::Preference)
    }

    /// Whether this kind has temporal urgency (deadline-driven).
    pub fn is_time_sensitive(self) -> bool {
        matches!(
            self,
            Self::Task | Self::Opportunity | Self::Need | Self::Risk
        )
    }

    /// All node kinds in discriminant order.
    pub const ALL: [NodeKind; Self::COUNT] = [
        Self::Entity,
        Self::Episode,
        Self::Belief,
        Self::Goal,
        Self::Task,
        Self::IntentHypothesis,
        Self::Routine,
        Self::Need,
        Self::Opportunity,
        Self::Risk,
        Self::Constraint,
        Self::Preference,
        Self::ConversationThread,
        Self::ActionSchema,
    ];
}

// ── Provenance ──

/// How a cognitive node was created — critical for trust and revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provenance {
    /// Directly observed from user interaction or sensor data.
    Observed,
    /// Inferred by the cognitive engine from patterns or correlations.
    Inferred,
    /// Explicitly told by the user ("I prefer X", "My birthday is Y").
    Told,
    /// Result of a safe self-experiment (action → observed outcome).
    Experimented,
    /// Extracted from a document, email, or external data source.
    Extracted,
    /// Inherited from a consolidated or merged source.
    Consolidated,
    /// System-generated default (e.g., initial constraint set).
    SystemDefault,
}

impl Provenance {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Observed => "observed",
            Self::Inferred => "inferred",
            Self::Told => "told",
            Self::Experimented => "experimented",
            Self::Extracted => "extracted",
            Self::Consolidated => "consolidated",
            Self::SystemDefault => "system_default",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "observed" => Self::Observed,
            "inferred" => Self::Inferred,
            "told" => Self::Told,
            "experimented" => Self::Experimented,
            "extracted" => Self::Extracted,
            "consolidated" => Self::Consolidated,
            "system_default" => Self::SystemDefault,
            _ => Self::Observed,
        }
    }

    /// Base reliability prior for this provenance type.
    /// Used as a multiplicative factor in belief revision.
    pub fn reliability_prior(self) -> f64 {
        match self {
            Self::Told => 0.95,       // User explicitly stated — highest trust
            Self::Observed => 0.90,    // Directly observed behavior
            Self::Experimented => 0.85, // Confirmed via controlled experiment
            Self::Extracted => 0.75,   // From external documents — may be outdated
            Self::Inferred => 0.60,    // Pattern-based inference — moderate trust
            Self::Consolidated => 0.80, // Merged from multiple sources
            Self::SystemDefault => 0.50, // Defaults — weakest, easily overridden
        }
    }
}

// ── Cognitive Attributes ──

/// Universal attribute set carried by every cognitive node.
///
/// These 11 dimensions define how a node participates in attention,
/// activation spreading, belief revision, and action selection.
/// All values are bounded and have clear semantic interpretations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveAttrs {
    /// Confidence in this node's validity [0.0, 1.0].
    /// For beliefs: posterior probability. For goals: likelihood of achievement.
    /// For routines: reliability of the pattern.
    pub confidence: f64,

    /// Current activation level [0.0, 1.0].
    /// High activation = node is relevant to current context.
    /// Decays over time; boosted by spreading activation.
    pub activation: f64,

    /// Intrinsic importance/salience [0.0, 1.0].
    /// How much this node matters regardless of current context.
    /// High for core beliefs, active goals, urgent needs.
    pub salience: f64,

    /// Persistence strength [0.0, 1.0].
    /// How resistant this node is to decay and eviction from working set.
    /// 1.0 = permanent (core identity), 0.0 = highly transient.
    pub persistence: f64,

    /// Emotional valence [-1.0, 1.0].
    /// Negative = associated with negative emotion. Positive = positive.
    /// Affects how the node influences response generation.
    pub valence: f64,

    /// Time-pressure signal [0.0, 1.0].
    /// 0.0 = no time pressure. 1.0 = critical deadline.
    /// Computed from deadline proximity for tasks/opportunities.
    pub urgency: f64,

    /// How new/surprising this node is [0.0, 1.0].
    /// High novelty boosts attention allocation.
    /// Decays as the system integrates the information.
    pub novelty: f64,

    /// Recency signal — seconds since last update.
    /// NOT stored directly; computed from `last_updated_ms`.
    /// Used in scoring: more recent = more relevant.
    pub last_updated_ms: u64,

    /// How much this node's state fluctuates [0.0, 1.0].
    /// High volatility = this belief/preference changes often.
    /// Volatile nodes get lower confidence ceilings.
    pub volatility: f64,

    /// How this node was created.
    pub provenance: Provenance,

    /// Number of distinct evidence sources supporting this node.
    /// For beliefs: number of corroborating observations.
    /// Influences confidence ceiling and revision sensitivity.
    pub evidence_count: u32,
}

impl CognitiveAttrs {
    /// Create attributes with sensible defaults for a given node kind.
    pub fn default_for(kind: NodeKind) -> Self {
        let (confidence, salience, persistence) = match kind {
            NodeKind::Entity => (0.80, 0.50, 0.90),
            NodeKind::Episode => (0.95, 0.40, 0.30),
            NodeKind::Belief => (0.50, 0.60, 0.80),
            NodeKind::Goal => (0.70, 0.80, 0.85),
            NodeKind::Task => (0.80, 0.70, 0.50),
            NodeKind::IntentHypothesis => (0.30, 0.90, 0.05),
            NodeKind::Routine => (0.50, 0.50, 0.70),
            NodeKind::Need => (0.60, 0.70, 0.40),
            NodeKind::Opportunity => (0.40, 0.60, 0.20),
            NodeKind::Risk => (0.40, 0.70, 0.60),
            NodeKind::Constraint => (0.90, 0.80, 0.95),
            NodeKind::Preference => (0.60, 0.50, 0.85),
            NodeKind::ConversationThread => (0.90, 0.80, 0.10),
            NodeKind::ActionSchema => (0.70, 0.40, 0.90),
        };

        Self {
            confidence,
            activation: 0.0,
            salience,
            persistence,
            valence: 0.0,
            urgency: 0.0,
            novelty: 1.0,
            last_updated_ms: now_ms(),
            volatility: 0.1,
            provenance: Provenance::Observed,
            evidence_count: 1,
        }
    }

    /// Clamp all values to their valid ranges.
    pub fn clamp(&mut self) {
        self.confidence = self.confidence.clamp(0.0, 1.0);
        self.activation = self.activation.clamp(0.0, 1.0);
        self.salience = self.salience.clamp(0.0, 1.0);
        self.persistence = self.persistence.clamp(0.0, 1.0);
        self.valence = self.valence.clamp(-1.0, 1.0);
        self.urgency = self.urgency.clamp(0.0, 1.0);
        self.novelty = self.novelty.clamp(0.0, 1.0);
        self.volatility = self.volatility.clamp(0.0, 1.0);
    }

    /// Age in seconds since last update.
    pub fn age_secs(&self) -> f64 {
        let now = now_ms();
        if now > self.last_updated_ms {
            (now - self.last_updated_ms) as f64 / 1000.0
        } else {
            0.0
        }
    }

    /// Composite relevance score for working set eviction decisions.
    /// Combines activation, salience, persistence, urgency, and recency.
    pub fn relevance_score(&self) -> f64 {
        let recency = (-self.age_secs() / (3600.0 * 24.0)).exp(); // 1-day half-life
        let novelty_boost = 1.0 + 0.2 * self.novelty;

        (0.35 * self.activation
            + 0.25 * self.salience
            + 0.15 * self.persistence
            + 0.15 * self.urgency
            + 0.10 * recency)
            * novelty_boost
    }

    /// Touch: update last_updated_ms and optionally boost activation.
    pub fn touch(&mut self, activation_boost: f64) {
        self.last_updated_ms = now_ms();
        self.activation = (self.activation + activation_boost).min(1.0);
    }

    /// Apply temporal decay to activation and novelty.
    /// Called during cognitive tick.
    pub fn decay(&mut self, elapsed_secs: f64) {
        // Activation decays with a half-life proportional to persistence
        let activation_half_life = 300.0 + 3300.0 * self.persistence; // 5min to 1hr
        self.activation *= f64::powf(2.0, -elapsed_secs / activation_half_life);

        // Novelty decays faster — 30min half-life
        self.novelty *= f64::powf(2.0, -elapsed_secs / 1800.0);

        // Urgency can increase with time for deadline-driven nodes
        // (handled externally by the agenda manager, not here)

        // Clamp small values to zero to avoid floating-point dust
        if self.activation < 1e-6 {
            self.activation = 0.0;
        }
        if self.novelty < 1e-6 {
            self.novelty = 0.0;
        }
    }
}

impl Default for CognitiveAttrs {
    fn default() -> Self {
        Self::default_for(NodeKind::Entity)
    }
}

// ── Node-Kind-Specific Payloads ──

/// Kind-specific data carried by each cognitive node.
/// This is the union of all node-type-specific fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodePayload {
    Entity(EntityPayload),
    Episode(EpisodePayload),
    Belief(BeliefPayload),
    Goal(GoalPayload),
    Task(TaskPayload),
    IntentHypothesis(IntentPayload),
    Routine(RoutinePayload),
    Need(NeedPayload),
    Opportunity(OpportunityPayload),
    Risk(RiskPayload),
    Constraint(ConstraintPayload),
    Preference(PreferencePayload),
    ConversationThread(ConversationPayload),
    ActionSchema(ActionSchemaPayload),
}

impl NodePayload {
    /// Get the NodeKind this payload corresponds to.
    pub fn kind(&self) -> NodeKind {
        match self {
            Self::Entity(_) => NodeKind::Entity,
            Self::Episode(_) => NodeKind::Episode,
            Self::Belief(_) => NodeKind::Belief,
            Self::Goal(_) => NodeKind::Goal,
            Self::Task(_) => NodeKind::Task,
            Self::IntentHypothesis(_) => NodeKind::IntentHypothesis,
            Self::Routine(_) => NodeKind::Routine,
            Self::Need(_) => NodeKind::Need,
            Self::Opportunity(_) => NodeKind::Opportunity,
            Self::Risk(_) => NodeKind::Risk,
            Self::Constraint(_) => NodeKind::Constraint,
            Self::Preference(_) => NodeKind::Preference,
            Self::ConversationThread(_) => NodeKind::ConversationThread,
            Self::ActionSchema(_) => NodeKind::ActionSchema,
        }
    }
}

/// Entity: a named thing from the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityPayload {
    /// Canonical name (e.g., "Alice", "Rust programming language").
    pub name: String,
    /// Entity classification (person, place, organization, concept, etc.).
    pub entity_type: String,
    /// Links to underlying memory rids that mention this entity.
    pub memory_rids: Vec<String>,
}

/// Episode: reference to a specific memory/event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodePayload {
    /// Reference to the memory rid in the memories table.
    pub memory_rid: String,
    /// Brief summary (extracted or generated).
    pub summary: String,
    /// When the episode occurred (unix timestamp).
    pub occurred_at: f64,
    /// Participants involved.
    pub participants: Vec<String>,
}

/// Belief: a held proposition about the world.
///
/// Beliefs use log-odds representation for principled Bayesian updating:
/// - log_odds = 0.0 → 50% confidence (maximum uncertainty)
/// - log_odds = 2.0 → ~88% confidence
/// - log_odds = -2.0 → ~12% confidence (strong disbelief)
/// - log_odds > 4.0 → very high confidence (>98%)
///
/// Advantages over raw probabilities:
/// - Additive updates: new_log_odds = old_log_odds + evidence_weight
/// - No clamping at 0/1 boundaries
/// - Natural handling of conflicting evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefPayload {
    /// The proposition this belief represents (natural language).
    pub proposition: String,
    /// Log-odds representation of confidence.
    /// P(true) = sigmoid(log_odds) = 1 / (1 + exp(-log_odds))
    pub log_odds: f64,
    /// Domain this belief belongs to (e.g., "work", "health", "preferences").
    pub domain: String,
    /// Source reliability-weighted evidence entries.
    /// Each entry: (source_description, evidence_weight, timestamp).
    pub evidence_trail: Vec<EvidenceEntry>,
    /// Whether this belief has been directly confirmed by the user.
    pub user_confirmed: bool,
}

impl BeliefPayload {
    /// Convert log-odds to probability.
    pub fn probability(&self) -> f64 {
        sigmoid(self.log_odds)
    }

    /// Update belief with new evidence.
    /// `weight` is the log-likelihood ratio: positive supports, negative contradicts.
    /// `reliability` scales the update (0.0 = ignore, 1.0 = full weight).
    pub fn update(&mut self, weight: f64, reliability: f64, source: &str, timestamp: f64) {
        let effective_weight = weight * reliability;
        self.log_odds += effective_weight;
        self.evidence_trail.push(EvidenceEntry {
            source: source.to_string(),
            weight: effective_weight,
            timestamp,
        });
    }

    /// How much evidence supports this belief (sum of positive weights).
    pub fn support_strength(&self) -> f64 {
        self.evidence_trail
            .iter()
            .filter(|e| e.weight > 0.0)
            .map(|e| e.weight)
            .sum()
    }

    /// How much evidence contradicts this belief (sum of negative weights).
    pub fn contradiction_strength(&self) -> f64 {
        self.evidence_trail
            .iter()
            .filter(|e| e.weight < 0.0)
            .map(|e| e.weight.abs())
            .sum()
    }
}

/// A single piece of evidence for or against a belief.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceEntry {
    /// Description of where this evidence came from.
    pub source: String,
    /// Log-likelihood ratio contribution (positive = supports, negative = contradicts).
    pub weight: f64,
    /// When this evidence was recorded.
    pub timestamp: f64,
}

/// Goal: a desired future state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalPayload {
    /// Natural language description of the goal.
    pub description: String,
    /// Current status.
    pub status: GoalStatus,
    /// Estimated progress toward completion [0.0, 1.0].
    pub progress: f64,
    /// Optional deadline (unix timestamp).
    pub deadline: Option<f64>,
    /// Priority tier.
    pub priority: Priority,
    /// Parent goal id (for goal decomposition).
    pub parent_goal: Option<NodeId>,
    /// Completion criteria (natural language).
    pub completion_criteria: String,
}

/// Goal lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalStatus {
    Active,
    Paused,
    Completed,
    Abandoned,
    Blocked,
}

impl GoalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
            Self::Completed => "completed",
            Self::Abandoned => "abandoned",
            Self::Blocked => "blocked",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "active" => Self::Active,
            "paused" => Self::Paused,
            "completed" => Self::Completed,
            "abandoned" => Self::Abandoned,
            "blocked" => Self::Blocked,
            _ => Self::Active,
        }
    }
}

/// Priority levels for goals, tasks, risks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

impl Priority {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "critical" => Self::Critical,
            _ => Self::Medium,
        }
    }

    /// Numeric weight for scoring [0.25, 1.0].
    pub fn weight(self) -> f64 {
        match self {
            Self::Low => 0.25,
            Self::Medium => 0.50,
            Self::High => 0.75,
            Self::Critical => 1.00,
        }
    }
}

/// Task: a concrete action item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPayload {
    /// What needs to be done.
    pub description: String,
    /// Current status.
    pub status: TaskStatus,
    /// Which goal this task advances.
    pub goal_id: Option<NodeId>,
    /// Optional deadline (unix timestamp).
    pub deadline: Option<f64>,
    /// Priority tier.
    pub priority: Priority,
    /// Estimated effort in minutes.
    pub estimated_minutes: Option<u32>,
    /// Prerequisite task IDs (must be completed first).
    pub prerequisites: Vec<NodeId>,
}

/// Task lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Cancelled,
    Blocked,
}

impl TaskStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Blocked => "blocked",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "pending" => Self::Pending,
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "cancelled" => Self::Cancelled,
            "blocked" => Self::Blocked,
            _ => Self::Pending,
        }
    }
}

/// IntentHypothesis: a guess about what the user wants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentPayload {
    /// Natural language description of the hypothesized intent.
    pub description: String,
    /// Feature vector for the linear classifier.
    /// Dimensions: [recency, frequency, entity_overlap, valence_match, ...]
    pub features: Vec<f64>,
    /// Posterior probability from the classifier [0.0, 1.0].
    pub posterior: f64,
    /// Which action schemas this intent maps to.
    pub candidate_actions: Vec<NodeId>,
    /// Conversation context that spawned this hypothesis.
    pub source_context: String,
}

/// Routine: a recurring behavioral pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutinePayload {
    /// Description of the routine (e.g., "Check email at 9am weekdays").
    pub description: String,
    /// Detected periodicity in seconds (e.g., 86400 for daily).
    pub period_secs: f64,
    /// Phase offset: seconds into the period when the event typically occurs.
    pub phase_offset_secs: f64,
    /// How reliable this routine is [0.0, 1.0].
    /// Computed from (observed_count / expected_count).
    pub reliability: f64,
    /// Number of times the routine has been observed.
    pub observation_count: u32,
    /// Last time the routine was triggered.
    pub last_triggered: f64,
    /// The action or event that constitutes this routine.
    pub action_description: String,
    /// Weekday mask: bit 0 = Monday, bit 6 = Sunday. 0x7F = every day.
    pub weekday_mask: u8,
}

impl RoutinePayload {
    /// Predict the next occurrence of this routine (unix timestamp).
    pub fn next_occurrence(&self, now: f64) -> f64 {
        if self.period_secs <= 0.0 {
            return f64::INFINITY;
        }
        let cycles_since_phase = ((now - self.phase_offset_secs) / self.period_secs).floor();
        let next = self.phase_offset_secs + (cycles_since_phase + 1.0) * self.period_secs;
        if next <= now {
            next + self.period_secs
        } else {
            next
        }
    }

    /// How soon until the next occurrence (seconds from now).
    pub fn time_until_next(&self, now: f64) -> f64 {
        self.next_occurrence(now) - now
    }
}

/// Need: an inferred user need.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeedPayload {
    /// What the user needs (e.g., "rest", "information about X", "social connection").
    pub description: String,
    /// Category of need.
    pub category: NeedCategory,
    /// Intensity of the need [0.0, 1.0]. Drives urgency.
    pub intensity: f64,
    /// When this need was last satisfied.
    pub last_satisfied: Option<f64>,
    /// How the need typically gets satisfied.
    pub satisfaction_pattern: String,
}

/// Categories of user needs (loosely based on Maslow but practical).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeedCategory {
    Informational,
    Social,
    Emotional,
    Organizational,
    Creative,
    Health,
    Financial,
    Professional,
}

impl NeedCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Informational => "informational",
            Self::Social => "social",
            Self::Emotional => "emotional",
            Self::Organizational => "organizational",
            Self::Creative => "creative",
            Self::Health => "health",
            Self::Financial => "financial",
            Self::Professional => "professional",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "informational" => Self::Informational,
            "social" => Self::Social,
            "emotional" => Self::Emotional,
            "organizational" => Self::Organizational,
            "creative" => Self::Creative,
            "health" => Self::Health,
            "financial" => Self::Financial,
            "professional" => Self::Professional,
            _ => Self::Informational,
        }
    }
}

/// Opportunity: a time-bounded chance for helpful action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpportunityPayload {
    /// What the opportunity is.
    pub description: String,
    /// When this opportunity expires (unix timestamp).
    pub expires_at: f64,
    /// Expected benefit if acted upon [0.0, 1.0].
    pub expected_benefit: f64,
    /// Required action to seize the opportunity.
    pub required_action: String,
    /// Which goals this opportunity advances.
    pub relevant_goals: Vec<NodeId>,
}

/// Risk: a potential problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskPayload {
    /// Description of the risk.
    pub description: String,
    /// Estimated severity [0.0, 1.0].
    pub severity: f64,
    /// Estimated likelihood [0.0, 1.0].
    pub likelihood: f64,
    /// What could mitigate this risk.
    pub mitigation: String,
    /// Which goals this risk threatens.
    pub threatened_goals: Vec<NodeId>,
}

impl RiskPayload {
    /// Expected impact = severity * likelihood. Used for prioritization.
    pub fn expected_impact(&self) -> f64 {
        self.severity * self.likelihood
    }
}

/// Constraint: a rule limiting acceptable actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintPayload {
    /// Natural language description of the constraint.
    pub description: String,
    /// Whether the constraint is hard (must not violate) or soft (prefer not to).
    pub constraint_type: ConstraintType,
    /// When this constraint is active (always, or specific conditions).
    pub condition: String,
    /// Source: who/what imposed this constraint.
    pub imposed_by: String,
}

/// Hard constraints must never be violated. Soft constraints are preferences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    Hard,
    Soft,
}

impl ConstraintType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hard => "hard",
            Self::Soft => "soft",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "hard" => Self::Hard,
            "soft" => Self::Soft,
            _ => Self::Soft,
        }
    }
}

/// Preference: a learned user preference (stronger than a single observation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferencePayload {
    /// What the preference is about (domain).
    pub domain: String,
    /// The preferred value/behavior.
    pub preferred: String,
    /// The dispreferred value/behavior (if any).
    pub dispreferred: Option<String>,
    /// Strength of preference [0.0, 1.0] — derived from observation count and consistency.
    pub strength: f64,
    /// Log-odds for belief-revision-style updating.
    pub log_odds: f64,
    /// Number of observations supporting this preference.
    pub observation_count: u32,
}

impl PreferencePayload {
    /// Probability that this preference is real (not noise).
    pub fn probability(&self) -> f64 {
        sigmoid(self.log_odds)
    }
}

/// ConversationThread: an active conversation with tracked state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationPayload {
    /// Current topic summary.
    pub topic: String,
    /// Emotional arc: recent valence samples.
    pub valence_history: Vec<f64>,
    /// Unresolved questions or tasks from this conversation.
    pub open_items: Vec<String>,
    /// Turn count.
    pub turn_count: u32,
    /// When the conversation started.
    pub started_at: f64,
}

/// ActionSchema: a template for an action the system can take.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSchemaPayload {
    /// Human-readable name (e.g., "send_reminder", "suggest_break").
    pub name: String,
    /// What this action does.
    pub description: String,
    /// The action category.
    pub action_kind: ActionKind,
    /// Preconditions that must be true for this action to be valid.
    /// Each is a natural language condition + optional node reference.
    pub preconditions: Vec<Precondition>,
    /// Expected effects of taking this action.
    pub effects: Vec<Effect>,
    /// Learned confidence threshold: only suggest if P(success) > threshold.
    pub confidence_threshold: f64,
    /// Historical success rate [0.0, 1.0].
    pub success_rate: f64,
    /// Number of times this action has been executed.
    pub execution_count: u32,
    /// Number of times the user accepted/approved this action.
    pub acceptance_count: u32,
}

/// The 8 core action categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionKind {
    /// Communicate with the user (message, notification, reminder).
    Communicate,
    /// Retrieve or present information.
    Inform,
    /// Create, modify, or organize data.
    Organize,
    /// Schedule or reschedule events.
    Schedule,
    /// Suggest a behavior change or new approach.
    Suggest,
    /// Proactively warn about a risk or problem.
    Warn,
    /// Perform a system operation (file, network, etc.).
    Execute,
    /// Do nothing — explicitly decide inaction is best.
    Abstain,
}

impl ActionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Communicate => "communicate",
            Self::Inform => "inform",
            Self::Organize => "organize",
            Self::Schedule => "schedule",
            Self::Suggest => "suggest",
            Self::Warn => "warn",
            Self::Execute => "execute",
            Self::Abstain => "abstain",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "communicate" => Self::Communicate,
            "inform" => Self::Inform,
            "organize" => Self::Organize,
            "schedule" => Self::Schedule,
            "suggest" => Self::Suggest,
            "warn" => Self::Warn,
            "execute" => Self::Execute,
            "abstain" => Self::Abstain,
            _ => Self::Abstain,
        }
    }

    /// Base cost of this action type (for utility calculations).
    /// Higher cost = more disruptive to the user.
    pub fn base_cost(self) -> f64 {
        match self {
            Self::Abstain => 0.0,
            Self::Inform => 0.05,
            Self::Organize => 0.10,
            Self::Suggest => 0.15,
            Self::Communicate => 0.20,
            Self::Schedule => 0.25,
            Self::Warn => 0.30,
            Self::Execute => 0.40,
        }
    }
}

/// A precondition for an action schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Precondition {
    /// Natural language description of the condition.
    pub description: String,
    /// Optional reference to a node that must be in a certain state.
    pub node_ref: Option<NodeId>,
    /// Whether this precondition is required (hard) or preferred (soft).
    pub required: bool,
}

/// An expected effect of an action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Effect {
    /// What changes when this action is taken.
    pub description: String,
    /// Probability that this effect actually occurs [0.0, 1.0].
    pub probability: f64,
    /// How much utility this effect provides [-1.0, 1.0].
    pub utility: f64,
}

// ── The Cognitive Node ──

/// A complete cognitive node: identity + attributes + payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveNode {
    /// Unique identifier with embedded type tag.
    pub id: NodeId,
    /// Universal cognitive attributes.
    pub attrs: CognitiveAttrs,
    /// Kind-specific payload.
    pub payload: NodePayload,
    /// Optional label for display/search.
    pub label: String,
    /// Free-form metadata (extensibility).
    pub metadata: HashMap<String, serde_json::Value>,
}

impl CognitiveNode {
    /// Create a new cognitive node.
    pub fn new(id: NodeId, label: String, payload: NodePayload) -> Self {
        let attrs = CognitiveAttrs::default_for(id.kind());
        Self {
            id,
            attrs,
            payload,
            label,
            metadata: HashMap::new(),
        }
    }

    /// Create with custom attributes.
    pub fn with_attrs(id: NodeId, label: String, payload: NodePayload, attrs: CognitiveAttrs) -> Self {
        Self {
            id,
            attrs,
            payload,
            label,
            metadata: HashMap::new(),
        }
    }

    /// Node kind (delegated to id).
    #[inline]
    pub fn kind(&self) -> NodeKind {
        self.id.kind()
    }

    /// Whether this node should persist to SQLite.
    pub fn is_persistent(&self) -> bool {
        self.id.kind().is_persistent()
    }
}

// ── Cognitive Edge Types ──

/// Typed edges in the cognitive state graph.
/// Each edge kind has semantic meaning that affects spreading activation
/// and reasoning. Weights are always in [-1.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveEdgeKind {
    // ── Epistemic edges ──
    /// Source provides evidence supporting target (positive weight).
    Supports,
    /// Source provides evidence contradicting target (weight typically negative).
    Contradicts,

    // ── Causal edges ──
    /// Source causes or leads to target.
    Causes,
    /// Source predicts target (weaker than causes — correlation, not causation).
    Predicts,
    /// Source prevents or blocks target.
    Prevents,

    // ── Goal/Task edges ──
    /// Source advances target goal.
    AdvancesGoal,
    /// Source blocks or threatens target goal.
    BlocksGoal,
    /// Source is a subtask/subgoal of target.
    SubtaskOf,
    /// Source requires target as prerequisite.
    Requires,

    // ── Associative edges ──
    /// General semantic association (e.g., entity co-occurrence).
    AssociatedWith,
    /// Source is an instance/example of target category.
    InstanceOf,
    /// Source is a part of target composite.
    PartOf,
    /// Source is similar to target.
    SimilarTo,

    // ── Temporal edges ──
    /// Source occurred before target.
    PrecedesTemporally,
    /// Source triggers target (routine/event chain).
    Triggers,

    // ── Preference/Constraint edges ──
    /// Source prefers target (positive affinity).
    Prefers,
    /// Source avoids target (negative affinity).
    Avoids,
    /// Source constrains target (limits what target can do).
    Constrains,
}

impl CognitiveEdgeKind {
    /// Number of distinct edge kinds.
    pub const COUNT: usize = 18;

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Supports => "supports",
            Self::Contradicts => "contradicts",
            Self::Causes => "causes",
            Self::Predicts => "predicts",
            Self::Prevents => "prevents",
            Self::AdvancesGoal => "advances_goal",
            Self::BlocksGoal => "blocks_goal",
            Self::SubtaskOf => "subtask_of",
            Self::Requires => "requires",
            Self::AssociatedWith => "associated_with",
            Self::InstanceOf => "instance_of",
            Self::PartOf => "part_of",
            Self::SimilarTo => "similar_to",
            Self::PrecedesTemporally => "precedes_temporally",
            Self::Triggers => "triggers",
            Self::Prefers => "prefers",
            Self::Avoids => "avoids",
            Self::Constrains => "constrains",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "supports" => Some(Self::Supports),
            "contradicts" => Some(Self::Contradicts),
            "causes" => Some(Self::Causes),
            "predicts" => Some(Self::Predicts),
            "prevents" => Some(Self::Prevents),
            "advances_goal" => Some(Self::AdvancesGoal),
            "blocks_goal" => Some(Self::BlocksGoal),
            "subtask_of" => Some(Self::SubtaskOf),
            "requires" => Some(Self::Requires),
            "associated_with" => Some(Self::AssociatedWith),
            "instance_of" => Some(Self::InstanceOf),
            "part_of" => Some(Self::PartOf),
            "similar_to" => Some(Self::SimilarTo),
            "precedes_temporally" => Some(Self::PrecedesTemporally),
            "triggers" => Some(Self::Triggers),
            "prefers" => Some(Self::Prefers),
            "avoids" => Some(Self::Avoids),
            "constrains" => Some(Self::Constrains),
            _ => None,
        }
    }

    /// Spreading activation transfer factor for this edge type.
    /// Determines how much activation flows from source to target.
    /// Negative values mean the edge *inhibits* the target.
    pub fn activation_transfer(self) -> f64 {
        match self {
            // Strong positive transfer
            Self::Supports => 0.7,
            Self::Causes => 0.8,
            Self::AdvancesGoal => 0.6,
            Self::Triggers => 0.7,
            Self::Requires => 0.5,
            Self::SubtaskOf => 0.4,

            // Moderate positive transfer
            Self::Predicts => 0.4,
            Self::AssociatedWith => 0.3,
            Self::SimilarTo => 0.3,
            Self::InstanceOf => 0.3,
            Self::PartOf => 0.3,
            Self::Prefers => 0.3,
            Self::PrecedesTemporally => 0.2,

            // Inhibitory transfer
            Self::Contradicts => -0.5,
            Self::Prevents => -0.6,
            Self::BlocksGoal => -0.5,
            Self::Avoids => -0.3,
            Self::Constrains => -0.2,
        }
    }

    /// Whether this edge type is inhibitory (suppresses target activation).
    pub fn is_inhibitory(self) -> bool {
        self.activation_transfer() < 0.0
    }

    /// Whether this edge participates in belief revision.
    pub fn is_epistemic(self) -> bool {
        matches!(self, Self::Supports | Self::Contradicts)
    }

    /// Whether this edge represents a causal relationship.
    pub fn is_causal(self) -> bool {
        matches!(self, Self::Causes | Self::Predicts | Self::Prevents)
    }

    /// All edge kinds.
    pub const ALL: [CognitiveEdgeKind; Self::COUNT] = [
        Self::Supports,
        Self::Contradicts,
        Self::Causes,
        Self::Predicts,
        Self::Prevents,
        Self::AdvancesGoal,
        Self::BlocksGoal,
        Self::SubtaskOf,
        Self::Requires,
        Self::AssociatedWith,
        Self::InstanceOf,
        Self::PartOf,
        Self::SimilarTo,
        Self::PrecedesTemporally,
        Self::Triggers,
        Self::Prefers,
        Self::Avoids,
        Self::Constrains,
    ];
}

/// A directed, typed, weighted edge in the cognitive state graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveEdge {
    /// Source node.
    pub src: NodeId,
    /// Target node.
    pub dst: NodeId,
    /// Relationship type.
    pub kind: CognitiveEdgeKind,
    /// Edge weight [-1.0, 1.0].
    /// Magnitude indicates strength; sign indicates support/opposition.
    pub weight: f64,
    /// When this edge was created (unix timestamp ms).
    pub created_at_ms: u64,
    /// When this edge was last confirmed/updated.
    pub last_confirmed_ms: u64,
    /// How many times this edge has been observed/confirmed.
    pub observation_count: u32,
    /// Confidence in this edge's existence [0.0, 1.0].
    pub confidence: f64,
}

impl CognitiveEdge {
    /// Create a new edge with default confidence.
    pub fn new(src: NodeId, dst: NodeId, kind: CognitiveEdgeKind, weight: f64) -> Self {
        let now = now_ms();
        Self {
            src,
            dst,
            kind,
            weight: weight.clamp(-1.0, 1.0),
            created_at_ms: now,
            last_confirmed_ms: now,
            observation_count: 1,
            confidence: 0.5,
        }
    }

    /// Effective weight = weight * confidence * edge-type transfer factor.
    /// Used during spreading activation.
    pub fn effective_activation_transfer(&self) -> f64 {
        self.weight * self.confidence * self.kind.activation_transfer()
    }

    /// Confirm this edge (increment observation count, update timestamp, boost confidence).
    pub fn confirm(&mut self) {
        self.last_confirmed_ms = now_ms();
        self.observation_count += 1;
        // Confidence grows with observations but saturates
        self.confidence = 1.0 - (1.0 - self.confidence) * 0.85;
    }
}

// ── Node ID Allocator ──

/// Thread-safe sequence allocator for NodeIds.
/// Maintains per-kind counters that persist across sessions via SQLite.
#[derive(Debug, Clone)]
pub struct NodeIdAllocator {
    /// Next available sequence per NodeKind.
    next_seq: [u32; NodeKind::COUNT],
}

impl NodeIdAllocator {
    /// Create a new allocator with all sequences starting at 1.
    /// (0 is reserved for NIL sentinel.)
    pub fn new() -> Self {
        Self {
            next_seq: [1; NodeKind::COUNT],
        }
    }

    /// Initialize from persisted high-water marks.
    pub fn from_high_water_marks(marks: &[(NodeKind, u32)]) -> Self {
        let mut alloc = Self::new();
        for &(kind, hwm) in marks {
            alloc.next_seq[kind.discriminant() as usize] = hwm + 1;
        }
        alloc
    }

    /// Allocate the next NodeId for a given kind.
    pub fn alloc(&mut self, kind: NodeKind) -> NodeId {
        let idx = kind.discriminant() as usize;
        let seq = self.next_seq[idx];
        assert!(seq <= NodeId::MAX_SEQ, "NodeId sequence exhausted for {:?}", kind);
        self.next_seq[idx] = seq + 1;
        NodeId::new(kind, seq)
    }

    /// Current high-water mark for a kind (for persistence).
    pub fn high_water_mark(&self, kind: NodeKind) -> u32 {
        let idx = kind.discriminant() as usize;
        if self.next_seq[idx] > 0 {
            self.next_seq[idx] - 1
        } else {
            0
        }
    }
}

impl Default for NodeIdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Utility Functions ──

/// Sigmoid function: maps log-odds to probability.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Logit function: maps probability to log-odds.
#[inline]
pub fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    (p / (1.0 - p)).ln()
}

/// Current time in milliseconds since Unix epoch.
pub fn now_ms() -> u64 {
    crate::time::now_ms()
}

/// Current time in seconds since Unix epoch (f64 for compatibility with existing code).
pub fn now_secs() -> f64 {
    crate::time::now_secs()
}

// ── Serialization Helpers ──

/// Serialize a CognitiveNode's payload to JSON for SQLite storage.
///
/// Stores the inner payload directly (without the enum tag), since the
/// node kind is stored separately in the `kind` column. This produces
/// cleaner, smaller JSON and avoids tag mismatches during deserialization.
pub fn serialize_payload(payload: &NodePayload) -> serde_json::Value {
    match payload {
        NodePayload::Entity(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::Episode(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::Belief(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::Goal(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::Task(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::IntentHypothesis(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::Routine(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::Need(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::Opportunity(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::Risk(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::Constraint(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::Preference(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::ConversationThread(p) => serde_json::to_value(p).unwrap_or_default(),
        NodePayload::ActionSchema(p) => serde_json::to_value(p).unwrap_or_default(),
    }
}

/// Deserialize a CognitiveNode's payload from JSON, using the node kind
/// to select the correct inner type.
pub fn deserialize_payload(kind: NodeKind, json: &serde_json::Value) -> Option<NodePayload> {
    match kind {
        NodeKind::Entity => serde_json::from_value(json.clone()).ok().map(NodePayload::Entity),
        NodeKind::Episode => serde_json::from_value(json.clone()).ok().map(NodePayload::Episode),
        NodeKind::Belief => serde_json::from_value(json.clone()).ok().map(NodePayload::Belief),
        NodeKind::Goal => serde_json::from_value(json.clone()).ok().map(NodePayload::Goal),
        NodeKind::Task => serde_json::from_value(json.clone()).ok().map(NodePayload::Task),
        NodeKind::IntentHypothesis => serde_json::from_value(json.clone()).ok().map(NodePayload::IntentHypothesis),
        NodeKind::Routine => serde_json::from_value(json.clone()).ok().map(NodePayload::Routine),
        NodeKind::Need => serde_json::from_value(json.clone()).ok().map(NodePayload::Need),
        NodeKind::Opportunity => serde_json::from_value(json.clone()).ok().map(NodePayload::Opportunity),
        NodeKind::Risk => serde_json::from_value(json.clone()).ok().map(NodePayload::Risk),
        NodeKind::Constraint => serde_json::from_value(json.clone()).ok().map(NodePayload::Constraint),
        NodeKind::Preference => serde_json::from_value(json.clone()).ok().map(NodePayload::Preference),
        NodeKind::ConversationThread => serde_json::from_value(json.clone()).ok().map(NodePayload::ConversationThread),
        NodeKind::ActionSchema => serde_json::from_value(json.clone()).ok().map(NodePayload::ActionSchema),
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_roundtrip() {
        for kind in NodeKind::ALL {
            let id = NodeId::new(kind, 42);
            assert_eq!(id.kind(), kind);
            assert_eq!(id.seq(), 42);

            // Raw roundtrip
            let raw = id.to_raw();
            let restored = NodeId::from_raw(raw);
            assert_eq!(restored.kind(), kind);
            assert_eq!(restored.seq(), 42);
        }
    }

    #[test]
    fn test_node_id_nil() {
        assert!(NodeId::NIL.is_nil());
        assert_eq!(NodeId::NIL.kind(), NodeKind::Entity);
        assert_eq!(NodeId::NIL.seq(), 0);
    }

    #[test]
    fn test_node_id_max_seq() {
        let id = NodeId::new(NodeKind::Belief, NodeId::MAX_SEQ);
        assert_eq!(id.kind(), NodeKind::Belief);
        assert_eq!(id.seq(), NodeId::MAX_SEQ);
    }

    #[test]
    #[should_panic(expected = "NodeId sequence overflow")]
    fn test_node_id_overflow() {
        NodeId::new(NodeKind::Entity, NodeId::MAX_SEQ + 1);
    }

    #[test]
    fn test_node_id_display() {
        let id = NodeId::new(NodeKind::Goal, 7);
        assert_eq!(format!("{id}"), "goal:7");
    }

    #[test]
    fn test_node_kind_roundtrip() {
        for kind in NodeKind::ALL {
            let s = kind.as_str();
            let parsed = NodeKind::from_str(s).unwrap();
            assert_eq!(parsed, kind);

            let d = kind.discriminant();
            let from_d = NodeKind::from_discriminant(d);
            assert_eq!(from_d, kind);
        }
    }

    #[test]
    fn test_provenance_reliability() {
        // Told > Observed > Experimented > Extracted > Consolidated > Inferred > SystemDefault
        assert!(Provenance::Told.reliability_prior() > Provenance::Observed.reliability_prior());
        assert!(Provenance::Observed.reliability_prior() > Provenance::Inferred.reliability_prior());
        assert!(Provenance::Inferred.reliability_prior() > Provenance::SystemDefault.reliability_prior());
    }

    #[test]
    fn test_sigmoid_logit_roundtrip() {
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let lo = logit(p);
            let p2 = sigmoid(lo);
            assert!((p - p2).abs() < 1e-10, "sigmoid(logit({p})) = {p2}");
        }
    }

    #[test]
    fn test_sigmoid_extremes() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_belief_update() {
        let mut belief = BeliefPayload {
            proposition: "User prefers dark mode".to_string(),
            log_odds: 0.0, // 50% prior
            domain: "preferences".to_string(),
            evidence_trail: vec![],
            user_confirmed: false,
        };

        assert!((belief.probability() - 0.5).abs() < 1e-10);

        // Add positive evidence
        belief.update(1.5, 0.9, "observed dark mode setting", 1000.0);
        assert!(belief.probability() > 0.7);

        // Add contradicting evidence
        belief.update(-0.5, 0.8, "switched to light mode once", 2000.0);
        let p = belief.probability();
        assert!(p > 0.5 && p < 0.9);

        assert_eq!(belief.evidence_trail.len(), 2);
        assert!(belief.support_strength() > 0.0);
        assert!(belief.contradiction_strength() > 0.0);
    }

    #[test]
    fn test_cognitive_attrs_decay() {
        let mut attrs = CognitiveAttrs {
            activation: 1.0,
            novelty: 1.0,
            persistence: 0.5,
            ..CognitiveAttrs::default_for(NodeKind::Belief)
        };

        // After 600 seconds with persistence=0.5, activation should decay significantly
        attrs.decay(600.0);
        assert!(attrs.activation < 1.0, "activation should decay");
        assert!(attrs.activation > 0.0, "activation shouldn't be zero yet");
        assert!(attrs.novelty < 1.0, "novelty should decay");

        // After a very long time, both should be near zero
        attrs.activation = 1.0;
        attrs.novelty = 1.0;
        attrs.decay(100_000.0);
        assert!(attrs.activation < 0.01);
        assert!(attrs.novelty < 0.001);
    }

    #[test]
    fn test_cognitive_attrs_clamp() {
        let mut attrs = CognitiveAttrs {
            confidence: 1.5,
            activation: -0.1,
            valence: 2.0,
            ..Default::default()
        };
        attrs.clamp();
        assert_eq!(attrs.confidence, 1.0);
        assert_eq!(attrs.activation, 0.0);
        assert_eq!(attrs.valence, 1.0);
    }

    #[test]
    fn test_cognitive_attrs_relevance_score() {
        let mut high_activation = CognitiveAttrs::default_for(NodeKind::Goal);
        high_activation.activation = 1.0;
        high_activation.urgency = 0.8;

        let low_activation = CognitiveAttrs::default_for(NodeKind::Goal);

        assert!(
            high_activation.relevance_score() > low_activation.relevance_score(),
            "high activation should have higher relevance"
        );
    }

    #[test]
    fn test_edge_kinds_completeness() {
        assert_eq!(CognitiveEdgeKind::ALL.len(), CognitiveEdgeKind::COUNT);
        for kind in CognitiveEdgeKind::ALL {
            let s = kind.as_str();
            let parsed = CognitiveEdgeKind::from_str(s).unwrap();
            assert_eq!(parsed, kind, "roundtrip failed for {s}");
        }
    }

    #[test]
    fn test_edge_activation_transfer() {
        // Supports should have positive transfer
        assert!(CognitiveEdgeKind::Supports.activation_transfer() > 0.0);
        // Contradicts should have negative transfer
        assert!(CognitiveEdgeKind::Contradicts.activation_transfer() < 0.0);
        // Causes should be the strongest positive
        assert!(
            CognitiveEdgeKind::Causes.activation_transfer()
                >= CognitiveEdgeKind::Supports.activation_transfer()
        );
    }

    #[test]
    fn test_edge_inhibitory() {
        let inhibitory: Vec<_> = CognitiveEdgeKind::ALL
            .iter()
            .filter(|k| k.is_inhibitory())
            .collect();
        assert!(inhibitory.contains(&&CognitiveEdgeKind::Contradicts));
        assert!(inhibitory.contains(&&CognitiveEdgeKind::Prevents));
        assert!(inhibitory.contains(&&CognitiveEdgeKind::BlocksGoal));
        assert!(inhibitory.contains(&&CognitiveEdgeKind::Avoids));
        assert!(inhibitory.contains(&&CognitiveEdgeKind::Constrains));
    }

    #[test]
    fn test_cognitive_edge_confirm() {
        let src = NodeId::new(NodeKind::Episode, 1);
        let dst = NodeId::new(NodeKind::Belief, 1);
        let mut edge = CognitiveEdge::new(src, dst, CognitiveEdgeKind::Supports, 0.8);

        let initial_confidence = edge.confidence;
        edge.confirm();
        assert_eq!(edge.observation_count, 2);
        assert!(edge.confidence > initial_confidence);

        // Confirm several more times — confidence should approach 1.0
        for _ in 0..20 {
            edge.confirm();
        }
        assert!(edge.confidence > 0.95);
    }

    #[test]
    fn test_cognitive_edge_effective_transfer() {
        let src = NodeId::new(NodeKind::Episode, 1);
        let dst = NodeId::new(NodeKind::Belief, 1);
        let edge = CognitiveEdge::new(src, dst, CognitiveEdgeKind::Supports, 0.8);

        let transfer = edge.effective_activation_transfer();
        // 0.8 (weight) * 0.5 (confidence) * 0.7 (supports transfer) = 0.28
        assert!((transfer - 0.28).abs() < 1e-10);
    }

    #[test]
    fn test_node_id_allocator() {
        let mut alloc = NodeIdAllocator::new();

        let id1 = alloc.alloc(NodeKind::Belief);
        let id2 = alloc.alloc(NodeKind::Belief);
        let id3 = alloc.alloc(NodeKind::Goal);

        assert_eq!(id1.kind(), NodeKind::Belief);
        assert_eq!(id1.seq(), 1);
        assert_eq!(id2.kind(), NodeKind::Belief);
        assert_eq!(id2.seq(), 2);
        assert_eq!(id3.kind(), NodeKind::Goal);
        assert_eq!(id3.seq(), 1);

        // High water marks
        assert_eq!(alloc.high_water_mark(NodeKind::Belief), 2);
        assert_eq!(alloc.high_water_mark(NodeKind::Goal), 1);
        assert_eq!(alloc.high_water_mark(NodeKind::Entity), 0);
    }

    #[test]
    fn test_node_id_allocator_restore() {
        let marks = vec![
            (NodeKind::Belief, 100),
            (NodeKind::Goal, 50),
        ];
        let mut alloc = NodeIdAllocator::from_high_water_marks(&marks);

        let id = alloc.alloc(NodeKind::Belief);
        assert_eq!(id.seq(), 101);

        let id = alloc.alloc(NodeKind::Goal);
        assert_eq!(id.seq(), 51);

        // Uninitialized kinds still start at 1
        let id = alloc.alloc(NodeKind::Entity);
        assert_eq!(id.seq(), 1);
    }

    #[test]
    fn test_cognitive_node_creation() {
        let mut alloc = NodeIdAllocator::new();
        let id = alloc.alloc(NodeKind::Belief);

        let node = CognitiveNode::new(
            id,
            "User prefers dark mode".to_string(),
            NodePayload::Belief(BeliefPayload {
                proposition: "User prefers dark mode".to_string(),
                log_odds: 0.0,
                domain: "preferences".to_string(),
                evidence_trail: vec![],
                user_confirmed: false,
            }),
        );

        assert_eq!(node.kind(), NodeKind::Belief);
        assert!(node.is_persistent());
        assert!((node.attrs.confidence - 0.50).abs() < 1e-10); // Belief default
    }

    #[test]
    fn test_routine_next_occurrence() {
        let routine = RoutinePayload {
            description: "Morning email check".to_string(),
            period_secs: 86400.0,    // daily
            phase_offset_secs: 32400.0, // 9am UTC
            reliability: 0.8,
            observation_count: 30,
            last_triggered: 0.0,
            action_description: "check email".to_string(),
            weekday_mask: 0x1F, // weekdays
        };

        // If it's 10am, next occurrence should be ~23 hours later
        let now = 36000.0; // 10am
        let next = routine.next_occurrence(now);
        assert!(next > now);
        assert!((next - now - 82800.0).abs() < 1.0); // ~23 hours
    }

    #[test]
    fn test_risk_expected_impact() {
        let risk = RiskPayload {
            description: "Server might crash".to_string(),
            severity: 0.9,
            likelihood: 0.3,
            mitigation: "Add monitoring".to_string(),
            threatened_goals: vec![],
        };
        assert!((risk.expected_impact() - 0.27).abs() < 1e-10);
    }

    #[test]
    fn test_action_kind_costs() {
        // Abstain should be cheapest, Execute most expensive
        assert_eq!(ActionKind::Abstain.base_cost(), 0.0);
        assert!(ActionKind::Execute.base_cost() > ActionKind::Communicate.base_cost());
        assert!(ActionKind::Communicate.base_cost() > ActionKind::Inform.base_cost());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Medium);
        assert!(Priority::Medium > Priority::Low);
    }

    #[test]
    fn test_priority_weight() {
        assert!((Priority::Critical.weight() - 1.0).abs() < 1e-10);
        assert!((Priority::Low.weight() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_goal_status_roundtrip() {
        let statuses = [
            GoalStatus::Active,
            GoalStatus::Paused,
            GoalStatus::Completed,
            GoalStatus::Abandoned,
            GoalStatus::Blocked,
        ];
        for s in statuses {
            assert_eq!(GoalStatus::from_str(s.as_str()), s);
        }
    }

    #[test]
    fn test_task_status_roundtrip() {
        let statuses = [
            TaskStatus::Pending,
            TaskStatus::InProgress,
            TaskStatus::Completed,
            TaskStatus::Cancelled,
            TaskStatus::Blocked,
        ];
        for s in statuses {
            assert_eq!(TaskStatus::from_str(s.as_str()), s);
        }
    }

    #[test]
    fn test_constraint_types() {
        assert_eq!(ConstraintType::from_str("hard"), ConstraintType::Hard);
        assert_eq!(ConstraintType::from_str("soft"), ConstraintType::Soft);
        assert_eq!(ConstraintType::from_str("unknown"), ConstraintType::Soft);
    }

    #[test]
    fn test_need_categories_roundtrip() {
        let categories = [
            NeedCategory::Informational,
            NeedCategory::Social,
            NeedCategory::Emotional,
            NeedCategory::Organizational,
            NeedCategory::Creative,
            NeedCategory::Health,
            NeedCategory::Financial,
            NeedCategory::Professional,
        ];
        for c in categories {
            assert_eq!(NeedCategory::from_str(c.as_str()), c);
        }
    }

    #[test]
    fn test_preference_payload_probability() {
        let pref = PreferencePayload {
            domain: "UI".to_string(),
            preferred: "dark mode".to_string(),
            dispreferred: Some("light mode".to_string()),
            strength: 0.8,
            log_odds: 2.0,
            observation_count: 10,
        };
        assert!(pref.probability() > 0.85);
    }

    #[test]
    fn test_node_payload_kind_consistency() {
        // Ensure NodePayload::kind() matches the variant
        let payloads: Vec<NodePayload> = vec![
            NodePayload::Entity(EntityPayload {
                name: "test".into(),
                entity_type: "person".into(),
                memory_rids: vec![],
            }),
            NodePayload::Belief(BeliefPayload {
                proposition: "test".into(),
                log_odds: 0.0,
                domain: "test".into(),
                evidence_trail: vec![],
                user_confirmed: false,
            }),
            NodePayload::Goal(GoalPayload {
                description: "test".into(),
                status: GoalStatus::Active,
                progress: 0.0,
                deadline: None,
                priority: Priority::Medium,
                parent_goal: None,
                completion_criteria: "test".into(),
            }),
        ];

        let expected_kinds = [NodeKind::Entity, NodeKind::Belief, NodeKind::Goal];
        for (payload, expected) in payloads.iter().zip(expected_kinds.iter()) {
            assert_eq!(payload.kind(), *expected);
        }
    }

    #[test]
    fn test_serialize_deserialize_payload() {
        let original = NodePayload::Belief(BeliefPayload {
            proposition: "The sky is blue".to_string(),
            log_odds: 3.5,
            domain: "science".to_string(),
            evidence_trail: vec![
                EvidenceEntry {
                    source: "observation".to_string(),
                    weight: 2.0,
                    timestamp: 1000.0,
                },
            ],
            user_confirmed: true,
        });

        let json = serialize_payload(&original);
        let restored = deserialize_payload(NodeKind::Belief, &json).unwrap();

        if let NodePayload::Belief(b) = restored {
            assert_eq!(b.proposition, "The sky is blue");
            assert!((b.log_odds - 3.5).abs() < 1e-10);
            assert_eq!(b.evidence_trail.len(), 1);
            assert!(b.user_confirmed);
        } else {
            panic!("Expected Belief payload");
        }
    }

    #[test]
    fn test_persistent_vs_transient() {
        assert!(NodeKind::Entity.is_persistent());
        assert!(NodeKind::Belief.is_persistent());
        assert!(NodeKind::Goal.is_persistent());
        assert!(!NodeKind::IntentHypothesis.is_persistent());
        assert!(!NodeKind::ConversationThread.is_persistent());
    }

    #[test]
    fn test_time_sensitive_kinds() {
        assert!(NodeKind::Task.is_time_sensitive());
        assert!(NodeKind::Opportunity.is_time_sensitive());
        assert!(NodeKind::Need.is_time_sensitive());
        assert!(NodeKind::Risk.is_time_sensitive());
        assert!(!NodeKind::Entity.is_time_sensitive());
        assert!(!NodeKind::Belief.is_time_sensitive());
    }

    #[test]
    fn test_action_schema_payload() {
        let schema = ActionSchemaPayload {
            name: "send_reminder".to_string(),
            description: "Send a reminder to the user".to_string(),
            action_kind: ActionKind::Communicate,
            preconditions: vec![
                Precondition {
                    description: "User has a pending task".to_string(),
                    node_ref: None,
                    required: true,
                },
            ],
            effects: vec![
                Effect {
                    description: "User is reminded of the task".to_string(),
                    probability: 0.95,
                    utility: 0.3,
                },
            ],
            confidence_threshold: 0.6,
            success_rate: 0.85,
            execution_count: 100,
            acceptance_count: 85,
        };

        assert_eq!(schema.action_kind, ActionKind::Communicate);
        assert_eq!(schema.preconditions.len(), 1);
        assert!(schema.preconditions[0].required);
    }

    #[test]
    fn test_conversation_payload() {
        let conv = ConversationPayload {
            topic: "Planning vacation".to_string(),
            valence_history: vec![0.5, 0.6, 0.3, 0.8],
            open_items: vec!["Book hotel".to_string(), "Check flights".to_string()],
            turn_count: 12,
            started_at: 1000.0,
        };
        assert_eq!(conv.open_items.len(), 2);
        assert_eq!(conv.turn_count, 12);
    }
}
