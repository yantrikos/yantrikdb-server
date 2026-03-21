// ── Directory modules ──
mod base;
pub mod engine;
mod cognition;
mod distributed;
mod knowledge;
mod vector;
pub(crate) mod time;
pub(crate) mod id;

// ── Re-exports at original crate paths ──
pub use base::{bench_utils, compression, encryption, error, hlc, schema, scoring, serde_helpers, types, vault};
pub use cognition::{action, agenda, analogy, attention, belief, belief_query, belief_network, benchmark, benchmark_ck4, calibration, causal, coherence, consolidate, contradiction, counterfactual, evaluator, experimenter, extractor, flywheel, hawkes, intent, introspection, metacognition, narrative, observer, patterns, perspective, personality, personality_bias, planner, policy, query_dsl, receptivity, replay, schema_induction, skills, suggest, surfacing, temporal, tick, state, triggers, world_model};
pub use distributed::{conflict, replication, sync};
pub use knowledge::{graph, graph_index};
pub use vector::hnsw;

// ── Convenience re-exports ──
pub use engine::YantrikDB;
pub use engine::tenant::{TenantManager, TenantConfig};
pub use error::YantrikDbError;
pub use types::*;
pub use consolidate::{consolidate, find_consolidation_candidates};
pub use triggers::{check_decay_triggers, check_consolidation_triggers, check_all_triggers};
pub use conflict::{scan_conflicts, detect_edge_conflicts, create_conflict};
pub use patterns::mine_patterns;
pub use state::{
    CognitiveNode, CognitiveAttrs, CognitiveEdge, CognitiveEdgeKind,
    NodeId, NodeKind, NodePayload, NodeIdAllocator, Provenance,
};
pub use attention::{WorkingSet, AttentionConfig};
pub use personality::{derive_personality, get_personality, set_personality_trait};
// V13: sessions, temporal helpers, entity profile, cross-domain types are re-exported via `pub use types::*;`
pub use engine::graph_state::{CognitiveGraphSaveResult, CognitiveGraphStats, CognitiveNodeFilter, CognitiveNodeOrder};
pub use belief::{Evidence, EvidenceResult, BeliefRevisionConfig, RevisionSummary, ThresholdDirection};
pub use belief_query::{BeliefPattern, BeliefOrder, BeliefExplanation, BeliefInventory};
pub use contradiction::{BeliefConflict, ContradictionConfig, ContradictionScanResult, ConflictDetectionMethod, ResolutionStrategy};
pub use intent::{IntentConfig, IntentInferenceResult, IntentSource, ScoredIntent};
pub use action::{ActionCandidate, ActionConfig, CandidateGenerationResult};
pub use evaluator::{EvaluatorConfig, EvaluatedAction, EvaluationResult};
pub use policy::{PolicyConfig, PolicyContext, PolicyDecision, PolicyResult, ReasoningTrace, SelectedAction};
pub use suggest::{NextStepRequest, NextStepResponse, ExecutionMode, ActionProposal, PipelineMetrics};
pub use agenda::{Agenda, AgendaConfig, AgendaId, AgendaItem, AgendaKind, AgendaStatus, UrgencyFn, TickResult};
pub use temporal::{TimeInterval, IntervalRelation, RecencyConfig, PeriodicityConfig, PeriodicityResult, EwmaTracker, BurstConfig, BurstResult, TemporalMotif, DeadlineUrgencyConfig, TemporalOrder, SeasonalHistogram, TemporalRelevanceConfig};
pub use hawkes::{HawkesParams, HawkesRegistry, HawkesRegistryConfig, EventTypeModel, EventPrediction, AnticipatedEvent, CircadianProfile, ModelSummary};
pub use receptivity::{ReceptivityModel, ReceptivityEstimate, ReceptivityFactor, ContextSnapshot, ActivityState, NotificationMode, SuggestionOutcome, QuietHoursConfig, AttentionBudgetConfig};
pub use tick::{TickConfig, TickState, TickReport, TickPhase, Anomaly, AnomalyKind, CachedSuggestion};
pub use surfacing::{SurfaceMode, SurfaceReason, SurfaceOutcome, SurfacingConfig, SurfacingResult, ProactiveSuggestion, SuppressedItem, SuppressionCause, SurfaceRateLimiter, SurfacingPreferences};
pub use observer::{SystemEvent, SystemEventData, EventKind, EventBuffer, EventFilter, EventCounters, CircadianHistogram, ObserverConfig, ObserverState, ObserverSummary, DerivedSignals};
pub use flywheel::{AutonomousBelief, BeliefCategory, BeliefStage, BeliefEvidence, BeliefStore, FlywheelConfig, FormationResult};
pub use world_model::{TransitionModel, StateFeatures, ActionKind as WorldActionKind, ActionOutcome as WorldActionOutcome, OutcomeDistribution, WorldModelSummary};
pub use experimenter::{Experiment, ExperimentId, ExperimentRegistry, ExperimentStatus, ExperimentVariable, SafetyBound, TrialOutcome, VariantValue, BetaPosterior};
pub use skills::{LearnedSkill, SkillId, SkillOrigin, SkillStage, SkillStep, SkillTrigger, SkillConfig, SkillRegistry, SkillMatch, SkillSummary, DiscoveryResult};
pub use extractor::{CognitiveUpdate, UpdateOp, ExtractorTier, ExtractionContext, ExtractionResponse, ExtractorConfig, TemplateStore, SerializableOpTemplate, LlmExtractionRequest, ExtractorSummary};
pub use calibration::{InteractionOutcome, InteractionRecord, UtilityWeights, ActionBandit, BanditRegistry, CalibrationMap, SourceReliability, ReliabilityRegistry, EvidenceSource, LearningState, LearningConfig, WeightSnapshot, LearningReport};
pub use introspection::{IntrospectionReport, Discovery, DiscoveryExplanation, LearningMilestone, MilestoneKind, BeliefStageBreakdown};
pub use causal::{CausalStore, CausalEdge, CausalNode, CausalStage, CausalConfig, CausalTrace, CausalEvidence, DiscoveryMethod, CausalSummary, CausalExplanation, EffectEstimate, EvidenceQuality, PredictedEffect, WhatIfResult, DiscoveryReport as CausalDiscoveryReport};
pub use planner::{Plan, PlanStep, PlanScore, PlanStore, PlanProposal, PlannerConfig, Blocker, BlockerKind, StepDerivation, BoundPrecondition};
pub use coherence::{CoherenceReport, CoherenceConfig, CoherenceHistory, CoherenceSnapshot, EnforcementReport, EnforcementAction, EnforcementKind, GoalConflict, BeliefContradiction, StaleNode, OrphanedItem, DependencyCycle, DeadlineAlert};
pub use metacognition::{MetaCognitiveReport, MetaCognitiveConfig, MetaCognitiveHistory, MetaCognitiveSnapshot, AbstainDecision, AbstainAction, AbstainReason, ConfidenceReport, ReasoningHealthReport, CoverageGap, CoverageGapKind, SignalStatus, SignalDetail};
pub use personality_bias::{PersonalityBiasVector, PersonalityPreset, PersonalityBiasStore, PersonalityBiasResult, BiasConfig, BiasContribution, ActionProperties, BondLevel, EvolutionConfig, LearnedPreferences, PersonalityImpactReport};
pub use query_dsl::{CognitivePipeline, CognitiveOperator, PipelineResult, PipelineStatus, StepResult, StepOutput, PipelineContext, PipelineExecutor, PipelinePatterns, ExecutionMode as PipelineExecutionMode, ProjectionHorizon, CandidateAction as PipelineCandidateAction, PolicyConstraint, ConstraintKind, EvidenceInput, ExplanationTrace};
pub use analogy::{AnalogyStore, StructuralMapping, NodeCorrespondence, EdgeCorrespondence, CandidateInference, ProjectedFact, TransferType, AnalogyScope, AnalogicalQuery, AnalogicalOpportunity, SubgraphGroup, TransferredStrategy, AnalogyMaintenanceReport};
pub use schema_induction::{SchemaId, InducedSchema, SchemaCondition, SchemaStore as InducedSchemaStore, ActionTemplate, ParameterSlot, ParamType, TemplateConstraint, ConstraintExpr, ExpectedOutcome, EpisodeData, ContextSnapshot as SchemaContext, SchemaMaintenanceReport, Direction as SchemaDirection};
pub use narrative::{ArcId, ArcTheme, ArcStatus, ChapterType, DirectionChange, Chapter, TurningPoint, NarrativeArc, Milestone, AutobiographicalTimeline, NarrativeEpisode, NarrativeQuery, NarrativeResult, ArcAlert, ArcAlertType};
pub use counterfactual::{CounterfactualType, Intervention, Observation, CounterfactualQuery, SimulatedStep, StateSnapshot, OutcomeDifference, NodeDelta, DeltaDirection, CounterfactualResult, RegretReport, DecisionRecord, SensitivityEntry, CounterfactualConfig};
pub use belief_network::{VariableId, FactorId, Distribution, FactorType, PotentialFunction, BeliefVariable, Factor as NetworkFactor, BeliefNetwork, BPConfig, BPResult, InferenceType, InferenceQuery, EvidenceContribution, InferenceResult, EdgeRelation, NetworkHealth};
pub use replay::{SamplingStrategy, ActionRecord, OutcomeData, ReplayEntry, BeliefDelta, CausalDelta, ReplayOutcome, DreamReport, ReplayBudget, ReplayBuffer, ReplayStats, ReplayEngine, ReplaySummary};
pub use perspective::{PerspectiveId, PerspectiveType, TemporalFocus, CognitiveStyle, SalienceTarget, SalienceOverride, EdgeWeightModifier, ActivationCondition, ActivationContext, Perspective, PerspectiveStack, PerspectiveStore, PerspectiveTransition, PerspectiveConflict, ConflictType};
