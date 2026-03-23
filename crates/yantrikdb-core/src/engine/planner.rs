//! Engine-level planning API.
//!
//! Wires the planner module into `YantrikDB` for persistence
//! and integration with the cognitive graph.

use crate::error::Result;
use crate::planner::{
    detect_blockers, instantiate_plan, next_plan_step, Blocker, ConstraintEntry,
    GoalEntry, Plan, PlanProposal, PlanStep, PlanStore, PlannerConfig, SchemaEntry,
    SkillStepInfo, SkillTemplate, TaskEntry, PlanningContext,
};
use crate::state::{
    CognitiveEdgeKind, CognitiveNode, ConstraintPayload, GoalPayload, NodeId,
    NodeKind, NodePayload, ActionSchemaPayload, TaskPayload,
};

use super::{now, YantrikDB};

/// Meta key for persisted plan store.
const PLAN_STORE_META_KEY: &str = "plan_store";
/// Meta key for persisted planner config.
const PLANNER_CONFIG_META_KEY: &str = "planner_config";

impl YantrikDB {
    // ── Persistence ──

    /// Load the plan store from the database.
    pub fn load_plan_store(&self) -> Result<PlanStore> {
        match Self::get_meta(&self.conn(), PLAN_STORE_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(PlanStore::new()),
        }
    }

    /// Persist the plan store.
    pub fn save_plan_store(&self, store: &PlanStore) -> Result<()> {
        let json = serde_json::to_string(store).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![PLAN_STORE_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the planner configuration.
    pub fn load_planner_config(&self) -> Result<PlannerConfig> {
        match Self::get_meta(&self.conn(), PLANNER_CONFIG_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(PlannerConfig::default()),
        }
    }

    /// Persist the planner configuration.
    pub fn save_planner_config(&self, config: &PlannerConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![PLANNER_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Planning API ──

    /// Generate plans to achieve a specific goal.
    ///
    /// Loads all relevant cognitive nodes and generates plans via
    /// the HTN-lite planner.
    pub fn plan_for_goal(&self, goal_id: NodeId) -> Result<PlanProposal> {
        let config = self.load_planner_config()?;
        let (schemas, goals, tasks, constraints, edges, skills) =
            self.load_planning_inputs()?;

        let ctx = PlanningContext {
            schemas: &schemas,
            goals: &goals,
            tasks: &tasks,
            constraints: &constraints,
            edges: &edges,
            skills: &skills,
            now: now(),
            config: &config,
        };

        let proposal = instantiate_plan(goal_id, &ctx);

        // Cache the best plan if viable.
        if let Some(best) = proposal.plans.first() {
            if best.viable {
                let mut store = self.load_plan_store()?;
                store.set_plan(best.clone());
                self.save_plan_store(&store)?;
            }
        }

        Ok(proposal)
    }

    /// Get the next recommended step toward a goal.
    pub fn next_step_for_goal(&self, goal_id: NodeId) -> Result<Option<PlanStep>> {
        // Try cached plan first.
        let store = self.load_plan_store()?;
        if let Some(cached) = store.get_plan(goal_id) {
            if cached.viable && !cached.steps.is_empty() {
                let step = cached.steps.first().cloned();
                if step.is_some() {
                    return Ok(step);
                }
            }
        }

        // Generate fresh.
        let config = self.load_planner_config()?;
        let (schemas, goals, tasks, constraints, edges, skills) =
            self.load_planning_inputs()?;

        let ctx = PlanningContext {
            schemas: &schemas,
            goals: &goals,
            tasks: &tasks,
            constraints: &constraints,
            edges: &edges,
            skills: &skills,
            now: now(),
            config: &config,
        };

        Ok(next_plan_step(goal_id, &ctx))
    }

    /// Detect all blockers preventing goal achievement.
    pub fn detect_goal_blockers(&self, goal_id: NodeId) -> Result<Vec<Blocker>> {
        let config = self.load_planner_config()?;
        let (schemas, goals, tasks, constraints, edges, skills) =
            self.load_planning_inputs()?;

        let ctx = PlanningContext {
            schemas: &schemas,
            goals: &goals,
            tasks: &tasks,
            constraints: &constraints,
            edges: &edges,
            skills: &skills,
            now: now(),
            config: &config,
        };

        Ok(detect_blockers(goal_id, &ctx))
    }

    /// Mark a goal's plan as succeeded.
    pub fn mark_plan_succeeded(&self, goal_id: NodeId) -> Result<Option<Plan>> {
        let mut store = self.load_plan_store()?;
        let plan = store.mark_succeeded(goal_id);
        self.save_plan_store(&store)?;
        Ok(plan)
    }

    /// Mark a goal's plan as failed.
    pub fn mark_plan_failed(&self, goal_id: NodeId) -> Result<Option<Plan>> {
        let mut store = self.load_plan_store()?;
        let plan = store.mark_failed(goal_id);
        self.save_plan_store(&store)?;
        Ok(plan)
    }

    /// Get planning statistics.
    pub fn planning_stats(&self) -> Result<PlanningStats> {
        let store = self.load_plan_store()?;
        Ok(PlanningStats {
            active_plans: store.active_count(),
            total_generated: store.total_generated,
            total_succeeded: store.total_succeeded,
            total_failed: store.total_failed,
        })
    }

    /// Reset the plan store.
    pub fn reset_plan_store(&self) -> Result<()> {
        self.save_plan_store(&PlanStore::new())
    }

    // ── Internal — load planning inputs from cognitive graph ──

    /// Load all data needed for planning from the cognitive graph.
    fn load_planning_inputs(
        &self,
    ) -> Result<(
        Vec<SchemaEntry>,
        Vec<GoalEntry>,
        Vec<TaskEntry>,
        Vec<ConstraintEntry>,
        Vec<crate::state::CognitiveEdge>,
        Vec<SkillTemplate>,
    )> {
        // Load cognitive nodes by kind.
        let schema_nodes = self.load_cognitive_nodes_by_kind(NodeKind::ActionSchema)?;
        let goal_nodes = self.load_cognitive_nodes_by_kind(NodeKind::Goal)?;
        let task_nodes = self.load_cognitive_nodes_by_kind(NodeKind::Task)?;
        let constraint_nodes = self.load_cognitive_nodes_by_kind(NodeKind::Constraint)?;

        // Convert to planner entry types.
        let schemas: Vec<SchemaEntry> = schema_nodes
            .into_iter()
            .filter_map(|n| {
                if let NodePayload::ActionSchema(payload) = n.payload {
                    Some(SchemaEntry {
                        node_id: n.id,
                        attrs: n.attrs,
                        payload,
                    })
                } else {
                    None
                }
            })
            .collect();

        let goals: Vec<GoalEntry> = goal_nodes
            .into_iter()
            .filter_map(|n| {
                if let NodePayload::Goal(payload) = n.payload {
                    Some(GoalEntry {
                        node_id: n.id,
                        attrs: n.attrs,
                        payload,
                    })
                } else {
                    None
                }
            })
            .collect();

        let tasks: Vec<TaskEntry> = task_nodes
            .into_iter()
            .filter_map(|n| {
                if let NodePayload::Task(payload) = n.payload {
                    Some(TaskEntry {
                        node_id: n.id,
                        attrs: n.attrs,
                        payload,
                    })
                } else {
                    None
                }
            })
            .collect();

        let constraints: Vec<ConstraintEntry> = constraint_nodes
            .into_iter()
            .filter_map(|n| {
                if let NodePayload::Constraint(payload) = n.payload {
                    Some(ConstraintEntry {
                        node_id: n.id,
                        payload,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Load relevant edges (goal/task/causal relationships).
        let mut edges = Vec::new();
        for kind in [
            CognitiveEdgeKind::AdvancesGoal,
            CognitiveEdgeKind::BlocksGoal,
            CognitiveEdgeKind::SubtaskOf,
            CognitiveEdgeKind::Requires,
            CognitiveEdgeKind::Causes,
            CognitiveEdgeKind::Prevents,
        ] {
            edges.extend(self.load_cognitive_edges_by_kind(kind)?);
        }

        // Load learned skills as decomposition templates.
        let skill_registry = self.load_skill_registry()?;
        let skills: Vec<SkillTemplate> = skill_registry
            .skills
            .values()
            .filter(|s| s.confidence >= 0.4 && !s.deprecated)
            .map(|s| SkillTemplate {
                skill_id: s.id,
                description: s.description.clone(),
                steps: s
                    .steps
                    .iter()
                    .map(|step| SkillStepInfo {
                        ordinal: step.ordinal,
                        action_kind: step.action_kind.clone(),
                        description: format!(
                            "{}{}",
                            step.action_kind,
                            step.tool_name
                                .as_ref()
                                .map(|t| format!(" ({})", t))
                                .unwrap_or_default()
                        ),
                        expected_duration_ms: step.expected_duration_ms,
                        optional: step.optional,
                    })
                    .collect(),
                confidence: s.confidence,
                success_rate: if s.offer_count > 0 {
                    s.success_count as f64 / s.offer_count as f64
                } else {
                    0.5
                },
            })
            .collect();

        Ok((schemas, goals, tasks, constraints, edges, skills))
    }
}

/// Compact planning statistics.
#[derive(Debug, Clone)]
pub struct PlanningStats {
    pub active_plans: usize,
    pub total_generated: u64,
    pub total_succeeded: u64,
    pub total_failed: u64,
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::state::NodeId;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_empty_plan_store() {
        let db = test_db();
        let stats = db.planning_stats().unwrap();
        assert_eq!(stats.active_plans, 0);
        assert_eq!(stats.total_generated, 0);
    }

    #[test]
    fn test_save_load_plan_store() {
        let db = test_db();

        let mut store = crate::planner::PlanStore::new();
        store.total_generated = 5;
        store.total_succeeded = 3;
        db.save_plan_store(&store).unwrap();

        let loaded = db.load_plan_store().unwrap();
        assert_eq!(loaded.total_generated, 5);
        assert_eq!(loaded.total_succeeded, 3);
    }

    #[test]
    fn test_save_load_planner_config() {
        let db = test_db();

        let mut config = crate::planner::PlannerConfig::default();
        config.max_depth = 6;
        config.beam_width = 8;
        db.save_planner_config(&config).unwrap();

        let loaded = db.load_planner_config().unwrap();
        assert_eq!(loaded.max_depth, 6);
        assert_eq!(loaded.beam_width, 8);
    }

    #[test]
    fn test_plan_for_goal_no_goals() {
        let db = test_db();
        // No cognitive nodes exist — should return no plans with blocker.
        let goal_id = NodeId::new(crate::state::NodeKind::Goal, 1);
        let proposal = db.plan_for_goal(goal_id).unwrap();
        assert!(proposal.plans.is_empty());
        assert!(!proposal.global_blockers.is_empty());
    }

    #[test]
    fn test_planning_stats_after_operations() {
        let db = test_db();

        let mut store = crate::planner::PlanStore::new();
        let plan = crate::planner::Plan {
            goal_id: NodeId::new(crate::state::NodeKind::Goal, 1),
            goal_description: "Test".to_string(),
            steps: Vec::new(),
            score: crate::planner::PlanScore {
                feasibility: 0.8,
                expected_utility: 0.6,
                simplicity: 1.0,
                schema_success_rate: 0.9,
                urgency: 0.5,
                composite: 0.7,
                estimated_total_secs: 30.0,
            },
            rationale: "Test".to_string(),
            created_at: 1000.0,
            viable: true,
            blockers: Vec::new(),
        };
        store.set_plan(plan);
        db.save_plan_store(&store).unwrap();

        let stats = db.planning_stats().unwrap();
        assert_eq!(stats.active_plans, 1);
        assert_eq!(stats.total_generated, 1);
    }

    #[test]
    fn test_reset_plan_store() {
        let db = test_db();

        let mut store = crate::planner::PlanStore::new();
        store.total_generated = 10;
        db.save_plan_store(&store).unwrap();

        db.reset_plan_store().unwrap();
        let stats = db.planning_stats().unwrap();
        assert_eq!(stats.total_generated, 0);
    }

    #[test]
    fn test_detect_goal_blockers_no_goals() {
        let db = test_db();
        let goal_id = NodeId::new(crate::state::NodeKind::Goal, 1);
        let blockers = db.detect_goal_blockers(goal_id).unwrap();
        assert!(!blockers.is_empty()); // Goal not found blocker.
    }
}
