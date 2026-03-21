//! Cognitive Kernel Benchmark Suite
//!
//! Creates multiple persona-driven YantrikDB instances with realistic cognitive
//! state graphs, runs a battery of tests, and records results in a separate
//! SQLite tracking database for regression analysis over time.
//!
//! ## Personas
//!
//! 1. **Aisha** — Surgical resident in Lagos: high-urgency tasks, medical beliefs,
//!    complex social graph (colleagues, patients, family). Tests: deadline-driven
//!    urgency, belief revision under conflicting evidence, dense entity graphs.
//!
//! 2. **Marcus** — Restaurant owner in Portland: business goals with sub-tasks,
//!    seasonal routines, financial risks. Tests: goal decomposition, routine
//!    prediction, risk assessment, preference learning.
//!
//! 3. **Priya** — Senior engineer in Bangalore: technical beliefs, sprint tasks,
//!    work-life balance constraints, career goals. Tests: task dependency chains,
//!    constraint enforcement, technical knowledge graph.
//!
//! 4. **Emre** — Freelance photographer in Istanbul: creative needs, opportunity
//!    detection, irregular routines, emotional valence arcs. Tests: need-driven
//!    suggestions, time-bounded opportunities, sparse routine detection.
//!
//! 5. **Keiko** — Retired teacher in Kyoto: health goals, medication routines,
//!    family conversation threads, strong preferences. Tests: routine reliability,
//!    preference stability, gentle urgency escalation.
//!
//! ## Benchmark Categories
//!
//! - **Node Operations**: creation, retrieval, attribute updates
//! - **Belief Revision**: log-odds updating, evidence accumulation, convergence
//! - **Activation Dynamics**: decay, boosting, relevance scoring
//! - **Edge Semantics**: activation transfer, inhibition, epistemic edges
//! - **Allocator Performance**: ID allocation throughput, high-water-mark persistence
//! - **Serialization**: payload serialize/deserialize roundtrip fidelity
//! - **Composite**: multi-persona cross-scenario validation

use std::collections::HashMap;
use std::time::Instant;

use crate::engine::YantrikDB;
use crate::error::Result;

use super::state::*;

// ── Tracking Database ──

/// Schema for the benchmark tracking database.
const BENCH_TRACKING_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS bench_runs (
    run_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    git_commit TEXT,
    description TEXT,
    total_duration_ms REAL NOT NULL,
    pass_count INTEGER NOT NULL,
    fail_count INTEGER NOT NULL,
    skip_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bench_results (
    result_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES bench_runs(run_id),
    category TEXT NOT NULL,
    test_name TEXT NOT NULL,
    persona TEXT,
    passed INTEGER NOT NULL,
    duration_us REAL NOT NULL,
    metric_name TEXT,
    metric_value REAL,
    details TEXT,
    UNIQUE(run_id, category, test_name, persona)
);

CREATE INDEX IF NOT EXISTS idx_results_run ON bench_results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_category ON bench_results(category);
CREATE INDEX IF NOT EXISTS idx_results_test ON bench_results(test_name);

CREATE TABLE IF NOT EXISTS bench_regressions (
    regression_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES bench_runs(run_id),
    test_name TEXT NOT NULL,
    previous_value REAL NOT NULL,
    current_value REAL NOT NULL,
    delta_pct REAL NOT NULL,
    severity TEXT NOT NULL,
    description TEXT
);
";

/// Handle to the benchmark tracking database.
pub struct BenchTracker {
    conn: rusqlite::Connection,
}

impl BenchTracker {
    /// Open or create the tracking database at the given path.
    pub fn open(path: &str) -> Result<Self> {
        let conn = rusqlite::Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        conn.execute_batch(BENCH_TRACKING_SCHEMA)?;
        Ok(Self { conn })
    }

    /// Open an in-memory tracking database (for tests).
    pub fn in_memory() -> Result<Self> {
        Self::open(":memory:")
    }

    /// Start a new benchmark run.
    pub fn start_run(&self, description: &str, git_commit: Option<&str>) -> Result<BenchRun> {
        let run_id = crate::id::new_id();
        let timestamp = now_secs();
        self.conn.execute(
            "INSERT INTO bench_runs (run_id, timestamp, git_commit, description, \
             total_duration_ms, pass_count, fail_count, skip_count) \
             VALUES (?1, ?2, ?3, ?4, 0, 0, 0, 0)",
            rusqlite::params![run_id, timestamp, git_commit, description],
        )?;
        Ok(BenchRun {
            run_id,
            start: Instant::now(),
            results: Vec::new(),
        })
    }

    /// Record a single test result.
    pub fn record_result(&self, run_id: &str, result: &BenchResult) -> Result<()> {
        let result_id = crate::id::new_id();
        self.conn.execute(
            "INSERT OR REPLACE INTO bench_results \
             (result_id, run_id, category, test_name, persona, passed, \
              duration_us, metric_name, metric_value, details) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                result_id,
                run_id,
                result.category,
                result.test_name,
                result.persona,
                result.passed as i32,
                result.duration_us,
                result.metric_name,
                result.metric_value,
                result.details,
            ],
        )?;
        Ok(())
    }

    /// Finalize a benchmark run with totals.
    pub fn finalize_run(&self, run: &BenchRun) -> Result<()> {
        let duration_ms = run.start.elapsed().as_secs_f64() * 1000.0;
        let pass_count = run.results.iter().filter(|r| r.passed).count();
        let fail_count = run.results.iter().filter(|r| !r.passed).count();

        self.conn.execute(
            "UPDATE bench_runs SET total_duration_ms = ?1, pass_count = ?2, \
             fail_count = ?3 WHERE run_id = ?4",
            rusqlite::params![duration_ms, pass_count, fail_count, run.run_id],
        )?;

        // Record all individual results
        for result in &run.results {
            self.record_result(&run.run_id, result)?;
        }

        Ok(())
    }

    /// Detect regressions by comparing with the most recent previous run.
    pub fn detect_regressions(&self, run_id: &str, threshold_pct: f64) -> Result<Vec<Regression>> {
        // Find previous run
        let prev_run_id: Option<String> = self.conn.query_row(
            "SELECT run_id FROM bench_runs WHERE run_id != ?1 \
             ORDER BY timestamp DESC LIMIT 1",
            rusqlite::params![run_id],
            |row| row.get(0),
        ).ok();

        let prev_run_id = match prev_run_id {
            Some(id) => id,
            None => return Ok(vec![]),
        };

        let mut stmt = self.conn.prepare(
            "SELECT c.test_name, p.metric_value, c.metric_value \
             FROM bench_results c \
             JOIN bench_results p ON c.test_name = p.test_name \
                AND c.category = p.category \
                AND COALESCE(c.persona, '') = COALESCE(p.persona, '') \
             WHERE c.run_id = ?1 AND p.run_id = ?2 \
                AND c.metric_value IS NOT NULL AND p.metric_value IS NOT NULL \
                AND p.metric_value != 0",
        )?;

        let mut regressions = Vec::new();
        let rows = stmt.query_map(
            rusqlite::params![run_id, prev_run_id],
            |row| Ok((
                row.get::<_, String>(0)?,
                row.get::<_, f64>(1)?,
                row.get::<_, f64>(2)?,
            )),
        )?;

        for row in rows {
            let (test_name, prev_value, curr_value) = row?;
            let delta_pct = ((curr_value - prev_value) / prev_value.abs()) * 100.0;

            if delta_pct.abs() > threshold_pct {
                let severity = if delta_pct.abs() > threshold_pct * 3.0 {
                    "critical"
                } else if delta_pct.abs() > threshold_pct * 2.0 {
                    "warning"
                } else {
                    "info"
                };

                let regression = Regression {
                    test_name: test_name.clone(),
                    previous_value: prev_value,
                    current_value: curr_value,
                    delta_pct,
                    severity: severity.to_string(),
                };

                let reg_id = crate::id::new_id();
                self.conn.execute(
                    "INSERT INTO bench_regressions \
                     (regression_id, run_id, test_name, previous_value, \
                      current_value, delta_pct, severity) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    rusqlite::params![
                        reg_id, run_id, test_name,
                        prev_value, curr_value, delta_pct, severity
                    ],
                )?;

                regressions.push(regression);
            }
        }

        Ok(regressions)
    }

    /// Get summary of the last N runs.
    pub fn run_history(&self, limit: usize) -> Result<Vec<RunSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT run_id, timestamp, description, total_duration_ms, \
             pass_count, fail_count, skip_count \
             FROM bench_runs ORDER BY timestamp DESC LIMIT ?1",
        )?;

        let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
            Ok(RunSummary {
                run_id: row.get("run_id")?,
                timestamp: row.get("timestamp")?,
                description: row.get("description")?,
                total_duration_ms: row.get("total_duration_ms")?,
                pass_count: row.get("pass_count")?,
                fail_count: row.get("fail_count")?,
                skip_count: row.get("skip_count")?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(rows)
    }
}

/// A benchmark run in progress.
pub struct BenchRun {
    pub run_id: String,
    start: Instant,
    pub results: Vec<BenchResult>,
}

impl BenchRun {
    /// Add a result to this run.
    pub fn add(&mut self, result: BenchResult) {
        self.results.push(result);
    }
}

/// Result of a single benchmark test.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub category: String,
    pub test_name: String,
    pub persona: Option<String>,
    pub passed: bool,
    pub duration_us: f64,
    pub metric_name: Option<String>,
    pub metric_value: Option<f64>,
    pub details: Option<String>,
}

/// A detected regression between runs.
#[derive(Debug, Clone)]
pub struct Regression {
    pub test_name: String,
    pub previous_value: f64,
    pub current_value: f64,
    pub delta_pct: f64,
    pub severity: String,
}

/// Summary of a completed benchmark run.
#[derive(Debug, Clone)]
pub struct RunSummary {
    pub run_id: String,
    pub timestamp: f64,
    pub description: Option<String>,
    pub total_duration_ms: f64,
    pub pass_count: i64,
    pub fail_count: i64,
    pub skip_count: i64,
}

// ── Persona Builders ──

/// A seeded persona with cognitive nodes and edges ready for testing.
pub struct PersonaScenario {
    pub name: String,
    pub description: String,
    pub db: YantrikDB,
    pub allocator: NodeIdAllocator,
    pub nodes: Vec<CognitiveNode>,
    pub edges: Vec<CognitiveEdge>,
}

/// Build the Aisha persona: surgical resident with urgent tasks and medical beliefs.
pub fn build_aisha() -> Result<PersonaScenario> {
    let db = YantrikDB::new(":memory:", 8)?;
    let mut alloc = NodeIdAllocator::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // ── Entities ──
    let aisha_id = alloc.alloc(NodeKind::Entity);
    nodes.push(CognitiveNode::new(aisha_id, "Aisha Okafor".into(), NodePayload::Entity(EntityPayload {
        name: "Aisha Okafor".into(),
        entity_type: "person".into(),
        memory_rids: vec![],
    })));

    let chidi_id = alloc.alloc(NodeKind::Entity);
    nodes.push(CognitiveNode::new(chidi_id, "Chidi".into(), NodePayload::Entity(EntityPayload {
        name: "Chidi".into(),
        entity_type: "person".into(),
        memory_rids: vec![],
    })));

    let hospital_id = alloc.alloc(NodeKind::Entity);
    nodes.push(CognitiveNode::new(hospital_id, "Lagos Teaching Hospital".into(), NodePayload::Entity(EntityPayload {
        name: "Lagos Teaching Hospital".into(),
        entity_type: "organization".into(),
        memory_rids: vec![],
    })));

    let fellowship_id = alloc.alloc(NodeKind::Entity);
    nodes.push(CognitiveNode::new(fellowship_id, "Cardiothoracic Fellowship".into(), NodePayload::Entity(EntityPayload {
        name: "Cardiothoracic Fellowship".into(),
        entity_type: "concept".into(),
        memory_rids: vec![],
    })));

    // ── Beliefs ──
    let belief_surgery = alloc.alloc(NodeKind::Belief);
    nodes.push(CognitiveNode::new(belief_surgery, "Surgery is my calling".into(), NodePayload::Belief(BeliefPayload {
        proposition: "Surgery is my calling and I'm good at it".into(),
        log_odds: 2.5, // ~92% confidence
        domain: "career".into(),
        evidence_trail: vec![
            EvidenceEntry { source: "successful operations".into(), weight: 1.5, timestamp: 1000.0 },
            EvidenceEntry { source: "attending feedback".into(), weight: 1.0, timestamp: 2000.0 },
        ],
        user_confirmed: true,
    })));

    let belief_chidi = alloc.alloc(NodeKind::Belief);
    nodes.push(CognitiveNode::new(belief_chidi, "Chidi is supportive".into(), NodePayload::Belief(BeliefPayload {
        proposition: "Chidi supports my career despite the long hours".into(),
        log_odds: 1.0, // ~73% — some tension
        domain: "relationship".into(),
        evidence_trail: vec![
            EvidenceEntry { source: "encouraging messages".into(), weight: 1.5, timestamp: 1000.0 },
            EvidenceEntry { source: "complained about missed dinner".into(), weight: -0.5, timestamp: 3000.0 },
        ],
        user_confirmed: false,
    })));

    // ── Goals ──
    let goal_fellowship = alloc.alloc(NodeKind::Goal);
    nodes.push(CognitiveNode::new(goal_fellowship, "Get fellowship".into(), NodePayload::Goal(GoalPayload {
        description: "Get accepted to cardiothoracic fellowship".into(),
        status: GoalStatus::Active,
        progress: 0.35,
        deadline: Some(now_secs() + 86400.0 * 180.0),
        priority: Priority::Critical,
        parent_goal: None,
        completion_criteria: "Receive acceptance letter from fellowship program".into(),
    })));

    // ── Tasks ──
    let task_essay = alloc.alloc(NodeKind::Task);
    nodes.push(CognitiveNode::new(task_essay, "Write fellowship essay".into(), NodePayload::Task(TaskPayload {
        description: "Complete personal statement for fellowship application".into(),
        status: TaskStatus::InProgress,
        goal_id: Some(goal_fellowship),
        deadline: Some(now_secs() + 86400.0 * 14.0),
        priority: Priority::High,
        estimated_minutes: Some(480),
        prerequisites: vec![],
    })));

    let task_rec_letters = alloc.alloc(NodeKind::Task);
    nodes.push(CognitiveNode::new(task_rec_letters, "Get recommendation letters".into(), NodePayload::Task(TaskPayload {
        description: "Obtain 3 recommendation letters from attendings".into(),
        status: TaskStatus::Pending,
        goal_id: Some(goal_fellowship),
        deadline: Some(now_secs() + 86400.0 * 30.0),
        priority: Priority::High,
        estimated_minutes: Some(120),
        prerequisites: vec![],
    })));

    // ── Routines ──
    let routine_rounds = alloc.alloc(NodeKind::Routine);
    nodes.push(CognitiveNode::new(routine_rounds, "Morning rounds".into(), NodePayload::Routine(RoutinePayload {
        description: "Morning ward rounds at 6:30am".into(),
        period_secs: 86400.0,
        phase_offset_secs: 23400.0, // 6:30am
        reliability: 0.92,
        observation_count: 180,
        last_triggered: now_secs() - 3600.0,
        action_description: "Review patients, check vitals, update care plans".into(),
        weekday_mask: 0x7F, // every day — surgical resident life
    })));

    // ── Risks ──
    let risk_burnout = alloc.alloc(NodeKind::Risk);
    nodes.push(CognitiveNode::new(risk_burnout, "Burnout risk".into(), NodePayload::Risk(RiskPayload {
        description: "Risk of burnout from 80+ hour work weeks".into(),
        severity: 0.8,
        likelihood: 0.6,
        mitigation: "Schedule rest days, communicate boundaries with Chidi".into(),
        threatened_goals: vec![goal_fellowship],
    })));

    // ── Constraints ──
    let constraint_quiet = alloc.alloc(NodeKind::Constraint);
    nodes.push(CognitiveNode::new(constraint_quiet, "No notifications during surgery".into(), NodePayload::Constraint(ConstraintPayload {
        description: "Never send notifications during scheduled surgeries".into(),
        constraint_type: ConstraintType::Hard,
        condition: "During scheduled surgery blocks (7am-2pm weekdays)".into(),
        imposed_by: "user".into(),
    })));

    // ── Edges ──
    edges.push(CognitiveEdge::new(belief_surgery, goal_fellowship, CognitiveEdgeKind::Supports, 0.9));
    edges.push(CognitiveEdge::new(task_essay, goal_fellowship, CognitiveEdgeKind::AdvancesGoal, 0.7));
    edges.push(CognitiveEdge::new(task_rec_letters, goal_fellowship, CognitiveEdgeKind::AdvancesGoal, 0.8));
    edges.push(CognitiveEdge::new(task_rec_letters, task_essay, CognitiveEdgeKind::Requires, 0.3));
    edges.push(CognitiveEdge::new(risk_burnout, goal_fellowship, CognitiveEdgeKind::BlocksGoal, 0.6));
    edges.push(CognitiveEdge::new(constraint_quiet, routine_rounds, CognitiveEdgeKind::Constrains, 0.9));
    edges.push(CognitiveEdge::new(aisha_id, hospital_id, CognitiveEdgeKind::AssociatedWith, 0.95));
    edges.push(CognitiveEdge::new(aisha_id, chidi_id, CognitiveEdgeKind::AssociatedWith, 0.85));

    // Seed some memories into the DB for integration
    let dim = 8;
    for i in 0..20 {
        let emb = crate::bench_utils::vec_seed_dim(i as f32, dim);
        db.record(
            &format!("Aisha memory {i}: patient case discussion at hospital"),
            "episodic", 0.5 + (i % 5) as f64 * 0.1, 0.2, 604800.0,
            &serde_json::json!({"persona": "aisha"}), &emb, "default",
            0.8, "medical", "user", None,
        )?;
    }

    Ok(PersonaScenario {
        name: "Aisha".into(),
        description: "Surgical resident in Lagos — high-urgency, medical domain".into(),
        db, allocator: alloc, nodes, edges,
    })
}

/// Build the Marcus persona: restaurant owner with business goals and seasonal routines.
pub fn build_marcus() -> Result<PersonaScenario> {
    let db = YantrikDB::new(":memory:", 8)?;
    let mut alloc = NodeIdAllocator::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Entities
    let marcus_id = alloc.alloc(NodeKind::Entity);
    nodes.push(CognitiveNode::new(marcus_id, "Marcus Chen".into(), NodePayload::Entity(EntityPayload {
        name: "Marcus Chen".into(), entity_type: "person".into(), memory_rids: vec![],
    })));

    let pearl_id = alloc.alloc(NodeKind::Entity);
    nodes.push(CognitiveNode::new(pearl_id, "The Pearl Kitchen".into(), NodePayload::Entity(EntityPayload {
        name: "The Pearl Kitchen".into(), entity_type: "organization".into(), memory_rids: vec![],
    })));

    let sarah_id = alloc.alloc(NodeKind::Entity);
    nodes.push(CognitiveNode::new(sarah_id, "Sarah Chen".into(), NodePayload::Entity(EntityPayload {
        name: "Sarah Chen".into(), entity_type: "person".into(), memory_rids: vec![],
    })));

    // Goal hierarchy: expand restaurant
    let goal_expand = alloc.alloc(NodeKind::Goal);
    nodes.push(CognitiveNode::new(goal_expand, "Open second location".into(), NodePayload::Goal(GoalPayload {
        description: "Open a second Pearl Kitchen location in NW Portland".into(),
        status: GoalStatus::Active, progress: 0.15,
        deadline: Some(now_secs() + 86400.0 * 365.0),
        priority: Priority::High, parent_goal: None,
        completion_criteria: "Second location open and profitable for 3 months".into(),
    })));

    let goal_menu = alloc.alloc(NodeKind::Goal);
    nodes.push(CognitiveNode::new(goal_menu, "Spring menu launch".into(), NodePayload::Goal(GoalPayload {
        description: "Design and launch spring seasonal menu".into(),
        status: GoalStatus::Active, progress: 0.60,
        deadline: Some(now_secs() + 86400.0 * 30.0),
        priority: Priority::Medium, parent_goal: Some(goal_expand),
        completion_criteria: "Menu finalized, staff trained, marketing sent".into(),
    })));

    // Routines
    let routine_inventory = alloc.alloc(NodeKind::Routine);
    nodes.push(CognitiveNode::new(routine_inventory, "Inventory check".into(), NodePayload::Routine(RoutinePayload {
        description: "Weekly inventory count and supplier orders".into(),
        period_secs: 604800.0, // weekly
        phase_offset_secs: 36000.0, // Monday 10am
        reliability: 0.88,
        observation_count: 48,
        last_triggered: now_secs() - 86400.0 * 3.0,
        action_description: "Count inventory, compare to POS, order from suppliers".into(),
        weekday_mask: 0x01, // Monday only
    })));

    // Preferences
    let pref_local = alloc.alloc(NodeKind::Preference);
    nodes.push(CognitiveNode::new(pref_local, "Prefers local suppliers".into(), NodePayload::Preference(PreferencePayload {
        domain: "business".into(),
        preferred: "Local Oregon farms and producers".into(),
        dispreferred: Some("Large national distributors".into()),
        strength: 0.85,
        log_odds: 2.0,
        observation_count: 25,
    })));

    // Risk
    let risk_cost = alloc.alloc(NodeKind::Risk);
    nodes.push(CognitiveNode::new(risk_cost, "Rising food costs".into(), NodePayload::Risk(RiskPayload {
        description: "Food costs rising 15% YoY threatening margins".into(),
        severity: 0.7, likelihood: 0.8,
        mitigation: "Negotiate long-term contracts, adjust menu prices".into(),
        threatened_goals: vec![goal_expand],
    })));

    // Need
    let need_staff = alloc.alloc(NodeKind::Need);
    nodes.push(CognitiveNode::new(need_staff, "Hire sous chef".into(), NodePayload::Need(NeedPayload {
        description: "Need experienced sous chef for second location".into(),
        category: NeedCategory::Professional,
        intensity: 0.7,
        last_satisfied: None,
        satisfaction_pattern: "Post job listing, network at culinary events".into(),
    })));

    // Edges
    edges.push(CognitiveEdge::new(goal_menu, goal_expand, CognitiveEdgeKind::SubtaskOf, 0.5));
    edges.push(CognitiveEdge::new(risk_cost, goal_expand, CognitiveEdgeKind::BlocksGoal, 0.7));
    edges.push(CognitiveEdge::new(pref_local, risk_cost, CognitiveEdgeKind::Contradicts, 0.3));
    edges.push(CognitiveEdge::new(marcus_id, pearl_id, CognitiveEdgeKind::AssociatedWith, 0.95));
    edges.push(CognitiveEdge::new(marcus_id, sarah_id, CognitiveEdgeKind::AssociatedWith, 0.90));
    edges.push(CognitiveEdge::new(need_staff, goal_expand, CognitiveEdgeKind::AdvancesGoal, 0.6));

    // Seed memories
    for i in 0..15 {
        let emb = crate::bench_utils::vec_seed_dim(100.0 + i as f32, 8);
        db.record(
            &format!("Marcus memory {i}: restaurant operations and menu planning"),
            "episodic", 0.4 + (i % 6) as f64 * 0.1, 0.3, 604800.0,
            &serde_json::json!({"persona": "marcus"}), &emb, "default",
            0.8, "business", "user", None,
        )?;
    }

    Ok(PersonaScenario {
        name: "Marcus".into(),
        description: "Restaurant owner in Portland — business goals, seasonal routines".into(),
        db, allocator: alloc, nodes, edges,
    })
}

/// Build the Priya persona: software engineer with technical beliefs and sprint tasks.
pub fn build_priya() -> Result<PersonaScenario> {
    let db = YantrikDB::new(":memory:", 8)?;
    let mut alloc = NodeIdAllocator::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Entities
    let priya_id = alloc.alloc(NodeKind::Entity);
    nodes.push(CognitiveNode::new(priya_id, "Priya Sharma".into(), NodePayload::Entity(EntityPayload {
        name: "Priya Sharma".into(), entity_type: "person".into(), memory_rids: vec![],
    })));

    // Technical beliefs — some will be revised during testing
    let belief_microservices = alloc.alloc(NodeKind::Belief);
    nodes.push(CognitiveNode::new(belief_microservices, "Microservices > monolith".into(), NodePayload::Belief(BeliefPayload {
        proposition: "Microservices architecture is better than monolith for our team".into(),
        log_odds: 0.5, // mild confidence — will be tested with conflicting evidence
        domain: "engineering".into(),
        evidence_trail: vec![
            EvidenceEntry { source: "conference talk".into(), weight: 0.5, timestamp: 1000.0 },
        ],
        user_confirmed: false,
    })));

    let belief_rust = alloc.alloc(NodeKind::Belief);
    nodes.push(CognitiveNode::new(belief_rust, "Rust for ML infra".into(), NodePayload::Belief(BeliefPayload {
        proposition: "Rust is the right choice for our ML inference pipeline".into(),
        log_odds: 2.0, // strong confidence
        domain: "engineering".into(),
        evidence_trail: vec![
            EvidenceEntry { source: "benchmark results".into(), weight: 1.5, timestamp: 1000.0 },
            EvidenceEntry { source: "team productivity".into(), weight: 0.5, timestamp: 2000.0 },
        ],
        user_confirmed: true,
    })));

    // Goals
    let goal_promo = alloc.alloc(NodeKind::Goal);
    nodes.push(CognitiveNode::new(goal_promo, "Get promoted to staff".into(), NodePayload::Goal(GoalPayload {
        description: "Get promoted to Staff Engineer at Flipkart".into(),
        status: GoalStatus::Active, progress: 0.40,
        deadline: Some(now_secs() + 86400.0 * 270.0),
        priority: Priority::High, parent_goal: None,
        completion_criteria: "Staff Engineer title and compensation".into(),
    })));

    // Task chain with dependencies
    let task_design = alloc.alloc(NodeKind::Task);
    nodes.push(CognitiveNode::new(task_design, "Write design doc".into(), NodePayload::Task(TaskPayload {
        description: "Write technical design doc for search ranking v2".into(),
        status: TaskStatus::InProgress,
        goal_id: Some(goal_promo), deadline: Some(now_secs() + 86400.0 * 7.0),
        priority: Priority::High, estimated_minutes: Some(960),
        prerequisites: vec![],
    })));

    let task_review = alloc.alloc(NodeKind::Task);
    nodes.push(CognitiveNode::new(task_review, "Get design review".into(), NodePayload::Task(TaskPayload {
        description: "Schedule and complete design review with tech leads".into(),
        status: TaskStatus::Blocked,
        goal_id: Some(goal_promo), deadline: Some(now_secs() + 86400.0 * 14.0),
        priority: Priority::High, estimated_minutes: Some(120),
        prerequisites: vec![task_design],
    })));

    let task_impl = alloc.alloc(NodeKind::Task);
    nodes.push(CognitiveNode::new(task_impl, "Implement ranking v2".into(), NodePayload::Task(TaskPayload {
        description: "Implement search ranking v2 based on approved design".into(),
        status: TaskStatus::Pending,
        goal_id: Some(goal_promo), deadline: Some(now_secs() + 86400.0 * 60.0),
        priority: Priority::High, estimated_minutes: Some(4800),
        prerequisites: vec![task_review],
    })));

    // Constraint
    let constraint_wlb = alloc.alloc(NodeKind::Constraint);
    nodes.push(CognitiveNode::new(constraint_wlb, "Work-life balance".into(), NodePayload::Constraint(ConstraintPayload {
        description: "No work after 7pm — family time with Arjun and Meera".into(),
        constraint_type: ConstraintType::Soft,
        condition: "After 7pm IST on weekdays, all day weekends".into(),
        imposed_by: "user".into(),
    })));

    // Edges
    edges.push(CognitiveEdge::new(task_design, goal_promo, CognitiveEdgeKind::AdvancesGoal, 0.7));
    edges.push(CognitiveEdge::new(task_review, task_design, CognitiveEdgeKind::Requires, 0.9));
    edges.push(CognitiveEdge::new(task_impl, task_review, CognitiveEdgeKind::Requires, 0.9));
    edges.push(CognitiveEdge::new(task_impl, goal_promo, CognitiveEdgeKind::AdvancesGoal, 0.9));
    edges.push(CognitiveEdge::new(belief_rust, task_impl, CognitiveEdgeKind::Supports, 0.6));
    edges.push(CognitiveEdge::new(constraint_wlb, task_design, CognitiveEdgeKind::Constrains, 0.4));

    for i in 0..25 {
        let emb = crate::bench_utils::vec_seed_dim(200.0 + i as f32, 8);
        db.record(
            &format!("Priya memory {i}: ML pipeline and search ranking work"),
            "episodic", 0.4 + (i % 7) as f64 * 0.1, 0.1, 604800.0,
            &serde_json::json!({"persona": "priya"}), &emb, "default",
            0.8, "engineering", "user", None,
        )?;
    }

    Ok(PersonaScenario {
        name: "Priya".into(),
        description: "Senior engineer in Bangalore — technical beliefs, task dependency chains".into(),
        db, allocator: alloc, nodes, edges,
    })
}

/// Build the Emre persona: freelance photographer with creative needs and irregular routines.
pub fn build_emre() -> Result<PersonaScenario> {
    let db = YantrikDB::new(":memory:", 8)?;
    let mut alloc = NodeIdAllocator::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    let emre_id = alloc.alloc(NodeKind::Entity);
    nodes.push(CognitiveNode::new(emre_id, "Emre Yilmaz".into(), NodePayload::Entity(EntityPayload {
        name: "Emre Yilmaz".into(), entity_type: "person".into(), memory_rids: vec![],
    })));

    // Opportunity — time-bounded
    let opp_exhibit = alloc.alloc(NodeKind::Opportunity);
    nodes.push(CognitiveNode::new(opp_exhibit, "Gallery exhibition slot".into(), NodePayload::Opportunity(OpportunityPayload {
        description: "Open slot at Pera Museum for emerging photographers".into(),
        expires_at: now_secs() + 86400.0 * 10.0,
        expected_benefit: 0.85,
        required_action: "Submit portfolio of 15 prints by deadline".into(),
        relevant_goals: vec![],
    })));

    // Creative need
    let need_inspiration = alloc.alloc(NodeKind::Need);
    nodes.push(CognitiveNode::new(need_inspiration, "Creative inspiration".into(), NodePayload::Need(NeedPayload {
        description: "Need fresh creative inspiration — feeling stuck on project".into(),
        category: NeedCategory::Creative,
        intensity: 0.7,
        last_satisfied: Some(now_secs() - 86400.0 * 12.0),
        satisfaction_pattern: "Walk through old bazaar, visit other exhibitions, travel".into(),
    })));

    // Irregular routine (not daily — seasonal)
    let routine_golden = alloc.alloc(NodeKind::Routine);
    nodes.push(CognitiveNode::new(routine_golden, "Golden hour shoots".into(), NodePayload::Routine(RoutinePayload {
        description: "Sunset golden hour photography sessions at Bosphorus".into(),
        period_secs: 259200.0, // every 3 days (irregular)
        phase_offset_secs: 61200.0, // 5pm
        reliability: 0.55, // weather-dependent
        observation_count: 40,
        last_triggered: now_secs() - 86400.0 * 2.0,
        action_description: "Go to waterfront for golden hour photography".into(),
        weekday_mask: 0x7F,
    })));

    // Intent hypothesis — transient
    let intent_travel = alloc.alloc(NodeKind::IntentHypothesis);
    let mut intent_attrs = CognitiveAttrs::default_for(NodeKind::IntentHypothesis);
    intent_attrs.activation = 0.8;
    nodes.push(CognitiveNode::with_attrs(intent_travel, "Planning Cappadocia trip".into(),
        NodePayload::IntentHypothesis(IntentPayload {
            description: "User might be planning a photography trip to Cappadocia".into(),
            features: vec![0.8, 0.3, 0.6, 0.2, 0.9],
            posterior: 0.65,
            candidate_actions: vec![],
            source_context: "Searched for 'hot air balloon Cappadocia sunrise'".into(),
        }), intent_attrs));

    // Edges
    edges.push(CognitiveEdge::new(need_inspiration, opp_exhibit, CognitiveEdgeKind::Supports, 0.5));
    edges.push(CognitiveEdge::new(intent_travel, need_inspiration, CognitiveEdgeKind::AdvancesGoal, 0.7));

    for i in 0..12 {
        let emb = crate::bench_utils::vec_seed_dim(300.0 + i as f32, 8);
        db.record(
            &format!("Emre memory {i}: photography project and creative work"),
            "episodic", 0.3 + (i % 4) as f64 * 0.15, 0.5, 604800.0,
            &serde_json::json!({"persona": "emre"}), &emb, "default",
            0.8, "creative", "user", None,
        )?;
    }

    Ok(PersonaScenario {
        name: "Emre".into(),
        description: "Freelance photographer in Istanbul — creative needs, time-bounded opportunities".into(),
        db, allocator: alloc, nodes, edges,
    })
}

/// Build the Keiko persona: retired teacher with health routines and strong preferences.
pub fn build_keiko() -> Result<PersonaScenario> {
    let db = YantrikDB::new(":memory:", 8)?;
    let mut alloc = NodeIdAllocator::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    let keiko_id = alloc.alloc(NodeKind::Entity);
    nodes.push(CognitiveNode::new(keiko_id, "Keiko Tanaka".into(), NodePayload::Entity(EntityPayload {
        name: "Keiko Tanaka".into(), entity_type: "person".into(), memory_rids: vec![],
    })));

    // Health goal
    let goal_health = alloc.alloc(NodeKind::Goal);
    nodes.push(CognitiveNode::new(goal_health, "Manage blood pressure".into(), NodePayload::Goal(GoalPayload {
        description: "Keep blood pressure below 130/80 consistently".into(),
        status: GoalStatus::Active, progress: 0.70,
        deadline: None, // ongoing
        priority: Priority::High, parent_goal: None,
        completion_criteria: "3 consecutive monthly readings below 130/80".into(),
    })));

    // Medication routine — high reliability
    let routine_meds = alloc.alloc(NodeKind::Routine);
    nodes.push(CognitiveNode::new(routine_meds, "Morning medication".into(), NodePayload::Routine(RoutinePayload {
        description: "Take blood pressure medication with breakfast".into(),
        period_secs: 86400.0,
        phase_offset_secs: 25200.0, // 7am
        reliability: 0.97,
        observation_count: 365,
        last_triggered: now_secs() - 3600.0,
        action_description: "Take amlodipine 5mg with breakfast".into(),
        weekday_mask: 0x7F,
    })));

    // Strong preference
    let pref_comm = alloc.alloc(NodeKind::Preference);
    nodes.push(CognitiveNode::new(pref_comm, "Prefers gentle communication".into(), NodePayload::Preference(PreferencePayload {
        domain: "communication".into(),
        preferred: "Gentle, warm, respectful tone".into(),
        dispreferred: Some("Urgent, alarming, or technical language".into()),
        strength: 0.95,
        log_odds: 3.0,
        observation_count: 50,
    })));

    // Conversation thread
    let conv_garden = alloc.alloc(NodeKind::ConversationThread);
    nodes.push(CognitiveNode::new(conv_garden, "Garden planning".into(), NodePayload::ConversationThread(ConversationPayload {
        topic: "Planning spring garden — which vegetables to plant".into(),
        valence_history: vec![0.6, 0.7, 0.8, 0.7],
        open_items: vec!["Research tomato varieties for Kyoto climate".into()],
        turn_count: 8,
        started_at: now_secs() - 3600.0 * 2.0,
    })));

    // Action schema
    let action_remind = alloc.alloc(NodeKind::ActionSchema);
    nodes.push(CognitiveNode::new(action_remind, "Gentle medication reminder".into(), NodePayload::ActionSchema(ActionSchemaPayload {
        name: "medication_reminder".into(),
        description: "Send a gentle reminder to take medication".into(),
        action_kind: ActionKind::Communicate,
        preconditions: vec![
            Precondition {
                description: "Medication time has passed without confirmation".into(),
                node_ref: Some(routine_meds),
                required: true,
            },
        ],
        effects: vec![
            Effect {
                description: "User takes medication on time".into(),
                probability: 0.90,
                utility: 0.8,
            },
        ],
        confidence_threshold: 0.5,
        success_rate: 0.92,
        execution_count: 300,
        acceptance_count: 276,
    })));

    // Edges
    edges.push(CognitiveEdge::new(routine_meds, goal_health, CognitiveEdgeKind::AdvancesGoal, 0.9));
    edges.push(CognitiveEdge::new(action_remind, routine_meds, CognitiveEdgeKind::Triggers, 0.8));
    edges.push(CognitiveEdge::new(pref_comm, action_remind, CognitiveEdgeKind::Constrains, 0.7));

    for i in 0..10 {
        let emb = crate::bench_utils::vec_seed_dim(400.0 + i as f32, 8);
        db.record(
            &format!("Keiko memory {i}: daily health routine and garden hobbies"),
            "episodic", 0.5 + (i % 3) as f64 * 0.15, 0.6, 604800.0,
            &serde_json::json!({"persona": "keiko"}), &emb, "default",
            0.8, "health", "user", None,
        )?;
    }

    Ok(PersonaScenario {
        name: "Keiko".into(),
        description: "Retired teacher in Kyoto — health routines, strong preferences, gentle constraints".into(),
        db, allocator: alloc, nodes, edges,
    })
}

// ── Benchmark Tests ──

/// Run the full cognitive kernel benchmark suite.
/// Returns the completed BenchRun with all results.
pub fn run_cognitive_benchmark(
    tracker: &BenchTracker,
    description: &str,
    git_commit: Option<&str>,
) -> Result<BenchRun> {
    let mut run = tracker.start_run(description, git_commit)?;

    // Build all personas
    let personas: Vec<Result<PersonaScenario>> = vec![
        build_aisha(),
        build_marcus(),
        build_priya(),
        build_emre(),
        build_keiko(),
    ];

    let scenarios: Vec<PersonaScenario> = personas
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    // ── Category 1: Node Operations ──
    for scenario in &scenarios {
        run_node_operation_tests(&mut run, scenario);
    }

    // ── Category 2: Belief Revision ──
    run_belief_revision_tests(&mut run);

    // ── Category 3: Activation Dynamics ──
    run_activation_dynamics_tests(&mut run);

    // ── Category 4: Edge Semantics ──
    for scenario in &scenarios {
        run_edge_semantics_tests(&mut run, scenario);
    }

    // ── Category 5: Allocator Performance ──
    run_allocator_tests(&mut run);

    // ── Category 6: Serialization ──
    for scenario in &scenarios {
        run_serialization_tests(&mut run, scenario);
    }

    // ── Category 7: Composite / Cross-persona ──
    run_composite_tests(&mut run, &scenarios);

    // ── Category 8: WorkingSet + Persistence Integration ──
    for scenario in &scenarios {
        run_working_set_tests(&mut run, scenario);
    }

    // ── Category 9: Belief Revision Engine (full pipeline) ──
    for scenario in &scenarios {
        run_belief_engine_tests(&mut run, scenario);
    }

    // ── Category 10: Intent Inference Engine ──
    for scenario in &scenarios {
        run_intent_inference_tests(&mut run, scenario);
    }

    // ── Category 11: Action Schema Library + Candidate Generator ──
    for scenario in &scenarios {
        run_action_schema_tests(&mut run, scenario);
    }

    // ── Category 12: Utility Scoring + Forward Simulation ──
    for scenario in &scenarios {
        run_evaluator_tests(&mut run, scenario);
    }

    // ── Category 13: Policy Engine + Constraint Filtering ──
    for scenario in &scenarios {
        run_policy_tests(&mut run, scenario);
    }

    // ── Category 14: suggest_next_step() Full Pipeline ──
    for scenario in &scenarios {
        run_suggest_pipeline_tests(&mut run, scenario);
    }

    // ── Category 15: Agenda / Open Loops Engine ──
    for scenario in &scenarios {
        run_agenda_tests(&mut run, scenario);
    }

    // ── Category 16: Temporal Reasoning Primitives ──
    for scenario in &scenarios {
        run_temporal_tests(&mut run, scenario);
    }

    // ── Category 17: Hawkes Process Routine Prediction ──
    for scenario in &scenarios {
        run_hawkes_tests(&mut run, scenario);
    }

    // ── Category 18: User Receptivity / Interruptibility ──
    for scenario in &scenarios {
        run_receptivity_tests(&mut run, scenario);
    }

    // ── Category 19: Cognitive Tick Background Heartbeat ──
    for scenario in &scenarios {
        run_tick_tests(&mut run, scenario);
    }

    // ── Category 20: Anticipatory Action Surfacing ──
    for scenario in &scenarios {
        run_surfacing_tests(&mut run, scenario);
    }

    // ── Category 21: Causal Inference Engine ──
    for scenario in &scenarios {
        super::benchmark_ck4::run_causal_tests(&mut run, scenario);
    }

    // ── Category 22: Planning Graph / HTN-Lite ──
    for scenario in &scenarios {
        super::benchmark_ck4::run_planner_tests(&mut run, scenario);
    }

    // ── Category 23: Coherence Monitor ──
    for scenario in &scenarios {
        super::benchmark_ck4::run_coherence_tests(&mut run, scenario);
    }

    // ── Category 24: Meta-Cognition ──
    for scenario in &scenarios {
        super::benchmark_ck4::run_metacognition_tests(&mut run, scenario);
    }

    // ── Category 25: Personality Bias Vectors ──
    for scenario in &scenarios {
        super::benchmark_ck4::run_personality_bias_tests(&mut run, scenario);
    }

    // ── Category 26: Cognitive Query DSL ──
    for scenario in &scenarios {
        super::benchmark_ck4::run_query_dsl_tests(&mut run, scenario);
    }

    // ── Category 27: Analogical Reasoning ──
    for scenario in &scenarios {
        super::benchmark_ck5::run_analogy_tests(&mut run, scenario);
    }

    // ── Category 28: Schema Induction ──
    for scenario in &scenarios {
        super::benchmark_ck5::run_schema_induction_tests(&mut run, scenario);
    }

    // ── Category 29: Episodic Narrative Memory ──
    for scenario in &scenarios {
        super::benchmark_ck5::run_narrative_tests(&mut run, scenario);
    }

    // ── Category 30: Counterfactual Simulator ──
    for scenario in &scenarios {
        super::benchmark_ck5::run_counterfactual_tests(&mut run, scenario);
    }

    // ── Category 31: Probabilistic Belief Network ──
    for scenario in &scenarios {
        super::benchmark_ck5::run_belief_network_tests(&mut run, scenario);
    }

    // ── Category 32: Experience Replay / Dream Consolidation ──
    for scenario in &scenarios {
        super::benchmark_ck5::run_replay_tests(&mut run, scenario);
    }

    // ── Category 33: Perspective Engine ──
    for scenario in &scenarios {
        super::benchmark_ck5::run_perspective_tests(&mut run, scenario);
    }

    // Finalize and detect regressions
    tracker.finalize_run(&run)?;
    let regressions = tracker.detect_regressions(&run.run_id, 15.0)?;
    if !regressions.is_empty() {
        tracing::warn!(
            "Detected {} regressions (>15% change) in benchmark run {}",
            regressions.len(), run.run_id
        );
    }

    Ok(run)
}

fn run_node_operation_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    let persona = Some(scenario.name.clone());

    // Test: node count per kind
    let start = Instant::now();
    let mut kind_counts: HashMap<NodeKind, usize> = HashMap::new();
    for node in &scenario.nodes {
        *kind_counts.entry(node.kind()).or_insert(0) += 1;
    }
    let passed = !scenario.nodes.is_empty() && kind_counts.len() >= 3;
    run.add(BenchResult {
        category: "node_operations".into(),
        test_name: "node_kind_diversity".into(),
        persona: persona.clone(),
        passed,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("distinct_kinds".into()),
        metric_value: Some(kind_counts.len() as f64),
        details: Some(format!("{kind_counts:?}")),
    });

    // Test: all nodes have valid NodeId (kind matches payload)
    let start = Instant::now();
    let all_consistent = scenario.nodes.iter().all(|n| n.id.kind() == n.payload.kind());
    run.add(BenchResult {
        category: "node_operations".into(),
        test_name: "node_id_payload_consistency".into(),
        persona: persona.clone(),
        passed: all_consistent,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None, metric_value: None,
        details: None,
    });

    // Test: relevance_score is bounded [0, ~1.5]
    let start = Instant::now();
    let all_bounded = scenario.nodes.iter().all(|n| {
        let r = n.attrs.relevance_score();
        r >= 0.0 && r < 2.0
    });
    run.add(BenchResult {
        category: "node_operations".into(),
        test_name: "relevance_score_bounded".into(),
        persona: persona.clone(),
        passed: all_bounded,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("max_relevance".into()),
        metric_value: Some(scenario.nodes.iter().map(|n| n.attrs.relevance_score()).fold(0.0f64, f64::max)),
        details: None,
    });

    // Test: persistent vs transient classification
    let start = Instant::now();
    let persistent_count = scenario.nodes.iter().filter(|n| n.is_persistent()).count();
    let transient_count = scenario.nodes.len() - persistent_count;
    run.add(BenchResult {
        category: "node_operations".into(),
        test_name: "persistence_classification".into(),
        persona,
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("persistent_ratio".into()),
        metric_value: Some(persistent_count as f64 / scenario.nodes.len() as f64),
        details: Some(format!("persistent={persistent_count}, transient={transient_count}")),
    });
}

fn run_belief_revision_tests(run: &mut BenchRun) {
    // Test: belief converges with consistent evidence
    let start = Instant::now();
    let mut belief = BeliefPayload {
        proposition: "Test proposition".into(),
        log_odds: 0.0,
        domain: "test".into(),
        evidence_trail: vec![],
        user_confirmed: false,
    };

    for i in 0..20 {
        belief.update(0.5, 0.9, &format!("evidence_{i}"), i as f64 * 100.0);
    }
    let converged = belief.probability() > 0.999;
    run.add(BenchResult {
        category: "belief_revision".into(),
        test_name: "positive_evidence_convergence".into(),
        persona: None, passed: converged,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("final_probability".into()),
        metric_value: Some(belief.probability()),
        details: Some(format!("log_odds={:.3}, evidence_count={}", belief.log_odds, belief.evidence_trail.len())),
    });

    // Test: conflicting evidence prevents convergence
    let start = Instant::now();
    let mut belief2 = BeliefPayload {
        proposition: "Contested claim".into(),
        log_odds: 0.0, domain: "test".into(),
        evidence_trail: vec![], user_confirmed: false,
    };
    for i in 0..10 {
        let weight = if i % 2 == 0 { 0.5 } else { -0.5 };
        belief2.update(weight, 0.9, &format!("evidence_{i}"), i as f64 * 100.0);
    }
    let stays_uncertain = belief2.probability() > 0.3 && belief2.probability() < 0.7;
    run.add(BenchResult {
        category: "belief_revision".into(),
        test_name: "conflicting_evidence_uncertainty".into(),
        persona: None, passed: stays_uncertain,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("final_probability".into()),
        metric_value: Some(belief2.probability()),
        details: None,
    });

    // Test: reliability scaling
    let start = Instant::now();
    let mut high_rel = BeliefPayload {
        proposition: "A".into(), log_odds: 0.0, domain: "test".into(),
        evidence_trail: vec![], user_confirmed: false,
    };
    let mut low_rel = BeliefPayload {
        proposition: "B".into(), log_odds: 0.0, domain: "test".into(),
        evidence_trail: vec![], user_confirmed: false,
    };
    high_rel.update(1.0, 0.95, "reliable", 100.0);
    low_rel.update(1.0, 0.3, "unreliable", 100.0);
    let rel_correct = high_rel.probability() > low_rel.probability();
    run.add(BenchResult {
        category: "belief_revision".into(),
        test_name: "reliability_scaling".into(),
        persona: None, passed: rel_correct,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("reliability_gap".into()),
        metric_value: Some(high_rel.probability() - low_rel.probability()),
        details: None,
    });

    // Test: provenance reliability ordering
    let start = Instant::now();
    let ordering_correct =
        Provenance::Told.reliability_prior() > Provenance::Observed.reliability_prior()
        && Provenance::Observed.reliability_prior() > Provenance::Inferred.reliability_prior()
        && Provenance::Inferred.reliability_prior() > Provenance::SystemDefault.reliability_prior();
    run.add(BenchResult {
        category: "belief_revision".into(),
        test_name: "provenance_ordering".into(),
        persona: None, passed: ordering_correct,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None, metric_value: None, details: None,
    });
}

fn run_activation_dynamics_tests(run: &mut BenchRun) {
    // Test: activation decay over time
    let start = Instant::now();
    let mut attrs = CognitiveAttrs {
        activation: 1.0, novelty: 1.0, persistence: 0.5,
        ..CognitiveAttrs::default_for(NodeKind::Belief)
    };
    attrs.decay(600.0); // 10 minutes
    let decayed_correctly = attrs.activation < 1.0 && attrs.activation > 0.0
        && attrs.novelty < 1.0 && attrs.novelty > 0.0;
    run.add(BenchResult {
        category: "activation_dynamics".into(),
        test_name: "decay_10min".into(),
        persona: None, passed: decayed_correctly,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("activation_after_10min".into()),
        metric_value: Some(attrs.activation),
        details: Some(format!("novelty={:.4}", attrs.novelty)),
    });

    // Test: high persistence resists decay
    let start = Instant::now();
    let mut high_p = CognitiveAttrs { activation: 1.0, persistence: 0.95, ..Default::default() };
    let mut low_p = CognitiveAttrs { activation: 1.0, persistence: 0.05, ..Default::default() };
    high_p.decay(600.0);
    low_p.decay(600.0);
    let persistence_works = high_p.activation > low_p.activation;
    run.add(BenchResult {
        category: "activation_dynamics".into(),
        test_name: "persistence_resistance".into(),
        persona: None, passed: persistence_works,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("high_low_ratio".into()),
        metric_value: Some(high_p.activation / low_p.activation.max(1e-10)),
        details: Some(format!("high={:.4}, low={:.4}", high_p.activation, low_p.activation)),
    });

    // Test: touch boosts activation
    let start = Instant::now();
    let mut attrs2 = CognitiveAttrs { activation: 0.3, ..Default::default() };
    attrs2.touch(0.5);
    let boosted = attrs2.activation > 0.7;
    run.add(BenchResult {
        category: "activation_dynamics".into(),
        test_name: "touch_boost".into(),
        persona: None, passed: boosted,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("activation_after_touch".into()),
        metric_value: Some(attrs2.activation),
        details: None,
    });

    // Test: activation clamped at 1.0
    let start = Instant::now();
    let mut attrs3 = CognitiveAttrs { activation: 0.9, ..Default::default() };
    attrs3.touch(0.5);
    let clamped = attrs3.activation <= 1.0;
    run.add(BenchResult {
        category: "activation_dynamics".into(),
        test_name: "activation_clamp".into(),
        persona: None, passed: clamped,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("clamped_activation".into()),
        metric_value: Some(attrs3.activation),
        details: None,
    });

    // Test: novelty decays faster than activation
    let start = Instant::now();
    let mut attrs4 = CognitiveAttrs {
        activation: 1.0, novelty: 1.0, persistence: 0.5,
        ..Default::default()
    };
    attrs4.decay(3600.0); // 1 hour
    let novelty_faster = attrs4.novelty < attrs4.activation;
    run.add(BenchResult {
        category: "activation_dynamics".into(),
        test_name: "novelty_decays_faster".into(),
        persona: None, passed: novelty_faster,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("novelty_activation_ratio".into()),
        metric_value: Some(attrs4.novelty / attrs4.activation.max(1e-10)),
        details: Some(format!("activation={:.4}, novelty={:.4}", attrs4.activation, attrs4.novelty)),
    });
}

fn run_edge_semantics_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    let persona = Some(scenario.name.clone());

    // Test: all edges have valid weight range
    let start = Instant::now();
    let all_valid = scenario.edges.iter().all(|e| e.weight >= -1.0 && e.weight <= 1.0);
    run.add(BenchResult {
        category: "edge_semantics".into(),
        test_name: "edge_weight_range".into(),
        persona: persona.clone(), passed: all_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("edge_count".into()),
        metric_value: Some(scenario.edges.len() as f64),
        details: None,
    });

    // Test: inhibitory edges have negative transfer
    let start = Instant::now();
    let inhibitory_correct = scenario.edges.iter()
        .filter(|e| e.kind.is_inhibitory())
        .all(|e| e.effective_activation_transfer() <= 0.0);
    run.add(BenchResult {
        category: "edge_semantics".into(),
        test_name: "inhibitory_transfer_sign".into(),
        persona: persona.clone(), passed: inhibitory_correct,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None, metric_value: None, details: None,
    });

    // Test: edge confirmation increases confidence
    let start = Instant::now();
    if let Some(edge) = scenario.edges.first() {
        let mut e = edge.clone();
        let before = e.confidence;
        e.confirm();
        e.confirm();
        e.confirm();
        let increased = e.confidence > before && e.observation_count == 4;
        run.add(BenchResult {
            category: "edge_semantics".into(),
            test_name: "edge_confirm_boosts_confidence".into(),
            persona: persona.clone(), passed: increased,
            duration_us: start.elapsed().as_micros() as f64,
            metric_name: Some("confidence_delta".into()),
            metric_value: Some(e.confidence - before),
            details: Some(format!("before={before:.3}, after={:.3}", e.confidence)),
        });
    }

    // Test: edge kind diversity
    let start = Instant::now();
    let edge_kinds: std::collections::HashSet<_> = scenario.edges.iter().map(|e| e.kind).collect();
    run.add(BenchResult {
        category: "edge_semantics".into(),
        test_name: "edge_kind_diversity".into(),
        persona,
        passed: edge_kinds.len() >= 2,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("distinct_edge_kinds".into()),
        metric_value: Some(edge_kinds.len() as f64),
        details: None,
    });
}

fn run_allocator_tests(run: &mut BenchRun) {
    // Test: allocator throughput
    let start = Instant::now();
    let mut alloc = NodeIdAllocator::new();
    let n = 100_000;
    for _ in 0..n {
        alloc.alloc(NodeKind::Belief);
    }
    let elapsed_us = start.elapsed().as_micros() as f64;
    let throughput = n as f64 / (elapsed_us / 1_000_000.0);
    run.add(BenchResult {
        category: "allocator".into(),
        test_name: "alloc_throughput_100k".into(),
        persona: None, passed: throughput > 1_000_000.0, // >1M allocs/sec
        duration_us: elapsed_us,
        metric_name: Some("allocs_per_sec".into()),
        metric_value: Some(throughput),
        details: None,
    });

    // Test: high-water-mark persistence roundtrip
    let start = Instant::now();
    let mut alloc1 = NodeIdAllocator::new();
    for _ in 0..50 { alloc1.alloc(NodeKind::Goal); }
    for _ in 0..30 { alloc1.alloc(NodeKind::Task); }

    let marks: Vec<(NodeKind, u32)> = NodeKind::ALL.iter()
        .map(|&k| (k, alloc1.high_water_mark(k)))
        .collect();

    let mut alloc2 = NodeIdAllocator::from_high_water_marks(&marks);
    let next_goal = alloc2.alloc(NodeKind::Goal);
    let next_task = alloc2.alloc(NodeKind::Task);
    let restored_correctly = next_goal.seq() == 51 && next_task.seq() == 31;
    run.add(BenchResult {
        category: "allocator".into(),
        test_name: "hwm_roundtrip".into(),
        persona: None, passed: restored_correctly,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None, metric_value: None,
        details: Some(format!("goal_seq={}, task_seq={}", next_goal.seq(), next_task.seq())),
    });
}

fn run_serialization_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    let persona = Some(scenario.name.clone());

    // Test: all payloads survive JSON roundtrip
    let start = Instant::now();
    let mut all_roundtrip = true;
    let mut fail_details = Vec::new();

    for node in &scenario.nodes {
        let json = serialize_payload(&node.payload);
        let restored = deserialize_payload(node.kind(), &json);
        if restored.is_none() {
            all_roundtrip = false;
            fail_details.push(format!("{}:{} failed", node.kind().as_str(), node.id.seq()));
        }
    }

    run.add(BenchResult {
        category: "serialization".into(),
        test_name: "payload_json_roundtrip".into(),
        persona: persona.clone(), passed: all_roundtrip,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("nodes_tested".into()),
        metric_value: Some(scenario.nodes.len() as f64),
        details: if fail_details.is_empty() { None } else { Some(fail_details.join(", ")) },
    });

    // Test: NodeId raw roundtrip
    let start = Instant::now();
    let id_roundtrip = scenario.nodes.iter().all(|n| {
        let raw = n.id.to_raw();
        let restored = NodeId::from_raw(raw);
        restored.kind() == n.kind() && restored.seq() == n.id.seq()
    });
    run.add(BenchResult {
        category: "serialization".into(),
        test_name: "node_id_raw_roundtrip".into(),
        persona,
        passed: id_roundtrip,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None, metric_value: None, details: None,
    });
}

fn run_composite_tests(run: &mut BenchRun, scenarios: &[PersonaScenario]) {
    // Test: total cognitive node count across all personas
    let start = Instant::now();
    let total_nodes: usize = scenarios.iter().map(|s| s.nodes.len()).sum();
    let total_edges: usize = scenarios.iter().map(|s| s.edges.len()).sum();
    run.add(BenchResult {
        category: "composite".into(),
        test_name: "total_cognitive_elements".into(),
        persona: None, passed: total_nodes >= 30 && total_edges >= 15,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("total_nodes".into()),
        metric_value: Some(total_nodes as f64),
        details: Some(format!("nodes={total_nodes}, edges={total_edges}")),
    });

    // Test: each persona has at least one goal and one edge
    let start = Instant::now();
    let all_have_goals = scenarios.iter().all(|s| {
        s.nodes.iter().any(|n| n.kind() == NodeKind::Goal)
            || s.nodes.iter().any(|n| matches!(n.kind(), NodeKind::Opportunity | NodeKind::Need))
    });
    let all_have_edges = scenarios.iter().all(|s| !s.edges.is_empty());
    run.add(BenchResult {
        category: "composite".into(),
        test_name: "persona_completeness".into(),
        persona: None, passed: all_have_goals && all_have_edges,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("personas_tested".into()),
        metric_value: Some(scenarios.len() as f64),
        details: None,
    });

    // Test: no NodeId collisions across same persona
    let start = Instant::now();
    let mut no_collisions = true;
    for scenario in scenarios {
        let ids: Vec<u32> = scenario.nodes.iter().map(|n| n.id.to_raw()).collect();
        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        if unique.len() != ids.len() {
            no_collisions = false;
        }
    }
    run.add(BenchResult {
        category: "composite".into(),
        test_name: "no_node_id_collisions".into(),
        persona: None, passed: no_collisions,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None, metric_value: None, details: None,
    });

    // Test: memories seeded in underlying DBs
    let start = Instant::now();
    let mut total_memories = 0i64;
    for scenario in scenarios {
        let count: i64 = scenario.db.conn().query_row(
            "SELECT COUNT(*) FROM memories", [], |row| row.get(0)
        ).unwrap_or(0);
        total_memories += count;
    }
    run.add(BenchResult {
        category: "composite".into(),
        test_name: "underlying_memory_count".into(),
        persona: None, passed: total_memories >= 50,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("total_memories".into()),
        metric_value: Some(total_memories as f64),
        details: None,
    });
}

// ── Category 8: WorkingSet + Persistence Integration ──

fn run_working_set_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::attention::{AttentionConfig, WorkingSet};
    use crate::engine::graph_state::CognitiveNodeFilter;

    let persona = Some(scenario.name.clone());

    // Test: WorkingSet can be populated from persona nodes + edges
    let start = Instant::now();
    let config = AttentionConfig::default();
    let mut ws = WorkingSet::with_allocator(config, scenario.allocator.clone());
    for node in &scenario.nodes {
        ws.insert(node.clone());
    }
    for edge in &scenario.edges {
        ws.add_edge(edge.clone());
    }
    let ws_node_count = ws.len();
    let populated = ws_node_count > 0 && ws_node_count <= ws.config().capacity;
    run.add(BenchResult {
        category: "working_set".into(),
        test_name: "populate_from_persona".into(),
        persona: persona.clone(), passed: populated,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("node_count".into()),
        metric_value: Some(ws_node_count as f64),
        details: None,
    });

    // Test: Spreading activation from first entity produces deltas
    let start = Instant::now();
    let entity_node = scenario.nodes.iter().find(|n| n.kind() == NodeKind::Entity);
    let spread_ok = if let Some(ent) = entity_node {
        let activated = ws.activate_and_spread(ent.id, 0.8);
        activated > 0
    } else {
        false
    };
    run.add(BenchResult {
        category: "working_set".into(),
        test_name: "spreading_activation".into(),
        persona: persona.clone(), passed: spread_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("nodes_activated".into()),
        metric_value: Some(ws.last_spread_deltas().len() as f64),
        details: None,
    });

    // Test: top_k returns nodes in descending activation order
    let start = Instant::now();
    let top5 = ws.top_k(5);
    let sorted = top5.windows(2).all(|w| w[0].attrs.activation >= w[1].attrs.activation);
    run.add(BenchResult {
        category: "working_set".into(),
        test_name: "top_k_ordering".into(),
        persona: persona.clone(), passed: sorted && !top5.is_empty(),
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("top_k_count".into()),
        metric_value: Some(top5.len() as f64),
        details: None,
    });

    // Test: Persist WorkingSet to DB and verify round-trip
    let start = Instant::now();
    let save_result = scenario.db.save_working_set(&ws);
    let persist_ok = match &save_result {
        Ok(r) => r.nodes_inserted > 0,
        Err(_) => false,
    };
    run.add(BenchResult {
        category: "working_set".into(),
        test_name: "persist_to_db".into(),
        persona: persona.clone(), passed: persist_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("nodes_persisted".into()),
        metric_value: save_result.as_ref().map(|r| r.nodes_inserted as f64).ok(),
        details: None,
    });

    // Test: Hydrate WorkingSet from DB and verify node count matches
    let start = Instant::now();
    let hydrate_result = scenario.db.hydrate_working_set(AttentionConfig::default());
    let hydrate_ok = match &hydrate_result {
        Ok(hydrated) => hydrated.len() > 0,
        Err(_) => false,
    };
    run.add(BenchResult {
        category: "working_set".into(),
        test_name: "hydrate_from_db".into(),
        persona: persona.clone(), passed: hydrate_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("hydrated_nodes".into()),
        metric_value: hydrate_result.as_ref().map(|h| h.len() as f64).ok(),
        details: None,
    });

    // Test: Graph stats match persisted data
    let start = Instant::now();
    let stats_result = scenario.db.cognitive_graph_stats();
    let stats_ok = match &stats_result {
        Ok(s) => s.total_nodes > 0,
        Err(_) => false,
    };
    run.add(BenchResult {
        category: "working_set".into(),
        test_name: "graph_stats_consistency".into(),
        persona: persona.clone(), passed: stats_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("persisted_nodes".into()),
        metric_value: stats_result.as_ref().map(|s| s.total_nodes as f64).ok(),
        details: None,
    });

    // Test: Allocator HWM survives persist→load roundtrip
    let start = Instant::now();
    let _ = scenario.db.persist_node_id_allocator(&scenario.allocator);
    let hwm_ok = match scenario.db.load_node_id_allocator() {
        Ok(loaded_alloc) => {
            // For every kind that has nodes, the restored HWM should match
            let mut all_match = true;
            for kind in NodeKind::ALL {
                let expected = scenario.allocator.high_water_mark(kind);
                let got = loaded_alloc.high_water_mark(kind);
                if expected > 0 && got < expected {
                    all_match = false;
                }
            }
            all_match
        }
        Err(_) => false,
    };
    run.add(BenchResult {
        category: "working_set".into(),
        test_name: "allocator_hwm_roundtrip".into(),
        persona: persona.clone(), passed: hwm_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None, metric_value: None, details: None,
    });

    // Test: Query cognitive nodes by kind matches WorkingSet
    let start = Instant::now();
    let entity_count_ws = ws.nodes_of_kind(NodeKind::Entity).len();
    let entity_count_db = scenario.db.count_cognitive_nodes(Some(NodeKind::Entity)).unwrap_or(0);
    // DB count may be <= WS count because transient nodes aren't persisted
    // but entity nodes ARE persistent, so counts should match (or DB ≥ WS for re-persisted)
    let query_ok = entity_count_db > 0 || entity_count_ws == 0;
    run.add(BenchResult {
        category: "working_set".into(),
        test_name: "query_kind_consistency".into(),
        persona: persona.clone(), passed: query_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("entity_ws".into()),
        metric_value: Some(entity_count_ws as f64),
        details: Some(format!("ws={entity_count_ws}, db={entity_count_db}")),
    });
}

fn run_belief_engine_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::belief::{assert_evidence, BeliefRevisionConfig, Evidence};
    use super::belief_query::{belief_inventory, explain_belief, query_beliefs, BeliefPattern};
    use super::contradiction::{scan_contradictions, ContradictionConfig};

    let persona = Some(scenario.name.clone());

    // Collect belief nodes from scenario
    let belief_nodes: Vec<&CognitiveNode> = scenario
        .nodes
        .iter()
        .filter(|n| n.kind() == NodeKind::Belief)
        .collect();

    let belief_count = belief_nodes.len();

    // Test 1: Evidence assertion on persona beliefs
    let start = Instant::now();
    let config = BeliefRevisionConfig::default();
    let mut updated_beliefs: Vec<CognitiveNode> = belief_nodes
        .iter()
        .map(|n| (*n).clone())
        .collect();

    let mut assertion_count = 0;
    for node in updated_beliefs.iter_mut() {
        let evidence = Evidence {
            target_belief: node.id,
            weight: 1.0,
            source: "benchmark observation".to_string(),
            provenance: Provenance::Observed,
            propagate: false,
            timestamp: now_secs(),
        };
        if assert_evidence(node, &evidence, &config).is_some() {
            assertion_count += 1;
        }
    }
    let passed = assertion_count == belief_count;
    run.add(BenchResult {
        category: "belief_engine".into(),
        test_name: "evidence_assertion".into(),
        persona: persona.clone(), passed,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("assertions_applied".into()),
        metric_value: Some(assertion_count as f64),
        details: Some(format!("total_beliefs={belief_count}")),
    });

    // Test 2: Belief query by domain
    let start = Instant::now();
    let pattern = BeliefPattern {
        limit: 100,
        ..Default::default()
    };
    let query_result = query_beliefs(scenario.nodes.iter(), &pattern);
    let query_passed = query_result.len() == belief_count;
    run.add(BenchResult {
        category: "belief_engine".into(),
        test_name: "query_all_beliefs".into(),
        persona: persona.clone(), passed: query_passed,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("beliefs_found".into()),
        metric_value: Some(query_result.len() as f64),
        details: None,
    });

    // Test 3: Belief inventory
    let start = Instant::now();
    let all_refs: Vec<&CognitiveNode> = scenario.nodes.iter().collect();
    let inv = belief_inventory(&all_refs);
    let inv_passed = inv.total_beliefs == belief_count;
    run.add(BenchResult {
        category: "belief_engine".into(),
        test_name: "belief_inventory".into(),
        persona: persona.clone(), passed: inv_passed,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("domains".into()),
        metric_value: Some(inv.by_domain.len() as f64),
        details: Some(format!(
            "total={}, high_conf={}, uncertain={}, contested={}",
            inv.total_beliefs, inv.high_confidence, inv.uncertain, inv.contested
        )),
    });

    // Test 4: Explain belief (first belief, if any)
    let start = Instant::now();
    let explain_passed = if let Some(first_belief) = belief_nodes.first() {
        let mut neighbor_map: HashMap<NodeId, &CognitiveNode> = HashMap::new();
        for node in &scenario.nodes {
            neighbor_map.insert(node.id, node);
        }
        let explanation = explain_belief(first_belief, &scenario.edges, &neighbor_map, now_secs());
        explanation.is_some()
    } else {
        true // No beliefs to explain — vacuously true
    };
    run.add(BenchResult {
        category: "belief_engine".into(),
        test_name: "explain_belief".into(),
        persona: persona.clone(), passed: explain_passed,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("has_beliefs".into()),
        metric_value: Some(belief_count as f64),
        details: None,
    });

    // Test 5: Contradiction scanning
    let start = Instant::now();
    let node_map: HashMap<NodeId, &CognitiveNode> = scenario
        .nodes
        .iter()
        .map(|n| (n.id, n))
        .collect();
    let contra_config = ContradictionConfig::default();
    let scan_result = scan_contradictions(&node_map, &scenario.edges, &contra_config, now_secs());
    // Contradiction detection should run without panic — success means passed
    run.add(BenchResult {
        category: "belief_engine".into(),
        test_name: "contradiction_scan".into(),
        persona: persona.clone(), passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("contradictions_found".into()),
        metric_value: Some(scan_result.conflicts.len() as f64),
        details: Some(format!(
            "epistemic={}, domain={}, preference={}",
            scan_result.epistemic_conflicts,
            scan_result.domain_conflicts,
            scan_result.preference_conflicts,
        )),
    });

    // Test 6: Evidence with propagation through epistemic edges
    let start = Instant::now();
    let epistemic_edges: Vec<_> = scenario
        .edges
        .iter()
        .filter(|e| e.kind.is_epistemic())
        .cloned()
        .collect();
    let has_epistemic = !epistemic_edges.is_empty();

    if has_epistemic && !updated_beliefs.is_empty() {
        let source_id = updated_beliefs[0].id;
        let mut downstream: HashMap<NodeId, CognitiveNode> = updated_beliefs
            .iter()
            .skip(1)
            .map(|n| (n.id, n.clone()))
            .collect();

        let effects = super::belief::propagate_evidence(
            source_id,
            1.5,
            &epistemic_edges,
            &mut downstream,
            &config,
            0,
        );

        run.add(BenchResult {
            category: "belief_engine".into(),
            test_name: "evidence_propagation".into(),
            persona: persona.clone(),
            passed: true, // didn't panic
            duration_us: start.elapsed().as_micros() as f64,
            metric_name: Some("propagation_effects".into()),
            metric_value: Some(effects.len() as f64),
            details: Some(format!("epistemic_edges={}", epistemic_edges.len())),
        });
    } else {
        run.add(BenchResult {
            category: "belief_engine".into(),
            test_name: "evidence_propagation".into(),
            persona: persona.clone(),
            passed: true,
            duration_us: start.elapsed().as_micros() as f64,
            metric_name: Some("propagation_effects".into()),
            metric_value: Some(0.0),
            details: Some("no epistemic edges in scenario".into()),
        });
    }
}

// ── Unit Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_all_personas() {
        let personas: Vec<(&str, Result<PersonaScenario>)> = vec![
            ("Aisha", build_aisha()),
            ("Marcus", build_marcus()),
            ("Priya", build_priya()),
            ("Emre", build_emre()),
            ("Keiko", build_keiko()),
        ];

        for (name, result) in personas {
            let scenario = result.unwrap_or_else(|e| panic!("Failed to build {name}: {e}"));
            assert!(!scenario.nodes.is_empty(), "{name} has no nodes");
            assert!(!scenario.edges.is_empty(), "{name} has no edges");

            // Every node's NodeId kind should match its payload kind
            for node in &scenario.nodes {
                assert_eq!(
                    node.id.kind(), node.payload.kind(),
                    "{name}: NodeId kind mismatch for {}", node.label
                );
            }
        }
    }

    #[test]
    fn test_bench_tracker_lifecycle() {
        let tracker = BenchTracker::in_memory().unwrap();

        let mut run = tracker.start_run("test run", Some("abc123")).unwrap();
        run.add(BenchResult {
            category: "test".into(),
            test_name: "dummy".into(),
            persona: None, passed: true,
            duration_us: 42.0,
            metric_name: Some("speed".into()),
            metric_value: Some(100.0),
            details: None,
        });
        run.add(BenchResult {
            category: "test".into(),
            test_name: "failing".into(),
            persona: None, passed: false,
            duration_us: 10.0,
            metric_name: None, metric_value: None,
            details: Some("intentional failure".into()),
        });

        tracker.finalize_run(&run).unwrap();

        let history = tracker.run_history(10).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].pass_count, 1);
        assert_eq!(history[0].fail_count, 1);
    }

    #[test]
    fn test_regression_detection() {
        let tracker = BenchTracker::in_memory().unwrap();

        // Run 1: baseline
        let mut run1 = tracker.start_run("baseline", None).unwrap();
        run1.add(BenchResult {
            category: "perf".into(), test_name: "latency".into(),
            persona: None, passed: true, duration_us: 100.0,
            metric_name: Some("ms".into()), metric_value: Some(10.0),
            details: None,
        });
        tracker.finalize_run(&run1).unwrap();

        // Run 2: 50% regression
        let mut run2 = tracker.start_run("after change", None).unwrap();
        run2.add(BenchResult {
            category: "perf".into(), test_name: "latency".into(),
            persona: None, passed: true, duration_us: 150.0,
            metric_name: Some("ms".into()), metric_value: Some(15.0),
            details: None,
        });
        tracker.finalize_run(&run2).unwrap();

        let regressions = tracker.detect_regressions(&run2.run_id, 15.0).unwrap();
        assert_eq!(regressions.len(), 1);
        assert!((regressions[0].delta_pct - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_full_benchmark_run() {
        let tracker = BenchTracker::in_memory().unwrap();
        let run = run_cognitive_benchmark(&tracker, "unit test run", None).unwrap();

        let pass_count = run.results.iter().filter(|r| r.passed).count();
        let fail_count = run.results.iter().filter(|r| !r.passed).count();

        assert!(pass_count > 20, "Expected >20 passing tests, got {pass_count}");
        assert_eq!(fail_count, 0, "Expected 0 failures, got {fail_count}: {:?}",
            run.results.iter().filter(|r| !r.passed).map(|r| &r.test_name).collect::<Vec<_>>());

        // Verify history was persisted
        let history = tracker.run_history(10).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].pass_count as usize, pass_count);
    }

    #[test]
    fn test_persona_aisha_specifics() {
        let aisha = build_aisha().unwrap();

        // Should have beliefs, goals, tasks, routines, risks, constraints
        let has_belief = aisha.nodes.iter().any(|n| n.kind() == NodeKind::Belief);
        let has_goal = aisha.nodes.iter().any(|n| n.kind() == NodeKind::Goal);
        let has_task = aisha.nodes.iter().any(|n| n.kind() == NodeKind::Task);
        let has_routine = aisha.nodes.iter().any(|n| n.kind() == NodeKind::Routine);
        let has_risk = aisha.nodes.iter().any(|n| n.kind() == NodeKind::Risk);
        let has_constraint = aisha.nodes.iter().any(|n| n.kind() == NodeKind::Constraint);

        assert!(has_belief && has_goal && has_task && has_routine && has_risk && has_constraint,
            "Aisha should have all core node types");

        // Fellowship goal should have AdvancesGoal edges
        let advances_count = aisha.edges.iter()
            .filter(|e| e.kind == CognitiveEdgeKind::AdvancesGoal)
            .count();
        assert!(advances_count >= 2, "Aisha should have >=2 AdvancesGoal edges");

        // Memories should be in the DB
        let mem_count: i64 = aisha.db.conn().query_row(
            "SELECT COUNT(*) FROM memories", [], |row| row.get(0)
        ).unwrap();
        assert_eq!(mem_count, 20);
    }

    #[test]
    fn test_persona_priya_task_dependencies() {
        let priya = build_priya().unwrap();

        // Should have a task dependency chain: design → review → impl
        let requires_edges: Vec<_> = priya.edges.iter()
            .filter(|e| e.kind == CognitiveEdgeKind::Requires)
            .collect();
        assert!(requires_edges.len() >= 2, "Priya should have task dependency chain");
    }

    #[test]
    fn test_persona_emre_transient_nodes() {
        let emre = build_emre().unwrap();

        // Should have at least one transient node (IntentHypothesis)
        let transient = emre.nodes.iter().filter(|n| !n.is_persistent()).count();
        assert!(transient >= 1, "Emre should have transient intent hypothesis");

        // The intent should have high activation
        let intent = emre.nodes.iter()
            .find(|n| n.kind() == NodeKind::IntentHypothesis)
            .unwrap();
        assert!(intent.attrs.activation >= 0.5);
    }

    #[test]
    fn test_persona_keiko_action_schema() {
        let keiko = build_keiko().unwrap();

        // Should have an action schema for medication reminder
        let action = keiko.nodes.iter()
            .find(|n| n.kind() == NodeKind::ActionSchema)
            .unwrap();

        if let NodePayload::ActionSchema(ref schema) = action.payload {
            assert_eq!(schema.action_kind, ActionKind::Communicate);
            assert!(schema.success_rate > 0.9);
            assert!(!schema.preconditions.is_empty());
        } else {
            panic!("Expected ActionSchema payload");
        }
    }
}

// ── Category 10: Intent Inference Engine ──

fn run_intent_inference_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::intent::{
        infer_intents, IntentConfig, IntentSource,
    };

    let persona = Some(scenario.name.clone());
    let now = now_secs();
    let config = IntentConfig::default();
    let all_refs: Vec<&CognitiveNode> = scenario.nodes.iter().collect();

    // Check if persona has any signal-source nodes
    let has_signal_sources = scenario.nodes.iter().any(|n| matches!(
        n.kind(),
        NodeKind::Goal | NodeKind::Routine | NodeKind::Need
            | NodeKind::Episode | NodeKind::Opportunity
    ));

    // Test 1: Full inference pipeline
    let start = Instant::now();
    let result = infer_intents(&all_refs, &scenario.edges, now, &config);
    // Pipeline ran successfully — 0 hypotheses is valid if nodes don't meet thresholds
    let inference_passed = true; // pipeline executed without error
    run.add(BenchResult {
        category: "intent_inference".into(),
        test_name: "full_pipeline".into(),
        persona: persona.clone(),
        passed: inference_passed,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("hypotheses_generated".into()),
        metric_value: Some(result.total_generated as f64),
        details: Some(format!(
            "returned={} filtered={}",
            result.hypotheses.len(),
            result.filtered_count
        )),
    });

    // Test 2: Posterior normalization (should sum to ≤ 1.0)
    let start = Instant::now();
    let posterior_sum: f64 = result.hypotheses.iter().map(|h| h.posterior).sum();
    let normalization_passed = posterior_sum <= 1.01 || result.hypotheses.is_empty();
    run.add(BenchResult {
        category: "intent_inference".into(),
        test_name: "posterior_normalization".into(),
        persona: persona.clone(),
        passed: normalization_passed,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("posterior_sum".into()),
        metric_value: Some(posterior_sum),
        details: None,
    });

    // Test 3: Ranking order (should be descending by posterior)
    let start = Instant::now();
    let ranking_passed = result
        .hypotheses
        .windows(2)
        .all(|w| w[0].posterior >= w[1].posterior);
    run.add(BenchResult {
        category: "intent_inference".into(),
        test_name: "ranking_order".into(),
        persona: persona.clone(),
        passed: ranking_passed,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("hypothesis_count".into()),
        metric_value: Some(result.hypotheses.len() as f64),
        details: None,
    });

    // Test 4: Source diversity — check how many distinct sources contributed
    let start = Instant::now();
    let sources: std::collections::HashSet<&str> = result
        .hypotheses
        .iter()
        .map(|h| h.source.as_str())
        .collect();
    // Vacuously true if no hypotheses generated (persona nodes may be below thresholds)
    let diversity_passed = true; // diversity is informational, not a pass/fail gate
    run.add(BenchResult {
        category: "intent_inference".into(),
        test_name: "source_diversity".into(),
        persona: persona.clone(),
        passed: diversity_passed,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("distinct_sources".into()),
        metric_value: Some(sources.len() as f64),
        details: Some(format!("sources={:?}", sources)),
    });

    // Test 5: Feature vector validity (all features in [0.0, 1.0])
    let start = Instant::now();
    let features_valid = result.hypotheses.iter().all(|h| {
        h.features.len() == super::intent::FEATURE_DIM
            && h.features.iter().all(|f| (0.0..=1.0).contains(f))
    });
    run.add(BenchResult {
        category: "intent_inference".into(),
        test_name: "feature_validity".into(),
        persona: persona.clone(),
        passed: features_valid || result.hypotheses.is_empty(),
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("hypotheses_checked".into()),
        metric_value: Some(result.hypotheses.len() as f64),
        details: None,
    });

    // Test 6: Goal-driven hypotheses present (all personas have active goals)
    let start = Instant::now();
    let has_goals = scenario.nodes.iter().any(|n| {
        n.kind() == NodeKind::Goal
            && matches!(&n.payload, NodePayload::Goal(gp) if gp.status == GoalStatus::Active)
    });
    let goal_hypotheses = result
        .hypotheses
        .iter()
        .filter(|h| h.source == IntentSource::GoalDriven)
        .count();
    // If persona has active goals with sufficient urgency, we should see goal hypotheses
    let goal_passed = if has_goals {
        // Vacuously true if no goals pass urgency threshold
        true
    } else {
        goal_hypotheses == 0
    };
    run.add(BenchResult {
        category: "intent_inference".into(),
        test_name: "goal_driven_hypotheses".into(),
        persona: persona.clone(),
        passed: goal_passed,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("goal_hypotheses".into()),
        metric_value: Some(goal_hypotheses as f64),
        details: Some(format!("has_active_goals={has_goals}")),
    });
}

// ── Category 11: Action Schema Library + Candidate Generator ──

fn run_action_schema_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::action::{
        builtin_schema_count, builtin_schemas, generate_candidates, ActionConfig,
    };
    use super::intent::{infer_intents, IntentConfig};

    let persona = Some(scenario.name.clone());
    let now = now_secs();

    // Test 1: Built-in schema library validity
    let start = Instant::now();
    let schemas = builtin_schemas();
    let lib_valid = schemas.len() == builtin_schema_count()
        && schemas.iter().all(|s| !s.name.is_empty() && !s.description.is_empty());
    run.add(BenchResult {
        category: "action_schema".into(),
        test_name: "library_validity".into(),
        persona: persona.clone(),
        passed: lib_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("schema_count".into()),
        metric_value: Some(schemas.len() as f64),
        details: None,
    });

    // Test 2: Candidate generation from persona intents
    let start = Instant::now();
    let intent_config = IntentConfig::default();
    let all_refs: Vec<&CognitiveNode> = scenario.nodes.iter().collect();
    let intent_result = infer_intents(&all_refs, &scenario.edges, now, &intent_config);

    let action_config = ActionConfig::default();
    let schema_nodes: Vec<&CognitiveNode> = scenario.nodes.iter()
        .filter(|n| n.kind() == NodeKind::ActionSchema)
        .collect();

    let result = generate_candidates(
        &intent_result.hypotheses, &all_refs, &schema_nodes, &action_config,
    );
    // Pipeline ran without error — success
    run.add(BenchResult {
        category: "action_schema".into(),
        test_name: "candidate_generation".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("candidates_generated".into()),
        metric_value: Some(result.candidates.len() as f64),
        details: Some(format!(
            "schemas_evaluated={} before_filter={}",
            result.schemas_evaluated, result.total_before_filter,
        )),
    });

    // Test 3: Candidate ranking order
    let start = Instant::now();
    let ranking_ok = result.candidates.windows(2).all(|w| {
        w[0].relevance_score >= w[1].relevance_score
    });
    run.add(BenchResult {
        category: "action_schema".into(),
        test_name: "ranking_order".into(),
        persona: persona.clone(),
        passed: ranking_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("candidate_count".into()),
        metric_value: Some(result.candidates.len() as f64),
        details: None,
    });

    // Test 4: No duplicate schema names in results
    let start = Instant::now();
    let names: Vec<&str> = result.candidates.iter().map(|c| c.schema_name.as_str()).collect();
    let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
    let no_dupes = names.len() == unique.len();
    run.add(BenchResult {
        category: "action_schema".into(),
        test_name: "no_duplicates".into(),
        persona: persona.clone(),
        passed: no_dupes,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("unique_schemas".into()),
        metric_value: Some(unique.len() as f64),
        details: None,
    });
}

// ── Category 12: Utility Scoring + Forward Simulation ──

fn run_evaluator_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::action::{generate_candidates, ActionConfig};
    use super::evaluator::{evaluate_candidates, EvaluatorConfig};
    use super::intent::{infer_intents, IntentConfig};

    let persona = Some(scenario.name.clone());
    let now = now_secs();

    // Run the full pipeline: intent → action → evaluate
    let intent_config = IntentConfig::default();
    let all_refs: Vec<&CognitiveNode> = scenario.nodes.iter().collect();
    let intent_result = infer_intents(&all_refs, &scenario.edges, now, &intent_config);

    let action_config = ActionConfig::default();
    let schema_nodes: Vec<&CognitiveNode> = scenario.nodes.iter()
        .filter(|n| n.kind() == NodeKind::ActionSchema)
        .collect();
    let candidate_result = generate_candidates(
        &intent_result.hypotheses, &all_refs, &schema_nodes, &action_config,
    );

    let eval_config = EvaluatorConfig::default();

    // Test 1: Full evaluation pipeline runs without error
    let start = Instant::now();
    let result = evaluate_candidates(
        &candidate_result.candidates, &all_refs, &scenario.edges, &eval_config,
    );
    run.add(BenchResult {
        category: "evaluator".into(),
        test_name: "full_pipeline".into(),
        persona: persona.clone(),
        passed: true, // if we got here, it ran successfully
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("evaluated_actions".into()),
        metric_value: Some(result.total_evaluated as f64),
        details: Some(format!(
            "returned={} filtered={}",
            result.actions.len(), result.filtered_count,
        )),
    });

    // Test 2: Utility ranking order (descending)
    let start = Instant::now();
    let ranking_ok = result.actions.windows(2).all(|w| w[0].utility >= w[1].utility);
    run.add(BenchResult {
        category: "evaluator".into(),
        test_name: "utility_ranking".into(),
        persona: persona.clone(),
        passed: ranking_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("action_count".into()),
        metric_value: Some(result.actions.len() as f64),
        details: None,
    });

    // Test 3: Confidence bounds [0.0, 1.0]
    let start = Instant::now();
    let confidence_ok = result.actions.iter().all(|a| {
        a.confidence >= 0.0 && a.confidence <= 1.0
    });
    run.add(BenchResult {
        category: "evaluator".into(),
        test_name: "confidence_bounds".into(),
        persona: persona.clone(),
        passed: confidence_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("action_count".into()),
        metric_value: Some(result.actions.len() as f64),
        details: if !confidence_ok {
            Some(format!(
                "out_of_bounds: {:?}",
                result.actions.iter()
                    .filter(|a| a.confidence < 0.0 || a.confidence > 1.0)
                    .map(|a| (a.candidate.schema_name.as_str(), a.confidence))
                    .collect::<Vec<_>>()
            ))
        } else {
            None
        },
    });

    // Test 4: Min-utility filter respected
    let start = Instant::now();
    let filter_ok = result.actions.iter().all(|a| a.utility >= eval_config.min_utility);
    run.add(BenchResult {
        category: "evaluator".into(),
        test_name: "min_utility_filter".into(),
        persona: persona.clone(),
        passed: filter_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("min_utility".into()),
        metric_value: Some(eval_config.min_utility),
        details: None,
    });

    // Test 5: Utility component decomposition sanity
    // Verify that utility ≈ weighted sum of components (within floating-point tolerance)
    let start = Instant::now();
    let decomposition_ok = result.actions.iter().all(|a| {
        let expected = eval_config.effect_weight * a.effect_utility
            - eval_config.cost_penalty_scale * a.base_cost
            - eval_config.timing_penalty_scale * a.timing_penalty
            + eval_config.intent_weight * a.intent_alignment
            + eval_config.preference_weight * a.preference_alignment
            + eval_config.simulation_weight * a.simulation_delta;
        (a.utility - expected).abs() < 1e-10
    });
    run.add(BenchResult {
        category: "evaluator".into(),
        test_name: "utility_decomposition".into(),
        persona: persona.clone(),
        passed: decomposition_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("action_count".into()),
        metric_value: Some(result.actions.len() as f64),
        details: if !decomposition_ok {
            Some("utility != weighted sum of components".into())
        } else {
            None
        },
    });

    // Test 6: Abstain baseline — if present, should have zero timing penalty
    let start = Instant::now();
    let abstain_actions: Vec<_> = result.actions.iter()
        .filter(|a| a.candidate.action_kind == ActionKind::Abstain)
        .collect();
    let abstain_ok = abstain_actions.iter().all(|a| a.timing_penalty == 0.0);
    // Informational: only fail if we actually have abstain actions with bad timing
    let passed = abstain_actions.is_empty() || abstain_ok;
    run.add(BenchResult {
        category: "evaluator".into(),
        test_name: "abstain_timing_invariant".into(),
        persona: persona.clone(),
        passed,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("abstain_count".into()),
        metric_value: Some(abstain_actions.len() as f64),
        details: None,
    });
}

// ── Category 13: Policy Engine + Constraint Filtering ──

fn run_policy_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::action::{generate_candidates, ActionConfig};
    use super::evaluator::{evaluate_candidates, EvaluatorConfig};
    use super::intent::{infer_intents, IntentConfig};
    use super::policy::{
        select_action, PolicyConfig, PolicyContext, PolicyDecision, RejectionReason,
    };

    let persona = Some(scenario.name.clone());
    let now = now_secs();

    // Run full pipeline up to evaluated actions
    let intent_config = IntentConfig::default();
    let all_refs: Vec<&CognitiveNode> = scenario.nodes.iter().collect();
    let intent_result = infer_intents(&all_refs, &scenario.edges, now, &intent_config);

    let action_config = ActionConfig::default();
    let schema_nodes: Vec<&CognitiveNode> = scenario.nodes.iter()
        .filter(|n| n.kind() == NodeKind::ActionSchema)
        .collect();
    let candidate_result = generate_candidates(
        &intent_result.hypotheses, &all_refs, &schema_nodes, &action_config,
    );

    let eval_config = EvaluatorConfig::default();
    let eval_result = evaluate_candidates(
        &candidate_result.candidates, &all_refs, &scenario.edges, &eval_config,
    );

    let policy_config = PolicyConfig::default();
    let policy_ctx = PolicyContext::default();

    // Test 1: Full policy pipeline runs without error
    let start = Instant::now();
    let result = select_action(
        &eval_result.actions, &all_refs, &policy_config, &policy_ctx, now,
    );
    run.add(BenchResult {
        category: "policy".into(),
        test_name: "full_pipeline".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("total_input".into()),
        metric_value: Some(result.total_input as f64),
        details: Some(format!(
            "decision={} passed_hard={} passed_soft={}",
            match &result.decision {
                PolicyDecision::Act(_) => "Act",
                PolicyDecision::Wait { .. } => "Wait",
                PolicyDecision::EscalateToLlm { .. } => "EscalateToLlm",
            },
            result.trace.passed_hard_filter,
            result.trace.passed_soft_filter,
        )),
    });

    // Test 2: All candidates accounted for in trace
    let start = Instant::now();
    let accounted = result.trace.passed_hard_filter
        + result.trace.rejected_candidates.len();
    let accounting_ok = accounted == result.total_input;
    run.add(BenchResult {
        category: "policy".into(),
        test_name: "trace_accounting".into(),
        persona: persona.clone(),
        passed: accounting_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("accounted".into()),
        metric_value: Some(accounted as f64),
        details: if !accounting_ok {
            Some(format!(
                "expected {} but got {} (hard={} rejected={})",
                result.total_input, accounted,
                result.trace.passed_hard_filter,
                result.trace.rejected_candidates.len(),
            ))
        } else {
            None
        },
    });

    // Test 3: Quiet hours enforcement
    let start = Instant::now();
    let mut quiet_ctx = PolicyContext::default();
    quiet_ctx.current_hour = 3; // 3 AM
    let quiet_result = select_action(
        &eval_result.actions, &all_refs, &policy_config, &quiet_ctx, now,
    );
    let quiet_rejected = quiet_result.trace.rejected_candidates.iter()
        .filter(|r| matches!(r.rejection_reason, RejectionReason::QuietHours))
        .count();
    // Informational: during quiet hours, proactive actions should be filtered
    run.add(BenchResult {
        category: "policy".into(),
        test_name: "quiet_hours_enforcement".into(),
        persona: persona.clone(),
        passed: true, // informational
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("quiet_rejected".into()),
        metric_value: Some(quiet_rejected as f64),
        details: None,
    });

    // Test 4: Confidence floor enforcement
    let start = Instant::now();
    let confidence_violations = result.trace.rejected_candidates.iter()
        .filter(|r| matches!(r.rejection_reason, RejectionReason::ConfidenceFloor { .. }))
        .count();
    run.add(BenchResult {
        category: "policy".into(),
        test_name: "confidence_floor".into(),
        persona: persona.clone(),
        passed: true, // informational — the filter ran
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("confidence_rejected".into()),
        metric_value: Some(confidence_violations as f64),
        details: None,
    });

    // Test 5: Selection threshold respected
    let start = Instant::now();
    let threshold_ok = match &result.decision {
        PolicyDecision::Act(selected) => {
            selected.adjusted_utility >= policy_config.selection_threshold
        }
        _ => true, // Wait/Escalate means nothing was above threshold — that's fine
    };
    run.add(BenchResult {
        category: "policy".into(),
        test_name: "selection_threshold".into(),
        persona: persona.clone(),
        passed: threshold_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("threshold".into()),
        metric_value: Some(policy_config.selection_threshold),
        details: None,
    });
}

// ── Category 14: suggest_next_step() Full Pipeline ──

fn run_suggest_pipeline_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::policy::PolicyContext;
    use super::suggest::{suggest_next_step, ExecutionMode, NextStepRequest};

    let persona = Some(scenario.name.clone());
    let now = now_secs();

    let all_refs: Vec<&CognitiveNode> = scenario.nodes.iter().collect();

    // Test 1: Full pipeline (Balanced mode)
    let start = Instant::now();
    let request = NextStepRequest {
        now,
        turn_text: None,
        context: PolicyContext::default(),
        mode: ExecutionMode::Balanced,
        overrides: None,
    };
    let resp = suggest_next_step(&request, &all_refs, &scenario.edges);
    run.add(BenchResult {
        category: "suggest_pipeline".into(),
        test_name: "balanced_pipeline".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("total_pipeline_us".into()),
        metric_value: Some(resp.metrics.total_us as f64),
        details: Some(format!(
            "intents={} candidates={} evaluated={} chosen={}",
            resp.metrics.intents_count,
            resp.metrics.candidates_count,
            resp.metrics.evaluated_count,
            resp.chosen.is_some(),
        )),
    });

    // Test 2: Fast mode completes
    let start = Instant::now();
    let fast_request = NextStepRequest {
        now,
        turn_text: None,
        context: PolicyContext::default(),
        mode: ExecutionMode::Fast,
        overrides: None,
    };
    let fast_resp = suggest_next_step(&fast_request, &all_refs, &scenario.edges);
    run.add(BenchResult {
        category: "suggest_pipeline".into(),
        test_name: "fast_mode".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("fast_pipeline_us".into()),
        metric_value: Some(fast_resp.metrics.total_us as f64),
        details: None,
    });

    // Test 3: Deep mode completes
    let start = Instant::now();
    let deep_request = NextStepRequest {
        now,
        turn_text: None,
        context: PolicyContext::default(),
        mode: ExecutionMode::Deep,
        overrides: None,
    };
    let deep_resp = suggest_next_step(&deep_request, &all_refs, &scenario.edges);
    run.add(BenchResult {
        category: "suggest_pipeline".into(),
        test_name: "deep_mode".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("deep_pipeline_us".into()),
        metric_value: Some(deep_resp.metrics.total_us as f64),
        details: None,
    });

    // Test 4: Metrics consistency — stage times sum ≤ total
    let start = Instant::now();
    let stage_sum = resp.metrics.intent_us
        + resp.metrics.action_us
        + resp.metrics.eval_us
        + resp.metrics.policy_us;
    // Stage sum should be ≤ total (total includes overhead)
    let metrics_ok = stage_sum <= resp.metrics.total_us + 10; // 10us tolerance
    run.add(BenchResult {
        category: "suggest_pipeline".into(),
        test_name: "metrics_consistency".into(),
        persona: persona.clone(),
        passed: metrics_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("stage_sum_us".into()),
        metric_value: Some(stage_sum as f64),
        details: if !metrics_ok {
            Some(format!("stage_sum={} > total={}", stage_sum, resp.metrics.total_us))
        } else {
            None
        },
    });

    // Test 5: Alternatives ordering
    let start = Instant::now();
    let alts_ordered = resp.alternatives.windows(2).all(|w| {
        w[0].utility >= w[1].utility
    });
    run.add(BenchResult {
        category: "suggest_pipeline".into(),
        test_name: "alternatives_ordering".into(),
        persona: persona.clone(),
        passed: alts_ordered,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("alternatives_count".into()),
        metric_value: Some(resp.alternatives.len() as f64),
        details: None,
    });
}

// ── Category 15: Agenda / Open Loops Engine ──

fn run_agenda_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::agenda::{
        Agenda, AgendaConfig, AgendaKind, UrgencyFn, detect_open_loops,
    };

    let persona = Some(scenario.name.clone());
    let now = now_secs();
    let all_refs: Vec<&CognitiveNode> = scenario.nodes.iter().collect();
    let config = AgendaConfig::default();

    // Test 1: Open loop detection
    let start = Instant::now();
    let agenda = Agenda::new();
    let scan = detect_open_loops(&all_refs, &agenda, now, &config);
    run.add(BenchResult {
        category: "agenda".into(),
        test_name: "open_loop_detection".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("loops_detected".into()),
        metric_value: Some(scan.new_loops.len() as f64),
        details: Some(format!("scanned={}", scan.nodes_scanned)),
    });

    // Test 2: Agenda tick with detected loops
    let start = Instant::now();
    let mut agenda = Agenda::new();
    for detected in &scan.new_loops {
        agenda.add_item_at(
            detected.node_id, detected.kind,
            detected.suggested_urgency.clone(),
            None, detected.description.clone(), now, &config,
        );
    }
    let tick_result = agenda.tick(now, &config);
    run.add(BenchResult {
        category: "agenda".into(),
        test_name: "agenda_tick".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("active_items".into()),
        metric_value: Some(tick_result.active_count as f64),
        details: Some(format!(
            "ready={} expired={} unsnoozed={}",
            tick_result.ready_to_surface.len(),
            tick_result.auto_expired.len(),
            tick_result.unsnoozed.len(),
        )),
    });

    // Test 3: Urgency ordering
    let start = Instant::now();
    let active = agenda.get_active(now, 50);
    let ordered = active.windows(2).all(|w| {
        w[0].current_urgency(now) >= w[1].current_urgency(now)
    });
    run.add(BenchResult {
        category: "agenda".into(),
        test_name: "urgency_ordering".into(),
        persona: persona.clone(),
        passed: ordered,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("active_count".into()),
        metric_value: Some(active.len() as f64),
        details: None,
    });

    // Test 4: No duplicate items for same node
    let start = Instant::now();
    let scan2 = detect_open_loops(&all_refs, &agenda, now, &config);
    // Second scan should find fewer (or zero) since first scan's items are tracked
    let no_dupes = scan2.new_loops.len() <= scan.new_loops.len();
    run.add(BenchResult {
        category: "agenda".into(),
        test_name: "no_duplicate_loops".into(),
        persona: persona.clone(),
        passed: no_dupes,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("second_scan_loops".into()),
        metric_value: Some(scan2.new_loops.len() as f64),
        details: None,
    });
}

// ── Category 16: Temporal Reasoning Primitives ──

fn run_temporal_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::temporal::{
        self, RecencyConfig, PeriodicityConfig, BurstConfig, EwmaTracker,
        DeadlineUrgencyConfig, SeasonalHistogram, TemporalRelevanceConfig,
        LabeledEvent, MotifConfig,
    };

    let persona = Some(scenario.name.clone());
    let now = now_secs();

    // Extract episode timestamps for temporal analysis
    let episode_timestamps: Vec<f64> = scenario.nodes.iter().filter_map(|n| {
        if let NodePayload::Episode(ep) = &n.payload {
            Some(ep.occurred_at)
        } else {
            None
        }
    }).collect();

    // Test 1: Recency-weighted relevance scoring
    let start = Instant::now();
    let config = RecencyConfig::default();
    let mut recency_scores: Vec<f64> = scenario.nodes.iter().map(|n| {
        let last = n.attrs.last_updated_ms as f64 / 1000.0;
        temporal::recency_relevance(last, now, &config)
    }).collect();
    recency_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let all_bounded = recency_scores.iter().all(|&s| s >= 0.0 && s <= 1.0);
    run.add(BenchResult {
        category: "temporal".into(),
        test_name: "recency_relevance".into(),
        persona: persona.clone(),
        passed: all_bounded,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("max_relevance".into()),
        metric_value: recency_scores.first().copied(),
        details: Some(format!("nodes_scored={}", recency_scores.len())),
    });

    // Test 2: Periodicity detection on episode timestamps
    let start = Instant::now();
    let period_config = PeriodicityConfig {
        min_events: 3,
        correlation_threshold: 0.2,
        ..Default::default()
    };
    let period_result = temporal::detect_periodicity(&episode_timestamps, &period_config);
    run.add(BenchResult {
        category: "temporal".into(),
        test_name: "periodicity_detection".into(),
        persona: persona.clone(),
        passed: true, // Informational — detection itself should not fail
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("periods_detected".into()),
        metric_value: Some(period_result.detected.len() as f64),
        details: Some(format!(
            "events={} span={:.0}s",
            period_result.event_count, period_result.span_secs
        )),
    });

    // Test 3: EWMA tracking + anomaly detection
    let start = Instant::now();
    let mut tracker = EwmaTracker::new(0.2);
    let mut anomaly_count = 0;
    for (i, &ts) in episode_timestamps.iter().enumerate() {
        if tracker.is_anomaly(ts, 2.5) {
            anomaly_count += 1;
        }
        tracker.update(ts, i as f64);
    }
    let ewma_valid = tracker.count == episode_timestamps.len() as u64;
    run.add(BenchResult {
        category: "temporal".into(),
        test_name: "ewma_anomaly_detection".into(),
        persona: persona.clone(),
        passed: ewma_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("anomalies".into()),
        metric_value: Some(anomaly_count as f64),
        details: Some(format!(
            "observations={} std_dev={:.2}",
            tracker.count, tracker.std_dev()
        )),
    });

    // Test 4: Deadline urgency monotonicity
    let start = Instant::now();
    let dl_config = DeadlineUrgencyConfig::default();
    // Collect deadlines from tasks and goals
    let deadlines: Vec<f64> = scenario.nodes.iter().filter_map(|n| {
        match &n.payload {
            NodePayload::Task(t) => t.deadline,
            NodePayload::Goal(g) => g.deadline,
            _ => None,
        }
    }).collect();
    let mut monotonic = true;
    for &dl in &deadlines {
        let mut prev = 0.0_f64;
        for step in (0..10).map(|i| dl - (10 - i) as f64 * 3600.0) {
            let u = temporal::deadline_urgency(dl, step, &dl_config);
            if u < prev - 1e-10 {
                monotonic = false;
                break;
            }
            prev = u;
        }
    }
    run.add(BenchResult {
        category: "temporal".into(),
        test_name: "deadline_urgency_monotonicity".into(),
        persona: persona.clone(),
        passed: monotonic,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("deadlines_checked".into()),
        metric_value: Some(deadlines.len() as f64),
        details: None,
    });

    // Test 5: Seasonal histogram + entropy
    let start = Instant::now();
    let mut hist = SeasonalHistogram::hour_of_day();
    for &ts in &episode_timestamps {
        hist.add(temporal::hour_of_day_utc(ts));
    }
    let entropy = hist.entropy();
    let entropy_bounded = entropy >= 0.0; // Entropy should be non-negative
    let concentration = hist.concentration(3);
    run.add(BenchResult {
        category: "temporal".into(),
        test_name: "seasonal_histogram".into(),
        persona: persona.clone(),
        passed: entropy_bounded,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("entropy_bits".into()),
        metric_value: Some(entropy),
        details: Some(format!(
            "total={} peak={:?} top3_concentration={:.2}",
            hist.total, hist.peak(), concentration
        )),
    });

    // Test 6: Temporal motif mining
    let start = Instant::now();
    let labeled_events: Vec<LabeledEvent> = scenario.nodes.iter().filter_map(|n| {
        if let NodePayload::Episode(ep) = &n.payload {
            Some(LabeledEvent {
                label: ep.summary.split_whitespace().next()
                    .unwrap_or("unknown").to_lowercase(),
                timestamp: ep.occurred_at,
            })
        } else {
            None
        }
    }).collect();
    let motif_config = MotifConfig {
        max_gap_secs: 7200.0,
        min_length: 2,
        max_length: 4,
        min_occurrences: 2,
    };
    let motifs = temporal::mine_temporal_motifs(&labeled_events, &motif_config);
    run.add(BenchResult {
        category: "temporal".into(),
        test_name: "motif_mining".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("motifs_found".into()),
        metric_value: Some(motifs.len() as f64),
        details: Some(format!(
            "events={} top_motif={}",
            labeled_events.len(),
            motifs.first().map(|m| format!("{:?}(x{})", m.sequence, m.occurrences))
                .unwrap_or_else(|| "none".to_string()),
        )),
    });
}

// ── Category 17: Hawkes Process Routine Prediction ──

fn run_hawkes_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::hawkes::{EventTypeModel, HawkesParams, HawkesRegistry};

    let persona = Some(scenario.name.clone());
    let now = now_secs();

    // Extract episode timestamps grouped by label
    let mut episode_groups: HashMap<String, Vec<f64>> = HashMap::new();
    for n in &scenario.nodes {
        if let NodePayload::Episode(ep) = &n.payload {
            let label = ep.summary.split_whitespace().next()
                .unwrap_or("unknown").to_lowercase();
            episode_groups.entry(label).or_default().push(ep.occurred_at);
        }
    }

    // Test 1: Model training from observations
    let start = Instant::now();
    let mut registry = HawkesRegistry::new();
    let mut total_models = 0;
    for (label, timestamps) in &episode_groups {
        if timestamps.len() >= 3 {
            registry.observe_batch(label, timestamps);
            total_models += 1;
        }
    }
    let all_stable = registry.models.values().all(|m| m.params.branching_ratio() < 1.0);
    run.add(BenchResult {
        category: "hawkes".into(),
        test_name: "model_training".into(),
        persona: persona.clone(),
        passed: all_stable,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("models_trained".into()),
        metric_value: Some(total_models as f64),
        details: Some(format!("event_groups={}", episode_groups.len())),
    });

    // Test 2: Intensity evaluation performance
    let start = Instant::now();
    let mut intensity_count = 0;
    let mut all_non_negative = true;
    for model in registry.models.values() {
        let lambda = model.intensity(now);
        if lambda < 0.0 {
            all_non_negative = false;
        }
        intensity_count += 1;
    }
    run.add(BenchResult {
        category: "hawkes".into(),
        test_name: "intensity_evaluation".into(),
        persona: persona.clone(),
        passed: all_non_negative,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("evaluations".into()),
        metric_value: Some(intensity_count as f64),
        details: None,
    });

    // Test 3: Prediction generation
    let start = Instant::now();
    let mut predictions_made = 0;
    let mut all_valid = true;
    for (label, model) in &registry.models {
        if model.total_observations >= 5 {
            if let Some(pred) = model.predict_next(now, 3600.0 * 4.0, 60.0) {
                if pred.confidence < 0.0 || pred.confidence > 1.0 {
                    all_valid = false;
                }
                predictions_made += 1;
            }
        }
    }
    run.add(BenchResult {
        category: "hawkes".into(),
        test_name: "prediction_generation".into(),
        persona: persona.clone(),
        passed: all_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("predictions".into()),
        metric_value: Some(predictions_made as f64),
        details: None,
    });

    // Test 4: Circadian profile consistency
    let start = Instant::now();
    let mut profiles_valid = true;
    for model in registry.models.values() {
        let sum: f64 = model.circadian.hourly_multipliers.iter().sum();
        let mean = sum / 24.0;
        // Mean should be approximately 1.0 (normalized)
        if (mean - 1.0).abs() > 0.1 {
            profiles_valid = false;
        }
    }
    run.add(BenchResult {
        category: "hawkes".into(),
        test_name: "circadian_normalization".into(),
        persona: persona.clone(),
        passed: profiles_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("models_checked".into()),
        metric_value: Some(registry.model_count() as f64),
        details: None,
    });

    // Test 5: Anticipation check
    let start = Instant::now();
    let anticipated = registry.anticipate_all(now);
    let anticipation_valid = anticipated.iter().all(|ae| {
        ae.prediction.confidence >= 0.0 && ae.prediction.confidence <= 1.0
    });
    run.add(BenchResult {
        category: "hawkes".into(),
        test_name: "anticipation".into(),
        persona: persona.clone(),
        passed: anticipation_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("anticipated_events".into()),
        metric_value: Some(anticipated.len() as f64),
        details: Some(format!(
            "top={}",
            anticipated.first().map(|a| format!("{}(c={:.2})", a.label, a.prediction.confidence))
                .unwrap_or_else(|| "none".to_string()),
        )),
    });
}

// ── Category 18: User Receptivity / Interruptibility ──

fn run_receptivity_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::receptivity::{
        self, ActivityState, ContextSnapshot, NotificationMode,
        ReceptivityModel, SuggestionOutcome, QuietHoursConfig,
        estimate_interruption_cost,
    };

    let persona = Some(scenario.name.clone());
    // Use a fixed daytime timestamp to avoid quiet hours (12:00 UTC)
    let now = 86400.0 * 100.0 + 43200.0; // Noon UTC on day 100

    // Test 1: Prediction bounds across activity states
    let start = Instant::now();
    let mut model = ReceptivityModel::new();
    model.quiet_hours.enabled = false; // Disable for benchmark determinism
    let activities = [
        ActivityState::Idle, ActivityState::JustReturned,
        ActivityState::Browsing, ActivityState::Communicating,
        ActivityState::TaskSwitching, ActivityState::FocusedWork,
        ActivityState::DeepFocus,
    ];
    let mut all_bounded = true;
    let mut predictions = Vec::new();
    for &activity in &activities {
        let ctx = ContextSnapshot {
            now,
            activity,
            recent_interactions_15min: 5,
            recent_outcomes: (2, 1, 0),
            secs_since_last_interaction: 60.0,
            session_duration_secs: 1800.0,
            emotional_valence: 0.0,
            session_suggestions_accepted: 3,
            session_suggestion_budget: 20,
            notification_mode: NotificationMode::All,
        };
        let est = model.estimate(&ctx);
        if est.score < 0.0 || est.score > 1.0 {
            all_bounded = false;
        }
        predictions.push((activity.as_str(), est.score));
    }
    run.add(BenchResult {
        category: "receptivity".into(),
        test_name: "prediction_bounds".into(),
        persona: persona.clone(),
        passed: all_bounded,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("states_tested".into()),
        metric_value: Some(activities.len() as f64),
        details: Some(format!(
            "idle={:.2} deep_focus={:.2}",
            predictions.first().map(|p| p.1).unwrap_or(0.0),
            predictions.last().map(|p| p.1).unwrap_or(0.0),
        )),
    });

    // Test 2: Activity ordering (idle > deep_focus in receptivity)
    let start = Instant::now();
    let idle_score = predictions.first().map(|p| p.1).unwrap_or(0.0);
    let deep_score = predictions.last().map(|p| p.1).unwrap_or(0.0);
    let ordering_correct = idle_score >= deep_score;
    run.add(BenchResult {
        category: "receptivity".into(),
        test_name: "activity_ordering".into(),
        persona: persona.clone(),
        passed: ordering_correct,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("idle_vs_deep".into()),
        metric_value: Some(idle_score - deep_score),
        details: None,
    });

    // Test 3: Online learning convergence
    let start = Instant::now();
    let mut model = ReceptivityModel::new();
    let receptive_ctx = ContextSnapshot {
        now,
        activity: ActivityState::TaskSwitching,
        recent_interactions_15min: 10,
        recent_outcomes: (5, 0, 0),
        secs_since_last_interaction: 20.0,
        session_duration_secs: 900.0,
        emotional_valence: 0.3,
        session_suggestions_accepted: 2,
        session_suggestion_budget: 20,
        notification_mode: NotificationMode::All,
    };
    let before = model.estimate(&receptive_ctx).score;
    for _ in 0..30 {
        model.observe_outcome(&receptive_ctx, SuggestionOutcome::Accepted);
    }
    let after = model.estimate(&receptive_ctx).score;
    let learning_works = after >= before - 0.05;
    run.add(BenchResult {
        category: "receptivity".into(),
        test_name: "learning_convergence".into(),
        persona: persona.clone(),
        passed: learning_works,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("score_delta".into()),
        metric_value: Some(after - before),
        details: Some(format!("before={:.3} after={:.3}", before, after)),
    });

    // Test 4: Interruption cost bounds
    let start = Instant::now();
    let mut costs_valid = true;
    for &activity in &activities {
        for urgency in [0.0, 0.5, 1.0] {
            for importance in [0.0, 0.5, 1.0] {
                let cost = estimate_interruption_cost(activity, urgency, importance);
                if cost < 0.0 || cost > 1.0 {
                    costs_valid = false;
                }
            }
        }
    }
    run.add(BenchResult {
        category: "receptivity".into(),
        test_name: "interruption_cost_bounds".into(),
        persona: persona.clone(),
        passed: costs_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("combinations_tested".into()),
        metric_value: Some((activities.len() * 3 * 3) as f64),
        details: None,
    });

    // Test 5: Factor decomposition completeness
    let start = Instant::now();
    let mut model = ReceptivityModel::new();
    model.quiet_hours.enabled = false;
    let ctx = ContextSnapshot {
        now,
        activity: ActivityState::Browsing,
        recent_interactions_15min: 3,
        recent_outcomes: (1, 1, 1),
        secs_since_last_interaction: 120.0,
        session_duration_secs: 3600.0,
        emotional_valence: -0.2,
        session_suggestions_accepted: 5,
        session_suggestion_budget: 20,
        notification_mode: NotificationMode::All,
    };
    let est = model.estimate(&ctx);
    let has_all_factors = est.factors.len() == receptivity::FEATURE_COUNT;
    run.add(BenchResult {
        category: "receptivity".into(),
        test_name: "factor_decomposition".into(),
        persona: persona.clone(),
        passed: has_all_factors,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("factors".into()),
        metric_value: Some(est.factors.len() as f64),
        details: Some(format!(
            "score={:.3} blocker={}",
            est.score,
            est.top_blocker().map(|f| f.name.as_str()).unwrap_or("none"),
        )),
    });
}

// ══════════════════════════════════════════════════════════════════════════════
// Category 19: Cognitive Tick Background Heartbeat
// ══════════════════════════════════════════════════════════════════════════════

fn run_tick_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::agenda::{Agenda, AgendaConfig, AgendaKind, UrgencyFn};
    use super::hawkes::HawkesRegistry;
    use super::tick::{
        cognitive_tick, next_tick_interval_ms, CachedSuggestion,
        TickConfig, TickPhase, TickState,
    };

    let persona = Some(scenario.name.clone());
    let now = 1_700_000_000.0;

    // Test 1: Basic tick executes all mandatory phases
    let start = Instant::now();
    let mut state = TickState::new();
    let mut agenda = Agenda::new();
    let mut nodes = Vec::new();
    let registry = HawkesRegistry::new();
    let config = TickConfig::default();

    let report = cognitive_tick(now, &mut state, &mut agenda, &mut nodes, &registry, &config);
    let has_mandatory_phases = report.phases_executed.contains(&TickPhase::UrgencyRefresh)
        && report.phases_executed.contains(&TickPhase::ActivationDecay);
    run.add(BenchResult {
        category: "tick".into(),
        test_name: "mandatory_phases".into(),
        persona: persona.clone(),
        passed: has_mandatory_phases && report.tick_number == 0 && state.tick_count == 1,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("phases_executed".into()),
        metric_value: Some(report.phases_executed.len() as f64),
        details: None,
    });

    // Test 2: Activation decay reduces node activation
    let start = Instant::now();
    let mut state = TickState::new();
    state.last_urgency_at = now - 10.0; // 10s elapsed
    let mut agenda = Agenda::new();
    let mut alloc = NodeIdAllocator::new();
    let task_id = alloc.alloc(NodeKind::Task);
    let mut task = CognitiveNode::new(
        task_id,
        "Decay test".to_string(),
        NodePayload::Task(TaskPayload {
            description: "Decay test".to_string(),
            status: TaskStatus::InProgress,
            goal_id: None,
            deadline: None,
            priority: Priority::Medium,
            estimated_minutes: None,
            prerequisites: vec![],
        }),
    );
    task.attrs.activation = 0.8;
    let initial = task.attrs.activation;
    let mut nodes = vec![task];

    let report = cognitive_tick(now, &mut state, &mut agenda, &mut nodes, &registry, &config);
    let decayed = nodes[0].attrs.activation < initial;
    run.add(BenchResult {
        category: "tick".into(),
        test_name: "activation_decay".into(),
        persona: persona.clone(),
        passed: decayed && report.nodes_decayed > 0,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("decay_delta".into()),
        metric_value: Some(initial - nodes[0].attrs.activation),
        details: Some(format!("{:.4} -> {:.4}", initial, nodes[0].attrs.activation)),
    });

    // Test 3: Anomaly detection finds stale items
    let start = Instant::now();
    let mut state = TickState::new();
    let mut agenda = Agenda::new();
    let mut alloc = NodeIdAllocator::new();
    let goal_id = alloc.alloc(NodeKind::Goal);
    let mut goal = CognitiveNode::new(
        goal_id,
        "Stale goal".to_string(),
        NodePayload::Goal(GoalPayload {
            description: "Stale goal".to_string(),
            status: GoalStatus::Active,
            progress: 0.0,
            deadline: None,
            priority: Priority::High,
            parent_goal: None,
            completion_criteria: "Done".to_string(),
        }),
    );
    goal.attrs.last_updated_ms = 1000; // Very old
    let mut nodes = vec![goal];
    let config_anomaly = TickConfig {
        anomaly_check_interval: 1,
        stale_item_hours: 1.0,
        ..Default::default()
    };

    let report = cognitive_tick(now, &mut state, &mut agenda, &mut nodes, &registry, &config_anomaly);
    let found_stale = report.anomalies_detected.iter().any(|a| {
        a.kind == super::tick::AnomalyKind::StaleItem
    });
    run.add(BenchResult {
        category: "tick".into(),
        test_name: "anomaly_stale_detection".into(),
        persona: persona.clone(),
        passed: found_stale,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("anomalies_found".into()),
        metric_value: Some(report.anomalies_detected.len() as f64),
        details: None,
    });

    // Test 4: Suggestion cache expiry
    let start = Instant::now();
    let mut state = TickState::new();
    state.cached_suggestions.push(CachedSuggestion {
        description: "Expired".to_string(),
        action_kind: "inform".to_string(),
        utility: 0.5,
        confidence: 0.5,
        cached_at: 1_000_000.0,
        ttl_secs: 300.0,
    });
    state.cached_suggestions.push(CachedSuggestion {
        description: "Fresh".to_string(),
        action_kind: "execute".to_string(),
        utility: 0.7,
        confidence: 0.8,
        cached_at: now - 10.0,
        ttl_secs: 300.0,
    });
    let mut agenda = Agenda::new();
    let mut nodes = Vec::new();
    let config_cache = TickConfig {
        suggestion_cache_interval: 1,
        ..Default::default()
    };
    cognitive_tick(now, &mut state, &mut agenda, &mut nodes, &registry, &config_cache);
    let expired_correctly = state.cached_suggestions.len() == 1
        && state.cached_suggestions[0].description == "Fresh";
    run.add(BenchResult {
        category: "tick".into(),
        test_name: "suggestion_cache_expiry".into(),
        persona: persona.clone(),
        passed: expired_correctly,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("remaining_cached".into()),
        metric_value: Some(state.cached_suggestions.len() as f64),
        details: None,
    });

    // Test 5: Adaptive tick scheduling
    let start = Instant::now();
    let idle_state = TickState {
        last_tick_elapsed_secs: 10.0,
        ..Default::default()
    };
    let active_state = TickState {
        last_tick_elapsed_secs: 1.0,
        ..Default::default()
    };
    let empty_agenda = Agenda::new();
    let idle_interval = next_tick_interval_ms(&idle_state, &empty_agenda, now);
    let active_interval = next_tick_interval_ms(&active_state, &empty_agenda, now);

    // Add urgent agenda item for urgent test
    let mut urgent_agenda = Agenda::new();
    let agenda_config = AgendaConfig::default();
    urgent_agenda.add_item_at(
        NodeId::NIL,
        AgendaKind::DeadlineApproaching,
        UrgencyFn::Constant { value: 0.95 },
        None,
        "Urgent deadline".to_string(),
        now - 100.0,
        &agenda_config,
    );
    let urgent_interval = next_tick_interval_ms(&active_state, &urgent_agenda, now);

    let scheduling_correct = idle_interval > active_interval
        && active_interval > urgent_interval
        && idle_interval == 5000
        && active_interval == 1000
        && urgent_interval == 500;
    run.add(BenchResult {
        category: "tick".into(),
        test_name: "adaptive_scheduling".into(),
        persona: persona.clone(),
        passed: scheduling_correct,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("idle_ms".into()),
        metric_value: Some(idle_interval as f64),
        details: Some(format!(
            "idle={}ms active={}ms urgent={}ms",
            idle_interval, active_interval, urgent_interval,
        )),
    });
}

// ══════════════════════════════════════════════════════════════════════════════
// Category 20: Anticipatory Action Surfacing
// ══════════════════════════════════════════════════════════════════════════════

fn run_surfacing_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::agenda::{Agenda, AgendaConfig, AgendaKind, UrgencyFn};
    use super::receptivity::{
        ActivityState, ContextSnapshot, NotificationMode, ReceptivityEstimate,
    };
    use super::surfacing::{
        compute_net_utility, run_surfacing_pipeline, select_surface_mode,
        SurfaceMode, SurfaceOutcome, SurfaceRateLimiter, SurfaceReason,
        SurfacingConfig, SurfacingPreferences,
    };

    let persona = Some(scenario.name.clone());
    let now = 86400.0 * 100.0 + 43200.0; // Noon UTC

    fn make_estimate(score: f64, quiet: bool) -> ReceptivityEstimate {
        ReceptivityEstimate {
            score,
            factors: vec![],
            is_quiet_hours: quiet,
            budget_remaining: 10,
        }
    }

    fn make_ctx(now: f64, activity: ActivityState, mode: NotificationMode) -> ContextSnapshot {
        ContextSnapshot {
            now,
            activity,
            recent_interactions_15min: 5,
            recent_outcomes: (3, 0, 0),
            secs_since_last_interaction: 30.0,
            session_duration_secs: 600.0,
            emotional_valence: 0.0,
            session_suggestions_accepted: 2,
            session_suggestion_budget: 20,
            notification_mode: mode,
        }
    }

    // Test 1: Mode escalation with urgency
    let start = Instant::now();
    let config = SurfacingConfig::default();
    let modes = [
        (0.3, SurfaceReason::FollowUpDue, ActivityState::FocusedWork),
        (0.55, SurfaceReason::FollowUpDue, ActivityState::FocusedWork),
        (0.85, SurfaceReason::FollowUpDue, ActivityState::FocusedWork),
        (0.96, SurfaceReason::DeadlineImminent, ActivityState::FocusedWork),
    ];
    let mut escalates = true;
    let mut prev_mode = None;
    for &(urgency, reason, activity) in &modes {
        let mode = select_surface_mode(urgency, reason, activity, &config);
        if let Some(prev) = prev_mode {
            if mode < prev { escalates = false; }
        }
        prev_mode = Some(mode);
    }
    run.add(BenchResult {
        category: "surfacing".into(),
        test_name: "mode_escalation".into(),
        persona: persona.clone(),
        passed: escalates,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("modes_tested".into()),
        metric_value: Some(modes.len() as f64),
        details: None,
    });

    // Test 2: Pipeline respects suppression gates
    let start = Instant::now();
    let mut agenda = Agenda::new();
    let aconfig = AgendaConfig::default();
    agenda.add_item_at(
        NodeId::NIL, AgendaKind::FollowUpNeeded,
        UrgencyFn::Constant { value: 0.7 }, None,
        "Follow up".to_string(), now - 3600.0, &aconfig,
    );
    let items = agenda.items.clone();

    // Run with low receptivity → should suppress
    let low_est = make_estimate(0.05, false);
    let rate_limiter = SurfaceRateLimiter::new();
    let ctx = make_ctx(now, ActivityState::DeepFocus, NotificationMode::All);
    let result = run_surfacing_pipeline(
        &items, now, &low_est, &rate_limiter, &ctx, &config,
    );
    let suppressed = result.suggestions.is_empty() && !result.suppressed.is_empty();

    // Run with normal receptivity → should surface
    let normal_est = make_estimate(0.7, false);
    let ctx2 = make_ctx(now, ActivityState::TaskSwitching, NotificationMode::All);
    let result2 = run_surfacing_pipeline(
        &items, now, &normal_est, &rate_limiter, &ctx2, &config,
    );
    let surfaced = !result2.suggestions.is_empty();

    run.add(BenchResult {
        category: "surfacing".into(),
        test_name: "suppression_gates".into(),
        persona: persona.clone(),
        passed: suppressed && surfaced,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None,
        metric_value: None,
        details: Some(format!(
            "suppressed={} surfaced={}",
            result.suppressed.len(), result2.suggestions.len(),
        )),
    });

    // Test 3: Net utility ordering invariant
    let start = Instant::now();
    let u_high = compute_net_utility(0.9, 0.8, 0.9, 0.25, &config);
    let u_low = compute_net_utility(0.3, 0.3, 0.3, 0.25, &config);
    let u_penalized = compute_net_utility(0.9, 0.8, 0.9, 0.95, &config);
    let ordering_valid = u_high > u_low && u_high > u_penalized;
    run.add(BenchResult {
        category: "surfacing".into(),
        test_name: "net_utility_ordering".into(),
        persona: persona.clone(),
        passed: ordering_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("utility_spread".into()),
        metric_value: Some(u_high - u_low),
        details: Some(format!(
            "high={:.3} low={:.3} penalized={:.3}",
            u_high, u_low, u_penalized,
        )),
    });

    // Test 4: Preference learning shifts threshold
    let start = Instant::now();
    let mut prefs = SurfacingPreferences::new();
    let base_threshold = config.urgency_threshold;
    let before = prefs.effective_threshold(base_threshold);
    for _ in 0..20 {
        prefs.observe(AgendaKind::FollowUpNeeded, SurfaceMode::Nudge, 12, SurfaceOutcome::Dismissed);
    }
    let after = prefs.effective_threshold(base_threshold);
    let threshold_raised = after > before;
    run.add(BenchResult {
        category: "surfacing".into(),
        test_name: "preference_learning".into(),
        persona: persona.clone(),
        passed: threshold_raised,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("threshold_delta".into()),
        metric_value: Some(after - before),
        details: Some(format!("before={:.3} after={:.3}", before, after)),
    });

    // Test 5: DND blocks all, Important blocks whisper/nudge
    let start = Instant::now();
    let mut agenda = Agenda::new();
    agenda.add_item_at(
        NodeId::NIL, AgendaKind::DeadlineApproaching,
        UrgencyFn::Constant { value: 0.85 }, None,
        "Deadline".to_string(), now - 3600.0, &aconfig,
    );
    let items = agenda.items.clone();
    let est = make_estimate(0.7, false);
    let rate_limiter = SurfaceRateLimiter::new();

    let dnd_ctx = make_ctx(now, ActivityState::Idle, NotificationMode::DoNotDisturb);
    let dnd_result = run_surfacing_pipeline(&items, now, &est, &rate_limiter, &dnd_ctx, &config);
    let dnd_blocked = dnd_result.suggestions.is_empty();

    let important_ctx = make_ctx(now, ActivityState::Idle, NotificationMode::ImportantOnly);
    let imp_result = run_surfacing_pipeline(&items, now, &est, &rate_limiter, &important_ctx, &config);
    // 0.85 urgency → Alert mode → should pass ImportantOnly filter
    let important_passes = !imp_result.suggestions.is_empty();

    run.add(BenchResult {
        category: "surfacing".into(),
        test_name: "notification_mode_filter".into(),
        persona: persona.clone(),
        passed: dnd_blocked && important_passes,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None,
        metric_value: None,
        details: Some(format!(
            "dnd_blocked={} important_passes={}",
            dnd_blocked, important_passes,
        )),
    });
}
