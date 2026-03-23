//! Engine-level agenda API.
//!
//! Wires the agenda/open-loops engine into `YantrikDB` methods that
//! persist agenda state and run open loop detection against the DB.

use crate::agenda::{
    self, Agenda, AgendaConfig, AgendaId, AgendaKind, AgendaItem,
    DetectedLoop, OpenLoopScanResult, TickResult, UrgencyFn,
};
use crate::error::Result;
use crate::state::{CognitiveNode, NodeKind};

use super::{now, YantrikDB};

/// Meta key for persisted agenda state.
const AGENDA_META_KEY: &str = "cognitive_agenda";

impl YantrikDB {
    // ── Agenda Persistence ──

    /// Load the agenda from the database (or create a new one).
    pub fn load_agenda(&self) -> Result<Agenda> {
        match Self::get_meta(&self.conn(), AGENDA_META_KEY)? {
            Some(json) => {
                serde_json::from_str(&json).map_err(|e| {
                    crate::error::YantrikDbError::Database(
                        rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                    )
                })
            }
            None => Ok(Agenda::new()),
        }
    }

    /// Persist the agenda to the database.
    pub fn save_agenda(&self, agenda: &Agenda) -> Result<()> {
        let json = serde_json::to_string(agenda).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![AGENDA_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Agenda Operations ──

    /// Add an item to the agenda and persist.
    pub fn agenda_add(
        &self,
        source_node: crate::state::NodeId,
        kind: AgendaKind,
        urgency_fn: UrgencyFn,
        due_at: Option<f64>,
        description: String,
    ) -> Result<AgendaId> {
        let mut agenda = self.load_agenda()?;
        let config = AgendaConfig::default();
        let ts = now();
        let id = agenda.add_item_at(
            source_node, kind, urgency_fn, due_at, description, ts, &config,
        );
        self.save_agenda(&agenda)?;
        Ok(id)
    }

    /// Tick the agenda: recompute urgencies, return items ready to surface.
    pub fn agenda_tick(&self) -> Result<TickResult> {
        let mut agenda = self.load_agenda()?;
        let config = AgendaConfig::default();
        let ts = now();
        let result = agenda.tick(ts, &config);
        self.save_agenda(&agenda)?;
        Ok(result)
    }

    /// Resolve an agenda item.
    pub fn agenda_resolve(&self, id: AgendaId) -> Result<bool> {
        let mut agenda = self.load_agenda()?;
        let resolved = agenda.resolve(id);
        if resolved {
            self.save_agenda(&agenda)?;
        }
        Ok(resolved)
    }

    /// Snooze an agenda item.
    pub fn agenda_snooze(&self, id: AgendaId, duration_secs: f64) -> Result<bool> {
        let mut agenda = self.load_agenda()?;
        let ts = now();
        let snoozed = agenda.snooze(id, ts, duration_secs);
        if snoozed {
            self.save_agenda(&agenda)?;
        }
        Ok(snoozed)
    }

    /// Dismiss an agenda item.
    pub fn agenda_dismiss(&self, id: AgendaId) -> Result<bool> {
        let mut agenda = self.load_agenda()?;
        let dismissed = agenda.dismiss(id);
        if dismissed {
            self.save_agenda(&agenda)?;
        }
        Ok(dismissed)
    }

    /// Get active agenda items sorted by urgency.
    pub fn agenda_active(&self, limit: usize) -> Result<Vec<AgendaItem>> {
        let agenda = self.load_agenda()?;
        let ts = now();
        Ok(agenda.get_active(ts, limit).into_iter().cloned().collect())
    }

    /// Scan the cognitive graph for new open loops and add them to the agenda.
    pub fn agenda_detect_loops(&self) -> Result<OpenLoopScanResult> {
        let mut agenda = self.load_agenda()?;
        let config = AgendaConfig::default();
        let ts = now();

        // Load nodes that can become open loops
        let mut all_nodes: Vec<CognitiveNode> = Vec::new();
        for kind in &[NodeKind::Task, NodeKind::Goal, NodeKind::Belief] {
            all_nodes.extend(self.load_cognitive_nodes_by_kind(*kind)?);
        }
        let node_refs: Vec<&CognitiveNode> = all_nodes.iter().collect();

        let scan = agenda::detect_open_loops(&node_refs, &agenda, ts, &config);

        // Auto-add detected loops to agenda
        for detected in &scan.new_loops {
            agenda.add_item_at(
                detected.node_id,
                detected.kind,
                detected.suggested_urgency.clone(),
                None,
                detected.description.clone(),
                ts,
                &config,
            );
        }

        if !scan.new_loops.is_empty() {
            self.save_agenda(&agenda)?;
        }

        Ok(scan)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::agenda::{AgendaId, AgendaKind, UrgencyFn};
    use crate::engine::YantrikDB;
    use crate::state::*;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_agenda_persistence_roundtrip() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let goal_id = alloc.alloc(NodeKind::Goal);
        let goal = CognitiveNode::new(
            goal_id,
            "Test goal".to_string(),
            NodePayload::Goal(GoalPayload {
                description: "Test goal".to_string(),
                status: GoalStatus::Active,
                progress: 0.3,
                deadline: None,
                priority: Priority::High,
                parent_goal: None,
                completion_criteria: "Done".to_string(),
            }),
        );
        db.persist_cognitive_node(&goal).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        // Add agenda item
        let id = db.agenda_add(
            goal_id,
            AgendaKind::StalledIntent,
            UrgencyFn::Constant { value: 0.7 },
            None,
            "Stalled goal".to_string(),
        ).unwrap();

        // Load and verify
        let agenda = db.load_agenda().unwrap();
        assert_eq!(agenda.active_count(), 1);
        let item = agenda.find(id).unwrap();
        assert_eq!(item.description, "Stalled goal");
    }

    #[test]
    fn test_agenda_resolve_persists() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();
        let goal_id = alloc.alloc(NodeKind::Goal);

        let id = db.agenda_add(
            goal_id,
            AgendaKind::PendingCommitment,
            UrgencyFn::Constant { value: 0.5 },
            None,
            "Commitment".to_string(),
        ).unwrap();

        db.agenda_resolve(id).unwrap();

        let agenda = db.load_agenda().unwrap();
        assert_eq!(agenda.active_count(), 0);
    }

    #[test]
    fn test_agenda_detect_stale_tasks() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let task_id = alloc.alloc(NodeKind::Task);
        let mut task = CognitiveNode::new(
            task_id,
            "Old task".to_string(),
            NodePayload::Task(TaskPayload {
                description: "Old task".to_string(),
                status: TaskStatus::InProgress,
                goal_id: None,
                deadline: None,
                priority: Priority::Medium,
                estimated_minutes: None,
                prerequisites: vec![],
            }),
        );
        // Make it stale — last_updated_ms very old relative to now()
        task.attrs.last_updated_ms = 0;
        db.persist_cognitive_node(&task).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let scan = db.agenda_detect_loops().unwrap();
        // Should detect the stale task
        assert!(scan.new_loops.iter().any(|l| l.kind == AgendaKind::AbandonedTask));
    }
}
