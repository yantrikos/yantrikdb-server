use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_types::*;

use super::{map_err, PyAIDB};

#[pymethods]
impl PyAIDB {
    // ── Cognition loop (V3) ──

    #[pyo3(signature = (config=None))]
    fn think(&self, py: Python<'_>, config: Option<&Bound<'_, PyDict>>) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let cfg = if let Some(d) = config {
            let mut c = aidb_core::ThinkConfig::default();
            if let Ok(Some(v)) = d.get_item("importance_threshold") { c.importance_threshold = v.extract()?; }
            if let Ok(Some(v)) = d.get_item("decay_threshold") { c.decay_threshold = v.extract()?; }
            if let Ok(Some(v)) = d.get_item("max_triggers") { c.max_triggers = v.extract()?; }
            if let Ok(Some(v)) = d.get_item("run_consolidation") { c.run_consolidation = v.extract()?; }
            if let Ok(Some(v)) = d.get_item("run_conflict_scan") { c.run_conflict_scan = v.extract()?; }
            if let Ok(Some(v)) = d.get_item("run_pattern_mining") { c.run_pattern_mining = v.extract()?; }
            if let Ok(Some(v)) = d.get_item("min_active_memories") { c.min_active_memories = v.extract()?; }
            c
        } else {
            aidb_core::ThinkConfig::default()
        };
        let result = db.think(&cfg).map_err(map_err)?;
        think_result_to_dict(py, &result)
    }

    fn deliver_trigger(&self, trigger_id: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.deliver_trigger(trigger_id).map_err(map_err)
    }

    fn acknowledge_trigger(&self, trigger_id: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.acknowledge_trigger(trigger_id).map_err(map_err)
    }

    fn act_on_trigger(&self, trigger_id: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.act_on_trigger(trigger_id).map_err(map_err)
    }

    fn dismiss_trigger(&self, trigger_id: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.dismiss_trigger(trigger_id).map_err(map_err)
    }

    #[pyo3(signature = (limit=10))]
    fn get_pending_triggers(&self, py: Python<'_>, limit: usize) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let triggers = db.get_pending_triggers(limit).map_err(map_err)?;
        triggers.iter().map(|t| persisted_trigger_to_dict(py, t)).collect()
    }

    #[pyo3(signature = (trigger_type=None, limit=50))]
    fn get_trigger_history(
        &self,
        py: Python<'_>,
        trigger_type: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let triggers = db.get_trigger_history(trigger_type, limit).map_err(map_err)?;
        triggers.iter().map(|t| persisted_trigger_to_dict(py, t)).collect()
    }

    #[pyo3(signature = (pattern_type=None, status=None, limit=50))]
    fn get_patterns(
        &self,
        py: Python<'_>,
        pattern_type: Option<&str>,
        status: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let patterns = db.get_patterns(pattern_type, status, limit).map_err(map_err)?;
        patterns.iter().map(|p| pattern_to_dict(py, p)).collect()
    }

    // ── Conflict resolution API (V2) ──

    #[pyo3(signature = (status=None, conflict_type=None, entity=None, priority=None, limit=50))]
    fn get_conflicts(
        &self,
        py: Python<'_>,
        status: Option<&str>,
        conflict_type: Option<&str>,
        entity: Option<&str>,
        priority: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let conflicts = db
            .get_conflicts(status, conflict_type, entity, priority, limit)
            .map_err(map_err)?;
        conflicts
            .iter()
            .map(|c| conflict_to_dict(py, c))
            .collect()
    }

    fn get_conflict(&self, py: Python<'_>, conflict_id: &str) -> PyResult<Option<PyObject>> {
        let db = self.get_inner()?;
        match db.get_conflict(conflict_id).map_err(map_err)? {
            Some(c) => Ok(Some(conflict_to_dict(py, &c)?)),
            None => Ok(None),
        }
    }

    #[pyo3(signature = (conflict_id, strategy, winner_rid=None, new_text=None, resolution_note=None))]
    fn resolve_conflict(
        &self,
        py: Python<'_>,
        conflict_id: &str,
        strategy: &str,
        winner_rid: Option<&str>,
        new_text: Option<&str>,
        resolution_note: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let result = db
            .resolve_conflict(conflict_id, strategy, winner_rid, new_text, resolution_note)
            .map_err(map_err)?;
        let dict = PyDict::new(py);
        dict.set_item("conflict_id", &result.conflict_id)?;
        dict.set_item("strategy", &result.strategy)?;
        dict.set_item("winner_rid", &result.winner_rid)?;
        dict.set_item("loser_tombstoned", result.loser_tombstoned)?;
        dict.set_item("new_memory_rid", &result.new_memory_rid)?;
        Ok(dict.into())
    }

    #[pyo3(signature = (conflict_id, note=None))]
    fn dismiss_conflict(&self, conflict_id: &str, note: Option<&str>) -> PyResult<()> {
        let db = self.get_inner()?;
        db.dismiss_conflict(conflict_id, note).map_err(map_err)
    }

    fn scan_conflicts(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let conflicts = aidb_core::scan_conflicts(db).map_err(map_err)?;
        conflicts
            .iter()
            .map(|c| conflict_to_dict(py, c))
            .collect()
    }
}
