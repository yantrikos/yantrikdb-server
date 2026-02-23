use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Convert an aidb-core Memory to a Python dict matching the Python engine's output exactly.
pub fn memory_to_dict(py: Python<'_>, mem: &aidb_core::Memory) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("rid", &mem.rid)?;
    dict.set_item("type", &mem.memory_type)?;
    dict.set_item("text", &mem.text)?;
    dict.set_item("created_at", mem.created_at)?;
    dict.set_item("importance", mem.importance)?;
    dict.set_item("valence", mem.valence)?;
    dict.set_item("half_life", mem.half_life)?;
    dict.set_item("last_access", mem.last_access)?;
    dict.set_item("consolidation_status", &mem.consolidation_status)?;
    dict.set_item("storage_tier", &mem.storage_tier)?;
    dict.set_item("consolidated_into", &mem.consolidated_into)?;
    dict.set_item("metadata", json_to_py(py, &mem.metadata)?)?;
    Ok(dict.into())
}

/// Convert an aidb-core RecallResult to a Python dict.
pub fn recall_result_to_dict(py: Python<'_>, r: &aidb_core::RecallResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("rid", &r.rid)?;
    dict.set_item("type", &r.memory_type)?;
    dict.set_item("text", &r.text)?;
    dict.set_item("created_at", r.created_at)?;
    dict.set_item("importance", r.importance)?;
    dict.set_item("valence", r.valence)?;
    dict.set_item("score", r.score)?;

    let scores = PyDict::new(py);
    scores.set_item("similarity", r.scores.similarity)?;
    scores.set_item("decay", r.scores.decay)?;
    scores.set_item("recency", r.scores.recency)?;
    scores.set_item("importance", r.scores.importance)?;
    scores.set_item("graph_proximity", r.scores.graph_proximity)?;
    dict.set_item("scores", scores)?;

    let why: Vec<&str> = r.why_retrieved.iter().map(|s| s.as_str()).collect();
    dict.set_item("why_retrieved", why)?;

    dict.set_item("metadata", json_to_py(py, &r.metadata)?)?;

    Ok(dict.into())
}

/// Convert an aidb-core Edge to a Python dict.
pub fn edge_to_dict(py: Python<'_>, e: &aidb_core::Edge) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("edge_id", &e.edge_id)?;
    dict.set_item("src", &e.src)?;
    dict.set_item("dst", &e.dst)?;
    dict.set_item("rel_type", &e.rel_type)?;
    dict.set_item("weight", e.weight)?;
    Ok(dict.into())
}

/// Convert an aidb-core DecayedMemory to a Python dict.
pub fn decayed_to_dict(py: Python<'_>, d: &aidb_core::DecayedMemory) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("rid", &d.rid)?;
    dict.set_item("text", &d.text)?;
    dict.set_item("type", &d.memory_type)?;
    dict.set_item("original_importance", d.original_importance)?;
    dict.set_item("current_score", d.current_score)?;
    dict.set_item("days_since_access", d.days_since_access)?;
    Ok(dict.into())
}

/// Convert aidb-core Stats to a Python dict.
pub fn stats_to_dict(py: Python<'_>, s: &aidb_core::Stats) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("active_memories", s.active_memories)?;
    dict.set_item("consolidated_memories", s.consolidated_memories)?;
    dict.set_item("tombstoned_memories", s.tombstoned_memories)?;
    dict.set_item("archived_memories", s.archived_memories)?;
    dict.set_item("edges", s.edges)?;
    dict.set_item("entities", s.entities)?;
    dict.set_item("operations", s.operations)?;
    dict.set_item("open_conflicts", s.open_conflicts)?;
    dict.set_item("resolved_conflicts", s.resolved_conflicts)?;
    dict.set_item("pending_triggers", s.pending_triggers)?;
    dict.set_item("active_patterns", s.active_patterns)?;
    dict.set_item("scoring_cache_entries", s.scoring_cache_entries)?;
    dict.set_item("vec_index_entries", s.vec_index_entries)?;
    Ok(dict.into())
}

/// Convert an aidb-core Conflict to a Python dict.
pub fn conflict_to_dict(py: Python<'_>, c: &aidb_core::Conflict) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("conflict_id", &c.conflict_id)?;
    dict.set_item("conflict_type", &c.conflict_type)?;
    dict.set_item("priority", &c.priority)?;
    dict.set_item("status", &c.status)?;
    dict.set_item("memory_a", &c.memory_a)?;
    dict.set_item("memory_b", &c.memory_b)?;
    dict.set_item("entity", &c.entity)?;
    dict.set_item("rel_type", &c.rel_type)?;
    dict.set_item("detected_at", c.detected_at)?;
    dict.set_item("detected_by", &c.detected_by)?;
    dict.set_item("detection_reason", &c.detection_reason)?;
    dict.set_item("resolved_at", c.resolved_at)?;
    dict.set_item("resolved_by", &c.resolved_by)?;
    dict.set_item("strategy", &c.strategy)?;
    dict.set_item("winner_rid", &c.winner_rid)?;
    dict.set_item("resolution_note", &c.resolution_note)?;
    Ok(dict.into())
}

/// Convert an aidb-core ThinkResult to a Python dict.
pub fn think_result_to_dict(py: Python<'_>, r: &aidb_core::ThinkResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    let triggers_list = pyo3::types::PyList::empty(py);
    for t in &r.triggers {
        let td = PyDict::new(py);
        td.set_item("trigger_type", &t.trigger_type)?;
        td.set_item("reason", &t.reason)?;
        td.set_item("urgency", t.urgency)?;
        let rids: Vec<&str> = t.source_rids.iter().map(|s| s.as_str()).collect();
        td.set_item("source_rids", rids)?;
        td.set_item("suggested_action", &t.suggested_action)?;
        let ctx = json_to_py(py, &serde_json::json!(t.context))?;
        td.set_item("context", ctx)?;
        triggers_list.append(td)?;
    }
    dict.set_item("triggers", triggers_list)?;

    dict.set_item("consolidation_count", r.consolidation_count)?;
    dict.set_item("conflicts_found", r.conflicts_found)?;
    dict.set_item("patterns_new", r.patterns_new)?;
    dict.set_item("patterns_updated", r.patterns_updated)?;
    dict.set_item("expired_triggers", r.expired_triggers)?;
    dict.set_item("duration_ms", r.duration_ms)?;
    Ok(dict.into())
}

/// Convert an aidb-core PersistedTrigger to a Python dict.
pub fn persisted_trigger_to_dict(py: Python<'_>, t: &aidb_core::PersistedTrigger) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("trigger_id", &t.trigger_id)?;
    dict.set_item("trigger_type", &t.trigger_type)?;
    dict.set_item("urgency", t.urgency)?;
    dict.set_item("status", &t.status)?;
    dict.set_item("reason", &t.reason)?;
    dict.set_item("suggested_action", &t.suggested_action)?;
    let rids: Vec<&str> = t.source_rids.iter().map(|s| s.as_str()).collect();
    dict.set_item("source_rids", rids)?;
    dict.set_item("context", json_to_py(py, &t.context)?)?;
    dict.set_item("created_at", t.created_at)?;
    dict.set_item("delivered_at", t.delivered_at)?;
    dict.set_item("acknowledged_at", t.acknowledged_at)?;
    dict.set_item("acted_at", t.acted_at)?;
    dict.set_item("expires_at", t.expires_at)?;
    Ok(dict.into())
}

/// Convert an aidb-core Pattern to a Python dict.
pub fn pattern_to_dict(py: Python<'_>, p: &aidb_core::Pattern) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("pattern_id", &p.pattern_id)?;
    dict.set_item("pattern_type", &p.pattern_type)?;
    dict.set_item("status", &p.status)?;
    dict.set_item("confidence", p.confidence)?;
    dict.set_item("description", &p.description)?;
    let evidence: Vec<&str> = p.evidence_rids.iter().map(|s| s.as_str()).collect();
    dict.set_item("evidence_rids", evidence)?;
    let entities: Vec<&str> = p.entity_names.iter().map(|s| s.as_str()).collect();
    dict.set_item("entity_names", entities)?;
    dict.set_item("context", json_to_py(py, &p.context)?)?;
    dict.set_item("first_seen", p.first_seen)?;
    dict.set_item("last_confirmed", p.last_confirmed)?;
    dict.set_item("occurrence_count", p.occurrence_count)?;
    Ok(dict.into())
}

/// Convert serde_json::Value to Python object.
pub fn json_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<PyObject> {
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.to_owned().into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.to_owned().into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.as_str().into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let list = pyo3::types::PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into())
        }
    }
}

/// Convert a Python object to serde_json::Value.
pub fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(serde_json::json!(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::json!(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
        let arr: Vec<serde_json::Value> = list
            .iter()
            .map(|item| py_to_json(&item))
            .collect::<PyResult<_>>()?;
        return Ok(serde_json::Value::Array(arr));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(&v)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    // Fallback: convert to string
    let s = obj.str()?.to_string();
    Ok(serde_json::Value::String(s))
}
