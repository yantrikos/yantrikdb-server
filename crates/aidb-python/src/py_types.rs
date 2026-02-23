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
    dict.set_item("edges", s.edges)?;
    dict.set_item("entities", s.entities)?;
    dict.set_item("operations", s.operations)?;
    dict.set_item("open_conflicts", s.open_conflicts)?;
    dict.set_item("resolved_conflicts", s.resolved_conflicts)?;
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
