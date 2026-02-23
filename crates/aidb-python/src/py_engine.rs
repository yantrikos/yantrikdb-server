use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use aidb_core::AIDB;

use crate::py_types::*;

/// Python wrapper for the AIDB engine.
#[pyclass(name = "AIDB", unsendable)]
pub struct PyAIDB {
    inner: Option<AIDB>,
    embedder: Option<PyObject>,
}

/// A thin proxy exposing execute() and commit() on the underlying connection.
/// This is needed because Python tests access db._conn.execute(...) directly.
#[pyclass]
pub struct ConnectionProxy {
    /// We store a raw pointer to the AIDB so we can call conn() on it.
    /// Safety: this proxy is only valid while the PyAIDB is alive.
    /// PyO3 prevents the user from dropping PyAIDB while ConnectionProxy exists
    /// because ConnectionProxy holds a Py<PyAIDB> reference.
    parent: Py<PyAIDB>,
}

/// A cursor-like object returned by ConnectionProxy.execute().
/// Holds the result rows so fetchall()/fetchone() can return them.
#[pyclass(unsendable)]
pub struct CursorProxy {
    rows: Vec<PyObject>,
    rowcount: usize,
}

#[pymethods]
impl CursorProxy {
    fn fetchall(&self, py: Python<'_>) -> Vec<PyObject> {
        self.rows.iter().map(|r| r.clone_ref(py)).collect()
    }

    fn fetchone(&self, py: Python<'_>) -> Option<PyObject> {
        self.rows.first().map(|r| r.clone_ref(py))
    }

    #[getter]
    fn rowcount(&self) -> usize {
        self.rowcount
    }
}

#[pymethods]
impl ConnectionProxy {
    #[pyo3(signature = (sql, params=None))]
    fn execute(&self, py: Python<'_>, sql: &str, params: Option<&Bound<'_, PyTuple>>) -> PyResult<CursorProxy> {
        let parent_ref = self.parent.borrow(py);
        let db = parent_ref.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        let conn = db.conn();

        let is_select = sql.trim_start().to_uppercase().starts_with("SELECT");

        let param_values: Vec<Box<dyn rusqlite::types::ToSql>> = if let Some(p) = params {
            p.iter().map(|item| py_to_sql_value(&item)).collect::<PyResult<_>>()?
        } else {
            vec![]
        };
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        if is_select {
            let mut stmt = conn.prepare(sql)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let col_count = stmt.column_count();
            let col_names: Vec<String> = (0..col_count)
                .map(|i| stmt.column_name(i).unwrap_or("").to_string())
                .collect();

            let rows_result = stmt.query_map(params_ref.as_slice(), |row| {
                let mut values: Vec<rusqlite::types::Value> = Vec::new();
                for i in 0..col_count {
                    values.push(row.get::<_, rusqlite::types::Value>(i)?);
                }
                Ok(values)
            }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let mut py_rows: Vec<PyObject> = Vec::new();
            for row_result in rows_result {
                let values = row_result.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let dict = PyDict::new(py);
                for (i, val) in values.iter().enumerate() {
                    let py_val = sqlite_value_to_py(py, val)?;
                    dict.set_item(&col_names[i], py_val)?;
                }
                // Also support index access by wrapping in a RowProxy
                py_rows.push(dict.into());
            }

            Ok(CursorProxy { rows: py_rows, rowcount: 0 })
        } else {
            // Non-SELECT: execute and return rowcount
            let changes = conn.execute(sql, params_ref.as_slice())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(CursorProxy { rows: vec![], rowcount: changes })
        }
    }

    fn executescript(&self, py: Python<'_>, sql: &str) -> PyResult<()> {
        let parent_ref = self.parent.borrow(py);
        let db = parent_ref.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        db.conn().execute_batch(sql)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    fn commit(&self, py: Python<'_>) -> PyResult<()> {
        let parent_ref = self.parent.borrow(py);
        let _db = parent_ref.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        // In rusqlite auto-commit mode, commit is a no-op
        Ok(())
    }
}

fn sqlite_value_to_py(py: Python<'_>, val: &rusqlite::types::Value) -> PyResult<PyObject> {
    match val {
        rusqlite::types::Value::Null => Ok(py.None()),
        rusqlite::types::Value::Integer(i) => Ok((*i).into_pyobject(py)?.to_owned().into_any().unbind()),
        rusqlite::types::Value::Real(f) => Ok((*f).into_pyobject(py)?.to_owned().into_any().unbind()),
        rusqlite::types::Value::Text(s) => Ok(s.as_str().into_pyobject(py)?.to_owned().into_any().unbind()),
        rusqlite::types::Value::Blob(b) => Ok(b.as_slice().into_pyobject(py)?.to_owned().into_any().unbind()),
    }
}

fn py_to_sql_value(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn rusqlite::types::ToSql>> {
    if obj.is_none() {
        return Ok(Box::new(rusqlite::types::Null));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Box::new(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Box::new(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Box::new(s));
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Box::new(b));
    }
    Err(PyRuntimeError::new_err(format!(
        "Unsupported SQL parameter type: {}",
        obj.get_type().name()?
    )))
}

fn map_err(e: aidb_core::AidbError) -> PyErr {
    match e {
        aidb_core::AidbError::NoEmbedder => {
            PyRuntimeError::new_err(e.to_string())
        }
        aidb_core::AidbError::NoQuery => {
            PyValueError::new_err(e.to_string())
        }
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

#[pymethods]
impl PyAIDB {
    #[new]
    #[pyo3(signature = (db_path=":memory:", embedding_dim=384, embedder=None))]
    fn new(
        db_path: &str,
        embedding_dim: usize,
        embedder: Option<PyObject>,
    ) -> PyResult<Self> {
        let inner = AIDB::new(db_path, embedding_dim).map_err(map_err)?;
        Ok(Self {
            inner: Some(inner),
            embedder,
        })
    }

    fn set_embedder(&mut self, embedder: PyObject) {
        self.embedder = Some(embedder);
    }

    /// The _conn property — returns a ConnectionProxy for test compatibility.
    #[getter]
    fn _conn(slf: Py<Self>, _py: Python<'_>) -> PyResult<ConnectionProxy> {
        Ok(ConnectionProxy { parent: slf })
    }

    #[pyo3(signature = (text, memory_type="episodic", importance=0.5, valence=0.0, half_life=604800.0, metadata=None, embedding=None))]
    fn record(
        &self,
        py: Python<'_>,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: Option<&Bound<'_, PyDict>>,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<String> {
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;

        let emb = match embedding {
            Some(e) => e,
            None => self.embed(py, text)?,
        };

        let meta = match metadata {
            Some(d) => py_to_json(&d.as_any())?,
            None => serde_json::json!({}),
        };

        db.record(text, memory_type, importance, valence, half_life, &meta, &emb)
            .map_err(map_err)
    }

    #[pyo3(signature = (query=None, query_embedding=None, top_k=10, time_window=None, memory_type=None, include_consolidated=false, expand_entities=true))]
    fn recall(
        &self,
        py: Python<'_>,
        query: Option<&str>,
        query_embedding: Option<Vec<f32>>,
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
    ) -> PyResult<Vec<PyObject>> {
        let _ = expand_entities; // Not used yet
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;

        let emb = match query_embedding {
            Some(e) => e,
            None => match query {
                Some(q) => self.embed(py, q)?,
                None => return Err(PyValueError::new_err("Must provide either query or query_embedding")),
            },
        };

        let results = db
            .recall(&emb, top_k, time_window, memory_type, include_consolidated)
            .map_err(map_err)?;

        results
            .iter()
            .map(|r| recall_result_to_dict(py, r))
            .collect()
    }

    #[pyo3(signature = (src, dst, rel_type="related_to", weight=1.0))]
    fn relate(
        &self,
        src: &str,
        dst: &str,
        rel_type: &str,
        weight: f64,
    ) -> PyResult<String> {
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        db.relate(src, dst, rel_type, weight).map_err(map_err)
    }

    #[pyo3(signature = (threshold=0.01))]
    fn decay(&self, py: Python<'_>, threshold: f64) -> PyResult<Vec<PyObject>> {
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        let decayed = db.decay(threshold).map_err(map_err)?;
        decayed.iter().map(|d| decayed_to_dict(py, d)).collect()
    }

    fn forget(&self, rid: &str) -> PyResult<bool> {
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        db.forget(rid).map_err(map_err)
    }

    fn get(&self, py: Python<'_>, rid: &str) -> PyResult<Option<PyObject>> {
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        match db.get(rid).map_err(map_err)? {
            Some(mem) => Ok(Some(memory_to_dict(py, &mem)?)),
            None => Ok(None),
        }
    }

    fn get_edges(&self, py: Python<'_>, entity: &str) -> PyResult<Vec<PyObject>> {
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        let edges = db.get_edges(entity).map_err(map_err)?;
        edges.iter().map(|e| edge_to_dict(py, e)).collect()
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        let s = db.stats().map_err(map_err)?;
        stats_to_dict(py, &s)
    }

    /// The actor_id of this AIDB instance (read-only).
    #[getter]
    fn actor_id(&self) -> PyResult<String> {
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        Ok(db.actor_id().to_string())
    }

    /// Exposed for Python consolidate.py compatibility.
    #[pyo3(signature = (op_type, target_rid, payload))]
    fn _log_op(
        &self,
        op_type: &str,
        target_rid: Option<&str>,
        payload: &Bound<'_, PyDict>,
    ) -> PyResult<String> {
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        let payload_json = py_to_json(&payload.as_any())?;
        db.log_op(op_type, target_rid, &payload_json, None).map_err(map_err)
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
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        let conflicts = db
            .get_conflicts(status, conflict_type, entity, priority, limit)
            .map_err(map_err)?;
        conflicts
            .iter()
            .map(|c| conflict_to_dict(py, c))
            .collect()
    }

    fn get_conflict(&self, py: Python<'_>, conflict_id: &str) -> PyResult<Option<PyObject>> {
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
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
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
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
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        db.dismiss_conflict(conflict_id, note).map_err(map_err)
    }

    fn scan_conflicts(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        let conflicts = aidb_core::scan_conflicts(db).map_err(map_err)?;
        conflicts
            .iter()
            .map(|c| conflict_to_dict(py, c))
            .collect()
    }

    #[pyo3(signature = (rid, new_text, new_importance=None, new_valence=None, embedding=None, correction_note=None))]
    fn correct(
        &self,
        py: Python<'_>,
        rid: &str,
        new_text: &str,
        new_importance: Option<f64>,
        new_valence: Option<f64>,
        embedding: Option<Vec<f32>>,
        correction_note: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })?;
        let emb = match embedding {
            Some(e) => e,
            None => self.embed(py, new_text)?,
        };
        let result = db
            .correct(rid, new_text, new_importance, new_valence, &emb, correction_note)
            .map_err(map_err)?;
        let dict = PyDict::new(py);
        dict.set_item("original_rid", &result.original_rid)?;
        dict.set_item("corrected_rid", &result.corrected_rid)?;
        dict.set_item("original_tombstoned", result.original_tombstoned)?;
        Ok(dict.into())
    }

    fn close(&mut self) -> PyResult<()> {
        if let Some(db) = self.inner.take() {
            db.close().map_err(map_err)?;
        }
        Ok(())
    }
}

impl PyAIDB {
    /// Get a reference to the inner AIDB engine (for use by consolidation/trigger wrappers).
    pub fn get_inner(&self) -> PyResult<&AIDB> {
        self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("AIDB is closed")
        })
    }

    fn embed(&self, py: Python<'_>, text: &str) -> PyResult<Vec<f32>> {
        match &self.embedder {
            Some(emb) => {
                let result = emb.call_method1(py, "encode", (text,))?;
                // Handle both list and numpy array returns
                if let Ok(list) = result.extract::<Vec<f32>>(py) {
                    Ok(list)
                } else {
                    // Try calling .tolist() for numpy arrays
                    let list = result.call_method0(py, "tolist")?;
                    list.extract::<Vec<f32>>(py)
                }
            }
            None => Err(PyRuntimeError::new_err(
                "No embedder configured. Pass an embedder to AIDB() or call set_embedder().",
            )),
        }
    }
}
