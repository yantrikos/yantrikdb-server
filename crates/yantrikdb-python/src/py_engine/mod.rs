mod cognition;
mod graph;
mod memory;
mod session_temporal;
mod sync;

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use yantrikdb_core::YantrikDB;

use crate::py_types::*;

/// Python wrapper for the YantrikDB engine.
#[pyclass(name = "YantrikDB")]
pub struct PyYantrikDB {
    pub(crate) inner: Option<Arc<YantrikDB>>,
    pub(crate) embedder: Option<PyObject>,
}

/// A thin proxy exposing execute() and commit() on the underlying connection.
/// This is needed because Python tests access db._conn.execute(...) directly.
/// Stores an Arc<YantrikDB> and acquires the connection lock on each call.
#[pyclass]
pub struct ConnectionProxy {
    db: Arc<YantrikDB>,
}

/// A cursor-like object returned by ConnectionProxy.execute().
/// Holds the result rows so fetchall()/fetchone() can return them.
#[pyclass]
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
    fn execute(
        &self,
        py: Python<'_>,
        sql: &str,
        params: Option<&Bound<'_, PyTuple>>,
    ) -> PyResult<CursorProxy> {
        let conn = self.db.conn();

        let is_select = sql.trim_start().to_uppercase().starts_with("SELECT");

        let param_values: Vec<Box<dyn rusqlite::types::ToSql>> = if let Some(p) = params {
            p.iter()
                .map(|item| py_to_sql_value(&item))
                .collect::<PyResult<_>>()?
        } else {
            vec![]
        };
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        if is_select {
            let mut stmt = conn
                .prepare(sql)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let col_count = stmt.column_count();
            let col_names: Vec<String> = (0..col_count)
                .map(|i| stmt.column_name(i).unwrap_or("").to_string())
                .collect();

            let rows_result = stmt
                .query_map(params_ref.as_slice(), |row| {
                    let mut values: Vec<rusqlite::types::Value> = Vec::new();
                    for i in 0..col_count {
                        values.push(row.get::<_, rusqlite::types::Value>(i)?);
                    }
                    Ok(values)
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let mut py_rows: Vec<PyObject> = Vec::new();
            for row_result in rows_result {
                let values = row_result.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let dict = PyDict::new(py);
                for (i, val) in values.iter().enumerate() {
                    let py_val = sqlite_value_to_py(py, val)?;
                    dict.set_item(&col_names[i], py_val)?;
                }
                py_rows.push(dict.into());
            }

            Ok(CursorProxy {
                rows: py_rows,
                rowcount: 0,
            })
        } else {
            let changes = conn
                .execute(sql, params_ref.as_slice())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(CursorProxy {
                rows: vec![],
                rowcount: changes,
            })
        }
    }

    fn executescript(&self, _py: Python<'_>, sql: &str) -> PyResult<()> {
        self.db
            .conn()
            .execute_batch(sql)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    fn commit(&self, _py: Python<'_>) -> PyResult<()> {
        // In rusqlite auto-commit mode, commit is a no-op
        Ok(())
    }
}

fn sqlite_value_to_py(py: Python<'_>, val: &rusqlite::types::Value) -> PyResult<PyObject> {
    match val {
        rusqlite::types::Value::Null => Ok(py.None()),
        rusqlite::types::Value::Integer(i) => {
            Ok((*i).into_pyobject(py)?.to_owned().into_any().unbind())
        }
        rusqlite::types::Value::Real(f) => {
            Ok((*f).into_pyobject(py)?.to_owned().into_any().unbind())
        }
        rusqlite::types::Value::Text(s) => {
            Ok(s.as_str().into_pyobject(py)?.to_owned().into_any().unbind())
        }
        rusqlite::types::Value::Blob(b) => Ok(b
            .as_slice()
            .into_pyobject(py)?
            .to_owned()
            .into_any()
            .unbind()),
    }
}

pub(crate) fn py_to_sql_value(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn rusqlite::types::ToSql>> {
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

pub(crate) fn map_err(e: yantrikdb_core::YantrikDbError) -> PyErr {
    match e {
        yantrikdb_core::YantrikDbError::NoEmbedder => PyRuntimeError::new_err(e.to_string()),
        yantrikdb_core::YantrikDbError::NoQuery => PyValueError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

#[pymethods]
impl PyYantrikDB {
    #[new]
    #[pyo3(signature = (db_path=":memory:", embedding_dim=384, embedder=None, encryption_key=None, model_dir=None))]
    fn new(
        db_path: &str,
        embedding_dim: usize,
        embedder: Option<PyObject>,
        encryption_key: Option<Vec<u8>>,
        model_dir: Option<&str>,
    ) -> PyResult<Self> {
        #[allow(unused_mut)]
        let mut inner = if let Some(key_bytes) = encryption_key {
            if key_bytes.len() != 32 {
                return Err(PyValueError::new_err(
                    "encryption_key must be exactly 32 bytes",
                ));
            }
            let mut key = [0u8; 32];
            key.copy_from_slice(&key_bytes);
            YantrikDB::new_encrypted(db_path, embedding_dim, &key).map_err(map_err)?
        } else {
            YantrikDB::new(db_path, embedding_dim).map_err(map_err)?
        };

        // If model_dir provided and candle feature enabled, use CandleEmbedder
        #[cfg(feature = "candle")]
        if let Some(dir) = model_dir {
            let candle_embedder = yantrik_ml::CandleEmbedder::from_dir(std::path::Path::new(dir))
                .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to load candle embedder: {e}"))
            })?;
            inner.set_embedder(Box::new(candle_embedder));
        }

        #[cfg(not(feature = "candle"))]
        if model_dir.is_some() {
            return Err(PyRuntimeError::new_err(
                "model_dir requires the 'candle' feature. Build with: maturin develop --features candle",
            ));
        }

        Ok(Self {
            inner: Some(Arc::new(inner)),
            embedder,
        })
    }

    /// Whether this instance has encryption enabled.
    #[getter]
    fn is_encrypted(&self) -> PyResult<bool> {
        let db = self.get_inner()?;
        Ok(db.is_encrypted())
    }

    fn set_embedder(&mut self, embedder: PyObject) -> PyResult<()> {
        self.embedder = Some(embedder);
        Ok(())
    }

    /// The _conn property — returns a ConnectionProxy for test compatibility.
    #[getter]
    fn _conn(&self) -> PyResult<ConnectionProxy> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        Ok(ConnectionProxy { db: Arc::clone(db) })
    }

    /// The actor_id of this YantrikDB instance (read-only).
    #[getter]
    fn actor_id(&self) -> PyResult<String> {
        let db = self.get_inner()?;
        Ok(db.actor_id().to_string())
    }

    #[pyo3(signature = (namespace=None))]
    fn stats(&self, py: Python<'_>, namespace: Option<&str>) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let s = db.stats(namespace).map_err(map_err)?;
        stats_to_dict(py, &s)
    }

    /// Exposed for Python consolidate.py compatibility.
    #[pyo3(signature = (op_type, target_rid, payload))]
    fn _log_op(
        &self,
        op_type: &str,
        target_rid: Option<&str>,
        payload: &Bound<'_, PyDict>,
    ) -> PyResult<String> {
        let db = self.get_inner()?;
        let payload_json = py_to_json(&payload.as_any())?;
        db.log_op(op_type, target_rid, &payload_json, None)
            .map_err(map_err)
    }

    fn close(&mut self) -> PyResult<()> {
        if let Some(arc) = self.inner.take() {
            // If we hold the only reference, unwrap and close explicitly.
            // Otherwise just drop our reference (closes on last drop).
            match Arc::try_unwrap(arc) {
                Ok(db) => db.close().map_err(map_err)?,
                Err(_arc) => {
                    // Other references still exist; dropping our ref is fine.
                }
            }
        }
        Ok(())
    }
}

impl PyYantrikDB {
    /// Get a reference to the inner YantrikDB engine (for use by consolidation/trigger wrappers).
    pub fn get_inner(&self) -> PyResult<&YantrikDB> {
        self.inner
            .as_deref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))
    }

    pub(crate) fn embed_text(&self, py: Python<'_>, text: &str) -> PyResult<Vec<f32>> {
        // Try Rust-native embedder first (candle or any Embedder impl)
        if let Some(db) = &self.inner {
            if db.has_embedder() {
                return db.embed(text).map_err(map_err);
            }
        }

        // Fall back to Python embedder
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
                "No embedder configured. Pass an embedder to YantrikDB() or call set_embedder().",
            )),
        }
    }
}
