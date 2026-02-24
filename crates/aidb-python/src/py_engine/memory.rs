use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_types::*;

use super::{map_err, PyAIDB};

#[pymethods]
impl PyAIDB {
    #[pyo3(signature = (text, memory_type="episodic", importance=0.5, valence=0.0, half_life=604800.0, metadata=None, embedding=None, namespace="default"))]
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
        namespace: &str,
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

        db.record(text, memory_type, importance, valence, half_life, &meta, &emb, namespace)
            .map_err(map_err)
    }

    #[pyo3(signature = (query=None, query_embedding=None, top_k=10, time_window=None, memory_type=None, include_consolidated=false, expand_entities=true, skip_reinforce=false, namespace=None))]
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
        skip_reinforce: bool,
        namespace: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
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
            .recall(&emb, top_k, time_window, memory_type, include_consolidated, expand_entities, query, skip_reinforce, namespace)
            .map_err(map_err)?;

        results
            .iter()
            .map(|r| recall_result_to_dict(py, r))
            .collect()
    }

    fn get(&self, py: Python<'_>, rid: &str) -> PyResult<Option<PyObject>> {
        let db = self.get_inner()?;
        match db.get(rid).map_err(map_err)? {
            Some(mem) => Ok(Some(memory_to_dict(py, &mem)?)),
            None => Ok(None),
        }
    }

    fn forget(&self, rid: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.forget(rid).map_err(map_err)
    }

    #[pyo3(signature = (threshold=0.01))]
    fn decay(&self, py: Python<'_>, threshold: f64) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let decayed = db.decay(threshold).map_err(map_err)?;
        decayed.iter().map(|d| decayed_to_dict(py, d)).collect()
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
        let db = self.get_inner()?;
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

    fn record_batch(&self, py: Python<'_>, inputs: Vec<Bound<'_, PyDict>>) -> PyResult<Vec<String>> {
        let db = self.get_inner()?;

        let mut record_inputs = Vec::with_capacity(inputs.len());
        for d in &inputs {
            let text: String = d.get_item("text")?.ok_or_else(|| {
                PyValueError::new_err("Each input must have a 'text' key")
            })?.extract()?;

            let memory_type: String = d.get_item("memory_type")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "episodic".to_string());

            let importance: f64 = d.get_item("importance")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(0.5);

            let valence: f64 = d.get_item("valence")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(0.0);

            let half_life: f64 = d.get_item("half_life")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(604800.0);

            let metadata = d.get_item("metadata")?
                .map(|v| py_to_json(&v))
                .transpose()?
                .unwrap_or(serde_json::json!({}));

            let embedding: Vec<f32> = match d.get_item("embedding")? {
                Some(v) => v.extract()?,
                None => self.embed(py, &text)?,
            };

            let namespace: String = d.get_item("namespace")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "default".to_string());

            record_inputs.push(aidb_core::RecordInput {
                text,
                memory_type,
                importance,
                valence,
                half_life,
                metadata,
                embedding,
                namespace,
            });
        }

        db.record_batch(&record_inputs).map_err(map_err)
    }
}
