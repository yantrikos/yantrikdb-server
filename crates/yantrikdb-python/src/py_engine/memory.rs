use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_types::*;

use super::{map_err, PyYantrikDB};

#[pymethods]
impl PyYantrikDB {
    #[pyo3(signature = (text, memory_type="episodic", importance=0.5, valence=0.0, half_life=604800.0, metadata=None, embedding=None, namespace="default", certainty=0.8, domain="general", source="user", emotional_state=None))]
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
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
    ) -> PyResult<String> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;

        let emb = match embedding {
            Some(e) => e,
            None => self.embed_text(py, text)?,
        };

        let meta = match metadata {
            Some(d) => py_to_json(&d.as_any())?,
            None => serde_json::json!({}),
        };

        db.record(
            text,
            memory_type,
            importance,
            valence,
            half_life,
            &meta,
            &emb,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
        )
        .map_err(map_err)
    }

    #[pyo3(signature = (query=None, query_embedding=None, top_k=10, time_window=None, memory_type=None, include_consolidated=false, expand_entities=true, skip_reinforce=false, namespace=None, domain=None, source=None))]
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
        domain: Option<&str>,
        source: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;

        let emb = match query_embedding {
            Some(e) => e,
            None => match query {
                Some(q) => self.embed_text(py, q)?,
                None => {
                    return Err(PyValueError::new_err(
                        "Must provide either query or query_embedding",
                    ))
                }
            },
        };

        let results = db
            .recall(
                &emb,
                top_k,
                time_window,
                memory_type,
                include_consolidated,
                expand_entities,
                query,
                skip_reinforce,
                namespace,
                domain,
                source,
            )
            .map_err(map_err)?;

        results
            .iter()
            .map(|r| recall_result_to_dict(py, r))
            .collect()
    }

    /// Recall with response including confidence scoring and refinement hints.
    #[pyo3(signature = (query=None, query_embedding=None, top_k=10, time_window=None, memory_type=None, include_consolidated=false, expand_entities=true, skip_reinforce=false, namespace=None, domain=None, source=None))]
    fn recall_with_response(
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
        domain: Option<&str>,
        source: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;

        let emb = match query_embedding {
            Some(e) => e,
            None => match query {
                Some(q) => self.embed_text(py, q)?,
                None => {
                    return Err(PyValueError::new_err(
                        "Must provide either query or query_embedding",
                    ))
                }
            },
        };

        let response = db
            .recall_with_response(
                &emb,
                top_k,
                time_window,
                memory_type,
                include_consolidated,
                expand_entities,
                query,
                skip_reinforce,
                namespace,
                domain,
                source,
            )
            .map_err(map_err)?;

        recall_response_to_dict(py, &response)
    }

    /// Refine a previous recall using a follow-up query.
    #[pyo3(signature = (original_query_embedding, refinement_text=None, refinement_embedding=None, original_rids=vec![], top_k=10, namespace=None, domain=None, source=None))]
    fn recall_refine(
        &self,
        py: Python<'_>,
        original_query_embedding: Vec<f32>,
        refinement_text: Option<&str>,
        refinement_embedding: Option<Vec<f32>>,
        original_rids: Vec<String>,
        top_k: usize,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;

        let ref_emb = match refinement_embedding {
            Some(e) => e,
            None => match refinement_text {
                Some(t) => self.embed_text(py, t)?,
                None => {
                    return Err(PyValueError::new_err(
                        "Must provide either refinement_text or refinement_embedding",
                    ))
                }
            },
        };

        let response = db
            .recall_refine(
                &original_query_embedding,
                &ref_emb,
                &original_rids,
                top_k,
                namespace,
                domain,
                source,
            )
            .map_err(map_err)?;

        recall_response_to_dict(py, &response)
    }

    /// Query builder API: composable recall with keyword arguments.
    ///
    /// ```python
    /// results = db.query(
    ///     embedding=emb,
    ///     top_k=10,
    ///     memory_type="episodic",
    ///     namespace="work",
    /// )
    /// ```
    #[pyo3(signature = (
        query=None, embedding=None, top_k=10, memory_type=None, namespace=None,
        time_window=None, expand_entities=false, include_consolidated=false,
        skip_reinforce=false, domain=None, source=None
    ))]
    fn query(
        &self,
        py: Python<'_>,
        query: Option<&str>,
        embedding: Option<Vec<f32>>,
        top_k: usize,
        memory_type: Option<&str>,
        namespace: Option<&str>,
        time_window: Option<(f64, f64)>,
        expand_entities: bool,
        include_consolidated: bool,
        skip_reinforce: bool,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;

        let emb = match embedding {
            Some(e) => e,
            None => match query {
                Some(q) => self.embed_text(py, q)?,
                None => {
                    return Err(PyValueError::new_err(
                        "Must provide either query or embedding",
                    ))
                }
            },
        };

        let mut q = yantrikdb_core::RecallQuery::new(emb).top_k(top_k);
        if let Some(mt) = memory_type {
            q = q.memory_type(mt);
        }
        if let Some(ns) = namespace {
            q = q.namespace(ns);
        }
        if let Some(tw) = time_window {
            q = q.time_window(tw.0, tw.1);
        }
        if expand_entities {
            q = q.expand_entities(query.unwrap_or(""));
        }
        if include_consolidated {
            q = q.include_consolidated();
        }
        if skip_reinforce {
            q = q.skip_reinforce();
        }
        if let Some(d) = domain {
            q = q.domain(d);
        }
        if let Some(s) = source {
            q = q.source(s);
        }

        let results = db.query(q).map_err(map_err)?;
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

    #[pyo3(signature = (limit=50, offset=0, domain=None, memory_type=None, namespace=None, sort_by="created_at"))]
    fn list_memories(
        &self,
        py: Python<'_>,
        limit: usize,
        offset: usize,
        domain: Option<&str>,
        memory_type: Option<&str>,
        namespace: Option<&str>,
        sort_by: &str,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let (memories, total) = db
            .list_memories(limit, offset, domain, memory_type, namespace, sort_by)
            .map_err(map_err)?;
        let dict = pyo3::types::PyDict::new(py);
        let items: Vec<PyObject> = memories
            .iter()
            .map(|m| memory_to_dict(py, m))
            .collect::<PyResult<Vec<_>>>()?;
        dict.set_item("memories", items)?;
        dict.set_item("total", total)?;
        dict.set_item("offset", offset)?;
        Ok(dict.into())
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
            None => self.embed_text(py, new_text)?,
        };
        let result = db
            .correct(
                rid,
                new_text,
                new_importance,
                new_valence,
                &emb,
                correction_note,
            )
            .map_err(map_err)?;
        let dict = PyDict::new(py);
        dict.set_item("original_rid", &result.original_rid)?;
        dict.set_item("corrected_rid", &result.corrected_rid)?;
        dict.set_item("original_tombstoned", result.original_tombstoned)?;
        Ok(dict.into())
    }

    fn record_batch(
        &self,
        py: Python<'_>,
        inputs: Vec<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<String>> {
        let db = self.get_inner()?;

        let mut record_inputs = Vec::with_capacity(inputs.len());
        for d in &inputs {
            let text: String = d
                .get_item("text")?
                .ok_or_else(|| PyValueError::new_err("Each input must have a 'text' key"))?
                .extract()?;

            let memory_type: String = d
                .get_item("memory_type")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "episodic".to_string());

            let importance: f64 = d
                .get_item("importance")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(0.5);

            let valence: f64 = d
                .get_item("valence")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(0.0);

            let half_life: f64 = d
                .get_item("half_life")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(604800.0);

            let metadata = d
                .get_item("metadata")?
                .map(|v| py_to_json(&v))
                .transpose()?
                .unwrap_or(serde_json::json!({}));

            let embedding: Vec<f32> = match d.get_item("embedding")? {
                Some(v) => v.extract()?,
                None => self.embed_text(py, &text)?,
            };

            let namespace: String = d
                .get_item("namespace")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "default".to_string());

            let certainty: f64 = d
                .get_item("certainty")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(0.8);

            let domain: String = d
                .get_item("domain")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "general".to_string());

            let source: String = d
                .get_item("source")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "user".to_string());

            let emotional_state: Option<String> = d
                .get_item("emotional_state")?
                .map(|v| v.extract::<Option<String>>())
                .transpose()?
                .flatten();

            record_inputs.push(yantrikdb_core::RecordInput {
                text,
                memory_type,
                importance,
                valence,
                half_life,
                metadata,
                embedding,
                namespace,
                certainty,
                domain,
                source,
                emotional_state,
            });
        }

        db.record_batch(&record_inputs).map_err(map_err)
    }

    /// Record feedback on a recall result for adaptive learning.
    #[pyo3(signature = (rid, feedback, query_text=None, query_embedding=None, score_at_retrieval=None, rank_at_retrieval=None))]
    fn recall_feedback(
        &self,
        rid: &str,
        feedback: &str,
        query_text: Option<&str>,
        query_embedding: Option<Vec<f32>>,
        score_at_retrieval: Option<f64>,
        rank_at_retrieval: Option<i32>,
    ) -> PyResult<()> {
        let db = self.get_inner()?;
        db.recall_feedback(
            query_text,
            query_embedding.as_deref(),
            rid,
            feedback,
            score_at_retrieval,
            rank_at_retrieval,
        )
        .map_err(map_err)
    }

    /// Get the current learned scoring weights.
    fn learned_weights(&self, py: Python<'_>) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let w = db.load_learned_weights().map_err(map_err)?;
        let dict = PyDict::new(py);
        dict.set_item("w_sim", w.w_sim)?;
        dict.set_item("w_decay", w.w_decay)?;
        dict.set_item("w_recency", w.w_recency)?;
        dict.set_item("gate_tau", w.gate_tau)?;
        dict.set_item("alpha_imp", w.alpha_imp)?;
        dict.set_item("keyword_boost", w.keyword_boost)?;
        dict.set_item("generation", w.generation)?;
        Ok(dict.into())
    }

    /// Embed text using the configured embedder. Returns a list of floats.
    ///
    /// ```python
    /// embedding = db.embed("some text")
    /// ```
    fn embed(&self, py: Python<'_>, text: &str) -> PyResult<Vec<f32>> {
        self.embed_text(py, text)
    }
}
