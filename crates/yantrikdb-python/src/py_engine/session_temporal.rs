//! Python bindings for sessions, temporal helpers, relationship depth,
//! entity profile, and procedural memory APIs.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_types::*;

use super::{map_err, PyYantrikDB};

#[pymethods]
impl PyYantrikDB {
    // ── Session API (V13) ──

    #[pyo3(signature = (namespace="default", client_id="default", metadata=None))]
    fn session_start(
        &self,
        namespace: &str,
        client_id: &str,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let db = self.get_inner()?;
        let meta = match metadata {
            Some(d) => py_to_json(&d.as_any())?,
            None => serde_json::json!({}),
        };
        db.session_start(namespace, client_id, &meta)
            .map_err(map_err)
    }

    #[pyo3(signature = (session_id, summary=None))]
    fn session_end(
        &self,
        py: Python<'_>,
        session_id: &str,
        summary: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let result = db.session_end(session_id, summary).map_err(map_err)?;
        session_summary_to_dict(py, &result)
    }

    #[pyo3(signature = (namespace="default", client_id="default"))]
    fn active_session(
        &self,
        py: Python<'_>,
        namespace: &str,
        client_id: &str,
    ) -> PyResult<Option<PyObject>> {
        let db = self.get_inner()?;
        match db.active_session(namespace, client_id).map_err(map_err)? {
            Some(s) => Ok(Some(session_to_dict(py, &s)?)),
            None => Ok(None),
        }
    }

    #[pyo3(signature = (namespace="default", client_id="default", limit=10))]
    fn session_history(
        &self,
        py: Python<'_>,
        namespace: &str,
        client_id: &str,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let sessions = db
            .session_history(namespace, client_id, limit)
            .map_err(map_err)?;
        sessions.iter().map(|s| session_to_dict(py, s)).collect()
    }

    #[pyo3(signature = (max_age_hours=24.0))]
    fn session_abandon_stale(&self, max_age_hours: f64) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.session_abandon_stale(max_age_hours).map_err(map_err)
    }

    // ── Temporal helpers (V13) ──

    #[pyo3(signature = (days=30.0, limit=50, namespace=None))]
    fn stale(
        &self,
        py: Python<'_>,
        days: f64,
        limit: usize,
        namespace: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let memories = db.stale(days, limit, namespace).map_err(map_err)?;
        memories.iter().map(|m| memory_to_dict(py, m)).collect()
    }

    #[pyo3(signature = (days=7.0, limit=50, namespace=None))]
    fn upcoming(
        &self,
        py: Python<'_>,
        days: f64,
        limit: usize,
        namespace: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let memories = db.upcoming(days, limit, namespace).map_err(map_err)?;
        memories.iter().map(|m| memory_to_dict(py, m)).collect()
    }

    #[pyo3(signature = (entity, days=90.0, namespace=None))]
    fn entity_profile(
        &self,
        py: Python<'_>,
        entity: &str,
        days: f64,
        namespace: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let profile = db
            .entity_profile(entity, days, namespace)
            .map_err(map_err)?;
        entity_profile_to_dict(py, &profile)
    }

    // ── Relationship depth (V14) ──

    #[pyo3(signature = (entity, namespace=None))]
    fn relationship_depth(
        &self,
        py: Python<'_>,
        entity: &str,
        namespace: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let depth = db.relationship_depth(entity, namespace).map_err(map_err)?;
        relationship_depth_to_dict(py, &depth)
    }

    // ── Procedural memory (V14) ──

    #[pyo3(signature = (query_embedding, query_text=None, domain=None, top_k=5, namespace=None))]
    fn surface_procedural(
        &self,
        py: Python<'_>,
        query_embedding: Vec<f32>,
        query_text: Option<&str>,
        domain: Option<&str>,
        top_k: usize,
        namespace: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let results = db
            .surface_procedural(&query_embedding, query_text, domain, top_k, namespace)
            .map_err(map_err)?;
        results
            .iter()
            .map(|r| recall_result_to_dict(py, r))
            .collect()
    }

    #[pyo3(signature = (text, embedding=None, domain="general", task_context="", effectiveness=0.5, namespace="default"))]
    fn record_procedural(
        &self,
        py: Python<'_>,
        text: &str,
        embedding: Option<Vec<f32>>,
        domain: &str,
        task_context: &str,
        effectiveness: f64,
        namespace: &str,
    ) -> PyResult<String> {
        let db = self.get_inner()?;
        let emb = match embedding {
            Some(e) => e,
            None => self.embed_text(py, text)?,
        };
        db.record_procedural(text, &emb, domain, task_context, effectiveness, namespace)
            .map_err(map_err)
    }

    #[pyo3(signature = (rid, outcome))]
    fn reinforce_procedural(&self, rid: &str, outcome: f64) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.reinforce_procedural(rid, outcome).map_err(map_err)
    }

    #[pyo3(signature = (namespace=None))]
    fn procedural_stats(&self, py: Python<'_>, namespace: Option<&str>) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let stats = db.procedural_stats(namespace).map_err(map_err)?;
        stats
            .iter()
            .map(|(domain, count, avg_imp)| {
                let dict = PyDict::new(py);
                dict.set_item("domain", domain)?;
                dict.set_item("count", count)?;
                dict.set_item("avg_effectiveness", avg_imp)?;
                Ok(dict.into())
            })
            .collect()
    }

    // ── Substitution category APIs (V14) ──

    #[pyo3(signature = (conflict_id, new_type))]
    fn reclassify_conflict(
        &self,
        py: Python<'_>,
        conflict_id: &str,
        new_type: &str,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let result = db
            .reclassify_conflict(conflict_id, new_type)
            .map_err(map_err)?;
        reclassify_result_to_dict(py, &result)
    }

    fn substitution_categories(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let cats = db.substitution_categories().map_err(map_err)?;
        cats.iter()
            .map(|c| substitution_category_to_dict(py, c))
            .collect()
    }

    #[pyo3(signature = (category_name))]
    fn substitution_members(&self, py: Python<'_>, category_name: &str) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let members = db.substitution_members(category_name).map_err(map_err)?;
        members
            .iter()
            .map(|m| substitution_member_to_dict(py, m))
            .collect()
    }

    #[pyo3(signature = (category_name, members, source="llm_suggested"))]
    fn learn_category_members(
        &self,
        category_name: &str,
        members: Vec<(String, f64)>,
        source: &str,
    ) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.learn_category_members(category_name, &members, source)
            .map_err(map_err)
    }

    #[pyo3(signature = (category_name))]
    fn reset_category_to_seed(&self, category_name: &str) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.reset_category_to_seed(category_name).map_err(map_err)
    }
}
