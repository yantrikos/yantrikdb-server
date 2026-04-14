use pyo3::prelude::*;
use pyo3::types::PyDict;

use yantrikdb_core::types::MemoryWithEmbedding;

use crate::py_engine::PyYantrikDB;
use crate::py_types::*;

/// Compute cosine similarity between two vectors.
#[pyfunction]
#[pyo3(name = "_cosine_similarity")]
pub fn py_cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> f64 {
    yantrikdb_core::consolidate::cosine_similarity(&a, &b)
}

/// Generate an extractive summary from a list of memory dicts.
#[pyfunction]
#[pyo3(name = "_extractive_summary")]
pub fn py_extractive_summary(memories: Vec<Bound<'_, PyDict>>) -> PyResult<String> {
    let mems: Vec<MemoryWithEmbedding> = memories
        .iter()
        .map(|d| dict_to_mem_with_embedding(d))
        .collect::<PyResult<_>>()?;
    Ok(yantrikdb_core::consolidate::extractive_summary(&mems))
}

/// Find clusters of related memories.
#[pyfunction]
#[pyo3(name = "_find_clusters", signature = (memories, sim_threshold=0.6, time_window_days=7.0, min_cluster_size=2, max_cluster_size=10))]
pub fn py_find_clusters(
    memories: Vec<Bound<'_, PyDict>>,
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    max_cluster_size: usize,
) -> PyResult<Vec<Vec<PyObject>>> {
    let mems: Vec<MemoryWithEmbedding> = memories
        .iter()
        .map(|d| dict_to_mem_with_embedding(d))
        .collect::<PyResult<_>>()?;

    let cluster_indices = yantrikdb_core::consolidate::find_clusters(
        &mems,
        None,
        sim_threshold,
        time_window_days,
        min_cluster_size,
        max_cluster_size,
    );

    // Convert back: each cluster is a list of the original dicts
    let result: Vec<Vec<PyObject>> = cluster_indices
        .into_iter()
        .map(|indices| {
            indices
                .into_iter()
                .map(|i| memories[i].clone().into_any().unbind())
                .collect()
        })
        .collect();

    Ok(result)
}

/// Find consolidation candidates.
#[pyfunction]
#[pyo3(signature = (db, sim_threshold=0.6, time_window_days=7.0, min_cluster_size=2, limit=100, require_entity_overlap=true))]
pub fn find_consolidation_candidates(
    py: Python<'_>,
    db: &PyYantrikDB,
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    limit: usize,
    require_entity_overlap: bool,
) -> PyResult<Vec<Vec<PyObject>>> {
    let inner = db.get_inner()?;
    let clusters = yantrikdb_core::consolidate::find_consolidation_candidates(
        inner,
        sim_threshold,
        time_window_days,
        min_cluster_size,
        limit,
        require_entity_overlap,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Convert clusters to Python dicts
    clusters
        .iter()
        .map(|cluster| {
            cluster
                .iter()
                .map(|m| mem_with_emb_to_dict(py, m))
                .collect::<PyResult<Vec<_>>>()
        })
        .collect()
}

/// Run the full consolidation pipeline.
#[pyfunction]
#[pyo3(name = "consolidate", signature = (db, sim_threshold=0.6, time_window_days=7.0, min_cluster_size=2, limit=100, require_entity_overlap=true, dry_run=false))]
pub fn py_consolidate(
    py: Python<'_>,
    db: &PyYantrikDB,
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    limit: usize,
    require_entity_overlap: bool,
    dry_run: bool,
) -> PyResult<Vec<PyObject>> {
    let inner = db.get_inner()?;
    let results = yantrikdb_core::consolidate::consolidate(
        inner,
        sim_threshold,
        time_window_days,
        min_cluster_size,
        limit,
        require_entity_overlap,
        dry_run,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    results
        .iter()
        .map(|v| json_to_py(py, v))
        .collect()
}

fn dict_to_mem_with_embedding(d: &Bound<'_, PyDict>) -> PyResult<MemoryWithEmbedding> {
    let rid: String = d.get_item("rid")?.map(|v| v.extract()).unwrap_or(Ok("".to_string()))?;
    let text: String = d.get_item("text")?.map(|v| v.extract()).unwrap_or(Ok("".to_string()))?;
    let memory_type: String = d.get_item("type")?.map(|v| v.extract()).unwrap_or(Ok("episodic".to_string()))?;
    let embedding: Vec<f32> = d.get_item("embedding")?.map(|v| v.extract()).unwrap_or(Ok(vec![]))?;
    let created_at: f64 = d.get_item("created_at")?.map(|v| v.extract()).unwrap_or(Ok(0.0))?;
    let importance: f64 = d.get_item("importance")?.map(|v| v.extract()).unwrap_or(Ok(0.5))?;
    let valence: f64 = d.get_item("valence")?.map(|v| v.extract()).unwrap_or(Ok(0.0))?;
    let half_life: f64 = d.get_item("half_life")?.map(|v| v.extract()).unwrap_or(Ok(604800.0))?;
    let last_access: f64 = d.get_item("last_access")?.map(|v| v.extract()).unwrap_or(Ok(0.0))?;
    let metadata: serde_json::Value = d
        .get_item("metadata")?
        .map(|v| py_to_json(&v))
        .unwrap_or(Ok(serde_json::json!({})))?;

    Ok(MemoryWithEmbedding {
        rid,
        memory_type,
        text,
        embedding,
        created_at,
        importance,
        valence,
        half_life,
        last_access,
        metadata,
        namespace: "default".to_string(),
    })
}

fn mem_with_emb_to_dict(py: Python<'_>, m: &MemoryWithEmbedding) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("rid", &m.rid)?;
    dict.set_item("type", &m.memory_type)?;
    dict.set_item("text", &m.text)?;
    dict.set_item("embedding", &m.embedding)?;
    dict.set_item("created_at", m.created_at)?;
    dict.set_item("importance", m.importance)?;
    dict.set_item("valence", m.valence)?;
    dict.set_item("half_life", m.half_life)?;
    dict.set_item("last_access", m.last_access)?;
    dict.set_item("metadata", json_to_py(py, &m.metadata)?)?;
    Ok(dict.into())
}
