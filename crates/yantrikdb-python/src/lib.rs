use pyo3::prelude::*;

pub mod py_consolidate;
pub mod py_engine;
pub mod py_tenant;
pub mod py_triggers;
pub mod py_types;

#[pymodule]
fn _yantrikdb_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Engine
    m.add_class::<py_engine::PyYantrikDB>()?;
    m.add_class::<py_tenant::PyTenantManager>()?;

    // Triggers
    m.add_class::<py_triggers::PyTrigger>()?;
    m.add_function(wrap_pyfunction!(py_triggers::check_decay_triggers, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_triggers::check_consolidation_triggers,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_triggers::check_all_triggers, m)?)?;

    // Consolidation
    m.add_function(wrap_pyfunction!(py_consolidate::py_consolidate, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_consolidate::find_consolidation_candidates,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_consolidate::py_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_consolidate::py_extractive_summary, m)?)?;
    m.add_function(wrap_pyfunction!(py_consolidate::py_find_clusters, m)?)?;

    Ok(())
}
