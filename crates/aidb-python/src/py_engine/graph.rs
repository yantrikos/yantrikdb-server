use pyo3::prelude::*;

use crate::py_types::*;

use super::{map_err, PyAIDB};

#[pymethods]
impl PyAIDB {
    #[pyo3(signature = (src, dst, rel_type="related_to", weight=1.0))]
    fn relate(
        &self,
        src: &str,
        dst: &str,
        rel_type: &str,
        weight: f64,
    ) -> PyResult<String> {
        let db = self.get_inner()?;
        db.relate(src, dst, rel_type, weight).map_err(map_err)
    }

    fn get_edges(&self, py: Python<'_>, entity: &str) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let edges = db.get_edges(entity).map_err(map_err)?;
        edges.iter().map(|e| edge_to_dict(py, e)).collect()
    }

    fn link_memory_entity(&self, memory_rid: &str, entity_name: &str) -> PyResult<()> {
        let db = self.get_inner()?;
        db.link_memory_entity(memory_rid, entity_name).map_err(map_err)
    }

    fn backfill_memory_entities(&self) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.backfill_memory_entities().map_err(map_err)
    }
}
