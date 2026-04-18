use pyo3::prelude::*;

use crate::py_types::*;

use super::{map_err, PyYantrikDB};

#[pymethods]
impl PyYantrikDB {
    #[pyo3(signature = (src, dst, rel_type="related_to", weight=1.0))]
    fn relate(&self, src: &str, dst: &str, rel_type: &str, weight: f64) -> PyResult<String> {
        let db = self.get_inner()?;
        db.relate(src, dst, rel_type, weight).map_err(map_err)
    }

    fn get_edges(&self, py: Python<'_>, entity: &str) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let edges = db.get_edges(entity).map_err(map_err)?;
        edges.iter().map(|e| edge_to_dict(py, e)).collect()
    }

    #[pyo3(signature = (pattern=None, entity_type=None, limit=20))]
    fn search_entities(
        &self,
        py: Python<'_>,
        pattern: Option<&str>,
        entity_type: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let entities = db
            .search_entities(pattern, entity_type, limit)
            .map_err(map_err)?;
        entities.iter().map(|e| entity_to_dict(py, e)).collect()
    }

    fn link_memory_entity(&self, memory_rid: &str, entity_name: &str) -> PyResult<()> {
        let db = self.get_inner()?;
        db.link_memory_entity(memory_rid, entity_name)
            .map_err(map_err)
    }

    fn backfill_memory_entities(&self) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.backfill_memory_entities().map_err(map_err)
    }
}
