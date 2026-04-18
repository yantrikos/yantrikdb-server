use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_types::*;

use super::{map_err, PyYantrikDB};

#[pymethods]
impl PyYantrikDB {
    // ── Storage tier operations ──

    fn archive(&self, rid: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.archive(rid).map_err(map_err)
    }

    fn hydrate(&self, rid: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.hydrate(rid).map_err(map_err)
    }

    #[pyo3(signature = (max_active,))]
    fn evict(&self, max_active: usize) -> PyResult<Vec<String>> {
        let db = self.get_inner()?;
        db.evict(max_active).map_err(map_err)
    }

    // ── Replication API (V5 P2P Sync) ──

    #[pyo3(signature = (since_hlc=None, since_op_id=None, exclude_actor=None, limit=1000))]
    fn extract_ops_since(
        &self,
        py: Python<'_>,
        since_hlc: Option<Vec<u8>>,
        since_op_id: Option<&str>,
        exclude_actor: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let ops = yantrikdb_core::replication::extract_ops_since(
            &*db.conn(),
            since_hlc.as_deref(),
            since_op_id,
            exclude_actor,
            limit,
        )
        .map_err(map_err)?;

        let mut result = Vec::with_capacity(ops.len());
        for op in &ops {
            let dict = PyDict::new(py);
            dict.set_item("op_id", &op.op_id)?;
            dict.set_item("op_type", &op.op_type)?;
            dict.set_item("timestamp", op.timestamp)?;
            dict.set_item("target_rid", &op.target_rid)?;
            dict.set_item("payload", json_to_py(py, &op.payload)?)?;
            dict.set_item("actor_id", &op.actor_id)?;
            dict.set_item("hlc", &op.hlc)?;
            dict.set_item("embedding_hash", &op.embedding_hash)?;
            dict.set_item("origin_actor", &op.origin_actor)?;
            result.push(dict.into());
        }
        Ok(result)
    }

    fn apply_ops(&self, py: Python<'_>, ops: Vec<Bound<'_, PyDict>>) -> PyResult<PyObject> {
        let db = self.get_inner()?;

        let mut entries = Vec::with_capacity(ops.len());
        for d in &ops {
            let op_id: String = d
                .get_item("op_id")?
                .ok_or_else(|| PyValueError::new_err("Each op must have 'op_id'"))?
                .extract()?;
            let op_type: String = d
                .get_item("op_type")?
                .ok_or_else(|| PyValueError::new_err("Each op must have 'op_type'"))?
                .extract()?;
            let timestamp: f64 = d
                .get_item("timestamp")?
                .ok_or_else(|| PyValueError::new_err("Each op must have 'timestamp'"))?
                .extract()?;
            let target_rid: Option<String> = d
                .get_item("target_rid")?
                .and_then(|v| if v.is_none() { None } else { Some(v) })
                .map(|v| v.extract())
                .transpose()?;
            let payload = d
                .get_item("payload")?
                .map(|v| py_to_json(&v))
                .transpose()?
                .unwrap_or(serde_json::json!({}));
            let actor_id: String = d
                .get_item("actor_id")?
                .ok_or_else(|| PyValueError::new_err("Each op must have 'actor_id'"))?
                .extract()?;
            let hlc: Vec<u8> = d
                .get_item("hlc")?
                .ok_or_else(|| PyValueError::new_err("Each op must have 'hlc'"))?
                .extract()?;
            let embedding_hash: Option<Vec<u8>> = d
                .get_item("embedding_hash")?
                .and_then(|v| if v.is_none() { None } else { Some(v) })
                .map(|v| v.extract())
                .transpose()?;
            let origin_actor: String = d
                .get_item("origin_actor")?
                .ok_or_else(|| PyValueError::new_err("Each op must have 'origin_actor'"))?
                .extract()?;

            entries.push(yantrikdb_core::replication::OplogEntry {
                op_id,
                op_type,
                timestamp,
                target_rid,
                payload,
                actor_id,
                hlc,
                embedding_hash,
                origin_actor,
            });
        }

        let stats = yantrikdb_core::replication::apply_ops(db, &entries).map_err(map_err)?;
        let dict = PyDict::new(py);
        dict.set_item("ops_applied", stats.ops_applied)?;
        dict.set_item("ops_skipped", stats.ops_skipped)?;
        Ok(dict.into())
    }

    fn get_peer_watermark(&self, py: Python<'_>, peer_actor: &str) -> PyResult<Option<PyObject>> {
        let db = self.get_inner()?;
        match yantrikdb_core::replication::get_peer_watermark(&*db.conn(), peer_actor)
            .map_err(map_err)?
        {
            Some((hlc, op_id)) => {
                let dict = PyDict::new(py);
                dict.set_item("hlc", &hlc)?;
                dict.set_item("op_id", &op_id)?;
                Ok(Some(dict.into()))
            }
            None => Ok(None),
        }
    }

    fn set_peer_watermark(&self, peer_actor: &str, hlc: Vec<u8>, op_id: &str) -> PyResult<()> {
        let db = self.get_inner()?;
        yantrikdb_core::replication::set_peer_watermark(&*db.conn(), peer_actor, &hlc, op_id)
            .map_err(map_err)
    }

    fn rebuild_vec_index(&self) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.rebuild_vec_index().map_err(map_err)
    }

    fn rebuild_graph_index(&self) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.rebuild_graph_index().map_err(map_err)
    }
}
