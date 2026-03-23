use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use yantrikdb_core::YantrikDB;

use crate::py_engine::{map_err, PyYantrikDB};

/// Per-tenant configuration stored in the manager.
struct TenantEntry {
    encryption_key: Option<[u8; 32]>,
    embedding_dim: Option<usize>,
}

/// Python multi-tenant manager. Creates isolated PyYantrikDB instances per tenant.
///
/// Each tenant gets a separate SQLite database file under `base_dir/`.
/// Optional per-tenant encryption keys provide defense-in-depth.
///
/// Usage:
///     mgr = TenantManager("/data/tenants", embedding_dim=384)
///     mgr.register_tenant("acme", encryption_key=os.urandom(32))
///     db = mgr.get("acme", embedder=my_embedder)
///     db.record(...)
#[pyclass(name = "TenantManager")]
pub struct PyTenantManager {
    base_dir: PathBuf,
    default_embedding_dim: usize,
    configs: HashMap<String, TenantEntry>,
}

#[pymethods]
impl PyTenantManager {
    #[new]
    #[pyo3(signature = (base_dir, embedding_dim=384))]
    fn new(base_dir: &str, embedding_dim: usize) -> PyResult<Self> {
        let path = PathBuf::from(base_dir);
        std::fs::create_dir_all(&path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to create tenant base dir: {e}"
            ))
        })?;
        Ok(Self {
            base_dir: path,
            default_embedding_dim: embedding_dim,
            configs: HashMap::new(),
        })
    }

    /// Register a tenant with optional encryption key and embedding dimension.
    #[pyo3(signature = (tenant_id, encryption_key=None, embedding_dim=None))]
    fn register_tenant(
        &mut self,
        tenant_id: &str,
        encryption_key: Option<Vec<u8>>,
        embedding_dim: Option<usize>,
    ) -> PyResult<()> {
        let key = if let Some(bytes) = encryption_key {
            if bytes.len() != 32 {
                return Err(PyValueError::new_err(
                    "encryption_key must be exactly 32 bytes",
                ));
            }
            let mut key = [0u8; 32];
            key.copy_from_slice(&bytes);
            Some(key)
        } else {
            None
        };
        self.configs.insert(
            tenant_id.to_string(),
            TenantEntry {
                encryption_key: key,
                embedding_dim,
            },
        );
        Ok(())
    }

    /// Get a PyYantrikDB instance for a tenant.
    ///
    /// Each call creates a new connection to the tenant's DB file.
    /// If the tenant has a registered encryption key, the DB is opened encrypted.
    #[pyo3(signature = (tenant_id, embedder=None))]
    fn get(&self, tenant_id: &str, embedder: Option<PyObject>) -> PyResult<PyYantrikDB> {
        let db_path = self
            .base_dir
            .join(format!("{tenant_id}.db"))
            .to_string_lossy()
            .to_string();

        let config = self.configs.get(tenant_id);
        let dim = config
            .and_then(|c| c.embedding_dim)
            .unwrap_or(self.default_embedding_dim);

        let inner = match config.and_then(|c| c.encryption_key.as_ref()) {
            Some(key) => YantrikDB::new_encrypted(&db_path, dim, key).map_err(map_err)?,
            None => YantrikDB::new(&db_path, dim).map_err(map_err)?,
        };

        Ok(PyYantrikDB {
            inner: Some(Arc::new(inner)),
            embedder,
        })
    }

    /// List all tenant DB files discovered in the base directory.
    fn discovered_tenants(&self) -> PyResult<Vec<String>> {
        let mut tenants = Vec::new();
        let entries = std::fs::read_dir(&self.base_dir).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to read tenant dir: {e}"
            ))
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "failed to read dir entry: {e}"
                ))
            })?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".db") {
                tenants.push(name_str.trim_end_matches(".db").to_string());
            }
        }
        tenants.sort();
        Ok(tenants)
    }
}
