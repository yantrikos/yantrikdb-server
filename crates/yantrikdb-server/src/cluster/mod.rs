//! Cluster / replication state machine.
//!
//! Implements Raft-lite leader election: only the election parts of Raft.
//! Log replication is handled by the existing CRDT oplog in yantrikdb-core,
//! which converges naturally without needing a strict log order.
//!
//! Roles:
//! - **Voter** — Full data node that can be elected leader
//! - **ReadReplica** — Consumes oplog, never votes, never accepts writes
//! - **Witness** — Vote-only, no data storage (separate binary)
//! - **Single** — Standalone, no replication
//!
//! This module intentionally exposes some methods for future use
//! and operational tooling, even when not currently called. A module-level
//! `#![allow(dead_code)]` covers those scaffolding points.

#![allow(dead_code)]

pub mod client;
pub mod election;
pub mod heartbeat;
pub mod peers;
pub mod replication;
pub mod server;
pub mod state;
pub mod sync_loop;

use std::sync::Arc;

pub use peers::PeerRegistry;
pub use state::{LeaderRole, NodeState};

use crate::config::ClusterSection;
use crate::tenant_pool::TenantPool;

/// Shared cluster context held by all background tasks and handlers.
pub struct ClusterContext {
    pub config: ClusterSection,
    pub state: Arc<NodeState>,
    pub peers: Arc<PeerRegistry>,
    pub pool: Arc<TenantPool>,
    pub control: Option<Arc<parking_lot::Mutex<crate::control::ControlDb>>>,
}

impl ClusterContext {
    pub fn new(
        config: ClusterSection,
        state: Arc<NodeState>,
        peers: Arc<PeerRegistry>,
        pool: Arc<TenantPool>,
        control: Option<Arc<parking_lot::Mutex<crate::control::ControlDb>>>,
    ) -> Self {
        Self {
            config,
            state,
            peers,
            pool,
            control,
        }
    }

    pub fn node_id(&self) -> u32 {
        self.state.node_id
    }

    pub fn quorum_size(&self) -> usize {
        self.config.quorum_size()
    }

    pub fn is_healthy(&self) -> bool {
        // We're healthy if we can reach a quorum (including ourselves).
        let reachable = self.peers.reachable_quorum_count() + 1; // +1 for self
        reachable >= self.quorum_size()
    }

    pub fn verify_secret(&self, presented: &str) -> bool {
        match &self.config.cluster_secret {
            Some(s) => s == presented,
            None => true, // no secret configured = accept any
        }
    }

    /// Get an engine for a named database. Loads it lazily if needed.
    pub fn engine_for(
        &self,
        db_name: &str,
    ) -> anyhow::Result<Arc<parking_lot::Mutex<yantrikdb::YantrikDB>>> {
        let db_record = if let Some(ref ctrl) = self.control {
            ctrl.lock()
                .get_database(db_name)?
                .ok_or_else(|| anyhow::anyhow!("database '{}' not found", db_name))?
        } else {
            // Fallback for tests / simple case
            crate::control::DatabaseRecord {
                id: 1,
                name: db_name.into(),
                path: db_name.into(),
                created_at: String::new(),
            }
        };
        self.pool.get_engine(&db_record)
    }

    /// Get a default-database engine for replication ops.
    pub fn default_engine(&self) -> anyhow::Result<Arc<parking_lot::Mutex<yantrikdb::YantrikDB>>> {
        self.engine_for("default")
    }

    /// List all replicable databases known to this node.
    pub fn list_databases(&self) -> Vec<String> {
        if let Some(ref ctrl) = self.control {
            if let Ok(dbs) = ctrl.lock().list_databases() {
                return dbs.into_iter().map(|d| d.name).collect();
            }
        }
        vec!["default".to_string()]
    }

    /// Ensure a database exists locally (used by followers when leader creates a new one).
    pub fn ensure_database(&self, name: &str) -> anyhow::Result<()> {
        let Some(ref ctrl) = self.control else {
            return Ok(()); // no control db, nothing to do
        };
        let ctrl = ctrl.lock();
        if ctrl.database_exists(name)? {
            return Ok(());
        }
        // Create the directory and the control entry
        let db_dir = self.pool.data_dir().join(name);
        std::fs::create_dir_all(&db_dir)?;
        ctrl.create_database(name, name)?;
        tracing::info!(database = %name, "auto-created database from cluster sync");
        Ok(())
    }

    /// Get our last HLC position from a specific database's oplog.
    pub fn last_hlc_for(&self, db_name: &str) -> anyhow::Result<Vec<u8>> {
        let engine = self.engine_for(db_name)?;
        let db = engine.lock();
        let conn = db.conn();
        let result: Option<Vec<u8>> = conn
            .query_row(
                "SELECT hlc FROM oplog ORDER BY hlc DESC, op_id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .ok();
        Ok(result.unwrap_or_default())
    }

    /// Get our last HLC position from the default database (for cluster status).
    pub fn last_hlc(&self) -> anyhow::Result<Vec<u8>> {
        self.last_hlc_for("default")
    }

    pub fn last_op_id_for(&self, db_name: &str) -> anyhow::Result<String> {
        let engine = self.engine_for(db_name)?;
        let db = engine.lock();
        let conn = db.conn();
        let result: Option<String> = conn
            .query_row(
                "SELECT op_id FROM oplog ORDER BY hlc DESC, op_id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .ok();
        Ok(result.unwrap_or_default())
    }

    pub fn last_op_id(&self) -> anyhow::Result<String> {
        self.last_op_id_for("default")
    }
}
