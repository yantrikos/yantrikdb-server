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

#![allow(dead_code)]
//! Cluster module — intentionally exposes some methods for future use
//! and operational tooling, even when not currently called.

pub mod client;
pub mod election;
pub mod heartbeat;
pub mod peers;
pub mod replication;
pub mod server;
pub mod state;
pub mod sync_loop;

use std::sync::Arc;

pub use peers::{PeerRegistry, PeerStatus};
pub use state::{LeaderRole, NodeState, RaftState};

use crate::config::ClusterSection;
use crate::tenant_pool::TenantPool;

/// Shared cluster context held by all background tasks and handlers.
pub struct ClusterContext {
    pub config: ClusterSection,
    pub state: Arc<NodeState>,
    pub peers: Arc<PeerRegistry>,
    pub pool: Arc<TenantPool>,
}

impl ClusterContext {
    pub fn new(
        config: ClusterSection,
        state: Arc<NodeState>,
        peers: Arc<PeerRegistry>,
        pool: Arc<TenantPool>,
    ) -> Self {
        Self {
            config,
            state,
            peers,
            pool,
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

    /// Get a default-database engine for replication ops.
    /// For now we replicate the "default" database. Multi-db replication
    /// will come in a follow-up.
    pub fn default_engine(&self) -> anyhow::Result<Arc<std::sync::Mutex<yantrikdb::YantrikDB>>> {
        // The default database always exists (created on server startup).
        let db_record = crate::control::DatabaseRecord {
            id: 1,
            name: "default".into(),
            path: "default".into(),
            created_at: String::new(),
        };
        self.pool.get_engine(&db_record)
    }

    /// Get our last HLC position from the local oplog.
    pub fn last_hlc(&self) -> anyhow::Result<Vec<u8>> {
        let engine = self.default_engine()?;
        let db = engine.lock().unwrap();
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

    pub fn last_op_id(&self) -> anyhow::Result<String> {
        let engine = self.default_engine()?;
        let db = engine.lock().unwrap();
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
}
