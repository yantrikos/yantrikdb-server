//! Wire-shape types for the openraft adapter.
//!
//! ## What's defined here
//!
//! - [`YantrikNodeId`] — identifier for a YantrikDB cluster node. `u64`
//!   for compatibility with openraft's default; we use a stable
//!   monotonic id from cluster config rather than UUIDs to keep log
//!   entries small and comparison cheap.
//! - [`YantrikNode`] — connection info for a peer (HTTP address). Stored
//!   in the openraft membership log so leadership transfers and joins
//!   know where to reach a node.
//! - [`YantrikLogEntry`] — application data carried in each Raft log
//!   entry. Wraps `(tenant_id, op_id, mutation)` because:
//!     1. **`tenant_id`** — Raft has a single global log per cluster,
//!        but our [`crate::commit::MemoryMutation`] is per-tenant. The
//!        log entry must carry the tenant so the apply path knows which
//!        tenant log to advance.
//!     2. **`op_id`** — preserved end-to-end for client idempotency
//!        across leader failover. A retry on the new leader hashes to
//!        the same `op_id` and the state machine returns the original
//!        receipt.
//!     3. **`mutation`** — the canonical RFC 010 grammar. Already
//!        version-pinned via `VersionedEvent`; openraft transports it
//!        opaquely.
//! - [`YantrikRaftResponse`] — what the state machine returns after
//!   applying an entry. Equivalent to a [`crate::commit::CommitReceipt`]
//!   minus the `op_id`/`tenant_id` (those are inputs, not outputs).
//! - [`YantrikRaftTypeConfig`] — pulls the above into openraft's
//!   `RaftTypeConfig` shape. Future sub-PRs (4-b/c/d) implement the
//!   storage / state-machine / network traits against this config.
//!
//! ## Why a separate `YantrikRaftResponse` instead of reusing `CommitReceipt`
//!
//! openraft's `R: AppDataResponse` is the type the state machine
//! returns. `CommitReceipt` includes `op_id` and `tenant_id` which the
//! caller already knows; baking them into `R` would duplicate them on
//! the wire. A slimmer response keeps replication payload minimal.
//!
//! The committer-facing `RaftCommitter` (PR-4-d) reconstructs the full
//! `CommitReceipt` from the input + `YantrikRaftResponse` so callers
//! see the same shape regardless of which `MutationCommitter` impl is
//! active.

use std::fmt;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::commit::{MemoryMutation, OpId, TenantId};

/// Cluster node identifier. Wraps `u64` so future expansion (e.g. node
/// generation tag for crash-restart disambiguation) doesn't require a
/// migration of the log format.
///
/// Stability: serialized in every membership-change log entry. Once a
/// cluster has used `YantrikNodeId(7)` for a node, that id MUST NOT be
/// recycled for a different physical node — fencing tokens depend on
/// id stability.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
#[serde(transparent)]
pub struct YantrikNodeId(pub u64);

impl YantrikNodeId {
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    pub const fn raw(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for YantrikNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node-{}", self.0)
    }
}

impl From<u64> for YantrikNodeId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<YantrikNodeId> for u64 {
    fn from(id: YantrikNodeId) -> Self {
        id.0
    }
}

/// Per-node connection info stored in the membership log so leaders +
/// new followers know where to reach peers across joins, transfers, and
/// crash-restart sequences.
///
/// `addr` is the cluster RPC URL (e.g. `https://10.0.0.5:7100`). HTTPS
/// is the production default once mTLS (RFC 014-A) is enforced; HTTP
/// only for dev / single-node loopback.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct YantrikNode {
    pub addr: String,
}

impl YantrikNode {
    pub fn new(addr: impl Into<String>) -> Self {
        Self { addr: addr.into() }
    }
}

impl fmt::Display for YantrikNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.addr)
    }
}

/// What every Raft log entry carries. See module docs for why each
/// field is required.
///
/// `#[serde(tag = "kind")]` is set on the inner `MemoryMutation`, so
/// this struct serializes as a flat object — easy to inspect via
/// `/v1/debug/history` (RFC 010 PR-5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct YantrikLogEntry {
    /// Per-tenant log scope. Apply path uses this to advance the
    /// per-tenant `log_index` watermark even though the Raft log itself
    /// is cluster-global.
    pub tenant_id: TenantId,
    /// Client idempotency token. Preserved across leader failover so
    /// retries return the original receipt (or `OpIdCollision` if the
    /// payload differs).
    pub op_id: OpId,
    /// The canonical RFC 010 mutation. Versioned via [`crate::version`].
    pub mutation: MemoryMutation,
}

impl YantrikLogEntry {
    /// Construct a new log entry. Used by the `RaftCommitter` (PR-4-d)
    /// before submitting to openraft, and by the log-storage adapter
    /// (PR-4-b) when reconstructing entries from `memory_commit_log`.
    pub fn new(tenant_id: TenantId, op_id: OpId, mutation: MemoryMutation) -> Self {
        Self {
            tenant_id,
            op_id,
            mutation,
        }
    }

    /// Decompose into the parts the apply path consumes.
    pub fn into_parts(self) -> (TenantId, OpId, MemoryMutation) {
        (self.tenant_id, self.op_id, self.mutation)
    }

    /// Borrow the inner mutation. Cheap — used by the state-machine
    /// (PR-4-c) to inspect without taking ownership.
    pub fn mutation(&self) -> &MemoryMutation {
        &self.mutation
    }
}

/// State-machine response after a successful apply. Slim on purpose —
/// see module docs.
///
/// `term` and `log_index` are populated by openraft; the apply path
/// does not assign them. We carry them in the response so the
/// `RaftCommitter` (PR-4-d) can reconstruct a full `CommitReceipt`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct YantrikRaftResponse {
    /// Raft term at which the entry was committed. Used by clients
    /// that pin reads to a (term, log_index) for linearizability.
    pub term: u64,
    /// Per-tenant log index assigned by the apply path. Distinct from
    /// the cluster-global Raft log index — see `YantrikLogEntry` docs.
    pub tenant_log_index: u64,
    /// Unix-microseconds at apply time. The committed_at field on
    /// `CommitReceipt` is reconstructed from this.
    pub applied_at_unix_micros: i64,
}

impl YantrikRaftResponse {
    /// Build a response. Used by the state-machine (PR-4-c).
    pub fn new(term: u64, tenant_log_index: u64, applied_at: SystemTime) -> Self {
        let micros = applied_at
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);
        Self {
            term,
            tenant_log_index,
            applied_at_unix_micros: micros,
        }
    }
}

/// openraft type-config for YantrikDB. Pulls the wire types together;
/// future sub-PRs implement the storage/state-machine/network traits
/// against `YantrikRaftTypeConfig`.
///
/// `Entry = openraft::Entry<Self>` uses openraft's built-in framing
/// (term + index + payload + membership sentinel) — we don't need a
/// custom entry type because [`YantrikLogEntry`] is the application
/// payload, not the framing.
///
/// `SnapshotData = std::io::Cursor<Vec<u8>>` is a placeholder. PR-4-c
/// will swap to a streaming SQLite checkpoint reader so install-snapshot
/// doesn't materialize the whole DB in memory.
///
/// `AsyncRuntime = openraft::TokioRuntime` matches our existing tokio
/// runtime (RFC 009 split-runtime keeps Raft tasks on the control
/// runtime).
openraft::declare_raft_types!(
    pub YantrikRaftTypeConfig:
        D = YantrikLogEntry,
        R = YantrikRaftResponse,
        NodeId = YantrikNodeId,
        Node = YantrikNode,
        Entry = openraft::Entry<YantrikRaftTypeConfig>,
        SnapshotData = std::io::Cursor<Vec<u8>>,
        AsyncRuntime = openraft::TokioRuntime,
);

#[cfg(test)]
mod tests {
    use super::*;

    fn upsert(rid: &str) -> MemoryMutation {
        MemoryMutation::UpsertMemory {
            rid: rid.into(),
            text: "x".into(),
            memory_type: "semantic".into(),
            importance: 0.5,
            valence: 0.0,
            half_life: 168.0,
            namespace: "default".into(),
            certainty: 1.0,
            domain: "general".into(),
            source: "user".into(),
            emotional_state: None,
            embedding: None,
            metadata: serde_json::json!({}),
        }
    }

    #[test]
    fn node_id_serde_is_transparent() {
        let id = YantrikNodeId::new(7);
        let json = serde_json::to_string(&id).unwrap();
        // Transparent => raw u64 in JSON, not an object wrapper.
        assert_eq!(json, "7");
        let back: YantrikNodeId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn node_id_display_includes_node_prefix() {
        // `node-7` in logs/metrics is more readable than bare `7`.
        assert_eq!(YantrikNodeId::new(7).to_string(), "node-7");
    }

    #[test]
    fn node_id_round_trips_through_u64_conversions() {
        let id = YantrikNodeId::from(42_u64);
        assert_eq!(id.raw(), 42);
        assert_eq!(u64::from(id), 42);
    }

    #[test]
    fn node_serde_round_trip() {
        let n = YantrikNode::new("https://10.0.0.5:7100");
        let json = serde_json::to_string(&n).unwrap();
        let back: YantrikNode = serde_json::from_str(&json).unwrap();
        assert_eq!(n, back);
        assert_eq!(n.to_string(), "https://10.0.0.5:7100");
    }

    #[test]
    fn log_entry_carries_all_three_required_fields() {
        // Per the doc on YantrikLogEntry: tenant_id, op_id, mutation
        // are all required for correct apply across failover.
        let tenant = TenantId::new(42);
        let op_id = OpId::new_random();
        let m = upsert("mem_a");
        let e = YantrikLogEntry::new(tenant, op_id, m.clone());
        assert_eq!(e.tenant_id, tenant);
        assert_eq!(e.op_id, op_id);
        assert_eq!(e.mutation(), &m);

        let (t, o, mm) = e.into_parts();
        assert_eq!(t, tenant);
        assert_eq!(o, op_id);
        assert_eq!(mm, m);
    }

    #[test]
    fn log_entry_serde_round_trip_is_lossless() {
        let entry = YantrikLogEntry::new(TenantId::new(1), OpId::new_random(), upsert("mem_a"));
        let json = serde_json::to_string(&entry).unwrap();
        let back: YantrikLogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn log_entry_with_tombstone_round_trips() {
        // RFC 011-B made TombstoneMemory implementable; the wire form
        // must round-trip through the Raft log just like any other
        // mutation.
        let entry = YantrikLogEntry::new(
            TenantId::new(1),
            OpId::new_random(),
            MemoryMutation::TombstoneMemory {
                rid: "mem_a".into(),
                reason: Some("test".into()),
                requested_at_unix_micros: 1_700_000_000_000_000,
            },
        );
        let json = serde_json::to_string(&entry).unwrap();
        let back: YantrikLogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn raft_response_micros_round_trip() {
        let when = std::time::UNIX_EPOCH + std::time::Duration::from_micros(1_700_000_000_000_000);
        let r = YantrikRaftResponse::new(7, 42, when);
        assert_eq!(r.term, 7);
        assert_eq!(r.tenant_log_index, 42);
        assert_eq!(r.applied_at_unix_micros, 1_700_000_000_000_000);
    }

    #[test]
    fn raft_response_serde_round_trip() {
        let r = YantrikRaftResponse {
            term: 3,
            tenant_log_index: 99,
            applied_at_unix_micros: 1_700_000_000_000_000,
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: YantrikRaftResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn type_config_associated_types_align() {
        // Compile-only: assert the macro wired the types correctly.
        // If anyone changes the declare_raft_types! invocation in a way
        // that breaks the AppData/AppDataResponse association, this
        // assertion stops compiling.
        fn _assert_assoc<C: openraft::RaftTypeConfig>() {}
        _assert_assoc::<YantrikRaftTypeConfig>();

        // Concrete type-id checks at runtime to pin the wire shape.
        let _: <YantrikRaftTypeConfig as openraft::RaftTypeConfig>::D =
            YantrikLogEntry::new(TenantId::new(1), OpId::new_random(), upsert("mem_a"));
        let _: <YantrikRaftTypeConfig as openraft::RaftTypeConfig>::R = YantrikRaftResponse {
            term: 0,
            tenant_log_index: 0,
            applied_at_unix_micros: 0,
        };
        let _: <YantrikRaftTypeConfig as openraft::RaftTypeConfig>::NodeId = YantrikNodeId::new(0);
        let _: <YantrikRaftTypeConfig as openraft::RaftTypeConfig>::Node =
            YantrikNode::new("http://localhost");
    }
}
