//! RFC 010 PR-4 — openraft adapter.
//!
//! ## Sub-PR layout (this PR is PR-4-a)
//!
//! | Sub-PR | Module | Purpose |
//! |---|---|---|
//! | **4-a (this)** | [`types`] | Type config + log-entry wrapper + node id/info. No I/O. |
//! | 4-b | `log_storage` | `RaftLogStorage` over `memory_commit_log`. |
//! | 4-c | `state_machine` | `RaftStateMachine` + snapshot. |
//! | 4-d | `network`, `committer` | HTTP `RaftNetwork` + `RaftCommitter` exposing [`crate::commit::MutationCommitter`]. |
//! | 4-e | (tests) | 3-node cluster boot, leader-kill failover, partition heal. |
//!
//! ## Why split openraft adoption into 5 sub-PRs
//!
//! openraft has four orthogonal trait surfaces (`RaftLogStorage`,
//! `RaftStateMachine`, `RaftNetwork`, plus the type config that ties
//! them together). Implementing all four in one PR would mean ~2000 LOC
//! of trait satisfaction with no test coverage until the very end —
//! exactly the kind of half-baked replication adapter the gpt-5.5
//! redteam flagged as the #1 architectural risk for YantrikDB.
//!
//! The split lets each sub-PR ship green tests and an isolated unit of
//! correctness: 4-a pins the wire types, 4-b verifies log durability
//! against `memory_commit_log`, 4-c verifies state-machine apply, 4-d
//! wires the network + committer, 4-e proves the whole thing under
//! Jepsen-style partition.
//!
//! ## What this sub-PR (4-a) does NOT include
//!
//! - No openraft `RaftLogStorage` impl yet (4-b).
//! - No state-machine apply (4-c).
//! - No network transport (4-d).
//! - No `RaftCommitter` wrapping `MutationCommitter` (4-d).
//! - No mTLS gate enforcement on cluster mode (4-d ties this to RFC 014-A).

pub mod assembly;
pub mod committer;
pub mod http_network;
pub mod log_storage;
pub mod network;
pub mod state_machine;
pub mod status;
pub mod types;

pub use assembly::{
    build_raft_cluster, AssemblyError, RaftAssembly, RaftAssemblyConfig, RaftClusterMode,
};
pub use committer::RaftCommitter;
pub use http_network::{raft_receive_router, HttpRaftNetwork, HttpRaftNetworkFactory};
pub use log_storage::SqliteRaftLogStorage;
pub use network::{StubRaftNetwork, StubRaftNetworkFactory};
pub use state_machine::{StateMachineState, YantrikSnapshotBuilder, YantrikStateMachine};
pub use status::{
    raft_status_router, record_openraft_status, spawn_raft_metrics_recorder, RaftMember,
    RaftStatus,
};
pub use types::{
    YantrikLogEntry, YantrikNode, YantrikNodeId, YantrikRaftResponse, YantrikRaftTypeConfig,
};
