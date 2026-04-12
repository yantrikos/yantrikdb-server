//! Cluster node state machine — Raft-lite for leader election.

use parking_lot::Mutex;
#[allow(unused_imports)]
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::config::NodeRole;

/// Whether this node is currently a leader, follower, or candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaderRole {
    /// Sole writer in the cluster (was elected).
    Leader,
    /// Following a leader, applying ops via push or pull.
    Follower,
    /// Currently running an election.
    Candidate,
    /// Read-only replica — never votes, never leads.
    ReadOnly,
    /// Witness — vote-only, never leads, no data.
    Witness,
    /// Standalone single-node mode.
    Standalone,
}

impl From<NodeRole> for LeaderRole {
    fn from(role: NodeRole) -> Self {
        match role {
            NodeRole::Single => LeaderRole::Standalone,
            NodeRole::Voter => LeaderRole::Follower, // start as follower
            NodeRole::ReadReplica => LeaderRole::ReadOnly,
            NodeRole::Witness => LeaderRole::Witness,
        }
    }
}

/// Persisted Raft state — survives restarts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RaftState {
    /// Current term (monotonically increases).
    pub current_term: u64,
    /// Node ID we voted for in the current term, if any.
    pub voted_for: Option<u32>,
}

impl RaftState {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// In-memory cluster state for this node.
pub struct NodeState {
    /// This node's ID (from config).
    pub node_id: u32,
    /// This node's configured role.
    pub configured_role: NodeRole,
    /// Current leader role (changes during failover).
    pub leader_role: Mutex<LeaderRole>,
    /// Persisted Raft state (term + vote).
    pub raft: Mutex<RaftState>,
    /// Path where Raft state is persisted.
    pub raft_state_path: PathBuf,
    /// Last time we heard from the leader.
    pub last_heartbeat: Mutex<Option<Instant>>,
    /// Current leader's node_id (if known).
    pub current_leader: Mutex<Option<u32>>,
}

impl NodeState {
    pub fn new(
        node_id: u32,
        configured_role: NodeRole,
        raft_state_path: PathBuf,
    ) -> anyhow::Result<Self> {
        let raft = RaftState::load(&raft_state_path)?;
        let leader_role = LeaderRole::from(configured_role);

        Ok(Self {
            node_id,
            configured_role,
            leader_role: Mutex::new(leader_role),
            raft: Mutex::new(raft),
            raft_state_path,
            last_heartbeat: Mutex::new(None),
            current_leader: Mutex::new(None),
        })
    }

    /// Whether this node accepts writes (i.e. is the current leader OR standalone).
    pub fn accepts_writes(&self) -> bool {
        let role = *self.leader_role.lock();
        matches!(role, LeaderRole::Leader | LeaderRole::Standalone)
    }

    /// Whether this node is currently the leader.
    pub fn is_leader(&self) -> bool {
        *self.leader_role.lock() == LeaderRole::Leader
    }

    /// Whether this node participates in elections (Voter or Witness).
    pub fn is_voter(&self) -> bool {
        matches!(self.configured_role, NodeRole::Voter | NodeRole::Witness)
    }

    pub fn current_term(&self) -> u64 {
        self.raft.lock().current_term
    }

    pub fn voted_for(&self) -> Option<u32> {
        self.raft.lock().voted_for
    }

    pub fn current_leader(&self) -> Option<u32> {
        *self.current_leader.lock()
    }

    pub fn leader_role(&self) -> LeaderRole {
        *self.leader_role.lock()
    }

    /// Transition this node to follower state for the given term.
    /// Persists the new term.
    pub fn become_follower(&self, term: u64, leader_id: Option<u32>) -> anyhow::Result<()> {
        // Read replicas and witnesses don't transition.
        if matches!(
            self.configured_role,
            NodeRole::ReadReplica | NodeRole::Witness | NodeRole::Single
        ) {
            return Ok(());
        }

        let mut raft = self.raft.lock();
        if term > raft.current_term {
            raft.current_term = term;
            raft.voted_for = None;
            raft.save(&self.raft_state_path)?;
        }
        drop(raft);

        *self.leader_role.lock() = LeaderRole::Follower;
        *self.current_leader.lock() = leader_id;
        *self.last_heartbeat.lock() = Some(Instant::now());

        tracing::info!(node_id = self.node_id, term, ?leader_id, "became follower");
        Ok(())
    }

    /// Transition this node to candidate state and start an election.
    /// Increments term, votes for self.
    pub fn become_candidate(&self) -> anyhow::Result<u64> {
        if !matches!(self.configured_role, NodeRole::Voter) {
            anyhow::bail!("only voters can become candidates");
        }

        let mut raft = self.raft.lock();
        raft.current_term += 1;
        raft.voted_for = Some(self.node_id);
        raft.save(&self.raft_state_path)?;
        let new_term = raft.current_term;
        drop(raft);

        *self.leader_role.lock() = LeaderRole::Candidate;
        *self.current_leader.lock() = None;

        tracing::info!(
            node_id = self.node_id,
            term = new_term,
            "became candidate (election started)"
        );
        Ok(new_term)
    }

    /// Transition this node to leader state.
    pub fn become_leader(&self) -> anyhow::Result<()> {
        if !matches!(self.configured_role, NodeRole::Voter) {
            anyhow::bail!("only voters can become leaders");
        }

        *self.leader_role.lock() = LeaderRole::Leader;
        *self.current_leader.lock() = Some(self.node_id);

        tracing::info!(
            node_id = self.node_id,
            term = self.current_term(),
            "became leader"
        );
        Ok(())
    }

    /// Record a vote granted to a candidate this term.
    /// Returns true if vote was granted, false if already voted differently.
    pub fn grant_vote(&self, term: u64, candidate_id: u32) -> anyhow::Result<bool> {
        let mut raft = self.raft.lock();

        // Reject if our term is higher
        if term < raft.current_term {
            return Ok(false);
        }

        // If new term, reset
        if term > raft.current_term {
            raft.current_term = term;
            raft.voted_for = None;
        }

        // Already voted for someone else this term?
        if let Some(voted) = raft.voted_for {
            if voted != candidate_id {
                return Ok(false);
            }
        }

        raft.voted_for = Some(candidate_id);
        raft.save(&self.raft_state_path)?;

        tracing::debug!(term, candidate_id, voter_id = self.node_id, "vote granted");
        Ok(true)
    }

    /// Update the last-heartbeat timestamp.
    pub fn record_heartbeat(&self, leader_id: u32, term: u64) -> anyhow::Result<()> {
        // If leader's term is newer, become follower.
        if term > self.current_term() {
            self.become_follower(term, Some(leader_id))?;
        } else {
            *self.current_leader.lock() = Some(leader_id);
            *self.last_heartbeat.lock() = Some(Instant::now());
        }
        Ok(())
    }

    /// How long since the last heartbeat from leader. None if never.
    pub fn time_since_heartbeat(&self) -> Option<std::time::Duration> {
        self.last_heartbeat.lock().as_ref().map(|t| t.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_state(role: NodeRole) -> (NodeState, TempDir) {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("raft.json");
        let state = NodeState::new(1, role, path).unwrap();
        (state, tmp)
    }

    #[test]
    fn standalone_accepts_writes() {
        let (state, _tmp) = make_state(NodeRole::Single);
        assert!(state.accepts_writes());
    }

    #[test]
    fn voter_starts_as_follower() {
        let (state, _tmp) = make_state(NodeRole::Voter);
        assert_eq!(state.leader_role(), LeaderRole::Follower);
        assert!(!state.accepts_writes());
    }

    #[test]
    fn read_replica_never_writes() {
        let (state, _tmp) = make_state(NodeRole::ReadReplica);
        assert_eq!(state.leader_role(), LeaderRole::ReadOnly);
        assert!(!state.accepts_writes());
        assert!(!state.is_voter());
    }

    #[test]
    fn become_candidate_increments_term() {
        let (state, _tmp) = make_state(NodeRole::Voter);
        assert_eq!(state.current_term(), 0);
        let new_term = state.become_candidate().unwrap();
        assert_eq!(new_term, 1);
        assert_eq!(state.current_term(), 1);
        assert_eq!(state.voted_for(), Some(1));
    }

    #[test]
    fn become_leader_only_after_candidate() {
        let (state, _tmp) = make_state(NodeRole::Voter);
        state.become_candidate().unwrap();
        state.become_leader().unwrap();
        assert!(state.is_leader());
        assert!(state.accepts_writes());
    }

    #[test]
    fn grant_vote_only_once_per_term() {
        let (state, _tmp) = make_state(NodeRole::Voter);
        assert!(state.grant_vote(1, 2).unwrap());
        // Same candidate same term — idempotent OK
        assert!(state.grant_vote(1, 2).unwrap());
        // Different candidate same term — denied
        assert!(!state.grant_vote(1, 3).unwrap());
    }

    #[test]
    fn higher_term_resets_vote() {
        let (state, _tmp) = make_state(NodeRole::Voter);
        state.grant_vote(1, 2).unwrap();
        // New term — can vote for different candidate
        assert!(state.grant_vote(2, 3).unwrap());
        assert_eq!(state.current_term(), 2);
        assert_eq!(state.voted_for(), Some(3));
    }

    #[test]
    fn raft_state_persists() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("raft.json");

        let state1 = NodeState::new(1, NodeRole::Voter, path.clone()).unwrap();
        state1.become_candidate().unwrap();
        assert_eq!(state1.current_term(), 1);
        drop(state1);

        // Reload
        let state2 = NodeState::new(1, NodeRole::Voter, path).unwrap();
        assert_eq!(state2.current_term(), 1);
        assert_eq!(state2.voted_for(), Some(1));
    }
}
