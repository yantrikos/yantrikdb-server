//! YRP — YantrikDB Replication Protocol (RFC 028 v2).
//!
//! Purpose-built clustering for the cognitive memory substrate. Per the RFC's
//! reframe: YRP rejects openraft's *operational posture* (wedge-on-confusion
//! boot, library-owned snapshot format, integration weight), **not Raft's
//! safety mathematics**. The safety core below is Raft-shaped, by the book,
//! and deliberately boring; the purpose-built parts (memory-native oplog
//! payloads, engine-checkpoint snapshots, quarantine-not-wedge recovery,
//! quorum-managed incarnations) layer on top in later phases.
//!
//! Phase A1 (merged): the election safety core as *pure logic* — no I/O, no
//! clocks, no tasks. Phase A1b (this revision): the canonical prefix log —
//! append continuity, conflict truncation, per-write quorum confirmation,
//! and the current-term commit rule — in the same pure state machine
//! ([`replica::ReplicaCore`]). A driver (production runtime or the
//! deterministic simulator in tests) feeds events in and executes the
//! returned effects; the pure shape is what makes Gate A provable.
//!
//! Gate A invariants covered (RFC 028 v2 §11):
//! 1. **Vote safety** (R2) — structural: granted votes only leave the node
//!    via [`replica::ReplicaCore::state_persisted`].
//! 2. **Authority safety** (R1) — per-write quorum confirmation: an entry
//!    commits only when a quorum has durably accepted it (acceptor acks are
//!    persist-gated), and only under the leader's current term; stale
//!    leaders are term-fenced at every acceptor. The sim proves it with a
//!    global committed-entry-uniqueness ledger.
//! 3. **Possibly-committed-suffix protection** (R3) — Raft last-`(term,
//!    index)` election freshness, never a watermark.

pub mod bootstrap;
pub mod driver;
pub mod engine_sink;
pub mod op;
pub mod replica;
pub mod runtime;
pub mod transport;
pub mod types;

#[cfg(test)]
mod sim;
