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
//! Phase A1 (this module): the **election safety core** as *pure logic* — no
//! I/O, no clocks, no tasks. A driver (production runtime or the deterministic
//! simulator in tests) feeds events in and executes the returned effects. The
//! pure shape is what makes Gate A provable: the simulator can crash a node at
//! any effect boundary and assert the invariants hold.
//!
//! Gate A invariants covered here (RFC 028 v2 §11):
//! 1. **Vote safety** — no node votes for two candidates in one term,
//!    including across crash/restart (review scenario R2).
//! 3. **Possibly-committed-suffix protection** — election freshness uses
//!    Raft's last-`(term, index)` rule, never a scalar watermark (R3).
//!
//! The structural trick for R2: [`election::ElectionCore`] *cannot* produce a
//! granted-vote response directly. Deciding to grant returns a
//! [`election::Effect::PersistHardState`]; only when the driver confirms
//! durability via [`election::ElectionCore::hard_state_persisted`] does the
//! response message materialize. Persist-before-respond is enforced by the
//! API's shape, not by driver discipline.

pub mod election;
pub mod types;

#[cfg(test)]
mod sim;
