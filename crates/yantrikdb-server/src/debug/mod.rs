//! RFC 010 PR-5 — Jepsen-friendly debug + fault-injection surface.
//!
//! ## Why this module exists
//!
//! Per gpt-5.5 brainstorm of RFC 010 (session 7c36ea8b): Jepsen tests
//! need fault injection points DESIGNED INTO the system, not bolted on
//! after. Specifically:
//!
//! 1. **History inspection** — Jepsen's linearizability checker needs to
//!    read the committed log (op_id, log_index, term, timestamps) for a
//!    tenant from any node. → [`history`] module.
//!
//! 2. **Fault injection** — Jepsen's nemesis framework needs admin
//!    endpoints to drop Raft messages, induce partitions, inject latency,
//!    corrupt log entries. → [`fault`] module + `FaultyNetwork` trait.
//!
//! 3. **Wire format stability** — log entries returned by /debug/history
//!    are at the v1.0 wire format (RFC 010 PR-3 conformance tests guard
//!    this). Jepsen scripts can deserialize against a stable contract.
//!
//! ## Endpoints
//!
//! | Method | Path | Purpose |
//! |---|---|---|
//! | `GET` | `/v1/debug/history/{tenant_id}?from=N&limit=K` | Read committed log entries |
//! | `POST` | `/v1/debug/fault/inject` | Add a fault to the registry |
//! | `GET` | `/v1/debug/fault` | List active faults |
//! | `POST` | `/v1/debug/fault/clear` | Remove all faults |
//! | `DELETE` | `/v1/debug/fault/{fault_id}` | Remove one fault |
//!
//! ## Authorization
//!
//! Debug endpoints are gated on the cluster master token in PR-5. Once
//! RFC 014-B (RBAC) lands, a dedicated `debug:fault-inject` scope replaces
//! the master-token check. PR-5's choice is intentionally restrictive:
//! debug endpoints are **destructive when used wrong**, so we err on the
//! side of "operator only" until we have proper RBAC.
//!
//! ## What lands later
//!
//! - **RFC 010 PR-4** (openraft): the cluster transport layer wraps its
//!   `Network` impl in `RegistryFaultyNetwork` so injected faults
//!   actually drop / delay / corrupt traffic.
//! - **RFC 016 PR-5** (Jepsen runner): packages a Clojure project that
//!   uses these endpoints as the nemesis interface.

pub mod fault;
pub mod history;

pub use fault::{
    FaultId, FaultKind, FaultRecord, FaultRegistry, FaultVerdict, FaultyNetwork, NoopFaultyNetwork,
    RegistryFaultyNetwork,
};
