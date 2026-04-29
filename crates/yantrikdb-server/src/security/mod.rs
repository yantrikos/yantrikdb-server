//! RFC 014-A — Security primitives.
//!
//! ## Why this module exists
//!
//! Security in YantrikDB is split into three layers, each landing in its
//! own RFC 014 sub-PR:
//!
//! 1. **014-A (this PR)**: cluster transport mTLS — the production gate
//!    for RFC 010 PR-4 openraft. Without valid mTLS certs configured,
//!    cluster mode in `cluster.mode = "openraft"` refuses to start.
//! 2. **014-B (later)**: API auth/RBAC + audit log
//! 3. **014-C (later)**: KeyProvider abstraction (Vault/KMS/HSM)
//!
//! ## What's in PR (014-A)
//!
//! - [`cluster_tls`] — `ClusterTlsConfig`, `build_acceptor`,
//!   `build_connector`, `CertificateRotator`
//! - [`cert_inspect`] — minimal cert-file verifier (loads + reports)
//!
//! ## Production gate
//!
//! When `cluster.mode = "openraft"` (RFC 010 PR-4 introduces this enum
//! variant), startup MUST find valid mTLS certs OR the operator MUST
//! explicitly set `cluster_tls.dev_mode = true` (which logs a loud
//! warning every minute). Default is "openraft requires real certs."
//!
//! ## Why mTLS specifically
//!
//! Cluster transport carries:
//! - Raft AppendEntries (the durable mutation log)
//! - Heartbeats (consensus liveness)
//! - Snapshot data (full state transfers)
//!
//! All three are existential to cluster correctness. A compromised peer
//! that can speak the cluster wire protocol can:
//! - Forge a leader (replicate fake mutations)
//! - Drop heartbeats to force unwanted elections
//! - Inject malformed snapshots
//!
//! Plain TLS (server-auth-only) wouldn't prevent any of those — a
//! malicious client with the right cluster_secret could still join.
//! mTLS requires every peer to present a cert signed by the cluster's
//! CA. Without a valid cert from the operator's PKI, the connection is
//! rejected at the TLS handshake — before any protocol bytes flow.

pub mod cert_inspect;
pub mod cluster_tls;

pub use cluster_tls::{
    build_acceptor, build_connector, CertificateRotator, ClusterTlsConfig, ClusterTlsError,
    ClusterTlsHandles,
};
