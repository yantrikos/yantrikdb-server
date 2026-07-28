//! RFC 029 — control-plane replication grammar + apply.
//!
//! Control mutations (create database / mint token / revoke token) ride the
//! **same** YRP log as data ops, as [`super::replica::Payload::Control`]
//! entries, and are applied to every node's `control.db` by
//! [`ControlApplySink`]. The per-node `control.db` thus becomes the
//! *materialized state of the replicated control log*: a token minted on the
//! leader exists on every follower and survives failover — closing the #1
//! enterprise-grade blocker (RFC 029 §The gap).
//!
//! ## Determinism + idempotency
//!
//! - **`db_id` is leader-assigned** and carried in the op, so every node
//!   inserts the identical id (per-node AUTOINCREMENT would diverge). Mirrors
//!   [`crate::control::ControlDb::import_snapshot`]'s explicit-id insert.
//! - **Timestamps are leader-assigned** (RFC-3339 strings in the op) so every
//!   node stores the same `created_at`/`revoked_at` — never `datetime('now')`
//!   at apply, which would differ per node.
//! - Apply is **idempotent on the natural key** (`databases.id`/name,
//!   `tokens.hash`) via `INSERT OR IGNORE`, so crash-replay of an
//!   already-durable index is a no-op — no separate op-id table needed.
//! - **Verifier material only**: `CreateToken` carries the SHA-256 token
//!   *hash*, never the plaintext (RFC 029 Invariant 3).
//!
//! ## Fail-stop
//!
//! A control apply error is returned as `Err` from the sink, which the apply
//! worker treats as fail-stop — the node stops applying (data too) and an
//! operator intervenes, rather than serving possibly-stale authorization
//! (RFC 029 Invariant 2). Because control and data share one apply marker, a
//! node is either caught up on the whole log or not; there is no independent
//! "control lag" to reason about.

use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::control::ControlDb;

/// A replicated control-plane mutation. serde_json-encoded into
/// `Payload::Control` (serde_json, not bincode — the grammar is small and
/// human-auditable, matching the data-plane [`super::op::YrpOp`] choice).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlOp {
    /// Create a database/tenant. `db_id` is leader-assigned (module docs).
    CreateDatabase {
        db_id: i64,
        name: String,
        path: String,
        /// Serialized JSON config; `"{}"` when unset.
        config: String,
        /// Leader RFC-3339 timestamp, replicated so all nodes agree.
        created_at: String,
    },
    /// Register a token by its SHA-256 hash (never plaintext).
    CreateToken {
        db_id: i64,
        token_hash: String,
        label: String,
        created_at: String,
    },
    /// Revoke a token by hash.
    RevokeToken {
        token_hash: String,
        revoked_at: String,
    },
}

impl ControlOp {
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|e| format!("encode ControlOp: {e}"))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes).map_err(|e| format!("decode ControlOp: {e}"))
    }

    /// The protocol-level claim key for exactly-once proposal, derived from
    /// the op's natural identity. Prefixed (`ctl:*`) so it lives in a
    /// disjoint region of the same u64 claim space the data plane uses for
    /// idempotency keys — a retried `CreateToken`/`RevokeToken` dedupes
    /// against the original entry rather than double-applying.
    pub fn claim_key(&self) -> u64 {
        let mut buf = Vec::new();
        match self {
            ControlOp::CreateDatabase { db_id, name, .. } => {
                buf.extend_from_slice(b"ctl:createdb:");
                buf.extend_from_slice(&db_id.to_le_bytes());
                buf.extend_from_slice(name.as_bytes());
            }
            ControlOp::CreateToken { token_hash, .. } => {
                buf.extend_from_slice(b"ctl:createtok:");
                buf.extend_from_slice(token_hash.as_bytes());
            }
            ControlOp::RevokeToken { token_hash, .. } => {
                buf.extend_from_slice(b"ctl:revoketok:");
                buf.extend_from_slice(token_hash.as_bytes());
            }
        }
        super::op::fnv1a64(&buf)
    }
}

/// Applies replicated [`ControlOp`]s to the node's `control.db`. Held by the
/// data-plane apply pipeline ([`super::engine_sink::EngineApplySink`]) and
/// invoked on every committed `Payload::Control` entry, in log order, on
/// every node. A control apply error is surfaced as `Err` (fail-stop).
pub struct ControlApplySink {
    control: Arc<Mutex<ControlDb>>,
}

impl ControlApplySink {
    pub fn new(control: Arc<Mutex<ControlDb>>) -> Self {
        Self { control }
    }

    /// Apply one control op to `control.db`. Idempotent (safe to replay).
    pub fn apply(&self, op: &ControlOp) -> Result<(), String> {
        let db = self.control.lock();
        match op {
            ControlOp::CreateDatabase {
                db_id,
                name,
                path,
                config,
                created_at,
            } => db
                .apply_create_database(*db_id, name, path, config, created_at)
                .map(|_| ())
                .map_err(|e| format!("control apply CreateDatabase({name}): {e}")),
            ControlOp::CreateToken {
                db_id,
                token_hash,
                label,
                created_at,
            } => db
                .apply_create_token(token_hash, *db_id, label, created_at)
                .map(|_| ())
                .map_err(|e| format!("control apply CreateToken(db={db_id}): {e}")),
            ControlOp::RevokeToken {
                token_hash,
                revoked_at,
            } => db
                .apply_revoke_token(token_hash, revoked_at)
                .map(|_| ())
                .map_err(|e| format!("control apply RevokeToken: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_op_round_trips() {
        let op = ControlOp::CreateToken {
            db_id: 7,
            token_hash: "abc123".into(),
            label: "svc".into(),
            created_at: "2026-07-27T00:00:00Z".into(),
        };
        let bytes = op.encode().unwrap();
        assert_eq!(ControlOp::decode(&bytes).unwrap(), op);
    }

    #[test]
    fn claim_keys_are_distinct_per_identity() {
        let a = ControlOp::CreateToken {
            db_id: 1,
            token_hash: "h1".into(),
            label: String::new(),
            created_at: String::new(),
        };
        let b = ControlOp::RevokeToken {
            token_hash: "h1".into(),
            revoked_at: String::new(),
        };
        // Same hash, different op kind → different claim key (create vs
        // revoke must not dedupe against each other).
        assert_ne!(a.claim_key(), b.claim_key());
    }

    #[test]
    fn apply_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let db = ControlDb::open(&tmp.path().join("control.db")).unwrap();
        let id = db.next_database_id().unwrap();
        db.apply_create_database(id, "acme", "/dev/null", "{}", "2026-07-27T00:00:00Z")
            .unwrap();
        let sink = ControlApplySink::new(Arc::new(Mutex::new(db)));
        let tok = ControlOp::CreateToken {
            db_id: id,
            token_hash: "deadbeef".into(),
            label: "t".into(),
            created_at: "2026-07-27T00:00:00Z".into(),
        };
        // Apply twice — the second is a no-op, not an error.
        sink.apply(&tok).unwrap();
        sink.apply(&tok).unwrap();
        assert_eq!(
            sink.control.lock().validate_token("deadbeef").unwrap(),
            Some(id)
        );
    }
}
