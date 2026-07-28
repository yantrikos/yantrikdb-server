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
    /// RFC 030: create an admin account. `password_hash` is argon2id,
    /// computed by the leader before proposing (never re-hashed per node).
    CreateUser {
        username: String,
        password_hash: String,
        role: String,
        created_at: String,
    },
    /// RFC 030: change an admin account's role.
    SetUserRole { username: String, role: String },
    /// RFC 030: rotate an admin account's password (argon2id hash).
    SetUserPassword {
        username: String,
        password_hash: String,
    },
    /// RFC 030: soft-disable an admin account (revoke the operator).
    DisableUser {
        username: String,
        disabled_at: String,
    },
    /// RFC 030 (H3): seed/rotate the replicated admin session-signing key.
    /// `value` is base64 of 32 random bytes; `kid` identifies it in tokens.
    SetAdminSessionKey { kid: String, value: String },
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
            // User ops key on username + a per-shape discriminator. Role /
            // password / disable each get a distinct claim so back-to-back
            // changes to the same user do NOT dedupe against each other;
            // the password/role content is also folded in so a retried
            // identical change dedupes but a new value re-appends.
            ControlOp::CreateUser { username, .. } => {
                buf.extend_from_slice(b"ctl:createuser:");
                buf.extend_from_slice(username.as_bytes());
            }
            ControlOp::SetUserRole { username, role } => {
                buf.extend_from_slice(b"ctl:userrole:");
                buf.extend_from_slice(username.as_bytes());
                buf.push(b':');
                buf.extend_from_slice(role.as_bytes());
            }
            ControlOp::SetUserPassword {
                username,
                password_hash,
            } => {
                buf.extend_from_slice(b"ctl:userpw:");
                buf.extend_from_slice(username.as_bytes());
                buf.push(b':');
                buf.extend_from_slice(password_hash.as_bytes());
            }
            ControlOp::DisableUser { username, .. } => {
                buf.extend_from_slice(b"ctl:userdisable:");
                buf.extend_from_slice(username.as_bytes());
            }
            ControlOp::SetAdminSessionKey { kid, .. } => {
                buf.extend_from_slice(b"ctl:sesskey:");
                buf.extend_from_slice(kid.as_bytes());
            }
        }
        super::op::fnv1a64(&buf)
    }

    /// Audit descriptor `(action, target)` for the replicated audit_log
    /// (RFC 030 M1). Secrets/hashes are never placed in the target.
    pub fn audit_action(&self) -> (&'static str, String) {
        match self {
            ControlOp::CreateDatabase { name, db_id, .. } => {
                ("create_database", format!("{name} (#{db_id})"))
            }
            ControlOp::CreateToken { db_id, label, .. } => {
                ("mint_token", format!("db #{db_id} [{label}]"))
            }
            ControlOp::RevokeToken { .. } => ("revoke_token", String::new()),
            ControlOp::CreateUser { username, role, .. } => {
                ("create_user", format!("{username} ({role})"))
            }
            ControlOp::SetUserRole { username, role } => {
                ("set_user_role", format!("{username} -> {role}"))
            }
            ControlOp::SetUserPassword { username, .. } => ("set_user_password", username.clone()),
            ControlOp::DisableUser { username, .. } => ("disable_user", username.clone()),
            ControlOp::SetAdminSessionKey { kid, .. } => ("rotate_session_key", kid.clone()),
        }
    }
}

/// RFC 030 (M1): the replicated unit inside `Payload::Control` — a control op
/// plus the `actor` who initiated it, so `ControlApplySink` can write a
/// quorum-durable, tamper-evident audit row as it applies on every node.
///
/// **Back-compat:** [`decode`](Self::decode) accepts BOTH this envelope and a
/// bare pre-RFC-030 `ControlOp` (actor defaults to empty), so control entries
/// already in the log (RFC 029) still replay on an RFC-030 node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlEnvelope {
    #[serde(default)]
    pub actor: String,
    /// Leader-stamped RFC-3339 timestamp (F3) so the replicated audit row is
    /// byte-identical on every node. Empty for legacy bare-op entries.
    #[serde(default)]
    pub at: String,
    pub op: ControlOp,
}

impl ControlEnvelope {
    /// Build an envelope, stamping `at` with the leader's clock once (the
    /// caller is the request-terminating/proposing node).
    pub fn new(actor: impl Into<String>, op: ControlOp) -> Self {
        Self {
            actor: actor.into(),
            at: chrono_now_rfc3339(),
            op,
        }
    }

    pub fn encode(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|e| format!("encode ControlEnvelope: {e}"))
    }

    /// Envelope first; on failure fall back to a legacy bare `ControlOp`
    /// (actor unknown → ""). The two shapes are unambiguous: the envelope has
    /// an `op` field, a bare op is an externally-tagged variant.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        match serde_json::from_slice::<ControlEnvelope>(bytes) {
            Ok(env) => Ok(env),
            Err(_) => ControlOp::decode(bytes).map(|op| ControlEnvelope {
                actor: String::new(),
                at: String::new(),
                op,
            }),
        }
    }

    pub fn claim_key(&self) -> u64 {
        self.op.claim_key()
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

    /// Apply one control envelope at YRP `index`. Writes the replicated audit
    /// row (M1) keyed on the index (idempotent on replay), then applies the
    /// op. Idempotent overall (safe to replay).
    pub fn apply(&self, index: u64, env: &ControlEnvelope) -> Result<(), String> {
        let db = self.control.lock();
        // M1: durable, deterministic audit — same row on every node. The
        // timestamp is leader-stamped in the envelope (F3) so the row is
        // byte-identical across nodes; legacy entries (no `at`) fall back to
        // apply-time, which is acceptable since they carry no actor anyway.
        if !env.actor.is_empty() {
            let (action, target) = env.op.audit_action();
            let at = if env.at.is_empty() {
                chrono_now_rfc3339()
            } else {
                env.at.clone()
            };
            db.apply_audit(index, &env.actor, action, &target, &at)
                .map_err(|e| format!("control audit write at {index}: {e}"))?;
        }
        let op = &env.op;
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
            ControlOp::CreateUser {
                username,
                password_hash,
                role,
                created_at,
            } => db
                .apply_create_user(username, password_hash, role, created_at)
                .map(|_| ())
                .map_err(|e| format!("control apply CreateUser({username}): {e}")),
            ControlOp::SetUserRole { username, role } => db
                .apply_set_user_role(index, username, role)
                .map(|_| ())
                .map_err(|e| format!("control apply SetUserRole({username}): {e}")),
            ControlOp::SetUserPassword {
                username,
                password_hash,
            } => db
                .apply_set_user_password(index, username, password_hash)
                .map(|_| ())
                .map_err(|e| format!("control apply SetUserPassword({username}): {e}")),
            ControlOp::DisableUser {
                username,
                disabled_at,
            } => db
                .apply_disable_user(index, username, disabled_at)
                .map(|_| ())
                .map_err(|e| format!("control apply DisableUser({username}): {e}")),
            ControlOp::SetAdminSessionKey { kid, value } => db
                .apply_set_admin_session_key(kid, value)
                .map_err(|e| format!("control apply SetAdminSessionKey({kid}): {e}")),
        }
    }
}

/// RFC-3339 now, used for audit timestamps at apply. (Server code, not a
/// workflow script — `chrono` is available.)
fn chrono_now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339()
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
        let env = ControlEnvelope::new("tester", tok);
        sink.apply(10, &env).unwrap();
        sink.apply(10, &env).unwrap();
        assert_eq!(
            sink.control.lock().validate_token("deadbeef").unwrap(),
            Some(id)
        );
        // The audit row was written (M1), keyed on the index.
        assert_eq!(sink.control.lock().list_audit(10).unwrap().len(), 1);
    }
}
