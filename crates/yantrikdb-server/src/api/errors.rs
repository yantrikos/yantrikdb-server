//! Structured HTTP error envelope (issue #39 brainstorm decision: Option B).
//!
//! Every error response across `/v1/*` carries this shape:
//!
//! ```json
//! {
//!   "error": {
//!     "code": "namespace_not_found",
//!     "message": "Namespace 'prod' not found.",
//!     "hint": "List visible namespaces via GET /v1/identity-scope."
//!   }
//! }
//! ```
//!
//! - `code` is a stable, machine-routable identifier from
//!   [`ApiErrorCode`]. Dashboards and clients should branch on `code`,
//!   not on `message` text.
//! - `message` is human-readable and may change wording between releases.
//! - `hint` is optional and actionable.
//!
//! See `docs/error-codes.md` for the registry.
//!
//! ## Migration note
//!
//! Pre-issue-#39, every endpoint emitted `{"error": "string"}` via
//! `http_gateway::app_error()`. That helper now emits this envelope
//! with [`ApiErrorCode::Generic`] for un-migrated call sites. Each site
//! moves to a specific code over time. This keeps the wire envelope
//! consistent across all endpoints — no mixed shapes — while letting
//! the code-naming migration happen incrementally.

use axum::{http::StatusCode, Json};
use serde_json::{json, Value};

/// Stable, machine-routable error codes.
///
/// The wire representation is the snake_case string from [`Self::as_str`].
/// Adding a variant is backwards-compatible. **Renaming an existing
/// variant's wire string breaks every consumer branching on
/// `error.code`** — treat the strings as a stable API surface.
///
/// New variants must be added to `docs/error-codes.md` in the same PR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiErrorCode {
    // ── Migration placeholder ─────────────────────────────────────────
    /// Placeholder for un-migrated `app_error()` call sites that haven't
    /// yet been replaced with a specific code. **New code MUST NOT emit
    /// this** — pick or add a specific variant.
    Generic,

    // ── 401 Unauthorized ──────────────────────────────────────────────
    /// Request did not present valid authentication credentials.
    Unauthenticated,

    // ── 403 Forbidden ─────────────────────────────────────────────────
    /// Authenticated but lacks permission for the requested action.
    Forbidden,
    /// Authenticated but the requested namespace is not visible to this
    /// token. Per the issue-#39 brainstorm policy: 403 (reveals
    /// existence), not 404 (indistinguishable). Pranab's call.
    NamespaceNotFound,
    /// Authenticated but the requested permission scope is not held.
    InsufficientScope,

    // ── 404 Not Found ─────────────────────────────────────────────────
    /// Memory with the given RID does not exist (or is not visible).
    MemoryNotFound,
    /// Conflict with the given ID does not exist.
    ConflictNotFound,
    /// Skill with the given ID does not exist.
    SkillNotFound,
    /// Session with the given ID does not exist.
    SessionNotFound,
    /// Entity with the given name does not exist.
    EntityNotFound,

    // ── 400 Bad Request / 422 Unprocessable ──────────────────────────
    /// Generic malformed request body or path.
    InvalidRequest,
    /// Query parameter failed validation (out-of-range, wrong type, etc.).
    InvalidQueryParameter,
    /// Pagination cursor is malformed or stale.
    InvalidCursor,
    /// `min_seq` parameter is malformed.
    InvalidMinSeq,
    /// Request body contained invalid JSON or didn't match the expected schema.
    InvalidBody,

    // ── 409 Conflict ──────────────────────────────────────────────────
    /// Op ID was reused with a different mutation (idempotency violation).
    OpIdCollision,
    /// Concurrent write race; client should re-read state and retry.
    UnexpectedLogIndex,

    // ── 412 Precondition Failed ───────────────────────────────────────
    /// Replica has not yet applied the requested `min_seq`. Retryable.
    ReplicaBehind,

    // ── 426 Upgrade Required ──────────────────────────────────────────
    /// Wire-version mismatch in a rolling upgrade.
    VersionUpgrade,

    // ── 429 Too Many Requests ─────────────────────────────────────────
    /// Request exceeded the per-principal or per-namespace rate limit.
    RateLimited,

    // ── 500 Internal Server Error ─────────────────────────────────────
    /// Catch-all for unexpected server failures. Include a `request_id`
    /// when emitted so operators can correlate logs.
    InternalError,

    // ── 307 Temporary Redirect ────────────────────────────────────────
    /// Request hit a follower that cannot serve it; client should redirect
    /// to the leader. Response body carries `leader_id`/`leader_addr`.
    NotLeader,

    // ── 503 Service Unavailable ───────────────────────────────────────
    /// Engine is currently unable to accept requests (e.g. mid-migration).
    EngineUnavailable,
    /// Cluster has no leader / no quorum.
    ClusterUnavailable,
    /// Leader is known but currently unreachable from this node.
    LeaderUnavailable,
    /// Write succeeded the commit log but didn't reach apply within the timeout.
    CommitTimeout,
}

impl ApiErrorCode {
    /// The stable snake_case wire string for this code.
    ///
    /// **DO NOT CHANGE existing return values.** Adding new variants is
    /// fine; renaming an existing variant's wire string breaks consumers.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Unauthenticated => "unauthenticated",
            Self::Forbidden => "forbidden",
            Self::NamespaceNotFound => "namespace_not_found",
            Self::InsufficientScope => "insufficient_scope",
            Self::MemoryNotFound => "memory_not_found",
            Self::ConflictNotFound => "conflict_not_found",
            Self::SkillNotFound => "skill_not_found",
            Self::SessionNotFound => "session_not_found",
            Self::EntityNotFound => "entity_not_found",
            Self::InvalidRequest => "invalid_request",
            Self::InvalidQueryParameter => "invalid_query_parameter",
            Self::InvalidCursor => "invalid_cursor",
            Self::InvalidMinSeq => "invalid_min_seq",
            Self::InvalidBody => "invalid_body",
            Self::OpIdCollision => "op_id_collision",
            Self::UnexpectedLogIndex => "unexpected_log_index",
            Self::ReplicaBehind => "replica_behind",
            Self::VersionUpgrade => "version_upgrade",
            Self::RateLimited => "rate_limited",
            Self::InternalError => "internal_error",
            Self::NotLeader => "not_leader",
            Self::EngineUnavailable => "engine_unavailable",
            Self::ClusterUnavailable => "cluster_unavailable",
            Self::LeaderUnavailable => "leader_unavailable",
            Self::CommitTimeout => "commit_timeout",
        }
    }
}

/// Construct a structured error response with the canonical envelope.
///
/// ```ignore
/// return Err(api_error(
///     StatusCode::NOT_FOUND,
///     ApiErrorCode::MemoryNotFound,
///     format!("memory {rid} not found"),
/// ));
/// ```
pub fn api_error(
    status: StatusCode,
    code: ApiErrorCode,
    message: impl Into<String>,
) -> (StatusCode, Json<Value>) {
    (
        status,
        Json(json!({
            "error": {
                "code": code.as_str(),
                "message": message.into(),
            }
        })),
    )
}

/// Same as [`api_error`] but with an actionable `hint`.
///
/// ```ignore
/// return Err(api_error_with_hint(
///     StatusCode::PRECONDITION_FAILED,
///     ApiErrorCode::ReplicaBehind,
///     "replica has not applied the requested min_seq",
///     "retry the request or route to the leader",
/// ));
/// ```
pub fn api_error_with_hint(
    status: StatusCode,
    code: ApiErrorCode,
    message: impl Into<String>,
    hint: impl Into<String>,
) -> (StatusCode, Json<Value>) {
    (
        status,
        Json(json!({
            "error": {
                "code": code.as_str(),
                "message": message.into(),
                "hint": hint.into(),
            }
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant must have a stable, snake_case wire string. This
    /// test is the gate against accidentally drifting a code rename.
    #[test]
    fn every_code_has_snake_case_wire_string() {
        let codes = [
            ApiErrorCode::Generic,
            ApiErrorCode::Unauthenticated,
            ApiErrorCode::Forbidden,
            ApiErrorCode::NamespaceNotFound,
            ApiErrorCode::InsufficientScope,
            ApiErrorCode::MemoryNotFound,
            ApiErrorCode::ConflictNotFound,
            ApiErrorCode::SkillNotFound,
            ApiErrorCode::SessionNotFound,
            ApiErrorCode::EntityNotFound,
            ApiErrorCode::InvalidRequest,
            ApiErrorCode::InvalidQueryParameter,
            ApiErrorCode::InvalidCursor,
            ApiErrorCode::InvalidMinSeq,
            ApiErrorCode::InvalidBody,
            ApiErrorCode::OpIdCollision,
            ApiErrorCode::UnexpectedLogIndex,
            ApiErrorCode::ReplicaBehind,
            ApiErrorCode::VersionUpgrade,
            ApiErrorCode::RateLimited,
            ApiErrorCode::InternalError,
            ApiErrorCode::NotLeader,
            ApiErrorCode::EngineUnavailable,
            ApiErrorCode::ClusterUnavailable,
            ApiErrorCode::LeaderUnavailable,
            ApiErrorCode::CommitTimeout,
        ];
        for code in codes {
            let s = code.as_str();
            assert!(!s.is_empty(), "code {code:?} has empty wire string");
            assert!(
                s.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                "code {code:?} wire string {s:?} not snake_case"
            );
            assert!(
                !s.starts_with('_') && !s.ends_with('_'),
                "code {code:?} wire string {s:?} has leading/trailing underscore"
            );
        }
    }

    /// Wire strings must be unique. Two variants with the same string
    /// would create silent collision in client routing.
    #[test]
    fn all_wire_strings_are_unique() {
        let codes = [
            ApiErrorCode::Generic,
            ApiErrorCode::Unauthenticated,
            ApiErrorCode::Forbidden,
            ApiErrorCode::NamespaceNotFound,
            ApiErrorCode::InsufficientScope,
            ApiErrorCode::MemoryNotFound,
            ApiErrorCode::ConflictNotFound,
            ApiErrorCode::SkillNotFound,
            ApiErrorCode::SessionNotFound,
            ApiErrorCode::EntityNotFound,
            ApiErrorCode::InvalidRequest,
            ApiErrorCode::InvalidQueryParameter,
            ApiErrorCode::InvalidCursor,
            ApiErrorCode::InvalidMinSeq,
            ApiErrorCode::InvalidBody,
            ApiErrorCode::OpIdCollision,
            ApiErrorCode::UnexpectedLogIndex,
            ApiErrorCode::ReplicaBehind,
            ApiErrorCode::VersionUpgrade,
            ApiErrorCode::RateLimited,
            ApiErrorCode::InternalError,
            ApiErrorCode::NotLeader,
            ApiErrorCode::EngineUnavailable,
            ApiErrorCode::ClusterUnavailable,
            ApiErrorCode::LeaderUnavailable,
            ApiErrorCode::CommitTimeout,
        ];
        let mut seen = std::collections::HashSet::new();
        for code in codes {
            let s = code.as_str();
            assert!(
                seen.insert(s),
                "duplicate wire string {s:?} for code {code:?}"
            );
        }
    }

    #[test]
    fn api_error_emits_canonical_envelope() {
        let (status, body) = api_error(
            StatusCode::NOT_FOUND,
            ApiErrorCode::MemoryNotFound,
            "memory abc123 not found",
        );
        assert_eq!(status, StatusCode::NOT_FOUND);
        let body = body.0;
        assert_eq!(body["error"]["code"], "memory_not_found");
        assert_eq!(body["error"]["message"], "memory abc123 not found");
        // No `hint` field when not provided.
        assert!(body["error"].get("hint").is_none());
    }

    #[test]
    fn api_error_with_hint_includes_hint() {
        let (status, body) = api_error_with_hint(
            StatusCode::PRECONDITION_FAILED,
            ApiErrorCode::ReplicaBehind,
            "replica has not applied seq 12345",
            "retry the request or route to the leader",
        );
        assert_eq!(status, StatusCode::PRECONDITION_FAILED);
        let body = body.0;
        assert_eq!(body["error"]["code"], "replica_behind");
        assert_eq!(
            body["error"]["hint"],
            "retry the request or route to the leader"
        );
    }
}
