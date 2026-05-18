//! Scope + namespace-access guards for the `/v1/*` HTTP layer.
//!
//! Established by issue #39: every read endpoint must (1) require a
//! specific scope from the token's `Principal`, and (2) resolve the
//! requested namespace against the token's authorized scope —
//! **query params can narrow, never broaden**.
//!
//! ## Why this lives here, not in `auth::scopes`
//!
//! The `auth::` module owns the token authentication model
//! (`Principal`, `ScopeSet`, `AuthProvider`). This module owns the
//! HTTP-layer policy on top of it: how a query parameter narrows a
//! token's authorized namespace, what error code surfaces on mismatch,
//! what status code goes back. Keeping those concerns in `api::` keeps
//! the auth core implementation-agnostic.
//!
//! ## Existing scope convention
//!
//! Per `auth::scopes` (RFC 014-B), scopes are typed: `Read`, `Write`,
//! `Recall`, `Forget`, `Admin`, `TenantManagement`. Phase 1's three
//! read endpoints all require `Scope::Read`. Don't introduce
//! finer-grain variants speculatively; if Phase 3's export endpoint
//! needs distinct authorization, add `Scope::Export` then.
//!
//! ## Existing namespace convention
//!
//! `Principal::tenant_id: Option<String>` is the token's authorized
//! namespace (the names "tenant" and "namespace" are aliased in this
//! codebase — the `tenant_id` field is what the dashboard fixture
//! calls `namespace`).
//!
//! - `None` ⇒ the token is cluster-wide (Admin / TenantManagement
//!   tokens). It may pick any namespace via the query parameter, or
//!   omit it.
//! - `Some(ns)` ⇒ the token is pinned to `ns`. Query parameter must
//!   either match `ns` exactly, or be omitted (in which case `ns` is
//!   the effective namespace). Any other value returns 403
//!   `namespace_not_found` per Pranab's policy.

use axum::http::StatusCode;

use crate::api::errors::{api_error, ApiErrorCode};
use crate::auth::principal::Principal;
use crate::auth::scopes::{Scope, ScopeSet};

/// The HTTP error tuple returned by these guards. Identical shape to
/// `http_gateway::AppError` — kept as a local alias so this module
/// doesn't depend on http_gateway internals.
type GuardError = (StatusCode, axum::Json<serde_json::Value>);

/// Guard: the principal must hold the given scope.
///
/// Returns `Err(403, error="insufficient_scope")` if the scope is
/// missing.
///
/// ```ignore
/// use crate::api::access::require_scope;
/// use crate::auth::scopes::Scope;
///
/// require_scope(&principal, Scope::Read)?;
/// ```
pub fn require_scope(principal: &Principal, scope: Scope) -> Result<(), GuardError> {
    if principal.has_scopes(ScopeSet::from_iter([scope])) {
        Ok(())
    } else {
        Err(api_error(
            StatusCode::FORBIDDEN,
            ApiErrorCode::InsufficientScope,
            format!("token does not hold required scope: {}", scope.as_str()),
        ))
    }
}

/// Guard: resolve the effective namespace for this request.
///
/// Policy:
///
/// - **Cluster-wide token** (`tenant_id == None`):
///   - If query param specified, return it (any namespace allowed).
///   - If query param omitted, return `Err(400, "namespace required")`.
///     Cluster-wide tokens MUST specify a target namespace explicitly
///     — defaulting to "default" would silently scope admin reads to
///     one namespace when the operator likely wanted to enumerate all.
///
/// - **Pinned token** (`tenant_id == Some(ns)`):
///   - If query param specified AND matches `ns`, return `ns`.
///   - If query param specified AND differs from `ns`, return
///     `Err(403, "namespace_not_found")` per Pranab's policy.
///   - If query param omitted, return `ns`.
///
/// **Query params narrow, never broaden.** A token pinned to "default"
/// cannot read from "private" by passing `?namespace=private`.
///
/// ```ignore
/// let ns = resolve_namespace(&principal, params.namespace.as_deref())?;
/// // Use `ns` to scope the engine query.
/// ```
pub fn resolve_namespace(
    principal: &Principal,
    query_namespace: Option<&str>,
) -> Result<String, GuardError> {
    match (&principal.tenant_id, query_namespace) {
        // Cluster-wide token. Operator must specify namespace explicitly.
        (None, Some(q)) => Ok(q.to_string()),
        (None, None) => Err(api_error(
            StatusCode::BAD_REQUEST,
            ApiErrorCode::InvalidQueryParameter,
            "namespace is required for cluster-wide tokens",
        )),
        // Pinned token. Query param must match or be omitted.
        (Some(ns), None) => Ok(ns.clone()),
        (Some(ns), Some(q)) if q == ns => Ok(ns.clone()),
        (Some(_ns), Some(q)) => Err(api_error(
            StatusCode::FORBIDDEN,
            ApiErrorCode::NamespaceNotFound,
            format!("namespace not found: {q}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::principal::Principal;
    use crate::auth::scopes::{Scope, ScopeSet};

    fn principal_with(scopes: impl IntoIterator<Item = Scope>, tenant: Option<&str>) -> Principal {
        let mut p = Principal::new("test-token").with_scopes(ScopeSet::from_iter(scopes));
        if let Some(t) = tenant {
            p = p.with_tenant(t);
        }
        p
    }

    fn body_field<'a>(err: &'a GuardError, field: &str) -> &'a serde_json::Value {
        &err.1 .0["error"][field]
    }

    // ── require_scope ────────────────────────────────────────────────

    #[test]
    fn require_scope_passes_when_principal_has_scope() {
        let p = principal_with([Scope::Read], None);
        assert!(require_scope(&p, Scope::Read).is_ok());
    }

    #[test]
    fn require_scope_fails_403_when_missing() {
        let p = principal_with([Scope::Read], None);
        let err = require_scope(&p, Scope::Admin).expect_err("must reject");
        assert_eq!(err.0, StatusCode::FORBIDDEN);
        assert_eq!(body_field(&err, "code"), "insufficient_scope");
    }

    #[test]
    fn require_scope_fails_for_empty_principal() {
        let p = principal_with([], None);
        let err = require_scope(&p, Scope::Read).expect_err("must reject");
        assert_eq!(err.0, StatusCode::FORBIDDEN);
    }

    // ── resolve_namespace: pinned token ──────────────────────────────

    #[test]
    fn resolve_namespace_returns_tenant_when_no_query_param() {
        let p = principal_with([Scope::Read], Some("default"));
        assert_eq!(resolve_namespace(&p, None).unwrap(), "default");
    }

    #[test]
    fn resolve_namespace_accepts_matching_query_param() {
        let p = principal_with([Scope::Read], Some("default"));
        assert_eq!(resolve_namespace(&p, Some("default")).unwrap(), "default");
    }

    #[test]
    fn resolve_namespace_rejects_403_when_query_differs() {
        let p = principal_with([Scope::Read], Some("default"));
        let err = resolve_namespace(&p, Some("private")).expect_err("must reject");
        assert_eq!(err.0, StatusCode::FORBIDDEN);
        assert_eq!(body_field(&err, "code"), "namespace_not_found");
    }

    // ── resolve_namespace: cluster-wide token ────────────────────────

    #[test]
    fn resolve_namespace_cluster_wide_with_query_returns_query() {
        let p = principal_with([Scope::Admin], None);
        assert_eq!(resolve_namespace(&p, Some("anything")).unwrap(), "anything");
        assert_eq!(resolve_namespace(&p, Some("private")).unwrap(), "private");
    }

    #[test]
    fn resolve_namespace_cluster_wide_without_query_requires_explicit() {
        let p = principal_with([Scope::Admin], None);
        let err = resolve_namespace(&p, None).expect_err("must reject");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert_eq!(body_field(&err, "code"), "invalid_query_parameter");
    }

    // ── attack surface ───────────────────────────────────────────────

    #[test]
    fn resolve_namespace_pinned_cannot_broaden_via_admin_query() {
        // Even if the request asks for "admin" or "default", a pinned
        // token only ever returns its pinned namespace. Query cannot
        // broaden.
        let p = principal_with([Scope::Read], Some("user_abc"));
        // Asking for the admin namespace is forbidden.
        let err = resolve_namespace(&p, Some("admin")).expect_err("must reject");
        assert_eq!(err.0, StatusCode::FORBIDDEN);
        assert_eq!(body_field(&err, "code"), "namespace_not_found");

        // Asking for the pinned name is allowed.
        assert_eq!(resolve_namespace(&p, Some("user_abc")).unwrap(), "user_abc");
    }
}
