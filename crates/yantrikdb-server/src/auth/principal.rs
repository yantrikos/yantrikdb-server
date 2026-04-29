//! RFC 014-B — authenticated principal type.
//!
//! A `Principal` is the resolved identity for a request: who they are,
//! which tenant they're operating in, and what they can do (scopes).
//! The HTTP layer attaches a `Principal` to each authenticated request
//! via tower middleware (deferred to consumer PR).

use super::scopes::{Scope, ScopeSet};

/// Authenticated principal. Lifetime: per-request. Cheap to clone
/// because all fields are owned strings and a `ScopeSet` (Copy).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    /// Stable opaque id. UUID for tokens, "system" for built-in
    /// internal callers (replication, background jobs).
    pub id: String,
    /// Tenant the token is scoped to. `None` for cluster-wide tokens
    /// (admin / tenant-management) that can act across tenants.
    pub tenant_id: Option<String>,
    /// Granted scopes. Used by the scope-check guard.
    pub scopes: ScopeSet,
    /// Display label for audit logs (e.g. "alice@example.com" or
    /// "automation-token-prod-1"). Not load-bearing; UI/audit only.
    pub label: Option<String>,
}

impl Principal {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tenant_id: None,
            scopes: ScopeSet::empty(),
            label: None,
        }
    }

    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    pub fn with_scopes(mut self, scopes: ScopeSet) -> Self {
        self.scopes = scopes;
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// True iff this principal has all of `required` scopes.
    pub fn has_scopes(&self, required: ScopeSet) -> bool {
        self.scopes.covers(required)
    }

    /// Convenience: check a single scope.
    pub fn has_scope(&self, scope: Scope) -> bool {
        self.scopes.contains(scope)
    }

    /// Predicate for cross-tenant operations: principal is acting on a
    /// tenant other than its scoped tenant. Returns `false` for
    /// principals without a tenant_id (cluster-wide tokens — those can
    /// touch any tenant).
    pub fn is_cross_tenant(&self, target_tenant: &str) -> bool {
        match &self.tenant_id {
            Some(t) => t != target_tenant,
            None => false,
        }
    }
}

/// Result of authenticating a presented credential. The auth middleware
/// converts this to a 401/403 response or attaches the `Principal` to
/// the request extensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthOutcome {
    /// Credential resolved to a principal.
    Authenticated(Principal),
    /// Credential is invalid (unknown token, malformed). HTTP 401.
    Unauthenticated,
    /// Credential is valid but revoked. HTTP 401 with distinct message
    /// so operators can grep audit logs for revoked-token usage.
    Revoked { id: String },
    /// Credential is valid but expired. HTTP 401.
    Expired { id: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_principal_has_empty_scopes() {
        let p = Principal::new("p1");
        assert!(p.scopes.is_empty());
        assert!(p.tenant_id.is_none());
        assert!(p.label.is_none());
    }

    #[test]
    fn builder_chain() {
        let p = Principal::new("p1")
            .with_tenant("acme")
            .with_scopes(ScopeSet::from_iter([Scope::Read, Scope::Recall]))
            .with_label("alice@example.com");
        assert_eq!(p.id, "p1");
        assert_eq!(p.tenant_id.as_deref(), Some("acme"));
        assert!(p.has_scope(Scope::Read));
        assert!(p.has_scope(Scope::Recall));
        assert!(!p.has_scope(Scope::Write));
        assert_eq!(p.label.as_deref(), Some("alice@example.com"));
    }

    #[test]
    fn has_scopes_checks_all() {
        let p = Principal::new("p1").with_scopes(ScopeSet::from_iter([Scope::Read, Scope::Recall]));
        assert!(p.has_scopes(ScopeSet::from_iter([Scope::Read])));
        assert!(p.has_scopes(ScopeSet::from_iter([Scope::Read, Scope::Recall])));
        assert!(!p.has_scopes(ScopeSet::from_iter([Scope::Read, Scope::Forget])));
    }

    #[test]
    fn is_cross_tenant_for_scoped_token() {
        let p = Principal::new("p1").with_tenant("acme");
        assert!(p.is_cross_tenant("widget-co"));
        assert!(!p.is_cross_tenant("acme"));
    }

    #[test]
    fn is_cross_tenant_false_for_cluster_wide_token() {
        // Cluster-wide token (no tenant_id) is allowed to touch any tenant.
        let p = Principal::new("admin").with_scopes(ScopeSet::from_iter([Scope::Admin]));
        assert!(!p.is_cross_tenant("acme"));
        assert!(!p.is_cross_tenant("widget-co"));
    }

    #[test]
    fn auth_outcomes_distinguish_revoked_from_expired() {
        // Audit operators rely on the distinction — a revoked token
        // means an operator pulled it (look at audit), an expired
        // token means the client just needs to renew.
        let revoked = AuthOutcome::Revoked { id: "p1".into() };
        let expired = AuthOutcome::Expired { id: "p1".into() };
        assert_ne!(revoked, expired);
    }
}
