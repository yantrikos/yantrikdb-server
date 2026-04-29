//! RFC 014-B — pluggable `AuthProvider` trait.
//!
//! The middleware (deferred) calls [`AuthProvider::authenticate`] on
//! each request, gets an [`AuthOutcome`], and either attaches the
//! `Principal` to request extensions or returns 401. The middleware
//! does NOT touch tokens directly — provider impls own that.
//!
//! Two provider impls live in the workspace:
//! - [`InMemoryAuthProvider`] (this file) — for tests and local dev.
//! - Control-DB-backed provider (deferred) — production. Hashes tokens
//!   with `super::hash_token`, looks up by hash in `control.tokens`
//!   table, returns the matching principal.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use parking_lot::RwLock;

use super::hash_token;
use super::principal::{AuthOutcome, Principal};
use super::scopes::ScopeSet;

/// Errors from the auth provider. These are *infrastructure* errors —
/// "control DB unreachable", not "token doesn't match". A failed
/// match returns `Ok(AuthOutcome::Unauthenticated)`, not `Err`.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("auth backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("auth backend error: {0}")]
    Backend(String),
}

/// Pluggable authenticator. Async because production impls hit a SQLite
/// pool (which is sync internally but sits behind an async wrapper to
/// keep the request handler non-blocking).
#[async_trait::async_trait]
pub trait AuthProvider: Send + Sync {
    /// Resolve a presented credential to an [`AuthOutcome`]. The
    /// credential is the raw bearer token string from the
    /// `Authorization: Bearer <tok>` header. Implementations MUST hash
    /// before comparison — never compare raw token strings.
    async fn authenticate(&self, presented: &str) -> Result<AuthOutcome, AuthError>;
}

/// Stored token record. Internal to in-memory provider; production
/// version lives in `control.tokens` table.
#[derive(Debug, Clone, PartialEq)]
struct TokenRecord {
    principal: Principal,
    /// Hash of the original token. Constructed via `super::hash_token`.
    hash: String,
    revoked: bool,
    /// `None` = no expiry. `Some(t)` = invalid after `t`.
    expires_at: Option<SystemTime>,
}

/// In-memory `AuthProvider` impl. Used by tests and as a reference
/// for the control-DB-backed production version.
///
/// Tokens are stored hashed (per RFC 014-B "Token hashing at rest").
/// Inserting a token returns the raw string ONCE — callers who lose
/// it must rotate, not look it up. This mirrors how the production
/// admin API will work.
#[derive(Default, Clone)]
pub struct InMemoryAuthProvider {
    inner: Arc<RwLock<InMemoryState>>,
}

#[derive(Default)]
struct InMemoryState {
    /// Indexed by hash for O(1) lookup. Realistic — production impl
    /// indexes on `tokens.hash`.
    by_hash: HashMap<String, TokenRecord>,
}

impl InMemoryAuthProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a token bound to the given principal and return the raw
    /// token string (the only chance to capture it). The principal is
    /// stored as-given.
    pub fn issue_token(&self, principal: Principal) -> String {
        let raw = super::generate_token();
        let hash = hash_token(&raw);
        let rec = TokenRecord {
            principal,
            hash: hash.clone(),
            revoked: false,
            expires_at: None,
        };
        self.inner.write().by_hash.insert(hash, rec);
        raw
    }

    /// Insert a token with an explicit expiry. Useful for tests of
    /// expiry semantics; production admin API calls this through a
    /// time picker.
    pub fn issue_token_with_expiry(
        &self,
        principal: Principal,
        expires_at: SystemTime,
    ) -> String {
        let raw = super::generate_token();
        let hash = hash_token(&raw);
        let rec = TokenRecord {
            principal,
            hash: hash.clone(),
            revoked: false,
            expires_at: Some(expires_at),
        };
        self.inner.write().by_hash.insert(hash, rec);
        raw
    }

    /// Mark a token revoked. Used by the admin "revoke this token" API.
    /// Returns `true` if the token existed (and is now revoked) or
    /// `false` if no token matches the hash.
    pub fn revoke(&self, raw: &str) -> bool {
        let hash = hash_token(raw);
        let mut guard = self.inner.write();
        match guard.by_hash.get_mut(&hash) {
            Some(rec) => {
                rec.revoked = true;
                true
            }
            None => false,
        }
    }

    /// Number of tokens stored (revoked or not). For dashboards / tests.
    pub fn token_count(&self) -> usize {
        self.inner.read().by_hash.len()
    }
}

#[async_trait::async_trait]
impl AuthProvider for InMemoryAuthProvider {
    async fn authenticate(&self, presented: &str) -> Result<AuthOutcome, AuthError> {
        let hash = hash_token(presented);
        let guard = self.inner.read();
        let Some(rec) = guard.by_hash.get(&hash) else {
            return Ok(AuthOutcome::Unauthenticated);
        };
        if rec.revoked {
            return Ok(AuthOutcome::Revoked {
                id: rec.principal.id.clone(),
            });
        }
        if let Some(exp) = rec.expires_at {
            if SystemTime::now() >= exp {
                return Ok(AuthOutcome::Expired {
                    id: rec.principal.id.clone(),
                });
            }
        }
        Ok(AuthOutcome::Authenticated(rec.principal.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::scopes::Scope;
    use super::*;
    use std::time::Duration;

    fn p(id: &str) -> Principal {
        Principal::new(id).with_scopes(ScopeSet::from_iter([Scope::Read]))
    }

    #[tokio::test]
    async fn authenticate_known_token_returns_principal() {
        let prov = InMemoryAuthProvider::new();
        let raw = prov.issue_token(p("alice"));
        let out = prov.authenticate(&raw).await.unwrap();
        match out {
            AuthOutcome::Authenticated(pp) => assert_eq!(pp.id, "alice"),
            other => panic!("expected Authenticated, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn authenticate_unknown_token_returns_unauthenticated() {
        let prov = InMemoryAuthProvider::new();
        prov.issue_token(p("alice"));
        let out = prov.authenticate("ydb_nonsense").await.unwrap();
        assert!(matches!(out, AuthOutcome::Unauthenticated));
    }

    #[tokio::test]
    async fn revoked_token_is_distinguished_from_unknown() {
        let prov = InMemoryAuthProvider::new();
        let raw = prov.issue_token(p("alice"));
        assert!(prov.revoke(&raw));
        let out = prov.authenticate(&raw).await.unwrap();
        assert!(matches!(out, AuthOutcome::Revoked { id } if id == "alice"));
    }

    #[tokio::test]
    async fn expired_token_returns_expired_outcome() {
        let prov = InMemoryAuthProvider::new();
        let past = SystemTime::now() - Duration::from_secs(60);
        let raw = prov.issue_token_with_expiry(p("alice"), past);
        let out = prov.authenticate(&raw).await.unwrap();
        assert!(matches!(out, AuthOutcome::Expired { id } if id == "alice"));
    }

    #[tokio::test]
    async fn future_expiry_still_authenticates() {
        let prov = InMemoryAuthProvider::new();
        let future = SystemTime::now() + Duration::from_secs(60);
        let raw = prov.issue_token_with_expiry(p("alice"), future);
        let out = prov.authenticate(&raw).await.unwrap();
        assert!(matches!(out, AuthOutcome::Authenticated(_)));
    }

    #[tokio::test]
    async fn revoke_unknown_token_returns_false() {
        let prov = InMemoryAuthProvider::new();
        assert!(!prov.revoke("ydb_nonsense"));
    }

    #[test]
    fn token_count_reflects_inserts() {
        let prov = InMemoryAuthProvider::new();
        assert_eq!(prov.token_count(), 0);
        prov.issue_token(p("a"));
        prov.issue_token(p("b"));
        assert_eq!(prov.token_count(), 2);
    }

    #[tokio::test]
    async fn raw_token_is_never_stored() {
        // Defensive: enforces "Token hashing at rest" via observable
        // contract. The only way to retrieve a token is via authentication
        // — there's no admin "show me the raw token" API. If somebody
        // refactors and leaks raw, this test still passes (we have no
        // raw-extraction API), but the more important guard is the
        // *non-existence* of any pub fn returning the raw string.
        let prov = InMemoryAuthProvider::new();
        let raw = prov.issue_token(p("alice"));
        // Authenticate against the issued raw — works.
        let ok = prov.authenticate(&raw).await.unwrap();
        assert!(matches!(ok, AuthOutcome::Authenticated(_)));
        // Authenticate against something else — fails.
        let no = prov.authenticate("ydb_garbage").await.unwrap();
        assert!(matches!(no, AuthOutcome::Unauthenticated));
    }

    #[tokio::test]
    async fn provider_is_dyn_dispatchable() {
        // Regression: trait must remain object-safe so middleware can
        // hold Arc<dyn AuthProvider>.
        let prov: Arc<dyn AuthProvider> = Arc::new(InMemoryAuthProvider::new());
        let _ = prov.authenticate("x").await;
    }
}
