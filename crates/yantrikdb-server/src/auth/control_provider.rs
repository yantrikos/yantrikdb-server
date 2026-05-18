//! Control-DB-backed [`AuthProvider`] for production use.
//!
//! Bridges the legacy `control.tokens` table to the RFC 014-B
//! [`Principal`] / [`ScopeSet`] substrate. Tokens predate the typed-scope
//! model: every persisted token is implicitly "full tenant ops on its
//! database", with no per-token scope storage yet. Until per-token
//! scopes ship, this provider hands every authenticated token the
//! tenant-data-plane bundle (`Read | Write | Recall | Forget`), and
//! recognizes the cluster master secret as a cluster-wide admin
//! principal (all scopes, no tenant pin).
//!
//! Per-token scope storage is the natural next step; this file is the
//! seam where it plugs in.

use std::sync::Arc;

use parking_lot::Mutex;

use super::principal::{AuthOutcome, Principal};
use super::provider::{AuthError, AuthProvider};
use super::scopes::{Scope, ScopeSet};
use crate::control::ControlDb;

/// Production [`AuthProvider`]. Two paths:
///
/// 1. If the presented credential matches `cluster_secret`, return a
///    cluster-wide admin principal — no `tenant_id`, all six scopes.
/// 2. Else hash the credential, look it up in `control.tokens`, fetch
///    the bound database name, return a tenant-pinned principal with
///    the data-plane bundle (`Read | Write | Recall | Forget`).
///
/// Provider [`AuthError`] is reserved for infrastructure failures
/// (control DB unreachable). A missing or revoked token returns
/// `Ok(AuthOutcome::Unauthenticated)`, mirroring [`InMemoryAuthProvider`].
pub struct ControlDbAuthProvider {
    control: Arc<Mutex<ControlDb>>,
    cluster_secret: Option<String>,
}

impl ControlDbAuthProvider {
    pub fn new(control: Arc<Mutex<ControlDb>>, cluster_secret: Option<String>) -> Self {
        Self {
            control,
            cluster_secret,
        }
    }

    fn cluster_admin_principal() -> Principal {
        Principal::new("cluster-admin")
            .with_scopes(ScopeSet::all())
            .with_label("cluster-master")
    }

    fn tenant_data_plane_scopes() -> ScopeSet {
        ScopeSet::from_iter([Scope::Read, Scope::Write, Scope::Recall, Scope::Forget])
    }
}

#[async_trait::async_trait]
impl AuthProvider for ControlDbAuthProvider {
    async fn authenticate(&self, presented: &str) -> Result<AuthOutcome, AuthError> {
        if let Some(secret) = &self.cluster_secret {
            if presented == secret {
                return Ok(AuthOutcome::Authenticated(Self::cluster_admin_principal()));
            }
        }

        let hash = super::hash_token(presented);

        let lookup = {
            let ctrl = self.control.lock();
            let db_id = match ctrl.validate_token(&hash) {
                Ok(Some(id)) => id,
                Ok(None) => return Ok(AuthOutcome::Unauthenticated),
                Err(e) => return Err(AuthError::Backend(e.to_string())),
            };
            ctrl.get_database_by_id(db_id)
                .map_err(|e| AuthError::Backend(e.to_string()))?
        };

        let Some(db) = lookup else {
            // Token row exists but its database row is gone — treat as
            // unauth rather than 500. Clients should rotate.
            return Ok(AuthOutcome::Unauthenticated);
        };

        // Token id surfaces in audit logs. Use a non-reversible prefix
        // of the hash so the raw token never leaks via the id field.
        let token_id = format!("tok_{}", &hash[..16]);
        let principal = Principal::new(token_id)
            .with_tenant(db.name)
            .with_scopes(Self::tenant_data_plane_scopes());
        Ok(AuthOutcome::Authenticated(principal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::hash_token;

    fn make_provider(
        cluster_secret: Option<&str>,
    ) -> (Arc<Mutex<ControlDb>>, ControlDbAuthProvider) {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("control.sqlite");
        let ctrl = ControlDb::open(&path).unwrap();
        let ctrl = Arc::new(Mutex::new(ctrl));
        let provider =
            ControlDbAuthProvider::new(Arc::clone(&ctrl), cluster_secret.map(String::from));
        // tmp must outlive the provider in the test — leak it.
        Box::leak(Box::new(tmp));
        (ctrl, provider)
    }

    #[tokio::test]
    async fn unknown_token_returns_unauthenticated() {
        let (_ctrl, provider) = make_provider(None);
        let out = provider.authenticate("ydb_nothing").await.unwrap();
        assert!(matches!(out, AuthOutcome::Unauthenticated));
    }

    #[tokio::test]
    async fn cluster_secret_returns_admin_principal() {
        let (_ctrl, provider) = make_provider(Some("super-secret-xyz"));
        let out = provider.authenticate("super-secret-xyz").await.unwrap();
        match out {
            AuthOutcome::Authenticated(p) => {
                assert_eq!(p.tenant_id, None);
                assert!(p.has_scope(Scope::Admin));
                assert!(p.has_scope(Scope::TenantManagement));
                assert!(p.has_scope(Scope::Read));
            }
            other => panic!("expected Authenticated, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn valid_db_token_returns_tenant_pinned_principal() {
        let (ctrl, provider) = make_provider(None);
        let (db_id, raw) = {
            let c = ctrl.lock();
            let id = c.create_database("acme", "/dev/null").unwrap();
            let raw = "ydb_test_token_for_acme";
            c.create_token(&hash_token(raw), id, "test").unwrap();
            (id, raw.to_string())
        };
        let _ = db_id;
        let out = provider.authenticate(&raw).await.unwrap();
        match out {
            AuthOutcome::Authenticated(p) => {
                assert_eq!(p.tenant_id.as_deref(), Some("acme"));
                assert!(p.has_scope(Scope::Read));
                assert!(p.has_scope(Scope::Write));
                assert!(p.has_scope(Scope::Recall));
                assert!(p.has_scope(Scope::Forget));
                assert!(!p.has_scope(Scope::Admin));
                assert!(!p.has_scope(Scope::TenantManagement));
                assert!(p.id.starts_with("tok_"));
            }
            other => panic!("expected Authenticated, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn revoked_token_is_unauthenticated() {
        // The legacy tokens table only exposes "not revoked" through
        // `validate_token`. We treat revoked-or-unknown identically as
        // Unauthenticated, since the row can't be distinguished without
        // schema work. Future PR may surface AuthOutcome::Revoked.
        let (ctrl, provider) = make_provider(None);
        let raw = "ydb_will_be_revoked";
        {
            let c = ctrl.lock();
            let id = c.create_database("acme", "/dev/null").unwrap();
            c.create_token(&hash_token(raw), id, "test").unwrap();
            assert!(c.revoke_token(&hash_token(raw)).unwrap());
        }
        let out = provider.authenticate(raw).await.unwrap();
        assert!(matches!(out, AuthOutcome::Unauthenticated));
    }

    #[tokio::test]
    async fn token_id_does_not_leak_raw_token() {
        let (ctrl, provider) = make_provider(None);
        let raw = "ydb_some_long_secret_string_xyz";
        {
            let c = ctrl.lock();
            let id = c.create_database("acme", "/dev/null").unwrap();
            c.create_token(&hash_token(raw), id, "test").unwrap();
        }
        let out = provider.authenticate(raw).await.unwrap();
        let p = match out {
            AuthOutcome::Authenticated(p) => p,
            o => panic!("{:?}", o),
        };
        assert!(!p.id.contains(raw));
        assert!(!p.id.contains("secret"));
    }
}
