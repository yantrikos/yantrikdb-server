//! Token generation and validation.
//!
//! Tokens are `ydb_<64 hex chars>` (32 bytes of randomness).
//! Stored as SHA-256 hashes in control.db.
//!
//! ## RFC 014-B Auth / RBAC substrate
//!
//! Submodules add the trait surface for the Auth/RBAC layer:
//! - [`scopes`] — typed scope enum + scope set (bitset).
//! - [`principal`] — [`principal::Principal`] + [`principal::AuthOutcome`].
//! - [`provider`] — [`provider::AuthProvider`] trait + [`provider::AuthError`].
//! - [`audit`] — [`audit::AuditEvent`] + [`audit::AuditSink`] trait.
//!
//! The `tower::Layer` middleware and the control-DB-backed
//! `AuthProvider` impl are NOT in this substrate slice — they're
//! deferred to a follow-up PR. Substrate composition is interface +
//! tests + reference impls (in-memory provider, in-memory audit sink)
//! so handlers and integration tests can plug them in immediately.

use rand::Rng;
use sha2::{Digest, Sha256};

pub mod admin;
pub mod audit;
pub mod control_provider;
pub mod middleware;
pub mod principal;
pub mod provider;
pub mod scopes;

pub use audit::{
    AuditEvent, AuditEventKind, AuditOutcome, AuditSink, InMemoryAuditSink, NoopAuditSink,
};
pub use control_provider::ControlDbAuthProvider;
pub use middleware::require_authenticated_principal;
pub use principal::{AuthOutcome, Principal};
pub use provider::{AuthError, AuthProvider, InMemoryAuthProvider};
pub use scopes::{Scope, ScopeSet};

/// Generate a new token: `ydb_<64 hex chars>`.
pub fn generate_token() -> String {
    let mut rng = rand::thread_rng();
    let mut bytes = [0u8; 32];
    rng.fill(&mut bytes);
    format!("ydb_{}", hex::encode(bytes))
}

/// Hash a token for storage (SHA-256).
pub fn hash_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_format() {
        let token = generate_token();
        assert!(token.starts_with("ydb_"));
        assert_eq!(token.len(), 4 + 64); // "ydb_" + 64 hex chars
    }

    #[test]
    fn hash_deterministic() {
        let token = "ydb_abc123";
        assert_eq!(hash_token(token), hash_token(token));
    }

    #[test]
    fn different_tokens_different_hashes() {
        let t1 = generate_token();
        let t2 = generate_token();
        assert_ne!(hash_token(&t1), hash_token(&t2));
    }
}
