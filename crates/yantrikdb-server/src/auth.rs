//! Token generation and validation.
//!
//! Tokens are `ydb_<64 hex chars>` (32 bytes of randomness).
//! Stored as SHA-256 hashes in control.db.

use rand::Rng;
use sha2::{Digest, Sha256};

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
