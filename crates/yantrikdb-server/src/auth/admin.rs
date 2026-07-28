//! RFC 030 — admin accounts, roles, and stateless session tokens.
//!
//! This is the security core of the admin RBAC layer. It is deliberately
//! small and self-contained so it can be audited:
//!
//! - **Passwords** are argon2id (slow, salted) — [`hash_password`] /
//!   [`verify_password`]. Never SHA-256 (that is for high-entropy tokens).
//! - **Sessions** are stateless, signed tokens: `b64url(payload).b64url(sig)`
//!   where `sig = HMAC-SHA256(session_key, payload_b64_bytes)`. No `alg`
//!   header (sidesteps JWT alg-confusion); the algorithm is hardcoded.
//!   Verification is **verify-before-parse** and **constant-time** (M5).
//! - The signing key is the replicated `admin_session_key` (H3), decoupled
//!   from `cluster_secret`; the token carries its `kid` so rotation cleanly
//!   invalidates prior sessions.
//! - The payload carries `ver` (the user's `token_version`) so a
//!   disabled/demoted/rotated user's live sessions die immediately (H1) —
//!   the caller checks it against `control.db`.
//! - Unknown `role` strings **fail closed** (L5).

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;

type HmacSha256 = Hmac<Sha256>;

/// Admin role, total-ordered `readonly < admin < owner`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Readonly,
    Admin,
    Owner,
}

impl Role {
    fn rank(self) -> u8 {
        match self {
            Role::Readonly => 0,
            Role::Admin => 1,
            Role::Owner => 2,
        }
    }

    /// Parse a role string. Returns `None` for anything not in the known set
    /// — the guard treats `None` as a hard deny (L5: never default-compare an
    /// unknown role).
    pub fn parse(s: &str) -> Option<Role> {
        match s {
            "readonly" => Some(Role::Readonly),
            "admin" => Some(Role::Admin),
            "owner" => Some(Role::Owner),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Role::Readonly => "readonly",
            Role::Admin => "admin",
            Role::Owner => "owner",
        }
    }

    /// Does this role satisfy a minimum required role?
    pub fn satisfies(self, min: Role) -> bool {
        self.rank() >= min.rank()
    }
}

/// The authenticated admin actor for a request (for RBAC + audit).
#[derive(Debug, Clone)]
pub struct AdminActor {
    pub name: String,
    pub role: Role,
}

/// Session token payload (the signed part).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionClaims {
    /// Subject — the admin username.
    pub sub: String,
    /// Role at mint time (re-checked against `ver`).
    pub role: String,
    /// The user's `token_version` at mint time (H1 revocation).
    pub ver: i64,
    /// Signing-key id (H3 rotation).
    pub kid: String,
    /// Issued-at / expiry, unix seconds (leader/edge assigned).
    pub iat: i64,
    pub exp: i64,
}

/// Sign a session token. `key` is the raw `admin_session_key` bytes.
pub fn sign_session(key: &[u8], claims: &SessionClaims) -> Result<String, String> {
    let payload = serde_json::to_vec(claims).map_err(|e| format!("encode session claims: {e}"))?;
    let payload_b64 = URL_SAFE_NO_PAD.encode(&payload);
    let mut mac =
        HmacSha256::new_from_slice(key).map_err(|_| "bad session key length".to_string())?;
    mac.update(payload_b64.as_bytes());
    let sig = mac.finalize().into_bytes();
    Ok(format!("{payload_b64}.{}", URL_SAFE_NO_PAD.encode(sig)))
}

/// Why a session token was rejected (all map to 401 at the HTTP layer).
#[derive(Debug, PartialEq, Eq)]
pub enum SessionError {
    Malformed,
    BadSignature,
    Expired,
    BadPayload,
}

/// Verify a session token against a signing key, constant-time, verifying the
/// signature over the exact received payload bytes BEFORE parsing JSON (M5).
/// Checks `exp` against `now` (unix seconds) with a small skew tolerance.
/// Does NOT check `ver`/role — the caller does that against `control.db`.
pub fn verify_session(
    key: &[u8],
    token: &str,
    now: i64,
    skew_secs: i64,
) -> Result<SessionClaims, SessionError> {
    let (payload_b64, sig_b64) = token.split_once('.').ok_or(SessionError::Malformed)?;
    if payload_b64.is_empty() || sig_b64.is_empty() {
        return Err(SessionError::Malformed);
    }
    let sig = URL_SAFE_NO_PAD
        .decode(sig_b64)
        .map_err(|_| SessionError::Malformed)?;
    // Constant-time verification over the raw received payload bytes.
    let mut mac = HmacSha256::new_from_slice(key).map_err(|_| SessionError::BadSignature)?;
    mac.update(payload_b64.as_bytes());
    mac.verify_slice(&sig)
        .map_err(|_| SessionError::BadSignature)?;
    // Signature is valid — now it is safe to parse.
    let payload = URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|_| SessionError::BadPayload)?;
    let claims: SessionClaims =
        serde_json::from_slice(&payload).map_err(|_| SessionError::BadPayload)?;
    if now > claims.exp + skew_secs {
        return Err(SessionError::Expired);
    }
    Ok(claims)
}

// ── Passwords (argon2id) ────────────────────────────────────────────

/// Hash a password with argon2id (random salt). Returns the PHC string
/// stored/replicated as `password_hash`. Runs at the request-terminating
/// node; plaintext never enters a control op (M3).
pub fn hash_password(password: &str) -> Result<String, String> {
    use argon2::password_hash::{rand_core::OsRng, PasswordHasher, SaltString};
    use argon2::Argon2;
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| format!("argon2 hash: {e}"))
}

/// Verify a password against a stored argon2 PHC hash (constant-time inside
/// argon2). Returns false on any parse/verify failure (fail closed).
pub fn verify_password(password: &str, phc: &str) -> bool {
    use argon2::password_hash::{PasswordHash, PasswordVerifier};
    use argon2::Argon2;
    match PasswordHash::new(phc) {
        Ok(parsed) => Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .is_ok(),
        Err(_) => false,
    }
}

/// A fixed decoy hash for unknown-user logins (M4 anti-enumeration): verify
/// against it so an unknown username costs the same argon2 time as a known
/// one and reveals nothing. Generated once per process (value irrelevant).
pub fn decoy_hash() -> &'static str {
    // A precomputed argon2id hash of a random string. Constant across the
    // process; only the *timing* matters, never a match.
    "$argon2id$v=19$m=19456,t=2,p=1$Y2xhdWRlZGVjb3lzYWx0$V7m1oq9pO7m3n5r8t0uWx2z4A6C8E0G2I4K6M8O0Q2S"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_ordering_and_parse() {
        assert!(Role::Owner.satisfies(Role::Admin));
        assert!(Role::Admin.satisfies(Role::Readonly));
        assert!(!Role::Readonly.satisfies(Role::Admin));
        assert_eq!(Role::parse("owner"), Some(Role::Owner));
        assert_eq!(Role::parse("root"), None); // unknown → fail closed
        assert_eq!(Role::parse(""), None);
    }

    #[test]
    fn password_roundtrip() {
        let h = hash_password("s3cret-pw").unwrap();
        assert!(verify_password("s3cret-pw", &h));
        assert!(!verify_password("wrong", &h));
        assert!(!verify_password("s3cret-pw", "not-a-hash"));
    }

    #[test]
    fn session_sign_verify_and_tamper() {
        let key = b"0123456789abcdef0123456789abcdef";
        let claims = SessionClaims {
            sub: "alice".into(),
            role: "owner".into(),
            ver: 3,
            kid: "k1".into(),
            iat: 1000,
            exp: 5000,
        };
        let tok = sign_session(key, &claims).unwrap();
        let got = verify_session(key, &tok, 2000, 30).unwrap();
        assert_eq!(got.sub, "alice");
        assert_eq!(got.ver, 3);
        // Expired.
        assert_eq!(
            verify_session(key, &tok, 6000, 30),
            Err(SessionError::Expired)
        );
        // Wrong key.
        assert_eq!(
            verify_session(b"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", &tok, 2000, 30),
            Err(SessionError::BadSignature)
        );
        // Tampered payload (flip a char in the payload segment).
        let mut parts: Vec<&str> = tok.split('.').collect();
        let mangled_payload = format!("{}A", parts[0]);
        parts[0] = &mangled_payload;
        let tampered = format!("{}.{}", parts[0], parts[1]);
        assert!(verify_session(key, &tampered, 2000, 30).is_err());
        // Malformed.
        assert_eq!(
            verify_session(key, "no-dot", 2000, 30),
            Err(SessionError::Malformed)
        );
    }
}
