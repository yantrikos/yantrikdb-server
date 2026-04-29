//! RFC 014-C — local `KeyProvider` implementation.
//!
//! Two modes:
//! - **In-memory**: keys live in process memory. Lost on restart.
//!   Useful for tests and short-lived dev workflows.
//! - **File-backed** (deferred to consumer PR): persists key material
//!   to a directory tree under `<root>/<tenant_id>/<purpose>/v<version>`
//!   with `0600` permissions. Restart-safe.
//!
//! This module ships the in-memory mode + the trait surface for the
//! file mode (struct exists, file ops left for a follow-up consumer
//! PR with proper file-perm tests on Unix-only CI).

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use rand::RngCore;

use super::{KeyError, KeyHandle, KeyMaterial, KeyProvider, KeyPurpose};

/// Standard DEK size for ChaCha20-Poly1305 / AES-256 = 32 bytes.
const DEFAULT_KEY_LEN_BYTES: usize = 32;

/// Errors specific to the local provider. Re-exposed from `mod.rs`.
#[derive(Debug, thiserror::Error)]
pub enum LocalKeyProviderError {
    #[error("key material generation failed: {0}")]
    Random(String),
}

/// In-memory key store. Cheap to clone (Arc-wrapped state).
#[derive(Default, Clone)]
pub struct LocalKeyProvider {
    state: Arc<RwLock<State>>,
    key_len: usize,
}

#[derive(Default)]
struct State {
    /// Per (tenant_id, purpose) → ordered versions. Vec index 0 = v1.
    /// `None` slots represent destroyed versions; we keep the Vec
    /// length stable so version numbers remain monotonic across a
    /// rotate-then-destroy cycle.
    keys: HashMap<(String, KeyPurpose), Vec<Option<Vec<u8>>>>,
}

impl LocalKeyProvider {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(State::default())),
            key_len: DEFAULT_KEY_LEN_BYTES,
        }
    }

    /// Override the key length. Almost always you want the default
    /// 32 bytes; this exists for tests that prefer shorter material
    /// for clarity.
    pub fn with_key_len(mut self, len: usize) -> Self {
        self.key_len = len;
        self
    }

    fn generate_bytes(&self, tenant_id: &str) -> Result<Vec<u8>, KeyError> {
        if tenant_id.is_empty() {
            return Err(KeyError::InvalidArgument("tenant_id is empty".into()));
        }
        if self.key_len == 0 {
            return Err(KeyError::InvalidArgument("key_len is 0".into()));
        }
        let mut buf = vec![0u8; self.key_len];
        rand::thread_rng().fill_bytes(&mut buf);
        Ok(buf)
    }

    /// Number of stored (tenant, purpose) entries — counting destroyed
    /// ones. For diagnostics + tests.
    pub fn entry_count(&self) -> usize {
        self.state.read().keys.len()
    }
}

#[async_trait::async_trait]
impl KeyProvider for LocalKeyProvider {
    async fn get_key(&self, tenant_id: &str, purpose: KeyPurpose) -> Result<KeyMaterial, KeyError> {
        let guard = self.state.read();
        let key = (tenant_id.to_string(), purpose);
        let Some(versions) = guard.keys.get(&key) else {
            return Err(KeyError::NotFound {
                tenant_id: tenant_id.into(),
                purpose: purpose.as_str(),
            });
        };
        // The "current" key is the highest non-destroyed version.
        for (idx, slot) in versions.iter().enumerate().rev() {
            if let Some(bytes) = slot {
                let handle = KeyHandle {
                    tenant_id: tenant_id.into(),
                    purpose,
                    version: (idx + 1) as u32,
                };
                return Ok(KeyMaterial::new(handle, bytes.clone()));
            }
        }
        Err(KeyError::NotFound {
            tenant_id: tenant_id.into(),
            purpose: purpose.as_str(),
        })
    }

    async fn get_key_version(
        &self,
        tenant_id: &str,
        purpose: KeyPurpose,
        version: u32,
    ) -> Result<KeyMaterial, KeyError> {
        if version == 0 {
            return Err(KeyError::InvalidArgument(
                "version must be >= 1 (1-indexed)".into(),
            ));
        }
        let guard = self.state.read();
        let key = (tenant_id.to_string(), purpose);
        let Some(versions) = guard.keys.get(&key) else {
            return Err(KeyError::UnknownVersion {
                tenant_id: tenant_id.into(),
                purpose: purpose.as_str(),
                version,
            });
        };
        let idx = (version - 1) as usize;
        let bytes =
            versions
                .get(idx)
                .and_then(|v| v.as_ref())
                .ok_or_else(|| KeyError::UnknownVersion {
                    tenant_id: tenant_id.into(),
                    purpose: purpose.as_str(),
                    version,
                })?;
        let handle = KeyHandle {
            tenant_id: tenant_id.into(),
            purpose,
            version,
        };
        Ok(KeyMaterial::new(handle, bytes.clone()))
    }

    async fn rotate_key(
        &self,
        tenant_id: &str,
        purpose: KeyPurpose,
    ) -> Result<KeyHandle, KeyError> {
        let bytes = self.generate_bytes(tenant_id)?;
        let mut guard = self.state.write();
        let key = (tenant_id.to_string(), purpose);
        let entry = guard.keys.entry(key).or_default();
        entry.push(Some(bytes));
        let version = entry.len() as u32;
        Ok(KeyHandle {
            tenant_id: tenant_id.into(),
            purpose,
            version,
        })
    }

    async fn destroy(&self, tenant_id: &str, purpose: KeyPurpose) -> Result<bool, KeyError> {
        let mut guard = self.state.write();
        let key = (tenant_id.to_string(), purpose);
        let Some(versions) = guard.keys.get_mut(&key) else {
            return Ok(false);
        };
        let mut destroyed_any = false;
        for slot in versions.iter_mut() {
            if let Some(bytes) = slot.take() {
                // Zeroize before drop. Same caveat as KeyMaterial::Drop —
                // best-effort, not a forensics-grade guarantee.
                let mut owned = bytes;
                for b in owned.iter_mut() {
                    unsafe { std::ptr::write_volatile(b, 0u8) };
                }
                destroyed_any = true;
            }
        }
        Ok(destroyed_any)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rotate_then_get_returns_current_key() {
        let p = LocalKeyProvider::new();
        let h = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        assert_eq!(h.version, 1);
        let m = p
            .get_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        assert_eq!(m.version(), 1);
        assert_eq!(m.len(), DEFAULT_KEY_LEN_BYTES);
    }

    #[tokio::test]
    async fn get_before_rotate_returns_not_found() {
        let p = LocalKeyProvider::new();
        let err = p
            .get_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap_err();
        assert!(matches!(err, KeyError::NotFound { .. }));
    }

    #[tokio::test]
    async fn rotate_increments_version() {
        let p = LocalKeyProvider::new();
        let v1 = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let v2 = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        assert_eq!(v1.version, 1);
        assert_eq!(v2.version, 2);
        // Current key reflects the latest.
        let cur = p
            .get_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        assert_eq!(cur.version(), 2);
    }

    #[tokio::test]
    async fn old_versions_remain_after_rotate() {
        let p = LocalKeyProvider::new();
        let _v1 = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let _v2 = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let v1_again = p
            .get_key_version("t1", KeyPurpose::TenantDataEncryption, 1)
            .await
            .unwrap();
        assert_eq!(v1_again.version(), 1);
    }

    #[tokio::test]
    async fn rotated_keys_have_distinct_bytes() {
        // 32 random bytes colliding twice across two rotate calls is
        // ~10^-77; treating equal bytes as a fatal regression is safe.
        let p = LocalKeyProvider::new();
        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let m1 = p
            .get_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let bytes_v1 = m1.as_bytes().to_vec();
        drop(m1);

        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let m2 = p
            .get_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        assert_ne!(bytes_v1, m2.as_bytes());
    }

    #[tokio::test]
    async fn destroy_makes_get_return_not_found() {
        // RFC 011 crypto-shred contract: after destroy, key fetch fails.
        let p = LocalKeyProvider::new();
        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        assert!(p
            .get_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .is_ok());

        let destroyed = p
            .destroy("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        assert!(destroyed);

        let err = p
            .get_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap_err();
        assert!(matches!(err, KeyError::NotFound { .. }));
    }

    #[tokio::test]
    async fn destroy_invalidates_all_versions_not_just_current() {
        let p = LocalKeyProvider::new();
        // Rotate twice → v1 + v2.
        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let _ = p
            .destroy("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        // v1 must also be unrecoverable — that's the whole point of
        // crypto-shred. If destroy left v1 readable, an attacker who
        // could decrypt old ciphertext would still recover the data.
        let err1 = p
            .get_key_version("t1", KeyPurpose::TenantDataEncryption, 1)
            .await
            .unwrap_err();
        assert!(matches!(err1, KeyError::UnknownVersion { .. }));
        let err2 = p
            .get_key_version("t1", KeyPurpose::TenantDataEncryption, 2)
            .await
            .unwrap_err();
        assert!(matches!(err2, KeyError::UnknownVersion { .. }));
    }

    #[tokio::test]
    async fn destroy_idempotent_returns_false_on_second_call() {
        let p = LocalKeyProvider::new();
        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let first = p
            .destroy("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let second = p
            .destroy("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        assert!(first);
        assert!(!second); // already destroyed → no-op
    }

    #[tokio::test]
    async fn destroy_unknown_pair_returns_false() {
        let p = LocalKeyProvider::new();
        let res = p
            .destroy("never-seen", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        assert!(!res);
    }

    #[tokio::test]
    async fn different_purposes_have_distinct_keys() {
        let p = LocalKeyProvider::new();
        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let _ = p
            .rotate_key("t1", KeyPurpose::BackupBlobEncryption)
            .await
            .unwrap();

        let m_data = p
            .get_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let m_backup = p
            .get_key("t1", KeyPurpose::BackupBlobEncryption)
            .await
            .unwrap();
        assert_ne!(m_data.as_bytes(), m_backup.as_bytes());
    }

    #[tokio::test]
    async fn destroying_one_purpose_preserves_others() {
        let p = LocalKeyProvider::new();
        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let _ = p.rotate_key("t1", KeyPurpose::ClusterTls).await.unwrap();

        let _ = p
            .destroy("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        // ClusterTls survives — load-bearing for the multi-purpose model.
        // If destroy(TenantDataEncryption) accidentally took out the TLS
        // key, a tenant delete would knock the cluster offline.
        assert!(p.get_key("t1", KeyPurpose::ClusterTls).await.is_ok());
        assert!(p
            .get_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn per_tenant_isolation() {
        let p = LocalKeyProvider::new();
        let _ = p
            .rotate_key("alice", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let _ = p
            .rotate_key("bob", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();

        let _ = p
            .destroy("alice", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        // Bob's key untouched.
        assert!(p
            .get_key("bob", KeyPurpose::TenantDataEncryption)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn rotate_with_empty_tenant_id_rejected() {
        let p = LocalKeyProvider::new();
        let err = p
            .rotate_key("", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap_err();
        assert!(matches!(err, KeyError::InvalidArgument(_)));
    }

    #[tokio::test]
    async fn get_key_version_zero_rejected() {
        let p = LocalKeyProvider::new();
        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let err = p
            .get_key_version("t1", KeyPurpose::TenantDataEncryption, 0)
            .await
            .unwrap_err();
        assert!(matches!(err, KeyError::InvalidArgument(_)));
    }

    #[tokio::test]
    async fn get_key_version_unknown_returns_unknown_version() {
        let p = LocalKeyProvider::new();
        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let err = p
            .get_key_version("t1", KeyPurpose::TenantDataEncryption, 99)
            .await
            .unwrap_err();
        assert!(matches!(err, KeyError::UnknownVersion { .. }));
    }

    #[tokio::test]
    async fn provider_is_dyn_dispatchable() {
        // Trait must remain object-safe so RFC 011 PR-4 can hold
        // Arc<dyn KeyProvider>.
        let p: Arc<dyn KeyProvider> = Arc::new(LocalKeyProvider::new());
        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        assert!(p
            .get_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn entry_count_reflects_unique_pairs() {
        let p = LocalKeyProvider::new();
        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        let _ = p.rotate_key("t1", KeyPurpose::ClusterTls).await.unwrap();
        let _ = p
            .rotate_key("t2", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        assert_eq!(p.entry_count(), 3);
        // Rotate again — same pair → same entry, counter unchanged.
        let _ = p
            .rotate_key("t1", KeyPurpose::TenantDataEncryption)
            .await
            .unwrap();
        assert_eq!(p.entry_count(), 3);
    }
}
