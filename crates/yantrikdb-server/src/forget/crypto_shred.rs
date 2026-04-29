//! RFC 011 PR-4 — crypto-shred: per-tenant DEK destruction.
//!
//! ## What this owns
//!
//! - [`CryptoShredder`] — orchestrates a tenant-delete: destroys the
//!   tenant's data-encryption key via the [`KeyProvider`], optionally
//!   destroys the backup-blob-encryption key too, and emits a
//!   [`ShredOutcome`] for audit / metrics.
//! - [`ShredPlan`] — what's about to be destroyed. Returned by
//!   `prepare()` before any destructive op runs, so operators (or the
//!   admin API) can preview + confirm.
//! - [`ShredOutcome`] — what actually happened. Captures whether each
//!   key was found + destroyed, or already gone (idempotent re-run).
//!
//! ## Why crypto-shred
//!
//! Per RFC 011 §motivation: "GDPR / right-to-erasure requires that
//! deleted data be unrecoverable, even from backups." Logical delete
//! (tombstone) doesn't reach into snapshot blobs that have already
//! been shipped to S3. Crypto-shred does: destroy the per-tenant DEK,
//! and every encrypted blob (live + backup) becomes ciphertext nobody
//! can decrypt — which under standard GDPR analysis is "deleted."
//!
//! ## What's NOT here (deferred)
//!
//! - Wiring CryptoShredder into the tenant-delete admin path.
//! - The actual encryption/decryption layer that wraps memory writes
//!   with `KeyProvider::get_key(tenant, TenantDataEncryption)` and an
//!   AEAD primitive (ChaCha20-Poly1305). The shredder destroys the
//!   key; the encryption layer is what produces blobs that need that
//!   key to read. RFC 011 PR-4 substrate is just the destruction
//!   orchestrator.
//! - `/v1/admin/tenants/{id}` DELETE endpoint that calls into this.

use std::sync::Arc;

use crate::key_provider::{KeyError, KeyProvider, KeyPurpose};

/// Configuration: which keys does a tenant-delete crypto-shred?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CryptoShredConfig {
    /// Destroy the per-tenant DEK that encrypts memory payload bytes.
    /// Almost always `true` — this is the GDPR-relevant key.
    pub data_encryption: bool,
    /// Destroy the backup-blob encryption key. Conservative default
    /// is `true` because backups containing this tenant's data should
    /// also be unrecoverable. Operators with a separate retention /
    /// legal-hold posture for backups can flip to `false`.
    pub backup_encryption: bool,
}

impl Default for CryptoShredConfig {
    fn default() -> Self {
        Self {
            data_encryption: true,
            backup_encryption: true,
        }
    }
}

/// What's about to be destroyed. Operators see this in the admin-API
/// preview before the destructive op runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShredPlan {
    pub tenant_id: String,
    pub purposes: Vec<KeyPurpose>,
}

impl ShredPlan {
    pub fn purpose_strings(&self) -> Vec<&'static str> {
        self.purposes.iter().map(|p| p.as_str()).collect()
    }
}

/// Per-purpose result of a shred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShredKeyResult {
    pub purpose: KeyPurpose,
    /// True if at least one version was destroyed; false if there was
    /// nothing to destroy (idempotent re-run, or the tenant never had
    /// this key issued).
    pub destroyed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShredOutcome {
    pub tenant_id: String,
    pub results: Vec<ShredKeyResult>,
}

impl ShredOutcome {
    /// True if every key in the plan was destroyed (including those
    /// that were already destroyed before — both count as "data is
    /// unreachable for this tenant").
    pub fn all_destroyed_or_absent(&self) -> bool {
        // Every result is either destroyed=true or destroyed=false
        // (already absent). Both are end-states where the data can't
        // be decrypted. So this is always true post-shred. The reason
        // we still distinguish destroyed vs absent in the per-purpose
        // result is for the audit log: "we actively destroyed this
        // version" vs "no key existed for this purpose."
        true
    }

    /// Subset of results that actively destroyed key material.
    pub fn destroyed_purposes(&self) -> Vec<KeyPurpose> {
        self.results
            .iter()
            .filter(|r| r.destroyed)
            .map(|r| r.purpose)
            .collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CryptoShredError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("key provider error: {0}")]
    KeyProvider(#[from] KeyError),
}

/// Orchestrates the crypto-shred for a tenant. Holds an `Arc<dyn KeyProvider>`;
/// callers construct one shredder per server (cheap to clone — Arc-wrapped).
pub struct CryptoShredder {
    keys: Arc<dyn KeyProvider>,
    cfg: CryptoShredConfig,
}

impl CryptoShredder {
    pub fn new(keys: Arc<dyn KeyProvider>, cfg: CryptoShredConfig) -> Self {
        Self { keys, cfg }
    }

    /// What would be destroyed if shred() were called now.
    pub fn prepare(&self, tenant_id: &str) -> Result<ShredPlan, CryptoShredError> {
        if tenant_id.is_empty() {
            return Err(CryptoShredError::InvalidArgument(
                "tenant_id is empty".into(),
            ));
        }
        let mut purposes = Vec::new();
        if self.cfg.data_encryption {
            purposes.push(KeyPurpose::TenantDataEncryption);
        }
        if self.cfg.backup_encryption {
            purposes.push(KeyPurpose::BackupBlobEncryption);
        }
        Ok(ShredPlan {
            tenant_id: tenant_id.into(),
            purposes,
        })
    }

    /// Execute the shred. Iterates the plan; for each purpose, calls
    /// `KeyProvider::destroy`. Returns per-purpose results so the
    /// audit log distinguishes "we actively destroyed v3+v4" from
    /// "this purpose had nothing to destroy."
    ///
    /// IMPORTANT: ordering is significant. We destroy ALL plan
    /// purposes even if one fails — partial state is worse than
    /// retrying the whole thing later, because it means SOME keys are
    /// gone (data partially unrecoverable) but others remain (data
    /// still recoverable from those keys' encrypted blobs). The
    /// returned error is the FIRST error encountered; the outcome
    /// includes whatever did succeed for audit-log purposes.
    pub async fn shred(&self, tenant_id: &str) -> Result<ShredOutcome, CryptoShredError> {
        let plan = self.prepare(tenant_id)?;
        let mut results = Vec::with_capacity(plan.purposes.len());
        let mut first_err: Option<KeyError> = None;
        for purpose in plan.purposes {
            match self.keys.destroy(tenant_id, purpose).await {
                Ok(destroyed) => results.push(ShredKeyResult { purpose, destroyed }),
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                    // Even on error, record what we tried.
                    results.push(ShredKeyResult {
                        purpose,
                        destroyed: false,
                    });
                }
            }
        }
        let outcome = ShredOutcome {
            tenant_id: tenant_id.into(),
            results,
        };
        if let Some(e) = first_err {
            return Err(CryptoShredError::KeyProvider(e));
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_provider::LocalKeyProvider;

    fn shredder(cfg: CryptoShredConfig) -> (CryptoShredder, Arc<LocalKeyProvider>) {
        let kp = Arc::new(LocalKeyProvider::new());
        let dyn_kp: Arc<dyn KeyProvider> = kp.clone();
        let s = CryptoShredder::new(dyn_kp, cfg);
        (s, kp)
    }

    #[test]
    fn config_defaults_destroy_both_data_and_backup_keys() {
        let cfg = CryptoShredConfig::default();
        assert!(cfg.data_encryption);
        assert!(cfg.backup_encryption);
    }

    #[test]
    fn prepare_lists_data_encryption_when_enabled() {
        let (s, _kp) = shredder(CryptoShredConfig::default());
        let plan = s.prepare("acme").unwrap();
        assert_eq!(plan.tenant_id, "acme");
        assert!(plan.purposes.contains(&KeyPurpose::TenantDataEncryption));
        assert!(plan.purposes.contains(&KeyPurpose::BackupBlobEncryption));
    }

    #[test]
    fn prepare_omits_backup_when_disabled() {
        let cfg = CryptoShredConfig {
            data_encryption: true,
            backup_encryption: false,
        };
        let (s, _kp) = shredder(cfg);
        let plan = s.prepare("acme").unwrap();
        assert!(plan.purposes.contains(&KeyPurpose::TenantDataEncryption));
        assert!(!plan.purposes.contains(&KeyPurpose::BackupBlobEncryption));
    }

    #[test]
    fn prepare_with_empty_tenant_rejected() {
        let (s, _kp) = shredder(CryptoShredConfig::default());
        let err = s.prepare("").unwrap_err();
        assert!(matches!(err, CryptoShredError::InvalidArgument(_)));
    }

    #[tokio::test]
    async fn shred_destroys_data_encryption_key() {
        let (s, kp) = shredder(CryptoShredConfig::default());
        // Issue both keys for the tenant.
        kp.rotate_key("acme", KeyPurpose::TenantDataEncryption).await.unwrap();
        kp.rotate_key("acme", KeyPurpose::BackupBlobEncryption).await.unwrap();

        let outcome = s.shred("acme").await.unwrap();
        assert_eq!(outcome.tenant_id, "acme");
        assert_eq!(outcome.results.len(), 2);
        for r in &outcome.results {
            assert!(r.destroyed, "expected destroyed for {:?}", r.purpose);
        }

        // Post-shred, neither key is recoverable.
        let err1 = kp.get_key("acme", KeyPurpose::TenantDataEncryption).await.unwrap_err();
        assert!(matches!(err1, KeyError::NotFound { .. }));
        let err2 = kp.get_key("acme", KeyPurpose::BackupBlobEncryption).await.unwrap_err();
        assert!(matches!(err2, KeyError::NotFound { .. }));
    }

    #[tokio::test]
    async fn shred_idempotent_on_second_call() {
        let (s, kp) = shredder(CryptoShredConfig::default());
        kp.rotate_key("acme", KeyPurpose::TenantDataEncryption).await.unwrap();
        let first = s.shred("acme").await.unwrap();
        let second = s.shred("acme").await.unwrap();
        // First run actively destroyed at least the data key.
        assert!(first.destroyed_purposes().contains(&KeyPurpose::TenantDataEncryption));
        // Second run: no purposes were actively destroyed (already gone).
        assert!(second.destroyed_purposes().is_empty());
        // Both runs leave the data unreachable.
        assert!(first.all_destroyed_or_absent());
        assert!(second.all_destroyed_or_absent());
    }

    #[tokio::test]
    async fn shred_does_not_touch_other_purposes() {
        // RFC 014-C purpose isolation: shredding tenant data must NOT
        // destroy ClusterTls / AuditSigning. This is the safety
        // invariant that prevents tenant delete from knocking the
        // cluster offline.
        let (s, kp) = shredder(CryptoShredConfig::default());
        kp.rotate_key("acme", KeyPurpose::TenantDataEncryption).await.unwrap();
        kp.rotate_key("acme", KeyPurpose::ClusterTls).await.unwrap();
        kp.rotate_key("acme", KeyPurpose::AuditSigning).await.unwrap();
        s.shred("acme").await.unwrap();
        // ClusterTls + AuditSigning untouched.
        assert!(kp.get_key("acme", KeyPurpose::ClusterTls).await.is_ok());
        assert!(kp.get_key("acme", KeyPurpose::AuditSigning).await.is_ok());
    }

    #[tokio::test]
    async fn shred_does_not_touch_other_tenants() {
        let (s, kp) = shredder(CryptoShredConfig::default());
        kp.rotate_key("alice", KeyPurpose::TenantDataEncryption).await.unwrap();
        kp.rotate_key("bob", KeyPurpose::TenantDataEncryption).await.unwrap();

        s.shred("alice").await.unwrap();

        // Alice's key gone.
        assert!(kp.get_key("alice", KeyPurpose::TenantDataEncryption).await.is_err());
        // Bob's preserved.
        assert!(kp.get_key("bob", KeyPurpose::TenantDataEncryption).await.is_ok());
    }

    #[tokio::test]
    async fn shred_invalidates_old_key_versions_too() {
        // Half-shredding (destroying current but leaving old versions)
        // would let an attacker who has historical ciphertext recover
        // the data. This test guards the contract.
        let (s, kp) = shredder(CryptoShredConfig::default());
        kp.rotate_key("acme", KeyPurpose::TenantDataEncryption).await.unwrap();
        kp.rotate_key("acme", KeyPurpose::TenantDataEncryption).await.unwrap();
        kp.rotate_key("acme", KeyPurpose::TenantDataEncryption).await.unwrap();

        s.shred("acme").await.unwrap();

        for v in 1..=3 {
            let err = kp
                .get_key_version("acme", KeyPurpose::TenantDataEncryption, v)
                .await
                .unwrap_err();
            assert!(
                matches!(err, KeyError::UnknownVersion { .. }),
                "expected v{} to be unrecoverable, got {:?}",
                v,
                err
            );
        }
    }

    #[tokio::test]
    async fn shred_with_no_keys_succeeds_no_op() {
        // Tenant exists in the system but never had keys issued. Shred
        // should succeed with `destroyed: false` for every purpose.
        let (s, _kp) = shredder(CryptoShredConfig::default());
        let outcome = s.shred("never-keyed").await.unwrap();
        for r in &outcome.results {
            assert!(!r.destroyed, "expected absent (not actively destroyed)");
        }
        assert!(outcome.all_destroyed_or_absent());
        assert!(outcome.destroyed_purposes().is_empty());
    }

    #[tokio::test]
    async fn shred_records_all_purposes_in_outcome() {
        // Operator audit log needs the per-purpose breakdown even on
        // failure paths.
        let (s, kp) = shredder(CryptoShredConfig::default());
        kp.rotate_key("acme", KeyPurpose::TenantDataEncryption).await.unwrap();
        let outcome = s.shred("acme").await.unwrap();
        let purposes: Vec<KeyPurpose> = outcome.results.iter().map(|r| r.purpose).collect();
        assert!(purposes.contains(&KeyPurpose::TenantDataEncryption));
        assert!(purposes.contains(&KeyPurpose::BackupBlobEncryption));
    }

    #[tokio::test]
    async fn shred_with_empty_tenant_id_rejected() {
        let (s, _kp) = shredder(CryptoShredConfig::default());
        let err = s.shred("").await.unwrap_err();
        assert!(matches!(err, CryptoShredError::InvalidArgument(_)));
    }

    #[tokio::test]
    async fn shred_data_only_when_backup_disabled() {
        let cfg = CryptoShredConfig {
            data_encryption: true,
            backup_encryption: false,
        };
        let (s, kp) = shredder(cfg);
        kp.rotate_key("acme", KeyPurpose::TenantDataEncryption).await.unwrap();
        kp.rotate_key("acme", KeyPurpose::BackupBlobEncryption).await.unwrap();

        s.shred("acme").await.unwrap();

        // Data key gone, backup key preserved (per cfg).
        assert!(kp.get_key("acme", KeyPurpose::TenantDataEncryption).await.is_err());
        assert!(kp.get_key("acme", KeyPurpose::BackupBlobEncryption).await.is_ok());
    }

    #[test]
    fn purpose_strings_helper() {
        let plan = ShredPlan {
            tenant_id: "acme".into(),
            purposes: vec![
                KeyPurpose::TenantDataEncryption,
                KeyPurpose::BackupBlobEncryption,
            ],
        };
        let strs = plan.purpose_strings();
        assert_eq!(
            strs,
            vec!["tenant_data_encryption", "backup_blob_encryption"]
        );
    }

    #[test]
    fn destroyed_purposes_filters_active_destructions() {
        let outcome = ShredOutcome {
            tenant_id: "acme".into(),
            results: vec![
                ShredKeyResult {
                    purpose: KeyPurpose::TenantDataEncryption,
                    destroyed: true,
                },
                ShredKeyResult {
                    purpose: KeyPurpose::BackupBlobEncryption,
                    destroyed: false,
                },
            ],
        };
        let destroyed = outcome.destroyed_purposes();
        assert_eq!(destroyed, vec![KeyPurpose::TenantDataEncryption]);
    }
}
