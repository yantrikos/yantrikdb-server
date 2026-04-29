//! RFC 021 PR-2 — per-tenant config overrides.
//!
//! ## What this owns
//!
//! - [`TenantConfigOverride`] — one tenant's per-key override map.
//!   Each override carries a [`super::ConfigVersion`] so replay,
//!   rollback, and divergence detection follow the same rules as
//!   global config (PR-1).
//! - [`TenantConfigStore`] trait + in-memory reference impl.
//! - [`OverrideValue`] — the typed leaf. Strings, ints, floats, bools,
//!   and durations cover the configurable surface; structured values
//!   (lists / maps) are intentionally excluded — overrides at this
//!   layer should be scalar tunables, not config replacement.
//! - [`resolve`] — "what's the effective value for (tenant, key)?":
//!   tenant override > None (caller falls back to global).
//!
//! ## What's NOT here (deferred)
//!
//! - The `MemoryMutation::SetTenantConfig` commit-log variant.
//! - The CLI: `yantrikdb config set --tenant X --key K=V`.
//! - The /v1/admin/tenants/{id}/config admin API.
//! - The actual integration where the admission middleware reads
//!   `resolve(tenant, "admission.max_concurrent_expanded_recall")`
//!   before each request — that lives in the admission consumer PR.
//!
//! ## Key naming
//!
//! Keys are dotted paths matching the global config struct shape:
//! `admission.max_concurrent_expanded_recall`,
//! `admission.cost_budget_per_sec`, etc. We don't enforce that
//! syntactically here — operators can set arbitrary keys, and the
//! caller-side resolver decides which ones to honor. This keeps the
//! substrate flexible (forward-compat: tomorrow's tunable doesn't
//! need a substrate change) while pushing the "is this key valid"
//! check to the consumer (which knows its own valid keys).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::versioned::ConfigVersion;

/// One scalar override leaf. Coverage chosen to match the runtime
/// tunables we know about today; new variants are additive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OverrideValue {
    String(String),
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    Duration(Duration),
}

impl OverrideValue {
    /// Stable type tag used in audit logs + admin API output. Pinned
    /// in tests.
    pub const fn type_str(&self) -> &'static str {
        match self {
            OverrideValue::String(_) => "string",
            OverrideValue::Int(_) => "int",
            OverrideValue::UInt(_) => "uint",
            OverrideValue::Float(_) => "float",
            OverrideValue::Bool(_) => "bool",
            OverrideValue::Duration(_) => "duration",
        }
    }
}

/// Per-tenant override bundle. Carries a version so the same
/// monotonic-replay rules from PR-1 apply at tenant scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TenantConfigOverride {
    pub tenant_id: String,
    pub version: ConfigVersion,
    /// Key → value. BTreeMap so iteration order is deterministic
    /// (matters for audit log diffs and dashboard rendering).
    pub overrides: BTreeMap<String, OverrideValue>,
}

impl TenantConfigOverride {
    pub fn new(tenant_id: impl Into<String>, version: ConfigVersion) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            version,
            overrides: BTreeMap::new(),
        }
    }

    pub fn set(&mut self, key: impl Into<String>, value: OverrideValue) {
        self.overrides.insert(key.into(), value);
    }

    pub fn unset(&mut self, key: &str) -> Option<OverrideValue> {
        self.overrides.remove(key)
    }

    pub fn get(&self, key: &str) -> Option<&OverrideValue> {
        self.overrides.get(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.overrides.keys()
    }

    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }

    pub fn len(&self) -> usize {
        self.overrides.len()
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum TenantConfigError {
    #[error("tenant `{0}` has no overrides")]
    NotFound(String),
    #[error(
        "stale version for tenant `{tenant}`: incoming `{incoming}`, observed `{observed}`"
    )]
    StaleVersion {
        tenant: String,
        incoming: ConfigVersion,
        observed: ConfigVersion,
    },
}

/// Pluggable backend for per-tenant override storage. Substrate trait
/// — production impl lives in control DB.
pub trait TenantConfigStore: Send + Sync {
    /// Replace the override bundle for a tenant. Versioned: rejects
    /// stale writes (incoming version <= observed) per the same
    /// monotonic rule used in PR-1.
    fn upsert(&self, ov: TenantConfigOverride) -> Result<(), TenantConfigError>;

    fn get(&self, tenant_id: &str) -> Option<TenantConfigOverride>;

    /// List all tenant ids that have overrides, in tenant-id-asc order.
    fn list(&self) -> Vec<String>;

    fn delete(&self, tenant_id: &str) -> bool;
}

/// In-memory ref impl. Wraps `Arc<RwLock<BTreeMap<...>>>` — cheap to
/// clone (the wrapper). Tests + dev.
#[derive(Default, Clone)]
pub struct InMemoryTenantConfigStore {
    inner: Arc<RwLock<BTreeMap<String, TenantConfigOverride>>>,
}

impl InMemoryTenantConfigStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }
}

impl TenantConfigStore for InMemoryTenantConfigStore {
    fn upsert(&self, ov: TenantConfigOverride) -> Result<(), TenantConfigError> {
        let mut g = self.inner.write();
        if let Some(existing) = g.get(&ov.tenant_id) {
            if ov.version <= existing.version {
                return Err(TenantConfigError::StaleVersion {
                    tenant: ov.tenant_id.clone(),
                    incoming: ov.version,
                    observed: existing.version,
                });
            }
        }
        g.insert(ov.tenant_id.clone(), ov);
        Ok(())
    }

    fn get(&self, tenant_id: &str) -> Option<TenantConfigOverride> {
        self.inner.read().get(tenant_id).cloned()
    }

    fn list(&self) -> Vec<String> {
        self.inner.read().keys().cloned().collect()
    }

    fn delete(&self, tenant_id: &str) -> bool {
        self.inner.write().remove(tenant_id).is_some()
    }
}

/// Effective value lookup. Returns:
/// - `Some(&OverrideValue)` — the tenant has set this key.
/// - `None` — no override for this (tenant, key) pair. Caller falls
///   back to the global config.
///
/// Substrate-only: this is the resolver, not the consumer. The caller
/// supplies its own `&TenantConfigOverride` (already loaded from the
/// store) — the resolver is a pure function so it can be called many
/// times per request without re-hitting the store.
pub fn resolve<'a>(
    overrides: &'a TenantConfigOverride,
    key: &str,
) -> Option<&'a OverrideValue> {
    overrides.get(key)
}

/// Convenience: lookup against a maybe-present override bundle.
/// `None` bundle means "no overrides for this tenant" → returns None
/// without touching anything.
pub fn resolve_opt<'a>(
    bundle: Option<&'a TenantConfigOverride>,
    key: &str,
) -> Option<&'a OverrideValue> {
    bundle.and_then(|b| b.get(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ov_at(v: u64) -> TenantConfigOverride {
        let mut o = TenantConfigOverride::new("acme", ConfigVersion(v));
        o.set(
            "admission.max_concurrent_expanded_recall",
            OverrideValue::UInt(8),
        );
        o
    }

    #[test]
    fn override_value_type_str_pinned() {
        assert_eq!(OverrideValue::String("x".into()).type_str(), "string");
        assert_eq!(OverrideValue::Int(0).type_str(), "int");
        assert_eq!(OverrideValue::UInt(0).type_str(), "uint");
        assert_eq!(OverrideValue::Float(0.0).type_str(), "float");
        assert_eq!(OverrideValue::Bool(true).type_str(), "bool");
        assert_eq!(OverrideValue::Duration(Duration::ZERO).type_str(), "duration");
    }

    #[test]
    fn override_set_get_round_trip() {
        let mut o = TenantConfigOverride::new("acme", ConfigVersion(1));
        o.set("k", OverrideValue::Int(42));
        assert_eq!(o.get("k"), Some(&OverrideValue::Int(42)));
    }

    #[test]
    fn override_unset_removes_and_returns_old() {
        let mut o = TenantConfigOverride::new("acme", ConfigVersion(1));
        o.set("k", OverrideValue::Int(42));
        assert_eq!(o.unset("k"), Some(OverrideValue::Int(42)));
        assert!(o.get("k").is_none());
    }

    #[test]
    fn override_keys_iterate_in_sorted_order() {
        let mut o = TenantConfigOverride::new("acme", ConfigVersion(1));
        o.set("zzz", OverrideValue::Bool(true));
        o.set("aaa", OverrideValue::Bool(false));
        o.set("mmm", OverrideValue::Bool(true));
        let keys: Vec<&String> = o.keys().collect();
        // BTreeMap → ascending.
        assert_eq!(keys, vec!["aaa", "mmm", "zzz"]);
    }

    #[test]
    fn store_upsert_first_writes_through() {
        let s = InMemoryTenantConfigStore::new();
        s.upsert(ov_at(1)).unwrap();
        let got = s.get("acme").unwrap();
        assert_eq!(got.version, ConfigVersion(1));
    }

    #[test]
    fn store_upsert_advances_version() {
        let s = InMemoryTenantConfigStore::new();
        s.upsert(ov_at(1)).unwrap();
        s.upsert(ov_at(2)).unwrap();
        assert_eq!(s.get("acme").unwrap().version, ConfigVersion(2));
    }

    #[test]
    fn store_rejects_stale_version() {
        let s = InMemoryTenantConfigStore::new();
        s.upsert(ov_at(2)).unwrap();
        // v1 < observed v2 → reject.
        let err = s.upsert(ov_at(1)).unwrap_err();
        assert!(matches!(err, TenantConfigError::StaleVersion { .. }));
        // Same-version is also rejected (monotonic strict).
        let err2 = s.upsert(ov_at(2)).unwrap_err();
        assert!(matches!(err2, TenantConfigError::StaleVersion { .. }));
    }

    #[test]
    fn store_list_returns_all_tenants_sorted() {
        let s = InMemoryTenantConfigStore::new();
        s.upsert(TenantConfigOverride::new("zzz", ConfigVersion(1))).unwrap();
        s.upsert(TenantConfigOverride::new("aaa", ConfigVersion(1))).unwrap();
        s.upsert(TenantConfigOverride::new("mmm", ConfigVersion(1))).unwrap();
        let listed = s.list();
        assert_eq!(listed, vec!["aaa".to_string(), "mmm".into(), "zzz".into()]);
    }

    #[test]
    fn store_delete_returns_existed_flag() {
        let s = InMemoryTenantConfigStore::new();
        s.upsert(ov_at(1)).unwrap();
        assert!(s.delete("acme"));
        assert!(!s.delete("acme")); // already gone
    }

    #[test]
    fn store_get_unknown_returns_none() {
        let s = InMemoryTenantConfigStore::new();
        assert!(s.get("ghost").is_none());
    }

    #[test]
    fn resolve_returns_set_value() {
        let mut o = TenantConfigOverride::new("acme", ConfigVersion(1));
        o.set("k", OverrideValue::Int(42));
        let v = resolve(&o, "k");
        assert_eq!(v, Some(&OverrideValue::Int(42)));
    }

    #[test]
    fn resolve_returns_none_for_unset_key() {
        let o = TenantConfigOverride::new("acme", ConfigVersion(1));
        assert!(resolve(&o, "unset_key").is_none());
    }

    #[test]
    fn resolve_opt_handles_missing_bundle() {
        // No bundle → None without touching anything.
        let v: Option<&OverrideValue> = resolve_opt(None, "k");
        assert!(v.is_none());
        // Bundle present but key not set.
        let o = TenantConfigOverride::new("acme", ConfigVersion(1));
        assert!(resolve_opt(Some(&o), "k").is_none());
    }

    #[test]
    fn store_dyn_dispatch() {
        let store: Arc<dyn TenantConfigStore> = Arc::new(InMemoryTenantConfigStore::new());
        store.upsert(ov_at(1)).unwrap();
        assert!(store.get("acme").is_some());
    }

    #[test]
    fn override_versions_are_independent_per_tenant() {
        // tenant A at v5 and tenant B at v1 don't conflict — store keys
        // by tenant_id so upserts at any version are independent.
        let s = InMemoryTenantConfigStore::new();
        let mut a = TenantConfigOverride::new("a", ConfigVersion(5));
        a.set("k", OverrideValue::Int(1));
        let mut b = TenantConfigOverride::new("b", ConfigVersion(1));
        b.set("k", OverrideValue::Int(2));
        s.upsert(a).unwrap();
        s.upsert(b).unwrap();
        assert_eq!(s.get("a").unwrap().version, ConfigVersion(5));
        assert_eq!(s.get("b").unwrap().version, ConfigVersion(1));
    }

    #[test]
    fn override_serialization_round_trip() {
        // Wire format must round-trip — the commit log persists the
        // serialized form. Catch accidental field renames at compile +
        // semver review time.
        let mut o = TenantConfigOverride::new("acme", ConfigVersion(7));
        o.set("k1", OverrideValue::Int(42));
        o.set("k2", OverrideValue::Bool(true));
        o.set("k3", OverrideValue::Duration(Duration::from_secs(60)));
        let json = serde_json::to_string(&o).unwrap();
        let back: TenantConfigOverride = serde_json::from_str(&json).unwrap();
        assert_eq!(o, back);
    }
}
