//! RFC 009 PR-2 — quota policy types + scope resolution.
//!
//! Per RFC 009 §2 ("quota_policies" table):
//!
//! ```sql
//! CREATE TABLE quota_policies (
//!     scope_type TEXT NOT NULL,    -- 'principal' | 'namespace' | 'global'
//!     scope_value TEXT NOT NULL,    -- the principal id, namespace name, or '*'
//!     rps_limit INTEGER,
//!     cost_budget_per_sec INTEGER,
//!     max_concurrent_expanded INTEGER,
//!     tier TEXT,                    -- 'gold' | 'silver' | 'bronze' (informational)
//!     PRIMARY KEY (scope_type, scope_value)
//! );
//! ```
//!
//! ## Defaults: provisional vs fallback
//!
//! - **`PROVISIONAL_DEFAULTS`** — what new tenants and the global default
//!   row land at on first contact. Per RFC §6 decision #5 ("Default global
//!   quotas"), these are *labeled provisional* until the PR-7 benchmark
//!   matrix promotes them to non-provisional. Operators are expected to
//!   tune per-tenant.
//!
//! - **`FALLBACK_DEFAULTS`** — half-throughput conservative defaults used
//!   when control DB is unreachable and there's no cached policy. Per
//!   RFC §6 ("Failure modes" → "Quota policy lookup → control DB
//!   unavailable"). Never fail-open to unlimited.

use std::time::Duration;

/// What the quota row keys on. The control DB stores this as a `(scope_kind,
/// scope_value)` pair; the in-memory representation keeps them split for
/// type-safety.
///
/// Resolution order at request time: `Principal(id)` → `Namespace(name)` →
/// `Global` ("*"). The first matching policy wins. [`PolicyResolver`] runs
/// the lookup chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeKind {
    Principal,
    Namespace,
    Global,
}

impl ScopeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ScopeKind::Principal => "principal",
            ScopeKind::Namespace => "namespace",
            ScopeKind::Global => "global",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "principal" => Some(ScopeKind::Principal),
            "namespace" => Some(ScopeKind::Namespace),
            "global" => Some(ScopeKind::Global),
            _ => None,
        }
    }
}

/// A fully-qualified scope: kind + value. The value is opaque for
/// `Principal`/`Namespace` and always `"*"` for `Global`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuotaScope {
    pub kind: ScopeKind,
    pub value: String,
}

impl QuotaScope {
    pub fn principal(id: impl Into<String>) -> Self {
        Self {
            kind: ScopeKind::Principal,
            value: id.into(),
        }
    }

    pub fn namespace(name: impl Into<String>) -> Self {
        Self {
            kind: ScopeKind::Namespace,
            value: name.into(),
        }
    }

    pub fn global() -> Self {
        Self {
            kind: ScopeKind::Global,
            value: "*".to_string(),
        }
    }
}

/// Tier label. Informational — does not change enforcement. Allows
/// operators to group policies for dashboards / billing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    Bronze,
    Silver,
    Gold,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Bronze => "bronze",
            Tier::Silver => "silver",
            Tier::Gold => "gold",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "bronze" => Some(Tier::Bronze),
            "silver" => Some(Tier::Silver),
            "gold" => Some(Tier::Gold),
            _ => None,
        }
    }
}

/// In-memory representation of a quota_policies row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuotaPolicy {
    pub scope: QuotaScope,
    /// Requests per second. None means "no RPS limit at this scope".
    pub rps_limit: Option<u64>,
    /// Cost-units per second. None means "no cost budget at this scope".
    pub cost_budget_per_sec: Option<u64>,
    /// Concurrent expanded recalls. None means "no concurrency cap at
    /// this scope".
    pub max_concurrent_expanded: Option<u32>,
    pub tier: Option<Tier>,
    /// Monotonic version. Bumped on every UPSERT. Used for divergence
    /// detection across the cluster (RFC 009 §2 "Limitations").
    pub policy_version: u64,
}

impl QuotaPolicy {
    /// Build a fully-default policy for the given scope using
    /// `PROVISIONAL_DEFAULTS`. Used by lazy-on-first-request backfill.
    pub fn provisional_for(scope: QuotaScope) -> Self {
        Self {
            scope,
            rps_limit: Some(PROVISIONAL_DEFAULTS.rps),
            cost_budget_per_sec: Some(PROVISIONAL_DEFAULTS.cost_budget),
            max_concurrent_expanded: Some(PROVISIONAL_DEFAULTS.max_concurrent_expanded),
            tier: None,
            policy_version: 1,
        }
    }

    /// Build a fallback-conservative policy for the given scope. Used
    /// when control DB is unreachable and no cache exists.
    pub fn fallback_for(scope: QuotaScope) -> Self {
        Self {
            scope,
            rps_limit: Some(FALLBACK_DEFAULTS.rps),
            cost_budget_per_sec: Some(FALLBACK_DEFAULTS.cost_budget),
            max_concurrent_expanded: Some(FALLBACK_DEFAULTS.max_concurrent_expanded),
            tier: None,
            policy_version: 0, // 0 = "synthetic, never persisted"
        }
    }
}

/// Default values applied to new tenants and the global row at rollout.
/// Per RFC §6 decision #5: provisional, may be retuned in PR-7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvisionalDefaults {
    pub rps: u64,
    pub cost_budget: u64,
    pub max_concurrent_expanded: u32,
    /// How long a cached policy stays valid before a control-DB re-fetch.
    pub cache_ttl: Duration,
}

pub const PROVISIONAL_DEFAULTS: ProvisionalDefaults = ProvisionalDefaults {
    rps: 100,
    cost_budget: 1000,
    max_concurrent_expanded: 4,
    cache_ttl: Duration::from_secs(600),
};

pub const FALLBACK_DEFAULTS: ProvisionalDefaults = ProvisionalDefaults {
    rps: 50,                    // half of provisional
    cost_budget: 500,           // half of provisional
    max_concurrent_expanded: 2, // half of provisional
    cache_ttl: Duration::from_secs(60),
};

/// Resolves the effective policy for a given (principal, namespace) pair.
/// Trait so the data plane can plug a control-DB-backed implementation
/// without admission middleware tests needing a SQLite instance.
pub trait PolicyResolver: Send + Sync {
    /// Look up the most-specific policy that applies. Resolution order:
    /// `Principal` → `Namespace` → `Global`.
    ///
    /// Returns the first match. If nothing matches, returns
    /// `Ok(provisional_global())` so callers always get a usable policy.
    fn resolve(
        &self,
        principal: Option<&str>,
        namespace: Option<&str>,
    ) -> Result<QuotaPolicy, ResolveError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    /// Control DB unreachable AND no cached policy. Caller should fall
    /// back to [`QuotaPolicy::fallback_for`] and emit
    /// `quota_lookup_fallback_total`.
    #[error("control DB unavailable and no cached policy")]
    Unavailable,
    #[error("backend error: {0}")]
    Backend(String),
}

/// Convenience helper to compute the provisional global policy. Useful
/// when initializing a fresh control DB.
pub fn provisional_global() -> QuotaPolicy {
    QuotaPolicy::provisional_for(QuotaScope::global())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_kind_round_trip() {
        for k in [
            ScopeKind::Principal,
            ScopeKind::Namespace,
            ScopeKind::Global,
        ] {
            assert_eq!(ScopeKind::parse(k.as_str()), Some(k));
        }
        assert!(ScopeKind::parse("nonsense").is_none());
    }

    #[test]
    fn tier_round_trip_case_insensitive() {
        assert_eq!(Tier::parse("Gold"), Some(Tier::Gold));
        assert_eq!(Tier::parse("SILVER"), Some(Tier::Silver));
        assert_eq!(Tier::parse("bronze"), Some(Tier::Bronze));
        assert!(Tier::parse("platinum").is_none());
    }

    #[test]
    fn provisional_for_is_full_policy() {
        let p = QuotaPolicy::provisional_for(QuotaScope::principal("alice"));
        assert_eq!(p.rps_limit, Some(PROVISIONAL_DEFAULTS.rps));
        assert_eq!(
            p.cost_budget_per_sec,
            Some(PROVISIONAL_DEFAULTS.cost_budget)
        );
        assert_eq!(
            p.max_concurrent_expanded,
            Some(PROVISIONAL_DEFAULTS.max_concurrent_expanded)
        );
        assert_eq!(p.policy_version, 1);
    }

    #[test]
    fn fallback_is_strictly_smaller_than_provisional() {
        // Defensive: if someone tunes provisional UP without thinking
        // about fallback, this test catches the regression. Fallback
        // must be ≤ provisional on every dimension.
        assert!(FALLBACK_DEFAULTS.rps <= PROVISIONAL_DEFAULTS.rps);
        assert!(FALLBACK_DEFAULTS.cost_budget <= PROVISIONAL_DEFAULTS.cost_budget);
        assert!(
            FALLBACK_DEFAULTS.max_concurrent_expanded
                <= PROVISIONAL_DEFAULTS.max_concurrent_expanded
        );
    }

    #[test]
    fn fallback_policy_has_synthetic_version_zero() {
        // policy_version=0 marks "synthetic, never persisted". Operators
        // can grep `policy_version=0` in audit logs to find requests
        // served during a control-DB outage.
        let p = QuotaPolicy::fallback_for(QuotaScope::namespace("hot_ns"));
        assert_eq!(p.policy_version, 0);
    }

    #[test]
    fn provisional_global_helper_returns_global_scope() {
        let p = provisional_global();
        assert_eq!(p.scope.kind, ScopeKind::Global);
        assert_eq!(p.scope.value, "*");
    }

    #[test]
    fn quota_scope_constructors_set_kind() {
        assert_eq!(QuotaScope::principal("alice").kind, ScopeKind::Principal);
        assert_eq!(QuotaScope::namespace("ns1").kind, ScopeKind::Namespace);
        assert_eq!(QuotaScope::global().kind, ScopeKind::Global);
        assert_eq!(QuotaScope::global().value, "*");
    }

    /// Test resolver impl: in-memory map keyed by scope. Useful as a
    /// reference implementation for the control-DB-backed resolver and
    /// for middleware tests.
    struct StaticResolver {
        principals: std::collections::HashMap<String, QuotaPolicy>,
        namespaces: std::collections::HashMap<String, QuotaPolicy>,
        global: QuotaPolicy,
    }

    impl PolicyResolver for StaticResolver {
        fn resolve(
            &self,
            principal: Option<&str>,
            namespace: Option<&str>,
        ) -> Result<QuotaPolicy, ResolveError> {
            if let Some(p) = principal {
                if let Some(pol) = self.principals.get(p) {
                    return Ok(pol.clone());
                }
            }
            if let Some(n) = namespace {
                if let Some(pol) = self.namespaces.get(n) {
                    return Ok(pol.clone());
                }
            }
            Ok(self.global.clone())
        }
    }

    #[test]
    fn resolver_picks_principal_first() {
        let r = StaticResolver {
            principals: std::collections::HashMap::from([(
                "alice".to_string(),
                QuotaPolicy {
                    rps_limit: Some(999),
                    ..QuotaPolicy::provisional_for(QuotaScope::principal("alice"))
                },
            )]),
            namespaces: std::collections::HashMap::new(),
            global: provisional_global(),
        };
        let pol = r.resolve(Some("alice"), Some("any_ns")).unwrap();
        assert_eq!(pol.rps_limit, Some(999));
    }

    #[test]
    fn resolver_falls_back_to_namespace_when_no_principal() {
        let r = StaticResolver {
            principals: std::collections::HashMap::new(),
            namespaces: std::collections::HashMap::from([(
                "shared".to_string(),
                QuotaPolicy {
                    rps_limit: Some(50),
                    ..QuotaPolicy::provisional_for(QuotaScope::namespace("shared"))
                },
            )]),
            global: provisional_global(),
        };
        let pol = r.resolve(Some("never_seen"), Some("shared")).unwrap();
        assert_eq!(pol.rps_limit, Some(50));
    }

    #[test]
    fn resolver_falls_back_to_global_when_no_match() {
        let r = StaticResolver {
            principals: std::collections::HashMap::new(),
            namespaces: std::collections::HashMap::new(),
            global: provisional_global(),
        };
        let pol = r.resolve(Some("ghost"), Some("nowhere")).unwrap();
        assert_eq!(pol.scope.kind, ScopeKind::Global);
    }
}
