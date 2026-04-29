//! RFC 014-B — typed scopes + scope set.
//!
//! Per RFC 014-B: "Scoped tokens (read | write | recall | forget |
//! admin | tenant-management)". This module defines the typed scope
//! enum and a bitset for compact "this principal has these scopes"
//! representation.
//!
//! ## Scopes are not roles
//!
//! Scopes describe individual operations. Roles (Reader, Writer,
//! Auditor, Admin) are conventional bundles of scopes that the
//! tenant-management UI hands out, but the data plane only checks
//! scopes. Roles never appear in the auth path.

/// One operation-class. The wire form is the lowercase `as_str()`
/// value; serialization uses `as_str` so adding a scope can never
/// silently change existing serializations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Scope {
    /// Read existing memories (recall, get, neighbors). Cheap path.
    Read = 0,
    /// Write new memories (remember, correct, relate). Mutation path.
    Write = 1,
    /// Issue recall queries — typically a superset of Read but kept
    /// distinct so a token can recall without being able to read raw
    /// memory bytes through the lower-level get API.
    Recall = 2,
    /// Issue forget operations (tombstone, purge). Destructive — the
    /// most-restricted data-plane scope.
    Forget = 3,
    /// Administrative ops: quota policy CRUD, breaker mode, retention
    /// config, key rotation. Cluster-wide, not tenant-scoped.
    Admin = 4,
    /// Tenant lifecycle ops: create tenant, suspend tenant, set tier,
    /// rotate per-tenant keys. Less powerful than Admin (no global
    /// config touch), but cross-tenant — not a regular data-plane
    /// scope.
    TenantManagement = 5,
}

impl Scope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Scope::Read => "read",
            Scope::Write => "write",
            Scope::Recall => "recall",
            Scope::Forget => "forget",
            Scope::Admin => "admin",
            Scope::TenantManagement => "tenant-management",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "read" => Some(Scope::Read),
            "write" => Some(Scope::Write),
            "recall" => Some(Scope::Recall),
            "forget" => Some(Scope::Forget),
            "admin" => Some(Scope::Admin),
            "tenant-management" => Some(Scope::TenantManagement),
            _ => None,
        }
    }

    pub const fn all() -> [Scope; 6] {
        [
            Scope::Read,
            Scope::Write,
            Scope::Recall,
            Scope::Forget,
            Scope::Admin,
            Scope::TenantManagement,
        ]
    }

    fn bit(self) -> u32 {
        1u32 << (self as u8)
    }
}

/// Compact bitset of scopes. Cheap to copy, cheap to compare, no
/// allocation. The bit positions are pinned by `Scope as u8` and tested
/// for stability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ScopeSet(u32);

impl ScopeSet {
    pub const fn empty() -> Self {
        Self(0)
    }

    pub fn all() -> Self {
        let mut s = Self::empty();
        for sc in Scope::all() {
            s.add(sc);
        }
        s
    }

    pub fn from_iter<I: IntoIterator<Item = Scope>>(iter: I) -> Self {
        let mut s = Self::empty();
        for sc in iter {
            s.add(sc);
        }
        s
    }

    pub fn add(&mut self, scope: Scope) {
        self.0 |= scope.bit();
    }

    pub fn remove(&mut self, scope: Scope) {
        self.0 &= !scope.bit();
    }

    pub fn contains(&self, scope: Scope) -> bool {
        self.0 & scope.bit() != 0
    }

    /// True iff this set has every scope in `required`.
    pub fn covers(&self, required: ScopeSet) -> bool {
        (self.0 & required.0) == required.0
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = Scope> + '_ {
        let bits = self.0;
        Scope::all()
            .into_iter()
            .filter(move |s| bits & s.bit() != 0)
    }

    /// Comma-separated wire-form representation, e.g. `"read,recall"`.
    /// Stable order: matches `Scope::all()`.
    pub fn to_csv(&self) -> String {
        let parts: Vec<&'static str> = self.iter().map(Scope::as_str).collect();
        parts.join(",")
    }

    pub fn parse_csv(s: &str) -> Result<Self, String> {
        let mut set = Self::empty();
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let sc = Scope::parse(part).ok_or_else(|| format!("unknown scope `{}`", part))?;
            set.add(sc);
        }
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_round_trip_via_string() {
        for s in Scope::all() {
            assert_eq!(Scope::parse(s.as_str()), Some(s));
        }
    }

    #[test]
    fn scope_parse_unknown_returns_none() {
        assert_eq!(Scope::parse("nonsense"), None);
        assert_eq!(Scope::parse(""), None);
        assert_eq!(Scope::parse("READ"), None); // case-sensitive
    }

    #[test]
    fn scope_string_form_pinned() {
        // Stability test — wire format breaks if these change.
        assert_eq!(Scope::Read.as_str(), "read");
        assert_eq!(Scope::Write.as_str(), "write");
        assert_eq!(Scope::Recall.as_str(), "recall");
        assert_eq!(Scope::Forget.as_str(), "forget");
        assert_eq!(Scope::Admin.as_str(), "admin");
        assert_eq!(Scope::TenantManagement.as_str(), "tenant-management");
    }

    #[test]
    fn empty_set_contains_nothing() {
        let s = ScopeSet::empty();
        assert!(s.is_empty());
        for sc in Scope::all() {
            assert!(!s.contains(sc));
        }
    }

    #[test]
    fn add_and_remove_round_trip() {
        let mut s = ScopeSet::empty();
        s.add(Scope::Read);
        s.add(Scope::Write);
        assert!(s.contains(Scope::Read));
        assert!(s.contains(Scope::Write));
        assert!(!s.contains(Scope::Forget));
        s.remove(Scope::Read);
        assert!(!s.contains(Scope::Read));
        assert!(s.contains(Scope::Write));
    }

    #[test]
    fn all_set_contains_all_scopes() {
        let s = ScopeSet::all();
        for sc in Scope::all() {
            assert!(s.contains(sc));
        }
    }

    #[test]
    fn covers_requires_every_required_scope() {
        let granted = ScopeSet::from_iter([Scope::Read, Scope::Recall]);
        assert!(granted.covers(ScopeSet::from_iter([Scope::Read])));
        assert!(granted.covers(ScopeSet::from_iter([Scope::Read, Scope::Recall])));
        assert!(!granted.covers(ScopeSet::from_iter([Scope::Read, Scope::Write])));
        assert!(granted.covers(ScopeSet::empty())); // vacuously true
    }

    #[test]
    fn csv_round_trip() {
        let s = ScopeSet::from_iter([Scope::Read, Scope::Forget, Scope::Admin]);
        let csv = s.to_csv();
        // Stable order from Scope::all() iteration.
        assert_eq!(csv, "read,forget,admin");
        let parsed = ScopeSet::parse_csv(&csv).unwrap();
        assert_eq!(parsed, s);
    }

    #[test]
    fn parse_csv_handles_whitespace_and_empty_segments() {
        let parsed = ScopeSet::parse_csv(" read , write ,, ").unwrap();
        assert!(parsed.contains(Scope::Read));
        assert!(parsed.contains(Scope::Write));
        assert!(!parsed.contains(Scope::Admin));
    }

    #[test]
    fn parse_csv_rejects_unknown_scope() {
        let err = ScopeSet::parse_csv("read,nonsense").unwrap_err();
        assert!(err.contains("nonsense"));
    }

    #[test]
    fn iter_returns_in_stable_order() {
        let s = ScopeSet::from_iter([Scope::Forget, Scope::Read, Scope::Admin]);
        let collected: Vec<Scope> = s.iter().collect();
        // Must match Scope::all() ordering.
        assert_eq!(collected, vec![Scope::Read, Scope::Forget, Scope::Admin]);
    }

    #[test]
    fn bit_positions_are_distinct() {
        // Defensive: if Scope discriminants collide, ScopeSet semantics
        // break silently.
        let mut bits = std::collections::HashSet::new();
        for sc in Scope::all() {
            assert!(bits.insert(sc.bit()), "duplicate bit for {:?}", sc);
        }
    }
}
