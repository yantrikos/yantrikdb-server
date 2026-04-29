//! RFC 021 PR-1 — versioned config primitives.
//!
//! ## What this owns
//!
//! - [`ConfigVersion`] — monotonic integer, the wire/persisted token
//!   for "which version is this".
//! - [`VersionedConfig<T>`] — wraps any config struct with its current
//!   version. Cheap to clone (Arc-wrapped inner T).
//! - [`ConfigDelta<T>`] — typed delta describing a change. Captures
//!   the new value AND the version it transitions FROM, so a stale
//!   apply (someone hands us a delta from v3 while we're already at
//!   v5) is detected at the apply site.
//!
//! ## Why versions live in the commit log
//!
//! The RFC 021 design says config changes are mutations on the commit
//! log: same replay path as data writes, same Raft-fed-by-leader
//! ordering. `ConfigVersion` increments monotonically in lock-step
//! with the log entry that wrote it. Any node can replay the log and
//! reach the same final config at the same version.
//!
//! This module ships the *types*; the commit-log integration (a new
//! `MemoryMutation::SetConfig` variant) is a follow-up consumer PR.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Monotonic, 1-indexed config version. v0 = "no config has been
/// applied" sentinel; v1 = first applied config.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct ConfigVersion(pub u64);

impl ConfigVersion {
    pub const SENTINEL: Self = ConfigVersion(0);

    pub fn next(self) -> Self {
        ConfigVersion(self.0.saturating_add(1))
    }

    pub fn is_sentinel(self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Display for ConfigVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// A T plus the version it represents. Cheap to clone (Arc<T>).
#[derive(Debug)]
pub struct VersionedConfig<T> {
    inner: Arc<T>,
    version: ConfigVersion,
}

impl<T> VersionedConfig<T> {
    pub fn new(value: T, version: ConfigVersion) -> Self {
        Self {
            inner: Arc::new(value),
            version,
        }
    }

    /// Construct at sentinel (v0). Useful in tests + initial state
    /// before any reload has occurred.
    pub fn sentinel(value: T) -> Self {
        Self::new(value, ConfigVersion::SENTINEL)
    }

    pub fn version(&self) -> ConfigVersion {
        self.version
    }

    pub fn value(&self) -> &T {
        &self.inner
    }

    pub fn arc(&self) -> Arc<T> {
        Arc::clone(&self.inner)
    }
}

impl<T> Clone for VersionedConfig<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            version: self.version,
        }
    }
}

impl<T: PartialEq> PartialEq for VersionedConfig<T> {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version && *self.inner == *other.inner
    }
}

/// Typed delta. Captures both the FROM and TO version so a stale apply
/// is detectable. The implementation deliberately uses full
/// replacement (`new_value: T`) rather than partial-patch semantics —
/// partial patches at this layer would require a per-field diff
/// machinery that's out of scope. The commit-log writer can choose to
/// optimize wire size by including only changed fields, but the
/// in-memory delta type is whole-T.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigDelta<T> {
    /// What we expect the current version to be. Apply fails if the
    /// observed version is not this.
    pub from_version: ConfigVersion,
    /// The new version this delta produces.
    pub to_version: ConfigVersion,
    pub new_value: T,
}

impl<T> ConfigDelta<T> {
    pub fn new(from_version: ConfigVersion, new_value: T) -> Self {
        Self {
            to_version: from_version.next(),
            from_version,
            new_value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinel_is_v0() {
        assert_eq!(ConfigVersion::SENTINEL, ConfigVersion(0));
        assert!(ConfigVersion::SENTINEL.is_sentinel());
        assert!(!ConfigVersion(1).is_sentinel());
    }

    #[test]
    fn next_is_monotonic() {
        let v = ConfigVersion(5);
        assert_eq!(v.next(), ConfigVersion(6));
    }

    #[test]
    fn next_saturates_at_max() {
        let max = ConfigVersion(u64::MAX);
        assert_eq!(max.next(), max);
    }

    #[test]
    fn version_display_format() {
        assert_eq!(format!("{}", ConfigVersion(7)), "v7");
        assert_eq!(format!("{}", ConfigVersion::SENTINEL), "v0");
    }

    #[test]
    fn versioned_config_clone_shares_arc() {
        let c = VersionedConfig::new(42u32, ConfigVersion(1));
        let c2 = c.clone();
        // Same backing allocation.
        assert!(Arc::ptr_eq(&c.arc(), &c2.arc()));
        assert_eq!(c.version(), c2.version());
    }

    #[test]
    fn versioned_config_sentinel_constructor() {
        let c = VersionedConfig::sentinel("hello".to_string());
        assert!(c.version().is_sentinel());
        assert_eq!(c.value(), "hello");
    }

    #[test]
    fn versioned_config_eq_compares_version_and_value() {
        let a = VersionedConfig::new(42u32, ConfigVersion(1));
        let b = VersionedConfig::new(42u32, ConfigVersion(1));
        let c = VersionedConfig::new(42u32, ConfigVersion(2));
        let d = VersionedConfig::new(43u32, ConfigVersion(1));
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn delta_to_version_is_one_past_from() {
        let d = ConfigDelta::new(ConfigVersion(7), "x");
        assert_eq!(d.from_version, ConfigVersion(7));
        assert_eq!(d.to_version, ConfigVersion(8));
    }

    #[test]
    fn version_ordering_is_numeric() {
        assert!(ConfigVersion(1) < ConfigVersion(2));
        assert!(ConfigVersion(2) > ConfigVersion(1));
        assert!(ConfigVersion(1) == ConfigVersion(1));
    }
}
