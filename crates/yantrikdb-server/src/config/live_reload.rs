//! RFC 021 PR-1 — `Reloadable` trait + apply orchestrator.
//!
//! ## What this owns
//!
//! - [`Reloadable`] trait — each runtime component that holds tunable
//!   state implements this. The orchestrator hands each component a
//!   delta; the component validates + swaps in the new value.
//! - [`ReloadOutcome`] — what happened on apply: applied, ignored
//!   (already at-or-past version), rejected (validation failed).
//! - [`ReloadError`] — error variants.
//!
//! ## Why apply, not get/set
//!
//! A `Reloadable` doesn't expose a setter. The orchestrator is the
//! ONLY caller, and it always goes through `apply(&delta)`. This:
//! - keeps the version-monotonicity invariant local to one method,
//! - lets validation reject the new value atomically (failing apply
//!   leaves the component on the old value),
//! - gives the testbed a single seam for "drive a config change
//!   through this component".

use super::versioned::{ConfigDelta, ConfigVersion};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadOutcome {
    /// Delta applied. New version is the delta's `to_version`.
    Applied {
        from_version: ConfigVersion,
        to_version: ConfigVersion,
    },
    /// The component is already at this version (or past it). No-op.
    /// Distinct from rejection — this is normal in replay scenarios.
    Ignored {
        observed_version: ConfigVersion,
        delta_to_version: ConfigVersion,
    },
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ReloadError {
    /// The delta says "transition from v3", but we observed v5. Either
    /// the delta is stale (caller should refresh from the log) or
    /// there's a fork in the log replay path.
    #[error("stale delta: from `{delta_from}`, observed `{observed}`")]
    StaleDelta {
        delta_from: ConfigVersion,
        observed: ConfigVersion,
    },
    /// Delta values failed component-specific validation. The component
    /// is unchanged.
    #[error("validation rejected: {0}")]
    Validation(String),
    /// Underlying reason from the component impl.
    #[error("component error: {0}")]
    Component(String),
}

/// One reloadable component. The trait method takes `&mut self` —
/// implementations are expected to wrap their hot state behind
/// `Arc<Mutex<...>>` or `parking_lot::RwLock<...>` so concurrent
/// readers see either the old or new value, never a torn write.
pub trait Reloadable: Send + Sync {
    /// The shape of the configuration this component consumes.
    type Config;

    /// Read the version this component is currently running.
    fn current_version(&self) -> ConfigVersion;

    /// Apply a delta. Returns:
    /// - `Ok(Applied)` on successful apply.
    /// - `Ok(Ignored)` if the delta's `to_version` is at-or-below the
    ///   current version (replay-safe no-op).
    /// - `Err(StaleDelta)` if the delta's `from_version` doesn't match
    ///   the current version AND the delta would advance us.
    /// - `Err(Validation)` if the new value is malformed.
    fn apply(&mut self, delta: &ConfigDelta<Self::Config>) -> Result<ReloadOutcome, ReloadError>;
}

/// Helper for the common case: a generic decision routine that an
/// impl can call before swapping. Returns `Ok(true)` to swap, `Ok(false)`
/// to ignore (replay no-op), `Err(...)` on stale.
pub fn classify_apply(
    observed: ConfigVersion,
    delta_from: ConfigVersion,
    delta_to: ConfigVersion,
) -> Result<DeltaDecision, ReloadError> {
    if delta_to <= observed {
        return Ok(DeltaDecision::Ignore);
    }
    if delta_from != observed {
        return Err(ReloadError::StaleDelta {
            delta_from,
            observed,
        });
    }
    Ok(DeltaDecision::Apply)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaDecision {
    /// New value should be installed.
    Apply,
    /// Already at-or-past this version. No-op.
    Ignore,
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use std::sync::Arc;

    /// Sample reloadable: a container around a u32 with a "must be
    /// non-zero" validation rule. Used to exercise the trait contract.
    #[derive(Default)]
    struct Counter {
        version: ConfigVersion,
        value: Arc<Mutex<u32>>,
    }

    impl Reloadable for Counter {
        type Config = u32;

        fn current_version(&self) -> ConfigVersion {
            self.version
        }

        fn apply(
            &mut self,
            delta: &ConfigDelta<Self::Config>,
        ) -> Result<ReloadOutcome, ReloadError> {
            match classify_apply(self.version, delta.from_version, delta.to_version)? {
                DeltaDecision::Ignore => Ok(ReloadOutcome::Ignored {
                    observed_version: self.version,
                    delta_to_version: delta.to_version,
                }),
                DeltaDecision::Apply => {
                    if delta.new_value == 0 {
                        return Err(ReloadError::Validation("value must be non-zero".into()));
                    }
                    *self.value.lock() = delta.new_value;
                    let from = self.version;
                    self.version = delta.to_version;
                    Ok(ReloadOutcome::Applied {
                        from_version: from,
                        to_version: delta.to_version,
                    })
                }
            }
        }
    }

    #[test]
    fn fresh_apply_succeeds_from_sentinel() {
        let mut c = Counter::default();
        let d = ConfigDelta::new(ConfigVersion::SENTINEL, 42);
        let out = c.apply(&d).unwrap();
        assert!(matches!(out, ReloadOutcome::Applied { .. }));
        assert_eq!(c.current_version(), ConfigVersion(1));
        assert_eq!(*c.value.lock(), 42);
    }

    #[test]
    fn second_apply_advances_version() {
        let mut c = Counter::default();
        c.apply(&ConfigDelta::new(ConfigVersion::SENTINEL, 42))
            .unwrap();
        c.apply(&ConfigDelta::new(ConfigVersion(1), 99)).unwrap();
        assert_eq!(c.current_version(), ConfigVersion(2));
        assert_eq!(*c.value.lock(), 99);
    }

    #[test]
    fn stale_delta_rejected() {
        let mut c = Counter::default();
        c.apply(&ConfigDelta::new(ConfigVersion::SENTINEL, 42))
            .unwrap();
        // Now at v1. A delta claiming to come from a divergent version
        // (v5 → v6) — would advance us, but from doesn't match observed.
        let stale = ConfigDelta::new(ConfigVersion(5), 7);
        let err = c.apply(&stale).unwrap_err();
        assert!(matches!(
            err,
            ReloadError::StaleDelta {
                delta_from: ConfigVersion(5),
                observed: ConfigVersion(1)
            }
        ));
        // Original value preserved.
        assert_eq!(*c.value.lock(), 42);
    }

    #[test]
    fn replay_below_current_version_is_ignored() {
        let mut c = Counter::default();
        c.apply(&ConfigDelta::new(ConfigVersion::SENTINEL, 42))
            .unwrap();
        c.apply(&ConfigDelta::new(ConfigVersion(1), 99)).unwrap();
        // Replay an older delta — already past, no-op.
        let stale_replay = ConfigDelta {
            from_version: ConfigVersion::SENTINEL,
            to_version: ConfigVersion(1),
            new_value: 42,
        };
        let out = c.apply(&stale_replay).unwrap();
        match out {
            ReloadOutcome::Ignored {
                observed_version,
                delta_to_version,
            } => {
                assert_eq!(observed_version, ConfigVersion(2));
                assert_eq!(delta_to_version, ConfigVersion(1));
            }
            other => panic!("expected Ignored, got {:?}", other),
        }
        // Value unchanged.
        assert_eq!(*c.value.lock(), 99);
    }

    #[test]
    fn validation_failure_preserves_old_value() {
        let mut c = Counter::default();
        c.apply(&ConfigDelta::new(ConfigVersion::SENTINEL, 42))
            .unwrap();
        // Apply with new_value=0 — validation rejects.
        let bad = ConfigDelta::new(ConfigVersion(1), 0);
        let err = c.apply(&bad).unwrap_err();
        assert!(matches!(err, ReloadError::Validation(_)));
        // Value AND version preserved.
        assert_eq!(*c.value.lock(), 42);
        assert_eq!(c.current_version(), ConfigVersion(1));
    }

    #[test]
    fn classify_apply_decisions() {
        // Forward — apply.
        assert_eq!(
            classify_apply(ConfigVersion(1), ConfigVersion(1), ConfigVersion(2)).unwrap(),
            DeltaDecision::Apply
        );
        // Already-past — ignore.
        assert_eq!(
            classify_apply(ConfigVersion(5), ConfigVersion(1), ConfigVersion(2)).unwrap(),
            DeltaDecision::Ignore
        );
        // Equal-to-current — ignore (delta_to == observed).
        assert_eq!(
            classify_apply(ConfigVersion(2), ConfigVersion(1), ConfigVersion(2)).unwrap(),
            DeltaDecision::Ignore
        );
        // Stale-from — error.
        let err = classify_apply(ConfigVersion(3), ConfigVersion(1), ConfigVersion(4)).unwrap_err();
        assert!(matches!(err, ReloadError::StaleDelta { .. }));
    }

    #[test]
    fn dyn_dispatch_via_trait_object() {
        // Trait must remain object-safe so the orchestrator can hold
        // Vec<Box<dyn Reloadable<Config = u32>>>.
        let counter: Box<dyn Reloadable<Config = u32>> = Box::new(Counter::default());
        assert_eq!(counter.current_version(), ConfigVersion::SENTINEL);
    }
}
