//! RFC 009 PR-2 — admission cost function.
//!
//! Per RFC 009 §2 ("Cost function (simplified per redteam)"):
//!
//! ```text
//! cost = top_k × E_expand
//! E_expand = 5 if expand_entities else 1
//! ```
//!
//! `namespace_size` is intentionally **not** in the default cost function:
//! coupling cost to data shape creates silent drift (the same query gets
//! more expensive as the dataset grows) and invites operator confusion.
//! If empirical benchmarks later show namespace density correlates with
//! resource cost, an opt-in `cost_function = "advanced"` config can be
//! added. Defaults stay shape-stable.

/// Default expansion multiplier. The hot cost of expanded recall (graph
/// walk, entity boost, fan-out) is empirically ~5× a no-expand recall on
/// the term=1423 incident traffic.
pub const DEFAULT_E_EXPAND: u32 = 5;

/// Inputs to the cost function. Constructed from the parsed recall
/// request before the handler runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostInputs {
    pub top_k: u32,
    pub expand_entities: bool,
}

/// Tunable cost-function parameters. Lives in `[admission]` config.
/// Default is the simplified `top_k × E_expand` formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostParams {
    pub e_expand: u32,
}

impl Default for CostParams {
    fn default() -> Self {
        Self {
            e_expand: DEFAULT_E_EXPAND,
        }
    }
}

impl CostParams {
    /// Validate. `e_expand` must be ≥ 1 (otherwise expanded recall costs
    /// less than a non-expanded recall, which is nonsense). 0 is rejected
    /// with a clear error so config typos surface at startup, not at
    /// 3am during a thrash.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.e_expand == 0 {
            return Err("admission.e_expand must be >= 1");
        }
        Ok(())
    }
}

/// Compute cost units for a recall request. Saturating arithmetic so a
/// (validated, but wildly out-of-range) `top_k` doesn't panic on overflow.
/// Returns `u64` because cost can exceed `u32` for very large `top_k` ×
/// `e_expand` combinations and we want headroom in budget arithmetic.
pub fn cost_units(inputs: CostInputs, params: CostParams) -> u64 {
    let multiplier = if inputs.expand_entities {
        params.e_expand
    } else {
        1
    };
    (inputs.top_k as u64).saturating_mul(multiplier as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_expand_is_top_k() {
        let cost = cost_units(
            CostInputs {
                top_k: 10,
                expand_entities: false,
            },
            CostParams::default(),
        );
        assert_eq!(cost, 10);
    }

    #[test]
    fn expand_multiplies_by_e_expand() {
        let cost = cost_units(
            CostInputs {
                top_k: 10,
                expand_entities: true,
            },
            CostParams::default(),
        );
        assert_eq!(cost, 10 * DEFAULT_E_EXPAND as u64);
    }

    #[test]
    fn term_1423_repro_pattern_is_expensive() {
        // 5 pollers × top_k=200 × expand=true = 1000 cost units per request.
        // A budget of 1000/s permits exactly one such request per second.
        // The dashboard delta between this and the cheap path is what
        // surfaces the noisy-neighbor signature.
        let cost = cost_units(
            CostInputs {
                top_k: 200,
                expand_entities: true,
            },
            CostParams::default(),
        );
        assert_eq!(cost, 1000);
    }

    #[test]
    fn custom_e_expand_is_honored() {
        let cost = cost_units(
            CostInputs {
                top_k: 10,
                expand_entities: true,
            },
            CostParams { e_expand: 3 },
        );
        assert_eq!(cost, 30);
    }

    #[test]
    fn zero_top_k_is_zero_cost() {
        // A degenerate but valid request — costs nothing.
        let cost = cost_units(
            CostInputs {
                top_k: 0,
                expand_entities: true,
            },
            CostParams::default(),
        );
        assert_eq!(cost, 0);
    }

    #[test]
    fn validate_rejects_zero_e_expand() {
        let params = CostParams { e_expand: 0 };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_accepts_one_or_higher() {
        assert!(CostParams { e_expand: 1 }.validate().is_ok());
        assert!(CostParams { e_expand: 100 }.validate().is_ok());
    }

    #[test]
    fn saturating_arithmetic_no_panic_at_extreme_inputs() {
        // u32::MAX × 5 overflows u32 but fits in u64 with headroom.
        let cost = cost_units(
            CostInputs {
                top_k: u32::MAX,
                expand_entities: true,
            },
            CostParams::default(),
        );
        assert_eq!(cost, u32::MAX as u64 * DEFAULT_E_EXPAND as u64);
    }
}
