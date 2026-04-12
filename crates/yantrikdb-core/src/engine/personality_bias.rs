//! Engine-level personality bias API.
//!
//! Wires the personality bias vector system into `YantrikDB` for
//! persistence and provides convenience methods for bias application.

use crate::error::Result;
use crate::personality_bias::{
    compute_bias, dampen_personality, personality_impact, ActionProperties,
    BiasConfig, BondLevel, PersonalityBiasResult, PersonalityBiasStore,
    PersonalityBiasVector, PersonalityImpactReport, PersonalityPreset,
};

use super::YantrikDB;

/// Meta key for persisted personality bias store.
const PERSONALITY_BIAS_STORE_META_KEY: &str = "personality_bias_store";

impl YantrikDB {
    // ── Persistence ──

    /// Load the personality bias store.
    pub fn load_personality_bias_store(&self) -> Result<PersonalityBiasStore> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), PERSONALITY_BIAS_STORE_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(PersonalityBiasStore::new()),
        }
    }

    /// Persist the personality bias store.
    pub fn save_personality_bias_store(&self, store: &PersonalityBiasStore) -> Result<()> {
        let json = serde_json::to_string(store).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![PERSONALITY_BIAS_STORE_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Personality Bias API ──

    /// Get the current personality bias vector.
    pub fn get_personality_bias(&self) -> Result<PersonalityBiasVector> {
        let store = self.load_personality_bias_store()?;
        Ok(store.current.clone())
    }

    /// Set the personality from a preset.
    pub fn set_personality_preset(&self, preset: PersonalityPreset) -> Result<()> {
        let store = PersonalityBiasStore::from_preset(preset);
        self.save_personality_bias_store(&store)
    }

    /// Set a custom personality bias vector.
    pub fn set_personality_bias(&self, vector: PersonalityBiasVector) -> Result<()> {
        let mut store = self.load_personality_bias_store()?;
        store.current = vector;
        store.base_preset = None;
        self.save_personality_bias_store(&store)
    }

    /// Update the bond level.
    pub fn set_bond_level(&self, bond: BondLevel) -> Result<()> {
        let mut store = self.load_personality_bias_store()?;
        store.set_bond_level(bond);
        self.save_personality_bias_store(&store)
    }

    /// Apply personality bias to an action.
    pub fn apply_personality_bias(
        &self,
        action: &ActionProperties,
    ) -> Result<PersonalityBiasResult> {
        let store = self.load_personality_bias_store()?;
        Ok(store.apply_bias(action))
    }

    /// Record user feedback and evolve personality.
    pub fn record_personality_feedback(
        &self,
        dimension_idx: usize,
        adjustment: f64,
    ) -> Result<()> {
        let mut store = self.load_personality_bias_store()?;
        store.record_feedback(dimension_idx, adjustment, super::now());
        self.save_personality_bias_store(&store)
    }

    /// Generate a personality impact report for candidate actions.
    pub fn personality_impact_report(
        &self,
        actions: &[(&str, &ActionProperties)],
    ) -> Result<PersonalityImpactReport> {
        let store = self.load_personality_bias_store()?;
        let profile_name = store.base_preset
            .map(|p| format!("{:?}", p))
            .unwrap_or_else(|| "Custom".to_string());

        Ok(personality_impact(
            &store.current,
            store.bond_level,
            &profile_name,
            actions,
            &store.bias_config,
        ))
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::personality_bias::{
        ActionProperties, BondLevel, PersonalityBiasStore, PersonalityBiasVector,
        PersonalityPreset,
    };

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_save_load_personality_bias_store() {
        let db = test_db();

        let store = PersonalityBiasStore::from_preset(PersonalityPreset::Companion);
        db.save_personality_bias_store(&store).unwrap();

        let loaded = db.load_personality_bias_store().unwrap();
        assert_eq!(loaded.base_preset, Some(PersonalityPreset::Companion));
        assert!((loaded.current.warmth - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_set_preset() {
        let db = test_db();

        db.set_personality_preset(PersonalityPreset::Guardian).unwrap();

        let bias = db.get_personality_bias().unwrap();
        assert!((bias.caution - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_set_custom_bias() {
        let db = test_db();

        let mut custom = PersonalityBiasVector::neutral();
        custom.curiosity = 0.95;
        db.set_personality_bias(custom).unwrap();

        let loaded = db.get_personality_bias().unwrap();
        assert!((loaded.curiosity - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_set_bond_level() {
        let db = test_db();

        db.set_bond_level(BondLevel::Bonded).unwrap();

        let store = db.load_personality_bias_store().unwrap();
        assert_eq!(store.bond_level, BondLevel::Bonded);
    }

    #[test]
    fn test_apply_bias() {
        let db = test_db();
        db.set_personality_preset(PersonalityPreset::Guardian).unwrap();
        db.set_bond_level(BondLevel::Trusted).unwrap();

        let action = ActionProperties {
            risk: 0.9,
            ..Default::default()
        };
        let result = db.apply_personality_bias(&action).unwrap();

        // Guardian should penalize risky actions.
        assert!(result.total_bias < 0.0);
    }

    #[test]
    fn test_record_feedback() {
        let db = test_db();
        db.set_personality_preset(PersonalityPreset::Assistant).unwrap();
        db.set_bond_level(BondLevel::Bonded).unwrap();

        let original = db.get_personality_bias().unwrap().curiosity;

        // Record enough feedback to trigger evolution.
        for _ in 0..15 {
            db.record_personality_feedback(0, 0.1).unwrap();
        }

        let updated = db.get_personality_bias().unwrap().curiosity;
        assert!(updated > original);
    }

    #[test]
    fn test_impact_report() {
        let db = test_db();
        db.set_personality_preset(PersonalityPreset::Companion).unwrap();
        db.set_bond_level(BondLevel::Familiar).unwrap();

        let action_a = ActionProperties {
            emotional_utility: 0.9,
            ..Default::default()
        };
        let action_b = ActionProperties {
            risk: 0.8,
            ..Default::default()
        };

        let report = db.personality_impact_report(
            &[("Comfort user", &action_a), ("Risky action", &action_b)],
        ).unwrap();

        assert_eq!(report.action_biases.len(), 2);
        assert_eq!(report.profile_name, "Companion");
    }
}
