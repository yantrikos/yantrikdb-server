//! Engine-level schema induction API.
//!
//! Wires the schema induction module into `YantrikDB` for persistence.

use crate::error::Result;
use crate::schema_induction::{
    SchemaId, SchemaMaintenanceReport, SchemaStore,
    match_schemas, observe_episode, schema_maintenance,
    EpisodeData, ContextSnapshot,
};

use super::YantrikDB;

const SCHEMA_STORE_META_KEY: &str = "induced_schema_store";

impl YantrikDB {
    // ── Persistence ──

    /// Load the induced schema store.
    pub fn load_induced_schema_store(&self) -> Result<SchemaStore> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), SCHEMA_STORE_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(SchemaStore::default()),
        }
    }

    /// Persist the induced schema store.
    pub fn save_induced_schema_store(&self, store: &SchemaStore) -> Result<()> {
        let json = serde_json::to_string(store).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![SCHEMA_STORE_META_KEY, json],
        )?;
        Ok(())
    }

    // ── API ──

    /// Observe an episode and update schemas.
    pub fn observe_episode_for_schema(
        &self,
        episode: &EpisodeData,
    ) -> Result<()> {
        let mut store = self.load_induced_schema_store()?;
        observe_episode(episode, &mut store);
        self.save_induced_schema_store(&store)?;
        Ok(())
    }

    /// Find schemas matching a given context snapshot.
    pub fn find_matching_schemas(
        &self,
        context: &ContextSnapshot,
    ) -> Result<Vec<(SchemaId, f64)>> {
        let store = self.load_induced_schema_store()?;
        Ok(match_schemas(context, &store))
    }

    /// Run schema maintenance (pruning weak schemas).
    pub fn run_schema_maintenance(
        &self,
        now_ms: u64,
        max_age_ms: u64,
    ) -> Result<SchemaMaintenanceReport> {
        let mut store = self.load_induced_schema_store()?;
        let report = schema_maintenance(&mut store, now_ms, max_age_ms);
        self.save_induced_schema_store(&store)?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_schema_store_save_load_roundtrip() {
        let db = test_db();
        let store = db.load_induced_schema_store().unwrap();
        assert!(store.is_empty());
        db.save_induced_schema_store(&store).unwrap();
        let loaded = db.load_induced_schema_store().unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_schema_store_default_on_missing() {
        let db = test_db();
        let store = db.load_induced_schema_store().unwrap();
        assert!(store.is_empty());
    }
}
