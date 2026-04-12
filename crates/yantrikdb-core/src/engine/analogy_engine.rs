//! Engine-level analogical reasoning API.
//!
//! Wires the analogy module into `YantrikDB` for persistence
//! and integration with the cognitive pipeline.

use crate::analogy::{
    AnalogyStore, AnalogicalQuery, AnalogicalOpportunity, AnalogyMaintenanceReport,
    StructuralMapping, SubgraphGroup, TransferredStrategy,
    analogy_strength_decay, detect_analogical_opportunities, find_analogies,
    transfer_strategy,
};
use crate::error::Result;
use crate::state::CognitiveNode;

use super::YantrikDB;

const ANALOGY_STORE_META_KEY: &str = "analogy_store";

impl YantrikDB {
    // ── Persistence ──

    /// Load the analogy store from the database.
    pub fn load_analogy_store(&self) -> Result<AnalogyStore> {
        // Scope the conn guard to the get_meta call so it drops before
        // the match body runs. Without this, arms that call self.*
        // methods (which re-acquire conn) will self-deadlock.
        let meta = Self::get_meta(&self.conn(), ANALOGY_STORE_META_KEY)?;
        match meta {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(AnalogyStore::default()),
        }
    }

    /// Persist the analogy store.
    pub fn save_analogy_store(&self, store: &AnalogyStore) -> Result<()> {
        let json = serde_json::to_string(store).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![ANALOGY_STORE_META_KEY, json],
        )?;
        Ok(())
    }

    // ── API ──

    /// Find analogies for a query against candidate subgraph groups.
    pub fn find_analogies_in_graph(
        &self,
        query: &AnalogicalQuery,
        source_node_data: &[&CognitiveNode],
        candidate_groups: &[SubgraphGroup],
    ) -> Result<Vec<StructuralMapping>> {
        Ok(find_analogies(query, source_node_data, candidate_groups))
    }

    /// Detect analogical opportunities across subgraph groups.
    pub fn detect_analogies(
        &self,
        recent_nodes: &[&CognitiveNode],
        recent_edges: &[crate::state::CognitiveEdge],
        groups: &[SubgraphGroup],
        min_quality: f64,
    ) -> Result<Vec<AnalogicalOpportunity>> {
        Ok(detect_analogical_opportunities(recent_nodes, recent_edges, groups, min_quality))
    }

    /// Transfer strategies from source schemas via an analogy mapping.
    pub fn transfer_strategy_via_analogy(
        &self,
        source_schemas: &[&CognitiveNode],
        mapping: &StructuralMapping,
    ) -> Result<Vec<TransferredStrategy>> {
        Ok(transfer_strategy(source_schemas, mapping))
    }

    /// Run maintenance on the analogy store (decay, pruning).
    pub fn run_analogy_maintenance(
        &self,
        now_ms: u64,
        max_age_ms: u64,
    ) -> Result<AnalogyMaintenanceReport> {
        let mut store = self.load_analogy_store()?;
        let report = analogy_strength_decay(&mut store, now_ms, max_age_ms);
        self.save_analogy_store(&store)?;
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
    fn test_analogy_store_save_load_roundtrip() {
        let db = test_db();
        let store = db.load_analogy_store().unwrap();
        assert!(store.mappings.is_empty());
        db.save_analogy_store(&store).unwrap();
        let loaded = db.load_analogy_store().unwrap();
        assert!(loaded.mappings.is_empty());
    }

    #[test]
    fn test_analogy_store_default_on_missing() {
        let db = test_db();
        let store = db.load_analogy_store().unwrap();
        assert!(store.mappings.is_empty());
    }
}
