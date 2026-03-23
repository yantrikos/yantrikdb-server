//! Engine-level probabilistic belief network API.
//!
//! Wires the belief_network module into `YantrikDB`, persisting
//! the factor graph and providing integrated inference queries.

use crate::belief_network::{
    BeliefNetwork, BPConfig, BPResult, InferenceQuery, InferenceResult,
    NetworkHealth, EdgeRelation,
    loopy_belief_propagation, query, information_gain, sensitivity_to_evidence,
    network_diagnostics, build_network_from_edges, most_probable_explanation,
};
use crate::belief_network::VariableId;
use crate::error::Result;
use crate::state::NodeId;

use super::YantrikDB;

const BELIEF_NETWORK_META_KEY: &str = "belief_network";
const BP_CONFIG_META_KEY: &str = "bp_config";

impl YantrikDB {
    // ── Persistence ──

    /// Load the belief network from the database.
    ///
    /// After deserialization the adjacency indices are rebuilt
    /// because they carry `#[serde(skip)]`.
    pub fn load_belief_network(&self) -> Result<BeliefNetwork> {
        match Self::get_meta(&self.conn(), BELIEF_NETWORK_META_KEY)? {
            Some(json) => {
                let mut net: BeliefNetwork = serde_json::from_str(&json).map_err(|e| {
                    crate::error::YantrikDbError::Database(
                        rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                    )
                })?;
                net.rebuild_indices();
                Ok(net)
            }
            None => Ok(BeliefNetwork::new()),
        }
    }

    /// Persist the belief network.
    pub fn save_belief_network(&self, network: &BeliefNetwork) -> Result<()> {
        let json = serde_json::to_string(network).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![BELIEF_NETWORK_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load belief-propagation configuration.
    pub fn load_bp_config(&self) -> Result<BPConfig> {
        match Self::get_meta(&self.conn(), BP_CONFIG_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(BPConfig::default()),
        }
    }

    /// Persist belief-propagation configuration.
    pub fn save_bp_config(&self, config: &BPConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![BP_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    // ── API ──

    /// Run loopy belief propagation on the persisted network.
    ///
    /// Loads the network, runs BP, saves the updated posteriors, and
    /// returns the convergence report.
    pub fn run_belief_propagation(&self) -> Result<BPResult> {
        let mut network = self.load_belief_network()?;
        let config = self.load_bp_config()?;
        let result = loopy_belief_propagation(&mut network, &config);
        self.save_belief_network(&network)?;
        Ok(result)
    }

    /// Run an inference query on the persisted network.
    ///
    /// Internally runs BP if needed, then computes the posterior
    /// for the target variable given evidence.
    pub fn run_belief_query(&self, inference: &InferenceQuery) -> Result<InferenceResult> {
        let mut network = self.load_belief_network()?;
        let config = self.load_bp_config()?;
        let result = query(&mut network, inference, &config);
        // Save updated posteriors after inference.
        self.save_belief_network(&network)?;
        Ok(result)
    }

    /// Compute the information gain of observing `candidate` on `target`.
    pub fn belief_information_gain(
        &self,
        candidate: VariableId,
        target: VariableId,
    ) -> Result<f64> {
        let mut network = self.load_belief_network()?;
        let config = self.load_bp_config()?;
        Ok(information_gain(&mut network, candidate, target, &config))
    }

    /// Rank all variables by their sensitivity (impact on `target`).
    pub fn belief_sensitivity(
        &self,
        target: VariableId,
    ) -> Result<Vec<(VariableId, f64)>> {
        let mut network = self.load_belief_network()?;
        let config = self.load_bp_config()?;
        Ok(sensitivity_to_evidence(&mut network, target, &config))
    }

    /// Most probable explanation for given evidence.
    pub fn belief_mpe(
        &self,
        evidence: &[(VariableId, f64)],
    ) -> Result<Vec<(VariableId, f64)>> {
        let mut network = self.load_belief_network()?;
        let config = self.load_bp_config()?;
        Ok(most_probable_explanation(&mut network, evidence, &config))
    }

    /// Get diagnostic health report for the persisted network.
    pub fn belief_network_health(&self) -> Result<NetworkHealth> {
        let network = self.load_belief_network()?;
        Ok(network_diagnostics(&network))
    }

    /// Build a belief network from cognitive graph edges and persist it.
    ///
    /// This bridges the cognitive graph (NodeId-based) into the factor
    /// graph representation used for probabilistic inference.
    pub fn build_and_save_belief_network(
        &self,
        beliefs: &[(NodeId, &str, f64)],
        edges: &[(NodeId, NodeId, EdgeRelation, f64)],
    ) -> Result<BeliefNetwork> {
        let network = build_network_from_edges(beliefs, edges);
        self.save_belief_network(&network)?;
        Ok(network)
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::belief_network::BPConfig;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_belief_network_save_load_roundtrip() {
        let db = test_db();
        let net = db.load_belief_network().unwrap();
        assert_eq!(net.variable_count(), 0);
        db.save_belief_network(&net).unwrap();
        let loaded = db.load_belief_network().unwrap();
        assert_eq!(loaded.variable_count(), 0);
    }

    #[test]
    fn test_bp_config_save_load() {
        let db = test_db();
        let config = db.load_bp_config().unwrap();
        assert_eq!(config.max_iterations, 30);
        db.save_bp_config(&config).unwrap();
        let loaded = db.load_bp_config().unwrap();
        assert_eq!(loaded.max_iterations, 30);
    }

    #[test]
    fn test_belief_network_health_empty() {
        let db = test_db();
        let health = db.belief_network_health().unwrap();
        assert_eq!(health.variable_count, 0);
        assert_eq!(health.factor_count, 0);
        assert!(health.healthy);
    }

    #[test]
    fn test_run_belief_propagation_empty() {
        let db = test_db();
        // Empty network should converge immediately.
        let result = db.run_belief_propagation().unwrap();
        assert!(result.converged);
    }
}
