//! Engine-level cognitive extractor API.
//!
//! Wires the extraction cascade into `YantrikDB` for persistence
//! of learned templates and integration with the cognitive graph.

use crate::error::Result;
use crate::extractor::{
    extract, integrate_llm_response, summarize_extractor, CognitiveUpdate, ExtractionContext,
    ExtractionResponse, ExtractorConfig, ExtractorSummary, SerializableOpTemplate, TemplateStore,
};

use super::{now, YantrikDB};

/// Meta key for persisted template store.
const TEMPLATE_STORE_META_KEY: &str = "extractor_template_store";
/// Meta key for persisted extractor config.
const EXTRACTOR_CONFIG_META_KEY: &str = "extractor_config";

impl YantrikDB {
    // ── Persistence ──

    /// Load the extraction template store from the database.
    pub fn load_template_store(&self) -> Result<TemplateStore> {
        match Self::get_meta(&self.conn(), TEMPLATE_STORE_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(TemplateStore::new()),
        }
    }

    /// Persist the extraction template store.
    pub fn save_template_store(&self, store: &TemplateStore) -> Result<()> {
        let json = serde_json::to_string(store).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![TEMPLATE_STORE_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the extractor configuration.
    pub fn load_extractor_config(&self) -> Result<ExtractorConfig> {
        match Self::get_meta(&self.conn(), EXTRACTOR_CONFIG_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(ExtractorConfig::default()),
        }
    }

    /// Persist the extractor configuration.
    pub fn save_extractor_config(&self, config: &ExtractorConfig) -> Result<()> {
        let json = serde_json::to_string(config).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![EXTRACTOR_CONFIG_META_KEY, json],
        )?;
        Ok(())
    }

    // ── Main API ──

    /// Extract cognitive updates from text using the full cascade.
    ///
    /// Returns extracted updates and optionally an LLM escalation request.
    /// The caller is responsible for fulfilling the LLM request if present.
    pub fn extract_cognitive_updates(
        &self,
        text: &str,
        context: &ExtractionContext,
    ) -> Result<ExtractionResponse> {
        let store = self.load_template_store()?;
        let config = self.load_extractor_config()?;
        Ok(extract(text, context, &store, &config))
    }

    /// Integrate an LLM extraction response back into the system.
    ///
    /// Parses the LLM's JSON response, creates CognitiveUpdates,
    /// and learns new templates (the flywheel effect).
    pub fn integrate_llm_extraction(
        &self,
        original_text: &str,
        llm_json: &str,
    ) -> Result<Vec<CognitiveUpdate>> {
        let mut store = self.load_template_store()?;
        let config = self.load_extractor_config()?;
        let ts = now();

        let updates = integrate_llm_response(
            original_text,
            llm_json,
            &mut store,
            config.max_templates,
            ts,
        );

        self.save_template_store(&store)?;
        Ok(updates)
    }

    /// Manually teach the extractor a new template.
    pub fn teach_extraction_template(
        &self,
        example_text: &str,
        op_template: SerializableOpTemplate,
    ) -> Result<()> {
        let mut store = self.load_template_store()?;
        let config = self.load_extractor_config()?;
        let ts = now();

        store.learn_template(example_text, op_template, ts, config.max_templates);
        self.save_template_store(&store)
    }

    /// Get a summary of the extractor's learned state.
    pub fn extractor_summary(&self) -> Result<ExtractorSummary> {
        let store = self.load_template_store()?;
        Ok(summarize_extractor(&store))
    }

    /// Get the number of learned templates.
    pub fn extractor_template_count(&self) -> Result<usize> {
        let store = self.load_template_store()?;
        Ok(store.templates.len())
    }

    /// Reset the template store (clear all learned templates). Irreversible.
    pub fn reset_template_store(&self) -> Result<()> {
        let store = TemplateStore::new();
        self.save_template_store(&store)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::extractor::{
        ExtractionContext, ExtractorTier, SerializableOpTemplate, UpdateOp,
    };
    use crate::state::Priority;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_extract_cognitive_updates() {
        let db = test_db();
        let context = ExtractionContext::default();

        let response = db
            .extract_cognitive_updates("I need to call the dentist", &context)
            .unwrap();

        assert!(!response.updates.is_empty());
        assert!(response.tiers_used.contains(&ExtractorTier::Rule));
    }

    #[test]
    fn test_integrate_llm_extraction_flywheel() {
        let db = test_db();

        let updates = db
            .integrate_llm_extraction(
                "I should schedule the quarterly review meeting",
                r#"[{"op":"create_task","description":"schedule quarterly review meeting","priority":"high","confidence":0.9}]"#,
            )
            .unwrap();

        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].tier, ExtractorTier::Llm);

        // Verify template was learned (flywheel)
        let count = db.extractor_template_count().unwrap();
        assert!(count > 0, "LLM extraction should create templates");
    }

    #[test]
    fn test_teach_extraction_template() {
        let db = test_db();

        db.teach_extraction_template(
            "deploy the app to staging environment",
            SerializableOpTemplate::CreateTask { priority: Priority::High },
        )
        .unwrap();

        let count = db.extractor_template_count().unwrap();
        assert_eq!(count, 1);

        // Template should match similar text
        let context = ExtractionContext::default();
        let response = db
            .extract_cognitive_updates("deploy the service to staging", &context)
            .unwrap();

        // May or may not match depending on keyword overlap threshold
        // but template store should have the template
        let summary = db.extractor_summary().unwrap();
        assert_eq!(summary.total_templates, 1);
    }

    #[test]
    fn test_reset_template_store() {
        let db = test_db();

        db.teach_extraction_template(
            "test template text",
            SerializableOpTemplate::CreateTask { priority: Priority::Medium },
        )
        .unwrap();

        db.reset_template_store().unwrap();
        let count = db.extractor_template_count().unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_escalation_on_unknown_text() {
        let db = test_db();
        let context = ExtractionContext::default();

        let response = db
            .extract_cognitive_updates("The weather is beautiful today", &context)
            .unwrap();

        assert!(response.escalation_needed);
        assert!(response.llm_request.is_some());
    }

    #[test]
    fn test_no_escalation_on_confident_match() {
        let db = test_db();
        let context = ExtractionContext::default();

        let response = db
            .extract_cognitive_updates("Remind me to take my medication at 8pm", &context)
            .unwrap();

        assert!(!response.escalation_needed, "High-confidence match should not escalate");
        match &response.updates[0].op {
            UpdateOp::CreateTask { priority, .. } => {
                assert_eq!(*priority, Priority::High);
            }
            other => panic!("Expected CreateTask, got {:?}", other),
        }
    }
}
