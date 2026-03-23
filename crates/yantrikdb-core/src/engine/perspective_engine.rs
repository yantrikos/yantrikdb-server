//! Engine-level perspective engine API.
//!
//! Wires the perspective module into `YantrikDB`, persisting the
//! perspective store and stack, and exposing context-dependent
//! salience/weight resolution.

use crate::perspective::{
    PerspectiveId, PerspectiveStore, PerspectiveStack, Perspective,
    PerspectiveTransition, PerspectiveConflict, ActivationContext,
    CognitiveStyle,
    activate_perspective, deactivate_perspective, resolve_salience,
    resolve_edge_weight, resolve_cognitive_style, detect_perspective_shift,
    perspective_conflict_check, create_preset,
};
use crate::error::Result;
use crate::state::{NodeId, NodeKind, CognitiveEdgeKind};

use super::YantrikDB;

const PERSPECTIVE_STORE_META_KEY: &str = "perspective_store";
const PERSPECTIVE_STACK_META_KEY: &str = "perspective_stack";

impl YantrikDB {
    // ── Persistence ──

    /// Load the perspective store from the database.
    pub fn load_perspective_store(&self) -> Result<PerspectiveStore> {
        match Self::get_meta(&self.conn(), PERSPECTIVE_STORE_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(PerspectiveStore::new()),
        }
    }

    /// Persist the perspective store.
    pub fn save_perspective_store(&self, store: &PerspectiveStore) -> Result<()> {
        let json = serde_json::to_string(store).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![PERSPECTIVE_STORE_META_KEY, json],
        )?;
        Ok(())
    }

    /// Load the active perspective stack from the database.
    pub fn load_perspective_stack(&self) -> Result<PerspectiveStack> {
        match Self::get_meta(&self.conn(), PERSPECTIVE_STACK_META_KEY)? {
            Some(json) => serde_json::from_str(&json).map_err(|e| {
                crate::error::YantrikDbError::Database(
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                )
            }),
            None => Ok(PerspectiveStack::new()),
        }
    }

    /// Persist the active perspective stack.
    pub fn save_perspective_stack(&self, stack: &PerspectiveStack) -> Result<()> {
        let json = serde_json::to_string(stack).map_err(|e| {
            crate::error::YantrikDbError::Database(
                rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            )
        })?;
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![PERSPECTIVE_STACK_META_KEY, json],
        )?;
        Ok(())
    }

    // ── API ──

    /// Activate a perspective by pushing it onto the stack.
    pub fn activate_perspective(&self, id: PerspectiveId) -> Result<()> {
        let store = self.load_perspective_store()?;
        let mut stack = self.load_perspective_stack()?;
        activate_perspective(&mut stack, id, &store);
        self.save_perspective_stack(&stack)?;
        Ok(())
    }

    /// Deactivate a perspective by removing it from the stack.
    pub fn deactivate_perspective(&self, id: PerspectiveId) -> Result<()> {
        let store = self.load_perspective_store()?;
        let mut stack = self.load_perspective_stack()?;
        deactivate_perspective(&mut stack, id, &store);
        self.save_perspective_stack(&stack)?;
        Ok(())
    }

    /// Resolve the effective salience of a node given the active perspectives.
    pub fn resolve_node_salience(
        &self,
        node_id: NodeId,
        node_kind: NodeKind,
        node_domain: Option<&str>,
        node_tags: &[String],
        base_salience: f64,
    ) -> Result<f64> {
        let store = self.load_perspective_store()?;
        let stack = self.load_perspective_stack()?;
        Ok(resolve_salience(
            &stack, node_id, node_kind, node_domain, node_tags, base_salience, &store,
        ))
    }

    /// Resolve the effective weight of an edge given the active perspectives.
    pub fn resolve_edge_weight_perspective(
        &self,
        edge_kind: CognitiveEdgeKind,
        base_weight: f64,
    ) -> Result<f64> {
        let store = self.load_perspective_store()?;
        let stack = self.load_perspective_stack()?;
        Ok(resolve_edge_weight(&stack, edge_kind, base_weight, &store))
    }

    /// Get the blended cognitive style from the active perspective stack.
    pub fn active_cognitive_style(&self) -> Result<CognitiveStyle> {
        let store = self.load_perspective_store()?;
        let stack = self.load_perspective_stack()?;
        Ok(resolve_cognitive_style(&stack, &store))
    }

    /// Detect whether the current context warrants a perspective shift.
    pub fn detect_perspective_shifts(
        &self,
        ctx: &ActivationContext,
    ) -> Result<Vec<PerspectiveTransition>> {
        let store = self.load_perspective_store()?;
        let stack = self.load_perspective_stack()?;
        Ok(detect_perspective_shift(ctx, &stack, &store))
    }

    /// Check for conflicts among the currently active perspectives.
    pub fn check_perspective_conflicts(&self) -> Result<Vec<PerspectiveConflict>> {
        let store = self.load_perspective_store()?;
        let stack = self.load_perspective_stack()?;
        Ok(perspective_conflict_check(&stack, &store))
    }

    /// Create a built-in preset perspective and insert it into the store.
    ///
    /// Available presets: `"creative"`, `"deadline"`, `"reflective"`.
    pub fn create_preset_perspective(
        &self,
        name: &str,
        now_ms: u64,
    ) -> Result<Option<PerspectiveId>> {
        let mut store = self.load_perspective_store()?;
        let id = store.alloc_id();
        if let Some(preset) = create_preset(name, id, now_ms) {
            let pid = preset.id;
            store.insert(preset);
            self.save_perspective_store(&store)?;
            Ok(Some(pid))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_perspective_store_save_load_roundtrip() {
        let db = test_db();
        let store = db.load_perspective_store().unwrap();
        assert!(store.is_empty());
        db.save_perspective_store(&store).unwrap();
        let loaded = db.load_perspective_store().unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_perspective_stack_save_load_roundtrip() {
        let db = test_db();
        let stack = db.load_perspective_stack().unwrap();
        assert_eq!(stack.depth(), 0);
        db.save_perspective_stack(&stack).unwrap();
        let loaded = db.load_perspective_stack().unwrap();
        assert_eq!(loaded.depth(), 0);
    }

    #[test]
    fn test_create_preset_perspective() {
        let db = test_db();
        let id = db.create_preset_perspective("creative", 1_000_000).unwrap();
        assert!(id.is_some());
        let store = db.load_perspective_store().unwrap();
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_create_preset_unknown() {
        let db = test_db();
        let id = db.create_preset_perspective("nonexistent", 1_000_000).unwrap();
        assert!(id.is_none());
    }

    #[test]
    fn test_active_cognitive_style_empty() {
        let db = test_db();
        let style = db.active_cognitive_style().unwrap();
        // Default style from empty stack.
        assert!(style.exploration_vs_exploitation >= 0.0);
    }

    #[test]
    fn test_no_conflicts_empty_stack() {
        let db = test_db();
        let conflicts = db.check_perspective_conflicts().unwrap();
        assert!(conflicts.is_empty());
    }
}
