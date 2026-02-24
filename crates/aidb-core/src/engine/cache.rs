use std::collections::HashMap;

use crate::error::Result;
use crate::types::ScoringRow;

use super::AIDB;

impl AIDB {
    /// Load scoring-relevant fields for all non-tombstoned memories into a HashMap.
    pub(crate) fn load_scoring_cache(
        conn: &rusqlite::Connection,
    ) -> Result<HashMap<String, ScoringRow>> {
        let mut stmt = conn.prepare(
            "SELECT rid, created_at, importance, half_life, last_access, \
             valence, consolidation_status, type, namespace, access_count \
             FROM memories \
             WHERE consolidation_status != 'tombstoned'",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                ScoringRow {
                    created_at: row.get(1)?,
                    importance: row.get(2)?,
                    half_life: row.get(3)?,
                    last_access: row.get(4)?,
                    valence: row.get(5)?,
                    consolidation_status: row.get(6)?,
                    memory_type: row.get(7)?,
                    namespace: row.get(8)?,
                    access_count: row.get::<_, i64>(9)? as u32,
                },
            ))
        })?;

        let mut cache = HashMap::new();
        for row in rows {
            let (rid, scoring_row) = row?;
            cache.insert(rid, scoring_row);
        }
        Ok(cache)
    }

    /// Insert a scoring row into the in-memory cache.
    pub fn cache_insert(&self, rid: String, row: ScoringRow) {
        self.scoring_cache.borrow_mut().insert(rid, row);
    }

    /// Remove a scoring row from the in-memory cache.
    pub fn cache_remove(&self, rid: &str) {
        self.scoring_cache.borrow_mut().remove(rid);
    }

    /// Mark a memory as consolidated in the cache and reduce its importance.
    pub fn cache_mark_consolidated(&self, rid: &str, importance_factor: f64) {
        let mut cache = self.scoring_cache.borrow_mut();
        if let Some(row) = cache.get_mut(rid) {
            row.consolidation_status = "consolidated".to_string();
            row.importance *= importance_factor;
        }
    }
}
