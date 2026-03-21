use rusqlite::params;

use crate::error::Result;
use crate::types::{DomainCount, EntityProfile, Memory};

use super::{now, YantrikDB};

impl YantrikDB {
    /// Memories not accessed in `days`, still active, ordered by importance.
    pub fn stale(
        &self,
        days: f64,
        limit: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<Memory>> {
        let ts = now();
        let cutoff = ts - days * 86400.0;

        let (sql, param_values) = if let Some(ns) = namespace {
            (
                "SELECT rid, type, text, created_at, importance, valence, half_life, \
                 last_access, access_count, consolidation_status, storage_tier, \
                 consolidated_into, metadata, namespace, certainty, domain, source, \
                 emotional_state, session_id, due_at, temporal_kind \
                 FROM memories \
                 WHERE consolidation_status = 'active' \
                 AND last_access < ?1 AND importance >= 0.3 AND namespace = ?2 \
                 ORDER BY importance DESC LIMIT ?3"
                    .to_string(),
                vec![
                    Box::new(cutoff) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(ns.to_string()) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(limit as i64) as Box<dyn rusqlite::types::ToSql>,
                ],
            )
        } else {
            (
                "SELECT rid, type, text, created_at, importance, valence, half_life, \
                 last_access, access_count, consolidation_status, storage_tier, \
                 consolidated_into, metadata, namespace, certainty, domain, source, \
                 emotional_state, session_id, due_at, temporal_kind \
                 FROM memories \
                 WHERE consolidation_status = 'active' \
                 AND last_access < ?1 AND importance >= 0.3 \
                 ORDER BY importance DESC LIMIT ?2"
                    .to_string(),
                vec![
                    Box::new(cutoff) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(limit as i64) as Box<dyn rusqlite::types::ToSql>,
                ],
            )
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let rows = stmt
            .query_map(params_ref.as_slice(), |row| row_to_memory(row))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Decrypt text and metadata
        let mut memories = Vec::with_capacity(rows.len());
        for mut m in rows {
            m.text = self.decrypt_text(&m.text)?;
            let meta_str = self.decrypt_text(&serde_json::to_string(&m.metadata)?)?;
            m.metadata = serde_json::from_str(&meta_str)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            memories.push(m);
        }

        Ok(memories)
    }

    /// Memories with due_at within `days` from now.
    pub fn upcoming(
        &self,
        days: f64,
        limit: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<Memory>> {
        let ts = now();
        let end = ts + days * 86400.0;

        let (sql, param_values) = if let Some(ns) = namespace {
            (
                "SELECT rid, type, text, created_at, importance, valence, half_life, \
                 last_access, access_count, consolidation_status, storage_tier, \
                 consolidated_into, metadata, namespace, certainty, domain, source, \
                 emotional_state, session_id, due_at, temporal_kind \
                 FROM memories \
                 WHERE consolidation_status = 'active' \
                 AND due_at IS NOT NULL AND due_at BETWEEN ?1 AND ?2 \
                 AND namespace = ?3 \
                 ORDER BY due_at ASC LIMIT ?4"
                    .to_string(),
                vec![
                    Box::new(ts) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(end) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(ns.to_string()) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(limit as i64) as Box<dyn rusqlite::types::ToSql>,
                ],
            )
        } else {
            (
                "SELECT rid, type, text, created_at, importance, valence, half_life, \
                 last_access, access_count, consolidation_status, storage_tier, \
                 consolidated_into, metadata, namespace, certainty, domain, source, \
                 emotional_state, session_id, due_at, temporal_kind \
                 FROM memories \
                 WHERE consolidation_status = 'active' \
                 AND due_at IS NOT NULL AND due_at BETWEEN ?1 AND ?2 \
                 ORDER BY due_at ASC LIMIT ?3"
                    .to_string(),
                vec![
                    Box::new(ts) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(end) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(limit as i64) as Box<dyn rusqlite::types::ToSql>,
                ],
            )
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let rows = stmt
            .query_map(params_ref.as_slice(), |row| row_to_memory(row))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut memories = Vec::with_capacity(rows.len());
        for mut m in rows {
            m.text = self.decrypt_text(&m.text)?;
            let meta_str = self.decrypt_text(&serde_json::to_string(&m.metadata)?)?;
            m.metadata = serde_json::from_str(&meta_str)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            memories.push(m);
        }

        Ok(memories)
    }

    /// Rich entity profile: valence, sessions, domains, frequency.
    pub fn entity_profile(
        &self,
        entity: &str,
        days: f64,
        namespace: Option<&str>,
    ) -> Result<EntityProfile> {
        let ts = now();
        let cutoff = ts - days * 86400.0;

        // Get entity metadata
        let (entity_type, first_seen, last_seen, total_mentions): (String, f64, f64, i64) =
            self.conn.query_row(
                "SELECT entity_type, first_seen, last_seen, mention_count \
                 FROM entities WHERE name = ?1",
                params![entity],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )?;

        // Get memories linked to this entity within window
        let ns_clause = if namespace.is_some() {
            " AND m.namespace = ?3"
        } else {
            ""
        };

        let sql = format!(
            "SELECT m.valence, m.created_at, m.domain, m.emotional_state, m.session_id \
             FROM memories m \
             JOIN memory_entities me ON me.memory_rid = m.rid \
             WHERE me.entity_name = ?1 AND m.created_at >= ?2 \
             AND m.consolidation_status = 'active'{ns_clause} \
             ORDER BY m.created_at"
        );

        let mut stmt = self.conn.prepare(&sql)?;

        struct Row {
            valence: f64,
            created_at: f64,
            domain: String,
            emotional_state: Option<String>,
            session_id: Option<String>,
        }

        let map_row = |row: &rusqlite::Row<'_>| -> rusqlite::Result<Row> {
            Ok(Row {
                valence: row.get(0)?,
                created_at: row.get(1)?,
                domain: row.get(2)?,
                emotional_state: row.get(3)?,
                session_id: row.get(4)?,
            })
        };

        let rows: Vec<Row> = if let Some(ns) = namespace {
            stmt.query_map(params![entity, cutoff, ns], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(params![entity, cutoff], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        let mention_count = rows.len() as i64;

        // Domains
        let mut domain_map: std::collections::HashMap<String, i64> =
            std::collections::HashMap::new();
        for r in &rows {
            *domain_map.entry(r.domain.clone()).or_insert(0) += 1;
        }
        let mut domains: Vec<DomainCount> = domain_map
            .into_iter()
            .map(|(domain, count)| DomainCount { domain, count })
            .collect();
        domains.sort_by(|a, b| b.count.cmp(&a.count));

        // Average valence
        let avg_valence = if mention_count > 0 {
            rows.iter().map(|r| r.valence).sum::<f64>() / mention_count as f64
        } else {
            0.0
        };

        // Valence trend: recent half vs older half
        let valence_trend = if rows.len() >= 4 {
            let mid = rows.len() / 2;
            let older_avg =
                rows[..mid].iter().map(|r| r.valence).sum::<f64>() / mid as f64;
            let recent_avg =
                rows[mid..].iter().map(|r| r.valence).sum::<f64>() / (rows.len() - mid) as f64;
            recent_avg - older_avg
        } else {
            0.0
        };

        // Session count
        let mut session_set: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for r in &rows {
            if let Some(ref sid) = r.session_id {
                session_set.insert(sid.clone());
            }
        }
        let session_count = session_set.len() as i64;

        // Dominant emotion
        let mut emotion_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for r in &rows {
            if let Some(ref e) = r.emotional_state {
                *emotion_counts.entry(e.clone()).or_insert(0) += 1;
            }
        }
        let dominant_emotion = emotion_counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(e, _)| e);

        // Interaction frequency (mentions per day)
        let window_days = days;
        let interaction_frequency = if window_days > 0.0 {
            mention_count as f64 / window_days
        } else {
            0.0
        };

        Ok(EntityProfile {
            entity: entity.to_string(),
            entity_type,
            mention_count,
            session_count,
            domains,
            avg_valence,
            valence_trend,
            dominant_emotion,
            interaction_frequency,
            last_mentioned_at: last_seen,
            first_seen,
            window_days,
        })
    }
}

fn row_to_memory(row: &rusqlite::Row<'_>) -> rusqlite::Result<Memory> {
    let meta_str: String = row.get(12)?;
    Ok(Memory {
        rid: row.get(0)?,
        memory_type: row.get(1)?,
        text: row.get(2)?,
        created_at: row.get(3)?,
        importance: row.get(4)?,
        valence: row.get(5)?,
        half_life: row.get(6)?,
        last_access: row.get(7)?,
        access_count: row.get::<_, i64>(8)? as u32,
        consolidation_status: row.get(9)?,
        storage_tier: row.get(10)?,
        consolidated_into: row.get(11)?,
        metadata: serde_json::from_str(&meta_str)
            .unwrap_or(serde_json::Value::Object(Default::default())),
        namespace: row.get(13)?,
        certainty: row.get(14)?,
        domain: row.get(15)?,
        source: row.get(16)?,
        emotional_state: row.get(17)?,
        session_id: row.get(18)?,
        due_at: row.get(19)?,
        temporal_kind: row.get(20)?,
    })
}
