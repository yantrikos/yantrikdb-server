use rusqlite::params;

use crate::error::Result;
use crate::serde_helpers::serialize_f32;

use super::{now, YantrikDB};

impl YantrikDB {
    /// Archive a hot memory to cold storage (compress embedding, remove from vec index).
    /// Returns true if the memory was archived, false if not found or already cold.
    #[tracing::instrument(skip(self))]
    pub fn archive(&self, rid: &str) -> Result<bool> {
        let ts = {
            let conn = self.conn();
            let row = conn.query_row(
                "SELECT embedding FROM memories WHERE rid = ?1 AND storage_tier = 'hot' AND consolidation_status = 'active'",
                params![rid],
                |row| row.get::<_, Vec<u8>>(0),
            );

            let stored_blob = match row {
                Ok(blob) => blob,
                Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(false),
                Err(e) => return Err(e.into()),
            };

            // Decrypt if encrypted, then compress, then re-encrypt for cold storage
            let raw_blob = self.decrypt_embedding(&stored_blob)?;
            let embedding = crate::serde_helpers::deserialize_f32(&raw_blob);
            let compressed = crate::compression::compress_embedding(&embedding);
            let stored_compressed = self.encrypt_embedding(&compressed)?;
            let ts = now();

            conn.execute(
                "UPDATE memories SET storage_tier = 'cold', embedding = ?1, updated_at = ?2 WHERE rid = ?3",
                params![stored_compressed, ts, rid],
            )?;

            ts
        }; // drop conn before acquiring vec_index write lock

        self.vec_index.write().unwrap().remove(rid);

        self.log_op(
            "archive",
            Some(rid),
            &serde_json::json!({
                "rid": rid,
                "updated_at": ts,
            }),
            None,
        )?;

        Ok(true)
    }

    /// Hydrate a cold memory back to hot storage (decompress embedding, re-insert into vec index).
    /// Returns true if the memory was hydrated, false if not found or already hot.
    #[tracing::instrument(skip(self))]
    pub fn hydrate(&self, rid: &str) -> Result<bool> {
        let (ts, embedding) = {
            let conn = self.conn();
            let row = conn.query_row(
                "SELECT embedding FROM memories WHERE rid = ?1 AND storage_tier = 'cold'",
                params![rid],
                |row| row.get::<_, Vec<u8>>(0),
            );

            let stored_blob = match row {
                Ok(blob) => blob,
                Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(false),
                Err(e) => return Err(e.into()),
            };

            // Decrypt if encrypted, decompress, then re-encrypt for hot storage
            let compressed_blob = self.decrypt_embedding(&stored_blob)?;
            let embedding = crate::compression::decompress_embedding(&compressed_blob);
            let raw_blob = serialize_f32(&embedding);
            let stored_raw = self.encrypt_embedding(&raw_blob)?;
            let ts = now();

            conn.execute(
                "UPDATE memories SET storage_tier = 'hot', embedding = ?1, updated_at = ?2 WHERE rid = ?3",
                params![stored_raw, ts, rid],
            )?;

            (ts, embedding)
        }; // drop conn before acquiring vec_index write lock

        self.vec_index.write().unwrap().insert(rid, &embedding)?;

        self.log_op(
            "hydrate",
            Some(rid),
            &serde_json::json!({
                "rid": rid,
                "updated_at": ts,
            }),
            None,
        )?;

        Ok(true)
    }

    /// Evict memories to cold storage based on decay scores.
    /// Archives the lowest-scoring memories until at most `max_active` hot memories remain.
    /// Returns the list of archived RIDs.
    #[tracing::instrument(skip(self))]
    pub fn evict(&self, max_active: usize) -> Result<Vec<String>> {
        let (mut scored, to_evict) = {
            let conn = self.conn();
            let hot_count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'active' AND storage_tier = 'hot'",
                [],
                |row| row.get(0),
            )?;

            if hot_count as usize <= max_active {
                return Ok(vec![]);
            }

            let to_evict = hot_count as usize - max_active;
            let ts = now();

            let mut stmt = conn.prepare(
                "SELECT rid, importance, half_life, last_access, created_at FROM memories \
                 WHERE consolidation_status = 'active' AND storage_tier = 'hot'",
            )?;

            let scored: Vec<(String, f64)> = stmt
                .query_map([], |row| {
                    let rid: String = row.get("rid")?;
                    let importance: f64 = row.get("importance")?;
                    let half_life: f64 = row.get("half_life")?;
                    let last_access: f64 = row.get("last_access")?;
                    let created_at: f64 = row.get("created_at")?;
                    let elapsed = ts - last_access;
                    let decay = crate::scoring::decay_score(importance, half_life, elapsed);
                    let age = ts - created_at;
                    let recency = crate::scoring::recency_score(age);
                    let score = crate::scoring::eviction_score(decay, recency);
                    Ok((rid, score))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            (scored, to_evict)
        }; // drop conn before archive() which re-acquires it

        // Sort ascending — lowest score = most evictable
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut archived_rids = Vec::new();
        for (rid, _) in scored.into_iter().take(to_evict) {
            if self.archive(&rid)? {
                archived_rids.push(rid);
            }
        }

        Ok(archived_rids)
    }
}
