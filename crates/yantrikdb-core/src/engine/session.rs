use rusqlite::params;

use crate::error::{Result, YantrikDbError};
use crate::types::{Session, SessionSummary};

use super::{now, YantrikDB};

impl YantrikDB {
    /// Start a new session. Returns session_id.
    /// Fails if an active session already exists for (namespace, client_id).
    pub fn session_start(
        &self,
        namespace: &str,
        client_id: &str,
        metadata: &serde_json::Value,
    ) -> Result<String> {
        let session_id = crate::id::new_id();
        let ts = now();
        let hlc = self.tick_hlc();
        let hlc_bytes = hlc.to_bytes().to_vec();
        let actor = self.actor_id().to_string();
        let meta_str = serde_json::to_string(metadata)?;

        {
            let conn = self.conn();
            conn.execute(
                "INSERT INTO sessions \
                 (session_id, namespace, client_id, status, started_at, metadata, hlc, origin_actor) \
                 VALUES (?1, ?2, ?3, 'active', ?4, ?5, ?6, ?7)",
                params![session_id, namespace, client_id, ts, meta_str, hlc_bytes, actor],
            ).map_err(|e| {
                if let rusqlite::Error::SqliteFailure(ref err, _) = e {
                    if err.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE {
                        return YantrikDbError::SessionConflict(format!(
                            "active session already exists for namespace={namespace}, client_id={client_id}"
                        ));
                    }
                }
                e.into()
            })?;
        } // drop conn before acquiring active_sessions write lock

        // Cache the active session
        self.active_sessions
            .write()
            .unwrap()
            .insert(namespace.to_string(), session_id.clone());

        self.log_op(
            "session_start",
            Some(&session_id),
            &serde_json::json!({
                "session_id": session_id,
                "namespace": namespace,
                "client_id": client_id,
                "started_at": ts,
            }),
            None,
        )?;

        Ok(session_id)
    }

    /// End an active session. Computes summary stats.
    pub fn session_end(
        &self,
        session_id: &str,
        summary: Option<&str>,
    ) -> Result<SessionSummary> {
        let ts = now();

        let (memory_count, avg_valence, topics, duration_secs) = {
            let conn = self.conn();

            // Gather stats from memories linked to this session
            let (memory_count, avg_valence): (i64, f64) = conn.query_row(
                "SELECT COUNT(*), COALESCE(AVG(valence), 0.0) \
                 FROM memories WHERE session_id = ?1",
                params![session_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;

            // Gather topics: distinct entity names from this session's memories
            let mut topic_stmt = conn.prepare(
                "SELECT DISTINCT e.name FROM entities e \
                 JOIN memory_entities me ON me.entity_name = e.name \
                 JOIN memories m ON m.rid = me.memory_rid \
                 WHERE m.session_id = ?1 \
                 LIMIT 20",
            )?;
            let topics: Vec<String> = topic_stmt
                .query_map(params![session_id], |row| row.get(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            let topics_json = serde_json::to_string(&topics)?;

            // Get started_at for duration calc
            let started_at: f64 = conn.query_row(
                "SELECT started_at FROM sessions WHERE session_id = ?1",
                params![session_id],
                |row| row.get(0),
            )?;

            let duration_secs = ts - started_at;

            // Update the session row
            conn.execute(
                "UPDATE sessions SET status = 'ended', ended_at = ?1, avg_valence = ?2, \
                 memory_count = ?3, topics = ?4, summary = ?5 \
                 WHERE session_id = ?6 AND status = 'active'",
                params![ts, avg_valence, memory_count, topics_json, summary, session_id],
            )?;

            (memory_count, avg_valence, topics, duration_secs)
        }; // drop conn before acquiring active_sessions write lock

        // Clear from active_sessions cache
        self.active_sessions.write().unwrap().retain(|_, sid| sid != session_id);

        self.log_op(
            "session_end",
            Some(session_id),
            &serde_json::json!({
                "session_id": session_id,
                "ended_at": ts,
                "memory_count": memory_count,
                "avg_valence": avg_valence,
                "duration_secs": duration_secs,
            }),
            None,
        )?;

        Ok(SessionSummary {
            session_id: session_id.to_string(),
            duration_secs,
            memory_count,
            avg_valence,
            topics,
        })
    }

    /// Get the currently active session for a (namespace, client_id).
    pub fn active_session(
        &self,
        namespace: &str,
        client_id: &str,
    ) -> Result<Option<Session>> {
        let conn = self.conn();
        let result = conn.query_row(
            "SELECT session_id, namespace, client_id, status, started_at, ended_at, \
             summary, avg_valence, memory_count, topics, metadata \
             FROM sessions \
             WHERE namespace = ?1 AND client_id = ?2 AND status = 'active'",
            params![namespace, client_id],
            |row| row_to_session(row),
        );

        match result {
            Ok(s) => Ok(Some(s)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get session history for a (namespace, client_id).
    pub fn session_history(
        &self,
        namespace: &str,
        client_id: &str,
        limit: usize,
    ) -> Result<Vec<Session>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT session_id, namespace, client_id, status, started_at, ended_at, \
             summary, avg_valence, memory_count, topics, metadata \
             FROM sessions \
             WHERE namespace = ?1 AND client_id = ?2 \
             ORDER BY started_at DESC LIMIT ?3",
        )?;

        let rows = stmt
            .query_map(params![namespace, client_id, limit as i64], |row| {
                row_to_session(row)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    /// Abandon stale sessions (active for longer than max_age_hours).
    pub fn session_abandon_stale(&self, max_age_hours: f64) -> Result<usize> {
        let ts = now();
        let cutoff = ts - max_age_hours * 3600.0;

        let changes = {
            let conn = self.conn();
            let changes = conn.execute(
                "UPDATE sessions SET status = 'abandoned', ended_at = ?1 \
                 WHERE status = 'active' AND started_at < ?2",
                params![ts, cutoff],
            )?;

            if changes > 0 {
                // Reload active_sessions cache while still holding conn
                let new_cache = Self::load_active_sessions(&conn)?;
                // conn < active_sessions in lock ordering, safe to hold both
                *self.active_sessions.write().unwrap() = new_cache;
            }
            changes
        };

        Ok(changes)
    }
}

fn row_to_session(row: &rusqlite::Row<'_>) -> rusqlite::Result<Session> {
    let topics_str: String = row.get(9)?;
    let meta_str: String = row.get(10)?;
    Ok(Session {
        session_id: row.get(0)?,
        namespace: row.get(1)?,
        client_id: row.get(2)?,
        status: row.get(3)?,
        started_at: row.get(4)?,
        ended_at: row.get(5)?,
        summary: row.get(6)?,
        avg_valence: row.get(7)?,
        memory_count: row.get(8)?,
        topics: serde_json::from_str(&topics_str).unwrap_or_default(),
        metadata: serde_json::from_str(&meta_str)
            .unwrap_or(serde_json::Value::Object(Default::default())),
    })
}
