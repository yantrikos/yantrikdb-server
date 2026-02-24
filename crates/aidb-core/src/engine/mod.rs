mod cache;
mod cognition;
mod conflict;
mod graph_ops;
mod indices;
mod lifecycle;
mod recall;
mod record;
mod stats;
mod storage;
#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::Rng;
use rusqlite::{params, Connection};

use crate::error::{AidbError, Result};
use crate::graph_index::GraphIndex;
use crate::hlc::{HLCTimestamp, HLC};
use crate::hnsw::HnswIndex;
use crate::schema::{
    MIGRATE_V1_TO_V2, MIGRATE_V2_TO_V3, MIGRATE_V3_TO_V4, MIGRATE_V4_TO_V5,
    MIGRATE_V5_TO_V6, MIGRATE_V6_TO_V7, SCHEMA_SQL, SCHEMA_VERSION,
};
use crate::types::*;

/// The AIDB cognitive memory engine.
pub struct AIDB {
    pub(crate) conn: Connection,
    pub(crate) embedding_dim: usize,
    pub(crate) hlc: RefCell<HLC>,
    pub(crate) actor_id: String,
    pub(crate) scoring_cache: RefCell<HashMap<String, ScoringRow>>,
    pub(crate) vec_index: RefCell<HnswIndex>,
    pub(crate) graph_index: RefCell<GraphIndex>,
}

pub(crate) fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Compute BLAKE3 hash of an embedding blob.
pub(crate) fn embedding_hash(embedding: &[f32]) -> Vec<u8> {
    let blob = crate::serde_helpers::serialize_f32(embedding);
    blake3::hash(&blob).as_bytes().to_vec()
}

/// Lightweight struct for fetching only text and metadata (post-scoring hydration).
pub(crate) struct TextMetadataRow {
    pub rid: String,
    pub text: String,
    pub metadata: String,
}

impl AIDB {
    /// Create a new AIDB instance with auto-generated actor_id.
    pub fn new(db_path: &str, embedding_dim: usize) -> Result<Self> {
        Self::open(db_path, embedding_dim, None)
    }

    /// Create a new AIDB instance with an explicit actor_id (for sync tests).
    pub fn new_with_actor(db_path: &str, embedding_dim: usize, actor_id: &str) -> Result<Self> {
        Self::open(db_path, embedding_dim, Some(actor_id.to_string()))
    }

    fn open(db_path: &str, embedding_dim: usize, actor_id: Option<String>) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        // Check existing schema version for migration
        let existing_version = Self::get_schema_version(&conn);

        if existing_version == Some(1) {
            conn.execute_batch(MIGRATE_V1_TO_V2)?;
            conn.execute_batch(MIGRATE_V2_TO_V3)?;
            conn.execute_batch(MIGRATE_V3_TO_V4)?;
            conn.execute_batch(MIGRATE_V4_TO_V5)?;
            conn.execute_batch(MIGRATE_V5_TO_V6)?;
            conn.execute_batch(MIGRATE_V6_TO_V7)?;
        } else if existing_version == Some(2) {
            conn.execute_batch(MIGRATE_V2_TO_V3)?;
            conn.execute_batch(MIGRATE_V3_TO_V4)?;
            conn.execute_batch(MIGRATE_V4_TO_V5)?;
            conn.execute_batch(MIGRATE_V5_TO_V6)?;
            conn.execute_batch(MIGRATE_V6_TO_V7)?;
        } else if existing_version == Some(3) {
            conn.execute_batch(MIGRATE_V3_TO_V4)?;
            conn.execute_batch(MIGRATE_V4_TO_V5)?;
            conn.execute_batch(MIGRATE_V5_TO_V6)?;
            conn.execute_batch(MIGRATE_V6_TO_V7)?;
        } else if existing_version == Some(4) {
            conn.execute_batch(MIGRATE_V4_TO_V5)?;
            conn.execute_batch(MIGRATE_V5_TO_V6)?;
            conn.execute_batch(MIGRATE_V6_TO_V7)?;
        } else if existing_version == Some(5) {
            conn.execute_batch(MIGRATE_V5_TO_V6)?;
            conn.execute_batch(MIGRATE_V6_TO_V7)?;
        } else if existing_version == Some(6) {
            conn.execute_batch(MIGRATE_V6_TO_V7)?;
        }

        conn.execute_batch(SCHEMA_SQL)?;

        // Set schema version
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?1)",
            params![SCHEMA_VERSION.to_string()],
        )?;

        // Resolve actor_id: explicit > stored in meta > generate new
        let actor_id = if let Some(id) = actor_id {
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('actor_id', ?1)",
                params![id],
            )?;
            id
        } else {
            match Self::get_meta(&conn, "actor_id")? {
                Some(id) => id,
                None => {
                    let id = uuid7::uuid7().to_string();
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES ('actor_id', ?1)",
                        params![id],
                    )?;
                    id
                }
            }
        };

        // Resolve node_id: stored in meta > generate random
        let node_id: u32 = match Self::get_meta(&conn, "node_id")? {
            Some(s) => s.parse().unwrap_or_else(|_| {
                let id: u32 = rand::thread_rng().gen();
                id
            }),
            None => {
                let id: u32 = rand::thread_rng().gen();
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('node_id', ?1)",
                    params![id.to_string()],
                )?;
                id
            }
        };

        let scoring_cache = Self::load_scoring_cache(&conn)?;
        let vec_index = Self::build_vec_index(&conn, embedding_dim)?;
        let graph_index = GraphIndex::build_from_db(&conn)?;

        Ok(Self {
            conn,
            embedding_dim,
            hlc: RefCell::new(HLC::new(node_id)),
            actor_id,
            scoring_cache: RefCell::new(scoring_cache),
            vec_index: RefCell::new(vec_index),
            graph_index: RefCell::new(graph_index),
        })
    }

    fn get_schema_version(conn: &Connection) -> Option<i32> {
        conn.query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| {
                let v: String = row.get(0)?;
                Ok(v.parse::<i32>().unwrap_or(0))
            },
        )
        .ok()
    }

    fn get_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
        match conn.query_row(
            "SELECT value FROM meta WHERE key = ?1",
            params![key],
            |row| row.get(0),
        ) {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get a new HLC timestamp (ticks the clock forward).
    pub fn tick_hlc(&self) -> HLCTimestamp {
        self.hlc.borrow_mut().now()
    }

    /// Merge a remote HLC timestamp into the local clock.
    pub fn merge_hlc(&self, remote: HLCTimestamp) -> HLCTimestamp {
        self.hlc.borrow_mut().recv(remote)
    }

    /// Get the actor_id of this instance.
    pub fn actor_id(&self) -> &str {
        &self.actor_id
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get a reference to the underlying connection (for test compatibility).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Get a mutable reference to the underlying connection.
    pub fn conn_mut(&mut self) -> &mut Connection {
        &mut self.conn
    }

    /// Close the database connection. After this, the engine cannot be used.
    pub fn close(self) -> Result<()> {
        self.conn.close().map_err(|(_, e)| AidbError::Database(e))
    }
}
