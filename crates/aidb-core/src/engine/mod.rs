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
pub mod tenant;
#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use rand::Rng;
use rusqlite::{params, Connection};

use crate::encryption::{self, EncryptionProvider};
use crate::error::{AidbError, Result};
use crate::graph_index::GraphIndex;
use crate::hlc::{HLCTimestamp, HLC};
use crate::hnsw::HnswIndex;
use crate::schema::{
    MIGRATE_V1_TO_V2, MIGRATE_V2_TO_V3, MIGRATE_V3_TO_V4, MIGRATE_V4_TO_V5,
    MIGRATE_V5_TO_V6, MIGRATE_V6_TO_V7, MIGRATE_V7_TO_V8, MIGRATE_V8_TO_V9,
    SCHEMA_SQL, SCHEMA_VERSION,
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
    pub(crate) enc: Option<EncryptionProvider>,
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
        Self::open(db_path, embedding_dim, None, None)
    }

    /// Create a new AIDB instance with an explicit actor_id (for sync tests).
    pub fn new_with_actor(db_path: &str, embedding_dim: usize, actor_id: &str) -> Result<Self> {
        Self::open(db_path, embedding_dim, Some(actor_id.to_string()), None)
    }

    /// Create a new encrypted AIDB instance.
    ///
    /// The 32-byte `master_key` is used to wrap/unwrap a per-database Data Encryption Key (DEK).
    /// All text, metadata, and embedding fields are encrypted at rest using AES-256-GCM.
    /// In-memory indexes operate on plaintext for full query performance.
    pub fn new_encrypted(db_path: &str, embedding_dim: usize, master_key: &[u8; 32]) -> Result<Self> {
        Self::open(db_path, embedding_dim, None, Some(master_key))
    }

    fn open(
        db_path: &str,
        embedding_dim: usize,
        actor_id: Option<String>,
        master_key: Option<&[u8; 32]>,
    ) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        // Check existing schema version for migration
        let existing_version = Self::get_schema_version(&conn);

        // Sequential migration chain — each version cascades.
        let migrations: &[(i32, &str)] = &[
            (1, MIGRATE_V1_TO_V2),
            (2, MIGRATE_V2_TO_V3),
            (3, MIGRATE_V3_TO_V4),
            (4, MIGRATE_V4_TO_V5),
            (5, MIGRATE_V5_TO_V6),
            (6, MIGRATE_V6_TO_V7),
            (7, MIGRATE_V7_TO_V8),
            (8, MIGRATE_V8_TO_V9),
        ];
        if let Some(v) = existing_version {
            for &(from_v, sql) in migrations {
                if v <= from_v {
                    conn.execute_batch(sql)?;
                }
            }
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

        // Initialize encryption (envelope pattern: master_key wraps DEK)
        let enc = if let Some(mk) = master_key {
            let provider = match Self::get_meta(&conn, "encrypted_dek")? {
                Some(wrapped_b64) => {
                    // Existing DB: unwrap DEK
                    let wrapped = base64::engine::general_purpose::STANDARD
                        .decode(&wrapped_b64)
                        .map_err(|e| AidbError::Encryption(format!("DEK base64: {e}")))?;
                    let dek = encryption::unwrap_dek(mk, &wrapped)?;
                    EncryptionProvider::from_dek(&dek)
                }
                None => {
                    // New DB: generate and store DEK
                    let dek = encryption::generate_key();
                    let wrapped = encryption::wrap_dek(mk, &dek)?;
                    let wrapped_b64 = base64::engine::general_purpose::STANDARD.encode(&wrapped);
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES ('encrypted_dek', ?1)",
                        params![wrapped_b64],
                    )?;
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES ('encryption_enabled', '1')",
                        [],
                    )?;
                    EncryptionProvider::from_dek(&dek)
                }
            };
            Some(provider)
        } else {
            // Verify we're not opening an encrypted DB without a key
            if Self::get_meta(&conn, "encryption_enabled")?.as_deref() == Some("1") {
                return Err(AidbError::Encryption(
                    "database is encrypted but no master_key provided".into(),
                ));
            }
            None
        };

        let scoring_cache = Self::load_scoring_cache(&conn)?;
        let vec_index = Self::build_vec_index_with_enc(&conn, embedding_dim, enc.as_ref())?;
        let graph_index = GraphIndex::build_from_db(&conn)?;

        Ok(Self {
            conn,
            embedding_dim,
            hlc: RefCell::new(HLC::new(node_id)),
            actor_id,
            scoring_cache: RefCell::new(scoring_cache),
            vec_index: RefCell::new(vec_index),
            graph_index: RefCell::new(graph_index),
            enc,
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

    /// Whether this instance has encryption enabled.
    pub fn is_encrypted(&self) -> bool {
        self.enc.is_some()
    }

    // ── Encryption helpers (transparent to callers) ──

    /// Encrypt a string field if encryption is enabled, otherwise pass through.
    pub(crate) fn encrypt_text(&self, plaintext: &str) -> Result<String> {
        match &self.enc {
            Some(e) => e.encrypt_string(plaintext),
            None => Ok(plaintext.to_string()),
        }
    }

    /// Decrypt a string field if encryption is enabled, otherwise pass through.
    pub(crate) fn decrypt_text(&self, stored: &str) -> Result<String> {
        match &self.enc {
            Some(e) => e.decrypt_string(stored),
            None => Ok(stored.to_string()),
        }
    }

    /// Encrypt an embedding blob if encryption is enabled.
    pub(crate) fn encrypt_embedding(&self, emb_blob: &[u8]) -> Result<Vec<u8>> {
        match &self.enc {
            Some(e) => e.encrypt_bytes(emb_blob),
            None => Ok(emb_blob.to_vec()),
        }
    }

    /// Decrypt an embedding blob if encryption is enabled.
    pub(crate) fn decrypt_embedding(&self, stored: &[u8]) -> Result<Vec<u8>> {
        match &self.enc {
            Some(e) => e.decrypt_bytes(stored),
            None => Ok(stored.to_vec()),
        }
    }

    /// Close the database connection. After this, the engine cannot be used.
    pub fn close(self) -> Result<()> {
        self.conn.close().map_err(|(_, e)| AidbError::Database(e))
    }
}
