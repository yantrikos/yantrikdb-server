mod belief;
mod cache;
mod intent;
mod action;
mod evaluator;
mod policy;
mod suggest;
mod agenda;
mod temporal;
mod hawkes;
mod receptivity;
mod tick;
mod surfacing;
mod observer;
mod flywheel;
mod world_model;
mod experimenter;
mod skills;
mod extractor;
mod calibration;
mod introspection;
mod causal;
mod planner;
mod cognition;
mod coherence;
mod metacognition;
mod personality_bias;
mod query_dsl;
mod conflict;
mod analogy_engine;
mod schema_induction_engine;
mod narrative_engine;
mod counterfactual_engine;
mod belief_network_engine;
mod replay_engine;
mod perspective_engine;
mod feedback;
pub mod graph_state;
mod graph_ops;
mod indices;
mod learning;
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
use crate::error::{YantrikDbError, Result};
use crate::graph_index::GraphIndex;
use crate::hlc::{HLCTimestamp, HLC};
use crate::hnsw::HnswIndex;
use crate::schema::{
    MIGRATE_V1_TO_V2, MIGRATE_V2_TO_V3, MIGRATE_V3_TO_V4, MIGRATE_V4_TO_V5,
    MIGRATE_V5_TO_V6, MIGRATE_V6_TO_V7, MIGRATE_V7_TO_V8, MIGRATE_V8_TO_V9,
    MIGRATE_V9_TO_V10, MIGRATE_V10_TO_V11, MIGRATE_V11_TO_V12,
    SCHEMA_SQL, SCHEMA_VERSION,
};
use crate::types::*;

/// The YantrikDB cognitive memory engine.
pub struct YantrikDB {
    pub(crate) conn: Connection,
    pub(crate) embedding_dim: usize,
    pub(crate) hlc: RefCell<HLC>,
    pub(crate) actor_id: String,
    pub(crate) scoring_cache: RefCell<HashMap<String, ScoringRow>>,
    pub(crate) vec_index: RefCell<HnswIndex>,
    pub(crate) graph_index: RefCell<GraphIndex>,
    pub(crate) enc: Option<EncryptionProvider>,
    /// Optional text-to-embedding converter. When set, enables `record_text()`
    /// and `recall_text()` which auto-embed text without an external server.
    embedder: Option<Box<dyn crate::types::Embedder>>,
}

pub(crate) fn now() -> f64 {
    crate::time::now_secs()
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

impl YantrikDB {
    /// Create a new YantrikDB instance with auto-generated actor_id.
    pub fn new(db_path: &str, embedding_dim: usize) -> Result<Self> {
        Self::open(db_path, embedding_dim, None, None)
    }

    /// Create a new YantrikDB instance with an explicit actor_id (for sync tests).
    pub fn new_with_actor(db_path: &str, embedding_dim: usize, actor_id: &str) -> Result<Self> {
        Self::open(db_path, embedding_dim, Some(actor_id.to_string()), None)
    }

    /// Create a new encrypted YantrikDB instance.
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
            (9, MIGRATE_V9_TO_V10),
            (10, MIGRATE_V10_TO_V11),
            (11, MIGRATE_V11_TO_V12),
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
                    let id = crate::id::new_id();
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
                        .map_err(|e| YantrikDbError::Encryption(format!("DEK base64: {e}")))?;
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
                return Err(YantrikDbError::Encryption(
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
            embedder: None,
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

    /// Get a reference to the encryption provider (for vault operations).
    pub fn encryption(&self) -> Option<&EncryptionProvider> {
        self.enc.as_ref()
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
        self.conn.close().map_err(|(_, e)| YantrikDbError::Database(e))
    }

    // ── Embedder integration ──

    /// Set the text-to-embedding converter. Enables `embed()`, `record_text()`,
    /// and `recall_text()` which auto-embed text without an external server.
    pub fn set_embedder(&mut self, embedder: Box<dyn crate::types::Embedder>) {
        self.embedder = Some(embedder);
    }

    /// Whether an embedder is configured.
    pub fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    /// Embed text using the configured embedder.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embedder
            .as_ref()
            .ok_or(YantrikDbError::NoEmbedder)?
            .embed(text)
            .map_err(|e| YantrikDbError::Inference(e.to_string()))
    }

    /// Record a memory with automatic embedding generation.
    pub fn record_text(
        &self,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
    ) -> Result<String> {
        let embedding = self.embed(text)?;
        self.record(
            text,
            memory_type,
            importance,
            valence,
            half_life,
            metadata,
            &embedding,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
        )
    }

    /// Recall memories by text query with automatic embedding.
    pub fn recall_text(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RecallResult>> {
        let embedding = self.embed(query)?;
        self.recall(
            &embedding,
            top_k,
            None,  // time_window
            None,  // memory_type
            false, // include_consolidated
            true,  // expand_entities
            Some(query),
            false, // skip_reinforce
            None,  // namespace
            None,  // domain
            None,  // source
        )
    }

    /// Recall memories with domain and source filters.
    ///
    /// Like `recall_text` but restricts results to a specific domain
    /// (e.g. `"session/summary"`, `"audit/tools"`) and/or source
    /// (e.g. `"self"`, `"companion"`, `"system"`).
    pub fn recall_text_filtered(
        &self,
        query: &str,
        top_k: usize,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> Result<Vec<RecallResult>> {
        let embedding = self.embed(query)?;
        self.recall(
            &embedding,
            top_k,
            None,  // time_window
            None,  // memory_type
            false, // include_consolidated
            true,  // expand_entities
            Some(query),
            false, // skip_reinforce
            None,  // namespace
            domain,
            source,
        )
    }
}
