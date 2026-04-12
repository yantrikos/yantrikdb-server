//! control.db — metadata database for multi-tenancy.
//!
//! Tracks databases, tokens, and server config in a dedicated SQLite file
//! separate from any tenant's data.

use rusqlite::{params, Connection};
use std::path::Path;

pub struct ControlDb {
    conn: Connection,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DatabaseRecord {
    pub id: i64,
    pub name: String,
    pub path: String,
    pub created_at: String,
}

/// Per-tenant resource quotas. Generous defaults ensure existing tenants
/// aren't broken; tighten per-database via the admin API.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TenantQuota {
    pub max_memories: i64,
    pub max_batch_size: i64,
    pub max_rps: i64,
}

impl Default for TenantQuota {
    fn default() -> Self {
        Self {
            max_memories: 1_000_000,
            max_batch_size: 10_000,
            max_rps: 1_000,
        }
    }
}

/// Metadata row for a token. Currently not returned by any code path —
/// the control DB operates by token hash, not by record. Reserved for
/// the `/v1/admin/tokens` listing endpoint (planned).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TokenRecord {
    pub hash: String,
    pub database_id: i64,
    pub label: String,
    pub created_at: String,
}

impl ControlDb {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let conn = Connection::open(path)?;
        // Same pragma hardening as tenant databases.
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; \
             PRAGMA synchronous=NORMAL; \
             PRAGMA foreign_keys=ON; \
             PRAGMA busy_timeout=5000;",
        )?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> anyhow::Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS databases (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                path        TEXT NOT NULL,
                config      TEXT NOT NULL DEFAULT '{}',
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS tokens (
                hash        TEXT PRIMARY KEY,
                database_id INTEGER NOT NULL REFERENCES databases(id),
                label       TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                revoked_at  TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_tokens_db ON tokens(database_id);

            CREATE TABLE IF NOT EXISTS quotas (
                database_id INTEGER PRIMARY KEY REFERENCES databases(id),
                max_memories    INTEGER NOT NULL DEFAULT 1000000,
                max_batch_size  INTEGER NOT NULL DEFAULT 10000,
                max_rps         INTEGER NOT NULL DEFAULT 1000,
                updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );
            ",
        )?;
        Ok(())
    }

    /// Create a new database entry. Returns the database ID.
    pub fn create_database(&self, name: &str, path: &str) -> anyhow::Result<i64> {
        self.conn.execute(
            "INSERT INTO databases (name, path) VALUES (?1, ?2)",
            params![name, path],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// List all databases.
    pub fn list_databases(&self) -> anyhow::Result<Vec<DatabaseRecord>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, path, created_at FROM databases ORDER BY id")?;
        let rows = stmt.query_map([], |row| {
            Ok(DatabaseRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                path: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Get a database by name.
    pub fn get_database(&self, name: &str) -> anyhow::Result<Option<DatabaseRecord>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, path, created_at FROM databases WHERE name = ?1")?;
        let mut rows = stmt.query_map(params![name], |row| {
            Ok(DatabaseRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                path: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        Ok(rows.next().transpose()?)
    }

    /// Get a database by ID.
    pub fn get_database_by_id(&self, id: i64) -> anyhow::Result<Option<DatabaseRecord>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, path, created_at FROM databases WHERE id = ?1")?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(DatabaseRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                path: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        Ok(rows.next().transpose()?)
    }

    /// Store a token hash mapped to a database.
    pub fn create_token(&self, hash: &str, database_id: i64, label: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO tokens (hash, database_id, label) VALUES (?1, ?2, ?3)",
            params![hash, database_id, label],
        )?;
        Ok(())
    }

    /// Validate a token hash. Returns the database ID if valid.
    pub fn validate_token(&self, hash: &str) -> anyhow::Result<Option<i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT database_id FROM tokens WHERE hash = ?1 AND revoked_at IS NULL")?;
        let mut rows = stmt.query_map(params![hash], |row| row.get::<_, i64>(0))?;
        Ok(rows.next().transpose()?)
    }

    /// Revoke a token.
    pub fn revoke_token(&self, hash: &str) -> anyhow::Result<bool> {
        let changed = self.conn.execute(
            "UPDATE tokens SET revoked_at = datetime('now') WHERE hash = ?1 AND revoked_at IS NULL",
            params![hash],
        )?;
        Ok(changed > 0)
    }

    /// Check if a database name already exists.
    pub fn database_exists(&self, name: &str) -> anyhow::Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM databases WHERE name = ?1",
            params![name],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    // ── Quota management ────────────────────────────────────────────

    /// Get the quota for a database. Returns defaults if no explicit quota set.
    pub fn get_quota(&self, database_id: i64) -> anyhow::Result<TenantQuota> {
        let result = self.conn.query_row(
            "SELECT max_memories, max_batch_size, max_rps FROM quotas WHERE database_id = ?1",
            params![database_id],
            |row| {
                Ok(TenantQuota {
                    max_memories: row.get(0)?,
                    max_batch_size: row.get(1)?,
                    max_rps: row.get(2)?,
                })
            },
        );
        match result {
            Ok(q) => Ok(q),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(TenantQuota::default()),
            Err(e) => Err(e.into()),
        }
    }

    /// Set or update the quota for a database.
    #[allow(dead_code)]
    pub fn set_quota(&self, database_id: i64, quota: &TenantQuota) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO quotas (database_id, max_memories, max_batch_size, max_rps, updated_at)
             VALUES (?1, ?2, ?3, ?4, datetime('now'))
             ON CONFLICT(database_id) DO UPDATE SET
                max_memories = excluded.max_memories,
                max_batch_size = excluded.max_batch_size,
                max_rps = excluded.max_rps,
                updated_at = excluded.updated_at",
            params![
                database_id,
                quota.max_memories,
                quota.max_batch_size,
                quota.max_rps,
            ],
        )?;
        Ok(())
    }

    /// Count total databases.
    ///
    /// Not currently called — reserved for startup banner and /metrics
    /// surfacing of tenant count.
    #[allow(dead_code)]
    pub fn database_count(&self) -> anyhow::Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM databases", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    // ── Control Plane Replication ──────────────────────────────────

    /// Export a full snapshot of databases + active tokens for replication.
    /// Called by the leader's HTTP admin endpoint.
    pub fn export_snapshot(&self) -> anyhow::Result<ControlSnapshot> {
        let databases = self.list_databases()?;

        let mut stmt = self.conn.prepare(
            "SELECT hash, database_id, label, created_at FROM tokens WHERE revoked_at IS NULL",
        )?;
        let tokens = stmt
            .query_map([], |row| {
                Ok(TokenSnapshot {
                    hash: row.get(0)?,
                    database_id: row.get(1)?,
                    label: row.get(2)?,
                    created_at: row.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ControlSnapshot { databases, tokens })
    }

    /// Import a control snapshot from the leader, upserting databases and
    /// tokens that don't exist locally. Does NOT delete local-only records
    /// — this is an additive merge, not a replace.
    ///
    /// Returns (databases_added, tokens_added).
    pub fn import_snapshot(&self, snapshot: &ControlSnapshot) -> anyhow::Result<(usize, usize)> {
        let mut dbs_added = 0;
        for db in &snapshot.databases {
            let exists = self.database_exists(&db.name)?;
            if !exists {
                self.conn.execute(
                    "INSERT INTO databases (id, name, path, created_at) VALUES (?1, ?2, ?3, ?4)",
                    params![db.id, db.name, db.path, db.created_at],
                )?;
                dbs_added += 1;
            }
        }

        let mut tokens_added = 0;
        for tok in &snapshot.tokens {
            // Upsert: insert if not exists (idempotent)
            let changed = self.conn.execute(
                "INSERT OR IGNORE INTO tokens (hash, database_id, label, created_at)
                 VALUES (?1, ?2, ?3, ?4)",
                params![tok.hash, tok.database_id, tok.label, tok.created_at],
            )?;
            if changed > 0 {
                tokens_added += 1;
            }
        }

        Ok((dbs_added, tokens_added))
    }
}

/// Snapshot of the control plane for replication between cluster nodes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ControlSnapshot {
    pub databases: Vec<DatabaseRecord>,
    pub tokens: Vec<TokenSnapshot>,
}

/// Token record as serialized for replication (no revoked tokens).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenSnapshot {
    pub hash: String,
    pub database_id: i64,
    pub label: String,
    pub created_at: String,
}
