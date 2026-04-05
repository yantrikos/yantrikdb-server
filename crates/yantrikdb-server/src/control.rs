//! control.db — metadata database for multi-tenancy.
//!
//! Tracks databases, tokens, and server config in a dedicated SQLite file
//! separate from any tenant's data.

use rusqlite::{params, Connection};
use std::path::Path;

pub struct ControlDb {
    conn: Connection,
}

#[derive(Debug, Clone)]
pub struct DatabaseRecord {
    pub id: i64,
    pub name: String,
    pub path: String,
    pub created_at: String,
}

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
            "
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
        let mut stmt = self.conn.prepare(
            "SELECT id, name, path, created_at FROM databases ORDER BY id"
        )?;
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
        let mut stmt = self.conn.prepare(
            "SELECT id, name, path, created_at FROM databases WHERE name = ?1"
        )?;
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
        let mut stmt = self.conn.prepare(
            "SELECT id, name, path, created_at FROM databases WHERE id = ?1"
        )?;
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
        let mut stmt = self.conn.prepare(
            "SELECT database_id FROM tokens WHERE hash = ?1 AND revoked_at IS NULL"
        )?;
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

    /// Count total databases.
    pub fn database_count(&self) -> anyhow::Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM databases",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }
}
