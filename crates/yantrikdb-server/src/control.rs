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
    /// Maximum oplog entries before writes are rejected until GC catches up.
    /// Default 500k entries (~200MB at avg 400 bytes/entry).
    pub max_oplog_entries: i64,
}

impl Default for TenantQuota {
    fn default() -> Self {
        Self {
            max_memories: 1_000_000,
            max_batch_size: 10_000,
            max_rps: 1_000,
            max_oplog_entries: 500_000,
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

            -- RFC 030: replicated admin accounts. password_hash is argon2id
            -- (never plaintext); role is 'owner'|'admin'|'readonly'.
            -- token_version bumps on password/role/disable change so live
            -- stateless sessions minted before the change fail verification.
            CREATE TABLE IF NOT EXISTS admin_users (
                username      TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL,
                token_version INTEGER NOT NULL DEFAULT 1,
                disabled_at   TEXT,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- RFC 030 (H3): replicated server secrets, decoupled from
            -- cluster_secret. Holds the admin session-signing key (id
            -- 'admin_session_key') as a kid + base64 value so sessions verify
            -- on every node and the key can rotate independently of peer auth.
            CREATE TABLE IF NOT EXISTS server_secrets (
                id      TEXT PRIMARY KEY,
                kid     TEXT NOT NULL,
                value   TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- RFC 030 (M1): replicated, tamper-evident admin audit. Written
            -- deterministically by ControlApplySink as each mutating control
            -- op applies, so the trail is quorum-durable + byte-identical on
            -- every node and survives failover/node loss.
            CREATE TABLE IF NOT EXISTS audit_log (
                yrp_index  INTEGER PRIMARY KEY,
                actor      TEXT NOT NULL,
                action     TEXT NOT NULL,
                target     TEXT NOT NULL,
                at         TEXT NOT NULL
            );

            -- RFC 031: replicated pack MOUNT MANIFEST — the cluster's intent
            -- of which pack (by content digest) is mounted into which database.
            -- Deliberately NO foreign key / no validation (review H3): manifest
            -- apply must be an unconditional idempotent UPSERT that can never
            -- fail on committed input and fence the cluster. The large pack
            -- FILES ride out-of-band (content-addressed store), never the log.
            CREATE TABLE IF NOT EXISTS db_packs (
                database_id  INTEGER NOT NULL,
                pack_digest  TEXT NOT NULL,
                pack_name    TEXT NOT NULL,
                mounted_at   TEXT NOT NULL,
                unmounted_at TEXT,
                PRIMARY KEY (database_id, pack_digest)
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

    /// Look up a token's `(database_id, label)` by hash (active tokens only)
    /// — used by rotation to mint a replacement for the same db + label.
    pub fn get_token_meta(&self, hash: &str) -> anyhow::Result<Option<(i64, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT database_id, label FROM tokens WHERE hash = ?1 AND revoked_at IS NULL",
        )?;
        let mut rows = stmt.query_map(params![hash], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?;
        Ok(rows.next().transpose()?)
    }

    /// List active tokens (hash included; the HTTP layer redacts to a prefix).
    pub fn list_active_tokens(&self) -> anyhow::Result<Vec<TokenSnapshot>> {
        let mut stmt = self.conn.prepare(
            "SELECT hash, database_id, label, created_at FROM tokens WHERE revoked_at IS NULL
             ORDER BY database_id, created_at",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(TokenSnapshot {
                hash: row.get(0)?,
                database_id: row.get(1)?,
                label: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
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
                    ..TenantQuota::default()
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

    // ── RFC 029: control-op apply (replicated control plane) ───────
    //
    // These are the deterministic apply primitives the YRP `ControlApplySink`
    // calls when a `Payload::Control` entry commits. Ids and timestamps are
    // **leader-assigned** and carried in the op, so every node writes the
    // identical row; apply is **idempotent on the natural key** so replaying
    // an already-durable index is a no-op.

    /// The next database id the leader will assign (`MAX(id)+1`). RFC 029:
    /// the leader allocates ids under a serializing lock so every node
    /// inserts the same one (local AUTOINCREMENT would diverge across nodes).
    pub fn next_database_id(&self) -> anyhow::Result<i64> {
        let n: i64 =
            self.conn
                .query_row("SELECT COALESCE(MAX(id), 0) + 1 FROM databases", [], |r| {
                    r.get(0)
                })?;
        Ok(n)
    }

    /// Apply a replicated database create: explicit leader-assigned id,
    /// idempotent on both the id PK and the name UNIQUE index. Returns
    /// `true` if a row was inserted, `false` if it already existed
    /// (crash-replay). A `false` is NOT an error — the natural key already
    /// holds the leader's value.
    pub fn apply_create_database(
        &self,
        id: i64,
        name: &str,
        path: &str,
        config: &str,
        created_at: &str,
    ) -> anyhow::Result<bool> {
        let changed = self.conn.execute(
            "INSERT OR IGNORE INTO databases (id, name, path, config, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, name, path, config, created_at],
        )?;
        Ok(changed > 0)
    }

    /// Apply a replicated token create: register a token hash idempotently
    /// (the `hash` PK dedupes crash-replay). Verifier material only — the
    /// plaintext token never reaches this layer (RFC 029 Invariant 3).
    pub fn apply_create_token(
        &self,
        hash: &str,
        database_id: i64,
        label: &str,
        created_at: &str,
    ) -> anyhow::Result<bool> {
        let changed = self.conn.execute(
            "INSERT OR IGNORE INTO tokens (hash, database_id, label, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![hash, database_id, label, created_at],
        )?;
        Ok(changed > 0)
    }

    /// Apply a replicated token revoke at a leader-supplied timestamp.
    /// Idempotent: only the first revoke sets `revoked_at`.
    pub fn apply_revoke_token(&self, hash: &str, revoked_at: &str) -> anyhow::Result<bool> {
        let changed = self.conn.execute(
            "UPDATE tokens SET revoked_at = ?2 WHERE hash = ?1 AND revoked_at IS NULL",
            params![hash, revoked_at],
        )?;
        Ok(changed > 0)
    }

    // ── RFC 030: replicated admin accounts (control-op apply + reads) ──

    /// Apply a replicated user create. Idempotent on the username PK.
    /// `password_hash` is argon2id, computed by the leader before proposing
    /// so every node stores the identical hash (never re-hashed per node).
    pub fn apply_create_user(
        &self,
        username: &str,
        password_hash: &str,
        role: &str,
        created_at: &str,
    ) -> anyhow::Result<bool> {
        let changed = self.conn.execute(
            "INSERT OR IGNORE INTO admin_users (username, password_hash, role, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![username, password_hash, role, created_at],
        )?;
        Ok(changed > 0)
    }

    /// Apply a role change, bumping `token_version` so the user's live
    /// stateless sessions (minted at the old version) stop verifying.
    ///
    /// **Owner-floor (H2), enforced HERE at apply time in log order:** if this
    /// change would demote the last enabled `owner`, it is a deterministic
    /// no-op on every node — never leave the cluster with zero owners. Because
    /// the check + update run under one lock at a fixed log position, two
    /// concurrent demotions proposed on different nodes cannot race to zero.
    /// `version` is the YRP log index at which this change applies. We SET
    /// `token_version = version` (not `+1`) so the bump is **crash-replay
    /// idempotent and identical on every node** (review F2): the apply and
    /// the durable marker are separate DB commits, so an index can re-apply
    /// after a crash — an unconditional increment would diverge, breaking the
    /// H1 revocation the value underpins. The index is monotonic, so this is
    /// still strictly-increasing per change.
    pub fn apply_set_user_role(
        &self,
        version: u64,
        username: &str,
        role: &str,
    ) -> anyhow::Result<bool> {
        if role != "owner" {
            let is_last_owner: bool = self.conn.query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM admin_users
                    WHERE username = ?1 AND role = 'owner' AND disabled_at IS NULL
                 ) AND (SELECT COUNT(*) FROM admin_users
                        WHERE role = 'owner' AND disabled_at IS NULL) <= 1",
                params![username],
                |r| r.get(0),
            )?;
            if is_last_owner {
                tracing::warn!(username, "owner-floor: refusing to demote the last owner");
                return Ok(false);
            }
        }
        // `role <> ?2` makes a same-role write a true no-op (review F5): it
        // neither changes the row nor invalidates the user's live sessions.
        let changed = self.conn.execute(
            "UPDATE admin_users SET role = ?2, token_version = ?3
             WHERE username = ?1 AND role <> ?2",
            params![username, role, version as i64],
        )?;
        Ok(changed > 0)
    }

    /// Apply a password rotation (argon2id hash). `version` = the log index
    /// (idempotent, see [`apply_set_user_role`]).
    pub fn apply_set_user_password(
        &self,
        version: u64,
        username: &str,
        hash: &str,
    ) -> anyhow::Result<bool> {
        let changed = self.conn.execute(
            "UPDATE admin_users SET password_hash = ?2, token_version = ?3
             WHERE username = ?1",
            params![username, hash, version as i64],
        )?;
        Ok(changed > 0)
    }

    /// Apply a soft-disable, setting `token_version` to the log index so live
    /// sessions die. Owner-floor (H2) at apply time: refuses to disable the
    /// last enabled owner (deterministic no-op on every node).
    pub fn apply_disable_user(
        &self,
        version: u64,
        username: &str,
        disabled_at: &str,
    ) -> anyhow::Result<bool> {
        let is_last_owner: bool = self.conn.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM admin_users
                WHERE username = ?1 AND role = 'owner' AND disabled_at IS NULL
             ) AND (SELECT COUNT(*) FROM admin_users
                    WHERE role = 'owner' AND disabled_at IS NULL) <= 1",
            params![username],
            |r| r.get(0),
        )?;
        if is_last_owner {
            tracing::warn!(username, "owner-floor: refusing to disable the last owner");
            return Ok(false);
        }
        let changed = self.conn.execute(
            "UPDATE admin_users SET disabled_at = ?2, token_version = ?3
             WHERE username = ?1 AND disabled_at IS NULL",
            params![username, disabled_at, version as i64],
        )?;
        Ok(changed > 0)
    }

    /// Look up an admin account (for login + session verification). Returns
    /// `None` for unknown users; the caller checks `disabled_at`.
    pub fn get_admin_user(&self, username: &str) -> anyhow::Result<Option<AdminUserRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT username, password_hash, role, token_version, disabled_at, created_at
             FROM admin_users WHERE username = ?1",
        )?;
        let mut rows = stmt.query_map(params![username], Self::map_admin_user)?;
        Ok(rows.next().transpose()?)
    }

    /// List admin accounts (password hashes included for internal callers;
    /// the HTTP layer strips them). Ordered by username.
    pub fn list_admin_users(&self) -> anyhow::Result<Vec<AdminUserRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT username, password_hash, role, token_version, disabled_at, created_at
             FROM admin_users ORDER BY username",
        )?;
        let rows = stmt.query_map([], Self::map_admin_user)?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Count enabled `owner` accounts — enforces the owner-floor invariant
    /// (never disable/demote the last owner and lock everyone out).
    pub fn count_enabled_owners(&self) -> anyhow::Result<i64> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM admin_users WHERE role = 'owner' AND disabled_at IS NULL",
            [],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// The admin session-signing key `(kid, base64_value)` if seeded (H3).
    pub fn get_admin_session_key(&self) -> anyhow::Result<Option<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT kid, value FROM server_secrets WHERE id = 'admin_session_key'")?;
        let mut rows =
            stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        Ok(rows.next().transpose()?)
    }

    /// Apply a replicated session-key set/rotate (H3). Last-writer-by-kid.
    pub fn apply_set_admin_session_key(&self, kid: &str, value: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO server_secrets (id, kid, value) VALUES ('admin_session_key', ?1, ?2)
             ON CONFLICT(id) DO UPDATE SET kid = excluded.kid, value = excluded.value,
                created_at = datetime('now')",
            params![kid, value],
        )?;
        Ok(())
    }

    /// Write a replicated audit row keyed on the YRP log index (M1),
    /// idempotent on replay. Called by `ControlApplySink` as it applies a
    /// mutating control op.
    pub fn apply_audit(
        &self,
        yrp_index: u64,
        actor: &str,
        action: &str,
        target: &str,
        at: &str,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO audit_log (yrp_index, actor, action, target, at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![yrp_index as i64, actor, action, target, at],
        )?;
        Ok(())
    }

    /// Most-recent audit rows (M1), newest first, for the audit view.
    pub fn list_audit(&self, limit: i64) -> anyhow::Result<Vec<AuditRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT yrp_index, actor, action, target, at FROM audit_log
             ORDER BY yrp_index DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok(AuditRow {
                yrp_index: row.get::<_, i64>(0)? as u64,
                actor: row.get(1)?,
                action: row.get(2)?,
                target: row.get(3)?,
                at: row.get(4)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    // ── RFC 031: pack mount manifest (unconditional, fail-stop-safe) ──

    /// Apply a replicated `MountPack`: UPSERT the manifest row and CLEAR
    /// `unmounted_at` (so a remount-after-unmount takes — L1). Unconditional:
    /// no FK, no db-existence check, nothing that can fail on committed input
    /// (H3). An orphaned row (db since gone) is harmless — the reconciler
    /// no-ops it.
    pub fn apply_mount_pack(
        &self,
        database_id: i64,
        pack_digest: &str,
        pack_name: &str,
        mounted_at: &str,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO db_packs (database_id, pack_digest, pack_name, mounted_at, unmounted_at)
             VALUES (?1, ?2, ?3, ?4, NULL)
             ON CONFLICT(database_id, pack_digest) DO UPDATE SET
                pack_name = excluded.pack_name, mounted_at = excluded.mounted_at,
                unmounted_at = NULL",
            params![database_id, pack_digest, pack_name, mounted_at],
        )?;
        Ok(())
    }

    /// Apply a replicated `UnmountPack`: soft-unmount (idempotent).
    pub fn apply_unmount_pack(
        &self,
        database_id: i64,
        pack_digest: &str,
        unmounted_at: &str,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE db_packs SET unmounted_at = ?3
             WHERE database_id = ?1 AND pack_digest = ?2 AND unmounted_at IS NULL",
            params![database_id, pack_digest, unmounted_at],
        )?;
        Ok(())
    }

    /// Every ACTIVE (not-unmounted) mount in the manifest — the reconciler's
    /// desired state across all databases.
    pub fn active_pack_mounts(&self) -> anyhow::Result<Vec<PackMountRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT database_id, pack_digest, pack_name FROM db_packs
             WHERE unmounted_at IS NULL ORDER BY database_id, pack_digest",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(PackMountRow {
                database_id: r.get(0)?,
                pack_digest: r.get(1)?,
                pack_name: r.get(2)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Active mounts for one database (pack-context / recall pending signal).
    pub fn active_pack_mounts_for(&self, database_id: i64) -> anyhow::Result<Vec<PackMountRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT database_id, pack_digest, pack_name FROM db_packs
             WHERE database_id = ?1 AND unmounted_at IS NULL ORDER BY pack_digest",
        )?;
        let rows = stmt.query_map(params![database_id], |r| {
            Ok(PackMountRow {
                database_id: r.get(0)?,
                pack_digest: r.get(1)?,
                pack_name: r.get(2)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    fn map_admin_user(row: &rusqlite::Row) -> rusqlite::Result<AdminUserRecord> {
        Ok(AdminUserRecord {
            username: row.get(0)?,
            password_hash: row.get(1)?,
            role: row.get(2)?,
            token_version: row.get(3)?,
            disabled_at: row.get(4)?,
            created_at: row.get(5)?,
        })
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

/// RFC 031 — an active pack mount from the replicated manifest.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PackMountRow {
    pub database_id: i64,
    pub pack_digest: String,
    pub pack_name: String,
}

/// RFC 030 (M1) — a replicated audit row (newest-first in the audit view).
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditRow {
    pub yrp_index: u64,
    pub actor: String,
    pub action: String,
    pub target: String,
    pub at: String,
}

/// RFC 030 — a replicated admin account row. `password_hash` (argon2id) is
/// never exposed by the HTTP layer.
#[derive(Debug, Clone)]
pub struct AdminUserRecord {
    pub username: String,
    pub password_hash: String,
    pub role: String,
    pub token_version: i64,
    pub disabled_at: Option<String>,
    pub created_at: String,
}

/// Token record as serialized for replication (no revoked tokens).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenSnapshot {
    pub hash: String,
    pub database_id: i64,
    pub label: String,
    pub created_at: String,
}
