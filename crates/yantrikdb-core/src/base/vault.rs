/// Secure credential vault baked into YantrikDB.
///
/// Uses AES-256-GCM encryption with an auto-generated vault-specific DEK.
/// Credentials are encrypted at rest — only the service name and metadata
/// are stored in plaintext for listing/search.

use rusqlite::Connection;
use rand::Rng;
use base64::{engine::general_purpose::STANDARD as B64, Engine};

use super::encryption::{self, EncryptionProvider};
use super::error::Result;

/// Get or create the vault's own EncryptionProvider.
///
/// The vault DEK is stored in the `vault_security` table, auto-generated on first access.
/// This is independent of the DB-level encryption — works even when the DB is opened
/// without a master key.
pub fn vault_encryption(conn: &Connection) -> Result<EncryptionProvider> {
    let existing: Option<String> = conn
        .query_row(
            "SELECT value FROM vault_security WHERE key = 'vault_dek'",
            [],
            |row| row.get(0),
        )
        .ok();

    if let Some(b64_dek) = existing {
        let dek_bytes = B64.decode(&b64_dek)
            .map_err(|e| super::error::YantrikDbError::Encryption(format!("vault DEK decode: {e}")))?;
        if dek_bytes.len() != 32 {
            return Err(super::error::YantrikDbError::Encryption(
                format!("vault DEK wrong length: {}", dek_bytes.len()),
            ));
        }
        let mut dek = [0u8; 32];
        dek.copy_from_slice(&dek_bytes);
        Ok(EncryptionProvider::from_dek(&dek))
    } else {
        let dek = encryption::generate_key();
        let b64 = B64.encode(dek);
        conn.execute(
            "INSERT OR REPLACE INTO vault_security (key, value) VALUES ('vault_dek', ?1)",
            rusqlite::params![b64],
        )?;
        Ok(EncryptionProvider::from_dek(&dek))
    }
}

/// Initialize vault tables. Called during DB setup.
pub fn init_tables(conn: &Connection) {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS vault_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service TEXT NOT NULL,
            username_enc TEXT NOT NULL,
            password_enc TEXT NOT NULL,
            url TEXT,
            notes_enc TEXT,
            category TEXT NOT NULL DEFAULT 'general',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(service, url)
        );
        CREATE INDEX IF NOT EXISTS idx_vault_service ON vault_entries(service);
        CREATE INDEX IF NOT EXISTS idx_vault_category ON vault_entries(category);
        CREATE TABLE IF NOT EXISTS vault_security (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );",
    )
    .expect("vault schema creation");
}

/// A decrypted vault entry.
#[derive(Debug, Clone)]
pub struct VaultEntry {
    pub id: i64,
    pub service: String,
    pub username: String,
    pub password: String,
    pub url: Option<String>,
    pub notes: Option<String>,
    pub category: String,
    pub created_at: f64,
    pub updated_at: f64,
}

/// A vault entry without sensitive fields (for listing).
#[derive(Debug, Clone)]
pub struct VaultListEntry {
    pub id: i64,
    pub service: String,
    pub url: Option<String>,
    pub category: String,
    pub updated_at: f64,
}

/// Store a credential in the vault.
pub fn store(
    conn: &Connection,
    enc: &EncryptionProvider,
    service: &str,
    username: &str,
    password: &str,
    url: Option<&str>,
    notes: Option<&str>,
    category: Option<&str>,
) -> Result<i64> {
    let now = crate::time::now_secs();

    let username_enc = enc.encrypt_string(username)?;
    let password_enc = enc.encrypt_string(password)?;
    let notes_enc = notes
        .map(|n| enc.encrypt_string(n))
        .transpose()?;
    let cat = category.unwrap_or("general");

    // Try UPDATE first for upsert — ON CONFLICT doesn't match NULLs in url
    let updated = conn.execute(
        "UPDATE vault_entries SET username_enc = ?1, password_enc = ?2, \
         notes_enc = ?3, category = ?4, updated_at = ?5 \
         WHERE service = ?6 AND (url IS ?7)",
        rusqlite::params![username_enc, password_enc, notes_enc, cat, now, service, url],
    )?;
    if updated == 0 {
        conn.execute(
            "INSERT INTO vault_entries \
             (service, username_enc, password_enc, url, notes_enc, category, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
            rusqlite::params![service, username_enc, password_enc, url, notes_enc, cat, now],
        )?;
    }

    let id = conn.last_insert_rowid();
    Ok(id)
}

/// Retrieve a credential by service name (decrypted).
pub fn get(
    conn: &Connection,
    enc: &EncryptionProvider,
    service: &str,
) -> Result<Vec<VaultEntry>> {
    let mut stmt = conn.prepare(
        "SELECT id, service, username_enc, password_enc, url, notes_enc, category, created_at, updated_at
         FROM vault_entries WHERE service = ?1 ORDER BY updated_at DESC"
    )?;

    let rows = stmt.query_map(rusqlite::params![service], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, String>(6)?,
            row.get::<_, f64>(7)?,
            row.get::<_, f64>(8)?,
        ))
    })?;

    let mut entries = Vec::new();
    for row in rows {
        let (id, svc, u_enc, p_enc, url, n_enc, cat, created, updated) = row?;

        let username = enc.decrypt_string(&u_enc)?;
        let password = enc.decrypt_string(&p_enc)?;
        let notes = n_enc.map(|n| enc.decrypt_string(&n)).transpose()?;

        entries.push(VaultEntry {
            id,
            service: svc,
            username,
            password,
            url,
            notes,
            category: cat,
            created_at: created,
            updated_at: updated,
        });
    }
    Ok(entries)
}

/// Search vault entries by service name pattern (case-insensitive).
pub fn search(
    conn: &Connection,
    enc: &EncryptionProvider,
    query: &str,
) -> Result<Vec<VaultEntry>> {
    let pattern = format!("%{query}%");
    let mut stmt = conn.prepare(
        "SELECT id, service, username_enc, password_enc, url, notes_enc, category, created_at, updated_at
         FROM vault_entries WHERE service LIKE ?1 ORDER BY updated_at DESC LIMIT 20"
    )?;

    let rows = stmt.query_map(rusqlite::params![pattern], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, String>(6)?,
            row.get::<_, f64>(7)?,
            row.get::<_, f64>(8)?,
        ))
    })?;

    let mut entries = Vec::new();
    for row in rows {
        let (id, svc, u_enc, p_enc, url, n_enc, cat, created, updated) = row?;

        let username = enc.decrypt_string(&u_enc)?;
        let password = enc.decrypt_string(&p_enc)?;
        let notes = n_enc.map(|n| enc.decrypt_string(&n)).transpose()?;

        entries.push(VaultEntry {
            id,
            service: svc,
            username,
            password,
            url,
            notes,
            category: cat,
            created_at: created,
            updated_at: updated,
        });
    }
    Ok(entries)
}

/// List all vault entries (without decrypted sensitive fields).
pub fn list(conn: &Connection) -> Result<Vec<VaultListEntry>> {
    let mut stmt = conn.prepare(
        "SELECT id, service, url, category, updated_at
         FROM vault_entries ORDER BY service ASC"
    )?;

    let rows = stmt.query_map([], |row| {
        Ok(VaultListEntry {
            id: row.get(0)?,
            service: row.get(1)?,
            url: row.get(2)?,
            category: row.get(3)?,
            updated_at: row.get(4)?,
        })
    })?;

    rows.into_iter()
        .map(|r| Ok(r?))
        .collect()
}

/// Delete a vault entry by ID.
pub fn delete(conn: &Connection, id: i64) -> Result<bool> {
    let changed = conn.execute(
        "DELETE FROM vault_entries WHERE id = ?1",
        rusqlite::params![id],
    )?;
    Ok(changed > 0)
}

/// Delete a vault entry by service name.
pub fn delete_by_service(conn: &Connection, service: &str) -> Result<usize> {
    let changed = conn.execute(
        "DELETE FROM vault_entries WHERE service = ?1",
        rusqlite::params![service],
    )?;
    Ok(changed)
}

// ── Security PIN ──

/// Set or update the vault security PIN. Stores a blake3 hash.
pub fn set_pin(conn: &Connection, pin: &str) -> Result<()> {
    let hash = blake3::hash(pin.as_bytes()).to_hex().to_string();
    conn.execute(
        "INSERT OR REPLACE INTO vault_security (key, value) VALUES ('pin_hash', ?1)",
        rusqlite::params![hash],
    )?;
    Ok(())
}

/// Verify a PIN against the stored hash. Returns true if correct or if no PIN is set.
pub fn verify_pin(conn: &Connection, pin: &str) -> bool {
    let stored: Option<String> = conn
        .query_row(
            "SELECT value FROM vault_security WHERE key = 'pin_hash'",
            [],
            |row| row.get(0),
        )
        .ok();

    match stored {
        None => true,
        Some(hash) => {
            let input_hash = blake3::hash(pin.as_bytes()).to_hex().to_string();
            hash == input_hash
        }
    }
}

/// Check if a PIN is configured.
pub fn has_pin(conn: &Connection) -> bool {
    conn.query_row(
        "SELECT 1 FROM vault_security WHERE key = 'pin_hash'",
        [],
        |_| Ok(()),
    )
    .is_ok()
}

/// Remove the vault PIN.
pub fn remove_pin(conn: &Connection) -> Result<()> {
    conn.execute(
        "DELETE FROM vault_security WHERE key = 'pin_hash'",
        [],
    )?;
    Ok(())
}

/// Generate a cryptographically secure random password.
pub fn generate_password(length: usize, include_special: bool) -> String {
    let length = length.clamp(8, 128);
    let mut rng = rand::thread_rng();

    let lowercase = b"abcdefghijkmnopqrstuvwxyz";  // no l (ambiguous)
    let uppercase = b"ABCDEFGHJKLMNPQRSTUVWXYZ";    // no I, O (ambiguous)
    let digits = b"23456789";                       // no 0, 1 (ambiguous)
    let special = b"!@#$%^&*-_=+?";

    let mut charset: Vec<u8> = Vec::new();
    charset.extend_from_slice(lowercase);
    charset.extend_from_slice(uppercase);
    charset.extend_from_slice(digits);
    if include_special {
        charset.extend_from_slice(special);
    }

    // Ensure at least one of each category
    let mut password: Vec<u8> = Vec::with_capacity(length);
    password.push(lowercase[rng.gen_range(0..lowercase.len())]);
    password.push(uppercase[rng.gen_range(0..uppercase.len())]);
    password.push(digits[rng.gen_range(0..digits.len())]);
    if include_special {
        password.push(special[rng.gen_range(0..special.len())]);
    }

    while password.len() < length {
        password.push(charset[rng.gen_range(0..charset.len())]);
    }

    // Shuffle (Fisher-Yates)
    for i in (1..password.len()).rev() {
        let j = rng.gen_range(0..=i);
        password.swap(i, j);
    }

    String::from_utf8(password).expect("ASCII password")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encryption::{generate_key, EncryptionProvider};

    fn setup() -> (Connection, EncryptionProvider) {
        let conn = Connection::open_in_memory().unwrap();
        init_tables(&conn);
        let dek = generate_key();
        let enc = EncryptionProvider::from_dek(&dek);
        (conn, enc)
    }

    #[test]
    fn test_store_and_get() {
        let (conn, enc) = setup();
        store(&conn, &enc, "github.com", "user123", "pass456", Some("https://github.com"), None, None).unwrap();
        let entries = get(&conn, &enc, "github.com").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].username, "user123");
        assert_eq!(entries[0].password, "pass456");
        assert_eq!(entries[0].url.as_deref(), Some("https://github.com"));
    }

    #[test]
    fn test_upsert() {
        let (conn, enc) = setup();
        store(&conn, &enc, "gmail", "old@gmail.com", "old_pass", None, None, None).unwrap();
        store(&conn, &enc, "gmail", "new@gmail.com", "new_pass", None, None, None).unwrap();
        let entries = get(&conn, &enc, "gmail").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].username, "new@gmail.com");
        assert_eq!(entries[0].password, "new_pass");
    }

    #[test]
    fn test_list_no_passwords() {
        let (conn, enc) = setup();
        store(&conn, &enc, "github.com", "user", "secret", None, None, Some("dev")).unwrap();
        store(&conn, &enc, "gmail.com", "user@gmail.com", "secret2", None, None, Some("email")).unwrap();
        let list_entries = list(&conn).unwrap();
        assert_eq!(list_entries.len(), 2);
    }

    #[test]
    fn test_search() {
        let (conn, enc) = setup();
        store(&conn, &enc, "github.com", "u", "p", None, None, None).unwrap();
        store(&conn, &enc, "gitlab.com", "u", "p", None, None, None).unwrap();
        store(&conn, &enc, "gmail.com", "u", "p", None, None, None).unwrap();
        let results = search(&conn, &enc, "git").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_delete() {
        let (conn, enc) = setup();
        store(&conn, &enc, "test", "u", "p", None, None, None).unwrap();
        let entries = get(&conn, &enc, "test").unwrap();
        assert!(delete(&conn, entries[0].id).unwrap());
        assert!(get(&conn, &enc, "test").unwrap().is_empty());
    }

    #[test]
    fn test_generate_password() {
        let p1 = generate_password(20, true);
        assert_eq!(p1.len(), 20);
        let p2 = generate_password(20, true);
        assert_ne!(p1, p2);

        let p_no_special = generate_password(16, false);
        assert_eq!(p_no_special.len(), 16);
        assert!(!p_no_special.contains('!') && !p_no_special.contains('@'));
    }

    #[test]
    fn test_wrong_key_cant_decrypt() {
        let (conn, enc) = setup();
        store(&conn, &enc, "secret_service", "admin", "super_secret", None, None, None).unwrap();

        let other_key = generate_key();
        let other_enc = EncryptionProvider::from_dek(&other_key);
        assert!(get(&conn, &other_enc, "secret_service").is_err());
    }

    #[test]
    fn test_pin_set_verify() {
        let (conn, _) = setup();
        assert!(!has_pin(&conn));

        set_pin(&conn, "1234").unwrap();
        assert!(has_pin(&conn));
        assert!(verify_pin(&conn, "1234"));
        assert!(!verify_pin(&conn, "wrong"));
    }

    #[test]
    fn test_pin_remove() {
        let (conn, _) = setup();
        set_pin(&conn, "9999").unwrap();
        assert!(has_pin(&conn));
        remove_pin(&conn).unwrap();
        assert!(!has_pin(&conn));
    }

    #[test]
    fn test_pin_change() {
        let (conn, _) = setup();
        set_pin(&conn, "old_pin").unwrap();
        assert!(verify_pin(&conn, "old_pin"));
        set_pin(&conn, "new_pin").unwrap();
        assert!(!verify_pin(&conn, "old_pin"));
        assert!(verify_pin(&conn, "new_pin"));
    }
}
