//! RFC 031 — content-addressed pack file store.
//!
//! Pack `.ydbpack` files are large opaque binaries that must NOT ride the YRP
//! consensus log (only the small mount *manifest* does — see RFC 031). They
//! live on disk under `data_dir/packs/<blake3-digest>.ydbpack`, keyed by their
//! content digest so identity == digest and duplicate uploads dedupe for free.
//!
//! The digest is validated as 64 lowercase hex chars before it ever touches a
//! path, so a digest supplied by a peer or a client can never traverse the
//! filesystem (`../`, absolute paths, etc. are rejected — RFC 031 Invariant 2).

use std::path::{Path, PathBuf};

/// A content-addressed store of pack files.
#[derive(Clone)]
pub struct PackStore {
    dir: PathBuf,
}

/// A stored pack's on-disk summary (for listings).
#[derive(Debug, Clone, serde::Serialize)]
pub struct StoredPack {
    pub digest: String,
    pub size: u64,
}

impl PackStore {
    /// Open (creating if absent) the pack store under `data_dir/packs/`.
    pub fn open(data_dir: &Path) -> std::io::Result<Self> {
        let dir = data_dir.join("packs");
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// True iff `s` is a well-formed blake3 hex digest (64 lowercase hex).
    /// Everything that turns a digest into a path goes through this first.
    pub fn is_valid_digest(s: &str) -> bool {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    /// The on-disk path for a digest, or `None` if the digest is malformed
    /// (which also makes path traversal impossible).
    pub fn path(&self, digest: &str) -> Option<PathBuf> {
        if Self::is_valid_digest(digest) {
            Some(self.dir.join(format!("{digest}.ydbpack")))
        } else {
            None
        }
    }

    pub fn digest_of(bytes: &[u8]) -> String {
        blake3::hash(bytes).to_hex().to_string()
    }

    pub fn has(&self, digest: &str) -> bool {
        self.path(digest).map(|p| p.exists()).unwrap_or(false)
    }

    /// Store bytes under their own content digest (atomic via temp+rename).
    /// Returns the digest. Idempotent: re-storing identical bytes is a no-op.
    pub fn store(&self, bytes: &[u8]) -> std::io::Result<String> {
        let digest = Self::digest_of(bytes);
        self.write_atomic(&digest, bytes)?;
        Ok(digest)
    }

    /// Store bytes that a PEER served for an EXPECTED digest — verifying the
    /// bytes actually hash to it before writing (RFC 031 Invariant 2). A
    /// mismatched stream is refused, never stored.
    pub fn store_verified(&self, expected: &str, bytes: &[u8]) -> std::io::Result<()> {
        let actual = Self::digest_of(bytes);
        if actual != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("pack digest mismatch: expected {expected}, got {actual}"),
            ));
        }
        self.write_atomic(expected, bytes)
    }

    /// Read a stored pack, re-verifying its digest on the way out (guards
    /// against on-disk corruption before we hand bytes to a peer or the engine).
    pub fn load(&self, digest: &str) -> std::io::Result<Vec<u8>> {
        let path = self
            .path(digest)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "bad digest"))?;
        let bytes = std::fs::read(path)?;
        if Self::digest_of(&bytes) != digest {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "stored pack failed digest re-verification (corrupt)",
            ));
        }
        Ok(bytes)
    }

    pub fn list(&self) -> Vec<StoredPack> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let name = e.file_name();
                let name = name.to_string_lossy();
                if let Some(d) = name.strip_suffix(".ydbpack") {
                    if Self::is_valid_digest(d) {
                        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                        out.push(StoredPack {
                            digest: d.to_string(),
                            size,
                        });
                    }
                }
            }
        }
        out.sort_by(|a, b| a.digest.cmp(&b.digest));
        out
    }

    fn write_atomic(&self, digest: &str, bytes: &[u8]) -> std::io::Result<()> {
        let path = self
            .path(digest)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "bad digest"))?;
        if path.exists() {
            return Ok(()); // content-addressed: identical bytes already present
        }
        let tmp = self.dir.join(format!(".{digest}.tmp"));
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_load_roundtrip_and_dedup() {
        let tmp = tempfile::tempdir().unwrap();
        let store = PackStore::open(tmp.path()).unwrap();
        let bytes = b"a sealed pack's bytes";
        let digest = store.store(bytes).unwrap();
        assert!(PackStore::is_valid_digest(&digest));
        assert!(store.has(&digest));
        assert_eq!(store.load(&digest).unwrap(), bytes);
        // dedup: storing again is a no-op returning the same digest
        assert_eq!(store.store(bytes).unwrap(), digest);
        assert_eq!(store.list().len(), 1);
    }

    #[test]
    fn verified_store_rejects_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let store = PackStore::open(tmp.path()).unwrap();
        let good = PackStore::digest_of(b"real bytes");
        assert!(store.store_verified(&good, b"real bytes").is_ok());
        // a peer claiming a digest but serving other bytes is refused
        assert!(store.store_verified(&good, b"TAMPERED").is_err());
    }

    #[test]
    fn malformed_digest_cannot_traverse() {
        let tmp = tempfile::tempdir().unwrap();
        let store = PackStore::open(tmp.path()).unwrap();
        assert!(store.path("../../etc/passwd").is_none());
        assert!(store.path("not-hex").is_none());
        assert!(store.path(&"g".repeat(64)).is_none()); // non-hex chars
        assert!(!store.has("../../etc/passwd"));
    }
}
