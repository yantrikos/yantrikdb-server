//! `[embedding] strategy = "auto"`: choose the embedder from the data
//! directory rather than from a configured `dim` (v0.21.0, Docker default).
//!
//! Every tenant in one server shares one embedding dimension (the pool
//! opens all of them at `embedding.dim`), so the choice is made once at
//! startup from what the existing databases record about themselves:
//!
//! | Data directory holds                         | Result                          |
//! |----------------------------------------------|---------------------------------|
//! | nothing with memories                        | `potion-base-8M`, else fallback |
//! | only `potion-base-2M` / 64-dim databases     | bundled `potion-base-2M`        |
//! | only `potion-base-8M` / 256-dim databases    | `potion-base-8M`                |
//! | anything else, a mix, or unidentified memory | refuse, naming the strategy     |
//!
//! The evidence is the embedder identity the engine records in each
//! database's `meta` table the first time it produces a vector. `meta` is
//! plaintext even in an encrypted database, which is why it is used here
//! instead of measuring vector bytes (encrypted vectors are padded).

use std::path::Path;

use crate::control::DatabaseRecord;
use crate::embedder::{ServerEmbedder, SharedDownloaded};

/// The registry model a fresh data directory gets: the engine's own
/// default for new stores, so a server and an embedded engine agree.
pub const AUTO_MODEL: &str = yantrikdb::engine::DEFAULT_NEW_STORE_EMBEDDER;
const BUNDLED_NAME: &str = "potion-base-2M";

/// What one database says about the embedder that built it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreEvidence {
    /// No file yet, or no memories in it: nothing to stay compatible with.
    Empty,
    /// The identity recorded in `meta`.
    Identity { name: Option<String>, dim: usize },
    /// Holds memories but records no embedder (vectors supplied by clients,
    /// or written before identities existed). Its vector space is unknown.
    Unidentified,
}

/// The embedder `auto` settled on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoChoice {
    Bundled,
    Downloaded,
}

/// The decision plus the message an operator must see, if any.
#[derive(Debug, PartialEq, Eq)]
pub struct AutoDecision {
    pub choice: AutoChoice,
    pub warning: Option<String>,
}

/// Pure decision over the evidence. `fetch` is only called when the answer
/// depends on whether the default model can be obtained.
pub fn decide(
    evidence: &[(String, StoreEvidence)],
    fetch: impl FnOnce() -> Result<(), String>,
) -> anyhow::Result<AutoDecision> {
    let unidentified: Vec<&str> = evidence
        .iter()
        .filter(|(_, e)| *e == StoreEvidence::Unidentified)
        .map(|(n, _)| n.as_str())
        .collect();
    if !unidentified.is_empty() {
        anyhow::bail!(
            "embedding strategy \"auto\": database(s) {unidentified:?} hold memories but record no \
             embedder, so auto cannot tell which vector space they are in. Set [embedding] strategy \
             explicitly: \"bundled\" with dim = 64 for potion-base-2M data, \"builtin\" with \
             dim = 384 for MiniLM data, or \"client_only\" if clients supply the vectors."
        );
    }

    let mut kinds: Vec<(Option<String>, usize)> = Vec::new();
    for (_, e) in evidence {
        if let StoreEvidence::Identity { name, dim } = e {
            let k = (name.clone(), *dim);
            if !kinds.contains(&k) {
                kinds.push(k);
            }
        }
    }

    match kinds.as_slice() {
        [] => match fetch() {
            Ok(()) => Ok(AutoDecision {
                choice: AutoChoice::Downloaded,
                warning: None,
            }),
            Err(e) => Ok(AutoDecision {
                choice: AutoChoice::Bundled,
                warning: Some(format!(
                    "embedding strategy \"auto\": could not obtain {AUTO_MODEL} ({e}); this fresh \
                     data directory will use the bundled {BUNDLED_NAME} (64-dim) instead. That is \
                     permanent for the data written now: switching later means re-embedding. If \
                     this host can reach GitHub, fix connectivity and restart BEFORE writing data."
                )),
            }),
        },
        [(name, dim)]
            if *dim == yantrikdb::embedder::BUNDLED_EMBEDDER_DIM
                && name.as_deref().is_none_or(|n| n == BUNDLED_NAME) =>
        {
            Ok(AutoDecision {
                choice: AutoChoice::Bundled,
                warning: None,
            })
        }
        [(Some(name), dim)]
            if name == AUTO_MODEL
                && Some(*dim)
                    == yantrikdb::embedder::DownloadedEmbedder::registry_dim(AUTO_MODEL) =>
        {
            // Existing data needs this exact model: no fallback, because the
            // bundled model cannot read these vectors.
            fetch().map_err(|e| {
                anyhow::anyhow!(
                    "embedding strategy \"auto\": the data directory was built by {AUTO_MODEL}, \
                     which could not be loaded ({e}). It is cached after the first download; \
                     restore network access to GitHub once, or copy the model cache."
                )
            })?;
            Ok(AutoDecision {
                choice: AutoChoice::Downloaded,
                warning: None,
            })
        }
        [(name, dim)] => anyhow::bail!(
            "embedding strategy \"auto\": the data directory was built by {} at {dim} dims, which \
             auto does not serve (it serves {BUNDLED_NAME} and {AUTO_MODEL}). Set [embedding] \
             strategy and dim explicitly for this data (\"builtin\" with dim = 384 for MiniLM).",
            name.as_deref().unwrap_or("an unnamed embedder")
        ),
        many => anyhow::bail!(
            "embedding strategy \"auto\": databases in this data directory were built by different \
             embedders {many:?}, but one server serves one dimension. Split them across servers \
             or re-embed to one model."
        ),
    }
}

/// Read one database's evidence. A missing file, or one without a
/// `memories` table, is `Empty`: there is nothing in it to stay compatible
/// with. Opened read-only from a fresh connection before any engine opens
/// it, so this never contends with the engine's own SQLite.
pub fn read_evidence(db_file: &Path) -> anyhow::Result<StoreEvidence> {
    if !db_file.exists() {
        return Ok(StoreEvidence::Empty);
    }
    let conn = rusqlite::Connection::open_with_flags(
        db_file,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let has_table = |t: &str| -> anyhow::Result<bool> {
        Ok(conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
            [t],
            |r| r.get::<_, bool>(0),
        )?)
    };
    if has_table("meta")? {
        let get = |k: &str| -> anyhow::Result<Option<String>> {
            match conn.query_row("SELECT value FROM meta WHERE key = ?1", [k], |r| r.get(0)) {
                Ok(v) => Ok(Some(v)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(e.into()),
            }
        };
        if let Some(dim) = get("embedder_dim")?.and_then(|d| d.parse::<usize>().ok()) {
            return Ok(StoreEvidence::Identity {
                name: get("embedder_name")?,
                dim,
            });
        }
    }
    if has_table("memories")? {
        let any: bool =
            conn.query_row("SELECT EXISTS(SELECT 1 FROM memories)", [], |r| r.get(0))?;
        if any {
            return Ok(StoreEvidence::Unidentified);
        }
    }
    Ok(StoreEvidence::Empty)
}

/// Resolve `auto` for a data directory: the embedder to attach and the
/// dimension every tenant must open at.
pub fn resolve(
    data_dir: &Path,
    databases: &[DatabaseRecord],
) -> anyhow::Result<(ServerEmbedder, usize)> {
    let mut evidence = Vec::with_capacity(databases.len());
    for db in databases {
        let file = data_dir.join(&db.path).join("yantrik.db");
        evidence.push((db.name.clone(), read_evidence(&file)?));
    }

    let mut loaded = None;
    let decision = decide(&evidence, || {
        yantrikdb::embedder::DownloadedEmbedder::fetch(AUTO_MODEL)
            .map(|e| loaded = Some(e))
            .map_err(|e| e.to_string())
    })?;
    if let Some(w) = &decision.warning {
        tracing::warn!("{w}");
        // Also on stderr: the Docker-log lesson from issue #3, where a
        // tracing-only warning never reached `docker logs`.
        eprintln!("[yantrikdb] WARNING: {w}");
    }
    Ok(match decision.choice {
        AutoChoice::Bundled => (
            ServerEmbedder::Bundled(yantrikdb::embedder::BundledEmbedder::new()),
            yantrikdb::embedder::BUNDLED_EMBEDDER_DIM,
        ),
        AutoChoice::Downloaded => {
            let emb = loaded.expect("decide() returns Downloaded only after a successful fetch");
            let dim = yantrikdb::types::Embedder::dim(&emb);
            (
                ServerEmbedder::Downloaded(SharedDownloaded(std::sync::Arc::new(emb))),
                dim,
            )
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(name: Option<&str>, dim: usize) -> StoreEvidence {
        StoreEvidence::Identity {
            name: name.map(str::to_string),
            dim,
        }
    }
    fn ev(items: &[StoreEvidence]) -> Vec<(String, StoreEvidence)> {
        items
            .iter()
            .enumerate()
            .map(|(i, e)| (format!("db{i}"), e.clone()))
            .collect()
    }
    fn ok() -> Result<(), String> {
        Ok(())
    }
    fn offline() -> Result<(), String> {
        Err("offline".into())
    }
    fn never() -> Result<(), String> {
        panic!("fetch must not be called for this evidence")
    }
    fn dim8m() -> usize {
        yantrikdb::embedder::DownloadedEmbedder::registry_dim(AUTO_MODEL).unwrap()
    }

    #[test]
    fn fresh_directory_gets_the_default_model() {
        let d = decide(&ev(&[]), ok).unwrap();
        assert_eq!(d.choice, AutoChoice::Downloaded);
        assert!(d.warning.is_none());
        let d = decide(&ev(&[StoreEvidence::Empty, StoreEvidence::Empty]), ok).unwrap();
        assert_eq!(d.choice, AutoChoice::Downloaded);
    }

    #[test]
    fn fresh_directory_offline_falls_back_loudly() {
        let d = decide(&ev(&[StoreEvidence::Empty]), offline).unwrap();
        assert_eq!(d.choice, AutoChoice::Bundled);
        let w = d.warning.expect("an offline fallback must warn");
        assert!(w.contains("permanent") && w.contains("offline"), "{w}");
    }

    #[test]
    fn existing_bundled_volume_keeps_its_embedder_without_touching_the_network() {
        // The upgrade case this strategy exists for: a v0.8.16+ Docker volume.
        for e in [id(Some("potion-base-2M"), 64), id(None, 64)] {
            let d = decide(&ev(&[e, StoreEvidence::Empty]), never).unwrap();
            assert_eq!(d.choice, AutoChoice::Bundled);
            assert!(d.warning.is_none());
        }
    }

    #[test]
    fn existing_default_model_volume_keeps_it_and_never_falls_back() {
        let e = id(Some(AUTO_MODEL), dim8m());
        assert_eq!(
            decide(&ev(&[e.clone()]), ok).unwrap().choice,
            AutoChoice::Downloaded
        );
        // Offline with data built by the default model must refuse, not
        // fall back: the bundled model cannot read those vectors.
        let err = decide(&ev(&[e]), offline).unwrap_err().to_string();
        assert!(err.contains("could not be loaded"), "{err}");
    }

    #[test]
    fn refuses_what_it_cannot_serve() {
        let cases = [
            ev(&[StoreEvidence::Unidentified]),
            ev(&[id(Some("all-MiniLM-L6-v2"), 384)]),
            ev(&[id(Some("some-other-64d"), 64)]),
            ev(&[
                id(Some("potion-base-2M"), 64),
                id(Some(AUTO_MODEL), dim8m()),
            ]),
            ev(&[id(Some("potion-base-2M"), 64), StoreEvidence::Unidentified]),
        ];
        for c in cases {
            let err = decide(&c, never).unwrap_err().to_string();
            assert!(err.contains("strategy"), "refusal must name the fix: {err}");
        }
    }

    #[test]
    fn evidence_reads_identity_memories_and_absence() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.db");
        assert_eq!(read_evidence(&missing).unwrap(), StoreEvidence::Empty);

        let p = dir.path().join("a.db");
        let c = rusqlite::Connection::open(&p).unwrap();
        c.execute_batch(
            "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
             CREATE TABLE memories(rid TEXT);",
        )
        .unwrap();
        assert_eq!(read_evidence(&p).unwrap(), StoreEvidence::Empty);
        c.execute("INSERT INTO memories VALUES ('r1')", []).unwrap();
        assert_eq!(read_evidence(&p).unwrap(), StoreEvidence::Unidentified);
        c.execute_batch(
            "INSERT INTO meta VALUES ('embedder_dim','64'), ('embedder_name','potion-base-2M');",
        )
        .unwrap();
        assert_eq!(read_evidence(&p).unwrap(), id(Some("potion-base-2M"), 64));
    }

    /// The real engine writes the identity this module reads: a store the
    /// bundled embedder filled must read back as potion-base-2M / 64, and
    /// `auto` must then keep the bundled embedder for it. Uses the engine,
    /// not a hand-built table, so a rename of the meta keys breaks here.
    #[test]
    fn a_real_bundled_store_reads_back_as_bundled() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("yantrik.db");
        {
            let db = yantrikdb::YantrikDB::new(
                p.to_str().unwrap(),
                yantrikdb::embedder::BUNDLED_EMBEDDER_DIM,
            )
            .unwrap();
            db.record_text(
                "the deploy key lives on node4",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        }
        let e = read_evidence(&p).unwrap();
        assert_eq!(e, id(Some("potion-base-2M"), 64));
        let d = decide(&[("default".into(), e)], never).unwrap();
        assert_eq!(d.choice, AutoChoice::Bundled);
    }
}
