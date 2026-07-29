//! RFC 031 — the pack reconciler.
//!
//! A per-node background actuator that makes this node's *physical* pack mounts
//! match the *replicated* `db_packs` manifest (the cluster's intent). This is
//! the best-effort half of the manifest/actuator split: the manifest apply is
//! fail-stop-safe consensus; the reconciler here can fail transiently (a file
//! not yet fetched) without fencing the cluster.
//!
//! Safety properties the adversarial review demanded (RFC 031 §Review):
//! - **Poison quarantine (C1):** a mount that panics or fails
//!   `MAX_ATTEMPTS` times moves that `(db, digest)` to a TERMINAL local
//!   quarantine (persisted, survives restart) so a bad pack can never
//!   crash-loop the cluster. The mount runs under `catch_unwind`.
//! - **Fail-visible (H1):** per-db status (mounted / pending / poisoned) is
//!   published for `/v1/health`, recall, and the admin API — never silent.
//! - **Verified-present (M2):** a file counts as present only if it hashes to
//!   its digest; a truncated file re-fetches. Fetch is single-flight (one
//!   reconcile at a time), size-capped, and digest-verified before store.
//! - **Bounded load (M2/H2):** only databases that actually have a manifest
//!   pack are engine-loaded — an intentional, bounded set, not every tenant.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Mutex, RwLock};

use crate::control::ControlDb;
use crate::pack_store::PackStore;
use crate::tenant_pool::TenantPool;

const RECONCILE_INTERVAL: Duration = Duration::from_secs(10);
const MOUNT_MAX_ATTEMPTS: u32 = 3;
/// Cap on a fetched pack file (bounds disk + the engine's HNSW build — H2/C1).
const PACK_MAX_BYTES: u64 = 64 * 1024 * 1024;
const FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Per-database physical-mount status, published for health / recall / API.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct DbPackStatus {
    /// Digests physically mounted and serving.
    pub mounted: Vec<String>,
    /// In the manifest but not yet mounted here (fetching / engine cold).
    pub pending: Vec<String>,
    /// Terminally quarantined on this node (a mount that crashed/failed).
    pub poisoned: Vec<String>,
}

/// Shared, node-local view of pack reconcile state. Cheap to read on hot paths.
#[derive(Default)]
pub struct PackStatus {
    per_db: RwLock<BTreeMap<i64, DbPackStatus>>,
}

impl PackStatus {
    pub fn get(&self, database_id: i64) -> DbPackStatus {
        self.per_db
            .read()
            .get(&database_id)
            .cloned()
            .unwrap_or_default()
    }
    /// Any database with packs not fully mounted (for /v1/health rollup).
    pub fn any_incomplete(&self) -> bool {
        self.per_db
            .read()
            .values()
            .any(|s| !s.pending.is_empty() || !s.poisoned.is_empty())
    }
    fn set(&self, database_id: i64, status: DbPackStatus) {
        self.per_db.write().insert(database_id, status);
    }
}

/// Fetches a missing pack file from a cluster peer (the leader first). `None`
/// in single-node mode, where the file is always already local.
pub trait PackFetcher: Send + Sync {
    /// Best (leader) base url + the peer secret, if a cluster is running.
    fn leader_base_and_secret(&self) -> Option<(String, String)>;
}

pub struct PackReconciler {
    control: Arc<Mutex<ControlDb>>,
    store: PackStore,
    pool: Arc<TenantPool>,
    status: Arc<PackStatus>,
    fetcher: Option<Arc<dyn PackFetcher>>,
    quarantine_path: std::path::PathBuf,
    /// (db_id, digest) -> failed attempts this process; promotes to quarantine.
    attempts: Mutex<BTreeMap<(i64, String), u32>>,
    quarantine: RwLock<BTreeSet<(i64, String)>>,
}

impl PackReconciler {
    pub fn new(
        control: Arc<Mutex<ControlDb>>,
        store: PackStore,
        pool: Arc<TenantPool>,
        status: Arc<PackStatus>,
        fetcher: Option<Arc<dyn PackFetcher>>,
        data_dir: &Path,
    ) -> Arc<Self> {
        let quarantine_path = data_dir.join("packs").join("quarantine.json");
        let quarantine = load_quarantine(&quarantine_path);
        Arc::new(Self {
            control,
            store,
            pool,
            status,
            fetcher,
            quarantine_path,
            attempts: Mutex::new(BTreeMap::new()),
            quarantine: RwLock::new(quarantine),
        })
    }

    /// Spawn the reconcile loop (single-flight: one tick at a time).
    pub fn spawn(self: Arc<Self>) {
        tokio::spawn(async move {
            loop {
                if let Err(e) = self.reconcile_once().await {
                    tracing::warn!(error = %e, "pack reconcile tick failed");
                }
                tokio::time::sleep(RECONCILE_INTERVAL).await;
            }
        });
    }

    /// One reconcile pass over the whole manifest. Public so it can be driven
    /// synchronously in tests.
    pub async fn reconcile_once(&self) -> anyhow::Result<()> {
        // Desired state grouped by database (unmounted rows are excluded).
        let desired = self.control.lock().active_pack_mounts()?;
        let mut by_db: BTreeMap<i64, Vec<(String, String)>> = BTreeMap::new();
        for m in desired {
            by_db
                .entry(m.database_id)
                .or_default()
                .push((m.pack_digest, m.pack_name));
        }

        for (db_id, wanted) in by_db {
            if let Err(e) = self.reconcile_db(db_id, &wanted).await {
                tracing::warn!(db_id, error = %e, "pack reconcile for db failed");
            }
        }
        Ok(())
    }

    async fn reconcile_db(&self, db_id: i64, wanted: &[(String, String)]) -> anyhow::Result<()> {
        // Resolve the tenant engine (lazy-loads — bounded: only dbs with packs
        // reach here). A missing/failed db is a no-op (H3), reported pending.
        let db_record = match self.control.lock().get_database_by_id(db_id)? {
            Some(r) => r,
            None => return Ok(()), // orphan manifest row (db gone) — ignore
        };
        let engine = match self.pool.get_engine(&db_record) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(db_id, error = %e, "engine load failed; packs pending");
                let mut st = DbPackStatus::default();
                st.pending = wanted.iter().map(|(d, _)| d.clone()).collect();
                self.status.set(db_id, st);
                return Ok(());
            }
        };

        let wanted_digests: BTreeSet<&str> = wanted.iter().map(|(d, _)| d.as_str()).collect();
        // Current physical mounts (digest recovered from the file path).
        let mounted = engine.mounted_packs();
        let mut mounted_by_digest: BTreeMap<String, String> = BTreeMap::new(); // digest -> pack_id
        for p in &mounted {
            if let Some(d) = digest_from_path(&p.path) {
                mounted_by_digest.insert(d, p.pack_id.clone());
            }
        }

        // 1. Unmount anything mounted that the manifest no longer wants.
        for (digest, pack_id) in &mounted_by_digest {
            if !wanted_digests.contains(digest.as_str()) {
                let _ = engine.unmount_pack(pack_id);
            }
        }

        let mut status = DbPackStatus::default();
        // 2. Mount everything wanted that isn't mounted and isn't quarantined.
        for (digest, _name) in wanted {
            if mounted_by_digest.contains_key(digest) {
                status.mounted.push(digest.clone());
                continue;
            }
            if self.quarantine.read().contains(&(db_id, digest.clone())) {
                status.poisoned.push(digest.clone());
                continue;
            }
            match self.ensure_file(digest).await {
                Ok(true) => {}
                Ok(false) | Err(_) => {
                    status.pending.push(digest.clone());
                    continue;
                }
            }
            // Physical mount under catch_unwind — a pack that panics the engine
            // must not take down the process (C1).
            let path = match self.store.path(digest) {
                Some(p) => p.to_string_lossy().to_string(),
                None => continue,
            };
            let engine2 = engine.clone();
            let path2 = path.clone();
            let result = tokio::task::spawn_blocking(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine2.mount_pack(&path2)
                }))
            })
            .await;
            let ok = matches!(result, Ok(Ok(Ok(_))));
            if ok {
                status.mounted.push(digest.clone());
                self.attempts.lock().remove(&(db_id, digest.clone()));
            } else {
                let attempts = {
                    let mut a = self.attempts.lock();
                    let n = a.entry((db_id, digest.clone())).or_insert(0);
                    *n += 1;
                    *n
                };
                // A panic (Err(_) from catch_unwind) quarantines immediately;
                // a plain error gets MAX_ATTEMPTS before terminal quarantine.
                let panicked = matches!(&result, Ok(Err(_)));
                if panicked || attempts >= MOUNT_MAX_ATTEMPTS {
                    tracing::error!(
                        db_id,
                        digest,
                        attempts,
                        panicked,
                        "pack mount terminally failed — quarantining (poison)"
                    );
                    self.quarantine.write().insert((db_id, digest.clone()));
                    self.persist_quarantine();
                    status.poisoned.push(digest.clone());
                } else {
                    status.pending.push(digest.clone());
                }
            }
        }
        self.status.set(db_id, status);
        Ok(())
    }

    /// Ensure the pack file is present-and-verified locally. Fetches from the
    /// leader if missing (cluster mode); size-capped + digest-verified.
    async fn ensure_file(&self, digest: &str) -> anyhow::Result<bool> {
        if self.store.has(digest) {
            // has() is name-present; confirm it verifies (heals a torn file).
            if self.store.load(digest).is_ok() {
                return Ok(true);
            }
            // Corrupt/torn — remove and re-fetch.
            if let Some(p) = self.store.path(digest) {
                let _ = std::fs::remove_file(p);
            }
        }
        let Some(fetcher) = &self.fetcher else {
            return Ok(false); // single-node: file should have been local
        };
        let Some((base, secret)) = fetcher.leader_base_and_secret() else {
            return Ok(false);
        };
        let client = reqwest::Client::builder().timeout(FETCH_TIMEOUT).build()?;
        let resp = client
            .get(format!("{base}/v1/packs/{digest}"))
            .bearer_auth(secret)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Ok(false);
        }
        if let Some(len) = resp.content_length() {
            if len > PACK_MAX_BYTES {
                anyhow::bail!("pack {digest} exceeds size cap ({len} bytes)");
            }
        }
        let bytes = resp.bytes().await?;
        if bytes.len() as u64 > PACK_MAX_BYTES {
            anyhow::bail!("pack {digest} stream exceeded size cap");
        }
        self.store.store_verified(digest, &bytes)?; // rejects a digest mismatch
        Ok(true)
    }

    fn persist_quarantine(&self) {
        let list: Vec<String> = self
            .quarantine
            .read()
            .iter()
            .map(|(db, d)| format!("{db}:{d}"))
            .collect();
        if let Ok(json) = serde_json::to_vec(&list) {
            let _ = std::fs::write(&self.quarantine_path, json);
        }
    }
}

fn load_quarantine(path: &Path) -> BTreeSet<(i64, String)> {
    let mut out = BTreeSet::new();
    if let Ok(bytes) = std::fs::read(path) {
        if let Ok(list) = serde_json::from_slice::<Vec<String>>(&bytes) {
            for e in list {
                if let Some((db, d)) = e.split_once(':') {
                    if let Ok(db) = db.parse::<i64>() {
                        out.insert((db, d.to_string()));
                    }
                }
            }
        }
    }
    out
}

/// A YRP node fetches missing pack files from the current leader (peer auth =
/// `cluster_secret`). On the leader itself the file is already local, so this
/// only fires on followers / rejoining nodes.
impl PackFetcher for crate::yrp::runtime::YrpHandle {
    fn leader_base_and_secret(&self) -> Option<(String, String)> {
        let (_, addr) = self.leader_hint();
        Some((addr?, self.cluster_secret.clone()?))
    }
}

/// Recover a pack's content digest from its stored file path (`<digest>.ydbpack`).
fn digest_from_path(path: &str) -> Option<String> {
    let name = Path::new(path).file_name()?.to_string_lossy();
    let d = name.strip_suffix(".ydbpack")?;
    if PackStore::is_valid_digest(d) {
        Some(d.to_string())
    } else {
        None
    }
}
