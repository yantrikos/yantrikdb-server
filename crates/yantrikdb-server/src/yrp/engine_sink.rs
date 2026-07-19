//! The production [`super::driver::ApplySink`] — where replicated YRP
//! entries land in real engine state.
//!
//! ## The atomicity story (the ApplySink contract, honestly)
//!
//! The sink's durable footprint spans three stores that cannot share one
//! SQLite transaction (per-tenant engine DBs, the shared commit log, and
//! the sink's own marker DB). The contract's guarantee — a crash never
//! separates an entry's effects from its applied-index/outcome record in
//! an observable way — is met by ORDER + IDEMPOTENCE instead:
//!
//! 1. **Commit-log first** ([`crate::commit::LocalSqliteCommitter`]):
//!    the `(tenant, op_id)` unique index makes this exactly-once — a
//!    crash-replay returns the ORIGINAL receipt (same shape openraft's
//!    state machine relied on).
//! 2. **Engine apply second** ([`crate::commit::Applier`]): the
//!    deterministic `*_with_rid` primitives are idempotent per rid/seq;
//!    `AlreadyApplied` is success.
//! 3. **Marker + outcome LAST**, in one transaction of the sink DB. The
//!    marker is what `durable_applied()` reports, so a crash anywhere
//!    before step 3 simply replays the entry through steps 1–2 (both
//!    no-ops) and completes step 3. The marker can never run AHEAD of
//!    the effects — which is the direction that would violate the
//!    contract (acking an apply that didn't happen).
//!
//! Client acks release only on the marker (the driver's `Applied` event),
//! so "success reported, effect missing" is structurally impossible.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use rusqlite::Connection;

use super::driver::ApplySink;
use super::op::YrpOp;
use super::replica::{LogEntry, Payload};
use crate::commit::{Applier, ApplyError, CommitError, CommitOptions, MemoryMutation, MutationCommitter, TenantId};

/// One applied entry's durable outcome — everything the gateway needs to
/// answer a deduped retry (the original rid), verify the full idempotency
/// key against hash collisions, and rebuild a commit receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct AppliedOutcome {
    pub yrp_index: u64,
    pub term: u64,
    pub tenant_id: i64,
    pub op_id: String,
    /// The protocol claim digest (`LogEntry.key`).
    pub key_hash: Option<u64>,
    /// The FULL client idempotency key (collision verification).
    pub key_str: Option<String>,
    /// Client-visible outcome id, when the mutation has one.
    pub rid: Option<String>,
    /// Per-tenant commit-log index assigned in step 1.
    pub tenant_log_index: u64,
    pub applied_at_unix_micros: i64,
}

/// Durable applied-index marker + outcome table (`yrp_apply.sqlite`).
pub struct OutcomeStore {
    conn: Mutex<Connection>,
    applied: AtomicU64,
}

impl OutcomeStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| format!("open yrp_apply: {e}"))?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self, String> {
        let conn = Connection::open_in_memory().map_err(|e| format!("open yrp_apply: {e}"))?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self, String> {
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             CREATE TABLE IF NOT EXISTS yrp_applied (
                 id INTEGER PRIMARY KEY CHECK (id = 1),
                 applied_index INTEGER NOT NULL
             );
             INSERT OR IGNORE INTO yrp_applied (id, applied_index) VALUES (1, 0);
             CREATE TABLE IF NOT EXISTS yrp_outcome (
                 yrp_index INTEGER PRIMARY KEY,
                 term INTEGER NOT NULL,
                 tenant_id INTEGER NOT NULL,
                 op_id TEXT NOT NULL,
                 key_hash INTEGER,
                 key_str TEXT,
                 rid TEXT,
                 tenant_log_index INTEGER NOT NULL,
                 applied_at_unix_micros INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_yrp_outcome_key
                 ON yrp_outcome (key_hash);",
        )
        .map_err(|e| format!("init yrp_apply schema: {e}"))?;
        let applied: i64 = conn
            .query_row("SELECT applied_index FROM yrp_applied WHERE id = 1", [], |r| {
                r.get(0)
            })
            .map_err(|e| format!("read applied marker: {e}"))?;
        Ok(Self {
            conn: Mutex::new(conn),
            applied: AtomicU64::new(applied as u64),
        })
    }

    /// Highest durably-applied YRP index.
    pub fn applied(&self) -> u64 {
        self.applied.load(Ordering::Acquire)
    }

    /// Record an applied op's outcome AND advance the marker, atomically.
    pub fn record(&self, out: &AppliedOutcome) -> Result<(), String> {
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction()
            .map_err(|e| format!("outcome tx: {e}"))?;
        tx.execute(
            "INSERT OR REPLACE INTO yrp_outcome
                 (yrp_index, term, tenant_id, op_id, key_hash, key_str, rid,
                  tenant_log_index, applied_at_unix_micros)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                out.yrp_index as i64,
                out.term as i64,
                out.tenant_id,
                out.op_id,
                out.key_hash.map(|k| k as i64),
                out.key_str,
                out.rid,
                out.tenant_log_index as i64,
                out.applied_at_unix_micros,
            ],
        )
        .map_err(|e| format!("insert outcome: {e}"))?;
        tx.execute(
            "UPDATE yrp_applied SET applied_index = MAX(applied_index, ?1) WHERE id = 1",
            [out.yrp_index as i64],
        )
        .map_err(|e| format!("advance marker: {e}"))?;
        tx.commit().map_err(|e| format!("commit outcome tx: {e}"))?;
        self.applied.fetch_max(out.yrp_index, Ordering::AcqRel);
        Ok(())
    }

    /// Advance the marker without an outcome row (protocol no-ops).
    pub fn advance(&self, index: u64) -> Result<(), String> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE yrp_applied SET applied_index = MAX(applied_index, ?1) WHERE id = 1",
            [index as i64],
        )
        .map_err(|e| format!("advance marker: {e}"))?;
        self.applied.fetch_max(index, Ordering::AcqRel);
        Ok(())
    }

    pub fn lookup_by_index(&self, yrp_index: u64) -> Result<Option<AppliedOutcome>, String> {
        let conn = self.conn.lock();
        Self::query_one(
            &conn,
            "SELECT yrp_index, term, tenant_id, op_id, key_hash, key_str, rid,
                    tenant_log_index, applied_at_unix_micros
             FROM yrp_outcome WHERE yrp_index = ?1",
            [yrp_index as i64],
        )
    }

    fn query_one(
        conn: &Connection,
        sql: &str,
        params: impl rusqlite::Params,
    ) -> Result<Option<AppliedOutcome>, String> {
        let mut stmt = conn.prepare(sql).map_err(|e| format!("prepare: {e}"))?;
        let mut rows = stmt
            .query(params)
            .map_err(|e| format!("query outcome: {e}"))?;
        match rows.next().map_err(|e| format!("row: {e}"))? {
            None => Ok(None),
            Some(r) => Ok(Some(AppliedOutcome {
                yrp_index: r.get::<_, i64>(0).map_err(|e| e.to_string())? as u64,
                term: r.get::<_, i64>(1).map_err(|e| e.to_string())? as u64,
                tenant_id: r.get(2).map_err(|e| e.to_string())?,
                op_id: r.get(3).map_err(|e| e.to_string())?,
                key_hash: r
                    .get::<_, Option<i64>>(4)
                    .map_err(|e| e.to_string())?
                    .map(|k| k as u64),
                key_str: r.get(5).map_err(|e| e.to_string())?,
                rid: r.get(6).map_err(|e| e.to_string())?,
                tenant_log_index: r.get::<_, i64>(7).map_err(|e| e.to_string())? as u64,
                applied_at_unix_micros: r.get(8).map_err(|e| e.to_string())?,
            })),
        }
    }
}

/// The rid a mutation makes client-visible, when it has one.
fn outcome_rid(m: &MemoryMutation) -> Option<String> {
    match m {
        MemoryMutation::UpsertMemory { rid, .. } => Some(rid.clone()),
        MemoryMutation::TombstoneMemory { rid, .. } => Some(rid.clone()),
        _ => None,
    }
}

/// Production sink: commit-log + engine + outcome marker, in the order
/// documented at module level.
pub struct EngineApplySink {
    committer: Arc<dyn MutationCommitter>,
    applier: Arc<dyn Applier>,
    outcomes: Arc<OutcomeStore>,
}

impl EngineApplySink {
    pub fn new(
        committer: Arc<dyn MutationCommitter>,
        applier: Arc<dyn Applier>,
        outcomes: Arc<OutcomeStore>,
    ) -> Self {
        Self {
            committer,
            applier,
            outcomes,
        }
    }
}

#[async_trait::async_trait]
impl ApplySink for EngineApplySink {
    async fn apply(&mut self, index: u64, entry: &LogEntry) -> Result<(), String> {
        let bytes = match &entry.payload {
            // Election no-ops and capability-activation carriers: nothing
            // to apply, just advance the durable marker.
            Payload::Noop => return self.outcomes.advance(index),
            // A Test payload on the production sink is a wiring bug —
            // fail-stop rather than silently skip (contract clause 3).
            Payload::Test(n) => {
                return Err(format!("Test payload {n} reached production sink at {index}"))
            }
            Payload::Op(b) => b,
        };
        let op = YrpOp::decode(bytes)?;

        // Step 1 — durable commit-log append, exactly-once on (tenant, op_id).
        let receipt = match self
            .committer
            .commit(
                op.tenant_id,
                op.mutation.clone(),
                CommitOptions::new().with_op_id(op.op_id),
            )
            .await
        {
            Ok(r) => r,
            // A collision here means a crash-replay carried a DIFFERENT
            // mutation for the same op_id — divergence evidence, fail-stop.
            Err(CommitError::OpIdCollision { op_id, .. }) => {
                return Err(format!("op_id collision during YRP apply: {op_id}"))
            }
            Err(e) => return Err(format!("commit-log append during YRP apply: {e}")),
        };

        // Step 2 — engine apply (live index updates inside). Seq = the
        // GLOBAL yrp index, mirroring openraft's use of the raft index.
        match self.applier.apply(op.tenant_id, index, &op.mutation).await {
            Ok(()) => {}
            Err(e) if e.is_idempotent_ok() => {}
            // Grammar variants without an apply path yet: non-fatal, same
            // posture as the openraft state machine (the commit-log row is
            // the durable truth; the engine wiring lands with its RFC).
            Err(ApplyError::NotYetWired { variant, .. }) => {
                tracing::warn!(variant, index, "YRP apply: mutation not yet wired to engine");
            }
            Err(e) => return Err(format!("engine apply at {index}: {e}")),
        }

        // Step 3 — outcome + marker, atomically, LAST.
        let applied_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);
        self.outcomes.record(&AppliedOutcome {
            yrp_index: index,
            term: entry.term.0,
            tenant_id: op.tenant_id.0,
            op_id: op.op_id.to_string(),
            key_hash: entry.key,
            key_str: op.idempotency_key.clone(),
            rid: outcome_rid(&op.mutation),
            tenant_log_index: receipt.log_index,
            applied_at_unix_micros: applied_at,
        })
    }

    fn durable_applied(&self) -> u64 {
        self.outcomes.applied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::{LocalSqliteCommitter, OpId};
    use crate::yrp::types::Term;

    fn op_entry(term: u64, key: Option<u64>, op: &YrpOp) -> LogEntry {
        LogEntry {
            term: Term(term),
            payload: Payload::Op(op.encode().unwrap()),
            key,
            activate: None,
        }
    }

    fn sample_op(key: Option<&str>) -> YrpOp {
        YrpOp {
            tenant_id: TenantId::new(1),
            op_id: OpId::new_random(),
            mutation: MemoryMutation::UpsertMemory {
                rid: "rid-sink-test".into(),
                text: "sink test".into(),
                memory_type: "semantic".into(),
                importance: 0.5,
                valence: 0.0,
                half_life: 168.0,
                metadata: serde_json::json!({}),
                namespace: "ns".into(),
                certainty: 1.0,
                domain: "work".into(),
                source: "user".into(),
                emotional_state: None,
                embedding: Some(vec![0.1, 0.2]),
                extracted_entities: vec![],
                created_at_unix_micros: Some(1),
                embedding_model: Some("default".into()),
            },
            idempotency_key: key.map(String::from),
        }
    }

    /// The crash-replay property end-to-end: applying the same index
    /// twice (as after a crash between engine apply and marker) yields
    /// ONE commit-log row, ONE outcome, and the same receipt fields.
    #[tokio::test]
    async fn replay_of_same_index_is_idempotent() {
        let committer: Arc<dyn MutationCommitter> =
            Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let applier: Arc<dyn Applier> = Arc::new(crate::commit::LocalApplier::new());
        let outcomes = Arc::new(OutcomeStore::open_in_memory().unwrap());
        let mut sink = EngineApplySink::new(committer.clone(), applier, outcomes.clone());

        let op = sample_op(Some("k1"));
        let key = Some(crate::yrp::op::claim_key_for_idempotency(op.tenant_id, "k1"));
        let entry = op_entry(3, key, &op);

        sink.apply(5, &entry).await.expect("first apply");
        assert_eq!(sink.durable_applied(), 5);
        let first = outcomes.lookup_by_index(5).unwrap().expect("outcome");
        assert_eq!(first.rid.as_deref(), Some("rid-sink-test"));
        assert_eq!(first.key_str.as_deref(), Some("k1"));
        assert_eq!(first.tenant_log_index, 1);

        // Crash-replay: same index, same entry.
        sink.apply(5, &entry).await.expect("replay apply");
        let again = outcomes.lookup_by_index(5).unwrap().expect("outcome");
        assert_eq!(first.tenant_log_index, again.tenant_log_index);
        assert_eq!(
            committer.high_watermark(op.tenant_id).await.unwrap(),
            1,
            "replay must not append a second commit-log row"
        );
    }

    /// Noop entries advance the marker without an outcome row; the marker
    /// survives reopen (recovered `durable_applied`).
    #[tokio::test]
    async fn noop_advances_marker_and_marker_is_durable() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("yrp_apply.sqlite");
        {
            let committer: Arc<dyn MutationCommitter> =
                Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
            let applier: Arc<dyn Applier> = Arc::new(crate::commit::LocalApplier::new());
            let outcomes = Arc::new(OutcomeStore::open(&path).unwrap());
            let mut sink = EngineApplySink::new(committer, applier, outcomes.clone());
            let noop = LogEntry {
                term: Term(1),
                payload: Payload::Noop,
                key: None,
                activate: None,
            };
            sink.apply(1, &noop).await.unwrap();
            assert_eq!(sink.durable_applied(), 1);
            assert!(outcomes.lookup_by_index(1).unwrap().is_none());
        }
        let reopened = OutcomeStore::open(&path).unwrap();
        assert_eq!(reopened.applied(), 1, "marker must survive restart");
    }
}
