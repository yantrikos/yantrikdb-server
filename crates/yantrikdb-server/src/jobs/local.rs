//! `LocalSqliteJobQueue` — single-node JobQueue backed by `durable_jobs`
//! table (migration m002).
//!
//! ## Concurrency model
//!
//! Same pattern as `LocalSqliteCommitter` (RFC 010 PR-2): rusqlite is
//! sync, wrapped in `Arc<Mutex<Connection>>`, all SQL inside
//! `tokio::task::spawn_blocking`. Lease acquisition uses a transaction
//! to make "find next Pending + flip to Leased" atomic; without that,
//! two workers could lease the same job simultaneously.
//!
//! ## Pickup ordering
//!
//! `acquire_lease` selects the highest-priority Pending job whose kind
//! matches the worker's filter, breaking ties by oldest `created_at`.
//! Index `idx_durable_jobs_pickup` covers this query.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};

use super::trait_def::{
    JobError, JobId, JobQueue, JobRecord, JobState, Lease, LeaseId, NewJob, Outcome, WorkerId,
};
use crate::commit::TenantId;
use crate::migrations::MigrationRunner;

pub struct LocalSqliteJobQueue {
    conn: Arc<Mutex<Connection>>,
}

impl LocalSqliteJobQueue {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, JobError> {
        let mut conn = Connection::open(path).map_err(|e| JobError::StorageFailure {
            message: format!("failed to open SQLite: {e}"),
        })?;
        Self::configure_pragmas(&conn)?;
        Self::run_migrations(&mut conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn open_in_memory() -> Result<Self, JobError> {
        let mut conn =
            Connection::open_in_memory().map_err(|e| JobError::StorageFailure {
                message: format!("failed to open in-memory SQLite: {e}"),
            })?;
        Self::configure_pragmas(&conn)?;
        Self::run_migrations(&mut conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn configure_pragmas(conn: &Connection) -> Result<(), JobError> {
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;\n\
             PRAGMA synchronous=NORMAL;\n\
             PRAGMA foreign_keys=ON;",
        )
        .map_err(|e| JobError::StorageFailure {
            message: format!("PRAGMA setup failed: {e}"),
        })?;
        Ok(())
    }

    fn run_migrations(conn: &mut Connection) -> Result<(), JobError> {
        MigrationRunner::run_pending(conn).map_err(|e| JobError::StorageFailure {
            message: format!("migration runner failed: {e}"),
        })?;
        Ok(())
    }
}

#[async_trait]
impl JobQueue for LocalSqliteJobQueue {
    async fn enqueue(&self, job: NewJob) -> Result<JobId, JobError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<JobId, JobError> {
            let conn = conn.lock();
            let id = JobId::new_random();
            let now_micros = systime_to_micros(SystemTime::now());
            let payload = serde_json::to_vec(&job.payload).map_err(|e| {
                JobError::StorageFailure {
                    message: format!("payload serialize failed: {e}"),
                }
            })?;
            conn.execute(
                "INSERT INTO durable_jobs (
                    id, tenant_id, kind, payload, state, priority, created_at_unix_micros
                 ) VALUES (?1,?2,?3,?4,?5,?6,?7)",
                params![
                    id.to_string(),
                    job.tenant_id.0,
                    job.kind,
                    payload,
                    JobState::Pending.as_str(),
                    job.priority as i64,
                    now_micros,
                ],
            )
            .map_err(|e| JobError::StorageFailure {
                message: format!("INSERT failed: {e}"),
            })?;
            Ok(id)
        })
        .await
        .map_err(|e| JobError::StorageFailure {
            message: format!("spawn_blocking join failed: {e}"),
        })?
    }

    async fn acquire_lease(
        &self,
        worker_id: WorkerId,
        kinds_filter: Option<Vec<String>>,
        tenant_filter: Option<TenantId>,
        lease_ttl_secs: u64,
    ) -> Result<Option<Lease>, JobError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<Option<Lease>, JobError> {
            let mut conn = conn.lock();
            // Transaction so "find next Pending + flip to Leased" is atomic.
            let tx = conn.transaction().map_err(|e| JobError::StorageFailure {
                message: format!("begin transaction: {e}"),
            })?;

            // Build the WHERE clause + params dynamically.
            // Pickup: state='Pending', highest priority, oldest created_at.
            let mut sql = String::from(
                "SELECT id, tenant_id, kind, payload, priority, created_at_unix_micros
                 FROM durable_jobs
                 WHERE state = 'Pending'",
            );
            // Track positional params; rusqlite needs trait-object'd values.
            let mut params_vec: Vec<rusqlite::types::Value> = Vec::new();

            if let Some(tid) = tenant_filter {
                sql.push_str(" AND tenant_id = ?");
                params_vec.push(rusqlite::types::Value::Integer(tid.0));
            }
            if let Some(ref kinds) = kinds_filter {
                if kinds.is_empty() {
                    // Empty filter = no kinds match = no job.
                    return Ok(None);
                }
                let placeholders: Vec<&'static str> = kinds.iter().map(|_| "?").collect();
                sql.push_str(&format!(" AND kind IN ({})", placeholders.join(",")));
                for k in kinds {
                    params_vec.push(rusqlite::types::Value::Text(k.clone()));
                }
            }
            sql.push_str(" ORDER BY priority DESC, created_at_unix_micros ASC LIMIT 1");

            let maybe_row: Option<(String, i64, String, Vec<u8>, i64, i64)> = {
                let mut stmt = tx.prepare(&sql).map_err(|e| JobError::StorageFailure {
                    message: format!("prepare pickup query: {e}"),
                })?;
                let param_refs: Vec<&dyn rusqlite::ToSql> =
                    params_vec.iter().map(|v| v as &dyn rusqlite::ToSql).collect();
                stmt.query_row(param_refs.as_slice(), |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Vec<u8>>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, i64>(5)?,
                    ))
                })
                .optional()
                .map_err(|e| JobError::StorageFailure {
                    message: format!("pickup query: {e}"),
                })?
            };

            let (id_str, tenant_id, kind, payload_bytes, priority, created_at_micros) =
                match maybe_row {
                    Some(r) => r,
                    None => return Ok(None),
                };

            let lease_id = LeaseId::new_random();
            let now = SystemTime::now();
            let now_micros = systime_to_micros(now);
            let expires_micros = now_micros + (lease_ttl_secs * 1_000_000) as i64;

            // Flip to Leased.
            let updated = tx.execute(
                "UPDATE durable_jobs
                 SET state = 'Leased',
                     leased_by = ?1,
                     leased_at_unix_micros = ?2,
                     lease_expires_at_unix_micros = ?3
                 WHERE id = ?4 AND state = 'Pending'",
                params![worker_id.0, now_micros, expires_micros, id_str],
            )
            .map_err(|e| JobError::StorageFailure {
                message: format!("UPDATE to Leased: {e}"),
            })?;

            if updated != 1 {
                // Race: another worker leased between our SELECT and
                // UPDATE. Roll back and return None — the caller can
                // retry with a fresh acquire_lease call.
                return Ok(None);
            }

            tx.commit().map_err(|e| JobError::StorageFailure {
                message: format!("commit: {e}"),
            })?;

            let payload: serde_json::Value = serde_json::from_slice(&payload_bytes).map_err(|e| {
                JobError::StorageFailure {
                    message: format!("payload deserialize: {e}"),
                }
            })?;
            let job_id = id_str
                .parse::<uuid7::Uuid>()
                .map(JobId)
                .map_err(|e| JobError::StorageFailure {
                    message: format!("job id parse: {e}"),
                })?;
            let job_record = JobRecord {
                id: job_id,
                tenant_id: TenantId::new(tenant_id),
                kind,
                payload,
                state: JobState::Leased,
                priority: priority.clamp(0, 255) as u8,
                created_at: micros_to_systime(created_at_micros),
                leased_by: Some(worker_id.clone()),
                leased_at: Some(now),
                lease_expires_at: Some(micros_to_systime(expires_micros)),
                completed_at: None,
                error_message: None,
            };
            Ok(Some(Lease {
                lease_id,
                job_id,
                worker_id,
                acquired_at: now,
                expires_at: micros_to_systime(expires_micros),
                job: job_record,
            }))
        })
        .await
        .map_err(|e| JobError::StorageFailure {
            message: format!("spawn_blocking join failed: {e}"),
        })?
    }

    async fn heartbeat(
        &self,
        lease_id: LeaseId,
        job_id: JobId,
        lease_ttl_secs: u64,
    ) -> Result<(), JobError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<(), JobError> {
            let conn = conn.lock();
            let now_micros = systime_to_micros(SystemTime::now());
            let new_expires = now_micros + (lease_ttl_secs * 1_000_000) as i64;
            // Only extend if still Leased and the row's expires_at is in
            // the future (i.e. lease hasn't been reclaimed).
            let updated = conn
                .execute(
                    "UPDATE durable_jobs
                     SET lease_expires_at_unix_micros = ?1
                     WHERE id = ?2 AND state = 'Leased'
                       AND lease_expires_at_unix_micros > ?3",
                    params![new_expires, job_id.to_string(), now_micros],
                )
                .map_err(|e| JobError::StorageFailure {
                    message: format!("heartbeat UPDATE: {e}"),
                })?;
            if updated == 0 {
                return Err(JobError::InvalidLease { job_id, lease_id });
            }
            Ok(())
        })
        .await
        .map_err(|e| JobError::StorageFailure {
            message: format!("spawn_blocking join failed: {e}"),
        })?
    }

    async fn complete(
        &self,
        lease_id: LeaseId,
        job_id: JobId,
        outcome: Outcome,
    ) -> Result<(), JobError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<(), JobError> {
            let conn = conn.lock();
            let now_micros = systime_to_micros(SystemTime::now());
            let (state, error_message) = match &outcome {
                Outcome::Succeeded => (JobState::Succeeded, None),
                Outcome::Failed { error_message } => (JobState::Failed, Some(error_message.clone())),
            };
            let updated = conn
                .execute(
                    "UPDATE durable_jobs
                     SET state = ?1,
                         completed_at_unix_micros = ?2,
                         outcome = ?3,
                         error_message = ?4
                     WHERE id = ?5 AND state = 'Leased'
                       AND lease_expires_at_unix_micros > ?6",
                    params![
                        state.as_str(),
                        now_micros,
                        match outcome {
                            Outcome::Succeeded => "Succeeded",
                            Outcome::Failed { .. } => "Failed",
                        },
                        error_message,
                        job_id.to_string(),
                        now_micros,
                    ],
                )
                .map_err(|e| JobError::StorageFailure {
                    message: format!("complete UPDATE: {e}"),
                })?;
            if updated == 0 {
                return Err(JobError::InvalidLease { job_id, lease_id });
            }
            Ok(())
        })
        .await
        .map_err(|e| JobError::StorageFailure {
            message: format!("spawn_blocking join failed: {e}"),
        })?
    }

    async fn cancel(&self, job_id: JobId) -> Result<(), JobError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<(), JobError> {
            let conn = conn.lock();
            // Look up current state first to give a clear error on
            // terminal-state transitions.
            let row: Option<(String,)> = conn
                .query_row(
                    "SELECT state FROM durable_jobs WHERE id = ?1",
                    params![job_id.to_string()],
                    |row| Ok((row.get::<_, String>(0)?,)),
                )
                .optional()
                .map_err(|e| JobError::StorageFailure {
                    message: format!("state lookup: {e}"),
                })?;
            let current_state = match row {
                Some((s,)) => JobState::from_str(&s).ok_or_else(|| JobError::StorageFailure {
                    message: format!("unknown state in DB: {s}"),
                })?,
                None => return Err(JobError::NotFound { id: job_id }),
            };
            if current_state.is_terminal() {
                return Err(JobError::TerminalState {
                    id: job_id,
                    state: current_state,
                    attempted: JobState::Cancelled,
                });
            }
            let now_micros = systime_to_micros(SystemTime::now());
            conn.execute(
                "UPDATE durable_jobs
                 SET state = 'Cancelled',
                     completed_at_unix_micros = ?1
                 WHERE id = ?2",
                params![now_micros, job_id.to_string()],
            )
            .map_err(|e| JobError::StorageFailure {
                message: format!("cancel UPDATE: {e}"),
            })?;
            Ok(())
        })
        .await
        .map_err(|e| JobError::StorageFailure {
            message: format!("spawn_blocking join failed: {e}"),
        })?
    }

    async fn get(&self, job_id: JobId) -> Result<JobRecord, JobError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<JobRecord, JobError> {
            let conn = conn.lock();
            let row = conn
                .query_row(
                    "SELECT id, tenant_id, kind, payload, state, priority,
                            created_at_unix_micros, leased_by,
                            leased_at_unix_micros, lease_expires_at_unix_micros,
                            completed_at_unix_micros, error_message
                     FROM durable_jobs WHERE id = ?1",
                    params![job_id.to_string()],
                    row_to_record,
                )
                .optional()
                .map_err(|e| JobError::StorageFailure {
                    message: format!("get query: {e}"),
                })?;
            row.ok_or(JobError::NotFound { id: job_id })
        })
        .await
        .map_err(|e| JobError::StorageFailure {
            message: format!("spawn_blocking join failed: {e}"),
        })?
    }

    async fn list(
        &self,
        tenant_filter: Option<TenantId>,
        state_filter: Option<JobState>,
        limit: usize,
    ) -> Result<Vec<JobRecord>, JobError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<Vec<JobRecord>, JobError> {
            let conn = conn.lock();
            let mut sql = String::from(
                "SELECT id, tenant_id, kind, payload, state, priority,
                        created_at_unix_micros, leased_by,
                        leased_at_unix_micros, lease_expires_at_unix_micros,
                        completed_at_unix_micros, error_message
                 FROM durable_jobs WHERE 1=1",
            );
            let mut params_vec: Vec<rusqlite::types::Value> = Vec::new();
            if let Some(tid) = tenant_filter {
                sql.push_str(" AND tenant_id = ?");
                params_vec.push(rusqlite::types::Value::Integer(tid.0));
            }
            if let Some(state) = state_filter {
                sql.push_str(" AND state = ?");
                params_vec.push(rusqlite::types::Value::Text(state.as_str().into()));
            }
            sql.push_str(" ORDER BY created_at_unix_micros DESC LIMIT ?");
            params_vec.push(rusqlite::types::Value::Integer(limit as i64));

            let mut stmt = conn.prepare(&sql).map_err(|e| JobError::StorageFailure {
                message: format!("prepare list: {e}"),
            })?;
            let param_refs: Vec<&dyn rusqlite::ToSql> =
                params_vec.iter().map(|v| v as &dyn rusqlite::ToSql).collect();
            let rows = stmt
                .query_map(param_refs.as_slice(), row_to_record)
                .map_err(|e| JobError::StorageFailure {
                    message: format!("list query: {e}"),
                })?;
            let mut out = Vec::with_capacity(limit);
            for r in rows {
                out.push(r.map_err(|e| JobError::StorageFailure {
                    message: format!("list row: {e}"),
                })?);
            }
            Ok(out)
        })
        .await
        .map_err(|e| JobError::StorageFailure {
            message: format!("spawn_blocking join failed: {e}"),
        })?
    }

    async fn expire_stale_leases(&self) -> Result<usize, JobError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<usize, JobError> {
            let conn = conn.lock();
            let now_micros = systime_to_micros(SystemTime::now());
            let n = conn
                .execute(
                    "UPDATE durable_jobs
                     SET state = 'Pending',
                         leased_by = NULL,
                         leased_at_unix_micros = NULL,
                         lease_expires_at_unix_micros = NULL
                     WHERE state = 'Leased'
                       AND lease_expires_at_unix_micros <= ?1",
                    params![now_micros],
                )
                .map_err(|e| JobError::StorageFailure {
                    message: format!("expire UPDATE: {e}"),
                })?;
            Ok(n)
        })
        .await
        .map_err(|e| JobError::StorageFailure {
            message: format!("spawn_blocking join failed: {e}"),
        })?
    }
}

fn row_to_record(row: &rusqlite::Row) -> rusqlite::Result<JobRecord> {
    let id_str: String = row.get(0)?;
    let tenant_id: i64 = row.get(1)?;
    let kind: String = row.get(2)?;
    let payload_bytes: Vec<u8> = row.get(3)?;
    let state_str: String = row.get(4)?;
    let priority: i64 = row.get(5)?;
    let created_at_micros: i64 = row.get(6)?;
    let leased_by: Option<String> = row.get(7)?;
    let leased_at_micros: Option<i64> = row.get(8)?;
    let lease_expires_at_micros: Option<i64> = row.get(9)?;
    let completed_at_micros: Option<i64> = row.get(10)?;
    let error_message: Option<String> = row.get(11)?;

    let id = id_str
        .parse::<uuid7::Uuid>()
        .map(JobId)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e)))?;
    let payload: serde_json::Value = serde_json::from_slice(&payload_bytes).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Blob, Box::new(e))
    })?;
    let state = JobState::from_str(&state_str).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            4,
            rusqlite::types::Type::Text,
            format!("unknown state: {state_str}").into(),
        )
    })?;

    Ok(JobRecord {
        id,
        tenant_id: TenantId::new(tenant_id),
        kind,
        payload,
        state,
        priority: priority.clamp(0, 255) as u8,
        created_at: micros_to_systime(created_at_micros),
        leased_by: leased_by.map(WorkerId::new),
        leased_at: leased_at_micros.map(micros_to_systime),
        lease_expires_at: lease_expires_at_micros.map(micros_to_systime),
        completed_at: completed_at_micros.map(micros_to_systime),
        error_message,
    })
}

fn systime_to_micros(t: SystemTime) -> i64 {
    t.duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

fn micros_to_systime(micros: i64) -> SystemTime {
    if micros < 0 {
        std::time::UNIX_EPOCH
    } else {
        std::time::UNIX_EPOCH + Duration::from_micros(micros as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_job() -> NewJob {
        NewJob {
            tenant_id: TenantId::new(1),
            kind: "rfc011.hnsw_delete".into(),
            payload: serde_json::json!({"rid": "mem_x"}),
            priority: 5,
        }
    }

    #[tokio::test]
    async fn enqueue_assigns_unique_ids() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        let id1 = q.enqueue(sample_job()).await.unwrap();
        let id2 = q.enqueue(sample_job()).await.unwrap();
        assert_ne!(id1, id2);
    }

    #[tokio::test]
    async fn acquire_lease_returns_none_for_empty_queue() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        let lease = q
            .acquire_lease(WorkerId::new("w1"), None, None, 60)
            .await
            .unwrap();
        assert!(lease.is_none());
    }

    #[tokio::test]
    async fn acquire_lease_returns_pending_job() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        let id = q.enqueue(sample_job()).await.unwrap();
        let lease = q
            .acquire_lease(WorkerId::new("w1"), None, None, 60)
            .await
            .unwrap()
            .expect("expected lease");
        assert_eq!(lease.job_id, id);
        assert_eq!(lease.worker_id, WorkerId::new("w1"));
        assert_eq!(lease.job.state, JobState::Leased);
        assert!(lease.expires_at > SystemTime::now());
    }

    #[tokio::test]
    async fn second_lease_picks_different_job() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        q.enqueue(sample_job()).await.unwrap();
        q.enqueue(sample_job()).await.unwrap();
        let l1 = q
            .acquire_lease(WorkerId::new("w1"), None, None, 60)
            .await
            .unwrap()
            .unwrap();
        let l2 = q
            .acquire_lease(WorkerId::new("w2"), None, None, 60)
            .await
            .unwrap()
            .unwrap();
        assert_ne!(l1.job_id, l2.job_id);
    }

    #[tokio::test]
    async fn third_lease_returns_none_when_all_leased() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        q.enqueue(sample_job()).await.unwrap();
        q.enqueue(sample_job()).await.unwrap();
        q.acquire_lease(WorkerId::new("w1"), None, None, 60)
            .await
            .unwrap();
        q.acquire_lease(WorkerId::new("w2"), None, None, 60)
            .await
            .unwrap();
        let l3 = q
            .acquire_lease(WorkerId::new("w3"), None, None, 60)
            .await
            .unwrap();
        assert!(l3.is_none());
    }

    #[tokio::test]
    async fn priority_orders_pickup() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        let _low = q
            .enqueue(NewJob {
                priority: 1,
                ..sample_job()
            })
            .await
            .unwrap();
        let high = q
            .enqueue(NewJob {
                priority: 9,
                ..sample_job()
            })
            .await
            .unwrap();
        let lease = q
            .acquire_lease(WorkerId::new("w1"), None, None, 60)
            .await
            .unwrap()
            .unwrap();
        // High-priority job should be picked first.
        assert_eq!(lease.job_id, high);
    }

    #[tokio::test]
    async fn kinds_filter_excludes_other_kinds() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        let snapshot_job = NewJob {
            kind: "rfc012.snapshot_create".into(),
            ..sample_job()
        };
        q.enqueue(sample_job()).await.unwrap(); // hnsw_delete
        q.enqueue(snapshot_job).await.unwrap();

        let lease = q
            .acquire_lease(
                WorkerId::new("snapshot-worker"),
                Some(vec!["rfc012.snapshot_create".into()]),
                None,
                60,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(lease.job.kind, "rfc012.snapshot_create");
    }

    #[tokio::test]
    async fn empty_kinds_filter_returns_none() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        q.enqueue(sample_job()).await.unwrap();
        let lease = q
            .acquire_lease(WorkerId::new("w"), Some(vec![]), None, 60)
            .await
            .unwrap();
        assert!(lease.is_none());
    }

    #[tokio::test]
    async fn complete_marks_succeeded() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        q.enqueue(sample_job()).await.unwrap();
        let lease = q
            .acquire_lease(WorkerId::new("w1"), None, None, 60)
            .await
            .unwrap()
            .unwrap();
        q.complete(lease.lease_id, lease.job_id, Outcome::Succeeded)
            .await
            .unwrap();
        let record = q.get(lease.job_id).await.unwrap();
        assert_eq!(record.state, JobState::Succeeded);
        assert!(record.completed_at.is_some());
    }

    #[tokio::test]
    async fn complete_marks_failed_with_message() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        q.enqueue(sample_job()).await.unwrap();
        let lease = q
            .acquire_lease(WorkerId::new("w1"), None, None, 60)
            .await
            .unwrap()
            .unwrap();
        q.complete(
            lease.lease_id,
            lease.job_id,
            Outcome::Failed {
                error_message: "rid not found".into(),
            },
        )
        .await
        .unwrap();
        let record = q.get(lease.job_id).await.unwrap();
        assert_eq!(record.state, JobState::Failed);
        assert_eq!(record.error_message.as_deref(), Some("rid not found"));
    }

    #[tokio::test]
    async fn cancel_pending_job() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        let id = q.enqueue(sample_job()).await.unwrap();
        q.cancel(id).await.unwrap();
        let record = q.get(id).await.unwrap();
        assert_eq!(record.state, JobState::Cancelled);
    }

    #[tokio::test]
    async fn cancel_terminal_state_fails() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        q.enqueue(sample_job()).await.unwrap();
        let lease = q
            .acquire_lease(WorkerId::new("w1"), None, None, 60)
            .await
            .unwrap()
            .unwrap();
        q.complete(lease.lease_id, lease.job_id, Outcome::Succeeded)
            .await
            .unwrap();
        let err = q.cancel(lease.job_id).await.unwrap_err();
        assert!(matches!(err, JobError::TerminalState { .. }));
    }

    #[tokio::test]
    async fn cancel_unknown_job_fails() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        let err = q.cancel(JobId::new_random()).await.unwrap_err();
        assert!(matches!(err, JobError::NotFound { .. }));
    }

    #[tokio::test]
    async fn expire_stale_leases_reclaims_to_pending() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        q.enqueue(sample_job()).await.unwrap();
        // Acquire with a 0-second TTL — immediately stale.
        let _ = q
            .acquire_lease(WorkerId::new("w1"), None, None, 0)
            .await
            .unwrap()
            .unwrap();
        // Need to wait at least 1us for `now > expires_at` check.
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let n = q.expire_stale_leases().await.unwrap();
        assert_eq!(n, 1);
        // Job should be Pending again.
        let pending = q
            .list(None, Some(JobState::Pending), 100)
            .await
            .unwrap();
        assert_eq!(pending.len(), 1);
    }

    #[tokio::test]
    async fn list_filters_by_tenant_and_state() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        q.enqueue(NewJob {
            tenant_id: TenantId::new(1),
            ..sample_job()
        })
        .await
        .unwrap();
        q.enqueue(NewJob {
            tenant_id: TenantId::new(2),
            ..sample_job()
        })
        .await
        .unwrap();

        let t1_only = q
            .list(Some(TenantId::new(1)), None, 100)
            .await
            .unwrap();
        assert_eq!(t1_only.len(), 1);
        assert_eq!(t1_only[0].tenant_id, TenantId::new(1));

        let pending_only = q
            .list(None, Some(JobState::Pending), 100)
            .await
            .unwrap();
        assert_eq!(pending_only.len(), 2);
    }

    #[tokio::test]
    async fn heartbeat_extends_lease() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        q.enqueue(sample_job()).await.unwrap();
        let lease = q
            .acquire_lease(WorkerId::new("w1"), None, None, 1)
            .await
            .unwrap()
            .unwrap();
        let original_expires = lease.expires_at;
        // Heartbeat with a longer TTL.
        q.heartbeat(lease.lease_id, lease.job_id, 60).await.unwrap();
        let record = q.get(lease.job_id).await.unwrap();
        assert!(record.lease_expires_at.unwrap() > original_expires);
    }

    #[tokio::test]
    async fn heartbeat_after_expiry_fails() {
        let q = LocalSqliteJobQueue::open_in_memory().unwrap();
        q.enqueue(sample_job()).await.unwrap();
        let lease = q
            .acquire_lease(WorkerId::new("w1"), None, None, 0)
            .await
            .unwrap()
            .unwrap();
        // Wait for the lease to expire + reclaim it.
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        q.expire_stale_leases().await.unwrap();
        let err = q.heartbeat(lease.lease_id, lease.job_id, 60).await.unwrap_err();
        assert!(matches!(err, JobError::InvalidLease { .. }));
    }

    #[tokio::test]
    async fn dyn_dispatch_works() {
        let q: Arc<dyn JobQueue> = Arc::new(LocalSqliteJobQueue::open_in_memory().unwrap());
        let id = q.enqueue(sample_job()).await.unwrap();
        let lease = q
            .acquire_lease(WorkerId::new("w1"), None, None, 60)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(lease.job_id, id);
    }
}
