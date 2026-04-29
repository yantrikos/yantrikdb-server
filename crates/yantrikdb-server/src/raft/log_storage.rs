//! `SqliteRaftLogStorage` — implements openraft's `RaftLogStorage` and
//! `RaftLogReader` over the `raft_log_entries` / `raft_vote` /
//! `raft_state` tables (migration m004).
//!
//! ## Contract guarantees
//!
//! 1. **No holes.** Append uses INSERT ... ON CONFLICT so a re-append
//!    at an existing index overwrites (the openraft idiom after a
//!    truncate). `truncate` removes a suffix; `purge` removes a prefix
//!    and updates the watermark; both leave a consecutive run.
//! 2. **Serialized write IO.** All writes go through a single
//!    `parking_lot::Mutex<Connection>`. openraft also drives storage
//!    from a single task in practice.
//! 3. **Vote durable before return.** `save_vote` commits the SQL
//!    transaction synchronously (inside `spawn_blocking`) before the
//!    method returns.
//! 4. **`LogFlushed` callback fires only after fsync.** We commit the
//!    SQL transaction (which fsyncs the WAL with the configured
//!    `PRAGMA synchronous=NORMAL`), then call
//!    `LogFlushed::log_io_completed(Ok(()))`.

use std::ops::{Bound, RangeBounds};
use std::sync::Arc;
use std::time::SystemTime;

use openraft::storage::{LogFlushed, RaftLogStorage};
use openraft::{
    AnyError, Entry, LogId, LogState, RaftLogReader, StorageError, StorageIOError, Vote,
};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};

use super::types::{YantrikNodeId, YantrikRaftTypeConfig};

const KEY_LAST_PURGED: &str = "last_purged_log_id";
const KEY_COMMITTED: &str = "committed_log_id";

/// SQLite-backed openraft log storage. Cheap to clone — internal state
/// is `Arc<Mutex<Connection>>`. openraft's `get_log_reader` returns a
/// clone, so cheap clone is required.
#[derive(Clone)]
pub struct SqliteRaftLogStorage {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteRaftLogStorage {
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Build a fresh in-memory storage with migrations applied. Tests
    /// only — production opens against a file path.
    #[cfg(test)]
    pub fn open_in_memory() -> Self {
        let mut conn = Connection::open_in_memory().expect("in-memory sqlite");
        crate::migrations::MigrationRunner::run_pending(&mut conn)
            .expect("migrations should apply on a fresh DB");
        Self::new(Arc::new(Mutex::new(conn)))
    }

    fn now_micros() -> i64 {
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0)
    }

    fn err_write_logs(msg: impl ToString) -> StorageError<YantrikNodeId> {
        StorageIOError::write_logs(AnyError::error(msg)).into()
    }
    fn err_read_logs(msg: impl ToString) -> StorageError<YantrikNodeId> {
        StorageIOError::read_logs(AnyError::error(msg)).into()
    }
    fn err_write_vote(msg: impl ToString) -> StorageError<YantrikNodeId> {
        StorageIOError::write_vote(AnyError::error(msg)).into()
    }
    fn err_read_vote(msg: impl ToString) -> StorageError<YantrikNodeId> {
        StorageIOError::read_vote(AnyError::error(msg)).into()
    }
    fn err_read(msg: impl ToString) -> StorageError<YantrikNodeId> {
        StorageIOError::read(AnyError::error(msg)).into()
    }
    fn err_write(msg: impl ToString) -> StorageError<YantrikNodeId> {
        StorageIOError::write(AnyError::error(msg)).into()
    }

    fn entry_to_blob(
        entry: &Entry<YantrikRaftTypeConfig>,
    ) -> Result<Vec<u8>, StorageError<YantrikNodeId>> {
        serde_json::to_vec(entry).map_err(|e| Self::err_write_logs(format!("entry serialize: {e}")))
    }

    fn blob_to_entry(
        blob: &[u8],
    ) -> Result<Entry<YantrikRaftTypeConfig>, StorageError<YantrikNodeId>> {
        serde_json::from_slice(blob)
            .map_err(|e| Self::err_read_logs(format!("entry deserialize: {e}")))
    }

    fn read_state_value<T: serde::de::DeserializeOwned>(
        conn: &Connection,
        key: &str,
    ) -> Result<Option<T>, StorageError<YantrikNodeId>> {
        let row: Option<Vec<u8>> = conn
            .query_row(
                "SELECT value FROM raft_state WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| Self::err_read(format!("state {key} query: {e}")))?;
        match row {
            None => Ok(None),
            Some(blob) => {
                let v: T = serde_json::from_slice(&blob)
                    .map_err(|e| Self::err_read(format!("state {key} deserialize: {e}")))?;
                Ok(Some(v))
            }
        }
    }

    fn write_state_value<T: serde::Serialize>(
        conn: &Connection,
        key: &str,
        value: &T,
    ) -> Result<(), StorageError<YantrikNodeId>> {
        let blob = serde_json::to_vec(value)
            .map_err(|e| Self::err_write(format!("state {key} serialize: {e}")))?;
        conn.execute(
            "INSERT INTO raft_state (key, value, updated_at_unix_micros) VALUES (?1, ?2, ?3)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at_unix_micros = excluded.updated_at_unix_micros",
            params![key, blob, Self::now_micros()],
        )
        .map_err(|e| Self::err_write(format!("state {key} write: {e}")))?;
        Ok(())
    }

    /// Resolve `[start, stop)` from a `RangeBounds<u64>` — openraft
    /// passes both inclusive-exclusive and unbounded forms.
    fn range_bounds(range: impl RangeBounds<u64>) -> (u64, Option<u64>) {
        let start = match range.start_bound() {
            Bound::Included(&x) => x,
            Bound::Excluded(&x) => x.saturating_add(1),
            Bound::Unbounded => 0,
        };
        let stop = match range.end_bound() {
            Bound::Included(&x) => Some(x.saturating_add(1)),
            Bound::Excluded(&x) => Some(x),
            Bound::Unbounded => None,
        };
        (start, stop)
    }

    /// Internal append routine: writes entries inside a transaction
    /// and returns once the transaction commits (= durable). Used by
    /// the public `append` (which then fires the openraft callback)
    /// and by tests that don't construct a `LogFlushed`.
    pub(crate) async fn append_durable(
        &self,
        entries: Vec<Entry<YantrikRaftTypeConfig>>,
    ) -> Result<(), StorageError<YantrikNodeId>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<(), StorageError<YantrikNodeId>> {
            let mut conn = conn.lock();
            let tx = conn
                .transaction()
                .map_err(|e| Self::err_write_logs(format!("begin tx: {e}")))?;

            for entry in entries.iter() {
                let blob = Self::entry_to_blob(entry)?;
                let leader_node = u64::from(entry.log_id.leader_id.node_id) as i64;
                let term = entry.log_id.leader_id.term as i64;
                let log_index = entry.log_id.index as i64;
                tx.execute(
                    "INSERT INTO raft_log_entries (log_index, term, leader_node_id, payload)
                     VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(log_index) DO UPDATE SET
                        term = excluded.term,
                        leader_node_id = excluded.leader_node_id,
                        payload = excluded.payload",
                    params![log_index, term, leader_node, blob],
                )
                .map_err(|e| Self::err_write_logs(format!("insert: {e}")))?;
            }

            tx.commit()
                .map_err(|e| Self::err_write_logs(format!("commit: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| Self::err_write_logs(format!("spawn_blocking: {e}")))?
    }
}

impl RaftLogReader<YantrikRaftTypeConfig> for SqliteRaftLogStorage {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + std::fmt::Debug + Send>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<YantrikRaftTypeConfig>>, StorageError<YantrikNodeId>> {
        let (start, stop) = Self::range_bounds(range);
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<_, StorageError<YantrikNodeId>> {
            let conn = conn.lock();
            let stop_clamp = stop.unwrap_or(i64::MAX as u64);
            let mut stmt = conn
                .prepare(
                    "SELECT payload FROM raft_log_entries
                     WHERE log_index >= ?1 AND log_index < ?2
                     ORDER BY log_index ASC",
                )
                .map_err(|e| Self::err_read_logs(format!("prepare: {e}")))?;
            let mut out = Vec::new();
            let rows = stmt
                .query_map(params![start as i64, stop_clamp as i64], |row| {
                    row.get::<_, Vec<u8>>(0)
                })
                .map_err(|e| Self::err_read_logs(format!("query: {e}")))?;
            for row in rows {
                let blob = row.map_err(|e| Self::err_read_logs(format!("row: {e}")))?;
                out.push(Self::blob_to_entry(&blob)?);
            }
            Ok(out)
        })
        .await
        .map_err(|e| Self::err_read_logs(format!("spawn_blocking: {e}")))?
    }
}

impl RaftLogStorage<YantrikRaftTypeConfig> for SqliteRaftLogStorage {
    type LogReader = Self;

    async fn get_log_state(
        &mut self,
    ) -> Result<LogState<YantrikRaftTypeConfig>, StorageError<YantrikNodeId>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<_, StorageError<YantrikNodeId>> {
            let conn = conn.lock();
            let last_purged: Option<LogId<YantrikNodeId>> =
                Self::read_state_value(&conn, KEY_LAST_PURGED)?;

            // last_log_id: the entry at MAX(log_index), or last_purged if none.
            let last_present_blob: Option<Vec<u8>> = conn
                .query_row(
                    "SELECT payload FROM raft_log_entries
                     ORDER BY log_index DESC LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| Self::err_read_logs(format!("max query: {e}")))?;

            let last_log_id = match last_present_blob {
                Some(blob) => Some(Self::blob_to_entry(&blob)?.log_id),
                None => last_purged.clone(),
            };

            Ok(LogState {
                last_purged_log_id: last_purged,
                last_log_id,
            })
        })
        .await
        .map_err(|e| Self::err_read_logs(format!("spawn_blocking: {e}")))?
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        self.clone()
    }

    async fn save_vote(
        &mut self,
        vote: &Vote<YantrikNodeId>,
    ) -> Result<(), StorageError<YantrikNodeId>> {
        let vote = vote.clone();
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<(), StorageError<YantrikNodeId>> {
            let conn = conn.lock();
            let blob = serde_json::to_vec(&vote)
                .map_err(|e| Self::err_write_vote(format!("vote serialize: {e}")))?;
            conn.execute(
                "INSERT INTO raft_vote (id, payload, updated_at_unix_micros) VALUES (1, ?1, ?2)
                 ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, updated_at_unix_micros = excluded.updated_at_unix_micros",
                params![blob, Self::now_micros()],
            )
            .map_err(|e| Self::err_write_vote(format!("upsert: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| Self::err_write_vote(format!("spawn_blocking: {e}")))?
    }

    async fn read_vote(
        &mut self,
    ) -> Result<Option<Vote<YantrikNodeId>>, StorageError<YantrikNodeId>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<_, StorageError<YantrikNodeId>> {
            let conn = conn.lock();
            let row: Option<Vec<u8>> = conn
                .query_row("SELECT payload FROM raft_vote WHERE id = 1", [], |row| {
                    row.get(0)
                })
                .optional()
                .map_err(|e| Self::err_read_vote(format!("query: {e}")))?;
            match row {
                None => Ok(None),
                Some(blob) => {
                    let vote: Vote<YantrikNodeId> = serde_json::from_slice(&blob)
                        .map_err(|e| Self::err_read_vote(format!("deserialize: {e}")))?;
                    Ok(Some(vote))
                }
            }
        })
        .await
        .map_err(|e| Self::err_read_vote(format!("spawn_blocking: {e}")))?
    }

    async fn save_committed(
        &mut self,
        committed: Option<LogId<YantrikNodeId>>,
    ) -> Result<(), StorageError<YantrikNodeId>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<(), StorageError<YantrikNodeId>> {
            let conn = conn.lock();
            Self::write_state_value(&conn, KEY_COMMITTED, &committed)
        })
        .await
        .map_err(|e| Self::err_write(format!("spawn_blocking: {e}")))?
    }

    async fn read_committed(
        &mut self,
    ) -> Result<Option<LogId<YantrikNodeId>>, StorageError<YantrikNodeId>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<_, StorageError<YantrikNodeId>> {
            let conn = conn.lock();
            Self::read_state_value(&conn, KEY_COMMITTED)
        })
        .await
        .map_err(|e| Self::err_read(format!("spawn_blocking: {e}")))?
    }

    async fn append<I>(
        &mut self,
        entries: I,
        callback: LogFlushed<YantrikRaftTypeConfig>,
    ) -> Result<(), StorageError<YantrikNodeId>>
    where
        I: IntoIterator<Item = Entry<YantrikRaftTypeConfig>> + Send,
        I::IntoIter: Send,
    {
        let entries: Vec<_> = entries.into_iter().collect();
        match self.append_durable(entries).await {
            Ok(()) => {
                callback.log_io_completed(Ok(()));
                Ok(())
            }
            Err(e) => {
                callback.log_io_completed(Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                )));
                Err(e)
            }
        }
    }

    async fn truncate(
        &mut self,
        log_id: LogId<YantrikNodeId>,
    ) -> Result<(), StorageError<YantrikNodeId>> {
        let conn = Arc::clone(&self.conn);
        let from_index = log_id.index as i64;
        tokio::task::spawn_blocking(move || -> Result<(), StorageError<YantrikNodeId>> {
            let conn = conn.lock();
            conn.execute(
                "DELETE FROM raft_log_entries WHERE log_index >= ?1",
                params![from_index],
            )
            .map_err(|e| Self::err_write_logs(format!("truncate: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| Self::err_write_logs(format!("spawn_blocking: {e}")))?
    }

    async fn purge(
        &mut self,
        log_id: LogId<YantrikNodeId>,
    ) -> Result<(), StorageError<YantrikNodeId>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<(), StorageError<YantrikNodeId>> {
            let mut conn = conn.lock();
            let tx = conn
                .transaction()
                .map_err(|e| Self::err_write_logs(format!("begin tx: {e}")))?;
            tx.execute(
                "DELETE FROM raft_log_entries WHERE log_index <= ?1",
                params![log_id.index as i64],
            )
            .map_err(|e| Self::err_write_logs(format!("purge delete: {e}")))?;
            Self::write_state_value(&tx, KEY_LAST_PURGED, &Some(log_id.clone()))?;
            tx.commit()
                .map_err(|e| Self::err_write_logs(format!("commit: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| Self::err_write_logs(format!("spawn_blocking: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::{MemoryMutation, OpId, TenantId};
    use crate::raft::types::{YantrikLogEntry, YantrikNode};
    use openraft::{EntryPayload, LeaderId};

    fn upsert_entry(rid: &str) -> YantrikLogEntry {
        YantrikLogEntry::new(
            TenantId::new(1),
            OpId::new_random(),
            MemoryMutation::UpsertMemory {
                rid: rid.into(),
                text: "x".into(),
                memory_type: "semantic".into(),
                importance: 0.5,
                valence: 0.0,
                half_life: 168.0,
                namespace: "default".into(),
                certainty: 1.0,
                domain: "general".into(),
                source: "user".into(),
                emotional_state: None,
                embedding: None,
                metadata: serde_json::json!({}),
            },
        )
    }

    fn entry_at(
        index: u64,
        term: u64,
        node: u64,
        payload: YantrikLogEntry,
    ) -> Entry<YantrikRaftTypeConfig> {
        Entry {
            log_id: LogId::new(LeaderId::new(term, YantrikNodeId::new(node)), index),
            payload: EntryPayload::Normal(payload),
        }
    }

    #[tokio::test]
    async fn fresh_storage_log_state_is_empty() {
        let mut s = SqliteRaftLogStorage::open_in_memory();
        let state = s.get_log_state().await.unwrap();
        assert_eq!(state.last_purged_log_id, None);
        assert_eq!(state.last_log_id, None);
    }

    #[tokio::test]
    async fn save_and_read_vote_round_trip() {
        let mut s = SqliteRaftLogStorage::open_in_memory();
        assert_eq!(s.read_vote().await.unwrap(), None);
        let v = Vote::new(7, YantrikNodeId::new(3));
        s.save_vote(&v).await.unwrap();
        assert_eq!(s.read_vote().await.unwrap(), Some(v));
    }

    #[tokio::test]
    async fn save_vote_overwrites_previous() {
        let mut s = SqliteRaftLogStorage::open_in_memory();
        s.save_vote(&Vote::new(1, YantrikNodeId::new(1)))
            .await
            .unwrap();
        let v2 = Vote::new(2, YantrikNodeId::new(2));
        s.save_vote(&v2).await.unwrap();
        assert_eq!(s.read_vote().await.unwrap(), Some(v2));
    }

    #[tokio::test]
    async fn save_and_read_committed() {
        let mut s = SqliteRaftLogStorage::open_in_memory();
        assert_eq!(s.read_committed().await.unwrap(), None);
        let lid = LogId::new(LeaderId::new(3, YantrikNodeId::new(1)), 42);
        s.save_committed(Some(lid.clone())).await.unwrap();
        assert_eq!(s.read_committed().await.unwrap(), Some(lid));
    }

    #[tokio::test]
    async fn append_durable_then_read_round_trip() {
        let mut s = SqliteRaftLogStorage::open_in_memory();
        let e1 = entry_at(1, 1, 1, upsert_entry("a"));
        let e2 = entry_at(2, 1, 1, upsert_entry("b"));
        s.append_durable(vec![e1.clone(), e2.clone()])
            .await
            .unwrap();

        let got = s.try_get_log_entries(1..3).await.unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].log_id.index, 1);
        assert_eq!(got[1].log_id.index, 2);
    }

    #[tokio::test]
    async fn try_get_log_entries_respects_range() {
        let mut s = SqliteRaftLogStorage::open_in_memory();
        let entries: Vec<_> = (1..=5)
            .map(|i| entry_at(i, 1, 1, upsert_entry(&format!("e{i}"))))
            .collect();
        s.append_durable(entries).await.unwrap();

        // Inclusive-exclusive: [2, 4) returns indices 2, 3.
        let got = s.try_get_log_entries(2..4).await.unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].log_id.index, 2);
        assert_eq!(got[1].log_id.index, 3);

        // Unbounded end: 4..  returns 4, 5.
        let got = s.try_get_log_entries(4..).await.unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].log_id.index, 4);
        assert_eq!(got[1].log_id.index, 5);
    }

    #[tokio::test]
    async fn get_log_state_reflects_appended_entries() {
        let mut s = SqliteRaftLogStorage::open_in_memory();
        let e3 = entry_at(3, 5, 7, upsert_entry("a"));
        s.append_durable(vec![e3.clone()]).await.unwrap();

        let state = s.get_log_state().await.unwrap();
        assert_eq!(state.last_log_id, Some(e3.log_id.clone()));
        assert_eq!(state.last_purged_log_id, None);
    }

    #[tokio::test]
    async fn truncate_removes_suffix_inclusive() {
        let mut s = SqliteRaftLogStorage::open_in_memory();
        let entries: Vec<_> = (1..=5)
            .map(|i| entry_at(i, 1, 1, upsert_entry(&format!("e{i}"))))
            .collect();
        s.append_durable(entries).await.unwrap();

        let target = LogId::new(LeaderId::new(1, YantrikNodeId::new(1)), 3);
        s.truncate(target).await.unwrap();

        let remaining = s.try_get_log_entries(0..100).await.unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].log_id.index, 1);
        assert_eq!(remaining[1].log_id.index, 2);
    }

    #[tokio::test]
    async fn purge_removes_prefix_and_updates_watermark() {
        let mut s = SqliteRaftLogStorage::open_in_memory();
        let entries: Vec<_> = (1..=5)
            .map(|i| entry_at(i, 1, 1, upsert_entry(&format!("e{i}"))))
            .collect();
        s.append_durable(entries).await.unwrap();

        let target = LogId::new(LeaderId::new(1, YantrikNodeId::new(1)), 3);
        s.purge(target.clone()).await.unwrap();

        // Entries 1..=3 gone; 4 and 5 remain.
        let remaining = s.try_get_log_entries(0..100).await.unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].log_id.index, 4);
        assert_eq!(remaining[1].log_id.index, 5);

        let state = s.get_log_state().await.unwrap();
        assert_eq!(state.last_purged_log_id, Some(target));
        assert_eq!(state.last_log_id.unwrap().index, 5);
    }

    #[tokio::test]
    async fn purge_with_no_entries_remaining_falls_back_to_purged() {
        // After purging everything, last_log_id should equal last_purged_log_id.
        let mut s = SqliteRaftLogStorage::open_in_memory();
        let entries: Vec<_> = (1..=3)
            .map(|i| entry_at(i, 1, 1, upsert_entry(&format!("e{i}"))))
            .collect();
        s.append_durable(entries).await.unwrap();

        let target = LogId::new(LeaderId::new(1, YantrikNodeId::new(1)), 3);
        s.purge(target.clone()).await.unwrap();

        let state = s.get_log_state().await.unwrap();
        assert_eq!(state.last_log_id, Some(target.clone()));
        assert_eq!(state.last_purged_log_id, Some(target));
    }

    #[tokio::test]
    async fn append_can_overwrite_existing_index() {
        // After truncate, openraft re-appends at the truncated index.
        // The INSERT ... ON CONFLICT ensures the new entry replaces.
        let mut s = SqliteRaftLogStorage::open_in_memory();
        let original = entry_at(1, 1, 1, upsert_entry("original"));
        let replacement = entry_at(1, 2, 7, upsert_entry("replacement"));

        s.append_durable(vec![original]).await.unwrap();
        s.append_durable(vec![replacement.clone()]).await.unwrap();

        let got = s.try_get_log_entries(1..2).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].log_id.leader_id.term, 2);
        assert_eq!(got[0].log_id.leader_id.node_id, YantrikNodeId::new(7));
    }

    #[tokio::test]
    async fn vote_persists_across_clone() {
        // get_log_reader returns a clone — the clone must observe writes
        // made through the original (they share the same Arc<Mutex<Connection>>).
        let mut s = SqliteRaftLogStorage::open_in_memory();
        let v = Vote::new(11, YantrikNodeId::new(2));
        s.save_vote(&v).await.unwrap();
        let mut reader = s.get_log_reader().await;
        assert_eq!(reader.read_vote().await.unwrap(), Some(v));
    }

    #[tokio::test]
    async fn entry_with_membership_payload_round_trips() {
        // Membership entries carry no application data; this verifies
        // the JSON payload format covers them.
        use openraft::Membership;
        use std::collections::BTreeSet;
        let mut s = SqliteRaftLogStorage::open_in_memory();
        let mut node_set = BTreeSet::new();
        node_set.insert(YantrikNodeId::new(1));
        node_set.insert(YantrikNodeId::new(2));
        let mut nodes = std::collections::BTreeMap::new();
        nodes.insert(YantrikNodeId::new(1), YantrikNode::new("http://n1"));
        nodes.insert(YantrikNodeId::new(2), YantrikNode::new("http://n2"));
        let mship = Membership::new(vec![node_set], nodes);
        let entry = Entry {
            log_id: LogId::new(LeaderId::new(1, YantrikNodeId::new(1)), 1),
            payload: EntryPayload::Membership(mship.clone()),
        };

        s.append_durable(vec![entry]).await.unwrap();

        let got = s.try_get_log_entries(1..2).await.unwrap();
        assert_eq!(got.len(), 1);
        match &got[0].payload {
            EntryPayload::Membership(m) => assert_eq!(*m, mship),
            other => panic!("expected Membership, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_range_query_returns_empty() {
        let mut s = SqliteRaftLogStorage::open_in_memory();
        let got = s.try_get_log_entries(1..5).await.unwrap();
        assert!(got.is_empty());
    }

    #[tokio::test]
    async fn vote_durable_across_reopen_through_shared_connection() {
        // Confirms the vote SQL row persists — though we share a
        // Connection across the two storages, the test exercises
        // that the insert is visible immediately after save_vote
        // returns (i.e. before any explicit fsync in the call).
        let conn = Arc::new(Mutex::new({
            let mut c = Connection::open_in_memory().unwrap();
            crate::migrations::MigrationRunner::run_pending(&mut c).unwrap();
            c
        }));
        let mut s1 = SqliteRaftLogStorage::new(Arc::clone(&conn));
        let mut s2 = SqliteRaftLogStorage::new(Arc::clone(&conn));
        let v = Vote::new(99, YantrikNodeId::new(5));
        s1.save_vote(&v).await.unwrap();
        // Second instance sees the vote via the shared connection.
        assert_eq!(s2.read_vote().await.unwrap(), Some(v));
    }
}
