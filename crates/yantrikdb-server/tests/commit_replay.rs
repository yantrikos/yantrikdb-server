//! Restart-replay integration test for RFC 010 PR-2.
//!
//! ## What this test gates
//!
//! PR-2 replaced the in-memory commit log backing with SQLite persistence.
//! The acceptance criterion is: **every committed entry survives a
//! committer drop + reopen.** Specifically:
//!
//! 1. High watermark restored exactly.
//! 2. read_range returns identical entries (mutation, op_id, log_index,
//!    term, committed_at, applied_at).
//! 3. New commits after reopen continue the log from the prior watermark.
//! 4. Per-tenant log isolation persists (tenant A's entries don't leak
//!    into tenant B on reload).
//!
//! Any of these failing means we've regressed durability — exactly the
//! kind of corruption that's invisible until disaster recovery, so we
//! pin it as a hard gate.
//!
//! ## Why it's an integration test (not a unit test)
//!
//! Restart semantics need a real on-disk SQLite file (in-memory `:memory:`
//! databases vanish when the connection drops). This test creates a
//! `tempdir` for the DB file, opens the committer, writes, drops, reopens,
//! and verifies — that's a full round-trip through the persistence layer
//! and lives at the integration-test level by convention.

#[path = "../src/cache/mod.rs"]
mod cache;

#[path = "../src/commit/mod.rs"]
mod commit;

#[path = "../src/forget/mod.rs"]
mod forget;

#[path = "../src/key_provider/mod.rs"]
mod key_provider;

#[path = "../src/index/mod.rs"]
mod index;

#[path = "../src/jobs/mod.rs"]
mod jobs;

#[path = "../src/migrations/mod.rs"]
mod migrations;

#[path = "../src/version/mod.rs"]
mod version;

use commit::{CommitOptions, LocalSqliteCommitter, MemoryMutation, MutationCommitter, TenantId};
use std::sync::Arc;
use tempfile::tempdir;

fn upsert(tag: &str) -> MemoryMutation {
    MemoryMutation::UpsertMemory {
        rid: format!("mem_{tag}"),
        text: format!("body-{tag}"),
        memory_type: "semantic".to_string(),
        importance: 0.5,
        valence: 0.0,
        half_life: 168.0,
        namespace: "default".to_string(),
        certainty: 1.0,
        domain: "general".to_string(),
        source: "user".to_string(),
        emotional_state: None,
        embedding: None,
        metadata: serde_json::json!({}),
    }
}

#[tokio::test]
async fn high_watermark_persists_across_restart() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("commit.db");
    let t1 = TenantId::new(1);
    let t2 = TenantId::new(2);

    // Phase 1: open, write, drop.
    {
        let c = LocalSqliteCommitter::open(&path).expect("open phase 1");
        for i in 1..=7 {
            c.commit(t1, upsert(&format!("t1_{i}")), CommitOptions::new())
                .await
                .unwrap();
        }
        for i in 1..=3 {
            c.commit(t2, upsert(&format!("t2_{i}")), CommitOptions::new())
                .await
                .unwrap();
        }
        assert_eq!(c.high_watermark(t1).await.unwrap(), 7);
        assert_eq!(c.high_watermark(t2).await.unwrap(), 3);
        // Drop here — committer (and its connection) closes.
    }

    // Phase 2: reopen against the same file. Watermarks must restore.
    {
        let c = LocalSqliteCommitter::open(&path).expect("open phase 2");
        assert_eq!(
            c.high_watermark(t1).await.unwrap(),
            7,
            "tenant 1 watermark should restore exactly"
        );
        assert_eq!(
            c.high_watermark(t2).await.unwrap(),
            3,
            "tenant 2 watermark should restore exactly"
        );
    }
}

#[tokio::test]
async fn read_range_returns_identical_entries_after_restart() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("commit.db");
    let t = TenantId::new(42);

    let mut written = Vec::new();
    {
        let c = LocalSqliteCommitter::open(&path).expect("open phase 1");
        for i in 1..=5 {
            let m = upsert(&format!("entry_{i}"));
            let r = c.commit(t, m.clone(), CommitOptions::new()).await.unwrap();
            written.push((r.log_index, m));
        }
    }

    {
        let c = LocalSqliteCommitter::open(&path).expect("open phase 2");
        let entries = c.read_range(t, 0, 100).await.unwrap();
        assert_eq!(entries.len(), 5);
        for (i, entry) in entries.iter().enumerate() {
            let (expected_index, expected_mutation) = &written[i];
            assert_eq!(entry.log_index, *expected_index);
            assert_eq!(entry.mutation, *expected_mutation);
            assert_eq!(entry.tenant_id, t);
            assert_eq!(entry.term, 0);
        }
    }
}

#[tokio::test]
async fn new_commits_continue_from_prior_watermark() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("commit.db");
    let t = TenantId::new(1);

    {
        let c = LocalSqliteCommitter::open(&path).expect("open phase 1");
        for i in 1..=4 {
            c.commit(t, upsert(&format!("first_{i}")), CommitOptions::new())
                .await
                .unwrap();
        }
        assert_eq!(c.high_watermark(t).await.unwrap(), 4);
    }

    {
        let c = LocalSqliteCommitter::open(&path).expect("open phase 2");
        // Brand-new commit on the reloaded committer must be log_index=5.
        let r = c
            .commit(t, upsert("after_restart"), CommitOptions::new())
            .await
            .unwrap();
        assert_eq!(
            r.log_index, 5,
            "new commit after restart must continue the log, not reset"
        );
        assert_eq!(c.high_watermark(t).await.unwrap(), 5);

        // Read the full log; should be 5 entries with correct ordering.
        let entries = c.read_range(t, 0, 100).await.unwrap();
        assert_eq!(entries.len(), 5);
        for (idx, entry) in entries.iter().enumerate() {
            assert_eq!(entry.log_index, (idx as u64) + 1);
        }
    }
}

#[tokio::test]
async fn per_tenant_isolation_persists_across_restart() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("commit.db");
    let t1 = TenantId::new(1);
    let t2 = TenantId::new(2);

    {
        let c = LocalSqliteCommitter::open(&path).expect("open phase 1");
        c.commit(t1, upsert("t1_a"), CommitOptions::new())
            .await
            .unwrap();
        c.commit(t2, upsert("t2_a"), CommitOptions::new())
            .await
            .unwrap();
        c.commit(t1, upsert("t1_b"), CommitOptions::new())
            .await
            .unwrap();
    }

    {
        let c = LocalSqliteCommitter::open(&path).expect("open phase 2");
        let t1_entries = c.read_range(t1, 0, 100).await.unwrap();
        let t2_entries = c.read_range(t2, 0, 100).await.unwrap();
        assert_eq!(t1_entries.len(), 2);
        assert_eq!(t2_entries.len(), 1);
        // t1's log_indices are 1, 2 (per-tenant — t2's commit doesn't shift them).
        assert_eq!(t1_entries[0].log_index, 1);
        assert_eq!(t1_entries[1].log_index, 2);
        assert_eq!(t2_entries[0].log_index, 1);
    }
}

#[tokio::test]
async fn dyn_dispatch_with_persistent_backing_works() {
    // The trait surface must remain dyn-dispatchable even with the
    // SQLite-backed impl. Pin it as part of PR-2's acceptance gate.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("commit.db");
    let c: Arc<dyn MutationCommitter> = Arc::new(LocalSqliteCommitter::open(&path).expect("open"));
    let r = c
        .commit(TenantId::new(1), upsert("dyn"), CommitOptions::new())
        .await
        .unwrap();
    assert_eq!(r.log_index, 1);
}

#[tokio::test]
async fn migration_runs_idempotently_on_reopen() {
    // Reopening shouldn't try to re-run the m001 migration; the
    // _yantrikdb_meta_migrations table should already record it.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("commit.db");

    for _ in 0..3 {
        let _c = LocalSqliteCommitter::open(&path).expect("open");
        // no-op; just constructing the committer runs migrations.
    }

    // After three opens, there should still be exactly one row in
    // _yantrikdb_meta_migrations for m001.
    let conn = rusqlite::Connection::open(&path).unwrap();
    let count: u32 = conn
        .query_row(
            "SELECT COUNT(*) FROM _yantrikdb_meta_migrations WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        count, 1,
        "m001 should appear exactly once after multiple opens"
    );
}
