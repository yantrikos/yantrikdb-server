//! `/v1/debug/history/{tenant_id}` — read committed log entries.
//!
//! ## Why this endpoint exists
//!
//! Jepsen linearizability checkers need a total-order history of
//! committed operations. Our [`crate::commit::MutationCommitter`] trait
//! already exposes `read_range`; this module is the HTTP wrapper that
//! makes it accessible to Jepsen runners (and to operators triaging
//! incidents).
//!
//! ## Wire format
//!
//! Returns a JSON object:
//!
//! ```jsonc
//! {
//!   "tenant_id": 42,
//!   "from_index": 100,
//!   "limit": 50,
//!   "high_watermark": 587,
//!   "entries": [
//!     {
//!       "op_id": "01...uuid7",
//!       "tenant_id": 42,
//!       "term": 0,
//!       "log_index": 100,
//!       "mutation": { "kind": "UpsertMemory", ... },  // RFC 010 PR-3 wire format
//!       "committed_at": <SystemTime>,
//!       "applied_at": <SystemTime|null>
//!     },
//!     ...
//!   ]
//! }
//! ```
//!
//! The `mutation` field follows the v1.0 wire format pinned by
//! `tests/wire_format_v1_0.rs`. Jepsen scripts can rely on it being
//! stable for the lifetime of v1.x.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::commit::{CommittedEntry, MutationCommitter, TenantId};

/// Request parameters for `GET /v1/debug/history/{tenant_id}`.
#[derive(Debug, Clone, Deserialize)]
pub struct HistoryParams {
    /// Inclusive starting log_index. 0 means "from the beginning".
    /// Defaults to 0 if not provided.
    #[serde(default)]
    pub from: Option<u64>,
    /// Maximum entries to return. Capped at 1000 server-side to prevent
    /// debug-endpoint amplification attacks. Defaults to 100.
    #[serde(default)]
    pub limit: Option<usize>,
}

impl HistoryParams {
    pub fn from_index(&self) -> u64 {
        self.from.unwrap_or(0)
    }
    pub fn effective_limit(&self) -> usize {
        self.limit.unwrap_or(100).min(MAX_HISTORY_LIMIT)
    }
}

/// Hard cap on history page size. Prevents `/v1/debug/history?limit=99999999`
/// from amplifying memory pressure on a node already being faulted by
/// the test harness.
pub const MAX_HISTORY_LIMIT: usize = 1000;

#[derive(Debug, Serialize)]
pub struct HistoryResponse {
    pub tenant_id: TenantId,
    pub from_index: u64,
    pub limit: usize,
    pub high_watermark: u64,
    pub entries: Vec<CommittedEntry>,
}

/// Implementation of the history-read logic. Separated from the HTTP
/// layer so unit tests can exercise it without spinning up axum.
pub async fn read_history(
    committer: &Arc<dyn MutationCommitter>,
    tenant_id: TenantId,
    params: &HistoryParams,
) -> Result<HistoryResponse, crate::commit::CommitError> {
    let from = params.from_index();
    let limit = params.effective_limit();
    let entries = committer.read_range(tenant_id, from, limit).await?;
    let high_watermark = committer.high_watermark(tenant_id).await?;
    Ok(HistoryResponse {
        tenant_id,
        from_index: from,
        limit,
        high_watermark,
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::{CommitOptions, LocalSqliteCommitter, MemoryMutation};

    fn upsert(tag: &str) -> MemoryMutation {
        MemoryMutation::UpsertMemory {
            rid: format!("mem_{tag}"),
            text: tag.into(),
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
        }
    }

    #[tokio::test]
    async fn read_history_returns_full_log_for_small_tenant() {
        let committer: Arc<dyn MutationCommitter> =
            Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let t = TenantId::new(1);
        for i in 1..=5 {
            committer
                .commit(t, upsert(&format!("e{i}")), CommitOptions::new())
                .await
                .unwrap();
        }
        let resp = read_history(
            &committer,
            t,
            &HistoryParams {
                from: None,
                limit: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(resp.tenant_id, t);
        assert_eq!(resp.from_index, 0);
        assert_eq!(resp.high_watermark, 5);
        assert_eq!(resp.entries.len(), 5);
        for (i, entry) in resp.entries.iter().enumerate() {
            assert_eq!(entry.log_index, (i as u64) + 1);
        }
    }

    #[tokio::test]
    async fn limit_is_capped_at_max_history_limit() {
        let committer: Arc<dyn MutationCommitter> =
            Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let resp = read_history(
            &committer,
            TenantId::new(1),
            &HistoryParams {
                from: Some(0),
                limit: Some(99_999_999),
            },
        )
        .await
        .unwrap();
        // No entries to return on an empty committer, but `limit` must
        // be the capped value, not the requested 99M.
        assert_eq!(resp.limit, MAX_HISTORY_LIMIT);
    }

    #[tokio::test]
    async fn read_history_respects_from_index() {
        let committer: Arc<dyn MutationCommitter> =
            Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let t = TenantId::new(1);
        for i in 1..=10 {
            committer
                .commit(t, upsert(&format!("e{i}")), CommitOptions::new())
                .await
                .unwrap();
        }
        let resp = read_history(
            &committer,
            t,
            &HistoryParams {
                from: Some(7),
                limit: Some(100),
            },
        )
        .await
        .unwrap();
        assert_eq!(resp.entries.len(), 4); // log_index 7, 8, 9, 10
        assert_eq!(resp.entries[0].log_index, 7);
        assert_eq!(resp.entries[3].log_index, 10);
        assert_eq!(resp.high_watermark, 10);
    }

    #[tokio::test]
    async fn read_history_for_unknown_tenant_returns_empty() {
        let committer: Arc<dyn MutationCommitter> =
            Arc::new(LocalSqliteCommitter::open_in_memory().unwrap());
        let resp = read_history(&committer, TenantId::new(999), &HistoryParams::default())
            .await
            .unwrap();
        assert!(resp.entries.is_empty());
        assert_eq!(resp.high_watermark, 0);
    }
}

impl Default for HistoryParams {
    fn default() -> Self {
        Self {
            from: None,
            limit: None,
        }
    }
}
