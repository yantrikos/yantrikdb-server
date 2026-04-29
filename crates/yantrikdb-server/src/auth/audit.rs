//! RFC 014-B — audit log substrate.
//!
//! Per RFC 014-B: "Audit log for data mutations and auth events."
//!
//! ## What gets audited
//!
//! - **Auth events**: token issued, authenticated success/failure,
//!   revoked, expired.
//! - **Mutations**: remember/correct/forget/relate (anything that changes
//!   stored memory state). The data-plane handler emits an `AuditEvent`
//!   on commit; the [`AuditSink`] persists it.
//! - **Admin events**: quota policy CRUD, breaker mode change, key
//!   rotation, tenant lifecycle.
//!
//! ## Why a sink trait
//!
//! The substrate doesn't pick a persistence target. Production writes
//! to control DB; tests use [`InMemoryAuditSink`]; future could write
//! to Kafka / OTLP / file. Keeping the trait surface narrow
//! (`record(event)`) lets the data plane stay agnostic.
//!
//! ## Why not just a metric
//!
//! Metrics aggregate and lose actor identity ("100 forgets in the last
//! hour" vs "alice forgot rid X at 10:42"). For compliance and
//! incident postmortem, the per-event record is required. The sink
//! emits structured records; metrics are derived separately.

use std::sync::Arc;
use std::time::SystemTime;

use parking_lot::Mutex;

use super::scopes::Scope;

/// Discriminator for the kind of audit event. Wire form (`as_str`) is
/// pinned by tests so dashboards and external consumers don't break
/// silently on refactor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditEventKind {
    /// Token created via admin API.
    TokenIssued,
    /// Token revoked via admin API.
    TokenRevoked,
    /// Successful authentication (request flowed through).
    AuthSuccess,
    /// Authentication failure (bad token, revoked, expired).
    AuthFailure,
    /// Data mutation (remember/correct/forget/relate).
    DataMutation,
    /// Admin operation (quota set, breaker mode change, retention config).
    AdminOp,
    /// Tenant lifecycle (create/suspend/delete tenant, rotate key).
    TenantLifecycle,
}

impl AuditEventKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            AuditEventKind::TokenIssued => "token_issued",
            AuditEventKind::TokenRevoked => "token_revoked",
            AuditEventKind::AuthSuccess => "auth_success",
            AuditEventKind::AuthFailure => "auth_failure",
            AuditEventKind::DataMutation => "data_mutation",
            AuditEventKind::AdminOp => "admin_op",
            AuditEventKind::TenantLifecycle => "tenant_lifecycle",
        }
    }
}

/// What happened with the operation. Operators grep on this to find
/// failed admin attempts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditOutcome {
    Success,
    /// Authorization denied (caller authenticated but lacked the scope).
    Forbidden,
    /// Authentication failed.
    Unauthenticated,
    /// Operation failed for a non-auth reason (validation, conflict,
    /// internal error).
    Error,
}

impl AuditOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            AuditOutcome::Success => "success",
            AuditOutcome::Forbidden => "forbidden",
            AuditOutcome::Unauthenticated => "unauthenticated",
            AuditOutcome::Error => "error",
        }
    }
}

/// One audit record. Constructed by handlers, consumed by [`AuditSink`].
/// Cheap to clone — fields are owned strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditEvent {
    pub at: SystemTime,
    pub kind: AuditEventKind,
    pub outcome: AuditOutcome,
    /// Principal id, if known. None for pre-auth failures (token
    /// didn't resolve to anything).
    pub principal_id: Option<String>,
    /// Tenant scope of the operation, if applicable.
    pub tenant_id: Option<String>,
    /// Scopes the operation required. Useful for "alice tried to forget
    /// without forget scope".
    pub required_scopes: Vec<Scope>,
    /// Free-form action label, e.g. "remember", "set_quota_policy".
    /// Stable strings — dashboards key on these.
    pub action: String,
    /// Optional structured detail. Resource ids, before/after policy
    /// versions, error codes. NEVER raw token bytes.
    pub detail: Option<String>,
}

impl AuditEvent {
    pub fn now(kind: AuditEventKind, outcome: AuditOutcome, action: impl Into<String>) -> Self {
        Self {
            at: SystemTime::now(),
            kind,
            outcome,
            principal_id: None,
            tenant_id: None,
            required_scopes: Vec::new(),
            action: action.into(),
            detail: None,
        }
    }

    pub fn with_principal(mut self, id: impl Into<String>) -> Self {
        self.principal_id = Some(id.into());
        self
    }

    pub fn with_tenant(mut self, id: impl Into<String>) -> Self {
        self.tenant_id = Some(id.into());
        self
    }

    pub fn with_required_scopes(mut self, scopes: impl IntoIterator<Item = Scope>) -> Self {
        self.required_scopes = scopes.into_iter().collect();
        self
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// Pluggable audit destination. Sync because most impls are write-and-
/// done (fire-and-forget); production control-DB sink can sit behind a
/// channel that batches into SQLite asynchronously.
pub trait AuditSink: Send + Sync {
    fn record(&self, event: AuditEvent);
}

/// No-op sink. Useful in tests that don't care about audit records and
/// in dev mode where audit isn't configured.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopAuditSink;

impl AuditSink for NoopAuditSink {
    fn record(&self, _event: AuditEvent) {}
}

/// In-memory ring sink for tests. Stores up to `cap` events, drops
/// oldest on overflow.
#[derive(Clone)]
pub struct InMemoryAuditSink {
    inner: Arc<Mutex<InMemoryRing>>,
}

struct InMemoryRing {
    events: std::collections::VecDeque<AuditEvent>,
    cap: usize,
}

impl InMemoryAuditSink {
    pub fn new(cap: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(InMemoryRing {
                events: std::collections::VecDeque::with_capacity(cap.min(1024)),
                cap,
            })),
        }
    }

    pub fn snapshot(&self) -> Vec<AuditEvent> {
        self.inner.lock().events.iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().events.is_empty()
    }

    pub fn clear(&self) {
        self.inner.lock().events.clear();
    }
}

impl AuditSink for InMemoryAuditSink {
    fn record(&self, event: AuditEvent) {
        let mut g = self.inner.lock();
        if g.events.len() >= g.cap {
            g.events.pop_front();
        }
        g.events.push_back(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev_data_success() -> AuditEvent {
        AuditEvent::now(AuditEventKind::DataMutation, AuditOutcome::Success, "remember")
            .with_principal("alice")
            .with_tenant("acme")
            .with_required_scopes([Scope::Write])
            .with_detail("rid=abc")
    }

    #[test]
    fn audit_event_kind_strings_pinned() {
        // Wire format — dashboards key on these.
        assert_eq!(AuditEventKind::TokenIssued.as_str(), "token_issued");
        assert_eq!(AuditEventKind::TokenRevoked.as_str(), "token_revoked");
        assert_eq!(AuditEventKind::AuthSuccess.as_str(), "auth_success");
        assert_eq!(AuditEventKind::AuthFailure.as_str(), "auth_failure");
        assert_eq!(AuditEventKind::DataMutation.as_str(), "data_mutation");
        assert_eq!(AuditEventKind::AdminOp.as_str(), "admin_op");
        assert_eq!(AuditEventKind::TenantLifecycle.as_str(), "tenant_lifecycle");
    }

    #[test]
    fn audit_outcome_strings_pinned() {
        assert_eq!(AuditOutcome::Success.as_str(), "success");
        assert_eq!(AuditOutcome::Forbidden.as_str(), "forbidden");
        assert_eq!(AuditOutcome::Unauthenticated.as_str(), "unauthenticated");
        assert_eq!(AuditOutcome::Error.as_str(), "error");
    }

    #[test]
    fn builder_chain_yields_full_event() {
        let ev = ev_data_success();
        assert_eq!(ev.kind, AuditEventKind::DataMutation);
        assert_eq!(ev.outcome, AuditOutcome::Success);
        assert_eq!(ev.principal_id.as_deref(), Some("alice"));
        assert_eq!(ev.tenant_id.as_deref(), Some("acme"));
        assert_eq!(ev.required_scopes, vec![Scope::Write]);
        assert_eq!(ev.detail.as_deref(), Some("rid=abc"));
        assert_eq!(ev.action, "remember");
    }

    #[test]
    fn noop_sink_swallows() {
        let s = NoopAuditSink;
        s.record(ev_data_success());
        // No-op — just compiles + doesn't panic.
    }

    #[test]
    fn in_memory_sink_collects_records() {
        let s = InMemoryAuditSink::new(8);
        assert!(s.is_empty());
        s.record(ev_data_success());
        s.record(ev_data_success());
        assert_eq!(s.len(), 2);
        assert_eq!(s.snapshot().len(), 2);
    }

    #[test]
    fn in_memory_sink_drops_oldest_on_overflow() {
        let s = InMemoryAuditSink::new(2);
        let mut e1 = ev_data_success();
        e1.action = "first".to_string();
        let mut e2 = ev_data_success();
        e2.action = "second".to_string();
        let mut e3 = ev_data_success();
        e3.action = "third".to_string();

        s.record(e1);
        s.record(e2);
        s.record(e3);
        assert_eq!(s.len(), 2);
        let snap = s.snapshot();
        assert_eq!(snap[0].action, "second");
        assert_eq!(snap[1].action, "third");
    }

    #[test]
    fn clear_empties_ring() {
        let s = InMemoryAuditSink::new(8);
        s.record(ev_data_success());
        assert_eq!(s.len(), 1);
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn sink_is_send_sync_for_arc_dyn() {
        let _: Arc<dyn AuditSink> = Arc::new(NoopAuditSink);
        let _: Arc<dyn AuditSink> = Arc::new(InMemoryAuditSink::new(4));
    }

    #[test]
    fn audit_event_carries_required_scopes_for_forbidden_diagnosis() {
        // Forbidden outcomes need to record what was *required* (not
        // just what the principal had) so operators can answer "why
        // was alice forbidden?" without re-running the request.
        let ev = AuditEvent::now(AuditEventKind::DataMutation, AuditOutcome::Forbidden, "forget")
            .with_principal("alice")
            .with_required_scopes([Scope::Forget]);
        assert_eq!(ev.required_scopes, vec![Scope::Forget]);
        assert_eq!(ev.outcome, AuditOutcome::Forbidden);
    }
}
