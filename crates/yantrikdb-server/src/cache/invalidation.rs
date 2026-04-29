//! Cache invalidation bus — fan-out channel for cache-relevant events.
//!
//! ## What it broadcasts
//!
//! Every event that should invalidate cache entries flows through this
//! bus. Consumers (concrete caches in RFC 015-B) subscribe; the
//! commit-log apply path (RFC 011-A wires this) publishes.
//!
//! Initial event types ([`InvalidationEvent`]):
//! - `Tombstoned` — a memory was tombstoned; caches with cached values
//!   covering this rid invalidate them.
//! - `Updated` — a memory was updated (UpdateMemoryPatch); caches with
//!   cached values for this rid invalidate.
//! - `EdgeChanged` — an entity edge added/removed; caches whose values
//!   depend on entity adjacency invalidate.
//! - `TenantConfigChanged` — tenant config patched; caches keyed on
//!   tenant config (e.g. namespace-scoped settings) invalidate.
//!
//! ## Why a broadcast bus instead of per-cache hooks
//!
//! - One commit-log apply call publishes one event; all caches see it.
//! - New caches subscribe at runtime without modifying the apply path.
//! - Slow consumers don't block the apply path: each subscriber drains
//!   its own channel; if a subscriber falls behind, it sees a lag warning
//!   but the apply path doesn't stall.
//!
//! ## Implementation
//!
//! [`tokio::sync::broadcast`] under the hood. Capacity bounded so a
//! stuck subscriber doesn't accumulate unbounded events; lagging
//! subscribers receive a `RecvError::Lagged(n)` and SHOULD respond by
//! flushing their cache wholesale (safer than missing events).

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::commit::TenantId;

/// Default broadcast channel capacity. Subscribers that fall this far
/// behind will see `Lagged` errors and SHOULD respond by clearing their
/// cache entirely (a safe over-invalidation that prevents serving stale
/// data after missed events).
pub const DEFAULT_BUS_CAPACITY: usize = 4096;

/// Cache-relevant events. Every variant carries enough info that any
/// subscriber can decide independently whether to invalidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum InvalidationEvent {
    /// A memory was tombstoned (RFC 011 forget). Caches with values
    /// covering this rid MUST invalidate them.
    Tombstoned {
        tenant_id: TenantId,
        rid: String,
    },

    /// A memory was updated in place. Caches with values for this rid
    /// MUST invalidate so the next read sees the new content.
    Updated {
        tenant_id: TenantId,
        rid: String,
    },

    /// An entity edge changed. Caches whose values depend on entity
    /// adjacency (e.g. expanded recall results) invalidate.
    EdgeChanged {
        tenant_id: TenantId,
        src: String,
        dst: String,
    },

    /// Tenant config patched. Caches keyed on per-tenant config invalidate.
    TenantConfigChanged {
        tenant_id: TenantId,
        key: String,
    },
}

impl InvalidationEvent {
    /// The tenant scope of this event. Used by per-tenant caches to
    /// filter events they don't care about.
    pub fn tenant_id(&self) -> TenantId {
        match self {
            InvalidationEvent::Tombstoned { tenant_id, .. }
            | InvalidationEvent::Updated { tenant_id, .. }
            | InvalidationEvent::EdgeChanged { tenant_id, .. }
            | InvalidationEvent::TenantConfigChanged { tenant_id, .. } => *tenant_id,
        }
    }

    /// Stable variant name for metrics/logs. Never rename.
    pub fn variant_name(&self) -> &'static str {
        match self {
            InvalidationEvent::Tombstoned { .. } => "Tombstoned",
            InvalidationEvent::Updated { .. } => "Updated",
            InvalidationEvent::EdgeChanged { .. } => "EdgeChanged",
            InvalidationEvent::TenantConfigChanged { .. } => "TenantConfigChanged",
        }
    }
}

/// Broadcast bus. Cheap to clone (Arc<broadcast::Sender> internally).
#[derive(Clone)]
pub struct InvalidationBus {
    sender: broadcast::Sender<InvalidationEvent>,
}

impl InvalidationBus {
    /// Construct with the default capacity. Use [`Self::with_capacity`]
    /// to override.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_BUS_CAPACITY)
    }

    pub fn with_capacity(cap: usize) -> Self {
        let (sender, _rx) = broadcast::channel(cap);
        Self { sender }
    }

    /// Publish an event. Returns the number of subscribers that received
    /// it. (Drops the count if no subscribers — this is a deliberate
    /// non-error: the commit path doesn't care if anyone's listening.)
    pub fn publish(&self, event: InvalidationEvent) -> usize {
        // tokio's broadcast::Sender::send returns Err only if there are
        // no active receivers. We treat that as "0 delivered" rather
        // than an error, so the commit path never fails because a
        // subscriber dropped.
        self.sender.send(event).unwrap_or(0)
    }

    /// Subscribe a new consumer. Each subscriber gets its own queue;
    /// publishing fan-outs to all subscribers.
    pub fn subscribe(&self) -> broadcast::Receiver<InvalidationEvent> {
        self.sender.subscribe()
    }

    /// Number of currently-active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl Default for InvalidationBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn empty_bus_starts_with_zero_subscribers() {
        let bus = InvalidationBus::new();
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[tokio::test]
    async fn publish_with_no_subscribers_returns_zero_not_error() {
        let bus = InvalidationBus::new();
        // Publishing into the void is fine — commit path can't fail
        // because nobody's listening.
        let n = bus.publish(InvalidationEvent::Tombstoned {
            tenant_id: TenantId::new(1),
            rid: "x".into(),
        });
        assert_eq!(n, 0);
    }

    #[tokio::test]
    async fn subscribe_receives_published_event() {
        let bus = InvalidationBus::new();
        let mut rx = bus.subscribe();
        bus.publish(InvalidationEvent::Tombstoned {
            tenant_id: TenantId::new(1),
            rid: "mem_a".into(),
        });
        let evt = rx.recv().await.expect("receive");
        match evt {
            InvalidationEvent::Tombstoned { tenant_id, rid } => {
                assert_eq!(tenant_id, TenantId::new(1));
                assert_eq!(rid, "mem_a");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[tokio::test]
    async fn multi_subscriber_fanout() {
        let bus = InvalidationBus::new();
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();
        let mut rx3 = bus.subscribe();
        let n = bus.publish(InvalidationEvent::Updated {
            tenant_id: TenantId::new(1),
            rid: "mem_a".into(),
        });
        assert_eq!(n, 3);
        // All three receive.
        for rx in [&mut rx1, &mut rx2, &mut rx3] {
            let evt = rx.recv().await.expect("receive");
            assert_eq!(evt.variant_name(), "Updated");
        }
    }

    #[tokio::test]
    async fn slow_subscriber_lags_but_does_not_block_publisher() {
        // capacity=2 so we can demonstrate lag handling.
        let bus = InvalidationBus::with_capacity(2);
        let mut rx = bus.subscribe();
        // Publish 5 events; subscriber's queue holds 2.
        for i in 0..5 {
            bus.publish(InvalidationEvent::Tombstoned {
                tenant_id: TenantId::new(1),
                rid: format!("mem_{i}"),
            });
        }
        // First recv: should hit Lagged because we filled past capacity.
        let result = rx.recv().await;
        match result {
            Err(broadcast::error::RecvError::Lagged(n)) => {
                assert!(n >= 1, "lagged by at least 1 message");
            }
            other => panic!("expected Lagged, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn subscriber_count_tracks_active_receivers() {
        let bus = InvalidationBus::new();
        assert_eq!(bus.subscriber_count(), 0);
        let _rx1 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 1);
        {
            let _rx2 = bus.subscribe();
            assert_eq!(bus.subscriber_count(), 2);
        }
        // rx2 dropped; count drops back.
        assert_eq!(bus.subscriber_count(), 1);
    }

    #[test]
    fn variant_names_are_stable() {
        // Pin metric labels.
        assert_eq!(
            InvalidationEvent::Tombstoned {
                tenant_id: TenantId::new(1),
                rid: "x".into(),
            }
            .variant_name(),
            "Tombstoned"
        );
        assert_eq!(
            InvalidationEvent::Updated {
                tenant_id: TenantId::new(1),
                rid: "x".into(),
            }
            .variant_name(),
            "Updated"
        );
        assert_eq!(
            InvalidationEvent::EdgeChanged {
                tenant_id: TenantId::new(1),
                src: "a".into(),
                dst: "b".into(),
            }
            .variant_name(),
            "EdgeChanged"
        );
        assert_eq!(
            InvalidationEvent::TenantConfigChanged {
                tenant_id: TenantId::new(1),
                key: "k".into(),
            }
            .variant_name(),
            "TenantConfigChanged"
        );
    }

    #[test]
    fn tenant_id_extraction_works_for_all_variants() {
        let t = TenantId::new(42);
        let cases = vec![
            InvalidationEvent::Tombstoned {
                tenant_id: t,
                rid: "x".into(),
            },
            InvalidationEvent::Updated {
                tenant_id: t,
                rid: "x".into(),
            },
            InvalidationEvent::EdgeChanged {
                tenant_id: t,
                src: "a".into(),
                dst: "b".into(),
            },
            InvalidationEvent::TenantConfigChanged {
                tenant_id: t,
                key: "k".into(),
            },
        ];
        for evt in cases {
            assert_eq!(evt.tenant_id(), t);
        }
    }
}
