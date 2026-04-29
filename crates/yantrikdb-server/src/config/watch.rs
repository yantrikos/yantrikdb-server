//! RFC 021 PR-1 — config watch substrate.
//!
//! Wraps `tokio::sync::watch` to expose a `(sender, receiver)` pair
//! tied to a `VersionedConfig<T>`. The recipe:
//!
//! 1. Server start: construct `(ConfigWatchSender, ConfigWatch)` from
//!    the initial config.
//! 2. The orchestrator (RFC 021 reload loop) calls `sender.publish(new)`
//!    each time a delta is applied.
//! 3. Subscribers (axum SSE handler, embedded components) call
//!    `watch.changed().await` to wait for the next version.
//!
//! ## Why `watch::channel`
//!
//! - Single-producer, multi-consumer.
//! - Latest-value semantics — slow consumers don't backpressure the
//!   producer; they just see the most recent version on next poll.
//!   That matches "config update" semantics: if 5 deltas land while a
//!   consumer is paused, they only need to see the final state.
//! - Cheap clone for the `Receiver` side. `axum::Router` can hand out
//!   one per SSE connection.
//!
//! ## What's NOT here
//!
//! - The axum SSE handler that streams `ConfigChange` events. Lives in
//!   `http_gateway.rs` consumer PR with full HTTP serialization.
//! - The mediation logic between `Reloadable::apply` and the watch
//!   send. The orchestrator (RFC 021 PR-1 consumer wiring) calls
//!   `sender.publish` after a successful apply.

use tokio::sync::watch;

use super::versioned::{ConfigVersion, VersionedConfig};

/// Sender half. Held by the reload orchestrator.
pub struct ConfigWatchSender<T> {
    inner: watch::Sender<VersionedConfig<T>>,
}

impl<T> ConfigWatchSender<T> {
    /// Publish a new versioned config. Returns `Err(())` if all
    /// receivers have been dropped — the caller can treat that as a
    /// "no-op" since no one's listening, or surface as a metric.
    pub fn publish(&self, value: VersionedConfig<T>) -> Result<(), ()> {
        self.inner.send(value).map_err(|_| ())
    }

    /// Number of currently-attached receivers. Useful for the
    /// `config_watchers_active` Prometheus gauge.
    pub fn receiver_count(&self) -> usize {
        self.inner.receiver_count()
    }
}

/// Receiver half. Cloneable — one per consumer.
pub struct ConfigWatch<T> {
    inner: watch::Receiver<VersionedConfig<T>>,
}

impl<T> Clone for ConfigWatch<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> ConfigWatch<T> {
    /// Return the current version + value snapshot. Doesn't block; the
    /// receiver always has a value (initial value at construction).
    /// `borrow()` gives a guard; this method clones to avoid lifetime
    /// surfaces in callers, since `VersionedConfig<T>` is cheap (Arc).
    pub fn current(&self) -> VersionedConfig<T>
    where
        T: Clone,
    {
        self.inner.borrow().clone()
    }

    pub fn current_version(&self) -> ConfigVersion {
        self.inner.borrow().version()
    }

    /// Wait for the next change. Returns the new value. Returns
    /// `Err(())` if the sender has been dropped.
    pub async fn changed(&mut self) -> Result<VersionedConfig<T>, ()>
    where
        T: Clone,
    {
        self.inner.changed().await.map_err(|_| ())?;
        Ok(self.inner.borrow().clone())
    }

    /// Mark the current value as seen. After calling, `changed()` will
    /// only return on the *next* publish.
    pub fn mark_seen(&mut self) {
        self.inner.mark_unchanged();
    }
}

/// Construct a paired sender + receiver seeded with `initial`. The
/// receiver starts with the initial value visible (its `current()`
/// returns `initial`).
pub fn channel<T>(initial: VersionedConfig<T>) -> (ConfigWatchSender<T>, ConfigWatch<T>) {
    let (tx, rx) = watch::channel(initial);
    (ConfigWatchSender { inner: tx }, ConfigWatch { inner: rx })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn vcfg(v: u64, x: u32) -> VersionedConfig<u32> {
        VersionedConfig::new(x, ConfigVersion(v))
    }

    #[test]
    fn current_returns_initial() {
        let (_tx, rx) = channel(vcfg(0, 42));
        assert_eq!(rx.current_version(), ConfigVersion(0));
        assert_eq!(*rx.current().value(), 42);
    }

    #[tokio::test]
    async fn changed_returns_after_publish() {
        let (tx, mut rx) = channel(vcfg(0, 42));
        rx.mark_seen();
        let publisher = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            tx.publish(vcfg(1, 99)).unwrap();
        });
        let next = rx.changed().await.unwrap();
        assert_eq!(next.version(), ConfigVersion(1));
        assert_eq!(*next.value(), 99);
        publisher.await.unwrap();
    }

    #[tokio::test]
    async fn changed_returns_err_when_sender_dropped() {
        let (tx, mut rx) = channel(vcfg(0, 42));
        rx.mark_seen();
        drop(tx);
        let err = rx.changed().await;
        assert!(err.is_err());
    }

    #[test]
    fn receiver_count_reflects_clones() {
        let (tx, rx) = channel(vcfg(0, 42));
        assert_eq!(tx.receiver_count(), 1);
        let _rx2 = rx.clone();
        assert_eq!(tx.receiver_count(), 2);
        drop(rx);
        assert_eq!(tx.receiver_count(), 1);
    }

    #[test]
    fn publish_with_no_receivers_returns_err() {
        let (tx, rx) = channel(vcfg(0, 42));
        drop(rx);
        let result = tx.publish(vcfg(1, 99));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn multiple_receivers_all_see_same_version() {
        let (tx, rx) = channel(vcfg(0, 42));
        let mut rx1 = rx.clone();
        let mut rx2 = rx.clone();
        rx1.mark_seen();
        rx2.mark_seen();
        tx.publish(vcfg(1, 99)).unwrap();
        let v1 = rx1.changed().await.unwrap();
        let v2 = rx2.changed().await.unwrap();
        assert_eq!(v1.version(), v2.version());
    }

    #[tokio::test]
    async fn slow_consumer_sees_only_latest_value() {
        // tokio::sync::watch has latest-value semantics. Publish v1, v2,
        // v3 quickly; the consumer wakes up and sees v3 — never v1 or v2.
        let (tx, mut rx) = channel(vcfg(0, 42));
        rx.mark_seen();
        tx.publish(vcfg(1, 1)).unwrap();
        tx.publish(vcfg(2, 2)).unwrap();
        tx.publish(vcfg(3, 3)).unwrap();
        let v = rx.changed().await.unwrap();
        assert_eq!(v.version(), ConfigVersion(3));
        assert_eq!(*v.value(), 3);
    }

    #[test]
    fn current_clone_shares_arc() {
        let (_tx, rx) = channel(vcfg(0, 42));
        let a = rx.current();
        let b = rx.current();
        // Both clones point at same Arc-wrapped inner.
        assert!(std::sync::Arc::ptr_eq(&a.arc(), &b.arc()));
    }
}
