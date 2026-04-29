//! `VersionedEvent` trait — every replicated/durable thing carries a version.
//!
//! Every event written to the commit log, replicated to peers, snapshotted
//! to backup, or persisted in any durable table MUST implement this trait.
//! Without it, future code can't know how to interpret historical events.
//!
//! ## Why this is a trait, not just two fields
//!
//! - Some events derive their version from their variant (e.g. a
//!   `MemoryMutation` enum where each variant declares its minimum
//!   `WireVersion`)
//! - Some events combine schema version of multiple tables (e.g. a backup
//!   manifest covers many tables)
//! - The trait keeps version concerns separate from event content,
//!   centralizing the compatibility check
//!
//! ## Standing acceptance criterion (RFC 017-B)
//!
//! Every new durable event type MUST implement [`VersionedEvent`]. The
//! `MemoryMutation` enum (RFC 010 PR-3) is the first user.

use super::schema::SchemaVersion;
use super::wire::WireVersion;

/// An event that can be replicated, persisted, or snapshotted.
/// Every such event carries a [`WireVersion`] (for inter-node compat) and
/// optionally a [`SchemaVersion`] for the table it's targeting.
pub trait VersionedEvent {
    /// The wire-protocol version of this event's serialization. Used for
    /// inter-node compatibility checks before apply.
    fn wire_version(&self) -> WireVersion;

    /// The minimum schema version of the target table needed to apply
    /// this event. `None` means the event is schema-independent (e.g.
    /// admin/config events that don't write to a specific data table).
    fn schema_version(&self) -> Option<(&'static str, SchemaVersion)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal example to verify the trait is usable. RFC 010 PR-3 will
    /// implement this for the real `MemoryMutation` enum.
    #[derive(Debug)]
    struct ExampleEvent {
        wire: WireVersion,
        schema: Option<(&'static str, SchemaVersion)>,
    }

    impl VersionedEvent for ExampleEvent {
        fn wire_version(&self) -> WireVersion {
            self.wire
        }
        fn schema_version(&self) -> Option<(&'static str, SchemaVersion)> {
            self.schema
        }
    }

    #[test]
    fn trait_is_implementable() {
        let e = ExampleEvent {
            wire: WireVersion::new(1, 0),
            schema: Some(("memory_commit_log", SchemaVersion::new(1))),
        };
        assert_eq!(e.wire_version(), WireVersion::new(1, 0));
        assert_eq!(
            e.schema_version(),
            Some(("memory_commit_log", SchemaVersion::new(1)))
        );
    }

    #[test]
    fn schema_version_default_is_none() {
        struct NoSchema;
        impl VersionedEvent for NoSchema {
            fn wire_version(&self) -> WireVersion {
                WireVersion::new(1, 0)
            }
        }
        let e = NoSchema;
        assert_eq!(e.schema_version(), None);
    }
}
