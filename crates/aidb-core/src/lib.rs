pub mod error;
pub mod types;
pub mod schema;
pub mod serde_helpers;
pub mod scoring;
pub mod hlc;
pub mod engine;
pub mod consolidate;
pub mod triggers;
pub mod replication;
pub mod sync;
pub mod conflict;

pub use engine::AIDB;
pub use error::AidbError;
pub use types::*;
pub use consolidate::{consolidate, find_consolidation_candidates};
pub use triggers::{check_decay_triggers, check_consolidation_triggers, check_all_triggers};
pub use conflict::{scan_conflicts, detect_edge_conflicts, create_conflict};
