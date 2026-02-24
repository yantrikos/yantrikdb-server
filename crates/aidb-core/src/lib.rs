// ── Directory modules ──
mod base;
pub mod engine;
mod cognition;
mod distributed;
mod knowledge;
mod vector;

// ── Re-exports at original crate paths ──
pub use base::{bench_utils, compression, error, hlc, schema, scoring, serde_helpers, types};
pub use cognition::{consolidate, patterns, triggers};
pub use distributed::{conflict, replication, sync};
pub use knowledge::{graph, graph_index};
pub use vector::hnsw;

// ── Convenience re-exports ──
pub use engine::AIDB;
pub use error::AidbError;
pub use types::*;
pub use consolidate::{consolidate, find_consolidation_candidates};
pub use triggers::{check_decay_triggers, check_consolidation_triggers, check_all_triggers};
pub use conflict::{scan_conflicts, detect_edge_conflicts, create_conflict};
pub use patterns::mine_patterns;
