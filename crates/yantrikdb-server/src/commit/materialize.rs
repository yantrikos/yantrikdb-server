//! Materialization — request body → deterministic [`MemoryMutation`].
//!
//! ## Why this exists (RFC 010 PR-6.2 §3)
//!
//! For follower apply to be byte-deterministic, the leader must do every
//! piece of nondeterministic work *before* the mutation enters the
//! commit log:
//!
//! - **Embedder runs only on the leader.** Followers consume the
//!   materialized embedding. Embedder version skew between nodes
//!   cannot diverge HNSW state.
//! - **NER runs only on the leader.** Same reason — NER models drift
//!   silently across upgrades; running on every node is a divergence
//!   risk.
//! - **Server-assigned timestamps stamped once at the leader.**
//!   Followers use the stamped value rather than reading their own
//!   wall clock. Apply on a follower five seconds after the leader
//!   does not reflect the follower's clock.
//! - **Server-assigned rids stamped once at the leader.** Mutation
//!   carries the rid so engine `record_with_rid` (PR 6.4) is purely
//!   deterministic — given the same mutation, every node produces
//!   the same row.
//!
//! ## Scope of PR 6.2 (this file)
//!
//! Trait shape + a [`LocalMaterializer`] that composes injected
//! [`Embedder`] and [`EntityExtractor`] traits. Engine wiring (passing
//! `yantrikdb::YantrikDB::embed` and the NER extractor as the trait
//! impls) lands in PR 6.4 once the handler migration goes through.
//!
//! Until PR 6.4: handlers continue to call `engine.record()` directly
//! and the materializer is unused on the production hot path. The
//! trait shape and the deterministic-mutation contract are what PR 6.2
//! commits to.

use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use thiserror::Error;
use uuid7::uuid7;

use super::mutation::MemoryMutation;

/// Embedder abstraction used by [`Materializer`]. PR 6.4 wires this to
/// `yantrikdb::YantrikDB::embed`. PR 6.2 ships the trait so PR 6.4 is
/// a thin wiring change rather than a refactor.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Encode `text` to a vector. The implementation MUST be
    /// deterministic given a fixed `model_id`. The leader calls this
    /// once at materialization time; followers never call it.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, MaterializeError>;

    /// Identifier for the embedding model in use. Stamped onto the
    /// mutation as `embedding_model` so RFC 013 model migration knows
    /// which embeddings need re-encoding when the cluster upgrades.
    fn model_id(&self) -> String;
}

/// NER abstraction used by [`Materializer`]. PR 6.4 wires this to the
/// engine's entity extractor. Empty default OK if NER isn't configured;
/// the Applier handles empty `extracted_entities` by leaving the
/// engine's own extractor to fill in (matches v1.0 behavior).
#[async_trait]
pub trait EntityExtractor: Send + Sync {
    /// Extract entity references from `text`. Return type is `Vec<String>`
    /// in PR 6.2 (the wire-format type for `MemoryMutation::UpsertMemory::extracted_entities`).
    /// PR 6.4 may swap to a richer struct once the engine's `EntityRef`
    /// shape stabilizes — that would be a wire-minor bump.
    async fn extract(&self, text: &str) -> Result<Vec<String>, MaterializeError>;
}

/// Materializer errors.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum MaterializeError {
    /// Embedder couldn't encode the text. Real production failure mode:
    /// ONNX runtime error, model not loaded, dimension mismatch.
    #[error("embed failed: {message}")]
    EmbedFailure { message: String },

    /// NER couldn't extract. Less catastrophic — the Applier's empty-fall-back
    /// path handles missing entities. But surface explicitly so the leader
    /// can decide whether to retry or proceed with empty entities.
    #[error("entity extraction failed: {message}")]
    NerFailure { message: String },

    /// Caller supplied a request that's missing a load-bearing field.
    /// PR 6.2 doesn't enforce many shape rules (the HTTP gateway already
    /// validates); reserved for invariants the materializer specifically
    /// needs (e.g. text length > 0).
    #[error("invalid request: {message}")]
    InvalidRequest { message: String },
}

/// Plain-data view of a `/v1/remember`-style request. Decoupled from the
/// HTTP layer so this module is testable without a full Axum stack.
///
/// PR 6.4 will populate this from the existing `Command::Remember` body
/// inside the migrated handler. PR 6.2 stops here.
#[derive(Debug, Clone, PartialEq)]
pub struct RememberRequest {
    pub text: String,
    pub memory_type: String,
    pub importance: f64,
    pub valence: f64,
    pub half_life: f64,
    pub namespace: String,
    pub certainty: f64,
    pub domain: String,
    pub source: String,
    pub emotional_state: Option<String>,
    pub metadata: serde_json::Value,
    /// Caller may pre-supply an embedding (rare — bulk import paths).
    /// When `None`, the materializer calls the embedder.
    pub client_embedding: Option<Vec<f32>>,
}

/// Convert a request into a fully-materialized [`MemoryMutation::UpsertMemory`].
///
/// Contract:
/// - **Deterministic given the same input.** rid is fresh (UUIDv7, time-
///   ordered + globally unique); timestamp is read once. Beyond that,
///   the output is byte-equivalent across every leader call with the
///   same request body and the same embedder/ner.
/// - **Idempotent on retry from caller side.** If the caller wants
///   client-side idempotency, they pass an op_id at the commit layer
///   ([`crate::commit::CommitOptions::with_op_id`]) and the committer
///   dedupes. The materializer itself doesn't track op_ids.
#[async_trait]
pub trait Materializer: Send + Sync {
    async fn materialize_remember(
        &self,
        req: RememberRequest,
    ) -> Result<MemoryMutation, MaterializeError>;
}

/// Concrete [`Materializer`] composing an [`Embedder`] and [`EntityExtractor`].
pub struct LocalMaterializer {
    embedder: Arc<dyn Embedder>,
    extractor: Arc<dyn EntityExtractor>,
}

impl LocalMaterializer {
    pub fn new(embedder: Arc<dyn Embedder>, extractor: Arc<dyn EntityExtractor>) -> Self {
        Self {
            embedder,
            extractor,
        }
    }
}

#[async_trait]
impl Materializer for LocalMaterializer {
    async fn materialize_remember(
        &self,
        req: RememberRequest,
    ) -> Result<MemoryMutation, MaterializeError> {
        if req.text.is_empty() {
            return Err(MaterializeError::InvalidRequest {
                message: "text is empty".into(),
            });
        }

        let embedding = match req.client_embedding {
            Some(emb) => emb,
            None => self.embedder.embed(&req.text).await?,
        };
        let extracted_entities = self.extractor.extract(&req.text).await?;
        let model_id = self.embedder.model_id();

        // System time read once on the leader. Followers use this stamped
        // value, never their own clock.
        let now_micros = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);

        // UUIDv7 — time-ordered, unique. The leader stamps the rid once;
        // followers receive it via the mutation and engine `record_with_rid`
        // (PR 6.4) writes the same row.
        let rid = uuid7().to_string();

        Ok(MemoryMutation::UpsertMemory {
            rid,
            text: req.text,
            memory_type: req.memory_type,
            importance: req.importance,
            valence: req.valence,
            half_life: req.half_life,
            namespace: req.namespace,
            certainty: req.certainty,
            domain: req.domain,
            source: req.source,
            emotional_state: req.emotional_state,
            embedding: Some(embedding),
            metadata: req.metadata,
            extracted_entities,
            created_at_unix_micros: Some(now_micros),
            embedding_model: Some(model_id),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    /// Test embedder that produces deterministic vectors from text by
    /// hashing — same text → same vector across calls. Real embedder
    /// has the same property (modulo model state) so the tests exercise
    /// the materializer's contract, not the embedder's nondeterminism.
    struct FakeEmbedder {
        model: String,
        call_count: Mutex<usize>,
    }
    impl FakeEmbedder {
        fn new(model: &str) -> Arc<Self> {
            Arc::new(Self {
                model: model.into(),
                call_count: Mutex::new(0),
            })
        }
        fn calls(&self) -> usize {
            *self.call_count.lock()
        }
    }
    #[async_trait]
    impl Embedder for FakeEmbedder {
        async fn embed(&self, text: &str) -> Result<Vec<f32>, MaterializeError> {
            *self.call_count.lock() += 1;
            // Deterministic 4-dim vector derived from text byte sum + length.
            let sum: u32 = text.bytes().map(|b| b as u32).sum();
            Ok(vec![
                sum as f32,
                text.len() as f32,
                (text.bytes().next().unwrap_or(0)) as f32,
                (text.bytes().last().unwrap_or(0)) as f32,
            ])
        }
        fn model_id(&self) -> String {
            self.model.clone()
        }
    }

    struct FakeExtractor;
    #[async_trait]
    impl EntityExtractor for FakeExtractor {
        async fn extract(&self, text: &str) -> Result<Vec<String>, MaterializeError> {
            // Pretend each word starting with uppercase is an entity.
            Ok(text
                .split_whitespace()
                .filter(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
                .map(|w| {
                    w.trim_end_matches(|c: char| !c.is_alphanumeric())
                        .to_string()
                })
                .collect())
        }
    }

    fn build_materializer() -> (LocalMaterializer, Arc<FakeEmbedder>) {
        let emb = FakeEmbedder::new("test-model.v1");
        let mat = LocalMaterializer::new(emb.clone(), Arc::new(FakeExtractor));
        (mat, emb)
    }

    fn req(text: &str) -> RememberRequest {
        RememberRequest {
            text: text.into(),
            memory_type: "semantic".into(),
            importance: 0.7,
            valence: 0.0,
            half_life: 86400.0,
            namespace: "test".into(),
            certainty: 1.0,
            domain: "general".into(),
            source: "test".into(),
            emotional_state: None,
            metadata: serde_json::json!({}),
            client_embedding: None,
        }
    }

    #[tokio::test]
    async fn materialize_populates_all_v1_1_fields() {
        // The v1.1 contract: every materialized mutation MUST have
        // embedding=Some, extracted_entities populated (or empty if
        // none), created_at=Some, embedding_model=Some. PR 6.4
        // followers depend on this — the Applier asserts it and
        // shuts down on a missing field.
        let (mat, _) = build_materializer();
        let m = mat
            .materialize_remember(req("Alice met Bob in Paris"))
            .await
            .unwrap();
        match m {
            MemoryMutation::UpsertMemory {
                rid,
                embedding,
                extracted_entities,
                created_at_unix_micros,
                embedding_model,
                ..
            } => {
                assert!(!rid.is_empty(), "rid stamped");
                assert!(embedding.is_some(), "embedding stamped");
                assert_eq!(extracted_entities, vec!["Alice", "Bob", "Paris"], "NER ran");
                assert!(created_at_unix_micros.is_some(), "timestamp stamped");
                assert_eq!(embedding_model.as_deref(), Some("test-model.v1"));
            }
            _ => panic!("expected UpsertMemory"),
        }
    }

    #[tokio::test]
    async fn client_supplied_embedding_skips_embedder() {
        // Bulk-import paths can supply their own embedding; the leader
        // must NOT re-embed (idempotent re-import cost). Verified by
        // observing the embedder's call_count.
        let (mat, emb) = build_materializer();
        let mut r = req("hello");
        r.client_embedding = Some(vec![9.0, 9.0, 9.0]);
        let m = mat.materialize_remember(r).await.unwrap();
        assert_eq!(emb.calls(), 0, "client embedding bypasses embedder");
        match m {
            MemoryMutation::UpsertMemory { embedding, .. } => {
                assert_eq!(embedding, Some(vec![9.0, 9.0, 9.0]));
            }
            _ => panic!("expected UpsertMemory"),
        }
    }

    #[tokio::test]
    async fn empty_text_is_invalid() {
        // Catches bad-shape callers before they pollute the commit log.
        let (mat, _) = build_materializer();
        let mut r = req("");
        r.client_embedding = Some(vec![]);
        let err = mat.materialize_remember(r).await.unwrap_err();
        assert!(matches!(err, MaterializeError::InvalidRequest { .. }));
    }

    #[tokio::test]
    async fn rid_is_fresh_each_call() {
        // UUIDv7 collision is astronomically unlikely; just confirm
        // the materializer doesn't accidentally cache or hard-code.
        let (mat, _) = build_materializer();
        let m1 = mat.materialize_remember(req("hello")).await.unwrap();
        let m2 = mat.materialize_remember(req("hello")).await.unwrap();
        match (&m1, &m2) {
            (
                MemoryMutation::UpsertMemory { rid: r1, .. },
                MemoryMutation::UpsertMemory { rid: r2, .. },
            ) => assert_ne!(r1, r2),
            _ => panic!("expected UpsertMemory"),
        }
    }

    #[tokio::test]
    async fn embedder_failure_surfaces_as_embed_failure() {
        struct AngryEmbedder;
        #[async_trait]
        impl Embedder for AngryEmbedder {
            async fn embed(&self, _: &str) -> Result<Vec<f32>, MaterializeError> {
                Err(MaterializeError::EmbedFailure {
                    message: "ONNX refused".into(),
                })
            }
            fn model_id(&self) -> String {
                "angry.v1".into()
            }
        }
        let mat = LocalMaterializer::new(Arc::new(AngryEmbedder), Arc::new(FakeExtractor));
        let err = mat.materialize_remember(req("hello")).await.unwrap_err();
        assert!(matches!(err, MaterializeError::EmbedFailure { .. }));
    }

    #[tokio::test]
    async fn ner_failure_surfaces_as_ner_failure() {
        struct AngryExtractor;
        #[async_trait]
        impl EntityExtractor for AngryExtractor {
            async fn extract(&self, _: &str) -> Result<Vec<String>, MaterializeError> {
                Err(MaterializeError::NerFailure {
                    message: "NER pipeline crashed".into(),
                })
            }
        }
        let mat = LocalMaterializer::new(FakeEmbedder::new("test.v1"), Arc::new(AngryExtractor));
        let err = mat.materialize_remember(req("hello")).await.unwrap_err();
        assert!(matches!(err, MaterializeError::NerFailure { .. }));
    }

    // Compile-time pin: Materializer is dyn-compatible. PR 6.4 will hold
    // `Arc<dyn Materializer>` on AppState.
    #[allow(dead_code)]
    fn _dyn_materializer_compile_check(_m: Arc<dyn Materializer>) {}
}
