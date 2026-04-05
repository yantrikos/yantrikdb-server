//! Built-in embedding via fastembed (all-MiniLM-L6-v2, ONNX).
//!
//! Implements the yantrikdb-core `Embedder` trait so engines can
//! auto-embed text without client-provided vectors.

use std::sync::{Arc, Mutex};

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// Inner model shared across all engines.
struct FastEmbedInner {
    model: Mutex<TextEmbedding>,
    dim: usize,
}

/// Shareable embedder — clone this to give each engine its own `Box<dyn Embedder>`.
#[derive(Clone)]
pub struct FastEmbedder {
    inner: Arc<FastEmbedInner>,
}

impl FastEmbedder {
    pub fn new() -> anyhow::Result<Self> {
        tracing::info!("loading embedding model (all-MiniLM-L6-v2)...");
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
        )?;
        tracing::info!("embedding model loaded (384 dim)");

        Ok(Self {
            inner: Arc::new(FastEmbedInner {
                model: Mutex::new(model),
                dim: 384,
            }),
        })
    }

    /// Create a boxed clone suitable for `YantrikDB::set_embedder()`.
    pub fn boxed(&self) -> Box<dyn yantrikdb::types::Embedder + Send + Sync> {
        Box::new(self.clone())
    }
}

impl yantrikdb::types::Embedder for FastEmbedder {
    fn embed(
        &self,
        text: &str,
    ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let mut model = self.inner.model.lock().unwrap();
        let results = model.embed(vec![text], None)?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| "empty embedding result".into())
    }

    fn embed_batch(
        &self,
        texts: &[&str],
    ) -> std::result::Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let mut model = self.inner.model.lock().unwrap();
        let owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let results = model.embed(owned, None)?;
        Ok(results)
    }

    fn dim(&self) -> usize {
        self.inner.dim
    }
}
