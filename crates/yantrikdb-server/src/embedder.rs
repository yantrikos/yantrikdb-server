//! Built-in embedding via fastembed (all-MiniLM-L6-v2, ONNX).
//!
//! Implements the yantrikdb-core `Embedder` trait so engines can
//! auto-embed text without client-provided vectors.

use parking_lot::Mutex;
use std::sync::Arc;

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

        // ort (the underlying ONNX Runtime binding) panics on dlopen failure
        // instead of returning an error. Catch the panic and convert it into
        // a clear actionable error so users don't see a raw stack trace on
        // first run.
        let model = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                    .with_show_download_progress(true),
            )
        }))
        .map_err(|panic_info| {
            let msg = panic_info
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic_info.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".to_string());

            let hint = if msg.contains("dlopen") || msg.contains("libonnxruntime") || msg.contains("onnxruntime.dll") {
                "\n\n\
                ONNX Runtime library not found. The built-in embedder requires it.\n\
                \n\
                Linux:   wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-1.24.4.tgz\n\
                         tar xzf onnxruntime-linux-x64-1.24.4.tgz\n\
                         sudo cp onnxruntime-linux-x64-1.24.4/lib/libonnxruntime*.so* /usr/local/lib/\n\
                         sudo ldconfig\n\
                         export ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so.1.24.4\n\
                \n\
                macOS:   brew install onnxruntime\n\
                \n\
                Windows: Download from https://github.com/microsoft/onnxruntime/releases\n\
                         and place onnxruntime.dll alongside the binary.\n\
                \n\
                Or use the Docker image (ghcr.io/yantrikos/yantrikdb) which bundles ONNX Runtime.\n\
                \n\
                Or skip the built-in embedder by setting [embedding] strategy = \"client_only\"\n\
                in your config file (you'll need to provide pre-computed embeddings to remember()).\n"
            } else {
                ""
            };

            anyhow::anyhow!(
                "embedder initialization failed: {}{}",
                msg,
                hint,
            )
        })??;

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
        let mut model = self.inner.model.lock();
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
        let mut model = self.inner.model.lock();
        let owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let results = model.embed(owned, None)?;
        Ok(results)
    }

    fn dim(&self) -> usize {
        self.inner.dim
    }
}
