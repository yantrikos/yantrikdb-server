use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub server: ServerSection,
    pub embedding: EmbeddingSection,
    pub background: BackgroundSection,
    pub limits: LimitsSection,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerSection {
    pub wire_port: u16,
    pub http_port: u16,
    pub data_dir: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingSection {
    pub strategy: EmbeddingStrategy,
    pub dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingStrategy {
    Builtin,
    ClientOnly,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BackgroundSection {
    pub consolidation_interval_minutes: u64,
    pub decay_sweep_interval_minutes: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LimitsSection {
    pub max_databases: usize,
    pub max_connections: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server: ServerSection::default(),
            embedding: EmbeddingSection::default(),
            background: BackgroundSection::default(),
            limits: LimitsSection::default(),
        }
    }
}

impl Default for ServerSection {
    fn default() -> Self {
        Self {
            wire_port: 7437,
            http_port: 7438,
            data_dir: PathBuf::from("./data"),
        }
    }
}

impl Default for EmbeddingSection {
    fn default() -> Self {
        Self {
            strategy: EmbeddingStrategy::Builtin,
            dim: 384,
        }
    }
}

impl Default for BackgroundSection {
    fn default() -> Self {
        Self {
            consolidation_interval_minutes: 30,
            decay_sweep_interval_minutes: 60,
        }
    }
}

impl Default for LimitsSection {
    fn default() -> Self {
        Self {
            max_databases: 100,
            max_connections: 1000,
        }
    }
}

impl ServerConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: ServerConfig = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn data_dir(&self) -> &Path {
        &self.server.data_dir
    }

    pub fn control_db_path(&self) -> PathBuf {
        self.server.data_dir.join("control.db")
    }
}
