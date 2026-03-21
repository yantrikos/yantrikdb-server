use thiserror::Error;

#[derive(Error, Debug)]
pub enum YantrikDbError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("No embedder configured. Pass an embedder to YantrikDB() or call set_embedder().")]
    NoEmbedder,

    #[error("Must provide either query or query_embedding")]
    NoQuery,

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("memory not found: {0}")]
    NotFound(String),

    #[error("sync error: {0}")]
    SyncError(String),

    #[error("invalid HLC timestamp: {0}")]
    HlcParseError(String),

    #[error("encryption error: {0}")]
    Encryption(String),

    #[error("model loading error: {0}")]
    ModelLoad(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("session conflict: {0}")]
    SessionConflict(String),
}

pub type Result<T> = std::result::Result<T, YantrikDbError>;
