use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("unknown opcode: 0x{0:02X}")]
    UnknownOpCode(u8),

    #[error("frame body too small: {0} bytes (minimum 6)")]
    FrameTooSmall(usize),

    #[error("frame body too large: {0} bytes")]
    FrameTooLarge(usize),

    #[error("payload serialization error: {0}")]
    Serialize(#[from] rmp_serde::encode::Error),

    #[error("payload deserialization error: {0}")]
    Deserialize(#[from] rmp_serde::decode::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
