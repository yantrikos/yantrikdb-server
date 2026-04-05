pub mod codec;
pub mod error;
pub mod frame;
pub mod messages;
pub mod opcodes;

pub use codec::YantrikCodec;
pub use error::ProtocolError;
pub use frame::Frame;
pub use opcodes::OpCode;

/// Helper: serialize a message to MessagePack bytes.
pub fn pack<T: serde::Serialize>(msg: &T) -> Result<bytes::Bytes, ProtocolError> {
    let data = rmp_serde::to_vec_named(msg)?;
    Ok(bytes::Bytes::from(data))
}

/// Helper: deserialize a message from MessagePack bytes.
pub fn unpack<'de, T: serde::Deserialize<'de>>(data: &'de [u8]) -> Result<T, ProtocolError> {
    Ok(rmp_serde::from_slice(data)?)
}

/// Build a complete frame from an opcode, stream ID, and serializable payload.
pub fn make_frame<T: serde::Serialize>(
    opcode: OpCode,
    stream_id: u32,
    msg: &T,
) -> Result<Frame, ProtocolError> {
    let payload = pack(msg)?;
    Ok(Frame::new(opcode, stream_id, payload))
}

/// Build a frame with an error response.
pub fn make_error(
    stream_id: u32,
    code: u16,
    message: impl Into<String>,
) -> Result<Frame, ProtocolError> {
    make_frame(
        OpCode::Error,
        stream_id,
        &messages::ErrorResponse {
            code,
            message: message.into(),
            details: None,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use messages::RememberRequest;

    #[test]
    fn pack_unpack_roundtrip() {
        let req = RememberRequest {
            text: "Alice leads engineering".into(),
            memory_type: "semantic".into(),
            importance: 0.9,
            valence: 0.0,
            half_life: 168.0,
            metadata: serde_json::json!({}),
            namespace: "default".into(),
            certainty: 1.0,
            domain: "work".into(),
            source: "user".into(),
            emotional_state: None,
            embedding: None,
        };

        let packed = pack(&req).unwrap();
        let unpacked: RememberRequest = unpack(&packed).unwrap();

        assert_eq!(unpacked.text, "Alice leads engineering");
        assert_eq!(unpacked.importance, 0.9);
        assert_eq!(unpacked.domain, "work");
    }

    #[test]
    fn make_frame_roundtrip() {
        let req = messages::RecallRequest {
            query: "who leads engineering?".into(),
            top_k: 5,
            memory_type: None,
            include_consolidated: false,
            expand_entities: true,
            namespace: None,
            domain: None,
            source: None,
            query_embedding: None,
        };

        let frame = make_frame(OpCode::Recall, 7, &req).unwrap();
        assert_eq!(frame.opcode, OpCode::Recall);
        assert_eq!(frame.stream_id, 7);

        let decoded: messages::RecallRequest = unpack(&frame.payload).unwrap();
        assert_eq!(decoded.query, "who leads engineering?");
        assert_eq!(decoded.top_k, 5);
    }

    #[test]
    fn make_error_frame() {
        let frame = make_error(0, messages::error_codes::AUTH_REQUIRED, "not authenticated").unwrap();
        assert_eq!(frame.opcode, OpCode::Error);

        let err: messages::ErrorResponse = unpack(&frame.payload).unwrap();
        assert_eq!(err.code, 1000);
        assert_eq!(err.message, "not authenticated");
    }
}
