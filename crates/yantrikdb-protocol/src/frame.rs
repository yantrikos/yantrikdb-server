use bytes::{Buf, BufMut, Bytes, BytesMut};

use crate::error::ProtocolError;
use crate::opcodes::OpCode;

/// Protocol version. Bit 7 = JSON payload mode (for debugging).
pub const PROTOCOL_VERSION: u8 = 0x01;
pub const JSON_MODE_FLAG: u8 = 0x80;

/// Minimum frame header size: 4 (length) + 1 (version) + 1 (opcode) + 4 (stream_id) = 10
pub const HEADER_SIZE: usize = 10;

/// Maximum payload size: 16 MiB
pub const MAX_PAYLOAD_SIZE: usize = 16 * 1024 * 1024;

/// A wire protocol frame.
///
/// ```text
/// ┌──────────┬──────────┬──────────┬──────────┬────────────┐
/// │ Length    │ Version  │ OpCode   │ StreamID │ Payload    │
/// │ (4 bytes)│ (1 byte) │ (1 byte) │ (4 bytes)│ (variable) │
/// └──────────┴──────────┴──────────┴──────────┴────────────┘
/// ```
///
/// Length covers everything after itself (version + opcode + stream_id + payload).
#[derive(Debug, Clone)]
pub struct Frame {
    pub version: u8,
    pub opcode: OpCode,
    pub stream_id: u32,
    pub payload: Bytes,
}

impl Frame {
    pub fn new(opcode: OpCode, stream_id: u32, payload: Bytes) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            opcode,
            stream_id,
            payload,
        }
    }

    /// Shorthand for a frame with no payload (e.g. PING/PONG).
    pub fn empty(opcode: OpCode, stream_id: u32) -> Self {
        Self::new(opcode, stream_id, Bytes::new())
    }

    /// Whether payloads should be JSON (debug mode).
    pub fn is_json_mode(&self) -> bool {
        self.version & JSON_MODE_FLAG != 0
    }

    /// Total wire size of this frame (including the 4-byte length prefix).
    pub fn wire_size(&self) -> usize {
        4 + 1 + 1 + 4 + self.payload.len()
    }

    /// Encode this frame into a byte buffer.
    pub fn encode(&self, dst: &mut BytesMut) {
        // Length = version(1) + opcode(1) + stream_id(4) + payload
        let body_len = 1 + 1 + 4 + self.payload.len();
        dst.reserve(4 + body_len);
        dst.put_u32(body_len as u32);
        dst.put_u8(self.version);
        dst.put_u8(self.opcode as u8);
        dst.put_u32(self.stream_id);
        dst.put_slice(&self.payload);
    }

    /// Try to decode a frame from a byte buffer.
    /// Returns `None` if not enough data yet, `Err` on malformed data.
    pub fn decode(src: &mut BytesMut) -> Result<Option<Self>, ProtocolError> {
        // Need at least 4 bytes for the length prefix
        if src.len() < 4 {
            return Ok(None);
        }

        // Peek at the length without advancing
        let body_len = u32::from_be_bytes([src[0], src[1], src[2], src[3]]) as usize;

        if body_len < 6 {
            return Err(ProtocolError::FrameTooSmall(body_len));
        }

        if body_len > MAX_PAYLOAD_SIZE + 6 {
            return Err(ProtocolError::FrameTooLarge(body_len));
        }

        // Do we have the full frame?
        let total = 4 + body_len;
        if src.len() < total {
            // Reserve space for the rest
            src.reserve(total - src.len());
            return Ok(None);
        }

        // Consume the frame
        let mut frame_buf = src.split_to(total);
        frame_buf.advance(4); // skip length prefix

        let version = frame_buf.get_u8();
        let opcode_byte = frame_buf.get_u8();
        let stream_id = frame_buf.get_u32();
        let payload = frame_buf.freeze();

        let opcode =
            OpCode::from_u8(opcode_byte).ok_or(ProtocolError::UnknownOpCode(opcode_byte))?;

        Ok(Some(Frame {
            version,
            opcode,
            stream_id,
            payload,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let frame = Frame::new(OpCode::Remember, 42, Bytes::from_static(b"hello world"));

        let mut buf = BytesMut::new();
        frame.encode(&mut buf);

        assert_eq!(buf.len(), frame.wire_size());

        let decoded = Frame::decode(&mut buf).unwrap().unwrap();
        assert_eq!(decoded.opcode, OpCode::Remember);
        assert_eq!(decoded.stream_id, 42);
        assert_eq!(decoded.payload, Bytes::from_static(b"hello world"));
        assert_eq!(decoded.version, PROTOCOL_VERSION);
    }

    #[test]
    fn decode_partial() {
        let frame = Frame::empty(OpCode::Ping, 0);
        let mut buf = BytesMut::new();
        frame.encode(&mut buf);

        // Feed only half the bytes
        let half = buf.len() / 2;
        let mut partial = buf.split_to(half);
        assert!(Frame::decode(&mut partial).unwrap().is_none());
    }

    #[test]
    fn decode_unknown_opcode() {
        let mut buf = BytesMut::new();
        buf.put_u32(6); // length: version(1) + opcode(1) + stream_id(4)
        buf.put_u8(PROTOCOL_VERSION);
        buf.put_u8(0xFF); // invalid opcode
        buf.put_u32(0);

        let err = Frame::decode(&mut buf).unwrap_err();
        assert!(matches!(err, ProtocolError::UnknownOpCode(0xFF)));
    }

    #[test]
    fn empty_frame() {
        let frame = Frame::empty(OpCode::Pong, 7);
        let mut buf = BytesMut::new();
        frame.encode(&mut buf);

        let decoded = Frame::decode(&mut buf).unwrap().unwrap();
        assert_eq!(decoded.opcode, OpCode::Pong);
        assert_eq!(decoded.stream_id, 7);
        assert!(decoded.payload.is_empty());
    }
}
