use bytes::BytesMut;
use tokio_util::codec::{Decoder, Encoder};

use crate::error::ProtocolError;
use crate::frame::Frame;

/// Tokio codec for YantrikDB wire protocol frames.
///
/// Handles length-delimited framing with the Frame encode/decode logic.
#[derive(Debug, Default)]
pub struct YantrikCodec;

impl YantrikCodec {
    pub fn new() -> Self {
        Self
    }
}

impl Decoder for YantrikCodec {
    type Item = Frame;
    type Error = ProtocolError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        Frame::decode(src)
    }
}

impl Encoder<Frame> for YantrikCodec {
    type Error = ProtocolError;

    fn encode(&mut self, item: Frame, dst: &mut BytesMut) -> Result<(), Self::Error> {
        item.encode(dst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcodes::OpCode;
    use bytes::Bytes;

    #[test]
    fn codec_roundtrip() {
        let mut codec = YantrikCodec::new();
        let frame = Frame::new(OpCode::Recall, 99, Bytes::from("test query"));

        let mut buf = BytesMut::new();
        codec.encode(frame, &mut buf).unwrap();

        let decoded = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(decoded.opcode, OpCode::Recall);
        assert_eq!(decoded.stream_id, 99);
        assert_eq!(decoded.payload, Bytes::from("test query"));
    }

    #[test]
    fn codec_partial_then_complete() {
        let mut codec = YantrikCodec::new();
        let frame = Frame::new(OpCode::Remember, 1, Bytes::from("data"));

        let mut full = BytesMut::new();
        codec.encode(frame, &mut full).unwrap();

        // Split into two parts
        let mut part1 = full.split_to(5);

        // First decode: not enough data
        assert!(codec.decode(&mut part1).unwrap().is_none());

        // Append the rest
        part1.unsplit(full);

        // Now it should decode
        let decoded = codec.decode(&mut part1).unwrap().unwrap();
        assert_eq!(decoded.opcode, OpCode::Remember);
    }

    #[test]
    fn codec_multiple_frames() {
        let mut codec = YantrikCodec::new();
        let mut buf = BytesMut::new();

        let f1 = Frame::empty(OpCode::Ping, 0);
        let f2 = Frame::new(OpCode::Remember, 1, Bytes::from("hello"));
        let f3 = Frame::empty(OpCode::Pong, 0);

        codec.encode(f1, &mut buf).unwrap();
        codec.encode(f2, &mut buf).unwrap();
        codec.encode(f3, &mut buf).unwrap();

        let d1 = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(d1.opcode, OpCode::Ping);

        let d2 = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(d2.opcode, OpCode::Remember);
        assert_eq!(d2.payload, Bytes::from("hello"));

        let d3 = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(d3.opcode, OpCode::Pong);

        // Nothing left
        assert!(codec.decode(&mut buf).unwrap().is_none());
    }
}
