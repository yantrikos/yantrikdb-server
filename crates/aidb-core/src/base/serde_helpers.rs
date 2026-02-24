/// Serialize a Vec<f32> to bytes (little-endian f32 array) for sqlite-vec.
pub fn serialize_f32(vector: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(vector.len() * 4);
    for &v in vector {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Deserialize bytes to Vec<f32>.
pub fn deserialize_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let v = vec![1.0f32, -2.5, 0.0, 3.14159, f32::MAX, f32::MIN];
        let blob = serialize_f32(&v);
        let back = deserialize_f32(&blob);
        assert_eq!(v, back);
    }

    #[test]
    fn test_empty() {
        let v: Vec<f32> = vec![];
        let blob = serialize_f32(&v);
        assert!(blob.is_empty());
        let back = deserialize_f32(&blob);
        assert!(back.is_empty());
    }

    #[test]
    fn test_single_element() {
        let v = vec![42.0f32];
        let blob = serialize_f32(&v);
        assert_eq!(blob.len(), 4);
        let back = deserialize_f32(&blob);
        assert_eq!(v, back);
    }
}
