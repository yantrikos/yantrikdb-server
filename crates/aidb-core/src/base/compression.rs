/// Zstd compression/decompression for embedding BLOBs.
///
/// Cold-tier memories store compressed embeddings to save space.
/// The zstd magic bytes (0x28 0xB5 0x2F 0xFD) identify compressed blobs.

use crate::serde_helpers::{deserialize_f32, serialize_f32};

/// Compress a float32 embedding into a zstd-compressed blob.
pub fn compress_embedding(embedding: &[f32]) -> Vec<u8> {
    let raw = serialize_f32(embedding);
    zstd::encode_all(raw.as_slice(), 3).expect("zstd compression failed")
}

/// Decompress a zstd-compressed blob back into a float32 embedding.
pub fn decompress_embedding(blob: &[u8]) -> Vec<f32> {
    let raw = zstd::decode_all(blob).expect("zstd decompression failed");
    deserialize_f32(&raw)
}

/// Check whether a blob is zstd-compressed by inspecting the magic bytes.
pub fn is_compressed(blob: &[u8]) -> bool {
    blob.len() >= 4 && blob[0] == 0x28 && blob[1] == 0xB5 && blob[2] == 0x2F && blob[3] == 0xFD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let embedding = vec![1.0f32, -2.5, 0.0, 3.14159, 42.0, -0.001];
        let compressed = compress_embedding(&embedding);
        let decompressed = decompress_embedding(&compressed);
        assert_eq!(embedding, decompressed);
    }

    #[test]
    fn test_is_compressed() {
        let embedding = vec![1.0f32, 2.0, 3.0];
        let compressed = compress_embedding(&embedding);
        assert!(is_compressed(&compressed));

        let raw = serialize_f32(&embedding);
        assert!(!is_compressed(&raw));
    }

    #[test]
    fn test_empty_embedding() {
        let embedding: Vec<f32> = vec![];
        let compressed = compress_embedding(&embedding);
        let decompressed = decompress_embedding(&compressed);
        assert_eq!(embedding, decompressed);
    }

    #[test]
    fn test_large_embedding() {
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32) * 0.001).collect();
        let compressed = compress_embedding(&embedding);
        let decompressed = decompress_embedding(&compressed);
        assert_eq!(embedding, decompressed);
        // Compressed should be smaller than raw for large vectors
        let raw_size = embedding.len() * 4;
        assert!(compressed.len() < raw_size, "compressed {} should be < raw {}", compressed.len(), raw_size);
    }
}
