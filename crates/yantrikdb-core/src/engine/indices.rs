use crate::encryption::EncryptionProvider;
use crate::error::Result;
use crate::hnsw::HnswIndex;
use crate::serde_helpers::deserialize_f32;

use super::YantrikDB;

impl YantrikDB {
    /// Build the HNSW vector index, optionally decrypting embeddings.
    pub(crate) fn build_vec_index_with_enc(
        conn: &rusqlite::Connection,
        embedding_dim: usize,
        enc: Option<&EncryptionProvider>,
    ) -> Result<HnswIndex> {
        let mut index = HnswIndex::new(embedding_dim);
        let mut stmt = conn.prepare(
            "SELECT rid, embedding FROM memories \
             WHERE consolidation_status IN ('active', 'consolidated') \
             AND storage_tier = 'hot' \
             AND embedding IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |row| {
            let rid: String = row.get(0)?;
            let emb_blob: Vec<u8> = row.get(1)?;
            Ok((rid, emb_blob))
        })?;
        for row in rows {
            let (rid, emb_blob) = row?;
            let raw_blob = if let Some(e) = enc {
                e.decrypt_bytes(&emb_blob)?
            } else {
                emb_blob
            };
            let embedding = deserialize_f32(&raw_blob);
            if embedding.len() == embedding_dim {
                index.insert(&rid, &embedding)?;
            }
        }
        Ok(index)
    }

    /// Build without encryption (backward-compatible helper).
    pub(crate) fn build_vec_index(
        conn: &rusqlite::Connection,
        embedding_dim: usize,
    ) -> Result<HnswIndex> {
        Self::build_vec_index_with_enc(conn, embedding_dim, None)
    }

    /// Rebuild the HNSW vector index from scratch. Called after replication.
    pub fn rebuild_vec_index(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let new_index = Self::build_vec_index_with_enc(&conn, self.embedding_dim, self.enc.as_ref())?;
        let count = new_index.len();
        drop(conn);
        *self.vec_index.write().unwrap() = new_index;
        Ok(count)
    }

    pub fn rebuild_graph_index(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let new_index = crate::graph_index::GraphIndex::build_from_db(&conn)?;
        let count = new_index.entity_count();
        drop(conn);
        *self.graph_index.write().unwrap() = new_index;
        Ok(count)
    }
}
