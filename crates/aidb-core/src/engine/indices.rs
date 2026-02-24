use crate::error::Result;
use crate::hnsw::HnswIndex;
use crate::serde_helpers::deserialize_f32;

use super::AIDB;

impl AIDB {
    /// Build the HNSW vector index from active hot-tier embeddings in SQLite.
    pub(crate) fn build_vec_index(
        conn: &rusqlite::Connection,
        embedding_dim: usize,
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
            let embedding = deserialize_f32(&emb_blob);
            if embedding.len() == embedding_dim {
                index.insert(&rid, &embedding)?;
            }
        }
        Ok(index)
    }

    /// Rebuild the HNSW vector index from scratch. Called after replication.
    pub fn rebuild_vec_index(&self) -> Result<usize> {
        let new_index = Self::build_vec_index(&self.conn, self.embedding_dim)?;
        let count = new_index.len();
        *self.vec_index.borrow_mut() = new_index;
        Ok(count)
    }

    pub fn rebuild_graph_index(&self) -> Result<usize> {
        let new_index = crate::graph_index::GraphIndex::build_from_db(&self.conn)?;
        let count = new_index.entity_count();
        *self.graph_index.borrow_mut() = new_index;
        Ok(count)
    }
}
