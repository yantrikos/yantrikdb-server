"""Tests for the AIDB engine — record, recall, relate, decay, forget."""

import math
import time

import pytest

from aidb import AIDB


# ── Helpers ──────────────────────────────────────────────

DIM = 8  # tiny embeddings for fast tests


def _vec(seed: float) -> list[float]:
    """Generate a deterministic unit-ish vector from a seed."""
    raw = [(seed + i) * 0.1 for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def db():
    """In-memory AIDB with no embedder (pre-computed vectors only)."""
    engine = AIDB(db_path=":memory:", embedding_dim=DIM)
    yield engine
    engine.close()


# ── record() ────────────────────────────────────────────

class TestRecord:
    def test_record_returns_rid(self, db):
        rid = db.record("hello world", embedding=_vec(1.0))
        assert isinstance(rid, str)
        assert len(rid) == 36  # UUIDv7 format

    def test_record_stores_memory(self, db):
        rid = db.record(
            "test memory",
            memory_type="semantic",
            importance=0.8,
            valence=-0.3,
            embedding=_vec(2.0),
        )
        mem = db.get(rid)
        assert mem is not None
        assert mem["text"] == "test memory"
        assert mem["type"] == "semantic"
        assert mem["importance"] == 0.8
        assert mem["valence"] == -0.3
        assert mem["consolidation_status"] == "active"

    def test_record_with_metadata(self, db):
        rid = db.record(
            "with meta",
            metadata={"source": "test", "tags": ["a", "b"]},
            embedding=_vec(3.0),
        )
        mem = db.get(rid)
        assert mem["metadata"]["source"] == "test"
        assert mem["metadata"]["tags"] == ["a", "b"]

    def test_record_without_embedding_raises(self, db):
        with pytest.raises(RuntimeError, match="No embedder configured"):
            db.record("no embedding")

    def test_record_updates_stats(self, db):
        assert db.stats()["active_memories"] == 0
        db.record("one", embedding=_vec(1.0))
        db.record("two", embedding=_vec(2.0))
        assert db.stats()["active_memories"] == 2


# ── recall() ────────────────────────────────────────────

class TestRecall:
    def test_recall_basic(self, db):
        db.record("the cat sat on the mat", embedding=_vec(1.0))
        db.record("dogs are loyal friends", embedding=_vec(5.0))
        db.record("cats love warm places", embedding=_vec(1.1))

        results = db.recall(query_embedding=_vec(1.0), top_k=2)
        assert len(results) == 2
        # Most similar to _vec(1.0) should be first
        assert "cat" in results[0]["text"]

    def test_recall_returns_scores(self, db):
        db.record("memory one", embedding=_vec(1.0))
        results = db.recall(query_embedding=_vec(1.0), top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "score" in r
        assert "scores" in r
        assert "similarity" in r["scores"]
        assert "decay" in r["scores"]
        assert "recency" in r["scores"]
        assert "why_retrieved" in r

    def test_recall_respects_top_k(self, db):
        for i in range(20):
            db.record(f"memory {i}", embedding=_vec(float(i)))
        results = db.recall(query_embedding=_vec(0.0), top_k=5)
        assert len(results) == 5

    def test_recall_filters_by_type(self, db):
        db.record("episodic mem", memory_type="episodic", embedding=_vec(1.0))
        db.record("semantic mem", memory_type="semantic", embedding=_vec(1.1))

        results = db.recall(
            query_embedding=_vec(1.0), top_k=10, memory_type="semantic"
        )
        assert all(r["type"] == "semantic" for r in results)

    def test_recall_reinforces_memories(self, db):
        rid = db.record("reinforce me", embedding=_vec(1.0), half_life=1000.0)
        original = db.get(rid)

        db.recall(query_embedding=_vec(1.0), top_k=1)
        after = db.get(rid)

        # half_life should increase by 20%
        assert after["half_life"] > original["half_life"]
        assert after["last_access"] >= original["last_access"]

    def test_recall_empty_db(self, db):
        results = db.recall(query_embedding=_vec(1.0), top_k=5)
        assert results == []

    def test_recall_requires_query(self, db):
        with pytest.raises(ValueError, match="Must provide"):
            db.recall(top_k=5)


# ── relate() ────────────────────────────────────────────

class TestRelate:
    def test_relate_creates_edge(self, db):
        edge_id = db.relate("Alice", "Bob", rel_type="knows")
        assert isinstance(edge_id, str)

        edges = db.get_edges("Alice")
        assert len(edges) == 1
        assert edges[0]["src"] == "Alice"
        assert edges[0]["dst"] == "Bob"
        assert edges[0]["rel_type"] == "knows"

    def test_relate_bidirectional_lookup(self, db):
        db.relate("Alice", "Bob", rel_type="knows")
        assert len(db.get_edges("Alice")) == 1
        assert len(db.get_edges("Bob")) == 1

    def test_relate_upserts_weight(self, db):
        db.relate("A", "B", rel_type="x", weight=0.5)
        db.relate("A", "B", rel_type="x", weight=0.9)

        edges = db.get_edges("A")
        assert len(edges) == 1
        assert edges[0]["weight"] == 0.9

    def test_relate_updates_stats(self, db):
        assert db.stats()["edges"] == 0
        db.relate("X", "Y")
        assert db.stats()["edges"] == 1
        assert db.stats()["entities"] == 2


# ── decay() ─────────────────────────────────────────────

class TestDecay:
    def test_decay_finds_old_memories(self, db):
        rid = db.record("old memory", importance=0.1, half_life=1.0, embedding=_vec(1.0))
        # Manually backdate last_access to simulate time passing
        db._conn.execute(
            "UPDATE memories SET last_access = ? WHERE rid = ?",
            (time.time() - 100, rid),
        )
        db._conn.commit()

        decayed = db.decay(threshold=0.01)
        assert len(decayed) >= 1
        assert any(d["rid"] == rid for d in decayed)

    def test_decay_skips_fresh_memories(self, db):
        db.record("fresh", importance=0.9, half_life=604800.0, embedding=_vec(1.0))
        decayed = db.decay(threshold=0.01)
        assert len(decayed) == 0

    def test_decay_returns_score_info(self, db):
        rid = db.record("decaying", importance=0.5, half_life=1.0, embedding=_vec(1.0))
        db._conn.execute(
            "UPDATE memories SET last_access = ? WHERE rid = ?",
            (time.time() - 50, rid),
        )
        db._conn.commit()

        decayed = db.decay(threshold=1.0)  # high threshold catches everything
        assert len(decayed) >= 1
        d = decayed[0]
        assert "current_score" in d
        assert "days_since_access" in d
        assert "original_importance" in d


# ── forget() ────────────────────────────────────────────

class TestForget:
    def test_forget_tombstones_memory(self, db):
        rid = db.record("forget me", embedding=_vec(1.0))
        assert db.forget(rid) is True

        mem = db.get(rid)
        assert mem["consolidation_status"] == "tombstoned"

    def test_forget_removes_from_vector_index(self, db):
        rid = db.record("forget vec", embedding=_vec(1.0))
        db.forget(rid)

        # Should not appear in recall results
        results = db.recall(query_embedding=_vec(1.0), top_k=10)
        assert all(r["rid"] != rid for r in results)

    def test_forget_nonexistent_returns_false(self, db):
        assert db.forget("nonexistent-rid") is False

    def test_forget_updates_stats(self, db):
        rid = db.record("bye", embedding=_vec(1.0))
        assert db.stats()["active_memories"] == 1
        db.forget(rid)
        assert db.stats()["active_memories"] == 0
        assert db.stats()["tombstoned_memories"] == 1


# ── stats() ──────────────────────────────────────────────

class TestStats:
    def test_stats_all_fields(self, db):
        s = db.stats()
        expected_keys = {
            "active_memories", "consolidated_memories", "tombstoned_memories",
            "archived_memories", "edges", "entities", "operations",
            "open_conflicts", "resolved_conflicts",
            "pending_triggers", "active_patterns",
            "scoring_cache_entries", "vec_index_entries",
            "graph_index_entities", "graph_index_edges",
        }
        assert set(s.keys()) == expected_keys

    def test_stats_tracks_operations(self, db):
        db.record("op1", embedding=_vec(1.0))
        db.relate("A", "B")
        s = db.stats()
        assert s["operations"] == 2  # 1 record + 1 relate


# ── Integration ──────────────────────────────────────────

class TestIntegration:
    def test_full_lifecycle(self, db):
        """Record -> recall -> relate -> decay -> forget lifecycle."""
        # Record
        rid1 = db.record("Python is great", importance=0.9, embedding=_vec(1.0))
        rid2 = db.record("Rust is fast", importance=0.7, embedding=_vec(5.0))
        rid3 = db.record("Python typing is improving", importance=0.5, embedding=_vec(1.2))

        # Recall — should find Python memories
        results = db.recall(query_embedding=_vec(1.0), top_k=2)
        assert len(results) == 2

        # Relate
        db.relate("Python", "typing", rel_type="has_feature")
        db.relate("Python", "Rust", rel_type="compared_with")

        assert db.stats()["edges"] == 2
        assert db.stats()["entities"] == 3

        # Decay — nothing should be decayed yet
        decayed = db.decay(threshold=0.01)
        assert len(decayed) == 0

        # Forget
        db.forget(rid2)
        assert db.stats()["active_memories"] == 2
        assert db.stats()["tombstoned_memories"] == 1

        # Verify forgotten memory not in recall
        results = db.recall(query_embedding=_vec(5.0), top_k=10)
        assert all(r["rid"] != rid2 for r in results)

    def test_valence_affects_ranking(self, db):
        """High-valence memories should rank higher when similarity is close."""
        db.record("neutral memory", valence=0.0, importance=0.5, embedding=_vec(1.0))
        db.record("emotional memory", valence=0.9, importance=0.5, embedding=_vec(1.01))

        results = db.recall(query_embedding=_vec(1.005), top_k=2)
        # The emotional memory should get a valence boost
        emotional = [r for r in results if "emotional" in r["text"]]
        assert len(emotional) == 1
        assert emotional[0]["score"] > results[-1]["score"]


# ── Graph-augmented recall integration tests ──────────

class TestGraphRecall:
    def test_recall_deterministic_with_skip_reinforce(self, db):
        """Same query with skip_reinforce=True returns identical results every time."""
        for i in range(10):
            db.record(f"memory {i}", embedding=_vec(float(i)))
        query = _vec(3.0)

        r1 = db.recall(query_embedding=query, top_k=5, skip_reinforce=True)
        r2 = db.recall(query_embedding=query, top_k=5, skip_reinforce=True)
        r3 = db.recall(query_embedding=query, top_k=5, skip_reinforce=True)

        rids1 = [r["rid"] for r in r1]
        rids2 = [r["rid"] for r in r2]
        rids3 = [r["rid"] for r in r3]
        assert rids1 == rids2 == rids3

    def test_skip_reinforce_prevents_mutation(self, db):
        """skip_reinforce=True should not modify half_life."""
        rid = db.record("test", embedding=_vec(1.0), half_life=1000.0)
        original = db.get(rid)

        db.recall(query_embedding=_vec(1.0), top_k=1, skip_reinforce=True)
        after = db.get(rid)
        assert after["half_life"] == original["half_life"]

    def test_graph_expansion_toggle(self, db):
        """expand_entities=True should set graph_proximity on connected memories."""
        r1 = db.record("Alice discussed the plan", embedding=_vec(1.0))
        r2 = db.record("Bob reviewed the code", embedding=_vec(5.0))
        db.relate("Alice", "Bob", rel_type="knows")
        db.link_memory_entity(r1, "Alice")
        db.link_memory_entity(r2, "Bob")

        # With expansion off
        results_off = db.recall(
            query="What is Alice working on?",
            query_embedding=_vec(1.0), top_k=10,
            expand_entities=False, skip_reinforce=True,
        )
        for r in results_off:
            assert r["scores"]["graph_proximity"] == 0.0

        # With expansion on
        results_on = db.recall(
            query="What is Alice working on?",
            query_embedding=_vec(1.0), top_k=10,
            expand_entities=True, skip_reinforce=True,
        )
        alice_result = [r for r in results_on if r["rid"] == r1]
        assert len(alice_result) == 1
        assert alice_result[0]["scores"]["graph_proximity"] > 0.0

    def test_entity_type_stored_after_relate(self, db):
        """relate() should classify and store entity_type correctly."""
        db.relate("Sarah", "data pipeline", rel_type="leads")
        db.relate("FAISS", "recommendation engine", rel_type="used_in")

        # Check entity types via internal DB access (rows are dicts)
        sarah_type = db._conn.execute(
            "SELECT entity_type FROM entities WHERE name = 'Sarah'"
        ).fetchone()["entity_type"]
        assert sarah_type == "person"

        faiss_type = db._conn.execute(
            "SELECT entity_type FROM entities WHERE name = 'FAISS'"
        ).fetchone()["entity_type"]
        assert faiss_type == "tech"

        pipeline_type = db._conn.execute(
            "SELECT entity_type FROM entities WHERE name = 'data pipeline'"
        ).fetchone()["entity_type"]
        assert pipeline_type == "unknown"

    def test_link_memory_entity_idempotent(self, db):
        """Linking same entity twice should not error or create duplicates."""
        rid = db.record("test", embedding=_vec(1.0))
        db.relate("Alice", "Bob", rel_type="knows")
        db.link_memory_entity(rid, "Alice")
        db.link_memory_entity(rid, "Alice")  # duplicate

        count = db._conn.execute(
            "SELECT COUNT(*) FROM memory_entities WHERE memory_rid = ? AND entity_name = 'Alice'",
            (rid,),
        ).fetchone()["COUNT(*)"]
        assert count == 1

    def test_recall_scores_non_negative(self, db):
        """All scores should be non-negative."""
        for i in range(10):
            db.record(
                f"memory {i}",
                importance=i * 0.1,
                valence=(i - 5) * 0.2,
                embedding=_vec(float(i)),
            )

        results = db.recall(query_embedding=_vec(5.0), top_k=10, skip_reinforce=True)
        for r in results:
            assert r["score"] >= 0.0, f"score should be non-negative, got {r['score']}"
            assert r["scores"]["similarity"] >= -1.0
            assert r["scores"]["decay"] >= 0.0
            assert r["scores"]["recency"] >= 0.0

    def test_recall_top_k_respected_with_graph(self, db):
        """top_k must be respected even when graph expansion adds candidates."""
        for i in range(15):
            rid = db.record(f"memory about topic {i}", embedding=_vec(float(i)))
            entity = f"Entity{i}"
            db.relate(entity, f"Entity{(i + 1) % 15}", rel_type="related_to")
            db.link_memory_entity(rid, entity)

        results = db.recall(
            query="Entity0 topic",
            query_embedding=_vec(0.0),
            top_k=5,
            expand_entities=True,
            skip_reinforce=True,
        )
        assert len(results) <= 5

    def test_backfill_memory_entities(self, db):
        """backfill_memory_entities should link memories to entities."""
        db.relate("Alice", "Bob", rel_type="knows")
        r1 = db.record("Alice discussed the plan", embedding=_vec(1.0))
        r2 = db.record("Bob reviewed the code", embedding=_vec(2.0))

        count = db.backfill_memory_entities()
        assert count > 0

        # Check links were created (rows are dicts)
        linked = db._conn.execute(
            "SELECT entity_name FROM memory_entities WHERE memory_rid = ?", (r1,)
        ).fetchall()
        entity_names = [row["entity_name"] for row in linked]
        assert "Alice" in entity_names


class TestStorageTier:
    def test_archive_hydrate_cycle(self, db):
        """Archive a memory to cold, verify invisible to recall, hydrate back."""
        emb = _vec(1.0)
        rid = db.record("archivable", embedding=emb)
        assert db.get(rid)["storage_tier"] == "hot"

        # Archive
        assert db.archive(rid) is True
        assert db.get(rid)["storage_tier"] == "cold"
        assert db.stats()["archived_memories"] == 1

        # Should not appear in recall
        results = db.recall(query_embedding=emb, top_k=10, skip_reinforce=True)
        assert all(r["rid"] != rid for r in results)

        # Hydrate back
        assert db.hydrate(rid) is True
        assert db.get(rid)["storage_tier"] == "hot"
        assert db.stats()["archived_memories"] == 0

        # Should appear in recall again
        results = db.recall(query_embedding=emb, top_k=10, skip_reinforce=True)
        assert any(r["rid"] == rid for r in results)

    def test_evict(self, db):
        """Evict memories to keep max_active, verify stats and recall."""
        for i in range(15):
            db.record(f"evict mem {i}", embedding=_vec(float(i)))

        assert db.stats()["active_memories"] == 15
        archived = db.evict(max_active=10)
        assert len(archived) == 5
        assert db.stats()["archived_memories"] == 5

        # Archived should not appear in recall
        results = db.recall(query_embedding=_vec(0.0), top_k=20, skip_reinforce=True)
        for r in results:
            assert r["rid"] not in archived

    def test_record_batch(self, db):
        """Batch record multiple memories at once."""
        inputs = [
            {"text": f"batch {i}", "embedding": _vec(float(i))}
            for i in range(10)
        ]
        rids = db.record_batch(inputs)
        assert len(rids) == 10
        assert db.stats()["active_memories"] == 10

        # All retrievable
        for rid in rids:
            mem = db.get(rid)
            assert mem is not None
            assert mem["storage_tier"] == "hot"
