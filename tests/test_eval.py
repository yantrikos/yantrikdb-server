"""Tests for the evaluation harness — uses a mock embedder for speed."""

import math

import pytest

from aidb import AIDB
from aidb.eval.harness import evaluate
from aidb.eval.synthetic import GOLDEN_QUERIES, SESSIONS, load_sessions_into_db


DIM = 64


class MockEmbedder:
    """Deterministic embedder that hashes text into a unit vector.

    Not semantically meaningful, but tests the harness plumbing.
    """

    def encode(self, text: str) -> list[float]:
        # Simple hash-based pseudo-embedding
        raw = []
        for i in range(DIM):
            h = hash(text + str(i)) % 10000
            raw.append(h / 10000.0 - 0.5)
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]


@pytest.fixture
def loaded_db():
    embedder = MockEmbedder()
    db = AIDB(db_path=":memory:", embedding_dim=DIM, embedder=embedder)
    text_to_rid = load_sessions_into_db(db, embedder=embedder)
    yield db, text_to_rid, embedder
    db.close()


class TestSyntheticData:
    def test_all_sessions_loaded(self, loaded_db):
        db, text_to_rid, _ = loaded_db
        total_mems = sum(len(s["memories"]) for s in SESSIONS)
        assert db.stats()["active_memories"] == total_mems
        assert len(text_to_rid) == total_mems
        assert len(SESSIONS) == 8, "Expected 8 sessions"
        assert total_mems == 32, "Expected 32 memories across 8 sessions"

    def test_entities_created(self, loaded_db):
        db, _, _ = loaded_db
        assert db.stats()["entities"] > 0
        assert db.stats()["edges"] > 0

    def test_temporal_ordering(self, loaded_db):
        db, text_to_rid, _ = loaded_db
        # First session memory should be older than last session memory
        first_rid = text_to_rid[SESSIONS[0]["memories"][0]["text"]]
        last_rid = text_to_rid[SESSIONS[-1]["memories"][-1]["text"]]
        first = db.get(first_rid)
        last = db.get(last_rid)
        assert first["created_at"] < last["created_at"]


class TestHarness:
    def test_harness_runs(self, loaded_db):
        db, text_to_rid, embedder = loaded_db
        report = evaluate(db, text_to_rid, top_k=10, embedder=embedder)

        assert report.total_queries == len(GOLDEN_QUERIES)
        assert report.total_memories > 0
        assert report.elapsed_seconds >= 0
        assert 0.0 <= report.mean_recall_at_k <= 1.0
        assert 0.0 <= report.mean_precision_at_k <= 1.0
        assert 0.0 <= report.mean_reciprocal_rank <= 1.0

    def test_per_query_results(self, loaded_db):
        db, text_to_rid, embedder = loaded_db
        report = evaluate(db, text_to_rid, top_k=10, embedder=embedder)

        for qr in report.query_results:
            assert qr.query_id
            assert qr.expected_count > 0
            assert len(qr.hits) + len(qr.misses) == qr.expected_count
            assert 0.0 <= qr.recall_at_k <= 1.0
            assert 0.0 <= qr.precision_at_k <= 1.0

    def test_report_summary(self, loaded_db):
        db, text_to_rid, embedder = loaded_db
        report = evaluate(db, text_to_rid, top_k=10, embedder=embedder)
        summary = report.summary()

        assert "Evaluation Report" in summary
        assert "Mean Recall@K" in summary
        assert "Mean MRR" in summary

    def test_recall_by_tag(self, loaded_db):
        db, text_to_rid, embedder = loaded_db
        report = evaluate(db, text_to_rid, top_k=10, embedder=embedder)

        assert "semantic" in report.recall_by_tag
        # With 40 queries across 12 categories, we expect several tags
        expected_tags = {"semantic", "graph", "temporal", "valence", "conflict"}
        for tag in expected_tags:
            assert tag in report.recall_by_tag, f"Tag '{tag}' missing from recall_by_tag"

    def test_all_40_queries_evaluated(self, loaded_db):
        """Verify all 40 golden queries are evaluated across 12 categories."""
        db, text_to_rid, embedder = loaded_db
        report = evaluate(db, text_to_rid, top_k=10, embedder=embedder)

        query_ids = {qr.query_id for qr in report.query_results}
        assert report.total_queries == 40

        # Spot-check one query from each of the 12 categories
        assert "q01_framework" in query_ids           # Direct Semantic
        assert "q05_sarah_all" in query_ids            # Entity Person
        assert "q11_faiss_scann" in query_ids          # Entity Tech
        assert "q15_last_meeting" in query_ids         # Temporal
        assert "q19_frustrated" in query_ids           # Emotional
        assert "q22_deadline" in query_ids             # Conflict
        assert "q25_work_habits" in query_ids          # Procedural
        assert "q28_data_problems" in query_ids        # Multi-hop
        assert "q31_model_perf" in query_ids           # Performance
        assert "q34_team_changes" in query_ids         # Team
        assert "q37_project_status" in query_ids       # Broad
        assert "q39_deployment" in query_ids           # Deployment

    def test_metrics_mathematically_valid(self, loaded_db):
        """Verify all metrics are in valid ranges and consistent."""
        db, text_to_rid, embedder = loaded_db
        report = evaluate(db, text_to_rid, top_k=10, embedder=embedder)

        # Aggregate metrics are averages of per-query metrics
        if report.query_results:
            expected_recall = sum(qr.recall_at_k for qr in report.query_results) / len(report.query_results)
            assert abs(report.mean_recall_at_k - expected_recall) < 1e-10

            expected_precision = sum(qr.precision_at_k for qr in report.query_results) / len(report.query_results)
            assert abs(report.mean_precision_at_k - expected_precision) < 1e-10

            expected_mrr = sum(qr.reciprocal_rank for qr in report.query_results) / len(report.query_results)
            assert abs(report.mean_reciprocal_rank - expected_mrr) < 1e-10

        # Per-query invariants
        for qr in report.query_results:
            assert 0.0 <= qr.recall_at_k <= 1.0
            assert 0.0 <= qr.precision_at_k <= 1.0
            assert 0.0 <= qr.reciprocal_rank <= 1.0
            assert len(qr.hits) + len(qr.misses) == qr.expected_count
            assert len(qr.retrieved_texts) <= 10  # top_k=10
            assert len(qr.scores) == len(qr.retrieved_texts)

    def test_skip_reinforce_in_eval(self, loaded_db):
        """Running eval twice should produce identical results (skip_reinforce)."""
        db, text_to_rid, embedder = loaded_db
        r1 = evaluate(db, text_to_rid, top_k=10, embedder=embedder)
        r2 = evaluate(db, text_to_rid, top_k=10, embedder=embedder)

        assert abs(r1.mean_recall_at_k - r2.mean_recall_at_k) < 1e-10
        assert abs(r1.mean_precision_at_k - r2.mean_precision_at_k) < 1e-10
        assert abs(r1.mean_reciprocal_rank - r2.mean_reciprocal_rank) < 1e-10

        for qr1, qr2 in zip(r1.query_results, r2.query_results):
            assert qr1.query_id == qr2.query_id
            assert abs(qr1.recall_at_k - qr2.recall_at_k) < 1e-10

    def test_memory_entities_linked(self, loaded_db):
        """After loading sessions, memory_entities table should have links."""
        db, _, _ = loaded_db
        count = db._conn.execute(
            "SELECT COUNT(*) AS cnt FROM memory_entities"
        ).fetchone()["cnt"]
        assert count > 0, "memory_entities should be populated after session loading"

    def test_entity_types_classified(self, loaded_db):
        """After loading sessions, entities should have classified types."""
        db, _, _ = loaded_db
        # Sarah should be classified as person
        result = db._conn.execute(
            "SELECT entity_type FROM entities WHERE name = 'Sarah'"
        ).fetchone()
        assert result is not None
        assert result["entity_type"] == "person"

        # FAISS should be classified as tech
        result = db._conn.execute(
            "SELECT entity_type FROM entities WHERE name = 'FAISS'"
        ).fetchone()
        assert result is not None
        assert result["entity_type"] == "tech"
