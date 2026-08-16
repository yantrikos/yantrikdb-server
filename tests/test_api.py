"""Tests for the YantrikDB REST API."""

import math
import threading
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from yantrikdb import YantrikDB

DIM = 8


def _vec(seed: float) -> list[float]:
    raw = [(seed + i) * 0.1 for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class _MockEmbedder:
    def encode(self, text: str) -> list[float]:
        seed = float(hash(text) % 1000) / 100.0
        return _vec(seed)


@contextmanager
def _make_client():
    """Build a test client with in-memory YantrikDB, bypassing lifespan.

    Routes are copied from the real app, so the app-level auth dependency
    (baked into each APIRoute at registration) is exercised as in production.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI as _FastAPI

    import yantrikdb.api as api_mod

    # Create a test lifespan that creates the DB on the event loop thread
    @asynccontextmanager
    async def test_lifespan(app):
        db = YantrikDB(db_path=":memory:", embedding_dim=DIM)
        db.set_embedder(_MockEmbedder())
        lock = threading.Lock()
        app.state.db = db
        app.state.lock = lock
        original = api_mod._db
        api_mod._db = lambda: (db, lock)
        try:
            yield
        finally:
            api_mod._db = original
            db.close()

    test_app = _FastAPI(title="YantrikDB", version="0.1.0", lifespan=test_lifespan)
    for route in api_mod.app.routes:
        test_app.routes.append(route)

    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def client(monkeypatch):
    """Test client with no API key configured — all routes open (legacy behavior)."""
    monkeypatch.delenv("YANTRIKDB_API_KEY", raising=False)
    with _make_client() as c:
        yield c


API_KEY = "test-secret-key"


@pytest.fixture
def auth_client(monkeypatch):
    """Test client with YANTRIKDB_API_KEY configured — bearer auth enforced."""
    monkeypatch.setenv("YANTRIKDB_API_KEY", API_KEY)
    with _make_client() as c:
        yield c


class TestMemoryEndpoints:
    def test_record_memory(self, client):
        resp = client.post("/memories", json={"text": "hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert "rid" in data

    def test_get_memory(self, client):
        rid = client.post("/memories", json={"text": "test"}).json()["rid"]
        resp = client.get(f"/memories/{rid}")
        assert resp.status_code == 200
        assert resp.json()["rid"] == rid
        assert resp.json()["text"] == "test"
        assert "embedding" not in resp.json()

    def test_get_memory_not_found(self, client):
        resp = client.get("/memories/nonexistent")
        assert resp.status_code == 404

    def test_recall_memories(self, client):
        client.post("/memories", json={"text": "the sky is blue"})
        client.post("/memories", json={"text": "grass is green"})

        resp = client.post("/memories/recall", json={"query": "sky color"})
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "results" in data

    def test_recall_empty(self, client):
        resp = client.post("/memories/recall", json={"query": "anything"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_forget_memory(self, client):
        rid = client.post("/memories", json={"text": "forget me"}).json()["rid"]
        resp = client.delete(f"/memories/{rid}")
        assert resp.status_code == 200
        assert resp.json()["forgotten"] is True

        # Memory is tombstoned — get() still returns it but it won't appear in recall
        resp = client.get(f"/memories/{rid}")
        assert resp.status_code == 200
        assert resp.json()["consolidation_status"] == "tombstoned"

    def test_correct_memory(self, client):
        rid = client.post("/memories", json={"text": "wrong"}).json()["rid"]
        resp = client.post(f"/memories/{rid}/correct", json={
            "new_text": "correct",
            "correction_note": "fixed",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "new_rid" in data or "corrected_rid" in data

    def test_record_with_fields(self, client):
        resp = client.post("/memories", json={
            "text": "important fact",
            "memory_type": "semantic",
            "importance": 0.9,
            "valence": 0.5,
            "metadata": {"source": "test"},
        })
        assert resp.status_code == 200


class TestEntityEndpoints:
    def test_relate(self, client):
        resp = client.post("/entities/relate", json={
            "src": "Alice", "dst": "Bob", "rel_type": "knows",
        })
        assert resp.status_code == 200
        assert "edge_id" in resp.json()

    def test_get_edges(self, client):
        client.post("/entities/relate", json={
            "src": "Alice", "dst": "Bob",
        })
        resp = client.get("/entities/Alice/edges")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_get_edges_empty(self, client):
        resp = client.get("/entities/Unknown/edges")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


class TestCognitionEndpoints:
    def test_think(self, client):
        resp = client.post("/think")
        assert resp.status_code == 200
        data = resp.json()
        assert "consolidation_count" in data

    def test_conflicts_empty(self, client):
        resp = client.get("/conflicts")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_triggers_empty(self, client):
        resp = client.get("/triggers")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


class TestSystemEndpoints:
    def test_stats(self, client):
        client.post("/memories", json={"text": "test"})
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_memories"] == 1

    def test_export(self, client):
        client.post("/memories", json={"text": "exportable"})
        resp = client.get("/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "yantrikdb-export-v1"
        assert len(data["memories"]) == 1

    def test_openapi_schema(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "YantrikDB"
        assert "/memories" in schema["paths"]
        assert "/memories/recall" in schema["paths"]
        assert "/stats" in schema["paths"]

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAuth:
    """Bearer auth when YANTRIKDB_API_KEY is configured."""

    AUTH = {"Authorization": f"Bearer {API_KEY}"}

    def test_export_requires_auth(self, auth_client):
        resp = auth_client.get("/export")
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "Bearer"

    def test_wrong_key_rejected(self, auth_client):
        resp = auth_client.get("/export", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401

    def test_wrong_scheme_rejected(self, auth_client):
        resp = auth_client.get("/export", headers={"Authorization": f"Basic {API_KEY}"})
        assert resp.status_code == 401

    def test_right_key_accepted(self, auth_client):
        resp = auth_client.get("/export", headers=self.AUTH)
        assert resp.status_code == 200
        assert resp.json()["version"] == "yantrikdb-export-v1"

    def test_mutating_routes_gated(self, auth_client):
        assert auth_client.post("/memories", json={"text": "x"}).status_code == 401
        assert auth_client.delete("/memories/some-rid").status_code == 401
        assert auth_client.post(
            "/memories/some-rid/correct", json={"new_text": "y"}
        ).status_code == 401
        assert auth_client.post(
            "/entities/relate", json={"src": "A", "dst": "B"}
        ).status_code == 401
        assert auth_client.post("/think").status_code == 401

    def test_read_routes_gated(self, auth_client):
        assert auth_client.get("/stats").status_code == 401
        assert auth_client.post("/memories/recall", json={"query": "q"}).status_code == 401
        assert auth_client.get("/memories/some-rid").status_code == 401

    def test_authed_roundtrip(self, auth_client):
        rid = auth_client.post(
            "/memories", json={"text": "secret"}, headers=self.AUTH
        ).json()["rid"]
        resp = auth_client.get(f"/memories/{rid}", headers=self.AUTH)
        assert resp.status_code == 200
        assert resp.json()["text"] == "secret"
        resp = auth_client.delete(f"/memories/{rid}", headers=self.AUTH)
        assert resp.status_code == 200

    def test_health_stays_open(self, auth_client):
        resp = auth_client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_no_key_configured_stays_open(self, client):
        # Legacy behavior: no YANTRIKDB_API_KEY → no auth required.
        assert client.get("/export").status_code == 200
        assert client.post("/memories", json={"text": "x"}).status_code == 200


class TestBindGuard:
    """Startup refusal for non-loopback binds without an API key."""

    def _run_main(self, monkeypatch):
        import yantrikdb.api as api_mod

        calls = []
        monkeypatch.setattr(
            api_mod.uvicorn, "run", lambda *a, **kw: calls.append((a, kw))
        )
        api_mod.main()
        return calls

    def test_non_loopback_without_key_refused(self, monkeypatch):
        monkeypatch.setenv("YANTRIKDB_HOST", "0.0.0.0")
        monkeypatch.delenv("YANTRIKDB_API_KEY", raising=False)
        monkeypatch.delenv("YANTRIKDB_ALLOW_INSECURE", raising=False)
        with pytest.raises(SystemExit):
            self._run_main(monkeypatch)

    def test_non_loopback_hostname_without_key_refused(self, monkeypatch):
        monkeypatch.setenv("YANTRIKDB_HOST", "myserver.lan")
        monkeypatch.delenv("YANTRIKDB_API_KEY", raising=False)
        monkeypatch.delenv("YANTRIKDB_ALLOW_INSECURE", raising=False)
        with pytest.raises(SystemExit):
            self._run_main(monkeypatch)

    def test_non_loopback_with_key_starts(self, monkeypatch):
        monkeypatch.setenv("YANTRIKDB_HOST", "0.0.0.0")
        monkeypatch.setenv("YANTRIKDB_API_KEY", "some-key")
        assert len(self._run_main(monkeypatch)) == 1

    def test_loopback_without_key_starts(self, monkeypatch):
        monkeypatch.delenv("YANTRIKDB_HOST", raising=False)
        monkeypatch.delenv("YANTRIKDB_API_KEY", raising=False)
        calls = self._run_main(monkeypatch)
        assert len(calls) == 1
        assert calls[0][1]["host"] == "127.0.0.1"

    def test_localhost_and_ipv6_loopback_without_key_start(self, monkeypatch):
        monkeypatch.delenv("YANTRIKDB_API_KEY", raising=False)
        for host in ("localhost", "::1", "127.0.0.1"):
            monkeypatch.setenv("YANTRIKDB_HOST", host)
            assert len(self._run_main(monkeypatch)) == 1

    def test_allow_insecure_escape_hatch(self, monkeypatch, caplog):
        monkeypatch.setenv("YANTRIKDB_HOST", "0.0.0.0")
        monkeypatch.delenv("YANTRIKDB_API_KEY", raising=False)
        monkeypatch.setenv("YANTRIKDB_ALLOW_INSECURE", "1")
        with caplog.at_level("WARNING", logger="yantrikdb.api"):
            calls = self._run_main(monkeypatch)
        assert len(calls) == 1
        assert any("SECURITY WARNING" in r.message for r in caplog.records)
