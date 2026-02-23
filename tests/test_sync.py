"""Tests for P2P sync transport and replication API."""

import json
import pytest
from aidb import AIDB


class MockEmbedder:
    """Simple deterministic embedder for testing."""
    def encode(self, text):
        # Simple hash-based embedding
        h = hash(text) & 0xFFFFFFFF
        return [(h >> i & 0xFF) / 255.0 for i in range(0, 32, 1)][:8]


@pytest.fixture
def embedder():
    return MockEmbedder()


@pytest.fixture
def db_a(embedder):
    db = AIDB(":memory:", 8, embedder)
    return db


@pytest.fixture
def db_b(embedder):
    db = AIDB(":memory:", 8, embedder)
    return db


class TestReplicationAPI:
    """Test the Python replication API bindings."""

    def test_extract_ops_empty(self, db_a):
        ops = db_a.extract_ops_since()
        assert isinstance(ops, list)
        assert len(ops) == 0

    def test_extract_ops_after_record(self, db_a):
        db_a.record("test memory", embedding=[1.0] * 8)
        ops = db_a.extract_ops_since()
        assert len(ops) >= 1
        record_op = next(o for o in ops if o["op_type"] == "record")
        assert record_op["payload"]["text"] == "test memory"

    def test_extract_ops_after_relate(self, db_a):
        db_a.relate("Alice", "Bob", "knows")
        ops = db_a.extract_ops_since()
        relate_ops = [o for o in ops if o["op_type"] == "relate"]
        assert len(relate_ops) == 1

    def test_apply_ops_round_trip(self, db_a, db_b):
        """Record in A, extract ops, apply to B, verify convergence."""
        rid = db_a.record("hello from A", embedding=[1.0] * 8)
        db_a.relate("Alice", "Bob", "knows")

        ops = db_a.extract_ops_since()
        assert len(ops) >= 2

        result = db_b.apply_ops(ops)
        assert result["ops_applied"] >= 2
        assert result["ops_skipped"] == 0

        # Verify B has the memory
        mem = db_b.get(rid)
        assert mem is not None
        assert mem["text"] == "hello from A"

        # Verify B has the edge
        edges = db_b.get_edges("Alice")
        assert any(e["dst"] == "Bob" for e in edges)

    def test_apply_ops_idempotent(self, db_a, db_b):
        """Applying the same ops twice should skip duplicates."""
        db_a.record("idempotent test", embedding=[1.0] * 8)
        ops = db_a.extract_ops_since()

        result1 = db_b.apply_ops(ops)
        assert result1["ops_applied"] >= 1

        result2 = db_b.apply_ops(ops)
        assert result2["ops_applied"] == 0
        assert result2["ops_skipped"] >= 1

    def test_extract_with_exclude_actor(self, db_a):
        db_a.record("local memory", embedding=[1.0] * 8)
        actor = db_a.actor_id

        ops_all = db_a.extract_ops_since()
        ops_filtered = db_a.extract_ops_since(exclude_actor=actor)

        assert len(ops_all) >= 1
        assert len(ops_filtered) == 0

    def test_watermark_tracking(self, db_a):
        """Test get/set peer watermark."""
        wm = db_a.get_peer_watermark("peer_1")
        assert wm is None

        db_a.set_peer_watermark("peer_1", b"\x00" * 16, "op-123")
        wm = db_a.get_peer_watermark("peer_1")
        assert wm is not None
        assert wm["op_id"] == "op-123"
        assert bytes(wm["hlc"]) == b"\x00" * 16

    def test_rebuild_vec_index(self, db_a):
        db_a.record("rebuild test", embedding=[1.0] * 8)
        count = db_a.rebuild_vec_index()
        assert count >= 1

    def test_rebuild_graph_index(self, db_a):
        db_a.relate("X", "Y", "linked")
        count = db_a.rebuild_graph_index()
        assert count >= 2  # at least X and Y


class TestSyncProtocol:
    """Test the sync protocol message handling."""

    def test_hello_message(self, db_a):
        from aidb.sync.transport import SyncProtocol

        proto = SyncProtocol(db_a)
        resp = json.loads(proto.handle_message(json.dumps({
            "id": 1,
            "method": "HELLO",
        })))
        assert resp["result"]["actor_id"] == db_a.actor_id

    def test_pull_ops_empty(self, db_a):
        from aidb.sync.transport import SyncProtocol

        proto = SyncProtocol(db_a)
        resp = json.loads(proto.handle_message(json.dumps({
            "id": 1,
            "method": "PULL_OPS",
            "params": {"limit": 10},
        })))
        assert resp["result"]["ops"] == []

    def test_pull_ops_with_data(self, db_a):
        from aidb.sync.transport import SyncProtocol

        db_a.record("sync test", embedding=[1.0] * 8)
        proto = SyncProtocol(db_a)

        resp = json.loads(proto.handle_message(json.dumps({
            "id": 1,
            "method": "PULL_OPS",
            "params": {"limit": 100},
        })))
        ops = resp["result"]["ops"]
        assert len(ops) >= 1
        # HLCs should be hex-encoded strings
        assert isinstance(ops[0]["hlc"], str)

    def test_push_ops(self, db_a, db_b):
        from aidb.sync.transport import SyncProtocol

        # Record in A, extract, serialize
        db_a.record("push test", embedding=[1.0] * 8)
        ops = db_a.extract_ops_since()
        serialized = []
        for op in ops:
            sop = dict(op)
            sop["hlc"] = bytes(sop["hlc"]).hex()
            if sop.get("embedding_hash"):
                sop["embedding_hash"] = bytes(sop["embedding_hash"]).hex()
            serialized.append(sop)

        # Push to B via protocol
        proto_b = SyncProtocol(db_b)
        resp = json.loads(proto_b.handle_message(json.dumps({
            "id": 1,
            "method": "PUSH_OPS",
            "params": {"ops": serialized},
        })))
        assert resp["result"]["ops_applied"] >= 1

    def test_ack_message(self, db_a):
        from aidb.sync.transport import SyncProtocol

        proto = SyncProtocol(db_a)
        resp = json.loads(proto.handle_message(json.dumps({
            "id": 1,
            "method": "ACK",
            "params": {
                "peer_actor": "remote_1",
                "hlc": "00" * 16,
                "op_id": "test-op-1",
            },
        })))
        assert resp["result"]["ok"] is True

        # Verify watermark was set
        wm = db_a.get_peer_watermark("remote_1")
        assert wm is not None
        assert wm["op_id"] == "test-op-1"

    def test_unknown_method(self, db_a):
        from aidb.sync.transport import SyncProtocol

        proto = SyncProtocol(db_a)
        resp = json.loads(proto.handle_message(json.dumps({
            "id": 1,
            "method": "INVALID",
        })))
        assert "error" in resp

    def test_invalid_json(self, db_a):
        from aidb.sync.transport import SyncProtocol

        proto = SyncProtocol(db_a)
        resp = json.loads(proto.handle_message("not json"))
        assert "error" in resp
