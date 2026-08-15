"""YantrikDB REST API — FastAPI server with OpenAPI docs."""

import hmac
import ipaddress
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

log = logging.getLogger("yantrikdb.api")


# ── Auth ──

# Paths that stay open even when an API key is configured. /health is a bare
# liveness probe (no DB access); everything else — including /export and all
# mutating routes — requires `Authorization: Bearer <YANTRIKDB_API_KEY>`.
_OPEN_PATHS = frozenset({"/health"})


def _configured_api_key() -> str | None:
    """Return the configured API key, or None when auth is disabled.

    Read at request time (not import time) so the key can be set/rotated
    per-process and so tests can monkeypatch the environment.
    """
    key = os.environ.get("YANTRIKDB_API_KEY", "").strip()
    return key or None


async def require_api_key(request: Request) -> None:
    """App-level dependency: enforce bearer auth when YANTRIKDB_API_KEY is set.

    No key configured → all routes stay open (backwards compatible for
    loopback-only local dev; non-loopback binds without a key are refused at
    startup by _check_bind_security).
    """
    expected = _configured_api_key()
    if expected is None:
        return
    if request.url.path in _OPEN_PATHS:
        return
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Pydantic Models ──


class RecordRequest(BaseModel):
    text: str
    memory_type: str = "episodic"
    importance: float = 0.5
    valence: float = 0.0
    metadata: dict | None = None


class RecordResponse(BaseModel):
    rid: str


class RecallRequest(BaseModel):
    query: str
    top_k: int = 10
    memory_type: str | None = None
    include_consolidated: bool = False
    expand_entities: bool = True


class CorrectRequest(BaseModel):
    new_text: str
    new_importance: float | None = None
    new_valence: float | None = None
    correction_note: str | None = None


class RelateRequest(BaseModel):
    src: str
    dst: str
    rel_type: str = "related_to"
    weight: float = 1.0


class RelateResponse(BaseModel):
    edge_id: str


class ThinkRequest(BaseModel):
    run_consolidation: bool = True
    run_conflict_scan: bool = True
    run_pattern_mining: bool = True


class ResolveRequest(BaseModel):
    strategy: str
    winner_rid: str | None = None
    new_text: str | None = None
    resolution_note: str | None = None


# ── App Setup ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize YantrikDB + embedder on startup."""
    from sentence_transformers import SentenceTransformer

    from yantrikdb import YantrikDB

    db_path = os.environ.get("YANTRIKDB_DB_PATH", str(Path.home() / ".yantrikdb" / "memory.db"))
    model_name = os.environ.get("YANTRIKDB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dim = int(os.environ.get("YANTRIKDB_EMBEDDING_DIM", "384"))

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading embedding model: %s", model_name)
    embedder = SentenceTransformer(model_name)

    log.info("Opening YantrikDB at: %s (dim=%d)", db_path, embedding_dim)
    db = YantrikDB(db_path=db_path, embedding_dim=embedding_dim, embedder=embedder)
    lock = threading.Lock()

    app.state.db = db
    app.state.lock = lock

    try:
        yield
    finally:
        log.info("Shutting down YantrikDB")
        db.close()


app = FastAPI(
    title="YantrikDB",
    description="A Cognitive Memory Engine for Persistent AI Systems",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)


def _db():
    return app.state.db, app.state.lock


# ── Health ──


@app.get("/health", tags=["system"])
async def health():
    """Bare liveness probe — always open, touches no data."""
    return {"status": "ok"}


# ── Memory Endpoints ──


@app.post("/memories", response_model=RecordResponse, tags=["memories"])
async def record_memory(req: RecordRequest):
    """Store a new memory."""
    db, lock = _db()
    with lock:
        rid = db.record(
            req.text,
            memory_type=req.memory_type,
            importance=req.importance,
            valence=req.valence,
            metadata=req.metadata or {},
        )
    return RecordResponse(rid=rid)


@app.post("/memories/recall", tags=["memories"])
async def recall_memories(req: RecallRequest):
    """Search memories by semantic similarity."""
    db, lock = _db()
    with lock:
        results = db.recall(
            query=req.query,
            top_k=req.top_k,
            memory_type=req.memory_type,
            include_consolidated=req.include_consolidated,
            expand_entities=req.expand_entities,
            skip_reinforce=True,
        )
    # Strip embeddings from response
    for r in results:
        r.pop("embedding", None)
    return {"count": len(results), "results": results}


@app.get("/memories/{rid}", tags=["memories"])
async def get_memory(rid: str):
    """Get a single memory by RID."""
    db, lock = _db()
    with lock:
        mem = db.get(rid)
    if mem is None:
        raise HTTPException(status_code=404, detail=f"Memory {rid} not found")
    mem.pop("embedding", None)
    return mem


@app.delete("/memories/{rid}", tags=["memories"])
async def forget_memory(rid: str):
    """Tombstone a memory (soft delete)."""
    db, lock = _db()
    with lock:
        found = db.forget(rid)
    if not found:
        raise HTTPException(status_code=404, detail=f"Memory {rid} not found")
    return {"rid": rid, "forgotten": True}


@app.post("/memories/{rid}/correct", tags=["memories"])
async def correct_memory(rid: str, req: CorrectRequest):
    """Correct a memory — tombstones original, creates corrected version."""
    db, lock = _db()
    with lock:
        result = db.correct(
            rid, req.new_text,
            new_importance=req.new_importance,
            new_valence=req.new_valence,
            correction_note=req.correction_note,
        )
    return result


# ── Entity / Graph Endpoints ──


@app.post("/entities/relate", response_model=RelateResponse, tags=["entities"])
async def relate_entities(req: RelateRequest):
    """Create or update an entity relationship."""
    db, lock = _db()
    with lock:
        edge_id = db.relate(req.src, req.dst, rel_type=req.rel_type, weight=req.weight)
    return RelateResponse(edge_id=edge_id)


@app.get("/entities/{entity}/edges", tags=["entities"])
async def get_edges(entity: str):
    """Get all relationships for an entity."""
    db, lock = _db()
    with lock:
        edges = db.get_edges(entity)
    return {"entity": entity, "count": len(edges), "edges": edges}


# ── Cognition Endpoints ──


@app.post("/think", tags=["cognition"])
async def think(req: ThinkRequest = ThinkRequest()):
    """Run the cognition loop."""
    db, lock = _db()
    config = {
        "run_consolidation": req.run_consolidation,
        "run_conflict_scan": req.run_conflict_scan,
        "run_pattern_mining": req.run_pattern_mining,
    }
    with lock:
        result = db.think(config=config)
    return result


@app.get("/conflicts", tags=["cognition"])
async def list_conflicts(status: str | None = None, limit: int = 50):
    """List memory conflicts."""
    db, lock = _db()
    with lock:
        conflicts = db.get_conflicts(status=status, limit=limit)
    return {"count": len(conflicts), "conflicts": conflicts}


@app.post("/conflicts/{conflict_id}/resolve", tags=["cognition"])
async def resolve_conflict(conflict_id: str, req: ResolveRequest):
    """Resolve a memory conflict."""
    db, lock = _db()
    with lock:
        result = db.resolve_conflict(
            conflict_id, req.strategy,
            winner_rid=req.winner_rid,
            new_text=req.new_text,
            resolution_note=req.resolution_note,
        )
    return result


@app.get("/triggers", tags=["cognition"])
async def list_triggers(limit: int = 10):
    """Get pending triggers."""
    db, lock = _db()
    with lock:
        triggers = db.get_pending_triggers(limit=limit)
    return {"count": len(triggers), "triggers": triggers}


# ── Stats ──


@app.get("/stats", tags=["system"])
async def get_stats():
    """Get engine statistics."""
    db, lock = _db()
    with lock:
        return db.stats()


# ── Export/Import ──


@app.get("/export", tags=["system"])
async def export_db(include_embeddings: bool = False):
    """Export the entire database as JSON."""
    from yantrikdb.cli import _export_db
    db, lock = _db()
    with lock:
        return _export_db(db, include_embeddings=include_embeddings)


# ── Entry Point ──


def _is_loopback_host(host: str) -> bool:
    """True only for hosts that provably resolve to loopback."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Unknown hostname — treat as non-loopback (fail closed).
        return False


def _check_bind_security(host: str) -> None:
    """Refuse to bind a non-loopback interface without an API key.

    Escape hatch: YANTRIKDB_ALLOW_INSECURE=1 downgrades the refusal to a
    prominent warning. Loopback binds without a key keep working unchanged.
    """
    if _is_loopback_host(host) or _configured_api_key() is not None:
        return
    if os.environ.get("YANTRIKDB_ALLOW_INSECURE", "").strip() == "1":
        log.warning(
            "SECURITY WARNING: binding to non-loopback host %r with NO API key "
            "(YANTRIKDB_ALLOW_INSECURE=1). Every endpoint — including GET /export "
            "(full DB dump) and DELETE /memories/{rid} — is reachable by anyone "
            "who can reach this interface.",
            host,
        )
        return
    raise SystemExit(
        f"Refusing to start: YANTRIKDB_HOST={host!r} is not a loopback address and "
        "no API key is configured. Anyone who can reach this interface could dump "
        "the database via GET /export or delete memories. Set YANTRIKDB_API_KEY "
        "to enable bearer auth, bind to 127.0.0.1, or set YANTRIKDB_ALLOW_INSECURE=1 "
        "to override (not recommended)."
    )


def main():
    """Entry point for the yantrikdb-server console script."""
    host = os.environ.get("YANTRIKDB_HOST", "127.0.0.1")
    port = int(os.environ.get("YANTRIKDB_PORT", "8321"))
    _check_bind_security(host)
    uvicorn.run(app, host=host, port=port, log_level="info")
