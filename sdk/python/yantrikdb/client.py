"""YantrikDB Python client — talks to the HTTP gateway."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from .types import (
    Edge,
    Memory,
    RecallResult,
    SessionSummary,
    Stats,
    ThinkResult,
)


def connect(url: str = "http://localhost:7438", *, token: str) -> YantrikClient:
    """Connect to a YantrikDB server.

    Args:
        url: Server URL. Supports:
            - http://host:port (HTTP gateway, default)
            - yantrik://host:port (wire protocol port — auto-adjusts to HTTP +1)
        token: Authentication token (ydb_...).
    """
    parsed = urlparse(url)
    if parsed.scheme in ("yantrik", "yantrik+tls"):
        port = (parsed.port or 7437) + 1
        http_url = f"http://{parsed.hostname}:{port}"
    else:
        http_url = url.rstrip("/")

    return YantrikClient(http_url, token)


class YantrikClient:
    """Client for YantrikDB HTTP gateway."""

    def __init__(self, base_url: str, token: str):
        self._base = base_url
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _post(self, path: str, json: dict) -> dict:
        r = self._client.post(path, json=json)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> dict:
        r = self._client.get(path)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str, json: dict | None = None) -> dict:
        r = self._client.request("DELETE", path, json=json)
        r.raise_for_status()
        return r.json()

    # ── Memory ────────────────────────────────────────────

    def remember(
        self,
        text: str,
        *,
        importance: float = 0.5,
        memory_type: str = "semantic",
        domain: str = "",
        source: str = "user",
        namespace: str = "",
        metadata: dict | None = None,
        valence: float = 0.0,
        half_life: float = 168.0,
        certainty: float = 1.0,
        emotional_state: str | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Store a memory. Returns the memory RID."""
        payload: dict[str, Any] = {
            "text": text,
            "importance": importance,
            "memory_type": memory_type,
            "domain": domain,
            "source": source,
            "namespace": namespace,
            "metadata": metadata or {},
            "valence": valence,
            "half_life": half_life,
            "certainty": certainty,
        }
        if emotional_state:
            payload["emotional_state"] = emotional_state
        if embedding:
            payload["embedding"] = embedding

        data = self._post("/v1/remember", payload)
        return data["rid"]

    def recall(
        self,
        query: str,
        *,
        top_k: int = 10,
        domain: str | None = None,
        source: str | None = None,
        namespace: str | None = None,
        memory_type: str | None = None,
        include_consolidated: bool = False,
        expand_entities: bool = True,
    ) -> RecallResult:
        """Semantic recall. Returns ranked results with explanations."""
        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "include_consolidated": include_consolidated,
            "expand_entities": expand_entities,
        }
        if domain:
            payload["domain"] = domain
        if source:
            payload["source"] = source
        if namespace:
            payload["namespace"] = namespace
        if memory_type:
            payload["memory_type"] = memory_type

        data = self._post("/v1/recall", payload)
        results = [Memory(**r) for r in data["results"]]
        return RecallResult(results=results, total=data["total"])

    def forget(self, rid: str) -> bool:
        """Tombstone a memory. Returns True if found."""
        data = self._post("/v1/forget", {"rid": rid})
        return data.get("found", False)

    # ── Graph ─────────────────────────────────────────────

    def relate(
        self,
        entity: str,
        target: str,
        relationship: str,
        *,
        weight: float = 1.0,
    ) -> str:
        """Create a knowledge graph edge. Returns edge ID."""
        data = self._post("/v1/relate", {
            "entity": entity,
            "target": target,
            "relationship": relationship,
            "weight": weight,
        })
        return data["edge_id"]

    # ── Session ───────────────────────────────────────────

    @contextmanager
    def session(
        self,
        namespace: str = "default",
        client_id: str = "",
        metadata: dict | None = None,
    ):
        """Context manager for cognitive sessions."""
        data = self._post("/v1/sessions", {
            "namespace": namespace,
            "client_id": client_id,
            "metadata": metadata or {},
        })
        sid = data["session_id"]
        try:
            yield _Session(self, sid)
        finally:
            self._delete(f"/v1/sessions/{sid}")

    # ── Cognition ─────────────────────────────────────────

    def think(
        self,
        *,
        run_consolidation: bool = True,
        run_conflict_scan: bool = True,
        run_pattern_mining: bool = False,
        run_personality: bool = False,
        consolidation_limit: int = 50,
    ) -> ThinkResult:
        """Trigger the cognitive loop."""
        data = self._post("/v1/think", {
            "run_consolidation": run_consolidation,
            "run_conflict_scan": run_conflict_scan,
            "run_pattern_mining": run_pattern_mining,
            "run_personality": run_personality,
            "consolidation_limit": consolidation_limit,
        })
        return ThinkResult(**data)

    # ── Info ──────────────────────────────────────────────

    def stats(self) -> Stats:
        """Get engine statistics."""
        data = self._get("/v1/stats")
        return Stats(**data)

    def personality(self) -> list[dict]:
        """Get derived personality traits."""
        data = self._get("/v1/personality")
        return data.get("traits", [])

    def conflicts(self) -> list[dict]:
        """List open conflicts."""
        data = self._get("/v1/conflicts")
        return data.get("conflicts", [])

    def health(self) -> dict:
        """Check server health."""
        return self._get("/v1/health")


class _Session:
    """A cognitive session — memories created within are linked."""

    def __init__(self, client: YantrikClient, session_id: str):
        self._client = client
        self.session_id = session_id

    def remember(self, text: str, **kwargs) -> str:
        return self._client.remember(text, **kwargs)

    def recall(self, query: str, **kwargs) -> RecallResult:
        return self._client.recall(query, **kwargs)
