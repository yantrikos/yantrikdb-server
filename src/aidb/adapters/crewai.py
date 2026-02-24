"""CrewAI memory adapter for AIDB.

Provides short-term (episodic), long-term (semantic), and entity (graph)
memory backends for CrewAI agents.

Usage:
    from aidb.adapters.crewai import AidbShortTermMemory, AidbLongTermMemory, AidbEntityMemory

    crew = Crew(
        short_term_memory=AidbShortTermMemory(db),
        long_term_memory=AidbLongTermMemory(db),
        entity_memory=AidbEntityMemory(db),
    )
"""

from __future__ import annotations

from typing import Any


class AidbShortTermMemory:
    """CrewAI short-term memory backed by AIDB episodic memories."""

    def __init__(self, db: Any, top_k: int = 5, namespace: str = "default"):
        self.db = db
        self.top_k = top_k
        self.namespace = namespace

    def save(self, value: str, metadata: dict | None = None, agent: str | None = None) -> None:
        """Store a short-term memory (episodic)."""
        meta = metadata or {}
        if agent:
            meta["agent"] = agent
        self.db.record(
            text=value,
            memory_type="episodic",
            importance=0.4,
            metadata=meta,
            namespace=self.namespace,
        )

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        """Search short-term memories."""
        k = limit or self.top_k
        results = self.db.recall(
            query=query,
            top_k=k,
            memory_type="episodic",
            namespace=self.namespace,
        )
        return [{"context": r["text"], "score": r["score"]} for r in results]

    def reset(self) -> None:
        """Reset is a no-op — AIDB uses decay, not deletion."""
        pass


class AidbLongTermMemory:
    """CrewAI long-term memory backed by AIDB semantic memories."""

    def __init__(self, db: Any, top_k: int = 5, namespace: str = "default"):
        self.db = db
        self.top_k = top_k
        self.namespace = namespace

    def save(self, value: str, metadata: dict | None = None, agent: str | None = None) -> None:
        """Store a long-term memory (semantic)."""
        meta = metadata or {}
        if agent:
            meta["agent"] = agent
        self.db.record(
            text=value,
            memory_type="semantic",
            importance=0.7,
            metadata=meta,
            namespace=self.namespace,
        )

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        """Search long-term memories."""
        k = limit or self.top_k
        results = self.db.recall(
            query=query,
            top_k=k,
            memory_type="semantic",
            namespace=self.namespace,
        )
        return [{"context": r["text"], "score": r["score"]} for r in results]

    def reset(self) -> None:
        """Reset is a no-op — AIDB uses decay, not deletion."""
        pass


class AidbEntityMemory:
    """CrewAI entity memory backed by AIDB knowledge graph."""

    def __init__(self, db: Any, top_k: int = 5, namespace: str = "default"):
        self.db = db
        self.top_k = top_k
        self.namespace = namespace

    def save(self, value: str, metadata: dict | None = None, agent: str | None = None) -> None:
        """Store an entity observation.

        If metadata contains 'entity' and 'relationship' keys,
        also creates a graph edge.
        """
        meta = metadata or {}
        if agent:
            meta["agent"] = agent

        rid = self.db.record(
            text=value,
            memory_type="semantic",
            importance=0.6,
            metadata=meta,
            namespace=self.namespace,
        )

        # Auto-link entities if provided
        entity = meta.get("entity")
        if entity:
            self.db.link_memory_entity(rid, entity)
            target = meta.get("target_entity")
            rel = meta.get("relationship", "related_to")
            if target:
                self.db.relate(src=entity, dst=target, rel_type=rel)
                self.db.link_memory_entity(rid, target)

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        """Search entity memories with graph expansion."""
        k = limit or self.top_k
        results = self.db.recall(
            query=query,
            top_k=k,
            expand_entities=True,
            namespace=self.namespace,
        )
        return [{"context": r["text"], "score": r["score"]} for r in results]

    def reset(self) -> None:
        """Reset is a no-op — AIDB uses decay, not deletion."""
        pass
