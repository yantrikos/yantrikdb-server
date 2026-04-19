#!/usr/bin/env python3
"""Simple structured-memory simulator for Phase 3C Condition C.

Stores (key, value, session) tuples. Retrieves by word-overlap similarity.
Instrumented for failure-taxonomy logging (stored? queried? retrieved? used?).

NOT yantrikdb. This simulator isolates the question "does structured
key/value storage + retrieval help?" without yantrikdb-specific features
like temporal validity or polarity opposition. If this wins, integrating
real yantrikdb is a higher-value follow-up with known-positive signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    idx: int
    key: str
    value: str
    session: int
    raw_text: str = ""
    # Populated on retrieval:
    last_retrieved_for_query: str | None = None
    last_score: float = 0.0


_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "of", "to", "for", "in", "on", "at", "by", "with",
    "from", "as", "it", "this", "that", "these", "those", "our", "we",
    "what", "which", "who", "when", "where", "why", "how",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "have", "has", "had", "not", "no", "so", "if", "then",
    "current", "currently", "latest", "now", "also", "just", "only",
})


def _tokenize(s: str) -> list[str]:
    # Treat hyphenated words as single tokens (go-live, dual-vendor).
    raw = re.findall(r"[A-Za-z0-9_\.\-]+", s or "")
    out = []
    for t in raw:
        t_low = t.lower()
        if t_low in _STOPWORDS or len(t_low) <= 1:
            continue
        out.append(t_low)
    return out


@dataclass
class MemoryStore:
    memories: list[Memory] = field(default_factory=list)
    write_log: list[dict] = field(default_factory=list)
    query_log: list[dict] = field(default_factory=list)
    # Per-store caps. Defaults match Phase 3C's stricter regime; callers
    # for larger-text corpora (LongMemEval turns) can override.
    value_cap: int = 120
    key_cap: int = 60

    def remember(self, key: str, value: str, session: int) -> dict:
        """Store a (key, value, session) tuple. Returns storage confirmation."""
        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return {"__error__": "empty key or value"}
        if len(value) > self.value_cap:
            value = value[:self.value_cap]
        if len(key) > self.key_cap:
            key = key[:self.key_cap]
        mem = Memory(
            idx=len(self.memories),
            key=key,
            value=value,
            session=session,
            raw_text=f"{key}: {value}",
        )
        self.memories.append(mem)
        self.write_log.append({
            "session": session,
            "idx": mem.idx,
            "key": key,
            "value": value,
        })
        return {"stored": True, "id": mem.idx, "key": key}

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k memories most similar to query by word-overlap score.
        Ties broken by recency (higher session first)."""
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []
        scored: list[tuple[float, int, Memory]] = []
        for m in self.memories:
            m_tokens = set(_tokenize(m.raw_text))
            if not m_tokens:
                continue
            overlap = q_tokens & m_tokens
            if not overlap:
                continue
            # Dice coefficient — bounded [0, 1], rewards matches relative to both sizes.
            score = (2 * len(overlap)) / (len(q_tokens) + len(m_tokens))
            scored.append((score, m.session, m))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        top = scored[:top_k]
        results = []
        for score, _sess, m in top:
            m.last_retrieved_for_query = query
            m.last_score = score
            results.append({
                "id": m.idx,
                "key": m.key,
                "value": m.value,
                "session": m.session,
                "score": round(score, 3),
            })
        self.query_log.append({
            "query": query,
            "top_k": top_k,
            "returned_ids": [r["id"] for r in results],
            "returned_scores": [r["score"] for r in results],
        })
        return results

    def summary(self) -> dict:
        return {
            "n_memories": len(self.memories),
            "by_session": {
                s: sum(1 for m in self.memories if m.session == s)
                for s in sorted({m.session for m in self.memories})
            },
            "n_writes": len(self.write_log),
            "n_queries": len(self.query_log),
            "total_chars": sum(len(m.raw_text) for m in self.memories),
        }


if __name__ == "__main__":
    # Quick self-test.
    s = MemoryStore()
    s.remember("titan.cio", "Maria Delgado is Titan's CIO", 1)
    s.remember("titan.cloud.provider", "AWS is the target cloud", 1)
    s.remember("titan.go_live", "target go-live Q3 2026", 1)
    s.remember("titan.go_live", "go-live pushed to Q4 2026", 3)
    results = s.recall("when is the current go-live date", top_k=3)
    for r in results:
        print(r)
    print("summary:", s.summary())
