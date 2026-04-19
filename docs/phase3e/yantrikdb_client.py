#!/usr/bin/env python3
"""Thin Python client for real yantrikdb HTTP endpoints.

Exposes the SAME shape as phase3c/memory_sim.py (remember/recall/summary)
but delegates to the actual yantrikdb server, computes embeddings
client-side via sentence-transformers, and supports think() — which the
simulator did NOT have.

Why this module exists: Phase 3A/3B/3C/3D all used memory_sim.py, which
is a Python list of (key, value, session) tuples with Dice word-overlap
retrieval. That is NOT yantrikdb — it has no HNSW, no multi-signal
scoring, no think() loop, no conflict detection, no decay. Running the
benchmarks against a simulator that omits the product's core features
was an invalid test. This client fixes that.

Env:
  YDB_URL (default: http://localhost:8420)
  YDB_TOKEN (default: reads from yantrikdb_local_token.txt or env)
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any

from sentence_transformers import SentenceTransformer

YDB_URL = os.environ.get("YDB_URL", "http://localhost:8420")
YDB_TOKEN = os.environ.get(
    "YDB_TOKEN",
    "ydb_32fe1f57be885f8b28727c9702d10d8358bdbab44c05cc9c24eba80d85266b8d",
)

# Singleton embedder — loaded once per process.
_EMBEDDER: SentenceTransformer | None = None
_EMBED_MODEL = "all-MiniLM-L6-v2"  # 384 dim, fast, well-studied


def embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(_EMBED_MODEL)
    return _EMBEDDER


def _http(method: str, path: str, body: dict | None = None, params: dict | None = None, timeout: int = 60) -> dict:
    url = YDB_URL + path
    if params:
        from urllib.parse import urlencode
        url += "?" + urlencode(params)
    headers = {
        "Authorization": f"Bearer {YDB_TOKEN}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            body_txt = exc.read().decode(errors="replace")[:300]
            if attempt == 2:
                return {"__error__": f"HTTP {exc.code}: {body_txt}"}
            time.sleep(1.5)
        except Exception as exc:
            if attempt == 2:
                return {"__error__": f"{type(exc).__name__}: {exc}"}
            time.sleep(1.5)
    return {"__error__": "unreachable"}


def _embed(text: str) -> list[float]:
    vec = embedder().encode(text, convert_to_numpy=True, normalize_embeddings=False)
    return [float(x) for x in vec.tolist()]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Encode a batch of texts. Much faster than per-call encode() because
    sentence-transformers amortizes tokenization/model init across the batch.
    On CPU this is 10-50x faster for batches of 100+ than sequential encode()."""
    if not texts:
        return []
    vecs = embedder().encode(texts, convert_to_numpy=True, normalize_embeddings=False, batch_size=64)
    return [[float(x) for x in v.tolist()] for v in vecs]


class YantrikStore:
    """Thin wrapper mimicking MemoryStore.remember/recall/summary/think."""

    def __init__(self, namespace: str, domain: str = "phase3e"):
        self.namespace = namespace
        self.domain = domain
        self.write_log: list[dict] = []
        self.query_log: list[dict] = []
        self.think_log: list[dict] = []

    def remember(
        self,
        key: str,
        value: str,
        session: int,
        importance: float = 0.5,
        valence: float = 0.0,
        embedding: list[float] | None = None,
    ) -> dict:
        """Store a memory. Key is passed via metadata so it's retrievable.
        Text for embedding = "{key}: {value}".

        If `embedding` is passed, skip the per-call embed() (use when you've
        batch-encoded already).
        """
        text = f"{key}: {value}" if key else value
        if not text.strip():
            return {"__error__": "empty text"}
        emb = embedding if embedding is not None else _embed(text)
        body = {
            "text": text,
            "memory_type": "semantic",
            "importance": importance,
            "valence": valence,
            "namespace": self.namespace,
            "domain": self.domain,
            "source": f"session_{session}",
            "embedding": emb,
            "metadata": {
                "session_n": session,
                "key": key,
            },
        }
        r = _http("POST", "/v1/remember", body=body)
        self.write_log.append({"session": session, "key": key, "value": value[:100], "result_keys": list(r.keys())[:5]})
        return r

    def remember_batch(
        self,
        items: list[dict],  # each: {key, value, session, importance?, valence?}
        chunk_size: int = 50,  # HTTP 413 body-size hits around 300+; safe at 50
    ) -> dict:
        """Batch-encode all texts, then POST to /v1/remember/batch in chunks.
        Each chunk = one HTTP round-trip. MASSIVELY faster than looped
        remember() at scale (550-turn haystack: ~10s vs ~20min on CPU)."""
        if not items:
            return {"count": 0, "rids": []}
        texts = [(item["key"] + ": " + item["value"]) if item.get("key") else item["value"] for item in items]
        vecs = embed_batch(texts)

        all_rids = []
        for i in range(0, len(items), chunk_size):
            batch_items = items[i:i + chunk_size]
            batch_vecs = vecs[i:i + chunk_size]
            memories = []
            for item, vec in zip(batch_items, batch_vecs):
                memories.append({
                    "text": (item["key"] + ": " + item["value"]) if item.get("key") else item["value"],
                    "memory_type": "semantic",
                    "importance": item.get("importance", 0.5),
                    "valence": item.get("valence", 0.0),
                    "namespace": self.namespace,
                    "domain": self.domain,
                    "source": f"session_{item['session']}",
                    "embedding": vec,
                    "metadata": {"session_n": item["session"], "key": item.get("key", "")},
                })
            r = _http("POST", "/v1/remember/batch", body={"memories": memories}, timeout=300)
            if "__error__" in r:
                # Fail LOUDLY so silent ingestion-skips can't fake success in logs
                self.write_log.append({"batch_error": r["__error__"], "chunk_start": i, "chunk_size": len(batch_items)})
                raise RuntimeError(f"remember_batch HTTP failure at chunk offset {i} (size {len(batch_items)}): {r['__error__']}")
            all_rids.extend(r.get("rids", []))
            self.write_log.append({"batch_count": len(batch_items), "rids_returned": len(r.get("rids", []))})
        return {"count": len(all_rids), "rids": all_rids}

    def recall(self, query: str, top_k: int = 10) -> list[dict]:
        """Run yantrikdb's multi-signal recall. Returns list of dicts with
        key (from metadata), value (from text), session (from source)."""
        emb = _embed(query)
        body = {
            "query": query,
            "top_k": top_k,
            "namespace": self.namespace,
            "query_embedding": emb,
        }
        r = _http("POST", "/v1/recall", body=body)
        self.query_log.append({"query": query, "top_k": top_k, "n_results": len(r.get("memories", []))})
        if "__error__" in r:
            return []
        results = []
        # Response shape: {"results": [...], "total": N} per yantrikdb HTTP API
        for mem in r.get("results", r.get("memories", [])):
            meta = mem.get("metadata") or {}
            text = mem.get("text", "")
            key = meta.get("key") or ""
            value = text[len(key) + 2:] if text.startswith(key + ": ") else text
            session = meta.get("session_n")
            if session is None:
                src = mem.get("source", "")
                session = int(src.split("_")[-1]) if src.startswith("session_") else 0
            results.append({
                "id": mem.get("rid") or mem.get("id") or "",
                "key": key,
                "value": value,
                "session": session,
                "score": mem.get("score", 0.0),
                "why_retrieved": mem.get("why_retrieved", []),
                "raw": mem,
            })
        return results

    def think(self) -> dict:
        """Run yantrikdb's think() loop: consolidation + conflict scan +
        pattern mining + trigger eval."""
        body = {"namespace": self.namespace}
        r = _http("POST", "/v1/think", body=body, timeout=120)
        self.think_log.append({"result": r})
        return r

    def stats(self) -> dict:
        return _http("GET", "/v1/stats")

    def summary(self) -> dict:
        s = self.stats()
        return {
            "n_writes": len(self.write_log),
            "n_queries": len(self.query_log),
            "n_thinks": len(self.think_log),
            "server_stats": s,
        }


if __name__ == "__main__":
    # Self-test: put in three memories, query, think.
    store = YantrikStore(namespace="selftest_001")
    print(">>> remember 3 items")
    print(store.remember("titan.cio", "Maria Delgado is Titan's CIO", 1))
    print(store.remember("titan.go_live", "go-live target: Q3 2026", 1))
    print(store.remember("titan.go_live", "go-live pushed: Q4 2026", 3))
    print(">>> recall go-live date")
    results = store.recall("when is the current go-live date", top_k=5)
    for r in results:
        print(f"  session={r['session']} score={r['score']:.3f} key={r['key']!r} value={r['value'][:60]!r}")
    print(">>> think()")
    print(store.think())
    print(">>> summary")
    print(store.summary())
