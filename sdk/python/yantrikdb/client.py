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
    Reflection,
    SessionSummary,
    Stats,
    ThinkResult,
)


def connect(
    url: str = "http://localhost:7438",
    *,
    token: str,
    embedder: str | None = "all-MiniLM-L6-v2",
) -> YantrikClient:
    """Connect to a YantrikDB server.

    Args:
        url: Server URL. Supports:
            - http://host:port (HTTP gateway, default)
            - yantrik://host:port (wire protocol port — auto-adjusts to HTTP +1)
        token: Authentication token (ydb_...).
        embedder: Sentence-transformers model name to use for client-side
            embedding. Default is 'all-MiniLM-L6-v2' (384-dim, fast, CPU-friendly).
            Pass None to disable auto-embedding — the caller must then
            provide `embedding=[...]` on every remember()/recall() call.
    """
    parsed = urlparse(url)
    if parsed.scheme in ("yantrik", "yantrik+tls"):
        port = (parsed.port or 7437) + 1
        http_url = f"http://{parsed.hostname}:{port}"
    else:
        http_url = url.rstrip("/")

    return YantrikClient(http_url, token, embedder=embedder)


class YantrikClient:
    """Client for YantrikDB HTTP gateway."""

    def __init__(self, base_url: str, token: str, *, embedder: str | None = None):
        self._base = base_url
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
        self._embedder_name = embedder
        self._embedder = None  # lazy

    def _embed(self, text: str) -> list[float] | None:
        """Lazily load sentence-transformers and encode. Returns None if
        auto-embedding is disabled (embedder=None at construction)."""
        if self._embedder_name is None:
            return None
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "Auto-embedding requires `sentence-transformers`. "
                    "Install it (`pip install sentence-transformers`) or "
                    "pass `embedder=None` to connect() and supply embeddings manually."
                ) from e
            self._embedder = SentenceTransformer(self._embedder_name)
        vec = self._embedder.encode(text, convert_to_numpy=True, normalize_embeddings=False)
        return [float(x) for x in vec.tolist()]

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
        if embedding is None:
            embedding = self._embed(text)
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

        # Auto-embed the query for vector recall
        emb = self._embed(query)
        if emb is not None:
            payload["query_embedding"] = emb

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

    # ── Character-substrate primitives ────────────────────────
    # Typed helpers over memory_type conventions. Engine storage is
    # identical to other memory types; these helpers give agents
    # intent-legible authoring and recall of the classes of state
    # that make longitudinal identity work.

    def remember_self(
        self,
        content: str,
        *,
        confidence: float = 0.8,
        namespace: str = "",
        domain: str = "",
        source: str = "agent_reflection",
        metadata: dict | None = None,
    ) -> str:
        """Store a self-model claim — what the agent is / tends to do /
        can't do / values. Example: `remember_self("I overtrust recent
        single-source reports")`.

        `confidence` goes into the `certainty` field; the agent should
        lower it when evidence contradicts and raise it on corroboration.
        """
        return self.remember(
            content, memory_type="self_model",
            importance=0.7, certainty=confidence,
            namespace=namespace, domain=domain, source=source,
            metadata={**(metadata or {}), "class": "self_model"},
        )

    def remember_rule(
        self,
        condition: str,
        action: str,
        *,
        confidence: float = 0.7,
        namespace: str = "",
        source: str = "agent_reflection",
        metadata: dict | None = None,
    ) -> str:
        """Store a policy rule: "when <condition>, <action>".

        Rules are the agent's learned heuristics. They should be
        revised when outcomes disconfirm them — record a learning
        signal and reduce the rule's certainty.
        """
        text = f"WHEN {condition} THEN {action}"
        return self.remember(
            text, memory_type="rule",
            importance=0.7, certainty=confidence,
            namespace=namespace, source=source,
            metadata={**(metadata or {}), "class": "rule",
                      "condition": condition, "action": action},
        )

    def remember_hypothesis(
        self,
        statement: str,
        *,
        confidence: float = 0.4,
        namespace: str = "",
        source: str = "agent_reflection",
        metadata: dict | None = None,
    ) -> str:
        """Store a tentative belief ("maybe A causes B"). Distinct from
        observation (fact) and rule (policy). Agents should elevate to
        belief on corroboration or retract on disconfirmation."""
        return self.remember(
            statement, memory_type="hypothesis",
            importance=0.5, certainty=confidence,
            namespace=namespace, source=source,
            metadata={**(metadata or {}), "class": "hypothesis"},
        )

    def remember_constraint(
        self,
        label: str,
        description: str,
        *,
        priority: float = 0.9,
        namespace: str = "",
        source: str = "user_authored",
        metadata: dict | None = None,
    ) -> str:
        """Store a user- or agent-authored commitment that should
        survive friction: "truthfulness over user-pleasing",
        "no irreversible action without confirmation".

        `priority` governs which constraint wins in conflicts. The
        engine does not enforce; it stores and ranks — enforcement
        is the caller's responsibility via recall before action.
        """
        text = f"[{label}] {description}"
        return self.remember(
            text, memory_type="constraint",
            importance=priority, certainty=1.0,
            namespace=namespace, source=source,
            metadata={**(metadata or {}), "class": "constraint",
                      "label": label, "priority": priority},
        )

    def remember_goal(
        self,
        label: str,
        description: str,
        *,
        deadline: float | None = None,
        priority: float = 0.7,
        namespace: str = "",
        source: str = "user_authored",
        metadata: dict | None = None,
    ) -> str:
        """Store a goal-state memory. Unlike constraints, goals are
        bounded objectives that can be achieved, abandoned, or
        superseded. Use `record_signal` on outcome to feed back."""
        text = f"[GOAL {label}] {description}"
        meta = {**(metadata or {}), "class": "goal", "label": label,
                "status": "active", "priority": priority}
        if deadline is not None:
            meta["deadline"] = deadline
        return self.remember(
            text, memory_type="goal",
            importance=priority, certainty=1.0,
            namespace=namespace, source=source, metadata=meta,
        )

    def remember_arc(
        self,
        name: str,
        description: str,
        *,
        status: str = "open",
        namespace: str = "",
        source: str = "agent_narrative",
        metadata: dict | None = None,
    ) -> str:
        """Store a narrative arc / open thread: a storyline the agent
        is currently inside. Self-model is "who I am"; narrative arc
        is "what story I'm in the middle of". Status transitions:
        open → tension → resolved | abandoned."""
        text = f"[ARC {name}] {description}"
        return self.remember(
            text, memory_type="narrative_arc",
            importance=0.6, certainty=1.0,
            namespace=namespace, source=source,
            metadata={**(metadata or {}), "class": "narrative_arc",
                      "name": name, "status": status},
        )

    def record_signal(
        self,
        kind: str,
        content: str,
        *,
        valence: float = 0.0,
        about_rid: str | None = None,
        namespace: str = "",
        source: str = "outcome_feedback",
        metadata: dict | None = None,
    ) -> str:
        """Record a learning signal — reward, punishment, confirmation,
        disconfirmation, source-trust delta, regret, calibration error.

        `valence` is the affective charge (+1 strong positive, -1
        strong negative). `about_rid`, if given, links the signal to
        the memory it updates — enabling "which rule just earned a
        reinforcement" queries.

        `kind` is a free-form tag: "reward", "disconfirm", "regret",
        "trust_up", "trust_down", "calibration_error", etc.
        """
        text = f"[{kind}] {content}"
        meta = {**(metadata or {}), "class": "learning_signal", "kind": kind}
        if about_rid is not None:
            meta["about_rid"] = about_rid
        return self.remember(
            text, memory_type="learning_signal",
            importance=0.5, valence=valence, certainty=1.0,
            namespace=namespace, source=source, metadata=meta,
        )

    def recall_typed(
        self,
        query: str,
        memory_type: str,
        *,
        top_k: int = 5,
        namespace: str | None = None,
    ) -> list[Memory]:
        """Convenience: recall filtered to a single character-substrate
        memory_type. Thin wrapper over `recall()` for legibility."""
        return self.recall(
            query, top_k=top_k, namespace=namespace,
            memory_type=memory_type, expand_entities=False,
        ).results

    def reflect(
        self,
        question: str,
        *,
        namespace: str | None = None,
        top_k_per_type: int = 5,
        include_conflicts: bool = True,
    ) -> Reflection:
        """Compose a meta-state view by running parallel type-filtered
        recalls against the same question. Returns a structured
        `Reflection` ready to render into an LLM prompt.

        This is the key operation for "agent reads its own memory to
        reflect". The agent sees WHO IT IS (self-model), WHAT IT
        BELIEVES (rules + hypotheses), WHAT IT'S COMMITTED TO
        (constraints + goals), WHAT STORY IT'S IN (arcs), and WHAT
        JUST HAPPENED (recent signals) — all relevant to `question`.
        """
        def pull(mt: str) -> list[Memory]:
            return self.recall_typed(
                question, mt, top_k=top_k_per_type, namespace=namespace,
            )
        reflection = Reflection(
            question=question,
            self_model=pull("self_model"),
            rules=pull("rule"),
            hypotheses=pull("hypothesis"),
            constraints=pull("constraint"),
            goals=pull("goal"),
            arcs=pull("narrative_arc"),
            recent_signals=pull("learning_signal"),
        )
        if include_conflicts:
            try:
                reflection.open_conflicts = self.conflicts()
            except Exception:
                reflection.open_conflicts = []
        return reflection


class _Session:
    """A cognitive session — memories created within are linked."""

    def __init__(self, client: YantrikClient, session_id: str):
        self._client = client
        self.session_id = session_id

    def remember(self, text: str, **kwargs) -> str:
        return self._client.remember(text, **kwargs)

    def recall(self, query: str, **kwargs) -> RecallResult:
        return self._client.recall(query, **kwargs)
