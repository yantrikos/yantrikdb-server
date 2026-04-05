from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Memory:
    rid: str
    text: str
    memory_type: str = "semantic"
    score: float = 0.0
    importance: float = 0.5
    created_at: float = 0.0
    why_retrieved: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    namespace: str = ""
    domain: str = ""
    source: str = "user"
    certainty: float = 1.0
    valence: float = 0.0


@dataclass
class RecallResult:
    results: list[Memory]
    total: int


@dataclass
class Edge:
    edge_id: str
    src: str
    dst: str
    rel_type: str
    weight: float = 1.0


@dataclass
class SessionSummary:
    session_id: str
    duration_secs: Optional[float] = None
    memory_count: Optional[int] = None
    topics: Optional[list[str]] = None


@dataclass
class ThinkResult:
    consolidation_count: int = 0
    conflicts_found: int = 0
    patterns_new: int = 0
    patterns_updated: int = 0
    personality_updated: bool = False
    duration_ms: int = 0
    triggers: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Stats:
    active_memories: int = 0
    consolidated_memories: int = 0
    tombstoned_memories: int = 0
    edges: int = 0
    entities: int = 0
    operations: int = 0
    open_conflicts: int = 0
    pending_triggers: int = 0
