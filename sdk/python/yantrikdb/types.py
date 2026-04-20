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


# ── Character-substrate primitives ────────────────────────────────
# These are conventions over memory_type: str. The engine treats them
# identically to "semantic"/"episodic"/"procedural"; the SDK adds
# typed helpers so agents can author and recall them with intent.

CHARACTER_TYPES = (
    "self_model",       # "I am / I tend to / I can't / I value"
    "rule",             # "when X, prefer/do Y" — policy
    "hypothesis",       # "maybe A causes B" — tentative, needs corroboration
    "constraint",       # user-authored commitment that should survive friction
    "goal",             # target-state with optional deadline + feedback
    "narrative_arc",    # open storyline/thread the agent is inside
    "learning_signal",  # reward/punishment/calibration delta
)


@dataclass
class Reflection:
    """Structured meta-state view of the agent at a moment in time.

    Built by composing type-filtered recalls. Each field is a list of
    Memory objects filtered to the relevant character-substrate type,
    ranked by relevance to the question asked.
    """
    question: str
    self_model: list[Memory] = field(default_factory=list)
    rules: list[Memory] = field(default_factory=list)
    hypotheses: list[Memory] = field(default_factory=list)
    constraints: list[Memory] = field(default_factory=list)
    goals: list[Memory] = field(default_factory=list)
    arcs: list[Memory] = field(default_factory=list)
    recent_signals: list[Memory] = field(default_factory=list)
    open_conflicts: list[dict] = field(default_factory=list)

    def render(self) -> str:
        """Human-readable summary suitable for feeding back to an LLM."""
        lines = [f"# Reflection on: {self.question}"]
        def section(name, items, fmt):
            if not items:
                return
            lines.append(f"\n## {name}")
            for m in items:
                lines.append(f"- {fmt(m)}")
        section("Self-model", self.self_model,
                lambda m: f"{m.text}  (confidence={m.certainty:.2f})")
        section("Rules / policies", self.rules,
                lambda m: f"{m.text}  (importance={m.importance:.2f})")
        section("Open hypotheses", self.hypotheses,
                lambda m: f"{m.text}  (certainty={m.certainty:.2f})")
        section("Constraints", self.constraints,
                lambda m: f"{m.text}  (priority={m.importance:.2f})")
        section("Active goals", self.goals,
                lambda m: m.text)
        section("Open narrative arcs", self.arcs,
                lambda m: f"{m.text}  (status={m.metadata.get('status','open')})")
        section("Recent learning signals", self.recent_signals,
                lambda m: f"{m.text}  (valence={m.valence:+.2f})")
        if self.open_conflicts:
            lines.append(f"\n## Open conflicts ({len(self.open_conflicts)})")
            for c in self.open_conflicts[:5]:
                lines.append(f"- {c.get('conflict_id','?')}: {c.get('summary','(no summary)')}")
        return "\n".join(lines)
