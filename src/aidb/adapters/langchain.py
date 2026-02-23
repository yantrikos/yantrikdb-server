"""LangChain memory adapter for AIDB.

Implements the BaseChatMemory protocol, allowing AIDB to serve as a
persistent memory backend for LangChain agents.

Usage:
    from aidb.adapters.langchain import AidbChatMemory

    memory = AidbChatMemory(db=aidb_instance)
    chain = ConversationChain(memory=memory, llm=llm)
"""

from __future__ import annotations

from typing import Any


class AidbChatMemory:
    """LangChain-compatible chat memory backed by AIDB.

    Stores conversation turns as episodic memories and recalls
    relevant context on each load.
    """

    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    return_messages: bool = False

    def __init__(
        self,
        db: Any,
        memory_key: str = "history",
        top_k: int = 5,
        importance: float = 0.4,
    ):
        self.db = db
        self.memory_key = memory_key
        self.top_k = top_k
        self.importance = importance

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Recall relevant memories for the current input."""
        query = inputs.get(self.input_key, "")
        if not query:
            return {self.memory_key: ""}

        results = self.db.recall(query=query, top_k=self.top_k)

        if self.return_messages:
            # Return as LangChain message objects
            try:
                from langchain_core.messages import AIMessage, HumanMessage

                messages = []
                for r in results:
                    text = r["text"]
                    if text.startswith("Human: "):
                        messages.append(HumanMessage(content=text[7:]))
                    elif text.startswith("AI: "):
                        messages.append(AIMessage(content=text[4:]))
                    else:
                        messages.append(HumanMessage(content=text))
                return {self.memory_key: messages}
            except ImportError:
                pass

        # Default: return as formatted string
        lines = [r["text"] for r in results]
        return {self.memory_key: "\n".join(lines)}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Store a conversation turn as an episodic memory."""
        human_input = inputs.get(self.input_key, "")
        ai_output = outputs.get(self.output_key, "")

        if human_input:
            self.db.record(
                text=f"Human: {human_input}",
                memory_type="episodic",
                importance=self.importance,
            )
        if ai_output:
            self.db.record(
                text=f"AI: {ai_output}",
                memory_type="episodic",
                importance=self.importance,
            )

    def clear(self) -> None:
        """Clear is a no-op — AIDB uses decay, not deletion."""
        pass
