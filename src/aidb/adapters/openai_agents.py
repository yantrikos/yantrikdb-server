"""OpenAI Agents SDK adapter for AIDB.

Generates function-calling tool definitions and dispatches tool calls
to the AIDB engine.

Usage:
    from aidb.adapters.openai_agents import get_tools, handle_tool_call

    tools = get_tools()
    # Add tools to your agent definition
    # When a tool call comes in:
    result = handle_tool_call(db, tool_name, arguments)
"""

from __future__ import annotations

from typing import Any


def get_tools() -> list[dict]:
    """Return OpenAI function-calling tool definitions for AIDB."""
    return [
        {
            "type": "function",
            "function": {
                "name": "memory_record",
                "description": "Store a new memory in the cognitive memory engine.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The memory content to store.",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "procedural"],
                            "description": "Type of memory. Default: episodic.",
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance score 0.0-1.0. Default: 0.5.",
                        },
                        "valence": {
                            "type": "number",
                            "description": "Emotional tone -1.0 to 1.0. Default: 0.0.",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional key-value metadata.",
                        },
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_recall",
                "description": "Search memories by semantic similarity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results. Default: 10.",
                        },
                        "memory_type": {
                            "type": "string",
                            "description": "Filter by type.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_forget",
                "description": "Tombstone a memory by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rid": {
                            "type": "string",
                            "description": "The memory ID to forget.",
                        },
                    },
                    "required": ["rid"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "entity_relate",
                "description": "Create a relationship between two entities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source entity name.",
                        },
                        "target": {
                            "type": "string",
                            "description": "Target entity name.",
                        },
                        "relationship": {
                            "type": "string",
                            "description": "Relationship type. Default: related_to.",
                        },
                        "weight": {
                            "type": "number",
                            "description": "Strength 0.0-1.0. Default: 1.0.",
                        },
                    },
                    "required": ["source", "target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "entity_edges",
                "description": "Get all relationships for an entity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "Entity name to look up.",
                        },
                    },
                    "required": ["entity"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_stats",
                "description": "Get memory engine statistics.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def handle_tool_call(db: Any, name: str, arguments: dict) -> Any:
    """Dispatch a tool call to the AIDB engine.

    Args:
        db: An AIDB instance (with embedder configured).
        name: Tool function name.
        arguments: Tool call arguments.

    Returns:
        The result of the tool call.
    """
    if name == "memory_record":
        rid = db.record(
            text=arguments["text"],
            memory_type=arguments.get("memory_type", "episodic"),
            importance=arguments.get("importance", 0.5),
            valence=arguments.get("valence", 0.0),
            metadata=arguments.get("metadata"),
        )
        return {"rid": rid}

    elif name == "memory_recall":
        results = db.recall(
            query=arguments["query"],
            top_k=arguments.get("top_k", 10),
            memory_type=arguments.get("memory_type"),
        )
        return {"memories": results}

    elif name == "memory_forget":
        success = db.forget(arguments["rid"])
        return {"forgotten": success}

    elif name == "entity_relate":
        edge_id = db.relate(
            src=arguments["source"],
            dst=arguments["target"],
            rel_type=arguments.get("relationship", "related_to"),
            weight=arguments.get("weight", 1.0),
        )
        return {"edge_id": edge_id}

    elif name == "entity_edges":
        edges = db.get_edges(arguments["entity"])
        return {"edges": edges}

    elif name == "memory_stats":
        return db.stats()

    else:
        raise ValueError(f"Unknown tool: {name}")
