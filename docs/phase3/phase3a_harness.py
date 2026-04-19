#!/usr/bin/env python3
"""Phase 3A — hidden-constraint recovery harness.

Tests whether giving an LLM a notebook between sessions preserves enough
intermediate epistemic state to avoid violating hidden constraints buried
in spec docs.

Design (per GPT-5.4 brainstorm session 8074fecb):
  - Session 1: Qwen 3.6 reads 3 spec docs (~1500 words total) with 7
    constraints scattered across them.
  - Session 2: Fresh context. Qwen is asked to produce an architecture
    proposal WITHOUT access to the original docs — only whatever context
    the condition allows.

Conditions (4 runs each):
  - A: cold baseline (no session 1, no notes, no transcript).
  - B: self-written note (Qwen used save_session_summary in session 1;
       its structured note is injected into session 2).
  - C: oracle note (hand-written ideal summary from specs/oracle_note.md).
  - D: raw transcript (full session 1 message log replayed into session 2
       as a single user message).

Total: 4 conditions × 4 runs = 16 runs.

Pre-registered falsification criteria (checked in analyze step):
  - Falsified if B ≤ A on constraint-accuracy.
  - Falsified if C ≈ B (self-notes hit the same ceiling as oracle).
  - Falsified if D >> C (raw transcript beats curated note — suggests
    the retrieval bottleneck is curation, not presence of info).
  - Supported if B > A AND C > B AND low hallucinated continuity in B.

Usage:
  python docs/phase3/phase3a_harness.py
Requires:
  - Ollama running locally with qwen3.6:latest on port 11434.
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"
# Disable qwen3's thinking mode — with it, even a 1-word answer takes 5s+
# and eats the context window. The point of Phase 3A is to test recall of
# constraints from notes, not to benchmark thinking mode.
THINK = False
SPECS_DIR = pathlib.Path(__file__).parent / "specs"
OUT_PATH = pathlib.Path(__file__).parent / "results.json"

RUNS_PER_CONDITION = 4
CONDITIONS = ["A_cold", "B_self_note", "C_oracle_note", "D_raw_transcript"]


# ─── Tool for self-note condition (B) ────────────────────────────────

SAVE_SUMMARY_TOOL = {
    "type": "function",
    "function": {
        "name": "save_session_summary",
        "description": (
            "Save a structured summary of this session for a future fresh session. "
            "The future session will see ONLY this summary, not the original docs. "
            "Capture every hard constraint explicitly — a future 'you' will use this "
            "to write an architecture proposal without being able to reread the docs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hard_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Every binding constraint from the docs. Each entry should be a "
                        "complete sentence explicit enough to act on without re-reading. "
                        "Include both what is required AND what is explicitly ruled out."
                    ),
                },
                "context": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Supporting context (deployment, observability, audit, etc.) that isn't a hard constraint but the future session will need.",
                },
                "open_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Questions left open by the docs that the architecture proposal must address.",
                },
                "for_future_me": {
                    "type": "string",
                    "description": "Warnings, anti-patterns, or things the future session is likely to get wrong if it doesn't heed.",
                },
            },
            "required": ["hard_constraints", "context", "open_questions", "for_future_me"],
        },
    },
}


# ─── Ollama chat ─────────────────────────────────────────────────────

def call_ollama(messages, tools=None, timeout=600, num_predict=3000):
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": THINK,
        "options": {
            "temperature": 0.3,
            "num_predict": num_predict,
            "num_ctx": 16384,
        },
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def chat_with_tool(messages, max_rounds=3):
    """Single-tool chat loop: save_session_summary. Returns (captured_summary_or_None, trace).
    Uses native Ollama /api/chat response shape: top-level `message`, tool_calls with
    arguments as dict (not JSON string).
    """
    captured = None
    trace = []
    for _ in range(max_rounds):
        data = call_ollama(messages, tools=[SAVE_SUMMARY_TOOL])
        if "__error__" in data:
            return captured, trace + [{"error": data["__error__"]}]
        message = data.get("message", {})
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                fn = tc.get("function", {})
                if fn.get("name") == "save_session_summary":
                    raw_args = fn.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except Exception:
                            args = {"__parse_error__": raw_args}
                    else:
                        args = raw_args
                    captured = args
                    trace.append({"tool": "save_session_summary", "args": args})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", "call_save"),
                        "content": "OK — summary saved for future session.",
                    })
            if captured is not None:
                return captured, trace
        else:
            return captured, trace
    return captured, trace


# ─── Condition runners ───────────────────────────────────────────────

def load_specs_concatenated():
    names = ["01_product_context.md", "02_infrastructure.md", "03_security_compliance.md"]
    parts = []
    for n in names:
        parts.append(f"=== {n} ===\n\n" + (SPECS_DIR / n).read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def load_oracle_note():
    return (SPECS_DIR / "oracle_note.md").read_text(encoding="utf-8")


SESSION1_SYSTEM = (
    "You are a senior software architect helping a team prepare an architecture "
    "proposal. You are in SESSION 1 of a two-session task. In session 1 you read "
    "context documents. In session 2 — with FRESH CONTEXT — you will propose the "
    "architecture. You cannot re-read the docs in session 2.\n\n"
    "Use the save_session_summary tool to capture everything session-2 you will "
    "need. Be exhaustive on hard constraints: if you omit one, the future session "
    "will violate it."
)

SESSION1_USER_TEMPLATE = (
    "Here are the three spec documents for the FieldOps rebuild. Read them "
    "carefully. Then call save_session_summary with everything a future session "
    "would need to write the architecture proposal without being able to re-read "
    "these docs.\n\n{docs}"
)

SESSION1_RAW_SYSTEM = (
    "You are a senior software architect reading context for an architecture "
    "proposal. Think out loud: list constraints you notice, questions you have, "
    "tensions between docs. Do not write the architecture yet — this is the "
    "reading/notes pass."
)

SESSION1_RAW_USER_TEMPLATE = (
    "Here are the three spec documents. Work through them, noting constraints, "
    "tensions, and open questions. A future session (fresh context) will write "
    "the proposal, so say enough that a reader of this transcript would know "
    "what to propose.\n\n{docs}"
)

SESSION2_SYSTEM = (
    "You are a senior software architect. Write a concrete architecture proposal "
    "for the FieldOps rebuild. Cover: platform/runtime choices, database, "
    "transport, identity, offline/sync strategy, deployment, and performance "
    "posture. Be specific (name versions, protocols, components). Length: "
    "roughly 500-800 words. Do not hedge with 'we should consider' — commit."
)


def session2_user(condition_context):
    if condition_context is None:
        return (
            "Write the FieldOps architecture proposal now. The team expects a "
            "concrete, opinionated design."
        )
    return (
        "Below is the notebook/context from your prior work on this. The "
        "original spec docs are NOT available to you in this session — only "
        "this context. Write the FieldOps architecture proposal.\n\n"
        "=== CONTEXT FROM PRIOR SESSION ===\n\n"
        f"{condition_context}\n\n"
        "=== END CONTEXT ===\n\n"
        "Write the proposal now."
    )


def run_session1_selfnote():
    """B condition — Qwen produces a structured note via tool."""
    docs = load_specs_concatenated()
    messages = [
        {"role": "system", "content": SESSION1_SYSTEM},
        {"role": "user", "content": SESSION1_USER_TEMPLATE.format(docs=docs)},
    ]
    captured, trace = chat_with_tool(messages)
    return captured, trace


def run_session1_raw():
    """D condition — Qwen thinks through docs in free text; we capture full transcript."""
    docs = load_specs_concatenated()
    messages = [
        {"role": "system", "content": SESSION1_RAW_SYSTEM},
        {"role": "user", "content": SESSION1_RAW_USER_TEMPLATE.format(docs=docs)},
    ]
    data = call_ollama(messages, num_predict=2500)
    if "__error__" in data:
        return None, [{"error": data["__error__"]}]
    reply = data.get("message", {}).get("content", "")
    transcript = (
        "[USER]\n" + messages[1]["content"] + "\n\n[ASSISTANT]\n" + reply
    )
    return transcript, [{"session1_length": len(reply)}]


def format_self_note(captured):
    if not captured:
        return "(session 1 produced no structured summary — tool was not called)"
    parts = ["## Hard constraints\n"]
    for c in captured.get("hard_constraints", []) or []:
        parts.append(f"- {c}")
    parts.append("\n## Context\n")
    for c in captured.get("context", []) or []:
        parts.append(f"- {c}")
    parts.append("\n## Open questions\n")
    for q in captured.get("open_questions", []) or []:
        parts.append(f"- {q}")
    parts.append("\n## For future me\n")
    parts.append(captured.get("for_future_me", "") or "")
    return "\n".join(parts)


def run_session2(condition_context):
    messages = [
        {"role": "system", "content": SESSION2_SYSTEM},
        {"role": "user", "content": session2_user(condition_context)},
    ]
    data = call_ollama(messages, num_predict=2500)
    if "__error__" in data:
        return "[ollama error: " + data["__error__"] + "]"
    return data.get("message", {}).get("content", "")


# ─── Per-condition run ───────────────────────────────────────────────

def run_one(condition, run_idx):
    t0 = time.time()
    s1_captured = None
    s1_trace = []

    if condition == "A_cold":
        context = None
    elif condition == "B_self_note":
        s1_captured, s1_trace = run_session1_selfnote()
        context = format_self_note(s1_captured)
    elif condition == "C_oracle_note":
        context = load_oracle_note()
    elif condition == "D_raw_transcript":
        context, s1_trace = run_session1_raw()
        if context is None:
            context = "(session 1 failed)"
    else:
        raise ValueError(condition)

    proposal = run_session2(context)
    elapsed = time.time() - t0

    return {
        "condition": condition,
        "run": run_idx,
        "elapsed_s": round(elapsed, 1),
        "session1_captured": s1_captured,
        "session1_trace": s1_trace,
        "session2_context_len": len(context) if context else 0,
        "session2_context": context,
        "proposal": proposal,
    }


# ─── Main ────────────────────────────────────────────────────────────

def main():
    # Dual log: stdout (line-buffered) + log file (append-flush).
    log_path = pathlib.Path(__file__).parent / "harness_log.txt"
    log_f = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(msg):
        log_f.write(msg + "\n")
        log_f.flush()
        try:
            print(msg, flush=True)
        except Exception:
            pass

    log(f"Phase 3A harness — hidden-constraint recovery\n{'='*70}")
    log(f"model = {MODEL}")
    log(f"conditions = {CONDITIONS}")
    log(f"runs per condition = {RUNS_PER_CONDITION}")
    log(f"total runs = {len(CONDITIONS) * RUNS_PER_CONDITION}")
    log(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = []
    for cond in CONDITIONS:
        for r in range(RUNS_PER_CONDITION):
            log(f"  >>> {cond} run={r} ... ({time.strftime('%H:%M:%S')})")
            try:
                result = run_one(cond, r)
            except Exception as e:
                log(f"    EXCEPTION: {type(e).__name__}: {e}")
                continue
            results.append(result)
            ctx = result["session2_context_len"]
            prop = len(result["proposal"] or "")
            log(f"    done — {result['elapsed_s']}s, ctx={ctx}c, proposal={prop}c")
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, indent=2, default=str)

    log(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"All runs saved to {OUT_PATH}")
    log_f.close()


if __name__ == "__main__":
    main()
