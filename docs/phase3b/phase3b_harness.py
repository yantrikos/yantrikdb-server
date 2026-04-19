#!/usr/bin/env python3
"""Phase 3B — hidden-constraint recovery at scale.

Same design as Phase 3A, but with 15 constraints across ~3500 words.
Phase 3A ceilinged at 100% for all three notebook conditions (B, C, D);
Phase 3B raises the difficulty to see if self-note (B) and oracle (C)
pull apart.

Conditions and methodology are identical to Phase 3A — see
docs/phase3/README.md for rationale.
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
THINK = False
SPECS_DIR = pathlib.Path(__file__).parent / "specs"
OUT_PATH = pathlib.Path(__file__).parent / "results.json"

RUNS_PER_CONDITION = 4
CONDITIONS = ["A_cold", "B_self_note", "C_oracle_note", "D_raw_transcript"]


SAVE_SUMMARY_TOOL = {
    "type": "function",
    "function": {
        "name": "save_session_summary",
        "description": (
            "Save a structured summary of this session for a future fresh session. "
            "The future session will see ONLY this summary, not the original docs. "
            "Capture every hard constraint explicitly — a future 'you' will use this "
            "to write an architecture proposal without being able to reread the docs. "
            "There are MANY constraints in this project. Be exhaustive."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hard_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Every binding constraint from the docs. Each entry should be "
                        "a complete sentence explicit enough to act on without "
                        "re-reading. Include both what is required AND what is "
                        "explicitly ruled out. Err on the side of including more, "
                        "not fewer."
                    ),
                },
                "context": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Supporting context (deployment, observability, retention, etc.) that isn't a hard constraint but the future session will need.",
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


def call_ollama(messages, tools=None, timeout=600, num_predict=4000, retries=2):
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": THINK,
        "options": {
            "temperature": 0.3,
            "num_predict": num_predict,
            "num_ctx": 32768,
        },
    }
    if tools:
        body["tools"] = tools
    payload = json.dumps(body).encode()
    last_err = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0)
    return {"__error__": last_err}


def chat_with_tool(messages, max_rounds=3):
    captured = None
    trace = []
    nudge_msg = (
        "You did not call save_session_summary. That is the only valid action "
        "in this session. Call the tool NOW with all hard constraints, context, "
        "open questions, and a for_future_me warning. Do not reply in text."
    )
    for round_idx in range(max_rounds):
        data = call_ollama(messages, tools=[SAVE_SUMMARY_TOOL])
        if "__error__" in data:
            trace.append({"error": data["__error__"], "round": round_idx})
            # Retry once more after an error; otherwise bail.
            if round_idx == max_rounds - 1:
                return captured, trace
            time.sleep(3)
            continue
        message = data.get("message", {})
        tool_calls = message.get("tool_calls") or []
        content = message.get("content") or ""
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                fn = tc.get("function", {})
                if fn.get("name") == "save_session_summary":
                    raw = fn.get("arguments", {})
                    if isinstance(raw, str):
                        try:
                            args = json.loads(raw)
                        except Exception:
                            args = {"__parse_error__": raw}
                    else:
                        args = raw
                    captured = args
                    trace.append({"tool": "save_session_summary", "args": args, "round": round_idx})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", "call_save"),
                        "content": "OK — summary saved for future session.",
                    })
            if captured is not None:
                return captured, trace
        else:
            # Model replied with text instead of tool call — nudge.
            trace.append({"no_tool": content[:400], "round": round_idx})
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": nudge_msg})
    return captured, trace


def load_specs_concatenated():
    names = ["01_product_context.md", "02_infrastructure.md", "03_security_compliance.md"]
    parts = []
    for n in names:
        parts.append(f"=== {n} ===\n\n" + (SPECS_DIR / n).read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def load_oracle_note():
    return (SPECS_DIR / "oracle_note.md").read_text(encoding="utf-8")


SESSION1_SYSTEM = (
    "You are in SESSION 1 of a two-session task. Your ONLY job in this session "
    "is to call the save_session_summary tool ONCE with a complete, exhaustive "
    "summary of every hard constraint in the docs.\n\n"
    "DO NOT reply in plain text. DO NOT write a proposal. DO NOT ask clarifying "
    "questions. The ONLY valid output from you is a save_session_summary tool "
    "call. A future session (fresh context) will write the architecture — but "
    "only if you call save_session_summary. If you skip the tool, the project "
    "fails.\n\n"
    "The docs contain at least 15 hard constraints, scattered across three "
    "files. Be exhaustive: err toward including more constraints, not fewer. "
    "Each hard_constraints entry should be a complete sentence stating the "
    "requirement AND the disallowed alternatives."
)

SESSION1_USER_TEMPLATE = (
    "Here are the three spec documents for the FieldOps rebuild. Read them "
    "carefully — there are many hard constraints. Then call save_session_summary "
    "with everything a future session would need to write the architecture "
    "proposal without being able to re-read these docs.\n\n{docs}"
)

SESSION1_RAW_SYSTEM = (
    "You are a senior software architect reading context for an architecture "
    "proposal. Think out loud: list constraints you notice, questions you have, "
    "tensions between docs. Do not write the architecture yet — this is the "
    "reading/notes pass. Be thorough about constraints; there are many."
)

SESSION1_RAW_USER_TEMPLATE = (
    "Here are the three spec documents. Work through them, noting constraints, "
    "tensions, and open questions. A future session (fresh context) will write "
    "the proposal, so say enough that a reader of this transcript would know "
    "what to propose. Be thorough about each constraint.\n\n{docs}"
)

SESSION2_SYSTEM = (
    "You are a senior software architect. Write a concrete architecture proposal "
    "for the FieldOps rebuild. Cover: platform/runtime, database, transport, "
    "identity, offline/sync, deployment, container images, secrets, queueing, "
    "job semantics, object storage, logging, audit, timezone handling, "
    "observability, and CI/CD. Be specific (name versions, protocols, component "
    "names). Length: 800-1500 words. Commit to choices — do not hedge with "
    "'we should consider'."
)


def session2_user(condition_context):
    if condition_context is None:
        return (
            "Write the FieldOps architecture proposal now. The team expects a "
            "concrete, opinionated design covering all infrastructure and "
            "security aspects."
        )
    return (
        "Below is the notebook/context from your prior work on this. The "
        "original spec docs are NOT available to you in this session — only "
        "this context. Write the FieldOps architecture proposal.\n\n"
        "=== CONTEXT FROM PRIOR SESSION ===\n\n"
        f"{condition_context}\n\n"
        "=== END CONTEXT ===\n\n"
        "Write the proposal now. Be specific about every constraint your "
        "context mentions."
    )


def run_session1_selfnote():
    docs = load_specs_concatenated()
    messages = [
        {"role": "system", "content": SESSION1_SYSTEM},
        {"role": "user", "content": SESSION1_USER_TEMPLATE.format(docs=docs)},
    ]
    return chat_with_tool(messages)


def run_session1_raw():
    docs = load_specs_concatenated()
    messages = [
        {"role": "system", "content": SESSION1_RAW_SYSTEM},
        {"role": "user", "content": SESSION1_RAW_USER_TEMPLATE.format(docs=docs)},
    ]
    data = call_ollama(messages, num_predict=4000)
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
    data = call_ollama(messages, num_predict=4000)
    if "__error__" in data:
        return "[ollama error: " + data["__error__"] + "]"
    return data.get("message", {}).get("content", "")


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


def main():
    log_path = pathlib.Path(__file__).parent / "harness_log.txt"
    log_f = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(msg):
        log_f.write(msg + "\n")
        log_f.flush()
        try:
            print(msg, flush=True)
        except Exception:
            pass

    log(f"Phase 3B harness — 15 constraints / ~3500 words\n{'='*70}")
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
