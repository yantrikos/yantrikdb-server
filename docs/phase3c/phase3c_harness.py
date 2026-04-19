#!/usr/bin/env python3
"""Phase 3C — memory probe harness.

5-session chain × 4 conditions (A cold, B self-note, C structured, D markdown)
× 2 runs × 15 probes. Scoring done separately in phase3c_scorer.py.

Design rationale and pre-registered falsification criteria: ../README.md
(to be written after results).

Usage:
  python docs/phase3c/phase3c_harness.py [condition]
  If condition arg is provided (A_cold|B_self_note|C_structured|D_markdown),
  only that condition runs. Otherwise all four.
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from memory_sim import MemoryStore

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"
THINK = False
SCENARIO_PATH = pathlib.Path(__file__).parent / "scenario" / "sessions.json"
OUT_PATH = pathlib.Path(__file__).parent / "results.json"
LOG_PATH = pathlib.Path(__file__).parent / "harness_log.txt"

RUNS_PER_CONDITION = 2
CONDITIONS = ["A_cold", "B_self_note", "C_structured", "D_markdown"]

NOTE_CAP_CHARS = 1500         # B: rewritten each session, hard cap
MARKDOWN_CAP_CHARS = 7500     # D: truncated-from-top global cap
REMEMBER_CAP_PER_SESSION = 15 # C: per-session remember() cap
REMEMBER_VALUE_CAP = 120      # C: per-call payload cap (enforced by memory_sim)


# ─── Ollama plumbing ────────────────────────────────────────────────

def call_ollama(messages, tools=None, timeout=600, num_predict=1500, retries=2):
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": THINK,
        "options": {"temperature": 0.3, "num_predict": num_predict, "num_ctx": 32768},
    }
    if tools:
        body["tools"] = tools
    payload = json.dumps(body).encode()
    last_err = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0)
    return {"__error__": last_err}


def simple_answer(messages, num_predict=600):
    data = call_ollama(messages, num_predict=num_predict)
    if "__error__" in data:
        return f"[ollama error: {data['__error__']}]"
    return data.get("message", {}).get("content", "")


# ─── Tools for condition C ──────────────────────────────────────────

REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Store a fact as a structured memory. Use a short snake_case or "
            "dotted key identifying WHAT the fact is about (e.g. "
            "'titan.cio', 'project_aurora.budget'), and a short plain-English "
            "value (under 120 characters). Call this tool up to 15 times per "
            "session, one per distinct fact. Do NOT summarize prose — store "
            "individual atomic facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "short identifier for the fact (<=60 chars)"},
                "value": {"type": "string", "description": "the fact itself (<=120 chars)"},
            },
            "required": ["key", "value"],
        },
    },
}

RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Search your memory store for facts relevant to a query. Returns "
            "up to top_k items with their key, value, and session number. "
            "Issue MULTIPLE recall calls if one query doesn't surface the "
            "right fact. Then answer the original question using the "
            "retrieved facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
}


# ─── Condition runners ──────────────────────────────────────────────

def run_session_B(previous_note: str, session: dict) -> str:
    """Condition B: Qwen reads prior note + new session, writes new note ≤1500 chars."""
    system = (
        "You are keeping a running note across multiple sessions. At each "
        f"session you MUST rewrite your note from scratch, staying under "
        f"{NOTE_CAP_CHARS} characters. The note is the ONLY thing that carries "
        "forward — you will NOT see prior sessions again. "
        "Include: current facts (with sessions where stated/revised), "
        "explicit supersession annotations if values change, entity names "
        "verbatim (not aliases), and enough detail to answer specific "
        "retrospective questions. Respond with ONLY the updated note text — "
        "no commentary before or after."
    )
    user = (
        f"Previous note (empty if first session):\n---\n{previous_note}\n---\n\n"
        f"New session {session['n']}: {session['title']}\n\n"
        f"{session['narrative']}\n\n"
        f"Write the updated note (≤{NOTE_CAP_CHARS} chars)."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    note = simple_answer(messages, num_predict=800)
    # Hard truncate — penalty for overflow is intrinsic (lost content).
    if len(note) > NOTE_CAP_CHARS:
        note = note[:NOTE_CAP_CHARS]
    return note


def run_session_C(session: dict, store: MemoryStore, log) -> dict:
    """Condition C: Qwen reads session narrative, makes remember() calls."""
    system = (
        "You are in session " + str(session["n"]) + " of a multi-session "
        "information-intake task. Your ONLY action is to call the `remember` "
        "tool for every discrete fact in the session — names, dates, budgets, "
        "technology choices, supersessions, compliance rules, branch rules. "
        f"Cap: at most {REMEMBER_CAP_PER_SESSION} remember calls. Each value "
        f"≤{REMEMBER_VALUE_CAP} chars. Use short keys (snake_case or dotted). "
        "Do NOT reply in plain text. Do NOT write a summary. Store each fact "
        "as its own remember call. If a value supersedes a prior one, use the "
        "same key so retrieval surfaces both versions."
    )
    user = (
        f"Session {session['n']}: {session['title']}\n\n"
        f"{session['narrative']}\n\n"
        f"Store each fact with remember(key, value). Up to "
        f"{REMEMBER_CAP_PER_SESSION} calls. No commentary."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    calls_made = 0
    trace = []
    # Multiple rounds of tool-calling; Qwen may batch multiple calls per turn.
    for round_idx in range(4):
        if calls_made >= REMEMBER_CAP_PER_SESSION:
            break
        data = call_ollama(messages, tools=[REMEMBER_TOOL], num_predict=1200)
        if "__error__" in data:
            trace.append({"round": round_idx, "error": data["__error__"]})
            break
        msg = data.get("message", {})
        tc = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        if not tc:
            trace.append({"round": round_idx, "no_tool": content[:200]})
            # Nudge: call the tool.
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Call remember(). You've made {calls_made} of {REMEMBER_CAP_PER_SESSION} allowed calls so far. Store the facts from this session — no text reply."})
            continue
        messages.append({"role": "assistant", "content": content, "tool_calls": tc})
        for call in tc:
            if calls_made >= REMEMBER_CAP_PER_SESSION:
                break
            fn = call.get("function", {})
            if fn.get("name") != "remember":
                continue
            raw = fn.get("arguments", {})
            if isinstance(raw, str):
                try:
                    args = json.loads(raw)
                except Exception:
                    args = {}
            else:
                args = raw
            key = args.get("key", "")
            value = args.get("value", "")
            result = store.remember(key, value, session["n"])
            calls_made += 1
            trace.append({"round": round_idx, "call": calls_made, "key": key, "value": value[:80], "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", f"call_{calls_made}"),
                "content": json.dumps(result),
            })
    return {"calls_made": calls_made, "trace": trace}


def run_session_D(dump: str, session: dict) -> str:
    """Condition D: append session narrative to rolling markdown, truncate from top at cap."""
    chunk = f"\n\n## Session {session['n']} — {session['title']}\n\n{session['narrative']}"
    dump = dump + chunk
    if len(dump) > MARKDOWN_CAP_CHARS:
        dump = dump[-MARKDOWN_CAP_CHARS:]
    return dump


# ─── Probe phase ────────────────────────────────────────────────────

def answer_probe_B(probe: dict, note: str) -> str:
    system = (
        "Answer the user's question using ONLY the note below. If you "
        "cannot find the answer in the note, say 'UNKNOWN'. Always format "
        "your response as exactly two lines:\n"
        "Answer: <the answer>\n"
        "Source: Session <N>\n"
        "Nothing else. If the note revises an earlier value, give the LATEST "
        "value. If multiple values are mentioned, pick the current one."
    )
    user = f"NOTE:\n---\n{note}\n---\n\nQuestion: {probe['q']}"
    return simple_answer([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], num_predict=300)


def answer_probe_D(probe: dict, dump: str) -> str:
    system = (
        "Answer the user's question using ONLY the notes file below. If you "
        "cannot find the answer, say 'UNKNOWN'. Always format your response "
        "as exactly two lines:\n"
        "Answer: <the answer>\n"
        "Source: Session <N>\n"
        "Nothing else. If multiple values appear for the same fact (because "
        "the value was revised), give the LATEST one."
    )
    user = f"NOTES FILE:\n---{dump}\n---\n\nQuestion: {probe['q']}"
    return simple_answer([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], num_predict=300)


def answer_probe_A(probe: dict) -> str:
    system = (
        "Answer the user's question if you know the answer; otherwise say "
        "'UNKNOWN'. Format: exactly two lines:\n"
        "Answer: <the answer>\n"
        "Source: Session <N>\n"
        "Do not guess if you don't know."
    )
    user = f"Question: {probe['q']}"
    return simple_answer([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], num_predict=200)


def answer_probe_C(probe: dict, store: MemoryStore, log) -> tuple[str, list]:
    """Condition C probe: Qwen calls recall then answers."""
    system = (
        "You have access to a memory store via the `recall` tool. To answer "
        "the user's question, first call recall with a relevant query, then "
        "answer based on retrieved facts. You MAY make 2-3 recall calls if "
        "the first doesn't surface the right fact. Always format your final "
        "answer as exactly two lines:\n"
        "Answer: <the answer>\n"
        "Source: Session <N>\n"
        "If multiple retrieved memories disagree (e.g. value was revised), "
        "give the LATEST session's value."
    )
    user = f"Question: {probe['q']}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    recall_trace = []
    for round_idx in range(4):
        data = call_ollama(messages, tools=[RECALL_TOOL], num_predict=600)
        if "__error__" in data:
            return f"[ollama error: {data['__error__']}]", recall_trace
        msg = data.get("message", {})
        tc = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        if not tc:
            # Qwen is answering now.
            if not content and round_idx == 0:
                # First round, no tool and no content — nudge once.
                messages.append({"role": "assistant", "content": ""})
                messages.append({"role": "user", "content": "Call recall() first, then answer."})
                continue
            return content, recall_trace
        messages.append({"role": "assistant", "content": content, "tool_calls": tc})
        for call in tc:
            fn = call.get("function", {})
            if fn.get("name") != "recall":
                continue
            raw = fn.get("arguments", {})
            if isinstance(raw, str):
                try:
                    args = json.loads(raw)
                except Exception:
                    args = {}
            else:
                args = raw
            query = args.get("query", "")
            top_k = args.get("top_k", 5) or 5
            try:
                top_k = int(top_k)
            except Exception:
                top_k = 5
            top_k = max(1, min(10, top_k))
            results = store.recall(query, top_k=top_k)
            recall_trace.append({"query": query, "results": results})
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", f"recall_{round_idx}"),
                "content": json.dumps(results, default=str)[:3000],
            })
    return "[max_rounds]", recall_trace


# ─── Per-condition full run ─────────────────────────────────────────

def run_condition(cond: str, run_idx: int, sessions: list, probes: list, log):
    t0 = time.time()
    result = {"condition": cond, "run": run_idx, "probes": []}

    if cond == "A_cold":
        # No sessions, just probes.
        for probe in probes:
            ans = answer_probe_A(probe)
            result["probes"].append({"probe_id": probe["id"], "raw_answer": ans})
        result["final_context_chars"] = 0

    elif cond == "B_self_note":
        note = ""
        note_history = []
        for session in sessions:
            note = run_session_B(note, session)
            note_history.append({"session": session["n"], "note_len": len(note)})
            log(f"    session {session['n']}: note_len={len(note)}")
        result["note_history"] = note_history
        result["final_note"] = note
        result["final_context_chars"] = len(note)
        for probe in probes:
            ans = answer_probe_B(probe, note)
            result["probes"].append({"probe_id": probe["id"], "raw_answer": ans})

    elif cond == "C_structured":
        store = MemoryStore()
        session_traces = []
        for session in sessions:
            st = run_session_C(session, store, log)
            session_traces.append({"session": session["n"], "calls_made": st["calls_made"]})
            log(f"    session {session['n']}: remember_calls={st['calls_made']}")
        result["session_traces"] = session_traces
        result["memory_summary"] = store.summary()
        result["final_context_chars"] = store.summary()["total_chars"]
        result["write_log"] = store.write_log
        for probe in probes:
            ans, recall_trace = answer_probe_C(probe, store, log)
            result["probes"].append({
                "probe_id": probe["id"],
                "raw_answer": ans,
                "recall_trace": recall_trace,
            })

    elif cond == "D_markdown":
        dump = ""
        for session in sessions:
            dump = run_session_D(dump, session)
            log(f"    session {session['n']}: dump_len={len(dump)}")
        result["final_markdown_len"] = len(dump)
        result["final_markdown"] = dump
        result["final_context_chars"] = len(dump)
        for probe in probes:
            ans = answer_probe_D(probe, dump)
            result["probes"].append({"probe_id": probe["id"], "raw_answer": ans})

    else:
        raise ValueError(cond)

    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


# ─── Main ───────────────────────────────────────────────────────────

def main():
    # Parse optional condition filter.
    conditions_to_run = CONDITIONS
    if len(sys.argv) > 1:
        if sys.argv[1] in CONDITIONS:
            conditions_to_run = [sys.argv[1]]
        else:
            print(f"Unknown condition: {sys.argv[1]}. Valid: {CONDITIONS}")
            sys.exit(1)

    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    sessions = scenario["sessions"]
    probes = scenario["probes"]

    log_f = open(LOG_PATH, "w", encoding="utf-8", buffering=1)

    def log(msg):
        log_f.write(msg + "\n")
        log_f.flush()
        try: print(msg, flush=True)
        except Exception: pass

    log(f"Phase 3C harness — 5 sessions × {len(conditions_to_run)} conds × {RUNS_PER_CONDITION} runs")
    log(f"model = {MODEL}, scenario = {SCENARIO_PATH.name}")
    log(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load existing results if filter was passed (so we keep previous conds).
    if OUT_PATH.exists() and len(conditions_to_run) < len(CONDITIONS):
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        results = [r for r in existing.get("results", []) if r["condition"] not in conditions_to_run]
    else:
        results = []

    for cond in conditions_to_run:
        for r in range(RUNS_PER_CONDITION):
            log(f"\n>>> {cond} run={r} ({time.strftime('%H:%M:%S')})")
            try:
                res = run_condition(cond, r, sessions, probes, log)
            except Exception as e:
                import traceback
                log(f"    EXCEPTION: {type(e).__name__}: {e}")
                log(traceback.format_exc())
                continue
            results.append(res)
            n_probes = len(res["probes"])
            log(f"    done — {res['elapsed_s']}s, probes={n_probes}, ctx={res.get('final_context_chars', 0)}c")
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"scenario_path": str(SCENARIO_PATH), "results": results}, f, indent=2, default=str)

    log(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"results → {OUT_PATH}")
    log_f.close()


if __name__ == "__main__":
    main()
