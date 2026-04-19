#!/usr/bin/env python3
"""Phase 3E — Phase 3C rerun with REAL yantrikdb.

Same scenario, same 4 conditions, same 15 probes, same seed. Only
condition C changes: instead of the phase3c memory_sim.py toy
(Python dict + Dice word-overlap, no think(), no conflict scan, no
multi-signal scoring), use real yantrikdb via HTTP with:
  - HNSW vector index (client-side MiniLM embeddings)
  - Multi-signal scoring (vector × decay × importance × graph × feedback)
  - think() called between each session (consolidation + conflict scan
    + pattern mining + trigger eval)

This is the experiment that should have been run before Phase 3C was
committed. The "40% stale-rate" finding was measured against a
stripped-down simulator, not yantrikdb.

Usage:
  python docs/phase3e/phase3e_3c_rerun.py [condition]
    condition: A_cold | B_self_note | C_yantrikdb | D_markdown | all
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
from yantrikdb_client import YantrikStore

# Reuse phase3c's scenario and conditions A/B/D exactly
SCENARIO_PATH = pathlib.Path(__file__).parent.parent / "phase3c" / "scenario" / "sessions.json"
_TAG = "_think_off" if "--think-off" in sys.argv else ("_freshdb" if "--fresh-db" in sys.argv else "")
OUT_PATH = pathlib.Path(__file__).parent / f"results_3c_rerun{_TAG}.json"
LOG_PATH = pathlib.Path(__file__).parent / f"harness_3c_rerun{_TAG}_log.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"

RUNS_PER_CONDITION = 2
CONDITIONS = ["A_cold", "B_self_note", "C_yantrikdb", "D_markdown"]

NOTE_CAP_CHARS = 1500
MARKDOWN_CAP_CHARS = 7500
REMEMBER_CAP_PER_SESSION = 15


# ── Ollama plumbing (same as phase3c harness) ──────────────────────

def call_ollama(messages, num_predict=600, timeout=300):
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3, "num_predict": num_predict, "num_ctx": 32768},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    last_err = ""
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode()).get("message", {}).get("content", "")
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0)
    return f"[ollama error: {last_err}]"


# ── Condition runners (A/B/D identical to phase3c) ─────────────────

def run_session_B(previous_note: str, session: dict) -> str:
    """B: rolling self-note, hard-capped at 1500 chars, rewritten each session."""
    system = (
        "You are keeping a running note across multiple sessions. At each "
        f"session you MUST rewrite your note from scratch, staying under "
        f"{NOTE_CAP_CHARS} characters. Include facts with originating sessions, "
        "explicit supersession annotations, entity names verbatim, and enough "
        "detail to answer retrospective questions. Respond with ONLY the "
        "updated note text — no commentary."
    )
    user = (
        f"Previous note (empty if first session):\n---\n{previous_note}\n---\n\n"
        f"New session {session['n']}: {session['title']}\n\n{session['narrative']}\n\n"
        f"Write the updated note (≤{NOTE_CAP_CHARS} chars)."
    )
    note = call_ollama([{"role": "system", "content": system}, {"role": "user", "content": user}], num_predict=800)
    return note[:NOTE_CAP_CHARS]


def run_session_C_yantrikdb(session: dict, store: YantrikStore, log) -> dict:
    """C: real yantrikdb. Qwen decides what to remember via tool-calls
    with same cap (15 per session, 120 char values). After each session,
    call think() to trigger consolidation + conflict scan."""
    REMEMBER_TOOL = {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Store a fact. Use short snake_case/dotted key (<=60 chars) and "
                "plain-English value (<=120 chars). Up to 15 calls per session. "
                "Same key for same-concept updates so the system can track "
                "supersession."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
    }
    system = (
        "You are in session " + str(session["n"]) + " of a multi-session intake. "
        "Your ONLY action is to call the `remember` tool for every discrete fact. "
        f"Cap: {REMEMBER_CAP_PER_SESSION} calls max. Value ≤120 chars. Use short "
        "keys. If a value supersedes a prior one, use the SAME key — the memory "
        "system tracks supersession. Do NOT reply in plain text."
    )
    user = (
        f"Session {session['n']}: {session['title']}\n\n{session['narrative']}\n\n"
        f"Store each fact with remember(). Up to {REMEMBER_CAP_PER_SESSION} calls. "
        "No commentary."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    calls_made = 0
    trace = []
    nudge = (
        f"Call remember(). {{calls_made}} of {REMEMBER_CAP_PER_SESSION} used. "
        "Store the facts — no text reply."
    )

    for round_idx in range(4):
        if calls_made >= REMEMBER_CAP_PER_SESSION:
            break
        # Call Ollama with tool
        body = {
            "model": MODEL, "messages": messages, "stream": False, "think": False,
            "tools": [REMEMBER_TOOL],
            "options": {"temperature": 0.3, "num_predict": 1200, "num_ctx": 32768},
        }
        req = urllib.request.Request(
            OLLAMA_URL, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            trace.append({"round": round_idx, "error": f"{type(exc).__name__}: {exc}"})
            break

        msg = data.get("message", {})
        tc = msg.get("tool_calls") or []
        content = msg.get("content") or ""

        if not tc:
            trace.append({"round": round_idx, "no_tool": content[:200]})
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": nudge.format(calls_made=calls_made)})
            continue

        messages.append({"role": "assistant", "content": content, "tool_calls": tc})
        for call in tc:
            if calls_made >= REMEMBER_CAP_PER_SESSION:
                break
            fn = call.get("function", {})
            if fn.get("name") != "remember":
                continue
            raw = fn.get("arguments", {})
            args = raw if isinstance(raw, dict) else json.loads(raw) if isinstance(raw, str) else {}
            key = args.get("key", "")[:60]
            value = args.get("value", "")[:120]
            result = store.remember(key, value, session["n"])
            calls_made += 1
            trace.append({"round": round_idx, "call": calls_made, "key": key, "value": value[:80]})
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", f"call_{calls_made}"),
                "content": json.dumps(result)[:500],
            })

    # After ingesting, run think() for consolidation + conflict scan — UNLESS --think-off
    if "--think-off" in sys.argv:
        think_result = {"skipped": True}
    else:
        think_result = store.think()
    return {"calls_made": calls_made, "trace": trace, "think_result": think_result}


def run_session_D(dump: str, session: dict) -> str:
    chunk = f"\n\n## Session {session['n']} — {session['title']}\n\n{session['narrative']}"
    dump = dump + chunk
    if len(dump) > MARKDOWN_CAP_CHARS:
        dump = dump[-MARKDOWN_CAP_CHARS:]
    return dump


# ── Probe phase ────────────────────────────────────────────────────

def answer_probe_with_context(probe: dict, context: str, label: str) -> str:
    if label == "cold":
        system = (
            "Answer the question if you know the answer; otherwise say 'UNKNOWN'. "
            "Format: Answer: <X>\nSource: Session <N>. Do not guess."
        )
        user = f"Question: {probe['q']}"
    else:
        system = (
            "Answer the user's question using ONLY the context below. If you "
            "cannot find the answer, say 'UNKNOWN'. Always format response as "
            "exactly two lines:\nAnswer: <X>\nSource: Session <N>\n"
            "If multiple values appear for the same fact (because it was "
            "revised), give the LATEST value."
        )
        user = f"CONTEXT:\n---\n{context}\n---\n\nQuestion: {probe['q']}"
    return call_ollama([{"role": "system", "content": system}, {"role": "user", "content": user}], num_predict=300)


def answer_probe_C_yantrikdb(probe: dict, store: YantrikStore) -> tuple[str, list]:
    """C: use yantrikdb's recall directly (no tool-calling loop; just call recall)."""
    results = store.recall(probe["q"], top_k=10)
    # Format retrieved memories for the LLM
    blocks = []
    for i, m in enumerate(results):
        blocks.append(f"[memory {i+1}, session {m['session']}, score {m['score']:.3f}] {m['key']}: {m['value']}")
    context = "\n".join(blocks) if blocks else "(no relevant memories retrieved)"
    system = (
        "Answer the user's question using ONLY the retrieved memories below. "
        "The memories are already sorted by yantrikdb's multi-signal scoring — "
        "the first one is most relevant. If multiple memories describe the same "
        "fact with different values (because it was revised), the one with the "
        "HIGHEST score and/or the LATEST session is most likely the current "
        "value. Format: Answer: <X>\nSource: Session <N>"
    )
    user = f"Retrieved memories:\n{context}\n\nQuestion: {probe['q']}"
    answer = call_ollama([{"role": "system", "content": system}, {"role": "user", "content": user}], num_predict=300)
    return answer, results


# ── Per-condition run ──────────────────────────────────────────────

def run_condition(cond: str, run_idx: int, sessions: list, probes: list, log):
    t0 = time.time()
    result = {"condition": cond, "run": run_idx, "probes": []}
    run_tag = f"{cond}_{run_idx}_{int(time.time())}"

    if cond == "A_cold":
        for probe in probes:
            ans = answer_probe_with_context(probe, "", "cold")
            result["probes"].append({"probe_id": probe["id"], "raw_answer": ans})
        result["final_context_chars"] = 0

    elif cond == "B_self_note":
        note = ""
        for session in sessions:
            note = run_session_B(note, session)
            log(f"    session {session['n']}: note_len={len(note)}")
        result["final_note"] = note
        result["final_context_chars"] = len(note)
        for probe in probes:
            ans = answer_probe_with_context(probe, note, "note")
            result["probes"].append({"probe_id": probe["id"], "raw_answer": ans})

    elif cond == "C_yantrikdb":
        store = YantrikStore(namespace=f"phase3e_3c_{run_tag}")
        session_traces = []
        for session in sessions:
            st = run_session_C_yantrikdb(session, store, log)
            session_traces.append({
                "session": session["n"],
                "calls_made": st["calls_made"],
                "think_conflicts": st["think_result"].get("conflicts_found", 0),
                "think_consolidation": st["think_result"].get("consolidation_count", 0),
                "think_triggers_n": len(st["think_result"].get("triggers", [])),
            })
            log(f"    session {session['n']}: remembered={st['calls_made']} "
                f"think: conflicts={st['think_result'].get('conflicts_found', 0)} "
                f"consolidated={st['think_result'].get('consolidation_count', 0)} "
                f"triggers={len(st['think_result'].get('triggers', []))}")
        result["session_traces"] = session_traces
        for probe in probes:
            ans, retrieved = answer_probe_C_yantrikdb(probe, store)
            result["probes"].append({
                "probe_id": probe["id"],
                "raw_answer": ans,
                "retrieved": [{"key": r["key"], "value": r["value"][:80], "session": r["session"], "score": r["score"], "why": r.get("why_retrieved", [])} for r in retrieved[:5]],
            })
        final_stats = store.stats()
        result["final_yantrikdb_stats"] = final_stats
        result["final_context_chars"] = final_stats.get("active_memories", 0) * 100  # rough estimate

    elif cond == "D_markdown":
        dump = ""
        for session in sessions:
            dump = run_session_D(dump, session)
            log(f"    session {session['n']}: dump_len={len(dump)}")
        result["final_markdown"] = dump
        result["final_context_chars"] = len(dump)
        for probe in probes:
            ans = answer_probe_with_context(probe, dump, "markdown")
            result["probes"].append({"probe_id": probe["id"], "raw_answer": ans})

    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main():
    cond_filter = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "all" else None
    conds_to_run = [cond_filter] if cond_filter else CONDITIONS
    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    sessions = scenario["sessions"]
    probes = scenario["probes"]

    log_f = open(LOG_PATH, "a", encoding="utf-8", buffering=1)

    def log(msg):
        log_f.write(msg + "\n")
        log_f.flush()
        try: print(msg, flush=True)
        except Exception: pass

    log(f"\nPhase 3E Phase 3C rerun with REAL yantrikdb")
    log(f"running: {conds_to_run}, runs each: {RUNS_PER_CONDITION}")
    log(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Preserve prior results if filter applied
    if OUT_PATH.exists() and cond_filter:
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        results = [r for r in existing.get("results", []) if r["condition"] != cond_filter]
    else:
        results = []

    for cond in conds_to_run:
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
            log(f"    done — {res['elapsed_s']}s, probes={len(res['probes'])}, ctx_chars={res.get('final_context_chars', 0)}")
            OUT_PATH.write_text(json.dumps({"results": results}, indent=2, default=str), encoding="utf-8")

    log(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"results → {OUT_PATH}")
    log_f.close()


if __name__ == "__main__":
    main()
