#!/usr/bin/env python3
"""Run v3 pipeline on Scenario 2 (medical case review).

PRE-REGISTERED pipeline per scenario2/prereg.md (committed at 3394391 before
this run). Alias pairs and resolve strategy are fixed; no post-hoc tweaks.

Same phase3e_3c_full_pipeline structure, just with different scenario + alias
list loaded from scenario2/sessions.json.
"""
from __future__ import annotations

import os
os.environ["YDB_TOKEN"] = "ydb_78977ba5690d9c60b979bc15afb57199405472ec9d440942e99ac9ac22c94899"

import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from yantrikdb_client import YantrikStore

SCENARIO_PATH = pathlib.Path(__file__).parent / "sessions.json"
OUT_PATH = pathlib.Path(__file__).parent / "results.json"
LOG_PATH = pathlib.Path(__file__).parent / "run_log.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"
RUNS = 2
REMEMBER_CAP_PER_SESSION = 15


def call_ollama(messages, tools=None, num_predict=600, timeout=300):
    body = {
        "model": MODEL, "messages": messages, "stream": False, "think": False,
        "options": {"temperature": 0.3, "num_predict": num_predict, "num_ctx": 32768},
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    last = ""
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
    return {"__error__": last}


REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Store a fact. Use short snake_case/dotted key (<=60 chars) and "
            "plain-English value (<=120 chars). Up to 15 calls per session. "
            "Same key for same-concept updates."
        ),
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
        },
    },
}


def run_session(session: dict, store: YantrikStore, log) -> dict:
    system = (
        "You are in session " + str(session["n"]) + " of a multi-session intake. "
        "Your ONLY action is to call the `remember` tool for every discrete fact. "
        f"Cap: {REMEMBER_CAP_PER_SESSION} calls max. Value <=120 chars. "
        "Use short keys. Same key for same-concept updates. Do NOT reply in plain text."
    )
    user = (
        f"Session {session['n']}: {session['title']}\n\n{session['narrative']}\n\n"
        f"Store each fact with remember(). Up to {REMEMBER_CAP_PER_SESSION} calls."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls_made = 0
    trace = []
    nudge = f"Call remember(). {{made}}/{REMEMBER_CAP_PER_SESSION} used. Store facts — no text reply."

    for round_idx in range(4):
        if calls_made >= REMEMBER_CAP_PER_SESSION: break
        data = call_ollama(messages, tools=[REMEMBER_TOOL], num_predict=1200)
        if "__error__" in data:
            trace.append({"round": round_idx, "error": data["__error__"]})
            break
        msg = data.get("message", {})
        tc = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        if not tc:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": nudge.format(made=calls_made)})
            continue
        messages.append({"role": "assistant", "content": content, "tool_calls": tc})
        for call in tc:
            if calls_made >= REMEMBER_CAP_PER_SESSION: break
            fn = call.get("function", {})
            if fn.get("name") != "remember": continue
            raw = fn.get("arguments", {})
            args = raw if isinstance(raw, dict) else (json.loads(raw) if isinstance(raw, str) else {})
            key = args.get("key", "")[:60]
            value = args.get("value", "")[:120]
            result = store.remember(key, value, session["n"])
            calls_made += 1
            trace.append({"round": round_idx, "call": calls_made, "key": key})
            messages.append({"role": "tool",
                             "tool_call_id": call.get("id", f"call_{calls_made}"),
                             "content": json.dumps(result)[:500]})

    think_result = store.think()
    return {"calls_made": calls_made, "trace": trace, "think_result": think_result}


def answer_probe(probe: dict, store: YantrikStore) -> tuple[str, list]:
    results = store.recall(probe["q"], top_k=10)
    blocks = []
    for i, m in enumerate(results):
        blocks.append(f"[memory {i+1}, session {m['session']}, score {m['score']:.3f}] {m['key']}: {m['value']}")
    context = "\n".join(blocks) if blocks else "(no relevant memories retrieved)"
    system = (
        "Answer the user's question using ONLY the retrieved memories below. "
        "If multiple memories describe the same fact with different values, "
        "the one with the HIGHEST score and/or the LATEST session is most "
        "likely current. Conflicts have been resolved by the memory "
        "substrate — trust the top-scored result for current state. "
        "Format: Answer: <X>\nSource: Session <N>"
    )
    user = f"Retrieved memories:\n{context}\n\nQuestion: {probe['q']}"
    data = call_ollama([{"role": "system", "content": system},
                        {"role": "user", "content": user}], num_predict=300)
    answer = data.get("message", {}).get("content", "") if "__error__" not in data else f"[ollama error]"
    return answer, results


def run_one(run_idx, scenario, log):
    t0 = time.time()
    run_tag = f"s2_v3_{run_idx}_{int(time.time())}"
    store = YantrikStore(namespace=f"scenario2_{run_tag}")

    # === Step 1: ingest 5 sessions ===
    session_traces = []
    for session in scenario["sessions"]:
        st = run_session(session, store, log)
        tr = st["think_result"]
        session_traces.append({
            "session": session["n"], "calls_made": st["calls_made"],
            "think_conflicts": tr.get("conflicts_found", 0),
            "think_consolidated": tr.get("consolidation_count", 0),
        })
        log(f"    session {session['n']}: remember={st['calls_made']} "
            f"conflicts={tr.get('conflicts_found', 0)} consolidated={tr.get('consolidation_count', 0)}")

    # === Step 2: alias relations (PRE-REGISTERED list from scenario metadata) ===
    relate_results = []
    for triple in scenario["alias_pairs_for_pipeline"]:
        entity, rel, target = triple
        r = store.relate(entity, target, rel)
        relate_results.append({"entity": entity, "target": target, "rel": rel, "ok": "__error__" not in r})
    log(f"    relate: {sum(1 for r in relate_results if r['ok'])}/{len(relate_results)} alias relations added")

    # === Step 3: conflict resolution (PRE-REGISTERED policy: latest-wins) ===
    resolve_result = store.resolve_all_latest_wins()
    log(f"    resolve: {resolve_result['resolved']}/{resolve_result['scanned']} conflicts resolved "
        f"({resolve_result['errors']} errors)")

    # === Step 4: probe phase ===
    probe_results = []
    for probe in scenario["probes"]:
        ans, retrieved = answer_probe(probe, store)
        probe_results.append({
            "probe_id": probe["id"], "raw_answer": ans,
            "retrieved": [{"key": r["key"], "value": r["value"][:80],
                           "session": r["session"], "score": r["score"]} for r in retrieved[:5]],
        })

    return {
        "condition": "scenario2_v3_prereg",
        "run": run_idx, "elapsed_s": round(time.time() - t0, 1),
        "session_traces": session_traces,
        "relate_results": relate_results, "resolve_result": resolve_result,
        "probes": probe_results,
    }


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8", buffering=1)
    def log(msg):
        log_f.write(msg + "\n"); log_f.flush()
        try: print(msg, flush=True)
        except Exception: pass

    log(f"Phase 3E Scenario 2 — v3 pipeline (PRE-REGISTERED, see prereg.md)")
    log(f"scenario: {SCENARIO_PATH.name}, runs={RUNS}")
    log(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    results = []
    for r in range(RUNS):
        log(f"\n>>> scenario2 v3 run={r} ({time.strftime('%H:%M:%S')})")
        try:
            res = run_one(r, scenario, log)
        except Exception as e:
            import traceback
            log(f"    EXCEPTION: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            continue
        results.append(res)
        log(f"    done — {res['elapsed_s']}s, probes={len(res['probes'])}")
        OUT_PATH.write_text(json.dumps({"results": results}, indent=2, default=str), encoding="utf-8")

    log(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_f.close()


if __name__ == "__main__":
    main()
