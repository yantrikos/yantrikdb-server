#!/usr/bin/env python3
"""Phase 3E v3: full yantrikdb pipeline on Phase 3C scenario.

Previous Phase 3E used ONLY remember + recall + think (~10% of yantrikdb's
surface area). This run adds:
  - explicit `/v1/relate` calls for known entity aliases (Aurora/Aurora-lite,
    hard_budget/target_budget, Nexus-North/Nexus-South, phase2_ship_date/
    phase2_announce_date)
  - post-ingest `scan_conflicts` + `resolve_all_latest_wins` — every conflict
    think() detected is resolved by keeping the newer memory
  - Uses fresh DB (fresh_p3e_v3) for isolation

Compares to v1 (0.850 current-DB, 0.917 fresh-DB, think-on) to see if
resolve_conflict closes the remaining gap and if relate() fixes alias
disambiguation.

Pre-registered expectation (per user's scale-dependent framing critique):
  - stale_error_rate should drop from 0.20 → <= 0.10 (conflict resolution)
  - alias_disambiguation should improve from 0.75 → >= 0.875 (explicit relate)
  - overall score should be >= 0.917 (match or exceed fresh-DB think-on)

If numbers don't shift, it suggests at this scale features don't compose.
If they shift substantially, the findings post framing changes materially.
"""
from __future__ import annotations

import os
# Use fresh_p3e_v3 DB
os.environ["YDB_TOKEN"] = "ydb_78977ba5690d9c60b979bc15afb57199405472ec9d440942e99ac9ac22c94899"

import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from yantrikdb_client import YantrikStore

SCENARIO_PATH = pathlib.Path(__file__).parent.parent / "phase3c" / "scenario" / "sessions.json"
OUT_PATH = pathlib.Path(__file__).parent / "results_3c_full_pipeline.json"
LOG_PATH = pathlib.Path(__file__).parent / "harness_3c_full_pipeline_log.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"
RUNS = 2
REMEMBER_CAP_PER_SESSION = 15

# Known entity-alias pairs from the scenario (hand-curated — these are the
# distinctions the probes test). In production yantrikdb would extract these
# automatically via entity resolution; here we pre-populate.
KNOWN_DISTINCT_ENTITIES = [
    ("Project Aurora", "distinct_from", "Aurora-lite"),
    ("Aurora-lite", "distinct_from", "Project Aurora"),
    ("hard_budget_cap", "distinct_from", "target_budget"),
    ("target_budget", "distinct_from", "hard_budget_cap"),
    ("Nexus-North", "distinct_from", "Nexus-South"),
    ("Nexus-South", "distinct_from", "Nexus-North"),
    ("phase2_ship_date", "distinct_from", "phase2_announce_date"),
    ("phase2_announce_date", "distinct_from", "phase2_ship_date"),
]


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
            "Same key for same-concept updates — the memory system tracks "
            "supersession automatically via think() + conflict resolution."
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


def run_session_C(session: dict, store: YantrikStore, log) -> dict:
    """Ingest via remember() + think() — unchanged from v1."""
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
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": nudge.format(made=calls_made)})
            continue

        messages.append({"role": "assistant", "content": content, "tool_calls": tc})
        for call in tc:
            if calls_made >= REMEMBER_CAP_PER_SESSION:
                break
            fn = call.get("function", {})
            if fn.get("name") != "remember":
                continue
            raw = fn.get("arguments", {})
            args = raw if isinstance(raw, dict) else (json.loads(raw) if isinstance(raw, str) else {})
            key = args.get("key", "")[:60]
            value = args.get("value", "")[:120]
            result = store.remember(key, value, session["n"])
            calls_made += 1
            trace.append({"round": round_idx, "call": calls_made, "key": key, "value": value[:80]})
            messages.append({"role": "tool",
                             "tool_call_id": call.get("id", f"call_{calls_made}"),
                             "content": json.dumps(result)[:500]})

    think_result = store.think()
    return {"calls_made": calls_made, "trace": trace, "think_result": think_result}


def answer_probe_C(probe: dict, store: YantrikStore) -> tuple[str, list]:
    """Same recall+generate as v1."""
    results = store.recall(probe["q"], top_k=10)
    blocks = []
    for i, m in enumerate(results):
        blocks.append(f"[memory {i+1}, session {m['session']}, score {m['score']:.3f}] {m['key']}: {m['value']}")
    context = "\n".join(blocks) if blocks else "(no relevant memories retrieved)"
    system = (
        "Answer the user's question using ONLY the retrieved memories below. "
        "Memories are sorted by yantrikdb's multi-signal score. If multiple "
        "memories describe the same fact with different values (because it "
        "was revised), the one with the HIGHEST score and/or the LATEST "
        "session is most likely the current value. Conflicts between "
        "memories have already been resolved by the memory substrate — "
        "trust the top-scored result for current state. "
        "Format: Answer: <X>\nSource: Session <N>"
    )
    user = f"Retrieved memories:\n{context}\n\nQuestion: {probe['q']}"
    data = call_ollama([{"role": "system", "content": system},
                        {"role": "user", "content": user}], num_predict=300)
    answer = data.get("message", {}).get("content", "") if "__error__" not in data else f"[ollama error]"
    return answer, results


def run_one(run_idx, sessions, probes, log):
    t0 = time.time()
    run_tag = f"full_pipeline_{run_idx}_{int(time.time())}"
    store = YantrikStore(namespace=f"phase3e_v3_{run_tag}")

    # === Step 1: ingest 5 sessions via remember + think ===
    session_traces = []
    for session in sessions:
        st = run_session_C(session, store, log)
        tr = st["think_result"]
        session_traces.append({
            "session": session["n"],
            "calls_made": st["calls_made"],
            "think_conflicts": tr.get("conflicts_found", 0),
            "think_consolidated": tr.get("consolidation_count", 0),
        })
        log(f"    session {session['n']}: remember={st['calls_made']} "
            f"conflicts={tr.get('conflicts_found', 0)} consolidated={tr.get('consolidation_count', 0)}")

    # === Step 2: NEW — explicit alias relations ===
    relate_results = []
    for entity, rel, target in KNOWN_DISTINCT_ENTITIES:
        r = store.relate(entity, target, rel)
        relate_results.append({"entity": entity, "target": target, "rel": rel, "ok": "__error__" not in r})
    log(f"    relate: {sum(1 for r in relate_results if r['ok'])}/{len(relate_results)} alias relations added")

    # === Step 3: NEW — post-ingest conflict resolution (latest-wins) ===
    resolve_result = store.resolve_all_latest_wins()
    log(f"    resolve: {resolve_result['resolved']}/{resolve_result['scanned']} conflicts resolved "
        f"({resolve_result['errors']} errors)")

    # === Step 4: probe phase ===
    probe_results = []
    for probe in probes:
        ans, retrieved = answer_probe_C(probe, store)
        probe_results.append({
            "probe_id": probe["id"],
            "raw_answer": ans,
            "retrieved": [{"key": r["key"], "value": r["value"][:80],
                           "session": r["session"], "score": r["score"]} for r in retrieved[:5]],
        })

    final_stats = store.stats()
    return {
        "condition": "C_yantrikdb_full_pipeline",
        "run": run_idx,
        "elapsed_s": round(time.time() - t0, 1),
        "session_traces": session_traces,
        "relate_results": relate_results,
        "resolve_result": resolve_result,
        "final_stats": final_stats,
        "probes": probe_results,
        "final_context_chars": final_stats.get("active_memories", 0) * 100,
    }


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8", buffering=1)

    def log(msg):
        log_f.write(msg + "\n")
        log_f.flush()
        try: print(msg, flush=True)
        except Exception: pass

    log(f"Phase 3E v3: FULL yantrikdb pipeline on 3C scenario")
    log(f"features: remember + think + relate (alias) + resolve_conflict (latest-wins)")
    log(f"DB: fresh_p3e_v3, runs={RUNS}")
    log(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    sessions = scenario["sessions"]
    probes = scenario["probes"]

    results = []
    for r in range(RUNS):
        log(f"\n>>> full_pipeline run={r} ({time.strftime('%H:%M:%S')})")
        try:
            res = run_one(r, sessions, probes, log)
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
