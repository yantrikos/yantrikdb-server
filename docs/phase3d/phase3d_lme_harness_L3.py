#!/usr/bin/env python3
"""Phase 3D Level 3: LongMemEval_s (~53 sessions, ~550 turns, ~120k tokens
per haystack) with the same Phase 3C memory simulator + Qwen 3.6.

Difference from L1 (phase3d_lme_harness.py):
  - Dataset: longmemeval_s_cleaned.json instead of longmemeval_oracle.json
  - top_k: 20 (not 10) — bigger haystack, need more recall
  - Otherwise identical: same memory_sim, same Qwen, same seed=42 subset

This is where retrieval actually matters. Oracle hides retrieval difficulty
by only including evidence sessions; _s puts the needle (1-few evidence
sessions) in a haystack of ~50 irrelevant ones.
"""
from __future__ import annotations

import io
import json
import pathlib
import random
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "phase3c"))
from memory_sim import MemoryStore

LME_DATA = pathlib.Path("c:/Users/sync/codes/LongMemEval/data/longmemeval_s_cleaned.json")
OUT_JSONL = pathlib.Path(__file__).parent / "hypotheses_L3.jsonl"
LOG_PATH = pathlib.Path(__file__).parent / "harness_L3_log.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"
SEED = 42
TOP_K = 20

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


def call_qwen(messages, num_predict=400, timeout=240):
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": num_predict, "num_ctx": 32768},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode()).get("message", {}).get("content", "")
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
    return f"[ollama error: {last}]"


def ingest_instance(instance: dict, store: MemoryStore) -> None:
    for si, session in enumerate(instance["haystack_sessions"]):
        session_id = instance["haystack_session_ids"][si] if si < len(instance["haystack_session_ids"]) else f"sess_{si}"
        date = instance["haystack_dates"][si] if si < len(instance["haystack_dates"]) else "unknown"
        for ti, turn in enumerate(session):
            content = turn.get("content", "").strip()
            if not content:
                continue
            role = turn.get("role", "unknown")
            key = f"{session_id}:t{ti}:{role}"
            value = f"[{date}] {role}: {content}"
            store.remember(key, value, si + 1)


def generate_answer(question: str, retrieved: list[dict], question_date: str) -> str:
    blocks = []
    for i, m in enumerate(retrieved):
        blocks.append(f"[memory {i+1}, session {m['session']}] {m['key']}: {m['value']}")
    context = "\n".join(blocks) if blocks else "(no relevant memories retrieved)"
    system = (
        "You are answering a question based on retrieved memory snippets from "
        "a long-running user conversation. The conversation has many sessions "
        "and the retrieved memories are the ones most relevant to the question "
        "by keyword overlap. Use ONLY the provided memories. If they do not "
        "contain the answer, say so briefly. Give a concise answer."
    )
    user = (
        f"Today's date: {question_date}\n\n"
        f"Retrieved memories (top-{len(retrieved)}):\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return call_qwen([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], num_predict=500).strip()


def sample_subset(data: list, n_per_type: int, seed: int = SEED) -> list:
    rng = random.Random(seed)
    by_type = {qt: [] for qt in QUESTION_TYPES}
    for inst in data:
        if inst["question_type"] in by_type:
            by_type[inst["question_type"]].append(inst)
    picks = []
    for qt, items in by_type.items():
        rng.shuffle(items)
        picks.extend(items[:n_per_type])
    rng.shuffle(picks)
    return picks


def main():
    n_per_type = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    log_f = open(LOG_PATH, "w", encoding="utf-8", buffering=1)

    def log(msg):
        log_f.write(msg + "\n")
        log_f.flush()
        try: print(msg, flush=True)
        except Exception: pass

    log(f"Phase 3D Level 3 — LongMemEval_s + memory_sim + Qwen 3.6")
    log(f"n_per_type={n_per_type}, TOP_K={TOP_K}, seed={SEED}")
    log(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = json.loads(LME_DATA.read_text(encoding="utf-8"))
    log(f"loaded {len(data)} instances from {LME_DATA.name}")
    subset = sample_subset(data, n_per_type)
    log(f"subset: {len(subset)} instances\n")

    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for idx, inst in enumerate(subset):
            t0 = time.time()
            log(f"[{idx+1}/{len(subset)}] {inst['question_id']} ({inst['question_type']}) — {time.strftime('%H:%M:%S')}")
            store = MemoryStore(value_cap=2000, key_cap=120)
            ingest_instance(inst, store)
            summary = store.summary()
            retrieved = store.recall(inst["question"], top_k=TOP_K)
            log(f"  stored {summary['n_memories']} turns | retrieved {len(retrieved)} (top score {retrieved[0]['score'] if retrieved else 'N/A'})")

            # Check if any retrieved memory came from an answer session.
            answer_sessions = set(inst.get("answer_session_ids", []))
            retrieved_sessions = set(r["key"].split(":")[0] for r in retrieved)
            recall_hit = bool(answer_sessions & retrieved_sessions)
            log(f"  answer_sessions: {list(answer_sessions)[:3]}; retrieved_sessions overlap: {recall_hit}")

            hyp = generate_answer(inst["question"], retrieved, inst.get("question_date", ""))
            elapsed = time.time() - t0
            log(f"  answer: {hyp[:140]!r}")
            log(f"  elapsed: {elapsed:.1f}s")
            entry = {
                "question_id": inst["question_id"],
                "hypothesis": hyp,
                "n_retrieved": len(retrieved),
                "retrieved_keys": [r["key"] for r in retrieved],
                "recall_hit_answer_session": recall_hit,
                "elapsed_s": round(elapsed, 1),
            }
            out.write(json.dumps(entry, default=str) + "\n")
            out.flush()

    log(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"hypotheses → {OUT_JSONL}")
    log_f.close()


if __name__ == "__main__":
    main()
