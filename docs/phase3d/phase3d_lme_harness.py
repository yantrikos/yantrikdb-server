#!/usr/bin/env python3
"""Phase 3D — Level 1: LongMemEval smoke with the Phase 3C memory simulator.

Goal: get a FIRST number on LongMemEval's oracle subset with our baseline
structured-memory retrieval (word-overlap, no embeddings, no yantrikdb,
no temporal logic). Establishes a floor. If Level 1 gets a terrible
number, Level 2 adds sentence-transformer embeddings. Level 3 adds real
yantrikdb. Level 4 adds RFC 006/008 temporal substrate.

Each LongMemEval instance has:
  - question_id, question_type, question, answer
  - haystack_sessions: list of sessions (each a list of turns)
Oracle subset = only evidence sessions included (~3 sessions avg).

Pipeline per question:
  1. Ingest every turn from every session into MemoryStore as a separate
     memory, keyed by a short summary of the turn's content.
  2. Recall top-k (default 10) for the question.
  3. Feed retrieved memories + question to Qwen 3.6 → hypothesis.
  4. Save to jsonl for eval.

Usage:
  python docs/phase3d/phase3d_lme_harness.py [N_SAMPLES_PER_TYPE]
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

LME_DATA = pathlib.Path("c:/Users/sync/codes/LongMemEval/data/longmemeval_oracle.json")
OUT_JSONL = pathlib.Path(__file__).parent / "hypotheses.jsonl"
LOG_PATH = pathlib.Path(__file__).parent / "harness_log.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"
SEED = 42
TOP_K = 10

# Question types to sample from; the 6 LongMemEval types
QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


def call_qwen(messages, num_predict=300, timeout=180):
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
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                return data.get("message", {}).get("content", "")
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
    return f"[ollama error: {last}]"


def ingest_instance(instance: dict, store: MemoryStore) -> None:
    """Store every turn as a memory. Key = '{session_id}:t{turn_idx}:{role}'."""
    for si, session in enumerate(instance["haystack_sessions"]):
        session_id = instance["haystack_session_ids"][si] if si < len(instance["haystack_session_ids"]) else f"sess_{si}"
        date = instance["haystack_dates"][si] if si < len(instance["haystack_dates"]) else "unknown"
        for ti, turn in enumerate(session):
            role = turn.get("role", "unknown")
            content = turn.get("content", "").strip()
            if not content:
                continue
            key = f"{session_id}:t{ti}:{role}"
            # Include date + role in the value so retrieval sees temporal/speaker cues.
            value = f"[{date}] {role}: {content}"
            store.remember(key, value, si + 1)


def generate_answer(question: str, retrieved: list[dict], question_date: str) -> str:
    """Feed retrieved memories + question to Qwen, get hypothesis."""
    context_blocks = []
    for i, m in enumerate(retrieved):
        context_blocks.append(f"[memory {i+1}, session {m['session']}] {m['key']}: {m['value']}")
    context = "\n".join(context_blocks) if context_blocks else "(no relevant memories retrieved)"
    system = (
        "You are answering a question based on retrieved memory snippets from "
        "a long-running user conversation. Use ONLY the memories provided. If "
        "the memories do not contain the answer, say so briefly. Give a "
        "concise answer; do not explain your reasoning extensively."
    )
    user = (
        f"Today's date: {question_date}\n\n"
        f"Retrieved memories (top-{len(retrieved)}):\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return call_qwen([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], num_predict=400).strip()


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

    log(f"Phase 3D Level 1 — LongMemEval oracle + memory_sim + Qwen 3.6")
    log(f"n_per_type={n_per_type} (6 types × {n_per_type} = {6*n_per_type} total)")
    log(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = json.loads(LME_DATA.read_text(encoding="utf-8"))
    log(f"loaded {len(data)} instances from {LME_DATA.name}")
    subset = sample_subset(data, n_per_type)
    log(f"subset: {len(subset)} instances\n")

    hyps = []
    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for idx, inst in enumerate(subset):
            t0 = time.time()
            log(f"[{idx+1}/{len(subset)}] {inst['question_id']} ({inst['question_type']}) — {time.strftime('%H:%M:%S')}")
            store = MemoryStore(value_cap=2000, key_cap=120)
            ingest_instance(inst, store)
            summary = store.summary()
            log(f"  stored {summary['n_memories']} turns as memories")
            retrieved = store.recall(inst["question"], top_k=TOP_K)
            log(f"  retrieved {len(retrieved)} memories (top score {retrieved[0]['score'] if retrieved else 'N/A'})")
            hyp = generate_answer(inst["question"], retrieved, inst.get("question_date", ""))
            elapsed = time.time() - t0
            log(f"  answer: {hyp[:140]!r}")
            log(f"  elapsed: {elapsed:.1f}s")
            entry = {
                "question_id": inst["question_id"],
                "hypothesis": hyp,
                "n_retrieved": len(retrieved),
                "retrieved_keys": [r["key"] for r in retrieved],
                "elapsed_s": round(elapsed, 1),
            }
            hyps.append(entry)
            out.write(json.dumps(entry, default=str) + "\n")
            out.flush()

    log(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"hypotheses → {OUT_JSONL}")
    log_f.close()


if __name__ == "__main__":
    main()
