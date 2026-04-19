#!/usr/bin/env python3
"""Phase 3E — LongMemEval L1 (oracle) rerun with REAL yantrikdb.

Phase 3D L1 ran against memory_sim.py (Python dict + word-overlap).
This version uses real yantrikdb: client-side MiniLM embeddings,
HNSW vector index, multi-signal scoring, think() after ingest.

Usage:
  python docs/phase3e/phase3e_L1_lme.py [N_PER_TYPE]
"""
from __future__ import annotations

import io
import json
import pathlib
import random
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from yantrikdb_client import YantrikStore

LME_DATA = pathlib.Path("c:/Users/sync/codes/LongMemEval/data/longmemeval_oracle.json")
OUT_JSONL = pathlib.Path(__file__).parent / "hypotheses_L1_ydb.jsonl"
LOG_PATH = pathlib.Path(__file__).parent / "harness_L1_ydb_log.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"
SEED = 42
TOP_K = 10

QUESTION_TYPES = [
    "single-session-user", "single-session-assistant", "single-session-preference",
    "multi-session", "temporal-reasoning", "knowledge-update",
]


def call_qwen(messages, num_predict=400, timeout=180):
    body = {
        "model": MODEL, "messages": messages, "stream": False, "think": False,
        "options": {"temperature": 0.2, "num_predict": num_predict, "num_ctx": 32768},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    last = ""
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode()).get("message", {}).get("content", "")
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
    return f"[ollama error: {last}]"


def ingest_instance(instance: dict, store: YantrikStore) -> int:
    count = 0
    for si, session in enumerate(instance["haystack_sessions"]):
        session_id = instance["haystack_session_ids"][si] if si < len(instance["haystack_session_ids"]) else f"sess_{si}"
        date = instance["haystack_dates"][si] if si < len(instance["haystack_dates"]) else "unknown"
        for ti, turn in enumerate(session):
            content = turn.get("content", "").strip()
            if not content:
                continue
            role = turn.get("role", "unknown")
            key = f"{session_id}:t{ti}:{role}"
            value = f"[{date}] {role}: {content}"[:500]
            store.remember(key, value, si + 1)
            count += 1
    return count


def generate_answer(question: str, retrieved: list[dict], question_date: str) -> str:
    blocks = []
    for i, m in enumerate(retrieved):
        why = m.get("why_retrieved", [])
        blocks.append(f"[memory {i+1}, session {m['session']}, score {m['score']:.3f}, {','.join(why)}] {m['value']}")
    context = "\n".join(blocks) if blocks else "(no relevant memories retrieved)"
    system = (
        "You are answering a question based on retrieved memory snippets from "
        "a user conversation. Memories are sorted by yantrikdb's multi-signal "
        "score (higher = more relevant). Use ONLY the provided memories. If "
        "they don't contain the answer, say so briefly. Give a concise answer."
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
    for qt in QUESTION_TYPES:
        rng.shuffle(by_type[qt])
        picks.extend(by_type[qt][:n_per_type])
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

    log(f"Phase 3E — LongMemEval L1 oracle rerun with REAL yantrikdb")
    log(f"n_per_type={n_per_type} (6 types × {n_per_type} = {6*n_per_type} total)")
    log(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = json.loads(LME_DATA.read_text(encoding="utf-8"))
    subset = sample_subset(data, n_per_type)
    log(f"subset: {len(subset)} instances\n")

    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for idx, inst in enumerate(subset):
            t0 = time.time()
            ns = f"lme_l1_ydb_{inst['question_id']}_{int(time.time())}"
            store = YantrikStore(namespace=ns)
            log(f"[{idx+1}/{len(subset)}] {inst['question_id']} ({inst['question_type']}) — {time.strftime('%H:%M:%S')}")

            ingested = ingest_instance(inst, store)
            # Optional think() after ingest — helps on knowledge-update
            think_res = store.think()
            conflicts = think_res.get("conflicts_found", 0)
            consolidated = think_res.get("consolidation_count", 0)
            log(f"  ingested {ingested} turns | think: conflicts={conflicts} consolidated={consolidated}")

            retrieved = store.recall(inst["question"], top_k=TOP_K)
            log(f"  retrieved {len(retrieved)} (top score {retrieved[0]['score']:.3f} if any)")

            hyp = generate_answer(inst["question"], retrieved, inst.get("question_date", ""))
            elapsed = time.time() - t0
            log(f"  answer: {hyp[:140]!r}")
            log(f"  elapsed: {elapsed:.1f}s")
            entry = {
                "question_id": inst["question_id"],
                "hypothesis": hyp,
                "question_type": inst["question_type"],
                "n_retrieved": len(retrieved),
                "retrieved_preview": [{"session": r["session"], "score": r["score"], "value": r["value"][:100]} for r in retrieved[:5]],
                "think_conflicts": conflicts,
                "think_consolidated": consolidated,
                "elapsed_s": round(elapsed, 1),
            }
            out.write(json.dumps(entry, default=str) + "\n")
            out.flush()

    log(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"hypotheses → {OUT_JSONL}")
    log_f.close()


if __name__ == "__main__":
    main()
