#!/usr/bin/env python3
"""Phase 3D L4 — scaling test: find the empirical tier transition point.

Method:
  - Start from longmemeval_s instances (baseline: ~550 turns per haystack).
  - For each target scale factor K in {1x, 2x, 5x, 10x, 20x, 50x}, amplify
    the haystack by appending distractor sessions sampled from OTHER
    instances' haystacks. Evidence sessions stay fixed so the "correct
    answer" remains the same; only the retrieval difficulty increases.
  - At each scale, measure:
      * recall@20: did the answer session appear in the top-20 retrieved?
      * answer accuracy: did Qwen produce a correct answer?
  - Fit the degradation curve; find where recall/accuracy actually break.

Hypothesis under test (the "tiered memory strategy" thesis):
  - At N ~= 500-5000 turns: plain retrieval + Qwen should stay high (tier 2).
  - At N ~= 5000-20000 turns: recall@20 should visibly degrade (tier 3).
  - At N >= 50000 turns: recall@20 should collapse (tier 4, where
    structured substrate is needed).

Results feed into the "tiered strategy" writeup. If degradation never
happens, the entire memory-substrate case is weaker than claimed. If
degradation happens sharply at some N, we've found the transition where
a structured substrate like yantrikdb should be tested next.

Runtime: ~15-20 min. 10 instances × 6 scales × (ingest + retrieve + generate).
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
OUT_JSONL = pathlib.Path(__file__).parent / "hypotheses_L4.jsonl"
LOG_PATH = pathlib.Path(__file__).parent / "harness_L4_log.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"
SEED = 42
TOP_K = 20

# Scale factors: 1x (base ~550 turns), up to 50x (~27k turns)
SCALE_FACTORS = [1, 2, 5, 10, 20, 50]
# Pick 10 instances — bias toward types where retrieval matters.
# Skip single-session-* since they're all at ceiling already.
TARGET_TYPES = ["multi-session", "temporal-reasoning", "knowledge-update"]
N_PER_TYPE = 3  # = 9 instances total

# We need enough distractor sessions for 50x amplification. At ~550 base turns,
# 50x means 27,500 turns. Each session avgs ~10 turns, so need ~2750 extra sessions.
# _s has 500 instances × ~53 sessions = 26,500 total sessions — plenty.


def call_qwen(messages, num_predict=500, timeout=300):
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


def build_distractor_pool(all_instances: list, exclude_qid: str) -> list[tuple[str, str, list]]:
    """Collect (session_id, date, turns) tuples from all OTHER instances."""
    pool = []
    for inst in all_instances:
        if inst["question_id"] == exclude_qid:
            continue
        for i, sess in enumerate(inst["haystack_sessions"]):
            sid = inst["haystack_session_ids"][i] if i < len(inst["haystack_session_ids"]) else f"{inst['question_id']}_s{i}"
            date = inst["haystack_dates"][i] if i < len(inst["haystack_dates"]) else "unknown"
            pool.append((f"distractor_{sid}", date, sess))
    return pool


def ingest_haystack(
    instance: dict,
    store: MemoryStore,
    distractor_pool: list,
    target_turns: int,
    rng: random.Random,
) -> int:
    """Ingest the instance's own sessions + distractors until target_turns reached.
    Returns total turns stored."""
    turns_stored = 0
    session_counter = 1

    # First: store the instance's own sessions (evidence + any filler from _s).
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
            store.remember(key, value, session_counter)
            turns_stored += 1
        session_counter += 1

    # Second: sample distractor sessions until we hit target_turns.
    shuffled = list(distractor_pool)
    rng.shuffle(shuffled)
    idx = 0
    while turns_stored < target_turns and idx < len(shuffled):
        dist_sid, dist_date, dist_turns = shuffled[idx]
        idx += 1
        for ti, turn in enumerate(dist_turns):
            if turns_stored >= target_turns:
                break
            content = turn.get("content", "").strip()
            if not content:
                continue
            role = turn.get("role", "unknown")
            key = f"{dist_sid}:t{ti}:{role}"
            value = f"[{dist_date}] {role}: {content}"
            store.remember(key, value, session_counter)
            turns_stored += 1
        session_counter += 1

    return turns_stored


def generate_answer(question: str, retrieved: list[dict], question_date: str) -> str:
    blocks = []
    for i, m in enumerate(retrieved):
        blocks.append(f"[memory {i+1}, session {m['session']}] {m['key']}: {m['value']}")
    context = "\n".join(blocks) if blocks else "(no relevant memories retrieved)"
    system = (
        "You are answering a question based on retrieved memory snippets from "
        "a long-running user conversation. Use ONLY the provided memories. "
        "If they do not contain the answer, say so briefly. Give a concise answer."
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


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8", buffering=1)

    def log(msg):
        log_f.write(msg + "\n")
        log_f.flush()
        try: print(msg, flush=True)
        except Exception: pass

    log(f"Phase 3D L4 — scaling test: plain retrieval at 1x/2x/5x/10x/20x/50x haystack")
    log(f"target types: {TARGET_TYPES}, n_per_type={N_PER_TYPE}, scales={SCALE_FACTORS}")
    log(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    all_data = json.loads(LME_DATA.read_text(encoding="utf-8"))
    log(f"loaded {len(all_data)} _s instances")

    # Pick our target instances (same seed as L1/L3 for reproducibility).
    rng = random.Random(SEED)
    by_type = {qt: [] for qt in TARGET_TYPES}
    for inst in all_data:
        if inst["question_type"] in by_type:
            by_type[inst["question_type"]].append(inst)
    targets = []
    for qt in TARGET_TYPES:
        rng.shuffle(by_type[qt])
        targets.extend(by_type[qt][:N_PER_TYPE])
    log(f"selected {len(targets)} target instances")

    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for inst_idx, inst in enumerate(targets):
            # Build distractor pool once per instance.
            pool_rng = random.Random(SEED + inst_idx)
            distractors = build_distractor_pool(all_data, inst["question_id"])
            base_turns = sum(len(s) for s in inst["haystack_sessions"])
            log(f"\n### Instance {inst_idx+1}/{len(targets)}: {inst['question_id']} ({inst['question_type']})")
            log(f"  base_turns={base_turns}, distractor_pool={len(distractors)}")

            answer_sessions = set(inst.get("answer_session_ids", []))

            for scale_k in SCALE_FACTORS:
                target_turns = base_turns * scale_k
                t0 = time.time()
                store = MemoryStore(value_cap=2000, key_cap=120)
                actual_turns = ingest_haystack(inst, store, distractors, target_turns, pool_rng)
                retrieved = store.recall(inst["question"], top_k=TOP_K)
                retrieved_sessions = set(r["key"].split(":")[0] for r in retrieved)
                # "answer session in top-k" check — distractor sessions are prefixed
                # with "distractor_" so they can't false-hit.
                recall_hit = bool(answer_sessions & retrieved_sessions)
                hyp = generate_answer(inst["question"], retrieved, inst.get("question_date", ""))
                elapsed = time.time() - t0
                log(f"  scale={scale_k}x  turns={actual_turns:>5}  recall_hit={recall_hit}  {elapsed:.1f}s  -> {hyp[:100]!r}")
                entry = {
                    "question_id": inst["question_id"],
                    "question_type": inst["question_type"],
                    "scale_factor": scale_k,
                    "target_turns": target_turns,
                    "actual_turns": actual_turns,
                    "base_turns": base_turns,
                    "recall_hit_answer_session": recall_hit,
                    "top_score": retrieved[0]["score"] if retrieved else None,
                    "n_retrieved": len(retrieved),
                    "hypothesis": hyp,
                    "elapsed_s": round(elapsed, 1),
                }
                out.write(json.dumps(entry, default=str) + "\n")
                out.flush()

    log(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"hypotheses → {OUT_JSONL}")
    log_f.close()


if __name__ == "__main__":
    main()
