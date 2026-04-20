#!/usr/bin/env python3
"""Phase 3F: Can Qwen learn from experience via yantrikdb memory?

Test: CRT-style trick problems where the intuitive answer is wrong.
  - Condition A (no memory): Qwen solves 5 test problems cold
  - Condition B (memory loop): Qwen solves 5 seed problems, gets feedback,
    extracts lessons, stores via yantrikdb.remember. Then solves 5 test
    problems WITH recall access before each.

All 10 problems share a pattern family (intuition-primes-wrong-answer).
Test problems vary the surface but preserve the abstract pattern.

Success metric: Condition B test accuracy > Condition A test accuracy.
If memory enables learning, B should improve over the seed-to-test arc.

Runs BOTH conditions with same Qwen 3.6 instance, different namespaces
on fresh_p3e_v3 DB.

Critical: this is NOT a proof of AGI. It's a minimal test that Qwen +
yantrikdb can produce behavior-improvement-over-time via memory-only
mechanisms, on ONE task type, with n=1 each condition. Not generalization.
Not transfer beyond the pattern family. Just: does the in-session
learning loop work at all?
"""
from __future__ import annotations

import os
os.environ["YDB_TOKEN"] = "ydb_78977ba5690d9c60b979bc15afb57199405472ec9d440942e99ac9ac22c94899"

import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "phase3e"))
from yantrikdb_client import YantrikStore

PROBLEMS_PATH = pathlib.Path(__file__).parent / "problems.json"
OUT_PATH = pathlib.Path(__file__).parent / "results.json"
LOG_PATH = pathlib.Path(__file__).parent / "run_log.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"


def call_qwen(messages, num_predict=500, timeout=180):
    body = {"model": MODEL, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0.3, "num_predict": num_predict, "num_ctx": 16384}}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode()).get("message", {}).get("content", "")
        except Exception as exc:
            time.sleep(2)
    return "[ollama error]"


def score_answer(answer: str, problem: dict) -> str:
    """Return 'correct' / 'intuitive_wrong' / 'other_wrong'."""
    a = answer.lower()
    for pat in problem["answer_patterns"]:
        if pat.lower() in a:
            return "correct"
    for pat in problem["wrong_patterns"]:
        if pat.lower() in a:
            return "intuitive_wrong"
    return "other_wrong"


def solve_cold(problem: dict, log) -> dict:
    """Ask Qwen to solve a problem with no memory context."""
    system = ("You are a careful problem-solver. Read the problem, think step by step, "
              "and give a final numeric answer. Format: 'Working: <brief reasoning>\\n"
              "Answer: <value with units>'")
    user = problem["problem"]
    t0 = time.time()
    response = call_qwen([{"role": "system", "content": system}, {"role": "user", "content": user}], num_predict=400)
    score = score_answer(response, problem)
    log(f"    {problem['id']} [{score}] {response[:120]!r}")
    return {"problem_id": problem["id"], "response": response, "score": score, "elapsed_s": round(time.time() - t0, 1)}


def solve_with_memory(problem: dict, store: YantrikStore, log) -> dict:
    """Solve a problem with yantrikdb recall first. Qwen sees any stored lessons."""
    # Recall relevant strategies
    recall_query = f"strategy for: {problem['problem'][:150]}"
    retrieved = store.recall(recall_query, top_k=5)
    memory_context = ""
    if retrieved:
        lines = ["Relevant stored strategies from past problems:"]
        for i, m in enumerate(retrieved):
            lines.append(f"  {i+1}. {m['value']}")
        memory_context = "\n".join(lines) + "\n\n"
    system = ("You are a careful problem-solver with access to a memory of strategies "
              "learned from past similar problems. CONSIDER the stored strategies — they "
              "may warn you about intuitive traps. Think step by step. Format: "
              "'Working: <brief reasoning>\\nAnswer: <value with units>'")
    user = f"{memory_context}Problem: {problem['problem']}"
    t0 = time.time()
    response = call_qwen([{"role": "system", "content": system}, {"role": "user", "content": user}], num_predict=500)
    score = score_answer(response, problem)
    log(f"    {problem['id']} [{score}] retrieved={len(retrieved)} {response[:120]!r}")
    return {"problem_id": problem["id"], "response": response, "score": score,
            "n_retrieved": len(retrieved), "elapsed_s": round(time.time() - t0, 1)}


def extract_lesson_after_feedback(problem: dict, qwen_wrong_response: str, log) -> str:
    """After Qwen got a problem wrong, give it feedback and ask it to distill a lesson."""
    system = ("You just attempted a trick problem and got it wrong. Based on the feedback, "
              "write a ONE-SENTENCE strategy that a future-you should recall when encountering "
              "similar-pattern problems. The strategy should be GENERAL (pattern-level), not "
              "specific to this problem's numbers. Output ONLY the strategy sentence.")
    user = (f"Problem: {problem['problem']}\n\n"
            f"Your answer was: {qwen_wrong_response[:400]}\n\n"
            f"Correct answer: {problem['correct_answer']}\n"
            f"The trick: {problem['lesson_hint']}\n\n"
            f"Write a one-sentence generalized strategy for future problems of this pattern:")
    lesson = call_qwen([{"role": "system", "content": system}, {"role": "user", "content": user}], num_predict=200)
    log(f"    → extracted lesson: {lesson[:150]!r}")
    return lesson.strip()


def run_condition_A(test_problems: list, log) -> list[dict]:
    """No memory. Solve all 5 test problems cold."""
    log("\n--- CONDITION A (no memory) ---")
    return [solve_cold(p, log) for p in test_problems]


def run_condition_B(seed_problems: list, test_problems: list, log) -> dict:
    """Memory loop: solve seeds with feedback, extract lessons, store. Then solve test with recall."""
    log("\n--- CONDITION B (memory loop) ---")
    store = YantrikStore(namespace=f"phase3f_B_{int(time.time())}")

    # Phase B1: seed with feedback + lesson storage
    log("\nB1 — SEED PHASE (solve cold, get feedback, store lessons)")
    seed_results = []
    lessons_stored = []
    for p in seed_problems:
        log(f"\n  [seed {p['id']}]")
        cold_result = solve_cold(p, log)
        # Feedback and lesson extraction happen regardless of correct/wrong, so Qwen
        # always stores the pattern — but richer reasoning when it got it wrong
        lesson_source = cold_result["response"] if cold_result["score"] != "correct" else p["lesson_hint"]
        lesson = extract_lesson_after_feedback(p, lesson_source, log)
        # Store as memory in yantrikdb
        store_result = store.remember(
            key=f"strategy_{p['id']}",
            value=lesson[:120],
            session=1,
            importance=0.9,
        )
        lessons_stored.append({"problem_id": p["id"], "lesson": lesson, "rid": store_result.get("rid")})
        seed_results.append({**cold_result, "lesson_stored": lesson})

    # Run think() to let the substrate consolidate and entity-extract
    think_result = store.think()
    log(f"\nthink() after seeds: conflicts={think_result.get('conflicts_found', 0)} "
        f"consolidated={think_result.get('consolidation_count', 0)}")

    # Phase B2: test with recall
    log("\nB2 — TEST PHASE (solve test problems WITH recall of stored strategies)")
    test_results = [solve_with_memory(p, store, log) for p in test_problems]

    return {
        "seed_results": seed_results,
        "lessons_stored": lessons_stored,
        "think_after_seeds": think_result,
        "test_results": test_results,
        "stats_final": store.stats(),
    }


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8", buffering=1)
    def log(msg):
        log_f.write(msg + "\n"); log_f.flush()
        try: print(msg, flush=True)
        except Exception: pass

    log("Phase 3F — self-learning via yantrikdb memory, CRT trick problems")
    log(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    problems = json.loads(PROBLEMS_PATH.read_text(encoding="utf-8"))
    seed, test = problems["seed"], problems["test"]
    log(f"{len(seed)} seed problems, {len(test)} test problems")

    # Run Condition A first (no memory at all for any problem)
    a_results = run_condition_A(test, log)
    a_correct = sum(1 for r in a_results if r["score"] == "correct")
    a_intuitive = sum(1 for r in a_results if r["score"] == "intuitive_wrong")
    log(f"\nCondition A: {a_correct}/{len(test)} correct ({a_intuitive} fell for intuitive trap)")

    # Run Condition B (memory loop)
    b = run_condition_B(seed, test, log)
    seed_correct = sum(1 for r in b["seed_results"] if r["score"] == "correct")
    test_correct = sum(1 for r in b["test_results"] if r["score"] == "correct")
    test_intuitive = sum(1 for r in b["test_results"] if r["score"] == "intuitive_wrong")
    log(f"\nCondition B seed: {seed_correct}/{len(seed)} correct (baseline)")
    log(f"Condition B test: {test_correct}/{len(test)} correct ({test_intuitive} fell for trap)")

    summary = {
        "condition_A": {
            "correct": a_correct, "total": len(test),
            "accuracy": a_correct / len(test), "intuitive_trap": a_intuitive,
            "results": a_results,
        },
        "condition_B": {
            "seed_correct": seed_correct, "seed_total": len(seed),
            "seed_accuracy": seed_correct / len(seed),
            "test_correct": test_correct, "test_total": len(test),
            "test_accuracy": test_correct / len(test),
            "test_intuitive_trap": test_intuitive,
            "b_full": b,
        },
        "learning_delta": (test_correct / len(test)) - (a_correct / len(test)),
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    log(f"\n=== SUMMARY ===")
    log(f"Condition A test accuracy:  {a_correct}/{len(test)} = {100*a_correct/len(test):.0f}%")
    log(f"Condition B test accuracy:  {test_correct}/{len(test)} = {100*test_correct/len(test):.0f}%")
    log(f"Learning delta (B - A):     {100*summary['learning_delta']:+.0f} pts")
    log(f"\nTest-set pattern-trap rate (both conditions):")
    log(f"  A intuitive-wrong:  {a_intuitive}/{len(test)} = {100*a_intuitive/len(test):.0f}%")
    log(f"  B intuitive-wrong:  {test_intuitive}/{len(test)} = {100*test_intuitive/len(test):.0f}%")
    log(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_f.close()


if __name__ == "__main__":
    main()
