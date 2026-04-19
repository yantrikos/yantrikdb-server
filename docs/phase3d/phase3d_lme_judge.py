#!/usr/bin/env python3
"""Phase 3D judge — grades hypotheses.jsonl against LongMemEval ground truth.

Uses LongMemEval's exact evaluation prompts (from src/evaluation/evaluate_qa.py),
but substitutes Qwen 3.6 for GPT-4o as judge. This introduces judge bias (Qwen
grading Qwen-generated answers) but gives a FREE first-pass number. Run with a
real GPT-4o pass before publishing any number.

Usage:
  python docs/phase3d/phase3d_lme_judge.py
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
import time
import urllib.request
from collections import defaultdict

LME_DATA = pathlib.Path("c:/Users/sync/codes/LongMemEval/data/longmemeval_oracle.json")
HYP_JSONL = pathlib.Path(__file__).parent / "hypotheses.jsonl"
OUT_JSONL = pathlib.Path(__file__).parent / "scored.jsonl"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"


def judge_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    """Verbatim from LongMemEval's evaluate_qa.py."""
    if abstention:
        t = (
            "I will give you an unanswerable question, an explanation, and a "
            "response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is "
            "given but the asked information is not.\n\n"
            "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )
        return t.format(question, answer, response)

    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        t = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to "
            "the correct answer or contains all the intermediate steps to get "
            "the correct answer, you should also answer yes. If the response "
            "only contains a subset of the information required by the "
            "answer, answer no.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
        return t.format(question, answer, response)
    if task == "temporal-reasoning":
        t = (
            "I will give you a question, a correct answer, and a response "
            "from a model. Please answer yes if the response contains the "
            "correct answer. Otherwise, answer no. If the response is "
            "equivalent to the correct answer or contains all the "
            "intermediate steps to get the correct answer, you should also "
            "answer yes. If the response only contains a subset of the "
            "information required by the answer, answer no. In addition, do "
            "not penalize off-by-one errors for the number of days. If the "
            "question asks for the number of days/weeks/months, etc., and "
            "the model makes off-by-one errors (e.g., predicting 19 days "
            "when the answer is 18), the model's response is still correct.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
        return t.format(question, answer, response)
    if task == "knowledge-update":
        t = (
            "I will give you a question, a correct answer, and a response "
            "from a model. Please answer yes if the response contains the "
            "correct answer. Otherwise, answer no. If the response contains "
            "some previous information along with an updated answer, the "
            "response should be considered as correct as long as the updated "
            "answer is the required answer.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
        return t.format(question, answer, response)
    if task == "single-session-preference":
        t = (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the "
            "user's personal information correctly.\n\n"
            "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
        return t.format(question, answer, response)
    raise ValueError(f"unknown task: {task}")


def call_qwen_judge(prompt: str, timeout: int = 60) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 10, "num_ctx": 8192},
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
                return data.get("message", {}).get("content", "").strip()
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
    return f"[error: {last}]"


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

    data = json.loads(LME_DATA.read_text(encoding="utf-8"))
    qid2inst = {inst["question_id"]: inst for inst in data}

    hyps = []
    with open(HYP_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                hyps.append(json.loads(line))

    print(f"Judging {len(hyps)} hypotheses with {MODEL} as substitute GPT-4o")
    print(f"(Qwen judging Qwen-generated answers — has bias; caveat in writeup)\n")

    scored = []
    by_type = defaultdict(list)
    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for i, hyp in enumerate(hyps):
            qid = hyp["question_id"]
            inst = qid2inst.get(qid)
            if not inst:
                print(f"  [{i+1}/{len(hyps)}] SKIP {qid} — not in dataset")
                continue
            abstention = "_abs" in qid
            prompt = judge_prompt(
                inst["question_type"], inst["question"], inst["answer"], hyp["hypothesis"], abstention
            )
            reply = call_qwen_judge(prompt)
            label = reply.lower().startswith("yes")
            scored_entry = {
                **hyp,
                "question_type": inst["question_type"],
                "question": inst["question"],
                "correct_answer": inst["answer"][:200],
                "judge_reply": reply,
                "autoeval_label": label,
            }
            scored.append(scored_entry)
            by_type[inst["question_type"]].append(label)
            out.write(json.dumps(scored_entry, default=str) + "\n")
            out.flush()
            marker = "+" if label else "-"
            print(f"  [{i+1}/{len(hyps)}] {marker} {qid} [{inst['question_type']}]: {hyp['hypothesis'][:60]!r} vs {inst['answer'][:60]!r} → judge={reply!r}")

    # Aggregate
    print("\n## Summary\n")
    total = len(scored)
    correct = sum(1 for s in scored if s["autoeval_label"])
    print(f"overall: {correct}/{total} = {correct/total:.2%}" if total else "no data")
    print()
    print(f"{'question_type':<30} {'correct/n':<12} {'acc':<8}")
    for qt, labels in sorted(by_type.items()):
        n = len(labels)
        c = sum(1 for x in labels if x)
        print(f"{qt:<30} {c}/{n:<10} {c/n:.2%}" if n else f"{qt}: 0/0")

    print(f"\nscored → {OUT_JSONL}")


if __name__ == "__main__":
    main()
