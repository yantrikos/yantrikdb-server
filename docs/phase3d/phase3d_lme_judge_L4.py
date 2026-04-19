#!/usr/bin/env python3
"""Judge for L4 scaling hypotheses — group by scale factor + compute recall and answer accuracy per scale."""
import io
import json
import pathlib
import sys
import time
import urllib.request
from collections import defaultdict

LME_DATA = pathlib.Path("c:/Users/sync/codes/LongMemEval/data/longmemeval_s_cleaned.json")
HYP_JSONL = pathlib.Path(__file__).parent / "hypotheses_L4.jsonl"
OUT_JSONL = pathlib.Path(__file__).parent / "scored_L4.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import phase3d_lme_judge as j  # reuse judge_prompt + call_qwen_judge


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    data = json.loads(LME_DATA.read_text(encoding="utf-8"))
    qid2inst = {inst["question_id"]: inst for inst in data}

    hyps = [json.loads(l) for l in open(HYP_JSONL, encoding="utf-8") if l.strip()]
    print(f"Judging {len(hyps)} L4 runs ({len({h['question_id'] for h in hyps})} unique instances × scales)")

    scored = []
    by_scale = defaultdict(lambda: {"hits": 0, "correct": 0, "n": 0})
    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for i, hyp in enumerate(hyps):
            qid = hyp["question_id"]
            inst = qid2inst.get(qid)
            if not inst:
                continue
            abstention = "_abs" in qid
            prompt = j.judge_prompt(inst["question_type"], inst["question"], inst["answer"], hyp["hypothesis"], abstention)
            reply = j.call_qwen_judge(prompt)
            label = reply.lower().startswith("yes")
            s = hyp["scale_factor"]
            by_scale[s]["n"] += 1
            by_scale[s]["correct"] += int(label)
            if hyp.get("recall_hit_answer_session"):
                by_scale[s]["hits"] += 1
            entry = {**hyp, "question_type": inst["question_type"], "correct_answer": inst["answer"][:200], "judge_reply": reply, "autoeval_label": label}
            scored.append(entry)
            out.write(json.dumps(entry, default=str) + "\n")
            out.flush()
            marker = "+" if label else "-"
            print(f"  [{i+1}/{len(hyps)}] {marker} {qid} scale={s}x  hit={hyp.get('recall_hit_answer_session')}  judge={reply!r}")

    print("\n## Per-scale summary\n")
    print(f"{'scale':>6} {'recall@20':>10} {'answer_acc':>12} {'n':>4}")
    for s in sorted(by_scale):
        d = by_scale[s]
        rec = d["hits"] / d["n"] if d["n"] else 0
        acc = d["correct"] / d["n"] if d["n"] else 0
        print(f"{s:>4}x  {d['hits']}/{d['n']} = {rec:.1%}   {d['correct']}/{d['n']} = {acc:.1%}  {d['n']:>3}")

    print(f"\nscored → {OUT_JSONL}")


if __name__ == "__main__":
    main()
