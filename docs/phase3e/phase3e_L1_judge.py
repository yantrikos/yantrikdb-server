#!/usr/bin/env python3
"""Judge L1 ydb hypotheses vs LongMemEval ground truth (Qwen as substitute
GPT-4o, same prompts as phase3d judge)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "phase3d"))
import phase3d_lme_judge as j

j.HYP_JSONL = pathlib.Path(__file__).parent / "hypotheses_L1_ydb.jsonl"
j.OUT_JSONL = pathlib.Path(__file__).parent / "scored_L1_ydb.jsonl"

if __name__ == "__main__":
    j.main()
