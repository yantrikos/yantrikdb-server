#!/usr/bin/env python3
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "phase3d"))
import phase3d_lme_judge as j
j.HYP_JSONL = pathlib.Path(__file__).parent / "hypotheses_L3_ydb.jsonl"
j.OUT_JSONL = pathlib.Path(__file__).parent / "scored_L3_ydb.jsonl"
if __name__ == "__main__": j.main()
