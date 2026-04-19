#!/usr/bin/env python3
"""Judge for L3 hypotheses. Same prompts + judge as L1, different input file."""
import pathlib
import sys

# Override paths for L3, then delegate to L1 judge module
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import phase3d_lme_judge as judge

judge.HYP_JSONL = pathlib.Path(__file__).parent / "hypotheses_L3.jsonl"
judge.OUT_JSONL = pathlib.Path(__file__).parent / "scored_L3.jsonl"

if __name__ == "__main__":
    judge.main()
