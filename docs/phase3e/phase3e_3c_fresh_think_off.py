#!/usr/bin/env python3
"""Fresh-DB + think-off cell of the 2x2 ablation."""
import os
os.environ["YDB_TOKEN"] = "ydb_0989d4b0d904501524c1dc735b4099e636e7e61201c64fd7bd0077211b4da4fb"
import sys
sys.argv.extend(["--fresh-db", "--think-off"])
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
# Override output path to disambiguate
import phase3e_3c_rerun as r
r.OUT_PATH = pathlib.Path(__file__).parent / "results_3c_rerun_fresh_think_off.json"
r.LOG_PATH = pathlib.Path(__file__).parent / "harness_3c_rerun_fresh_think_off_log.txt"

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "C_yantrikdb"] + [a for a in sys.argv[1:] if a.startswith("--")]
    r.main()
