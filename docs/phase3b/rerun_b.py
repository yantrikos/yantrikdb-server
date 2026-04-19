#!/usr/bin/env python3
"""Re-run only B_self_note (original matrix run had Qwen skip the tool
in session 1 for all 4 B runs). Merge with existing results.json."""
import io
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import phase3b_harness as h

RESULTS = pathlib.Path(__file__).parent / "results.json"
LOG = pathlib.Path(__file__).parent / "rerun_b_log.txt"
log_f = open(LOG, "w", encoding="utf-8", buffering=1)

def log(msg):
    log_f.write(msg + "\n")
    log_f.flush()
    try: print(msg, flush=True)
    except Exception: pass

with open(RESULTS, encoding="utf-8") as f:
    data = json.load(f)

# Drop old B entries, keep A/C/D
data["results"] = [r for r in data["results"] if r["condition"] != "B_self_note"]

log(f"Re-running B_self_note x 4\nstart: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

for r_idx in range(4):
    log(f">>> B_self_note run={r_idx} ({time.strftime('%H:%M:%S')})")
    try:
        result = h.run_one("B_self_note", r_idx)
    except Exception as e:
        log(f"  EXCEPTION: {e}")
        continue
    data["results"].append(result)
    ctx = result["session2_context_len"]
    prop = len(result["proposal"] or "")
    captured = result["session1_captured"] or {}
    nc = len(captured.get("hard_constraints", []))
    log(f"  done — {result['elapsed_s']}s, ctx={ctx}c, proposal={prop}c, n_constraints={nc}")
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

log(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log_f.close()
