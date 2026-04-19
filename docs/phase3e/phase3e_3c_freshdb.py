#!/usr/bin/env python3
"""Phase 3E contamination audit: Phase 3C rerun on a FRESH yantrikdb database.

The default DB has accumulated entities/edges from Phase 2 and Phase 3E
test runs. The knowledge graph is per-DB (confirmed 2026-04-19), so
rerunning in a newly-created `fresh_p3e` DB isolates the scenario
results from prior test residue.

This script is identical to phase3e_3c_rerun.py except it points at
the fresh_p3e token. If the 3C gap holds here, residue wasn't helping.
If it collapses, the earlier 0.850 was partly artifact.

Usage:
  python docs/phase3e/phase3e_3c_freshdb.py
"""
import os
# Set fresh-DB token BEFORE importing the client (the client reads YDB_TOKEN at module load)
os.environ["YDB_TOKEN"] = "ydb_0989d4b0d904501524c1dc735b4099e636e7e61201c64fd7bd0077211b4da4fb"

import sys
sys.argv.append("--fresh-db")  # triggers the _freshdb output path tag in rerun
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import phase3e_3c_rerun as r

# Override to run C only
r.CONDITIONS = ["C_yantrikdb"]
r.RUNS_PER_CONDITION = 2

if __name__ == "__main__":
    # Also override sys.argv to filter to C_yantrikdb for the rerun's arg parser
    sys.argv = [sys.argv[0], "C_yantrikdb"] + [a for a in sys.argv[1:] if a.startswith("--")]
    r.main()
