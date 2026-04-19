#!/usr/bin/env python3
"""Wipe the local yantrikdb data dir and restart with a fresh database.

USED FOR contamination audits. After running Phase 2/3A-D work, the
local DB has 30+ entities and 85+ edges from prior tests. Some of
these may cross-contaminate Phase 3E benchmarks via yantrikdb's
global entity graph. This script cleanly resets the server state.

Usage (from repo root):
  # 1. Stop the running yantrikdb server (the backgrounded one)
  # 2. Run this to wipe data/
  # 3. Restart server with yantrikdb serve --config yantrikdb_local.toml
  # 4. Create a new token
  # 5. Update YDB_TOKEN env or yantrikdb_client.py default

This script only wipes; it does NOT restart the server.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import time

DATA_DIR = pathlib.Path("data")
BACKUP_DIR = pathlib.Path("data.backup." + time.strftime("%Y%m%d_%H%M%S"))


def main():
    if not DATA_DIR.exists():
        print(f"No {DATA_DIR} directory — nothing to wipe.")
        return

    if "--force" not in sys.argv:
        print(f"Will MOVE {DATA_DIR}/ to {BACKUP_DIR}/ (so original is recoverable).")
        print("Re-run with --force to actually do it.")
        return

    print(f"Moving {DATA_DIR}/ -> {BACKUP_DIR}/ ...")
    shutil.move(str(DATA_DIR), str(BACKUP_DIR))
    DATA_DIR.mkdir(exist_ok=True)
    print(f"done. Fresh {DATA_DIR}/ ready. Start server + create token next.")


if __name__ == "__main__":
    main()
