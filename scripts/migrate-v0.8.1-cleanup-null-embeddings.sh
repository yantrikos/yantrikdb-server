#!/usr/bin/env bash
# v0.8.1 migration script — cleanup NULL-embedding rows.
#
# Background: v0.7.x and v0.8.0 had a writer-side bug (issue #19)
# where /v1/remember and /v1/remember/batch could silently store rows
# with `embedding=NULL` if the embedder service hiccupped. v0.8.1
# fixed the writer side. This script cleans up rows already on disk
# from before the upgrade.
#
# Without cleanup, NULL rows poison /v1/recall on the namespace
# (the similarity-scan path bails on NULL with "Invalid column type
# Null at index: 1, name: embedding"). Run this script once per data
# directory after upgrading to v0.8.1.
#
# Usage:
#   ./scripts/migrate-v0.8.1-cleanup-null-embeddings.sh --data-dir /var/lib/yantrikdb [--apply]
#
# Default mode is DRY-RUN: prints what would be deleted, makes no
# changes. Pass --apply to actually delete.
#
# Idempotent: running twice with --apply is safe — the second run
# finds 0 rows and does nothing.
#
# Requires: bash, sqlite3 (apt-get install sqlite3 / brew install sqlite3).
#
# Issue: https://github.com/yantrikos/yantrikdb-server/issues/19

set -euo pipefail

DATA_DIR=""
APPLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --help|-h)
      sed -n '2,/^$/p' "$0" | sed 's/^#//; s/^ //'
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 --data-dir <path> [--apply]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "error: --data-dir is required" >&2
  echo "usage: $0 --data-dir <path> [--apply]" >&2
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "error: data dir does not exist: $DATA_DIR" >&2
  exit 1
fi

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "error: sqlite3 binary not found in PATH" >&2
  echo "install: apt-get install sqlite3 / brew install sqlite3 / yum install sqlite" >&2
  exit 1
fi

mode="DRY-RUN"
if [[ "$APPLY" -eq 1 ]]; then
  mode="APPLY"
fi

echo "=========================================="
echo "v0.8.1 NULL-embedding cleanup — mode: $mode"
echo "data dir: $DATA_DIR"
echo "=========================================="
echo

# Find all tenant DBs. Layout: <data-dir>/<tenant>/yantrik.db
# Skip the control DB and any non-tenant files.
total_null=0
total_tenants=0
deleted_total=0

shopt -s nullglob
for tenant_dir in "$DATA_DIR"/*/; do
  db="$tenant_dir/yantrik.db"
  if [[ ! -f "$db" ]]; then
    continue
  fi
  tenant_name="$(basename "$tenant_dir")"

  # Count NULL embeddings.
  null_count="$(sqlite3 "$db" 'SELECT COUNT(*) FROM memories WHERE embedding IS NULL;' 2>/dev/null || echo "?")"

  if [[ "$null_count" == "?" ]]; then
    echo "  [skip] $tenant_name: cannot read DB (locked? bad schema?)"
    continue
  fi

  total_tenants=$((total_tenants + 1))

  if [[ "$null_count" -eq 0 ]]; then
    echo "  [ok]    $tenant_name: 0 NULL rows"
    continue
  fi

  total_null=$((total_null + null_count))

  echo "  [found] $tenant_name: $null_count NULL-embedding row(s)"

  # Show a sample for operator visibility.
  echo "          sample rids:"
  sqlite3 "$db" \
    "SELECT '            ' || rid || ' (namespace=' || COALESCE(namespace,'<null>') || ', text_len=' || length(text) || ')'
     FROM memories WHERE embedding IS NULL LIMIT 3;" 2>/dev/null || true

  if [[ "$APPLY" -eq 1 ]]; then
    deleted="$(sqlite3 "$db" \
      'DELETE FROM memories WHERE embedding IS NULL; SELECT changes();' 2>/dev/null || echo "?")"
    if [[ "$deleted" == "?" ]]; then
      echo "          [error] DELETE failed — DB likely held by a running yantrikdb-server. Stop the server, retry."
      exit 2
    fi
    echo "          [deleted] $deleted row(s)"
    deleted_total=$((deleted_total + deleted))
  fi
done

echo
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "  tenants scanned:     $total_tenants"
echo "  total NULL rows:     $total_null"
if [[ "$APPLY" -eq 1 ]]; then
  echo "  rows deleted:        $deleted_total"
  if [[ "$deleted_total" -gt 0 ]]; then
    echo
    echo "Cleanup complete. /v1/recall on affected namespaces should now"
    echo "succeed. Verify by issuing a query against each affected tenant."
  fi
else
  if [[ "$total_null" -gt 0 ]]; then
    echo
    echo "DRY-RUN: no changes made. Re-run with --apply to delete the rows."
    echo "Note: stop the yantrikdb-server process before running --apply"
    echo "      (the script needs exclusive access to the SQLite files)."
  fi
fi
