# Runbook: Disk Full / Data Dir Full

## Symptoms
- Writes fail with 500 / "database or disk is full"
- SQLite WAL file growing without bound
- `df -h /var/lib/yantrikdb` shows >95% usage

## Diagnosis

```bash
ssh root@<node> '
  df -h /var/lib/yantrikdb
  du -sh /var/lib/yantrikdb/*
  ls -lh /var/lib/yantrikdb/default/yantrik.db*
'
```

Common culprits:
- `yantrik.db-wal` — WAL not checkpointing (can grow to GB under steady writes)
- `yantrik.db` — database itself is large (many memories + oplog)
- `/var/log/yantrik-diag/` — watchdog diagnostic bundles accumulating

## Immediate Mitigation

1. **Checkpoint the WAL** (reclaims WAL space immediately):
   ```bash
   ssh root@<node> 'sqlite3 /var/lib/yantrikdb/default/yantrik.db "PRAGMA wal_checkpoint(TRUNCATE)"'
   ```

2. **Clean old diagnostics** (safe — these are only for debugging):
   ```bash
   ssh root@<node> 'find /var/log/yantrik-diag -mtime +7 -exec rm -rf {} + 2>/dev/null'
   ```

3. **Run oplog GC manually** (removes old applied entries):
   ```bash
   ssh root@<node> 'sqlite3 /var/lib/yantrikdb/default/yantrik.db "
     DELETE FROM oplog WHERE op_id IN (
       SELECT op_id FROM oplog WHERE applied = 1
       ORDER BY hlc ASC LIMIT 50000
     )
   "'
   ```

## Permanent Fix

- **Grow the volume**: Proxmox LXC → Resources → Root Disk → resize
- **Add scheduled WAL checkpoint**: the background worker does this automatically (task #71 in hardening roadmap)
- **Reduce oplog retention**: configure `keep_recent` in the oplog GC worker (default 100k entries)
