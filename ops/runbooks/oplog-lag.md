# Runbook: Follower Oplog Lag / Can't Catch Up

## Symptoms
- `/v1/cluster` on the follower shows `reachable: true` but data is stale
- Leader has memories that the follower doesn't return via recall
- Follower logs show `sync_loop` activity but last_op_id is far behind leader

## Diagnosis

1. **Compare oplog positions:**
   ```bash
   # On leader
   curl -sS http://192.168.4.141:7438/v1/cluster | python3 -c "
     import sys,json;d=json.load(sys.stdin)
     for p in d.get('peers',[]): print(p['addr'], 'last_seen:', p.get('last_seen_secs_ago'))
   "
   ```

2. **Check follower sync logs:**
   ```bash
   ssh root@192.168.4.140 'journalctl -u yantrikdb --since "10 minutes ago" --no-pager | grep sync_loop'
   ```

3. **Check oplog sizes:**
   ```bash
   ssh root@192.168.4.140 'sqlite3 /var/lib/yantrikdb/default/yantrik.db "SELECT COUNT(*) FROM oplog"'
   ssh root@192.168.4.141 'sqlite3 /var/lib/yantrikdb/default/yantrik.db "SELECT COUNT(*) FROM oplog"'
   ```

## Recovery

1. **Wait.** sync_loop runs every 10 seconds and pulls in batches. A newly restarted follower catches up automatically. For 1000 ops, expect ~30 seconds.

2. **If the follower is permanently behind** (sync_loop errors in log):
   - Check network: `ping 192.168.4.141` from follower
   - Check cluster secret matches on both nodes
   - Check disk space on follower

3. **Nuclear option — full re-sync from snapshot:**
   ```bash
   ssh root@192.168.4.140 'systemctl stop yantrikdb && rm -rf /var/lib/yantrikdb/default/ && systemctl start yantrikdb'
   ```
   The follower will recreate the database directory and pull all data from the leader via oplog replication. This is safe but slow for large databases.
