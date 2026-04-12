# Runbook: Cluster Unreachable / All Nodes Down

## Symptoms
- All HTTP endpoints on all voters return timeout or connection refused
- Watchdog firing on every node
- Swarmcode messages from clients reporting "can't reach YantrikDB"

## Diagnosis

1. **SSH into each node and check systemd:**
   ```bash
   ssh root@192.168.4.140 'systemctl is-active yantrikdb; journalctl -u yantrikdb -n 30 --no-pager'
   ssh root@192.168.4.141 'systemctl is-active yantrikdb; journalctl -u yantrikdb -n 30 --no-pager'
   ```

2. **Check if processes are alive but hung:**
   ```bash
   ssh root@192.168.4.140 'pidof yantrikdb && curl -sS --max-time 5 http://localhost:7438/v1/health'
   ```
   - If pidof returns a PID but curl times out → **hung** (see [hang-captured.md](hang-captured.md))
   - If pidof returns nothing → **crashed** (check journal for panic/OOM)

3. **Check disk space** (SQLite fails silently when full):
   ```bash
   ssh root@192.168.4.140 'df -h /var/lib/yantrikdb'
   ```

4. **Check if watchdog captured diagnostics:**
   ```bash
   ssh root@192.168.4.140 'ls -lt /var/log/yantrik-diag/ | head -5'
   ```

## Recovery

1. **Restart leader first** (the node that was most recently leader — check `raft.json`):
   ```bash
   ssh root@192.168.4.141 'systemctl restart yantrikdb && sleep 5 && curl -sS http://localhost:7438/v1/cluster'
   ```

2. **Then restart follower:**
   ```bash
   ssh root@192.168.4.140 'systemctl restart yantrikdb && sleep 5 && curl -sS http://localhost:7438/v1/cluster'
   ```

3. **Wait for election** (~10s). Verify both nodes see a leader:
   ```bash
   for h in 192.168.4.140 192.168.4.141; do
     curl -sS http://$h:7438/v1/cluster | python3 -c "import sys,json;d=json.load(sys.stdin);print('$h:',d['role'],'leader=',d['leader_id'])"
   done
   ```

4. **Smoke test:**
   ```bash
   curl -sS -X POST http://<leader>:7438/v1/recall \
     -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
     -d '{"query":"test","top_k":1}'
   ```

## Data Safety
- SQLite WAL mode protects against corruption from unclean shutdown
- Oplog replication will catch up followers automatically after restart
- HNSW index rebuilds from SQLite on restart if needed
- No manual intervention required for data recovery
