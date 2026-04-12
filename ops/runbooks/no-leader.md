# Runbook: No Leader Elected / Election Stuck

## Symptoms
- `/v1/cluster` shows `leader_id: null` on all nodes
- All write endpoints return 503 "no leader elected"
- Reads still work (followers serve stale-OK reads)
- Heartbeat log shows repeated `leader timeout — starting election`

## Diagnosis

1. **Check cluster state on all voters:**
   ```bash
   for h in 192.168.4.140 192.168.4.141; do
     echo "=== $h ==="
     curl -sS http://$h:7438/v1/cluster | python3 -c "
       import sys,json;d=json.load(sys.stdin)
       print('role:', d['role'], 'term:', d['current_term'], 'voted_for:', d.get('voted_for'))
       for p in d.get('peers',[]): print(' peer:', p['addr'], 'reachable:', p['reachable'], 'term:', p.get('current_term'))
     "
   done
   ```

2. **Common causes:**
   - **Witness unreachable** → no quorum (need 2 of 3 for election). Check witness: `curl -sS http://192.168.4.142:7438/v1/cluster`
   - **Split vote** → both voters voted for themselves. Will resolve on next election timeout (~10s).
   - **Term mismatch** → one node at much higher term. Lower-term node needs to catch up.
   - **Corrupted raft.json** → node can't persist vote state.

3. **Check raft.json:**
   ```bash
   ssh root@192.168.4.140 'cat /var/lib/yantrikdb/raft.json'
   ssh root@192.168.4.141 'cat /var/lib/yantrikdb/raft.json'
   ```

## Recovery

1. **Wait 30 seconds.** Most election stalls self-resolve after 2-3 election timeouts.

2. **If witness is down**, restart it:
   ```bash
   ssh root@192.168.4.142 'systemctl restart yantrikdb-witness'
   ```

3. **Force election from a specific node** (makes that node a candidate immediately):
   ```bash
   curl -X POST http://192.168.4.140:7438/v1/cluster/promote \
     -H "Authorization: Bearer <cluster_master_token>"
   ```

4. **If raft.json is corrupted**, delete it and restart — the node will start as follower at term 0 and catch up:
   ```bash
   ssh root@192.168.4.140 'systemctl stop yantrikdb && rm /var/lib/yantrikdb/raft.json && systemctl start yantrikdb'
   ```
   **Warning**: only do this on ONE node at a time. The other node must remain as the authority for the current term.
