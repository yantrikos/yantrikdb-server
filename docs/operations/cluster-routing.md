# Cluster client routing — interim runbook (pre-PR-6)

Status: **interim** until RFC 010 PR-6 lands (target v0.8.13).

## Why this document exists

In v0.8.x with `cluster.raft_mode=openraft`, the server has an asymmetry
that bites HTTP clients which round-robin or fall back across nodes:

- **Writes** to a follower return `503 not the leader`. Clients that
  retry against the next URL in their list silently land on the leader
  → write succeeds, but the operator has no signal that the topology
  matters.
- **Reads** to a follower return `200 OK` from local SQLite/HNSW state.
  No 503, no fallback. If replication is broken (the cosmetic-openraft
  bug RFC 010 PR-6 fixes), follower returns *stale* data with a 200.

Net effect: **writes consistently land on the leader; reads return
whatever the node-you-hit happens to have**. Clients that list the
follower first see "I wrote it, I can't read it" — the ghosting bug.

This is structurally fixed in RFC 010 PR-6 (handlers route writes
through openraft consensus → followers replicate → reads on any node
return the same data). Until PR-6 ships, the mitigation is **list the
leader first in your client's URL config**.

## Symptom checklist

You probably have this bug if:

- Your client config has more than one cluster URL.
- Writes appear to succeed (no errors at the application layer).
- Reads sometimes return data, sometimes return "not found" or empty
  results, with no apparent pattern from the client's perspective.
- A direct `curl http://leader:7438/v1/recall` returns the data; a
  direct `curl http://follower:7438/v1/recall` does not.

## Diagnostic

The probe below runs against the live cluster and discriminates
follower-first-read from leader-side HNSW lag.

```sh
# Find the current leader.
LEADER=$(curl -s http://NODE_A:7438/v1/cluster/raft | jq -r '.current_leader')
LEADER_ADDR=$(curl -s http://NODE_A:7438/v1/cluster/raft \
  | jq -r ".members[] | select(.node_id==$LEADER) | .addr")
FOLLOWER_ADDR=$(curl -s http://NODE_A:7438/v1/cluster/raft \
  | jq -r ".members[] | select(.node_id!=$LEADER) | .addr")

echo "Leader:   $LEADER_ADDR"
echo "Follower: $FOLLOWER_ADDR"

# 1. Write a probe to the leader directly.
RID=$(curl -s -X POST "$LEADER_ADDR/v1/remember" \
  -H "Authorization: Bearer $YDB_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"cluster-routing probe","memory_type":"semantic","namespace":"_probe"}' \
  | jq -r .rid)
echo "Wrote rid=$RID"

# 2. Read from leader (should hit immediately).
echo "--- leader recall ---"
curl -s -X POST "$LEADER_ADDR/v1/recall" \
  -H "Authorization: Bearer $YDB_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"cluster-routing probe","namespace":"_probe","top_k":3}' \
  | jq '.results[] | {rid, score}'

# 3. Read from follower (should miss until PR-6 lands).
echo "--- follower recall ---"
curl -s -X POST "$FOLLOWER_ADDR/v1/recall" \
  -H "Authorization: Bearer $YDB_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"cluster-routing probe","namespace":"_probe","top_k":3}' \
  | jq '.results[] | {rid, score}'
```

| Leader recall | Follower recall | Verdict |
|---|---|---|
| Hit | Miss | **Cosmetic-openraft (this runbook applies).** Follower replication broken. Apply mitigation below. |
| Hit | Hit | Cluster healthy — your bug is elsewhere. |
| Miss | Miss | New bug. Check `last_log_index` on both nodes; if 0 or unchanged across writes, escalate. |
| Miss | Hit | Routing reversed — your "leader" is actually the follower. Re-check `/v1/cluster/raft`. |

## Mitigation: leader-first URL ordering

Set your client's URL list with the **current leader first**.

### Lane B (`/etc/lane-b.env`) example

```sh
# Before:
YDB_URLS=http://192.168.4.141:7438,http://192.168.4.140:7438

# After (assuming .140 is the leader; verify via /v1/cluster/raft):
YDB_URLS=http://192.168.4.140:7438,http://192.168.4.141:7438
```

Always backup before editing:

```sh
cp /etc/lane-b.env /etc/lane-b.env.bak.$(date +%Y%m%d-%H%M%S)
$EDITOR /etc/lane-b.env
systemctl restart lane-b   # or whichever unit owns the client
```

### Generic clients

If the client posts to multiple URLs and falls back on errors (most
HTTP clients do this with retry middleware), put the leader URL first.
The 503-fallback then keeps you alive if the leader transitions, and
the leader-read avoids the stale-follower trap.

If your client doesn't have fallback logic at all, just point it at
the leader. You lose HA on read, but you also avoid the silent-stale
problem.

## When the leader changes

openraft elections happen on leader failure / restart. The leader can
move from .140 to .141 (or any node in the voter set). When that
happens:

1. Your client's writes start returning 503 from the old leader.
2. Reads to the old leader become stale (mirror image of the original
   bug).

Operators have two options today:

- **Manual:** monitor `/v1/cluster/raft` on any node; when leader_id
  changes, update client config + restart.
- **Pin via DNS:** point a DNS name (e.g. `yantrikdb-leader.internal`)
  at the current leader; switch the DNS record on leader change. Your
  client URLs reference the DNS name. Trade-off: extra DNS lookup
  latency + TTL window of stale routing.

PR-6.6 (saga task 189) introduces HTTP 307 redirects with `Location`
headers pointing at the current leader, so clients that follow
redirects (every standard HTTP library does) will route automatically.
Until then, manual or DNS pinning is the operator's job.

## Witness node (legacy)

The witness node (port 7440) is **not** part of the openraft cluster
in v0.8.x. It's a legacy raft-lite component scheduled for removal in
v0.9.0. Do not include witness URLs in client config. The openraft
member set is visible at `/v1/cluster/raft`:

```sh
curl -s http://leader:7438/v1/cluster/raft | jq .members
```

Should return only the voter nodes (typically 2 in a 2-voter cluster,
3 in a 3-voter cluster). If you see a witness URL there, the cluster
is misconfigured.

## When PR-6 ships

This runbook becomes obsolete at v0.8.13 (PR 6.4 — handler migration
through MutationCommitter). After that:

- Writes go through openraft consensus and replicate to all nodes.
- Reads from any node return the same data (within bounded apply lag,
  typically <1 s).
- 307 redirects make leader-routing transparent for new clients.
- Boot invariants reject `raft_mode=openraft` with a non-Raft handler
  config, so cosmetic-openraft mode is structurally impossible.

Once your operator surface confirms `replication_lag_log_entries: 0`
on `/v1/health` for all followers, you can revert to round-robin URL
ordering or drop the leader-first hack entirely.

## See also

- RFC 010 PR-6: `docs/rfcs/rfc_010_pr6_write_path_migration.md`
- Saga Epic #53 — task tracker for PR 6.1–6.9
- ROADMAP.md — v0.8.12–v0.8.14 sequence
- Memory rid `019deed1-5588-70c0-b14b-a2059cb960eb` — the 2026-05-03
  Lane B incident this runbook was written from.
