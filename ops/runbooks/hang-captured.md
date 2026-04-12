# Runbook: Hang Captured by Watchdog

## Symptoms
- Watchdog alert received via ntfy: "YantrikDB HANG on <node>"
- Process alive (pidof returns PID), port listening (ss shows LISTEN), but HTTP requests hang
- Diagnostics directory created at `/var/log/yantrik-diag/hang-<timestamp>/`

## Diagnosis — Read Before Restarting

The diagnostics captured by the watchdog are the most valuable artifact. **Do not restart without inspecting them first** — a hang you restart without understanding is a bug you will hit again.

### 1. Open the diagnostic bundle

```bash
D=/var/log/yantrik-diag/hang-<timestamp>
ls -la $D/
```

Files:
- `info.txt` — PID, uptime, CPU%, MEM%, stat
- `gdb-bt.txt` — **THE KEY FILE** — full thread backtraces from gdb
- `threads.txt` — per-thread kernel stacks from /proc
- `sockets.txt` — established TCP connections
- `journal.txt` — last 200 journal lines
- `kstack.txt` — main thread kernel stack
- `status.txt` — /proc/PID/status

### 2. Analyze gdb-bt.txt

```bash
grep -E "^Thread|#0|#1|#2|#3" $D/gdb-bt.txt | head -80
```

**What to look for:**

| Pattern | Meaning |
|---|---|
| Many threads at `futex::Mutex::lock_contended` in `handler::execute` | All blocking threads waiting for engine lock |
| One thread deep inside `engine::cognition::think` or `consolidate` | That thread HOLDS the engine lock |
| Thread in `conflict_exists` or `check_redundancy` at `lock_contended` | **Self-deadlock** (same class as v0.5.8 bug) |
| Thread in `conn.lock()` inside a `match Self::get_meta(...)` | **Self-deadlock** on conn mutex (same class as v0.5.9 fix) |
| All threads in `epoll_wait` / sleeping | Not a lock issue — maybe OOM or resource starvation |

### 3. Check if parking_lot deadlock detector logged anything

```bash
grep "DEADLOCK DETECTED" $D/journal.txt
```

If present, parking_lot found a cycle. The log includes thread IDs and backtraces for every thread in the cycle. This is definitive — you know exactly which locks are involved.

### 4. Check metrics for leading indicators

```bash
curl -sS http://localhost:7438/metrics | grep lock_wait
```

If `yantrikdb_lock_wait_seconds` p99 was climbing before the hang, the lock contention was building gradually (not a sudden deadlock).

## Recovery

### If you identified a self-deadlock (fix needed)
1. Archive the diagnostic bundle: `tar czf /root/hang-<ts>.tar.gz $D`
2. Restart: `systemctl restart yantrikdb`
3. File a bug with the gdb-bt.txt and journal.txt

### If the watchdog auto-restarted (3 consecutive hangs)
1. Check `/var/log/yantrik-watchdog.log` for the sequence of events
2. The diagnostics from the FIRST hang detection are the most useful (before any restart confusion)
3. Verify the server recovered: `curl -sS http://localhost:7438/v1/health/deep`

### If you can't identify the cause
1. Leave the process running (don't restart yet)
2. Grab a fresh gdb snapshot: `gdb -batch -p $(pidof yantrikdb) -ex 'thread apply all bt' > /tmp/manual-bt.txt`
3. Check `/proc/$(pidof yantrikdb)/status` for thread count, memory
4. Check `ss -tnp` for connection backlog
5. Escalate with all captured data

## History of Resolved Hangs

| Version | Root Cause | Fix |
|---|---|---|
| v0.5.7 | `check_redundancy` self-deadlock: `if let` scrutinee held `MutexGuard<Connection>` through body that re-entered `db.conn()` | v0.5.8: scoped guard explicitly |
| v0.5.8 | Same class in 40 `get_meta` call sites + `save_working_set` holding conn across `log_op` | v0.5.9: parking_lot + full audit + match→let pattern |
