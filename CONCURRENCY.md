# Concurrency & Locking Invariants

**Audience**: contributors touching shared state inside `yantrikdb` or `yantrikdb-server`.

**Status**: enforced as of v0.5.9 (parking_lot migration). Violations of this document are the #1 class of bug in YantrikDB.

---

## Why this document exists

We had two production hangs (v0.5.7 / v0.5.8 series) caused by locking mistakes:

1. **Self-deadlock on `Mutex<Connection>`** — a call site held a `db.conn()` guard in an `if let` scrutinee whose temporary lifetime extended through the body, and the body called a helper that tried to take `db.conn()` again on the same thread. `std::sync::Mutex` is non-reentrant, so a same-thread re-acquire blocks forever. See [crates/yantrikdb-core/src/cognition/triggers.rs:313](crates/yantrikdb-core/src/cognition/triggers.rs#L313) and [commit c4c2d9d](https://github.com/yantrikos/yantrikdb-server/commit/c4c2d9d).

2. **Tokio worker wedged by synchronous engine call** — HTTP handlers called `engine.lock()` directly on a tokio worker thread; a slow internal call (`think`, `consolidate`, `embed`) parked the worker for its entire duration. With 2 CPU cores, two concurrent slow ops wedged the whole runtime. See [commit c9d943a](https://github.com/yantrikos/yantrikdb-server/commit/c9d943a).

Both are now structurally prevented. This document codifies the rules that keep them that way.

---

## Rule 1 — `parking_lot` everywhere, never `std::sync::{Mutex,RwLock}`

All `Mutex<T>` and `RwLock<T>` inside `yantrikdb` and `yantrikdb-server` MUST come from `parking_lot`, not `std::sync`.

**Why**:

- `parking_lot` does not poison on panic. A panic inside a lock guard does not cascade into every other thread panicking on its next `lock()` call. `std::sync::Mutex` poisoning has caused every major Rust production DB to eventually hit a cascade failure.
- `parking_lot::deadlock::check_deadlock()` lets us run a background task that detects circular lock dependencies at runtime and logs full thread backtraces. We enable this via the `deadlock_detection` feature; the detector lives in `crates/yantrikdb-server/src/main.rs::run_server` and runs every 10 seconds.
- `parking_lot::Mutex::try_lock_for(Duration)` is available — use it in places where "wait forever" is never the right answer.
- `parking_lot::MutexGuard` derefs to `&T` / `&mut T` without the `.unwrap()` ritual, which makes lock sites visually cleaner and harder to misread.

**Mechanical rule**: if you write `use std::sync::{Mutex, ...}`, delete it. Use `use parking_lot::Mutex;` instead. `std::sync::Arc` is fine and encouraged — it is a smart pointer, not a lock.

---

## Rule 2 — Never hold a `std::sync::Mutex` guard across an `.await`

(Also applies to `parking_lot::Mutex`, but especially std.)

**Why**: synchronous mutex guards are not `Send` — well, they are, but holding one across an `.await` transitively makes the future `!Send`, so tokio refuses to schedule it on the multi-threaded runtime. You will either get a compile error or, worse, a deadlock that only triggers under load.

**Structural prevention**: all HTTP + wire-protocol handlers in this codebase route through `tokio::task::spawn_blocking` before touching the engine. See [crates/yantrikdb-server/src/http_gateway.rs::execute_cmd](crates/yantrikdb-server/src/http_gateway.rs) and [crates/yantrikdb-server/src/server.rs::handle_connection](crates/yantrikdb-server/src/server.rs). Because the engine work happens on a blocking thread, the async runtime cannot be wedged by a synchronous lock, AND a mutex guard cannot possibly span an `.await` — there are no awaits inside the spawn_blocking closure.

**Rule for new code**: if you are writing an `async fn` that needs `engine.lock()`, your first instinct should be wrong. Refactor to wrap the synchronous section in `spawn_blocking`. If you cannot, talk to another maintainer before merging.

---

## Rule 3 — Lock ordering invariant

Always acquire locks in this global order. Never acquire in reverse order.

```
control  >  tenant_pool  >  engine  >  conn  >  vec_index / graph_index / scoring_cache  >  active_sessions  >  hlc
```

Read this as: "if you already hold `control`, you may take any lock to its right. If you already hold `engine`, you may NOT take `control` — you must drop `engine` first."

**Why this specific order**: it mirrors the natural call graph. Public-facing code starts by resolving a token (control → engine), then performs engine operations that may touch internal indices and the connection. The `conn` mutex is always taken AFTER the outer engine mutex because the engine wraps the connection logically.

**If you feel tempted to violate this order**, the correct fix is almost always: drop the outer guard first, do the inner work, then re-acquire. Use explicit `drop(guard)` or a tight `{ let g = ...; ... }` block to control scope.

---

## Rule 4 — Never hold `db.conn()` across a call that takes `&YantrikDB`

This is the specific pattern that caused the v0.5.7/v0.5.8 hang.

```rust
// WRONG — deadlocks if helper() calls db.conn() internally.
let conn = db.conn();
some_helper(db);  // <-- may transitively call db.conn() → SAME-THREAD DEADLOCK

// ALSO WRONG — `if let` scrutinee extends the temporary guard lifetime
// through the entire body, including the call to conflict_exists(db, ...).
if let Some(x) = helper_taking_conn(&*db.conn(), ...) {
    conflict_exists(db, ...);  // <-- db.conn() acquired while outer still held
}

// CORRECT — scope the guard tightly, drop before calling into db.*
let x = {
    let conn = db.conn();
    helper_taking_conn(&conn, ...)
};
conflict_exists(db, ...);  // guard is dropped, safe to reacquire
```

**Why**: `YantrikDB::conn()` returns `MutexGuard<rusqlite::Connection>` from a non-reentrant `Mutex`. If the same thread tries to acquire it a second time while already holding it, it blocks forever (parking_lot's default Mutex is non-reentrant; use `ReentrantMutex` if you genuinely need recursion, which you almost never do).

**Structural prevention**:

- The audit in [crates/yantrikdb-core/src/cognition/](crates/yantrikdb-core/src/cognition/) confirmed no remaining instances of this pattern as of v0.5.9.
- Runtime deadlock detection (Rule 1) will catch any future instance within 10 seconds of it firing.
- If you write a new helper that takes `db: &YantrikDB`, document in the doc comment whether it internally takes `db.conn()`, so callers know not to hold a conn guard across it.

---

## Rule 5 — Read-heavy fields use `RwLock`, write-heavy use `Mutex`

YantrikDB's engine struct uses:

| Field | Lock type | Rationale |
|---|---|---|
| `conn: Mutex<Connection>` | `Mutex` | rusqlite::Connection is `!Sync`; all access serialized |
| `hlc: Mutex<HLC>` | `Mutex` | HLC tick mutates state, almost always a write |
| `scoring_cache: RwLock<HashMap>` | `RwLock` | read-heavy; many recalls read, consolidation writes |
| `vec_index: RwLock<HnswIndex>` | `RwLock` | read-heavy; recall reads, rebuild writes |
| `graph_index: RwLock<GraphIndex>` | `RwLock` | read-heavy; expand_entities reads, relate writes |
| `active_sessions: RwLock<HashMap>` | `RwLock` | read-heavy; session lookup reads, session start writes |

When introducing a new shared field, pick the lock type based on expected read:write ratio. If you are not sure, use `Mutex` — it is simpler and RwLock's write-starvation semantics are subtle.

---

## Rule 6 — Guard lifetime: tight scoping with explicit blocks

Always bind a lock guard to a named variable and keep the scope as small as possible. Prefer explicit `{ ... }` blocks to force early drop.

```rust
// Preferred
let result = {
    let conn = db.conn();
    conn.query_row(...)?
};

// Acceptable if the function ends immediately after
let conn = db.conn();
conn.execute(...)?;

// Discouraged — guard lives across later ops
let conn = db.conn();
conn.execute(...)?;
do_other_things();  // <-- conn still held
more_stuff();
```

**Why**: the shorter the scope, the smaller the window for accidentally calling into code that re-acquires the same lock.

---

## Rule 7 — Metrics on every lock site

As of v0.5.9, every lock acquisition site emits a histogram sample to `/metrics`:

- `yantrikdb_lock_wait_seconds{lock="engine",site="<fn>"}` — time from `lock()` called to guard acquired
- `yantrikdb_lock_hold_seconds{lock="engine",site="<fn>"}` — time from acquire to drop

Alerts trigger when p99 > 5s sustained for 1 minute. This gives us ~30s to know about any future deadlock, versus the two days it took to find the v0.5.7 self-deadlock.

When adding new lock sites, wire them through the same macro (`lock_with_metrics!`) so they get instrumented automatically.

---

## Checklist — adding a new `Mutex<T>` or `RwLock<T>` field

1. Is the lock actually needed? Could you use `Arc<T>` + immutable data, or message-passing via `tokio::sync::mpsc`?
2. If yes, import from `parking_lot`, not `std::sync`.
3. Where does the new lock fit in the ordering from Rule 3? Update this document.
4. Document on the struct field what it protects and what invariant holds while held.
5. Audit every acquisition site for Rule 4 (no guard held across `&self` calls that might re-enter).
6. Wire metrics via `lock_with_metrics!`.
7. If the lock is taken from `async` code, verify Rule 2 (use spawn_blocking).
8. Add a targeted test that exercises the contention path.

---

## Checklist — when debugging a hang

1. Is the process responsive to `/v1/cluster` (HTTP gateway) or `/v1/health`?
2. Check `yantrikdb_lock_wait_seconds` in `/metrics`. If any lock has p99 > 1s, that's your deadlock candidate.
3. Check the deadlock detector log: `grep "DEADLOCK DETECTED" journalctl -u yantrikdb`. parking_lot will have logged backtraces of every deadlocked thread.
4. If the detector shows nothing but the server is stuck, grab `gdb -p $(pidof yantrikdb) -batch -ex 'thread apply all bt'` and look for threads in `lock_contended`. The thread NOT in `lock_contended` is the lock holder — look at what it's doing.
5. The watchdog at `/usr/local/bin/yantrik-watchdog.sh` captures this diagnostic bundle automatically to `/var/log/yantrik-diag/hang-<timestamp>/`.
6. Read the captures before restarting. A hang you restart without inspecting is a bug you will hit again.

---

## History

- **v0.5.7** — fixed `/metrics` holding control lock across engine lock
- **v0.5.7** — moved all engine calls to `spawn_blocking` in HTTP + wire handlers
- **v0.5.8** — fixed `check_redundancy` self-deadlock on `Mutex<Connection>` re-entrance
- **v0.5.9** — migrated all Mutex/RwLock to `parking_lot`, enabled deadlock detection at runtime, added lock-wait metrics, published this document
