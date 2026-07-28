# RFC 030 — Multi-user admin accounts + RBAC (control-plane extension)

**Status:** Draft · **Depends on:** RFC 029 (control-plane replication) · **Supersedes:** the master-token-only admin gate

## The gap

RFC 029 made the control plane replicate, but admin authorization is still
a single shared **master token** (`cluster_secret`) pasted into the studio.
That is a bootstrap credential, not an enterprise auth model: no named
operators, no roles, no least-privilege, no per-actor audit attribution, no
way to revoke one operator without rotating the cluster secret for everyone.

This RFC adds **named admin accounts with roles**, a **session-based login**
(the raw secret is never stored in the browser), and **RBAC enforcement** on
every admin route — all replicated through the same YRP control plane so
identity-of-operators survives failover exactly like tokens/databases do.

## Model

### Admin users are replicated control-plane entities

A new `admin_users` table in `control.db`, materialized from replicated
control ops (mirroring tokens/databases — RFC 029):

```sql
CREATE TABLE admin_users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,      -- argon2id, NEVER plaintext
    role          TEXT NOT NULL,      -- 'owner' | 'admin' | 'readonly'
    disabled_at   TEXT,               -- soft-disable (tombstone-shaped)
    created_at    TEXT NOT NULL
);
```

New `ControlOp` variants (serde_json in `Payload::Control`, replicated +
idempotent on the `username` key, verifier-hash only):

- `CreateUser { username, password_hash, role, created_at }`
- `SetUserRole { username, role }`
- `SetUserPassword { username, password_hash }`   — rotation
- `DisableUser { username, disabled_at }`          — revoke an operator

**Password hashing is argon2id, not SHA-256.** Tokens are 256-bit random →
SHA-256 is fine; passwords are low-entropy human input → they need a slow,
salted KDF. Add the `argon2` crate. Only the hash is replicated (Invariant 3
extends to passwords).

### Roles

| Role | Can |
|---|---|
| `owner` | everything, incl. user management (create/role/disable users) |
| `admin` | databases + tokens + quotas (mint/rotate/revoke), view audit — **no** user management |
| `readonly` | view databases/tokens/users/audit — **no** mutations |

Roles are a total order (`readonly < admin < owner`); a route declares the
minimum role it needs.

### Session auth (login, not secret-pasting)

- `POST /v1/admin/session { username, password }` → argon2-verify against
  `admin_users` → issue a **signed session token**: `base64(payload).sig`
  where `payload = { sub: username, role, iat, exp }` and
  `sig = HMAC-SHA256(server_session_key, payload)`, TTL ~60 min.
- **`server_session_key = HMAC-SHA256(cluster_secret, "ydb-admin-session-v1")`** —
  derived from the shared secret, so a session minted on any node verifies on
  every node (stateless; survives restart and failover; no session table).
- Admin routes accept `Authorization: Bearer <session_token>`; the guard
  verifies signature + expiry + role, and sets the audit actor to `sub`.

### Master token = break-glass owner + bootstrap

The `cluster_secret` continues to authorize admin routes as an implicit
**owner** (break-glass, and the only way in before any user exists). The
studio's first-run flow: log in with the master token → create the first
`owner` account → subsequent logins use accounts. The master token is never
removed (it is the recovery path if all accounts are lost/disabled).

## Enforcement

A single guard replaces `require_master_token`:

```
fn require_role(state, headers, min: Role) -> Result<AdminActor, AppError>
```

Accepts, in order: (1) a valid session token (verify HMAC+exp, check
`role >= min`); (2) the master token (→ owner). Returns an `AdminActor
{ name, role }` used for audit attribution. Every admin route declares its
`min` role. Constant-time compare for the master token (closes the earlier
timing nit).

## New / changed endpoints

- `POST /v1/admin/session` — login → session token (public; rate-limited).
- `GET  /v1/admin/me` — current actor + role (drives the studio's role-aware UI).
- `GET/POST /v1/admin/users`, `POST /v1/admin/users/{u}/role`,
  `/password`, `/disable` — **owner** only.
- `GET  /v1/admin/databases`, `GET /v1/admin/tokens` — clean list endpoints
  gated by `require_role(readonly)` (fixes the RFC 029 gap where the studio
  listed via `control-snapshot`, which only checks `state.cluster` and 401s
  in yrp mode).
- `POST /v1/admin/databases`, `/tokens`, `/tokens/revoke` — **admin**.
- `POST /v1/admin/tokens/rotate { token|hash }` — mint a new token for the
  same db+label, revoke the old (both replicated); returns the new plaintext
  once. **admin**.
- `GET/PUT /v1/admin/databases/{id}/quota` — view/set quotas (the
  `control.db` `quotas` table already exists). **admin** for PUT.
- `GET /v1/admin/audit?limit=&kind=` — read the audit trail. **admin**.
  Requires a **persistent audit sink** (control.db `audit_log` table) since
  admin actions must be durably attributable; wire admin ops to emit
  `AuditEvent`s with the `AdminActor` as actor.

## Invariants (security)

1. **Replicate verifier material only** — argon2 password hashes + token
   hashes, never plaintext (extends RFC 029 Invariant 3).
2. **Session tokens are stateless + signed** — no server session store to
   leak; compromise of one node's memory does not yield a forgeable session
   beyond what the shared `cluster_secret` already implies.
3. **Fail closed** — an unparseable/expired/under-privileged session is 401/403,
   never a silent downgrade. The control-freshness gate (RFC 029 inc2-A) still
   applies: a catching-up node refuses admin auth too.
4. **Owner floor** — the last enabled `owner` cannot be disabled or demoted
   (prevents locking everyone out); the master token remains the recovery path.
5. **Bootstrap ordering (F4)** — user control ops are new `Payload::Control`
   shapes; all nodes must run the RFC-030 build before the first user op is
   minted (same operational rule RFC 029 established, honored during the
   rolling redeploy).

## Delivery (release cadence)

- **0.13.1** — the foundation: replicated users + argon2 + session login +
  RBAC guard + clean list endpoints + token rotation + quotas + persistent
  audit + audit endpoint. Studio v2 login + user/db/token/quota/audit
  management, role-aware.
- **0.14** — polish + depth (audit filtering/export, token last-used,
  bulk ops, SSO/OIDC mapping onto roles — the next enterprise rung).

## Review hardening (adversarial pass — resolutions folded in)

- **H1 — real session revocation.** `admin_users` carries a monotonic
  `token_version`; the session payload includes `ver`; the guard rejects
  unless `ver == current`. `SetUserRole`/`SetUserPassword`/`DisableUser` bump
  it, so a demoted/disabled/rotated operator's live sessions die immediately,
  cluster-wide (the check rides the RFC-029 control-apply state).
- **H2 — owner-floor enforced deterministically at APPLY time**, in log
  order, inside `ControlApplySink`: `DisableUser`/`SetUserRole` re-evaluate
  "would enabled-owner count drop below 1?" against state as of that log
  position and become an identical no-op on every node if so. Never
  propose-time-only (that races two ops on different nodes to zero owners).
- **H3 — session key decoupled from `cluster_secret`.** A dedicated 32-byte
  `admin_session_key` is replicated control-plane state (a `server_secrets`
  row, seeded by a `SetAdminSessionKey { kid, key }` control op the leader
  proposes at first boot if absent). Session tokens carry `kid`; rotation is
  a new key op that invalidates all prior sessions without touching peer auth
  or the master token.
- **M1 — replicated, tamper-evident audit.** Every *mutating* `ControlOp`
  carries an `actor` (serde `default` for back-compat); `ControlApplySink`
  writes an `audit_log` row deterministically as it applies, so admin-action
  audit is quorum-durable, byte-identical on all nodes, and survives failover
  / node loss. Login / failed-login / read events stay per-node (documented).
- **M2 — dedupe by op-id, last-in-log-order wins** for role/password/disable;
  `CreateUser` against an existing username **errors** (409) at the endpoint,
  never a silent no-op.
- **M3 — argon2id runs at the request-terminating node**; only the hash is
  proposed/replicated; plaintext never enters a `ControlOp` nor peer traffic.
- **M4 — login DoS/enumeration hardening.** Per-IP + per-username rate limit
  *before* argon2; a global semaphore caps concurrent argon2 verifications;
  unknown users get a constant-time decoy verify (no enumeration).
- **M5 — session verification** does constant-time HMAC compare, verifies the
  signature over the exact received bytes *before* JSON-parsing, and rejects
  malformed tokens (segment count / base64 / empty) as 401.
- **L-fixes:** bounded clock-skew tolerance + leader-assigned `iat/exp` (L1);
  admin routes assume TLS, `Authorization` never logged (L2); a stale/
  quarantined node refuses admin *reads/mutations* even for the master token,
  surfacing staleness rather than acting on stale RBAC (L3); list endpoints
  **redact** `password_hash`/`token_hash` (L4); unknown `role` strings fail
  closed in the guard's ordering (L5); every master-token (break-glass) use
  emits a loud `actor="master-token"` audit event (L6).

## Out of scope

External IdP (OIDC/SAML) — layers on top by mapping an external identity to a
role, replicated via the same control log. RBAC is the prerequisite. Full
`admin_session_key` rotation UI + audit export/filtering land in 0.14.
