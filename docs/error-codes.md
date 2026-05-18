# yantrikdb-server HTTP error codes

Every error response across `/v1/*` carries a structured envelope:

```json
{
  "error": {
    "code": "stable_id",
    "message": "human-readable text",
    "hint": "actionable hint (optional)"
  }
}
```

**Branch on `code`, not `message`.** The message wording is allowed to
change between releases; the code is part of the stable API surface.

The source of truth for the code enum is
[`crates/yantrikdb-server/src/api/errors.rs`](../crates/yantrikdb-server/src/api/errors.rs)
(`ApiErrorCode`). This document mirrors it. **Adding a new code requires
updating both files in the same PR**, gated by the
`every_code_has_snake_case_wire_string` + `all_wire_strings_are_unique`
unit tests.

---

## 401 Unauthorized

### `unauthenticated`

The request did not present valid authentication credentials.

- Retryable: no (until credentials change)
- Client action: present a valid token

---

## 403 Forbidden

### `forbidden`

Authenticated but lacks permission for the requested action.

- Retryable: no (until token grants change)
- Client action: request a token with the required permission

### `namespace_not_found`

The requested namespace is not visible to this token. Per the issue-#39
policy: **403 reveals existence** rather than 404 indistinguishable.
This is a deliberate choice — most internal tooling already enumerates
namespaces via `/v1/identity-scope`, and distinguishing
visible-but-forbidden from non-existent helps debugging.

- Retryable: no
- Client action: call `GET /v1/identity-scope` to see visible namespaces

### `insufficient_scope`

Authenticated and namespace is visible, but the specific permission
scope required for this endpoint is not held by the token.

- Retryable: no
- Client action: request a token with the required scope
  (`memory:read`, `memory:export`, `memory:graph:read`, `scope:read`,
  `admin:read`)

---

## 404 Not Found

### `memory_not_found`

Memory with the given RID does not exist (or is not visible to this
token, when 403 disclosure would leak existence-status from a
restricted namespace).

- Retryable: no
- Client action: confirm the RID; if it was just written, retry with
  `?min_seq=N` from the write response (see `replica_behind` below)

### `conflict_not_found`

Conflict with the given ID does not exist.

### `skill_not_found`

Skill with the given ID does not exist.

### `session_not_found`

Session with the given ID does not exist.

### `entity_not_found`

Entity with the given name does not exist.

---

## 400 Bad Request / 422 Unprocessable Entity

### `invalid_request`

Generic malformed request body or path.

### `invalid_query_parameter`

A query parameter failed validation: out-of-range value, wrong type,
unknown enum variant, etc. The `message` should identify the parameter.

### `invalid_cursor`

Pagination cursor is malformed or stale (cursor encoding changed
between releases).

- Retryable: yes, after dropping the cursor (start a fresh listing)

### `invalid_min_seq`

`min_seq` query parameter is malformed.

### `invalid_body`

Request body is not valid JSON or doesn't match the expected schema.

---

## 409 Conflict

### `op_id_collision`

The same `op_id` was used with a different mutation.

- Retryable: no — client bug; don't retry

### `unexpected_log_index`

Concurrent write race; client should re-read state and retry.

- Retryable: yes

---

## 412 Precondition Failed

### `replica_behind`

The replica receiving the request has not yet applied the `min_seq`
the client requested. This is the read-your-writes signal: the write
succeeded, but the read landed on a node that hasn't caught up.

- Retryable: yes, with backoff
- Client action: retry the request, optionally route to the leader

---

## 426 Upgrade Required

### `version_upgrade`

Wire-version mismatch during a rolling cluster upgrade.

- Retryable: no (operator must bring the cluster to a uniform version)

---

## 429 Too Many Requests

### `rate_limited`

Request exceeded the per-principal or per-namespace rate limit. The
response includes a `Retry-After` header.

- Retryable: yes, after the `Retry-After` duration

---

## 500 Internal Server Error

### `internal_error`

Catch-all for unexpected server failures. When emitted, include a
`request_id` in the response and a correlating log line so operators
can trace the failure.

- Retryable: maybe — depends on what failed

---

## 307 Temporary Redirect

### `not_leader`

The receiving node is not the cluster leader. Standard HTTP clients
will follow the redirect; the response body also carries `leader_id`
and `leader_addr` for clients that don't.

---

## 503 Service Unavailable

### `engine_unavailable`

Engine is currently unable to accept requests (e.g. mid-migration,
mid-reembed Swapping phase).

- Retryable: yes, with backoff

### `cluster_unavailable`

Cluster has no leader / no quorum.

- Retryable: yes, with backoff (election should resolve within seconds)

### `leader_unavailable`

Leader is known but currently unreachable from this node (network
partition, leader process restarting, etc.).

- Retryable: yes, with backoff

### `commit_timeout`

Write succeeded reaching the commit log but didn't apply within the
timeout. The response body carries the `op_id` — retry is idempotent.

- Retryable: yes, reusing the `op_id`

---

## Migration placeholder

### `generic`

**Do not emit from new code.** This is the placeholder code emitted by
the pre-issue-#39 `http_gateway::app_error()` helper for the ~125 call
sites that haven't yet been migrated to a specific code from this
registry. Each site migrates over time as its semantics become clear.

The wire envelope shape is identical to specific-coded errors, so
dashboards branching on specific codes treat `generic` as a fallback
("we don't know what failed, see the message").

---

## Adding a new code

1. Add a variant to `ApiErrorCode` in
   [`crates/yantrikdb-server/src/api/errors.rs`](../crates/yantrikdb-server/src/api/errors.rs).
2. Add the snake_case wire string in `ApiErrorCode::as_str()`.
3. Add a section to this document with HTTP status, retryability, and
   client action.
4. The unit tests in `api::errors::tests` will fail-loudly if the new
   variant is missing from the test array — keep both lists in sync.
5. Existing wire strings must never be renamed — they are a stable API
   surface.
