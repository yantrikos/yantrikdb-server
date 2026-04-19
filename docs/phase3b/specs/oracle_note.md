# Oracle note for FieldOps architecture proposal (15 constraints)

(Ideal hand-written summary — used as condition C injected into session 2.)

## Hard constraints (must not violate)

1. **Platform targets:** Windows 11 Pro (ThinkPad X1 Carbon) and Ubuntu
   22.04 LTS (Dell XPS 13) clients only. Mac is NOT a target — declined
   by IT in 2023. Do not plan for macOS.

2. **Language/runtime:** Python 3.11+ on the backend (3.11.6 in prod).
   Team uses match statements and exception groups. Do NOT propose Go,
   Rust, Node, Java, C# or downgrade Python.

3. **Database:** PostgreSQL 15 or newer. Locked — the managed tier only
   supports PG 15+. Do NOT propose MySQL, MariaDB, SQLite-as-primary,
   MongoDB, or any document database for the server tier. (SQLite on
   the client laptop for offline-first is fine.)

4. **Performance:** p99 server-side latency under 200ms under realistic
   load. Hard acceptance criterion, not aspirational.

5. **Transport:** TLS end-to-end. No plaintext HTTP anywhere, including
   service-to-service inside the cluster. mTLS between internal
   services. Security review will block mixed content.

6. **Identity:** Self-hosted Keycloak at keycloak.corp.internal ONLY.
   Auth0, Okta, Google Workspace as IdP, Azure AD, Cognito, Firebase
   Auth — all explicitly disallowed. No "dev shortcut" managed SSO.

7. **Offline-first client:** Must work fully offline on laptop, reconcile
   on reconnect to a known-good network. Do NOT propose a design where
   the client holds a live connection to the server during field work.

8. **Container registry:** Internal Artifactory at
   `artifactory.corp.internal/docker/` only. Public Docker Hub is blocked
   at corporate egress. Every base image (`python:3.11-slim`, postgres,
   etc.) must be from the internal mirror.

9. **Secrets:** HashiCorp Vault at `vault.corp.internal` only. Pulled
   via Vault Agent injecting to tmpfs at startup. Do NOT put secrets in
   K8s Secrets directly, in environment variables, or in config files.
   12-factor env-var conventions do NOT apply — compliance trumps.

10. **Queue:** PostgreSQL LISTEN/NOTIFY plus a jobs table with
    `FOR UPDATE SKIP LOCKED`. Do NOT propose Redis, RabbitMQ, SQS, NATS,
    Kafka, or any dedicated message broker. Redis in particular is
    explicitly banned (operational pain in 2022).

11. **Jobs must be idempotent, at-least-once:** Every background job
    needs an idempotence key. Design for at-least-once delivery, not
    exactly-once. Test by submitting each job type twice in staging.

12. **Object storage:** S3-compatible MinIO at `objects.corp.internal`,
    with AES-256 server-side encryption (SSE-S3 equivalent). Do NOT
    store media on the application pod's filesystem, do NOT use
    PostgreSQL bytea, do NOT route through external cloud object stores.

13. **Audit log format:** `audit-format-v3` — already defined by the
    compliance desk. Do NOT invent a new schema, do NOT propose custom
    event-sourcing JSON, do NOT "improve" the format.

14. **Logging library:** `company-common-logging` (installed from
    `artifactory.corp.internal/pypi/`). Do NOT use stdlib `logging`
    directly, loguru, or structlog — Tempo/Loki correlation will break
    silently.

15. **UTC in storage, local TZ for display:** All PostgreSQL and audit
    log timestamps in UTC. Client displays in user's local TZ from
    Keycloak profile's `zoneinfo`. Be explicit at API layer.

## Supporting context

- Deployment: internal Kubernetes (`k8s-prod.corp.internal`).
- Observability: existing Prometheus + Tempo + Loki stack — don't
  propose Datadog/NewRelic/Honeycomb.
- CI/CD: self-hosted GitLab CI (not GitHub Actions).
- Data classification: Internal by default, Confidential for
  restricted-site geolocation; differentiated by Keycloak role at the
  app layer (same physical storage).
- Data retention: raw captures 7 years then hard delete with
  tombstone; audit log forever.
- Local client store: SQLite, encrypted at rest, keys from Keycloak
  session identity.

## Open questions for the proposal to address

- Sync conflict resolution: last-write-wins vs CRDT vs something
  domain-specific?
- Audit log write: same DB transaction as the primary write (two-phase)
  or outbox pattern with async shipping to Loki?
- Confidential-record routing: does the local SQLite store hold
  Confidential rows in a separate encrypted partition, or the same
  one?

## For future you

Anti-patterns that will trigger rejection if proposed: any macOS target,
any third-party SaaS identity, any direct Docker Hub base image, any
Redis/RabbitMQ, any Datadog-style observability, any exactly-once
semantics, any stdlib logging direct, any bytea photo storage, any
custom audit log format, any env-var secrets. If the proposal drifts
toward any of these, stop and re-read this.
