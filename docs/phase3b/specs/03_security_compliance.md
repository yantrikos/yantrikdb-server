# FieldOps — Security & Compliance

## Identity

The compliance desk has reviewed this program and approved use of our
self-hosted Keycloak deployment at `keycloak.corp.internal` as the
identity plane. Two other FieldOps-adjacent tools already federate into
that Keycloak — the permit-intake service and the asset-catalog
service — and the realm, roles, and group structure are already in
place. External identity providers — Auth0, Okta, Google Workspace as
an IdP, Azure AD, Cognito, Firebase Authentication, any managed SSO —
are NOT approved for this program. The compliance reasoning: field data
includes client-site geolocation traces and asset photographs that
were scoped under contract as data-in-our-custody, and the contract
negotiation explicitly ruled out routing identity claims through a
third-party SaaS. Do not propose any of those third-party IdPs as a
fallback, as a staging shortcut, or as a "for local dev only"
convenience. The dev environment federates into the same Keycloak via
VPN.

If a future program wants to re-evaluate third-party identity that will
be a separate compliance engagement. Out of scope here.

## Photo and attachment storage

Captured photos and attachments are stored in an S3-compatible object
store — specifically, our internal MinIO deployment at
`objects.corp.internal`. Every object must be written with AES-256
server-side encryption (SSE-S3 equivalent, using the MinIO-managed key
hierarchy). We do not store media files on the application pod's local
filesystem, we do not store them in PostgreSQL as bytea, and we do not
route them through a separate cloud provider's object store. The MinIO
deployment is what we have; use it. Inline references to objects go in
the PostgreSQL row as canonical URIs like `s3://fieldops/<capture-id>/
<asset-id>`.

## Data classification

Most captured data is Internal classification. A small subset — the
geolocation traces for restricted client sites — is Confidential. The
platform must tag a record as Confidential at capture time and route it
through a stricter retention and access path. The storage backend is
the same (same PostgreSQL cluster, same MinIO bucket); differentiation
is enforced at the application layer against Keycloak roles. A user
without the `fieldops.confidential` role should get a 403 on read
attempts for Confidential records, logged to the audit trail.

## Audit log

Every write action must be logged to an append-only audit log. The
audit format is `audit-format-v3` — already defined in the internal
wiki by the compliance desk, and it is the canonical format across our
services. Do not invent a different format; do not use
event-sourcing-style JSON payloads with bespoke fields; do not propose
a custom "better" schema. The compliance team reads the audit log
quarterly and they consume `audit-format-v3` with their existing tools.
The format's key fields include actor_id (Keycloak user_id), action
verb, target_id, target_classification, and a monotonic timestamp in
UTC. The audit log sink is a dedicated Loki stream with long retention
(see below).

## Logging

All application logging uses our internal `company-common-logging`
Python library, not stdlib `logging` directly and not `loguru` or
`structlog`. The company-common-logging library wraps stdlib logging
with our required fields (trace_id injection from Tempo, service
identity from the env, structured JSON output for Loki). Every Python
service at this company uses this library, which is how Loki/Tempo
correlation works — if you propose stdlib logging or a third-party
logging library, trace correlation will silently break and the on-call
dashboards won't work. It is installed as
`company-common-logging @ artifactory.corp.internal/pypi/`.

## Transport

See the infrastructure brief for TLS posture: all channels TLS
end-to-end, no plaintext internal HTTP, mTLS between internal services.

## Offline capture & at-rest encryption

Because the client works offline, the on-device store holds some
Confidential data between sync cycles. The device store is encrypted
at rest. Key material is derived from the user's Keycloak session — so
when a user rotates out of the program (access revoked in Keycloak),
the keys are unusable on any laptop they previously held. Specifics of
the KDF will be defined in a follow-up brief; for this architecture
proposal, treat "local store encrypted at rest with keys derived from
Keycloak session identity" as a binding requirement. The local SQLite
file on the laptop is encrypted at rest.

## Time and timezone

All timestamps stored in PostgreSQL and in the audit log are in UTC.
The client displays timestamps in the user's local timezone (derived
from the Keycloak profile's `zoneinfo` claim), never in server TZ,
never in "whatever the laptop clock says." UTC-in-storage,
local-TZ-for-display: write it this way and be explicit about it in
the API layer, because every previous project here has bungled
timezone handling at least once.

## Data retention

Captured field records (rows plus media objects) are retained for 7
years from capture date, then hard-deleted with tombstone rows
preserving the record ID and classification. The audit log is retained
indefinitely — it is the system of record for who did what, and
compliance requires it be kept. Do not propose "retain everything
forever" or "7-year retention on everything including audit"; the
split is deliberate.

## Out of scope

Red-team penetration testing is scheduled for pre-launch. Not a
concern for the architecture proposal.
