# FieldOps — Security & Compliance Brief

## Identity

The central compliance desk has reviewed this program and approved a
self-hosted identity plane. We operate an internal Keycloak deployment that
already serves two other FieldOps-adjacent tools and we will federate into
that. External identity providers — Auth0, Okta, Google Workspace as an IdP,
Azure AD — are **not approved** for this program. The reasoning in the
compliance note: the field data includes client-site geolocation traces and
asset photographs that were scoped under contract as data-in-our-custody;
routing identity claims through a third-party SaaS was scoped out in the
contract negotiation and we are not reopening that.

If a future phase wants to re-evaluate third-party identity, that will be a
separate compliance engagement and is out of scope here.

## Data classification

Most captured data is Internal. A small subset — the geolocation traces for
restricted client sites — is Confidential. The platform must be able to tag a
record as Confidential and route it through the stricter retention and access
paths; but the storage backend is the same, and the access differentiation is
enforced at the application layer against Keycloak roles.

## Audit

Every write action goes into an append-only audit log. The audit format is
already defined elsewhere (see `audit-format-v3` in the internal wiki) and
that format is canonical — do not invent a different one. The audit log is
read by the compliance team quarterly.

## Transport

See the infrastructure brief for the TLS posture. Briefly: all channels
encrypted end-to-end, no plaintext. Client certificates are not required for
the field-laptop use case; we rely on Keycloak session tokens for client
identity.

## Offline capture and confidentiality

Because the client works offline, the on-device store holds some Confidential
data between sync cycles. The device store must be encrypted at rest. Key
material is derived from the user's Keycloak session — when a user rotates
out of the program, we want the keys unusable on any laptop they previously
held. Specifics on the KDF will be defined later; for the architecture
proposal, treat "at-rest encryption keyed to Keycloak identity" as a
requirement, not a nice-to-have.

## Out of scope

Red-team penetration testing of the production deployment is scheduled for
pre-launch. For the architecture proposal you are writing now, that is out
of scope — do not try to pre-empt it.
