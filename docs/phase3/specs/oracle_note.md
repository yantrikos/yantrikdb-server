# Oracle note for FieldOps architecture proposal

(Ideal hand-written summary — used as condition C injected into session 2.)

## Hard constraints (must not violate)

1. **Platform targets:** Windows 11 Pro and Ubuntu 22.04 only. Mac is NOT a
   target — IT declined it in 2023 and compliance won't certify a third
   platform. Do not plan for Mac.

2. **Language/runtime:** Python 3.11+ on the backend. Team uses match
   statements and exception groups. Do NOT propose Go, Rust, Node, or
   downgrade Python.

3. **Database:** PostgreSQL 15 or newer. This is locked — the managed tier
   only supports 15+ and there is no budget for a separate tier. Do NOT
   propose MySQL, SQLite-as-primary, or document stores.

4. **Transport:** TLS end-to-end. No plain HTTP anywhere, including
   internal service-to-service calls. Security review will block mixed
   content.

5. **Identity:** Self-hosted Keycloak ONLY. Auth0, Okta, Google Workspace,
   Azure AD are explicitly disallowed by compliance. Reason: field data is
   contractually data-in-our-custody.

6. **Offline-first client:** Must work offline against local on-device
   store; sync on reconnect. Do NOT propose a design where the client
   stays connected to the server during field work.

7. **Performance:** p99 server-side latency under 200ms under realistic
   load. This is a hard acceptance criterion, not aspirational.

## Other context

- Deployment: internal Kubernetes (prod), smaller staging cluster, Docker
  Compose for dev.
- Observability: Prometheus + Tempo + Loki (existing stack — do not invent
  new).
- Audit: append-only log, format `audit-format-v3` (already defined — do
  not redesign).
- On-device store for offline capture must be encrypted at rest, keyed to
  Keycloak identity so keys are revocable on user rotation.
- Data classification: most Internal, restricted-site geolocation is
  Confidential — differentiate at app layer via Keycloak roles.
- Red-team pen-testing is pre-launch and out of scope for the proposal.

## Open questions for you to handle in the proposal

- How exactly does the client sync reconcile write conflicts between
  offline edits and server state?
- What is the event model for the audit log writes relative to the primary
  DB writes (two-phase? outbox?)
- How does the Confidential-classification routing interact with sync —
  does the client hold Confidential data in the same local store as
  Internal?

## For future you

If you find yourself proposing Mac support, Okta SSO, MongoDB, or
always-online clients, stop. The docs were explicit.
