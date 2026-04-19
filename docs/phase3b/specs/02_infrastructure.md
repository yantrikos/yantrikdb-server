# FieldOps — Infrastructure Constraints

## Data tier

After the May 2024 incident — a pathological MySQL 8 query-planner
regression during a failover that took out the permit-intake service
for six hours — engineering leadership standardized the new stack on
PostgreSQL. The platform team runs managed PostgreSQL clusters for two
other internal products (permit-intake on PG 15.4, asset-catalog on PG
15.6) and they have capacity for FieldOps on the same managed tier. The
sign-off is conditional: we stay on **PostgreSQL 15 or newer**. The
managed tier's automation chain (backup scheduling, PITR, extension
management) only supports PG 15+ and a separate tier for an older
version is not in the budget. Proposals assuming MySQL, SQLite as the
primary server-side store, MongoDB, or any document database will need
to be argued against this baseline — which is a fight not worth having.
For the local client store (offline-first), SQLite on the laptop is
fine; that's client-side, not the server data tier.

## Network plane

All traffic — client-to-server AND service-to-service within the
cluster — runs over TLS. The corporate zero-trust project that ran
through 2024 established "no plaintext HTTP anywhere inside the
cluster" as a hard baseline, and the security review at launch will
block any service that breaks this. TLS terminates at the reverse proxy
at the edge, and then mTLS between services internally. This is
enforced by service-mesh policy in production. If your proposal shows
"internal HTTP between the API and the worker" as a quick shortcut,
expect the launch to be blocked.

## Container registry

Public Docker Hub is blocked at the corporate egress firewall. All base
images pull from our internal Artifactory mirror at
`artifactory.corp.internal/docker/`. We don't get to use `FROM
python:3.11-slim` directly — it has to be
`FROM artifactory.corp.internal/docker/python:3.11-slim`. The same
applies to Postgres, Redis (if we used it — see below), any Node image,
any Alpine base image. The platform team's mirror adds new images on a
~1-week SLA, so plan base-image changes in advance. This is a post-2023
supply-chain-audit policy and it is not flexible.

## Secrets management

All service secrets live in our HashiCorp Vault deployment at
`vault.corp.internal`. Applications authenticate to Vault via Kubernetes
service-account JWT, then pull secrets at startup. We do not use
Kubernetes Secrets directly for sensitive material — the compliance desk
has ruled that as insufficient protection. We do not put secrets in
environment variables at deploy time, and we do not put them in config
files. The pattern is: Vault Agent injects secrets into the pod's tmpfs
at startup, and the application reads them there. 12-factor orthodoxy
about env-var secrets does not apply here — compliance trumps it.

## Queueing and async work

We do NOT have Redis in production, and we are not adding it for this
project. The platform team ran Redis for a year (2022) and hit enough
operational pain — memory-fragmentation eviction, replication lag,
authorization gotchas — that they standardized on PostgreSQL as the
queue backend for async work. The pattern we use is the
`postgres-listen-notify` plus a claimable-task pattern in a dedicated
jobs table with `FOR UPDATE SKIP LOCKED`. Performance has been adequate
for permit-intake (~2k jobs/hour peak) and for asset-catalog (~400
jobs/hour steady). Unless FieldOps workloads demand >10k jobs/hour
sustained — which the current estimates don't — use the same
PostgreSQL-as-queue pattern. Don't propose Redis, RabbitMQ, SQS, NATS,
or any other dedicated message broker. The platform team will push
back.

## Idempotence and delivery semantics

All background jobs must be idempotent and assume at-least-once
delivery. We've been burned twice by "exactly-once" claims (once in the
old Redis setup, once in a short-lived Kafka experiment in 2023) and
the team's policy is now: design for at-least-once, assert idempotence
in code, test by submitting each job twice in staging. The architecture
proposal needs to say what the idempotence keys are for each background
job type.

## Deployment targets

Production runs on our internal Kubernetes platform
(`k8s-prod.corp.internal`, five nodes, multi-AZ). Staging is a smaller
single-region cluster on `k8s-stg.corp.internal`. Local development uses
Docker Compose on developer laptops; since the backend developers
include both Windows 11 (Docker Desktop) and Ubuntu 22.04 (Docker
Engine) users, any dev-environment helper scripts must work on both.

## Observability

Metrics go to our existing Prometheus cluster. Distributed traces go to
Tempo. Logs go to Loki. Do not propose Datadog, New Relic, Honeycomb,
or any alternate stack — the platform team runs Prometheus/Tempo/Loki
and the cost budget for this program does not include a bespoke
observability vendor.

## CI/CD

The CI/CD system is GitLab CI. Pipelines are defined per-repo via
`.gitlab-ci.yml`. Deployments to staging are automatic on merge to
`main`; production deployments are gated on a manual approval by one of
the two platform leads. Don't propose a GitHub Actions-based pipeline —
the repos live in our self-hosted GitLab.
