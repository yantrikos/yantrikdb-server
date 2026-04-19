# FieldOps — Infrastructure Constraints

## Data tier

After the last outage postmortem — specifically the May 2024 incident where
our MySQL 8 cluster hit a pathological query-planner regression during a
failover — engineering leadership chose to standardize the new stack on
PostgreSQL. The platform team already operates managed PostgreSQL 15 clusters
for two other internal products, and they have capacity on those clusters for
FieldOps. We have explicit sign-off to reuse that capacity, but only if we
stay on PostgreSQL 15 or newer — the managed tier does not support older
major versions, and there is no budget to stand up a separate tier for an
older release. Proposals that assume MySQL, SQLite-as-primary, or a document
store would need to be argued against this baseline, which is not a battle
worth picking here.

## Network plane

All service-to-service traffic and all client-to-server traffic runs over
TLS. The corporate perimeter terminates TLS at our reverse proxies but the
policy team has confirmed that TLS must be end-to-end: no plain HTTP inside
the cluster, no mixed-content tolerated at the edge. The zero-trust project
that ran last year established this as a baseline and any new service must
align. If a diagram shows "internal HTTP" between services, expect the
security review to block the launch.

## Edge connectivity

Because the field team operates in connectivity-poor environments, the client
must be able to function against a local on-device store and reconcile on
reconnect. We will not ship a design where the client stays open against a
remote server during field work. The sync protocol runs when the laptop
reaches a known-good network (office or hotel Wi-Fi); during field work the
client writes to local storage and the server is unaware.

## Deployment targets

Production runs on our internal Kubernetes platform. Staging is a smaller
single-region cluster. Local development runs on developer laptops against
Docker Compose; since the backend developers use both Windows and Ubuntu,
Docker Desktop or Docker Engine respectively, any dev-environment scripts
must work on both.

## Observability

Metrics land in our existing Prometheus stack, traces in Tempo, logs in Loki.
Do not pick a bespoke stack.
