# FieldOps — Product & Team Context

## Why we are rebuilding

FieldOps is the internal tooling used by our field operations team —
surveyors, drillers, inspectors — across roughly 260 users. The current
platform is a VB.NET tool first shipped in 2011, then patched heavily
through three different vendor handovers. By 2023 the codebase had
accumulated about fourteen years of architectural debt and nobody on the
current internal team can reliably extend it without breaking something
else. The 2024 "billing cycle" outage — where a patch to the timesheet
export broke the inbound photo ingest — cost the team four days of field
capture. That was the trigger. Engineering leadership signed off on a
rewrite in Q4 2024; target launch is Q3 of next year with a soft internal
rollout to two regional teams first, then full cutover by Q1 of the year
after.

## Who actually uses it

The field team is not a software team. They use laptops as tools. The
central office issues ThinkPad X1 Carbons running Windows 11 Pro — that
is the default, and the one most users run. The regional offices in the
western territory standardized on Dell XPS 13 laptops running Ubuntu
22.04 LTS, because the local IT lead at the time preferred open-source
and won that argument back in 2021. Procurement is not going to undo
that; both platforms are permanent. A Mac request came up during the
2023 planning round from the analytics group, who wanted to use the data
export from their own MacBooks; that request was declined by IT because
the procurement contract doesn't include Apple hardware and the
compliance desk doesn't want to certify a third client platform. Mac is
not a target. Any proposal that plans for Mac support is planning for
work we will not staff.

Most of our users spend two to five days in the office in a cycle, then
multi-week stretches in remote locations — offshore rigs, mountain
passes, rural transmission corridors, desert survey lines. In those
locations the connection is either expensive satellite (high-latency,
metered), intermittent LTE (usable only in the morning hours when a
distant tower has capacity), or entirely absent for days. Our own
internal survey of field laptops across Q2 2024 found a mean
connectivity availability of 31% of the working day, with the 5th
percentile at 0%. The platform must let field staff do their full day
offline and reconcile whenever the laptop reaches a known-good network
(office Wi-Fi, hotel Wi-Fi on a travel day, regional office on a
stopover). We are not shipping a design where the client requires an
open connection to the server during field work. That is a non-starter.

## Developer team

The backend team is five engineers: three of them built our previous two
Python services (the permit-intake service and the asset-catalog
service), and they have asked that we standardize on Python going
forward. Our build images are on Python 3.11.6 in production and the
team wants to use the newer match-statement and exception-group
features. We are not downgrading Python to 3.9 or 3.10 to match some
external library — pin the library, not the runtime. Front-end is one
engineer plus a contractor; they use TypeScript and the choice of UI
framework is being made separately and is not in scope for the
architecture proposal.

## Performance expectations

Users capture data at 1-second cadence for certain field surveys, and
the existing tool's 3–4 second save latency is the single most common
complaint in our user-satisfaction survey. For the rebuild, the product
manager has made p99 server-side latency under 200 milliseconds a hard
acceptance criterion — this is written into the sign-off documentation,
not a stretch goal. Meeting this is non-negotiable; if an architectural
choice pushes latency above 200ms p99, pick a different choice.

## Image distribution

All our service container images live in our internal JFrog Artifactory
at `artifactory.corp.internal/docker/`. Public Docker Hub is blocked at
the corporate egress firewall — builds that reference `python:3.11-slim`
or `postgres:15-alpine` from Docker Hub will fail in CI because the
builder image cannot reach Docker Hub. Every base image we use must be
mirrored into Artifactory first, and the mirror process runs weekly.
This is a policy from the 2023 supply-chain audit and it is not moving.
So the architecture needs to specify its base images in terms of the
internal mirror — and if a new base image is needed, the platform team
has a ~1-week SLA on adding it to the Artifactory mirror, which the
project timeline must accommodate.

## What we are building

The core of the system is a record-of-work database: field staff capture
observations, photos, and structured measurements on the laptop; the
client reconciles offline captures on reconnect; findings are submitted
up the chain to the central operations team; the central team reviews,
annotates, and either finalizes the record or sends it back to the
field. The system has a web portal for the central team, a thick-client
desktop app for the field team, and an API surface for a handful of
downstream systems (the billing system, the asset catalog, the
compliance desk's audit tools).

## Related briefs

You will write the architecture proposal. Before drafting, read the
infrastructure brief (database, network, deployment, container policy,
queueing) and the security-and-compliance brief (identity, data
classification, storage, audit). Specifics in those briefs bind your
proposal — there are a number of hard constraints and several
anti-patterns to avoid.
