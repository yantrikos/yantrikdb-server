# FieldOps — Product Context Brief

## Background

FieldOps is an internal tooling platform used by our field operations team
(surveyors, drillers, inspectors). We are rebuilding the platform from scratch
because the current VB.NET-based tool has accumulated fourteen years of patches
and nobody on the current team can reliably modify it. The rewrite target is
launch in Q3 next year, with a soft internal rollout to two teams first.

## Who uses FieldOps

The field team's laptops are a mix: the central office issues ThinkPads running
Windows 11 Pro, while the regional offices standardized on Dell XPS machines
running Ubuntu 22.04 LTS. A small team once requested Mac support during the
2023 planning round; that request was declined by IT because the procurement
chain doesn't include Apple and the compliance desk doesn't want to certify a
third platform. Do not plan for Mac as a target.

## Operating environments

The surveyors spend multi-week stretches in remote locations — offshore rigs,
mountain passes, or rural transmission corridors — where the connection is
either satellite (high latency, expensive per-MB) or absent entirely. The
platform must let them work their full day offline, then reconcile when they
are back on a normal connection. Any design that assumes the client stays
online during operation is a non-starter.

## Developer team

The backend team is five engineers. Three of them built the previous Python
services at this company and we have standardized on Python for everything
server-side. The team has asked that we use the newer match-statement and
exception-group features — the current production Python on our build images
is 3.11.6, and we will not downgrade.

## Performance expectations

The field team often captures data at 1-second cadence for field surveys, and
the existing tool sometimes takes three to four seconds to acknowledge a save,
which they hate. For the rebuild, the product manager has written into the
acceptance criteria that p99 server-side latency must come in under 200
milliseconds under realistic load. This is a hard acceptance criterion, not an
aspirational target.

## What we are building

The core of the system is a record-of-work database that lets field staff
record observations, photos, and structured measurements; reconcile offline
capture on reconnect; and submit findings up the chain to the central ops
team. You will be designing the overall server architecture. Related briefs
cover infrastructure and security; read them both before proposing.
