#!/usr/bin/env python3
"""Multi-Agent Ops Coordinator — stale beliefs and the database that caught them.

Black Friday, 15:20 UTC. The on-call engineer asks the coordinator:
  "Is the checkout rollout active? Are customers impacted?"

Five sub-agents monitor five different sources. They disagree. YantrikDB
stores every claim with source attribution, validity windows, and
confidence bands — and surfaces exactly which agents' memory is stale.

A normal agent stack would collapse the disagreement into one averaged
answer. YantrikDB keeps every claim alive long enough to ask: "which
source should dominate right now, and which one is holding yesterday's
truth?"

Requires yantrikdb-server v0.7.2+ and yantrikdb 0.6.1+.

Usage:
  python multi_agent_ops_engine.py <token> [base_url]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

TOKEN = sys.argv[1] if len(sys.argv) > 1 else None
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:7438"
NS = "multi-agent-ops-blackfriday"


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


if not TOKEN:
    die("missing token. usage: python multi_agent_ops_engine.py <token> [base_url]")


def request_json(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        die(f"{method} {path} -> HTTP {exc.code}: {exc.read().decode(errors='replace')}")


def remember(text, *, source, importance=0.8, certainty=0.85):
    r = request_json("POST", "/v1/remember", {
        "text": text, "memory_type": "episodic", "importance": importance,
        "valence": 0.0, "domain": "agent_ops", "source": source,
        "namespace": NS, "certainty": certainty,
    })
    return r.get("rid", "")


def claim(src, rel, dst, *, source, polarity=1, modality="asserted",
          valid_from=None, valid_to=None, confidence="medium"):
    body = {
        "src": src, "rel_type": rel, "dst": dst, "namespace": NS,
        "polarity": polarity, "modality": modality, "extractor": source,
        "confidence_band": confidence,
    }
    if valid_from is not None:
        body["valid_from"] = valid_from
    if valid_to is not None:
        body["valid_to"] = valid_to
    r = request_json("POST", "/v1/claim", body)
    return r.get("claim_id", "")


def t(hour, minute=0):
    """Unix seconds for 2026-11-27 (Black Friday) at HH:MM UTC."""
    return datetime(2026, 11, 27, hour, minute, tzinfo=timezone.utc).timestamp()


def fmt_time(ts):
    if ts is None:
        return "--:--"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M")


def banner(title):
    print("\n" + "=" * 74)
    print(f" {title}")
    print("=" * 74)


def sub(title):
    print(f"\n-- {title} " + "-" * max(0, 70 - len(title)))


# ==================================================================
# Phase 1: Seed each agent's observations
# ==================================================================
#
# The five sub-agents and their stated confidence tiers:
#   agent.deploy    — CI/CD system events        (high, authoritative)
#   agent.telemetry — Prometheus / metrics bus   (high, fresh)
#   agent.support   — customer support inbox     (medium)
#   agent.config    — feature-flag API snapshot  (high BUT STALE at query time)
#   agent.status    — public status page scrape  (medium AND STALE)

def seed_deploy_agent():
    sub("agent.deploy — CI/CD pipeline events")
    remember(
        "[CI/CD] 15:10:00 UTC — checkout-rollout-v8 pipeline succeeded. "
        "Feature flag 'checkout_v8' was toggled ON in production.",
        source="agent.deploy", certainty=0.98, importance=0.95)
    claim("checkout_rollout", "is_active", "true", source="agent.deploy",
          polarity=1, valid_from=t(15, 10), confidence="high")


def seed_telemetry_agent():
    sub("agent.telemetry — Prometheus / metrics bus")
    remember(
        "[METRICS] 15:15:32 UTC — checkout_request_duration_p99 = 4.8s "
        "(baseline 0.6s). 5xx error rate 38% on /api/checkout. Spike began "
        "approximately 3 minutes after the 15:10 deploy.",
        source="agent.telemetry", certainty=0.97, importance=1.0)
    claim("checkout_service", "is_healthy", "true", source="agent.telemetry",
          polarity=-1,  # DEGRADED — telemetry says NOT healthy
          valid_from=t(15, 13), confidence="high")
    claim("checkout_v8", "degraded_at", "15:13", source="agent.telemetry",
          valid_from=t(15, 13), confidence="high")


def seed_support_agent():
    sub("agent.support — customer support inbox")
    remember(
        "[SUPPORT] Three new tickets since 15:12: 'checkout hangs then errors', "
        "'cart items disappear on payment step', 'cannot complete order'. "
        "Routing these to the on-call engineer.",
        source="agent.support", certainty=0.85, importance=0.9)
    claim("customer_impact", "exists", "true", source="agent.support",
          polarity=1, valid_from=t(15, 12), confidence="medium")


def seed_config_agent_stale():
    sub("agent.config — feature-flag API (STALE: snapshot from 14:50)")
    remember(
        "[CONFIG SNAPSHOT @ 14:50] Feature flag 'checkout_v8' = DISABLED. "
        "Rollout plan scheduled for 15:10 window.",
        source="agent.config", certainty=0.95, importance=0.7)
    # This claim was accurate when captured but is now stale.
    claim("checkout_rollout", "is_active", "true", source="agent.config",
          polarity=-1,  # DISABLED per the 14:50 snapshot
          valid_from=t(14, 50), valid_to=t(15, 10),  # validity ENDS at 15:10
          confidence="high")


def seed_status_page_stale():
    sub("agent.status — public status page (STALE: last updated 14:40)")
    remember(
        "[STATUS PAGE @ 14:40] All systems operational. No ongoing incidents.",
        source="agent.status", certainty=0.7, importance=0.5)
    claim("checkout_service", "is_healthy", "true", source="agent.status",
          polarity=1, valid_from=t(14, 40), valid_to=t(15, 13),
          confidence="medium")


# ==================================================================
# Phase 2: The coordinator's query
# ==================================================================

def run_think():
    banner("PHASE 2  COORDINATOR WAKES UP AT 15:20")
    print("  Engineer asks: 'Is the checkout rollout active? Customers impacted?'")
    print("  Running think() to surface agent contradictions...\n")
    r = request_json("POST", "/v1/think", {
        "run_consolidation": False, "run_conflicts": True, "run_patterns": False,
    })
    print(f"    conflicts_found: {r.get('conflicts_found', 0)}")
    print(f"    duration_ms:     {r.get('duration_ms', 0):.1f}")


def detect_polarity_contradictions():
    banner("PHASE 3  POLARITY CONTRADICTIONS — agents disagree")
    print("  Walking the claims ledger for each key entity...\n")

    entities = ["checkout_rollout", "checkout_service", "customer_impact"]
    contradictions = []

    for entity in entities:
        r = request_json("GET", f"/v1/claims?entity={entity}&namespace={NS}")
        claims = r.get("claims", [])
        groups = {}
        for c in claims:
            key = (c["src"], c["rel_type"], c["dst"])
            groups.setdefault(key, []).append(c)

        for (src, rel, dst), group in groups.items():
            pos = [c for c in group if c["polarity"] == 1]
            neg = [c for c in group if c["polarity"] == -1]
            if pos and neg:
                for p in pos:
                    for n in neg:
                        contradictions.append({
                            "subject": src, "relation": rel, "object": dst,
                            "positive_source": p["extractor"],
                            "positive_from": p.get("valid_from"),
                            "positive_to": p.get("valid_to"),
                            "positive_conf": p["confidence_band"],
                            "negative_source": n["extractor"],
                            "negative_from": n.get("valid_from"),
                            "negative_to": n.get("valid_to"),
                            "negative_conf": n["confidence_band"],
                        })

    if not contradictions:
        print("  (no contradictions)")
        return []

    for i, c in enumerate(contradictions, 1):
        pf, pt_ = fmt_time(c["positive_from"]), fmt_time(c["positive_to"])
        nf, nt = fmt_time(c["negative_from"]), fmt_time(c["negative_to"])
        print(f"  [{i}] POLARITY_CONTRADICTION")
        print(f"      {c['subject']} --{c['relation']}--> {c['object']}")
        print(f"      ({c['positive_source']:18}) CLAIMS YES  "
              f"[{pf} - {pt_}]  conf={c['positive_conf']}")
        print(f"      ({c['negative_source']:18}) CLAIMS NO   "
              f"[{nf} - {nt}]  conf={c['negative_conf']}")
        print()
    return contradictions


# ==================================================================
# Phase 4: THE TEMPORAL QUERY — the killer feature
# ==================================================================

def temporal_query(at_time):
    """What did the agent fleet believe at a specific point in time?"""
    banner(f"PHASE 4  TEMPORAL QUERY — what did we believe at {fmt_time(at_time)}?")
    print()

    entities = ["checkout_rollout", "checkout_service", "customer_impact"]
    for entity in entities:
        r = request_json("GET", f"/v1/claims?entity={entity}&namespace={NS}")
        claims = r.get("claims", [])
        # Filter to claims valid at this instant
        active_then = []
        for c in claims:
            vf = c.get("valid_from")
            vt = c.get("valid_to")
            if vf is None:
                continue
            if vf <= at_time and (vt is None or at_time < vt):
                active_then.append(c)

        if not active_then:
            continue
        print(f"  {entity}")
        for c in active_then:
            pol = "YES" if c["polarity"] == 1 else "NO "
            vt_str = fmt_time(c.get("valid_to")) if c.get("valid_to") else "now"
            print(f"    [{c['extractor']:18}] {pol}  "
                  f"({fmt_time(c['valid_from'])}–{vt_str})  "
                  f"conf={c['confidence_band']}")
        print()


# ==================================================================
# Phase 5: Evidence chain via recall
# ==================================================================

def evidence_chain():
    banner("PHASE 5  WHAT EACH AGENT REPORTED (via recall)")
    r = request_json("POST", "/v1/recall", {
        "query": "checkout rollout status and customer impact right now",
        "top_k": 8, "namespace": NS,
    })
    print(f"  Query: 'checkout rollout status and customer impact right now'\n")
    for i, m in enumerate(r.get("results", []), 1):
        src = m.get("source", "?")
        text = m.get("text", "")
        if len(text) > 140:
            text = text[:137] + "..."
        print(f"  [{i}] {src:18} score={m.get('score',0):.2f}")
        print(f"      {text}")
        print()


# ==================================================================
# Phase 6: The reconciled answer
# ==================================================================

def reconcile(contradictions, at_time):
    banner("PHASE 6  RECONCILED ANSWER (synthesized from the claims ledger)")
    print()

    # For each contradiction, pick the winner: whichever claim is
    # (a) still valid at the query time and (b) highest confidence.
    stale_agents = set()
    winners = []

    for c in contradictions:
        pos_valid = (
            c.get("positive_to") is None or at_time < c["positive_to"]
        ) and c.get("positive_from") and c["positive_from"] <= at_time
        neg_valid = (
            c.get("negative_to") is None or at_time < c["negative_to"]
        ) and c.get("negative_from") and c["negative_from"] <= at_time

        if pos_valid and not neg_valid:
            winners.append(("YES", c["positive_source"], c["subject"], c["relation"], c["object"]))
            stale_agents.add(c["negative_source"])
        elif neg_valid and not pos_valid:
            winners.append(("NO", c["negative_source"], c["subject"], c["relation"], c["object"]))
            stale_agents.add(c["positive_source"])
        elif pos_valid and neg_valid:
            # Both valid — higher confidence wins
            conf_rank = {"high": 3, "medium": 2, "low": 1}
            if conf_rank[c["positive_conf"]] >= conf_rank[c["negative_conf"]]:
                winners.append(("YES", c["positive_source"], c["subject"], c["relation"], c["object"]))
                stale_agents.add(c["negative_source"])
            else:
                winners.append(("NO", c["negative_source"], c["subject"], c["relation"], c["object"]))
                stale_agents.add(c["positive_source"])

    print("  Current facts (chosen by source-freshness + confidence):")
    for verdict, src, subj, rel, obj in winners:
        print(f"    {subj} --{rel}--> {obj}  = {verdict}   [authority: {src}]")
    print()

    if stale_agents:
        print("  Agents with STALE or superseded beliefs (excluded from verdict):")
        for a in sorted(stale_agents):
            print(f"    - {a}")
        print()

    print("  Recommended action for the on-call engineer:")
    print("    * checkout_v8 rollout IS live (deploy completed at 15:10)")
    print("    * checkout service IS degraded (telemetry confirms error spike)")
    print("    * customer impact IS real (support tickets arriving)")
    print("    * ROLL BACK checkout_v8 via feature flag")
    print()
    print("  Stale reports from agent.config and agent.status MUST NOT be")
    print("  used to decide rollback. A vector-memory agent would blend them.")
    print("  YantrikDB's validity windows make the staleness explicit.")


# ==================================================================
# Main
# ==================================================================

def main():
    banner(f"MULTI-AGENT OPS — which agent memory is stale?")
    print(f"  Cluster:    {BASE}")
    print(f"  Namespace:  {NS}")
    print(f"  Scenario:   Black Friday 2026, 15:20 UTC, on-call query")

    banner("PHASE 1  EACH SUB-AGENT REPORTS WHAT IT SEES")
    seed_deploy_agent()
    seed_telemetry_agent()
    seed_support_agent()
    seed_config_agent_stale()
    seed_status_page_stale()
    time.sleep(2)

    run_think()
    contradictions = detect_polarity_contradictions()

    # The killer feature: a temporal query showing belief at different times.
    query_time = t(15, 20)  # current coordinator query
    temporal_query(query_time)
    temporal_query(t(14, 55))  # 25 minutes earlier — belief was different

    evidence_chain()
    reconcile(contradictions, query_time)

    banner("DONE — agents disagreed. The database knew which memory was stale.")


if __name__ == "__main__":
    main()
