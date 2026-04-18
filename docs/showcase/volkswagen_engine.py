#!/usr/bin/env python3
"""Volkswagen emissions — the public-claims-vs-records showcase.

Between 2009 and 2015, Volkswagen sold ~11 million diesel vehicles certified
as meeting strict emissions standards. Internally, engineers had designed
defeat-device software that detected emissions test conditions and switched
to a compliant mode. On the road, the same vehicles emitted up to 40x the
legal NOx limit.

Every public claim (certifications, press releases, sustainability reports)
said the cars complied. Every internal engineering document, ICCT field
test result, EPA Notice of Violation, and subsequent court filing said
otherwise.

This showcase feeds YantrikDB a slice of the public record across all
those sources and surfaces the polarity contradictions that unraveled
Dieselgate.

All sources are public record: EPA NOV (2015-09-18), DOJ court filings,
ICCT West Virginia Univ. field-test report (2014), Volkswagen public
sustainability reports (2010–2015).

Requires yantrikdb-server v0.7.2+ and yantrikdb 0.6.1+.

Usage:
  python volkswagen_engine.py <token> [base_url]
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
NS = "volkswagen-dieselgate"


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


if not TOKEN:
    die("missing token. usage: python volkswagen_engine.py <token> [base_url]")


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


def remember(text, *, source, importance=0.8, certainty=0.9):
    r = request_json("POST", "/v1/remember", {
        "text": text, "memory_type": "episodic", "importance": importance,
        "valence": 0.0, "domain": "compliance_research", "source": source,
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


def d(year, month, day):
    return datetime(year, month, day, tzinfo=timezone.utc).timestamp()


def fmt_date(ts):
    if ts is None:
        return "----"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def banner(title):
    print("\n" + "=" * 74)
    print(f" {title}")
    print("=" * 74)


def sub(title):
    print(f"\n-- {title} " + "-" * max(0, 70 - len(title)))


# ==================================================================
# Phase 1: Seed sources from the public record
# ==================================================================

def seed_vw_public():
    sub("VW public claims (press releases, sustainability reports, certification)")
    remember(
        "[VW Sustainability Report 2010] 'Our modern TDI Clean Diesel engines "
        "comply with the most stringent U.S. emissions regulations — the "
        "Environmental Protection Agency Tier 2 Bin 5 standard.'",
        source="vw.public", importance=0.95)
    remember(
        "[VW press release 2012-05] 'Volkswagen's BlueTDI technology has "
        "redefined diesel: clean, efficient, and fully compliant with "
        "U.S. and European emissions limits.'",
        source="vw.public", importance=0.9)
    remember(
        "[VW public statement 2015-05, in response to early ICCT findings] "
        "'There is no software that could produce results inconsistent with "
        "our certification.'",
        source="vw.public", importance=0.95)

    # Structured claims from VW's public position
    claim("VW_TDI_diesels_2009_2015", "complies_with",
          "US_EPA_Tier2_Bin5_standard", source="vw.public",
          polarity=1, valid_from=d(2009, 1, 1), valid_to=d(2015, 9, 18),
          confidence="high")
    claim("VW_TDI_diesels_2009_2015", "contains", "defeat_device_software",
          source="vw.public", polarity=-1,  # VW publicly denies
          valid_from=d(2009, 1, 1), valid_to=d(2015, 9, 18), confidence="high")


def seed_internal_engineering():
    sub("VW internal engineering documents (disclosed via court filings)")
    remember(
        "[Internal VW technical documentation, dated 2006-2007, produced at trial] "
        "Engineering teams document a 'dual-mode' engine calibration: a "
        "low-emissions mode triggered when the vehicle detects steering "
        "angle, barometric pressure, and wheel-speed patterns consistent "
        "with a dynamometer test.",
        source="vw.internal", importance=1.0, certainty=0.95)
    remember(
        "[Internal VW email, 2014, cited in DOJ complaint] Senior engineer "
        "writes to a colleague noting that 'the software cannot pass US real-"
        "world testing without the acoustic condition' — a reference to the "
        "test-detection logic.",
        source="vw.internal", importance=0.95, certainty=0.9)

    # Internal engineering acknowledges defeat device
    claim("VW_TDI_diesels_2009_2015", "contains", "defeat_device_software",
          source="vw.internal", polarity=1,
          valid_from=d(2006, 1, 1), confidence="high")
    claim("VW_TDI_diesels_2009_2015", "complies_with",
          "US_EPA_Tier2_Bin5_standard", source="vw.internal",
          polarity=-1,  # internally known to not comply under normal driving
          valid_from=d(2006, 1, 1), confidence="high")


def seed_iccT_findings():
    sub("ICCT + West Virginia Univ. on-road field test (May 2014)")
    remember(
        "[ICCT / WVU CAFEE report, May 2014] Portable Emissions Measurement "
        "on VW TDI vehicles under real-world driving found NOx emissions "
        "5–35x the Tier 2 Bin 5 standard. The same vehicles passed the "
        "lab dynamometer test.",
        source="iccT.report", importance=1.0, certainty=0.98)
    claim("VW_TDI_diesels_2009_2015", "complies_with",
          "US_EPA_Tier2_Bin5_standard", source="iccT.report",
          polarity=-1,  # field test says NO
          valid_from=d(2014, 5, 15), confidence="high")


def seed_epa_nov():
    sub("EPA Notice of Violation (2015-09-18)")
    remember(
        "[EPA Notice of Violation, 2015-09-18] EPA formally charges that "
        "certain VW vehicles from model years 2009–2015 contain 'defeat "
        "devices' and violate the Clean Air Act. Approximately 482,000 "
        "diesel vehicles in the U.S. affected. EPA recall order issued.",
        source="epa.nov", importance=1.0, certainty=1.0)
    claim("VW_TDI_diesels_2009_2015", "contains", "defeat_device_software",
          source="epa.nov", polarity=1,
          valid_from=d(2015, 9, 18), confidence="high")
    claim("VW_TDI_diesels_2009_2015", "complies_with",
          "US_EPA_Tier2_Bin5_standard", source="epa.nov",
          polarity=-1,
          valid_from=d(2015, 9, 18), confidence="high")


def seed_doj_settlement():
    sub("DOJ consent decree + criminal pleas (2017)")
    remember(
        "[DOJ consent decree 2017-01-11] Volkswagen AG pleads guilty to "
        "conspiracy to defraud the United States, wire fraud, and violating "
        "the Clean Air Act. Company agrees to pay $4.3 billion in criminal "
        "and civil penalties. Total Dieselgate cost to VW exceeds $33 billion.",
        source="doj.decree", importance=1.0, certainty=1.0)
    claim("Volkswagen_AG", "admits_to", "defeat_device_fraud",
          source="doj.decree", polarity=1,
          valid_from=d(2017, 1, 11), confidence="high")


# ==================================================================
# Phase 2: Analysis
# ==================================================================

def run_think():
    banner("PHASE 2  THINK() — SCAN THE COMPLIANCE RECORD")
    r = request_json("POST", "/v1/think", {
        "run_consolidation": False, "run_conflicts": True, "run_patterns": False,
    })
    print(f"  conflicts_found: {r.get('conflicts_found', 0)}")
    print(f"  duration_ms:     {r.get('duration_ms', 0):.1f}")


def detect_polarity_contradictions():
    banner("PHASE 3  POLARITY CONTRADICTIONS (public vs private vs regulator)")
    entities = ["VW_TDI_diesels_2009_2015", "Volkswagen_AG"]
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
                            "pos_source": p["extractor"],
                            "pos_from": p.get("valid_from"),
                            "neg_source": n["extractor"],
                            "neg_from": n.get("valid_from"),
                        })
    if not contradictions:
        print("  (no polarity contradictions)")
        return []
    for i, c in enumerate(contradictions, 1):
        print(f"\n  [{i}] POLARITY_CONTRADICTION")
        print(f"      {c['subject']}")
        print(f"        --{c['relation']}-->  {c['object']}")
        print(f"      ({c['neg_source']:15}) CLAIMS NO  from {fmt_date(c['neg_from'])}")
        print(f"      ({c['pos_source']:15}) CLAIMS YES from {fmt_date(c['pos_from'])}")
    return contradictions


def temporal_query(at_time):
    banner(f"PHASE 4  TEMPORAL QUERY — what was the public state of belief on {fmt_date(at_time)}?")
    print()
    r = request_json("GET", f"/v1/claims?entity=VW_TDI_diesels_2009_2015&namespace={NS}")
    claims = r.get("claims", [])
    # Filter to claims valid at this date AND from public-facing sources
    public_like = {"vw.public", "iccT.report", "epa.nov", "doj.decree"}
    for c in claims:
        vf = c.get("valid_from")
        vt = c.get("valid_to")
        if c["extractor"] not in public_like:
            continue
        if vf is None or vf > at_time:
            continue
        if vt is not None and at_time >= vt:
            continue
        pol = "YES" if c["polarity"] == 1 else "NO "
        vt_str = fmt_date(vt) if vt else "now"
        print(f"    [{c['extractor']:15}] {pol}  "
              f"{c['src']} --{c['rel_type']}--> {c['dst']}")
        print(f"    {'':17} ({fmt_date(c['valid_from'])} – {vt_str}, conf={c['confidence_band']})")


def evidence_chain():
    banner("PHASE 5  EVIDENCE CHAIN — the defeat device itself")
    r = request_json("POST", "/v1/recall", {
        "query": "defeat device software test detection emissions",
        "top_k": 6, "namespace": NS,
    })
    print("  Query: 'defeat device software test detection emissions'\n")
    for i, m in enumerate(r.get("results", []), 1):
        src = m.get("source", "?")
        text = m.get("text", "")
        if len(text) > 150:
            text = text[:147] + "..."
        print(f"  [{i}] {src:15} score={m.get('score',0):.2f}")
        print(f"      {text}\n")


def verdict(contradictions):
    banner("PHASE 6  VERDICT")
    print()
    print(f"  {len(contradictions)} polarity contradiction(s) between public claims")
    print(f"  and authoritative records.")
    print()
    print("  The pattern: from 2009 to 2015, VW's public position was that the")
    print("  TDI diesels complied with US Tier 2 Bin 5 and contained no defeat")
    print("  device. Internal engineering documents showed the opposite as early")
    print("  as 2006. The ICCT field test revealed the gap in 2014. The EPA")
    print("  Notice of Violation made the contradiction official in 2015. The")
    print("  DOJ consent decree confirmed it in 2017.")
    print()
    print("  YantrikDB stored every claim with polarity, validity, and source.")
    print("  The pre-2014 belief state (public = compliant, internal = not)")
    print("  is preserved. The post-2015 reality (public admits, regulator")
    print("  confirms) is preserved. Both are queryable. That's what makes this")
    print("  a cognitive memory database, not a scandal log.")


def main():
    banner(f"VOLKSWAGEN DIESELGATE — public claims vs records")
    print(f"  Cluster:    {BASE}")
    print(f"  Namespace:  {NS}")
    print(f"  Period:     2006 (internal defeat-device design) → 2017 (DOJ plea)")

    banner("PHASE 1  SEED PUBLIC RECORD")
    seed_vw_public()
    seed_internal_engineering()
    seed_iccT_findings()
    seed_epa_nov()
    seed_doj_settlement()
    time.sleep(2)

    run_think()
    contradictions = detect_polarity_contradictions()

    # Temporal queries — belief state at two different points
    temporal_query(d(2013, 6, 1))   # mid-scandal, before ICCT report
    temporal_query(d(2016, 1, 1))   # post-NOV, pre-DOJ

    evidence_chain()
    verdict(contradictions)

    banner("DONE — the car knew when it was being tested. The database knew too.")


if __name__ == "__main__":
    main()
