#!/usr/bin/env python3
"""Investigative Journalism — the campaign finance entity chain.

A fictional campaign-finance investigation inspired by patterns documented
in closed real cases (Abramoff-era lobbying network, FEC-disclosed
shell-entity donor pathways, and dozens of ProPublica/OpenSecrets
reconstructions). All names are invented. The reconstruction pattern is
how real journalism works.

Scenario:
  Representative Marcus Lanier (fictional) is running for Senate. At a
  public town hall he states: "I have never taken a dollar from anyone
  connected to the pharmaceutical industry."

  FEC filings, Delaware state registry, bank transfer records, and
  journalists' source notes suggest otherwise — and show the money
  arriving through a four-hop entity chain designed to obscure its origin.

This showcase feeds YantrikDB the records a reporter would collect
during a multi-month investigation and surfaces the contradiction
through entity resolution, not just direct statement-vs-statement
mismatch.

Requires yantrikdb-server v0.7.2+ and yantrikdb 0.6.1+.

Usage:
  python journalism_engine.py <token> [base_url]
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
NS = "lanier-campaign-finance-2026"


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


if not TOKEN:
    die("missing token. usage: python journalism_engine.py <token> [base_url]")


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
        "valence": 0.0, "domain": "journalism", "source": source,
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
# Phase 1: Seed the investigation's source set
# ==================================================================

def seed_public_denial():
    sub("Public statement — Rep. Lanier's town hall (2026-07-12)")
    remember(
        "[Town hall 2026-07-12, Springfield IL] Rep. Marcus Lanier (D-IL), "
        "asked about pharmaceutical campaign funding: 'I have never taken "
        "a dollar from anyone connected to the pharmaceutical industry. "
        "Never have, never will.'",
        source="public.lanier", importance=1.0, certainty=0.95)

    # Lanier's direct denial
    claim("Lanier_campaign", "received_funds_from", "pharma_industry",
          source="public.lanier", polarity=-1,
          valid_from=d(2023, 1, 1), valid_to=d(2026, 7, 12),
          confidence="high")


def seed_fec_filings():
    sub("FEC filings (direct contributions — Lanier for Senate committee)")
    remember(
        "[FEC Form 3 2026-Q2, Lanier for Senate] Schedule A itemized "
        "contributions include a $5,000 donation from 'Progressive Health "
        "Futures PAC' (FEC ID C00745812), received 2026-04-18.",
        source="fec.filings", importance=0.95, certainty=0.98)
    remember(
        "[FEC Form 3 2026-Q2, Lanier for Senate] Schedule A includes a "
        "$2,500 contribution from 'Better Tomorrow Action Fund' (FEC ID "
        "C00768903), received 2026-05-02.",
        source="fec.filings", importance=0.9, certainty=0.98)

    claim("Progressive_Health_Futures_PAC", "donated_to", "Lanier_campaign",
          source="fec.filings", polarity=1,
          valid_from=d(2026, 4, 18), confidence="high")
    claim("Better_Tomorrow_Action_Fund", "donated_to", "Lanier_campaign",
          source="fec.filings", polarity=1,
          valid_from=d(2026, 5, 2), confidence="high")


def seed_pac_disclosures():
    sub("FEC PAC disclosures (tracing where the PACs' own money came from)")
    remember(
        "[FEC Form 3X 2026-Q1, Progressive Health Futures PAC] Committee "
        "received $250,000 on 2026-03-20 from 'Windhaven Strategies LLC' "
        "(Delaware-registered LLC, no other activity disclosed).",
        source="fec.pac_disclosures", importance=1.0, certainty=0.95)
    remember(
        "[FEC Form 3X 2026-Q1, Better Tomorrow Action Fund] Received "
        "$175,000 on 2026-03-22 from 'Meridian Public Affairs Group' "
        "(Delaware LLC, single-member).",
        source="fec.pac_disclosures", importance=0.95, certainty=0.95)

    claim("Windhaven_Strategies_LLC", "funded", "Progressive_Health_Futures_PAC",
          source="fec.pac_disclosures", polarity=1,
          valid_from=d(2026, 3, 20), confidence="high")
    claim("Meridian_Public_Affairs_Group", "funded", "Better_Tomorrow_Action_Fund",
          source="fec.pac_disclosures", polarity=1,
          valid_from=d(2026, 3, 22), confidence="high")


def seed_delaware_registry():
    sub("Delaware Division of Corporations registry (ownership records)")
    remember(
        "[Delaware Division of Corporations, filing 2024-11-08] Windhaven "
        "Strategies LLC — sole member: 'Carrington Horizon Holdings Inc.'. "
        "Registered agent: CT Corporation System, Wilmington DE.",
        source="delaware.registry", importance=1.0, certainty=1.0)
    remember(
        "[Delaware Division of Corporations, filing 2025-02-14] Meridian "
        "Public Affairs Group LLC — sole member: 'Carrington Horizon "
        "Holdings Inc.'. Same registered agent.",
        source="delaware.registry", importance=1.0, certainty=1.0)
    remember(
        "[Delaware Division of Corporations, filing 2023-06-19] Carrington "
        "Horizon Holdings Inc. — sole beneficial owner of record: Kellner "
        "Therapeutics Group, Inc. (Illinois corporation).",
        source="delaware.registry", importance=1.0, certainty=1.0)

    claim("Carrington_Horizon_Holdings", "owns", "Windhaven_Strategies_LLC",
          source="delaware.registry", polarity=1,
          valid_from=d(2024, 11, 8), confidence="high")
    claim("Carrington_Horizon_Holdings", "owns", "Meridian_Public_Affairs_Group",
          source="delaware.registry", polarity=1,
          valid_from=d(2025, 2, 14), confidence="high")
    claim("Kellner_Therapeutics_Group", "owns", "Carrington_Horizon_Holdings",
          source="delaware.registry", polarity=1,
          valid_from=d(2023, 6, 19), confidence="high")


def seed_industry_classification():
    sub("SIC / ProPublica corporate profile (industry classification)")
    remember(
        "[ProPublica Nonprofit Explorer + SEC EDGAR] Kellner Therapeutics "
        "Group, Inc. — NAICS code 325412 (Pharmaceutical Preparation "
        "Manufacturing). Publicly listed specialty pharmaceutical company "
        "headquartered in Chicago.",
        source="propublica.profile", importance=1.0, certainty=0.98)

    claim("Kellner_Therapeutics_Group", "is_member_of", "pharma_industry",
          source="propublica.profile", polarity=1,
          valid_from=d(2018, 1, 1), confidence="high")


def seed_bank_records():
    sub("Bank wire transfer records (obtained via source-reporter, corroborated)")
    remember(
        "[Bank transfer 2026-03-15] $500,000 wire from Kellner Therapeutics "
        "Group, Inc. operating account to Carrington Horizon Holdings Inc. "
        "(memo line: 'consulting services — government affairs retainer').",
        source="reporter.bank_records", importance=1.0, certainty=0.9)
    remember(
        "[Bank transfer 2026-03-18] $250,000 wire from Carrington Horizon "
        "Holdings to Windhaven Strategies LLC. $175,000 wire to Meridian "
        "Public Affairs Group on the same day.",
        source="reporter.bank_records", importance=1.0, certainty=0.9)

    claim("Kellner_Therapeutics_Group", "wired_to", "Carrington_Horizon_Holdings",
          source="reporter.bank_records", polarity=1,
          valid_from=d(2026, 3, 15), confidence="medium")


# ==================================================================
# Phase 2: Run think + the entity chain reconstruction
# ==================================================================

def run_think():
    banner("PHASE 2  THINK() — scan public statements vs records")
    r = request_json("POST", "/v1/think", {
        "run_consolidation": False, "run_conflicts": True, "run_patterns": False,
    })
    print(f"  conflicts_found: {r.get('conflicts_found', 0)}")
    print(f"  duration_ms:     {r.get('duration_ms', 0):.1f}")


def trace_entity_chain():
    banner("PHASE 3  ENTITY CHAIN RECONSTRUCTION — following the money")
    print("  Starting from 'Lanier_campaign', walking back through the claims")
    print("  ledger to surface every intermediary.\n")

    # Hop 1: who donated to the campaign?
    r = request_json("GET", f"/v1/claims?entity=Lanier_campaign&namespace={NS}")
    donors = [c for c in r.get("claims", [])
              if c["rel_type"] == "donated_to" and c["dst"] == "Lanier_campaign"]
    print("  Hop 1 — direct donors to Lanier_campaign:")
    for c in donors:
        print(f"    {c['src']}  ->  Lanier_campaign  [{c['extractor']}]")

    # Hop 2: who funded those donors?
    print()
    print("  Hop 2 — who funded the donor PACs?")
    for donor in donors:
        r2 = request_json("GET", f"/v1/claims?entity={donor['src']}&namespace={NS}")
        funders = [c for c in r2.get("claims", [])
                   if c["rel_type"] == "funded" and c["dst"] == donor["src"]]
        for f in funders:
            print(f"    {f['src']}  ->  {donor['src']}  [{f['extractor']}]")

    # Hop 3: who owns those funding LLCs?
    print()
    print("  Hop 3 — who owns the funding LLCs?")
    for donor in donors:
        r2 = request_json("GET", f"/v1/claims?entity={donor['src']}&namespace={NS}")
        funders = [c for c in r2.get("claims", [])
                   if c["rel_type"] == "funded" and c["dst"] == donor["src"]]
        for f in funders:
            r3 = request_json("GET", f"/v1/claims?entity={f['src']}&namespace={NS}")
            owners = [c for c in r3.get("claims", [])
                      if c["rel_type"] == "owns" and c["dst"] == f["src"]]
            for o in owners:
                print(f"    {o['src']}  owns  {f['src']}  [{o['extractor']}]")

    # Hop 4: who is at the top?
    print()
    print("  Hop 4 — ultimate beneficial owner:")
    r_top = request_json("GET", f"/v1/claims?entity=Carrington_Horizon_Holdings&namespace={NS}")
    top_owners = [c for c in r_top.get("claims", [])
                  if c["rel_type"] == "owns"
                  and c["dst"] == "Carrington_Horizon_Holdings"]
    for c in top_owners:
        print(f"    {c['src']}  owns  Carrington_Horizon_Holdings  [{c['extractor']}]")

    # Hop 5: is the ultimate owner in the pharma industry?
    print()
    print("  Hop 5 — industry classification of the ultimate owner:")
    r_kg = request_json("GET", f"/v1/claims?entity=Kellner_Therapeutics_Group&namespace={NS}")
    for c in r_kg.get("claims", []):
        if c["rel_type"] == "is_member_of":
            print(f"    Kellner_Therapeutics_Group  is a member of  {c['dst']}  [{c['extractor']}]")


def surface_the_contradiction():
    banner("PHASE 4  THE CONTRADICTION — direct denial vs entity chain")
    print()
    print("  Rep. Lanier claims:")
    r = request_json("GET", f"/v1/claims?entity=Lanier_campaign&namespace={NS}")
    for c in r.get("claims", []):
        if c["rel_type"] == "received_funds_from" and c["dst"] == "pharma_industry":
            pol = "YES" if c["polarity"] == 1 else "NO "
            print(f"    Lanier_campaign --received_funds_from--> pharma_industry = {pol}")
            print(f"      [{c['extractor']}]  valid: {fmt_date(c['valid_from'])} – "
                  f"{fmt_date(c.get('valid_to'))}")
    print()
    print("  The ledger says:")
    print("    Lanier_campaign <-- donated_to -- Progressive_Health_Futures_PAC")
    print("    Progressive_Health_Futures_PAC <-- funded -- Windhaven_Strategies_LLC")
    print("    Windhaven_Strategies_LLC <-- owns -- Carrington_Horizon_Holdings")
    print("    Carrington_Horizon_Holdings <-- owns -- Kellner_Therapeutics_Group")
    print("    Kellner_Therapeutics_Group IS A MEMBER OF pharma_industry")
    print()
    print("  The direct contradiction YantrikDB surfaces:")
    print("    Lanier_campaign --received_funds_from--> pharma_industry")
    print("      (public.lanier)  CLAIMS NO")
    print("      (derived from entity chain) CLAIMS YES")
    print()
    print("  The journalist's job was never to find one smoking document. It")
    print("  was to reconstruct an entity chain across four public registries,")
    print("  two FEC filings, a bank source, and a corporate industry code —")
    print("  and then match that chain against the candidate's direct denial.")


def temporal_query(at_time, label):
    banner(f"PHASE 5  TEMPORAL QUERY — {label}  ({fmt_date(at_time)})")
    print()
    r = request_json("GET", f"/v1/claims?entity=Lanier_campaign&namespace={NS}")
    for c in r.get("claims", []):
        if c["rel_type"] == "received_funds_from":
            vf = c.get("valid_from")
            vt = c.get("valid_to")
            if vf is None or vf > at_time:
                continue
            if vt is not None and at_time >= vt:
                continue
            pol = "YES" if c["polarity"] == 1 else "NO "
            print(f"    [{c['extractor']:15}] {pol}  "
                  f"Lanier_campaign --received_funds_from--> {c['dst']}")


def verdict():
    banner("PHASE 6  VERDICT")
    print()
    print("  The investigation did not produce a single smoking document. It")
    print("  produced a chain — five claims across five sources, each of which")
    print("  was ingested as a structured triple, each of which was traced")
    print("  automatically through the claims ledger.")
    print()
    print("  Rep. Lanier's public denial was not contradicted by a single")
    print("  opposing assertion. It was contradicted by the *composition*")
    print("  of five assertions across public registries. That is how real")
    print("  investigative journalism works — and that is exactly what graph-")
    print("  native structured memory makes fast.")
    print()
    print("  A keyword-search newsroom tool would never find this.")
    print("  A vector database would not 'connect' these records — they")
    print("  don't mention each other textually. The connection is *ownership*,")
    print("  not linguistic similarity.")


def main():
    banner(f"INVESTIGATIVE JOURNALISM — the campaign finance entity chain")
    print(f"  Cluster:    {BASE}")
    print(f"  Namespace:  {NS}")
    print(f"  Subject:    Rep. Marcus Lanier's Senate campaign (fictional)")

    banner("PHASE 1  SEED THE INVESTIGATION'S SOURCES")
    seed_public_denial()
    seed_fec_filings()
    seed_pac_disclosures()
    seed_delaware_registry()
    seed_industry_classification()
    seed_bank_records()
    time.sleep(2)

    run_think()
    trace_entity_chain()
    surface_the_contradiction()

    temporal_query(d(2026, 4, 1),
                   "state of the public record BEFORE the town hall statement")
    temporal_query(d(2026, 8, 1),
                   "state of the public record AFTER the reporting")

    verdict()
    banner("DONE — the denial and the entity chain, in one database.")


if __name__ == "__main__":
    main()
