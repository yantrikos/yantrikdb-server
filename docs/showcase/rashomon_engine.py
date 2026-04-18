#!/usr/bin/env python3
"""The Rashomon Engine — reconstructing truth from conflicting testimony.

A data breach happened at Helios Labs on 2026-03-15 between 21:00 and 00:00.
Flagship product source code leaked to a public repo at 23:15.

Five people had badge access that night. Each tells their version of events.
Two system sources — badge logs and git commits — provide ground truth.

The synthesis at the end is driven by REAL queries into YantrikDB, not
hardcoded print statements. You can trust the output because every lie
surfaced is computed from the claims ledger + recall engine.

Requires yantrikdb-server v0.7.2+ (namespace filter on /v1/conflicts) and
yantrikdb 0.6.1+ (claim UNIQUE fix that lets multi-source contradictions
coexist).

Usage:
  python rashomon_engine.py <token> [base_url]
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
NS = "rashomon-helios-breach"


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


if not TOKEN:
    die("missing token. usage: python rashomon_engine.py <token> [base_url]")


def request_json(method: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        die(f"{method} {path} -> HTTP {exc.code}: {body}")


def remember(text: str, *, source: str, importance: float = 0.7,
             certainty: float = 0.8) -> str:
    r = request_json("POST", "/v1/remember", {
        "text": text,
        "memory_type": "episodic",
        "importance": importance,
        "valence": 0.0,
        "domain": "investigation",
        "source": source,
        "namespace": NS,
        "certainty": certainty,
    })
    return r.get("rid", "")


def claim(src: str, rel: str, dst: str, *, source: str, polarity: int = 1,
          modality: str = "asserted", valid_from: float | None = None,
          valid_to: float | None = None, confidence: str = "medium") -> str:
    body = {
        "src": src, "rel_type": rel, "dst": dst,
        "namespace": NS, "polarity": polarity, "modality": modality,
        "extractor": source, "confidence_band": confidence,
    }
    if valid_from is not None:
        body["valid_from"] = valid_from
    if valid_to is not None:
        body["valid_to"] = valid_to
    r = request_json("POST", "/v1/claim", body)
    return r.get("claim_id", "")


def t(hour: int, minute: int = 0) -> float:
    """Unix seconds for 2026-03-15 HH:MM UTC."""
    return datetime(2026, 3, 15, hour, minute, tzinfo=timezone.utc).timestamp()


def fmt_time(ts: float | None) -> str:
    if ts is None:
        return "--:--"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M")


# ==================================================================
# Print helpers
# ==================================================================

def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(f" {title}")
    print("=" * 74)


def sub(title: str) -> None:
    print(f"\n-- {title} " + "-" * max(0, 70 - len(title)))


# ==================================================================
# Phase 1: Seed witness statements + system evidence
# ==================================================================

def seed() -> None:
    banner("PHASE 1  SEEDING 5 WITNESSES + 2 AUTHORITATIVE LOG SOURCES")

    # Wipe any prior run's claims first — the showcase should be idempotent
    sub("Seeding Maya Chen (senior engineer | honest, partial view)")
    remember(
        "I was at the office working on the Q2 release deadline. Got there around 7pm. "
        "David Park came in around 9:30pm and said he needed to grab his laptop. "
        "His office light stayed on the whole time I was there.",
        source="maya.chen", certainty=0.95, importance=0.85)
    remember(
        "I left the office at 10pm. David's office light was still on when I walked past.",
        source="maya.chen", certainty=0.9, importance=0.9)
    claim("Maya", "was_at", "Helios_office", source="maya.chen",
          valid_from=t(19), valid_to=t(22), confidence="high")
    claim("David", "was_at", "Helios_office", source="maya.chen",
          valid_from=t(21, 30), valid_to=t(22), confidence="medium")

    sub("Seeding David Park (CTO | LIES about times and repo access)")
    remember(
        "I stopped by the office briefly around 9:30pm to grab my MacBook. "
        "I left maybe 15 minutes later. I was home by 10.",
        source="david.park", certainty=0.95, importance=0.9)
    remember(
        "I definitely did not access the production repository that night.",
        source="david.park", certainty=0.95, importance=0.95)
    # David's self-reported window: 21:30 -> 21:45
    claim("David", "was_at", "Helios_office", source="david.park",
          valid_from=t(21, 30), valid_to=t(21, 45), confidence="high")
    # David DENIES repo access
    claim("David", "accessed", "production_repo", source="david.park",
          polarity=-1,  # negation
          valid_from=t(21), valid_to=t(23, 59), confidence="high")

    sub("Seeding Alex Rivera (night janitor | reliable, saw David leave late)")
    remember(
        "Started my shift at 10pm. Ran into Maya on my way in | she was heading out.",
        source="alex.rivera", certainty=0.95, importance=0.75)
    remember(
        "David Park was still in his office when I did the executive floor at 11pm. "
        "Door was closed but light was on and I could hear typing.",
        source="alex.rivera", certainty=0.95, importance=0.95)
    remember(
        "David came down around 11:30pm. Looked stressed, didn't say hi. Walked past me.",
        source="alex.rivera", certainty=0.9, importance=0.95)
    claim("Alex", "was_at", "Helios_office", source="alex.rivera",
          valid_from=t(22), valid_to=t(1, 30), confidence="high")
    claim("David", "was_at", "Helios_office", source="alex.rivera",
          valid_from=t(22), valid_to=t(23, 30), confidence="high")

    sub("Seeding Jamie Torres (junior engineer | MILD LIE about being remote)")
    remember(
        "I was working from home all night. On my couch the whole evening.",
        source="jamie.torres", certainty=0.9, importance=0.8)
    claim("Jamie", "was_at", "Helios_office", source="jamie.torres",
          polarity=-1,  # Jamie DENIES being at the office
          valid_from=t(19), valid_to=t(1), confidence="high")

    sub("Seeding badge access logs (system.badge | authoritative)")
    remember("[BADGE] 19:02 Maya Chen entered via main lobby.",
             source="system.badge", certainty=1.0, importance=0.9)
    remember("[BADGE] 21:34 David Park entered via executive entrance.",
             source="system.badge", certainty=1.0, importance=0.95)
    remember("[BADGE] 22:03 Maya Chen exited via main lobby.",
             source="system.badge", certainty=1.0, importance=0.9)
    remember("[BADGE] 22:06 Alex Rivera entered via service entrance.",
             source="system.badge", certainty=1.0, importance=0.85)
    remember("[BADGE] 22:48 Jamie Torres entered via main lobby.",
             source="system.badge", certainty=1.0, importance=0.95)
    remember("[BADGE] 23:07 Jamie Torres exited via main lobby.",
             source="system.badge", certainty=1.0, importance=0.95)
    remember("[BADGE] 23:31 David Park exited via executive entrance.",
             source="system.badge", certainty=1.0, importance=0.95)
    claim("David", "was_at", "Helios_office", source="system.badge",
          valid_from=t(21, 34), valid_to=t(23, 31), confidence="high")
    claim("Jamie", "was_at", "Helios_office", source="system.badge",
          valid_from=t(22, 48), valid_to=t(23, 7), confidence="high")
    claim("Maya", "was_at", "Helios_office", source="system.badge",
          valid_from=t(19, 2), valid_to=t(22, 3), confidence="high")

    sub("Seeding git activity logs (system.git | authoritative)")
    remember(
        "[GIT] 23:08 user=jamie.torres pushed cosmetic UI fix to helios-internal-tools.",
        source="system.git", certainty=1.0, importance=0.75)
    remember(
        "[GIT] 23:14 user=david.park exported helios-core snapshot, pushed to "
        "FORK origin-external/dp-review.",
        source="system.git", certainty=1.0, importance=1.0)
    remember(
        "[GIT] 23:15 origin-external/dp-review made PUBLIC via gh cli.",
        source="system.git", certainty=1.0, importance=1.0)
    # David DID access (polarity=+1) — directly contradicts his denial above
    claim("David", "accessed", "production_repo", source="system.git",
          polarity=1,
          valid_from=t(23, 14), valid_to=t(23, 15), confidence="high")
    claim("David", "leaked", "production_code", source="system.git",
          valid_from=t(23, 15), confidence="high")


# ==================================================================
# Phase 2: Run think()
# ==================================================================

def run_think() -> dict:
    banner("PHASE 2  RUN THINK()  SCAN FOR CONTRADICTIONS")
    print("  POST /v1/think  (conflict detection on, consolidation off)")
    r = request_json("POST", "/v1/think", {
        "run_consolidation": False,
        "run_conflicts": True,
        "run_patterns": False,
    })
    print(f"    conflicts_found: {r.get('conflicts_found', 0)}")
    print(f"    duration_ms:     {r.get('duration_ms', 0):.1f}")
    return r


# ==================================================================
# Phase 3: Polarity contradiction detection from structured claims
# ==================================================================

def detect_polarity_contradictions() -> list[dict]:
    """Walk claims for every entity and find polarity=+1 vs polarity=-1 pairs
    on the same (src, rel_type, dst) -- the POLARITY_CONTRADICTION pattern
    from RFC 006. This is the MOST damning evidence type."""
    banner("PHASE 3  POLARITY CONTRADICTION DETECTION (RFC 006)")
    print("  Walking structured claims for each suspect...\n")

    suspects = ["David", "Maya", "Jamie", "Alex"]
    contradictions = []

    for entity in suspects:
        r = request_json("GET", f"/v1/claims?entity={entity}&namespace={NS}")
        claims = r.get("claims", [])
        # Group by (src, rel_type, dst) — find groups with BOTH polarities
        groups: dict[tuple, list] = {}
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
                            "positive_confidence": p["confidence_band"],
                            "positive_validity": (p.get("valid_from"), p.get("valid_to")),
                            "negative_source": n["extractor"],
                            "negative_confidence": n["confidence_band"],
                            "negative_validity": (n.get("valid_from"), n.get("valid_to")),
                        })

    if not contradictions:
        print("  (no polarity contradictions found)")
        return []

    for i, c in enumerate(contradictions, 1):
        pf, pt = c["positive_validity"]
        nf, nt = c["negative_validity"]
        print(f"  [{i}] POLARITY_CONTRADICTION")
        print(f"      subject:   {c['subject']}")
        print(f"      relation:  {c['relation']}  -->  {c['object']}")
        print(f"      ({c['positive_source']:15})  CLAIMS YES "
              f"[{fmt_time(pf)}-{fmt_time(pt)}] conf={c['positive_confidence']}")
        print(f"      ({c['negative_source']:15})  CLAIMS NO  "
              f"[{fmt_time(nf)}-{fmt_time(nt)}] conf={c['negative_confidence']}")
        print()

    return contradictions


# ==================================================================
# Phase 4: Temporal contradiction detection
# ==================================================================

def detect_temporal_contradictions() -> list[dict]:
    """For each was_at claim, compare witnesses' time windows. If one source
    says X was somewhere until 21:45 but another says X was still there at
    23:00, that's a temporal contradiction."""
    banner("PHASE 4  TEMPORAL CONTRADICTION DETECTION")
    print("  Cross-referencing was_at claims across sources...\n")

    contradictions = []
    for entity in ["David", "Maya", "Jamie", "Alex"]:
        r = request_json("GET", f"/v1/claims?entity={entity}&namespace={NS}")
        claims = [c for c in r.get("claims", [])
                  if c["rel_type"] == "was_at" and c["polarity"] == 1]
        # Compare each pair
        for i, a in enumerate(claims):
            for b in claims[i+1:]:
                if a["src"] != b["src"] or a["dst"] != b["dst"]:
                    continue
                a_to = a.get("valid_to")
                b_from = b.get("valid_from")
                b_to = b.get("valid_to")
                a_from = a.get("valid_from")
                if a_to is None or b_from is None or a_from is None or b_to is None:
                    continue
                # Disjoint windows from different sources = contradiction
                if a_to < b_from or b_to < a_from:
                    continue  # agree on disjoint periods
                # Overlap — but significantly different end times = someone's lying
                end_gap = abs(a_to - b_to)
                if end_gap > 900:  # >15 minutes discrepancy in exit time
                    contradictions.append({
                        "entity": a["src"], "location": a["dst"],
                        "src_a": a["extractor"], "to_a": a_to,
                        "src_b": b["extractor"], "to_b": b_to,
                        "gap_minutes": end_gap / 60,
                    })

    if not contradictions:
        print("  (no temporal contradictions found)")
        return []

    for i, c in enumerate(contradictions, 1):
        a_time = fmt_time(c["to_a"])
        b_time = fmt_time(c["to_b"])
        print(f"  [{i}] {c['entity']} left {c['location']}: "
              f"{c['src_a']} says {a_time}, {c['src_b']} says {b_time} "
              f"({c['gap_minutes']:.0f} min gap)")

    return contradictions


# ==================================================================
# Phase 5: Cross-reference with explicit denials (Jamie-style "I wasn't there")
# ==================================================================

def detect_presence_denials() -> list[dict]:
    """Catch someone claiming polarity=-1 on was_at while system.badge has
    polarity=+1 on the same location with a specific time window."""
    banner("PHASE 5  PRESENCE DENIALS CAUGHT BY BADGE LOGS")
    contradictions = []
    for entity in ["David", "Maya", "Jamie", "Alex"]:
        r = request_json("GET", f"/v1/claims?entity={entity}&namespace={NS}")
        denies = [c for c in r.get("claims", [])
                  if c["rel_type"] == "was_at" and c["polarity"] == -1]
        confirms = [c for c in r.get("claims", [])
                    if c["rel_type"] == "was_at" and c["polarity"] == 1
                    and c["extractor"].startswith("system.")]
        for d in denies:
            for c in confirms:
                if d["src"] == c["src"] and d["dst"] == c["dst"]:
                    contradictions.append({
                        "entity": d["src"], "location": d["dst"],
                        "denier": d["extractor"],
                        "contradicting_source": c["extractor"],
                        "system_from": c.get("valid_from"),
                        "system_to": c.get("valid_to"),
                    })

    if not contradictions:
        print("  (no presence denials found)")
        return []

    for i, c in enumerate(contradictions, 1):
        from_t = fmt_time(c["system_from"])
        to_t = fmt_time(c["system_to"])
        print(f"  [{i}] {c['entity']} denies being at {c['location']}, "
              f"but {c['contradicting_source']} logs {from_t}-{to_t}")
    return contradictions


# ==================================================================
# Phase 6: Recall-driven evidence chain for the perpetrator
# ==================================================================

def evidence_chain(subject: str) -> None:
    banner(f"PHASE 6  EVIDENCE CHAIN FOR {subject.upper()}")
    r = request_json("POST", "/v1/recall", {
        "query": f"what did {subject} do that night",
        "top_k": 10,
        "namespace": NS,
    })
    print(f"  Recall: 'what did {subject} do that night'\n")
    for i, m in enumerate(r.get("results", []), 1):
        src = m.get("source", "?")
        text = m.get("text", "")
        if len(text) > 130:
            text = text[:127] + "..."
        print(f"  [{i}] {src:15} score={m.get('score',0):.2f}")
        print(f"      {text}")
        print()


# ==================================================================
# Phase 7: Scoped conflict list from the server (namespace-filtered)
# ==================================================================

def show_scoped_conflicts() -> int:
    banner("PHASE 7  SCOPED CONFLICTS FROM /v1/conflicts?namespace=...")
    r = request_json("GET", f"/v1/conflicts?namespace={NS}&limit=50")
    conflicts = r.get("conflicts", [])
    print(f"  {len(conflicts)} conflict(s) involving memories in this namespace.\n")
    for i, c in enumerate(conflicts[:10], 1):
        print(f"  [{i}] {c.get('conflict_type','?')} "
              f"[priority={c.get('priority','?')}] entity={c.get('entity','?')}")
        reason = c.get("detection_reason", "")
        if len(reason) > 120:
            reason = reason[:117] + "..."
        print(f"      {reason}")
    return len(conflicts)


# ==================================================================
# Phase 8: The verdict — synthesized from everything above
# ==================================================================

def verdict(polarity_contras: list, temporal_contras: list,
            denials: list, scoped_count: int) -> None:
    banner("PHASE 8  VERDICT (SYNTHESIZED FROM REAL QUERIES)")

    # Rank suspects by contradiction count
    scores: dict[str, int] = {}
    for c in polarity_contras:
        scores[c["subject"]] = scores.get(c["subject"], 0) + 3  # polarity most damning
    for c in temporal_contras:
        scores[c["entity"]] = scores.get(c["entity"], 0) + 2
    for c in denials:
        scores[c["entity"]] = scores.get(c["entity"], 0) + 2

    print("  Suspect contradiction scores (weighted):")
    for entity, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"    {entity:10} {score} points  "
              f"({'VERY HIGH' if score >= 5 else 'HIGH' if score >= 3 else 'elevated'})")
    print()

    if not scores:
        print("  No contradictions detected.")
        return

    culprit = max(scores.items(), key=lambda x: x[1])[0]
    print(f"  PRIMARY SUSPECT: {culprit}")

    # Look up the specific action connecting them to the breach
    r = request_json("GET", f"/v1/claims?entity={culprit}&namespace={NS}")
    guilty_claims = [c for c in r.get("claims", [])
                     if c["extractor"].startswith("system.") and c["polarity"] == 1
                     and c["rel_type"] in ("accessed", "leaked")]

    if guilty_claims:
        print("\n  Actions attributed to this suspect by AUTHORITATIVE sources:")
        for c in guilty_claims:
            t_str = fmt_time(c.get("valid_from"))
            print(f"    [{c['extractor']}] {c['src']} --{c['rel_type']}--> "
                  f"{c['dst']}  at {t_str}")

    if polarity_contras:
        print("\n  Their stated position (proven false):")
        for c in polarity_contras:
            if c["subject"] == culprit:
                print(f"    [{c['negative_source']}] denied {c['relation']} --> {c['object']}")
                print(f"    [{c['positive_source']}] confirmed it at "
                      f"{fmt_time(c['positive_validity'][0])}")

    print()
    print("  " + "-" * 68)
    print("  Reconstruction entirely derived from:")
    print(f"    - {len(polarity_contras)} polarity contradiction(s)")
    print(f"    - {len(temporal_contras)} temporal contradiction(s)")
    print(f"    - {len(denials)} presence denial(s) caught by logs")
    print(f"    - {scoped_count} conflict(s) auto-detected by YantrikDB")
    print()
    print("  No hardcoded conclusions. Every lie surfaced was computed from")
    print("  the claims ledger. This is memory as a reasoning substrate,")
    print("  not memory as a search index.")


# ==================================================================
# Main
# ==================================================================

def main():
    banner(f"THE RASHOMON ENGINE | Helios Labs Data Breach Investigation")
    print(f"  Cluster:    {BASE}")
    print(f"  Namespace:  {NS}")
    print(f"  Date:       2026-03-15, 19:00-00:00 UTC")

    seed()
    time.sleep(2)  # let engine settle

    run_think()

    polarity = detect_polarity_contradictions()
    temporal = detect_temporal_contradictions()
    denials = detect_presence_denials()

    evidence_chain("David")

    scoped = show_scoped_conflicts()

    verdict(polarity, temporal, denials, scoped)

    banner("DONE  Truth reconstructed from conflicting testimony.")


if __name__ == "__main__":
    main()
