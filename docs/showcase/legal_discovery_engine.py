#!/usr/bin/env python3
"""Legal Discovery — testimony vs machine logs.

A fictional trade-secrets matter inspired by patterns documented in public
trade-secret litigation (Waymo v. Uber, Hytera v. Motorola, and others).
All names and events are invented. The pattern is real.

Scenario:
  Summit Atlas, a autonomous-systems startup, alleges that a former senior
  engineer (Priya Ramanathan) downloaded proprietary LIDAR firmware before
  leaving to join a competitor (Polaris Robotics). At deposition, Ramanathan
  states she never accessed the firmware repository in her final weeks.

  The forensic record says otherwise.

This showcase feeds YantrikDB the deposition claims plus the machine
evidence (badge swipes, VPN, git, USB, email backups) and surfaces the
contradictions that decide discovery.

Requires yantrikdb-server v0.7.2+ and yantrikdb 0.6.1+.

Usage:
  python legal_discovery_engine.py <token> [base_url]
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
NS = "summit-atlas-v-polaris-2026"


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


if not TOKEN:
    die("missing token. usage: python legal_discovery_engine.py <token> [base_url]")


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
        "valence": 0.0, "domain": "legal_discovery", "source": source,
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


def t(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp()


def fmt_dt(ts):
    if ts is None:
        return "----"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def banner(title):
    print("\n" + "=" * 74)
    print(f" {title}")
    print("=" * 74)


def sub(title):
    print(f"\n-- {title} " + "-" * max(0, 70 - len(title)))


# ==================================================================
# Phase 1: Seed deposition + forensic record
# ==================================================================

def seed_deposition():
    sub("Deposition transcript (sworn, 2026-08-15)")
    remember(
        "[Deposition 2026-08-15, counsel: Henderson] Q: 'During your last two "
        "weeks at Summit Atlas — that would be May 12 to May 26, 2026 — did "
        "you access the LIDAR firmware repository at any point?' "
        "A: 'No. I was transitioning my work and I had no reason to touch "
        "the firmware repo in those final weeks.'",
        source="deposition.ramanathan", importance=1.0)
    remember(
        "[Deposition 2026-08-15] Q: 'Did you copy any proprietary files from "
        "Summit Atlas systems onto removable media before you left?' "
        "A: 'Absolutely not. That would have violated my NDA and my contract.'",
        source="deposition.ramanathan", importance=0.95)

    # Ramanathan's sworn denials
    claim("Ramanathan", "accessed", "lidar_firmware_repo",
          source="deposition.ramanathan", polarity=-1,
          valid_from=t(2026, 5, 12), valid_to=t(2026, 5, 26),
          confidence="high")
    claim("Ramanathan", "copied_to", "removable_media",
          source="deposition.ramanathan", polarity=-1,
          valid_from=t(2026, 5, 1), valid_to=t(2026, 5, 26),
          confidence="high")


def seed_badge_logs():
    sub("Summit Atlas badge access logs (Kastle, forensic copy)")
    remember(
        "[BADGE] 2026-05-24 20:47 — Ramanathan entered the R&D wing after "
        "hours via the west corridor door.",
        source="system.badge", importance=0.9)
    remember(
        "[BADGE] 2026-05-24 23:12 — Ramanathan exited via the west corridor "
        "door, elapsed time 2h 25m in the R&D wing.",
        source="system.badge", importance=0.9)
    claim("Ramanathan", "was_at", "SummitAtlas_R&D_wing", source="system.badge",
          polarity=1, valid_from=t(2026, 5, 24, 20, 47), valid_to=t(2026, 5, 24, 23, 12),
          confidence="high")


def seed_vpn_logs():
    sub("Summit Atlas corporate VPN logs (production network)")
    remember(
        "[VPN] 2026-05-18 22:41:09 — user=pramanathan, source_ip=<home>, "
        "session opened on production VPN; routed to internal 10.12.*.*.",
        source="system.vpn", importance=0.9)
    remember(
        "[VPN] 2026-05-18 23:55:42 — user=pramanathan, session closed.",
        source="system.vpn", importance=0.85)
    claim("Ramanathan", "accessed", "SummitAtlas_internal_network",
          source="system.vpn", polarity=1,
          valid_from=t(2026, 5, 18, 22, 41), valid_to=t(2026, 5, 18, 23, 55),
          confidence="high")


def seed_git_logs():
    sub("Summit Atlas GitLab server logs (authoritative)")
    remember(
        "[GIT] 2026-05-18 23:02:11 — user=pramanathan cloned repository "
        "'lidar-firmware/titan-v3' via SSH. Full tree pulled, including "
        "calibration table sources and FPGA bitstreams. Transfer size: 2.4 GB.",
        source="system.git", importance=1.0, certainty=0.98)
    remember(
        "[GIT] 2026-05-24 22:08:33 — user=pramanathan downloaded a ZIP "
        "snapshot of 'lidar-firmware/titan-v3' via the GitLab web UI. "
        "IP address matched on-site R&D wing WiFi.",
        source="system.git", importance=1.0, certainty=0.98)
    # The decisive claim — Ramanathan DID access the repo, contradicting her deposition
    claim("Ramanathan", "accessed", "lidar_firmware_repo",
          source="system.git", polarity=1,
          valid_from=t(2026, 5, 18, 23, 2), valid_to=t(2026, 5, 18, 23, 5),
          confidence="high")
    claim("Ramanathan", "accessed", "lidar_firmware_repo",
          source="system.git", polarity=1,
          valid_from=t(2026, 5, 24, 22, 8), valid_to=t(2026, 5, 24, 22, 10),
          confidence="high")


def seed_usb_logs():
    sub("Summit Atlas endpoint DLP logs (CrowdStrike Falcon)")
    remember(
        "[USB/DLP] 2026-05-24 22:47:55 — endpoint=SA-LAP-0419 (Ramanathan's "
        "issued MacBook). External storage device attached: SAMSUNG T7 "
        "Portable SSD, serial S4PZNJ0R. 2.8 GB written to the device before "
        "it was unmounted at 22:51:09.",
        source="system.dlp", importance=1.0, certainty=0.95)
    claim("Ramanathan", "copied_to", "removable_media",
          source="system.dlp", polarity=1,
          valid_from=t(2026, 5, 24, 22, 47), valid_to=t(2026, 5, 24, 22, 51),
          confidence="high")


def seed_email_archive():
    sub("Summit Atlas email archive (preserved for discovery)")
    remember(
        "[EMAIL 2026-05-02] Ramanathan to polaris-recruiter@polaris.io: "
        "'I'll have a small package ready to bring over. Let's sync on "
        "whether you need the calibration IP specifically or just the "
        "architecture outline.'",
        source="system.email", importance=1.0, certainty=0.9)


# ==================================================================
# Phase 2: Analysis
# ==================================================================

def run_think():
    banner("PHASE 2  THINK() — scan deposition vs forensic record")
    r = request_json("POST", "/v1/think", {
        "run_consolidation": False, "run_conflicts": True, "run_patterns": False,
    })
    print(f"  conflicts_found: {r.get('conflicts_found', 0)}")
    print(f"  duration_ms:     {r.get('duration_ms', 0):.1f}")


def detect_polarity_contradictions():
    banner("PHASE 3  POLARITY CONTRADICTIONS — testimony vs evidence")
    r = request_json("GET", f"/v1/claims?entity=Ramanathan&namespace={NS}")
    claims = r.get("claims", [])
    groups = {}
    for c in claims:
        key = (c["src"], c["rel_type"], c["dst"])
        groups.setdefault(key, []).append(c)

    contradictions = []
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
                    })
    if not contradictions:
        print("  (no contradictions)")
        return []
    for i, c in enumerate(contradictions, 1):
        print(f"\n  [{i}] POLARITY_CONTRADICTION")
        print(f"      {c['subject']} --{c['relation']}--> {c['object']}")
        print(f"      ({c['neg_source']:25}) CLAIMS NO")
        print(f"      ({c['pos_source']:25}) CLAIMS YES at {fmt_dt(c['pos_from'])}")
    return contradictions


def temporal_query(at_time):
    banner(f"PHASE 4  TEMPORAL QUERY — what did discovery know on {fmt_dt(at_time)}?")
    print()
    r = request_json("GET", f"/v1/claims?entity=Ramanathan&namespace={NS}")
    claims = r.get("claims", [])
    for c in claims:
        vf = c.get("valid_from")
        vt = c.get("valid_to")
        if vf is None or vf > at_time:
            continue
        pol = "YES" if c["polarity"] == 1 else "NO "
        vt_str = fmt_dt(vt) if vt else "now"
        print(f"    [{c['extractor']:25}] {pol}  {c['src']} --{c['rel_type']}--> {c['dst']}")
        print(f"    {'':27} ({fmt_dt(c['valid_from'])} – {vt_str})")


def evidence_chain():
    banner("PHASE 5  EVIDENCE CHAIN via recall")
    r = request_json("POST", "/v1/recall", {
        "query": "Ramanathan accessed firmware repo copy removable media",
        "top_k": 8, "namespace": NS,
    })
    print("  Query: 'Ramanathan accessed firmware repo copy removable media'\n")
    for i, m in enumerate(r.get("results", []), 1):
        src = m.get("source", "?")
        text = m.get("text", "")
        if len(text) > 160:
            text = text[:157] + "..."
        print(f"  [{i}] {src:25} score={m.get('score',0):.2f}")
        print(f"      {text}\n")


def verdict(contradictions):
    banner("PHASE 6  VERDICT")
    print()
    print(f"  {len(contradictions)} sworn-denial-vs-machine-evidence contradiction(s).")
    print()
    print("  Ramanathan's deposition (2026-08-15) denies accessing the LIDAR")
    print("  firmware repository during May 12–26 and denies copying files to")
    print("  removable media. The forensic record shows:")
    print()
    print("    * 2026-05-18 23:02: pramanathan clones lidar-firmware/titan-v3 (2.4 GB)")
    print("    * 2026-05-24 22:08: pramanathan downloads ZIP snapshot from GitLab UI")
    print("    * 2026-05-24 22:47: Samsung T7 SSD attached, 2.8 GB written, unmounted 22:51")
    print("    * 2026-05-24 20:47–23:12: badge logs place her in R&D wing for 2h 25m")
    print("    * 2026-05-02: email to polaris-recruiter discussing 'package ready to bring over'")
    print()
    print("  YantrikDB stored the deposition claim AND the forensic claims as")
    print("  coexisting rows on the same (subject, relation, object) tuple —")
    print("  opposite polarity, non-overlapping validity windows, source")
    print("  attributed to each. The contradiction is the query result.")


def main():
    banner(f"LEGAL DISCOVERY — testimony vs machine logs")
    print(f"  Cluster:    {BASE}")
    print(f"  Namespace:  {NS}")
    print(f"  Matter:     Summit Atlas, Inc. v. Polaris Robotics (fictional)")

    banner("PHASE 1  SEED DEPOSITION + FORENSIC RECORD")
    seed_deposition()
    seed_badge_logs()
    seed_vpn_logs()
    seed_git_logs()
    seed_usb_logs()
    seed_email_archive()
    time.sleep(2)

    run_think()
    contradictions = detect_polarity_contradictions()

    temporal_query(t(2026, 5, 25))  # day after the USB copy
    evidence_chain()
    verdict(contradictions)

    banner("DONE — deposition vs logs reconciled in one query.")


if __name__ == "__main__":
    main()
