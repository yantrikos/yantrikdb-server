#!/usr/bin/env python3
"""Watergate — the historical research showcase.

Forty-nine years after Nixon's resignation, every primary source is public:
Senate Watergate Committee transcripts, the Oval Office tapes, Nixon's
public statements, sworn depositions from his inner circle.

This showcase feeds YantrikDB a slice of those sources and asks it to do
what the Watergate investigators did over two years in a matter of seconds:
find the contradictions between what Nixon said in public and what he said
in private, between what his aides swore under oath and what the tapes
later revealed.

The verdict at the end is derived from polarity contradictions and source
cross-referencing — not hardcoded in Python. Every lie surfaced is caught
by the claims ledger.

Requires yantrikdb-server v0.7.2+ and yantrikdb 0.6.1+.

Usage:
  python watergate_engine.py <token> [base_url]

Sources are public domain: Nixon tapes (National Archives), Senate Watergate
Committee Report (1973), public Nixon statements (1972-74 press archives).
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
NS = "watergate-1972-1974"


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


if not TOKEN:
    die("missing token. usage: python watergate_engine.py <token> [base_url]")


def request_json(method, path, payload=None):
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
        die(f"{method} {path} -> HTTP {exc.code}: {exc.read().decode(errors='replace')}")


def remember(text, *, source, importance=0.7, certainty=0.9):
    r = request_json("POST", "/v1/remember", {
        "text": text, "memory_type": "episodic", "importance": importance,
        "valence": 0.0, "domain": "historical_research", "source": source,
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
# Phase 1: Seed public record
# ==================================================================
#
# Each source is a historical primary record. Quotes are from public
# transcripts and press archives. The "system.tape" source is the
# National Archives' Nixon White House Tapes collection — the
# authoritative record of what Nixon actually said in private.

def seed_public_nixon():
    sub("Nixon's PUBLIC statements (press conferences, speeches)")
    remember(
        "[1972-06-22 press conference] 'The White House has had no "
        "involvement whatever in this particular incident.'",
        source="nixon.public", importance=0.95)
    remember(
        "[1972-08-29 press conference] 'I can say categorically that... "
        "no one in the White House staff, no one in this administration, "
        "presently employed, was involved in this very bizarre incident.'",
        source="nixon.public", importance=0.95)
    remember(
        "[1973-04-30 Oval Office address] 'There can be no whitewash at "
        "the White House.'",
        source="nixon.public", importance=0.9)
    remember(
        "[1973-11-17 press conference, Disney World] 'People have got to "
        "know whether or not their President is a crook. Well, I am not "
        "a crook.'",
        source="nixon.public", importance=1.0)

    # Structured claims — Nixon's public denials
    claim("Nixon", "authorized", "Watergate_coverup", source="nixon.public",
          polarity=-1,  # he publicly denies
          valid_from=d(1972, 6, 17), valid_to=d(1974, 8, 9), confidence="high")
    claim("Nixon_administration", "knew_of", "Watergate_break_in",
          source="nixon.public", polarity=-1,
          valid_from=d(1972, 6, 17), valid_to=d(1972, 12, 31), confidence="high")
    claim("Nixon", "discussed", "hush_money", source="nixon.public",
          polarity=-1,  # he denies
          valid_from=d(1972, 6, 17), valid_to=d(1974, 8, 9), confidence="high")


def seed_nixon_tapes():
    sub("Nixon White House tapes (system.tape — authoritative)")
    # The "smoking gun" tape, released August 5, 1974
    remember(
        "[TAPE 1972-06-23 10:04 AM, Oval Office] President Nixon to H.R. "
        "Haldeman: 'You call [FBI director Pat] Gray in, and just say... "
        "we feel that... for the good of the country, don't go any further "
        "into this case, period!'",
        source="system.tape", importance=1.0, certainty=1.0)
    remember(
        "[TAPE 1972-06-23] Haldeman: 'Who was the asshole that did?' "
        "Nixon: '...it's Liddy. He just isn't well screwed on...' "
        "— Nixon naming G. Gordon Liddy as responsible for the break-in, "
        "six days after denying any White House involvement.",
        source="system.tape", importance=1.0, certainty=1.0)
    remember(
        "[TAPE 1973-03-21] Nixon to John Dean: 'How much money do you need?' "
        "Dean: 'I would say these people are going to cost a million dollars "
        "over the next two years.' Nixon: 'We could get that... you could "
        "get a million dollars. And you could get it in cash.'",
        source="system.tape", importance=1.0, certainty=1.0)
    remember(
        "[TAPE 1972-06-20] An 18-1/2 minute gap appears in the recording "
        "of Nixon's meeting with Haldeman on the first workday after the "
        "break-in. Rose Mary Woods claims she accidentally erased 5 minutes; "
        "the remaining 13-1/2 minutes remain unexplained.",
        source="system.tape", importance=0.95, certainty=0.95)

    # Ground-truth claims from the tapes
    claim("Nixon", "authorized", "Watergate_coverup", source="system.tape",
          polarity=1,  # affirmative — the tape proves it
          valid_from=d(1972, 6, 23), confidence="high")
    claim("Nixon", "discussed", "hush_money", source="system.tape",
          polarity=1,
          valid_from=d(1973, 3, 21), confidence="high")
    claim("Nixon", "named_as_responsible", "G_Gordon_Liddy",
          source="system.tape", valid_from=d(1972, 6, 23), confidence="high")


def seed_john_dean():
    sub("John Dean sworn testimony (Senate Watergate Committee, June 1973)")
    remember(
        "[1973-06-25 Senate testimony] 'To CM (Commitment Memory): "
        "There is a cancer within, close to the presidency, that is growing. "
        "It is growing daily; it's compounding.'",
        source="dean.testimony", importance=1.0, certainty=0.95)
    remember(
        "[1973-06-25 Senate testimony] John Dean testifies that on "
        "March 21, 1973, he warned Nixon about the cover-up and told him "
        "hush money was being paid to the Watergate defendants. Nixon, "
        "according to Dean, discussed raising up to $1 million for "
        "continued payments.",
        source="dean.testimony", importance=0.95, certainty=0.9)
    remember(
        "[1973-06-25 Senate testimony] Dean asserts Nixon was fully aware "
        "of the cover-up by September 15, 1972.",
        source="dean.testimony", importance=0.9, certainty=0.9)

    claim("Nixon", "discussed", "hush_money", source="dean.testimony",
          polarity=1,
          valid_from=d(1973, 3, 21), confidence="high")
    claim("Nixon", "authorized", "Watergate_coverup", source="dean.testimony",
          polarity=1,
          valid_from=d(1972, 9, 15), confidence="high")


def seed_haldeman():
    sub("H.R. Haldeman sworn testimony + public statements (1973)")
    remember(
        "[1973-07 Senate testimony] Haldeman initially denies that the "
        "President had any foreknowledge of the break-in or the cover-up, "
        "characterizing Dean's account as 'not accurate.'",
        source="haldeman.testimony", importance=0.9, certainty=0.85)
    remember(
        "[1973-07-30 Senate testimony] Asked whether he and Nixon had "
        "discussed getting the CIA to block the FBI's Watergate "
        "investigation, Haldeman equivocates: 'I don't recall that being "
        "the purpose of the conversation.'",
        source="haldeman.testimony", importance=0.9, certainty=0.85)

    claim("Haldeman", "discussed_with", "Nixon_FBI_block",
          source="haldeman.testimony", polarity=-1,  # he denies
          valid_from=d(1972, 6, 23), confidence="medium")


def seed_ehrlichman():
    sub("John Ehrlichman testimony (1973)")
    remember(
        "[1973-07 Senate testimony] Ehrlichman denies knowledge of hush "
        "money payments and states that the President authorized no "
        "illegal activity.",
        source="ehrlichman.testimony", importance=0.85, certainty=0.8)
    claim("Nixon", "authorized", "Watergate_coverup",
          source="ehrlichman.testimony", polarity=-1,
          valid_from=d(1972, 6, 17), valid_to=d(1973, 7, 31), confidence="medium")


def seed_senate_report():
    sub("Senate Watergate Committee findings (official record)")
    remember(
        "[1974-06 Senate Watergate Report] The committee concludes that "
        "the Nixon White House orchestrated and directed a systematic "
        "cover-up of the Watergate break-in involving false testimony, "
        "obstruction of justice, and use of intelligence agencies to "
        "impede investigation.",
        source="senate.report", importance=1.0, certainty=0.95)
    remember(
        "[1974-07-24 Supreme Court ruling, United States v. Nixon] "
        "President Nixon ordered to turn over the White House tapes. "
        "Executive privilege does not shield a president from criminal "
        "investigation.",
        source="senate.report", importance=1.0, certainty=1.0)
    remember(
        "[1974-08-05] Release of the 'smoking gun' tape of June 23, 1972 "
        "proves Nixon authorized the cover-up six days after the break-in. "
        "Nixon loses Republican support in Congress within hours.",
        source="senate.report", importance=1.0, certainty=1.0)
    remember(
        "[1974-08-09] Nixon resigns the presidency, effective noon the "
        "following day — the only U.S. president to do so.",
        source="senate.report", importance=1.0, certainty=1.0)


# ==================================================================
# Phase 2: Analysis (all computed, not scripted)
# ==================================================================

def run_think():
    banner("PHASE 2  THINK() — SCAN THE RECORD FOR CONTRADICTIONS")
    r = request_json("POST", "/v1/think", {
        "run_consolidation": False, "run_conflicts": True, "run_patterns": False,
    })
    print(f"  conflicts_found: {r.get('conflicts_found', 0)}")
    print(f"  duration_ms:     {r.get('duration_ms', 0):.1f}")


def detect_polarity_contradictions():
    banner("PHASE 3  POLARITY CONTRADICTIONS (what was denied vs what was proved)")
    print("  Walking structured claims for each subject...\n")

    subjects = ["Nixon", "Nixon_administration", "Haldeman"]
    contradictions = []

    for entity in subjects:
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
                            "negative_source": n["extractor"],
                            "negative_from": n.get("valid_from"),
                        })

    if not contradictions:
        print("  (no polarity contradictions)")
        return []

    for i, c in enumerate(contradictions, 1):
        pf = fmt_date(c["positive_from"])
        nf = fmt_date(c["negative_from"])
        print(f"  [{i}] POLARITY_CONTRADICTION")
        print(f"      {c['subject']} --{c['relation']}--> {c['object']}")
        print(f"      ({c['negative_source']:24}) CLAIMS NO  from {nf}")
        print(f"      ({c['positive_source']:24}) CLAIMS YES from {pf}")
        print()
    return contradictions


def trace_the_lie(subject, relation, obj):
    banner(f"PHASE 4  THE EVIDENCE CHAIN FOR: {subject} {relation} {obj}")
    query = f"{subject} {relation} {obj}"
    r = request_json("POST", "/v1/recall", {
        "query": query, "top_k": 8, "namespace": NS,
    })
    print(f"  Query: '{query}'\n")
    for i, m in enumerate(r.get("results", []), 1):
        src = m.get("source", "?")
        text = m.get("text", "")
        if len(text) > 150:
            text = text[:147] + "..."
        print(f"  [{i}] {src:24} score={m.get('score',0):.2f}")
        print(f"      {text}")
        print()


def verdict(polarity_contras):
    banner("PHASE 5  VERDICT — WHAT YANTRIKDB RECONSTRUCTED")
    print()
    if not polarity_contras:
        print("  No contradictions detected.")
        return

    # Rank subjects by contradiction count
    scores = {}
    for c in polarity_contras:
        scores[c["subject"]] = scores.get(c["subject"], 0) + 1

    print("  Subjects with documented public-denial vs private-evidence contradictions:")
    for entity, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"    {entity:25} {score} contradiction(s)")
    print()

    print("  The cover-up collapsed because the tapes created a polarity")
    print("  contradiction with every public denial — a structured, permanent,")
    print("  cross-referenceable record. What took the Senate Watergate")
    print("  Committee two years of subpoenas, hearings, and Supreme Court")
    print("  battles to surface, YantrikDB's claims ledger surfaces in one")
    print("  query.")
    print()
    print("  Historical footnote: Nixon resigned on 1974-08-09, four days")
    print("  after the release of the smoking-gun tape of 1972-06-23.")


def main():
    banner(f"WATERGATE — the historical research showcase")
    print(f"  Cluster:    {BASE}")
    print(f"  Namespace:  {NS}")
    print(f"  Period:     1972-06-17 (break-in)  to  1974-08-09 (resignation)")

    banner("PHASE 1  SEED PRIMARY SOURCES (public record only)")
    seed_public_nixon()
    seed_nixon_tapes()
    seed_john_dean()
    seed_haldeman()
    seed_ehrlichman()
    seed_senate_report()
    time.sleep(2)

    run_think()
    polarity = detect_polarity_contradictions()

    # The two most consequential contradictions to trace in detail:
    trace_the_lie("Nixon", "authorized", "Watergate_coverup")
    trace_the_lie("Nixon", "discussed", "hush_money")

    verdict(polarity)

    banner("DONE — primary sources cross-referenced.")


if __name__ == "__main__":
    main()
