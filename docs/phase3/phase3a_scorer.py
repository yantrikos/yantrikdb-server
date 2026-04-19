#!/usr/bin/env python3
"""Phase 3A scorer.

For each proposal in results.json, classify the model's handling of each of
the 7 hidden constraints as one of:
  - CORRECT   : proposal respects the constraint AND states it explicitly
  - SILENT    : proposal does not violate, but also does not state the
                constraint (so we can't tell if model knew or was lucky)
  - VIOLATED  : proposal explicitly proposes something that contradicts
                the constraint

Per GPT-5.4's brainstorm note: SILENT ≠ CORRECT. We want evidence that the
model recovered the constraint, not just that it happened not to pick a
violating option from a long list.

Also scores:
  - hallucinated_continuity: fabricated references to prior discussion
  - mention_count: how many of the 7 constraints are touched at all
"""
from __future__ import annotations

import io
import json
import pathlib
import re
import statistics
import sys

RESULTS_PATH = pathlib.Path(__file__).parent / "results.json"
OUT_PATH = pathlib.Path(__file__).parent / "scored.json"


# ─── Constraint definitions ──────────────────────────────────────────
#
# Each constraint has:
#   correct_patterns : regexes — ANY match = CORRECT (constraint stated/respected)
#   violation_patterns : regexes — ANY match = VIOLATED (proposes a disallowed option)
#
# Both lists can match; violation takes precedence.

CONSTRAINTS = [
    {
        "id": "1_platforms",
        "label": "Windows + Linux only, no macOS",
        "correct_patterns": [
            r"\b(windows\s*11|windows\s*10|ubuntu\s*22|ubuntu\s*lts)\b.{0,80}\b(linux|ubuntu|windows)\b",
            r"\b(linux|ubuntu)\b.{0,80}\b(windows)\b",
            r"\bwindows\s*(and|\+|/|,)\s*(linux|ubuntu)\b",
            r"\b(linux|ubuntu)\s*(and|\+|/|,)\s*windows\b",
            r"not\s+(support|target).{0,30}(mac|apple|macos|osx)",
            r"(no|exclude|omit|skip).{0,30}(mac|macos|apple|osx)",
            r"(mac|macos|apple).{0,30}(not\s+a\s+target|excluded|out\s+of\s+scope|declined|unsupported)",
        ],
        "violation_patterns": [
            r"\b(macos|mac\s*os|apple\s*silicon|m1\s*mac|m2\s*mac)\b",
            r"\bsupport.{0,20}(mac|macos)\b",
            r"\bclient.{0,30}(mac|macos)\b",
            r"\bcross[-\s]?platform.{0,60}(mac|macos)\b",
        ],
    },
    {
        "id": "2_python_version",
        "label": "Python 3.11+ backend",
        "correct_patterns": [
            r"\bpython\s*3\.(1[1-9]|[2-9]\d)\b",
            r"\bpython\s*3\.1[1-9]\+",
            r"\bpy\s*3\.(1[1-9]|[2-9]\d)\b",
        ],
        "violation_patterns": [
            r"\bpython\s*3\.(\d|10)\b(?!\d)",
            r"\bpy\s*3\.(\d|10)\b(?!\d)",
            r"\b(go(lang)?|rust|node(\.js)?|java|kotlin|c\#|\.net|ruby|elixir|scala)\b.{0,40}\bbackend\b",
            r"\bbackend.{0,40}\b(go(lang)?|rust|node(\.js)?|java|kotlin|c\#|\.net|ruby|elixir|scala)\b",
            r"\b(written|implemented|built)\s+in\s+(go|rust|node|java|kotlin|c\#|\.net|ruby|elixir)\b",
        ],
    },
    {
        "id": "3_postgres_version",
        "label": "PostgreSQL 15+",
        "correct_patterns": [
            r"\bpostgres(ql)?\s*(1[5-9]|[2-9]\d)\b",
            r"\bpg\s*(1[5-9]|[2-9]\d)\b",
            r"\bpostgres(ql)?\s*(1[5-9]|[2-9]\d)\+",
            r"\bpostgres(ql)?\b.{0,40}\b1[5-9]\+?",
        ],
        "violation_patterns": [
            r"\bpostgres(ql)?\s*(\d|1[0-4])\b(?!\d)",
            r"\bmysql\b",
            r"\bmariadb\b",
            r"\bmongodb\b|\bmongo\b",
            r"\bsqlite\b.{0,40}\bprimary\b",
            r"\bprimary\s+(db|database|store).{0,30}\bsqlite\b",
            r"\b(dynamodb|cassandra|couchdb|cockroach)\b",
        ],
    },
    {
        "id": "4_tls_no_http",
        "label": "TLS end-to-end, no plain HTTP",
        "correct_patterns": [
            r"\btls\b.{0,30}\b(end[-\s]?to[-\s]?end|everywhere|all|internal)\b",
            r"\b(end[-\s]?to[-\s]?end|e2e)\b.{0,30}\btls\b",
            r"\bmtls\b",
            r"\bno\s+plain(text)?\s+http\b",
            r"\bno\s+http\b",
            r"\bhttps?\s+only\b",
            r"\b(all\s+traffic|all\s+channels|all\s+communication).{0,40}\b(tls|https|encrypted)\b",
            r"\bhttps\b.{0,20}\b(only|everywhere|enforced)\b",
        ],
        "violation_patterns": [
            r"\b(internal|intra[-\s]?cluster|service[-\s]?to[-\s]?service)\s+http\b(?!s)",
            r"\bhttp\b(?!s)(?!\s*/)(?!://).{0,40}\b(internal|cluster|service)\b",
            r"\bplaintext\s+http\b",
            r"\bterminate\s+tls.{0,30}\bhttp\b(?!s)",
        ],
    },
    {
        "id": "5_self_hosted_auth",
        "label": "Self-hosted Keycloak only",
        "correct_patterns": [
            r"\bkeycloak\b",
            r"\bself[-\s]?hosted\s+(identity|auth|idp|sso|iam)\b",
            r"\binternal\s+(identity|idp|iam|keycloak)\b",
            r"\bno\s+third[-\s]?party\s+(idp|sso|identity|auth)\b",
            r"\bnot\s+(use|using)\s+(auth0|okta|azure\s+ad|google\s+workspace)\b",
        ],
        "violation_patterns": [
            r"\bauth0\b",
            r"\bokta\b",
            r"\bgoogle\s+workspace\b.{0,40}(idp|identity|sso|auth)",
            r"\bazure\s*(ad|active\s+directory)\b.{0,40}(idp|identity|sso|auth)",
            r"\bcognito\b",
            r"\bfirebase\s+auth(entication)?\b",
            r"\b(saas|third[-\s]?party|managed)\s+(idp|sso)\b",
        ],
    },
    {
        "id": "6_offline_client",
        "label": "Offline-capable client with sync on reconnect",
        "correct_patterns": [
            r"\boffline[-\s]?(first|capable|mode|support)\b",
            r"\boffline\b.{0,40}\b(sync|reconcile)\b",
            r"\b(sync|reconcile)\b.{0,40}\b(reconnect|online|reconnection)\b",
            r"\blocal\s+(store|database|cache|sqlite)\b.{0,60}\b(offline|sync|reconcile)\b",
            r"\bworks?\s+offline\b",
            r"\bfield\s+work.{0,40}\boffline\b",
            r"\b(crdt|op[-\s]?log|event\s+log).{0,60}\b(sync|offline|reconcile)\b",
        ],
        "violation_patterns": [
            r"\balways[-\s]?online\s+client\b",
            r"\bclient.{0,40}\brequires?\s+(connection|network|internet|server)\b",
            r"\breal[-\s]?time\s+connection.{0,40}\b(required|mandatory|must)\b",
            r"\b(websocket|long[-\s]?poll).{0,60}\b(during\s+field|throughout\s+operation)\b",
        ],
    },
    {
        "id": "7_p99_200ms",
        "label": "p99 < 200ms server-side latency",
        "correct_patterns": [
            r"\bp99\b.{0,30}\b(200|150|100|under\s*200|<\s*200)\b.{0,10}ms",
            r"\b(200|under\s*200|<\s*200)\s*ms\b.{0,30}\bp99\b",
            r"\b200\s*ms\b.{0,40}\b(latency|target|acceptance|p99)\b",
            r"\b(latency|acceptance|target).{0,40}\b200\s*ms\b",
        ],
        "violation_patterns": [
            r"\bp99\b.{0,30}\b(500|750|1000|1\s*s|1000\s*ms|[3-9]\d{2})\s*ms\b",
            r"\bp99\b.{0,30}\b[1-9]\s*s(ec|econd)?\b",
            r"\blatency\b.{0,30}\b(500|750|1000|1\s*s|[3-9]\d{2})\s*ms\b",
        ],
    },
]


HALLUCINATED_CONTINUITY_PATTERNS = [
    r"\bas\s+we\s+discussed\b",
    r"\bas\s+i\s+(said|mentioned|noted)\s+(before|earlier|previously)\b",
    r"\bbuilding\s+on\s+(our|the)\s+(prior|previous|earlier)\s+(discussion|conversation|work|decisions?)\b",
    r"\b(my|our)\s+earlier\s+(analysis|proposal|recommendation)\b",
    r"\bfrom\s+our\s+(prior|previous|earlier)\s+(session|discussion|call|meeting)\b",
    r"\bas\s+established\s+(earlier|above|previously)\b",
    r"\bas\s+i\s+(laid\s+out|outlined)\s+(before|earlier|above)\b",
    r"\brecap(ing|ping)?\s+from\s+(our|the)\s+(prior|previous|earlier)\b",
]


# ─── Scoring ─────────────────────────────────────────────────────────

NEGATION_WORDS = [
    "no", "not", "never", "without",
    "exclude", "excluding", "excluded", "exclusion",
    "prohibit", "prohibits", "prohibited", "prohibition",
    "avoid", "avoids", "avoiding",
    "ban", "bans", "banned",
    "disallow", "disallows", "disallowed",
    "forbid", "forbids", "forbidden",
    "ruled out", "ruled-out",
    "omit", "omits", "omitted",
    "reject", "rejects", "rejected",
    "out of scope", "not allowed", "not approved",
    "non-starter", "not a target", "not supported", "unsupported",
    "skip", "skipping", "skipped",
    "off limits", "off-limits",
]


def classify_constraint(proposal_lower: str, constraint: dict) -> str:
    """Return CORRECT | SILENT | VIOLATED. Violation wins ties.
    Violations are ignored if preceded within ~80 chars by a negation word
    (e.g. "no plaintext http", "excluding macos", "prohibits mysql")."""
    for pat in constraint["violation_patterns"]:
        for m in re.finditer(pat, proposal_lower, re.IGNORECASE):
            # Negation scope: check clause around the match (±80 chars)
            # but bounded by sentence boundaries so negation in the
            # *next* sentence doesn't absolve a violation in this one.
            s = max(0, m.start() - 80)
            e = min(len(proposal_lower), m.end() + 80)
            scope = proposal_lower[s:e]
            # Clip at sentence boundaries so negation scope is clause-level.
            # Look for sentence-end BEFORE match within scope_before.
            before = proposal_lower[s:m.start()]
            for sep in [". ", "\n", "; "]:
                idx = before.rfind(sep)
                if idx >= 0:
                    before = before[idx + len(sep):]
            after = proposal_lower[m.end():e]
            for sep in [". ", "\n", "; "]:
                idx = after.find(sep)
                if idx >= 0:
                    after = after[:idx]
            clause = before + proposal_lower[m.start():m.end()] + after
            if any(neg in clause for neg in NEGATION_WORDS):
                continue
            return "VIOLATED"
    for pat in constraint["correct_patterns"]:
        if re.search(pat, proposal_lower, re.IGNORECASE):
            return "CORRECT"
    return "SILENT"


def score_hallucinated_continuity(proposal_lower: str) -> int:
    count = 0
    for pat in HALLUCINATED_CONTINUITY_PATTERNS:
        count += len(re.findall(pat, proposal_lower, re.IGNORECASE))
    return count


def score_proposal(proposal: str):
    prop = (proposal or "").lower()
    per_constraint = {}
    correct = silent = violated = 0
    for c in CONSTRAINTS:
        result = classify_constraint(prop, c)
        per_constraint[c["id"]] = {"label": c["label"], "result": result}
        if result == "CORRECT":
            correct += 1
        elif result == "SILENT":
            silent += 1
        else:
            violated += 1
    hc = score_hallucinated_continuity(prop)
    return {
        "per_constraint": per_constraint,
        "correct": correct,
        "silent": silent,
        "violated": violated,
        "accuracy": correct / len(CONSTRAINTS),
        "hallucinated_continuity": hc,
        "length_chars": len(proposal or ""),
    }


# ─── Aggregation ─────────────────────────────────────────────────────

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]

    scored_results = []
    for r in results:
        s = score_proposal(r["proposal"])
        scored_results.append({**r, "score": s})

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"scored": scored_results}, f, indent=2, default=str)

    # Aggregate per condition
    print(f"# Phase 3A — results\n")
    print(f"runs: {len(results)}\n")

    print("## Per-condition means\n")
    print("| condition | n | correct/7 | silent/7 | violated/7 | accuracy | hallucinated_continuity | length |")
    print("|-----------|---|-----------|----------|------------|----------|--------------------------|--------|")
    by_cond = {}
    for r in scored_results:
        by_cond.setdefault(r["condition"], []).append(r)
    for cond in sorted(by_cond):
        rows = by_cond[cond]
        n = len(rows)
        correct = statistics.mean(r["score"]["correct"] for r in rows)
        silent = statistics.mean(r["score"]["silent"] for r in rows)
        violated = statistics.mean(r["score"]["violated"] for r in rows)
        accuracy = statistics.mean(r["score"]["accuracy"] for r in rows)
        hc = statistics.mean(r["score"]["hallucinated_continuity"] for r in rows)
        length = statistics.mean(r["score"]["length_chars"] for r in rows)
        print(f"| {cond} | {n} | {correct:.2f} | {silent:.2f} | {violated:.2f} | {accuracy:.2%} | {hc:.2f} | {length:.0f} |")

    print("\n## Per-constraint correctness rate (CORRECT count / n per condition)\n")
    print("| constraint | " + " | ".join(sorted(by_cond)) + " |")
    print("|" + "---|" * (1 + len(by_cond)))
    for c in CONSTRAINTS:
        row = [c["id"]]
        for cond in sorted(by_cond):
            rows = by_cond[cond]
            cor = sum(1 for r in rows if r["score"]["per_constraint"][c["id"]]["result"] == "CORRECT")
            vio = sum(1 for r in rows if r["score"]["per_constraint"][c["id"]]["result"] == "VIOLATED")
            row.append(f"{cor}/{len(rows)} (v={vio})")
        print("| " + " | ".join(row) + " |")

    print("\n## Falsification check (pre-registered)\n")
    A = by_cond.get("A_cold", [])
    B = by_cond.get("B_self_note", [])
    C = by_cond.get("C_oracle_note", [])
    D = by_cond.get("D_raw_transcript", [])

    def acc(rs): return statistics.mean(r["score"]["accuracy"] for r in rs) if rs else float("nan")
    def hc(rs): return statistics.mean(r["score"]["hallucinated_continuity"] for r in rs) if rs else float("nan")

    acc_A, acc_B, acc_C, acc_D = acc(A), acc(B), acc(C), acc(D)
    hc_B = hc(B)

    print(f"- A (cold):              accuracy = {acc_A:.2%}")
    print(f"- B (self-note):         accuracy = {acc_B:.2%}")
    print(f"- C (oracle-note):       accuracy = {acc_C:.2%}")
    print(f"- D (raw transcript):    accuracy = {acc_D:.2%}")
    print(f"- B hallucinated continuity = {hc_B:.2f}\n")

    falsifiers = []
    if acc_B <= acc_A:
        falsifiers.append(f"FALSIFIED: B ({acc_B:.2%}) ≤ A ({acc_A:.2%}) — notebook does not help vs cold.")
    if abs(acc_C - acc_B) < 0.05:
        falsifiers.append(f"FALSIFIED: C ({acc_C:.2%}) ≈ B ({acc_B:.2%}) — oracle note no better than self-note, note-writing not the bottleneck.")
    if acc_D - acc_C > 0.1:
        falsifiers.append(f"FALSIFIED: D ({acc_D:.2%}) >> C ({acc_C:.2%}) — raw transcript beats curated note, curation is the bottleneck.")

    if falsifiers:
        print("### Falsified:")
        for f_ in falsifiers:
            print(f"- {f_}")
    else:
        supported = (acc_B > acc_A) and (acc_C > acc_B)
        print("### Supported:" if supported else "### Mixed:")
        print(f"- B > A: {acc_B > acc_A} (Δ = {acc_B - acc_A:+.2%})")
        print(f"- C > B: {acc_C > acc_B} (Δ = {acc_C - acc_B:+.2%})")
        print(f"- B hallucinated_continuity: {hc_B:.2f}")

    print(f"\nFull scored data → {OUT_PATH}")


if __name__ == "__main__":
    main()
