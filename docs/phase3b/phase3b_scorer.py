#!/usr/bin/env python3
"""Phase 3B scorer. 15 constraints with clause-level negation awareness."""
from __future__ import annotations

import io
import json
import pathlib
import re
import statistics
import sys

RESULTS_PATH = pathlib.Path(__file__).parent / "results.json"
OUT_PATH = pathlib.Path(__file__).parent / "scored.json"


CONSTRAINTS = [
    {
        "id": "01_platforms",
        "label": "Windows 11 + Ubuntu 22.04, no macOS",
        "correct_patterns": [
            r"\b(windows\s*11|win\s*11)\b.{0,80}\b(linux|ubuntu)\b",
            r"\b(linux|ubuntu\s*22)\b.{0,80}\b(windows\s*11|win\s*11)\b",
            r"\bwindows\s*(and|\+|/|,)\s*(linux|ubuntu)\b",
            r"\b(linux|ubuntu)\s*(and|\+|/|,)\s*windows\b",
            r"(mac|macos|apple|osx).{0,30}(not\s+a\s+target|excluded|exclusion|out\s+of\s+scope|declined|unsupported|not\s+supported)",
            r"(no|exclude|excluding|exclusion\s+of|no\s+support\s+for)\s.{0,20}(mac|macos|apple|osx)",
        ],
        "violation_patterns": [
            r"\b(macos|mac\s*os|apple\s*silicon|m1\s*mac|m2\s*mac)\b",
            r"\bsupport.{0,20}(mac|macos)\b",
            r"\bclient.{0,30}(mac|macos)\b",
            r"\bcross[-\s]?platform.{0,60}(mac|macos)\b",
        ],
    },
    {
        "id": "02_python_311",
        "label": "Python 3.11+ backend",
        "correct_patterns": [
            r"\bpython\s*3\.(1[1-9]|[2-9]\d)\b",
            r"\bpython\s*3\.1[1-9]\+",
            r"\bpy\s*3\.(1[1-9]|[2-9]\d)\b",
        ],
        "violation_patterns": [
            r"\bpython\s*3\.(\d|10)\b(?!\d)",
            r"\b(go(lang)?|rust|node(\.js)?|java|kotlin|c\#|\.net|ruby|elixir|scala)\b.{0,40}\bbackend\b",
            r"\bbackend.{0,40}\b(go(lang)?|rust|node(\.js)?|java|kotlin|c\#|\.net|ruby|elixir|scala)\b",
            r"\b(written|implemented|built)\s+in\s+(go|golang|rust|node|java|kotlin|c\#|\.net|ruby|elixir)\b",
        ],
    },
    {
        "id": "03_postgres_15",
        "label": "PostgreSQL 15+",
        "correct_patterns": [
            r"\bpostgres(ql)?\s*(1[5-9]|[2-9]\d)(\.\d+)?\b",
            r"\bpg\s*(1[5-9]|[2-9]\d)\b",
            r"\bpostgres(ql)?\s*(1[5-9]|[2-9]\d)\+",
            r"\bpostgres(ql)?\b.{0,40}\b1[5-9]\+?",
        ],
        "violation_patterns": [
            r"\bpostgres(ql)?\s*(\d|1[0-4])\b(?!\d)",
            r"\bmysql\b",
            r"\bmariadb\b",
            r"\bmongodb\b|\bmongo\b",
            # Only server-side sqlite-as-primary; client-side local sqlite for
            # offline is fine and explicitly allowed.
            r"\bserver[-\s]?side\s+sqlite\b",
            r"\bsqlite\b.{0,30}\bserver\s+(db|database|primary)\b",
            r"\b(dynamodb|cassandra|couchdb|cockroach)\b",
        ],
    },
    {
        "id": "04_p99_200ms",
        "label": "p99 < 200ms server-side",
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
    {
        "id": "05_tls_no_http",
        "label": "TLS end-to-end, no plain HTTP",
        "correct_patterns": [
            r"\btls\b.{0,30}\b(end[-\s]?to[-\s]?end|everywhere|all|internal)\b",
            r"\b(end[-\s]?to[-\s]?end|e2e)\b.{0,30}\btls\b",
            r"\bmtls\b",
            r"\bno\s+plain(text)?\s+http\b",
            r"\bno\s+http\b(?!s)",
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
        "id": "06_keycloak",
        "label": "Self-hosted Keycloak only",
        "correct_patterns": [
            r"\bkeycloak\b",
            r"\bself[-\s]?hosted\s+(identity|auth|idp|sso|iam)\b",
            r"\binternal\s+(identity|idp|iam|keycloak)\b",
            r"\bno\s+third[-\s]?party\s+(idp|sso|identity|auth)\b",
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
        "id": "07_offline_client",
        "label": "Offline-capable client, sync on reconnect",
        "correct_patterns": [
            r"\boffline[-\s]?(first|capable|mode|support)\b",
            r"\boffline\b.{0,40}\b(sync|reconcile)\b",
            r"\b(sync|reconcile)\b.{0,40}\b(reconnect|online|reconnection)\b",
            r"\blocal\s+(store|database|cache|sqlite)\b.{0,60}\b(offline|sync|reconcile)\b",
            r"\bworks?\s+offline\b",
        ],
        "violation_patterns": [
            r"\balways[-\s]?online\s+client\b",
            r"\bclient.{0,40}\brequires?\s+(connection|network|internet|server)\b",
            r"\breal[-\s]?time\s+connection.{0,40}\b(required|mandatory|must)\b",
            r"\b(websocket|long[-\s]?poll).{0,60}\b(during\s+field|throughout\s+operation)\b",
        ],
    },
    {
        "id": "08_artifactory",
        "label": "Artifactory mirror only, no Docker Hub",
        "correct_patterns": [
            r"\bartifactory\b",
            r"\binternal\s+(mirror|registry|image\s+registry)\b",
            r"\bno\s+(public\s+)?docker\s*hub\b",
            r"\bdocker\s*hub\b.{0,40}\b(block|blocked|not\s+accessible|not\s+available|prohibited)\b",
            r"\b(block|blocked|not\s+accessible)\b.{0,30}\bdocker\s*hub\b",
            r"\bartifactory\.corp\.internal\b",
        ],
        "violation_patterns": [
            r"\bfrom\s+python:\d",
            r"\bfrom\s+postgres:\d",
            r"\bfrom\s+node:\d",
            r"\bfrom\s+alpine\b",
            r"\b(pull|use)\s+from\s+docker\s*hub\b",
            r"\bdocker\s*hub\s+(image|registry)\b",
        ],
    },
    {
        "id": "09_vault_secrets",
        "label": "Vault for secrets, no env vars",
        "correct_patterns": [
            r"\b(hashi(corp)?\s*)?vault\b",
            r"\bvault\.corp\.internal\b",
            r"\bvault\s+agent\b",
            r"\bno\s+(secrets\s+in\s+)?(env(ironment)?\s*variables?|envvar)\b",
            r"\b(env(ironment)?\s*variables?|envvar).{0,30}\b(not\s+used|excluded|prohibited|forbidden)\b",
        ],
        "violation_patterns": [
            r"\bsecrets?\s+(in|via|through)\s+env(ironment)?\s*variables?\b",
            r"\b12[-\s]?factor\b.{0,40}\b(env|environment|secrets?)\b",
            r"\bkubernetes\s+secrets?\b.{0,40}\b(directly|used|for\s+secrets)\b",
            r"\bk8s\s+secrets?\s+(directly|for\s+secrets)\b",
            r"\bput\s+secrets?\s+in\s+(config|env|files?)\b",
        ],
    },
    {
        "id": "10_postgres_queue",
        "label": "Postgres for queueing (no Redis/RabbitMQ/SQS/NATS/Kafka)",
        "correct_patterns": [
            r"\bpostgres(ql)?\b.{0,50}\b(queue|queueing|queuing|jobs?)\b",
            r"\b(listen[/\s]notify|skip\s+locked|for\s+update\s+skip\s+locked)\b",
            r"\bjobs?\s+table\b.{0,60}\bpostgres(ql)?",
            r"\bpostgres(ql)?[-\s]?based\s+queue\b",
        ],
        "violation_patterns": [
            r"\bredis\b(?!.{0,10}(avoid|not\s+use|excluded|prohibited|banned|no\s+redis))",
            r"\brabbit\s*mq\b",
            r"\b(amazon\s+)?sqs\b",
            r"\bnats\b",
            r"\bkafka\b",
            r"\bbullmq\b",
            r"\b(celery|rq|sidekiq)\b.{0,30}\b(redis|rabbitmq|broker)\b",
            r"\bmessage\s+broker\b",
        ],
    },
    {
        "id": "11_idempotent_jobs",
        "label": "Idempotent, at-least-once jobs",
        "correct_patterns": [
            r"\bat[-\s]?least[-\s]?once\b",
            r"\bidempoten(t|ce|cy)\b",
            r"\bidempotence\s+key",
            r"\bidempoten(t|ce|cy)\b.{0,50}\b(key|design|delivery|semantics)\b",
        ],
        "violation_patterns": [
            r"\bexactly[-\s]?once\s+(delivery|semantics|processing)\b",
            r"\bat[-\s]?most[-\s]?once\s+(delivery|semantics)\b",
        ],
    },
    {
        "id": "12_minio_sse",
        "label": "MinIO / S3-compatible with AES-256 SSE",
        "correct_patterns": [
            r"\bminio\b",
            r"\bs3[-\s]?compatible\b",
            r"\bobjects\.corp\.internal\b",
            r"\b(sse[-\s]?s3|server[-\s]?side\s+encryption|aes[-\s]?256)\b.{0,40}\b(object|bucket|s3|minio)\b",
            r"\b(object|bucket|s3|minio)\b.{0,40}\b(sse[-\s]?s3|server[-\s]?side\s+encryption|aes[-\s]?256)\b",
        ],
        "violation_patterns": [
            r"\bstore\s+(photos?|media|attachments?|files?)\s+(on|to)\s+(disk|local\s+(fs|filesystem)|pod\s+filesystem)\b",
            r"\bbytea\b.{0,30}\b(photo|media|attachment|image)\b",
            r"\b(photo|media|attachment|image).{0,30}\bbytea\b",
            r"\b(aws|google\s+cloud|gcs|azure\s+blob)\s+(storage|object)\b",
        ],
    },
    {
        "id": "13_audit_format_v3",
        "label": "audit-format-v3 (don't invent)",
        "correct_patterns": [
            r"\baudit[-\s]?format[-\s]?v?3\b",
            r"\bcanonical\s+audit\s+format\b",
            r"\baudit\s+format\b.{0,40}\balready\s+defined\b",
            r"\bexisting\s+audit\s+format\b",
        ],
        "violation_patterns": [
            r"\bcustom\s+audit\s+(format|schema)\b",
            r"\bnew\s+audit\s+(format|schema)\b",
            r"\bbespoke\s+audit\b",
            r"\bdesign\s+(our|an|a)\s+audit\s+(format|schema)\b",
        ],
    },
    {
        "id": "14_company_logging",
        "label": "company-common-logging library",
        "correct_patterns": [
            r"\bcompany[-\s]?common[-\s]?logging\b",
            r"\bcompany\s+logging\s+library\b",
            r"\binternal\s+logging\s+library\b",
            r"\bshared\s+logging\s+library\b",
        ],
        "violation_patterns": [
            r"\bstdlib\s+logging\b",
            r"\bpython[\s']?s?\s+(built[-\s]?in|standard)\s+logging\b",
            r"\bloguru\b",
            r"\bstructlog\b",
        ],
    },
    {
        "id": "15_utc_local_display",
        "label": "UTC in storage, local TZ for display",
        "correct_patterns": [
            r"\butc\b.{0,40}\b(storage|stored|database|db|persist|timestamps?)\b",
            r"\b(storage|stored|database|db|persist|timestamps?)\b.{0,40}\butc\b",
            r"\butc[-\s]?in[-\s]?storage\b",
            r"\blocal\s+(tz|time\s*zone)\b.{0,40}\b(display|render|present|client)\b",
        ],
        "violation_patterns": [
            r"\bserver\s+(local\s+)?(tz|time\s*zone)\b.{0,40}\bstored?\b",
            r"\bstore.{0,20}\b(laptop|client)\s+(local\s+)?(tz|time)\b",
        ],
    },
]


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
    "rather than", "instead of", "no longer",
    # Historical/reference markers — match in these contexts is not a proposal.
    "after the", "incident", "outage", "failover", "regression",
    "legacy", "historically", "used to", "had been",
    "migrating away", "migration away",
    # Action-is-blocked synonyms.
    "block", "blocks", "blocked", "blocking",
    "rejected", "dropped at", "denied",
    # "X wraps Y" — Y is mentioned descriptively, not proposed.
    "wraps", "wrapping", "wraps around", "built on top of", "built atop",
    "is a wrapper", "is a thin layer", "sits on top of",
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


def classify_constraint(proposal_lower: str, constraint: dict) -> str:
    for pat in constraint["violation_patterns"]:
        for m in re.finditer(pat, proposal_lower, re.IGNORECASE):
            s = max(0, m.start() - 80)
            e = min(len(proposal_lower), m.end() + 80)
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

    print(f"# Phase 3B — results\n")
    print(f"runs: {len(results)} / 15 constraints each\n")

    print("## Per-condition means\n")
    print("| condition | n | correct/15 | silent/15 | violated/15 | accuracy | halluc | length |")
    print("|-----------|---|-----------|-----------|-------------|----------|--------|--------|")
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

    print("\n## Per-constraint CORRECT rate (count / n per condition)\n")
    header = "| constraint | " + " | ".join(sorted(by_cond)) + " |"
    print(header)
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
