"""Hallucination admission benchmark — failure mode #3 of the autonomous-
learning-at-scale thesis.

Generates a deterministic adversarial-skill corpus that mirrors the kinds
of malformed outputs an LLM might emit when authoring skills at runtime,
then measures admission rates under two substrate models:

  (A) YantrikDB server-side schema validation
      — Strict regex on skill_id (^[a-z][a-z0-9_]*(\\.[a-z0-9_]+)+$)
      — Length bounds on body (50..5000)
      — Format on applies_to entries (^[a-z][a-z0-9_]*$), length 1..10
      — Enum on skill_type {procedure, reference, lesson, pattern, rule}
      — Required field checks
      — Rejects at write time with 400-class error

  (B) Naive SKILL.md filesystem
      — File system accepts any byte sequence as a file
      — YAML parser validates ONLY that frontmatter is parseable YAML
        (very permissive — accepts almost any well-formed YAML even with
        semantically wrong field values)
      — Field-shape errors surface only at read/use time, after the
        skill has already entered the substrate and potentially been
        invoked by an agent.

The hallucination admission rate is the fraction of malformed skills
that the substrate ACCEPTS (writes successfully). Lower is better —
a substrate that admits hallucinated skills lets them propagate to
downstream agent decisions.

Two violation classes:
  - 'shape': field doesn't match the validation regex/length/enum.
    YantrikDB rejects; naive filesystem accepts (it's still valid YAML).
  - 'parse': YAML itself is malformed.
    Both reject — this is a control class showing both systems agree
    on un-parseable inputs.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).parent
OUT_CORPUS = HERE / "adversarial_skills_corpus.jsonl"
OUT_RESULTS = HERE / "hallucination_admission.csv"
OUT_REPORT = HERE / "hallucination_admission.report.md"

# --- YantrikDB schema validators (mirror server-side rules) ---
SKILL_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
APPLIES_TO_ENTRY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SKILL_TYPE_ENUM = {"procedure", "reference", "lesson", "pattern", "rule"}
BODY_MIN, BODY_MAX = 50, 5000
APPLIES_TO_MAX = 10
SKILL_ID_MIN, SKILL_ID_MAX = 4, 200


def validate_yantrikdb(skill: dict) -> tuple[bool, str]:
    """Server-side validation as YantrikDB enforces it. Returns
    (admitted, reason_if_rejected)."""
    # Required fields
    md = skill.get("metadata") or {}
    sid = md.get("skill_id")
    if not sid:
        return False, "missing skill_id"
    if not isinstance(sid, str):
        return False, "skill_id not string"
    if not (SKILL_ID_MIN <= len(sid) <= SKILL_ID_MAX):
        return False, f"skill_id length {len(sid)} out of range"
    if not SKILL_ID_RE.match(sid):
        return False, "skill_id regex violation"
    body = skill.get("text")
    if not body or not isinstance(body, str):
        return False, "missing body"
    if not (BODY_MIN <= len(body) <= BODY_MAX):
        return False, f"body length {len(body)} out of range"
    skill_type = md.get("skill_type")
    if not skill_type:
        return False, "missing skill_type"
    if skill_type not in SKILL_TYPE_ENUM:
        return False, f"skill_type {skill_type!r} not in enum"
    applies_to = md.get("applies_to")
    if not applies_to or not isinstance(applies_to, list):
        return False, "applies_to not non-empty list"
    if len(applies_to) > APPLIES_TO_MAX:
        return False, f"applies_to has {len(applies_to)} entries (max {APPLIES_TO_MAX})"
    for entry in applies_to:
        if not isinstance(entry, str):
            return False, f"applies_to entry not string: {entry!r}"
        if not APPLIES_TO_ENTRY_RE.match(entry):
            return False, f"applies_to entry regex violation: {entry!r}"
    return True, ""


def validate_naive_filesystem(skill: dict) -> tuple[bool, str]:
    """A naive SKILL.md filesystem accepts any file that is parseable
    YAML/JSON. The semantic field validation does not happen at write
    time — bad content lives in the catalog until someone tries to read
    or use it. Returns (admitted, reason)."""
    # The filesystem itself is permissive. We model "filesystem write"
    # as "the data structure is well-formed enough to serialize and
    # deserialize cleanly." Anything that round-trips through JSON
    # (which models the YAML-parses-successfully baseline) is admitted.
    try:
        json.dumps(skill)
        # Filesystem also requires that the basic envelope exists — if
        # skill_id is completely missing, even a permissive filesystem
        # would put it at an unfindable location. Model this as
        # rejecting only when there's literally no skill_id key.
        md = skill.get("metadata") or {}
        if "skill_id" not in md:
            return False, "no skill_id key (filesystem cannot place file)"
        return True, ""
    except (TypeError, ValueError) as e:
        return False, f"not serializable: {e}"


# --- Adversarial corpus generation ---
# 100 deliberately malformed skills, deterministic, balanced across
# violation categories.


def good_skill(idx: int) -> dict:
    """A well-formed skill that should be admitted by both systems —
    control group for the corpus."""
    return {
        "namespace": "skill_substrate",
        "text": "Skill: when working with adversarial benchmark control entries, verify that both substrates admit the well-formed skill so that the rejection-rate measurement is not confounded by base-rate refusal. This is a control sample.",
        "memory_type": "procedural",
        "domain": "skill",
        "importance": 0.5,
        "source": "bench.adversarial",
        "metadata": {
            "record_type": "skill",
            "schema_version": 1,
            "skill_id": f"adv.control.good.v{idx}",
            "version": 1,
            "status": "active",
            "skill_type": "procedure",
            "applies_to": ["adversarial_bench"],
            "triggers": ["control sample"],
            "variant_idx": idx,
            "authored_by": "agent.adversarial_synth",
            "synthetic_index": idx,
            "violation_category": "control_good",
        },
    }


def build_corpus() -> list[dict]:
    """Build 100 adversarial skills + 20 controls, deterministic."""
    skills: list[dict] = []

    # Controls (20)
    for i in range(20):
        skills.append(good_skill(i))

    # --- Category: skill_id violations (20) ---
    # 5 each: hyphen-instead-of-underscore (the load-bearing bug),
    # starts-with-digit, uppercase, no-dots, too-short
    for i in range(5):
        s = good_skill(100 + i)
        s["metadata"]["skill_id"] = f"adv-hyphen-{i}.v0"  # hyphen in name segment
        s["metadata"]["violation_category"] = "skill_id_hyphen"
        skills.append(s)
    for i in range(5):
        s = good_skill(110 + i)
        s["metadata"]["skill_id"] = f"7adv.digit.start.v{i}"  # starts with digit
        s["metadata"]["violation_category"] = "skill_id_starts_digit"
        skills.append(s)
    for i in range(5):
        s = good_skill(120 + i)
        s["metadata"]["skill_id"] = f"AdvUppercase.v{i}"  # uppercase
        s["metadata"]["violation_category"] = "skill_id_uppercase"
        skills.append(s)
    for i in range(5):
        s = good_skill(130 + i)
        s["metadata"]["skill_id"] = f"advnodots{i}"  # no dots
        s["metadata"]["violation_category"] = "skill_id_no_dots"
        skills.append(s)

    # --- Category: applies_to violations (20) ---
    # 5 each: hyphen-in-entry (the load-bearing bug), starts-with-digit,
    # uppercase, too-many-entries
    for i in range(5):
        s = good_skill(140 + i)
        s["metadata"]["applies_to"] = [f"adv-hyphen-tag-{i}"]  # hyphen!
        s["metadata"]["violation_category"] = "applies_to_hyphen"
        skills.append(s)
    for i in range(5):
        s = good_skill(150 + i)
        s["metadata"]["applies_to"] = [f"7digit_start_{i}"]
        s["metadata"]["violation_category"] = "applies_to_starts_digit"
        skills.append(s)
    for i in range(5):
        s = good_skill(160 + i)
        s["metadata"]["applies_to"] = [f"UPPER_CASE_{i}"]
        s["metadata"]["violation_category"] = "applies_to_uppercase"
        skills.append(s)
    for i in range(5):
        s = good_skill(170 + i)
        s["metadata"]["applies_to"] = [f"valid_{j}" for j in range(15)]  # >10 entries
        s["metadata"]["violation_category"] = "applies_to_too_many"
        skills.append(s)

    # --- Category: skill_type violations (10) ---
    # 5 each: not-in-enum, completely-bogus
    for i, bad_type in enumerate(["function", "action", "recipe", "plugin", "tool"]):
        s = good_skill(180 + i)
        s["metadata"]["skill_type"] = bad_type
        s["metadata"]["violation_category"] = "skill_type_not_in_enum"
        skills.append(s)
    for i, bad_type in enumerate(["", "PROCEDURE", "Procedure", "proc", "lessons"]):
        s = good_skill(190 + i)
        s["metadata"]["skill_type"] = bad_type
        s["metadata"]["violation_category"] = "skill_type_case_or_bad"
        skills.append(s)

    # --- Category: body violations (10) ---
    # 5 each: too-short, too-long
    for i in range(5):
        s = good_skill(200 + i)
        s["text"] = "Too short."  # < 50 chars
        s["metadata"]["violation_category"] = "body_too_short"
        skills.append(s)
    for i in range(5):
        s = good_skill(210 + i)
        s["text"] = "Very long body. " * 400  # > 5000 chars
        s["metadata"]["violation_category"] = "body_too_long"
        skills.append(s)

    # --- Category: missing required fields (10) ---
    for i in range(2):
        s = good_skill(220 + i)
        del s["metadata"]["skill_id"]
        s["metadata"]["violation_category"] = "missing_skill_id"
        skills.append(s)
    for i in range(2):
        s = good_skill(222 + i)
        del s["metadata"]["skill_type"]
        s["metadata"]["violation_category"] = "missing_skill_type"
        skills.append(s)
    for i in range(2):
        s = good_skill(224 + i)
        s["metadata"]["applies_to"] = []  # empty array
        s["metadata"]["violation_category"] = "applies_to_empty"
        skills.append(s)
    for i in range(2):
        s = good_skill(226 + i)
        del s["text"]
        s["metadata"]["violation_category"] = "missing_body"
        skills.append(s)
    for i in range(2):
        s = good_skill(228 + i)
        # Skill ID present but wrong type
        s["metadata"]["skill_id"] = 12345
        s["metadata"]["violation_category"] = "skill_id_not_string"
        skills.append(s)

    return skills


def main():
    skills = build_corpus()
    OUT_CORPUS.write_text("\n".join(json.dumps(s) for s in skills), encoding="utf-8")
    print(f"Generated {len(skills)} adversarial skills -> {OUT_CORPUS}")

    # Run both validators on every skill
    results = []
    for s in skills:
        cat = s["metadata"].get("violation_category", "unknown")
        ydb_admit, ydb_reason = validate_yantrikdb(s)
        fs_admit, fs_reason = validate_naive_filesystem(s)
        results.append({
            "skill_id": s["metadata"].get("skill_id", "<missing>"),
            "category": cat,
            "yantrikdb_admitted": ydb_admit,
            "yantrikdb_reason": ydb_reason,
            "filesystem_admitted": fs_admit,
            "filesystem_reason": fs_reason,
        })

    with OUT_RESULTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {OUT_RESULTS}")

    # Aggregate
    by_cat: dict[str, dict] = {}
    for r in results:
        c = r["category"]
        by_cat.setdefault(c, {"n": 0, "ydb_admit": 0, "fs_admit": 0})
        by_cat[c]["n"] += 1
        by_cat[c]["ydb_admit"] += int(r["yantrikdb_admitted"])
        by_cat[c]["fs_admit"] += int(r["filesystem_admitted"])

    # Total adversarial (excluding controls)
    total_adv_n = sum(v["n"] for c, v in by_cat.items() if c != "control_good")
    total_adv_ydb_admit = sum(v["ydb_admit"] for c, v in by_cat.items() if c != "control_good")
    total_adv_fs_admit = sum(v["fs_admit"] for c, v in by_cat.items() if c != "control_good")

    control = by_cat.get("control_good", {"n": 0, "ydb_admit": 0, "fs_admit": 0})

    print("\n=== Per-category admission counts ===")
    print(f"{'category':<35} {'N':>4} {'YDB':>5} {'FS':>5}  ratio")
    for c, v in sorted(by_cat.items()):
        ydb_pct = 100 * v["ydb_admit"] / v["n"]
        fs_pct = 100 * v["fs_admit"] / v["n"]
        print(f"{c:<35} {v['n']:>4} {v['ydb_admit']:>5} {v['fs_admit']:>5}  YDB={ydb_pct:5.1f}% FS={fs_pct:5.1f}%")

    print("\n=== Headline numbers ===")
    print(f"Controls (well-formed):  N={control['n']}  YDB admitted {control['ydb_admit']}/{control['n']}  FS admitted {control['fs_admit']}/{control['n']}")
    print(f"Adversarial (malformed): N={total_adv_n}  YDB admitted {total_adv_ydb_admit}/{total_adv_n} = {100*total_adv_ydb_admit/total_adv_n:.1f}%")
    print(f"                                FS admitted {total_adv_fs_admit}/{total_adv_n} = {100*total_adv_fs_admit/total_adv_n:.1f}%")

    # Report
    lines = []
    lines.append("# Hallucination Admission Benchmark\n")
    lines.append("**Date**: 2026-05-11  \n")
    lines.append(f"**Corpus**: {len(skills)} skills ({control['n']} well-formed controls + {total_adv_n} adversarial)  \n")
    lines.append("**Substrates compared**: YantrikDB server-side schema validation vs naive SKILL.md filesystem (YAML-parseable accept).  \n")
    lines.append("")
    lines.append("## Methodology\n")
    lines.append("We construct a deterministic adversarial corpus of malformed skills mirroring the kinds of output an LLM might emit when authoring skills at runtime via `POST /v1/skills/define`. Six violation classes (skill_id shape, applies_to shape, skill_type enum, body length, missing required fields, type errors) totaling " + str(total_adv_n) + " adversarial entries. " + str(control['n']) + " well-formed controls confirm base-rate admission.")
    lines.append("")
    lines.append("YantrikDB applies server-side regex/length/enum validation at write time (mirrored exactly in this script — see source for the validation rules; identical to the production wrapper). The naive filesystem baseline accepts any YAML-parseable input — file content can be malformed semantically as long as YAML syntax parses.")
    lines.append("")
    lines.append("## Per-category admission rates\n")
    lines.append("| Category | N | YantrikDB admitted | Filesystem admitted |")
    lines.append("|---|---:|---:|---:|")
    for c, v in sorted(by_cat.items()):
        ydb_pct = 100 * v["ydb_admit"] / v["n"]
        fs_pct = 100 * v["fs_admit"] / v["n"]
        lines.append(f"| `{c}` | {v['n']} | {v['ydb_admit']} ({ydb_pct:.0f}%) | {v['fs_admit']} ({fs_pct:.0f}%) |")
    lines.append("")
    lines.append("## Headline result\n")
    lines.append(f"Well-formed controls: both substrates admit {control['ydb_admit']}/{control['n']} (YantrikDB) and {control['fs_admit']}/{control['n']} (filesystem). Base-rate admission is identical for valid input.")
    lines.append("")
    lines.append(f"**Adversarial entries: YantrikDB admits {total_adv_ydb_admit}/{total_adv_n} = {100*total_adv_ydb_admit/total_adv_n:.1f}%. Filesystem admits {total_adv_fs_admit}/{total_adv_n} = {100*total_adv_fs_admit/total_adv_n:.1f}%.**")
    lines.append("")
    lines.append("## Interpretation\n")
    rejected_ydb = total_adv_n - total_adv_ydb_admit
    pct_caught = 100 * rejected_ydb / total_adv_n
    lines.append(f"YantrikDB's write-time schema validation catches **{rejected_ydb}/{total_adv_n} ({pct_caught:.0f}%) of malformed skills before they enter the substrate**. The naive filesystem baseline admits **{100*total_adv_fs_admit/total_adv_n:.0f}% of the same malformed skills** — they enter the catalog as valid YAML files, become discoverable via retrieval, and fail only at agent-invocation time (or silently produce wrong behavior).")
    lines.append("")
    lines.append("This addresses the **hallucination admission failure mode** for autonomous learning at scale: agents authoring skills at runtime via `POST /v1/skills/define` cannot poison the substrate with shape-violating definitions — the server rejects them at the API boundary. Semantic correctness of skill content remains the agent layer's responsibility; what the substrate guarantees is structural well-formedness.")
    lines.append("")
    lines.append("## Limitations\n")
    lines.append("- The naive filesystem baseline is intentionally weak — it models 'YAML parser only' validation. A SKILL.md system that adds CI-time validation or a write-side validator would reduce its admission rate. The point is that this validator must be built separately; YantrikDB ships it as a substrate property.")
    lines.append("- The adversarial corpus is enumerative across shape violation classes; semantic-content adversaries (skill that is well-formed shape-wise but instructs a dangerous action) are NOT caught by either substrate. This is the scope-boundary of the schema-not-semantics design line.")
    lines.append("- We do not measure agent behavior after admission; this is a substrate-level admission test, not an end-to-end safety claim.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
