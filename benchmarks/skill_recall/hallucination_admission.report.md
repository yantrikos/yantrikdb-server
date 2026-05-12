# Hallucination Admission Benchmark

**Date**: 2026-05-11  

**Corpus**: 90 skills (20 well-formed controls + 70 adversarial)  

**Substrates compared**: YantrikDB server-side schema validation vs naive SKILL.md filesystem (YAML-parseable accept).  


## Methodology

We construct a deterministic adversarial corpus of malformed skills mirroring the kinds of output an LLM might emit when authoring skills at runtime via `POST /v1/skills/define`. Six violation classes (skill_id shape, applies_to shape, skill_type enum, body length, missing required fields, type errors) totaling 70 adversarial entries. 20 well-formed controls confirm base-rate admission.

YantrikDB applies server-side regex/length/enum validation at write time (mirrored exactly in this script — see source for the validation rules; identical to the production wrapper). The naive filesystem baseline accepts any YAML-parseable input — file content can be malformed semantically as long as YAML syntax parses.

## Per-category admission rates

| Category | N | YantrikDB admitted | Filesystem admitted |
|---|---:|---:|---:|
| `applies_to_empty` | 2 | 0 (0%) | 2 (100%) |
| `applies_to_hyphen` | 5 | 0 (0%) | 5 (100%) |
| `applies_to_starts_digit` | 5 | 0 (0%) | 5 (100%) |
| `applies_to_too_many` | 5 | 0 (0%) | 5 (100%) |
| `applies_to_uppercase` | 5 | 0 (0%) | 5 (100%) |
| `body_too_long` | 5 | 0 (0%) | 5 (100%) |
| `body_too_short` | 5 | 0 (0%) | 5 (100%) |
| `control_good` | 20 | 20 (100%) | 20 (100%) |
| `missing_body` | 2 | 0 (0%) | 2 (100%) |
| `missing_skill_id` | 2 | 0 (0%) | 0 (0%) |
| `missing_skill_type` | 2 | 0 (0%) | 2 (100%) |
| `skill_id_hyphen` | 5 | 0 (0%) | 5 (100%) |
| `skill_id_no_dots` | 5 | 0 (0%) | 5 (100%) |
| `skill_id_not_string` | 2 | 0 (0%) | 2 (100%) |
| `skill_id_starts_digit` | 5 | 0 (0%) | 5 (100%) |
| `skill_id_uppercase` | 5 | 0 (0%) | 5 (100%) |
| `skill_type_case_or_bad` | 5 | 0 (0%) | 5 (100%) |
| `skill_type_not_in_enum` | 5 | 0 (0%) | 5 (100%) |

## Headline result

Well-formed controls: both substrates admit 20/20 (YantrikDB) and 20/20 (filesystem). Base-rate admission is identical for valid input.

**Adversarial entries: YantrikDB admits 0/70 = 0.0%. Filesystem admits 68/70 = 97.1%.**

## Interpretation

YantrikDB's write-time schema validation catches **70/70 (100%) of malformed skills before they enter the substrate**. The naive filesystem baseline admits **97% of the same malformed skills** — they enter the catalog as valid YAML files, become discoverable via retrieval, and fail only at agent-invocation time (or silently produce wrong behavior).

This addresses the **hallucination admission failure mode** for autonomous learning at scale: agents authoring skills at runtime via `POST /v1/skills/define` cannot poison the substrate with shape-violating definitions — the server rejects them at the API boundary. Semantic correctness of skill content remains the agent layer's responsibility; what the substrate guarantees is structural well-formedness.

## Limitations

- The naive filesystem baseline is intentionally weak — it models 'YAML parser only' validation. A SKILL.md system that adds CI-time validation or a write-side validator would reduce its admission rate. The point is that this validator must be built separately; YantrikDB ships it as a substrate property.
- The adversarial corpus is enumerative across shape violation classes; semantic-content adversaries (skill that is well-formed shape-wise but instructs a dangerous action) are NOT caught by either substrate. This is the scope-boundary of the schema-not-semantics design line.
- We do not measure agent behavior after admission; this is a substrate-level admission test, not an end-to-end safety claim.