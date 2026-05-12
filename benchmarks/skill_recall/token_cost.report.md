# Token-Cost Benchmark — Skill Discovery at Scale

**Date**: 2026-05-11

**Tokenizer**: OpenAI cl100k_base (GPT-4 family)

**Corpus**: 5000 synthetic skills (seed=42)

**Queries**: 100 ground-truth queries, target_skill_ids[0:5]


## Per-skill token cost (median)

| Representation | Median tokens |
|---|---|
| Full SKILL.md (YAML frontmatter + body) | 183.0 |
| SKILL.md metadata only (frontmatter) | 50.0 |
| YantrikDB body (substrate row text) | 123.0 |

YantrikDB top-5 body cost across 100 queries: mean=369 p95=635


## Catalog-size sweep

| Catalog size N | (A) Full SKILL.md catalog | (B) Metadata + top-K body (Anthropic progressive disclosure) | (C) YantrikDB top-K body | A/C reduction | B/C reduction |
|---|---|---|---|---|---|
| 100 | 18,250 tok | 5,388 tok | 368 tok | 50× | 14.6× |
| 500 | 91,450 tok | 25,268 tok | 368 tok | 248× | 68.5× |
| 1,000 | 181,200 tok | 49,868 tok | 368 tok | 492× | 135.3× |
| 2,500 | 455,150 tok | 124,768 tok | 368 tok | 1235× | 338.5× |
| 5,000 | 919,200 tok | 252,868 tok | 368 tok | 2494× | 686.0× |

## Interpretation

- **(A)** is the naive worst-case filesystem pattern: load every SKILL.md upfront. Doesn't scale past a few hundred skills before exceeding standard context windows.
- **(B)** is the realistic Anthropic Agent Skills pattern: load metadata only (frontmatter), fetch body on demand for matching skills. Still requires linear catalog scan in tokens.
- **(C)** is the YantrikDB substrate pattern: semantic top-K retrieval over the database, no catalog upfront. **Constant** in catalog size — same token cost at 100 skills as at 5,000 skills.

The substrate pattern's cost is **independent of catalog size** because the engine handles the index. The filesystem patterns' costs are **linear in catalog size**.

At 5,000 skills (Lane B's projected scale within ~3 months), the substrate pattern costs ~368 tokens vs ~252,868 for the metadata-pattern — a **686× reduction** in context consumption for skill discovery alone.