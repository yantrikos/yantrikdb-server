# Skill as Memory, Not Document — Zenodo Deposit Bundle

**Paper**: *Skill as Memory, Not Document: A Database-Native Substrate for Agent Skill Catalogs*
**Author**: Pranab Sarkar, Independent Researcher (ORCID 0009-0009-8683-1481)
**License**: CC BY 4.0
**Date**: 2026-05-11
**DOI**: [10.5281/zenodo.20128887](https://doi.org/10.5281/zenodo.20128887)
**Related system DOI**: YantrikDB software — https://doi.org/10.5281/zenodo.18793952

This is the complete reproducible artifact bundle for the paper. Upload the entire directory contents to Zenodo as a new deposit; Zenodo will assign a DOI specific to this paper.

## Files

| File | Purpose |
|---|---|
| `skill-substrate-experience-report-v4-FINAL.pdf` | Final paper PDF (248 KB, 16-18 pages, table of contents, embedded figure) — primary artifact |
| `skill-substrate-experience-report-v4-source.md` | Markdown source for the paper (47 KB) — editability + machine-readable record |
| `figure-token-cost.png` | Figure 1 (Section 5): per-query top-K=5 prompt-token cost vs catalog size, log-log axes |
| `token_cost_bench.py` | Section 5 disclosure-pattern sweep benchmark — generates `token_cost.csv` |
| `kill_test_ablation.py` | Section 5.3 body-only ablation — generates `kill_test_ablation.report.md` |
| `k_sensitivity_and_tokenizer_bench.py` | Section 5.4–5.5 K-sweep + multi-tokenizer benchmark — generates `k_sensitivity.csv` + `tokenizer_validation.csv` |
| `hallucination_admission_bench.py` | Section 7 invalid-skill admission benchmark on 90-skill adversarial corpus — generates `hallucination_admission.csv` |
| `token_cost.csv` | Raw output: Section 5.2 table data |
| `k_sensitivity.csv` | Raw output: Section 5.4 table data |
| `tokenizer_validation.csv` | Raw output: Section 5.5 table data |
| `hallucination_admission.csv` | Raw output: Section 7 per-skill admission results |

## Reproducibility

All benchmarks are deterministic (seed=42 for the corpus generator, fixed query set). To reproduce:

```bash
# Clone the YantrikDB server repository
git clone https://github.com/yantrikos/yantrikdb-server.git
cd yantrikdb-server/benchmarks/skill_recall

# Install dependencies
pip install tiktoken matplotlib

# Re-run all benchmarks
python token_cost_bench.py
python kill_test_ablation.py
python k_sensitivity_and_tokenizer_bench.py
python hallucination_admission_bench.py
```

The 5,000-skill corpus (`skills_corpus.jsonl`) and 100 ground-truth queries (`queries_groundtruth.jsonl`) live in the same directory in the public repository.

## Section 6 latency baseline

The Section 6 retrieval-latency measurements (p50 = 87.3 ms, p95 = 106.3 ms at 5K-skill scale) were collected at YantrikDB engine commit `c886e9e` (2026-04-29) on a Windows 11 + Docker Desktop deployment with the bundled MiniLM-L6-v2 embedder. Methodology and raw results are in `benchmarks/skill_recall/RESULTS.md` in the public repository.

## Companion paper (forthcoming)

A follow-up paper is pre-specified for delivery by 2026-08-04 covering:
- Postgres+pgvector+JSON-schema+audit-table boring baseline (the most important deferred comparison)
- End-to-end agent task success
- Adversarial near-duplicate retrieval corpus
- Real-world skill subset
- Multi-tenant contention
- Threat model and security analysis
- Semantic-violation invalid-skill admission
- Task-success ablation: YAML-frontmatter-intact vs frontmatter-stripped retrieved content

## Suggested Zenodo metadata

When uploading, suggested values:

- **Resource type**: Publication → Preprint
- **Title**: *Skill as Memory, Not Document: A Database-Native Substrate for Agent Skill Catalogs*
- **Authors**: Pranab Sarkar — Independent Researcher — ORCID: 0009-0009-8683-1481
- **Description**: Use the abstract from the paper
- **Keywords**: autonomous agent learning, agent memory, skill management, vector retrieval, database substrate, schema validation, context window, experience report, LLM, YantrikDB
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Related identifiers**:
  - "Is supplement to" → https://doi.org/10.5281/zenodo.18793952 (YantrikDB software)
  - "Cites" → arxiv IDs in the References section
- **Communities**: (leave blank or join relevant CS communities)
- **Funding**: (none — independent research)
