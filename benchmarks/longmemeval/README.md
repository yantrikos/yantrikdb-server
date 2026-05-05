# LongMemEval Benchmark for YantrikDB

End-to-end retrieval-quality benchmark on the **LongMemEval** corpus
(Wu et al., 2024) — the field-standard agent-memory benchmark used by
Zep, Memento, Mastra, MemGPT and others.

## Dataset

500 questions across 7 question types: single-session-user,
single-session-assistant, single-session-preference, temporal-reasoning,
knowledge-update, multi-session, abstention. Three configurations:

| Variant | Size | Sessions per question | Use |
|---|---|---|---|
| `oracle` | 15 MB | only evidence sessions | harness validation, tiny |
| `s` | 278 MB | ~40 sessions (~115k tokens) | headline benchmark |
| `m` | 2.75 GB | ~500 sessions | scale stress |

Source: <https://github.com/xiaowu0162/LongMemEval> · License: MIT.

## What this benchmark measures

For each question:

1. Boot a fresh per-question YantrikDB namespace (`lme_<question_id>`)
2. Write every turn from every haystack session as a memory, tagged
   with `(session_id, turn_idx, has_answer)` in metadata
3. Issue the question's `question` text against `/v1/recall` with a
   range of top-K values (1, 5, 10, 20)
4. Score:
   - **Turn-level recall@K**: did top-K results include at least one
     turn where `has_answer=true`?
   - **Session-level recall@K**: did top-K results include at least
     one turn from `answer_session_ids`?

QA-correctness scoring (LongMemEval's GPT-4o judge layer) is **not**
included in this harness. That's a separable concern that needs an
OpenAI API key and adds ~$5-15 per full run; retrieval recall is what
yantrikdb's substrate provides, and it's the metric the indexing
roadmap moves.

## Methodology choices (and why)

- **Per-question namespace.** LongMemEval is structured as 500
  independent retrieval tasks; sessions don't bleed across questions.
  Per-question namespaces keep the recall test cleanly scoped and
  match how production agents partition memory.
- **Embedder = builtin MiniLM-L6-v2 (384-dim).** Same as the
  skill_recall benchmark; comparable across YantrikDB benchmarks.
- **`expand_entities=false`.** LongMemEval evaluates pure retrieval,
  not entity-graph augmentation. Setting `false` matches the
  comparison shape used by Zep/Memento (none ship with entity
  expansion enabled by default in their public benchmark numbers).
- **Single-node, no replication.** Matches LongMemEval's published
  runs and the skill_recall benchmark.

## Running

```bash
# 1. Boot YantrikDB (same single-node config as skill_recall).
cd benchmarks/longmemeval
docker compose up -d

# 2. Mint a token (Windows / Git Bash users: prefix with MSYS_NO_PATHCONV=1).
docker exec yantrikdb-lme yantrikdb db --data-dir /var/lib/yantrikdb create default
docker exec yantrikdb-lme yantrikdb token --data-dir /var/lib/yantrikdb create --db default --label bench

export YDB_TOKEN=<token>

# 3. Pull the oracle variant (15 MB, fastest harness validation).
python fetch.py --variant oracle

# 4. Run the harness end-to-end.
python run.py --variant oracle --port 7438 --top-k 10 --limit 50

# 5. Aggregate metrics.
python metrics.py results_oracle.jsonl
```

After the harness shape is validated on `oracle`, escalate to `_s`:

```bash
python fetch.py --variant s
python run.py --variant s --port 7438 --top-k 10
python metrics.py results_s.jsonl
```

## What to expect

`oracle` runs in minutes (only evidence sessions). `_s` runs
in tens of minutes to a few hours depending on hardware (5000+ total
sessions × tens of turns each, all written through the embedder).

## Comparison

LongMemEval published numbers for competitor systems are referenced
in the original paper; we record yantrikdb's numbers per question type
and link to the paper Table for head-to-head context.
