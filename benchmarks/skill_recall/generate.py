#!/usr/bin/env python3
"""Generate synthetic skill records + ground-truth queries for the
skill-recall benchmark.

Schema: topic x action x variant
  - 50 topics: broad spread covering agent-system concerns (recipe
    authoring, template resolution, vector recall, leader election,
    model routing, ...).
  - 10 actions per topic (diagnose, mitigate, retry, batch, parallelize,
    dedupe, version, audit, recover, escalate).
  - 10 variants per (topic, action): each with a distinguishing trigger
    phrase + counterexample so sibling skills are confusably-similar
    by design - this stresses the embedder's ability to discriminate.

5000 skills total at full scale. Generation is deterministic (random
seed = 42 in main()) so benchmark runs are reproducible across machines.

Outputs (written to this script's directory):
  skills_corpus.jsonl       - one skill per line, ready for seed.py
  queries_groundtruth.jsonl - 100 ground-truth queries with target_skill_ids

Usage:
  python generate.py 5000   # default if argument omitted
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path


TOPICS = [
    ("recipe_authoring", "writing recipes via the runtime DSL"),
    ("template_resolver", "resolving {{var}} templates in recipe args"),
    ("ydb_recall", "semantic recall against the YDB cluster"),
    ("ydb_remember", "writing memories to the YDB cluster"),
    ("web_search", "searching the web via SearXNG"),
    ("gdelt_search", "fetching news articles from GDELT"),
    ("file_io", "reading and writing files in the workspace"),
    ("ssh_access", "SSH commands to LXC containers"),
    ("telegram_send", "sending messages over Telegram"),
    ("prediction_tracking", "tagging and validating predictions"),
    ("character_simulation", "simulating character pipelines"),
    ("news_pulse", "summarising news firehose for digest"),
    ("cluster_election", "Raft leader election on the YDB cluster"),
    ("leader_failover", "redirecting writes when YDB leader changes"),
    ("embedder_latency", "embedding service latency under load"),
    ("sqlite_locking", "SQLite WAL and BEGIN IMMEDIATE locking"),
    ("json_parsing", "parsing JSON payloads safely"),
    ("regex_pitfalls", "regex backtracking and DoS pitfalls"),
    ("http_timeout", "HTTP timeout handling and retry behaviour"),
    ("retry_backoff", "exponential backoff and jitter strategies"),
    ("memory_correlation", "correlating memories via thread_id and dedupe_key"),
    ("dedupe_keys", "deduplication keys for idempotent writes"),
    ("vector_search", "vector similarity ranking and cutoff"),
    ("namespace_isolation", "per-project namespace isolation in YDB"),
    ("top_k_cutoff", "top_k cutoff bug when corpus grows"),
    ("schema_versioning", "evolving schema_version field on records"),
    ("recipe_invocation", "invoking recipes via recipe_submit"),
    ("comm_substrate", "writing comm.item records to comm_substrate"),
    ("comm_subscription", "registering comm.subscription records"),
    ("event_trigger", "event-driven triggers via subscription matching"),
    ("audit_trail", "audit comm items for provenance chains"),
    ("observability", "structured logs and metrics for substrate"),
    ("lane_b_authoring", "Lane B authoring its own recipes and tools"),
    ("tier_classification", "tier 1/2/3 classification of authored content"),
    ("verifier_gate", "dual-LLM verifier gate for tier 2 authorship"),
    ("ast_validation", "AST whitelist validation for tool source code"),
    ("sandbox_isolation", "subprocess sandbox isolation for tool execution"),
    ("qwen_tool_calls", "Qwen 3.6 native tool-call generation quirks"),
    ("model_routing", "routing model calls to OpenAI/Anthropic/Ollama"),
    ("ollama_5xx_errors", "Ollama 500/502 errors and recovery"),
    ("openai_rate_limit", "OpenAI 429 rate limit and quota management"),
    ("deepseek_response_shape", "DeepSeek API response shape variations"),
    ("telegram_chunking", "Telegram 4096-char message chunking"),
    ("github_search_quota", "GitHub Search API rate quota under unauth"),
    ("arxiv_atom_parser", "parsing arxiv Atom XML responses"),
    ("hn_firebase_api", "HackerNews Firebase API consumption"),
    ("gdelt_artlist_format", "GDELT ArtList JSON structure"),
    ("leader_election_storm", "election thrashing under CPU saturation"),
    ("recall_load_test", "load testing recall under high concurrency"),
    ("skill_consolidation", "consolidating overlapping skills"),
]

ACTIONS = [
    ("diagnose", "diagnose the root cause of"),
    ("mitigate", "mitigate the impact of"),
    ("escalate", "escalate when you encounter"),
    ("retry", "retry safely on transient failures of"),
    ("batch", "batch operations to amortise the cost of"),
    ("parallelize", "parallelise calls to overcome latency in"),
    ("dedupe", "deduplicate to avoid double-processing of"),
    ("version", "version and supersede prior knowledge about"),
    ("audit", "audit the provenance chain involving"),
    ("recover", "recover from interrupted state in"),
]

# Per-variant distinguishing phrases (10 variants per topic+action)
VARIANT_DETAILS = [
    ("symptom is timeout exceeding 20 seconds", "watchdog log shows leader-timeout"),
    ("symptom is empty result list with no error", "filter reduced corpus below cutoff"),
    ("symptom is malformed payload with extra newlines", "encoding mismatch UTF-8 vs UTF-16"),
    ("symptom is duplicate writes despite dedupe_key", "dedupe_key colliding under race"),
    ("symptom is silent truncation at 4000 chars", "single-line break_points heuristic missed"),
    ("symptom is partial response with HTTP 200 but body cut off", "chunked transfer aborted mid-stream"),
    ("symptom is correlation thread_id missing on response", "thread_id not propagated through retry"),
    ("symptom is rate limit 429 from upstream API", "quota exhausted in burst window"),
    ("symptom is schema_version mismatch in metadata", "consumer expecting v1 receives v2 envelope"),
    ("symptom is sub-millisecond response that looks fake", "cached negative result returned without re-query"),
]


def gen_skill_text(topic: str, topic_desc: str, action: str, action_phrase: str,
                    variant_idx: int, detail_a: str, detail_b: str) -> str:
    return (
        f"Skill: when working with {topic_desc}, {action_phrase} the {variant_idx}-th "
        f"failure mode where the {detail_a}. The recognition trigger is that {detail_b}. "
        f"Concrete fix: apply the {action} pattern specific to {topic} variant {variant_idx}. "
        f"Counter-example: do NOT confuse this with variant {(variant_idx + 5) % 10} of "
        f"the same family — those have a different remediation. Example incident: "
        f"{topic}-{action}-{variant_idx} surfaced during a recent run; the engineer "
        f"recognised {detail_a[:50]} and applied {action}-pattern v{variant_idx + 1}."
    )


def gen_skill_record(topic_idx: int, action_idx: int, variant_idx: int,
                     skill_index: int) -> dict:
    topic, topic_desc = TOPICS[topic_idx]
    action, action_phrase = ACTIONS[action_idx]
    detail_a, detail_b = VARIANT_DETAILS[variant_idx]
    skill_id = f"skill.{topic}.{action}.v{variant_idx}"
    text = gen_skill_text(topic, topic_desc, action, action_phrase,
                          variant_idx, detail_a, detail_b)
    metadata = {
        "record_type": "skill",
        "schema_version": 1,
        "skill_id": skill_id,
        "version": 1,
        "status": "active",
        "skill_type": "lesson",
        "applies_to": [topic],
        "triggers": [detail_a, detail_b],
        "variant_idx": variant_idx,
        "topic_family": topic,
        "action_family": action,
        "success_count": 0,
        "failure_count": 0,
        "authored_by": "agent.bench_synth",
        "synthetic_index": skill_index,
    }
    return {
        "namespace": "skill_test_substrate",
        "text": text,
        "memory_type": "procedural",
        "domain": "skill",
        "importance": 0.55,
        "source": "bench.skill_synth",
        "metadata": metadata,
    }


def gen_query_for_family(topic_idx: int, action_idx: int) -> dict:
    """Two query forms per family — one paraphrased, one trigger-phrase based.
    Either ought to retrieve at least one of the 10 variants in that family."""
    topic, topic_desc = TOPICS[topic_idx]
    action, action_phrase = ACTIONS[action_idx]
    target_skill_ids = [f"skill.{topic}.{action}.v{v}" for v in range(10)]
    # Paraphrased natural-language query
    q_para = f"How do I {action_phrase} issues in {topic_desc}?"
    return {
        "query": q_para,
        "target_skill_ids": target_skill_ids,
        "topic_family": topic,
        "action_family": action,
        "expected_min_rank": 5,  # at least 1 of 10 must appear in top-5
    }


def gen_specific_variant_query(topic_idx: int, action_idx: int, variant_idx: int) -> dict:
    """Trigger-phrase query that should hit a specific variant (sharper test)."""
    topic, _ = TOPICS[topic_idx]
    action, _ = ACTIONS[action_idx]
    detail_a, detail_b = VARIANT_DETAILS[variant_idx]
    target = f"skill.{topic}.{action}.v{variant_idx}"
    q = f"I see {detail_a} during {topic}. What's the right {action} pattern?"
    return {
        "query": q,
        "target_skill_ids": [target],
        "topic_family": topic,
        "action_family": action,
        "variant_idx": variant_idx,
        "expected_min_rank": 5,
        "query_kind": "specific_variant",
    }


def main():
    if len(sys.argv) > 1:
        scale = int(sys.argv[1])
    else:
        scale = 5000  # default

    n_topics = len(TOPICS)
    n_actions = len(ACTIONS)
    n_variants = len(VARIANT_DETAILS)
    full_n = n_topics * n_actions * n_variants  # 5000

    if scale > full_n:
        print(f"requested scale {scale} > max combinatorial {full_n}; clamping", file=sys.stderr)
        scale = full_n

    out_dir = Path(__file__).parent
    skills_path = out_dir / "skills_corpus.jsonl"
    queries_path = out_dir / "queries_groundtruth.jsonl"

    # Generate skills (deterministic order)
    skills = []
    idx = 0
    for t in range(n_topics):
        for a in range(n_actions):
            for v in range(n_variants):
                if idx >= scale:
                    break
                skills.append(gen_skill_record(t, a, v, idx))
                idx += 1
            if idx >= scale:
                break
        if idx >= scale:
            break

    with open(skills_path, "w", encoding="utf-8") as f:
        for s in skills:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(skills)} skills -> {skills_path}")

    # Sample queries — 50 family queries + 50 specific-variant queries = 100 total
    rng = random.Random(42)
    family_pairs = [(t, a) for t in range(n_topics) for a in range(n_actions)]
    rng.shuffle(family_pairs)
    family_q = [gen_query_for_family(t, a) for t, a in family_pairs[:50]]

    variant_triples = [(t, a, v) for t in range(n_topics) for a in range(n_actions) for v in range(n_variants)]
    rng.shuffle(variant_triples)
    specific_q = [gen_specific_variant_query(t, a, v) for t, a, v in variant_triples[:50]]

    queries = family_q + specific_q
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    print(f"wrote {len(queries)} ground-truth queries -> {queries_path}")
    print(f"  family-level (broad)  : {len(family_q)}")
    print(f"  variant-specific (sharp): {len(specific_q)}")


if __name__ == "__main__":
    main()
