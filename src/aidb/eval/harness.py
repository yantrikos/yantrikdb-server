"""Evaluation harness for AIDB recall quality.

Measures: Recall@K, Precision@K, MRR, and per-query breakdown.
Compares AIDB multi-signal retrieval vs vector-only baseline.
"""

import time
from dataclasses import dataclass, field

from aidb import AIDB
from aidb.eval.synthetic import GOLDEN_QUERIES, SESSIONS, load_sessions_into_db


@dataclass
class QueryResult:
    """Result of evaluating a single golden query."""

    query_id: str
    query_text: str
    description: str
    test_tags: list[str]
    expected_count: int
    retrieved_texts: list[str]
    retrieved_rids: list[str]
    scores: list[float]
    why_retrieved: list[list[str]]
    hits: list[str]  # expected texts that were found
    misses: list[str]  # expected texts that were missed
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float  # 1/rank of first relevant result


@dataclass
class EvalReport:
    """Full evaluation report."""

    mode: str  # "aidb" or "vector_only"
    total_memories: int
    total_queries: int
    query_results: list[QueryResult]
    mean_recall_at_k: float = 0.0
    mean_precision_at_k: float = 0.0
    mean_reciprocal_rank: float = 0.0
    elapsed_seconds: float = 0.0
    recall_by_tag: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Evaluation Report: {self.mode} ===",
            f"Memories: {self.total_memories} | Queries: {self.total_queries}",
            f"Mean Recall@K:    {self.mean_recall_at_k:.3f}",
            f"Mean Precision@K: {self.mean_precision_at_k:.3f}",
            f"Mean MRR:         {self.mean_reciprocal_rank:.3f}",
            f"Elapsed:          {self.elapsed_seconds:.3f}s",
            "",
            "── By Tag ──",
        ]
        for tag, score in sorted(self.recall_by_tag.items()):
            lines.append(f"  {tag:20s} recall={score:.3f}")

        lines.append("")
        lines.append("── Per Query ──")
        for qr in self.query_results:
            status = "PASS" if qr.recall_at_k >= 0.5 else "FAIL"
            lines.append(
                f"  [{status}] {qr.query_id}: recall={qr.recall_at_k:.2f} "
                f"prec={qr.precision_at_k:.2f} mrr={qr.reciprocal_rank:.2f} "
                f"({len(qr.hits)}/{qr.expected_count} found)"
            )
            if qr.misses:
                for m in qr.misses:
                    lines.append(f"        MISS: {m[:80]}...")

        return "\n".join(lines)


def evaluate(db: AIDB, text_to_rid: dict, top_k: int = 10, embedder=None) -> EvalReport:
    """Run all golden queries against a loaded AIDB and measure quality.

    Args:
        db: AIDB instance with loaded synthetic data.
        text_to_rid: Mapping from memory text to RID.
        top_k: Number of results to retrieve per query.
        embedder: SentenceTransformer for embedding queries.

    Returns:
        EvalReport with per-query and aggregate metrics.
    """
    start = time.time()
    query_results = []

    for gq in GOLDEN_QUERIES:
        query_embedding = None
        if embedder is not None:
            vec = embedder.encode(gq["query"])
            query_embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)

        results = db.recall(
            query=gq["query"],
            query_embedding=query_embedding,
            top_k=top_k,
            skip_reinforce=True,
        )

        retrieved_texts = [r["text"] for r in results]
        retrieved_rids = [r["rid"] for r in results]
        scores = [r["score"] for r in results]
        why = [r["why_retrieved"] for r in results]

        # Calculate hits/misses — with consolidation-aware provenance
        expected_rids = {text_to_rid[t] for t in gq["expected_texts"] if t in text_to_rid}
        rid_to_text = {v: k for k, v in text_to_rid.items()}

        # Build a set of "covered" source RIDs by checking consolidated_from metadata
        covered_rids = set(retrieved_rids)
        for r in results:
            consolidated_from = r.get("metadata", {}).get("consolidated_from", [])
            covered_rids.update(consolidated_from)

        hits = [t for t in gq["expected_texts"] if text_to_rid.get(t) in covered_rids]
        misses = [t for t in gq["expected_texts"] if text_to_rid.get(t) not in covered_rids]

        # Recall@K: fraction of expected results found (directly or via consolidation)
        recall_at_k = len(hits) / len(gq["expected_texts"]) if gq["expected_texts"] else 0.0

        # Precision@K: fraction of retrieved results that are relevant (direct or covering)
        relevant_retrieved = 0
        for r in results:
            r_rid = r["rid"]
            if r_rid in expected_rids:
                relevant_retrieved += 1
            else:
                # Check if this consolidated memory covers any expected sources
                consolidated_from = r.get("metadata", {}).get("consolidated_from", [])
                if any(src_rid in expected_rids for src_rid in consolidated_from):
                    relevant_retrieved += 1
        precision_at_k = relevant_retrieved / len(results) if results else 0.0

        # MRR: reciprocal rank of first relevant result (direct or covering)
        reciprocal_rank = 0.0
        for rank, r in enumerate(results, 1):
            r_rid = r["rid"]
            is_relevant = r_rid in expected_rids
            if not is_relevant:
                consolidated_from = r.get("metadata", {}).get("consolidated_from", [])
                is_relevant = any(src_rid in expected_rids for src_rid in consolidated_from)
            if is_relevant:
                reciprocal_rank = 1.0 / rank
                break

        query_results.append(QueryResult(
            query_id=gq["id"],
            query_text=gq["query"],
            description=gq["description"],
            test_tags=gq["test_tags"],
            expected_count=len(gq["expected_texts"]),
            retrieved_texts=retrieved_texts,
            retrieved_rids=retrieved_rids,
            scores=scores,
            why_retrieved=why,
            hits=hits,
            misses=misses,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            reciprocal_rank=reciprocal_rank,
        ))

    elapsed = time.time() - start

    # Aggregate metrics
    mean_recall = sum(qr.recall_at_k for qr in query_results) / len(query_results)
    mean_precision = sum(qr.precision_at_k for qr in query_results) / len(query_results)
    mean_mrr = sum(qr.reciprocal_rank for qr in query_results) / len(query_results)

    # Recall by tag
    tag_scores: dict[str, list[float]] = {}
    for qr in query_results:
        for tag in qr.test_tags:
            tag_scores.setdefault(tag, []).append(qr.recall_at_k)
    recall_by_tag = {tag: sum(s) / len(s) for tag, s in tag_scores.items()}

    stats = db.stats()

    return EvalReport(
        mode="aidb",
        total_memories=stats["active_memories"],
        total_queries=len(query_results),
        query_results=query_results,
        mean_recall_at_k=mean_recall,
        mean_precision_at_k=mean_precision,
        mean_reciprocal_rank=mean_mrr,
        elapsed_seconds=elapsed,
        recall_by_tag=recall_by_tag,
    )


def run_comparison(embedding_dim: int = 384, embedder=None, top_k: int = 10):
    """Run side-by-side comparison: AIDB multi-signal vs vector-only baseline.

    Returns (aidb_report, baseline_report).
    """
    # ── AIDB multi-signal ──
    db_aidb = AIDB(db_path=":memory:", embedding_dim=embedding_dim, embedder=embedder)
    text_to_rid_aidb = load_sessions_into_db(db_aidb, embedder=embedder)
    report_aidb = evaluate(db_aidb, text_to_rid_aidb, top_k=top_k, embedder=embedder)
    report_aidb.mode = "aidb_multi_signal"
    db_aidb.close()

    return report_aidb
