"""Retrieval metrics: evidence recall@k, MRR@10, document recall@5 and latency."""

from __future__ import annotations

import time

import numpy as np

from ..retrieve import Retriever
from .relevance import is_relevant

KS = (1, 3, 5, 10)


def evaluate_retrieval(dataset: list[dict], retriever: Retriever, mode: str) -> dict:
    depth = max(KS)
    recall = {k: 0 for k in KS}
    doc_recall5 = 0
    rr = 0.0
    latencies: list[float] = []
    misses: list[str] = []

    for item in dataset:
        t0 = time.perf_counter()
        hits = retriever.search(item["question"], k=depth, mode=mode)
        latencies.append((time.perf_counter() - t0) * 1000)
        rel = [is_relevant(h.chunk, item) for h in hits]
        first = rel.index(True) if True in rel else None
        for k in KS:
            recall[k] += first is not None and first < k
        rr += 1 / (first + 1) if first is not None else 0
        doc_recall5 += any(h.chunk.doc_path == item["doc_path"] for h in hits[:5])
        if first is None:
            misses.append(item["id"])

    n = len(dataset)
    lat = np.array(latencies)
    return {
        "mode": mode,
        "n": n,
        **{f"recall@{k}": recall[k] / n for k in KS},
        "mrr@10": rr / n,
        "doc_recall@5": doc_recall5 / n,
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "misses": misses,
    }


def results_table(rows: list[dict]) -> str:
    head = "| Chunking | Retrieval | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Doc recall@5 | p50 ms | p95 ms |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['index']} | {r['mode']} | {r['recall@1']:.1%} | {r['recall@3']:.1%} | {r['recall@5']:.1%} "
            f"| {r['recall@10']:.1%} | {r['mrr@10']:.3f} | {r['doc_recall@5']:.1%} "
            f"| {r['latency_p50_ms']:.1f} | {r['latency_p95_ms']:.1f} |"
        )
    return "\n".join(lines)
