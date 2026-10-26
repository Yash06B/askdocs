"""Retrieval: BM25, dense, hybrid (reciprocal rank fusion) and cross-encoder reranking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .index import Index

MODES = ("bm25", "dense", "hybrid", "hybrid-rerank")

TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in into is it its of on or "
    "so that the their then there these this to was what when where which while who why "
    "will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked lists of ids. RRF only looks at ranks, so BM25 and cosine
    scores never need to be put on the same scale."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass
class Hit:
    chunk: Chunk
    score: float


def _top(scores: np.ndarray, n: int) -> list[int]:
    n = min(n, len(scores))
    idx = np.argpartition(-scores, n - 1)[:n]
    return idx[np.argsort(-scores[idx])].tolist()


class Retriever:
    def __init__(
        self,
        index: Index,
        embedder: QueryEmbedder,
        rerank_model: str | None = None,
        candidates: int = 30,
    ):
        self.index = index
        self.embedder = embedder
        self.rerank_model = rerank_model
        self.candidates = candidates
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in index.chunks])
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            if not self.rerank_model:
                raise ValueError("hybrid-rerank needs a rerank model")
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(self.rerank_model)
        return self._reranker

    def bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

    def dense_scores(self, query: str) -> np.ndarray:
        return self.index.embeddings @ self.embedder.embed_query(query)

    def search(self, query: str, k: int = 5, mode: str = "hybrid-rerank") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        chunks = self.index.chunks

        if mode == "bm25":
            s = self.bm25_scores(query)
            return [Hit(chunks[i], float(s[i])) for i in _top(s, k)]
        if mode == "dense":
            s = self.dense_scores(query)
            return [Hit(chunks[i], float(s[i])) for i in _top(s, k)]

        pool = max(k, self.candidates)
        fused = reciprocal_rank_fusion(
            [_top(self.bm25_scores(query), pool), _top(self.dense_scores(query), pool)]
        )
        if mode == "hybrid":
            return [Hit(chunks[i], s) for i, s in fused[:k]]

        cand = [i for i, _ in fused[:pool]]
        scores = list(self.reranker.rerank(query, [chunks[i].embed_text for i in cand]))
        order = sorted(zip(cand, scores), key=lambda t: t[1], reverse=True)
        return [Hit(chunks[i], float(s)) for i, s in order[:k]]
