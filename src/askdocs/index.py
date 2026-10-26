"""Embedding chunks and persisting/loading an index to disk.

An index is a directory with three files:

* ``chunks.jsonl``    - one Chunk per line
* ``embeddings.npy``  - float32 matrix, L2-normalised, row i = chunk i
* ``meta.json``       - embed model, chunker, corpus URL base, build stats

A flat numpy matrix is deliberate: at documentation scale (thousands of chunks) an
exact dot-product search takes well under a millisecond, so an ANN database would
add infrastructure without changing any result.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import CHUNKERS, Chunk, Document

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name)

    def _query_prefix(self) -> str:
        return BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vecs = np.array(list(self._model.embed(texts, batch_size=batch_size)), dtype=np.float32)
        return _normalize(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.array(next(iter(self._model.embed([self._query_prefix() + text]))), dtype=np.float32)
        return _normalize(vec[None, :])[0]


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray
    meta: dict

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(d / "embeddings.npy", self.embeddings)
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> Index:
        d = Path(directory)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"No index at {d}. Build one with `askdocs ingest`.")
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        embeddings = np.load(d / "embeddings.npy")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return cls(chunks=chunks, embeddings=embeddings, meta=meta)


def build_index(
    docs: list[Document],
    chunker: str,
    embedder: Embedder,
    url_base: str | None = None,
) -> Index:
    chunk_fn = CHUNKERS[chunker]
    chunks = [c for doc in docs for c in chunk_fn(doc)]
    t0 = time.perf_counter()
    embeddings = embedder.embed_documents([c.embed_text for c in chunks])
    words = [len(c.text.split()) for c in chunks]
    meta = {
        "chunker": chunker,
        "embed_model": embedder.model_name,
        "url_base": url_base,
        "documents": len(docs),
        "chunks": len(chunks),
        "avg_chunk_words": round(sum(words) / max(1, len(words)), 1),
        "embed_seconds": round(time.perf_counter() - t0, 1),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return Index(chunks=chunks, embeddings=embeddings, meta=meta)
