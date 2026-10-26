import re

import numpy as np

from askdocs.chunking import Chunk
from askdocs.eval.relevance import is_relevant, quote_coverage
from askdocs.eval.retrieval import evaluate_retrieval
from askdocs.index import Index
from askdocs.retrieve import Retriever, reciprocal_rank_fusion, tokenize

VOCAB = ["cors", "middleware", "origins", "query", "parameter", "optional", "docker", "image", "deploy"]


class BagOfWordsEmbedder:
    """Deterministic stand-in for the ONNX model, so tests need no downloads."""

    def _vec(self, text: str) -> np.ndarray:
        words = re.findall(r"[a-z]+", text.lower())
        v = np.array([words.count(w) for w in VOCAB], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def make_retriever() -> Retriever:
    texts = [
        ("cors.md", "Use CORSMiddleware middleware to allow origins from other domains. cors cors"),
        ("query.md", "A query parameter becomes optional when you give it a default of None."),
        ("docker.md", "Build a docker image for your app and deploy it to a server."),
    ]
    chunks = [Chunk(id=f"{p}#0", doc_path=p, title=p, section="", anchor=None, text=t) for p, t in texts]
    emb = BagOfWordsEmbedder()
    index = Index(chunks=chunks, embeddings=np.stack([emb._vec(c.embed_text) for c in chunks]), meta={})
    return Retriever(index, emb)


def test_tokenize_drops_stopwords():
    assert tokenize("How do I make the Query optional?") == ["make", "query", "optional"]


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    ids = [i for i, _ in fused]
    assert set(ids[:2]) == {1, 2}
    assert ids[-1] in {3, 4}


def test_each_mode_finds_obvious_match():
    r = make_retriever()
    for mode in ("bm25", "dense", "hybrid"):
        assert r.search("make a query parameter optional", k=1, mode=mode)[0].chunk.doc_path == "query.md"


def test_quote_matching_is_chunk_boundary_tolerant():
    chunk = Chunk(id="a#0", doc_path="a.md", title="", section="", anchor=None,
                  text="Some text. A query parameter becomes optional when you give it a default of None. More.")
    item = {"doc_path": "a.md", "quote": "A `query` parameter becomes optional when you give it a default of `None`."}
    assert quote_coverage(chunk.text, item["quote"]) == 1.0
    assert is_relevant(chunk, item)
    assert not is_relevant(chunk, {**item, "doc_path": "b.md"})
    # only the first half of the quote made it into this chunk
    half = Chunk(id="a#1", doc_path="a.md", title="", section="", anchor=None, text="becomes optional when you")
    assert not is_relevant(half, item)


def test_evaluate_retrieval_metrics():
    r = make_retriever()
    dataset = [
        {"id": "q1", "question": "allow cross origin requests", "doc_path": "cors.md",
         "quote": "Use CORSMiddleware middleware to allow origins from other domains."},
        {"id": "q2", "question": "allowed origins inside a container", "doc_path": "docker.md",
         "quote": "Build a docker image for your app and deploy it to a server."},
    ]
    res = evaluate_retrieval(dataset, r, "bm25")
    assert res["recall@1"] == 0.5  # q2 shares words with the CORS page, none with the docker page
    assert res["recall@3"] == 1.0
    assert 0.5 < res["mrr@10"] < 1.0
