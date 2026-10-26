"""Grounded answer generation with inline citations."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .chunking import doc_url
from .llm import LLM
from .retrieve import Hit, Retriever

SYSTEM_PROMPT = """You answer developer questions using documentation excerpts.

Rules:
- Use ONLY the numbered sources. Do not add facts from outside knowledge.
- After each sentence that states a fact, cite the source(s) it came from, like [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "I couldn't find that in the docs."
- Be concise: a few sentences, plus a short code snippet only if the sources contain one."""

REFUSAL = "I couldn't find that in the docs."
CITE_RE = re.compile(r"\[(\d+)\]")


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.chunk.section or h.chunk.title}\n{h.chunk.text}" for i, h in enumerate(hits, 1))


def cited_indices(text: str, n_sources: int) -> list[int]:
    """1-based source numbers the answer cites, ignoring out-of-range numbers."""
    return sorted({int(m) for m in CITE_RE.findall(text) if 1 <= int(m) <= n_sources})


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    cited: list[int]
    retrieval_ms: float
    generation_ms: float

    @property
    def refused(self) -> bool:
        return REFUSAL.lower().rstrip(".") in self.text.lower()

    def to_dict(self, url_base: str | None = None) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "refused": self.refused,
            "timings_ms": {"retrieval": round(self.retrieval_ms, 1), "generation": round(self.generation_ms, 1)},
            "sources": [
                {
                    "n": i,
                    "cited": i in self.cited,
                    "section": h.chunk.section or h.chunk.title,
                    "doc_path": h.chunk.doc_path,
                    "url": doc_url(url_base, h.chunk.doc_path, h.chunk.anchor),
                    "score": round(h.score, 4),
                    "text": h.chunk.text,
                }
                for i, h in enumerate(self.hits, 1)
            ],
        }


def answer_question(
    question: str,
    retriever: Retriever,
    llm: LLM,
    k: int = 5,
    mode: str = "hybrid-rerank",
) -> Answer:
    t0 = time.perf_counter()
    hits = retriever.search(question, k=k, mode=mode)
    t1 = time.perf_counter()
    prompt = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    text = llm.complete(SYSTEM_PROMPT, prompt).strip()
    t2 = time.perf_counter()
    return Answer(
        question=question,
        text=text,
        hits=hits,
        cited=cited_indices(text, len(hits)),
        retrieval_ms=(t1 - t0) * 1000,
        generation_ms=(t2 - t1) * 1000,
    )
