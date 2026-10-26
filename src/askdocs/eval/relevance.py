"""Deciding whether a retrieved chunk contains a question's evidence.

Each eval item stores a verbatim ``quote`` from the docs that answers it. A chunk is
relevant when it comes from the same document and contains most of that quote as a
contiguous run of tokens. Matching on the quote, not on a chunk id, is what makes the
labels independent of how the corpus was chunked, so different chunkers can be
compared on the same question set.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..chunking import Chunk

_TOK = re.compile(r"[a-z0-9]+")


def norm_tokens(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def quote_coverage(chunk_text: str, quote: str) -> float:
    """Fraction of the quote found as one contiguous token run in the chunk."""
    q = norm_tokens(quote)
    if not q:
        return 0.0
    c = norm_tokens(chunk_text)
    m = SequenceMatcher(None, c, q, autojunk=False).find_longest_match(0, len(c), 0, len(q))
    return m.size / len(q)


def is_relevant(chunk: Chunk, item: dict, threshold: float = 0.6) -> bool:
    return chunk.doc_path == item["doc_path"] and quote_coverage(chunk.text, item["quote"]) >= threshold
