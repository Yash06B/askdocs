"""Synthesise a golden question set from the corpus.

For each sampled chunk an LLM writes a realistic question, a short reference answer
and a verbatim quote containing the answer. Items are kept only if the quote really
occurs in the source text and the question stands on its own.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from ..chunking import Chunk
from ..llm import LLM, parse_json
from .relevance import norm_tokens, quote_coverage

SYSTEM = "You write evaluation questions for a documentation search engine. You reply in JSON."

PROMPT = """Here is a passage from the "{title}" page of a software project's documentation (section: {section}).

<passage>
{text}
</passage>

Write ONE question that a developer who has NOT read this passage might type into a docs search box, and that this passage answers.

Rules:
- The question must make sense on its own. Never refer to "the passage", "this section", "the example above" or "the text".
- Phrase it the way a developer with a problem would, in your own words. Do not copy distinctive phrases from the passage.
- "quote": copy, exactly character for character, ONE sentence from the passage that contains the answer.
- "answer": a one or two sentence answer to the question.

Return JSON: {{"question": "...", "answer": "...", "quote": "..."}}
If the passage has nothing substantive to ask about (e.g. it is only links or a code listing), return {{"skip": true}}."""

BAD_PHRASES = re.compile(r"\b(passage|this section|the section|the text|above|below|the example)\b", re.I)


def _is_mostly_code(text: str) -> bool:
    lines = text.splitlines()
    in_fence, code = False, 0
    for line in lines:
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        code += in_fence
    return code > 0.5 * max(1, len(lines))


def _candidates(chunks: list[Chunk], seed: int, min_words: int) -> list[Chunk]:
    """Round-robin over documents so the set covers the whole corpus."""
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if len(c.text.split()) >= min_words and not _is_mostly_code(c.text):
            by_doc[c.doc_path].append(c)
    rng = random.Random(seed)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    for d in docs:
        rng.shuffle(by_doc[d])
    ordered: list[Chunk] = []
    while any(by_doc[d] for d in docs):
        for d in docs:
            if by_doc[d]:
                ordered.append(by_doc[d].pop())
    return ordered


def validate(item: dict, chunk: Chunk) -> str | None:
    """Return a rejection reason, or None if the item is usable."""
    if item.get("skip"):
        return "skipped by model"
    q, quote, ans = (str(item.get(k, "")).strip() for k in ("question", "quote", "answer"))
    if not q or not quote or not ans:
        return "missing field"
    if not q.endswith("?"):
        return "not a question"
    if BAD_PHRASES.search(q):
        return "question not standalone"
    if len(norm_tokens(quote)) < 5:
        return "quote too short"
    if quote_coverage(chunk.text, quote) < 0.9:
        return "quote not found in source"
    return None


def generate_dataset(
    chunks: list[Chunk],
    llm: LLM,
    n: int,
    out_path: str | Path,
    seed: int = 7,
    min_words: int = 60,
    log=print,
) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    rejected: dict[str, int] = defaultdict(int)
    for chunk in _candidates(chunks, seed, min_words):
        if len(items) >= n:
            break
        prompt = PROMPT.format(title=chunk.title, section=chunk.section, text=chunk.text)
        try:
            raw = parse_json(llm.complete(SYSTEM, prompt, json_mode=True, max_tokens=400))
        except Exception as e:  # malformed JSON, timeouts: just move on
            rejected[f"error: {type(e).__name__}"] += 1
            continue
        reason = validate(raw, chunk)
        key = str(raw.get("question", "")).lower().strip()
        if reason is None and key in seen:
            reason = "duplicate"
        if reason:
            rejected[reason] += 1
            continue
        seen.add(key)
        items.append(
            {
                "id": f"q{len(items) + 1:03d}",
                "question": raw["question"].strip(),
                "answer": raw["answer"].strip(),
                "quote": raw["quote"].strip(),
                "doc_path": chunk.doc_path,
                "section": chunk.section,
                "source_chunk_id": chunk.id,
            }
        )
        log(f"[{len(items)}/{n}] {items[-1]['question']}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    log(f"wrote {len(items)} items to {out}; rejected: {dict(rejected)}")
    return items


def load_dataset(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
