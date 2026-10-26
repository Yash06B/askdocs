"""Loading markdown docs and splitting them into retrievable chunks.

Two chunkers are provided so the eval harness can compare them:

* ``fixed``     - naive sliding window over words, no awareness of structure.
* ``markdown``  - splits on headings (ignoring ``#`` inside code fences), keeps the
                  heading breadcrumb with each chunk, and packs long sections by
                  paragraph so chunks never cut a sentence in half.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import asdict, dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
ANCHOR_RE = re.compile(r"\s*\{\s*#([\w-]+)\s*\}\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
INCLUDE_RE = re.compile(r"^\s*\{\*.*\*\}\s*$")  # mkdocs code-include directive
ADMONITION_RE = re.compile(r"^\s*///\s*(\w*)")  # "/// tip | Title" ... "///"
TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


@dataclass
class Document:
    path: str  # posix path relative to the corpus root, e.g. "tutorial/query-params.md"
    title: str
    text: str  # cleaned markdown


@dataclass
class Chunk:
    id: str
    doc_path: str
    title: str
    section: str  # heading breadcrumb, e.g. "Query Parameters > Defaults"
    anchor: str | None
    text: str

    @property
    def embed_text(self) -> str:
        """Text that gets embedded and BM25-indexed: breadcrumb + body."""
        return f"{self.section}\n\n{self.text}" if self.section else self.text

    def to_dict(self) -> dict:
        return asdict(self)


def clean_markdown(raw: str) -> str:
    """Strip mkdocs-specific syntax that only adds noise to retrieval."""
    out: list[str] = []
    in_fence = False
    for line in raw.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if INCLUDE_RE.match(line):
            continue
        m = ADMONITION_RE.match(line)
        if m:
            kind = m.group(1)
            if kind:
                out.append(f"**{kind.capitalize()}:**")
            continue
        out.append(TAG_RE.sub("", line))
    return "\n".join(out).strip() + "\n"


def parse_heading(line: str) -> tuple[int, str, str | None] | None:
    m = HEADING_RE.match(line)
    if not m:
        return None
    text = m.group(2)
    anchor = None
    a = ANCHOR_RE.search(text)
    if a:
        anchor = a.group(1)
        text = text[: a.start()]
    return len(m.group(1)), text.strip(), anchor


def load_documents(root: str | Path, exclude: list[str] | None = None) -> list[Document]:
    root = Path(root)
    exclude = exclude or []
    docs: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in exclude):
            continue
        text = clean_markdown(path.read_text(encoding="utf-8"))
        title = path.stem
        for line in text.splitlines():
            h = parse_heading(line)
            if h and h[0] == 1:
                title = h[1]
                break
        docs.append(Document(path=rel, title=title, text=text))
    return docs


@dataclass
class _Section:
    path: list[str]
    anchor: str | None
    body: str


def split_sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str, str | None]] = []
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        sections.append(
            _Section(
                path=[h for _, h, _ in stack],
                anchor=stack[-1][2] if stack else None,
                body="\n".join(body).strip(),
            )
        )
        body.clear()

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else parse_heading(line)
        if heading is None:
            body.append(line)
            continue
        flush()
        level = heading[0]
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append(heading)
    flush()
    return sections


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines, but never inside a code fence."""
    paras: list[str] = []
    cur: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if cur:
                paras.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        paras.append("\n".join(cur))
    return paras


def _pack(paragraphs: list[str], max_words: int) -> list[str]:
    windows: list[str] = []
    cur: list[str] = []
    n = 0
    for para in paragraphs:
        words = len(para.split())
        if words > max_words:
            if cur:
                windows.append("\n\n".join(cur))
                cur, n = [], 0
            toks = para.split()
            windows.extend(" ".join(toks[i : i + max_words]) for i in range(0, len(toks), max_words))
            continue
        if n + words > max_words and cur:
            windows.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(para)
        n += words
    if cur:
        windows.append("\n\n".join(cur))
    return windows


def markdown_chunks(doc: Document, max_words: int = 300, min_words: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in split_sections(doc.text):
        if not sec.body:
            continue
        section = " > ".join(sec.path) or doc.title
        words = len(sec.body.split())
        prev = chunks[-1] if chunks else None
        # Fold tiny sections into the previous chunk instead of emitting slivers.
        if words < min_words and prev and len(prev.text.split()) + words <= max_words:
            heading = sec.path[-1] if sec.path else ""
            prev.text += f"\n\n{heading}\n\n{sec.body}" if heading else f"\n\n{sec.body}"
            continue
        for window in _pack(_paragraphs(sec.body), max_words):
            chunks.append(
                Chunk(
                    id=f"{doc.path}#{len(chunks)}",
                    doc_path=doc.path,
                    title=doc.title,
                    section=section,
                    anchor=sec.anchor,
                    text=window,
                )
            )
    return chunks


def fixed_chunks(doc: Document, size: int = 250, overlap: int = 50) -> list[Chunk]:
    words = doc.text.split()
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    for start in range(0, max(1, len(words) - overlap), step):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(
            Chunk(
                id=f"{doc.path}#{len(chunks)}",
                doc_path=doc.path,
                title=doc.title,
                section="",
                anchor=None,
                text=" ".join(window),
            )
        )
    return chunks


CHUNKERS = {"markdown": markdown_chunks, "fixed": fixed_chunks}


def doc_url(base: str | None, path: str, anchor: str | None = None) -> str | None:
    """Map a docs source path to its published URL (mkdocs layout)."""
    if not base:
        return None
    p = path[:-3] if path.endswith(".md") else path
    if p == "index":
        p = ""
    elif p.endswith("/index"):
        p = p[: -len("index")]
    else:
        p += "/"
    return base.rstrip("/") + "/" + p + (f"#{anchor}" if anchor else "")
