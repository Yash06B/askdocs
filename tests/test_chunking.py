from askdocs.chunking import Document, clean_markdown, doc_url, fixed_chunks, markdown_chunks, split_sections

RAW = """# Query Parameters { #query-parameters }

Intro paragraph about query parameters that is long enough to stand alone as a chunk of text.

{* ../../docs_src/query_params/tutorial001_py310.py hl[9] *}

/// tip

Use <dfn title="x">defaults</dfn> wisely.

///

## Defaults { #defaults }

```python
# this is a comment, not a heading
skip: int = 0
```

Defaults body.
"""


def test_clean_markdown_strips_mkdocs_syntax():
    text = clean_markdown(RAW)
    assert "{*" not in text
    assert "**Tip:**" in text
    assert "///" not in text
    assert "<dfn" not in text and "defaults wisely" in text
    assert "# this is a comment" in text  # code is untouched


def test_split_sections_ignores_hash_in_code_fence():
    sections = split_sections(clean_markdown(RAW))
    paths = [s.path for s in sections if s.body]
    assert paths == [["Query Parameters"], ["Query Parameters", "Defaults"]]
    assert sections[-1].anchor == "defaults"
    assert "# this is a comment" in sections[-1].body


def test_markdown_chunks_keep_breadcrumb_and_fold_tiny_sections():
    doc = Document(path="tutorial/query-params.md", title="Query Parameters", text=clean_markdown(RAW))
    chunks = markdown_chunks(doc, max_words=300, min_words=40)
    # "Defaults" is tiny, so it is folded into the previous chunk
    assert len(chunks) == 1
    assert chunks[0].section == "Query Parameters"
    assert "Defaults body." in chunks[0].text
    assert chunks[0].embed_text.startswith("Query Parameters\n\n")


def test_markdown_chunks_respect_max_words():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    doc = Document(path="a.md", title="A", text=f"# A\n\n## Big {{ #big }}\n\n{body}")
    chunks = markdown_chunks(doc, max_words=100)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 100 for c in chunks)
    assert all(c.anchor == "big" and c.section == "A > Big" for c in chunks)


def test_fixed_chunks_overlap_and_cover_everything():
    words = [f"w{i}" for i in range(600)]
    doc = Document(path="a.md", title="A", text=" ".join(words))
    chunks = fixed_chunks(doc, size=250, overlap=50)
    assert [c.text.split()[0] for c in chunks] == ["w0", "w200", "w400"]
    assert chunks[-1].text.split()[-1] == "w599"


def test_doc_url():
    base = "https://fastapi.tiangolo.com/"
    assert doc_url(base, "index.md") == "https://fastapi.tiangolo.com/"
    assert doc_url(base, "tutorial/index.md") == "https://fastapi.tiangolo.com/tutorial/"
    assert doc_url(base, "tutorial/query-params.md", "defaults") == "https://fastapi.tiangolo.com/tutorial/query-params/#defaults"
    assert doc_url(None, "a.md") is None
