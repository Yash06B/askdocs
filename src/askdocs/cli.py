"""Command line: fetch, ingest, search, ask, serve, eval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import Settings

FASTAPI_REPO = "https://github.com/fastapi/fastapi.git"
FASTAPI_DOCS = "docs/en/docs"
FASTAPI_URL = "https://fastapi.tiangolo.com/"


def _retriever(settings: Settings, index_name: str):
    from .index import Embedder, Index
    from .retrieve import Retriever

    index = Index.load(settings.index_dir(index_name))
    return Retriever(index, Embedder(index.meta["embed_model"]), rerank_model=settings.rerank_model)


def _llm(settings: Settings, provider: str | None, model: str | None):
    from .llm import make_llm

    provider = provider or settings.provider
    return make_llm(provider, model or (settings.model if provider == settings.provider else None), settings.ollama_url)


def cmd_fetch(args, settings: Settings) -> None:
    dest = settings.data_dir / "corpus" / "fastapi"
    if dest.exists():
        print(f"{dest} already exists; delete it to re-fetch")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(a, check=True)  # noqa: E731
    run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", FASTAPI_REPO, str(dest))
    run("git", "-C", str(dest), "sparse-checkout", "set", FASTAPI_DOCS)
    print(f"docs at {dest / FASTAPI_DOCS}")


def cmd_ingest(args, settings: Settings) -> None:
    from .chunking import load_documents
    from .index import Embedder, build_index

    source = Path(args.source or settings.data_dir / "corpus" / "fastapi" / FASTAPI_DOCS)
    docs = load_documents(source, exclude=args.exclude)
    print(f"loaded {len(docs)} documents from {source}")
    index = build_index(docs, args.chunker, Embedder(settings.embed_model), url_base=args.url_base)
    name = args.name or args.chunker
    index.save(settings.index_dir(name))
    print(f"index '{name}': {json.dumps(index.meta)}")


def cmd_search(args, settings: Settings) -> None:
    from .chunking import doc_url

    r = _retriever(settings, args.index)
    for i, h in enumerate(r.search(args.question, k=args.k, mode=args.mode), 1):
        url = doc_url(r.index.meta.get("url_base"), h.chunk.doc_path, h.chunk.anchor) or h.chunk.doc_path
        print(f"{i}. [{h.score:.3f}] {h.chunk.section or h.chunk.title}\n   {url}")


def cmd_ask(args, settings: Settings) -> None:
    from .answer import answer_question
    from .chunking import doc_url

    r = _retriever(settings, args.index)
    ans = answer_question(args.question, r, _llm(settings, args.provider, args.model), k=args.k, mode=args.mode)
    print(ans.text, "\n")
    for n in ans.cited:
        c = ans.hits[n - 1].chunk
        print(f"[{n}] {c.section or c.title} - {doc_url(r.index.meta.get('url_base'), c.doc_path, c.anchor) or c.doc_path}")


def cmd_serve(args, settings: Settings) -> None:
    import os

    import uvicorn

    if args.index:
        os.environ["ASKDOCS_INDEX"] = args.index
    uvicorn.run("askdocs.api:app", host=args.host, port=args.port)


def cmd_eval_generate(args, settings: Settings) -> None:
    from .eval.generate import generate_dataset
    from .index import Index

    index = Index.load(settings.index_dir(args.index))
    generate_dataset(index.chunks, _llm(settings, args.provider, args.model), n=args.n, out_path=args.out, seed=args.seed)


def cmd_eval_retrieval(args, settings: Settings) -> None:
    from .eval.generate import load_dataset
    from .eval.retrieval import evaluate_retrieval, results_table
    from .index import Embedder, Index
    from .retrieve import Retriever

    dataset = load_dataset(args.dataset)
    rows = []
    embedders: dict[str, Embedder] = {}
    rerankers = args.rerank_models or [settings.rerank_model]
    for name in args.indexes:
        index = Index.load(settings.index_dir(name))
        model = index.meta["embed_model"]
        embedders.setdefault(model, Embedder(model))
        for mode in args.modes:
            # Each reranker is its own row; the other modes don't use one.
            for reranker in rerankers if mode == "hybrid-rerank" else [None]:
                r = Retriever(index, embedders[model], rerank_model=reranker)
                r.search("warm up", k=1, mode=mode)  # load ONNX sessions before timing
                res = evaluate_retrieval(dataset, r, mode)
                res["index"] = name
                if reranker:
                    res["reranker"] = reranker
                    if len(rerankers) > 1:
                        res["mode"] = f"{mode} ({reranker.split('/')[-1]})"
                rows.append(res)
                print(f"{name:10s} {res['mode']:40s} R@5={res['recall@5']:.1%} MRR={res['mrr@10']:.3f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    table = results_table(rows)
    (out / "retrieval.md").write_text(f"{len(dataset)} questions from `{args.dataset}`\n\n{table}\n", encoding="utf-8")
    print("\n" + table)


def cmd_eval_answers(args, settings: Settings) -> None:
    from .eval.answers import answers_table, evaluate_answers
    from .eval.generate import load_dataset

    dataset = load_dataset(args.dataset)[: args.n]
    r = _retriever(settings, args.index)
    llm = _llm(settings, args.provider, args.model)
    judge = _llm(settings, args.judge_provider, args.judge_model)
    summaries, records = [], []
    for mode in args.modes:
        s, recs = evaluate_answers(dataset, r, llm, judge, mode, k=args.k)
        summaries.append(s)
        records.extend(recs)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers.json").write_text(json.dumps({"summary": summaries, "records": records}, indent=2), encoding="utf-8")
    table = answers_table(summaries)
    header = f"{len(dataset)} questions, index `{args.index}`, generator `{llm.name}`, judge `{judge.name}`, k={args.k}"
    (out / "answers.md").write_text(f"{header}\n\n{table}\n", encoding="utf-8")
    print("\n" + table)


def main(argv: list[str] | None = None) -> None:
    from .retrieve import MODES

    settings = Settings()
    p = argparse.ArgumentParser(prog="askdocs", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fetch", help="shallow-clone the FastAPI docs as a demo corpus")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("ingest", help="chunk + embed a folder of markdown into an index")
    s.add_argument("--source", help="folder of .md files (default: the fetched FastAPI docs)")
    s.add_argument("--chunker", choices=["markdown", "fixed"], default="markdown")
    s.add_argument("--name", help="index name (default: the chunker name)")
    s.add_argument("--url-base", default=FASTAPI_URL, help="published docs URL, for citation links")
    s.add_argument("--exclude", nargs="*", default=["release-notes.md"], help="glob patterns to skip")
    s.set_defaults(fn=cmd_ingest)

    def add_retrieval_args(s, default_mode="hybrid-rerank"):
        s.add_argument("--index", default=settings.index_name)
        s.add_argument("--mode", choices=MODES, default=default_mode)
        s.add_argument("--k", type=int, default=5)

    def add_llm_args(s, prefix=""):
        s.add_argument(f"--{prefix}provider", choices=["ollama", "anthropic"])
        s.add_argument(f"--{prefix}model")

    s = sub.add_parser("search", help="retrieve chunks only (no LLM)")
    s.add_argument("question")
    add_retrieval_args(s)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("ask", help="answer a question with citations")
    s.add_argument("question")
    add_retrieval_args(s)
    add_llm_args(s)
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("serve", help="run the web UI + JSON API")
    s.add_argument("--index")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    ev = sub.add_parser("eval", help="evaluation harness").add_subparsers(dest="eval_cmd", required=True)

    s = ev.add_parser("generate", help="synthesise a golden QA set from an index")
    s.add_argument("--index", default="markdown")
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="eval/datasets/fastapi_qa.jsonl")
    add_llm_args(s)
    s.set_defaults(fn=cmd_eval_generate)

    s = ev.add_parser("retrieval", help="score every chunker x retrieval mode")
    s.add_argument("--dataset", default="eval/datasets/fastapi_qa.jsonl")
    s.add_argument("--indexes", nargs="+", default=["fixed", "markdown"])
    s.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    s.add_argument("--rerank-models", nargs="+", help="compare several cross-encoders (default: ASKDOCS_RERANK_MODEL)")
    s.add_argument("--out-dir", default="eval/results")
    s.set_defaults(fn=cmd_eval_retrieval)

    s = ev.add_parser("answers", help="end-to-end answers scored by an LLM judge")
    s.add_argument("--dataset", default="eval/datasets/fastapi_qa.jsonl")
    s.add_argument("--index", default="markdown")
    s.add_argument("--modes", nargs="+", choices=MODES, default=["bm25", "hybrid-rerank"])
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--out-dir", default="eval/results")
    add_llm_args(s)
    add_llm_args(s, prefix="judge-")
    s.set_defaults(fn=cmd_eval_answers)

    args = p.parse_args(argv)
    try:
        args.fn(args, settings)
    except FileNotFoundError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
