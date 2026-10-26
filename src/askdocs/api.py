"""FastAPI app: JSON API plus a single-page UI at ``/``."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .answer import answer_question
from .chunking import doc_url
from .config import Settings
from .index import Embedder, Index
from .llm import make_llm
from .retrieve import MODES, Retriever

WEB_DIR = Path(__file__).parent / "web"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    index = Index.load(settings.index_dir())
    state["retriever"] = Retriever(index, Embedder(index.meta["embed_model"]), rerank_model=settings.rerank_model)
    state["llm"] = make_llm(settings.provider, settings.model or None, settings.ollama_url)
    state["meta"] = index.meta
    yield
    state.clear()


app = FastAPI(title="AskDocs", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str = "hybrid-rerank"
    k: int = Field(default=5, ge=1, le=10)


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "llm": state["llm"].name, "index": state["meta"]}


@app.get("/api/search")
async def search(q: str = Query(min_length=2), mode: str = "hybrid-rerank", k: int = Query(5, ge=1, le=20)) -> dict:
    _check_mode(mode)
    r: Retriever = state["retriever"]
    hits = await run_in_threadpool(r.search, q, k, mode)
    base = state["meta"].get("url_base")
    return {
        "query": q,
        "mode": mode,
        "hits": [
            {
                "section": h.chunk.section or h.chunk.title,
                "url": doc_url(base, h.chunk.doc_path, h.chunk.anchor),
                "score": round(h.score, 4),
                "text": h.chunk.text,
            }
            for h in hits
        ],
    }


@app.post("/api/ask")
async def ask(req: AskRequest) -> dict:
    _check_mode(req.mode)
    try:
        ans = await run_in_threadpool(answer_question, req.question, state["retriever"], state["llm"], req.k, req.mode)
    except Exception as e:  # LLM unreachable, model missing, etc.
        raise HTTPException(502, f"LLM call failed: {e}") from e
    return ans.to_dict(state["meta"].get("url_base"))
