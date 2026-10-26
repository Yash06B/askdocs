"""Settings read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("ASKDOCS_DATA", "data")))
    index_name: str = field(default_factory=lambda: _env("ASKDOCS_INDEX", "markdown"))
    provider: str = field(default_factory=lambda: _env("ASKDOCS_PROVIDER", "ollama"))
    model: str = field(default_factory=lambda: _env("ASKDOCS_MODEL", ""))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    embed_model: str = field(default_factory=lambda: _env("ASKDOCS_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(
        default_factory=lambda: _env("ASKDOCS_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    )

    @property
    def index_root(self) -> Path:
        return self.data_dir / "indexes"

    def index_dir(self, name: str | None = None) -> Path:
        return self.index_root / (name or self.index_name)
