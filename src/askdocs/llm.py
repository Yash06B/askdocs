"""Thin LLM clients. Ollama runs locally for free; Anthropic is optional."""

from __future__ import annotations

import json
import re
from typing import Protocol

import httpx

DEFAULT_MODELS = {"ollama": "llama3.2:3b", "anthropic": "claude-sonnet-5-5"}


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 1024) -> str: ...


class OllamaLLM:
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: float = 600):
        self.name = f"ollama/{model}"
        self.model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def complete(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 1024) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 8192},
        }
        if json_mode:
            body["format"] = "json"
        r = self._client.post("/api/chat", json=body)
        r.raise_for_status()
        return r.json()["message"]["content"]


class AnthropicLLM:
    def __init__(self, model: str):
        import anthropic

        self.name = f"anthropic/{model}"
        self.model = model
        self._client = anthropic.Anthropic()

    def complete(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 1024) -> str:
        if json_mode:
            system += "\nRespond with a single JSON object and nothing else."
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")


def make_llm(provider: str, model: str | None = None, ollama_url: str = "http://localhost:11434") -> LLM:
    model = model or DEFAULT_MODELS[provider]
    if provider == "ollama":
        return OllamaLLM(model, ollama_url)
    if provider == "anthropic":
        return AnthropicLLM(model)
    raise ValueError(f"unknown provider {provider!r}")


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences or chatter around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))
