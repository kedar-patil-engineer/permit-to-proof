"""LLM backends. The pipeline depends only on the LLMBackend protocol, so any
of these is interchangeable at runtime (Part B7)."""

from __future__ import annotations

from typing import List

from .base import LLMBackend
from .mock import MockBackend
from .ollama_backend import OllamaBackend
from .openai_backend import OpenAIBackend

BACKEND_NAMES: List[str] = ["Mock", "OpenAI", "Ollama"]


def make_backend(name: str, *, model: str | None = None, **kwargs) -> LLMBackend:
    """Construct a backend by display name. Defaults to the Mock backend."""
    key = (name or "Mock").strip().lower()
    if key == "openai":
        return OpenAIBackend(model=model or OpenAIBackend.DEFAULT_MODEL, **kwargs)
    if key == "ollama":
        return OllamaBackend(model=model or OllamaBackend.DEFAULT_MODEL, **kwargs)
    return MockBackend()


__all__ = [
    "LLMBackend",
    "MockBackend",
    "OpenAIBackend",
    "OllamaBackend",
    "BACKEND_NAMES",
    "make_backend",
]
