"""Ollama backend: local, private extraction for the cost and privacy comparison.

Talks to a locally running Ollama server over HTTP. Imports safely with no
server running and degrades to a clear error only when actually invoked, so it
never blocks the app or the offline test suite (Part B7, B11).
"""

from __future__ import annotations

import os
from typing import List

import requests

from app.core.schema import Obligation, Segment
from app.llm.base import (
    SYSTEM_PROMPT,
    build_user_prompt,
    extract_json_object,
    parse_obligations_payload,
)

DEFAULT_MODEL = "llama3.1"
DEFAULT_HOST = "http://localhost:11434"


class OllamaBackend:
    """Local extraction via an Ollama server."""

    name = "Ollama"
    DEFAULT_MODEL = DEFAULT_MODEL
    DEFAULT_HOST = DEFAULT_HOST

    def __init__(self, model: str = DEFAULT_MODEL, host: str | None = None,
                 timeout: float = 120.0):
        self.model = model or DEFAULT_MODEL
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.timeout = timeout

    @staticmethod
    def is_available(host: str | None = None) -> bool:
        """True only if an Ollama server answers at the configured host."""
        base = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        try:
            resp = requests.get(base + "/api/tags", timeout=2.0)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def extract_obligations(self, segments: List[Segment]) -> List[Obligation]:
        prompt = build_user_prompt(segments)
        try:
            resp = requests.post(
                self.host + "/api/chat",
                json={
                    "model": self.model,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not reach Ollama at %s. Is it running? (%s)" % (self.host, exc)
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                "Ollama returned HTTP %d: %s" % (resp.status_code, resp.text[:200])
            )
        content = resp.json().get("message", {}).get("content", "")
        payload = extract_json_object(content)
        return parse_obligations_payload(payload, segments, id_prefix="OLL")
