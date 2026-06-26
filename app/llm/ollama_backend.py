"""Ollama backend: local, private extraction for the cost and privacy comparison.

Talks to a locally running Ollama server over HTTP. Imports safely with no
server running and degrades to a clear error only when actually invoked, so it
never blocks the app or the offline test suite (Part B7, B11).
"""

from __future__ import annotations

import os
import time
from typing import List

import requests

from app.core.schema import Obligation, Segment
from app.llm.base import (
    DEFAULT_BATCH_SIZE,
    SYSTEM_PROMPT,
    build_user_prompt,
    run_batched_extraction,
)

DEFAULT_MODEL = "llama3.1"
DEFAULT_HOST = "http://localhost:11434"


class OllamaBackend:
    """Local extraction via an Ollama server."""

    name = "Ollama"
    DEFAULT_MODEL = DEFAULT_MODEL
    DEFAULT_HOST = DEFAULT_HOST

    def __init__(self, model: str = DEFAULT_MODEL, host: str | None = None,
                 timeout: float = 120.0, batch_size: int = DEFAULT_BATCH_SIZE):
        self.model = model or DEFAULT_MODEL
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        # Latency/throughput for the local-vs-hosted comparison. Local runs are
        # free, so cost is reported as zero rather than a token price.
        self.last_run_stats: dict = {}

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
        stats = {"calls": 0, "total_tokens": 0}

        def call(batch):
            stats["calls"] += 1
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
                            {"role": "user", "content": build_user_prompt(batch)},
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
            body = resp.json()
            ev = body.get("eval_count", 0) or 0
            pv = body.get("prompt_eval_count", 0) or 0
            stats["total_tokens"] += ev + pv
            return body.get("message", {}).get("content", "")

        start = time.perf_counter()
        obligations = run_batched_extraction(segments, call, "OLL", self.batch_size)
        stats["latency_seconds"] = round(time.perf_counter() - start, 3)
        stats["model"] = self.model
        stats["n_segments"] = len(segments)
        stats["n_obligations"] = len(obligations)
        stats["estimated_cost_usd"] = 0.0  # local inference is free
        self.last_run_stats = stats
        return obligations
