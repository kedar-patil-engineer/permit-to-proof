"""OpenAI backend: real hosted extraction with structured JSON output.

Imports safely even when the openai package is absent or no key is configured,
so the rest of the app and the test suite are never blocked (Part B7, B11). It
asks the model for JSON matching the Obligation schema and insists the exact
supporting text be copied into source_quote, then parses defensively so a bad
response becomes flagged data rather than a crash.
"""

from __future__ import annotations

import os
import time
from typing import List

from app.core.schema import Obligation, Segment
from app.llm.base import (
    DEFAULT_BATCH_SIZE,
    SYSTEM_PROMPT,
    build_user_prompt,
    run_batched_extraction,
)

DEFAULT_MODEL = "gpt-4o-mini"

# USD per 1M tokens, by model. Used only to turn measured token counts into a
# reported cost for the paper's cost/latency table; never affects extraction.
PRICING_PER_MTOK = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


class OpenAIBackend:
    """Hosted extraction via the OpenAI Chat Completions API."""

    name = "OpenAI"
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        self.model = model or DEFAULT_MODEL
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.batch_size = batch_size
        # Populated by extract_obligations so callers (the eval harness) can
        # report real cost/latency for the paper instead of estimates.
        self.last_run_stats: dict = {}

    @staticmethod
    def is_available() -> bool:
        """True only if the package is importable and a key is configured."""
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "The 'openai' package is not installed. Run: pip install openai"
            ) from exc
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        return OpenAI(api_key=self._api_key)

    def extract_obligations(self, segments: List[Segment]) -> List[Obligation]:
        client = self._client()
        stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                 "total_tokens": 0}

        def call(batch):
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(batch)},
                ],
            )
            stats["calls"] += 1
            usage = getattr(response, "usage", None)
            if usage is not None:
                stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                stats["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
            return response.choices[0].message.content or ""

        start = time.perf_counter()
        obligations = run_batched_extraction(segments, call, "OAI", self.batch_size)
        stats["latency_seconds"] = round(time.perf_counter() - start, 3)
        stats["model"] = self.model
        stats["n_segments"] = len(segments)
        stats["n_obligations"] = len(obligations)
        stats["estimated_cost_usd"] = self._estimate_cost(stats)
        self.last_run_stats = stats
        return obligations

    def _estimate_cost(self, stats: dict) -> float:
        price = PRICING_PER_MTOK.get(self.model)
        if not price:
            return 0.0
        cost = (stats["prompt_tokens"] / 1_000_000) * price["input"] + \
               (stats["completion_tokens"] / 1_000_000) * price["output"]
        return round(cost, 6)
