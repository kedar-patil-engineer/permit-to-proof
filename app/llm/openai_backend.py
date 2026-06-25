"""OpenAI backend: real hosted extraction with structured JSON output.

Imports safely even when the openai package is absent or no key is configured,
so the rest of the app and the test suite are never blocked (Part B7, B11). It
asks the model for JSON matching the Obligation schema and insists the exact
supporting text be copied into source_quote, then parses defensively so a bad
response becomes flagged data rather than a crash.
"""

from __future__ import annotations

import os
from typing import List

from app.core.schema import Obligation, Segment
from app.llm.base import (
    DEFAULT_BATCH_SIZE,
    SYSTEM_PROMPT,
    build_user_prompt,
    run_batched_extraction,
)

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIBackend:
    """Hosted extraction via the OpenAI Chat Completions API."""

    name = "OpenAI"
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        self.model = model or DEFAULT_MODEL
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.batch_size = batch_size

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
            return response.choices[0].message.content or ""

        return run_batched_extraction(segments, call, "OAI", self.batch_size)
