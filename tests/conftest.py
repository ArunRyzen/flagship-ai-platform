"""Shared fixtures: a fully offline assistant (hashing embedder + heuristic policy)."""

from __future__ import annotations

import pytest

from flagship.config import Settings
from flagship.factory import build_assistant
from flagship.pipeline import Assistant
from flagship.sample_data import SAMPLE_DOCS


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test offline even if the developer has real keys in their environment.

    Without this, having GEMINI_API_KEY exported locally would make the factory pick the
    live GeminiPolicy/GeminiEmbedder during tests — slow, flaky, and costly. Deleting the
    variables per-test guarantees the offline fakes are always selected.
    """
    for var in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # Also ignore any local `.env` file for the duration of each test.
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def make_assistant() -> Assistant:
    # Explicit None keys → offline embedder + heuristic policy, regardless of any .env file.
    settings = Settings(gemini_api_key=None, anthropic_api_key=None, openai_api_key=None)
    assistant = build_assistant(settings)
    for doc_id, text in SAMPLE_DOCS.items():
        assistant.ingest(doc_id, text)
    return assistant


@pytest.fixture
def assistant() -> Assistant:
    return make_assistant()
