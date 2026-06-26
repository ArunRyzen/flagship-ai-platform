"""Shared fixtures: a fully offline assistant (hashing embedder + heuristic policy)."""

from __future__ import annotations

import pytest

from flagship.config import Settings
from flagship.factory import build_assistant
from flagship.pipeline import Assistant
from flagship.sample_data import SAMPLE_DOCS


def make_assistant() -> Assistant:
    assistant = build_assistant(Settings())  # no keys → offline embedder + heuristic policy
    for doc_id, text in SAMPLE_DOCS.items():
        assistant.ingest(doc_id, text)
    return assistant


@pytest.fixture
def assistant() -> Assistant:
    return make_assistant()
