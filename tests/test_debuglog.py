"""LLM_DEBUG: silent when unset, narrates every offline AI call site to stderr when set."""

from __future__ import annotations

import pytest

from flagship.debuglog import debug_enabled, log_block
from flagship.retrieval import HashingEmbedder
from tests.conftest import make_assistant


def test_debug_disabled_by_default_and_for_falsy_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # conftest's autouse fixture already deleted LLM_DEBUG → off.
    assert not debug_enabled()
    for falsy in ("0", "false", "FALSE", "  False  "):
        monkeypatch.setenv("LLM_DEBUG", falsy)
        assert not debug_enabled()
    monkeypatch.setenv("LLM_DEBUG", "1")
    assert debug_enabled()
    monkeypatch.setenv("LLM_DEBUG", "yes")
    assert debug_enabled()


def test_silent_when_unset(capsys: pytest.CaptureFixture[str]) -> None:
    assistant = make_assistant()
    capsys.readouterr()  # drop anything from ingest
    answer = assistant.ask("What do guardrails defend against?")
    captured = capsys.readouterr()
    assert not answer.blocked
    assert "=== AI REQUEST" not in captured.err
    assert "=== AI RESPONSE" not in captured.err
    assert captured.err == ""


def test_agent_and_embedder_blocks_on_stderr_when_set(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assistant = make_assistant()  # built (and ingested) before debug is switched on
    monkeypatch.setenv("LLM_DEBUG", "1")
    capsys.readouterr()
    answer = assistant.ask("What do guardrails defend against?")
    captured = capsys.readouterr()
    assert not answer.blocked
    # The whole ask() pipeline narrated itself on stderr, nothing on stdout.
    assert captured.out == ""
    err = captured.err
    # Agent turn 1: the offline policy saw the task + tools, decided to search.
    assert "=== AI REQUEST (agent step 1) ===" in err
    assert "HeuristicPolicy (offline, no API call)" in err
    assert "knowledge_search" in err
    assert "calculator" in err
    assert "=== AI RESPONSE (agent step 1) ===" in err
    assert "=== TOOL RESULT (step 1) ===" in err
    # Agent turn 2: the policy answered from the tool result.
    assert "=== AI REQUEST (agent step 2) ===" in err
    assert "final answer" in err
    # The knowledge_search tool embedded the query via the offline embedder.
    assert "=== AI REQUEST (embeddings) ===" in err
    assert "HashingEmbedder (offline, no API call)" in err
    assert "=== AI RESPONSE (embeddings) ===" in err


def test_embedder_logs_shape_not_raw_vectors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    embedder = HashingEmbedder(dim=32)
    vectors = embedder.embed(["guardrails block prompt injection", "x " * 300])
    err = capsys.readouterr().err
    assert "text_count: 2" in err
    assert "guardrails block prompt injection"[:80] in err
    assert "vector_count: 2" in err
    assert "dims: 32" in err
    # Never the raw numbers — only the shape.
    assert str(vectors[0][:3]) not in err


def test_log_block_truncates_huge_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    log_block("AI REQUEST (test)", payload="x" * 5000, small="ok")
    err = capsys.readouterr().err
    assert "... [truncated]" in err
    assert "x" * 5000 not in err
    assert "small: ok" in err
