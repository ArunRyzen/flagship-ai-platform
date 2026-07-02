"""Gemini live-path seams, tested offline with mocked clients (no network, no key).

Mirrors how the other live paths are covered: we fake the `google-genai` client so we can
verify our *own* code — request shaping, response parsing, provider selection — without
ever calling the real API.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from flagship.agent import GeminiPolicy, HeuristicPolicy, KnowledgeTool, build_registry
from flagship.config import Settings
from flagship.errors import PolicyError
from flagship.factory import build_embedder, build_policy
from flagship.models import Step, ToolCall, ToolResult
from flagship.retrieval import GeminiEmbedder, HashingEmbedder, Retriever

# --- helpers ------------------------------------------------------------------------


def _registry() -> Any:
    retriever = Retriever(HashingEmbedder())
    return build_registry(KnowledgeTool(retriever))


def _text_part(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, function_call=None)


def _call_part(name: str, args: dict, call_id: str | None = None) -> SimpleNamespace:
    call = SimpleNamespace(id=call_id, name=name, args=args)
    return SimpleNamespace(text=None, function_call=call)


def _response(parts: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])


class _FakeModels:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs = kwargs
        return self.response


def _patch_client(monkeypatch: pytest.MonkeyPatch, models: _FakeModels) -> None:
    import google.genai as genai

    monkeypatch.setattr(genai, "Client", lambda api_key=None: SimpleNamespace(models=models))


def _policy() -> GeminiPolicy:
    return GeminiPolicy(registry=_registry(), model="gemini-2.5-flash", max_tokens=64, api_key="k")


# --- GeminiPolicy -------------------------------------------------------------------


def test_gemini_policy_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _FakeModels(_response([_call_part("knowledge_search", {"query": "rag"})]))
    _patch_client(monkeypatch, models)
    decision = _policy().decide("what is rag", [])
    assert not decision.is_final
    assert decision.tool_calls[0].name == "knowledge_search"
    assert decision.tool_calls[0].args == {"query": "rag"}
    assert decision.tool_calls[0].id  # ids are synthesized when Gemini omits them


def test_gemini_policy_parses_final_text(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _FakeModels(_response([_text_part("RAG grounds answers.")]))
    _patch_client(monkeypatch, models)
    decision = _policy().decide("what is rag", [])
    assert decision.is_final
    assert decision.finish == "RAG grounds answers."


def test_gemini_policy_disables_automatic_function_calling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Our run_agent loop must own tool execution — the SDK's auto-calling stays off.
    models = _FakeModels(_response([_text_part("ok")]))
    _patch_client(monkeypatch, models)
    _policy().decide("hi", [])
    config = models.kwargs["config"]
    assert config.automatic_function_calling.disable is True
    declared = config.tools[0].function_declarations
    assert {d.name for d in declared} == {"knowledge_search", "calculator"}


def test_gemini_policy_replays_history(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _FakeModels(_response([_text_part("done")]))
    _patch_client(monkeypatch, models)
    step = Step(
        tool_calls=[ToolCall(id="c0", name="knowledge_search", args={"query": "rag"})],
        results=[ToolResult(id="c0", name="knowledge_search", output="[rag] RAG grounds ...")],
    )
    _policy().decide("what is rag", [step])
    contents = models.kwargs["contents"]
    # task, then one model turn (the tool call) and one user turn (the tool result)
    assert len(contents) == 3
    assert contents[1].parts[0].function_call.name == "knowledge_search"
    assert contents[2].parts[0].function_response.response == {"result": "[rag] RAG grounds ..."}


def test_gemini_policy_wraps_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import google.genai as genai
    from google.genai import errors

    class _Boom:
        def generate_content(self, **kwargs: Any) -> SimpleNamespace:
            raise errors.APIError(500, {"error": {"message": "boom"}})

    monkeypatch.setattr(genai, "Client", lambda api_key=None: SimpleNamespace(models=_Boom()))
    with pytest.raises(PolicyError):
        _policy().decide("hi", [])


# --- GeminiEmbedder -----------------------------------------------------------------


def test_gemini_embedder_returns_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _EmbedModels:
        def embed_content(self, **kwargs: Any) -> SimpleNamespace:
            assert kwargs["model"] == "gemini-embedding-001"
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2]), SimpleNamespace(values=[0.3, 0.4])]
            )

    import google.genai as genai

    monkeypatch.setattr(
        genai, "Client", lambda api_key=None: SimpleNamespace(models=_EmbedModels())
    )
    embedder = GeminiEmbedder(model="gemini-embedding-001", api_key="k")
    assert embedder.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]


# --- Factory selection ---------------------------------------------------------------


def test_factory_prefers_gemini_when_key_set() -> None:
    settings = Settings(gemini_api_key="k", anthropic_api_key=None, openai_api_key=None)
    assert isinstance(build_embedder(settings), GeminiEmbedder)
    assert isinstance(build_policy(settings, _registry()), GeminiPolicy)


def test_factory_stays_offline_without_keys() -> None:
    settings = Settings(gemini_api_key=None, anthropic_api_key=None, openai_api_key=None)
    assert isinstance(build_embedder(settings), HashingEmbedder)
    assert isinstance(build_policy(settings, _registry()), HeuristicPolicy)
