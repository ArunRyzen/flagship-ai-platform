"""Agent: tools execute, the scripted policy drives the loop, calculator is safe."""

from __future__ import annotations

from flagship.agent import (
    KnowledgeTool,
    ScriptedPolicy,
    build_registry,
    calculator,
    run_agent,
)
from flagship.models import Decision, ToolCall
from flagship.retrieval import HashingEmbedder, Retriever, chunk_text


def _registry() -> tuple:
    retriever = Retriever(HashingEmbedder())
    retriever.add(chunk_text(doc_id="rag", text="RAG grounds an LLM in retrieved documents."))
    knowledge = KnowledgeTool(retriever)
    return build_registry(knowledge), knowledge


def test_calculator_safe() -> None:
    assert calculator("2 * (3 + 4)") == "14.0"


def test_scripted_agent_runs_tool_then_finishes() -> None:
    registry, knowledge = _registry()
    policy = ScriptedPolicy(
        [
            Decision(
                tool_calls=[ToolCall(id="c0", name="knowledge_search", args={"query": "rag"})]
            ),
            Decision(finish="RAG grounds an LLM in retrieved documents."),
        ]
    )
    text, history = run_agent(task="what is rag", registry=registry, policy=policy)
    assert "grounds an LLM" in text
    assert len(history) == 1
    assert knowledge.last_results  # the tool recorded what it retrieved (for citations)


def test_unknown_tool_is_error() -> None:
    registry, _ = _registry()
    result = registry.execute(ToolCall(id="x", name="nope", args={}))
    assert result.is_error
