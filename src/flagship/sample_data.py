"""A built-in knowledge base + golden eval set, so the assistant works out of the box."""

from __future__ import annotations

SAMPLE_DOCS: dict[str, str] = {
    "rag": "Retrieval-Augmented Generation grounds an LLM in retrieved documents, so it "
    "answers from your data with citations instead of from parametric memory.",
    "agents": "An agent uses the ReAct loop: reason, call a tool, observe the result, and repeat — "
    "with a step budget to prevent runaway loops.",
    "guardrails": "Guardrails defend against prompt injection by filtering input and validating "
    "output; indirect prompt injection is the new XSS.",
    "evaluation": "Evaluation uses a ship gate: a versioned dataset, a numeric score, and a "
    "regression alarm to block changes that hurt quality.",
}

GOLDEN: list[dict[str, str]] = [
    {
        "id": "rag",
        "input": "What does RAG do?",
        "reference": "grounds an LLM in retrieved documents",
    },
    {"id": "agent", "input": "What loop do agents use?", "reference": "ReAct"},
    {"id": "guard", "input": "What do guardrails defend against?", "reference": "prompt injection"},
    {"id": "eval", "input": "What is a ship gate?", "reference": "versioned dataset"},
]
