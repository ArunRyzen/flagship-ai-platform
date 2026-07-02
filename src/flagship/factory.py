"""Composition root: build the fully-wired `Assistant` from settings (offline-capable defaults).

Beginner note: this is the ONLY place that decides which concrete implementation fills each
seam (which embedder, which policy). Everything else codes against the small Protocols
(`Embedder`, `Policy`), so swapping providers means editing these few lines — nothing else.

Selection order, both seams: Gemini key present → Gemini; else OpenAI/Anthropic key → that
provider; else the free offline fake. One GEMINI_API_KEY lights up the whole live path.
"""

from __future__ import annotations

from flagship.agent import (
    AnthropicPolicy,
    GeminiPolicy,
    HeuristicPolicy,
    KnowledgeTool,
    Policy,
    ToolRegistry,
    build_registry,
)
from flagship.config import Settings
from flagship.pipeline import Assistant
from flagship.retrieval import Embedder, GeminiEmbedder, HashingEmbedder, OpenAIEmbedder, Retriever


def build_embedder(settings: Settings) -> Embedder:
    """Pick the embedder: Gemini if its key is set, then OpenAI, else the offline hasher."""
    if settings.gemini_api_key:
        return GeminiEmbedder(
            model=settings.gemini_embedding_model, api_key=settings.gemini_api_key
        )
    if settings.openai_api_key:
        return OpenAIEmbedder(model=settings.embedding_model, api_key=settings.openai_api_key)
    return HashingEmbedder()


def build_policy(settings: Settings, registry: ToolRegistry) -> Policy:
    """Pick the agent's brain: a live LLM when POLICY + a key allow it, else the heuristic.

    Gemini is preferred whenever GEMINI_API_KEY is set (the beginner path — one free key),
    mirroring the original Anthropic selection. With no keys at all this returns the
    HeuristicPolicy, which is why every test and CI run works offline.
    """
    if settings.policy in {"gemini", "anthropic"} and settings.gemini_api_key:
        return GeminiPolicy(
            registry=registry,
            model=settings.gemini_model,
            max_tokens=settings.max_tokens,
            api_key=settings.gemini_api_key,
        )
    if settings.policy == "anthropic" and settings.anthropic_api_key:
        return AnthropicPolicy(
            registry=registry,
            model=settings.anthropic_model,
            max_tokens=settings.max_tokens,
            api_key=settings.anthropic_api_key,
        )
    return HeuristicPolicy()


def build_assistant(settings: Settings) -> Assistant:
    """Assemble the machine: retriever → knowledge tool → tool registry → policy → Assistant."""
    retriever = Retriever(build_embedder(settings), candidate_k=settings.candidate_k)
    knowledge = KnowledgeTool(retriever, k=settings.top_k)
    registry = build_registry(knowledge)
    policy = build_policy(settings, registry)
    return Assistant(
        retriever=retriever,
        registry=registry,
        knowledge=knowledge,
        policy=policy,
        max_steps=settings.max_steps,
    )
