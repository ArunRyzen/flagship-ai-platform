"""Composition root: build the fully-wired `Assistant` from settings (offline-capable defaults)."""

from __future__ import annotations

from flagship.agent import (
    AnthropicPolicy,
    HeuristicPolicy,
    KnowledgeTool,
    Policy,
    ToolRegistry,
    build_registry,
)
from flagship.config import Settings
from flagship.pipeline import Assistant
from flagship.retrieval import Embedder, HashingEmbedder, OpenAIEmbedder, Retriever


def build_embedder(settings: Settings) -> Embedder:
    if settings.openai_api_key:
        return OpenAIEmbedder(model=settings.embedding_model, api_key=settings.openai_api_key)
    return HashingEmbedder()


def build_policy(settings: Settings, registry: ToolRegistry) -> Policy:
    if settings.policy == "anthropic" and settings.anthropic_api_key:
        return AnthropicPolicy(
            registry=registry,
            model=settings.anthropic_model,
            max_tokens=settings.max_tokens,
            api_key=settings.anthropic_api_key,
        )
    return HeuristicPolicy()


def build_assistant(settings: Settings) -> Assistant:
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
