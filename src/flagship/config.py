"""Configuration from the environment / `.env`. Runs fully offline with no keys."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Embeddings (offline hashing embedder if no key).
    openai_api_key: str | None = Field(default=None)
    embedding_model: str = Field(default="text-embedding-3-small")

    # Agent policy: "anthropic" for real tool-use, else the offline heuristic.
    policy: str = Field(default="anthropic")
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-opus-4-8")
    max_tokens: int = Field(default=1024)

    # Retrieval + agent + gate knobs.
    top_k: int = Field(default=4)
    candidate_k: int = Field(default=12)
    max_steps: int = Field(default=6)
    eval_threshold: float = Field(default=0.75)

    # Serving.
    rate_limit_max: int = Field(default=60)
    rate_limit_window_s: float = Field(default=60.0)


def load_settings() -> Settings:
    return Settings()
