"""Configuration from the environment / `.env`. Runs fully offline with no keys.

Beginner note: this file is the single "control panel" for the whole platform. Every knob
(which model to use, how many chunks to retrieve, how many agent steps to allow) lives here.
Pydantic's `BaseSettings` reads each field from an environment variable of the same name
(e.g. `GEMINI_API_KEY` fills `gemini_api_key`), falling back to a `.env` file, falling back
to the default written below. No key set → every default keeps the system offline and free.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Live models (all optional) ---------------------------------------------------
    # Gemini is the preferred live provider: one free GEMINI_API_KEY powers BOTH the
    # agent's decision-making (gemini-2.5-flash) and real embeddings (gemini-embedding-001).
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-flash")
    gemini_embedding_model: str = Field(default="gemini-embedding-001")

    # Embeddings via OpenAI (used only if no Gemini key; else offline hashing embedder).
    openai_api_key: str | None = Field(default=None)
    embedding_model: str = Field(default="text-embedding-3-small")

    # Agent policy selector: "gemini" or "anthropic" enables live tool-use *if* the matching
    # key is present; anything else (or no key) falls back to the offline heuristic policy.
    policy: str = Field(default="gemini")
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-opus-4-8")
    max_tokens: int = Field(default=1024)  # cap on how long a live model's reply may be

    # --- Retrieval + agent + gate knobs ------------------------------------------------
    top_k: int = Field(default=4)  # how many chunks the knowledge_search tool returns
    candidate_k: int = Field(default=12)  # how many candidates each ranker feeds into fusion
    max_steps: int = Field(default=6)  # the agent's step budget (prevents runaway loops)
    eval_threshold: float = Field(default=0.75)  # ship-gate: min pass-rate to allow a release

    # --- Serving ------------------------------------------------------------------------
    rate_limit_max: int = Field(default=60)  # requests allowed per client...
    rate_limit_window_s: float = Field(default=60.0)  # ...within this many seconds


def load_settings() -> Settings:
    return Settings()
