"""FastAPI service: the assistant behind a semantic cache, rate limiting, and tracing.

Beginner note: this is the "edge" of the system — HTTP concerns (rate limits, caching,
metrics) live HERE, while the core logic stays in the library. Order of checks in /ask
matters and is deliberate: rate limit first (cheapest, protects everything), then the
cache (skip all real work on a hit), and only then the full guardrails → agent → RAG run.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from flagship.cache import SemanticCache
from flagship.config import Settings, load_settings
from flagship.evaluation import Dataset, EvalCase, run_eval
from flagship.factory import build_assistant, build_embedder
from flagship.pipeline import Assistant
from flagship.sample_data import GOLDEN, SAMPLE_DOCS
from flagship.tracing import Tracer

app = FastAPI(title="flagship-ai-platform", version="0.1.0")
_metrics = {"ask_requests": 0}  # simple in-process counter, exposed via GET /metrics


# `@lru_cache` on zero-argument functions = build once, reuse forever (a lazy singleton).
# The assistant, cache, and limiter are constructed on the first request, not at import time.
@lru_cache
def _settings() -> Settings:
    return load_settings()


@lru_cache
def _assistant() -> Assistant:
    assistant = build_assistant(_settings())
    for doc_id, text in SAMPLE_DOCS.items():  # preload the built-in docs so /ask works instantly
        assistant.ingest(doc_id, text)
    return assistant


@lru_cache
def _cache() -> SemanticCache:
    return SemanticCache(build_embedder(_settings()))


@lru_cache
def _limiter() -> object:
    from flagship.ratelimit import RateLimiter

    s = _settings()
    return RateLimiter(s.rate_limit_max, s.rate_limit_window_s)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest, http_request: Request) -> dict:
    # 1) Rate limit per client IP — reject floods before doing anything expensive.
    client = http_request.client.host if http_request.client else "unknown"
    limiter = _limiter()
    if not limiter.allow(client):  # type: ignore[attr-defined]
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    # 2) Semantic cache — a similar-enough earlier question returns instantly.
    _metrics["ask_requests"] += 1
    cached = _cache().get(request.question)
    if cached is not None:
        return {**cached, "cached": True}

    # 3) The real work: guardrails → agent → RAG → cited answer, traced end to end.
    tracer = Tracer()
    answer = _assistant().ask(request.question, tracer=tracer)
    payload = {**answer.model_dump(), "trace": tracer.to_list()}
    if not answer.blocked:  # never cache refusals — a blocked probe shouldn't "stick"
        _cache().put(request.question, payload)
    return {**payload, "cached": False}


@app.get("/metrics")
def metrics() -> dict:
    cache = _cache()
    return {
        "ask_requests": _metrics["ask_requests"],
        "cache_hits": cache.stats.hits,
        "cache_misses": cache.stats.misses,
        "cache_hit_rate": round(cache.stats.hit_rate, 3),
        "cache_size": cache.size,
    }


@app.get("/eval")
def evaluate(threshold: float = 0.75) -> dict:
    assistant = _assistant()
    dataset = Dataset(name="flagship-golden", cases=[EvalCase(**c) for c in GOLDEN])
    report = run_eval(lambda q: assistant.ask(q).text, dataset, threshold=threshold)
    return {
        "dataset": report.dataset,
        "pass_rate": report.pass_rate,
        "passed": report.passed,
        "threshold": report.threshold,
        "passed_cases": report.passed_cases,
        "failed_cases": report.failed_cases,
    }
