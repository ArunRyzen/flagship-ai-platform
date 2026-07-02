"""A semantic response cache (serves cached answers for similar queries).

Beginner note: a normal cache only hits on the *exact same* string. A semantic cache embeds
each query into a vector and hits when a new query's vector is nearly identical (cosine
similarity ≥ threshold) to a stored one — so "What does RAG do?" can be served from the answer
to "what does rag do ?". Each hit skips the whole agent + retrieval run: faster and cheaper.
The threshold is deliberately high (0.97): serving a *wrong* cached answer is far worse than
missing the cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from flagship.retrieval import Embedder


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class SemanticCache:
    def __init__(self, embedder: Embedder, *, threshold: float = 0.97, max_size: int = 512) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._max_size = max_size
        self._entries: list[tuple[list[float], dict]] = []
        self.stats = CacheStats()

    def get(self, query: str) -> dict | None:
        """Return the stored answer whose query is most similar — if similar *enough*."""
        if not self._entries:
            self.stats.misses += 1
            return None
        emb = self._embedder.embed([query])[0]
        best, best_sim = None, -1.0
        for cached_emb, answer in self._entries:
            sim = _cosine(emb, cached_emb)
            if sim > best_sim:
                best_sim, best = sim, answer
        if best is not None and best_sim >= self._threshold:
            self.stats.hits += 1
            return best
        self.stats.misses += 1
        return None

    def put(self, query: str, answer: dict) -> None:
        """Store (query-embedding, answer); evict the oldest entry once the cache is full."""
        self._entries.append((self._embedder.embed([query])[0], answer))
        if len(self._entries) > self._max_size:
            self._entries.pop(0)

    @property
    def size(self) -> int:
        return len(self._entries)
