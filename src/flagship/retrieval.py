"""Hybrid RAG retrieval, condensed: chunk → embed → dense + BM25 → Reciprocal Rank Fusion.

The same pattern as the standalone RAG project, compacted into one module so the agent can use it
as a tool. Offline by default (a hashing embedder); set an OpenAI key for real embeddings.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Protocol

import numpy as np

from flagship.models import Chunk, RetrievedChunk

if TYPE_CHECKING:
    from openai import OpenAI

_TOKEN = re.compile(r"[a-z0-9]+")
_K1, _B = 1.5, 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def chunk_text(*, doc_id: str, text: str, size: int = 600) -> list[Chunk]:
    words = text.split()
    chunks, buf, idx = [], [], 0
    for word in words:
        buf.append(word)
        if len(" ".join(buf)) >= size:
            chunks.append(
                Chunk(id=f"{doc_id}::{idx}", doc_id=doc_id, text=" ".join(buf), index=idx)
            )
            buf, idx = [], idx + 1
    if buf:
        chunks.append(Chunk(id=f"{doc_id}::{idx}", doc_id=doc_id, text=" ".join(buf), index=idx))
    return chunks or [Chunk(id=f"{doc_id}::0", doc_id=doc_id, text=text, index=0)]


# --- Embedders ----------------------------------------------------------------------


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic offline bag-of-words embedder (real lexical similarity, no key)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in _tokenize(text):
                idx = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "big") % self.dim  # noqa: S324
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class OpenAIEmbedder:
    def __init__(self, *, model: str, api_key: str | None) -> None:
        from openai import OpenAI

        self._client: OpenAI = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


# --- Retriever ----------------------------------------------------------------------


def _rrf(rankings: list[list[RetrievedChunk]], rrf_k: int = 60) -> list[RetrievedChunk]:
    fused: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item.chunk.id] = fused.get(item.chunk.id, 0.0) + 1.0 / (rrf_k + rank)
            best.setdefault(item.chunk.id, item)
    order = sorted(fused.items(), key=lambda kv: -kv[1])
    return [RetrievedChunk(chunk=best[cid].chunk, score=s, source="hybrid") for cid, s in order]


class Retriever:
    """Hybrid (dense + BM25) retrieval with RRF fusion over an in-memory index."""

    def __init__(self, embedder: Embedder, *, candidate_k: int = 12) -> None:
        self._embedder = embedder
        self._candidate_k = candidate_k
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._tokens: list[list[str]] = []
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = np.asarray(self._embedder.embed([c.text for c in chunks]), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])
        for chunk in chunks:
            toks = _tokenize(chunk.text)
            self._chunks.append(chunk)
            self._tokens.append(toks)
            for term in set(toks):
                self._df[term] += 1
        lengths = [len(t) for t in self._tokens]
        self._avgdl = sum(lengths) / len(lengths) if lengths else 0.0

    def _dense(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._matrix is None:
            return []
        q = np.asarray(self._embedder.embed([query])[0], dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        scores = self._matrix @ q
        top = np.argsort(-scores)[:k]
        return [RetrievedChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top]

    def _sparse(self, query: str, k: int) -> list[RetrievedChunk]:
        n = len(self._chunks)
        if not n:
            return []
        q_terms = _tokenize(query)
        scored = []
        for chunk, toks in zip(self._chunks, self._tokens, strict=True):
            tf, dl, score = Counter(toks), len(toks), 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                denom = tf[term] + _K1 * (1 - _B + _B * dl / (self._avgdl or 1))
                score += idf * (tf[term] * (_K1 + 1)) / denom
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=score))
        scored.sort(key=lambda r: -r.score)
        return scored[:k]

    def retrieve(self, query: str, *, k: int = 4) -> list[RetrievedChunk]:
        dense = self._dense(query, self._candidate_k)
        sparse = self._sparse(query, self._candidate_k)
        return _rrf([dense, sparse])[:k]
