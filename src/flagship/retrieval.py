"""Hybrid RAG retrieval, condensed: chunk → embed → dense + BM25 → Reciprocal Rank Fusion.

The same pattern as the standalone RAG project, compacted into one module so the agent can use it
as a tool. Offline by default (a hashing embedder); set a Gemini (or OpenAI) key for real
embeddings.

Beginner note — why "hybrid"? Two different searchers look at every query:
  * **Dense** search compares *meanings* using embedding vectors (finds "car" for "automobile"),
  * **BM25 (sparse)** search compares *exact words* like a classic search engine (great for
    names, error codes, and rare terms embeddings may blur).
Each produces its own ranked list; **RRF** (Reciprocal Rank Fusion) merges the two lists by
rank position, so a chunk that scores well in *both* rises to the top.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Protocol

import numpy as np

from flagship.debuglog import log_block
from flagship.models import Chunk, RetrievedChunk

if TYPE_CHECKING:
    from openai import OpenAI

_TOKEN = re.compile(r"[a-z0-9]+")
_K1, _B = 1.5, 0.75  # standard BM25 tuning constants (term-frequency saturation, length norm)


def _tokenize(text: str) -> list[str]:
    # Lowercase and split into simple word tokens — used by BM25 and the hashing embedder.
    return _TOKEN.findall(text.lower())


def chunk_text(*, doc_id: str, text: str, size: int = 600) -> list[Chunk]:
    """Split a document into ~600-character chunks (retrieval works on chunks, not documents).

    Why chunk at all? Embedding a whole book into one vector blurs its meaning, and an LLM
    only needs the few relevant paragraphs. Each chunk gets a stable id like "mydoc::2"
    (document id + position) so citations can point back to the exact passage.
    """
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
    """The seam: anything that turns a list of texts into a list of number-vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _log_embed_request(model: str, texts: list[str]) -> None:
    """LLM_DEBUG block for an embedding call: which model, how many texts, short previews."""
    previews = " | ".join(t[:80] for t in texts[:5])
    log_block("AI REQUEST (embeddings)", model=model, text_count=len(texts), previews=previews)


def _log_embed_response(vectors: list[list[float]]) -> None:
    """LLM_DEBUG block for the result: only the shape — raw vectors would be noise."""
    dims = len(vectors[0]) if vectors else 0
    log_block("AI RESPONSE (embeddings)", vector_count=len(vectors), dims=dims)


class HashingEmbedder:
    """Deterministic offline bag-of-words embedder (real lexical similarity, no key).

    Each word is hashed to one of `dim` buckets and counted; the vector is then normalized.
    Texts sharing many words get similar vectors. It can't understand synonyms like a real
    embedding model, but it is free, instant, and identical on every run — perfect for tests.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        _log_embed_request("HashingEmbedder (offline, no API call)", texts)
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in _tokenize(text):
                # md5 here is a bucket-picker, not security — hence the noqa.
                idx = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "big") % self.dim  # noqa: S324
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        _log_embed_response(out)
        return out


class GeminiEmbedder:
    """Real semantic embeddings via Google Gemini (`gemini-embedding-001`).

    Used automatically when GEMINI_API_KEY is set. Same `embed` shape as HashingEmbedder,
    so the Retriever never knows (or cares) which one it holds.
    """

    def __init__(self, *, model: str, api_key: str | None) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        _log_embed_request(f"Gemini {self._model}", texts)
        resp = self._client.models.embed_content(model=self._model, contents=texts)
        vectors = [list(e.values or []) for e in resp.embeddings or []]
        _log_embed_response(vectors)
        return vectors


class OpenAIEmbedder:
    """Real semantic embeddings via OpenAI (used only when no Gemini key is present)."""

    def __init__(self, *, model: str, api_key: str | None) -> None:
        from openai import OpenAI

        self._client: OpenAI = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        _log_embed_request(f"OpenAI {self._model}", texts)
        resp = self._client.embeddings.create(model=self._model, input=texts)
        vectors = [d.embedding for d in resp.data]
        _log_embed_response(vectors)
        return vectors


# --- Retriever ----------------------------------------------------------------------


def _rrf(rankings: list[list[RetrievedChunk]], rrf_k: int = 60) -> list[RetrievedChunk]:
    """Merge several ranked lists into one, using only each item's *position* in each list.

    A chunk ranked 1st contributes 1/(60+1), ranked 2nd 1/(60+2), and so on; scores add up
    across lists. Position-based fusion sidesteps the problem that dense scores (cosine)
    and BM25 scores live on completely different numeric scales.
    """
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
        self._candidate_k = candidate_k  # how many hits each ranker feeds into fusion
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # one embedding row per chunk (dense index)
        self._tokens: list[list[str]] = []  # tokenized chunks (BM25 index)
        self._df: Counter[str] = Counter()  # document frequency: how many chunks contain a term
        self._avgdl = 0.0  # average chunk length in tokens (BM25 length normalization)

    def add(self, chunks: list[Chunk]) -> None:
        """Index new chunks: embed them for dense search and tokenize them for BM25."""
        if not chunks:
            return
        vecs = np.asarray(self._embedder.embed([c.text for c in chunks]), dtype=np.float32)
        # Normalize each row to length 1 so a plain dot product below equals cosine similarity.
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
        """Meaning-based search: embed the query, score every chunk by cosine similarity."""
        if self._matrix is None:
            return []
        q = np.asarray(self._embedder.embed([query])[0], dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        scores = self._matrix @ q  # one similarity score per indexed chunk
        top = np.argsort(-scores)[:k]
        return [RetrievedChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top]

    def _sparse(self, query: str, k: int) -> list[RetrievedChunk]:
        """Keyword search (BM25): reward chunks containing the query's words, with two twists —
        rare words count more than common ones (idf), and short chunks beat long rambling ones
        (length normalization)."""
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
        """The public entry point: run both searchers, fuse their rankings, return the top k."""
        dense = self._dense(query, self._candidate_k)
        sparse = self._sparse(query, self._candidate_k)
        return _rrf([dense, sparse])[:k]
