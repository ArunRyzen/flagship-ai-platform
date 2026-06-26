"""Hybrid retrieval finds the relevant document; chunking is stable."""

from __future__ import annotations

from flagship.retrieval import HashingEmbedder, Retriever, chunk_text
from flagship.sample_data import SAMPLE_DOCS


def test_chunking_produces_ids() -> None:
    chunks = chunk_text(doc_id="d", text="one two three four five")
    assert chunks[0].id == "d::0"
    assert chunks[0].doc_id == "d"


def test_hybrid_retrieves_relevant_doc() -> None:
    retriever = Retriever(HashingEmbedder())
    for doc_id, text in SAMPLE_DOCS.items():
        retriever.add(chunk_text(doc_id=doc_id, text=text))
    results = retriever.retrieve("what do guardrails defend against?", k=2)
    assert results
    assert any(r.chunk.doc_id == "guardrails" for r in results)
