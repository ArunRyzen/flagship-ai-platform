"""HTTP surface: health, ask (+ cache + trace), metrics, eval."""

from __future__ import annotations

from fastapi.testclient import TestClient

from flagship import api

client = TestClient(api.app)


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_with_trace() -> None:
    resp = client.post("/ask", json={"question": "What does RAG do?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"]
    assert body["cached"] is False
    assert isinstance(body["trace"], list)


def test_repeated_question_is_cached() -> None:
    q = {"question": "What loop do agents use?"}
    client.post("/ask", json=q)
    second = client.post("/ask", json=q).json()
    assert second["cached"] is True


def test_metrics_and_eval() -> None:
    assert "cache_hit_rate" in client.get("/metrics").json()
    body = client.get("/eval").json()
    assert body["passed"] is True  # the assistant clears the gate on the golden set


def test_ask_rejects_empty() -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422
