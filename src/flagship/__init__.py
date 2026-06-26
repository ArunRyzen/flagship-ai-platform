"""flagship-ai-platform — the capstone that composes the whole portfolio into one service.

A single request flows: **guardrails** (block injection, redact PII) → a **ReAct agent** that uses
**hybrid RAG retrieval** as a tool → a grounded, cited **answer** — with the whole path **traced**,
the system **evaluated behind a CI gate**, and the API **cached + rate-limited + observable**.

Each capability mirrors a milestone project; here they're integrated behind one `Assistant`.
"""

from flagship.models import Answer
from flagship.pipeline import Assistant

__version__ = "0.1.0"

__all__ = ["Assistant", "Answer", "__version__"]
