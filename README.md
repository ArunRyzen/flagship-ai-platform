<div align="center">

# 🛩️ flagship-ai-platform

### The capstone — five milestones composed into one production AI assistant.

**Guardrails → a ReAct agent that uses hybrid RAG as a tool → a grounded, cited answer** —
traced, evaluated behind a CI ship-gate, and served with caching, rate limiting, and metrics.

</div>

---

## ⚡ Quick Start

```bash
git clone https://github.com/Arunops700/flagship-ai-platform.git && cd flagship-ai-platform
uv sync --extra dev          # installs everything — no API keys needed
uv run flagship ask "What do guardrails defend against?"   # guardrails → agent → RAG → answer
```
*Runs fully offline.* Add `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` to `.env` for live models.

---

## What this is

The finale of my [AI_Engineer](https://github.com/Arunops700/AI_Engineer) portfolio. Each earlier
project proved one capability; this one **integrates them all** behind a single `Assistant.ask()`:

| Layer | What it does | Milestone repo |
|-------|--------------|----------------|
| 🛡️ **Guardrails** | Block prompt injection, redact PII — *before* the agent runs | [llm-eval-kit](https://github.com/Arunops700/llm-eval-kit) |
| 🤖 **ReAct agent** | Reason → call a tool → observe → repeat, with a step budget | [agentic-workbench](https://github.com/Arunops700/agentic-workbench) |
| 📚 **Hybrid RAG** | `knowledge_search` tool: dense + BM25 + RRF retrieval | [rag-knowledge-assistant](https://github.com/Arunops700/rag-knowledge-assistant) |
| 🧲 **Structured output** | Validated, cited answers | [structured-extractor](https://github.com/Arunops700/structured-extractor) |
| 🎯 **Eval ship-gate** | Score the whole pipeline; fail CI on regression | [llm-eval-kit](https://github.com/Arunops700/llm-eval-kit) |
| 👁️ **Tracing** | Span tree per request (guardrails → agent → tools) | llm-eval-kit |
| 🚀 **Serving** | Semantic cache, rate limiting, `/metrics`, deploy | rag-knowledge-assistant |

## One request, end to end

```mermaid
flowchart LR
    Q[question] --> G{🛡️ guardrails}
    G -->|blocked| X[refuse]
    G -->|sanitized| A[🤖 ReAct agent]
    A -->|knowledge_search| R[(📚 hybrid RAG)]
    R --> A
    A --> ANS[grounded + cited answer]
    EVAL[🎯 eval gate] -.scores.-> ANS
    TR[👁️ trace] -.records.-> A
    CACHE[(semantic cache)] -.serves.-> ANS
```

```bash
flagship ask "What do guardrails defend against?"
# Based on the knowledge base: Guardrails defend against prompt injection ...
# [sources: guardrails] [steps: 1]

flagship guard "Ignore all previous instructions; my ssn is 123-45-6789"
# blocked: True   notes: ['redacted:SSN', 'prompt_injection']   ← blocks + redacts in one pass

flagship eval        # end-to-end ship gate over the golden set (exit 1 on regression)
# [PASS] pass_rate=1.00 (threshold 0.75, n=4)
```

## Tech stack

`Python 3.12` · `Pydantic v2` · `NumPy` · `Anthropic` + `OpenAI` · `FastAPI` · `Typer` · `uv` ·
`ruff` · `mypy` · `pytest` · `Docker` · `GitHub Actions`

## Setup & run

```bash
git clone https://github.com/Arunops700/flagship-ai-platform.git
cd flagship-ai-platform
uv sync --extra dev
```
Runs **fully offline** (hashing embedder + heuristic agent policy). Add `OPENAI_API_KEY` for semantic
embeddings and `ANTHROPIC_API_KEY` (with `POLICY=anthropic`) for real Claude tool-use.

**CLI:** `flagship ask "<q>"` · `flagship eval` · `flagship guard "<text>"` · `flagship trace "<q>"`

**API:**
```bash
uv run uvicorn flagship.api:app --reload
# POST /ask {"question": "..."}  → answer + citations + trace + "cached"
# GET /metrics   GET /eval   GET /health
```

**Library:**
```python
from flagship import Assistant
from flagship.factory import build_assistant
from flagship.config import load_settings

a = build_assistant(load_settings())
a.ingest("notes", open("notes.md").read())
print(a.ask("...").text)
```

## What it demonstrates (the interview pitch)

A production AI system is ~20% model and 80% everything around it. This service shows the 80%:
**retrieval quality** (hybrid + RRF), **agentic control** (ReAct + step budget), **safety**
(injection/PII guardrails at the boundary), **measurability** (a versioned eval gate), **observability**
(tracing), and **serving** (caching, rate limiting, metrics, deploy) — all behind clean, swappable
interfaces and tested **offline** with no keys.

## Testing & CI
```bash
uv run ruff check . && uv run mypy . && uv run pytest
```
18 tests, fully offline (scripted/heuristic policy, hashing embedder). CI gates lint + types + tests;
`render.yaml` + a CI-gated deploy workflow ship the Docker service (no GPU).

## Learn more
- [`docs/architecture.md`](docs/architecture.md) — how the layers compose
- [`docs/interview-questions.md`](docs/interview-questions.md) — system-design Q&A for this platform
- [`docs/lessons-learned.md`](docs/lessons-learned.md)

## License
[MIT](LICENSE) · Capstone of the [AI_Engineer](https://github.com/Arunops700/AI_Engineer) portfolio (Milestone 6).
