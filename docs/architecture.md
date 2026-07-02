# Architecture & Design Decisions

How the five milestones compose into one service. Read alongside the source.

## The request path
`Assistant.ask(question)` runs a single, traced path:

```
question ─▶ guardrails ─▶ ReAct agent ──knowledge_search──▶ hybrid RAG ─▶ grounded, cited answer
              (block /        (reason → act → observe,            (dense + BM25
               redact)         step budget)                        → RRF)
```

The API wraps that with a **semantic cache**, **rate limiting**, and **/metrics**; CI wraps it with
an **eval ship-gate**.

## Why a single `Assistant` over five services
Each capability is proven in its own repo; the capstone's job is to show they **compose** without
becoming a tangle. The trick is that every layer is a small interface:

- `Embedder` (hashing | Gemini | OpenAI), `Policy` (scripted | heuristic | Gemini | Anthropic),
  `Tool` / `ToolRegistry`, `Tracer`. Swapping any implementation touches only `factory.py`.

So the integrated pipeline stays readable, and the whole thing runs **offline** for tests (hashing
embedder + heuristic policy) — the same testability principle from every milestone.

## Key decisions

### 1. Guardrails run first, before the agent
Input is sanitized (PII redacted) and injection is blocked **before** the agent or any tool executes
— so a hidden instruction in the input can't steer the agent. Blocked requests short-circuit with
`steps == 0`.

### 2. RAG is a *tool*, not a fixed step
The agent decides when to call `knowledge_search`. That's the agentic shape: the model controls the
trajectory, and the retriever is one capability it can use (and the tool records what it retrieved so
the answer carries citations).

### 3. The eval gate scores the *whole* pipeline
`run_eval` calls `Assistant.ask` over a golden set and `gate()` fails CI below threshold. It's not a
unit test of one layer — it's a regression alarm on the integrated behavior, and it runs in CI.

### 4. Tracing spans the composition
One `Tracer` records nested spans (`ask` → `guardrails` → `agent`), so a request is inspectable end
to end. Same shape you'd export to Langfuse/Phoenix in production.

### 5. Serving concerns live at the edge
Caching, rate limiting, and metrics are in the API layer, not the library — so the core stays a clean
library and the transport owns operational concerns.

## What's intentionally condensed
This repo *re-implements* compact versions of retrieval, agent, guardrails, eval, and tracing rather
than importing five packages — so it's a self-contained showcase. The standalone repos (linked in the
README) go deeper on each (LangGraph + MCP for agents, pgvector + reranking for RAG, LLM-as-judge for
evals, QLoRA for fine-tuning).

## If this were a real product
Swap the in-memory store for pgvector; the heuristic policy for LangGraph + Claude tool-use; the
pattern guardrails for a model classifier; the in-memory cache/limiter for Redis; and export traces
to a real backend. Every one of those is a one-interface change — which is the point.
