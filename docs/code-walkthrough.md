# Code Walkthrough — a beginner's tour of the whole codebase

This document assumes you can read Python but have never built an LLM system. It explains
every file in plain English, gives you a reading order, and — most importantly — traces **one
real question through the five boxes** of the pipeline so you can see exactly where each thing
happens.

---

## The big picture in one sentence

> A question walks in; a **guardrail** frisks it; an **agent** (a decision-maker) chooses to
> **search the knowledge base**; the search results come back; the answer is written **with
> citations**; and a **tracer** wrote down everything that happened, with timings.

Everything in `src/flagship/` exists to serve one of those clauses.

## Recommended reading order

Read the files in this order — each one only depends on the ones before it:

| # | File | What you'll learn | Lines |
|---|------|-------------------|-------|
| 1 | `src/flagship/models.py` | The nouns: `Chunk`, `ToolCall`, `Decision`, `Answer`... | tiny |
| 2 | `src/flagship/guardrails.py` | Box 1 — screening input | small |
| 3 | `src/flagship/retrieval.py` | Box 3 — hybrid search (dense + BM25 + RRF) | medium |
| 4 | `src/flagship/agent.py` | Box 2 — tools, policies, and the ReAct loop | medium |
| 5 | `src/flagship/tracing.py` | Box 5 — spans and the trace tree | small |
| 6 | `src/flagship/pipeline.py` | **All five boxes composed** — the heart of the repo | small |
| 7 | `src/flagship/evaluation.py` | The ship gate (quality tests for answers) | small |
| 8 | `src/flagship/factory.py` | How the pieces get chosen and wired together | small |
| 9 | `src/flagship/config.py` | Every knob, and which env var sets it | small |
| 10 | `src/flagship/cache.py`, `ratelimit.py`, `api.py` | The serving edge (HTTP concerns) | small |
| 11 | `src/flagship/cli.py`, `sample_data.py`, `errors.py` | The CLI, built-in docs, exceptions | tiny |

---

## ⭐ One question through the five boxes

This is the section to study. We follow **one** call:

```python
assistant.ask("What do guardrails defend against?")
```

Every box below lists *the file and function where it happens*, so you can put a breakpoint
(or a `print`) on each and watch the question travel. The composed path lives entirely inside
**`src/flagship/pipeline.py` → `Assistant.ask`** — the boxes are just its paragraphs.

### Box 1 — Guardrail screening
**Where:** `pipeline.py` → `Assistant.ask` (the first `with tracer.span("guardrails")` block),
calling **`guardrails.py` → `guard_input(question)`**.

The raw question is checked before *anything* else runs:

```python
guard = guard_input(question)
if guard.blocked:
    return Answer(..., blocked=True, ...)   # refuse; the agent never runs (steps == 0)
```

`guard_input` does two things in one pass: it rewrites PII (emails, phone numbers, SSNs) into
`[REDACTED_*]` placeholders, and it scans for prompt-injection phrases like "ignore all
previous instructions". Injection → `blocked=True` → the pipeline returns a refusal
immediately. Our question is clean, so we continue — and note that the agent will receive
`guard.text` (the *sanitized* text), never the raw input.

### Box 2 — The agent decides to call `knowledge_search`
**Where:** `pipeline.py` → `Assistant.ask` (the `with tracer.span("agent")` block) calls
**`agent.py` → `run_agent`**, which asks the **policy** to decide:
- offline: `agent.py` → `HeuristicPolicy.decide`
- live Gemini: `agent.py` → `GeminiPolicy.decide`

`run_agent` is a loop: *ask the brain → run the tools it asked for → repeat*. On turn one,
the policy sees an empty history and (heuristic or LLM alike) decides the right move is:

```python
Decision(tool_calls=[ToolCall(id="c0", name="knowledge_search", args={"query": task})])
```

That `Decision` is not an answer — it's a request: "please run this tool for me". The loop
obliges via `ToolRegistry.execute(call)` (also in `agent.py`).

### Box 3 — Hybrid retrieval runs
**Where:** executing the tool call lands in **`agent.py` → `KnowledgeTool.__call__`**, which
calls **`retrieval.py` → `Retriever.retrieve(query, k=4)`**.

`retrieve` runs two searches over the indexed chunks and merges them:

```python
dense = self._dense(query, self._candidate_k)     # meaning-based (embedding cosine similarity)
sparse = self._sparse(query, self._candidate_k)   # keyword-based (BM25)
return _rrf([dense, sparse])[:k]                  # fuse the two rankings, keep the top k
```

The winning chunks (here, the "guardrails" doc) are formatted as `[doc_id] text` lines and
returned to the agent loop as the tool's output. Crucially, `KnowledgeTool` also stores the
raw hits in `self.last_results` — that's the breadcrumb Box 4 will pick up. On the *next*
loop turn the policy sees the tool output in `history` and returns
`Decision(finish="Based on the knowledge base: ...")` — the loop ends after **1 step**.

### Box 4 — Citations attached
**Where:** back in **`pipeline.py` → `Assistant.ask`**, right after `run_agent` returns:

```python
citations = [
    Citation(chunk_id=r.chunk.id, doc_id=r.chunk.doc_id)
    for r in self._knowledge.last_results
]
```

The pipeline reads what the knowledge tool retrieved *last* and turns each hit into a
`Citation` (defined in `models.py`) — e.g. `chunk_id="guardrails::0", doc_id="guardrails"`.
Notice the pipeline set `self._knowledge.last_results = []` *before* running the agent: that
reset guarantees citations can only come from this question, never a previous one.

### Box 5 — Every step traced
**Where:** the spans opened throughout `Assistant.ask`, implemented in
**`tracing.py` → `Tracer.span`**.

Boxes 1–4 all happened *inside* `with tracer.span(...)` blocks, so the request produced a
nested timeline tree:

```json
[{ "name": "ask", "attributes": {"question": "..."},
   "children": [
     { "name": "guardrails", "attributes": {"blocked": false, "notes": []} },
     { "name": "agent",      "attributes": {"steps": 1} }
   ] }]
```

See it yourself: `uv run flagship trace "What do guardrails defend against?"` (the `trace`
command in `cli.py` prints this JSON), or `POST /ask` — the API returns the trace in the
response body.

---

## The agent step budget (you will experiment with this)

The budget is the `max_steps` loop bound in **`agent.py` → `run_agent`**:

```python
for _ in range(max_steps):          # ← the fuse
    decision = policy.decide(task, history)
    ...
return "(stopped: reached max steps)", history   # budget exhausted → give up gracefully
```

Where the number comes from, end to end:
`MAX_STEPS` env var → `config.py` → `Settings.max_steps` (default **6**) →
`factory.py` → `build_assistant(...)` → `pipeline.py` → `Assistant(max_steps=...)` →
`run_agent(max_steps=self._max_steps)`.

**To set it to 1:** put `MAX_STEPS=1` in your `.env` (or pass `max_steps=1` when constructing
`Assistant`). With a budget of 1 the agent can call `knowledge_search` once but never gets a
second turn to phrase its answer — so `ask` returns `"(stopped: reached max steps)"`.
Predict what `flagship eval` will say before you run it.

## How `flagship eval` catches a planted regression

The gate lives in **`evaluation.py`** and the command in **`cli.py` → `evaluate`**:

1. `sample_data.py` → `GOLDEN` is the versioned golden set — 4 questions with expected
   answer fragments (e.g. *"What do guardrails defend against?"* → `"prompt injection"`).
2. `run_eval` asks the real assistant every question and grades each answer with
   `_case_passes` (fragment appears verbatim, or ≥ 50% word overlap).
3. `gate(report)` raises `GateFailure` when the pass rate drops below the threshold
   (default 0.75 = at least 3 of the 4 must pass), and `cli.py` converts that into
   **exit code 1** — which is exactly what fails the `Eval ship-gate` step in
   `.github/workflows/ci.yml`.

**Try planting a regression:** break retrieval on purpose — e.g. in `retrieval.py` →
`Retriever.retrieve`, return `[]` instead of the fused results. Every answer becomes
"Based on the knowledge base: " with no facts, `_case_passes` fails all 4 cases, and:

```
$ uv run flagship eval
[FAIL] pass_rate=0.00 (threshold 0.75, n=4)
flagship-golden: pass_rate 0.00 < threshold 0.75
$ echo $?   # → 1  (this non-zero exit is what turns CI red)
```

Note what just happened: no unit test of `retrieve` had to anticipate this bug. The gate
measures *end-to-end answer quality*, so anything that quietly degrades answers — a bad
prompt, a broken ranker, an over-eager guardrail — trips the same alarm. Revert your change
and the gate goes green again.

---

## File-by-file tour

### `src/flagship/models.py` — the shared vocabulary
Small Pydantic classes every other file speaks in. `Chunk` (a piece of a document),
`RetrievedChunk` (a chunk + its search score), `ToolCall`/`ToolResult` (what the agent asks
for / what it got back), `Step` (one loop turn: calls + results), `Decision` (the policy's
verdict: either `finish` text or more `tool_calls`), `Citation`, and `Answer` (the final
product: text + citations + steps + guardrail notes). Start here; everything else is verbs.

### `src/flagship/guardrails.py` — Box 1
Two regex tables — `_PII` (what to redact) and `_INJECTION` (what to block) — and one
function, `guard_input`, that applies both and reports what it did in `notes`. Deterministic
on purpose: it's the cheap, fast first layer (production would add an ML classifier behind it).

### `src/flagship/retrieval.py` — Box 3
The search engine. `chunk_text` splits documents into ~600-character pieces with stable ids
(`"mydoc::2"`). Three interchangeable `Embedder`s turn text into vectors: `HashingEmbedder`
(offline word-counting — free and deterministic), `GeminiEmbedder` (`gemini-embedding-001`,
used when `GEMINI_API_KEY` is set), `OpenAIEmbedder` (fallback). The `Retriever` keeps two
indexes over the same chunks — an embedding matrix for *meaning* search (`_dense`) and token
lists for *keyword* search (`_sparse`, BM25) — and `_rrf` merges the two ranked lists by rank
position. `retrieve` = run both, fuse, return top-k.

### `src/flagship/agent.py` — Box 2
Four ideas in one file:
1. **Tools** — `Tool` (name + description + schema + function) and `ToolRegistry` (looks up
   and safely runs a `ToolCall`; errors become results, not crashes). Two tools are
   registered: `knowledge_search` and a `calculator` (a safe AST-walking evaluator — no
   `eval`).
2. **`KnowledgeTool`** — the agent-facing wrapper around the `Retriever`; remembers
   `last_results` so the pipeline can build citations (the Box 3 → Box 4 handoff).
3. **Policies** — the swappable "brain" behind the `Policy` protocol: `ScriptedPolicy`
   (tests), `HeuristicPolicy` (offline rules), `GeminiPolicy` (live `gemini-2.5-flash`
   tool-use; note it *disables the SDK's automatic function calling* so our loop stays in
   charge of budgets, tracing, and citations), `AnthropicPolicy` (live Claude).
4. **`run_agent`** — the ReAct loop itself, with the `max_steps` fuse.

### `src/flagship/tracing.py` — Box 5
`Span` = a named stopwatch with attributes and children; `Tracer.span(...)` is a `with`-block
that starts/stops it and nests it under whichever span is currently open. `to_list()` dumps
the tree as JSON-friendly dicts.

### `src/flagship/pipeline.py` — all five boxes
`Assistant.ingest` chunks + indexes a document. `Assistant.ask` is the composition — read it
top to bottom and you'll recognize each box from the traced walkthrough above.

### `src/flagship/evaluation.py` — the ship gate
`Dataset`/`EvalCase` (the golden set), `run_eval` (ask + grade every case), `EvalReport`
(pass rate), `gate` (raise below threshold). See the regression exercise above.

### `src/flagship/factory.py` — the wiring room
The only file that knows about concrete providers. `build_embedder`: Gemini key → Gemini,
else OpenAI key → OpenAI, else hashing. `build_policy`: Gemini key → `GeminiPolicy`, else
Anthropic key (+`POLICY=anthropic`) → `AnthropicPolicy`, else heuristic. `build_assistant`
assembles retriever → knowledge tool → registry → policy → `Assistant`.

### `src/flagship/config.py` — the control panel
One `Settings` class; each field maps to an env var (`GEMINI_API_KEY`, `MAX_STEPS`, ...) with
offline-safe defaults. Nothing else in the codebase reads `os.environ`.

### `src/flagship/cache.py`, `ratelimit.py`, `api.py` — the serving edge
`SemanticCache`: embeds queries and returns a stored answer when a new query is ≥ 0.97
cosine-similar (a paraphrase hit skips the whole pipeline). `RateLimiter`: sliding window of
timestamps per client. `api.py` (FastAPI) applies them in cost order in `POST /ask` —
rate limit → cache → real pipeline — and exposes `/health`, `/metrics`, `/eval`. Blocked
answers are never cached.

### `src/flagship/cli.py`, `sample_data.py`, `errors.py`
`cli.py`: the `flagship` command (`ask`, `eval`, `guard`, `trace`). `sample_data.py`: 4 tiny
built-in docs + the `GOLDEN` eval set, so everything works with zero setup. `errors.py`: the
exception family (`GateFailure` is the one CI cares about).

### `tests/` — 26 offline tests
One test file per concern, mirroring the modules. `conftest.py` builds a fully offline
assistant *and* deletes any real API keys from the environment per test, so the suite is
identical on your laptop and in CI. `test_gemini.py` covers the live Gemini seams with
**mocked** clients — verifying our request shaping, response parsing, and factory selection
without any network.

---

## Where to find X

| X | File | Symbol |
|---|------|--------|
| The whole pipeline in one method | `src/flagship/pipeline.py` | `Assistant.ask` |
| Prompt-injection block / PII redaction | `src/flagship/guardrails.py` | `guard_input`, `_INJECTION`, `_PII` |
| The agent loop (ReAct) | `src/flagship/agent.py` | `run_agent` |
| **The agent step budget** | `src/flagship/agent.py` / `config.py` / `.env` | `run_agent(max_steps=...)` ← `Settings.max_steps` ← `MAX_STEPS` |
| The offline "brain" | `src/flagship/agent.py` | `HeuristicPolicy.decide` |
| The live Gemini "brain" | `src/flagship/agent.py` | `GeminiPolicy.decide` |
| Where a tool call is executed | `src/flagship/agent.py` | `ToolRegistry.execute` |
| The `knowledge_search` tool | `src/flagship/agent.py` | `KnowledgeTool.__call__`, `build_registry` |
| Hybrid search (dense + BM25 + fusion) | `src/flagship/retrieval.py` | `Retriever.retrieve`, `_dense`, `_sparse`, `_rrf` |
| Chunking (and chunk ids for citations) | `src/flagship/retrieval.py` | `chunk_text` |
| Embedders (offline / Gemini / OpenAI) | `src/flagship/retrieval.py` | `HashingEmbedder`, `GeminiEmbedder`, `OpenAIEmbedder` |
| **Citation assembly** | `src/flagship/pipeline.py` | end of `Assistant.ask` (reads `KnowledgeTool.last_results`) |
| Spans / the trace tree | `src/flagship/tracing.py` | `Tracer.span`, `Span.to_dict` |
| **The eval ship gate** | `src/flagship/evaluation.py` | `run_eval`, `_case_passes`, `gate` |
| The golden dataset | `src/flagship/sample_data.py` | `GOLDEN` |
| The `flagship eval` command (exit 1) | `src/flagship/cli.py` | `evaluate` |
| CI runs the gate | `.github/workflows/ci.yml` | step "Eval ship-gate" |
| Which provider gets picked (and why) | `src/flagship/factory.py` | `build_embedder`, `build_policy` |
| All env vars / defaults | `src/flagship/config.py` + `.env.example` | `Settings` |
| Semantic cache | `src/flagship/cache.py` | `SemanticCache.get` / `.put` |
| Rate limiting | `src/flagship/ratelimit.py` | `RateLimiter.allow` |
| HTTP endpoints | `src/flagship/api.py` | `ask`, `metrics`, `evaluate`, `health` |
| Offline-test guarantee | `tests/conftest.py` | `_offline_env`, `make_assistant` |
| Mocked Gemini tests | `tests/test_gemini.py` | all |

---

## Suggested first experiments

1. `uv run flagship trace "What is RAG?"` — read the span tree, match it to the five boxes.
2. Set `MAX_STEPS=1` in `.env` and re-ask — watch the step budget bite; then run
   `uv run flagship eval` and explain the result.
3. Plant the regression from the eval section, watch the gate fail, revert.
4. Add a Gemini key to `.env` and re-run `flagship ask` — same code path, real brain.
