# Lessons Learned

Notes from building the capstone (Milestone 6).

## Technical
- **Composition is a design skill of its own.** Each capability was already proven; the hard part was
  making them meet behind one `Assistant.ask()` *without* a tangle. Small interfaces at every seam
  (Embedder, Policy, Tool, Tracer) are what made the composition stay readable.
- **Guardrails-first matters.** Sanitizing/blocking before the agent runs — not after — is the
  difference between defense and theater. Blocked requests short-circuit with zero steps.
- **RAG-as-a-tool is the right abstraction.** Letting the agent *decide* to retrieve (vs a fixed
  pre-step) is what makes it agentic, and recording the tool's results gives citations for free.
- **An eval gate over the whole pipeline catches what unit tests can't.** Unit tests check layers;
  the gate checks the *integrated behavior*, and running it in CI makes regressions loud.
- **Offline-first scaled across six projects.** The hashing embedder + scripted/heuristic policy made
  the entire integrated system testable with no keys — the single most valuable pattern of the program.

## Process
- **Condense, don't copy five repos.** Re-implementing compact versions kept the capstone a
  self-contained showcase; the standalone repos go deeper. That's an honest, navigable structure.
- **Library core, edge concerns at the edge.** Caching/rate-limiting/metrics live in the API, not the
  library — so the core stays a clean, reusable `Assistant`.

## If I did it again
- Wire in the real components directly (LangGraph + MCP, pgvector + reranking, LLM-as-judge) behind
  the same interfaces, as installable packages.
- Add streaming responses and a small UI.
- Deploy it live and put the trace export into Langfuse.

## The program, in one line
The portfolio went from "can call an LLM" to "can design, build, evaluate, secure, serve, and reason
about a production AI system" — and this capstone is the proof it composes.
