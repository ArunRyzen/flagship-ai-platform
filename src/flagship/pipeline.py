"""The `Assistant` — the integrated pipeline that composes the whole stack.

One `ask` runs: **guardrails** (block injection / redact PII) → a **ReAct agent** that calls
**hybrid RAG retrieval** as a tool → a grounded, cited **answer** — all under a **trace**. This is
where the five milestones meet behind one method.

Beginner note: if you read only one file in this repo, read `ask` below — it is the whole
product in ~30 lines. Everything else in the codebase exists to serve one of its five moments:
screen the input, let the agent think, let it search, attach the receipts, record the timeline.
"""

from __future__ import annotations

from flagship.agent import KnowledgeTool, Policy, ToolRegistry, run_agent
from flagship.guardrails import guard_input
from flagship.models import Answer, Citation
from flagship.retrieval import Retriever, chunk_text
from flagship.tracing import Tracer


class Assistant:
    def __init__(
        self,
        *,
        retriever: Retriever,
        registry: ToolRegistry,
        knowledge: KnowledgeTool,
        policy: Policy,
        max_steps: int = 6,
    ) -> None:
        self._retriever = retriever
        self._registry = registry
        self._knowledge = knowledge  # kept separately so we can read its last_results for citations
        self._policy = policy
        self._max_steps = max_steps  # the agent's step budget, passed straight to run_agent

    def ingest(self, doc_id: str, text: str) -> int:
        """Add a document to the knowledge base: split it into chunks, index every chunk."""
        chunks = chunk_text(doc_id=doc_id, text=text)
        self._retriever.add(chunks)
        return len(chunks)

    def ask(self, question: str, *, tracer: Tracer | None = None) -> Answer:
        """Answer one question end to end. The five boxes, in order, live right here."""
        tracer = tracer or Tracer()
        with tracer.span("ask", question=question):  # Box 5: the trace wraps everything
            # Box 1 — guardrail screening: sanitize/deny BEFORE any model or tool runs.
            with tracer.span("guardrails") as gspan:
                guard = guard_input(question)
                gspan.set(blocked=guard.blocked, notes=guard.notes)
            if guard.blocked:
                # Refuse early: the agent never sees an injected prompt, no tokens are spent.
                return Answer(
                    question=question,
                    text="Request blocked by input guardrails (possible prompt injection).",
                    blocked=True,
                    guard_notes=guard.notes,
                )

            # Boxes 2 + 3 — the agent loop. The policy decides to call knowledge_search
            # (box 2); executing that tool runs hybrid retrieval (box 3). We clear
            # last_results first so citations can only come from THIS question.
            self._knowledge.last_results = []
            with tracer.span("agent") as aspan:
                text, history = run_agent(
                    task=guard.text,  # note: the *sanitized* text, not the raw question
                    registry=self._registry,
                    policy=self._policy,
                    max_steps=self._max_steps,
                )
                aspan.set(steps=len(history))

            # Box 4 — citation assembly: whatever the knowledge tool retrieved last
            # becomes the answer's receipts (chunk id + document id per source).
            citations = [
                Citation(chunk_id=r.chunk.id, doc_id=r.chunk.doc_id)
                for r in self._knowledge.last_results
            ]
            return Answer(
                question=question,
                text=text,
                citations=citations,
                steps=len(history),
                blocked=False,
                guard_notes=guard.notes,
            )
