"""The `Assistant` — the integrated pipeline that composes the whole stack.

One `ask` runs: **guardrails** (block injection / redact PII) → a **ReAct agent** that calls
**hybrid RAG retrieval** as a tool → a grounded, cited **answer** — all under a **trace**. This is
where the five milestones meet behind one method.
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
        self._knowledge = knowledge
        self._policy = policy
        self._max_steps = max_steps

    def ingest(self, doc_id: str, text: str) -> int:
        chunks = chunk_text(doc_id=doc_id, text=text)
        self._retriever.add(chunks)
        return len(chunks)

    def ask(self, question: str, *, tracer: Tracer | None = None) -> Answer:
        tracer = tracer or Tracer()
        with tracer.span("ask", question=question):
            with tracer.span("guardrails") as gspan:
                guard = guard_input(question)
                gspan.set(blocked=guard.blocked, notes=guard.notes)
            if guard.blocked:
                return Answer(
                    question=question,
                    text="Request blocked by input guardrails (possible prompt injection).",
                    blocked=True,
                    guard_notes=guard.notes,
                )

            self._knowledge.last_results = []
            with tracer.span("agent") as aspan:
                text, history = run_agent(
                    task=guard.text,
                    registry=self._registry,
                    policy=self._policy,
                    max_steps=self._max_steps,
                )
                aspan.set(steps=len(history))

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
