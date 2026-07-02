"""A lightweight in-memory tracer (spans, duration, tokens, cost) — observability for the pipeline.

Same shape you'd export to Langfuse / Phoenix / OpenTelemetry in production.

Beginner note: a **span** is a stopwatch with a name — "this named piece of work started here,
ended there, and here are some facts about it". Spans nest: the "ask" span contains a
"guardrails" span and an "agent" span, so one request becomes a little timeline tree you can
read to see exactly where time went and what each stage decided.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class Span:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    start: float = 0.0
    end: float | None = None
    children: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return ((self.end or self.start) - self.start) * 1000.0

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
        }


class Tracer:
    def __init__(self) -> None:
        self._roots: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        """Start a span with `with tracer.span("name"):` — it times the block automatically.

        The stack tracks which span is currently "open": a new span becomes a child of the
        one on top (that's how nesting happens), and the `finally` guarantees the stopwatch
        stops even if the code inside raises.
        """
        span = Span(name=name, attributes=dict(attributes), start=perf_counter())
        (self._stack[-1].children if self._stack else self._roots).append(span)
        self._stack.append(span)
        try:
            yield span
        finally:
            span.end = perf_counter()
            self._stack.pop()

    @property
    def roots(self) -> list[Span]:
        return self._roots

    def to_list(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._roots]
