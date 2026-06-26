"""Core domain types shared across the integrated pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    index: int = 0


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    source: str = "hybrid"


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    id: str
    name: str
    output: str
    is_error: bool = False


class Step(BaseModel):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    results: list[ToolResult] = Field(default_factory=list)


class Decision(BaseModel):
    finish: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @property
    def is_final(self) -> bool:
        return self.finish is not None


class Citation(BaseModel):
    chunk_id: str
    doc_id: str


class Answer(BaseModel):
    """The end-to-end result: the answer, what it cited, agent steps, and guardrail notes."""

    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    steps: int = 0
    blocked: bool = False
    guard_notes: list[str] = Field(default_factory=list)
