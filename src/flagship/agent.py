"""The ReAct agent: it uses RAG retrieval as a tool, with a swappable Policy and a step budget.

This is the M3 pattern wired to the M2 retriever: `knowledge_search` is a tool backed by the hybrid
retriever, so the agent *decides* when to look things up. The Policy seam keeps it testable offline
(ScriptedPolicy / HeuristicPolicy) and real with Claude tool-use (AnthropicPolicy).
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from flagship.errors import PolicyError
from flagship.models import Decision, RetrievedChunk, Step, ToolCall, ToolResult

if TYPE_CHECKING:
    from flagship.retrieval import Retriever

# --- Tools --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(id=call.id, name=call.name, output="Unknown tool", is_error=True)
        try:
            return ToolResult(id=call.id, name=call.name, output=str(tool.func(**call.args)))
        except Exception as exc:
            return ToolResult(id=call.id, name=call.name, output=f"Error: {exc}", is_error=True)

    def to_anthropic_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in self._tools.values()
        ]


_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _calc(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_calc(node.left), _calc(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_calc(node.operand)
    raise ValueError("arithmetic only")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    return str(_calc(ast.parse(expression, mode="eval").body))


class KnowledgeTool:
    """A retrieval-backed tool that remembers what it returned (for citations)."""

    def __init__(self, retriever: Retriever, *, k: int = 4) -> None:
        self._retriever = retriever
        self._k = k
        self.last_results: list[RetrievedChunk] = []

    def __call__(self, query: str) -> str:
        self.last_results = self._retriever.retrieve(query, k=self._k)
        if not self.last_results:
            return "No relevant information found."
        return "\n".join(f"[{r.chunk.doc_id}] {r.chunk.text}" for r in self.last_results)


def build_registry(knowledge: KnowledgeTool) -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                "knowledge_search",
                "Search the knowledge base for relevant passages.",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                knowledge,
            ),
            Tool(
                "calculator",
                "Evaluate a basic arithmetic expression.",
                {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
                calculator,
            ),
        ]
    )


# --- Policies -----------------------------------------------------------------------


class Policy(Protocol):
    def decide(self, task: str, history: list[Step]) -> Decision: ...


class ScriptedPolicy:
    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = decisions

    def decide(self, task: str, history: list[Step]) -> Decision:
        idx = len(history)
        return self._decisions[idx] if idx < len(self._decisions) else Decision(finish="(done)")


class HeuristicPolicy:
    """Offline policy: look it up (or calculate), then answer from the result."""

    def decide(self, task: str, history: list[Step]) -> Decision:
        if history:
            last = history[-1].results[-1].output if history[-1].results else ""
            return Decision(
                finish=f"Based on the knowledge base: {last.splitlines()[0] if last else ''}"
            )
        if re.search(r"\d\s*[-+*/]\s*\d", task):
            expr = re.search(r"[-+*/()\d.\s]+", task)
            arg = expr.group().strip() if expr else task
            return Decision(
                tool_calls=[ToolCall(id="c0", name="calculator", args={"expression": arg})]
            )
        return Decision(
            tool_calls=[ToolCall(id="c0", name="knowledge_search", args={"query": task})]
        )


_SYSTEM = (
    "You are a helpful assistant with tools. Use knowledge_search to ground answers in the "
    "knowledge base; answer only from retrieved context or say you don't know. Be concise."
)


class AnthropicPolicy:
    def __init__(self, *, registry: ToolRegistry, model: str, max_tokens: int, api_key: str | None):
        self._registry, self._model, self._max_tokens, self._api_key = (
            registry,
            model,
            max_tokens,
            api_key,
        )

    def _messages(self, task: str, history: list[Step]) -> list[dict]:
        messages: list[dict] = [{"role": "user", "content": task}]
        for step in history:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                        for c in step.tool_calls
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.id,
                            "content": r.output,
                            "is_error": r.is_error,
                        }
                        for r in step.results
                    ],
                }
            )
        return messages

    def decide(self, task: str, history: list[Step]) -> Decision:
        import anthropic

        try:
            resp = anthropic.Anthropic(api_key=self._api_key).messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM,
                tools=cast(Any, self._registry.to_anthropic_tools()),
                messages=cast(Any, self._messages(task, history)),
            )
        except anthropic.APIError as exc:
            raise PolicyError(f"Anthropic request failed: {exc}") from exc
        calls = [
            ToolCall(id=b.id, name=b.name, args=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        if calls:
            return Decision(tool_calls=calls)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Decision(finish=text or "(no answer)")


def run_agent(
    *, task: str, registry: ToolRegistry, policy: Policy, max_steps: int = 6
) -> tuple[str, list[Step]]:
    """Run the ReAct loop until the policy finishes or the step budget is hit."""
    history: list[Step] = []
    for _ in range(max_steps):
        decision = policy.decide(task, history)
        if decision.is_final:
            return decision.finish or "", history
        results = [registry.execute(c) for c in decision.tool_calls]
        history.append(Step(tool_calls=decision.tool_calls, results=results))
    return "(stopped: reached max steps)", history
