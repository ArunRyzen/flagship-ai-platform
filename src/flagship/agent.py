"""The ReAct agent: it uses RAG retrieval as a tool, with a swappable Policy and a step budget.

This is the M3 pattern wired to the M2 retriever: `knowledge_search` is a tool backed by the hybrid
retriever, so the agent *decides* when to look things up. The Policy seam keeps it testable offline
(ScriptedPolicy / HeuristicPolicy) and real with live tool-use (GeminiPolicy / AnthropicPolicy).

Beginner note — the mental model for this whole file:
  * A **Tool** is just a named Python function the agent is allowed to call.
  * A **Policy** is the "brain": given the task and what happened so far, it decides the next move —
    either "call these tools" or "here is my final answer". The brain is swappable: a scripted one
    for tests, a rule-based one for offline use, or a real LLM (Gemini/Claude) for live use.
  * `run_agent` is the **loop** that alternates brain-decision → tool-execution until the brain
    says "done" or we run out of steps (the step budget).
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
    """One capability the agent may use: a name, a description the LLM reads to decide when
    to use it, a JSON-schema describing its arguments, and the Python function to run."""

    name: str
    description: str
    parameters: dict
    func: Callable[..., str]


class ToolRegistry:
    """The phone book of tools. The agent loop hands it a ToolCall ("name + args") and it
    runs the matching function, catching errors so one bad call never crashes the loop."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            # The LLM asked for a tool we don't have — report it as an error result
            # instead of raising, so the agent can see the mistake and recover.
            return ToolResult(id=call.id, name=call.name, output="Unknown tool", is_error=True)
        try:
            return ToolResult(id=call.id, name=call.name, output=str(tool.func(**call.args)))
        except Exception as exc:
            return ToolResult(id=call.id, name=call.name, output=f"Error: {exc}", is_error=True)

    def to_anthropic_tools(self) -> list[dict]:
        """Describe our tools in the shape the Anthropic API expects (`input_schema`)."""
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in self._tools.values()
        ]

    def to_gemini_tools(self) -> list[dict]:
        """Describe our tools in the shape the Gemini API expects (`parameters`).

        Same information as `to_anthropic_tools`, different envelope — each provider
        names the schema field differently.
        """
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]


# A whitelist of the only arithmetic operators the calculator tool will evaluate.
_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _calc(node: ast.AST) -> float:
    # Walk the parsed expression tree and only allow numbers and the four operators above.
    # This is why the calculator is safe: `eval("__import__('os')...")` is impossible here.
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
    """A retrieval-backed tool that remembers what it returned (for citations).

    This is the bridge between the agent and the RAG retriever: when the agent calls
    `knowledge_search`, this object runs hybrid retrieval and formats the hits as
    "[doc_id] text" lines for the LLM to read. It also stashes the raw hits in
    `last_results` so the pipeline can turn them into citations afterwards.
    """

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
    """Wire up the two tools this assistant ships with: knowledge search and a calculator."""
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
    """The seam: anything with a `decide(task, history) -> Decision` method can drive the agent."""

    def decide(self, task: str, history: list[Step]) -> Decision: ...


class ScriptedPolicy:
    """Test double: plays back a pre-written list of decisions, one per loop turn."""

    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = decisions

    def decide(self, task: str, history: list[Step]) -> Decision:
        idx = len(history)  # turn number == how many steps have already happened
        return self._decisions[idx] if idx < len(self._decisions) else Decision(finish="(done)")


class HeuristicPolicy:
    """Offline policy: look it up (or calculate), then answer from the result.

    No LLM at all — simple rules: on the first turn, call the calculator if the task
    looks like math, otherwise search the knowledge base; on the second turn, answer
    with whatever the tool returned. Deterministic, free, and good enough for tests.
    """

    def decide(self, task: str, history: list[Step]) -> Decision:
        if history:
            # Turn 2+: we already ran a tool. Answer from its first output line.
            last = history[-1].results[-1].output if history[-1].results else ""
            return Decision(
                finish=f"Based on the knowledge base: {last.splitlines()[0] if last else ''}"
            )
        if re.search(r"\d\s*[-+*/]\s*\d", task):
            # Looks like "2 + 2" → extract the math part and call the calculator.
            expr = re.search(r"[-+*/()\d.\s]+", task)
            arg = expr.group().strip() if expr else task
            return Decision(
                tool_calls=[ToolCall(id="c0", name="calculator", args={"expression": arg})]
            )
        # Default first move: search the knowledge base with the question itself.
        return Decision(
            tool_calls=[ToolCall(id="c0", name="knowledge_search", args={"query": task})]
        )


# The system prompt every live policy uses: it tells the model to ground its answers in
# retrieved context (the anti-hallucination instruction) and keep replies short.
_SYSTEM = (
    "You are a helpful assistant with tools. Use knowledge_search to ground answers in the "
    "knowledge base; answer only from retrieved context or say you don't know. Be concise."
)


class GeminiPolicy:
    """Live policy backed by Google Gemini tool-use (`google-genai` SDK).

    Same job as HeuristicPolicy — turn (task, history) into a Decision — but the choice is
    made by gemini-2.5-flash. We *disable* the SDK's automatic function calling on purpose:
    the SDK would otherwise run tools and loop internally, and we want OUR `run_agent` loop
    to stay in charge (so the step budget, tracing, and citations keep working).
    """

    def __init__(self, *, registry: ToolRegistry, model: str, max_tokens: int, api_key: str | None):
        self._registry, self._model, self._max_tokens, self._api_key = (
            registry,
            model,
            max_tokens,
            api_key,
        )

    def _contents(self, task: str, history: list[Step]) -> list[Any]:
        """Replay the conversation so far in Gemini's format.

        Gemini's transcript alternates: user text → model function_call parts → user
        function_response parts → ... The LLM is stateless, so every turn we resend the
        whole story and ask "what next?".
        """
        from google.genai import types

        contents: list[Any] = [types.Content(role="user", parts=[types.Part(text=task)])]
        for step in history:
            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part(function_call=types.FunctionCall(name=c.name, args=dict(c.args)))
                        for c in step.tool_calls
                    ],
                )
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=r.name,
                                response={"error" if r.is_error else "result": r.output},
                            )
                        )
                        for r in step.results
                    ],
                )
            )
        return contents

    def decide(self, task: str, history: list[Step]) -> Decision:
        from google import genai
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            max_output_tokens=self._max_tokens,
            tools=[types.Tool(function_declarations=cast(Any, self._registry.to_gemini_tools()))],
            # Keep the agent loop in run_agent: never let the SDK call tools by itself.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            resp = genai.Client(api_key=self._api_key).models.generate_content(
                model=self._model, contents=self._contents(task, history), config=config
            )
        except errors.APIError as exc:
            raise PolicyError(f"Gemini request failed: {exc}") from exc

        # Read the model's reply parts: function_call parts mean "run these tools",
        # text parts mean "here is my final answer".
        candidate = resp.candidates[0] if resp.candidates else None
        parts = list(candidate.content.parts or []) if candidate and candidate.content else []
        calls = [
            ToolCall(
                # Gemini doesn't always assign call ids, so we synthesize stable ones.
                id=part.function_call.id or f"g{len(history)}-{i}",
                name=part.function_call.name or "",
                args=dict(part.function_call.args or {}),
            )
            for i, part in enumerate(parts)
            if part.function_call is not None
        ]
        if calls:
            return Decision(tool_calls=calls)
        text = "".join(part.text for part in parts if part.text)
        return Decision(finish=text or "(no answer)")


class AnthropicPolicy:
    """Live policy backed by Claude tool-use (kept for users with an Anthropic key)."""

    def __init__(self, *, registry: ToolRegistry, model: str, max_tokens: int, api_key: str | None):
        self._registry, self._model, self._max_tokens, self._api_key = (
            registry,
            model,
            max_tokens,
            api_key,
        )

    def _messages(self, task: str, history: list[Step]) -> list[dict]:
        # Same replay idea as GeminiPolicy._contents, in Anthropic's message format:
        # assistant "tool_use" blocks followed by user "tool_result" blocks.
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
    """Run the ReAct loop until the policy finishes or the step budget is hit.

    This is the entire agent in five lines: ask the brain what to do; if it says "done",
    return its answer; otherwise run the requested tools, append the (calls, results) pair
    to `history`, and ask again. `max_steps` is the safety fuse — without it a confused
    policy could call tools forever (and burn money on a live API).
    """
    history: list[Step] = []
    for _ in range(max_steps):
        decision = policy.decide(task, history)
        if decision.is_final:
            return decision.finish or "", history
        results = [registry.execute(c) for c in decision.tool_calls]
        history.append(Step(tool_calls=decision.tool_calls, results=results))
    return "(stopped: reached max steps)", history
