"""Command-line interface for the integrated assistant."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from flagship.config import load_settings
from flagship.evaluation import Dataset, EvalCase, gate, run_eval
from flagship.factory import build_assistant
from flagship.guardrails import guard_input
from flagship.pipeline import Assistant
from flagship.sample_data import GOLDEN, SAMPLE_DOCS
from flagship.tracing import Tracer

app = typer.Typer(
    help="Flagship AI platform — RAG + agent + guardrails + evals, served.", no_args_is_help=True
)


def _assistant() -> Assistant:
    assistant = build_assistant(load_settings())
    for doc_id, text in SAMPLE_DOCS.items():
        assistant.ingest(doc_id, text)
    return assistant


@app.command()
def ask(question: Annotated[str, typer.Argument(help="Your question.")]) -> None:
    """Answer a question end to end (guardrails → agent+RAG → cited answer)."""
    answer = _assistant().ask(question)
    typer.echo(answer.text)
    if answer.guard_notes:
        typer.echo(f"[guardrails: {answer.guard_notes}]", err=True)
    if answer.citations:
        cites = ", ".join(c.doc_id for c in answer.citations)
        typer.echo(f"[sources: {cites}] [steps: {answer.steps}]", err=True)


@app.command(name="eval")
def evaluate(threshold: Annotated[float, typer.Option(help="Ship-gate threshold.")] = 0.75) -> None:
    """Run the end-to-end eval and apply the ship gate (exit 1 on failure)."""
    assistant = _assistant()
    dataset = Dataset(name="flagship-golden", cases=[EvalCase(**c) for c in GOLDEN])
    report = run_eval(lambda q: assistant.ask(q).text, dataset, threshold=threshold)
    verdict = "PASS" if report.passed else "FAIL"
    typer.echo(
        f"[{verdict}] pass_rate={report.pass_rate:.2f} (threshold {threshold}, n={report.n})"
    )
    typer.echo(f"  passed={report.passed_cases} failed={report.failed_cases}", err=True)
    try:
        gate(report)
    except Exception as exc:  # GateFailure
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def guard(text: Annotated[str, typer.Argument(help="Text to check.")]) -> None:
    """Run input guardrails (prompt-injection + PII) on text."""
    result = guard_input(text)
    typer.echo(f"blocked: {result.blocked}")
    typer.echo(f"notes: {result.notes}")
    typer.echo(f"sanitized: {result.text}")


@app.command()
def trace(question: Annotated[str, typer.Argument(help="Question to trace.")]) -> None:
    """Answer a question and print the execution trace (spans)."""
    tracer = Tracer()
    answer = _assistant().ask(question, tracer=tracer)
    typer.echo(answer.text)
    typer.echo(json.dumps(tracer.to_list(), indent=2), err=True)


if __name__ == "__main__":
    app()
