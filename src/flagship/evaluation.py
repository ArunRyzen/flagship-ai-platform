"""End-to-end evaluation + CI ship-gate over the assistant.

Scores the assistant's answers against a golden set and fails the gate below a threshold — the same
"versioned dataset + numeric score + regression alarm" discipline, applied to the whole pipeline.

Beginner note: think of this as unit tests for *answer quality*. A "golden set" is a small,
version-controlled list of (question, expected-answer-fragment) pairs. `run_eval` asks the real
assistant every question and grades each answer; `gate` turns the overall score into a hard
pass/fail. CI runs `flagship eval`, so a code change that quietly makes answers worse (say,
someone breaks retrieval) drops the pass rate below the threshold and the build goes red —
BEFORE the regression ships.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from flagship.errors import GateFailure


class EvalCase(BaseModel):
    id: str
    input: str
    reference: str


class Dataset(BaseModel):
    name: str
    cases: list[EvalCase] = Field(default_factory=list)


class EvalReport(BaseModel):
    dataset: str
    threshold: float
    passed_cases: list[str] = Field(default_factory=list)
    failed_cases: list[str] = Field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.passed_cases) + len(self.failed_cases)

    @property
    def pass_rate(self) -> float:
        return len(self.passed_cases) / self.n if self.n else 0.0

    @property
    def passed(self) -> bool:
        return self.pass_rate >= self.threshold


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _case_passes(output: str, reference: str) -> bool:
    """Grade one answer, cheaply and deterministically (no LLM judge needed here).

    Pass if the expected fragment appears verbatim in the answer, OR if the two share
    at least half their words (Jaccard overlap ≥ 0.5) — a forgiving-but-honest match.
    """
    if reference.strip().lower() in output.lower():
        return True
    a, b = _tokens(output), _tokens(reference)
    union = a | b
    return (len(a & b) / len(union) if union else 0.0) >= 0.5


def run_eval(
    answer_fn: Callable[[str], str], dataset: Dataset, *, threshold: float = 0.75
) -> EvalReport:
    report = EvalReport(dataset=dataset.name, threshold=threshold)
    for case in dataset.cases:
        if _case_passes(answer_fn(case.input), case.reference):
            report.passed_cases.append(case.id)
        else:
            report.failed_cases.append(case.id)
    return report


def gate(report: EvalReport) -> bool:
    """The ship gate: raise (→ non-zero exit → red CI) if the pass rate is below threshold."""
    if not report.passed:
        raise GateFailure(
            f"{report.dataset}: pass_rate {report.pass_rate:.2f} < threshold {report.threshold:.2f}"
        )
    return True
