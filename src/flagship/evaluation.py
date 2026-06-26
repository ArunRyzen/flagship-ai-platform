"""End-to-end evaluation + CI ship-gate over the assistant.

Scores the assistant's answers against a golden set and fails the gate below a threshold — the same
"versioned dataset + numeric score + regression alarm" discipline, applied to the whole pipeline.
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
    if not report.passed:
        raise GateFailure(
            f"{report.dataset}: pass_rate {report.pass_rate:.2f} < threshold {report.threshold:.2f}"
        )
    return True
