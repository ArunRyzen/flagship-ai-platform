"""End-to-end evaluation + gate over the real assistant."""

from __future__ import annotations

import pytest

from flagship.errors import GateFailure
from flagship.evaluation import Dataset, EvalCase, gate, run_eval
from flagship.sample_data import GOLDEN
from tests.conftest import make_assistant


def _dataset() -> Dataset:
    return Dataset(name="golden", cases=[EvalCase(**c) for c in GOLDEN])


def test_assistant_passes_the_gate() -> None:
    assistant = make_assistant()
    report = run_eval(lambda q: assistant.ask(q).text, _dataset(), threshold=0.75)
    assert report.passed
    assert gate(report) is True


def test_gate_raises_when_threshold_unmet() -> None:
    # An assistant that always answers wrong fails the gate.
    report = run_eval(lambda q: "irrelevant", _dataset(), threshold=0.75)
    assert not report.passed
    with pytest.raises(GateFailure):
        gate(report)
