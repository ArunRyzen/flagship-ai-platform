"""Input guardrails: block prompt injection, redact PII. The first thing every request hits.

Pattern-based and deterministic (a fast first layer; production adds a model classifier). Indirect
prompt injection is "the new XSS" — so the assistant sanitizes input before the agent ever runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PII = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,2}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}
_INJECTION = [
    re.compile(r"ignore\s+(all\s+|the\s+)?(previous|above|prior)\s+(instructions|prompts)", re.I),
    re.compile(r"disregard\s+(the\s+)?(previous|above|system)", re.I),
    re.compile(r"reveal\s+(your\s+)?(instructions|system\s+prompt|prompt)", re.I),
    re.compile(r"forget\s+(everything|all|your\s+instructions)", re.I),
    re.compile(r"you\s+are\s+now\b", re.I),
]


@dataclass
class GuardResult:
    text: str  # sanitized (PII redacted)
    notes: list[str] = field(default_factory=list)
    blocked: bool = False


def guard_input(text: str) -> GuardResult:
    """Redact PII and block prompt-injection attempts."""
    notes: list[str] = []
    sanitized = text
    for label, pattern in _PII.items():
        if pattern.search(sanitized):
            notes.append(f"redacted:{label}")
            sanitized = pattern.sub(f"[REDACTED_{label}]", sanitized)
    injection = any(p.search(text) for p in _INJECTION)
    if injection:
        notes.append("prompt_injection")
    return GuardResult(text=sanitized, notes=notes, blocked=injection)
