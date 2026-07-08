"""LLM_DEBUG — a learning aid that prints every AI request/response as a plain-text block.

Set the environment variable `LLM_DEBUG=1` and every model call site (the agent's policy
calls, the embedders — offline fakes included) prints a labelled block to **stderr**, so you
can watch the pipeline think without touching a debugger. Unset it and the platform is
completely silent again.

Beginner note — why stderr? The CLI prints answers on stdout; keeping debug chatter on
stderr means you can still pipe/redirect the real output cleanly. And why plain ASCII
blocks instead of a logging framework? Because the point is to *read* them, top to bottom,
like a story: `=== AI REQUEST ... ===` then `=== AI RESPONSE ... ===`.

Safety: callers never pass API keys into `log_block`, and any oversized field is truncated
to ~2000 characters so a huge document can't flood your terminal.
"""

from __future__ import annotations

import os
import sys

_MAX_FIELD_CHARS = 2000


def debug_enabled() -> bool:
    """True when the LLM_DEBUG env var is set to anything truthy ("0"/"false" don't count)."""
    value = os.environ.get("LLM_DEBUG", "")
    return value.strip().lower() not in {"", "0", "false"}


def _clip(value: object) -> str:
    """Render one field as text, truncating anything over ~2000 chars."""
    text = str(value)
    if len(text) > _MAX_FIELD_CHARS:
        return text[:_MAX_FIELD_CHARS] + "... [truncated]"
    return text


def log_block(title: str, **fields: object) -> None:
    """Print one `=== TITLE ===` block with `key: value` lines to stderr (no-op when off)."""
    if not debug_enabled():
        return
    lines = [f"=== {title} ==="]
    lines.extend(f"{key}: {_clip(value)}" for key, value in fields.items())
    print("\n".join(lines) + "\n", file=sys.stderr)
