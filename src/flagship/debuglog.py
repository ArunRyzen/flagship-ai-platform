"""LLM_DEBUG — a learning aid that prints every AI request/response as a plain-text block.

Set the environment variable `LLM_DEBUG=1` — or put `LLM_DEBUG=1` in your project's `.env`
file — and every model call site (the agent's policy calls, the embedders — offline fakes
included) prints a labelled block to **stderr**, so you can watch the pipeline think without
touching a debugger. Unset it and the platform is completely silent again.

Precedence: a *real* environment variable always wins when it is set (even `LLM_DEBUG=0`,
which force-disables); only when the variable is completely unset do we fall back to
reading `LLM_DEBUG` from a `.env` file in the current working directory.

Beginner note — why stderr? The CLI prints answers on stdout; keeping debug chatter on
stderr means you can still pipe/redirect the real output cleanly. And why plain ASCII
blocks instead of a logging framework? Because the point is to *read* them, top to bottom,
like a story: `=== AI REQUEST ... ===` then `=== AI RESPONSE ... ===`.

Safety: callers never pass API keys into `log_block`, and any oversized field is truncated
to ~2000 characters so a huge document can't flood your terminal.
"""

from __future__ import annotations

import functools
import os
import sys

_MAX_FIELD_CHARS = 2000


def _is_truthy(value: str) -> bool:
    """Shared truthiness rule: anything except ""/"0"/"false" (case-insensitive) counts."""
    return value.strip().lower() not in {"", "0", "false"}


@functools.lru_cache(maxsize=1)
def _dotenv_debug_value() -> str:
    """Read `LLM_DEBUG` from a `.env` file in the current working directory (or "").

    Beginner notes:
    - `dotenv_values` parses the file *without* mutating `os.environ`, so this stays a
      read-only fallback — it can never shadow real environment variables elsewhere.
    - The import is lazy and wrapped in try/except: `python-dotenv` arrives as a
      transitive dependency of `pydantic-settings`, but if it's ever missing the feature
      simply degrades to "disabled" instead of crashing the whole platform.
    - `lru_cache` means we hit the filesystem once per process, not on every model call;
      tests call `_dotenv_debug_value.cache_clear()` when they change `.env` or the cwd.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return ""
    return dotenv_values(".env", encoding="utf-8-sig").get("LLM_DEBUG") or ""


def debug_enabled() -> bool:
    """True when LLM_DEBUG is truthy, checking the real env var first, then `.env`.

    Precedence (beginner-friendly): if the *real* environment variable `LLM_DEBUG` is set
    at all, it decides — so `LLM_DEBUG=0` on the command line force-disables debug even if
    your `.env` says `LLM_DEBUG=1`. Only when the variable is completely unset do we fall
    back to the `.env` file in the current directory.
    """
    env_value = os.environ.get("LLM_DEBUG")
    if env_value is not None:
        return _is_truthy(env_value)
    return _is_truthy(_dotenv_debug_value())


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
