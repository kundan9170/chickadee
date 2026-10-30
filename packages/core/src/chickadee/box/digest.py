"""Deterministic evidence digest: a failure's output, its tail
redacted.

Moved from `verifier/digest.py` when the verifier retired: its last reader is
the syntax check's failure (`box/syntax.py`).

Caps exist so a 400-line compiler dump cannot blow the retry prompt. Tails rather than
heads: the diagnostic that names the error is almost always at the end. Same input bytes
produce the same digest bytes — the moving-vs-stable comparison depends on that.

**The digest is cut from redacted bytes.** the secrets rules say digests
pass the redactor. The redaction happens *inside* `build_digest`, so no caller can hand a model
unredacted bytes by forgetting. The redactor is deterministic, so stability comparisons are
untouched; `stdout_sha` is the pre-redaction identity of the blob and is not
this function's.

**A digest is its tail and what it was cut from.** `output` is the whole input,
redacted: what the box journals beside a failing gate's line, and what the worker reads the
rest of by name. The cut is values (`lines`, `kept`), never a marker in the text, so a
reader can say how much was left out and how to reach it.
"""

from __future__ import annotations

from dataclasses import dataclass

from chickadee.edits.lines import split_lines
from chickadee.secrets.redact import redact_text

__all__ = ["MAX_CHARS", "MAX_LINES", "Digest", "build_digest"]

MAX_LINES = 40
MAX_CHARS = 4000


@dataclass(frozen=True)
class Digest:
    """A failure's evidence: the tail a prompt shows, and the whole it was cut from."""

    text: str
    """The `failing:` header when one was given, then the last lines of `output`: at most
    `MAX_LINES` lines and `MAX_CHARS` characters, whole lines except when one line alone is
    over the cap, whose tail is kept."""
    output: str
    """The whole input, redacted, its newlines `\\n`: what `text` was cut from."""
    lines: int
    """Lines in `output`, as `edits.lines.split_lines` numbers them."""
    kept: int
    """How many of those lines `text` shows, the last of them: lines `lines - kept + 1` to
    `lines`. A line shown only in part counts as shown."""
    cut: bool
    """Whether `text` leaves anything of `output` out: a dropped line, or part of one."""


_FAILING = "failing: "


def build_digest(
    raw: str, failing: tuple[str, ...] = (), *, max_chars: int = MAX_CHARS
) -> Digest:
    """The last `MAX_LINES` lines and `max_chars` characters of `raw`, redacted. A smaller
    `max_chars` is a reader's narrower room (the retry's share of a 4 KB budget), cut by
    the same rule."""
    # Redact FIRST, on the whole input, so a token split by the line or character cap
    # cannot survive as a fragment on either side of the cut. The placeholders carry the
    # secret's fingerprint, so "the same token as turn 3" stays recognisable.
    redacted, _ = redact_text(raw)
    output = redacted.replace("\r\n", "\n").replace("\r", "\n")
    header = f"{_FAILING}{', '.join(failing)}\n" if failing else ""
    lines = split_lines(output, keepends=True)
    room = max(min(max_chars, MAX_CHARS) - len(header), 0)
    tail: list[str] = []
    used = 0
    partial = False
    for line in reversed(lines[-MAX_LINES:]):
        if used + len(line) > room:
            if not tail:
                # One line alone over the cap: its tail, which is where an error names itself.
                tail.append(line[-room:] if room else "")
                partial = True
            break
        tail.append(line)
        used += len(line)
    body = "".join(reversed(tail))
    return Digest(
        text=header + body.removesuffix("\n"),
        output=output,
        lines=len(lines),
        kept=len(tail),
        cut=partial or len(tail) < len(lines),
    )
