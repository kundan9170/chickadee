"""Previews: the bounded, redacted text a tool or model record carries for a reader.

A record's shas are its identity; a preview is what a person reads in the REPL while the box
works: what a call asked, what it was given back, what a model turn said. Never compared,
never replayed from.

**Redacted before it is clipped** (`secrets.redact`): a secret cut in half by the clip would
slip past the detectors and half of it would be written. A preview is written to the journal,
so this is the boundary SE-4 asks for, not a display filter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Final, cast

from chickadee.edits.lines import clip
from chickadee.secrets.redact import redact_text

__all__ = ["PREVIEW_CHARS", "args_preview", "preview"]

PREVIEW_CHARS: Final = 400
"""What one preview holds, before the clip's marker: enough to read a call's arguments and the
head of its result, and a fixed bound on what each record adds to the journal."""

_WINDOW: Final = 16 * 1024
"""How much of a text is redacted before the clip. Far past `PREVIEW_CHARS`, so placeholders
shorter than the secrets they replace cannot pull an unredacted character into view, and a large
file read is not scanned whole for 400 characters of it."""


def preview(text: str) -> str:
    """`text`, redacted, then cut to `PREVIEW_CHARS` and marked with what was cut."""
    redacted, _ = redact_text(text[:_WINDOW])
    beyond = len(text) - len(text[:_WINDOW])
    if not beyond:
        return clip(redacted, PREVIEW_CHARS)
    hidden = len(redacted) - PREVIEW_CHARS + beyond
    return f"{redacted[:PREVIEW_CHARS]}…[+{hidden:,} chars]"


def args_preview(arguments: Mapping[str, object]) -> str:
    """A call's arguments as compact, key-sorted JSON, previewed. Each string is redacted with
    its key beside it first (`token=…`): in JSON the key is quoted, and a detector that reads
    `token = …` does not match `"token": …`."""
    return preview(
        json.dumps(_named(arguments), sort_keys=True, ensure_ascii=False, default=str)
    )


def _named(value: object) -> object:
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in cast("Mapping[object, object]", value).items():
            name = str(key)
            if isinstance(item, str):
                redacted, found = redact_text(f"{name}={item}")
                out[name] = redacted[len(name) + 1 :] if found else item
            else:
                out[name] = _named(item)
        return out
    if isinstance(value, list | tuple):
        return [_named(v) for v in cast("Sequence[object]", value)]
    return value
