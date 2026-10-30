"""The argument a call may carry beside its tool's own: Zeus's `findings`, what each earlier
result a call relies on established. One home for every role's surface.

No call carries a purpose line: no `why`, no `intent`. A role other than Zeus is shown
its tools' schemas as they are; a tool decoded under a grammar sends free text, not an object,
and is shown as it is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Final, cast

from chickadee.errors import HarnessBug

if TYPE_CHECKING:
    from chickadee.client.request import ToolSchema

__all__ = [
    "FINDINGS",
    "is_line",
    "shown",
    "split_meta",
    "with_meta",
]

FINDINGS: Final = "findings"
"""Zeus's alone: what each earlier result a call relies on established."""
_FINDINGS_SCHEMA: Final[Mapping[str, object]] = {
    "type": "object",
    "description": (
        "for each earlier result this call relies on, by its call id: what that result "
        "established"
    ),
    "additionalProperties": {"type": "string", "minLength": 1},
}


def is_line(value: object) -> bool:
    """A meta value that says something: a non-blank string."""
    return isinstance(value, str) and bool(value.strip())


def with_meta(schema: Mapping[str, object], *, findings: bool = False) -> dict[str, object]:
    """A tool's schema as the model is shown it: its own arguments, then `findings` (optional)
    for Zeus; any other role's as it is. A tool naming `findings` itself is a harness bug: the
    two would mean different things in one object."""
    raw = schema.get("properties")
    none: Mapping[str, object] = {}
    own = cast("Mapping[str, object]", raw) if isinstance(raw, Mapping) else none
    if not findings:
        return dict(schema)
    if FINDINGS in own:
        raise HarnessBug(f"a tool schema names the meta argument {FINDINGS!r} itself")
    return {**schema, "properties": {**own, FINDINGS: dict(_FINDINGS_SCHEMA)}}


def shown(tools: tuple[ToolSchema, ...], *, findings: bool = False) -> tuple[ToolSchema, ...]:
    """A role's resident schemas with the meta arguments, each as `with_meta` shows it; a
    tool decoded under a grammar as it is."""
    return tuple(
        t if t.grammar else replace(t, schema=with_meta(t.schema, findings=findings))
        for t in tools
    )


def split_meta(args: Mapping[str, object]) -> tuple[dict[str, object], object]:
    """`(the tool's own arguments, findings)`: `findings` as sent, or None when absent. The call
    itself is never edited: this is a view for validating and dispatching."""
    own = {k: v for k, v in args.items() if k != FINDINGS}
    return own, args.get(FINDINGS)
