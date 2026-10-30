"""A turn's `bound` lines.

The bounds themselves are the shared rules Zeus's pipeline runs (`tools/catalogue.py`:
`CallCap`, `ResultTokenCap`, `PerResponse`, `Repetition`). What stays here is Zeus's
own record of meeting them: the session log holds a turn's bound once per resource and
outcome, with the count, never a line per refused call. A per-response refusal (a second
`coder`, `locate` past the fan-out) writes none: the response's refusal is its own record.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

from pydantic import Field, with_config

from chickadee.context.lines import LINE, SessionKind
from chickadee.tools.rules import BoundReached

__all__ = ["BoundLine", "BoundLines"]

_UNSET = Field(default=None, exclude_if=lambda v: v is None)


@with_config(LINE)
@dataclass(frozen=True)
class BoundLine:
    """The `bound` line: a turn's bound, once per resource and outcome, with
    the count. A repetition names the key it counted under; every other line has no key."""

    KIND: ClassVar[SessionKind] = "bound"
    resource: str
    outcome: str
    used: int
    limit: int
    scope: Literal["turn"] = "turn"
    tool: Annotated[str | None, _UNSET] = None
    args_digest: Annotated[str | None, _UNSET] = None
    generation: Annotated[int | None, _UNSET] = None


class BoundLines:
    """One turn's `bound` lines, written through `write` (`Session.append`), each resource and
    outcome once. A new turn is a new `BoundLines`."""

    def __init__(self, write: Callable[[BoundLine], object]) -> None:
        self._write = write
        self._written: set[tuple[str, str]] = set()

    def refused(self, bound: BoundReached, key: tuple[str, str, int] | None) -> None:
        """A turn's bound refused a call (D98's `2N`). `key` is the repetition's
        `(tool, args_digest, generation)`; no other resource names one."""
        if bound.scope != "turn":
            return
        self._line(bound.resource, "refused", bound.used, bound.limit, key)

    def noticed(self, count: int, limit: int, key: tuple[str, str, int]) -> None:
        """The `N`th identical call against an unchanged workspace was served with a
        note."""
        self._line("repetition", "noticed", count, limit, key)

    def _line(
        self,
        resource: str,
        outcome: str,
        used: int,
        bound: int,
        key: tuple[str, str, int] | None,
    ) -> None:
        if (resource, outcome) in self._written:
            return
        self._written.add((resource, outcome))
        line = BoundLine(resource, outcome, used, bound)
        if resource == "repetition" and key is not None:
            line = dataclasses.replace(line, tool=key[0], args_digest=key[1], generation=key[2])
        self._write(line)
