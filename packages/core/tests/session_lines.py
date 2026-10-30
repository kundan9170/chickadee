"""A session-log line given as its payload, for tests that write what no typed writer would.

`Session.append` takes a typed line, so a test that must write a line no writer can
produce — a malformed one the reparse has to refuse, or a kind whose writer does not exist
yet (`design_record`, `note`) — writes it through `raw`. The line still goes through the
append's render, reparse and check, so a refusal is the same refusal a writer would meet.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, ClassVar

from pydantic import JsonValue, model_serializer, model_validator, with_config

from chickadee.context.lines import LINE, Line, SessionKind


@cache
def _of(kind: SessionKind) -> type[Any]:
    @with_config(LINE)
    @dataclass(frozen=True)
    class Raw:
        KIND: ClassVar[SessionKind] = kind
        payload: dict[str, JsonValue]

        @model_validator(mode="before")
        @classmethod
        def _wrap(cls, value: object) -> object:
            return {"payload": value}

        @model_serializer(mode="plain")
        def _unwrap(self) -> dict[str, JsonValue]:
            return self.payload

    return Raw


def raw(kind: SessionKind, payload: dict[str, Any]) -> Line:
    """`payload` as a `kind` line, exactly as given."""
    line: Line = _of(kind)(payload)
    return line
