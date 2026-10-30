"""Another harness's tool calls, in ours: what a refusal names as the call to send.

Most open coding-agent training data was collected under OpenHands, so a small model's
strongest habits are its tool names, its `command` switch, its argument names and absolute
paths under a mount. A call in that shape is still refused, recorded exactly as sent and never
run (a call is never corrected to the nearest tool); its refusal names the call in our
terms instead of only saying no.

One table, OpenHands' tools only, written by hand from OpenHands'
`software-agent-sdk` at commit `3f32945` (`llm/mixins/fn_call_converter.py`, the tool names and
their arguments; `agent/utils.py`, `normalize_tool_call`'s aliases and its inference of a
missing `command`). Plain data: no model and no similarity search, at build or at run time, so
the answer is the same every time. Other harnesses join it only if the count says the idea
works.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, cast

from chickadee.paths import Place
from chickadee.tools.spec import ToolSpec

__all__ = [
    "ROWS",
    "Row",
    "Suggestion",
    "foreign_argument",
    "ours_for",
    "translate",
]


@dataclass(frozen=True)
class Row:
    """One OpenHands call (`tool`, and its `command` for `str_replace_editor`) and ours: the
    tool (None when nothing here does that), its arguments by their name and ours, and what
    the refusal adds: why nothing is to be called, or which of their arguments has no
    equivalent here."""

    tool: str
    command: str | None
    ours: str | None
    args: Mapping[str, str] = field(default_factory=dict[str, str])
    note: str = ""
    dropped: tuple[str, ...] = ()
    """Their arguments with no equivalent here: named in the note when the call sends one."""
    directory: bool = False
    """The row for a `view` of a directory, which is our `glob`, not `read`."""


ROWS: Final[tuple[Row, ...]] = (
    Row("str_replace_editor", "view", "read", {"path": "path", "view_range": "lines"}),
    Row("str_replace_editor", "view", "glob", {"path": "pattern"}, directory=True),
    Row("str_replace_editor", "create", "create", {"path": "path", "file_text": "content"}),
    Row(
        "str_replace_editor",
        "str_replace",
        "edit",
        {"path": "path", "old_str": "old", "new_str": "new"},
    ),
    Row(
        "str_replace_editor",
        "insert",
        "insert_lines",
        {"path": "path", "insert_line": "after", "new_str": "content"},
        note="use after as the line number read shows",
    ),
    Row(
        "str_replace_editor",
        "undo_edit",
        None,
        note="the harness keeps every step; there is nothing to call",
    ),
    Row(
        "execute_bash",
        None,
        "shell",
        {"command": "command"},
        note="no equivalent here",
        dropped=("is_input", "timeout"),
    ),
    Row(
        "bash",
        None,
        "shell",
        {"command": "command"},
        note="no equivalent here",
        dropped=("is_input", "timeout"),
    ),
    Row("think", None, None, note="reason in your reply"),
    Row("finish", None, "finish", {"message": "report"}),
)
"""D209 decision 2's table, exactly."""

_EDITOR: Final = "str_replace_editor"


@dataclass(frozen=True)
class Suggestion:
    """What to send instead: our tool and its arguments in our names, and what that tool takes;
    or, `tool` None, why nothing here is to be called."""

    tool: str | None
    arguments: Mapping[str, object] = field(default_factory=dict[str, object])
    takes: tuple[str, ...] = ()
    note: str = ""

    @property
    def call(self) -> str | None:
        """The call as the model writes it: `read {"path": "config.py"}`."""
        if self.tool is None:
            return None
        return call_text(self.tool, self.arguments)


SHOWN: Final = 80
"""A longer value is shown as `<as you sent it>`: the model has it in its own call, and a
file's content echoed back would cost its length again on every refusal."""


def call_text(tool: str, args: Mapping[str, object]) -> str:
    """A call to send, as the model writes it: `create {"path": "core/ai.py", "content": <as
    you sent it>}`. The one rendering of a suggested call, for both refusals that name one."""
    shown = (
        f"{json.dumps(k)}: "
        + (
            "<as you sent it>"
            if isinstance(v, str) and len(v) > SHOWN
            else json.dumps(v, ensure_ascii=False)
        )
        for k, v in args.items()
    )
    return f"{tool} {{{', '.join(shown)}}}"


def translate(
    name: str,
    args: Mapping[str, object],
    offered: Mapping[str, ToolSpec],
    place: Callable[[str], Place | None] | None = None,
) -> Suggestion | None:
    """The call to send for an OpenHands call (`name`, `args` as sent), in our terms; None when
    the name is not in the table, or its target is not in `offered` (Zeus has no `edit`), so a
    refusal never names a tool the role cannot call. `offered` is the scope's specs by our
    tool's name, each under the name the model is shown it by, which the suggestion uses.

    A missing `command` on `str_replace_editor` is inferred from the arguments present, as
    OpenHands' `normalize_tool_call` does; the inference only shapes the suggestion, and
    nothing runs. `place` says where a path lands in the repository (the role's file view), so
    the suggested call carries `config.py`, not the mount the model sent."""
    command = _command(name, args)
    rows = [r for r in ROWS if r.tool == name and r.command == command]
    if not rows:
        return None
    sent = args.get("path")
    where = place(sent) if place is not None and isinstance(sent, str) else None
    directory = where.directory if where is not None else _looks_like_directory(sent)
    row = next((r for r in rows if r.directory == directory), rows[0])
    if row.ours is None:
        return Suggestion(None, note=row.note)
    spec = offered.get(row.ours)
    if spec is None:
        return None
    # `offered` is keyed by our name; the spec carries the name the model is shown it by.
    ours: dict[str, object] = {}
    for theirs, mine in row.args.items():
        if theirs in args:
            ours[mine] = _value(theirs, args[theirs], where, row)
    if "lines" in ours and ours["lines"] is None:
        del ours["lines"]
    dropped = [a for a in row.dropped if a in args]
    note = (
        f"do not send {', '.join(dropped)}: {row.note}"
        if dropped
        else ("" if row.dropped else row.note)
    )
    return Suggestion(spec.name, ours, _takes(spec), note)


def foreign_argument(arg: str) -> str | None:
    """Which OpenHands tool an argument is of, when ours take none of that name: `command` is
    `str_replace_editor`'s, `file_text` too. None for one the table does not
    name, and for one it keeps under the same name (`path`), which is ours too."""
    if arg == "command":
        return _EDITOR
    # A name the table keeps as it is (`path`) is ours as much as theirs: not named as theirs.
    if any(row.args.get(arg) == arg for row in ROWS):
        return None
    for row in ROWS:
        if arg in row.args or arg in row.dropped:
            return row.tool
    return None


def ours_for(tool: str, arg: str) -> str | None:
    """Our name for OpenHands' argument `arg` on a call to our `tool`: `old_str` on `edit` is
    `old`. None when the table maps none."""
    for row in ROWS:
        # A directory's `path` becomes `glob`'s pattern by its value, not by a new name.
        if row.ours == tool and not row.directory and row.args.get(arg, arg) != arg:
            return row.args[arg]
    return None


def _command(name: str, args: Mapping[str, object]) -> str | None:
    """The `command` of a `str_replace_editor` call, inferred when missing as OpenHands does:
    `old_str` is a replace, `file_text` a create, `insert_line` an insert, anything else a
    view. None for every other tool."""
    if name != _EDITOR:
        return None
    command = args.get("command")
    if isinstance(command, str):
        return command
    if "old_str" in args:
        return "str_replace"
    if "file_text" in args:
        return "create"
    if "insert_line" in args:
        return "insert"
    return "view"


def _looks_like_directory(sent: object) -> bool:
    return isinstance(sent, str) and sent.rstrip().endswith("/")


def _value(theirs: str, value: object, where: Place | None, row: Row) -> object:
    """An argument's value in our shape: a path where it lands, a directory as `glob`'s pattern,
    a `view_range` as `lines`."""
    if theirs == "path" and isinstance(value, str):
        path = where.path if where is not None else value
        if not row.directory:
            return path
        base = path.rstrip("/")
        return "*" if base in ("", ".") else f"{base}/*"
    if theirs == "view_range":
        return _lines(value)
    return value


def _lines(value: object) -> str | None:
    """`[a, b]` as `"a-b"`; None for a range `lines` cannot say (OpenHands' `-1`: the end)."""
    if not isinstance(value, list):
        return None
    items = cast("list[object]", value)
    if len(items) != 2 or not all(_whole(n) for n in items):
        return None
    a, b = cast("list[int]", items)
    return f"{a}-{b}" if 1 <= a <= b else None


def _whole(n: object) -> bool:
    return isinstance(n, int) and not isinstance(n, bool)


def _takes(spec: ToolSpec) -> tuple[str, ...]:
    """The tool's arguments, in its schema's order."""
    props = spec.schema.get("properties")
    return tuple(cast("Mapping[str, object]", props)) if isinstance(props, Mapping) else ()
