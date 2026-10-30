"""The command surface as values (as data): every subcommand, its
flags, and the exit codes it may return.

`main.py` registers its commands from this table and leaves each one through `leave`, which
refuses a code the command's row does not hold, so a command cannot return a code the command's
table
does not list for it. `test_command_surface.py` asserts that this table equals the documented
table and that the Typer
app registers exactly it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NoReturn

import typer

__all__ = [
    "BARE",
    "DONE",
    "EXIT_MEANING",
    "HELD",
    "JOURNAL",
    "NOT_A_TTY",
    "PRECONDITION",
    "STOPPED",
    "SURFACE",
    "USAGE",
    "VERSION",
    "SurfaceRow",
    "command",
    "leave",
]


@dataclass(frozen=True)
class SurfaceRow:
    """One row of the table. `name` is the words after `chickadee`: `""` for bare `chickadee`,
    `"mcp trust"` for a nested one, and `"--version"` for the root flag that the table gives a
    row of its own."""

    name: str
    flags: frozenset[str]
    exits: frozenset[int]


def _row(name: str, flags: tuple[str, ...], exits: tuple[int, ...]) -> SurfaceRow:
    return SurfaceRow(name, frozenset(flags), frozenset(exits))


DONE: Final = 0
USAGE: Final = 1
STOPPED: Final = 2
JOURNAL: Final = 3
PRECONDITION: Final = 4
HELD: Final = 5
NOT_A_TTY: Final = 6
"""The codes by name: the one home every command, and `chickadee_serve.refusals` (pinned by
`test_command_surface.py`), takes them from."""

EXIT_MEANING: Final[dict[int, str]] = {
    DONE: "the command completed",
    USAGE: "usage or configuration error before anything started",
    STOPPED: "the session ended by refusal or abort before a handoff; for replay, it diverged",
    JOURNAL: "a journal could not be written or read",
    PRECONDITION: "a precondition refused the start",
    HELD: "the repository is held by another process",
    NOT_A_TTY: "not a TTY, so no REPL",
}
"""The closed table of exit codes."""

_TURN_EXITS = (DONE, USAGE, STOPPED, JOURNAL, PRECONDITION, HELD)

BARE: Final = ""
VERSION: Final = "--version"

SURFACE: Final[tuple[SurfaceRow, ...]] = (
    _row(
        BARE,
        ("--continue", "--session", "--tcp", "--new", "--worktree", "--in-place"),
        (0, 1, 4, 5, 6),
    ),
    _row(
        "attach",
        ("--continue", "--session", "--tcp", "--new", "--worktree", "--in-place"),
        (0, 1, 4),
    ),
    _row("tui", ("--continue", "--session", "--tcp", "--server"), (0, 1, 4, 5, 6)),
    _row("init", ("--force",), (0, 4, 5)),
    _row("setup", ("--install-servers",), (0, 1)),
    _row(
        "run",
        (
            "--yes",
            "--continue",
            "--session",
            "--in-place",
            "--file",
            "--allow-secret-file",
            "--json",
            "--new",
            "--worktree",
        ),
        _TURN_EXITS,
    ),
    _row(
        "ask",
        ("--yes", "--continue", "--session", "--file", "--allow-secret-file", "--json"),
        _TURN_EXITS,
    ),
    _row("fork", ("--at", "--reason"), (0, 1, 4)),
    _row("resume", ("--yes", "--json"), _TURN_EXITS),
    _row("replay", (), (0, 1, 2, 3)),
    _row(
        "serve", ("--repo", "--tcp", "--host", "--replay", "--session", "--speed"), (0, 1, 4, 5)
    ),
    _row("trust", (), (0, 1, 4, 6)),
    _row("hooks", (), (0, 1, 4)),
    _row("mcp trust", (), (0, 1)),
    _row("gc", ("--older-than", "--class", "--run", "--dry-run"), (0, 1)),
    _row("inspect", ("--repo", "--state", "--host", "--port", "--web"), (0, 1)),
    _row(VERSION, (), (0,)),
)

_BY_NAME: Final = {c.name: c for c in SURFACE}


def command(name: str) -> SurfaceRow:
    return _BY_NAME[name]


def leave(name: str, code: int) -> NoReturn:
    """End command `name` with `code`, which its row must hold: a code the table does not list
    for
    the command is a harness bug, raised as one rather than returned to a script."""
    if code not in command(name).exits:
        raise AssertionError(f"`chickadee {name}` may not exit {code}: §6a lists no such code")
    raise typer.Exit(code)
