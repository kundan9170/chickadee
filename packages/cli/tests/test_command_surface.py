"""The command surface is §6a's: the table in the spec, `SURFACE`, and
the Typer app name the same commands, flags and exit codes. A command added to one and not the
others fails here; so does a command that tries to leave with a code its row does not list."""

from __future__ import annotations

from typing import Any

import pytest
import typer

from chickadee_cli.main import app
from chickadee_cli.surface import (
    BARE,
    HELD,
    PRECONDITION,
    SURFACE,
    USAGE,
    VERSION,
    leave,
)
from chickadee_serve.__main__ import parser as serve_parser
from chickadee_serve.refusals import REFUSAL_EXIT

r"""A cell boundary: a `|` not escaped as `\|` inside a cell."""


def _options(command: Any) -> set[str]:
    # Typer vendors its own Click, so its classes are read by their attributes, not types.
    return {
        opt
        for param in command.params
        if param.param_type_name == "option"
        for opt in param.opts
        if opt.startswith("--") and opt != "--help"
    }


def _registered() -> dict[str, set[str]]:
    """Every command the app registers, with its flags; the root's as `BARE`."""
    root: Any = typer.main.get_command(app)
    found: dict[str, set[str]] = {BARE: _options(root) - {VERSION}, VERSION: set()}
    assert VERSION in _options(root)

    def walk(group: Any, prefix: str) -> None:
        for name, sub in group.commands.items():
            full = f"{prefix}{name}"
            if hasattr(sub, "commands"):
                walk(sub, f"{full} ")
            else:
                found[full] = _options(sub)

    walk(root, "")
    return found


def test_the_app_registers_exactly_the_surface() -> None:
    registered = _registered()
    serve_flags = registered.pop("serve")
    assert serve_flags == set(), "serve passes its whole command line to chickadee-serve"
    expected = {c.name: set(c.flags) for c in SURFACE if c.name != "serve"}
    assert registered == expected


def test_serve_flags_are_chickadee_serves_own() -> None:
    parsed = {opt for action in serve_parser()._actions for opt in action.option_strings}
    row = next(c for c in SURFACE if c.name == "serve")
    assert parsed - {"-h", "--help"} == set(row.flags)


def test_every_code_is_in_the_closed_table() -> None:
    assert {code for c in SURFACE for code in c.exits} <= set(range(7))


def test_a_command_cannot_leave_with_a_code_its_row_lacks() -> None:
    with pytest.raises(typer.Exit) as left:
        leave("replay", 2)
    assert left.value.exit_code == 2
    with pytest.raises(AssertionError, match="may not exit 6"):
        leave("run", 6)


def test_serve_takes_its_refusal_codes_from_the_surface() -> None:
    """`chickadee_serve.refusals` cannot import the CLI, so its copy is pinned here."""
    assert REFUSAL_EXIT == {
        "held": HELD,
        "unbindable": USAGE,
        "not_a_repository": PRECONDITION,
        "config": USAGE,
    }
