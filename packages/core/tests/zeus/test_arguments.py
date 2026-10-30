"""`Arguments`, the view `validate` returns for a valid call.

It reads what the schema promised and nothing else: an absent optional is its default, and a
shape the schema let through but names otherwise is a harness bug, never a guess.
"""

from __future__ import annotations

import pytest

from chickadee.errors import HarnessBug
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.spec import spec
from chickadee.tools.validate import Arguments, validate


def _valid(tool: str, args: dict[str, object]) -> Arguments:
    found = spec(tool)
    assert found is not None
    result = validate(found, args)
    assert isinstance(result, Arguments), result
    return result


def test_each_read_returns_what_was_sent() -> None:
    args = _valid("shell", {"command": "ls", "network": True})
    assert args.text("command") == "ls"
    assert args.boolean("network", False) is True
    assert args.boolean("idempotent", False) is False
    assert args.text_or_none("cwd") is None
    assert _valid("impact", {"symbols": ["a.f", "b.g"]}).texts("symbols") == ("a.f", "b.g")


def test_an_absent_optional_is_its_default() -> None:
    args = _valid("read", {"path": "a.py"})
    assert args.integer("max_bytes", None) is None
    assert args.text_or_none("depth") is None
    assert _valid("neighbors", {"target": "go"}).integer("depth", 1) == 1
    assert _valid("grep_in", {"handle": "h_1", "pattern": "x"}).integer("context", 2) == 2


def test_an_object_is_read_through_the_same_view() -> None:
    """No tool of Zeus's takes an object since `coder`'s `request` went; the edit tools'
    `edits` are each read as the arguments they are."""
    edit = BUILTINS.get("edit")
    assert edit is not None
    args = validate(edit.spec, {"path": "a.py", "edits": [{"old": "x", "new": "y"}]})
    assert isinstance(args, Arguments), args
    (one,) = args.objects("edits")
    assert one.tool == "edit.edits[0]"
    assert (one.text("old"), one.text("new")) == ("x", "y")


@pytest.mark.parametrize(
    ("read", "sent"),
    [
        (lambda a: a.text("k"), {"k": 1}),
        (lambda a: a.text("k"), {}),
        (lambda a: a.text_or_none("k"), {"k": 1}),
        (lambda a: a.integer("k", 0), {"k": True}),
        (lambda a: a.integer("k", 0), {"k": "1"}),
        (lambda a: a.boolean("k", False), {"k": 1}),
        (lambda a: a.texts("k"), {"k": "a"}),
        (lambda a: a.texts("k"), {"k": ["a", 1]}),
        (lambda a: a.object("k"), {"k": []}),
    ],
)
def test_a_shape_the_schema_names_otherwise_is_a_harness_bug(
    read: object, sent: dict[str, object]
) -> None:
    with pytest.raises(HarnessBug, match="passed validation as"):
        read(Arguments("probe", sent))  # type: ignore[operator]
