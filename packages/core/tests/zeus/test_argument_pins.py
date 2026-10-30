"""Pins of how every tool reads its arguments, Zeus's and the scout's, before the records theme
has them read through what validated them.

`T-3.57` makes `validate` return the typed `Arguments` every tool reads through, and deletes
each tool's own narrowing. These pin what that must not change: the values each tool hands on,
its defaults (`depth` 1, `context` 2, an absent optional as `None`), and the calls `validate`
refuses before any tool runs. A later task may change these tests' setup when the code they
call changes shape, never their assertions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from tool_args import zeus_answer

from chickadee.context.session import Session
from chickadee.paths import Place
from chickadee.schema import ToolCall
from chickadee.schema.journal import EventKind, P
from chickadee.scout.search import Search
from chickadee.scout.tools import ScoutTools
from chickadee.tools.validate import Arguments

from .zeus_helpers import ScriptedGraph, answer, call, calls, rig

# -- recording fakes -------------------------------------------------------------------------


@dataclass(frozen=True)
class _Said:
    text: str

    def render(self) -> str:
        return self.text


@dataclass
class _Recorder:
    """The read and navigate ports, recording exactly what each call handed them."""

    asked: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(
        default_factory=list[tuple[str, tuple[Any, ...], dict[str, Any]]]
    )

    def _note(self, name: str, *args: Any, **kwargs: Any) -> _Said:
        self.asked.append((name, args, kwargs))
        return _Said(f"{name} answered")

    def read(self, target: str, **kwargs: Any) -> _Said:
        return self._note("read", target, **kwargs)

    def repeat_key(self, target: str, **kwargs: Any) -> None:
        return None

    def grep_in(self, handle: str, pattern: str, **kwargs: Any) -> _Said:
        return self._note("grep_in", handle, pattern, **kwargs)

    def place(self, sent: str) -> Place | None:
        return None

    # The navigator port answers text (`tools.context.Navigator`).
    def symbol(self, name: str) -> str:
        return self._note("symbol", name).text

    def neighbors(self, target: str, depth: int = 1) -> str:
        return self._note("neighbors", target, depth).text

    def impact(self, symbols: Sequence[str]) -> str:
        return self._note("impact", tuple(symbols)).text

    def tests_for(self, paths: Sequence[str]) -> str:
        return self._note("tests_for", tuple(paths)).text


class _Sink:
    def __init__(self) -> None:
        self.n = 0

    def record(self, kind: EventKind, payload: P) -> int:
        self.n += 1
        return self.n


# -- Zeus's read tools ------------------------------------------------------------------------

_NO_REGION: dict[str, Any] = {"lines": None, "at": None, "depth": None, "max_bytes": None}
"""A read's arguments as the port is handed them, none named. Each read hands on its call's
id too, so a box role's home can say which call showed which lines."""

READS: list[tuple[str, dict[str, Any], tuple[str, tuple[Any, ...], dict[str, Any]]]] = [
    (
        "read",
        {"path": "src/a.py"},
        ("read", ("src/a.py",), {**_NO_REGION, "call_id": "c1"}),
    ),
    (
        "read",
        {"path": "src/a.py", "lines": "1-9", "max_bytes": 400},
        (
            "read",
            ("src/a.py",),
            {**_NO_REGION, "lines": "1-9", "max_bytes": 400, "call_id": "c1"},
        ),
    ),
    (
        "read",
        {"path": "a.Thing", "at": "a.Thing.go", "depth": "body"},
        (
            "read",
            ("a.Thing",),
            {**_NO_REGION, "at": "a.Thing.go", "depth": "body", "call_id": "c1"},
        ),
    ),
    ("grep_in", {"handle": "h_1", "pattern": "x"}, ("grep_in", ("h_1", "x"), {"context": 2})),
    (
        "grep_in",
        {"handle": "h_1", "pattern": "x", "context": 0},
        ("grep_in", ("h_1", "x"), {"context": 0}),
    ),
]


@pytest.mark.parametrize(("tool", "args", "handed"), READS)
def test_zeus_s_read_tools_hand_on_what_they_did(
    tmp_path: Path,
    tool: str,
    args: dict[str, Any],
    handed: tuple[str, tuple[Any, ...], dict[str, Any]],
) -> None:
    reader = _Recorder()
    with Session.create(tmp_path, "S-args") as session:
        session.record_assistant("look", [ToolCall(id="c1", name=tool, arguments=args)])
        # Built directly, not through `validate`: `depth="body"` is a value the schema refuses,
        # pinned to show the tool hands on what it is given.
        zeus_answer(session, reader, "c1", tool, Arguments(tool, args))  # type: ignore[arg-type]
    assert reader.asked == [handed]


# -- Zeus's navigation, through the loop -------------------------------------------------------


def test_zeus_s_navigation_hands_on_what_it_did(tmp_path: Path) -> None:
    graph = ScriptedGraph()
    r = rig(
        tmp_path,
        [
            calls(
                call("symbol", name="a.Thing"),
                call("grep", pattern="retry", paths=["src/**"]),
                call("glob", pattern="**/*.py"),
                call("neighbors", target="a.go"),
                call("neighbors", target="a.go", depth=3),
                call("impact", symbols=["a.go", "b.stop"]),
            ),
            answer("ok"),
        ],
        graph=graph,
    )
    r.run("look around")
    assert graph.asked == [
        ("symbol", "a.Thing"),
        ("grep", ("retry", ("src/**",))),
        ("glob", "**/*.py"),
        ("neighbors", ("a.go", 1)),
        ("neighbors", ("a.go", 3)),
        ("impact", ("a.go", "b.stop")),
    ]


def test_a_call_validate_refuses_never_reaches_the_tool(tmp_path: Path) -> None:
    graph = ScriptedGraph()
    r = rig(
        tmp_path,
        [
            calls(call("grep", "f1"), call("neighbors", "n1", target="a.go", depth="2")),
            answer(),
        ],
        graph=graph,
    )
    r.run("look")
    assert graph.asked == []
    for cid in ("f1", "n1"):
        result = r.session.state.result(cid)
        assert result is not None and not result.ran
        assert result.text.startswith("INVALID_ARGUMENTS")


# -- the scout's shared tools -----------------------------------------------------------------

SCOUT: list[tuple[str, dict[str, Any], tuple[str, tuple[Any, ...], dict[str, Any]]]] = [
    ("symbol", {"name": "a.Thing"}, ("symbol", ("a.Thing",), {})),
    ("neighbors", {"target": "a.go"}, ("neighbors", ("a.go", 1), {})),
    ("neighbors", {"target": "a.go", "depth": 2}, ("neighbors", ("a.go", 2), {})),
    ("impact", {"symbols": ["a.go"]}, ("impact", (("a.go",),), {})),
    (
        "read",
        {"path": "src/a.py"},
        ("read", ("src/a.py",), {**_NO_REGION, "call_id": "s1"}),
    ),
    ("grep_in", {"handle": "h_1", "pattern": "x"}, ("grep_in", ("h_1", "x"), {"context": 2})),
]


@pytest.mark.parametrize(("tool", "args", "handed"), SCOUT)
def test_the_scout_s_tools_hand_on_what_they_did(
    tmp_path: Path,
    tool: str,
    args: dict[str, Any],
    handed: tuple[str, tuple[Any, ...], dict[str, Any]],
) -> None:
    ports = _Recorder()
    tools = ScoutTools(ports, ports, Search(root=tmp_path, rg=None))  # type: ignore[arg-type]
    tools.call(tool, args, _Sink(), offer_exec=False)
    assert ports.asked == [handed]


def test_the_scout_refuses_a_call_its_schema_refuses(tmp_path: Path) -> None:
    ports = _Recorder()
    tools = ScoutTools(ports, ports, Search(root=tmp_path, rg=None))  # type: ignore[arg-type]
    answer_ = tools.call(
        "neighbors", {"target": "a.go", "depth": "2"}, _Sink(), offer_exec=False
    )
    assert answer_.text.startswith("INVALID_ARGUMENTS")
    assert ports.asked == []
