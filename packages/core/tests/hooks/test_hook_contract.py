"""The hook contract: the points as data, the one precedence, the
matchers, and `HookSpec`'s refusals.

What these pin: every point has a payload model, a level and a blocking verdict it takes; the
precedence, pair by pair; notes and warnings kept in order and labelled; two `Continue`s
joined and the first `RetryTask` kept; each matcher narrowing; a hook is a built-in or a script.
"""

from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from chickadee.errors import HarnessBug
from chickadee.hooks.match import matches
from chickadee.hooks.payloads import (
    PAYLOADS,
    AfterStep,
    BeforeRequest,
    BeforeTool,
    HookEvent,
)
from chickadee.hooks.points import POINTS, HookPoint
from chickadee.hooks.rule import tool_payload
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import (
    PRECEDENCE,
    Ask,
    Continue,
    Deny,
    Halt,
    Note,
    Proceed,
    RetryTask,
    Warn,
    combine,
    label,
    labelled,
)
from chickadee.paths import Place
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.rules import Pending, ScopeState
from chickadee.tools.tool import Invocation


def test_every_point_has_a_payload_and_takes_its_blocking_verdict() -> None:
    assert set(POINTS) == set(HookPoint) == set(PAYLOADS)
    for point, spec in POINTS.items():
        assert Proceed in spec.accepts and Warn in spec.accepts, point
        assert spec.blocking in spec.accepts, point


def test_retry_and_continue_are_taken_only_where_the_contract_says() -> None:
    assert [p for p, s in POINTS.items() if RetryTask in s.accepts] == [HookPoint.ROUND_ENDED]
    assert [p for p, s in POINTS.items() if Continue in s.accepts] == [HookPoint.BEFORE_STOP]
    assert Note not in POINTS[HookPoint.BEFORE_REQUEST].accepts


def test_an_event_refuses_another_points_payload() -> None:
    with pytest.raises(HarnessBug):
        HookEvent(HookPoint.AFTER_TOOL, "worker", _before_tool())


_ONE = {
    Halt: Halt("h"),
    Deny: Deny("d"),
    Ask: Ask("a"),
    RetryTask: RetryTask("r"),
    Continue: Continue("c"),
}


@pytest.mark.parametrize(("strong", "weak"), list(combinations(PRECEDENCE, 2)))
def test_the_stronger_verdict_decides_in_either_order(strong: type, weak: type) -> None:
    for said in (
        [("a", _ONE[strong]), ("b", _ONE[weak])],
        [("b", _ONE[weak]), ("a", _ONE[strong])],
    ):
        assert type(combine(said).decided) is strong


def test_notes_and_warnings_are_kept_in_order() -> None:
    got = combine([("x", Note("one")), ("y", Warn("w")), ("z", Note("two")), ("p", Proceed())])
    assert got.decided is None
    assert got.notes == ("one", "two") and got.warns == ("w",)


def test_a_scripts_words_are_labelled_and_the_verdict_kept() -> None:
    assert labelled("lint", Note("x")) == Note(label("lint", "x")) == Note("[hook lint] x")
    assert labelled("g", RetryTask("again", fresh=("a.py",))) == RetryTask(
        "[hook g] again", ("a.py",)
    )
    assert labelled("g", Deny("no", "do y")) == Deny("[hook g] no", "do y")
    assert labelled("g", Proceed()) == Proceed()


def test_two_continues_are_joined_and_the_first_retry_kept() -> None:
    joined = combine([("a", Continue("fix x")), ("b", Continue("fix y"))]).decided
    assert isinstance(joined, Continue)
    assert "fix x" in joined.text and "fix y" in joined.text
    first = combine([("a", RetryTask("first")), ("b", RetryTask("second"))]).decided
    assert first == RetryTask("first")


def _before_tool(**kw: object) -> BeforeTool:
    fields: dict[str, object] = {
        "tool": "shell",
        "tool_sent": "bash",
        "input": {"command": "pytest -q"},
        "resolved": ("src/app.py",),
        "programs": ("pytest",),
    }
    fields.update(kw)
    return BeforeTool.model_validate(fields)


def _spec(**kw: object) -> HookSpec:
    fields: dict[str, object] = {"event": "before_tool", "name": "h", "use": "x"}
    fields.update(kw)
    return HookSpec.model_validate(fields)


def _event(payload: BeforeTool | None = None) -> HookEvent:
    return HookEvent(
        HookPoint.BEFORE_TOOL, "worker", payload or _before_tool(), model="qwen3-coder"
    )


@pytest.mark.parametrize(
    ("spec", "hit"),
    [
        ({}, True),
        ({"role": "worker"}, True),
        ({"role": "athena"}, False),
        ({"model": "qwen*"}, True),
        ({"model": "gpt*"}, False),
        ({"tool": "shell"}, True),
        # The registry name, never the name the model wrote.
        ({"tool": "bash"}, False),
        ({"program": ["pytest"]}, True),
        ({"program": ["ruff"]}, False),
        ({"paths": ["src/**"]}, True),
        ({"paths": ["docs/**"]}, False),
        ({"enabled": False}, False),
        ({"event": "after_tool"}, False),
    ],
)
def test_each_matcher_narrows(spec: dict[str, object], hit: bool) -> None:
    assert matches(_spec(**spec), _event()) is hit


def test_a_step_is_matched_by_the_paths_it_changed() -> None:
    step = HookEvent(
        HookPoint.AFTER_STEP, "worker", AfterStep(tool="shell", changed=("src/a.py",))
    )
    assert matches(_spec(event="after_step", paths=["src/*.py"]), step)
    assert not matches(_spec(event="after_step", paths=["tests/*"]), step)


def test_a_point_with_no_paths_never_matches_a_paths_hook() -> None:
    request = HookEvent(
        HookPoint.BEFORE_REQUEST, "worker", BeforeRequest(messages=3, tokens_est=9)
    )
    assert not matches(_spec(event="before_request", paths=["**"]), request)


@pytest.mark.parametrize(
    "fields",
    [
        {"use": None},
        {"command": ["x.sh"]},
        {"use": None, "command": []},
        {"use": None, "command": ["x.sh"], "with": {"a": 1}},
        {"name": "Bad Name"},
        {"timeout_s": 121},
        {"timeout_s": True},
        {"enabled": "false"},
        {"event": "pre_action"},
        {"unknown": 1},
    ],
)
def test_a_spec_is_a_builtin_or_a_script_and_is_refused_otherwise(
    fields: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _spec(**fields)


def test_a_script_spec_reads_with_its_alias() -> None:
    spec = HookSpec.model_validate(
        {"event": "round_ended", "name": "c", "command": ["./c.sh"], "on_error": "deny"}
    )
    assert spec.kind == "script" and spec.key == ("round_ended", "c")
    built = _spec(**{"with": {"max": 2}})
    assert built.kind == "builtin" and built.with_ == {"max": 2}


@pytest.mark.parametrize(
    ("tool", "raw", "named"),
    [
        ("read", {"path": "src/a.py"}, ("src/a.py",)),
        ("move", {"path": "src/a.py", "destination": "lib/b.py"}, ("src/a.py", "lib/b.py")),
        ("tests_for", {"paths": ["src/a.py", "src/b.py"]}, ("src/a.py", "src/b.py")),
        ("shell", {"command": "/usr/bin/git status && pytest -q"}, ()),
    ],
)
def test_a_calls_files_are_the_ones_its_spec_names(
    tool: str, raw: dict[str, object], named: tuple[str, ...]
) -> None:
    """Review finding: a hook's `paths` reads a call by its tool's `ToolSpec.paths`, a list
    argument included, never a second list of argument names."""
    spec = BUILTINS.get(tool)
    assert spec is not None
    pending = Pending(Invocation("c1", "c1", 0), tool, raw, tool, spec.spec)
    state = ScopeState({})
    state.place = lambda path: Place(path)
    payload = tool_payload(pending, state)
    assert payload.resolved == named
    if tool == "shell":
        assert payload.programs == ("git", "pytest")
