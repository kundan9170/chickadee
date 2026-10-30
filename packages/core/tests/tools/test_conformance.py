"""Every registered tool, as a plug-in must be: an object
schema; a template whose first line is its stub; `INVALID_ARGUMENTS` on malformed arguments;
every failure it returns within its declaration; no read but through its port (a denied path
answers the port's refusal); a text answer in its bound's unit; a mutating spec with an
effect lifecycle; its ports real ones, and needed.

A new built-in that passes this file and the load checks is safe to enable for an
experiment (decision 15). That every `SURFACE` spec's failures are in the docs' vocabulary is
`zeus/test_tool_surface.py`'s.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal

import pytest

from chickadee.context.results import (
    IndexUnavailable,
    InvalidArgument,
    ToolFailure,
    UrlRefused,
)
from chickadee.edits.draft import CandidateDraft
from chickadee.edits.formats.base import Targets
from chickadee.edits.lines import NUMBER, decode
from chickadee.errors import HarnessBug
from chickadee.schema.adapter import ToolCall
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.context import DraftCall, JobOutput, ToolContext
from chickadee.tools.rules import Refuse
from chickadee.tools.tool import PORTS, Answer, Invocation, Said, TextAnswer, Tool, unit_of
from chickadee.tools.validate import Arguments

from .tools_helpers import DictFiles, DictNavigator, FakePorts, dispatcher

TEMPLATES = Path(__file__).resolve().parents[2] / "src/chickadee/adapters/frontier/templates"

TOOLS = [BUILTINS.get(name) for name in BUILTINS.names()]
IDS = list(BUILTINS.names())

ARGS: dict[str, dict[str, object]] = {
    "atlas": {},
    "grep": {"pattern": "q"},
    "glob": {"pattern": "*.py"},
    "finish": {"report": "done"},
    "symbol": {"name": "n"},
    "neighbors": {"target": "t"},
    "impact": {"symbols": ["s"]},
    "read": {"path": "src/a.py"},
    "grep_in": {"handle": "src/a.py", "pattern": "a"},
    "load_skill": {"name": "migrations"},
    "find_skill": {"query": "migrations"},
    "spawn_agent": {"task": "where do we retry?", "profile": "reader"},
    "follow_up": {"agent": "A1", "message": "and where is it bounded?"},
    "recall": {"query": "retry"},
    "locate": {"question": "where do we retry?"},
    "tools": {"query": "read"},
    "shell": {"command": "ls"},
    "job": {"action": "tail", "id": "j1"},
    "coder": {"work": "2026-10-18-add-a-retry"},
    "ask_user": {"question": "which branch? default: main"},
    "note_concern": {"text": "the retry is unbounded", "anchor": ["src/a.py"]},
    "note": {"text": "s2 is cut at the parser because the lexer is shared"},
    "note_rejected": {"text": "a queue", "reason": "too heavy"},
    "request_exec": {"command": "ls", "reason": "r"},
    "write": {"path": "src/a.py", "content": "a = 2\n"},
    "edit": {"path": "src/a.py", "old": "a = 1", "new": "a = 2"},
    "apply_patch": {
        "patch": "*** Begin Patch\n*** Update File: src/a.py\n-a = 1\n+a = 2\n*** End Patch\n"
    },
    "replace_lines": {"path": "src/a.py", "start": 1, "end": 1, "content": "a = 2"},
    "insert_lines": {"path": "src/a.py", "after": 0, "content": "b = 2"},
    "replace_symbol": {"path": "src/a.py", "symbol": "a", "content": "a = 2"},
    "insert_after_symbol": {"path": "src/a.py", "symbol": "a", "content": "b = 2"},
    "create": {"path": "src/b.py", "content": "b = 1\n"},
    "delete": {"path": "src/a.py"},
    "move": {"path": "src/a.py", "destination": "src/c.py"},
    "tests_for": {"paths": ["src/a.py"]},
    "plan": {"text": "# Plan: one task\n\n1. **Do it.**\n\nFinal check: the suite."},
    "delegate": {"title": "Do it", "brief": "## Task 1 of 1: do it\n\n### Goal\nIt is done."},
    "review": {"task": "T1", "answer": "accept"},
    "outgrown": {},
    "doc": {"op": "write", "file": "README.md", "content": "# Design\n"},
    "worktree": {"action": "land", "how": "squash"},
    "pause": {},
    "tell": {"text": "use argon2 instead", "from": [1]},
    "reply": {"tell": "tell-1", "outcome": "applied", "text": "T1 uses argon2"},
    "web_fetch": {"url": "https://a.example.com/"},
    "web_search": {"query": "a proxy error"},
}

EXECUTORS = {
    name: (lambda pending: Said(f"streamed {pending.call.id}"))
    for name in BUILTINS.names()
    if (t := BUILTINS.get(name)) is not None and t.streams
}
"""A streaming tool runs only through its role's executor; this one stands in for Zeus's."""


class _Search:
    def __call__(
        self,
        query: str,
        *,
        kind: Literal["text", "path"] = "text",
        paths: Sequence[str] = (),
        cancel: object = None,
    ) -> str:
        return f"search {kind} {query!r}"


class _Edits:
    """The `edits` port."""

    def edit(self) -> AbstractContextManager[DraftCall]:
        """A real draft over one file, `src/a.py`, holding `a = 1`, in the default anchor
        style a line tool brings."""
        draft = CandidateDraft(
            Targets(writable=("src/a.py",)),
            lambda _: decode("a = 1\n"),
            lambda _: None,
            style=NUMBER,
        )
        return draft.call()


class _Docs:
    """The `docs` port: one part folder, its files drafted from nothing."""

    def resolve(self, file: str, title: str | None) -> str:
        return f"docs/chickadee/w/design/{file}"

    def edit(self, call: Invocation, approved_sha: str) -> AbstractContextManager[DraftCall]:
        draft = CandidateDraft(
            Targets(writable=("docs/chickadee/w/design/README.md",)),
            lambda _: None,
            lambda _: None,
        )
        return draft.call()

    def noted(self, answer: Answer) -> Answer:
        return answer


class _Skills:
    """The `skills` port: one skill indexed, anything else refused by name."""

    def load(self, name: str, call_id: str) -> str | InvalidArgument:
        if name != "migrations":
            return InvalidArgument("name", f"no skill is named {name!r}")
        return f"[skill migrations · loaded by {call_id}]"

    def find(self, query: str) -> str:
        return f"- migrations · knowledge, load_skill · for {query}"


class _Web:
    """The `web` port: one page it was given, anything else refused by its first guard."""

    def fetch(
        self, url: str, offset: int, call_id: str, cancel: threading.Event | None = None
    ) -> str | UrlRefused:
        if url != "https://a.example.com/":
            return UrlRefused(url, "not_seen")
        return f"<web_content url={url!r} trust='untrusted'>\npage\n</web_content>"

    def search(self, query: str, source: str | None, call_id: str) -> str:
        return f"<web_results query={query!r} trust='untrusted'>\nno results\n</web_results>"

    def saw(self, text: str) -> None:
        del text

    def fetched(self, url: str) -> str | None:
        return "page" if url == "https://a.example.com/" else None

    def carry(self, url: str, shown: Sequence[str]) -> None:
        """No earlier round: nothing to carry."""


class _Agents:
    """The `agents` port: a reader runs, anything else is refused by name."""

    def spawn(
        self,
        skill: str | None,
        profile: str | None,
        task: str,
        handles: object,
        *,
        call_id: str,
        cancel: object = None,
    ) -> str | InvalidArgument:
        if profile != "reader":
            return InvalidArgument("profile", f"no profile is named {profile!r}")
        return f"[agent A1 · round 0 · reader · complete] {task}"

    def follow_up(
        self, agent: str, message: str, *, call_id: str, cancel: object = None
    ) -> str | InvalidArgument:
        if agent != "A1":
            return InvalidArgument("agent", f"no agent {agent} in this session")
        return f"[agent A1 · round 1 · reader · complete] {message}"


class _Desk:
    """The `desk` port: each task tool answered by name."""

    def plan(self, text: str | None, *, call_id: str) -> str:
        return f"plan {call_id}"

    def delegate(
        self,
        title: str | None = None,
        brief: str | None = None,
        *,
        task: str | None = None,
        call_id: str,
    ) -> str:
        return f"T1 delegated: {title}"

    def review(
        self, task: str, answer: str, text: str | None, *, call_id: str, fresh: object = None
    ) -> str:
        return f"{task} {answer}"

    def end_outgrown(self, *, call_id: str) -> str:
        return "outgrown"

    def reply(self, tell: str, outcome: str, text: str, *, call_id: str) -> str:
        return f"replied to {tell}: {outcome}"


class _Finish:
    def finish(self, report: str) -> str:
        return f"reported: {report}"


class _Jobs:
    """The `jobs` port: one job, j1, whose output is two lines."""

    inline_bytes = 8192
    max_wait_s = 600

    def known(self, job_id: str) -> bool:
        return job_id == "j1"

    def is_run(self, job_id: str) -> bool:
        return False

    def run_board(self, job_id: str) -> str:
        return ""

    def listing(self) -> str:
        return "j1 · task · running"

    def output(self, job_id: str) -> JobOutput:
        return JobOutput("one\ntwo\n", "", 0)

    def await_exit(self, job_id: str, timeout_s: float, cancel: object) -> str | None:
        return "j1 exited 0"

    def kill(self, job_id: str, by: str) -> bool:
        return True


class _Worktree:
    """The `worktree` port: every move answers what it did."""

    def enter(self, *, operation_id: str, cancel: object) -> Said:
        return Said("entered the worktree chickadee/s1")

    def leave(self) -> Said:
        return Said("back in the user's checkout")

    def land(self, how: str, *, operation_id: str, cancel: object) -> Said:
        return Said(f"landed chickadee/s1 ({how})")

    def discard(self, *, operation_id: str, cancel: object) -> Said:
        return Said("discarded chickadee/s1")


class _Team:
    """The `team` port: a pause and a tell answer what they did."""

    def pause(self) -> Said:
        return Said("run r1 paused")

    def tell(self, text: str, sources: tuple[int, ...]) -> Said:
        return Said("told the team (tell-1)")


class _Escapes:
    def request(self, command: str, reason: str) -> str:
        return f"request_exec `{command}` · exit 0"


def _ctx(navigator: DictNavigator | None = None) -> ToolContext:
    files = DictFiles({"src/a.py": "a = 1\n"}, denied=frozenset({".env"}))
    rest = FakePorts()
    return ToolContext(
        role="test",
        files=files,
        handles=files,
        exec=rest,
        locator=rest,
        journal=rest,
        notes=rest,
        catalog=rest,
        user=rest,
        navigator=navigator
        or DictNavigator(
            {k: f"{k} answer" for k in ("atlas", "symbol", "neighbors", "impact")}
        ),
        search=_Search(),
        exec_request=_Escapes(),
        edits=_Edits(),
        skills=_Skills(),
        design_notes=rest,
        desk=_Desk(),
        finish=_Finish(),
        docs=_Docs(),
        jobs=_Jobs(),
        worktree=_Worktree(),
        team=_Team(),
        agents=_Agents(),
        web=_Web(),
    )


def _failure_name(answer: object) -> str | None:
    return getattr(answer, "failure", None)


def test_every_builtin_has_its_arguments_here() -> None:
    assert set(ARGS) == set(BUILTINS.names())


@pytest.mark.parametrize("tool", TOOLS, ids=IDS)
def test_an_object_schema_and_a_template_whose_first_line_is_its_stub(tool: Tool) -> None:
    assert tool.spec.schema["type"] == "object"
    template = TEMPLATES / tool.spec.template
    assert template.is_file(), template
    assert template.read_text(encoding="utf-8").strip().splitlines()[0].strip()


@pytest.mark.parametrize("tool", TOOLS, ids=IDS)
def test_malformed_arguments_are_invalid_arguments(tool: Tool) -> None:
    d = dispatcher([tool], ctx=_ctx(), executors=EXECUTORS)
    outcome = d.call(ToolCall(id="c", name=tool.spec.name, arguments={"no_such": 1}))
    decided = outcome.pending.decided
    assert isinstance(decided, Refuse) and decided.failure.failure == "INVALID_ARGUMENTS"


@pytest.mark.parametrize("tool", TOOLS, ids=IDS)
def test_a_valid_call_answers_and_text_is_in_its_bounds_unit(tool: Tool) -> None:
    d = dispatcher([tool], ctx=_ctx(), executors=EXECUTORS)
    answer = d.call(
        ToolCall(id="c", name=tool.spec.name, arguments=ARGS[tool.spec.name])
    ).answer
    assert answer is not None
    name = _failure_name(answer)
    assert name is None or name in tool.spec.failures, (tool.spec.name, name)
    if isinstance(answer, TextAnswer):
        assert answer.unit == unit_of(tool.spec)


@pytest.mark.parametrize("tool", TOOLS, ids=IDS)
def test_a_failure_from_its_port_is_one_it_declares(tool: Tool) -> None:
    def unavailable(*_: object, **__: object) -> ToolFailure:
        return IndexUnavailable("", "no index")

    class Down(DictNavigator):
        atlas = find = symbol = neighbors = impact = unavailable  # type: ignore[assignment]

    ctx = _ctx(Down({}))
    if "navigator" not in tool.needs:
        args = dict(ARGS[tool.spec.name])
        if "target" in args:
            args["target"] = ".env"
    else:
        args = ARGS[tool.spec.name]
    d = dispatcher([tool], ctx=ctx, executors=EXECUTORS)
    answer = d.call(ToolCall(id="c", name=tool.spec.name, arguments=args)).answer
    name = _failure_name(answer)
    if name is not None:
        assert name in tool.spec.failures, (tool.spec.name, name)


def test_a_denied_path_is_refused_by_the_port_not_read_around_it() -> None:
    read = BUILTINS.get("read")
    assert read is not None
    d = dispatcher([read], ctx=_ctx())
    answer = d.call(ToolCall(id="c", name="read", arguments={"path": ".env"})).answer
    assert _failure_name(answer) == "SECRET_PATH"


@pytest.mark.parametrize("tool", TOOLS, ids=IDS)
def test_its_ports_are_real_and_it_cannot_run_without_them(tool: Tool) -> None:
    arguments = Arguments(tool.spec.name, ARGS[tool.spec.name])
    if tool.streams:
        # A streaming tool touches no port itself: its role's executor runs it, never `run`.
        assert tool.needs <= PORTS
        with pytest.raises(HarnessBug, match="its role runs it"):
            tool.run(Invocation("c", "c", 0), arguments, ToolContext(role="test"))
        return
    assert tool.needs and tool.needs <= PORTS
    with pytest.raises(HarnessBug, match="without the"):
        tool.run(Invocation("c", "c", 0), arguments, ToolContext(role="test"))


@pytest.mark.parametrize("tool", TOOLS, ids=IDS)
def test_its_effect_lifecycle_and_retry_rule_are_declared(tool: Tool) -> None:
    spec = tool.spec
    assert isinstance(spec.idempotent, bool)
    assert spec.mutates == "nothing" or spec.effects != "none"
