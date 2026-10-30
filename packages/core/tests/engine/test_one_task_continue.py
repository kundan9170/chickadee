"""A one-task plan continues as a design's work does (its brief in the design's
place): `coder(title, brief)` with the same brief resumes its suspended run, no stop asked; a
changed brief is an amendment, asked at stop 2 with what changed, and the suspended run is
ended as superseded so it is never resumed beside the new one."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import (
    SID,
    Approving,
    CoderModel,
    LeavesAsked,
    athena_accepts,
    box_call,
    build,
    make_repo,
)
from zeus_turn_helpers import WAIT_S, answer, call, calls, state

import chickadee.delegation as delegation_module
from chickadee.engine import PacketEngine
from chickadee.errors import EngineBusy, SessionError
from chickadee.journal.paths import runs_root
from chickadee.journal.runs import RunProjections
from chickadee.schema.interaction import AnswerSource, ApprovalStop, StopAnswer, StopDecision
from chickadee.state.roots import roots as resolve_roots

TITLE = "Return 2"
BRIEF = "## Goal\n`go()` returns 2.\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


@pytest.fixture
def raw(raw: dict[str, Any]) -> dict[str, Any]:
    raw["workspace"] = {"mode": "worktree"}
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    return raw


class _Stops(Approving):
    def __init__(self) -> None:
        super().__init__()
        self.stops: list[ApprovalStop] = []

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        self.stops.append(stop)
        return super().confirm_stop(stop, cancel=cancel)


def _projection(repo: Path, run_id: str) -> Any:
    return RunProjections(resolve_roots(repo)).readable(run_id)


class _EditsThenLeaves(LeavesAsked):
    """The person edits the brief at stop 2 and approves the edit; then, asked about a command,
    leaves without answering."""

    def __init__(self, edited: str) -> None:
        super().__init__()
        self._given = [
            StopDecision(answer=StopAnswer.EDIT, edited=edited, source=AnswerSource.HUMAN),
            StopDecision(answer=StopAnswer.APPROVE, source=AnswerSource.HUMAN),
        ]

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        return self._given.pop(0) if self._given else super().confirm_stop(stop, cancel=cancel)


def _suspended(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    leaving: LeavesAsked | None = None,
) -> str:
    """A one-task run left suspended at a question the person never answered: its id."""
    leaving = leaving if leaving is not None else LeavesAsked()
    engine, _ = build(
        monkeypatch,
        repo,
        dict(raw),
        [
            calls(call("coder", "c1", title=TITLE, brief=BRIEF)),
            calls(box_call("shell", "w1", command="git checkout -- app.py")),
        ],
        interaction=leaving,
    )
    engine.turn(SID, "make go return 2")
    assert leaving.asked.wait(30), "the worker's command was put to the person"
    engine.close()
    [first] = sorted(d.name for d in runs_root(resolve_roots(repo)).iterdir() if d.is_dir())
    assert _projection(repo, first).end is None, "left suspended"
    return first


def test_the_same_brief_resumes_its_run_and_asks_nothing(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    first = _suspended(monkeypatch, repo, raw)
    person = _Stops()
    engine, _ = build(
        monkeypatch,
        repo,
        raw,
        [calls(call("coder", "c2", title=TITLE, brief=BRIEF)), answer("Done.")],
        interaction=person,
    )
    box = CoderModel(
        [
            calls(box_call("shell", "w2", command="true")),
            answer("T1: ran it."),
            *athena_accepts(),
        ]
    )
    monkeypatch.setattr(delegation_module, "client_for", lambda _endpoint: box)
    try:
        engine.turn(SID, "continue").wait(timeout=WAIT_S)
        said = state(engine, SID).result("c2")
        assert said is not None and f"run {first}" in said.text
        assert person.stops == [], "the brief was approved; nothing is asked again"
        assert _projection(repo, first).end is not None, "the same run ran to its end"
    finally:
        engine.close()


def test_a_changed_brief_is_asked_with_what_changed_and_supersedes_the_run(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    first = _suspended(monkeypatch, repo, raw)
    person = _Stops()
    changed = BRIEF + "And `stop()` returns 0.\n"
    engine, _ = build(
        monkeypatch,
        repo,
        raw,
        [calls(call("coder", "c2", title=TITLE, brief=changed)), answer("Done.")],
        interaction=person,
    )
    box = CoderModel([answer("T1: nothing to do."), *athena_accepts()])
    monkeypatch.setattr(delegation_module, "client_for", lambda _endpoint: box)
    try:
        engine.turn(SID, "also stop").wait(timeout=WAIT_S)
        [stop] = person.stops
        assert stop.notice is not None
        assert f"Changed since the brief approved in run {first}" in stop.notice
        assert "+And `stop()` returns 0." in stop.notice
        end = _projection(repo, first).end
        assert end is not None and end.outcome == "ABORTED" and "superseded" in (end.why or "")
        said = state(engine, SID).result("c2")
        assert said is not None and f"run {first}" not in said.text, "a new run did the work"
    finally:
        engine.close()


def test_another_title_is_refused_while_the_one_task_run_is_paused(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """Only the same title continues a paused one-task plan; another is another work,
    refused naming the paused one and `/drop`, never shown as an amendment of its brief."""
    first = _suspended(monkeypatch, repo, raw)
    person = _Stops()
    engine, _ = build(
        monkeypatch,
        repo,
        raw,
        [
            calls(call("coder", "c2", title="Add b", brief="## Goal\n`b.py` exists.\n")),
            answer("It is paused."),
        ],
        interaction=person,
    )
    try:
        engine.turn(SID, "add b").wait(timeout=WAIT_S)
        said = state(engine, SID).result("c2")
        assert said is not None and said.text.startswith("BLOCKED")
        assert f"run {first} of the one-task plan 'Return 2' is paused" in said.text
        assert "coder(title='Return 2', brief)" in said.text and "/drop" in said.text
        assert person.stops == [], "nothing is asked"
        assert _projection(repo, first).end is None, "never superseded"
    finally:
        engine.close()


def _resumes_to_its_end() -> CoderModel:
    return CoderModel(
        [
            calls(box_call("shell", "w2", command="true")),
            answer("T1: ran it."),
            *athena_accepts(),
        ]
    )


def test_the_brief_zeus_sent_resumes_a_run_the_person_edited(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """The person edited the brief at stop 2, so the approved brief is theirs; Zeus sending his
    own again changed nothing since, and the run resumes with the approved one, no stop."""
    edited = BRIEF + "Keep `go()` pure.\n"
    first = _suspended(monkeypatch, repo, raw, _EditsThenLeaves(edited))
    person = _Stops()
    engine, _ = build(
        monkeypatch,
        repo,
        raw,
        [calls(call("coder", "c2", title=TITLE, brief=BRIEF)), answer("Done.")],
        interaction=person,
    )
    monkeypatch.setattr(delegation_module, "client_for", lambda _e: _resumes_to_its_end())
    try:
        engine.turn(SID, "continue").wait(timeout=WAIT_S)
        said = state(engine, SID).result("c2")
        assert said is not None and f"run {first}" in said.text
        assert person.stops == [], "nothing changed since the person approved it"
    finally:
        engine.close()


def test_an_amendment_that_cannot_open_leaves_the_paused_run_to_continue(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """The paused run is superseded only once the new run is open: one that cannot open (the
    engine taken while the person answered stop 2) leaves the paused run as it was."""
    first = _suspended(monkeypatch, repo, raw)
    engine, _ = build(
        monkeypatch,
        repo,
        raw,
        [
            calls(call("coder", "c2", title=TITLE, brief=BRIEF + "And `stop()`.\n")),
            answer("It could not start."),
        ],
        interaction=_Stops(),
    )

    def taken(*_args: object, **_kwargs: object) -> Any:
        raise EngineBusy("run r9 is in flight on this engine")

    monkeypatch.setattr(PacketEngine, "coder", taken)
    try:
        engine.turn(SID, "also stop").wait(timeout=WAIT_S)
        assert _projection(repo, first).end is None, "never superseded"
    finally:
        engine.close()


def test_a_dropped_run_is_never_continued(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """`/drop` and a resume decide under the same lock: a resume that comes after the drop
    finds the run ended and refuses, never writing past its `run_end`."""
    first = _suspended(monkeypatch, repo, raw)
    engine, _ = build(monkeypatch, repo, raw, [answer("Hello.")], interaction=_Stops())
    try:
        engine.turn(SID, "hi").wait(timeout=WAIT_S)
        assert engine.drop_run(SID).startswith(f"run {first} dropped")
        opened = engine._conversations.get(SID).opened()  # type: ignore[union-attr] # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]
        assert opened is not None
        with pytest.raises(SessionError, match="has ended"):
            engine._continue_run(opened.session, SID, first)  # pyright: ignore[reportPrivateUsage]
    finally:
        engine.close()
