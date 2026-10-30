"""A stop over a folder (`approval/stops.ask_stop`'s `reread`): what is shown
and approved is always the folder as it is, re-read at every answer; an edit of any of its
files is a new showing; a folder whose shape is wrong is shown again naming what is wrong; an
approval of a folder changed since it was shown is refused; approved, `version` is the tree."""

from __future__ import annotations

import threading
from collections.abc import Iterator

from chickadee.approval.stops import Reread, StopOutcome, WorkspacePreview, ask_stop, stop_of
from chickadee.schema.interaction import (
    AnswerSource,
    ApprovalStop,
    StopAnswer,
    StopDecision,
    StopKind,
)

WHERE = WorkspacePreview("worktree", "main")


class Port:
    """Answers each showing in turn, and keeps what it was shown."""

    def __init__(self, *answers: StopAnswer) -> None:
        self._answers: Iterator[StopAnswer] = iter(answers)
        self.shown: list[ApprovalStop] = []

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        self.shown.append(stop)
        answer = next(self._answers)
        edited = "x" if answer is StopAnswer.EDIT else None
        return StopDecision(answer=answer, edited=edited, source=AnswerSource.HUMAN)


class Folder:
    """The folder's state at each read: a list the test moves through."""

    def __init__(self, *states: Reread) -> None:
        self._states = list(states)
        self.reads = 0

    def __call__(self) -> Reread:
        state = self._states[min(self.reads, len(self._states) - 1)]
        self.reads += 1
        return state


def _first(text: str = "# Plan\n") -> ApprovalStop:
    return stop_of(StopKind.PLAN, "athena", text, WHERE, folder="/w/plan")


def test_an_approval_of_the_folder_as_shown_carries_its_tree() -> None:
    folder = Folder(Reread("# Plan\n", "t1"))
    got = ask_stop(_first(), Port(StopAnswer.APPROVE), reread=folder)
    assert got == StopOutcome(StopAnswer.APPROVE, "# Plan\n", got.sha, version="t1")


def test_an_edit_of_any_file_is_a_new_showing_of_the_folder_as_it_is() -> None:
    folder = Folder(
        Reread("# Plan\n", "t1"), Reread("# Plan v2\n", "t2"), Reread("# Plan v2\n", "t2")
    )
    port = Port(StopAnswer.EDIT, StopAnswer.APPROVE)
    got = ask_stop(_first(), port, reread=folder)
    assert [s.text for s in port.shown] == ["# Plan\n", "# Plan v2\n"]
    assert (got.text, got.version) == ("# Plan v2\n", "t2")


def test_an_edit_that_changed_no_file_is_said_and_the_same_folder_shown() -> None:
    folder = Folder(Reread("# Plan\n", "t1"))
    port = Port(StopAnswer.EDIT, StopAnswer.APPROVE)
    ask_stop(_first(), port, reread=folder)
    assert port.shown[1].refused == "the edit changed nothing in the folder"


def test_a_folder_out_of_shape_is_shown_again_naming_what_is_wrong() -> None:
    folder = Folder(
        Reread("# Plan\n", "t1"),
        Reread("# Plan\n", "t2", ("plan/README.md: the Tasks section is missing",)),
        Reread("# Plan\n", "t3"),
    )
    port = Port(StopAnswer.EDIT, StopAnswer.EDIT, StopAnswer.APPROVE)
    got = ask_stop(_first(), port, reread=folder)
    assert port.shown[1].refused == (
        "the folder is not in shape: plan/README.md: the Tasks section is missing"
    )
    assert got.version == "t3"


def test_an_approval_of_a_folder_changed_since_it_was_shown_is_refused() -> None:
    folder = Folder(Reread("# Plan\n", "t1"), Reread("# Plan b\n", "t2"))
    port = Port(StopAnswer.APPROVE, StopAnswer.APPROVE)
    got = ask_stop(_first(), port, reread=folder)
    assert port.shown[1].refused == "the folder changed after it was shown: this is it now"
    assert (got.text, got.version) == ("# Plan b\n", "t2")


def test_a_note_or_a_cancel_ends_it_as_a_text_stop_does() -> None:
    folder = Folder(Reread("# Plan\n", "t1"))
    got = ask_stop(_first(), Port(StopAnswer.CANCEL), reread=folder)
    assert got.answer is StopAnswer.CANCEL and got.version is None
