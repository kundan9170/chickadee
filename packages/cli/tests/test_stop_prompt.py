"""A stop in the REPL: `approval.stop_lines` and `parse_stop`, and the
`Interaction` that shows one as it opens and sends its answer.

What these pin: the markdown is shown exactly, with who asks, where the work runs, why the
last answer was not used and the commands; `a`, `n <words>` and `c [reason]` answer it; `e`
sends the person's saved markdown, and an unchanged or unsaved edit sends nothing; a note
needs words; the permission prompt names who asks and for which task; a stop counts as an
approval in the stop rate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chickadee_cli.approval import (
    STOP_HINT,
    Invalid,
    StopReply,
    parse_stop,
    stop_lines,
    stop_rate,
)
from chickadee_cli.interaction import Interaction
from chickadee_cli.render.question import question_lines
from chickadee_cli.render.style import Style
from chickadee_serve.frames import RecordFrame
from chickadee_wire.models import ApprovalStop
from chickadee_wire.questions import OpenQuestion, Questions

PIPE = Style(tty=False)
TEXT = "# Plan\n\n1. T1 — add the parser\n2. T2 — use it in stats\n"
STOP: dict[str, Any] = {
    "id": "stop-plan-1",
    "kind": "plan",
    "author": "athena",
    "text": TEXT,
    "sha": "a" * 64,
    "round": 2,
    "refused": "the last edit changed nothing",
    "workspace": "worktree",
    "source_ref": "main",
}


def _stop() -> ApprovalStop:
    return ApprovalStop.model_validate(STOP)


def no_editor(_text: str) -> str | None:
    return None


def test_the_stop_shows_its_markdown_exactly_and_what_to_answer() -> None:
    lines = stop_lines(_stop(), PIPE)
    assert "athena asks you to approve the plan (round 2)" in lines[0]
    assert any("a new worktree from main" in line for line in lines)
    body = "\n".join(lines)
    assert TEXT.rstrip("\n") in body, "the markdown, line for line"
    assert any("the last edit changed nothing" in line for line in lines)
    assert STOP_HINT in lines[-1]


def test_approve_note_and_cancel() -> None:
    stop = _stop()
    assert parse_stop(stop, "a", no_editor) == StopReply("approve")
    # Moved: `a <words>` sent an approval whose words reached nobody; it is
    # refused, naming where words go.
    refused = parse_stop(stop, "a but skip the migrations", no_editor)
    assert isinstance(refused, Invalid) and "n <words>" in refused.why
    assert parse_stop(stop, "n keep T2 smaller", no_editor) == StopReply(
        "note", note="keep T2 smaller"
    )
    assert parse_stop(stop, "c not today", no_editor) == StopReply("cancel", note="not today")
    assert parse_stop(stop, "c", no_editor) == StopReply("cancel")


def test_a_note_needs_words_and_an_unknown_line_answers_nothing() -> None:
    stop = _stop()
    assert isinstance(parse_stop(stop, "n", no_editor), Invalid)
    assert isinstance(parse_stop(stop, "", no_editor), Invalid)
    assert isinstance(parse_stop(stop, "yes", no_editor), Invalid)


def test_an_edit_sends_the_saved_markdown_and_an_unchanged_one_nothing() -> None:
    stop = _stop()
    edited = TEXT + "3. T3 — docs\n"
    assert parse_stop(stop, "e", lambda text: text + "3. T3 — docs\n") == StopReply(
        "edit", text=edited
    )
    assert isinstance(parse_stop(stop, "e", lambda text: text), Invalid)
    assert isinstance(parse_stop(stop, "e", no_editor), Invalid)


class _SentAnswers:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def answer(
        self, sid: str, qid: str, choice: str, note: str | None = None, **kw: Any
    ) -> Any:
        self.sent.append({"qid": qid, "choice": choice, "note": note, **kw})


def _asked(seq: int, payload: dict[str, Any]) -> RecordFrame:
    return RecordFrame(
        scope="S-1", seq=seq, at="2026-10-15T10:00:00Z", kind="question_asked", payload=payload
    )


def test_the_repl_shows_a_stop_as_it_opens_and_sends_its_edit() -> None:
    wire = _SentAnswers()
    out: list[str] = []
    edited = TEXT.replace("T2", "T2 (small)")
    questions = Interaction(
        wire,  # type: ignore[arg-type]
        "S-1",
        PIPE,
        out.append,
        edit=lambda _text: edited,
    )
    questions.feed(_asked(4, {"stop": STOP}))
    assert any("asks you to approve the plan" in line for line in out), "shown as it opens"
    assert questions.answer("x") is None and wire.sent == []
    assert questions.answer("e") == "stop-plan-1"
    assert wire.sent == [
        {
            "qid": "stop-plan-1",
            "choice": "edit",
            "note": None,
            "text": edited,
        }
    ]


def test_the_permission_prompt_names_who_asks_and_for_which_task() -> None:
    fold = Questions()
    fold.feed(
        _asked(
            5,
            {
                "approval": {
                    "action_id": "shell-1",
                    "operation_id": "op",
                    "task_id": "T2",
                    "asker": "worker",
                    "kind": "command",
                    "display": "pip install requests",
                    "spec_sha": "b" * 64,
                    "network": True,
                    "idempotent": False,
                    "reversible": False,
                    "tier": "t2",
                }
            },
        )
    )
    [asked] = fold.open
    assert isinstance(asked, OpenQuestion)
    assert question_lines(asked, PIPE)[0].startswith(
        "! worker on T2 asks to run command: pip install requests"
    )


def _answered(seq: int, qid: str, choice: str, source: str) -> RecordFrame:
    return RecordFrame(
        scope="S-1",
        seq=seq,
        at="2026-10-15T10:00:00Z",
        kind="question_answered",
        payload={"id": qid, "choice": choice, "source": source, "elapsed_ms": 0},
    )


PERMISSION: dict[str, Any] = {
    "approval": {
        "action_id": "shell-1",
        "operation_id": "op",
        "task_id": "T2",
        "asker": "worker",
        "kind": "command",
        "display": "git checkout -- tests/test_pose.py",
        "spec_sha": "b" * 64,
        "network": False,
        "idempotent": False,
        "reversible": False,
        "tier": "t2",
    }
}


def test_a_session_joined_late_shows_only_the_questions_still_open() -> None:
    """`--continue` after leaving at a permission question: the stop was approved and the
    question withdrawn when its run stopped. Shown as the history is folded, both read as
    open, and the person's `1` went to Zeus as words. Neither is shown, and the line is not
    taken as an answer."""
    wire = _SentAnswers()
    out: list[str] = []
    questions = Interaction(wire, "S-1", PIPE, out.append)  # type: ignore[arg-type]
    questions.fold(
        [
            _asked(1, {"stop": STOP}),
            _answered(2, "stop-plan-1", "approve", "human"),
            _asked(3, PERMISSION),
            _answered(4, "shell-1", "", "withdrawn"),
        ]
    )
    assert out == [], "nothing closed is shown again"
    assert questions.current is None, "so the next line is the person's words"


def test_a_session_joined_late_shows_the_question_it_still_waits_on() -> None:
    out: list[str] = []
    questions = Interaction(_SentAnswers(), "S-1", PIPE, out.append)  # type: ignore[arg-type]
    questions.fold(
        [
            _asked(1, {"stop": STOP}),
            _answered(2, "stop-plan-1", "approve", "human"),
            _asked(3, PERMISSION),
        ]
    )
    assert out[0].startswith("! worker on T2 asks to run command: git checkout")
    assert not any("asks you to approve" in line for line in out), "the approved stop is not"
    assert questions.current is not None and questions.current.id == "shell-1"


def test_a_stop_a_person_answered_is_not_a_permission_stop() -> None:
    """Moved (T-3.157 review): every attended delegation passes a stop by design, so counting
    them pinned R-8 at 100%; the rate counts the tier floor's questions, at a command."""
    frames = [
        _asked(1, {"stop": STOP}),
        RecordFrame(
            scope="S-1",
            seq=2,
            at="2026-10-15T10:00:01Z",
            kind="question_answered",
            payload={
                "id": "stop-plan-1",
                "choice": "approve",
                "source": "human",
                "elapsed_ms": 1,
            },
            run_id="R-1",
        ),
        RecordFrame(
            scope="S-1",
            seq=3,
            at="2026-10-15T10:00:02Z",
            kind="episode",
            payload={
                "run_id": "R-1",
                "path": "planned",
                "tier": "t1",
                "design_sha": None,
                "constraints": [],
                "opened_at_seq": 1,
            },
        ),
    ]
    rate = stop_rate(frames)
    assert (rate.stopped, rate.delegations) == (0, 1)


def test_a_folder_stop_names_its_folder_and_e_opens_its_files(tmp_path: Path) -> None:
    """`e` opens every file of the folder, saved in place, and sends the
    README as saved; the engine reads the folder again."""
    folder = tmp_path / "plan"
    (folder / "tasks").mkdir(parents=True)
    (folder / "README.md").write_text(TEXT)
    (folder / "tasks" / "T1-x.md").write_text("# T1: x\n")
    stop = ApprovalStop.model_validate({**STOP, "folder": str(folder), "notice": "said once"})
    lines = "\n".join(stop_lines(stop, PIPE))
    assert str(folder) in lines and "e opens them all" in lines and "said once" in lines

    opened: list[Path] = []

    def edit_folder(at: Path) -> bool:
        opened.append(at)
        (at / "README.md").write_text(TEXT + "3. T3 — docs\n")
        return True

    got = parse_stop(stop, "e", no_editor, edit_folder)
    assert got == StopReply("edit", text=TEXT + "3. T3 — docs\n")
    assert opened == [folder]
    unopened = parse_stop(stop, "e", no_editor, lambda _at: False)
    assert isinstance(unopened, Invalid) and "could not open the files" in unopened.why
