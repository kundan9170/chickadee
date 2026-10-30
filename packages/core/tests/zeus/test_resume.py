"""Resume restores the reason, not only the work.

A session's resume brings back the ratified design and the constraint the user stated turns
ago, because the log is the context and a resumed loop assembles from the same lines. A crash
that left a call without its result is repaired before anything else is said: the harness
answers it, as a fact — for a `coder` call with what the run's journal reconciles, and for any
other call that what it did is unknown."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

from chickadee.client.request import text_of
from chickadee.context.ledger import note_constraint
from chickadee.context.lines import EpisodeLine
from chickadee.context.session import Session
from chickadee.context.standing import journaled_effects
from chickadee.context.work_line import WorkLine
from chickadee.schema import ToolCall
from chickadee.work import record_work
from chickadee.zeus.delegate import Delegation, OpenedEpisode, open_episode

from .zeus_helpers import SESSION_ID, FakeBox, FakeRun, answer, call, calls, rig

GOAL = "one bucket per client id; refill 10 tokens a second; 429 when empty"
DESIGN = f"# Rate limiting\n\n## What changes\n- `api`: {GOAL}.\n"
ASKED = "add rate limiting to the API. whatever you do, never rename a field of UserDTO"
CONSTRAINT = "no field of UserDTO is renamed or removed"
WORK = "2026-10-18-rate-limiting"


def test_resume_restores_the_design_and_the_constraint(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    delegate = call("coder", "c1", work=WORK)
    script = [
        answer("I will design a token bucket."),
        calls(delegate),
        answer("built; tests pass"),
    ]
    first = rig(tmp_path, script)
    first.run(ASKED)
    said = next(t.seq for t in first.session.state.turns() if t.role == "user")
    # T-3.19's `note_constraint` tool is not on the surface yet; its writer is the ledger's.
    cid = note_constraint(first.session, CONSTRAINT, [said], "session")
    assert isinstance(cid, str)
    record_work(first.session, WorkLine(name=WORK, dir="docs/chickadee", state="open"))
    first.run("approved — build it")
    first.session.close()

    reopened = Session.open(sessions, SESSION_ID)
    second = rig(tmp_path, [answer("resumed")], session=reopened, workspace=first.workspace)
    second.run("where were we?")
    wire = second.model.wire(0)
    assert ASKED in wire, "the user's words"
    assert CONSTRAINT in wire, "the constraint, stated turns ago"
    assert WORK in wire, "the design, delegated by its work's name (D204)"
    assert "COMPLETED · run R-1" in wire, "and the work"
    state = reopened.state
    assert [c.text for c in state.ledger.live_constraints()] == [CONSTRAINT]
    # What stop 1 showed, the README with the turn and the constraint, is composed where the
    # folder is (`engine/test_work_folder.py`); the delegation carries the ledger.
    assert state.episodes()[0].constraints == (cid,), "the delegation carried the ledger"
    reopened.close()


def _crashed_mid_call(tmp_path: Path, tool_call: ToolCall, *, episode: str | None) -> None:
    """A session that stopped between dispatching a call and recording its result."""
    session = Session.create(tmp_path / "sessions", SESSION_ID)
    session.record_user("ship it")
    turn = session.record_assistant("", [tool_call])
    if episode is not None:
        session.record_episode(
            EpisodeLine(
                run_id=episode,
                path="planned",
                tier=None,
                design_sha="b" * 64,
                constraints=(),
                opened_at_seq=turn.seq,
            )
        )
    session.close()


def test_a_crash_mid_delegation_is_reconciled_before_the_resumed_turn(tmp_path: Path) -> None:
    coder = call("coder", "c7", design="# Rate limiting\n")
    _crashed_mid_call(tmp_path, coder, episode="R-9")
    reopened = Session.open(tmp_path / "sessions", SESSION_ID)
    journal = {"R-9": "T1-T2 committed · T3 delegated, its round cut off"}
    r = rig(
        tmp_path, [answer("T3 needs a retry")], session=reopened, reconcile=journal.__getitem__
    )
    events = r.run("what happened?")
    assert events[0].kind == "tool_result", "the tail is repaired before anything else"
    line = reopened.state.result("c7")
    assert line is not None and line.run_id == "R-9" and line.ran
    assert line.text.startswith("INTERRUPTED{tool: coder")
    assert journal["R-9"] in line.text
    messages = r.model.requests[0].messages
    roles = [m.role for m in messages]
    at = roles.index("tool")
    assert messages[at - 1].role == "assistant" and messages[at].tool_call_id == "c7"
    assert "what happened?" in text_of(messages[-1])
    reopened.close()


def test_a_crash_mid_delegation_with_no_journal_says_what_it_did_is_unknown(
    tmp_path: Path,
) -> None:
    """The episode line is written before the run does any work, so an open run may have
    done nothing or everything. With no journal to reconcile it from, the repair names the
    run and says so, rather than call it in flight."""
    coder = call("coder", "c7", design="# Rate limiting\n")
    _crashed_mid_call(tmp_path, coder, episode="R-9")
    reopened = Session.open(tmp_path / "sessions", SESSION_ID)
    r = rig(tmp_path, session=reopened)
    r.run("what happened?")
    line = reopened.state.result("c7")
    assert line is not None and line.run_id == "R-9"
    assert line.text.startswith("INTERRUPTED{tool: coder")
    assert "R-9" in line.text and "unknown" in line.text
    assert "in flight" not in line.text
    reopened.close()


def test_a_crash_mid_shell_call_says_what_it_did_is_unknown(tmp_path: Path) -> None:
    """`ran: false` is a refusal's, and the command was not refused. A push that may have
    happened stays in a later fork's effects notice."""
    shell = call("shell", "s3", command="git push", network=["github.com"])
    _crashed_mid_call(tmp_path, shell, episode=None)
    reopened = Session.open(tmp_path / "sessions", SESSION_ID)
    r = rig(tmp_path, session=reopened)
    r.run("did it push?")
    state = reopened.state
    line = state.result("s3")
    assert line is not None and line.ran, "not refused, so it may have run"
    assert line.text.startswith("INTERRUPTED{tool: shell")
    assert "unknown" in line.text
    effects = journaled_effects(state, None, hi=line.seq)
    assert [(e.kind, e.detail) for e in effects] == [("network", "git push")]
    reopened.close()


def test_a_crash_mid_question_is_not_taken_for_the_persons_answer(tmp_path: Path) -> None:
    asked = call("ask_user", "q2", question="bucket size? default 10; or 100")
    _crashed_mid_call(tmp_path, asked, episode=None)
    reopened = Session.open(tmp_path / "sessions", SESSION_ID)
    r = rig(tmp_path, session=reopened)
    r.run("go on")
    line = reopened.state.result("q2")
    assert line is not None and not line.ran, "nobody answered"
    assert line.text.startswith("INTERRUPTED{tool: ask_user")
    assert not reopened.state.is_user_words(line.seq)
    reopened.close()


# -- a failed episode open gives the lease back ---------------------------


def _coder_turn(session: Session) -> int:
    """The assistant turn that called `coder`: an `episode` line opens at it."""
    session.record_user("do it")
    turn = session.record_assistant("", [ToolCall(id="c1", name="coder", arguments={})])
    return turn.seq


def _delegation() -> Delegation:
    return Delegation(
        request={"title": "t", "brief": "b"},
        kind="task",
        text="b",
        title="t",
        constraints=(),
    )


def test_an_episode_line_that_cannot_be_written_gives_the_lease_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`open` took the repository lease; the `episode` line then fails to write (a full
    disk). The run is abandoned before the error propagates, and the next `coder` call
    opens a run of its own."""
    box = FakeBox()
    with Session.create(tmp_path / "sessions", SESSION_ID) as session:

        def full_disk(_line: EpisodeLine) -> EpisodeLine:
            raise OSError(errno.ENOSPC, "No space left on device")

        at = _coder_turn(session)
        monkeypatch.setattr(session, "record_episode", full_disk)
        with pytest.raises(OSError, match="No space"):
            open_episode(session, box, _delegation(), opened_at_seq=at, cancel=None)
        assert box.abandoned == ["R-1"]

        monkeypatch.undo()
        opened = open_episode(session, box, _delegation(), opened_at_seq=at, cancel=None)
        assert isinstance(opened, OpenedEpisode) and opened.run.run_id == "R-2"
        assert box.abandoned == ["R-1"], "the run that opened is not handed back"


def test_a_sync_that_fails_gives_the_lease_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    box = FakeBox()
    with Session.create(tmp_path / "sessions", SESSION_ID) as session:

        def broken() -> None:
            raise OSError(errno.EIO, "I/O error")

        at = _coder_turn(session)
        monkeypatch.setattr(session, "sync", broken)
        with pytest.raises(OSError, match="I/O"):
            open_episode(session, box, _delegation(), opened_at_seq=at, cancel=None)
        assert box.abandoned == ["R-1"]


def test_an_abandon_that_fails_too_does_not_hide_why_the_open_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full disk that broke the `episode` line is what the person is told; an abandon
    that then fails on the same disk is a note on it, not the error in its place."""
    box = FakeBox()
    with Session.create(tmp_path / "sessions", SESSION_ID) as session:

        def full_disk(_line: EpisodeLine) -> EpisodeLine:
            raise OSError(errno.ENOSPC, "No space left on device")

        def cannot_abandon(_run: FakeRun) -> None:
            raise OSError(errno.ENOSPC, "the run's journal cannot be written either")

        at = _coder_turn(session)
        monkeypatch.setattr(session, "record_episode", full_disk)
        monkeypatch.setattr(FakeRun, "abandon", cannot_abandon)
        with pytest.raises(OSError, match="No space") as err:
            open_episode(session, box, _delegation(), opened_at_seq=at, cancel=None)

    assert any("cannot be written either" in note for note in err.value.__notes__)
