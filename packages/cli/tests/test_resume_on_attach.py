"""Opening a session again resumes the run left paused in it: `commands.resume`, as the
REPL calls it on attach and as `/resume` does. Which run that is, the server says in the
session's liveness (`paused_run`); the rule is core's (`context/project.paused_runs`) and
tested there.

What these pin: a paused run is said and followed from the last record the person saw, read
before it goes on; on attach, only the run liveness names is resumed, never one still going; a
session with nothing paused says nothing; `/resume` says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from chickadee_cli.commands import resume
from chickadee_cli.render.style import Tone
from chickadee_serve.frames import RecordFrame, Resumed
from chickadee_wire.models import Liveness

SID = "S-1"
AT = "2026-10-20T10:00:00Z"


def _exec(seq: int) -> RecordFrame:
    return RecordFrame(scope="R-1", seq=seq, at=AT, kind="exec", payload={})


class _Session:
    """The wire as `resume` reads it: the session's liveness and the run it names paused, a
    run's journal, and the engine's resume, which continues `in_flight`."""

    def __init__(
        self,
        paused: str | None = None,
        *,
        in_flight: str | None = None,
        journal: Sequence[RecordFrame] = (),
        live: str = "idle",
    ) -> None:
        self.paused = paused
        self.in_flight = in_flight
        self.journal = tuple(journal)
        self.live = live
        self.resumed = 0

    def liveness(self, sid: str) -> Liveness:
        return Liveness.model_validate(
            {"kind": self.live, "last_seq": 4, "paused_run": self.paused}
        )

    def run_journal(self, run_id: str, after: int = 0) -> tuple[RecordFrame, ...]:
        assert self.resumed == 0, "read before the run goes on"
        return self.journal

    def resume(self, sid: str) -> Resumed:
        self.resumed += 1
        reconciled = (self.in_flight,) if self.in_flight is not None else ()
        return Resumed(session_id=SID, reconciled=reconciled, in_flight=self.in_flight)


class _Ctx:
    session_id = SID

    def __init__(self, client: _Session) -> None:
        self.client: Any = client
        self.said: list[tuple[str, Tone]] = []
        self.followed: list[tuple[str, int]] = []

    def say(self, text: str, tone: Tone = Tone.PLAIN, *, indent: str = "") -> None:
        self.said.append((text, tone))

    def follow_run(self, run_id: str, after: int) -> None:
        self.followed.append((run_id, after))


def test_a_paused_run_is_continued_and_followed_from_what_was_seen() -> None:
    wire = _Session(
        "R-1",
        in_flight="R-1",
        journal=[_exec(1), _exec(2), _exec(7)],
    )
    ctx = _Ctx(wire)
    resume(ctx, attaching=True)  # type: ignore[arg-type]
    assert ctx.followed == [("R-1", 7)], "from the last record, not replayed"
    [(text, tone)] = ctx.said
    assert "continuing run R-1" in text and tone is Tone.GOOD


def test_a_run_liveness_does_not_name_paused_is_not_resumed_on_attach() -> None:
    """A harness failure is reported BLOCKED and left without `run_end`, for the person to
    resume if they choose; liveness names no paused run, so opening the session does not run it
    again on its own."""
    wire = _Session(None, in_flight="R-1")
    resume(_Ctx(wire), attaching=True)  # type: ignore[arg-type]
    assert wire.resumed == 0


def test_a_run_still_going_is_not_resumed_on_attach() -> None:
    """Another terminal joining a session whose run is in flight follows it; it does not ask the
    engine to continue what it is already running."""
    wire = _Session("R-1", in_flight="R-1", live="working")
    ctx = _Ctx(wire)
    resume(ctx, attaching=True)  # type: ignore[arg-type]
    assert (wire.resumed, ctx.said) == (0, [])


def test_opening_a_session_with_nothing_paused_says_nothing() -> None:
    wire = _Session()
    ctx = _Ctx(wire)
    resume(ctx, attaching=True)  # type: ignore[arg-type]
    assert (wire.resumed, ctx.said, ctx.followed) == (0, [], [])


def test_the_command_says_when_nothing_is_in_flight() -> None:
    ctx = _Ctx(_Session())
    resume(ctx, attaching=False)  # type: ignore[arg-type]
    assert ctx.said == [("nothing in flight to continue", Tone.DIM)]
