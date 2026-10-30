"""Which session the REPL opens: new, named, or the latest; a new one
continuing the repository's open work; and what it saw, written to the transcript."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from engines import answer
from hosting import SID, hosted
from repl_driver import Script, Typed, printed, repl

from chickadee_cli.render.style import Style
from chickadee_cli.repl import Repl
from chickadee_cli.session import (
    NoSessionToContinue,
    ask_which,
    continuing_line,
    open_session,
)
from chickadee_serve.bind import Uds
from chickadee_serve.frames import ErrorBody, ErrorCode
from chickadee_wire.client import Client
from chickadee_wire.errors import WireError
from chickadee_wire.models import OpenWorkInfo, SessionInfo


def test_new_named_and_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    with hosted(monkeypatch) as host, host.client() as client:
        named = open_session(client, session_id=SID, resume_latest=False)
        assert named.session_id == SID
        fresh = open_session(client, session_id=None, resume_latest=False)
        assert fresh.session_id != SID
        latest = open_session(client, session_id=None, resume_latest=True)
        assert latest.session_id == fresh.session_id, "the one updated last"


def test_continue_with_no_session_says_so() -> None:
    client = Client(
        Uds(Path("/x.sock")),
        "t",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=[])),
    )
    with pytest.raises(NoSessionToContinue):
        open_session(client, session_id=None, resume_latest=True)


def test_the_transcript_holds_what_was_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [Typed("hello"), Typed("/exit", after=printed("hi there"))]
    with repl(monkeypatch, [answer("hi there")], lines) as (source, _code):
        assert source.transcript is not None
        text = source.transcript.read_text(encoding="utf-8")
    assert "> hello" in text and "hi there" in text
    assert all(line in text for line in source.out)


# -- a new session starts where the work is ----------------------


class _Works:
    """A client holding open work: what `open_session` asked, and what it created."""

    def __init__(self, *keys: str) -> None:
        self.works = tuple(
            OpenWorkInfo(
                key=k,
                title=f"work {k}",
                work=k,
                mode="in_place",
                worktree_path=None,
                session_id="S-old",
                at=f"2026-10-24T10:00:0{n}Z",
            )
            for n, k in enumerate(keys)
        )
        self.listed = 0
        self.created: list[str | None] = []

    def open_works(self) -> tuple[OpenWorkInfo, ...]:
        self.listed += 1
        return self.works

    def create_session(self, session_id: str | None = None, *, adopt: str | None = None) -> Any:
        self.created.append(adopt)
        found = next((w for w in self.works if w.key == adopt), None)
        return SessionInfo(session_id="S-new", last_seq=0, continuing=found)


def _open(client: _Works, **kwargs: Any) -> SessionInfo:
    return open_session(client, session_id=None, resume_latest=False, **kwargs)  # type: ignore[arg-type]


def test_with_no_open_work_a_new_session_continues_nothing() -> None:
    client = _Works()
    assert _open(client).continuing is None
    assert client.created == [None]


def test_one_open_work_is_continued_without_asking() -> None:
    client = _Works("a")
    asked: list[object] = []
    info = _open(client, choose=lambda works: asked.append(works) or None)
    assert client.created == ["a"] and not asked
    assert continuing_line(info) == (
        'Continuing "work a", in your checkout. /handoff to land it; '
        "start with --new to continue none of it."
    )


def test_a_batch_run_is_told_only_what_it_can_do() -> None:
    """Review of 2026-10-25: `chickadee run` was told to type `/handoff`, a REPL command, and
    to run `chickadee --new`, which is the REPL's spelling."""
    info = _open(_Works("a"))
    assert continuing_line(info, repl=False) == (
        'Continuing "work a", in your checkout; --new continues none of it.'
    )


class _OldServer(_Works):
    """A server of an earlier 4.x wire, which has no `/works` route."""

    def open_works(self) -> tuple[OpenWorkInfo, ...]:
        self.listed += 1
        raise WireError(404, ErrorBody(code=ErrorCode.NOT_FOUND, message="Not Found"))


def test_a_server_without_open_work_starts_a_fresh_session() -> None:
    """Review of 2026-10-25: a newer CLI against an older server could not start a session at
    all, since the start read `/works` first and its 404 ended the command."""
    client = _OldServer("a")
    assert _open(client).continuing is None
    assert client.created == [None] and client.listed == 1


def test_any_other_refusal_from_open_work_is_still_said() -> None:
    class Refusing(_Works):
        def open_works(self) -> tuple[OpenWorkInfo, ...]:
            raise WireError(500, ErrorBody(code=ErrorCode.INTERNAL, message="broken"))

    with pytest.raises(WireError):
        _open(Refusing("a"))


def test_several_are_chosen_from_and_the_choice_may_be_none() -> None:
    client = _Works("newest", "older")
    _open(client, choose=lambda works: works[1].key)
    _open(client, choose=lambda _works: None)
    assert client.created == ["older", None]


def test_with_nothing_to_choose_a_batch_run_takes_the_newest() -> None:
    client = _Works("newest", "older")
    _open(client)
    assert client.created == ["newest"]


def test_new_continues_nothing_and_reads_no_work() -> None:
    client = _Works("a")
    _open(client, fresh=True)
    assert client.created == [None] and client.listed == 0


@pytest.mark.parametrize(
    ("typed", "chosen"), [("", "newest"), ("2", "older"), ("n", None), ("what", "newest")]
)
def test_the_question_takes_the_newest_by_default(typed: str, chosen: str | None) -> None:
    works = _Works("newest", "older").works
    shown: list[str] = []
    assert ask_which(works, read=lambda _prompt: typed, write=shown.append) == chosen
    assert shown[0] == "Open work in this repository:"
    assert shown[1] == '  1. "work newest", in your checkout'
    assert shown[-1] == "  n. start fresh"


@pytest.mark.parametrize("stop", [EOFError, KeyboardInterrupt])
def test_the_question_ended_without_an_answer(stop: type[BaseException]) -> None:
    """Review of 2026-10-25: Ctrl-D at the question raised `EOFError` out of the start. An end
    of input starts fresh, as `n` does; Ctrl-C is the person stopping, and is raised for the
    REPL to end quietly."""
    works = _Works("newest", "older").works

    def read(_prompt: str) -> str:
        raise stop

    if stop is EOFError:
        assert ask_which(works, read=read, write=lambda _line: None) is None
    else:
        with pytest.raises(KeyboardInterrupt):
            ask_which(works, read=read, write=lambda _line: None)


# -- a turn says where its work runs -----------------


@pytest.mark.parametrize(
    ("worktree", "in_place", "sent"),
    [(False, False, None), (True, False, "worktree"), (False, True, "in_place")],
)
def test_the_flags_say_where_the_session_works(
    worktree: bool, in_place: bool, sent: str | None
) -> None:
    from chickadee_cli.main import _workspace  # pyright: ignore[reportPrivateUsage]

    assert _workspace("run", worktree, in_place) == sent


@pytest.mark.parametrize(
    ("in_place", "sent", "only"),
    [(False, "worktree", None), (True, "in_place", "in_place")],
)
def test_run_yes_defaults_to_a_worktree_and_filters_nothing_it_was_not_told(
    monkeypatch: pytest.MonkeyPatch, in_place: bool, sent: str, only: str | None
) -> None:
    """D230 question 4: an unattended run (`--yes`) runs in a worktree unless `--in-place`.
    That default is not a flag the person typed, so it keeps no open work out (Amendment 4
    decision 4); `--in-place` does."""
    import chickadee_cli.main as main

    seen: dict[str, Any] = {}
    monkeypatch.setattr(main, "_turn", lambda *_a, **kw: seen.update(kw))  # pyright: ignore[reportUnknownLambdaType]
    main.run("tidy up", yes=True, in_place=in_place)
    assert (seen["workspace"], seen["only"]) == (sent, only)


def test_both_flags_name_two_places_and_are_refused() -> None:
    import typer

    from chickadee_cli.main import _workspace  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(typer.Exit) as left:
        _workspace("run", True, True)
    assert left.value.exit_code == 1


def test_the_flag_reaches_the_persons_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    from chickadee_cli.batch import BatchIO, follow
    from chickadee_cli.render.style import Style

    how = BatchIO(json=False, out=lambda _l: None, err=lambda _l: None, style=Style(tty=False))
    with hosted(monkeypatch, [answer("ok")]) as host, host.client() as client:
        after = client.session_info(SID).last_seq
        follow(client, SID, how, after=after, send="tidy up", workspace="worktree")
        records = list(client.session_journal(SID, after))
    (turn,) = [r for r in records if r.kind == "turn" and r.payload.get("role") == "user"]
    assert turn.payload["workspace"] == "worktree"


# -- the flag moves the session ------------------------------------------


def _mixed() -> _Works:
    client = _Works("newest", "older")
    newest = client.works[0].model_copy(update={"mode": "worktree", "worktree_path": "/wt"})
    client.works = (newest, client.works[1])
    return client


def test_in_place_never_continues_worktree_work() -> None:
    client = _mixed()
    _open(client, only="in_place")
    assert client.created == ["older"], "the newest is a worktree's, so it is passed over"


def test_worktree_never_continues_work_in_the_checkout() -> None:
    client = _Works("a")
    _open(client, only="worktree")
    assert client.created == [None], "nothing of that kind: a fresh session"


def test_the_repl_sends_the_flag_with_its_first_turn_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later word moves the session itself: a flag typed at the start never pulls it back."""
    sent: list[object] = []
    with hosted(monkeypatch, [answer("one"), answer("two")]) as host, host.client() as client:
        real = client.turn

        def turn(sid: str, text: str, **kw: Any) -> Any:
            sent.append(kw.get("workspace"))
            return real(sid, text, **kw)

        monkeypatch.setattr(client, "turn", turn)
        source = Script(
            [
                Typed("first", after=printed("Ctrl-D to leave")),
                Typed("second", after=printed("one")),
            ]
        )
        Repl(
            client,
            SID,
            source=source,
            write=source.out.append,
            style=Style(tty=False),
            workspace="in_place",
        ).run()
    assert sent == ["in_place", None]
