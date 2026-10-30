"""An escape recurring in more than 30% of `locate` calls is promoted
into the snapshot, and a promoted command needs no request."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

from scout_helpers import ListSink

from chickadee.executor.executor import Executor
from chickadee.graph.queries import Graph
from chickadee.isolation.none import Passthrough
from chickadee.process.supervisor import Completed, ProcessSupervisor
from chickadee.schema.journal import EventKind, ScoutEscapeP
from chickadee.scout.escapes import EscapeRequest, Escapes
from chickadee.scout.hints import HINTS_FILE, PROMOTE_ABOVE, Hints
from chickadee.scout.snapshot import build_snapshot


class _Echo(ProcessSupervisor):
    def __init__(self) -> None:
        self.ran: list[str] = []

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: int,
        stdin: bytes | None = None,
        cancel: threading.Event | None = None,
    ) -> Completed:
        self.ran.append(argv[-1])
        return Completed(f"output of {argv[-1]}\n".encode(), b"", 0, False)


def _hints(tmp_path: Path) -> Hints:
    return Hints(tmp_path / "state" / HINTS_FILE, now=lambda: "2026-09-30T00:00:00+00:00")


def test_more_than_thirty_percent_promotes_and_thirty_does_not(tmp_path: Path) -> None:
    hints = _hints(tmp_path)
    for n in range(10):
        asked: list[tuple[str, str | None]] = []
        if n < 4:
            asked.append(("make help", "sha-a"))
        if n < 3:
            asked.append(("git log --oneline -5", None))
        if n == 0:
            asked.append(("ls docs", None))
        hints.record_run(asked)
    assert PROMOTE_ABOVE == 0.30
    assert hints.promoted() == ("make help",)
    help_hint = hints.get("make help")
    assert help_hint is not None and help_hint.hit_rate == 0.4
    log_hint = hints.get("git log --oneline -5")
    assert log_hint is not None and log_hint.hit_rate == 0.3


def test_a_command_counts_once_per_call(tmp_path: Path) -> None:
    hints = _hints(tmp_path)
    hints.record_run([("make help", None), ("make help", None)])
    hints.record_run([])
    hint = hints.get("make help")
    assert hint is not None and hint.hits == 1 and hint.hit_rate == 0.5


def test_the_record_survives_a_restart_with_its_four_fields(tmp_path: Path) -> None:
    hints = _hints(tmp_path)
    hints.record_run([("make help", "sha-a")])
    hints.record_run([("make help", "sha-b")])
    hints.record_run([])
    hints.save()
    again = _hints(tmp_path)
    assert again.runs == 3 and again.promoted() == ("make help",)
    hint = again.get("make help")
    assert hint is not None
    assert (hint.command, hint.added_at, hint.last_output_sha) == (
        "make help",
        "2026-09-30T00:00:00+00:00",
        "sha-b",
    )
    assert abs(hint.hit_rate - 2 / 3) < 1e-9


def test_an_unreadable_record_starts_empty(tmp_path: Path) -> None:
    path = tmp_path / "state" / HINTS_FILE
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    hints = Hints(path)
    assert hints.runs == 0 and hints.promoted() == ()


def test_a_promoted_command_is_in_the_snapshot_and_needs_no_request(
    tmp_path: Path, repo: Path, graph: Graph
) -> None:
    hints = _hints(tmp_path)
    for _ in range(3):
        hints.record_run([("make help", None)])
    sink = ListSink()
    sup = _Echo()
    memo: dict[str, str] = {}
    session = Escapes(
        executor=Executor(backend=Passthrough(), supervisor=sup),
        cwd=repo,
        memo=memo,
        allow_unsandboxed=True,
    )
    promoted = session.run_promoted(hints.promoted(), sink)
    assert promoted == {"make help": "output of make help\n"}
    snap = build_snapshot(repo, [], graph, promoted=promoted, probe=lambda _b: None)
    assert "$ make help" in snap.render()
    # The next locate call in the session: a request for it is answered from memory.
    call = Escapes(
        executor=Executor(backend=Passthrough(), supervisor=sup),
        cwd=repo,
        memo=memo,
        allow_unsandboxed=True,
    )
    out = call.request(EscapeRequest("make help", "task names"), sink)
    assert out.known and call.used == 0
    assert sup.ran == ["make help"], "it ran once, for the snapshot"
    first = sink.of(EventKind.SCOUT_ESCAPE)[0]
    assert isinstance(first, ScoutEscapeP) and "promoted" in first.reason


def test_a_promoted_command_keeps_counting_while_it_is_promoted(
    tmp_path: Path, repo: Path, sink: ListSink
) -> None:
    """The model never asks for what the snapshot already holds; counting only its asks
    would demote a command that is useful on every call, and it would flap in and out."""
    hints = _hints(tmp_path)
    for _ in range(3):
        hints.record_run([("make help", None)])
    session = Escapes(
        executor=Executor(backend=Passthrough(), supervisor=_Echo()),
        cwd=repo,
        allow_unsandboxed=True,
    )
    session.run_promoted(hints.promoted(), sink)
    for _ in range(20):
        session.begin()  # a new locate call; the model asks for nothing
        hints.record_run(session.asked())
    assert hints.promoted() == ("make help",)
    help_hint = hints.get("make help")
    assert help_hint is not None and help_hint.hits == 23


def test_a_promoted_command_that_now_fails_stops_counting(repo: Path, sink: ListSink) -> None:
    session = Escapes(
        executor=Executor(backend=Passthrough(), supervisor=_Echo()),
        cwd=repo,
        allow_unsandboxed=True,
    )
    session.run_promoted(["make help"], sink)
    assert dict(session.asked()).keys() == {"make help"}
    session.run_promoted(["pip install x"], sink)  # the next snapshot: refused, left out
    session.begin()
    assert session.asked() == ()


def test_a_promoted_command_now_refused_is_left_out(repo: Path) -> None:
    sup = _Echo()
    esc = Escapes(
        executor=Executor(backend=Passthrough(), supervisor=sup),
        cwd=repo,
        allow_unsandboxed=True,
    )
    out = esc.run_promoted(["pip install x", "make help"], ListSink())
    assert out is not None and list(out) == ["make help"] and sup.ran == ["make help"]
