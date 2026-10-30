"""The recordings the CLI's tests read: the recorded run, framed as the server frames
it or installed into a state directory, and the demo session with that run, served as a
replay over a real socket. One home for what `test_render_lines`, `test_run_follower`,
`test_batch` and `tui/` each kept a copy of.

**The recordings are `packages/serve/tests`'s**, built and held by their own tests
(`recorded_run.py`, `demo_session.py`, `test_serve_fixture.py`): regenerated from the code,
never edited by hand. This module only reads them, and reads its ids from them (the session's
directory, the run journal's first line) rather than restating them.

**Core-free at import**: the replay's server, which reaches the engine, is imported inside
`replayed` only, so a test that reads a recording loads no engine.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from chickadee_serve.frames import RecordFrame
from chickadee_wire.client import Client

__all__ = [
    "DEMO_FIXTURE",
    "RUN_FIXTURE",
    "Replayed",
    "demo_session_id",
    "install_run",
    "recorded_run_id",
    "recording",
    "replayed",
    "run_frames",
]

_SERVE_TESTS: Final = Path(__file__).resolve().parents[2] / "serve" / "tests"
RUN_FIXTURE: Final = _SERVE_TESTS / "fixtures" / "run"
"""The recorded task run (`serve/tests/recorded_run.py`): Athena plans two tasks,
T1 fails its tests, passes on feedback and is committed, T2 ends its round at its limit and is
abandoned, and the run ends `PARTIAL`."""
DEMO_FIXTURE: Final = _SERVE_TESTS / "fixtures" / "demo"
"""The demo session (`serve/tests/demo_session.py`), whose coder call is the recorded run."""

_JOURNAL: Final = "journal.jsonl"


def recorded_run_id() -> str:
    """The recorded run's id, as its journal holds it."""
    with (RUN_FIXTURE / _JOURNAL).open(encoding="utf-8") as fh:
        return str(json.loads(fh.readline())["run_id"])


def demo_session_id() -> str:
    """The demo recording's one session, by its directory."""
    (session,) = (DEMO_FIXTURE / "sessions").iterdir()
    return session.name


def run_frames() -> list[RecordFrame]:
    """The recorded run's journal, framed as `/run/{id}/events` frames it: the scope and
    `run_id` the run's, the envelope's `task_id` and `round` kept."""
    frames: list[RecordFrame] = []
    for line in (RUN_FIXTURE / _JOURNAL).read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        frames.append(
            RecordFrame(
                scope=event["run_id"],
                run_id=event["run_id"],
                seq=event["seq"],
                at=event["ts"],
                kind=event["kind"],
                payload=event["payload"],
                task_id=event.get("task_id"),
                round=event.get("round"),
            )
        )
    return frames


def install_run(runs: Path, run_id: str | None = None) -> Path:
    """The recorded run's journal copied to `runs/<run_id>/journal.jsonl`, as a repository's
    run journals are kept, and its blobs beside it, which a replay serves; the
    journal's path. `run_id` defaults to the run's own."""
    target = runs / (run_id or recorded_run_id())
    target.mkdir(parents=True)
    shutil.copytree(RUN_FIXTURE / "blobs", target / "blobs")
    return Path(shutil.copy(RUN_FIXTURE / _JOURNAL, target / _JOURNAL))


def recording(dest: Path) -> Path:
    """A recording a replay can serve whole: the demo session and, under `runs/`, the run its
    `episode` names, which the demo fixture alone does not keep."""
    shutil.copytree(DEMO_FIXTURE, dest)
    install_run(dest / "runs")
    return dest


@dataclass(frozen=True)
class Replayed:
    """A replay being served: a client of it, its session, and the directory it serves from
    (its socket and token file, as `chickadee serve --replay` announces them)."""

    client: Client
    session_id: str
    directory: Path


@contextmanager
def replayed(source: Path, *, speed: float = 0.0) -> Iterator[Replayed]:
    """`source` (a `recording`) served as `chickadee serve --replay` serves it, on a socket in
    a short directory of its own, and a client attached to it there. Speed 0 reveals every
    record at once, waiting only at a question, as live, for an answer."""
    from chickadee_serve.auth import TokenStore
    from chickadee_serve.server import replay, running, serve_in_thread, socket_in

    # A short root: AF_UNIX paths are limited to about 100 bytes (`hosting.short_root`).
    directory = Path(tempfile.mkdtemp(prefix="ck-r-"))
    session_id = demo_session_id()
    try:
        store = TokenStore()
        served, facade = replay(source, session_id, directory / "state", store, speed=speed)
        with running(store, directory, lambda: socket_in(directory)) as up:
            server = serve_in_thread(served, up.bind)
            facade.start()  # once it listens, so a client sees the replay from its start
            try:
                with Client.attach(directory) as client:
                    yield Replayed(client, session_id, directory)
            finally:
                facade.stop()
                server.stop()
    finally:
        shutil.rmtree(directory, ignore_errors=True)
