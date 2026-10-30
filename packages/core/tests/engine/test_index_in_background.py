"""A repository's index built beside the person (`PacketEngine.index_in_background`):
the screen opens at once, and the engine says what it is doing until it is done.

What these pin: a repository with no index starts a build and one initialised starts none;
liveness says it is building, and how far (preparing before any file is read, then
each file of the total, then linking), then nothing; a session composed while it built is
rebuilt at its next turn, so its graph is the index; a build that fails is said, with where to
look, and with no progress.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import APP, build, worktree_count, zeus_delegates
from repos import init_repo
from zeus_turn_helpers import SID, WAIT_S, answer, config_for, start, state, with_sessions

import chickadee.engine as engine_module
from chickadee.engine import IndexProgress, PacketEngine
from chickadee.init import orchestrate


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = init_repo(tmp_path / "repo", {"app.py": "def go() -> int:\n    return 1\n"})
    return with_sessions(root, SID)


def _held_build(monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, list[int]]:
    """The build's `init` held until the test releases it, and how many times it ran."""
    release = threading.Event()
    ran: list[int] = []
    real = engine_module._init  # pyright: ignore[reportPrivateImportUsage]

    def held(*args: Any, **kwargs: Any) -> Any:
        assert release.wait(WAIT_S), "the test never released the build"
        ran.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(engine_module, "_init", held)
    return release, ran


def _engine(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    script: list[Any],
) -> PacketEngine:
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)
    return engine


def _until(done: Any) -> None:
    deadline = threading.Event()
    timer = threading.Timer(WAIT_S, deadline.set)
    timer.start()
    try:
        while not done():
            assert not deadline.wait(0.02), "never happened"
    finally:
        timer.cancel()


def test_the_index_is_built_beside_the_person_and_said_until_it_is_done(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    release, ran = _held_build(monkeypatch)
    engine = _engine(monkeypatch, repo, raw, engines, [answer("hello"), answer("again")])

    assert engine.index_in_background() is True
    engine.turn(SID, "hi").wait(timeout=WAIT_S)  # a turn while it builds: nothing waits
    assert engine.liveness(SID).indexing == "building this repository's index"
    assert engine.liveness(SID).index_progress == IndexProgress("preparing")

    # Each composition opens the checkout's graph: the count of compositions.
    opened: list[object] = []
    real = engine_module.opened_until_closed  # pyright: ignore[reportPrivateImportUsage]

    def counted(root: Path, stack: Any) -> Any:
        graph = real(root, stack)
        opened.append(graph)
        return graph

    monkeypatch.setattr(engine_module, "opened_until_closed", counted)
    release.set()
    _until(lambda: engine.liveness(SID).indexing is None)
    assert ran == [1]
    engine.turn(SID, "again").wait(timeout=WAIT_S)
    assert len(opened) == 1, "composed while it built: rebuilt once it is"
    assert opened[0] is not None, "and its graph is the index now"
    assert engine.index_in_background() is False, "initialised: nothing more to build"


def test_a_build_that_fails_is_said(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    def fails(*args: Any, **kwargs: Any) -> Any:
        raise engine_module.InitError("the deny list could not be read")

    monkeypatch.setattr(engine_module, "_init", fails)
    engine = _engine(monkeypatch, repo, raw, engines, [answer("hello")])
    engine.index_in_background()
    engine.turn(SID, "hi").wait(timeout=WAIT_S)
    _until(lambda: engine.liveness(SID).indexing != "building this repository's index")
    said = engine.liveness(SID).indexing
    assert said is not None and "the deny list could not be read" in said
    assert "`chickadee init` says why" in said
    assert engine.liveness(SID).index_progress is None, "failed: said with no progress"


def test_a_run_started_while_it_builds_waits_for_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The build holds the repository's lease: a run waits for it, rather than failing on the
    lease, and makes its worktree only once the index is there."""
    # Committed and never initialised: `coder_helpers.make_repo` builds its graph.
    repo = init_repo(
        tmp_path / "repo",
        {"app.py": APP, ".gitignore": "__pycache__/\n"},
    )
    repo = with_sessions(repo, SID)
    # Held inside `init`, the lease taken: what a run meets while the index is built.
    release = threading.Event()
    ran: list[int] = []
    real = orchestrate._build_index  # pyright: ignore[reportPrivateUsage]

    def held(*args: Any, **kwargs: Any) -> Any:
        assert release.wait(WAIT_S), "the test never released the build"
        ran.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(orchestrate, "_build_index", held)
    engine, _ = build(monkeypatch, repo, raw, [*zeus_delegates(), answer("done")])
    engines.append(engine)
    assert engine.index_in_background() is True

    handle = engine.turn(SID, "make go return 2")
    with pytest.raises(TimeoutError):
        handle.wait(timeout=1.0)
    assert worktree_count(repo) == 1, "no worktree while the index is built"

    release.set()
    handle.wait(timeout=WAIT_S)
    assert ran == [1]
    assert worktree_count(repo) == 2
    replies = [t.text for t in state(engine).turns() if t.role == "assistant"]
    assert replies[-1] == "done"


def test_the_build_says_how_far_it_has_got(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Each report the pass makes reaches liveness while it runs; none once it is done."""
    seen: list[IndexProgress | None] = []
    real = engine_module._init  # pyright: ignore[reportPrivateImportUsage]

    def watched(*args: Any, progress: Any, **kwargs: Any) -> Any:
        def told(p: IndexProgress) -> None:
            progress(p)
            seen.append(engine.liveness(SID).index_progress)

        return real(*args, progress=told, **kwargs)

    monkeypatch.setattr(engine_module, "_init", watched)
    engine = _engine(monkeypatch, repo, raw, engines, [answer("hello")])
    engine.index_in_background()
    _until(lambda: engine.liveness(SID).indexing is None)
    assert seen[0] == IndexProgress("reading", 0, len(seen) - 1)
    assert seen[-1] == IndexProgress("linking", len(seen) - 1, len(seen) - 1)
    assert [p.done for p in seen if p is not None and p.stage == "reading"] == list(
        range(len(seen) - 1)
    )
    assert engine.liveness(SID).index_progress is None
