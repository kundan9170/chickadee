"""`PacketEngine.coder` through the real box:
it names the run and takes the lease and does nothing else; the run starts when its result is
asked for; and a second episode while one is in flight is `EngineBusy`, naming the run."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from coder_helpers import APP, SID, build, make_repo, one_task, solved
from zeus_turn_helpers import WAIT_S

from chickadee.delegation import BoxEpisode, OneEpisode
from chickadee.engine import PacketEngine
from chickadee.errors import EngineBusy
from chickadee.init import detect
from chickadee.init.lease import LeaseBusy, RepoLease
from chickadee.journal import paths
from chickadee.state.roots import roots as resolve_roots


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def test_coder_names_the_run_and_does_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Item 10: Zeus's `episode` line must be durable before the run does anything, so
    `coder()` writes no journal; the first `result()` starts it."""
    engine, _ = build(monkeypatch, repo, raw, solved("return 1", "return 2"))
    engines.append(engine)
    handle = engine.coder(one_task(), session_id=SID)
    run_dir = paths.run_dir(resolve_roots(repo), handle.run_id)
    assert not run_dir.exists(), "coder() wrote a journal before its result was asked for"

    result = handle.result()

    assert (result.outcome, result.why) == ("COMPLETED", None)
    assert (result.tasks_committed, result.tasks_total) == (1, 1)
    assert run_dir.exists()
    assert (repo / "app.py").read_text() == APP, "a worktree run left the checkout alone"


def test_a_second_episode_in_flight_is_engine_busy(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [*solved("return 1", "return 2"), *solved("return 2", "return 3")]
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    first = engine.coder(one_task(), session_id=SID)

    with pytest.raises(EngineBusy, match=first.run_id):
        engine.coder(one_task("Return 3"), session_id=SID)

    first.result()
    # The session's one worktree: the second call builds on the first's edit.
    second = engine.coder(one_task("Return 3"), session_id=SID)
    assert second.result().outcome == "COMPLETED"


def test_an_abandoned_episode_frees_the_engine_and_the_lease(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A handle whose result will never be asked for gives back what `coder()` took."""
    engine, _ = build(monkeypatch, repo, raw, solved("return 1", "return 2"))
    engines.append(engine)
    first = engine.coder(one_task(), session_id=SID)
    first.abandon()
    assert not paths.run_dir(resolve_roots(repo), first.run_id).exists()

    again = engine.coder(one_task(), session_id=SID)
    assert again.result().outcome == "COMPLETED"


def test_reads_stay_concurrent_with_an_episode(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Only a second episode is refused: the session's budget and its header still answer
    while one runs."""
    engine, _ = build(monkeypatch, repo, raw, solved("return 1", "return 2"))
    engines.append(engine)
    handle = engine.coder(one_task(), session_id=SID)
    done = threading.Event()
    reader = threading.Thread(target=lambda: (engine.budget(SID), done.set()))
    reader.start()
    handle.result()
    reader.join(WAIT_S)
    assert done.is_set()


def test_closing_the_engine_gives_back_an_unstarted_episodes_lease(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
) -> None:
    """`close()` releases leases: an episode named and never asked for
    is given up, so the lease is back before `close()` returns."""
    engine, _ = build(monkeypatch, repo, raw)
    engine.coder(one_task(), session_id=SID)
    assert not lease_is_free(repo)
    engine.close()
    assert lease_is_free(repo)


def lease_is_free(repo: Path) -> bool:
    identity = detect.resolve(repo)
    try:
        RepoLease.acquire(identity.git_common_dir, identity.repository_id).release()
    except LeaseBusy:
        return False
    return True


def test_no_episode_opens_once_the_engine_is_closing() -> None:
    """Close pauses the run in flight, and Zeus's turn outlives that for a moment; a
    `coder` call made in it is refused, never started only to be ended by the close."""
    episodes = OneEpisode()
    episodes.stop()
    with pytest.raises(EngineBusy, match="closing"):
        episodes.open(
            lambda _release: pytest.fail("an episode opened during close"), session_id="s1"
        )


def test_the_run_in_flight_is_named_only_to_its_own_session() -> None:
    """D240 §2.3, §2.11: one engine serves several sessions, and what one may do to the run in
    flight (pause, drop, list, be refused for it) asks whose it is."""
    episodes = OneEpisode()
    run = cast(BoxEpisode, SimpleNamespace(run_id="R1"))
    episodes.open(lambda _release: run, session_id="s1")
    assert episodes.busy() == episodes.busy("s1") == "run R1"
    assert episodes.busy("s2") is None
    assert episodes.current("s1") is run
    assert episodes.current("s2") is None
    assert episodes.current() is run
