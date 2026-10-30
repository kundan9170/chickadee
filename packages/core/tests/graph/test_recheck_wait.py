"""The stop's short wait on the background re-check (decision 10).

What these pin: with a real (fake) server, the indexer settles once the re-check has judged
the dependents, and they read fresh; a queue that does not empty in time says so rather than
waiting on; an index with no servers has nothing to wait for.
"""

from __future__ import annotations

import time
from pathlib import Path

from test_servers import (  # noqa: F401 # pyright: ignore[reportUnusedImport]
    _episode,
    _repo,
    fake,
)

from chickadee.graph.diagnostics import DiagCoverage
from chickadee.graph.indexer import Indexer
from chickadee.graph.servers import EpisodeServers, Recheck, ServerSlots
from chickadee.process.supervisor import ProcessSupervisor


def test_the_index_settles_once_the_dependents_are_judged(tmp_path: Path, fake: None) -> None:  # noqa: F811
    root = _repo(tmp_path)
    ix, graph, servers = _episode(root)
    try:
        (root / "util.py").write_text("def helper(x, y):\n    return x\n", encoding="utf-8")
        ix.reindex([root / "util.py"])
        assert ix.settle(30.0)
        assert graph.diagnostics_in("app.py").coverage is DiagCoverage.TOTAL
    finally:
        servers.close()


def test_a_queue_that_does_not_empty_in_time_says_so(tmp_path: Path) -> None:
    servers = EpisodeServers(tmp_path, ProcessSupervisor(), slots=ServerSlots())
    # Queued with no worker to ask it: it stays unfinished.
    servers._queue.put(Recheck("a.py", None, 1))  # type: ignore[arg-type] # pyright: ignore[reportPrivateUsage,reportArgumentType]
    started = time.monotonic()
    assert servers.wait_rechecked(0.05) is False
    assert time.monotonic() - started < 5
    servers._queue.task_done()  # pyright: ignore[reportPrivateUsage]
    assert servers.wait_rechecked(0.05) is True


def test_an_index_with_no_servers_has_nothing_to_wait_for(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.py").write_text("X = 1\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    assert ix.settle(0.0)
    ix.close()
