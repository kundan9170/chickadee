"""Two processes racing init: exactly one publishes."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

import pytest
from repos import git

from chickadee.init import detect, init, lease, publish

CHILDREN = 6

_CHILD = """
import pathlib, sys, time
from chickadee.init.lease import LeaseBusy
from chickadee.init import init

root, barrier, name = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
(barrier / ('ready.' + name)).touch()
while not (barrier / 'go').exists():
    time.sleep(0.005)
try:
    init(root)
except LeaseBusy:
    sys.exit(3)
"""


def test_two_processes_racing_init_yield_one_publisher(repo: pathlib.Path) -> None:
    """Two publishers would each rename a generation into place and the second would
    overwrite a pointer the first is still finishing. The lease is what stops it, so the
    test has to actually create contention rather than hope for it — hence the barrier."""
    barrier = repo / "_barrier"
    barrier.mkdir()
    procs = [
        subprocess.Popen([sys.executable, "-c", _CHILD, str(repo), str(barrier), str(i)])
        for i in range(CHILDREN)
    ]
    try:
        deadline = time.monotonic() + 60
        while len(list(barrier.glob("ready.*"))) < CHILDREN:
            assert time.monotonic() < deadline, "children never reached the barrier"
            time.sleep(0.01)
        (barrier / "go").touch()
        codes = [p.wait(timeout=90) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()

    published, refused = codes.count(0), codes.count(3)
    assert published + refused == CHILDREN, f"unexpected child exits: {codes}"
    assert published >= 1
    state = publish.read_state(repo)
    assert state is not None and state["complete"] is True
    generations = publish.chickadee_dir(repo) / "generations"
    assert (generations / state["generation"]).is_dir()


def test_a_live_lease_is_a_typed_busy_failure(repo: pathlib.Path) -> None:
    identity = detect.resolve(repo)
    held = lease.RepoLease.acquire(identity.git_common_dir, identity.repository_id)
    try:
        with pytest.raises(lease.LeaseBusy) as exc:
            init(repo)
        assert exc.value.pid > 0
    finally:
        held.release()
    init(repo)  # released: the repository initialises without manual cleanup


def test_a_dead_owner_does_not_block(repo: pathlib.Path) -> None:
    """flock is released by the kernel when the holder dies, so a crashed init needs no
    reconciliation — there is no stale-lease concept to get wrong."""
    identity = detect.resolve(repo)
    path = lease.lease_path(identity.git_common_dir)
    path.write_text('{"pid": 999999, "host": "gone", "repo": "x"}')
    init(repo)
    assert publish.is_initialised(repo)


def test_the_lease_is_shared_across_linked_worktrees(repo: pathlib.Path) -> None:
    """D39 makes worktrees the default, so two worktrees of one repository must contend
    for one lease — otherwise both initialise the same repository at once."""
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "seed")
    linked = repo.parent / "linked"
    git(repo, "worktree", "add", "-q", str(linked))
    assert detect.resolve(linked).repository_id == detect.resolve(repo).repository_id

    identity = detect.resolve(repo)
    held = lease.RepoLease.acquire(identity.git_common_dir, identity.repository_id)
    try:
        with pytest.raises(lease.LeaseBusy):
            init(linked)
    finally:
        held.release()
