"""The repository lease beside a background index build (`init.lease.building`).

What these pin: while this process's own build holds the lease, a taker in this process waits
for it and then has it; a lease another process holds is refused at once, as before, even with
a build of this process registered.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from repos import init_repo

from chickadee.errors import RepoLeaseBusy
from chickadee.init.detect import resolve
from chickadee.init.lease import LeaseBusy, RepoLease, building, take_write_lease


def test_this_process_s_build_is_waited_for(tmp_path: Path) -> None:
    identity = resolve(init_repo(tmp_path / "repo"))
    taken = threading.Event()

    def build() -> None:
        with building(identity.repository_id):
            lease = RepoLease.acquire(identity.git_common_dir, identity.repository_id)
            taken.set()
            time.sleep(0.5)
            lease.release()

    builder = threading.Thread(target=build)
    builder.start()
    assert taken.wait(5)
    started = time.monotonic()
    with take_write_lease(identity):
        waited = time.monotonic() - started
    builder.join(5)
    assert waited >= 0.3, "it waited for the build, never refused it"


def test_a_lease_another_process_holds_is_refused_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with a build of this process registered: only this process's own hold is waited
    for, so another process's is refused as before."""
    identity = resolve(init_repo(tmp_path / "repo"))

    def held_elsewhere(_dir: Path, _repo: str) -> RepoLease:
        raise LeaseBusy(os.getpid() + 1, "elsewhere")

    monkeypatch.setattr(RepoLease, "acquire", staticmethod(held_elsewhere))
    started = time.monotonic()
    with building(identity.repository_id), pytest.raises(RepoLeaseBusy, match="elsewhere"):
        take_write_lease(identity)
    assert time.monotonic() - started < 1.0
