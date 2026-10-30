"""What the wire and CLI tests share: a repository with an engine config, hosted in this
process with a scripted model behind its zeus rung, and waits on a session's records."""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from engines import Step, config_toml, raw_config, scripted, with_prices, with_sessions
from repos import init_repo

from chickadee_serve.frames import RecordFrame
from chickadee_wire.client import Client
from chickadee_wire.host import Hosted, host_in_process

SID = "S-wire"
WAIT_S = 20.0


def short_root() -> Path:
    """A short root: the socket lives in the repository's git dir, and AF_UNIX paths are
    limited to about 100 bytes, which pytest's own temporary paths can exceed."""
    return Path(tempfile.mkdtemp(prefix="ck-"))


def repository(
    monkeypatch: pytest.MonkeyPatch, root: Path, *, auto_title: bool = False
) -> Path:
    """A repository holding session `SID`, and the engine config the host loads for it. A
    session's name is a model call that would take a scripted answer meant for a turn:
    off unless `auto_title`, which also prices the title role's rung so the call is made."""
    xdg = root / "xdg"
    (xdg / "chickadee").mkdir(parents=True)
    raw = with_prices(raw_config())
    raw["interaction"] = {**raw.get("interaction", {}), "auto_title": auto_title}
    if auto_title:
        cheap = raw["rungs"]["low-param"]
        raw["prices"].setdefault(cheap, {"price_in": 0.1, "price_out": 0.2})
    (xdg / "chickadee" / "config.toml").write_text(config_toml(raw))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    for name in [k for k in os.environ if k.startswith("CHICKADEE_")]:
        monkeypatch.delenv(name)
    return with_sessions(init_repo(root / "repo", commit=None), SID)


@contextmanager
def hosted(
    monkeypatch: pytest.MonkeyPatch, script: Sequence[Step] = (), *, auto_title: bool = False
) -> Iterator[Hosted]:
    """`host_in_process` over a fresh repository, its model answering from `script` (the title
    role's call too, after the first turn, with `auto_title`)."""
    repo = repository(monkeypatch, short_root(), auto_title=auto_title)
    scripted(monkeypatch, script)
    with host_in_process(repo) as host:
        yield host


def until(
    client: Client, kind: str, *, count: int = 1, sid: str = SID, wait_s: float = WAIT_S
) -> RecordFrame:
    """The `count`th record of `kind` on the session, polled for over `/journal`."""
    deadline = time.monotonic() + wait_s
    while True:
        seen = [f for f in client.session_journal(sid) if f.kind == kind]
        if len(seen) >= count:
            return seen[count - 1]
        assert time.monotonic() < deadline, f"{len(seen)} of {count} {kind} records"
        time.sleep(0.02)


def answered(client: Client, count: int, *, sid: str = SID, wait_s: float = WAIT_S) -> None:
    """Wait until the session holds `count` final answers: assistant turns calling no tool,
    each the end of one user turn."""
    deadline = time.monotonic() + wait_s
    while True:
        finals = [
            f
            for f in client.session_journal(sid)
            if f.kind == "turn"
            and f.payload.get("role") == "assistant"
            and not f.payload.get("calls")
        ]
        if len(finals) >= count:
            return
        assert time.monotonic() < deadline, f"{len(finals)} of {count} answers on {sid}"
        time.sleep(0.02)
