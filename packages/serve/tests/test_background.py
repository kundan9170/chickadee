"""`serve_in_thread`'s server stops when asked, and a stop that cannot happen says so rather
than letting its caller tear down what a live server still uses."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import cast

import httpx
import pytest
import uvicorn
from engines import config_for, raw_config, start, with_sessions
from repos import init_repo

from chickadee_serve import server
from chickadee_serve.auth import TokenStore
from chickadee_serve.bind import Uds
from chickadee_serve.frames import TokenClass
from chickadee_serve.server import Background, serve_in_thread

SID = "S-stop"


class _Fake:
    """What `Background.stop` uses of a uvicorn server."""

    def __init__(self) -> None:
        self.should_exit = False
        self.force_exit = False

    def handle_exit(self, sig: int, frame: object) -> None:
        self.should_exit = True


def test_a_server_that_will_not_stop_is_forced_and_then_refused() -> None:
    release = threading.Event()
    fake = _Fake()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="did not stop"):
            Background(cast("uvicorn.Server", fake), thread).stop(wait_s=0.05)
        assert fake.should_exit and fake.force_exit, "asked, then forced"
    finally:
        release.set()


def test_a_server_that_stops_when_forced_is_stopped() -> None:
    fake = _Fake()

    def run() -> None:
        while not fake.force_exit:
            time.sleep(0.01)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    Background(cast("uvicorn.Server", fake), thread).stop(wait_s=0.05)
    assert not thread.is_alive()


def test_an_attached_stream_does_not_hold_the_server_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A client attached to a live session's `/events` and never leaving: the server still
    stops at once, because its exit ends every attached stream first (`Served.stopping`).
    Without that, a graceful shutdown waits on the client."""
    repo = with_sessions(init_repo(tmp_path / "repo", commit=None), SID)
    engine, _ = start(monkeypatch, repo, config_for(raw_config()))
    tokens = TokenStore()
    token = tokens.issue(TokenClass.INTERACTIVE)
    sock = Path(tempfile.mkdtemp(prefix="ck-")) / "serve.sock"
    running = serve_in_thread(server.live(engine, tokens), Uds(sock))
    client = httpx.Client(
        transport=httpx.HTTPTransport(uds=str(sock)),
        base_url="http://serve",
        timeout=30,
        headers={"authorization": f"Bearer {token}"},
    )
    try:
        with client.stream("GET", f"/session/{SID}/events") as attached:
            assert attached.status_code == 200
            started = time.monotonic()
            running.stop(wait_s=5.0)
            assert time.monotonic() - started < 3.0, "stopped without waiting on the client"
            assert not running.thread.is_alive()
    finally:
        client.close()
        engine.close()
