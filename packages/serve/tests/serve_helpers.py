"""What the serve tests share: the app over a replay of the demo session, or over a live engine
with a scripted model, each with its tokens."""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from demo_session import FIXTURE, SESSION_ID
from starlette.testclient import TestClient

from chickadee.engine import PacketEngine
from chickadee_serve import server
from chickadee_serve.app import Served, build_app
from chickadee_serve.auth import TokenClass, TokenStore
from chickadee_serve.bind import Uds
from chickadee_serve.frames import RecordFrame, decode_frame
from chickadee_serve.interaction import PendingQuestions
from chickadee_serve.progress import ProgressHub
from chickadee_serve.replay import ReplayFacade


@dataclass
class Harness:
    client: TestClient
    facade: ReplayFacade
    served: Served
    interactive: str
    automation: str
    scratch: Path

    def headers(self, token_class: TokenClass = TokenClass.INTERACTIVE) -> dict[str, str]:
        token = self.interactive if token_class is TokenClass.INTERACTIVE else self.automation
        return {"authorization": f"Bearer {token}"}

    def journal(self, after: int = 0) -> list[RecordFrame]:
        """`/journal` from `after`: the records revealed so far, then closed."""
        response = self.client.get(
            f"/session/{SESSION_ID}/journal", params={"from": after}, headers=self.headers()
        )
        assert response.status_code == 200, response.text
        return [
            f
            for f in (decode_frame(line) for line in response.text.splitlines() if line)
            if isinstance(f, RecordFrame)
        ]


@contextmanager
def replay(
    tmp: Path,
    *,
    speed: float = 0.0,
    start: bool = True,
    timeout_s: float = 5.0,
    depth: int | None = None,
) -> Iterator[Harness]:
    """The demo recording served by a replay app, the pacing thread started unless `start` is
    False. Speed 0 reveals every record as fast as it can, waiting only at the question."""
    hub = ProgressHub() if depth is None else ProgressHub(depth=depth)
    pending = PendingQuestions()
    facade = ReplayFacade(
        FIXTURE,
        SESSION_ID,
        tmp / "scratch",
        speed=speed,
        hub=hub,
        pending=pending,
        timeout_s=timeout_s,
    )
    tokens = TokenStore()
    interactive = tokens.issue(TokenClass.INTERACTIVE)
    automation = tokens.issue(TokenClass.AUTOMATION)
    served = Served(facade, tokens, hub=hub, pending=pending, replay=True)
    with TestClient(build_app(served)) as client:
        if start:
            facade.start()
        try:
            yield Harness(client, facade, served, interactive, automation, tmp / "scratch")
        finally:
            facade.stop()


@dataclass
class Live:
    """The app over a live engine: its client and its tokens."""

    client: TestClient
    interactive: str
    automation: str

    def headers(self, token_class: TokenClass = TokenClass.INTERACTIVE) -> dict[str, str]:
        token = self.interactive if token_class is TokenClass.INTERACTIVE else self.automation
        return {"authorization": f"Bearer {token}"}

    def journal(self, sid: str, after: int = 0) -> list[RecordFrame]:
        response = self.client.get(
            f"/session/{sid}/journal", params={"from": after}, headers=self.headers()
        )
        assert response.status_code == 200, response.text
        return [
            f
            for f in (decode_frame(line) for line in response.text.splitlines() if line)
            if isinstance(f, RecordFrame)
        ]

    def until(
        self, sid: str, kind: str, *, count: int = 1, wait_s: float = 20.0
    ) -> RecordFrame:
        """The `count`th record of `kind` on the session, polled for as a client without the
        live channel would (polling alone is correct)."""
        deadline = time.monotonic() + wait_s
        while True:
            seen = [f for f in self.journal(sid) if f.kind == kind]
            if len(seen) >= count:
                return seen[count - 1]
            assert time.monotonic() < deadline, (
                f"{len(seen)} of {count} {kind} records on {sid}"
            )
            time.sleep(0.02)


@contextmanager
def served_engine(engine: PacketEngine) -> Iterator[Live]:
    """`engine` behind the app as `chickadee-serve` serves it (`server.live`)."""
    tokens = TokenStore()
    interactive = tokens.issue(TokenClass.INTERACTIVE)
    automation = tokens.issue(TokenClass.AUTOMATION)
    with TestClient(build_app(server.live(engine, tokens))) as client:
        yield Live(client, interactive, automation)


def recorded_lines() -> list[dict[str, object]]:
    """The recording's own log lines, read as bytes: what a client must end up holding."""
    log = FIXTURE / "sessions" / SESSION_ID / "session.jsonl"
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


@contextmanager
def live(served: Served) -> Iterator[httpx.Client]:
    """`served` on a Unix socket, as `chickadee serve` serves it, and a client of it.
    Starlette's `TestClient` buffers a whole response, so a stream that never ends can only be
    tested against a real server."""
    sock = Path(tempfile.mkdtemp(prefix="ck-")) / "serve.sock"
    running = server.serve_in_thread(served, Uds(sock))
    try:
        with httpx.Client(
            transport=httpx.HTTPTransport(uds=str(sock)), base_url="http://serve", timeout=10
        ) as client:
            yield client
    finally:
        running.stop()
