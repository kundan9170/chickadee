"""The two servers `chickadee serve` runs, composed in one place.

- **`live`** puts a `PacketEngine` behind the app: its questions wait on the wire
  (`RemoteInteraction`, CL-5's timeout from its own config), and each turn's events are pumped
  onto the progress channel (`TurnPumps`).
- **`replay`** puts a recorded session behind the same app, read-only.
- **`listen`** runs either on its bind until a signal stops it, and **`Running`** removes what
  the server left on disk (the socket and the token file), however it stopped.
- **`serve_in_thread`** runs either on a thread of the caller's process, for a frontend that
  hosts its own engine (`chickadee_wire.host`) and for tests; `Background.stop` ends it.

`__main__` parses arguments and calls these; a test or an in-process host calls them
directly, so no second composition of a served engine exists.
"""

from __future__ import annotations

import contextlib
import signal
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import FrameType
from typing import Final

import uvicorn

from chickadee.durable import FileLock, LockHeld, make_dir
from chickadee.engine import PacketEngine
from chickadee.errors import ConfigError
from chickadee.state.layers import LayerReadError
from chickadee.state.merge import MergeError
from chickadee.state.roots import NotARepository
from chickadee_serve.app import Served, build_app
from chickadee_serve.auth import TOKEN_FILE, TokenClass, TokenStore
from chickadee_serve.bind import (
    SOCKET_NAME,
    AlreadyServed,
    Bind,
    BindRefused,
    Tcp,
    Uds,
    claim_socket,
)
from chickadee_serve.interaction import PendingQuestions, RemoteInteraction
from chickadee_serve.progress import ProgressHub, TurnPumps
from chickadee_serve.refusals import Refusal
from chickadee_serve.replay import ReplayFacade

__all__ = [
    "SERVE_LOCK",
    "STOP_SIGNALS",
    "Background",
    "Running",
    "engine_version",
    "listen",
    "live",
    "refused",
    "replay",
    "running",
    "scratch_dir",
    "serve_in_thread",
    "socket_in",
]


def engine_version() -> str:
    """The installed core's version, for `GET /`. A source tree with no metadata says so."""
    try:
        return metadata.version("chickadee-core")
    except metadata.PackageNotFoundError:
        return "unknown"


def live(engine: PacketEngine, tokens: TokenStore) -> Served:
    """`engine` served: every question it asks waits for a client's answer, up to its
    `[interaction] question_timeout_s`, then resolves to its default. Call before
    the engine opens a session: a session's port wraps the engine's when it is opened."""
    hub = ProgressHub()
    pending = PendingQuestions()
    timeout_s = engine.config.interaction.question_timeout_s
    engine.interaction = RemoteInteraction(pending, timeout_s=timeout_s)
    # A person is attached: `coder` hands the work over in the background (as
    # amended 2026-10-27), and they talk to Zeus while the team works.
    engine.attended = True
    return Served(
        engine,
        tokens,
        hub=hub,
        pending=pending,
        engine_version=engine_version(),
        question_timeout_s=timeout_s,
        on_turn=TurnPumps(hub).start,
    )


def replay(
    state_dir: Path, session_id: str, scratch: Path, tokens: TokenStore, *, speed: float
) -> tuple[Served, ReplayFacade]:
    """The recording under `state_dir` served read-only, its answers kept in `scratch`. Not
    yet started: the caller starts it once it listens, so a client sees it from the start."""
    hub = ProgressHub()
    pending = PendingQuestions()
    facade = ReplayFacade(state_dir, session_id, scratch, speed=speed, hub=hub, pending=pending)
    served = Served(
        facade,
        tokens,
        hub=hub,
        pending=pending,
        engine_version=engine_version(),
        replay=True,
        question_timeout_s=facade.timeout_s,
    )
    return served, facade


@dataclass
class Running:
    """What a server leaves on disk while it runs, removed by `running` when it stops."""

    bind: Bind
    token_file: Path
    tokens: dict[TokenClass, str] = field(default_factory=dict[TokenClass, str])


def refused(exc: BaseException) -> tuple[Refusal, str] | None:
    """Why starting an engine failed, when `exc` is a refusal and not a bug: the reason and
    what to say. The one translation of the engine's start errors (`refusals.py`)."""
    match exc:
        case AlreadyServed():
            return "held", str(exc)
        case BindRefused():
            return "unbindable", str(exc)
        case NotARepository():
            return "not_a_repository", f"not a repository: {exc}"
        case ConfigError() | LayerReadError() | MergeError():
            return "config", f"the config does not load: {exc}"
        case _:
            return None


SERVE_LOCK: Final = "serve.lock"
"""Held for a server's whole life, beside its token file: one server per state directory,
whichever bind it uses. A loopback TCP server claims no socket, so without it a second
server would overwrite a running one's tokens and delete them when it stopped."""


@contextlib.contextmanager
def running(store: TokenStore, state_dir: Path, bind: Callable[[], Bind]) -> Iterator[Running]:
    """Take the server lock in `state_dir`, then the bind (`bind()`, under the lock, so two
    servers never race for a socket), then issue one token per class and write them owner-only
    to `state_dir / TOKEN_FILE`. On exit, whatever ended the server, remove the token file and
    the socket and release the lock. A client reads its token from the file: the tokens are
    never printed. `AlreadyServed` when another server holds the lock."""
    make_dir(state_dir)
    try:
        lock = FileLock.acquire(state_dir / SERVE_LOCK)
    except LockHeld as exc:
        raise AlreadyServed(f"a server already serves this repository ({exc})") from exc
    token_file = state_dir / TOKEN_FILE
    where: Bind | None = None
    try:
        where = bind()
        tokens = {cls: store.issue(cls) for cls in TokenClass}
        store.write(token_file, tokens)
        yield Running(where, token_file, tokens)
    finally:
        TokenStore.remove(token_file)
        if isinstance(where, Uds):
            with contextlib.suppress(FileNotFoundError):
                where.path.unlink()
        lock.release()


def socket_in(directory: Path) -> Uds:
    """The default bind: `SOCKET_NAME` in `directory`, a dead server's left-over reclaimed."""
    return claim_socket(directory / SOCKET_NAME)


def scratch_dir() -> Path:
    """A replay's own directory: its socket, its token file and the answers given to it, so
    serving a recording never writes beside it."""
    return Path(tempfile.mkdtemp(prefix="ck-replay-"))


def listen(served: Served, bind: Bind) -> None:
    """Serve until SIGINT or SIGTERM, which uvicorn turns into a graceful shutdown: open
    streams are closed and the call returns.

    uvicorn re-raises the signal into the handler it found once it has shut down. Under the
    default handlers that would kill the process (SIGTERM) or raise `KeyboardInterrupt`
    (SIGINT) before the caller removes the socket and the token file, so for the call those
    handlers are `_stopped`, which has nothing left to do. Call from the main thread."""
    previous = {sig: signal.signal(sig, _stopped) for sig in STOP_SIGNALS}
    try:
        _Server(served, bind).run()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class _Server(uvicorn.Server):
    """The one way a served app is put on a bind, whichever runs it. Its exit, by a signal or
    by `Background.stop`, first tells the app it is stopping (`Served.stopping`), so attached
    streams end and the graceful shutdown that follows is not left waiting on clients."""

    def __init__(self, served: Served, bind: Bind) -> None:
        app = build_app(served)
        if isinstance(bind, Tcp):
            config = uvicorn.Config(
                app, host=bind.host, port=bind.port, log_level="warning", lifespan="on"
            )
        else:
            config = uvicorn.Config(app, uds=str(bind.path), log_level="warning", lifespan="on")
        super().__init__(config)
        self.served = served

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        self.served.stopping.set()
        super().handle_exit(sig, frame)


START_WAIT_S: Final = 10.0
"""How long `serve_in_thread` waits for uvicorn to listen before it gives up."""


@dataclass
class Background:
    """A server on a thread of this process."""

    server: uvicorn.Server
    thread: threading.Thread
    bound: Bind | None = None
    """Where it listens, once it does: the bind it was given, with the port the kernel chose
    when that was `0` (`serve_in_thread`)."""

    def stop(self, wait_s: float = START_WAIT_S) -> None:
        """Ask uvicorn to shut down, as SIGTERM would, and wait for the thread: open streams
        are closed and the bind is released when this returns. A shutdown still waiting after
        `wait_s` is forced, as a second signal would; one that outlives that too raises, so
        its caller never removes the socket or closes the engine under a live server."""
        self.server.handle_exit(signal.SIGTERM, None)
        self.thread.join(timeout=wait_s)
        if self.thread.is_alive():
            self.server.force_exit = True
            self.thread.join(timeout=wait_s)
        if self.thread.is_alive():
            raise RuntimeError(f"the server thread did not stop within {2 * wait_s}s")


def serve_in_thread(served: Served, bind: Bind, *, wait_s: float = START_WAIT_S) -> Background:
    """`served` on `bind`, from a thread, returned once it listens (`Background.bound` says
    where, a TCP port `0` resolved). uvicorn installs no signal
    handlers off the main thread, so the caller's are untouched and `stop` is the only way it
    ends. `RuntimeError` when it does not start (its bind was taken, say)."""
    server = _Server(served, bind)
    thread = threading.Thread(target=server.run, name="chickadee-serve", daemon=True)
    thread.start()
    deadline = time.monotonic() + wait_s
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            Background(server, thread).stop()
            raise RuntimeError(f"the server did not start on {bind}")
        time.sleep(0.01)
    return Background(server, thread, _bound(server, bind))


def _bound(server: uvicorn.Server, bind: Bind) -> Bind:
    """`bind` as listened on: a TCP port `0` is the one the kernel chose."""
    if not isinstance(bind, Tcp) or bind.port != 0:
        return bind
    port: int = server.servers[0].sockets[0].getsockname()[1]
    return Tcp(bind.host, port)


STOP_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)
"""What stops a server, gracefully."""


def _stopped(signum: int, frame: FrameType | None) -> None:
    """The handler uvicorn re-raises into after its shutdown: the server has stopped, and the
    caller's cleanup runs next."""
