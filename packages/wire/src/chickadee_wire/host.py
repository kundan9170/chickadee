"""A frontend that hosts its own engine: the server `chickadee serve`
runs, on a thread of this process, and nothing a separate server would not do.

**The one place a frontend touches `chickadee.*`** (`test_confined.py`). Everything else in
this package speaks the wire, so a frontend built on `host_in_process` and one attached to a
running server differ only in where the server is.

**Composed from the server's own parts, never a second composition** (`chickadee_serve.server`):
the repository's config as a server loads it (`load_host_config`), its lock and bind and tokens
(`running`: one server per repository), the engine served as a live server serves it
(`live`), on a thread (`serve_in_thread`).

**A live socket is refused, a dead one reclaimed.** `running` holds the repository's
`serve.lock` for the server's life, so a second host, or a `chickadee serve` already running,
is refused (`HostRefused`, reason `held`) before an engine is built. A socket a dead server
left is reclaimed by `socket_in`'s probe: nothing answers on it, so it is removed and bound
again.

**A socket path too long falls back to loopback TCP** on a port the kernel chooses, when the
caller named no address: a deep checkout should not stop a frontend that can say where
it listens instead. `Hosted.fallback` says that it did, for the frontend to tell the person.

`Hosted.close()` stops the server, closes the engine and removes the socket and the token
file, in that order, however far `host_in_process` got.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Self

from chickadee.engine import PacketEngine
from chickadee.state.load import load_host_config
from chickadee.state.roots import NotARepository, roots
from chickadee_serve.auth import TokenStore
from chickadee_serve.bind import Bind, SocketPathTooLong, Tcp
from chickadee_serve.frames import TokenClass
from chickadee_serve.server import live, refused, running, serve_in_thread, socket_in
from chickadee_wire.client import Client
from chickadee_wire.errors import HostRefused
from chickadee_wire.transport import Address, token_in

__all__ = ["LOOPBACK", "Hosted", "host_in_process", "state_dir_of"]

LOOPBACK: Final = "127.0.0.1"
"""Where a host falls back to when its socket path is too long."""


@dataclass
class Hosted:
    """A served engine in this process: where it listens, and its tokens."""

    address: Address
    repo: Path
    """The repository's working tree, as the engine serves it."""
    state_dir: Path
    """The repository's repo-local state: the socket, the token file and the lock are here."""
    tokens: dict[TokenClass, str]
    _stack: contextlib.ExitStack = field(repr=False)
    fallback: str | None = None
    """Why it listens on loopback TCP though no address was asked for: the socket path the
    repository's state would need is too long. None when it listens where it was asked."""

    def token(self, token_class: TokenClass = TokenClass.INTERACTIVE) -> str:
        return self.tokens[token_class]

    def client(self, token_class: TokenClass = TokenClass.INTERACTIVE) -> Client:
        """A connected client of this host, answering as `token_class`. Its token is read
        from the server's file, as `Client.attach` reads it, so the client outlives a restart
        of the host on this repository."""
        state = self.state_dir
        return Client.connect(self.address, lambda: token_in(state, token_class))

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def host_in_process(repo: Path | None = None, *, tcp: Tcp | None = None) -> Hosted:
    """Serve the repository containing `repo` (the working directory by default) from this
    process: on its repo-local socket, or on `tcp`, a loopback address; with no `tcp` and a
    socket path too long, on a free loopback port (`Hosted.fallback`). Returns once it
    listens. `HostRefused` when it will not start: a server already serves the repository,
    the address cannot be bound, the path is not in one, or the
    engine config does not load (`server.refused`, the one translation), so a frontend never
    imports the engine's own errors."""
    try:
        return _host(repo if repo is not None else Path.cwd(), tcp)
    except Exception as exc:
        why = refused(exc)
        if why is None:
            raise
        raise HostRefused(*why) from exc


def state_dir_of(path: Path) -> Path:
    """The repo-local state of the repository containing `path`, where its server's socket
    and token file are: what a frontend attaching to a running server looks in. Asked of
    the engine's own layout (`state.roots`), never restated. `HostRefused` outside a
    repository."""
    try:
        return roots(path).repo_local
    except NotARepository as exc:
        why = refused(exc)
        assert why is not None
        raise HostRefused(*why) from exc


def _host(repo: Path, tcp: Tcp | None) -> Hosted:
    host = load_host_config(repo)
    state = host.roots.repo_local
    store = TokenStore()
    fallback: list[str] = []

    def bind() -> Bind:
        if tcp is not None:
            return tcp
        try:
            return socket_in(state)
        except SocketPathTooLong as exc:
            fallback.append(
                f"its socket path is longer than a Unix socket path may be ({exc.path})"
            )
            return Tcp(LOOPBACK, 0)

    with contextlib.ExitStack() as stack:
        up = stack.enter_context(running(store, state, bind))
        engine = PacketEngine(host.repo, host.engine, trust=host.trust)
        stack.callback(engine.close)
        # A repository with no index has it built beside the person, never before them
        # its liveness says so until it is done.
        engine.index_in_background()
        served = serve_in_thread(live(engine, store), up.bind)
        stack.callback(served.stop)
        address = served.bound or up.bind
        # Until here a failure unwinds what was started; from here `Hosted.close` does.
        hosted = Hosted(
            address,
            host.repo,
            state,
            dict(up.tokens),
            stack.pop_all(),
            fallback=fallback[0] if fallback else None,
        )
    return hosted
