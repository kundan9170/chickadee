"""Where the server listens.

**A Unix domain socket by default**, at `SOCKET_PATH` in the repository's repo-local state: its
access control is the filesystem's, and no other host can reach it. **Loopback TCP is opt-in**,
for a client that cannot open a socket file (a browser); it still needs a token, as every
route but `GET /` does. **Any other address is refused** in v1, before a socket is opened:
serving a repository to a network is a different security problem than this one solves.

A socket path left by a server that died is reclaimed: if nothing answers on it, it is
removed and bound again. If something answers, another server owns it and this one refuses,
naming the path, rather than stealing the address from a live process.
"""

from __future__ import annotations

import contextlib
import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "SOCKET_NAME",
    "UDS_PATH_MAX",
    "AlreadyServed",
    "Bind",
    "BindRefused",
    "SocketPathTooLong",
    "Tcp",
    "Uds",
    "claim_socket",
    "loopback_tcp",
]

SOCKET_NAME: Final = "serve.sock"
"""`SOCKET_PATH`'s file name, in the repository's repo-local state (`Roots.repo_local`,
`<git common dir>/chickadee/`). The state rules assign it there: one server
serves a repository and its state is repo-local, so every worktree of the clone finds
the same socket; and its token file beside it must never be in a working tree, where a commit
would carry a credential. That answers an earlier security review for this caller."""

UDS_PATH_MAX: Final = 103
"""The longest socket path bound: `sun_path` holds 104 bytes on macOS and 108 on Linux, each
with its terminating NUL. A longer path is refused before anything is bound, naming `--tcp`."""


class BindRefused(Exception):
    """The server will not listen there, and says why."""


class SocketPathTooLong(BindRefused):
    """The socket path is longer than a socket address holds (`UDS_PATH_MAX`). A host that was
    given no address may serve on loopback TCP instead (`chickadee_wire.host`)."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"{path} is longer than a Unix socket path may be ({UDS_PATH_MAX} bytes); "
            "serve on loopback TCP with --tcp instead"
        )
        self.path = path


class AlreadyServed(BindRefused):
    """Another server serves the repository already: the one refusal a client answers
    by attaching to that server rather than by choosing another address."""


@dataclass(frozen=True)
class Uds:
    path: Path


@dataclass(frozen=True)
class Tcp:
    host: str
    port: int

    @property
    def url(self) -> str:
        """`http://host:port`, an IPv6 host in brackets (`http://[::1]:8765`), as a URL
        must spell it: the one place a TCP bind is written as one."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


Bind = Uds | Tcp


def loopback_tcp(host: str, port: int) -> Tcp:
    """A TCP bind, refused unless `host` is a loopback address. `localhost` is
    resolved, and every address it resolves to must be loopback."""
    if not 0 <= port <= 65535:
        raise BindRefused(f"port {port} is not a port")
    for address in _addresses(host):
        if not ipaddress.ip_address(address).is_loopback:
            raise BindRefused(
                f"{host} is not a loopback address ({address}); the server binds a Unix socket "
                f"or loopback TCP only, never a network address (30-serve SV-6)"
            )
    return Tcp(host, port)


def _addresses(host: str) -> tuple[str, ...]:
    try:
        return (str(ipaddress.ip_address(host)),)
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise BindRefused(f"{host} does not resolve: {exc}") from exc
    return tuple(sorted({str(info[4][0]) for info in infos}))


def claim_socket(path: Path) -> Uds:
    """The socket path, free to bind: a stale file from a dead server removed, a live one
    refused. The directory is made if missing. A path too long for a socket address is refused
    first: the kernel would refuse it at bind, after the stale file was already gone."""
    if len(str(path).encode()) > UDS_PATH_MAX:
        raise SocketPathTooLong(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if _answers(path):
            raise AlreadyServed(f"a server is already listening on {path}")
        path.unlink()
    return Uds(path)


def _answers(path: Path) -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(1.0)
        probe.connect(str(path))
    except OSError:
        return False
    else:
        return True
    finally:
        with contextlib.suppress(OSError):
            probe.close()
