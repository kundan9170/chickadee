"""`host_in_process`: the repository served from this process, on its
repo-local socket; a live server refused, a dead one's socket reclaimed; nothing left behind."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from hosting import SID, hosted, repository, scripted, short_root

from chickadee_serve.bind import SOCKET_NAME, SocketPathTooLong, Tcp, Uds
from chickadee_serve.frames import TOKEN_FILE, TokenClass
from chickadee_serve.server import refused
from chickadee_wire.errors import HostRefused
from chickadee_wire.host import LOOPBACK, host_in_process
from chickadee_wire.transport import discover, token_in


def test_a_hosted_engine_answers_on_the_repositorys_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with hosted(monkeypatch) as host:
        sock, tokens = host.state_dir / SOCKET_NAME, host.state_dir / TOKEN_FILE
        assert host.address == Uds(sock)
        assert discover(host.state_dir) == host.address
        assert token_in(host.state_dir) == host.token(TokenClass.INTERACTIVE)
        with host.client() as client:
            assert client.server is not None and client.server.replay is False
            assert [s.session_id for s in client.sessions()] == [SID]
    assert not sock.exists() and not tokens.exists(), "nothing left behind"


def test_a_second_host_on_the_same_repository_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with hosted(monkeypatch) as first:
        with pytest.raises(HostRefused, match="already serves this repository") as refused:
            host_in_process(first.repo)
        assert refused.value.reason == "held"
        assert isinstance(first.address, Uds) and first.address.path.exists()
        with first.client() as client:
            client.sessions()


def test_a_dead_servers_socket_is_reclaimed(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = repository(monkeypatch, short_root())
    scripted(monkeypatch, ())
    state = repo / ".git" / "chickadee"
    state.mkdir(parents=True, exist_ok=True)
    left = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    left.bind(str(state / SOCKET_NAME))
    left.close()  # the file stays; nothing answers on it
    assert (state / SOCKET_NAME).exists()
    with host_in_process(repo) as host, host.client() as client:
        assert client.sessions()


def test_a_path_outside_any_repository_is_refused_by_reason(tmp_path: Path) -> None:
    with pytest.raises(HostRefused) as refused:
        host_in_process(tmp_path)
    assert refused.value.reason == "not_a_repository"


def test_a_socket_path_too_long_falls_back_to_a_free_loopback_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deep checkout no longer stops a host that was given no address. It listens
    on loopback TCP on a port the kernel chose, says why, and leaves no socket behind."""
    root = short_root() / ("d" * 90)
    root.mkdir()
    repo = repository(monkeypatch, root)
    scripted(monkeypatch, ())
    with host_in_process(repo) as host, host.client() as client:
        assert isinstance(host.address, Tcp)
        assert host.address.host == LOOPBACK and host.address.port > 0
        assert host.fallback is not None and str(host.state_dir) in host.fallback
        assert client.sessions() is not None
        assert not (host.state_dir / SOCKET_NAME).exists()


def test_a_socket_path_too_long_is_still_unbindable_for_a_server() -> None:
    """`chickadee serve` falls back to nothing: it was asked for a socket, and its exit code
    hangs on the reason, never "a server already serves this repository"."""
    why = refused(SocketPathTooLong(Path("/deep") / ("d" * 120)))
    assert why is not None and why[0] == "unbindable"


def test_a_host_on_its_own_socket_says_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    with hosted(monkeypatch) as host:
        assert host.fallback is None
