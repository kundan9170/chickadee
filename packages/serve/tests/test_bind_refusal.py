"""A Unix socket by default, loopback TCP on request, nothing else, and why.
A dead server's socket is reclaimed; a live one's is never stolen."""

from __future__ import annotations

import socket
import tempfile
from pathlib import Path

import pytest

from chickadee_serve.bind import BindRefused, Tcp, Uds, claim_socket, loopback_tcp


@pytest.mark.parametrize("host", ["0.0.0.0", "8.8.8.8", "::", "192.168.1.10"])
def test_a_network_address_is_refused_and_says_why(host: str) -> None:
    with pytest.raises(BindRefused, match="SV-6"):
        loopback_tcp(host, 8765)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_is_allowed(host: str) -> None:
    assert loopback_tcp(host, 8765) == Tcp(host, 8765)


def test_a_port_out_of_range_is_refused() -> None:
    with pytest.raises(BindRefused):
        loopback_tcp("127.0.0.1", 70000)


def _short_dir() -> Path:
    """AF_UNIX paths are limited to about 100 bytes; pytest's tmp_path can exceed that."""
    return Path(tempfile.mkdtemp(prefix="ck-"))


def test_a_stale_socket_is_reclaimed() -> None:
    path = _short_dir() / "serve.sock"
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(path))
    dead.close()  # the file stays; nothing listens
    assert claim_socket(path) == Uds(path)
    assert not path.exists()


def test_a_live_socket_is_refused() -> None:
    path = _short_dir() / "serve.sock"
    live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live.bind(str(path))
    live.listen(1)
    try:
        with pytest.raises(BindRefused, match="already listening"):
            claim_socket(path)
        assert path.exists()
    finally:
        live.close()
