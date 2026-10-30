"""A server is asked and, approved, runs outside the sandbox; a port the sandbox refused
says so, with the way out."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from repos import init_repo
from sandboxes import Enforcing, Scripted
from tool_args import checked

from chickadee.context.homes import Scratch
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import ToolResult
from chickadee.context.terms import Vision
from chickadee.executor.executor import Executor
from chickadee.schema.authorization import Tier
from chickadee.shell import RefusedAboveT0, Shell, ShellClaim

REFUSED_BIND = (
    b'  File "/usr/lib/python3.12/socketserver.py", line 472, in server_bind\n'
    b"    self.socket.bind(self.server_address)\n"
    b"PermissionError: [Errno 1] Operation not permitted\n"
)


class _Yes:
    def __init__(self) -> None:
        self.claims: list[ShellClaim] = []

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        self.claims.append(claim)
        return True


def _shell(
    tmp_path: Path, out: bytes = b"", code: int = 0, authority: _Yes | None = None
) -> tuple[Shell, Enforcing]:
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    own = Scratch()
    reader = Reader(own, HandleResolver(root, own.blob), vision=Vision("z"))
    backend = Enforcing()
    shell = Shell(
        Executor(backend, supervisor=Scripted(out, exit_code=code)),
        root,
        lambda data: reader.store("output", data),
        authority=authority,
    )
    return shell, backend


def _run(shell: Shell, command: str, **args: object) -> object:
    return shell.run(checked("shell", {"command": command, **args}), operation_id="op-1")


def test_a_server_is_asked_and_refused_with_nobody_to_ask(tmp_path: Path) -> None:
    shell, backend = _shell(tmp_path)
    refused = _run(shell, "python3 -m http.server 8000 --bind 127.0.0.1")
    assert isinstance(refused, RefusedAboveT0)
    assert refused.tier is Tier.T1 and refused.classification == "dev_server"
    assert backend.seen == [], "nothing ran"


def test_an_approved_server_runs_outside_the_sandbox(tmp_path: Path) -> None:
    asked = _Yes()
    shell, backend = _shell(tmp_path, b"Serving HTTP on 127.0.0.1 port 8000\n", authority=asked)
    done = _run(shell, "python3 -m http.server 8000 --bind 127.0.0.1")
    assert isinstance(done, ToolResult)
    [claim] = asked.claims
    assert (claim.tier, claim.classification) == (Tier.T1, "dev_server")
    assert backend.seen == [], "approved, it is not wrapped (D207 item 1)"
    assert "sandbox none · network on" in done.header


def test_a_refused_port_names_the_sandbox_and_the_way_out(tmp_path: Path) -> None:
    """The 2026-10-27 report: a sealed server's bind is EPERM on macOS. A script the table does
    not know as a server runs sealed, T0, and fails the same way."""
    shell, _ = _shell(tmp_path, REFUSED_BIND, code=1)
    done = _run(shell, "python3 serve.py")
    assert isinstance(done, ToolResult)
    text = done.render()
    assert "NETWORK_DENIED" in text and "refused the server its port" in text
    assert "sandbox: false" in text
    assert "background" not in text, "this shell keeps no jobs, so it is never told of them"
    assert "network: true" not in text, "one shape, not the network's as well"


def test_a_refused_port_outside_the_sandbox_is_no_hint(tmp_path: Path) -> None:
    """Approved, nothing sandboxed it: a refused bind there is the port's own (in use,
    privileged), and no shape is the sandbox's."""
    shell, _ = _shell(tmp_path, REFUSED_BIND, code=1, authority=_Yes())
    done = _run(shell, "python3 -m http.server 80")
    assert isinstance(done, ToolResult)
    assert "NETWORK_DENIED" not in done.render()


@pytest.mark.parametrize(
    ("output", "said"),
    [
        (REFUSED_BIND, "PermissionError: [Errno 1] Operation not permitted"),
        (b"Error: listen EPERM: operation not permitted 127.0.0.1:3000\n", "listen EPERM"),
        (b"listen tcp 127.0.0.1:8080: bind: operation not permitted\n", "bind: operation"),
    ],
    ids=["python", "node", "go"],
)
def test_each_runtimes_refused_port_is_read(tmp_path: Path, output: bytes, said: str) -> None:
    shell, _ = _shell(tmp_path, output, code=1)
    done = _run(shell, "./serve")
    assert isinstance(done, ToolResult)
    text = done.render()
    assert "refused the server its port" in text and said in text


@pytest.mark.parametrize(
    "output",
    [b"cat: x: Permission denied\n", b"curl: (6) Could not resolve host: example.com\n"],
    ids=["a file", "dns"],
)
def test_what_is_not_a_refused_port_is_not_read_as_one(tmp_path: Path, output: bytes) -> None:
    shell, _ = _shell(tmp_path, output, code=1)
    done = _run(shell, "./serve")
    assert isinstance(done, ToolResult)
    assert "refused the server its port" not in done.render()
