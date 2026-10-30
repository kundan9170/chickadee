"""`chickadee-serve` as a person runs it: a process of its own, over a real repository or the
demo recording. It answers `GET /` on its socket, a SIGTERM stops it
with exit 0 and nothing left behind, and every refusal exits with its §6a code
saying why."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from demo_session import FIXTURE, SESSION_ID
from engines import config_toml
from repos import init_repo

from chickadee_serve.bind import SOCKET_NAME
from chickadee_serve.frames import TOKEN_FILE, TokenClass, read_token

WAIT_S = 30.0
SERVING = re.compile(r"serving on (?P<where>[^;]+); tokens in (?P<tokens>\S+)")


def _env(xdg: Path) -> dict[str, str]:
    """This machine's environment, but the engine config in `xdg` and no `CHICKADEE_*`."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("CHICKADEE_")}
    env["XDG_CONFIG_HOME"] = str(xdg)
    return env


@pytest.fixture
def home() -> Path:
    """A short root: the socket lives in the repository's git dir, and AF_UNIX paths are
    limited to about 100 bytes, which pytest's own temporary paths can exceed."""
    root = Path(tempfile.mkdtemp(prefix="ck-"))
    (root / "xdg" / "chickadee").mkdir(parents=True)
    (root / "xdg" / "chickadee" / "config.toml").write_text(config_toml())
    return root


@contextmanager
def serving(
    args: list[str], env: dict[str, str]
) -> Iterator[tuple[subprocess.Popen[str], Path, Path]]:
    """The command running, once it says where it serves: the process, the socket and the
    token file. Stopped with SIGTERM if the test has not stopped it."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "chickadee_serve", *args],
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stderr is not None
        line = proc.stderr.readline()
        found = SERVING.search(line)
        assert found is not None, f"no address announced: {line!r}"
        sock, tokens = Path(found["where"]), Path(found["tokens"])
        _answers(sock)
        yield proc, sock, tokens
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=WAIT_S)


def _client(sock: Path) -> httpx.Client:
    return httpx.Client(
        transport=httpx.HTTPTransport(uds=str(sock)), base_url="http://serve", timeout=10
    )


def _answers(sock: Path) -> None:
    """Wait until the socket answers `GET /`: the address is announced before uvicorn binds."""
    deadline = time.monotonic() + WAIT_S
    while True:
        try:
            with _client(sock) as client:
                if client.get("/").status_code == 200:
                    return
        except httpx.TransportError:
            pass
        assert time.monotonic() < deadline, f"{sock} never answered"
        time.sleep(0.05)


def _stop(proc: subprocess.Popen[str]) -> int:
    proc.send_signal(signal.SIGTERM)
    return proc.wait(timeout=WAIT_S)


def test_live_serves_the_repository_from_its_repo_local_state(home: Path) -> None:
    repo = init_repo(home / "repo")
    with serving(["--repo", str(repo)], _env(home / "xdg")) as (proc, sock, tokens):
        assert sock == repo / ".git" / "chickadee" / SOCKET_NAME
        assert tokens == sock.parent / TOKEN_FILE
        with _client(sock) as client:
            hello = client.get("/").json()
            assert hello["replay"] is False
            token = read_token(tokens, TokenClass.INTERACTIVE)
            listed = client.get("/session", headers={"authorization": f"Bearer {token}"})
            assert listed.status_code == 200 and listed.json() == []
        assert _stop(proc) == 0
    assert not sock.exists() and not tokens.exists(), "nothing left behind"


def test_a_second_server_on_the_same_repository_is_refused(home: Path) -> None:
    repo = init_repo(home / "repo")
    env = _env(home / "xdg")
    with serving(["--repo", str(repo)], env) as (_proc, sock, tokens):
        second = subprocess.run(
            [sys.executable, "-m", "chickadee_serve", "--repo", str(repo)],
            env=env,
            capture_output=True,
            text=True,
            timeout=WAIT_S,
        )
        assert second.returncode == 5, "held: §6a's code for a served repository"
        assert "already serves this repository" in second.stderr
        assert sock.exists(), "the first server's socket is not stolen"
        loopback = subprocess.run(
            [sys.executable, "-m", "chickadee_serve", "--repo", str(repo), "--tcp", "0"],
            env=env,
            capture_output=True,
            text=True,
            timeout=WAIT_S,
        )
        assert loopback.returncode == 5, "a TCP server would overwrite the first's tokens"
        assert tokens.exists(), "and delete them when it stopped"


def test_replay_serves_the_recording_and_cleans_up(home: Path) -> None:
    args = ["--replay", str(FIXTURE), "--session", SESSION_ID, "--speed", "0"]
    with serving(args, _env(home / "xdg")) as (proc, sock, _tokens):
        with _client(sock) as client:
            assert client.get("/").json()["replay"] is True
        assert _stop(proc) == 0
    assert not sock.parent.exists(), "a replay's scratch goes with it"


@pytest.mark.parametrize(
    ("args", "says", "code"),
    [
        (["--tcp", "8765", "--host", "0.0.0.0"], "SV-6", 1),
        (["--repo", "/"], "not a repository", 4),
        (["--speed", "fast"], "invalid float value", 1),
    ],
)
def test_a_refusal_exits_with_its_code_and_says_why(
    home: Path, args: list[str], says: str, code: int
) -> None:
    if args[0] == "--tcp":
        args = ["--replay", str(FIXTURE), "--session", SESSION_ID, *args]
    done = subprocess.run(
        [sys.executable, "-m", "chickadee_serve", *args],
        env=_env(home / "xdg"),
        capture_output=True,
        text=True,
        timeout=WAIT_S,
    )
    assert done.returncode == code, done.stderr
    assert says in done.stderr
