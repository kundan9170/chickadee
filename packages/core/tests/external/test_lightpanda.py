"""The `lightpanda` engine (`external/fetch/lightpanda.py`): one fixed command
per fetch, run through the supervisor and a wrap with the network on and the repository masked;
its JSON answer read as an engine's; and which engine `web.fetch.engine` gives (`page_engine`).
The browser itself is never run here: what it is asked, and how its answer is read, is."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from chickadee.context.results import FetchFailed
from chickadee.external.fetch import Fetched, Moved
from chickadee.external.fetch.http import HttpEngine
from chickadee.external.fetch.lightpanda import LightpandaEngine
from chickadee.external.urls import Admitted
from chickadee.external.web import WebLimits, page_engine
from chickadee.isolation.none import Passthrough
from chickadee.isolation.types import Isolation, SandboxLevel
from chickadee.process.supervisor import Completed

URL = "https://docs.example.org/guide"
TARGET = Admitted(URL, "docs.example.org", 443, ("93.184.216.34",), False)
BINARY = Path("/opt/lightpanda/lightpanda")


class Sealing:
    """A sandbox backend that seals, and keeps what it was asked to wrap."""

    def __init__(self) -> None:
        self.wrapped: list[Isolation] = []

    def available(self) -> bool:
        return True

    def level(self) -> SandboxLevel:
        return SandboxLevel.NATIVE

    def wrap(self, isolation: Isolation) -> tuple[str, ...]:
        self.wrapped.append(isolation)
        return ("sandbox", *isolation.argv)


class CannedRun:
    """Answers every run with `done`, keeping what it ran and with which environment."""

    def __init__(self, done: Completed) -> None:
        self.done = done
        self.runs: list[tuple[tuple[str, ...], dict[str, str], int]] = []

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: int,
        stdin: bytes | None = None,
        cancel: object = None,
    ) -> Completed:
        del cwd, stdin, cancel
        self.runs.append((argv, dict(env), timeout_s))
        return self.done


def answered(**fields: Any) -> Completed:
    body = {
        "url": URL,
        "http_status": 200,
        "headers": [{"name": "content-type", "value": "text/html"}],
        "content": "\n# Guide\n\nRead this.\n",
        "error": None,
        **fields,
    }
    return Completed(json.dumps(body).encode(), b"", 0, False)


def engine(
    done: Completed, env: Mapping[str, str] | None = None
) -> tuple[LightpandaEngine, Sealing, CannedRun]:
    wrap, supervisor = Sealing(), CannedRun(done)
    made = LightpandaEngine(
        BINARY,
        wrap=wrap,
        connect_ms=8000,
        timeout_ms=15000,
        page_ms=20000,
        max_bytes=160000,
        masked=(Path("/work/repo"),),
        supervisor=supervisor,  # type: ignore[arg-type]
        env=env or {},
    )
    return made, wrap, supervisor


def test_only_fetch_is_run_with_every_fixed_flag() -> None:
    made, _, _ = engine(answered())
    assert made.argv(TARGET) == (
        str(BINARY),
        "fetch",
        URL,
        "--json",
        "--dump",
        "markdown",
        "--strip-mode",
        "clutter",
        "--dump-max-bytes",
        "160000",
        "--block-private-networks",
        "--http-connect-timeout",
        "8000",
        "--http-timeout",
        "15000",
        "--terminate-ms",
        "20000",
        "--obey-robots",
        "--log-level",
        "fatal",
        "--user-agent-suffix",
        " (+https://github.com/kundan9170/chickadee)",
    )


def test_behind_a_proxy_the_proxy_is_the_one_private_address_it_may_reach() -> None:
    made, _, _ = engine(
        answered(), {"HTTPS_PROXY": "http://127.0.0.1:3128", "SSL_CERT_FILE": "/ca.pem"}
    )
    argv = made.argv(TARGET)
    tail = argv[argv.index("--http-proxy") :]
    assert tail == (
        "--http-proxy",
        "http://127.0.0.1:3128",
        "--block-cidrs",
        "-127.0.0.1/32",
        "--ca-cert",
        "/ca.pem",
    )
    no_proxy, _, _ = engine(
        answered(), {"HTTPS_PROXY": "http://p:1", "NO_PROXY": "example.org"}
    )
    assert "--http-proxy" not in no_proxy.argv(TARGET)


def test_it_runs_wrapped_with_the_network_on_the_repository_masked_and_no_telemetry() -> None:
    made, wrap, supervisor = engine(answered())
    made.get(TARGET)
    (isolation,) = wrap.wrapped
    assert (isolation.network_off, isolation.deny_read) == (False, (Path("/work/repo"),))
    ((argv, env, timeout_s),) = supervisor.runs
    assert argv[0] == "sandbox" and argv[1:4] == (str(BINARY), "fetch", URL)
    assert env["LIGHTPANDA_DISABLE_TELEMETRY"] == "true"
    assert env["HOME"] == str(isolation.cwd) and env["HOME"] != str(Path.home())
    assert timeout_s == 8 + 20 + 5
    assert not isolation.cwd.exists(), "its home is thrown away after the fetch"


@pytest.mark.parametrize(
    ("done", "read"),
    [
        (answered(), Fetched(URL, "# Guide\n\nRead this.", False)),
        (
            answered(content="# Guide\n\nlong text\n\n[truncated]\n"),
            Fetched(URL, "# Guide\n\nlong text", True),
        ),
        (
            answered(url="https://docs.example.org/guide/"),
            Fetched("https://docs.example.org/guide/", "# Guide\n\nRead this.", False),
        ),
        (
            answered(url="https://elsewhere.example/landing"),
            Moved(URL, "https://elsewhere.example/landing", 0),
        ),
        (answered(http_status=404), FetchFailed(URL, "http_status", 404)),
        (
            answered(headers=[{"name": "Content-Type", "value": "application/pdf"}]),
            FetchFailed(URL, "not_text", 200),
        ),
        (answered(error="RobotsBlocked", http_status=0), FetchFailed(URL, "robots")),
        (answered(error="OperationTimedout", http_status=0), FetchFailed(URL, "timeout")),
        (answered(error="CouldntConnect", http_status=0), FetchFailed(URL, "connection")),
        (Completed(b"", b"", -9, True), FetchFailed(URL, "timeout")),
        (Completed(b"not json", b"crash", 134, False), FetchFailed(URL, "connection")),
    ],
)
def test_its_answer_is_read_as_an_engine_s(done: Completed, read: object) -> None:
    made, _, _ = engine(done)
    assert made.get(TARGET) == read


class Unsealed(Sealing):
    def level(self) -> SandboxLevel:
        return SandboxLevel.NONE


@pytest.mark.parametrize(
    ("choice", "backend", "binary", "require", "gets"),
    [
        ("http", Sealing(), BINARY, False, HttpEngine),
        ("auto", Sealing(), BINARY, False, LightpandaEngine),
        ("auto", Sealing(), None, False, HttpEngine),
        ("auto", Unsealed(), BINARY, False, HttpEngine),
        ("lightpanda", Sealing(), BINARY, True, LightpandaEngine),
        ("lightpanda", Unsealed(), BINARY, False, LightpandaEngine),
        ("lightpanda", Passthrough(), BINARY, False, LightpandaEngine),
    ],
)
def test_the_engine_web_fetch_engine_gives(
    choice: str, backend: Sealing | Passthrough, binary: Path | None, require: bool, gets: type
) -> None:
    made = page_engine(
        choice,  # type: ignore[arg-type]
        WebLimits(),
        backend=backend,
        require_sandbox=require,
        binary=binary,
    )
    assert not isinstance(made, str)
    assert isinstance(made(()), gets)


@pytest.mark.parametrize(
    ("backend", "binary", "require", "says"),
    [
        (Sealing(), None, False, "no lightpanda is installed"),
        (Unsealed(), BINARY, True, "sandbox.require_sandbox is set with no sandbox to seal it"),
    ],
)
def test_lightpanda_named_and_not_to_be_had_says_why(
    backend: Sealing, binary: Path | None, require: bool, says: str
) -> None:
    made = page_engine(
        "lightpanda", WebLimits(), backend=backend, require_sandbox=require, binary=binary
    )
    assert isinstance(made, str) and says in made
