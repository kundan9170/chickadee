"""The `lightpanda` engine: a page whose scripts run, as text.

Lightpanda is a headless browser, run here as a separate, unmodified process, one per fetch:
its browser is AGPL-3.0, so the harness never bundles it, and uses it only when the person
installed it (`chickadee-core[web]`, or a `lightpanda` on the `PATH`).

**Its process is bounded the way every process the harness starts is:** through
`ProcessSupervisor` and an `Isolation` wrap, network on, its home a throwaway directory, the
repository and the person's secrets masked unreadable, and an environment with no credential
in it and `LIGHTPANDA_DISABLE_TELEMETRY=true`.

**Its flags are fixed.** Only `fetch` is ever run: never `evaluate`, `fill` or `press` (the
researcher reads a site, it never acts on one) and never its keyless `search`, whose service is
not ours to choose. Every call passes `--block-private-networks`, `--obey-robots`, `--json`,
`--dump markdown --strip-mode clutter`, `--dump-max-bytes` and the three timeouts. Behind a
proxy, the proxy is the one private address it may reach (`--block-cidrs -<address>`), so the
proxy, not the browser, makes every connection, as with the `http` engine.

The browser follows a redirect itself. One that ends on another host is answered as the `http`
engine answers it (`Moved`): the page is not shown, and the model fetches the target if it wants
it, so a page cannot move a fetch to a host the model never chose.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import json
import math
import os
import shutil
import socket
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import urlsplit

from chickadee.context.results import FetchFailed
from chickadee.external.fetch import CONTACT, Fetched, Moved
from chickadee.external.fetch.proxy import proxy_for
from chickadee.external.fetch.text import is_text
from chickadee.external.urls import Admitted, normalise
from chickadee.isolation.base import SandboxBackend
from chickadee.isolation.types import Isolation, harness_cache
from chickadee.process.supervisor import Completed, ProcessSupervisor, scrub_env

__all__ = ["LightpandaEngine", "find_lightpanda"]

_TRUNCATED: Final = "[truncated]"
"""What Lightpanda appends to a dump it cut at `--dump-max-bytes`."""
_TIMEOUTS: Final = ("OperationTimedout", "Timeout", "TimedOut")
_ROBOTS: Final = "RobotsBlocked"
_SPARE_S: Final = 5
"""Seconds the supervisor waits past the browser's own deadlines before it ends the process."""


def find_lightpanda() -> Path | None:
    """The `lightpanda` binary: on the `PATH`, or the one its Python package carries."""
    on_path = shutil.which("lightpanda")
    if on_path is not None:
        return Path(on_path)
    spec = importlib.util.find_spec("lightpanda")
    if spec is None or spec.origin is None:
        return None
    packaged = Path(spec.origin).parent / "lightpanda"
    return packaged if packaged.is_file() and os.access(packaged, os.X_OK) else None


class LightpandaEngine:
    """One `lightpanda fetch` per call. `wrap` is the sandbox backend to run it under, chosen
    by the caller (`engine.py`, which never picks this engine where a sandbox is required and
    none seals); `masked` is what the wrap makes unreadable: the repository and the person's
    secrets."""

    provider: str = "web.fetch.lightpanda"

    def __init__(
        self,
        binary: Path,
        *,
        wrap: SandboxBackend,
        connect_ms: int,
        timeout_ms: int,
        page_ms: int,
        max_bytes: int,
        masked: tuple[Path, ...] = (),
        supervisor: ProcessSupervisor | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._binary = binary
        self._wrap = wrap
        self._connect_ms = connect_ms
        self._timeout_ms = timeout_ms
        self._page_ms = page_ms
        self._max_bytes = max_bytes
        self._masked = masked
        self._supervisor = supervisor or ProcessSupervisor()
        self._env = os.environ if env is None else env

    def argv(self, target: Admitted) -> tuple[str, ...]:
        """The one command a fetch runs, its flags fixed."""
        argv = [
            str(self._binary),
            "fetch",
            target.url,
            "--json",
            "--dump",
            "markdown",
            "--strip-mode",
            "clutter",
            "--dump-max-bytes",
            str(self._max_bytes),
            "--block-private-networks",
            "--http-connect-timeout",
            str(self._connect_ms),
            "--http-timeout",
            str(self._timeout_ms),
            "--terminate-ms",
            str(self._page_ms),
            "--obey-robots",
            "--log-level",
            "fatal",
            "--user-agent-suffix",
            f" ({CONTACT})",
        ]
        proxy = proxy_for(urlsplit(target.url).scheme, target.host, self._env)
        if proxy is not None:
            argv += ["--http-proxy", proxy]
            for address in _addresses(urlsplit(proxy).hostname):
                argv += ["--block-cidrs", f"-{address}"]
        certificates = self._env.get("SSL_CERT_FILE")
        if certificates:
            argv += ["--ca-cert", certificates]
        return tuple(argv)

    def get(self, target: Admitted) -> Fetched | Moved | FetchFailed:
        homes = harness_cache() / "web"
        homes.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="lightpanda-", dir=homes) as made:
            home = Path(made)
            wrapped = self._wrap.wrap(
                Isolation(
                    argv=self.argv(target),
                    cwd=home,
                    network_off=False,
                    repo_writable=True,
                    deny_read=self._masked,
                )
            )
            done = self._supervisor.run(
                wrapped,
                cwd=home,
                env=self._environment(home),
                timeout_s=math.ceil(
                    (self._connect_ms + max(self._timeout_ms, self._page_ms)) / 1000
                )
                + _SPARE_S,
            )
        return _read(target, done)

    def _environment(self, home: Path) -> dict[str, str]:
        env = scrub_env(network=True)
        env.update(
            HOME=str(home),
            XDG_CONFIG_HOME=str(home),
            XDG_CACHE_HOME=str(home),
            XDG_DATA_HOME=str(home),
            LIGHTPANDA_DISABLE_TELEMETRY="true",
        )
        return env


def _addresses(host: str | None) -> tuple[str, ...]:
    """`host`'s addresses as single-address CIDRs, for the one private exemption a proxy is."""
    if not host:
        return ()
    try:
        found = {str(info[4][0]) for info in socket.getaddrinfo(host, None)}
    except OSError:
        return ()
    out: list[str] = []
    for raw in sorted(found):
        address = ipaddress.ip_address(raw.split("%", 1)[0])
        out.append(f"{address}/{address.max_prefixlen}")
    return tuple(out)


def _read(target: Admitted, done: Completed) -> Fetched | Moved | FetchFailed:
    """The browser's JSON answer as an engine's: a page, a redirect to another host, or why
    not."""
    if done.timed_out:
        return FetchFailed(target.url, "timeout")
    try:
        answer = cast("dict[str, Any]", json.loads(done.stdout.decode("utf-8", "replace")))
        error = answer.get("error")
        status = int(answer.get("http_status") or 0)
        landed = str(answer.get("url") or target.url)
        content = str(answer.get("content") or "")
        headers = cast("list[dict[str, str]]", answer.get("headers") or [])
    except (ValueError, TypeError, AttributeError):
        return FetchFailed(target.url, "connection")
    if error is not None:
        name = str(error)
        if name == _ROBOTS:
            return FetchFailed(target.url, "robots")
        if any(t in name for t in _TIMEOUTS):
            return FetchFailed(target.url, "timeout")
        return FetchFailed(target.url, "connection")
    if status >= 400:
        return FetchFailed(target.url, "http_status", status)
    kind = next(
        (h["value"] for h in headers if h.get("name", "").lower() == "content-type"), ""
    )
    if not is_text(kind):
        return FetchFailed(target.url, "not_text", status)
    final = normalise(landed) or target.url
    if urlsplit(final).hostname != target.host:
        return Moved(target.url, final, 0)
    text = content.strip("\n")
    cut = text.endswith(_TRUNCATED)
    if cut:
        text = text[: -len(_TRUNCATED)].rstrip()
    return Fetched(final, text, cut)
