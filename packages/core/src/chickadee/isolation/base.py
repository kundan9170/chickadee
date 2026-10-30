"""Backend protocol, detection, and selection order.

**Availability is not application.** Detection says what this machine *could* enforce;
`ExecResult.sandbox` says what a run actually got. Reporting the first as the second is the
mistake `packages/eval` already made once and had to fix, and it is
what turns "the sandbox held" into a claim nobody checked.

**Available means it runs.** A backend whose binary is on the path can still fail every
wrap: on Ubuntu 24.04 AppArmor refuses `bwrap` a user namespace. So a real backend's
`available()` is `probe(self).ran`: one harmless `true` under it, once per process.
"""

from __future__ import annotations

import platform
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from chickadee.errors import InfraError
from chickadee.isolation.types import Isolation, SandboxLevel
from chickadee.process import Completed, ProcessSupervisor, scrub_env


class SandboxBackend(Protocol):
    def available(self) -> bool: ...
    def level(self) -> SandboxLevel: ...
    def wrap(self, isolation: Isolation) -> tuple[str, ...]: ...


def select(preferred: str | None = None) -> SandboxBackend:
    """Strongest available backend, or the honest passthrough.

    Selection order is container → native → none. `preferred` forces a choice so a config
    can pin one; an unavailable preference falls through rather than pretending.
    """
    from chickadee.isolation import container, none

    # An explicit preference is a decision, so it is honoured before availability ranking.
    # `none` in particular must mean "run unwrapped and say so", not "fall through to
    # whatever this machine happens to have" — which is the opposite of the request.
    if preferred == SandboxLevel.NONE.value:
        return none.Passthrough()

    backends: list[SandboxBackend] = [
        container.Container(enabled=preferred == SandboxLevel.CONTAINER.value)
    ]
    backends.append(native())

    if preferred:
        for backend in backends:
            if backend.level().value == preferred and backend.available():
                return backend
    for backend in backends:
        if backend.available():
            return backend
    return none.Passthrough()


def has(binary: str) -> bool:
    return shutil.which(binary) is not None


def configured(backend: str) -> SandboxBackend:
    """The backend `[sandbox] backend` names: `auto` is `select()`'s ranking, and any
    other value is `select(that)`, which falls through to the next backend, or to `none`, when
    it does not run here. The one place the engine chooses: every spawn site it builds and the
    run header take this, so the header names the backend the commands run under."""
    return select(None if backend == "auto" else backend)


def level_of(backend: SandboxBackend) -> SandboxLevel:
    """The level a command wrapped by `backend` runs at: its own when it runs here, `none`
    when it does not. The reading every spawn site makes before it spawns."""
    return backend.level() if backend.available() else SandboxLevel.NONE


def reads_masked(level: SandboxLevel, *, allow_secret_read: bool) -> bool:
    """Whether a wrapped read of a deny-listed path is masked: there is a sandbox to
    mask it, and `allow_secret_read` has not resolved the mask to nothing. What Zeus's prefix
    says and the header's sandbox line says, from one rule."""
    return level is not SandboxLevel.NONE and not allow_secret_read


def native() -> SandboxBackend:
    """This platform's own backend: Seatbelt on macOS, bubblewrap elsewhere. What `setup`
    probes, since it is the one a person can install or unblock."""
    from chickadee.isolation import linux, macos

    return macos.Seatbelt() if platform.system() == "Darwin" else linux.Bubblewrap()


# -- the probe ---------------------------------------------------------------------

PROBE_TIMEOUT_S = 10
"""Long enough for a container runtime to start `true`; a native wrap takes milliseconds."""


@dataclass(frozen=True)
class Probe:
    """Whether a backend ran one harmless command, and when it did not, why and what to do.

    `chickadee setup` prints it. `fix` names what a person would do; nothing here
    does it."""

    level: SandboxLevel
    ran: bool
    reason: str = ""
    fix: str = ""


@runtime_checkable
class Probed(Protocol):
    """What a real backend tells the probe about itself. The probe owns running and caching;
    a backend owns only the facts about its own binary."""

    def probe_key(self) -> tuple[str, ...]:
        """What makes two probes of this backend the same question: the binary that would
        run, as `PATH` finds it now, and anything else the wrap depends on."""
        ...

    def not_ready(self) -> tuple[str, str] | None:
        """Why the probe cannot even try, as (reason, fix), or None when it can."""
        ...

    def diagnose(self, done: Completed) -> tuple[str, str]:
        """Why `true` did not run under this backend, as (reason, fix)."""
        ...


_probes: dict[tuple[str, ...], Probe] = {}
_probes_lock = threading.Lock()
"""Guards `_probes`. One probe per backend per process: every spawn asks `available()`, and
the answer does not change under a session. Sessions run on their own worker threads,
and a probe runs under the lock, so two threads never spawn the same one.

Only a stable answer is kept: it ran, it is absent or not enabled, or it failed and said why.
A timeout or a failed spawn is asked again next time, because caching one slow start would
leave every later command in the process unsandboxed."""


def probe(backend: SandboxBackend) -> Probe:
    """Run `true` under `backend`, read-only and network off, through the supervisor, once per
    process. The passthrough has nothing to try. A backend that does not describe
    itself to the probe (a test double) is taken at its own `available()`."""
    level = backend.level()
    if level is SandboxLevel.NONE:
        return Probe(level, ran=True)
    if not isinstance(backend, Probed):
        return Probe(level, ran=backend.available())
    key = (type(backend).__qualname__, *backend.probe_key())
    with _probes_lock:
        found = _probes.get(key)
        if found is None:
            found, stable = _run_probe(backend, backend, level)
            if stable:
                _probes[key] = found
        return found


def probe_native() -> Probe:
    """The probe of this platform's own backend: what `chickadee setup` reports."""
    return probe(native())


def forget_probes() -> None:
    """Drop every cached probe, so the next `available()` asks again: for setup after the
    person has applied a fix, and for tests."""
    with _probes_lock:
        _probes.clear()


def _run_probe(
    backend: SandboxBackend, facts: Probed, level: SandboxLevel
) -> tuple[Probe, bool]:
    """The probe, and whether its answer is stable enough to keep for the process."""
    blocked = facts.not_ready()
    if blocked is not None:
        return Probe(level, False, *blocked), True
    with tempfile.TemporaryDirectory(prefix="chickadee-probe-") as scratch:
        cwd = Path(scratch)
        wrapped = backend.wrap(Isolation(argv=("true",), cwd=cwd))
        try:
            done = ProcessSupervisor().run(
                wrapped, cwd=cwd, env=scrub_env(), timeout_s=PROBE_TIMEOUT_S
            )
        except InfraError as exc:
            # The supervisor's spawn failure: the binary vanished or is not executable.
            fix = f"check that {wrapped[0]} is installed and runs"
            return Probe(level, False, str(exc), fix), False
    if done.timed_out:
        why = f"`true` under {wrapped[0]} did not finish in {PROBE_TIMEOUT_S}s"
        fix = f"run `{' '.join(wrapped)}` to see where it waits"
        return Probe(level, False, why, fix), False
    if done.exit_code == 0:
        return Probe(level, ran=True), True
    return Probe(level, False, *facts.diagnose(done)), True


def first_line(done: Completed) -> str:
    """The first line a failed probe printed, so its reason reads as one line."""
    for raw in (done.stderr, done.stdout):
        for line in raw.decode("utf-8", "replace").splitlines():
            if line.strip():
                return line.strip()
    return f"exit {done.exit_code}"
