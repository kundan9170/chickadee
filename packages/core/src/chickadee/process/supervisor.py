"""The only `Popen` in core.

This module is spawn, env scrub, timeout, and process-group kill: batch until exit, a
short-lived session, or a job that runs on past its call. Callers pass the argv
already wrapped. LSP framing stays in
`graph/lsp.py`. This package must not import `chickadee.executor`.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chickadee.errors import InfraError
from chickadee.paths import within

log = logging.getLogger(__name__)

_CREDENTIAL_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "APIKEY",
    "API_KEY",
    "CREDENTIAL",
    "AUTH",
)
_ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SHELL", "USER", "TERM")

NETWORK_ENV: tuple[str, ...] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "PIP_CERT",
    "GIT_SSL_CAINFO",
    "NODE_EXTRA_CA_CERTS",
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX",
    "UV_CACHE_DIR",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "NPM_CONFIG_REGISTRY",
    "npm_config_registry",
    "NPM_CONFIG_CACHE",
    "XDG_CACHE_HOME",
)
"""How a machine reaches the network, kept beyond the keep-list for a spawn whose network is
open: the worktree's bootstrap, and a model's command once its network is open. A
proxy, a corporate CA bundle, a private index: without them an install fails behind a proxy even
with the network on, as it did on 2026-10-18. Codex and Claude Code set these same variables
when their network opens. The credential scrub still applies to each, by name; a proxy URL with
embedded auth is the operator's own environment, and reaching the network is the point."""

GRACE_S = 2.0
"""How long a group gets to honour SIGTERM before SIGKILL."""

CLOSE_S = 0.5
"""How long a session's process gets to leave by itself once its stdin is closed, before the
group is signalled: a language server told `shutdown` and `exit` leaves in milliseconds, and
SIGTERM sent at once cost basedpyright's wrapper and Node ~2 s to unwind."""

CANCEL_POLL_MS = 100
"""The longest a wait goes without looking at the cancellation token. A stop reaches a
running command within this plus the reap's grace, rather than at the command's timeout."""

JOB_CHECK_S = 1.0
"""How often a job's waiter looks at its runtime bound and its output's size:
a job can overshoot `max_log_bytes` by at most this much output. Its exit is not polled: the
waiter blocks on it between looks."""


class Cancelled(Exception):
    """A session's blocking call returned because the token was set.

    Raised, never returned: a session read has no "partial result" a caller could mistake
    for the child's answer. The process is **not** reaped — a turn interrupt is not the end
    of a dev server, and `close()` or the `with` block's exit still owns that.
    """


@dataclass(frozen=True)
class Completed:
    stdout: bytes
    stderr: bytes
    exit_code: int
    timed_out: bool
    cancelled: bool = False
    """The token was set and the group was reaped. Not a timeout — the
    argument that an exit code cannot carry the verdict, applied to the user's stop
    versus the harness's limit. The default keeps every existing constructor unchanged."""


Ended = Literal["exited", "timed_out", "killed", "ceiling"]
"""How a job ended: by itself; at its runtime bound (a task's); killed by
whoever named itself in `by`; or past its output ceiling. `lost`, a job the harness
did not see end, is resume's word, never this module's."""


@dataclass(frozen=True)
class JobExit:
    """What a job's waiter saw at its end, set once."""

    exit_code: int
    """The child's return code as `Completed` carries one: a signal's death is `-n`."""
    ended: Ended
    wall_ms: int
    by: str | None
    """Who asked for the end (`Job.kill`), named before the signal was sent: an exit someone
    asked for never notifies."""
    seen: bool
    """A `wait(observe=True)` was blocked on the job when it ended, so the exit was handed to
    whoever waited. Decided under the lock the outcome is set under."""


class Job:
    """A child that runs on past its call: stdout and stderr in two files,
    stdin `/dev/null`, its own process group. One waiter thread blocks on its exit, reaps the
    group at every exit, and holds the bounds.

    **Guarded by `_lock`:** `_exit`, `_by`, `_gone` and `_observers`. The outcome is set
    exactly once, under it; `wait` sleeps on `_done`, its condition."""

    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        pgid: int | None,
        *,
        out: Path,
        err: Path,
        max_runtime_s: int | None,
        max_log_bytes: int,
        on_exit: Callable[[Job, JobExit], None] | None,
    ) -> None:
        self._proc = proc
        self.pid = proc.pid
        self.pgid = pgid
        self.out = out
        self.err = err
        self.started = process_started(proc.pid)
        """The kernel's start time for the pid, which resume compares before it signals."""
        self._t0 = time.monotonic()
        self._max_runtime_s = max_runtime_s
        self._max_log_bytes = max_log_bytes
        self._on_exit = on_exit
        self._lock = threading.Lock()
        self._done = threading.Condition(self._lock)
        self._exit: JobExit | None = None
        self._by: str | None = None
        self._observers = 0
        self._gone = False
        """The waiter has seen the process end: a kill after that came after the exit."""
        # The waiter keeps the session's binding: a thread starts with an empty context.
        context = contextvars.copy_context()
        self._waiter = threading.Thread(
            target=context.run, args=(self._watch,), name=f"job-{proc.pid}", daemon=True
        )
        self._waiter.start()

    @property
    def outcome(self) -> JobExit | None:
        with self._lock:
            return self._exit

    def alive(self) -> bool:
        return self.outcome is None

    def elapsed_ms(self) -> int:
        """Milliseconds since it started: a running job's, for the answer that yields it."""
        return int((time.monotonic() - self._t0) * 1000)

    def wait(
        self,
        timeout_s: float | None,
        cancel: threading.Event | None = None,
        *,
        observe: bool = False,
    ) -> JobExit | None:
        """The exit, or `None` at `timeout_s` with the job still running. Waits in slices of
        `CANCEL_POLL_MS` and looks at the token between them: on it, raises `Cancelled`,
        the job left running. `observe`: whoever waits will hand the exit on, so one that
        arrives during this wait is `seen`."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._lock:
            if observe:
                self._observers += 1
            try:
                while self._exit is None:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled
                    left = None if deadline is None else deadline - time.monotonic()
                    if left is not None and left <= 0:
                        return None
                    poll = CANCEL_POLL_MS / 1000 if cancel is not None else None
                    bounds = [b for b in (left, poll) if b is not None]
                    self._done.wait(min(bounds) if bounds else None)
                return self._exit
            finally:
                if observe:
                    self._observers -= 1

    def kill(self, by: str) -> bool:
        """End the job, naming who asked before the signal is sent. The
        waiter sees the exit and records it as `killed`. False, and nothing is sent, when the
        process had already ended: that exit is its own, however soon after it this came."""
        with self._lock:
            if self._gone:
                return False
            if self._by is None:
                self._by = by
        reap_group(self.pgid, self._proc)
        return True

    def join(self, timeout_s: float) -> None:
        """Until the waiter has set the outcome and run `on_exit`, or `timeout_s`."""
        self._waiter.join(timeout_s)

    def _watch(self) -> None:
        ended: Ended = "exited"
        while True:
            try:
                self._proc.wait(timeout=JOB_CHECK_S)
                break
            except subprocess.TimeoutExpired:
                pass
            ran = time.monotonic() - self._t0
            if self._max_runtime_s is not None and ran >= self._max_runtime_s:
                ended = "timed_out"
            elif self._log_bytes() > self._max_log_bytes:
                ended = "ceiling"
            else:
                continue
            reap_group(self.pgid, self._proc)
            self._proc.wait()
            break
        # Who asked for the end is read as the end is seen, before the reap below, which can
        # take `GRACE_S`: a kill in that time came after the exit and is not its cause.
        with self._lock:
            self._gone = True
            by = self._by
        # Every exit reaps the group, as `run` does: a child the leader left behind dies too.
        reap_group(self.pgid, self._proc)
        code = self._proc.returncode
        with self._lock:
            exit_ = JobExit(
                exit_code=code,
                ended="killed" if by is not None and ended == "exited" else ended,
                wall_ms=int((time.monotonic() - self._t0) * 1000),
                by=by,
                seen=self._observers > 0,
            )
            self._exit = exit_
            self._done.notify_all()
        if self._on_exit is not None:
            try:
                self._on_exit(self, exit_)
            except Exception:
                # Nothing above this thread would see it: the outcome stands, and the failure
                # to record it is the log's to show.
                log.exception("job %s: recording its exit failed", self.pid)

    def _log_bytes(self) -> int:
        total = 0
        for path in (self.out, self.err):
            with contextlib.suppress(OSError):
                total += path.stat().st_size
        return total


def process_started(pid: int) -> str | None:
    """The kernel's start time for `pid`, or `None` when it is gone or cannot be read. Resume
    compares it with the recorded one before it signals anything, so a reused pid is never
    signalled. Linux reads field 22 of `/proc/<pid>/stat`, in clock ticks
    since boot; elsewhere `ps -o lstart=`, through this module's own `run`."""
    if sys.platform.startswith("linux"):
        stat = Path(f"/proc/{pid}/stat")
        try:
            text = stat.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        # The command name is in parentheses and may hold spaces: fields follow the last `)`.
        fields = text[text.rfind(")") + 2 :].split()
        return fields[19] if len(fields) > 19 else None
    done = ProcessSupervisor().run(
        ("ps", "-o", "lstart=", "-p", str(pid)), cwd=Path("/"), env=scrub_env(), timeout_s=5
    )
    started = done.stdout.decode("utf-8", "replace").strip()
    return started if done.exit_code == 0 and started else None


_session: ContextVar[str] = ContextVar("chickadee_session", default="")
"""The session the spawns made in this context belong to, `""` outside any session.
A context variable rather than a process-wide value, because one engine holds several
sessions open at once, each driven on its own worker thread, and every thread has its
own context: no lock is needed, and one session's binding never reaches another's children."""


def bind_session(session_id: str) -> None:
    """Name the session every later spawn in this context belongs to.

    A binding rather than an argument because `scrub_env` is called from every spawn path,
    several of them (`verifier.infra`, `graph`, `process.harness`) with no session in hand;
    threading an argument through each would be the edit the one-function design
    avoids. Each conversation binds its worker thread once, before its build runs
    (`zeus/conversation.py`), so everything the session spawns on that thread carries its id.
    A thread started from there begins with an empty context: it inherits the binding only
    through `contextvars.copy_context().run`.
    """
    _session.set(session_id)


def bound_session() -> str:
    """The session this context's spawns belong to (`bind_session`), `""` outside any."""
    return _session.get()


def _without_own_venv(path: str, workspace: Path) -> str:
    """`path` without the `bin` of the virtualenv this process runs in, unless that
    environment's directory holds `workspace` (a project venv the harness was installed
    into, whose `pytest` the work wants). Outside a virtualenv, `path` unchanged. A harness
    started with `uv run` or from an activated venv puts its own `bin` first, and a model's
    `python3` would otherwise be the harness's interpreter."""
    prefix = Path(sys.prefix)
    if prefix == Path(sys.base_prefix):
        return path
    if within(prefix.parent, workspace) is not None:
        return path
    own = {os.path.realpath(prefix / d) for d in ("bin", "Scripts")}
    kept = [entry for entry in path.split(os.pathsep) if os.path.realpath(entry) not in own]
    return os.pathsep.join(kept)


def scrub_env(
    *, human: bool = False, workspace: Path | None = None, network: bool = False
) -> dict[str, str]:
    """Drop credentials. HUMAN keeps the user's shell; everyone else gets a keep-list.

    `workspace`: the directory a model's command runs in (`Executor.run`). Given, the harness's
    own virtualenv leaves `PATH` unless it is the work's (`_without_own_venv`),
    so `python3 -m pip` is the project's Python, not the one the harness was started with. The
    harness's own spawns pass none: its syntax checks and language servers may live there.

    Then, **after** both filters, the two markers put in every spawn: `CHICKADEE=1`,
    so a child knows it is under an agent (no prompts, no colour, no `$EDITOR`), and
    `CHICKADEE_SESSION`, so two sessions' children are told apart. Set last, so neither a
    keep-list nor the user's own shell can shadow them. A marker, never a capability, and
    `CI` is deliberately not set: it means a CI job to other tools, and this is not one.
    `CHICKADEE_SESSION` is empty outside a session: `init_repo`'s spawns, and anything this
    context's `bind_session` has not named.

    `network`: the spawn's network is open, so `NETWORK_ENV` is kept too.
    """
    kept = (*_ENV_KEEP, *NETWORK_ENV) if network else _ENV_KEEP
    env = dict(os.environ) if human else {k: v for k, v in os.environ.items() if k in kept}
    scrubbed = {
        k: v
        for k, v in env.items()
        if not any(marker in k.upper() for marker in _CREDENTIAL_MARKERS)
    }
    if workspace is not None and not human and "PATH" in scrubbed:
        scrubbed["PATH"] = _without_own_venv(scrubbed["PATH"], workspace)
    scrubbed["CHICKADEE"] = "1"
    scrubbed["CHICKADEE_SESSION"] = _session.get()
    if not human:
        # A model's `python -m py_compile` or `import` writes no `__pycache__` into the
        # person's tree: bytecode is a cache, never the work.
        scrubbed["PYTHONDONTWRITEBYTECODE"] = "1"
    return scrubbed


def person_env(*, withheld: frozenset[str], workspace: Path) -> dict[str, str]:
    """The environment of a command the person approved to run outside the sandbox: theirs,
    whole, so a credential helper, an SSH agent or a private registry's token
    works as it does in their own shell. Less `withheld`, the harness's own model endpoints'
    keys (`EngineConfig.endpoint_keys`), which no model's command may read. The harness's own
    virtualenv still leaves `PATH`, and the markers are set last, as
    `scrub_env` sets them."""
    env = {k: v for k, v in os.environ.items() if k not in withheld}
    if "PATH" in env:
        env["PATH"] = _without_own_venv(env["PATH"], workspace)
    env["CHICKADEE"] = "1"
    env["CHICKADEE_SESSION"] = _session.get()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - exists, not ours
        return True
    return True


def reap_group(pgid: int | None, leader: subprocess.Popen[bytes] | None = None) -> None:
    """SIGTERM, wait up to GRACE_S, then SIGKILL. Verify the group is gone.

    `leader` is the group's leader when it is this process's child: polled as the group is
    watched, so its exit is reaped. Unreaped, it is a zombie, and a group whose only member is
    a zombie still answers `killpg(pgid, 0)`: every reap waited the whole `GRACE_S` and then
    killed what had already exited (2 s on every language server's close)."""
    if pgid is None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + GRACE_S
    while time.monotonic() < deadline:
        if leader is not None:
            leader.poll()
        if not _group_alive(pgid):
            return
        time.sleep(0.05)
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pgid, signal.SIGKILL)


class ProcessSession:
    """Live stdin/stdout until `close`. Does not outlive the caller's `with` block."""

    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        pgid: int | None,
        deadline: float,
        cancel: threading.Event | None = None,
    ) -> None:
        self._proc = proc
        self._pgid = pgid
        self._deadline = deadline
        self._cancel = cancel
        self._closed = False

    def _cancelled(self) -> bool:
        return self._cancel is not None and self._cancel.is_set()

    def _remaining(self) -> float:
        return self._deadline - time.monotonic()

    def write(self, data: bytes) -> None:
        """Write all of `data` by the session's deadline, or raise `TimeoutError`. In pieces
        no larger than the pipe's atomic size, each only once the pipe can take it, so a child
        that stops reading (a full stdout of its own) cannot block the harness past the
        deadline: a blocking write had none. Its stderr is drained meanwhile (`_drain`)."""
        if self._closed or self._proc.stdin is None:
            raise InfraError("process session stdin is closed")
        if self._cancelled():
            raise Cancelled("the session's cancellation token is set; nothing was written")
        fd = self._proc.stdin.fileno()
        view = memoryview(data)
        while view:
            if self._cancelled():
                raise Cancelled("the session's cancellation token is set; part was written")
            remaining = self._remaining()
            if remaining <= 0:
                raise TimeoutError("process session timed out")
            readers = self._stderr_fds()
            readable, writable, _ = select.select(
                readers, [fd], [], min(remaining, CANCEL_POLL_MS / 1000)
            )
            if readable:
                self._drain()
            if writable:
                view = view[os.write(fd, view[: select.PIPE_BUF]) :]

    def read(self, n: int) -> bytes:
        """Wait for output in slices of at most `CANCEL_POLL_MS`, looking at the token
        between them. On the token, raise `Cancelled` and leave the child running. The
        child's stderr is drained meanwhile (`_drain`)."""
        return self._read_by(n, self._deadline)

    def read_within(self, n: int, timeout_s: float) -> bytes:
        """`read`, bounded by `timeout_s` as well as the session's deadline: a session kept for
        a long life (an episode's language server) still bounds each answer it waits on. The
        session stays open after this one's `TimeoutError`."""
        return self._read_by(n, min(self._deadline, time.monotonic() + timeout_s))

    def _read_by(self, n: int, deadline: float) -> bytes:
        if n <= 0 or self._proc.stdout is None:
            return b""
        fd = self._proc.stdout.fileno()
        while True:
            if self._cancelled():
                raise Cancelled("the session's cancellation token is set; the process runs on")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("process session timed out")
            ready, _, _ = select.select(
                [fd, *self._stderr_fds()], [], [], min(remaining, CANCEL_POLL_MS / 1000)
            )
            if fd in ready:
                return os.read(fd, n)
            if ready:
                self._drain()

    def _stderr_fds(self) -> list[int]:
        stderr = self._proc.stderr
        return [stderr.fileno()] if stderr is not None and not stderr.closed else []

    def _drain(self) -> None:
        """Read and drop what the child wrote to stderr. A session's caller speaks over stdout
        and never reads stderr, and a child whose stderr pipe filled would block mid-answer
        (a language server's log, a stack trace): the session would end at its deadline with
        the child stuck, not answering."""
        stderr = self._proc.stderr
        if stderr is None or stderr.closed:
            return
        with contextlib.suppress(OSError, ValueError):
            if not os.read(stderr.fileno(), 65536):
                stderr.close()  # EOF: the child closed it; stop selecting on it

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc.stdin is not None:
            with contextlib.suppress(OSError, BrokenPipeError):
                self._proc.stdin.close()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.wait(timeout=CLOSE_S)
        reap_group(self._pgid, self._proc)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.wait(timeout=1)
        # Drain so a child blocked on a full pipe is not why wait hung.
        if self._proc.stdout is not None:
            with contextlib.suppress(OSError, ValueError):
                self._proc.stdout.close()
        if self._proc.stderr is not None:
            with contextlib.suppress(OSError, ValueError):
                self._proc.stderr.close()


class ProcessSupervisor:
    """One spawn implementation. Batch for wrap callers; session for LSP/MCP stdio."""

    GRACE_S = GRACE_S

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: int,
        stdin: bytes | None = None,
        cancel: threading.Event | None = None,
    ) -> Completed:
        """Run until exit, the timeout, or the token. Stdout is available only after the
        child is gone.

        The wait is in slices of at most `CANCEL_POLL_MS`, and the token is looked at
        between them; on set, the group is reaped and the result says `cancelled`, with
        whatever output was captured. `communicate` keeps what it read across a retried
        timeout, so slicing loses nothing. `cancel=None` never cancels, which is the
        behaviour every existing caller had. The timeout path is unchanged and sets
        `timed_out`, never `cancelled`.
        """
        proc, pgid = self._popen(argv, cwd=cwd, env=env, stdin_pipe=stdin is not None)
        deadline = time.monotonic() + timeout_s
        pending = stdin
        while True:
            if cancel is not None and cancel.is_set() and proc.poll() is None:
                # Only a child still running is cancelled. One that finished in the slice
                # the token was set in finished: its own exit is the truth, read below.
                reap_group(pgid, proc)
                stdout, stderr = proc.communicate()
                return Completed(
                    stdout or b"", stderr or b"", proc.returncode, False, cancelled=True
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reap_group(pgid, proc)
                stdout, stderr = proc.communicate()
                return Completed(stdout or b"", stderr or b"", proc.returncode or 124, True)
            wait = remaining if cancel is None else min(remaining, CANCEL_POLL_MS / 1000)
            try:
                stdout, stderr = proc.communicate(input=pending, timeout=wait)
            except subprocess.TimeoutExpired:
                # Input goes in once: a retried `communicate` refuses new input, and the
                # first call has already written it.
                pending = None
                continue
            reap_group(pgid, proc)
            return Completed(stdout or b"", stderr or b"", proc.returncode or 0, False)

    @contextmanager
    def session(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_s: int,
        env: Mapping[str, str] | None = None,
        cancel: threading.Event | None = None,
    ) -> Generator[ProcessSession]:
        """Turn-taking handle. Reaped when the `with` block ends, timeout or not — and not
        when the token is set: a blocking call raises `Cancelled` and the child lives on,
        because stopping a turn is not stopping a server."""
        proc, pgid = self._popen(argv, cwd=cwd, env=env or scrub_env(), stdin_pipe=True)
        sess = ProcessSession(proc, pgid, time.monotonic() + timeout_s, cancel)
        try:
            yield sess
        finally:
            sess.close()

    def start(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        out: Path,
        err: Path,
        max_runtime_s: int | None,
        max_log_bytes: int,
        on_exit: Callable[[Job, JobExit], None] | None = None,
    ) -> Job:
        """A child that may run on past its call: stdout to `out`, stderr to
        `err`, stdin `/dev/null`, so a full pipe never stalls it, its output outlives the
        harness, and it never reads the harness's terminal. `max_runtime_s=None` is a
        server's: no runtime bound. `on_exit` runs once, on the job's waiter thread, after
        the outcome is set."""
        proc, pgid = self._popen(argv, cwd=cwd, env=env, stdin_pipe=False, out=out, err=err)
        return Job(
            proc,
            pgid,
            out=out,
            err=err,
            max_runtime_s=max_runtime_s,
            max_log_bytes=max_log_bytes,
            on_exit=on_exit,
        )

    def _popen(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin_pipe: bool,
        out: Path | None = None,
        err: Path | None = None,
    ) -> tuple[subprocess.Popen[bytes], int | None]:
        """`out` and `err` given, the child writes to those files and reads `/dev/null`;
        otherwise both are pipes, as for `run` and `session`."""
        with contextlib.ExitStack() as files:
            if out is not None and err is not None:
                stdin: int | None = subprocess.DEVNULL
                # Descriptors the child writes, closed here once it holds them: its output,
                # never a durable write of the harness's, so not `chickadee.durable`'s.
                stdout = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                files.callback(os.close, stdout)
                stderr = os.open(err, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                files.callback(os.close, stderr)
            else:
                stdin = subprocess.PIPE if stdin_pipe else None
                stdout = stderr = subprocess.PIPE
            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=dict(env),
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except OSError as exc:
                raise InfraError(f"could not spawn {argv[0]!r}: {exc}") from exc
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):  # pragma: no cover - exited instantly
            pgid = None
        return proc, pgid


__all__ = [
    "CANCEL_POLL_MS",
    "JOB_CHECK_S",
    "NETWORK_ENV",
    "Cancelled",
    "Completed",
    "Ended",
    "Job",
    "JobExit",
    "ProcessSession",
    "ProcessSupervisor",
    "bind_session",
    "bound_session",
    "person_env",
    "process_started",
    "reap_group",
    "scrub_env",
]
