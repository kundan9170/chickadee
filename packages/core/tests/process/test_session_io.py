"""A session's writes keep its deadline, and its stderr never fills: a language
server that stops reading, or logs a lot, must not hang the index past its timeout."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from chickadee.process.supervisor import ProcessSupervisor, scrub_env


def _script(tmp_path: Path, body: str) -> tuple[str, ...]:
    path = tmp_path / "child.py"
    path.write_text(body, encoding="utf-8")
    return (sys.executable, str(path))


def test_a_write_to_a_child_that_never_reads_times_out(tmp_path: Path) -> None:
    """A blocking write had no deadline: a child that stopped reading hung the caller."""
    argv = _script(tmp_path, "import time\ntime.sleep(30)\n")
    started = time.monotonic()
    with (
        ProcessSupervisor().session(argv, cwd=tmp_path, timeout_s=2, env=scrub_env()) as sess,
        pytest.raises(TimeoutError),
    ):
        sess.write(b"x" * (4 * 1024 * 1024))
    assert time.monotonic() - started < 10


def test_a_child_that_floods_stderr_still_answers(tmp_path: Path) -> None:
    """Nothing read the child's stderr: once its pipe filled, the child blocked before its
    answer, and the session ended at its deadline with nothing said."""
    body = (
        "import sys\n"
        "sys.stderr.write('log line\\n' * 100_000)\n"
        "sys.stderr.flush()\n"
        "line = sys.stdin.readline()\n"
        "sys.stdout.write('echo:' + line)\n"
        "sys.stdout.flush()\n"
    )
    argv = _script(tmp_path, body)
    with ProcessSupervisor().session(argv, cwd=tmp_path, timeout_s=10, env=scrub_env()) as sess:
        sess.write(b"hello\n")
        got = b""
        while not got.endswith(b"\n"):
            got += sess.read(4096)
    assert got == b"echo:hello\n"


def test_a_bounded_read_times_out_and_the_session_answers_after(tmp_path: Path) -> None:
    """An episode's language server lives long, and each answer it is waited on still has a
    deadline (`read_within`); the session is still open after one passes."""
    body = (
        "import sys\n"
        "for line in sys.stdin:\n"
        "    sys.stdout.write(line)\n"
        "    sys.stdout.flush()\n"
    )
    argv = _script(tmp_path, body)
    with ProcessSupervisor().session(argv, cwd=tmp_path, timeout_s=60, env=scrub_env()) as sess:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            sess.read_within(64, 0.3)
        assert time.monotonic() - started < 5, "bounded by its own timeout, not the session's"
        sess.write(b"ping\n")
        assert sess.read_within(64, 10) == b"ping\n"
