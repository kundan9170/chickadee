"""ProcessSupervisor: batch, session, timeout. The only `Popen` in core."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from chickadee.process.supervisor import (
    GRACE_S,
    ProcessSupervisor,
    bind_session,
    reap_group,
    scrub_env,
)


def test_batch_returns_stdout_after_exit(tmp_path: Path) -> None:
    done = ProcessSupervisor().run(
        ("echo", "hi"),
        cwd=tmp_path,
        env=scrub_env(),
        timeout_s=5,
    )
    assert b"hi" in done.stdout
    assert not done.timed_out
    assert done.exit_code == 0


def test_session_turn_taking(tmp_path: Path) -> None:
    script = tmp_path / "echo.py"
    script.write_text(
        "import sys\n"
        "for _ in range(2):\n"
        "    line = sys.stdin.buffer.readline()\n"
        "    sys.stdout.buffer.write(line)\n"
        "    sys.stdout.buffer.flush()\n",
        encoding="utf-8",
    )
    with ProcessSupervisor().session(
        (sys.executable, str(script)),
        cwd=tmp_path,
        timeout_s=5,
    ) as sess:
        sess.write(b"one\n")
        assert sess.read(4) == b"one\n"
        sess.write(b"two\n")
        assert sess.read(4) == b"two\n"


def test_session_timeout_kills_the_child(tmp_path: Path) -> None:
    with (
        pytest.raises(TimeoutError),
        ProcessSupervisor().session(("sleep", "30"), cwd=tmp_path, timeout_s=1) as sess,
    ):
        sess.read(1)


# --- D113: every spawn knows it is under the harness -----------------------------------


@pytest.fixture
def session() -> Iterator[str]:
    bind_session("s-0113")
    yield "s-0113"
    bind_session("")


def _child_env(stdout: bytes) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.decode().splitlines() if "=" in line)


def test_a_batch_spawn_carries_both_markers(tmp_path: Path, session: str) -> None:
    done = ProcessSupervisor().run(("env",), cwd=tmp_path, env=scrub_env(), timeout_s=5)
    child = _child_env(done.stdout)
    assert child["CHICKADEE"] == "1"
    assert child["CHICKADEE_SESSION"] == session


def test_an_agent_spawn_writes_no_bytecode_and_a_human_one_is_left_alone() -> None:
    """A model's `py_compile` or `import` left `__pycache__` in the person's tree."""
    assert scrub_env()["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "PYTHONDONTWRITEBYTECODE" not in scrub_env(human=True) or os.environ.get(
        "PYTHONDONTWRITEBYTECODE"
    )


def test_a_session_spawn_carries_both_markers(tmp_path: Path, session: str) -> None:
    """`session()` builds its own environment when handed none; that path is marked too."""
    script = tmp_path / "markers.py"
    script.write_text(
        "import os, sys\n"
        "env = os.environ\n"
        "sys.stdout.write(env['CHICKADEE'] + ' ' + env['CHICKADEE_SESSION'] + '\\n')\n"
        "sys.stdout.flush()\n",
        encoding="utf-8",
    )
    with ProcessSupervisor().session(
        (sys.executable, str(script)), cwd=tmp_path, timeout_s=5
    ) as s:
        expected = f"1 {session}\n".encode()
        assert s.read(len(expected)) == expected


def test_a_human_spawn_carries_both_and_the_users_shell_cannot_shadow_them(
    tmp_path: Path, session: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HUMAN keeps the user's shell, and a value the user exported is replaced, not
    inherited: the markers are set after both filters."""
    monkeypatch.setenv("CHICKADEE", "0")
    monkeypatch.setenv("CHICKADEE_SESSION", "someone-elses")
    env = scrub_env(human=True)
    assert env["CHICKADEE"] == "1"
    assert env["CHICKADEE_SESSION"] == session
    done = ProcessSupervisor().run(("env",), cwd=tmp_path, env=env, timeout_s=5)
    assert _child_env(done.stdout)["CHICKADEE"] == "1"


def test_ci_is_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """`CI` means a CI job to hundreds of tools, and this is not one."""
    monkeypatch.delenv("CI", raising=False)
    assert "CI" not in scrub_env()
    assert "CI" not in scrub_env(human=True)


def test_an_unbound_session_is_marked_empty_not_absent() -> None:
    """A child still learns it is under the harness before any session is named."""
    env = scrub_env()
    assert env["CHICKADEE"] == "1"
    assert env["CHICKADEE_SESSION"] == ""


# -- the harness's own virtualenv -----------------------------------------


def _in_venv(monkeypatch: pytest.MonkeyPatch, venv: Path, others: str) -> None:
    """The harness running from `venv`, with its `bin` first on `PATH`, as `uv run` puts it."""
    (venv / "bin").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setenv("PATH", os.pathsep.join((str(venv / "bin"), others)))


def test_a_models_command_never_gets_the_harnesss_own_venv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The owner's run: `python3 -m pip` was `…/project-chickadee/.venv/bin/python3`, which has
    no pip, because the harness's `PATH` came through whole."""
    _in_venv(monkeypatch, tmp_path / "chickadee" / ".venv", "/usr/bin")
    work = tmp_path / "ai_agent"
    work.mkdir()
    assert scrub_env(workspace=work)["PATH"] == "/usr/bin"


def test_a_project_venv_the_harness_was_installed_into_is_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The venv is the work's when its directory holds the workspace: its `pytest` is wanted."""
    venv = tmp_path / "project" / ".venv"
    _in_venv(monkeypatch, venv, "/usr/bin")
    work = tmp_path / "project" / "src"
    work.mkdir(parents=True)
    assert scrub_env(workspace=work)["PATH"].split(os.pathsep)[0] == str(venv / "bin")


def test_the_harnesss_own_spawns_and_the_persons_keep_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A syntax check or a language server may live in the harness's venv, and a `!` line is
    the person's shell: neither names a workspace to judge it by."""
    venv = tmp_path / "chickadee" / ".venv"
    _in_venv(monkeypatch, venv, "/usr/bin")
    work = tmp_path / "elsewhere"
    work.mkdir()
    assert str(venv / "bin") in scrub_env()["PATH"]
    assert str(venv / "bin") in scrub_env(human=True, workspace=work)["PATH"]


def test_outside_a_venv_the_path_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    assert scrub_env(workspace=tmp_path)["PATH"] == "/usr/local/bin:/usr/bin"


# -- an exit is seen when it happens -----------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX")
def test_a_session_that_leaves_on_eof_closes_at_once(tmp_path: Path) -> None:
    """A child that leaves when its stdin closes (a language server told `exit`) is never
    signalled, and its exit is seen at once: the group was taken as alive while its leader was
    an unreaped zombie, and every close waited `GRACE_S`."""
    with ProcessSupervisor().session(
        ("cat",), cwd=tmp_path, timeout_s=10, env=scrub_env()
    ) as s:
        started = time.monotonic()
        s.close()
    assert time.monotonic() - started < GRACE_S / 2


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX")
def test_a_group_reaped_with_its_leader_is_gone_once_it_exits(tmp_path: Path) -> None:
    proc = subprocess.Popen(("sleep", "30"), cwd=tmp_path, start_new_session=True)
    started = time.monotonic()
    reap_group(os.getpgid(proc.pid), proc)
    assert time.monotonic() - started < GRACE_S / 2
    assert proc.returncode is not None, "reaped, not left a zombie"
