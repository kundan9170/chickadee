"""bubblewrap argv construction, and a real denial where bwrap exists."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from repos import git

from chickadee.isolation import base, linux
from chickadee.isolation.base import Probe, forget_probes, has, probe, select
from chickadee.isolation.linux import BINARY, Bubblewrap, bwrap_argv
from chickadee.isolation.types import Isolation, SandboxLevel, mask_entries
from chickadee.process import Completed, ProcessSupervisor

from .isolation_helpers import RUNS, fake_binary, iso_for

bwrap_only = pytest.mark.skipif(not has(BINARY), reason="bubblewrap unavailable")


def test_read_only_wrap_binds_the_root_read_only(work: Path) -> None:
    argv = bwrap_argv(iso_for(work, ("echo", "x"), repo_writable=False))
    assert "--ro-bind" in argv
    assert f"--bind {work} {work}" not in " ".join(argv)


def test_writable_wrap_rebinds_the_worktree(work: Path) -> None:
    argv = " ".join(bwrap_argv(iso_for(work, ("echo", "x"), repo_writable=True)))
    assert f"--bind {work} {work}" in argv


def test_read_only_wrap_rebinds_the_worktree_after_the_tmpfs(work: Path) -> None:
    """A worktree under `/tmp` is not hidden by the private tmpfs mounted over it."""
    argv = list(bwrap_argv(iso_for(work, ("echo", "x"), repo_writable=False)))
    bound = argv.index(str(work))
    assert argv[bound - 1] == "--ro-bind" and argv[bound + 1] == str(work)
    assert argv.index("--tmpfs") < bound


@pytest.mark.parametrize("cwd", ["/tmp", "/"])
def test_a_read_only_wrap_never_binds_the_host_tmp_back(cwd: str) -> None:
    """The private tmpfs is the point: a cwd of `/tmp` or `/` is not bound back over it."""
    argv = bwrap_argv(Isolation(argv=("true",), cwd=Path(cwd), repo_writable=False))
    tmpfs = argv.index("--tmpfs")
    assert "--ro-bind" not in argv[tmpfs:], argv


def test_proc_and_dev_are_mounted_after_the_root_bind(work: Path) -> None:
    """G3 F1: bubblewrap applies mounts in order, so a `/dev` mounted before `--ro-bind / /`
    is covered by the host's, bound read-only and `nodev`: no device opens, and a sandboxed
    `git` fails to open `/dev/null`. Mounted after it, the wrap has its own."""
    argv = list(bwrap_argv(iso_for(work, ("echo", "x"), repo_writable=False)))
    root = next(i for i in range(len(argv) - 2) if argv[i : i + 3] == ["--ro-bind", "/", "/"])
    assert root < argv.index("--proc") and root < argv.index("--dev"), argv


def test_network_is_unshared_when_off(work: Path) -> None:
    assert "--unshare-net" in bwrap_argv(iso_for(work, ("echo", "x")))


def test_the_level_is_native() -> None:
    assert Bubblewrap().level() is SandboxLevel.NATIVE


@bwrap_only
def test_bwrap_denies_a_real_repo_write(work: Path) -> None:  # pragma: no cover - platform
    isolation = iso_for(work, ("sh", "-c", f"echo x > {work}/denied.txt"), repo_writable=False)
    # The C locale, so the shell's error is strerror's English whatever the machine's.
    env = {**os.environ, "LC_ALL": "C"}
    out = subprocess.run(
        bwrap_argv(isolation), capture_output=True, text=True, check=False, env=env
    )
    assert out.returncode != 0
    assert "Read-only file system" in out.stderr, "refused, not a missing directory"
    assert not (work / "denied.txt").exists()


@bwrap_only
def test_bwrap_read_only_runs_git_and_writes_dev_null(work: Path) -> None:  # pragma: no cover
    """G3 F1: gate A3's `git status` in a read-only wrap opens `/dev/null`; before the fix
    every patch escalated as infrastructure."""
    git(work, "init", "-q")
    script = "git status --porcelain >/dev/null && echo x > /dev/null && echo ok"
    isolation = iso_for(work, ("sh", "-c", script), repo_writable=False)
    env = {**os.environ, "LC_ALL": "C"}
    out = subprocess.run(
        bwrap_argv(isolation), capture_output=True, text=True, check=False, env=env
    )
    assert out.returncode == 0 and out.stdout == "ok\n", out.stderr


@bwrap_only
def test_bwrap_read_only_sees_the_worktree(work: Path) -> None:  # pragma: no cover - platform
    (work / "seen.txt").write_text("here\n")
    isolation = iso_for(work, ("cat", str(work / "seen.txt")), repo_writable=False)
    out = subprocess.run(bwrap_argv(isolation), capture_output=True, text=True, check=False)
    assert out.returncode == 0 and out.stdout == "here\n"


# --- D91: the mask, as argv --------------------------------------------------------


def test_deny_read_ro_binds_a_mode_000_entry_over_each_path(
    work: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    secret = work / ".env"
    secret.write_text("x")
    keys = work / ".ssh"
    keys.mkdir()
    argv = bwrap_argv(Isolation(argv=("cat",), cwd=work, deny_read=(secret, keys)))
    file_, dir_ = mask_entries()
    joined = " ".join(argv)
    assert f"--ro-bind {file_} {secret}" in joined, "a file gets the file mask"
    assert f"--ro-bind {dir_} {keys}" in joined, "a directory gets the directory mask"
    if sys.platform != "win32":  # mode bits are POSIX; the argv shape is not
        assert (file_.stat().st_mode & 0o777) == 0


def test_the_mask_comes_after_the_writable_bind(
    work: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A secret inside the writable worktree must still be masked: the later bind wins."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    secret = work / ".env"
    secret.write_text("x")
    argv = list(
        bwrap_argv(Isolation(argv=("cat",), cwd=work, repo_writable=True, deny_read=(secret,)))
    )
    assert argv.index("--bind") < argv.index(str(secret))


def test_no_deny_read_no_mask(work: Path) -> None:
    argv = bwrap_argv(iso_for(work, ("echo", "x")))
    assert "mask-file" not in " ".join(argv) and "mask-dir" not in " ".join(argv)


# -- the probe -------------------------------------------------------------

USERNS = 'echo "bwrap: setting up uid map: Permission denied" >&2\nexit 1\n'


@pytest.fixture
def fresh_probes() -> Iterator[None]:
    forget_probes()
    yield
    forget_probes()


def _path_with(monkeypatch: pytest.MonkeyPatch, bin_dir: Path) -> None:
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")


def _path_without_bwrap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A PATH holding `true` and `sh`, and no `bwrap`, wherever this runs."""
    empty = tmp_path / "nobwrap"
    empty.mkdir()
    for tool in ("true", "sh"):
        found = shutil.which(tool)
        assert found is not None
        (empty / tool).symlink_to(found)
    monkeypatch.setenv("PATH", str(empty))


def test_a_bwrap_that_runs_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_probes: None
) -> None:
    _path_with(monkeypatch, tmp_path / "bin")
    fake_binary(tmp_path / "bin", BINARY, RUNS)
    found = probe(Bubblewrap())
    assert found == Probe(SandboxLevel.NATIVE, ran=True)
    assert Bubblewrap().available()
    assert isinstance(select(), Bubblewrap)


def test_a_bwrap_on_the_path_that_cannot_run_is_not_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_probes: None
) -> None:
    """Ubuntu 24.04: the binary is there and AppArmor refuses it a user namespace."""
    _path_with(monkeypatch, tmp_path / "bin")
    fake_binary(tmp_path / "bin", BINARY, USERNS)
    found = probe(Bubblewrap())
    assert found.level is SandboxLevel.NATIVE and not found.ran
    assert "user namespace" in found.reason
    assert "setting up uid map: Permission denied" in found.reason
    assert "user namespaces" in found.fix
    assert not Bubblewrap().available()
    assert select().level() is SandboxLevel.NONE


def test_the_fix_names_apparmor_and_what_lifting_it_means(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_probes: None
) -> None:
    sysctl = tmp_path / "proc_sys"
    (sysctl / "kernel").mkdir(parents=True)
    (sysctl / "kernel" / "apparmor_restrict_unprivileged_userns").write_text("1\n")
    monkeypatch.setattr(linux, "PROC_SYS", sysctl)
    _path_with(monkeypatch, tmp_path / "bin")
    fake_binary(tmp_path / "bin", BINARY, USERNS)
    fix = probe(Bubblewrap()).fix
    assert "AppArmor" in fix
    assert "kernel.apparmor_restrict_unprivileged_userns" in fix
    assert "every program on this machine" in fix


def test_with_no_bwrap_the_result_is_none_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_probes: None
) -> None:
    _path_without_bwrap(monkeypatch, tmp_path)
    found = probe(Bubblewrap())
    assert not found.ran
    assert "bwrap is not on the PATH" in found.reason
    assert "bubblewrap" in found.fix
    assert select().level() is SandboxLevel.NONE
    assert probe(select()) == Probe(SandboxLevel.NONE, ran=True)


def test_the_probe_runs_once_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_probes: None
) -> None:
    _path_with(monkeypatch, tmp_path / "bin")
    count = tmp_path / "count"
    fake_binary(tmp_path / "bin", BINARY, f'echo x >> "{count}"\n' + RUNS)
    for _ in range(3):
        assert Bubblewrap().available()
    assert count.read_text().count("x") == 1


def test_the_probe_runs_true_under_the_backend_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_probes: None
) -> None:
    """It reports; the person decides. One spawn, through the supervisor:
    `true` under the backend, read-only and network off. Nothing installed, nothing
    privileged, no setting changed."""
    _path_with(monkeypatch, tmp_path / "bin")
    fake_binary(tmp_path / "bin", BINARY, USERNS)
    spawned: list[tuple[str, ...]] = []
    real = ProcessSupervisor.run

    def record(self: ProcessSupervisor, argv: tuple[str, ...], **kw: Any) -> Completed:
        spawned.append(argv)
        return real(self, argv, **kw)

    monkeypatch.setattr(ProcessSupervisor, "run", record)
    probe(Bubblewrap())
    assert len(spawned) == 1
    argv = spawned[0]
    assert argv[0] == BINARY and argv[-2:] == ("--", "true")
    assert "--unshare-net" in argv and "--bind" not in argv


def test_a_probe_that_timed_out_is_asked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_probes: None
) -> None:
    """A slow start is not an answer: caching it would leave the whole process unsandboxed
    after one busy moment. Only a stable answer (absent, refused, not enabled) is kept."""
    monkeypatch.setattr(base, "PROBE_TIMEOUT_S", 1)
    _path_with(monkeypatch, tmp_path / "bin")
    fake_binary(tmp_path / "bin", BINARY, "sleep 30\n")
    slow = probe(Bubblewrap())
    assert not slow.ran and "did not finish" in slow.reason
    fake_binary(tmp_path / "bin", BINARY, RUNS)
    assert probe(Bubblewrap()).ran
