"""`Isolation.read_only`: paths bound read-only over a writable workspace, after it,
so the box shell's commands write the worktree but never its `.git`. Every backend renders the
field after its writable binds, only when the workspace is writable; `isolation_for` passes it
through for a request whose workspace is writable; and bubblewrap, where the machine has it,
refuses the write for real."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from chickadee.executor.policy import isolation_for
from chickadee.executor.types import ExecRequest, PolicyClass
from chickadee.isolation.base import has
from chickadee.isolation.container import Container
from chickadee.isolation.linux import BINARY, bwrap_argv
from chickadee.isolation.macos import profile_for
from chickadee.isolation.types import Isolation

CWD = Path("/work/tree")
KEPT = Path("/work/tree/.git")

bwrap_only = pytest.mark.skipif(not has(BINARY), reason="bubblewrap unavailable")


def wrap(*, writable: bool) -> Isolation:
    return Isolation(argv=("true",), cwd=CWD, repo_writable=writable, read_only=(KEPT,))


def test_bubblewrap_binds_it_read_only_after_the_writable_bind() -> None:
    argv = bwrap_argv(wrap(writable=True))
    writable = argv.index("--bind")
    kept = [i for i, a in enumerate(argv) if a == "--ro-bind" and argv[i + 1] == str(KEPT)]
    assert len(kept) == 1 and kept[0] > writable
    assert str(KEPT) not in bwrap_argv(wrap(writable=False))


def test_seatbelt_denies_writes_under_it_after_the_allow() -> None:
    profile = profile_for(wrap(writable=True))
    allow = profile.index(f'(allow file-write* (subpath "{CWD}"))')
    deny = profile.index(f'(deny file-write* (subpath "{KEPT}"))')
    assert deny > allow
    assert str(KEPT) not in profile_for(wrap(writable=False))


def test_a_container_mounts_it_read_only_after_the_workspace() -> None:
    argv = Container().wrap(wrap(writable=True))
    assert argv.index(f"{KEPT}:{KEPT}:ro") > argv.index(f"{CWD}:{CWD}:rw")
    assert not any(str(KEPT) in a for a in Container().wrap(wrap(writable=False)))


def test_isolation_for_passes_it_only_where_the_workspace_is_writable() -> None:
    def request(policy: PolicyClass) -> ExecRequest:
        return ExecRequest(
            command="make", cwd=CWD, policy=policy, timeout_s=10, read_only=(KEPT,)
        )

    assert isolation_for(request(PolicyClass.WORKTREE)).read_only == (KEPT,)
    sealed = isolation_for(request(PolicyClass.ACTION))
    assert sealed.read_only == () and not sealed.repo_writable


def test_an_unenforced_class_is_given_its_also_writable_paths() -> None:
    """The box shell names the worktree root whatever the `cwd`, since every backend binds only
    the `cwd` writable."""
    request = ExecRequest(
        command="make",
        cwd=CWD / "sub",
        policy=PolicyClass.WORKTREE,
        timeout_s=10,
        also_writable=(CWD,),
    )
    assert isolation_for(request).writable == (CWD,)


@bwrap_only
def test_bwrap_keeps_it_read_only_under_a_writable_workspace(
    work: Path,
) -> None:  # pragma: no cover - platform
    kept = work / ".git"
    kept.mkdir()
    (kept / "config").write_text("[core]\n", encoding="utf-8")
    script = f"echo x > {work}/new.txt && echo y > {kept}/config"
    isolation = Isolation(
        argv=("sh", "-c", script), cwd=work, repo_writable=True, read_only=(kept,)
    )
    env = {**os.environ, "LC_ALL": "C"}
    out = subprocess.run(
        bwrap_argv(isolation), capture_output=True, text=True, check=False, env=env
    )
    assert out.returncode != 0
    assert "Read-only file system" in out.stderr
    assert (work / "new.txt").read_text(encoding="utf-8") == "x\n"
    assert (kept / "config").read_text(encoding="utf-8") == "[core]\n"
