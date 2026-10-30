"""`Isolation.writable` (design 4a): paths bound writable beside a writable `cwd`, so a
declared git operation in a linked worktree can write the repository's common git directory.
Every backend binds them only when the repository is writable, and `isolation_for` gives them
only to a request whose writes are declared."""

from __future__ import annotations

from pathlib import Path

from chickadee.executor.policy import isolation_for
from chickadee.executor.types import ExecRequest, PolicyClass
from chickadee.isolation.container import Container
from chickadee.isolation.linux import bwrap_argv
from chickadee.isolation.macos import profile_for
from chickadee.isolation.types import Isolation

CWD = Path("/work/tree")
GIT = Path("/work/repo/.git")


def wrap(*, writable: bool) -> Isolation:
    return Isolation(argv=("true",), cwd=CWD, repo_writable=writable, writable=(GIT,))


def binds(argv: tuple[str, ...], flag: str) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def test_bubblewrap_binds_the_extra_path_only_beside_a_writable_cwd() -> None:
    assert binds(bwrap_argv(wrap(writable=True)), "--bind") == [str(CWD), str(GIT)]
    assert str(GIT) not in bwrap_argv(wrap(writable=False))


def test_seatbelt_allows_writes_under_the_extra_path_only_beside_a_writable_cwd() -> None:
    assert f'(allow file-write* (subpath "{GIT}"))' in profile_for(wrap(writable=True))
    assert str(GIT) not in profile_for(wrap(writable=False))


def test_a_container_mounts_the_extra_path_only_beside_a_writable_cwd() -> None:
    assert f"{GIT}:{GIT}:rw" in Container().wrap(wrap(writable=True))
    assert not any(str(GIT) in a for a in Container().wrap(wrap(writable=False)))


def test_only_a_declared_write_is_given_the_extra_path() -> None:
    def request(effects: tuple[str, ...]) -> ExecRequest:
        return ExecRequest(
            command="git switch -c x",
            cwd=CWD,
            policy=PolicyClass.ACTION,
            timeout_s=10,
            effects=effects,
            also_writable=(GIT,),
        )

    assert isolation_for(request(("**",))).writable == (GIT,)
    sealed = isolation_for(request(()))
    assert sealed.writable == () and not sealed.repo_writable
