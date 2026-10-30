"""The mask property, end to end.

A wrapped process that opens a masked path gets a **permission error** and not one byte of
the file. `ls` of a masked directory fails the same way; `chmod` on the mask is refused
because the bind is read-only; an unmasked sibling still reads; `deny_read=()` masks
nothing. Skipped, never faked, where no backend can enforce — a green test on a machine
that could not enforce is the claim EX-2 exists to prevent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from chickadee.isolation.base import has
from chickadee.isolation.linux import BINARY, bwrap_argv
from chickadee.isolation.none import reads_unmasked
from chickadee.isolation.types import Isolation, SandboxLevel, mask_entries

bwrap_only = pytest.mark.skipif(not has(BINARY), reason="bubblewrap unavailable")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="mode bits are POSIX")

SECRET = "AKIA-not-a-real-key-but-must-never-appear"


def _run(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _iso(work: Path, argv: tuple[str, ...], deny: tuple[Path, ...]) -> Isolation:
    return Isolation(argv=argv, cwd=work, network_off=True, repo_writable=True, deny_read=deny)


def _bwrap_works(work: Path) -> bool:
    """Some hosts have the binary and no user namespaces (CI containers, some WSL
    kernels). Then even an unmasked wrap fails, and a masked one failing proves nothing."""
    return _run(bwrap_argv(_iso(work, ("true",), ()))).returncode == 0


@pytest.fixture
def secret_tree(work: Path) -> tuple[Path, Path, Path]:
    env = work / ".env"
    env.write_text(f"KEY={SECRET}\n")
    ssh = work / ".ssh"
    ssh.mkdir()
    (ssh / "id_rsa").write_text(SECRET)
    sibling = work / "README"
    sibling.write_text("readable\n")
    return env, ssh, sibling


@posix_only
def test_mask_entries_are_mode_000_under_the_cache_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    file_, dir_ = mask_entries()
    assert file_.is_file() and dir_.is_dir()
    # Under the cache root the harness owns — never a bare `/tmp/...`, which the wrap
    # replaces with a private tmpfs. (pytest's own tmp_path sits under /tmp, so the
    # assertion is about the root we set, not the literal prefix.)
    assert file_.parent == tmp_path / "chickadee" / "mask"
    assert (file_.stat().st_mode & 0o777) == 0
    assert (dir_.stat().st_mode & 0o777) == 0
    # Re-applied every call: an outside chmod must not silently un-mask.
    file_.chmod(0o644)
    mask_entries()
    assert (file_.stat().st_mode & 0o777) == 0


def test_reads_unmasked_is_a_fact_about_the_wrap(work: Path) -> None:
    masked = Isolation(argv=("x",), cwd=work, deny_read=(work / ".env",))
    assert reads_unmasked(SandboxLevel.NONE, masked)
    assert not reads_unmasked(SandboxLevel.NATIVE, masked)
    assert not reads_unmasked(SandboxLevel.NONE, Isolation(argv=("x",), cwd=work))


@bwrap_only
def test_a_masked_file_is_a_permission_error_and_no_bytes(  # pragma: no cover - platform
    work: Path, secret_tree: tuple[Path, Path, Path]
) -> None:
    env, _, _ = secret_tree
    if not _bwrap_works(work):
        pytest.skip("bubblewrap present but cannot create a user namespace here")
    out = _run(bwrap_argv(_iso(work, ("cat", str(env)), (env,))))
    assert out.returncode != 0
    assert "ermission denied" in out.stderr
    assert SECRET not in out.stdout
    assert out.stdout == ""


@bwrap_only
def test_a_masked_directory_fails_the_same_way(  # pragma: no cover - platform
    work: Path, secret_tree: tuple[Path, Path, Path]
) -> None:
    _, ssh, _ = secret_tree
    if not _bwrap_works(work):
        pytest.skip("bubblewrap present but cannot create a user namespace here")
    listing = _run(bwrap_argv(_iso(work, ("ls", str(ssh)), (ssh,))))
    assert listing.returncode != 0 and "ermission denied" in listing.stderr
    assert "id_rsa" not in listing.stdout
    read = _run(bwrap_argv(_iso(work, ("cat", str(ssh / "id_rsa")), (ssh,))))
    assert read.returncode != 0 and SECRET not in read.stdout


@bwrap_only
def test_the_mask_cannot_be_chmodded_away_from_inside(  # pragma: no cover - platform
    work: Path, secret_tree: tuple[Path, Path, Path]
) -> None:
    env, _, _ = secret_tree
    if not _bwrap_works(work):
        pytest.skip("bubblewrap present but cannot create a user namespace here")
    out = _run(bwrap_argv(_iso(work, ("sh", "-c", f"chmod 644 {env} && cat {env}"), (env,))))
    assert out.returncode != 0
    assert SECRET not in out.stdout


@bwrap_only
def test_an_unmasked_sibling_still_reads(  # pragma: no cover - platform
    work: Path, secret_tree: tuple[Path, Path, Path]
) -> None:
    env, _, sibling = secret_tree
    if not _bwrap_works(work):
        pytest.skip("bubblewrap present but cannot create a user namespace here")
    out = _run(bwrap_argv(_iso(work, ("cat", str(sibling)), (env,))))
    assert out.returncode == 0 and out.stdout == "readable\n"


@bwrap_only
def test_an_empty_deny_read_masks_nothing(  # pragma: no cover - platform
    work: Path, secret_tree: tuple[Path, Path, Path]
) -> None:
    env, _, _ = secret_tree
    if not _bwrap_works(work):
        pytest.skip("bubblewrap present but cannot create a user namespace here")
    out = _run(bwrap_argv(_iso(work, ("cat", str(env)), ())))
    assert out.returncode == 0 and SECRET in out.stdout


def test_the_wrap_drops_every_capability(work: Path) -> None:
    """Without it, root inside the wrap keeps `CAP_SYS_ADMIN` and unmounts the mask."""
    argv = bwrap_argv(_iso(work, ("true",), (work / ".env",)))
    i = argv.index("--cap-drop")
    assert argv[i + 1] == "ALL"
    assert i < argv.index("--"), "a bwrap option, not part of the wrapped command"


@bwrap_only
def test_the_cover_cannot_be_removed_from_inside(  # pragma: no cover - platform
    work: Path, secret_tree: tuple[Path, Path, Path]
) -> None:
    """D173, as whoever runs the suite, root included: unmounting the mask fails, binding the
    worktree elsewhere to read under the mask fails, and the secret never appears."""
    env, _, _ = secret_tree
    if not _bwrap_works(work):
        pytest.skip("bubblewrap present but cannot create a user namespace here")
    other = work / "under"
    script = (
        f"umount {env}; echo umount=$?; "
        f"mkdir -p {other} && mount --bind {work} {other}; echo bind=$?; "
        f"cat {env}; cat {other}/.env"
    )
    out = _run(bwrap_argv(_iso(work, ("sh", "-c", script), (env,))))
    assert "umount=0" not in out.stdout
    assert "bind=0" not in out.stdout
    assert SECRET not in out.stdout and SECRET not in out.stderr
