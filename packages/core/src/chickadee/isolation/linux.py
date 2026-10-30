"""bubblewrap argv construction, the approach Claude Code and Codex CLI also use."""

from __future__ import annotations

import shutil
from pathlib import Path

from chickadee.isolation.base import first_line, has, probe
from chickadee.isolation.types import Isolation, SandboxLevel, mask_for
from chickadee.paths import within
from chickadee.process import Completed

BINARY = "bwrap"

_TMP = Path("/tmp")


def bwrap_argv(isolation: Isolation) -> tuple[str, ...]:
    argv: list[str] = [BINARY, "--die-with-parent"]
    # Read-only bind of the root, then re-bind the writable set on top. Building it the
    # other way — binding only what is needed — breaks every toolchain that reads a config
    # from a path nobody predicted.
    argv += ["--ro-bind", "/", "/"]
    # A `/proc` and a minimal `/dev` of the wrap's own, mounted AFTER the root bind (G3 F1):
    # bubblewrap applies mounts in order, so mounted before it they are covered by the host's,
    # bound read-only and `nodev`, where no device opens and a sandboxed `git` cannot open
    # `/dev/null`.
    argv += ["--proc", "/proc", "--dev", "/dev"]
    # Never the host `/tmp`: binding the real one lets a sandboxed process modify another
    # process's temporary files. The session's own directory where there is one, else a
    # private tmpfs per command.
    if isolation.tmp is not None:
        # The session's own directory over `/tmp`, so what one command leaves
        # there the next finds; still never the host's `/tmp`. Bound at its own path too,
        # which `TMPDIR` names.
        argv += ["--bind", str(isolation.tmp), str(_TMP)]
        argv += ["--bind", str(isolation.tmp), str(isolation.tmp)]
    else:
        argv += ["--tmpfs", str(_TMP)]
    # The harness's scratch, writable whatever the wrap: a package manager's
    # downloads, never the person's state.
    for scratch in isolation.scratch:
        argv += ["--bind", str(scratch), str(scratch)]
    if isolation.repo_writable:
        argv += ["--bind", str(isolation.cwd), str(isolation.cwd)]
        for extra in isolation.writable:
            argv += ["--bind", str(extra), str(extra)]
        # After the writable binds, so each wins over the bind it lies under.
        for kept in isolation.read_only:
            argv += ["--ro-bind", str(kept), str(kept)]
    elif _below_tmp(isolation.cwd):
        # A read-only worktree under `/tmp` is bound back over the tmpfs, or it would be an
        # empty directory: a search there finds nothing, and a refused write looks refused
        # only because its directory is missing. Only strictly below `/tmp`: a cwd of `/tmp`
        # or `/` would bind the host's `/tmp` back over the private one.
        argv += ["--ro-bind", str(isolation.cwd), str(isolation.cwd)]
    # A mode-000 scratch entry bound read-only OVER each denied path, after the
    # root and the worktree's bind, either kind, so it wins. A read is EACCES, `ls` of a
    # masked directory is EACCES, and `chmod` inside is EROFS because the bind is read-only —
    # a permission error every time, never an empty file. Ordered after the worktree bind
    # deliberately: a secret inside the worktree must still be masked.
    for denied in isolation.deny_read:
        argv += ["--ro-bind", str(mask_for(denied)), str(denied)]
    if isolation.network_off:
        argv += ["--unshare-net"]
    # No capabilities inside the wrap, for anyone. bwrap already drops them for a
    # normal user; run as root it keeps them all, and root with CAP_SYS_ADMIN unmounts the
    # mask above, or binds the worktree elsewhere, and reads the secret under it. Without
    # capabilities root obeys modes like everyone else, so a masked read is a permission
    # error for root too.
    argv += ["--cap-drop", "ALL"]
    argv += ["--chdir", str(isolation.cwd), "--"]
    return (*argv, *isolation.argv)


def _below_tmp(cwd: Path) -> bool:
    """Strictly below `/tmp`, by `paths.within`: `/tmp` itself is not."""
    landed = within(_TMP, cwd)
    return landed is not None and landed != within(_TMP, _TMP)


class Bubblewrap:
    def available(self) -> bool:
        """It runs, not that it is on the path."""
        return probe(self).ran

    def level(self) -> SandboxLevel:
        return SandboxLevel.NATIVE

    def wrap(self, isolation: Isolation) -> tuple[str, ...]:
        return bwrap_argv(isolation)

    # -- what the probe asks -------------------------------------------------------

    def probe_key(self) -> tuple[str, ...]:
        return (shutil.which(BINARY) or "",)

    def not_ready(self) -> tuple[str, str] | None:
        if has(BINARY):
            return None
        return f"{BINARY} is not on the PATH", INSTALL

    def diagnose(self, done: Completed) -> tuple[str, str]:
        said = first_line(done)
        if any(sign in said for sign in _USERNS_SIGNS):
            return f"{BINARY} cannot create a user namespace: {said}", _userns_fix()
        return (
            f"{BINARY} exited {done.exit_code}: {said}",
            f"run `{BINARY} --ro-bind / / true` to see the error",
        )


INSTALL = (
    "install the bubblewrap package: `sudo apt install bubblewrap` (Debian, Ubuntu), "
    "`sudo dnf install bubblewrap` (Fedora), `sudo pacman -S bubblewrap` (Arch)"
)
"""Named for the person to run. Chickadee installs nothing."""

PROC_SYS = Path("/proc/sys")
"""Where the kernel's user-namespace settings are read. Read, never written."""

_USERNS_SIGNS = (
    "uid map",
    "user namespace",
    "new namespace",
    "No permissions to creat",
    "Operation not permitted",
)
"""What bwrap prints when the kernel refuses it a user namespace, across its versions."""

_LIFTING = (
    "Lifting it lets every program on this machine create user namespaces, which widens "
    "the kernel surface any local process can reach; that is the machine owner's call."
)


def _userns_fix() -> str:
    """Which setting blocks it, read from the kernel, and what changing it means."""
    if _sysctl("kernel/apparmor_restrict_unprivileged_userns") == "1":
        return (
            "AppArmor restricts unprivileged user namespaces (Ubuntu 23.10 and later). Either "
            "give bwrap an AppArmor profile that allows them (`/etc/apparmor.d/bwrap`, with "
            "`userns,`), or set kernel.apparmor_restrict_unprivileged_userns=0. " + _LIFTING
        )
    if _sysctl("kernel/unprivileged_userns_clone") == "0":
        return (
            "the kernel disables unprivileged user namespaces: set "
            "kernel.unprivileged_userns_clone=1. " + _LIFTING
        )
    if _sysctl("user/max_user_namespaces") == "0":
        return (
            "the kernel allows no user namespaces: raise user.max_user_namespaces. " + _LIFTING
        )
    return (
        "the kernel or a container runtime refuses unprivileged user namespaces; inside a "
        "container, run it with user namespaces allowed. " + _LIFTING
    )


def _sysctl(name: str) -> str | None:
    try:
        return (PROC_SYS / name).read_text(encoding="utf-8").strip()
    except OSError:
        return None
