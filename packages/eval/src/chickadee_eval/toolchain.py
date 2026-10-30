"""Dependency provisioning for sealed runs.

Two requirements that appear to contradict each other:

**The sandbox has no network.** `runner.seal()` gives benchmark code a fresh network
namespace with a down loopback, so `connect()` fails at the kernel. That is the whole point:
runtime contamination was measured at 63% of one frontier model's SWE-bench Pro resolutions,
and an unsealed number is not a number.

**JavaScript needs jest, and jest is 127MB of `node_modules`.** The exercises are jest
suites; `npm install` inside the seal cannot work, and until this module existed all 49
JavaScript problems resolved to `infra_skip` — honest, correctly excluded from the
denominator, and half the benchmark missing while E1 claimed two languages.

The resolution is ordering. Dependencies are installed **before** the seal, once, into a
store outside any workspace; each problem's workspace then gets a symlink to it. The
install needs the network and does not run benchmark code; the test run runs benchmark code
and has no network. Neither ever does both.

**Why a symlink is not a hole in the seal.** The seal is a *network* namespace — `bwrap`
runs with `--dev-bind / /`, so there is no filesystem isolation to break, and a sandboxed
test already sees the whole filesystem. The property a shared store could genuinely damage
is a different one: §5.2 item 5, *"no state carries between problems within an arm."* A
writable shared `node_modules` is exactly such a channel — problem 1 writes, problem 12
executes. So the store is made read-only after installation and checked before use.

**Pinned, because "whatever npm resolved that day" is not a benchmark.** The upstream
exercises ship no lockfile and depend on `^`/`~` ranges, so two runs a month apart would
score against different jest versions with nothing recording it. `toolchains/javascript/`
carries a generated lockfile and installs with `npm ci`, which refuses to deviate from it.
`store_digest()` goes into the archive so a result can say which tree ran it.
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from chickadee_eval.manifest import Language

TOOLCHAIN_ROOT = Path(__file__).resolve().parent / "toolchains"
"""Where the pinned manifests live: **inside the importable package**, beside this module.

The first version put them at `packages/eval/toolchains/`, a sibling of `src/`, and claimed
in this very docstring that the location made the eval runnable from a wheel. It did the
opposite — `[tool.hatch.build.targets.wheel] packages = ["src/chickadee_eval"]` ships only
what is under the package directory, so an installed eval would have had the code to
provision and no manifest to provision from. Caught in review; the claim was written before
the packaging was checked."""


class ProvisionError(RuntimeError):
    """Dependencies could not be installed. Raised rather than degraded.

    The caller's alternative is to run without them, which `polyglot.run_tests` reports as
    `infra_skip` — a correct outcome that must be *chosen*, not arrived at by a failed
    install nobody noticed.
    """


@dataclass(frozen=True)
class DependencySet:
    """A pinned set of third-party packages one language needs to run its tests."""

    name: str
    manifest_dir: Path
    """Holds the committed manifest and lockfile. Copied into the store, never installed
    in place — an install writes `node_modules` and would put 127MB inside the repo."""
    install_argv: tuple[str, ...]
    """`npm ci`, not `npm install`: `ci` fails if the lockfile and manifest disagree,
    which is the difference between a pinned tree and a hopeful one."""
    binary: str
    """The executable whose absence means the language cannot be provisioned at all."""
    payload: str
    """The directory the install produces, and the one linked into each workspace."""
    manifest_files: tuple[tuple[str, str], ...]
    """`(committed name, installed name)` pairs, and the two differ on purpose.

    The repo's `.gitignore` carries bare `package.json` and `package-lock.json` entries — a
    reasonable guard against stray npm artefacts in a Python repo — which matched these and
    made `git add -A` skip them silently. Worse, hatchling honours VCS ignore rules when
    selecting files, so even force-added they were **excluded from the built wheel**: an
    installed eval had the code to provision and no manifest to provision from. Verified by
    building one and listing it.

    A `.pinned` suffix sidesteps the pattern instead of fighting it. The alternatives were a
    `.gitignore` negation (a file no task owns) or a hatch `artifacts` override (build config
    carrying a rule about a rule); renaming keeps the whole problem inside this task's own
    files, and `provision` writes the real names into the store where npm expects them."""


DEPENDENCY_SETS: dict[Language, DependencySet] = {
    Language.JAVASCRIPT: DependencySet(
        name="javascript",
        manifest_dir=TOOLCHAIN_ROOT / "javascript",
        # `--ignore-scripts` because an install runs third-party lifecycle hooks as this
        # user, outside the sandbox, with the network up — the one moment in a run where
        # arbitrary code executes unsealed. Verified that jest and babel still work without
        # them; if a future dependency needs one, that is a decision to record, not a flag
        # to flip.
        install_argv=("npm", "ci", "--silent", "--no-audit", "--no-fund", "--ignore-scripts"),
        binary="npm",
        payload="node_modules",
        manifest_files=(
            ("package.json.pinned", "package.json"),
            ("package-lock.json.pinned", "package-lock.json"),
        ),
    ),
}


@dataclass(frozen=True)
class Store:
    """A provisioned dependency tree, ready to be linked into workspaces."""

    language: Language
    path: Path
    """The `payload` directory itself, e.g. `<root>/javascript/node_modules`."""
    digest: str
    """Of the pinned manifest, not of the installed tree. Hashing 127MB per run would cost
    more than it tells: `npm ci` is a function of the lockfile, so the lockfile's digest is
    what identifies the tree, and it is what belongs in the archive."""
    packages: int


def manifest_digest(language: Language) -> str:
    """A stable id for the pinned dependency set."""
    dep = DEPENDENCY_SETS[language]
    h = hashlib.blake2b(digest_size=16)
    for committed, installed in dep.manifest_files:
        h.update(installed.encode("utf-8"))
        h.update((dep.manifest_dir / committed).read_bytes())
    return h.hexdigest()


def needs_provisioning(language: Language) -> bool:
    return language in DEPENDENCY_SETS


def _set_readonly(root: Path) -> None:
    """Drop write bits across the tree.

    §5.2 item 5 — no state carries between problems. A shared, writable `node_modules` is a
    channel from problem 1 to problem 12, and one that executes: a test that dropped a file
    in there would be imported by every later problem. Read-only turns that from a silent
    channel into a failed write.

    Not a security boundary, and not claimed as one. Two reasons, both worth stating rather
    than leaving for someone to discover:

    - `bwrap --dev-bind / /` gives benchmark code the whole filesystem regardless.
    - `unshare -rn` maps the caller to **root inside the namespace**, and root bypasses mode
      bits on files it owns. So this stops a stray write from an ordinary user and does not
      stop a determined one from inside the sandbox.

    It is a correctness boundary against accidents — a test dropping a file into
    `node_modules` where every later problem would import it — which is the failure that
    actually happens. Making it hold against the sandbox too means a read-only bind mount
    (`bwrap --ro-bind`), which is available only on the container backend and belongs with a
    decision about what `wrap()` promises.
    """
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                continue
            path.chmod(stat.S_IMODE(mode) & ~0o222)
        except OSError:
            # A tree that cannot be fully locked down is reported by `is_readonly`, not
            # silently accepted here.
            continue
    with contextlib.suppress(OSError):
        root.chmod(stat.S_IMODE(root.lstat().st_mode) & ~0o222)


def _set_writable(root: Path) -> None:
    """Restore write bits so the store can be replaced or removed."""
    for path in [root, *root.rglob("*")]:
        try:
            mode = path.lstat().st_mode
            if not stat.S_ISLNK(mode):
                path.chmod(stat.S_IMODE(mode) | 0o200)
        except OSError:
            continue


def is_readonly(store: Store) -> bool:
    """Whether the store's write bits are cleared. Checked before use, not just after install.

    Mode bits, deliberately, and **not** `os.access(..., W_OK)`. `os.access` answers "can
    *this* process write here", and for root the answer is yes whatever the mode says — so
    it reported a correctly locked store as writable the first time this ran, under the
    container's root user. The question worth asking is about the store, not about the
    caller.
    """
    mode = store.path.lstat().st_mode
    return not (stat.S_IMODE(mode) & 0o222)


def provision(
    language: Language, root: Path, *, force: bool = False, timeout_s: int = 900
) -> Store:
    """Install a language's pinned dependencies under `root`. **Needs the network.**

    Call this *before* `seal()` and before any workspace exists. Idempotent: an existing
    store built from the same manifest digest is reused, because reinstalling 127MB per run
    is time spent proving something the lockfile already fixed.
    """
    if language not in DEPENDENCY_SETS:
        raise ProvisionError(f"{language} has no dependency set; nothing to provision")
    dep = DEPENDENCY_SETS[language]
    if shutil.which(dep.binary) is None:
        raise ProvisionError(
            f"{dep.binary} is not installed, so {language} cannot be provisioned. The run "
            "may continue — `polyglot.run_tests` reports these problems as infra_skip — but "
            "that decision belongs to the caller, not to a failed install."
        )

    home = root / dep.name
    payload = home / dep.payload
    digest = manifest_digest(language)
    stamp = home / ".chickadee-digest"

    if (
        not force
        and payload.is_dir()
        and stamp.is_file()
        and stamp.read_text(encoding="utf-8").strip() == digest
    ):
        return Store(language, payload, digest, _count(payload))

    if home.exists():
        _set_writable(home)
        shutil.rmtree(home)
    home.mkdir(parents=True)
    for committed, installed in dep.manifest_files:
        shutil.copyfile(dep.manifest_dir / committed, home / installed)

    try:
        proc = subprocess.run(
            list(dep.install_argv),
            cwd=home,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProvisionError(f"{dep.name}: install did not complete: {exc}") from exc
    if proc.returncode != 0:
        raise ProvisionError(
            f"{dep.name}: {' '.join(dep.install_argv)} exited {proc.returncode}\n"
            f"{proc.stdout[-800:]}\n{proc.stderr[-800:]}"
        )
    if not payload.is_dir():
        raise ProvisionError(f"{dep.name}: install reported success but no {dep.payload}/")

    stamp.write_text(digest + "\n", encoding="utf-8")
    _set_readonly(payload)
    return Store(language, payload, digest, _count(payload))


def _count(payload: Path) -> int:
    """Installed packages, for the record. Top level plus scoped, which is what `npm ls`
    would call the tree's breadth without walking 127MB to find out."""
    total = 0
    for entry in payload.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.name.startswith("@") and entry.is_dir():
            total += sum(1 for _ in entry.iterdir())
        else:
            total += 1
    return total


def link_into(workspace: Path, store: Store) -> None:
    """Make the store's payload reachable from a workspace.

    A symlink rather than a copy. Copying 127MB per problem would add minutes to an 83
    problem run to defend a boundary that is not there — the seal is a network namespace,
    and `bwrap --dev-bind / /` gives benchmark code the whole filesystem either way. What a
    shared store could actually damage is the "no state carries between problems", and
    read-only is what answers that.
    """
    dep = DEPENDENCY_SETS[store.language]
    target = workspace / dep.payload
    if target.is_symlink() or target.exists():
        # A problem that ships its own is left alone: overwriting it would substitute a
        # dependency tree the exercise did not ask for.
        return
    target.symlink_to(store.path, target_is_directory=True)


def prepare(
    languages: set[Language], root: Path, *, timeout_s: int = 900
) -> dict[Language, Store]:
    """Provision every language in `languages` that needs it. Skips those that do not.

    Raises on failure rather than returning a partial map: a run that quietly provisioned
    one of two languages would report `infra_skip` for the other and look like a property of
    the benchmark rather than of the setup.
    """
    return {
        language: provision(language, root, timeout_s=timeout_s)
        for language in sorted(languages)
        if needs_provisioning(language)
    }
