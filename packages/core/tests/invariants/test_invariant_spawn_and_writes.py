"""Invariants 1 and 2, asserted over the whole of `core` rather than one package.

The project's invariants make two monopoly claims about the repository: one module owns
model-authored writes,
and `ProcessSupervisor` is the only spawn. Both are stated repository-wide and neither is
checked repository-wide. `tests/executor/test_no_popen.py` asserts the spawn half for
`executor/` alone, and the fs-watchdog enforces
the mutation half dynamically — which sees only the import graph a given test happened to
execute, so a lazy import inside an error path passes on every run until the day it does
not. Parsing every source file sees all of them, which is the argument `test_boundaries.py`
already makes for the packaging rule.

**These are allow-lists, not patterns.** Every module permitted to spawn or to write is
named here with the reason it is permitted, so a new one fails this file and gets read by a
person. A prefix or a glob would be a standing licence — the same objection `tasks/GRANTS.toml`
makes to granting a directory instead of a file.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

import chickadee

CORE_SRC = pathlib.Path(chickadee.__file__).resolve().parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


def _load_resolver() -> ModuleType:
    """The alias resolver, shared with the storage guard (`scripts/guards/_resolve.py`), loaded
    by file path as `test_guard_runner.py` loads the runner: `scripts/`
    is not a package core imports."""
    path = REPO_ROOT / "scripts" / "guards" / "_resolve.py"
    spec = importlib.util.spec_from_file_location("_guard_resolve", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_resolve = _load_resolver()


def _sources() -> list[pathlib.Path]:
    return sorted(p for p in CORE_SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(CORE_SRC).as_posix()


def test_sources_were_found() -> None:
    """A glob that silently matches nothing turns every assertion below into a no-op."""
    assert len(_sources()) > 50


# --- invariant 2: ProcessSupervisor is the only spawn --------

POPEN_OWNER = "process/supervisor.py"
"""`chickadee.process` holds the only `Popen`. Isolation wraps, Executor decides
policy, and neither spawns."""

SUBPROCESS_RUN_OK = {
    "process/supervisor.py": "the spawn implementation itself",
    "handoff.py": "no longer applicable — handoff goes through ProcessSupervisor (T-2.11)",
}
"""Modules permitted to call `subprocess.run`, each with the reason.

`subprocess.run` is not `Popen` and D70's letter is about `Popen`, but an unexamined
`subprocess` import is how the letter gets lost. Listing them by name means the set can
only grow through a review that has to write down why.

T-2.24 drained workspace, applier, reconcile, detect and toolchain through
`run_git`/`run_harness`. They are not listed: a reintroduced `import subprocess` must
fail this file rather than hide behind a stale reason.

`handoff.py` is listed with a reason that says it does not apply: T-2.11 routed every git
call through `ProcessSupervisor`, so if it ever reappears here that is a regression, and
the entry documents the expectation rather than blessing the module.
"""


_bindings = _resolve.bindings
_is_write_open = _resolve.is_write_open


def _calls(tree: ast.Module, module: str, attr: str) -> bool:
    """Whether this module calls `module.attr` under any name it bound for it."""
    return bool(_resolve.call_sites(tree, module, attr))


def _calls_popen(tree: ast.Module) -> bool:
    return _calls(tree, "subprocess", "Popen")


def _imports_subprocess(tree: ast.Module) -> bool:
    """An import, not the word. `handoff.py`'s docstring explains why it does *not* spawn,
    and a text match would read that sentence as the violation it describes."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            return True
        if not isinstance(node, ast.Import):
            continue
        names = [alias.name for alias in node.names]
        if any(n == "subprocess" or n.startswith("subprocess.") for n in names):
            return True
    return False


def test_popen_exists_in_exactly_one_module() -> None:
    """One spawn implementation, two callers. Not 'few' — one."""
    holders = [
        _rel(p) for p in _sources() if _calls_popen(ast.parse(p.read_text(encoding="utf-8")))
    ]
    assert holders == [POPEN_OWNER], (
        f"`Popen(` must appear only in {POPEN_OWNER} (D70). Found in: {holders}. "
        "Isolation wraps and Executor decides policy; neither may spawn."
    )


@pytest.mark.parametrize("path", _sources(), ids=_rel)
def test_subprocess_users_are_named_and_justified(path: pathlib.Path) -> None:
    rel = _rel(path)
    if not _imports_subprocess(ast.parse(path.read_text(encoding="utf-8"))):
        return
    assert rel in SUBPROCESS_RUN_OK, (
        f"{rel} reaches for `subprocess` and is not in SUBPROCESS_RUN_OK. Harness-owned "
        "processes go through `ProcessSupervisor` (D69); LLM-proposed commands go through "
        "`Executor.run`. If this call is genuinely neither, add it here with its reason so "
        "the exception is reviewed rather than inherited."
    )


def _imports_name_from(tree: ast.Module, module: str, name: str) -> bool:
    """Whether `name` is imported *from `module`* — the module is the load-bearing half.

    Dropping it meant `from unrelated import ProcessSupervisor` satisfied the assertion, so
    the test would have accepted a look-alike bound from anywhere.
    """
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == module
        and any(a.name == name for a in node.names)
        for node in ast.walk(tree)
    )


def test_handoff_no_longer_spawns_directly() -> None:
    """T-2.11 drained it. Asserted rather than assumed, because the allow-list entry above
    would otherwise let it silently come back.

    Checked through the AST rather than by substring. `handoff.py`'s module docstring
    discusses `ProcessSupervisor` at length, so `"ProcessSupervisor" in text` stays true
    after the import is deleted — the same mistake this file already made once with
    `subprocess`, which is exactly why it is worth not making twice.
    """
    tree = ast.parse((CORE_SRC / "handoff.py").read_text(encoding="utf-8"))
    assert not _imports_subprocess(tree)
    assert _imports_name_from(tree, "chickadee.process", "ProcessSupervisor"), (
        "handoff.py no longer imports ProcessSupervisor. Either it stopped spawning "
        "entirely, or it found another way to — check which before editing this test."
    )


# --- invariant 1: the edit port owns model-authored writes ------------------

WRITERS = {
    "box/edits.py": (
        "the box's edit port (D196 §10, T-3.155): the task loop's one home of model-authored "
        "writes to the worktree, each inside a recorded step. The Applier it replaced retired "
        "with the packet loop (T-3.157)"
    ),
    "durable/blobs.py": (
        "the one content-addressed blob store (D149, DU-5); like `files.py` it writes for its "
        "callers, each listed here for what it stores"
    ),
    "durable/log.py": (
        "the one append-only log (D149, DU-7); like `files.py` it writes for its callers, "
        "each listed here for the log it keeps"
    ),
    "durable/files.py": (
        "the one durable-write primitive (D149): it writes nothing of its own choosing, so "
        "every module that calls it is listed here for what it writes (DU-9)"
    ),
    "journal/writer.py": "the Journal is the source of truth and owns its own file",
    "journal/paths.py": "creates the run's journal directory under .chickadee/, not the tree",
    "journal/gc.py": (
        "unlinks bulk blobs a person asked to prune and appends gc.log (D111): harness "
        "bookkeeping under the repo-local root, never the tree, never the journal itself"
    ),
    "init/lease.py": "creates the lease file's parent directory; the WRITE lease itself (D39)",
    "zeus/jobs.py": (
        "a job's output directory under the session's own, and the removal of a command's two "
        "output files once its result is in blobs (D201, T-3.161): the child writes the files, "
        "the harness never writes content into them"
    ),
    "graph/store.py": "the index the packer reads; harness bookkeeping, never model output",
    "workspace.py": "worktree setup and pre-run snapshots (D39)",
    "git_guard.py": (
        "git's own config and hooks, put back byte for byte as a command found them (T-3.144, "
        "moved from zeus/writes.py by T-3.178): the harness's restore, never content a model "
        "chose"
    ),
    "init/publish.py": "generated config at init, before any model has run",
    "secrets/walk_memo.py": (
        "the deny-list walk's memo under the harness's cache (T-3.224): file names it judged, "
        "never their content, and never a model's"
    ),
    "init/prepare.py": (
        "the exclude file of a repository Chickadee made where there was none (T-3.224): "
        "the disposable directories and the secret paths it left out, never a model's"
    ),
    "init/servers.py": (
        "creates Chickadee's servers directory before an install the person said yes to "
        "(`chickadee setup`, T-3.224); the installer writes it, never a model"
    ),
    "schema/export.py": "committed goldens, developer-invoked",
    "box/compose.py": (
        "builds a run's preimage store over the run's own blobs/ and operations/ (D160); "
        "the edit port writes pre-images through it, never model content into the tree"
    ),
    "goldens.py": "committed goldens of the contracts a part owns, developer-invoked (D54)",
    "isolation/types.py": (
        "the deny-read mask's two mode-000 scratch entries under the harness cache (D91): "
        "contentless, never model-authored, never in the worktree — the mode is the content"
    ),
    "context/store.py": (
        "the session log and its content-addressed blobs under the repo-local root "
        "(28-context §5.1): like the Journal it records model turns, never into the tree"
    ),
    "work.py": (
        "B7's one-time notice marker under the repo-local root (D204): harness bookkeeping, "
        "never the tree, never model output"
    ),
    "handoff.py": (
        "the index row's text as a blob file in a temporary directory outside every tree, "
        "for `git hash-object` (D204 B1): harness bookkeeping, never model output"
    ),
    "work_records.py": (
        "the index on the person's branch at landing (D204 B1), and B7's "
        "private copy under the repo-local root: harness projections of the journal and the "
        "landed folders; a work's README and report.md go through the edit port instead"
    ),
    "scout/hints.py": (
        "snapshot_hints.json, the per-repository escape counts SC-8 promotes on "
        "(14-scout §6): harness bookkeeping at a path its caller names, never the tree"
    ),
    "external/fetch/lightpanda.py": (
        "the `lightpanda` engine's throwaway home, one empty directory per fetch under the "
        "harness cache, removed when the fetch ends (D232 decision 6): never the tree, never "
        "model output"
    ),
    "approval/grants.py": (
        "a person's *always for this repo*: one `[[approval.grants]]` line in "
        "`.chickadee/approval.toml` and this clone's `trusted.toml` (D106 §3, T-4.12). "
        "Rendered from the router's binding, never model-authored"
    ),
    "state/local.py": (
        "this clone's `work.toml`, the work folder the person chose for this repository when "
        "the configured one could not be used (D230 Amendment 3, T-3.215): one line, on the "
        "person's yes to the harness's question, never model-authored"
    ),
    "state/user.py": (
        "a frontend's own file of the person's, beside their `config.toml` "
        "(`write_user_file`): the TUI's `tui.toml`, its full-screen choice (T-5.28), rendered "
        "by the frontend from the person's toggle, never model-authored"
    ),
}
"""Every module in `core` that writes to disk, with what it writes.

Invariant 1 is narrower than "only the edit port writes files", and the watchdog's own
docstring says so: it is about *the source tree a run mutates*, not the harness's own
bookkeeping. The Journal must write the Journal; the indexer must write the index. What the
invariant forbids is a second path by which **model-authored content** reaches the worktree.

So this is a closed-set assertion rather than a monopoly assertion. A new writer is not
necessarily a violation — it is a question, and failing here is how the question gets asked.
"""

_WRITE_METHODS = frozenset(
    {"write_text", "write_bytes", "rmtree", "unlink", "rename", "mkdir", "touch"}
)
"""Methods that only ever mutate. Deliberately excludes `replace`: `str.replace` is
everywhere and `os.replace` is the rare one, so matching the name alone reports the whole
repository and therefore reports nothing."""


def _writes(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in _WRITE_METHODS:
                return True
            if func.attr == "open" and _is_write_open(node):
                return True
        elif isinstance(func, ast.Name) and func.id == "open" and _is_write_open(node):
            return True
    # `os.replace` goes through `_calls` rather than a literal receiver comparison: both
    # `import os as fs` and `from os import replace as move` are ordinary and both invisible
    # to a literal check, while `text.replace(...)` has to stay invisible to this one —
    # matching the bare method name would report the entire repository.
    return _calls(tree, "os", "replace") or _calls_durable_write(tree)


_DURABLE_MODULES = (
    "chickadee.durable",
    "chickadee.durable.files",
    "chickadee.durable.blobs",
    "chickadee.durable.lock",
    "chickadee.durable.log",
)
_DURABLE_WRITES = frozenset(
    {
        "write_atomic",
        "rename_durably",
        "unlink_durably",
        "make_dir",
        "write_all",
        "repair_tail",
        # Constructing one is the write: its methods are not resolvable statically, and a
        # store, a lock or a log exists only to write its directory or its file.
        "BlobStore",
        "FileLock",
        "AppendLog",
    }
)
"""Every `chickadee.durable` name that writes. DU-9: the package grants no permission to
write, so calling one is the caller's write, and the caller must be in `WRITERS`. Without
this, moving the applier's primitive into a shared module would have let any module write
through it unseen. `sync_dir` is absent: it persists what a writer already did."""


_DURABLE_CLASSES = frozenset({"BlobStore", "FileLock", "AppendLog"})


def _calls_durable_write(tree: ast.Module) -> bool:
    return any(
        _calls(tree, module, name)
        or (name in _DURABLE_CLASSES and _uses_class(tree, module, name))
        for module in _DURABLE_MODULES
        for name in _DURABLE_WRITES
    )


def _uses_class(tree: ast.Module, module: str, name: str) -> bool:
    """A call through the class itself: `FileLock.acquire(p)`, or `d.FileLock.acquire(p)`
    after `import chickadee.durable as d`. `_calls` sees only `FileLock(...)`."""
    bound = _bindings(tree, module, name)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id in bound.direct:
            return True
        if (
            isinstance(owner, ast.Attribute)
            and owner.attr == name
            and isinstance(owner.value, ast.Name)
            and owner.value.id in bound.modules
        ):
            return True
    return False


@pytest.mark.parametrize("path", _sources(), ids=_rel)
def test_every_disk_writer_is_declared(path: pathlib.Path) -> None:
    rel = _rel(path)
    if not _writes(ast.parse(path.read_text(encoding="utf-8"))):
        return
    assert rel in WRITERS, (
        f"{rel} writes to disk and is not in WRITERS. Invariant 1 gives the edit port "
        "(`box/edits.py`) the monopoly on model-authored writes; harness bookkeeping "
        "(journal, index, worktree setup) is permitted but is enumerated so a new path is "
        "read by a person. If this write can carry model-authored content, it belongs in the "
        "edit port."
    )


def test_the_declared_writers_all_still_exist() -> None:
    """An allow-list that outlives its entries stops being a list of exceptions and starts
    being a list of names nobody checks."""
    missing = [rel for rel in WRITERS if not (CORE_SRC / rel).exists()]
    assert not missing, f"WRITERS names modules that no longer exist: {missing}"


def test_the_declared_subprocess_users_all_still_exist() -> None:
    missing = [rel for rel in SUBPROCESS_RUN_OK if not (CORE_SRC / rel).exists()]
    assert not missing, f"SUBPROCESS_RUN_OK names modules that no longer exist: {missing}"


# --- the detectors themselves -----------------------------------------------
#
# A structural test is only worth its detector. Each case below is an evasion that the
# first version of this file missed, found in review on PR #137 — so they are pinned here
# rather than left as a claim that they were fixed once.


@pytest.mark.parametrize(
    ("source", "spawns"),
    [
        ("import subprocess\nsubprocess.Popen(['x'])", True),
        ("from subprocess import Popen\nPopen(['x'])", True),
        # The evasion: a renamed import produces an `ast.Name` the literal string misses.
        ("from subprocess import Popen as spawn\nspawn(['x'])", True),
        # Bound by assignment rather than import, and again one hop further along.
        ("import subprocess\nspawn = subprocess.Popen\nspawn(['x'])", True),
        ("import subprocess\na = subprocess.Popen\nb = a\nb(['x'])", True),
        # Chained assignment is one node with two Name targets, not two Assigns.
        ("import subprocess\na = b = subprocess.Popen\nb(['x'])", True),
        ("import subprocess\nsp = subprocess\nsp.Popen(['x'])", True),
        ("import subprocess as sp\nsp.Popen(['x'])", True),
        ("from subprocess import run\nrun(['x'])", False),
        ("import subprocess", False),
    ],
    ids=[
        "attribute",
        "bare-name",
        "aliased-import",
        "assigned",
        "assigned-twice",
        "chained-assign",
        "assigned-module",
        "aliased-module",
        "run-is-not-popen",
        "import-only",
    ],
)
def test_the_spawn_detector_sees_through_an_alias(source: str, spawns: bool) -> None:
    assert _calls_popen(ast.parse(source)) is spawns


@pytest.mark.parametrize(
    ("source", "writes"),
    [
        ("open('p', 'w')", True),
        ("open('p', mode='a')", True),
        ("import pathlib\npathlib.Path('p').write_text('x')", True),
        ("import pathlib\npathlib.Path('p').mkdir()", True),
        ("import pathlib\npathlib.Path('p').touch()", True),
        ("import os\nos.replace('a', 'b')", True),
        # The same call reached through an aliased module, and through an aliased name.
        ("import os as fs\nfs.replace('a', 'b')", True),
        ("from os import replace as move\nmove('a', 'b')", True),
        # Module alias by assignment, not by `import os as`.
        ("import os\nfs = os\nfs.replace('a', 'b')", True),
        ("open('p')", False),
        ("open('p', 'r')", False),
        # `"ascii"` contains an `"a"`. Reading every positional argument called this a write.
        ("open('p', 'r', -1, 'ascii')", False),
        # `str.replace` is everywhere; only `os.replace` touches disk.
        ("text = 'a'.replace('a', 'b')", False),
    ],
    ids=[
        "open-w",
        "open-mode-kw",
        "write_text",
        "mkdir",
        "touch",
        "os-replace",
        "os-replace-aliased-module",
        "os-replace-aliased-name",
        "os-replace-assigned-module",
        "open-default-read",
        "open-r",
        "encoding-is-not-a-mode",
        "str-replace-is-not-a-write",
    ],
)
def test_the_write_detector_reads_only_the_mode(source: str, writes: bool) -> None:
    assert _writes(ast.parse(source)) is writes


@pytest.mark.parametrize(
    ("source", "writes"),
    [
        ("from chickadee.durable import write_atomic\nwrite_atomic(p, b'')", True),
        ("from chickadee.durable import write_atomic as put\nput(p, b'')", True),
        ("import chickadee.durable as durable\ndurable.write_atomic(p, b'')", True),
        ("from chickadee.durable.files import unlink_durably\nunlink_durably(p)", True),
        ("from chickadee.durable import rename_durably\nrename_durably(a, b)", True),
        ("from chickadee.durable import make_dir\nmake_dir(p)", True),
        ("from chickadee.durable import BlobStore\nBlobStore(p).put(b'')", True),
        ("from chickadee.durable.lock import FileLock\nFileLock.acquire(p)", True),
        # A probe takes and drops a lock and writes nothing.
        ("from chickadee.durable import lock_is_held\nlock_is_held(p)", False),
        # Persisting what a writer already did is not a write of its own.
        ("from chickadee.durable import sync_dir\nsync_dir(p)", False),
        # Importing the name without calling it writes nothing.
        ("from chickadee.durable import write_atomic", False),
    ],
    ids=[
        "write_atomic",
        "write_atomic-aliased",
        "module-aliased",
        "from-files-module",
        "rename_durably",
        "make_dir",
        "BlobStore",
        "FileLock",
        "lock_is_held-is-not-a-write",
        "sync_dir-is-not-a-write",
        "import-only",
    ],
)
def test_a_call_into_durable_is_the_callers_write(source: str, writes: bool) -> None:
    """DU-9, shown failing: a module that calls `write_atomic` without being listed in
    `WRITERS` is reported, however it bound the name."""
    assert _writes(ast.parse(source)) is writes


@pytest.mark.parametrize(
    ("source", "imported"),
    [
        ("from chickadee.process import ProcessSupervisor", True),
        ("from chickadee.process import Completed, ProcessSupervisor", True),
        # The module is the load-bearing half: a look-alike bound from anywhere else is
        # not the supervisor, however convincing the name.
        ("from unrelated import ProcessSupervisor", False),
        ("from chickadee.process import Completed", False),
    ],
    ids=["direct", "among-others", "look-alike-elsewhere", "different-name"],
)
def test_the_import_check_keeps_the_module(source: str, imported: bool) -> None:
    tree = ast.parse(source)
    assert _imports_name_from(tree, "chickadee.process", "ProcessSupervisor") is imported
