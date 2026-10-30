"""The storage guard: every durable write in core goes through
`chickadee.durable`.

Every part that kept state on disk wrote its own atomic write, directory sync, blob store,
lock and log, and the copies drifted (#276 B1, B2, B3). `T-3.48` to `T-3.50` moved them into
one package. This fails the next copy: a file-mutating primitive anywhere in
`packages/core/src` outside `chickadee/durable/`.

It parses each file and counts, per call site:

- calls to `os.fsync`, `os.fdatasync`, `os.replace`, `os.rename`, `os.ftruncate`,
  `os.truncate`, `fcntl.flock`, `fcntl.lockf`, `tempfile.mkstemp` and
  `tempfile.NamedTemporaryFile`, through any alias (`_resolve.py`, which invariant 1's check
  shares);
- reads of `os.O_EXCL` and `os.O_APPEND`;
- the methods `.write_text`, `.write_bytes`, `.truncate`, `.rename`, and `.replace` with one
  positional argument and no keyword (`Path.replace(target)`, never `str.replace(a, b)` or
  `datetime.replace(tzinfo=...)`);
- `open` or `.open` with a write mode.

Core only: EN-1 lets a surface import only the engine facade and the schema, so `eval`
and the CLI keep their own writes. `.unlink` and `.mkdir` are not counted: removing a file and
making a directory are not the torn-write class `durable` exists for, and `make_dir` and
`unlink_durably` are there for a caller that needs them durable.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

from guards import Guard, Violation
from guards._resolve import call_sites, is_write_open, name_uses

REPO_ROOT = Path(__file__).resolve().parents[2]
HOME = "packages/core/src/chickadee/durable/"

EXEMPT: dict[str, str] = {
    "packages/core/src/chickadee/schema/export.py": (
        "the frozen contract tree, T-0.2's alone (D54); its golden write is developer-invoked, "
        "and a torn golden shows in `git diff`"
    ),
}
"""Files outside the rule for good, each with its reason. An exemption is a line a reviewer
reads, never a count: a file here is not scanned for sites. An entry that no longer exempts
anything (its file is gone, or holds no site) is itself a violation, because an exemption
that covers nothing is a standing licence for the next write in that file.

D149 also named `graph/store.py` (SQLite), `isolation/types.py` (contentless mode-000
entries) and `workspace.py` (Git owns worktrees). None holds a counted site: their writes are
`sqlite3`'s, `touch` and `chmod`, and git's. So they are scanned like any file, and a copy of
the durable primitives added to one fails here like anywhere else."""

_CALLS: tuple[tuple[str, str], ...] = (
    ("os", "fsync"),
    ("os", "fdatasync"),
    ("os", "replace"),
    ("os", "rename"),
    ("os", "ftruncate"),
    ("os", "truncate"),
    ("fcntl", "flock"),
    ("fcntl", "lockf"),
    ("tempfile", "mkstemp"),
    ("tempfile", "NamedTemporaryFile"),
)
_CONSTANTS: tuple[tuple[str, str], ...] = (("os", "O_EXCL"), ("os", "O_APPEND"))
_METHODS = frozenset({"write_text", "write_bytes", "truncate", "rename"})


def sites(tree: ast.Module) -> list[ast.AST]:
    """Every node in `tree` that writes the way only `durable` should, each once."""
    found: dict[int, ast.AST] = {}
    for module, attr in _CALLS:
        found.update((id(n), n) for n in call_sites(tree, module, attr))
    for module, attr in _CONSTANTS:
        found.update((id(n), n) for n in name_uses(tree, module, attr))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _writes(node):
            found.setdefault(id(node), node)
    return sorted(found.values(), key=lambda n: (getattr(n, "lineno", 0), id(n)))


def _writes(node: ast.Call) -> bool:
    """A mutating method, or `open` in a write mode. `.replace` only with one positional
    argument and no keyword: `Path.replace(target)`, never `str.replace(a, b)` or
    `datetime.replace(tzinfo=...)`."""
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr in _METHODS:
            return True
        if func.attr == "replace":
            return len(node.args) == 1 and not node.keywords
        return func.attr == "open" and is_write_open(node)
    return isinstance(func, ast.Name) and func.id == "open" and is_write_open(node)


def find(files: Sequence[Path]) -> list[Violation]:
    """Every site outside `durable/`, and every exemption that no longer exempts one.

    Parses core's source, which is Python 3.12: run it with 3.12 or later, as CI's
    `python3` is."""
    violations: list[Violation] = []
    exempted: set[str] = set()
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        if rel.startswith(HOME):
            continue
        source = path.read_text(encoding="utf-8")
        found = sites(ast.parse(source))
        if rel in EXEMPT:
            if found:
                exempted.add(rel)
            continue
        lines = source.splitlines()
        for node in found:
            line = getattr(node, "lineno", 0)
            text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            violations.append(Violation(path=rel, line=line, text=text))
    for rel in sorted(set(EXEMPT) - exempted):
        why = "its file holds no counted site" if (REPO_ROOT / rel).is_file() else "no file"
        violations.append(Violation(path=rel, line=0, text=f"stale EXEMPT entry: {why}"))
    return violations


BASELINE: dict[str, int] = {}

GUARD = Guard(
    name="storage",
    rule=(
        "a durable write (fsync, rename, replace, truncate, flock, temp file, append or write "
        "mode) goes through `chickadee.durable`, never a copy of it (D149)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
