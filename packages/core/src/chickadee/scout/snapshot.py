"""Layer 1: the snapshot — deterministic, zero model calls, and never over
its budget, `locate.snapshot_tokens`.

Built by code on a session's first `locate` and reused while the workspace `generation()`
holds (it said *"at run start"* when a run was the top-level flow). It is the scout's
map, which is why the scout's surface has no `atlas`. Every field is present in the render
even when empty: a snapshot that omitted "no tests found" would read the same as one that
never looked.

**Not to be confused with `TreeSnapshot`** (contracts C-10), the applier's pre-write
enumeration of dirty and untracked paths.
"""

from __future__ import annotations

import json
import os
import re
import threading
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import cast

from chickadee.errors import InfraError
from chickadee.graph.ignore import Ignore
from chickadee.graph.indexer import MAX_FILES
from chickadee.graph.queries import Graph
from chickadee.init.toolchain import version_of
from chickadee.limits import limit, tokens_for_chars
from chickadee.paths import glob_match
from chickadee.process.harness import run_git
from chickadee.profiles.model import LanguageProfile
from chickadee.schema import sha256_hex
from chickadee.schema.isolation import WorkspaceMode

__all__ = [
    "SNAPSHOT_TOKENS",
    "Snapshot",
    "SnapshotCache",
    "build_snapshot",
]

SNAPSHOT_TOKENS = int(limit("locate.snapshot_tokens").value)
"""`locate.snapshot_tokens`'s default, read from the register, which holds the one
copy: the owner's call (raised from D133's 800, which collapsed a real repository's map
to its top-level directories). Still unmeasured; a later measurement recalibrates it.
Measured with P-2's estimator, like every other token count here. A caller holding the
resolved register passes its value as `build_snapshot(budget=)`."""

_SUBJECT_CHARS = 72
_NAME_CHARS = 80

_DEPTH = 3
_SMALL_DIR = 20
_DIRTY_SHOWN = 10
_COMMITS_SHOWN = 5
_ENTRYPOINTS_SHOWN = 10
_DEPS_SHOWN = 12

_ENTRY_NAMES = frozenset(
    {
        "__main__.py", "main.py", "cli.py", "app.py", "server.py", "manage.py", "wsgi.py",
        "asgi.py", "main.go", "main.rs", "main.ts", "main.js", "index.ts", "index.js",
        "server.ts", "server.js", "app.ts", "app.js",
    }
)  # fmt: skip
"""The `entrypoints` row: files that are conventionally where a program starts."""

_RUNTIMES: Mapping[str, str] = {
    "python": "python3",
    "typescript": "node",
    "javascript": "node",
    "go": "go",
    "rust": "rustc",
}
"""The interpreter or compiler whose version the `toolchain` row names, per profile. A profile
names its runner and its bootstrap; neither is the runtime."""

_MAKE = ("Makefile", "makefile", "GNUmakefile")
_JUST = ("justfile", "Justfile", ".justfile")
_DOCKER = (
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml",
)  # fmt: skip


@dataclass(frozen=True)
class Snapshot:
    """The table as a type — flat on purpose, plus `index_truncated`: invariant 20 says
    a partial index may not present itself as complete, and this is the first thing a model
    sees."""

    # repo
    root: str
    source_ref: str
    head: str
    execution_mode: WorkspaceMode
    dirty_count: int
    dirty_first: tuple[str, ...]
    recent_commits: tuple[str, ...]
    # tree — depth 3; full file lists only for directories under 20 entries
    dir_counts: Mapping[str, int]
    small_dir_files: Mapping[str, tuple[str, ...]]
    # languages
    languages: tuple[str, ...]
    profile_name: str
    # manifests — declared dependency names, never lockfile bodies
    manifests: Mapping[str, tuple[str, ...]]
    # tests
    test_dirs: tuple[str, ...]
    test_runner: str | None
    test_count: int
    test_naming: str | None
    # toolchain
    toolchain: Mapping[str, str]
    has_make: bool
    has_just: bool
    has_docker: bool
    # graph
    files_indexed: int
    symbols_indexed: int
    edge_classes: Mapping[str, int]
    index_truncated: bool
    # the rest
    entrypoints: tuple[str, ...]
    promoted: Mapping[str, str]
    tokens: int
    """Measured over `render()`; never above the budget it was built for."""
    omitted: Mapping[str, int]
    """Entries the last trim steps cut, per row (`tree`, `manifests`, `tests`, `commits`,
    `entrypoints`, `promoted`), rendered as `(+N more)` — a cut list says it was cut."""

    def render(self) -> str:
        """What the scout's model reads: every row, `none` where a row is empty."""
        lines = [
            "## SNAPSHOT",
            f"repo: {self.root} · {self.source_ref} @ {self.head[:12] or 'no commits'} · "
            f"{self.execution_mode}",
            f"dirty: {self.dirty_count}" + _list(self.dirty_first, self.dirty_count),
            "recent commits:"
            + ("" if self.recent_commits else " none")
            + self._more("commits"),
            *(f"  - {s}" for s in self.recent_commits),
            "tree:" + ("" if self.dir_counts else " none") + self._more("tree"),
            *(_tree_row(d, n, self.small_dir_files.get(d)) for d, n in self.dir_counts.items()),
            f"languages: {_csv(self.languages)} · profile: {self.profile_name or 'none'}",
            "manifests:" + ("" if self.manifests else " none") + self._more("manifests"),
            *(f"  {p}: {_csv(deps)}" for p, deps in self.manifests.items()),
            f"tests: {self.test_count} file(s) in {_csv(self.test_dirs)}{self._more('tests')}"
            f" · runner: {self.test_runner or 'none'} · naming: {self.test_naming or 'none'}",
            "toolchain: "
            + (_csv(tuple(f"{k} {v}" for k, v in self.toolchain.items())))
            + f" · make: {_yn(self.has_make)} · just: {_yn(self.has_just)} · docker: "
            f"{_yn(self.has_docker)}",
            f"graph: {self.files_indexed} files, {self.symbols_indexed} symbols · edges "
            + (_csv(tuple(f"{k} {v}" for k, v in self.edge_classes.items())))
            + (
                " · INDEX TRUNCATED: some files are not indexed" if self.index_truncated else ""
            ),
            f"entrypoints: {_csv(self.entrypoints)}{self._more('entrypoints')}",
            "promoted:" + ("" if self.promoted else " none") + self._more("promoted"),
            *(f"  $ {cmd}\n    {_first(out)}" for cmd, out in self.promoted.items()),
        ]
        return "\n".join(lines)

    def _more(self, row: str) -> str:
        n = self.omitted.get(row, 0)
        return f" (+{n} more)" if n else ""

    @property
    def sha(self) -> str:
        return sha256_hex(self.render())


def _csv(items: Sequence[str]) -> str:
    return ", ".join(items) if items else "none"


def _yn(flag: bool) -> str:
    return "yes" if flag else "no"


def _list(shown: Sequence[str], total: int) -> str:
    if not shown:
        return ""
    more = f" (+{total - len(shown)} more)" if total > len(shown) else ""
    return f" — {', '.join(shown)}{more}"


def _tree_row(directory: str, count: int, files: tuple[str, ...] | None) -> str:
    depth = 0 if directory == "." else directory.count("/") + 1
    row = f"{'  ' * depth}{directory}/ ({count})"
    return f"{row}: {', '.join(files)}" if files else row


def _first(output: str) -> str:
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    if not lines:
        return "(no output)"
    more = f" (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
    return lines[0][:120] + more


# -- building ---------------------------------------------------------------------------


@dataclass
class _Walk:
    files: list[str]
    dir_counts: dict[str, int]
    direct: dict[str, list[str]]


def _walk(root: Path) -> _Walk:
    """One gitignore-aware pass (`graph.ignore`, the indexer's own rules): every file's
    path, recursive counts for directories to depth 3, and each directory's direct files."""
    rules = Ignore(root)
    files: list[str] = []
    counts: dict[str, int] = {}
    direct: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        current = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not rules.skipped(current / d))
        rel_dir = current.relative_to(root).as_posix()
        kept = sorted(f for f in filenames if not rules.skipped(current / f))
        direct[rel_dir] = kept
        for name in kept:
            files.append(name if rel_dir == "." else f"{rel_dir}/{name}")
        parts = [] if rel_dir == "." else rel_dir.split("/")
        for i in range(min(len(parts), _DEPTH) + 1):
            key = "/".join(parts[:i]) or "."
            counts[key] = counts.get(key, 0) + len(kept)
    return _Walk(files, counts, direct)


def _git(root: Path, *args: str) -> str | None:
    try:
        done = run_git(root, *args)
    except (InfraError, OSError):
        return None
    if done.timed_out or done.exit_code != 0:
        return None
    return done.stdout.decode("utf-8", "replace")


def _dirty(root: Path) -> list[str]:
    out = _git(root, "status", "--porcelain=v1", "-z")
    if not out:
        return []
    names: list[str] = []
    entries = out.split("\0")
    skip = False
    for entry in entries:
        if skip:
            skip = False
            continue
        if len(entry) < 4:
            continue
        names.append(entry[3:])
        # A rename or copy carries its source as the next NUL-separated field.
        skip = entry[0] in "RC"
    return names


_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _declared(path: Path) -> tuple[str, ...]:
    """Declared dependency names, never versions and never lockfile bodies. A manifest
    this does not read is still listed, with no names, rather than left out."""
    name = path.name
    try:
        if name == "pyproject.toml":
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            project = cast(dict[str, object], data.get("project", {}))
            deps = cast(list[object], project.get("dependencies", []))
            names = [
                m.group(1) for d in deps if isinstance(d, str) if (m := _REQ_NAME.match(d))
            ]
            tool = cast(dict[str, object], data.get("tool", {}))
            poetry = cast(dict[str, object], tool.get("poetry", {}))
            names += [
                k
                for k in cast(dict[str, object], poetry.get("dependencies", {}))
                if k != "python"
            ]
            return tuple(dict.fromkeys(names))
        if name == "package.json":
            data = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
            keys: list[str] = []
            for section in ("dependencies", "devDependencies"):
                keys += list(cast(dict[str, object], data.get(section, {})))
            return tuple(dict.fromkeys(keys))
        if name == "Cargo.toml":
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            return tuple(cast(dict[str, object], data.get("dependencies", {})))
        if name == "go.mod":
            text = path.read_text(encoding="utf-8")
            return tuple(
                m.group(1)
                for m in re.finditer(r"^\s*(?:require\s+)?([\w.\-/]+)\s+v\d", text, re.M)
            )
        if name.startswith("requirements") and name.endswith(".txt"):
            rows = path.read_text(encoding="utf-8").splitlines()
            return tuple(
                m.group(1)
                for r in rows
                if not r.lstrip().startswith(("#", "-"))
                if (m := _REQ_NAME.match(r))
            )
    except (OSError, ValueError, tomllib.TOMLDecodeError):
        return ()
    return ()


def _graph_facts(graph: Graph | None) -> tuple[int, int, dict[str, int], tuple[str, ...]]:
    if graph is None:
        return 0, 0, {}, ()
    conn = graph.conn
    files = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
    symbols = int(conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0])
    edges = {
        str(k): int(n)
        for k, n in conn.execute(
            "SELECT edge_class, COUNT(*) FROM edges GROUP BY edge_class ORDER BY edge_class"
        )
    }
    langs = tuple(
        str(r[0])
        for r in conn.execute(
            "SELECT lang FROM files WHERE lang IS NOT NULL GROUP BY lang "
            "ORDER BY COUNT(*) DESC, lang"
        )
    )
    return files, symbols, edges, langs


def build_snapshot(
    repo: Path,
    profiles: Sequence[LanguageProfile],
    graph: Graph | None,
    *,
    execution_mode: WorkspaceMode = "worktree",
    promoted: Mapping[str, str] | None = None,
    probe: Callable[[str], str | None] = version_of,
    budget: int = SNAPSHOT_TOKENS,
) -> Snapshot:
    """The table for `repo`. `profiles` is every profile the repository activates — several
    may — and `promoted` the promoted escapes' outputs, run by
    `Escapes.run_promoted` under the same sandbox. `graph` None is an unavailable index,
    reported as zeros rather than guessed."""
    root = repo.resolve()
    walk = _walk(root)
    dirty = _dirty(root)
    log = _git(root, "log", f"-{_COMMITS_SHOWN}", "--format=%s") or ""
    head = (_git(root, "rev-parse", "HEAD") or "").strip()
    ref = (_git(root, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() or "unknown"

    globs = tuple(g for p in profiles for g in p.tests.globs)
    tests = [f for f in walk.files if glob_match(f, globs) is not None]
    test_dirs = tuple(sorted({f.split("/", 1)[0] for f in tests if "/" in f}))
    runners = [
        p.tests.runner.argv[0] for p in profiles if p.tests.runner and p.tests.runner.argv
    ]
    naming = [g for g in globs if "*" in g.rsplit("/", 1)[-1] and "**" not in g]

    manifest_names = {m for p in profiles for m in p.deps.manifests}
    manifests = {
        f: _declared(root / f)[:_DEPS_SHOWN]
        for f in walk.files
        if f.rsplit("/", 1)[-1] in manifest_names and f.count("/") < _DEPTH
    }

    toolchain: dict[str, str] = {}
    binaries = [_RUNTIMES[p.name] for p in profiles if p.name in _RUNTIMES]
    binaries += [p.toolchain.bootstrap.argv[0] for p in profiles if p.toolchain.bootstrap]
    binaries += runners
    for binary in dict.fromkeys(binaries):
        version = probe(binary)
        if version is not None:
            toolchain[binary] = version

    files_n, symbols_n, edges, langs = _graph_facts(graph)
    top = set(walk.direct.get(".", []))
    entry = tuple(
        f
        for f in walk.files
        if (f.rsplit("/", 1)[-1] in _ENTRY_NAMES or f.startswith("bin/"))
        and glob_match(f, globs) is None
    )[:_ENTRYPOINTS_SHOWN]

    snap = Snapshot(
        root=str(root),
        source_ref=ref,
        head=head,
        execution_mode=execution_mode,
        dirty_count=len(dirty),
        dirty_first=tuple(dirty[:_DIRTY_SHOWN]),
        recent_commits=tuple(s for s in log.splitlines() if s.strip())[:_COMMITS_SHOWN],
        dir_counts=dict(sorted(walk.dir_counts.items(), key=lambda kv: _tree_key(kv[0]))),
        small_dir_files={
            d: tuple(names)
            for d, names in walk.direct.items()
            if d in walk.dir_counts and 0 < len(names) < _SMALL_DIR
        },
        languages=langs or tuple(p.name for p in profiles),
        profile_name=", ".join(p.name for p in profiles),
        manifests=manifests,
        test_dirs=test_dirs,
        test_runner=", ".join(dict.fromkeys(runners)) or None,
        test_count=len(tests),
        test_naming=", ".join(naming) or None,
        toolchain=toolchain,
        has_make=bool(top & set(_MAKE)),
        has_just=bool(top & set(_JUST)),
        has_docker=bool(top & set(_DOCKER)),
        files_indexed=files_n,
        symbols_indexed=symbols_n,
        edge_classes=edges,
        # The indexer keeps the newest MAX_FILES and says so only in its report, which the
        # store does not keep; an index holding exactly the cap may be complete, but saying
        # so would be the claim invariant 20 forbids.
        index_truncated=files_n >= MAX_FILES,
        entrypoints=entry,
        promoted=dict(promoted or {}),
        tokens=0,
        omitted={},
    )
    return _fit(snap, budget)


def _tree_key(directory: str) -> tuple[str, ...]:
    return () if directory == "." else tuple(directory.split("/"))


def _measured(snap: Snapshot) -> Snapshot:
    return replace(snap, tokens=tokens_for_chars(len(snap.render())))


def _fit(snap: Snapshot, budget: int) -> Snapshot:
    """Trim toward the budget in the order that loses least: file lists first (a directory's
    count survives), then the deepest tree rows, then the long lists, then promoted output —
    and then, by construction, every list with no count cap of its own, cut to the
    most entries that still fit, with a `(+N more)` marker. Every row stays — a trimmed row
    is shorter, never absent. "The most that fit" is searched for, not stepped down to: a
    fixed ladder sized for a small budget throws most of a large one away.

    The rows themselves are the floor: a budget smaller than the empty skeleton (well under a
    hundred tokens) cannot be met, and the skeleton is returned. Any budget above it is."""
    snap = _measured(snap)
    steps: list[Callable[[Snapshot], Snapshot]] = [
        lambda s: replace(
            s, small_dir_files={d: f for d, f in s.small_dir_files.items() if d == "."}
        ),
        lambda s: replace(s, small_dir_files={}),
        lambda s: replace(
            s, dir_counts={d: n for d, n in s.dir_counts.items() if d.count("/") < 2}
        ),
        lambda s: replace(
            s, dir_counts={d: n for d, n in s.dir_counts.items() if d.count("/") < 1}
        ),
        lambda s: replace(s, manifests={p: d[:4] for p, d in s.manifests.items()}),
        lambda s: replace(
            s, dirty_first=s.dirty_first[:3], recent_commits=s.recent_commits[:2]
        ),
        lambda s: replace(s, entrypoints=s.entrypoints[:3]),
        lambda s: replace(s, promoted={c: _first(o) for c, o in s.promoted.items()}),
    ]
    for step in steps:
        if snap.tokens <= budget:
            return snap
        snap = _measured(step(snap))
    if snap.tokens <= budget:
        return snap
    base = snap
    best = _measured(_capped(base, 0))
    if best.tokens > budget:
        return best  # the skeleton: the floor the rows themselves set
    # More entries never render shorter, so the largest k that fits is a binary search.
    lo, hi = 0, _longest(base)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = _measured(_capped(base, mid))
        if candidate.tokens <= budget:
            lo, best = mid, candidate
        else:
            hi = mid - 1
    return best


def _longest(s: Snapshot) -> int:
    """The longest list `_capped` cuts; the tree's count leaves out the root's row, which
    `_capped` always keeps."""
    return max(
        len(s.dir_counts) - 1,
        len(s.manifests),
        len(s.promoted),
        len(s.test_dirs),
        len(s.recent_commits),
        len(s.entrypoints),
        len(s.dirty_first),
        0,
    )


def _capped(base: Snapshot, k: int) -> Snapshot:
    """`base` with each uncapped list cut to `k` entries and long strings shortened. The tree
    keeps the root's row, whose count covers everything cut below it."""
    tree = dict(list(base.dir_counts.items())[: k + 1])
    manifests = dict(list(base.manifests.items())[:k])
    promoted = {c[:_NAME_CHARS]: o for c, o in list(base.promoted.items())[:k]}
    tests = base.test_dirs[:k]
    commits = tuple(c[:_SUBJECT_CHARS] for c in base.recent_commits[:k])
    entry = base.entrypoints[:k]
    cut = {
        "tree": len(base.dir_counts) - len(tree),
        "manifests": len(base.manifests) - len(manifests),
        "promoted": len(base.promoted) - len(promoted),
        "tests": len(base.test_dirs) - len(tests),
        "commits": len(base.recent_commits) - len(commits),
        "entrypoints": len(base.entrypoints) - len(entry),
    }
    return replace(
        base,
        dir_counts=tree,
        small_dir_files={},
        manifests=manifests,
        promoted=promoted,
        test_dirs=tests,
        recent_commits=commits,
        dirty_first=tuple(d[:_NAME_CHARS] for d in base.dirty_first[:k]),
        entrypoints=entry,
        omitted={row: n for row, n in cut.items() if n},
    )


def field_names() -> tuple[str, ...]:
    """Every field the type carries — what the completeness test holds the build to."""
    return tuple(f.name for f in fields(Snapshot))


# -- reuse ------------------------------------------------------------------------------


class SnapshotCache:
    """One session's snapshot: built on the first `locate`, reused while `generation()`
    returns what it returned at the build, rebuilt when it moves — the invalidation
    signal.

    A build the session's stop cut short returns `None` and is not kept: what it holds
    is missing what the stop cut, so the next call builds afresh rather than reuse it."""

    def __init__(
        self,
        build: Callable[[threading.Event | None], Snapshot | None],
        generation: Callable[[], int],
    ) -> None:
        self._build = build
        self._generation = generation
        self._held: tuple[int, Snapshot] | None = None
        self.builds = 0

    def get(self, cancel: threading.Event | None = None) -> Snapshot | None:
        now = self._generation()
        if self._held is None or self._held[0] != now:
            built = self._build(cancel)
            if built is None:
                return None
            self._held = (now, built)
            self.builds += 1
        return self._held[1]
