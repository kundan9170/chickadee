"""Full and incremental indexing.

**The index always completes.** A parse error or an unprofiled language is a *gap* that gets
reported, never an exception that ends the pass. A repository always has some file the
harness cannot read, and an indexer that refuses to finish because of one is an indexer that
never runs on the repositories this is for.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from chickadee.clock import stamp
from chickadee.errors import InfraError
from chickadee.graph import diagnostics, fts, ignore, store
from chickadee.graph import edges as edge_mod
from chickadee.graph.dependents import dependents
from chickadee.graph.diagnostics import Verdict
from chickadee.graph.lsp import DefQuery
from chickadee.graph.parse import ParsedFile, parse_file
from chickadee.paths import within
from chickadee.process.supervisor import ProcessSupervisor, scrub_env
from chickadee.profiles.loader import ProfileSet
from chickadee.profiles.resolve import FileLanguage
from chickadee.schema.ids import sha256_hex
from chickadee.secrets.denylist import Denylist, read_denied

SKIP_DIRS = ignore.ALWAYS_SKIP
"""Always-skipped path components. Gitignore is additive on top of this set."""

MAX_FILES = 20_000
"""Breadth cap. Hitting it sets `IndexReport.truncated` rather than silently indexing a
prefix of the repository."""


@dataclass(frozen=True)
class Gap:
    path: str
    reason: str


@dataclass(frozen=True)
class IndexReport:
    files_indexed: int
    symbols: int
    edges: int
    edge_classes: dict[str, int]
    gaps: tuple[Gap, ...]
    truncated: bool
    wall_ms: int


@dataclass(frozen=True)
class IndexProgress:
    """How far a pass has got: what a person watching a build is shown."""

    stage: Literal["preparing", "reading", "linking"]
    """`preparing`: no file read yet (the caller's work before the pass: the repository, the
    deny list, the profiles). `reading`: `done` of `total` files taken, the part of a build
    that grows with the repository. `linking`: every file read; the edges, the servers'
    verdicts and the commit left."""
    done: int = 0
    total: int = 0


def walk(root: Path) -> Iterable[Path]:
    """Source files, gitignore-aware. Sorted so an uncapped pass is byte-identical.

    Directories the rules skip are not descended into — `node_modules` and `.git` are
    never listed, not listed and then discarded.
    """
    rules = ignore.Ignore(root)
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = sorted(name for name in dirnames if not rules.skipped(current / name))
        for name in filenames:
            path = current / name
            if not rules.skipped(path):
                found.append(path)
    yield from sorted(found)


def _cap(
    paths: list[Path], root: Path, languages: Mapping[Path, FileLanguage | None]
) -> tuple[list[Path], bool]:
    """Keep the newest *indexable* files when the repo exceeds MAX_FILES. Spec §6.

    Unprofiled files become gaps either way. Spending a cap slot on a lockfile or
    README would drop source the parser could have used.
    """
    indexable = [p for p in paths if languages[p] is not None]
    if len(indexable) <= MAX_FILES:
        return paths, False
    recency = git_recency(root)
    tracked = git_tracked(root)
    dirty = git_dirty(root)
    oldest = len(recency) + 1

    def key(path: Path) -> tuple[int, str]:
        rel = path.relative_to(root).as_posix()
        if rel in recency:
            return (recency[rel], rel)
        # Untracked or dirty: the working set. Tracked files outside the 200-commit
        # window are older than everything in it — ranking them as -1 kept lockfiles
        # from 2015 and dropped yesterday's edit (search already uses `len+1`).
        if rel not in tracked or rel in dirty:
            return (-1, rel)
        return (oldest, rel)

    return sorted(indexable, key=key)[:MAX_FILES], True


def sha256_of(path: Path) -> str:
    return sha256_hex(path.read_bytes())


@runtime_checkable
class _Judging(Protocol):
    """A resolver that judges bytes other than the disk's (`lsp.CheckerResolver.judge`)."""

    def judge(self, texts: Mapping[str, bytes]) -> tuple[Verdict, ...]: ...


@runtime_checkable
class _Settling(Protocol):
    """A resolver whose servers re-check in the background (`lsp.Resolver`)."""

    def wait_rechecked(self, timeout_s: float) -> bool: ...


class Indexer:
    def __init__(
        self,
        repo: Path,
        resolver: edge_mod.Resolver | None = None,
        *,
        denylist: Denylist | None = None,
        profiles: ProfileSet | None = None,
        progress: Callable[[IndexProgress], None] | None = None,
    ) -> None:
        self.repo = repo
        self.conn = store.connect(repo)
        self.resolver: edge_mod.Resolver = resolver or edge_mod.NoResolver()
        """Each production index is given its languages' servers (`box/spawn.index_resolver`);
        `NoResolver`, the default, is for tests and an index that opts out."""
        self.profiles = profiles if profiles is not None else ProfileSet.load(repo=repo)
        """Which language each file is and the grammar that parses it: the
        run's frozen set when the composition passes one, else the repository's."""
        self._marked: list[str] = []
        """Files this pass marked stale, nearest first: re-checked once it commits."""
        self.keeping: str | None = None
        """The task whose start each pass keeps: set for a worker's rounds, so a
        verdict a pass or its re-check replaces is copied aside first."""
        self.keeping_from: str | None = None
        """The tree the kept task began at: where a changed file with no verdict then has its
        start read from, and asked of its server."""
        self.denylist = denylist if denylist is not None else Denylist()
        """The session's list, which the composition passes; the default is the built-ins.
        A source file it denies is a `Gap`, never parsed: its symbols and signatures would
        reach the planner."""
        self.progress = progress
        """Told how far each pass has got: once per file taken, a gap included, and
        once as the files give way to the edges. Called on the pass's thread."""

    def settle(self, timeout_s: float) -> bool:
        """The re-check of the files earlier passes outdated, given at most `timeout_s` to
        finish: what a task's stop waits on before it compares with the start. Whether it
        finished; True where nothing re-checks."""
        if isinstance(self.resolver, _Settling):
            return self.resolver.wait_rechecked(timeout_s)
        return True

    def since_start(self, task: str) -> diagnostics.SinceStart:
        """`task`'s kept files now against their start (`diagnostics.since_start`)."""
        return diagnostics.since_start(self.conn, task)

    def close(self) -> None:
        self.conn.close()

    def full_index(self, root: Path | None = None) -> IndexReport:
        root = root or self.repo
        self.conn.execute("DELETE FROM files")  # cascades to symbols/edges/imports/globals
        fts.clear(self.conn)
        # A full index judges the repository afresh: no verdict from before is carried.
        diagnostics.clear(self.conn)
        return self._index(list(walk(root)), root)

    def reindex(self, changed: Sequence[Path] | None = None) -> IndexReport:
        """Reparse only what changed. `changed=None` rescans and compares sha256."""
        root = self.repo
        if changed is None:
            candidates = [p for p in walk(root) if self._is_dirty(root, p)]
        else:
            # A caller's watcher can hand us a path from outside the repository.
            # `relative_to` would raise on it, and GR-4 says the index always completes.
            candidates = [p for p in changed if p.is_file() and within(root, p) is not None]
        self._forget_deleted(root)
        extra = self._sources_needing_reresolve(root)
        merged: list[Path] = []
        seen: set[Path] = set()
        for path in (*candidates, *extra):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            merged.append(path)
        report = self._index(merged, root)
        # Restore: the unique name exists only after this pass writes it. Unknown callers
        # of that name are reclassified now, not left as a permanent gap (item 6).
        follow = [p for p in self._sources_needing_reresolve(root) if p.resolve() not in seen]
        if not follow:
            return report
        extra_report = self._index(follow, root)
        return IndexReport(
            files_indexed=report.files_indexed + extra_report.files_indexed,
            symbols=extra_report.symbols,
            edges=extra_report.edges,
            edge_classes=extra_report.edge_classes,
            gaps=report.gaps + extra_report.gaps,
            truncated=report.truncated or extra_report.truncated,
            wall_ms=report.wall_ms + extra_report.wall_ms,
        )

    # -- internals ----------------------------------------------------------

    def _is_dirty(self, root: Path, path: Path) -> bool:
        rel = path.relative_to(root).as_posix()
        try:
            return store.indexed_sha(self.conn, rel) != sha256_of(path)
        except OSError:
            return True  # unreadable now; let `_index` record it as a gap

    def _forget_deleted(self, root: Path) -> None:
        """Files gone from disk lose their rows, and edges pointing at their symbols are
        marked stale+unknown rather than left dangling. What depended on them is marked
        stale first, while their rows still say who did."""
        on_disk = {p.relative_to(root).as_posix() for p in walk(root)}
        known = {r[0] for r in self.conn.execute("SELECT path FROM files")}
        gone = sorted(known - on_disk)
        self._outdate(gone)
        diagnostics.forget(self.conn, gone)
        removed: set[str] = set()
        for path in gone:
            removed |= self._drop_file(path)
        self._mark_vanished(removed)

    def _drop_file(self, rel: str) -> set[str]:
        """Remove a file's rows, returning the qualified names it had defined.

        **Dropping is not deleting.** Reindexing a file drops its rows and writes them
        again, so the names returned here are "gone *right now*", not "gone from the
        repository". Marking staleness from inside this method treated every edit as a
        deletion: editing a file downgraded every edge pointing at its symbols to
        `unknown`+`stale` even though the symbols came straight back. `_mark_vanished`
        makes that judgement afterwards, when the rewritten rows are visible.
        """
        gone = {
            r[0]
            for r in self.conn.execute(
                "SELECT qualified_name FROM symbols WHERE file = ?", (rel,)
            )
        }
        fts.drop_file(self.conn, rel)
        self.conn.execute("DELETE FROM files WHERE path = ?", (rel,))
        return gone

    def _mark_vanished(self, candidates: set[str]) -> int:
        """Of the names dropped this pass, mark stale only those with no surviving row."""
        if not candidates:
            return 0
        survivors: set[str] = set()
        for batch in edge_mod.chunks(sorted(candidates), edge_mod.batch_size(self.conn)):
            survivors |= {
                r[0]
                for r in self.conn.execute(
                    "SELECT qualified_name FROM symbols WHERE qualified_name IN "
                    f"({','.join('?' * len(batch))})",
                    batch,
                )
            }
        return edge_mod.mark_stale_for_deleted(self.conn, candidates - survivors)

    def _sources_needing_reresolve(self, root: Path) -> list[Path]:
        """Caller files whose outgoing edges are no longer trustworthy (item 6).

        Stale edges are the delete/rename case: the callee vanished, the caller file did
        not change. Unknown edges whose bare name is now unique are the restore case: a
        later pass created the only `helper`, and the unchanged caller must pick it up.
        Builtins (`str`) stay unknown and are not reparsed on every pass.
        """
        files: list[Path] = []
        seen: set[str] = set()

        def add(rel: str) -> None:
            if rel in seen:
                return
            path = root / rel
            if path.is_file():
                seen.add(rel)
                files.append(path)

        for (rel,) in self.conn.execute("SELECT DISTINCT file FROM edges WHERE stale = 1"):
            add(rel)

        indexes: dict[str, dict[str, str]] = {}
        rows = self.conn.execute(
            "SELECT e.file, e.dst, f.lang FROM edges e JOIN files f ON f.path = e.file "
            "WHERE e.edge_class = 'unknown' AND e.stale = 0"
        )
        for rel, dst, lang in rows:
            if lang not in indexes:
                indexes[lang] = edge_mod.name_index(self.conn, lang)
            known = indexes[lang]
            bare = dst.rsplit(".", 1)[-1]
            if dst in known or bare in known:
                add(rel)
        return files

    def _index(self, paths: list[Path], root: Path) -> IndexReport:
        started = time.perf_counter()
        # Each file's language once per pass: the cap and the parse ask the same question.
        languages = {path: self.profiles.language_of(path) for path in paths}
        paths, truncated = _cap(paths, root, languages)
        real_root = Path(os.path.realpath(root))
        gaps: list[Gap] = []
        parsed: list[ParsedFile] = []
        dropped: set[str] = set()
        # Files whose bytes differ from the index's, new ones included: what makes their
        # dependents' verdicts stale.
        changed: list[str] = []
        previous: dict[str, str | None] = {}
        """Each changed file's indexed sha before this pass: what its kept verdict judged."""

        for done, path in enumerate(paths):
            self._said("reading", done, len(paths))
            rel = path.relative_to(root).as_posix()
            language = languages[path]
            if language is None:
                gaps.append(Gap(rel, "unprofiled language"))
                continue
            denial = read_denied(rel, path, self.denylist, workspace=real_root)
            if denial is not None:
                # Parsing is reading. A row from before the file was denied goes too, and the
                # verdict on its bytes with it; what depended on it is marked first, while its
                # rows still say who did, as a deleted file's dependents are.
                if store.indexed_sha(self.conn, rel) is not None:
                    self._outdate([rel])
                dropped |= self._drop_file(rel)
                diagnostics.forget(self.conn, (rel,))
                gaps.append(Gap(rel, denial.reason))
                continue
            try:
                sha = sha256_of(path)
            except OSError as exc:
                # The index always completes. A file the process cannot read is a
                # gap like any other — permissions, a race with an editor, a broken
                # symlink. Letting this escape made one unreadable file end the pass.
                gaps.append(Gap(rel, f"unreadable: {exc.strerror or exc}"))
                continue
            before = store.indexed_sha(self.conn, rel)
            if before != sha:
                changed.append(rel)
                previous[rel] = before
            dropped |= self._drop_file(rel)
            result = parse_file(root, path, language)
            parsed.append(result)
            self._write_file_row(rel, result, sha)
            if not result.parse_ok:
                gaps.append(Gap(rel, result.error or "parse error"))

        # Edges resolve in a second pass, after every symbol from this batch is in the
        # table: a name defined in the last file walked must be visible to a reference in
        # the first, or file order would decide an edge's class.
        # Per language, not repository-wide: a Python definition is not a candidate for a
        # JavaScript call, and a fallback edge across that boundary is a wrong edge.
        # Now that every file in this batch has been rewritten, a dropped name that has no
        # row is genuinely gone rather than merely reparsed.
        self._mark_vanished(dropped)
        self._said("linking", len(paths), len(paths))
        if self.keeping is not None:
            self._ask_start(self.keeping, previous)
        verdicts = self._prefetch(parsed)

        names: dict[str, dict[str, str]] = {}
        edge_total = 0
        for result in parsed:
            if result.lang not in names:
                names[result.lang] = edge_mod.name_index(self.conn, result.lang)
            classified = edge_mod.classify(
                result.references, result.path, names[result.lang], self.resolver
            )
            edge_mod.insert(self.conn, classified)
            edge_total += len(classified)

        # Marked against this pass's edges, then cleared for every file a server judged in it:
        # it judged them with the changed files already on disk. A task's start is kept first,
        # before this pass or its re-check replaces a verdict.
        self._outdate(changed, previous)
        diagnostics.record(self.conn, verdicts)
        # A file judged in this pass is fresh: nothing is left for a re-check to ask of it.
        judged = {v.path for v in verdicts if not v.local}
        self._marked = [p for p in self._marked if p not in judged]
        generation = store.bump_generation(self.conn)
        self.conn.commit()
        self._recheck(generation)
        return IndexReport(
            files_indexed=len(parsed),
            symbols=self._count("symbols"),
            edges=self._count("edges"),
            edge_classes=edge_mod.class_counts(self.conn),
            gaps=tuple(gaps),
            truncated=truncated,
            wall_ms=int((time.perf_counter() - started) * 1000),
        )

    def _said(self, stage: Literal["reading", "linking"], done: int, total: int) -> None:
        if self.progress is not None:
            self.progress(IndexProgress(stage, done, total))

    def _ask_start(self, task: str, previous: Mapping[str, str | None]) -> None:
        """Each changed file with no fresh verdict at its first change has its start asked: its
        bytes at the task's tree (`keeping_from`) judged by its server, or nothing for a file
        the task created. A tree git cannot read leaves them "no record"."""
        needs = diagnostics.unkept(self.conn, task, previous)
        tree = self.keeping_from
        if not needs or tree is None or _git_stdout(self.repo, "cat-file", "-t", tree) == b"":
            return
        texts: dict[str, bytes] = {}
        for rel in needs:
            if _git_stdout(self.repo, "ls-tree", tree, "--", rel) == b"":
                diagnostics.keep_found(self.conn, task, rel, ())
                continue
            texts[rel] = _git_stdout(self.repo, "cat-file", "blob", f"{tree}:{rel}")
        if texts and isinstance(self.resolver, _Judging):
            for verdict in self.resolver.judge(texts):
                diagnostics.keep_found(self.conn, task, verdict.path, verdict.diagnostics)

    def _outdate(
        self, changed: Sequence[str], previous: Mapping[str, str | None] | None = None
    ) -> None:
        """Every judged file depending on one of `changed`, marked stale by it at this pass's
        generation (A2); nothing to walk while no file has a verdict. Where a
        task's start is kept, `changed` (each at its `previous` sha) and those dependents are
        kept first, before this pass or its re-check replaces their verdicts."""
        outdated = (
            dependents(self.conn, changed)
            if changed and diagnostics.judged_any(self.conn)
            else {}
        )
        if self.keeping is not None:
            now = {p: store.indexed_sha(self.conn, p) for p in outdated}
            diagnostics.keep(self.conn, self.keeping, {**now, **(previous or {})})
        if outdated:
            this_pass = store.generation(self.conn) + 1
            marked = diagnostics.mark_stale(self.conn, outdated, this_pass)
            self._marked.extend(p for p in marked if p not in self._marked)

    def _recheck(self, generation: int) -> None:
        """The files this pass marked, handed to the resolver's servers to ask again in the
        background, once the marks are committed; with no such servers they
        stay marked until a later pass judges them."""
        marked, self._marked = self._marked, []
        if marked and isinstance(self.resolver, edge_mod.Rechecking):
            self.resolver.recheck(marked, generation)

    def _prefetch(self, parsed: Sequence[ParsedFile]) -> tuple[Verdict, ...]:
        """The resolver's batch of questions, and the verdicts its servers published for this
        pass's files."""
        if not isinstance(self.resolver, edge_mod.Prefetching) or not parsed:
            return ()
        # Only a name some indexed symbol has, or an import binds (`from a import f as g`):
        # a server's answer for any other (`len`, a library's function) lands outside the
        # index, where no symbol can be `resolved`, so asking would spend the server's time
        # for an answer `lookup` throws away.
        declared = {str(r[0]) for r in self.conn.execute("SELECT DISTINCT name FROM symbols")}
        declared |= {
            str(r[0])
            for r in self.conn.execute(
                "SELECT DISTINCT alias FROM imports WHERE alias IS NOT NULL"
            )
        }
        queries = [
            DefQuery(result.path, ref.line, ref.name, ref.column)
            for result in parsed
            for ref in result.references
            if ref.name.rsplit(".", 1)[-1] in declared
        ]
        return self.resolver.prefetch(
            queries, self._symbol_at, [result.path for result in parsed]
        )

    def _symbol_at(self, file: str, line: int) -> str | None:
        """The symbol declared on `line` of `file`, innermost first, or None. A definition a
        server finds elsewhere in a symbol's body is a local (a variable, a parameter), not the
        symbol: taking the innermost one covering the line made a function's own locals edges
        to itself, labelled `resolved`."""
        row = self.conn.execute(
            "SELECT qualified_name FROM symbols WHERE file = ? AND line_start = ? "
            "ORDER BY (line_end - line_start) ASC LIMIT 1",
            (file, line),
        ).fetchone()
        return str(row[0]) if row else None

    def _write_file_row(self, rel: str, result: ParsedFile, sha: str) -> None:
        self.conn.execute(
            "INSERT INTO files(path, sha256, lang, parse_ok, indexed_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, lang=excluded.lang, "
            "parse_ok=excluded.parse_ok, indexed_at=excluded.indexed_at",
            (rel, sha, result.lang, int(result.parse_ok), stamp()),
        )
        ids: dict[str, int] = {}
        for sym in result.symbols:
            parent_id = ids.get(sym.parent) if sym.parent else None
            cur = self.conn.execute(
                "INSERT INTO symbols(name, qualified_name, file, kind, line_start, "
                "line_end, sig, doc_first_line, parent_id, exported) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    sym.name,
                    sym.qualified_name,
                    rel,
                    sym.kind,
                    sym.line_start,
                    sym.line_end,
                    sym.sig,
                    sym.doc_first_line,
                    parent_id,
                    int(sym.exported),
                ),
            )
            rowid = cur.lastrowid
            if rowid is None:  # rowid is the join key; 0 would be an unreachable orphan row
                raise InfraError(
                    f"sqlite returned no rowid for {sym.qualified_name!r} in {rel}"
                )
            ids[sym.qualified_name] = int(rowid)
            fts.index_symbol(self.conn, rowid, sym.qualified_name)
        for imp in result.imports:
            self.conn.execute(
                "INSERT INTO imports(file, module, alias, line) VALUES(?,?,?,?)",
                (rel, imp.module, imp.alias, imp.line),
            )
        for glob in result.globals:
            self.conn.execute(
                "INSERT INTO globals(qualified_name, name, file, line, kind, type_text) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(qualified_name) DO UPDATE SET "
                "file=excluded.file, line=excluded.line, kind=excluded.kind, "
                "type_text=excluded.type_text",
                (glob.qualified_name, glob.name, rel, glob.line, glob.kind, glob.type_text),
            )

    def _count(self, table: str) -> int:
        row = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # fixed literals
        return int(row[0])


GIT_TIMEOUT_S = 10


def _git_stdout(repo: Path, *args: str) -> bytes:
    """One git argv through ProcessSupervisor. Empty on any failure."""
    try:
        done = ProcessSupervisor().run(
            ("git", *args),
            cwd=repo,
            env=scrub_env(),
            timeout_s=GIT_TIMEOUT_S,
        )
    except (OSError, InfraError):
        return b""
    if done.timed_out or done.exit_code != 0:
        return b""
    return done.stdout


def git_recency(repo: Path) -> dict[str, int]:
    """Commit recency per path, newest first, for `search` ranking. Absent git is not an
    error — ranking falls back to BM25 alone."""
    out = _git_stdout(repo, "log", "--name-only", "--pretty=format:", "-n", "200")
    if not out:
        return {}
    order: dict[str, int] = {}
    for rank, line in enumerate(out.decode("utf-8", "replace").split("\n")):
        path = line.strip()
        if path and path not in order:
            order[path] = rank
    return order


def git_tracked(repo: Path) -> set[str]:
    return _git_nul(repo, "ls-files", "-z")


def git_dirty(repo: Path) -> set[str]:
    """Unstaged modifications plus staged tracked edits. `ls-files -m` misses the index."""
    return _git_nul(repo, "ls-files", "-m", "-z") | _git_nul(
        repo, "diff", "--cached", "--name-only", "-z"
    )


def _git_nul(repo: Path, *args: str) -> set[str]:
    out = _git_stdout(repo, *args)
    if not out:
        return set()
    return {p for p in out.decode("utf-8", "replace").split("\0") if p}
