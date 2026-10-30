"""Text or path search over the repository, including files the graph does not index: the
`search` port's implementation, for every role that holds `grep` and `glob`
(`tools/builtin/grep.py`, `glob.py`).
"""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from chickadee.context.result_data import Match, Matches, Paths, ResultData, Structured
from chickadee.edits.lines import split_lines
from chickadee.graph.ignore import ALWAYS_SKIP, walk
from chickadee.isolation.base import SandboxBackend, select
from chickadee.isolation.none import Passthrough
from chickadee.isolation.types import Isolation
from chickadee.paths import glob_match, spelled_out, within
from chickadee.process.supervisor import ProcessSupervisor, scrub_env
from chickadee.secrets.denylist import Denylist, read_denied
from chickadee.tools.builtin.grep import SPEC as GREP_SPEC
from chickadee.tools.spec import INLINE_BYTES_DEFAULT, RESULT_LINES_DEFAULT
from chickadee.tools.validate import ArgumentViolation, InvalidArguments

__all__ = ["SEARCH_TIMEOUT_S", "Search", "SearchHit"]

SEARCH_TIMEOUT_S = 20
_FILE_BYTES = 1 << 20
"""The in-process fallback skips files over a megabyte: a text search that reads a build
artefact whole is not answering a question about the source."""


@dataclass(frozen=True)
class SearchHit:
    path: str
    line: int | None
    """None for a path hit."""
    text: str = ""


@dataclass
class Search:
    """The scout's own bounded search: ripgrep as harness argv through Isolation and the
    supervisor — gitignore-aware, network off, the repository read-only, the deny list
    masked — and, where `rg` is not installed, an in-process literal scan of the same tree.
    Path search is always in-process: it reads names, never contents.

    The two engines search the same files the same way: both include hidden files and skip
    the indexer's `ALWAYS_SKIP` directories, and both match case-sensitively; they differ
    only in that the fallback matches the query as literal text, which its first line says.

    Every hit is judged by `secrets.read_denied` whichever engine ran, because at
    `sandbox=none` the mask is not there to stop `rg` reading one: its name, then where it
    lands. The fallback asks before it reads a file, and it reads no link that leaves the
    root."""

    root: Path
    backend: SandboxBackend | None = None
    supervisor: ProcessSupervisor | None = None
    denylist: Denylist = field(default_factory=Denylist)
    deny_read: tuple[Path, ...] = ()
    max_hits: int = RESULT_LINES_DEFAULT
    max_bytes: int = INLINE_BYTES_DEFAULT
    timeout_s: int = SEARCH_TIMEOUT_S
    rg: str | None = field(default_factory=lambda: shutil.which("rg"))
    cancel: threading.Event | None = None

    def __call__(
        self,
        query: str,
        *,
        kind: Literal["text", "path"] = "text",
        paths: Sequence[str] = (),
        cancel: threading.Event | None = None,
    ) -> str:
        """`cancel` is the call's own stop, which ends a running `rg`; without one,
        the search's own."""
        timed_out = False
        if kind == "path":
            hits, engine = list(self._paths(query, paths)), "paths"
        elif self.rg is not None:
            found = self._ripgrep(query, paths, cancel or self.cancel)
            if isinstance(found, str):
                return found
            (hits, timed_out), engine = found, "ripgrep"
        else:
            hits, engine = list(self._literal(query, paths)), "in-process, literal (no rg)"
        real_root = Path(os.path.realpath(self.root))
        kept = [h for h in hits if self._readable(h.path, real_root)]
        return self._render(query, kind, engine, kept, timed_out=timed_out)

    # -- engines ------------------------------------------------------------------------

    def _ripgrep(
        self, query: str, paths: Sequence[str], cancel: threading.Event | None
    ) -> tuple[list[SearchHit], bool] | str:
        argv: tuple[str, ...] = (
            cast(str, self.rg),
            "--line-number",
            "--no-heading",
            "--color=never",
            "--no-messages",
            "--max-columns=200",
            "--hidden",
            *(f"--glob={spelled_out(g)}" for g in paths),
            # After the caller's globs, so a skip wins over an include (rg: the later glob).
            *(f"--glob=!{d}/" for d in sorted(ALWAYS_SKIP)),
            "-e",
            query,
            "--",
            ".",
        )
        backend = self.backend or select()
        wrap = backend if backend.available() else Passthrough()
        listed = self.denylist
        wrapped = wrap.wrap(
            Isolation(
                argv=argv,
                cwd=self.root,
                network_off=True,
                repo_writable=False,
                deny_read=() if listed.allow_secret_read else self.deny_read,
                deny_patterns=listed.name_patterns(),
            )
        )
        done = (self.supervisor or ProcessSupervisor()).run(
            wrapped,
            cwd=self.root,
            env=scrub_env(),
            timeout_s=self.timeout_s,
            cancel=cancel,
        )
        out = done.stdout.decode("utf-8", "replace")
        if done.exit_code == 2 and not out and not done.timed_out:
            # rg's own complaint — an unparsable pattern, most often — as the typed failure
            # the model corrects against.
            why = done.stderr.decode("utf-8", "replace").strip() or "rg exited 2"
            violation = ArgumentViolation("pattern", why[:300])
            return InvalidArguments(GREP_SPEC.name, (violation,), GREP_SPEC.schema).as_content()
        hits = [h for h in (_rg_line(line) for line in out.splitlines()) if h is not None]
        return hits, done.timed_out

    def _files(self, paths: Sequence[str]) -> Iterator[str]:
        for rel in walk(self.root):
            if not paths or glob_match(rel, paths) is not None:
                yield rel

    def _paths(self, query: str, paths: Sequence[str]) -> Iterator[SearchHit]:
        for rel in self._files(paths):
            if glob_match(rel, (query,)) is not None:
                yield SearchHit(path=rel, line=None)

    def _literal(self, query: str, paths: Sequence[str]) -> Iterator[SearchHit]:
        real_root = Path(os.path.realpath(self.root))
        for rel in self._files(paths):
            full = self.root / rel
            if full.is_symlink() and within(self.root, full) is None:
                # It reads through the link, and a landing outside the root is not the
                # repository's to search.
                continue
            if not self._readable(rel, real_root):
                continue
            try:
                if full.stat().st_size > _FILE_BYTES:
                    continue
                data = full.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:8192]:
                continue
            for n, line in enumerate(split_lines(data.decode("utf-8", "replace")), 1):
                if query in line:
                    yield SearchHit(path=rel, line=n, text=line.strip()[:200])

    def _readable(self, rel: str, real_root: Path) -> bool:
        """The one read rule: the hit's name, then where it lands, so a link cannot
        launder a secret into a result and a nested `.docker/config.json` is the entry it
        is. Asked of every hit, whichever engine found it."""
        return read_denied(rel, self.root / rel, self.denylist, workspace=real_root) is None

    # -- the result ---------------------------------------------------------------------

    def _render(
        self, query: str, kind: str, engine: str, hits: list[SearchHit], *, timed_out: bool
    ) -> str:
        # Named as the tool the model called: `glob` reads paths, `grep` contents.
        tool = "glob" if kind == "path" else "grep"
        head = f"{tool} {query!r} · {engine}"
        if not hits:
            where = "file paths" if kind == "path" else "file contents"
            note = f" (timed out after {self.timeout_s}s)" if timed_out else ""
            text = f"{head}\nUNRESOLVED{{pattern: {query!r}, searched: {where}}}{note}"
            return Structured(text, _data(kind, []))
        rows: list[str] = []
        shown: list[SearchHit] = []
        size = len(head)
        for h in hits[: self.max_hits]:
            row = h.path if h.line is None else f"{h.path}:{h.line}: {h.text}"
            size += len(row.encode("utf-8")) + 1
            if size > self.max_bytes:
                break
            rows.append(row)
            shown.append(h)
        body = [head, *rows]
        if len(rows) < len(hits):
            body.append(
                f"[truncated: {len(hits) - len(rows):,} of {len(hits):,} hits not shown "
                "(result_cap) · narrow the pattern or the paths]"
            )
        if timed_out:
            body.append(f"[{tool} timed out after {self.timeout_s}s; hits are partial]")
        return Structured("\n".join(body), _data(kind, shown))


def _data(kind: str, shown: list[SearchHit]) -> ResultData:
    """The rows the text shows, typed: the deny list and the bound have run,
    so nothing here is a hit the text left out."""
    if kind == "path":
        return Paths(tuple(h.path for h in shown))
    return Matches(tuple(Match(h.path, h.line or 0, h.text) for h in shown))


def _rg_line(line: str) -> SearchHit | None:
    path, sep, rest = line.partition(":")
    number, sep2, text = rest.partition(":")
    if not (sep and sep2 and number.isdigit()):
        return None
    return SearchHit(path=path.removeprefix("./"), line=int(number), text=text.strip()[:200])
