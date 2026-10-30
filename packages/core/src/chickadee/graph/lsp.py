"""What the index asks a file's language server: go-to-definition
for the edges it classifies, and each file's verdict.

Each language's questions go to one server: the episode's, kept alive between passes
(`graph/servers.py`), or else **one bounded session per language per pass**, started and shut
down here. Either way the conversation is `graph/conversation.py`'s, and the graph never spawns
the server itself: which server a file's language has, and its `languageId`, are its profile's
(`[lsp]`), and it runs under the harness's wrap (`box/spawn.harness_wrap`):
network off, the repository read-only, the deny list masked.

Returning `None` means *this resolver could not say* — missing binary, timeout,
handshake failure, unparseable output, or a location that is not a symbol we indexed.
`classify` then falls back. Labelling any of those as `resolved` would be a wrong edge.

**A verdict is asked for.** Each file a pass sends is
asked `textDocument/diagnostic`; only a `full` report is returned as its verdict, stamped with
the sha256 of the bytes the server holds. A server that cannot answer gives none, never a clean
file.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Final, cast

from chickadee.edits.lines import split_lines
from chickadee.errors import InfraError
from chickadee.graph.conversation import (
    EXIT,
    INITIALIZED,
    LspConversation,
    Request,
    Supervisor,
    did_open,
    frame_message,
    initialize,
    json_object,
    parse_frames,
    read_file,
    rel_of,
    shutdown,
    uri_of,
)
from chickadee.graph.diagnostics import Verdict
from chickadee.graph.servers import EpisodeServers, Recheck, Server
from chickadee.process.supervisor import scrub_env
from chickadee.profiles.loader import ProfileSet
from chickadee.profiles.model import LanguageProfile, Template
from chickadee.profiles.tools import find_tool

Lookup = Callable[[str, int], str | None]
"""file + 1-based line → innermost indexed qualified name, or None."""


@dataclass(frozen=True)
class DefQuery:
    file: str
    line: int
    name: str
    column: int | None = None
    """The byte column the parse placed `name` at on `line` (`Reference.column`)."""


Wrap = Callable[[tuple[str, ...], Path], tuple[str, ...]]
"""A server's argv as the harness runs it, from the repository root (`box/spawn.harness_wrap`):
raises `InfraError` when it may not run, which leaves that language's edges to fall back."""

SESSION_TIMEOUT_S: Final = 120
"""One language's whole per-pass session, its start included: pyright analyses the workspace
before its first answer. At the deadline every unanswered question falls back."""

MAX_QUERIES: Final = 5_000
"""Questions asked of one language's server per index pass, in file order; the rest fall back,
which the report's `edge_classes` count shows. A server answers one at a time, so this bounds a
pass on a large repository."""

MAX_JUDGED: Final = 200
"""Files of one language whose verdict a pass asks for, beyond those it asks definitions in,
in path order: each is a full analysis inside the pass. A file past the bound keeps the verdict
it had, stale or unavailable as it reads."""


def _column(root: Path, rel: str, line: int, name: str, column: int | None = None) -> int:
    """The LSP `character` of `name` on `line` (1-based), in UTF-16 code units as LSP counts
    them: at `column`, the byte column the parse placed it at, else its first whole-word
    spelling on the line. 0 if the line is missing."""
    path = root / rel
    try:
        rows = split_lines(path.read_bytes().decode("utf-8", "replace"))
    except OSError:
        return 0
    if line < 1 or line > len(rows):
        return 0
    row = rows[line - 1]
    if column is not None:
        before = row.encode("utf-8")[:column].decode("utf-8", "replace")
    else:
        found = re.search(rf"(?<!\w){re.escape(name)}(?!\w)", row)
        before = row[: found.start()] if found is not None else ""
    return len(before.encode("utf-16-le")) // 2


def _definition_path(result: object) -> tuple[str, int] | None:
    """Pull (uri, 1-based line) out of Location | Location[] | LocationLink[]."""
    if isinstance(result, list):
        items = cast("list[object]", result)
        if not items:
            return None
        return _definition_path(items[0])
    obj = json_object(result)
    if obj is None:
        return None
    if "uri" in obj and "range" in obj:
        uri, rng = obj["uri"], obj["range"]
    elif "targetUri" in obj and "targetSelectionRange" in obj:
        uri, rng = obj["targetUri"], obj["targetSelectionRange"]
    elif "targetUri" in obj and "targetRange" in obj:
        uri, rng = obj["targetUri"], obj["targetRange"]
    else:
        return None
    rng_obj = json_object(rng)
    if not isinstance(uri, str) or rng_obj is None:
        return None
    start = json_object(rng_obj.get("start"))
    if start is None:
        return None
    line = start.get("line")
    if not isinstance(line, int):
        return None
    return uri, line + 1


def _definition(root: Path, fid: int, q: DefQuery) -> bytes:
    return frame_message(
        {
            "jsonrpc": "2.0",
            "id": fid,
            "method": "textDocument/definition",
            "params": {
                "textDocument": {"uri": uri_of(root, q.file)},
                "position": {
                    "line": max(q.line - 1, 0),
                    "character": _column(root, q.file, q.line, q.name, q.column),
                },
            },
        }
    )


def encode_dump(
    root: Path, queries: Sequence[DefQuery], language_id: Callable[[str], str]
) -> bytes:
    """Every frame a session sends for `queries`, in order, then EOF: the T-2.17 dump control,
    which tests compare against. `language_id` is a file's LSP `languageId`."""
    return b"".join(
        (
            initialize(root),
            INITIALIZED,
            *_body_frames(root, queries, language_id),
            shutdown(len(queries) + 1),
            EXIT,
        )
    )


def _body_frames(
    root: Path, queries: Sequence[DefQuery], language_id: Callable[[str], str]
) -> Iterator[bytes]:
    opened: set[str] = set()
    for i, q in enumerate(queries, start=1):
        if q.file not in opened:
            yield did_open(root, q.file, language_id(q.file), read_file(root, q.file) or b"")
            opened.add(q.file)
        yield _definition(root, i, q)


def _from(server: Server, verdict: Verdict) -> Verdict:
    """A verdict as its server's: a linter's is the file's own (`Verdict.local`)."""
    return replace(verdict, local=True) if server.local else verdict


def _empty_cache() -> dict[tuple[str, int, str], str | None]:
    return {}


def _apply_definition(
    cache: dict[tuple[str, int, str], str | None],
    query: DefQuery,
    frame: dict[str, object],
    root: Path,
    lookup: Lookup,
) -> None:
    loc = _definition_path(frame.get("result"))
    qname: str | None = None
    if loc is not None:
        rel = rel_of(root, loc[0])
        if rel is not None:
            qname = lookup(rel, loc[1])
    cache[(query.file, query.line, query.name)] = qname


@dataclass
class CheckerResolver:
    """Prefetching Resolver over each language's own server. `definition` is a cache
    read after `prefetch`. A query's language is the profile set's answer for its file; a
    language whose profile names no server, or whose server is not on `PATH`, or whose wrap
    refuses, answers nothing, so its edges fall back. Never a guess labelled `resolved`."""

    root: Path
    supervisor: Supervisor
    profiles: ProfileSet
    wrap: Wrap
    timeout_s: int = SESSION_TIMEOUT_S
    max_queries: int = MAX_QUERIES
    servers: EpisodeServers | None = None
    """The episode's servers, kept between passes; None, and every pass asks
    a session of its own, as `init` and Zeus's checkout index do."""
    _cache: dict[tuple[str, int, str], str | None] = field(default_factory=_empty_cache)

    def prefetch(
        self, queries: Sequence[DefQuery], lookup: Lookup, files: Sequence[str] = ()
    ) -> tuple[Verdict, ...]:
        """For each language with a server: its first `max_queries` questions, and the verdict
        of each file it is asked about and of the pass's other `files` of its language, at most
        `MAX_JUDGED` of those. Every verdict, in path order."""
        # A timeout or missing reply must not keep a previous pass's qualified name —
        # classify would label that stale location `resolved`.
        self._cache.clear()
        verdicts: dict[tuple[str, str], Verdict] = {}
        for profile, asked, judged in self._by_language(queries, files):
            # The language's server answers its questions and judges; its linter judges alone
            # A restarted server answers again: its last verdict stands.
            for server, questions in (
                (self._server(profile), asked),
                (self._linter(profile), ()),
            ):
                if server is None:
                    continue
                into: list[Verdict] = []
                build = partial(self.questions, questions, judged, lookup, into)
                if self.servers is None or not self.servers.ask(server, build):
                    self._session(server, build)
                verdicts.update(((v.path, v.source), _from(server, v)) for v in into)
        return tuple(verdicts[key] for key in sorted(verdicts))

    def judge(self, texts: Mapping[str, bytes]) -> tuple[Verdict, ...]:
        """Each file's verdict on the bytes given rather than those on disk: a task's start,
        for a file that had none when the task first changed it. The next pass's
        `refresh` brings each file back to disk. Every verdict that landed."""
        verdicts: list[Verdict] = []
        for profile, _, judged in self._by_language((), list(texts)):
            for server in (self._server(profile), self._linter(profile)):
                if server is None:
                    continue
                into: list[Verdict] = []

                def build(
                    conversation: LspConversation,
                    judged: list[str] = judged,
                    into: list[Verdict] = into,
                ) -> list[Request]:
                    return [
                        conversation.verdict(r, conversation.send(r, texts[r]), into)
                        for r in judged
                    ]

                if self.servers is None or not self.servers.ask(server, build):
                    self._session(server, build)
                verdicts.extend(_from(server, v) for v in into)
        return tuple(verdicts)

    def wait_rechecked(self, timeout_s: float) -> bool:
        """The background re-check given at most `timeout_s` to finish; True with no servers,
        where nothing is queued (`EpisodeServers.wait_rechecked`)."""
        return True if self.servers is None else self.servers.wait_rechecked(timeout_s)

    def recheck(self, paths: Sequence[str], asked_at: int) -> None:
        """The episode's servers asked again, in the background, for each of `paths`, which a
        change marked stale; nothing without them (`Rechecking`)."""
        if self.servers is None:
            return
        items: list[Recheck] = []
        for rel in paths:
            language = self.profiles.language_of(rel)
            server = None if language is None else self._server(language.profile)
            if server is not None:
                items.append(Recheck(rel, server, asked_at))
        self.servers.recheck(items)

    def _server(self, profile: LanguageProfile) -> Server | None:
        """The profile's server under the wrap; None when it names none, none is installed
        where `find_tool` looks, or the wrap refuses (its edges then fall back)."""
        return self._wrapped(profile, profile.lsp.server, local=False)

    def _linter(self, profile: LanguageProfile) -> Server | None:
        """The profile's linter's server under the wrap; None as `_server` is, and a
        linter not installed is no source: its findings are absent, never clean."""
        return self._wrapped(profile, profile.lint.server, local=True)

    def _wrapped(
        self, profile: LanguageProfile, spec: Template | None, *, local: bool
    ) -> Server | None:
        found = None if spec is None else find_tool(spec.argv[0], self.root, own=True)
        if spec is None or found is None:
            return None
        try:
            # The binary found, by its path: the server's environment need not have it on PATH.
            wrapped = self.wrap((str(found), *spec.argv[1:]), self.root.resolve())
        except InfraError:
            return None
        return Server(profile, wrapped, Path(spec.argv[0]).name, local=local)

    def definition(self, name: str, file: str, line: int) -> str | None:
        return self._cache.get((file, line, name))

    def _by_language(
        self, queries: Sequence[DefQuery], files: Sequence[str]
    ) -> list[tuple[LanguageProfile, list[DefQuery], list[str]]]:
        """Each language with a server, its questions (each asked once, at most
        `max_queries`), and the files whose verdicts it is asked: those it has questions in,
        then the others, at most `MAX_JUDGED`."""
        groups: dict[str, tuple[LanguageProfile, list[DefQuery], list[str]]] = {}

        def group(rel: str) -> tuple[LanguageProfile, list[DefQuery], list[str]] | None:
            language = self.profiles.language_of(rel)
            if language is None or (
                language.profile.lsp.server is None and language.profile.lint.server is None
            ):
                return None
            return groups.setdefault(language.name, (language.profile, [], []))

        seen: set[tuple[str, int, str]] = set()
        for q in queries:
            key = (q.file, q.line, q.name)
            if key not in seen and (found := group(q.file)) is not None:
                seen.add(key)
                found[1].append(q)
        for rel in sorted(set(files)):
            if (found := group(rel)) is not None:
                found[2].append(rel)
        out: list[tuple[LanguageProfile, list[DefQuery], list[str]]] = []
        for profile, asked, others in groups.values():
            asked = asked[: self.max_queries]
            in_asked = sorted({q.file for q in asked})
            asked_files = set(in_asked)
            rest = [rel for rel in others if rel not in asked_files][:MAX_JUDGED]
            out.append((profile, asked, [*in_asked, *rest]))
        return out

    def _session(
        self, server: Server, build: Callable[[LspConversation], list[Request]]
    ) -> None:
        """One language's questions in a session of its own, started and shut down here. A
        failure ends it: what it had not answered falls back, and the verdicts are those that
        landed."""
        root = self.root.resolve()
        profile = server.profile
        try:
            with self.supervisor.session(
                server.argv, cwd=root, timeout_s=self.timeout_s, env=scrub_env()
            ) as sess:
                conversation = LspConversation(
                    sess,
                    root,
                    lambda rel: profile.language_id_for(rel) or profile.name,
                    server.name,
                    settings=server.settings,
                )
                if conversation.start():
                    conversation.ask(build(conversation))
                    conversation.stop()
        except (TimeoutError, OSError, InfraError):
            return

    def questions(
        self,
        queries: Sequence[DefQuery],
        files: Sequence[str],
        lookup: Lookup,
        into: list[Verdict],
        conversation: LspConversation,
    ) -> list[Request]:
        """What one pass asks one language's server, over `conversation`: each of `files` sent
        as it is on disk, its definitions, then each file's verdict into `into`. Shared by a
        per-pass session and the episode's server, whose conversation persists."""
        root = self.root.resolve()
        conversation.refresh()
        sent = {rel: sha for rel in files if (sha := conversation.sync(rel)) is not None}
        asked = [q for q in queries if q.file in sent]
        definitions = [self._definition_request(root, q, lookup) for q in asked]
        verdicts = [conversation.verdict(rel, sha, into) for rel, sha in sent.items()]
        return [*definitions, *verdicts]

    def _definition_request(self, root: Path, q: DefQuery, lookup: Lookup) -> Request:
        def answered(frame: dict[str, object]) -> None:
            _apply_definition(self._cache, q, frame, root, lookup)

        return Request(lambda fid: _definition(root, fid, q), answered)


__all__ = [
    "MAX_JUDGED",
    "MAX_QUERIES",
    "SESSION_TIMEOUT_S",
    "CheckerResolver",
    "DefQuery",
    "Lookup",
    "Supervisor",
    "Wrap",
    "encode_dump",
    "frame_message",
    "parse_frames",
]
