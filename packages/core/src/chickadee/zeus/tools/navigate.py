"""Family A — navigate: the ports Zeus's navigation tools are given (`33-tools.md` §3 rows 1-8).

The tools themselves are built-ins (`tools/builtin/`), dispatched by Zeus's pipeline; what is
Zeus's is what they reach: the graph's navigator (or `NoIndex`), the scout as `locate`'s
`Locator` (or `NoLocator`), `recall`'s `SessionJournal`, and `tools`' `ZeusCatalog`.
Read-only, and never file content: every answer is pointers, names, signatures and edges,
which is what lets Zeus decide what to `read` without reading everything (invariant 13). A
session with no index attached answers `INDEX_UNAVAILABLE` with that as its coverage, which
is a fact, rather than an empty answer, which would be a claim about the repository.

**`ScoutLocator` is the `Locator` a session attaches**. It wires the scout to
what Zeus already has, and forks none of it: the same graph answers Zeus's `symbol`,
`find`, `neighbors` and `impact` reach; `read` and `grep_in` go through a reader of its own
for each `locate`, which shares the session's deny list and caps and nothing of Zeus's
context; the session log takes the scout's records; and its `request_exec` runs under
the `Escapes` built here, from the person's `sandbox.allow_unsandboxed_escapes`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import with_config

from chickadee.adapters.base import tool_descriptions
from chickadee.context.lines import LINE, SessionKind
from chickadee.context.reader import Reader
from chickadee.context.recall import Recall, Recalled
from chickadee.context.record_lines import RecordLine
from chickadee.context.results import CallCancelled, IndexUnavailable, ToolFailure, ToolResult
from chickadee.hooks.fire import Hooks, RoleHooks
from chickadee.init.toolchain import version_of
from chickadee.journal.effects import EFFECT_EVENTS
from chickadee.schema.findings import Findings
from chickadee.schema.isolation import WorkspaceMode
from chickadee.schema.journal import EventKind, P
from chickadee.scout import (
    Checker,
    Escapes,
    Hints,
    LocateCaps,
    Report,
    Scout,
    ScoutSink,
    ScoutTools,
    Search,
    Snapshot,
    SnapshotCache,
    build_snapshot,
    scout_rung,
)
from chickadee.scout.sink import check_record
from chickadee.tools.context import HandleStore, Navigator
from chickadee.tools.meta import with_meta
from chickadee.tools.spec import (
    INLINE_BYTES_DEFAULT,
    RESULT_LINES_DEFAULT,
    SURFACE,
    ToolSpec,
)
from chickadee.zeus.prefix import TEMPLATES

if TYPE_CHECKING:
    from chickadee.budget.manager import Ledger
    from chickadee.client import ModelClient
    from chickadee.config import EngineConfig, RungEndpoint
    from chickadee.context.session import Session
    from chickadee.executor.executor import Executor
    from chickadee.graph.queries import Graph
    from chickadee.profiles.model import LanguageProfile

__all__ = [
    "NoIndex",
    "NoLocator",
    "ScoutLocator",
    "SessionJournal",
    "SessionScoutSink",
    "UnavailableLocator",
    "ZeusCatalog",
    "scout_locator",
]

_NO_INDEX: Final = "no index is attached to this session"
_NO_LOCATOR: Final = "no locate agent is attached"

LOCATE: Final = "locate"


class NoIndex:
    """The navigator of a session with no index attached: every answer is
    `INDEX_UNAVAILABLE`, which is a fact, rather than an empty answer, which would be a claim
    about the repository (a port present with nothing behind it answers its
    own failure)."""

    def atlas(self) -> ToolFailure:
        return IndexUnavailable("", _NO_INDEX)

    def symbol(self, name: str) -> ToolFailure:
        return IndexUnavailable("", _NO_INDEX)

    def neighbors(self, target: str, depth: int) -> ToolFailure:
        return IndexUnavailable("", _NO_INDEX)

    def impact(self, symbols: Sequence[str]) -> ToolFailure:
        return IndexUnavailable("", _NO_INDEX)

    def tests_for(self, paths: Sequence[str]) -> ToolFailure:
        return IndexUnavailable("", _NO_INDEX)


class NoLocator:
    """The locator of a session with no scout attached: `INDEX_UNAVAILABLE`, as above."""

    def locate(
        self, question: str, *, cancel: threading.Event | None
    ) -> str | ToolFailure | CallCancelled:
        return IndexUnavailable("", _NO_LOCATOR)


class UnavailableLocator:
    """The locator of a session whose scout cannot run at all, for a reason known before any
    call: `UNAVAILABLE{rung, reason, detail}`, rendered by the scout's own `Report`, so
    Zeus reads the same answer a scout that failed would give, and the session goes on. The
    engine attaches it when the scout's model has no price row: a call would be paid for and
    then refused at its charge."""

    def __init__(self, model: str, detail: str) -> None:
        self._report = Report(
            question="",
            findings=Findings(truncated=True),
            outcome="unavailable",
            model=model,
            unavailable="unreachable",
            detail=detail,
        )

    def locate(
        self, question: str, *, cancel: threading.Event | None
    ) -> str | ToolFailure | CallCancelled:
        return self._report.render()


class SessionJournal:
    """`recall`'s port over one session: every session under the root, read through the
    engine's one cache of session folds, with this session's own state as
    the current one. Recall reads the session as it stands, so it holds the session's lock;
    what it finds is stored behind one handle, its head inline."""

    def __init__(
        self,
        session: Session,
        recall: Recall,
        handles: HandleStore,
        *,
        inline_bytes: int = INLINE_BYTES_DEFAULT,
    ) -> None:
        self._session = session
        self._recall = recall
        self._handles = handles
        self._bytes = inline_bytes

    def recall(self, query: str) -> ToolResult | ToolFailure:
        with self._session.lock:
            found = self._recall(query, current=self._session.state)
            if isinstance(found, Recalled):
                return found.result(self._handles.store, self._bytes)
        return found


class ZeusCatalog:
    """The `tools` tool's port: Zeus's surface, its description templates, a spec's schema as
    Zeus is shown it (its own arguments and `findings`, `tools/meta.with_meta`), and
    `result_cap` as the brief count."""

    def __init__(
        self,
        surface: tuple[ToolSpec, ...] = SURFACE,
        descriptions: Mapping[str, str] | None = None,
        *,
        entry_cap: int = RESULT_LINES_DEFAULT,
    ) -> None:
        self._surface = surface
        # Read from the templates when the `tools` tool first asks, not when a turn builds
        # its catalog: a turn opens no file it does not use.
        self._descriptions = descriptions
        self._entry_cap = entry_cap

    def offered(self) -> tuple[ToolSpec, ...]:
        return self._surface

    def descriptions(self) -> Mapping[str, str]:
        if self._descriptions is None:
            self._descriptions = tool_descriptions(self._surface, TEMPLATES)
        return self._descriptions

    def shown_schema(self, spec: ToolSpec) -> Mapping[str, object]:
        return with_meta(spec.schema, findings=True)

    @property
    def entry_cap(self) -> int:
        return self._entry_cap


# -- locate: the scout, wired ---------------------------------------------------


_DURABLE: Final = EFFECT_EVENTS
"""The records synced before `record` returns. An escape's intent must be on disk before its
command is dispatched; every other line is flushed like any session line — the run
journal's own policy for the same records."""


@with_config(LINE)
@dataclass(frozen=True)
class ScoutLine(RecordLine):
    """The `scout` line: one scout record, as its kind and the frozen
    journal payload `KIND_TABLE` types it by (`RecordLine`). A `locate` inside a coder episode
    writes the same payload to the run journal, so it is read here as the journal reads it."""

    KIND: ClassVar[SessionKind] = "scout"


class SessionScoutSink:
    """`ScoutSink` for a `locate` from Zeus: the session log.

    One `scout` line per record, carrying the scout's frozen journal payload unchanged — the
    same shapes a `locate` inside a coder episode writes to the run journal, so nothing reads
    them differently by where they landed."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def record(self, kind: EventKind, payload: P) -> int:
        check_record(kind, payload)
        event = self._session.append(ScoutLine(kind, payload))
        if kind in _DURABLE:
            self._session.sync()
        return event.seq


class ScoutLocator:
    """`Locator` over the scout: one per session.

    The snapshot is the session's: built by code on the first `locate` and reused while the
    graph's generation holds. Promoted escapes run into each build, and every
    call's escapes are counted toward promotion. What Zeus reads is the report as the
    scout renders it — findings, `UNRESOLVED`, `CANCELLED`, or `UNAVAILABLE` with what Zeus
    can use instead.

    Each `locate` is a new scout with what one call owns: a reader of its own, a
    checker, and an escape budget and history. What one `locate` read is gone when it
    ends and never reaches Zeus's session.

    What the calls share is the session's, and each part is safe to share whether calls run
    one after another or together (`locate` is `capped`):
    - the snapshot and the hints are touched only under `_shared`, so two calls never both
      rebuild the snapshot, run the promoted commands twice, or lose each other's counts;
    - the promoted set is replaced by `run_promoted`, inside the snapshot build, and read by
      `asked()` when the hints count a call: both under `_shared`, so no call reads it while
      it is being replaced (the replacement itself is two steps, not one);
    - the escapes' memo is a dict each call reads and adds to one key at a time, and each
      such step is whole;
    - the sink and the ledger hold their own locks.
    `escapes` is the session's: its policy, memo and promotions.

    A stop is the session's token, so it reaches every call at once, the one holding
    `_shared` included. That call's build ends between, or inside, its promoted commands, or
    between its version probes, and keeps nothing; what is left to wait for is one probe or
    the snapshot's git reads, each bounded by its own timeout. A call stopped before its
    scout started is `CANCELLED{tool: locate}`; one stopped while its scout searched is the
    scout's partial report, and its asks are counted like any other call's."""

    def __init__(
        self,
        scout: Callable[[], Scout],
        sink: ScoutSink,
        snapshot: Callable[[threading.Event | None], Snapshot | None],
        *,
        escapes: Escapes | None = None,
        hints: Hints | None = None,
    ) -> None:
        self._scout = scout
        self._sink = sink
        self._snapshot = snapshot
        self.escapes = escapes
        self._hints = hints
        self._shared = threading.RLock()
        """Held for the session's shared parts of a `locate`: the snapshot and the hints."""

    def locate(
        self, question: str, *, cancel: threading.Event | None
    ) -> str | ToolFailure | CallCancelled:
        with self._shared:
            stopped = cancel is not None and cancel.is_set()
            snapshot = None if stopped else self._snapshot(cancel)
        if snapshot is None:
            return CallCancelled(LOCATE)
        scout = self._scout()
        report = scout.run(question, snapshot, self._sink, cancel)
        asked = scout.tools.escapes
        if self._hints is not None and asked is not None:
            with self._shared:
                self._hints.record_run(asked.asked())
                self._hints.save()
        return report.render()


def scout_locator(
    config: EngineConfig,
    session: Session,
    *,
    workspace: Path,
    graph: Graph,
    navigator: Navigator,
    reader: Reader,
    executor: Executor,
    client_for: Callable[[RungEndpoint], ModelClient],
    profiles: Sequence[LanguageProfile] = (),
    ledger: Ledger | None = None,
    hints: Hints | None = None,
    execution_mode: WorkspaceMode = "worktree",
    probe: Callable[[str], str | None] = version_of,
    hooks: Hooks | None = None,
) -> ScoutLocator:
    """The session's `locate`, from the person's config.

    `request_exec` runs under `executor` — the session's, with its deny list and sandbox —
    and only where that sandbox enforces DISCOVERY, unless the person set
    `sandbox.allow_unsandboxed_escapes`. That key is read here and nowhere else, which is what
    joins it to `EngineConfig`. The caps are the limits register's `locate` rows,
    sized on the model the scout's rung names. `probe` reads a toolchain
    binary's version for the snapshot (`build_snapshot`)."""
    caps = LocateCaps.from_resolved(config.resolved_limits)
    escapes = Escapes(
        executor=executor,
        cwd=workspace,
        allow_unsandboxed=config.sandbox.allow_unsandboxed_escapes,
    )
    search = Search(
        root=workspace,
        backend=executor.backend,
        denylist=executor.denylist,
        deny_read=executor.deny_read,
    )
    rung = scout_rung(config, client_for)
    # The scout rung's own sight: the rung `scout_rung` resolved, and whether
    # the model it names can see.
    vision = config.vision_for(rung.rung_name)
    policy = config.tool_policy("locate")
    # The session's hooks, as the scout holds them: its role, on its rung's model.
    scouting = (
        None
        if hooks is None
        else RoleHooks(hooks, "locate", rung.model, session=session.session_id)
    )

    def scout() -> Scout:
        # What one `locate` owns: a reader of its own, whose handles and bytes
        # are dropped when the call ends and which shares the session's deny list and caps; a
        # checker; and an escape budget and history over the session's memo.
        own = ScoutTools(
            navigator,
            reader.scratch(vision=vision),
            search,
            escapes=escapes.for_call(),
            policy=policy,
            hooks=scouting,
        )
        checker = Checker(workspace, graph, denylist=executor.denylist)
        return Scout(rung, own, checker, caps, ledger=ledger, hooks=scouting)

    sink = SessionScoutSink(session)

    def build(cancel: threading.Event | None) -> Snapshot | None:
        # A build the stop reached is not kept. It ends between the promoted commands,
        # or between the version probes (up to VERSION_TIMEOUT_S each): once the stop is set
        # no further probe runs, and what the build holds is missing what the stop cut. The
        # stop is seen here, where the build is dropped, so `build_snapshot` has no part in it.
        def stopped() -> bool:
            return cancel is not None and cancel.is_set()

        def probe_unless_stopped(binary: str) -> str | None:
            return None if stopped() else probe(binary)

        promoted: dict[str, str] | None = None
        if hints is not None:
            promoted = escapes.run_promoted(hints.promoted(), sink, cancel)
            if promoted is None:
                return None
        snapshot = build_snapshot(
            workspace,
            profiles,
            graph,
            execution_mode=execution_mode,
            promoted=promoted,
            probe=probe_unless_stopped,
            budget=caps.snapshot_tokens,
        )
        return None if stopped() else snapshot

    cache = SnapshotCache(build, graph.generation)
    return ScoutLocator(scout, sink, cache.get, escapes=escapes, hints=hints)
