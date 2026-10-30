"""Plain helpers for the context suite — a uniquely named module, never `conftest`."""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from files import pdf as pdf
from files import png as png
from tool_args import checked, zeus_answer

from chickadee.config import PriceRow
from chickadee.context.compaction.cut import CacheState, Compaction, compact
from chickadee.context.compaction.residue import NO_RESIDUE, ResidueSource
from chickadee.context.diagnostics import DiagnosticsIndex
from chickadee.context.handle import content_sha
from chickadee.context.homes import SessionHome
from chickadee.context.lines import EpisodeLine
from chickadee.context.project import RunRecords
from chickadee.context.reader import Reader
from chickadee.context.recall import Recall
from chickadee.context.replay import ReplayCache
from chickadee.context.request import Ctx, assemble
from chickadee.context.resolver import HandleResolver
from chickadee.context.session import Session, ToolLine, Turn
from chickadee.context.terms import Caps, ChangedLines, ReadRules, SymbolIndex, Vision
from chickadee.journal.effects import EffectKind
from chickadee.journal.errors import BlobMissing
from chickadee.journal.projection import RunProjection, project
from chickadee.schema import EventKind, JournalEvent, Tier, ToolCall
from chickadee.schema.adapter import CacheMode, CacheObs
from chickadee.schema.journal import (
    EffectCompletedP,
    EffectIntendedP,
    EffectStartedP,
    P,
    RunEndP,
    RunStartP,
)
from chickadee.secrets.denylist import Denylist

SESSION_ID = "S-test"
TS = "2026-09-26T12:00:00"


def sessions_dir(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


def recall_of(sessions: Path, *, records: RunRecords | None = None) -> Recall:
    """`recall` over `sessions`, with a cache of its own: the engine passes its one."""
    return Recall(sessions, ReplayCache(sessions), records=records)


def new_session(tmp_path: Path, session_id: str = SESSION_ID) -> Session:
    return Session.create(sessions_dir(tmp_path), session_id)


def reopen(session: Session) -> Session:
    """Close and open again: everything the reopened session holds came from the log."""
    root, sid = session.root, session.session_id
    session.close()
    return Session.open(root, sid)


def write_files(base: Path, files: Mapping[str, str | bytes]) -> Path:
    for name, body in files.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            path.write_bytes(body)
        else:
            path.write_text(body, encoding="utf-8")
    return base


def numbered(n: int, word: str = "line") -> str:
    """`n` distinct lines, `line 1` to `line n`, newline-terminated."""
    return "".join(f"{word} {i}\n" for i in range(1, n + 1))


ROW = re.compile(r"^ *(\d+)│", re.MULTILINE)
"""A numbered file line as every read prints it: the number is the file's."""
MARK = re.compile(r"\[elided: lines? (\d+)(?:-(\d+))?\]")


def shown(text: str) -> set[int]:
    """The file lines a result's text shows, by the numbers it prints."""
    return {int(n) for n in ROW.findall(text)}


def marked(text: str) -> set[int]:
    """The file lines a result's text says it left out."""
    out: set[int] = set()
    for a, b in MARK.findall(text):
        out.update(range(int(a), int(b or a) + 1))
    return out


def run_id(name: str) -> str:
    """A well-formed run id (`<date>-<8 hex>`), the same for the same name on every run."""
    return f"2026-09-26-{hashlib.sha256(name.encode()).hexdigest()[:8]}"


# -- the graph, as the reader sees it -----------------------------------------------------


@dataclass(frozen=True)
class Sym:
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int
    sig: str | None = None
    doc_first_line: str | None = None


@dataclass
class FakeIndex:
    """`SymbolIndex` over a fixed table: a path it has symbols for is known, and the table
    describes that file's bytes under `root` as they are when read. With no `root` it describes
    no bytes, so every known file reads as edited since it was indexed."""

    symbols: dict[str, tuple[Sym, ...]] = field(default_factory=dict[str, tuple[Sym, ...]])
    root: Path | None = None

    def file_known(self, path: str) -> bool:
        return path in self.symbols

    def symbols_in(self, path: str) -> Sequence[Sym]:
        return self.symbols.get(path, ())

    def indexed_sha(self, path: str) -> str | None:
        if self.root is None or path not in self.symbols:
            return None
        return content_sha((self.root / path).read_bytes())


def python_module(functions: int, body: int = 3) -> tuple[str, tuple[Sym, ...]]:
    """A module of `functions` functions, `body` lines each and a blank after, with the
    symbols an index would report for it."""
    lines: list[str] = []
    syms: list[Sym] = []
    for i in range(1, functions + 1):
        start = len(lines) + 1
        lines.append(f"def f{i}(x):")
        lines.extend(f"    y{j} = x + {i * 100 + j}" for j in range(1, body))
        lines.append(f"    return {i}")
        syms.append(Sym(f"f{i}", f"mod.f{i}", "function", start, len(lines), f"def f{i}(x)"))
        lines.append("")
    return "\n".join(lines) + "\n", tuple(syms)


@dataclass(frozen=True)
class Diag:
    """A `graph` diagnostic row, structurally."""

    line: int
    severity: str = "error"
    message: str = "something is wrong"
    col: int | None = None
    code: str | None = None
    source: str = "pyright"


@dataclass(frozen=True)
class DiagSet:
    in_span: tuple[Diag, ...] = ()
    out_of_span: tuple[Diag, ...] = ()
    coverage: str = "unavailable"
    analyzed_sha: str | None = None
    stale_by: str | None = None


@dataclass
class FakeDiagnostics:
    """`DiagnosticsIndex` over a fixed table; a path it has no row for was never analyzed."""

    by_path: dict[str, DiagSet] = field(default_factory=dict[str, DiagSet])

    def diagnostics_in(self, path: str, lines: range | None = None) -> DiagSet:
        return self.by_path.get(path, DiagSet())


# -- calling tools the way the loop will ---------------------------------------------------


def reader_for(
    session: Session,
    ws: Path,
    *,
    vision: Vision | None = None,
    index: SymbolIndex | None = None,
    diagnostics: DiagnosticsIndex | None = None,
    changes: ChangedLines | None = None,
    denylist: Denylist | None = None,
    caps: Caps | None = None,
) -> Reader:
    return Reader(
        SessionHome(session),
        HandleResolver(ws, session.blob),
        vision=vision or Vision("zeus"),
        index=index,
        diagnostics=diagnostics,
        changes=changes,
        denylist=denylist,
        rules=None if caps is None else ReadRules(caps),
    )


class Caller:
    """Record an assistant turn making one call, then answer and record it through Zeus's own
    dispatcher and tools (`zeus_answer`) — the order the loop uses, so every result lands as
    the durable line the fold reads."""

    def __init__(self, session: Session, reader: Reader) -> None:
        self.session = session
        self.reader = reader
        self._ids = itertools.count(1)

    def next_id(self) -> str:
        return f"c{next(self._ids)}"

    def __call__(self, tool: str, **args: object) -> ToolLine:
        cid = self.next_id()
        self.session.record_assistant("", [ToolCall(id=cid, name=tool, arguments=args)])
        recall = recall_of(self.session.root) if tool == "recall" else None
        return zeus_answer(
            self.session, self.reader, cid, tool, checked(tool, args), recall=recall
        )


# -- whole exchanges, for the cut ---------------------------------------------------------


def at_length(ask: str) -> str:
    """A user turn as long as people write them. A cut that frees nothing is not taken, so a
    region of three-word turns — cheaper than the block that would replace it — is no
    region at all: a test that needs a cut states its asks at length."""
    return f"{ask} " + "(with the context and caveats a real request carries) " * 12


def exchange(
    session: Session,
    call_id: str,
    tool: str,
    args: Mapping[str, object],
    result: str,
    *,
    finding: str | None = "seen",
) -> Turn:
    """One assistant call, its result, and the assistant turn that drew a finding from it."""
    session.record_assistant("", [ToolCall(id=call_id, name=tool, arguments=dict(args))])
    session.record_tool(call_id, result)
    findings = {call_id: finding} if finding is not None else None
    return session.record_assistant("noted", findings=findings)


def work_unit(session: Session, name: str) -> str:
    """A user ask, a look around, a `coder` dispatch with its episode, and the result: one
    completed unit. Returns its run id."""
    run = run_id(name)
    session.record_user(f"please do {name}")
    exchange(
        session, f"r-{name}", "shell", {"command": f"ls {name}"}, f"files of {name}\n" * 30
    )
    dispatch = session.record_assistant(
        "dispatching",
        [ToolCall(id=f"c-{name}", name="coder", arguments={"request": {"goal": name}})],
    )
    session.record_episode(EpisodeLine(run, "planned", Tier.T1, None, (), dispatch.seq))
    session.record_tool(f"c-{name}", f"run {run} SOLVED", run_id=run)
    session.record_assistant(f"{name} is done", findings={f"c-{name}": f"{name} solved"})
    return run


# -- the cache, the window, the journal ---------------------------------------------------


def price() -> PriceRow:
    return PriceRow(price_in=3.0, price_out=15.0, price_cached_in=0.3, cache_ttl_s=300)


def no_cache_signal() -> CacheState:
    """A provider that reports nothing about its cache: `warm` is unknowable."""
    return CacheState(price(), CacheObs(mode=CacheMode.NONE))


def over_threshold(tokens: int, *, fraction: float = 0.9) -> Ctx:
    """A context `tokens` large in a window it fills to `fraction`: past the size trigger,
    with a response reserve small enough that the request still fits, so the cut is the
    economic kind rather than MUST_FIT."""
    window = max(int(tokens / fraction), 1)
    return Ctx(tokens, window, "context_limit", max(window // 50, 1))


def cut(
    session: Session,
    ws: Path,
    *,
    records: FakeRecords | None = None,
    layer3: ResidueSource = NO_RESIDUE,
    pinned: Sequence[int] = (),
) -> Compaction:
    """One compaction over the session as it stands: over the size trigger, with no cache
    signal and nothing pending, so the cut is taken on size and nothing else."""
    held = assemble(session.state, session.blob).tokens
    return compact(
        session,
        over_threshold(held),
        no_cache_signal(),
        0,
        resolver=HandleResolver(ws, session.blob),
        records=records,
        layer3=layer3,
        pinned=pinned,
    )


def event(
    run: str,
    seq: int,
    kind: EventKind,
    payload: P,
    *,
    task_id: str | None = None,
    round: int | None = None,
) -> JournalEvent:
    return JournalEvent(
        run_id=run,
        seq=seq,
        ts=TS,
        kind=kind,
        payload=payload,
        task_id=task_id,
        round=round,
    )


def run_start(
    run: str, goal: str, *, isolation: Literal["worktree", "in_place"] = "worktree"
) -> JournalEvent:
    return event(
        run,
        1,
        EventKind.RUN_START,
        RunStartP(
            goal=goal,
            work_sha="0" * 64,
            config_sha="0" * 64,
            repo_head="0" * 40,
            engine_version="0",
            schema_version=2,
            isolation=isolation,
            hard_cost_cap_usd=5.0,
        ),
    )


def run_end(run: str, seq: int, outcome: str = "COMPLETED", cost: float = 0.42) -> JournalEvent:
    """A run's end: one task, committed when the run completed; `why` set otherwise."""
    completed = outcome == "COMPLETED"
    return event(
        run,
        seq,
        EventKind.RUN_END,
        RunEndP.model_validate(
            {
                "outcome": outcome,
                "why": None if completed else "the task was not accepted",
                "tasks": 1,
                "committed": 1 if completed else 0,
                "tokens": 1000,
                "cost_usd": cost,
                "wall_ms": 2500,
                "handoff_path": "handoff.md",
            }
        ),
    )


def effect(
    run: str,
    seq: int,
    operation_id: str,
    kind: EffectKind,
    outcome: Literal["succeeded", "failed", "ambiguous"] = "succeeded",
) -> tuple[JournalEvent, JournalEvent, JournalEvent]:
    """One effect's lifecycle: intended, started and completed, from `seq`."""
    return (
        event(
            run,
            seq,
            EventKind.EFFECT_INTENDED,
            EffectIntendedP(
                operation_id=operation_id,
                effect_kind=kind.value,
                approved_sha="0" * 64,
                preconditions=(),
                idempotent=False,
                declared_surface=(),
            ),
        ),
        event(
            run, seq + 1, EventKind.EFFECT_STARTED, EffectStartedP(operation_id=operation_id)
        ),
        event(
            run,
            seq + 2,
            EventKind.EFFECT_COMPLETED,
            EffectCompletedP(operation_id=operation_id, outcome=outcome, observed_surface=()),
        ),
    )


@dataclass
class FakeRecords:
    """`RunRecords` over events held in memory; a run with no events has no journal."""

    runs: dict[str, tuple[JournalEvent, ...]] = field(
        default_factory=dict[str, tuple[JournalEvent, ...]]
    )
    blobs: dict[tuple[str, str], bytes] = field(default_factory=dict[tuple[str, str], bytes])
    reads: list[str] = field(default_factory=list[str])

    def projection(self, run_id: str) -> RunProjection:
        self.reads.append(run_id)
        if run_id not in self.runs:
            raise FileNotFoundError(run_id)
        return project(self.runs[run_id])

    def blob(self, run_id: str, sha: str) -> bytes:
        try:
            return self.blobs[(run_id, sha)]
        except KeyError:
            raise BlobMissing(sha) from None
