"""The run journal, folded once, into one typed projection every reader reads.

**One fold, and only the fold reads a line's kind.** Every reader asks the projection by
payload type (`lines(ExecP)`, `last(PackedP)`) and is answered with typed lines. A line's kind
and its payload's type name each other one to one (`KIND_TABLE`), so a type is a kind. A
reader holding a line of more than one type narrows its payload with `isinstance`.

**It records what the journal says and refuses nothing.** An operation reused, started without
intent or completed without start, a seq gap, a line after `run_end`: each is a `Defect` with
its seq, never raised. The readers that refuse such a journal (`gc`, `Journal.open_for_resume`,
`Box.resume`) call `require_consistent()`, which raises `JournalCorrupt`. Every other reader
reads such a journal as it did.

**It holds no in-flight task state.** A task's state is `box/tasks.TaskBook`'s fold of its
lines, which the projection keeps by task (`tasks`).

**Pure.** No file, no clock, no randomness: the same lines fold to the same projection.
Commands and external calls are not re-run; their recorded results are payloads like any other
(cassette semantics), and `cassetted_kinds` names which kinds were read that way.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel

from chickadee.journal.errors import JournalCorrupt, JournalError
from chickadee.schema import EventKind, JournalEvent
from chickadee.schema.journal import (
    KIND_TABLE,
    BudgetP,
    EffectCompletedP,
    EffectIntendedP,
    EffectStartedP,
    ExecP,
    McpCallP,
    ModelCallP,
    RedactionP,
    RunEndP,
    RunStartP,
    ScoutToolP,
    SecretDeniedP,
    ToolCallP,
    UntrustedReadP,
)

__all__ = [
    "Defect",
    "Fold",
    "Operation",
    "Placed",
    "RunProjection",
    "RunSpend",
    "kind_of",
    "project",
    "project_before",
    "project_readable",
    "tool_call",
]

_KIND_OF: Final[Mapping[type[BaseModel], EventKind]] = MappingProxyType(
    {model: kind for kind, (model, _) in KIND_TABLE.items()}
)


def kind_of(payload: BaseModel) -> EventKind:
    """The kind a payload is written under: `KIND_TABLE` read the other way, one model per
    kind. For a reader that must write a line it holds again (resume's recorded decision,
    which resume compares), never for dispatching on what a line is."""
    return _KIND_OF[type(payload)]


@dataclass(frozen=True)
class Placed[P: BaseModel]:
    """One line, as the fold keeps it: its place in the journal and its typed payload."""

    seq: int
    payload: P
    task_id: str | None = None
    """The task the line is about."""
    round: int | None = None

    def narrow[Q: BaseModel](self, model: type[Q]) -> Placed[Q] | None:
        """This line as a line of `model`, or None when its payload is another type."""
        p = self.payload
        if isinstance(p, model):
            return Placed(self.seq, p, self.task_id, self.round)
        return None


@dataclass(frozen=True)
class Defect:
    """Something the fold found inconsistent, at `seq`. Recorded, never raised."""

    seq: int
    why: str


def tool_call(p: ScoutToolP | ToolCallP) -> ToolCallP:
    """A tool call's line, whichever kind it was written as, read as a `ToolCallP`: the one
    reader of both. A `scout_tool` line is the scout's (`locate`),
    served, and a repeat when `dedup_hit`."""
    if isinstance(p, ToolCallP):
        return p
    return ToolCallP(
        role="locate",
        tool=p.tool,
        args_sha=p.args_sha,
        outcome="served",
        result_tokens=p.result_tokens,
        dedup_hit=p.dedup_hit,
    )


# -- the projection -------------------------------------------------------------------------


def _of[P: BaseModel](
    lines: Iterable[Placed[BaseModel]], model: type[P]
) -> tuple[Placed[P], ...]:
    return tuple(n for n in (line.narrow(model) for line in lines) if n is not None)


@dataclass(frozen=True)
class Operation:
    """One effect's lifecycle, as its lines say it."""

    operation_id: str
    intended: EffectIntendedP | None
    """None on a damaged journal whose `effect_intended` line is missing: the operation is
    still seen, and the fold records the defect."""
    started: bool = False
    completed: EffectCompletedP | None = None
    task_id: str | None = None
    """The task a box step belongs to (`worker_step`); None for an effect outside
    a task (one of Athena's steps outside one, a bootstrap, Zeus's shell)."""

    @property
    def kind(self) -> str | None:
        """`effect_kind` as intended; None when the intent is missing."""
        return self.intended.effect_kind if self.intended is not None else None

    @property
    def surface(self) -> tuple[str, ...]:
        """What the operation declared it would touch; nothing when its intent is missing."""
        return tuple(self.intended.declared_surface) if self.intended is not None else ()

    @property
    def phase(self) -> Literal["intended", "started", "completed"]:
        if self.completed is not None:
            return "completed"
        return "started" if self.started else "intended"

    @property
    def outcome(self) -> Literal["succeeded", "failed", "ambiguous"] | None:
        return self.completed.outcome if self.completed is not None else None

    @property
    def needs_reconciliation(self) -> bool:
        """Started or intended and never completed: what reached the world is unknown until
        recovery looks (intended is never "safe to redo")."""
        return self.phase != "completed"


@dataclass(frozen=True)
class RunSpend:
    """The run's model spend, read once (decision 5) from its `model_call` lines."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    by_rung: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    """`model_call.rung`, or its role for a line that names none."""
    by_role: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    """`model_call.role`: Athena's and the worker's shares, from the same
    lines as `by_rung`."""
    by_task: Mapping[str, tuple[int, float]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Tokens (in and out) and cost per task: the calls made inside one."""
    floor_usd: float = 0.0
    """The largest run-scope `budget` line: the lineage's own amount, written with the charge
    and before its `model_call`."""

    @property
    def proven_usd(self) -> float:
        return max(self.cost_usd, self.floor_usd)


@dataclass(frozen=True)
class RunProjection:
    run_id: str
    last_seq: int
    start: RunStartP | None
    end: RunEndP | None
    """The run ended iff this is set."""
    operations: Mapping[str, Operation]
    """In the order each was intended; one whose intent is missing, at its first line."""
    spend: RunSpend
    defects: tuple[Defect, ...]
    """What the fold found inconsistent; never raised."""
    by_type: Mapping[type[BaseModel], tuple[Placed[BaseModel], ...]] = field(repr=False)
    tasks: Mapping[str, tuple[Placed[BaseModel], ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """The task loop's lines by task: every line naming a task, in order, in
    the order each task first appears. What `box/tasks.TaskBook` folds."""

    def lines[P: BaseModel](self, model: type[P]) -> tuple[Placed[P], ...]:
        """Every line of `model`, in order."""
        return _of(self.by_type.get(model, ()), model)

    def first[P: BaseModel](self, model: type[P]) -> P | None:
        found = self.lines(model)
        return found[0].payload if found else None

    def require_consistent(self) -> None:
        """`JournalCorrupt` naming the first defect: for the readers that refuse such a
        journal (`gc`, `Journal.open_for_resume`, `Box.resume`)."""
        if self.defects:
            first = self.defects[0]
            raise JournalCorrupt(first.seq, "<projection>", first.why)

    # The fields `ReplayState` folded and no production code read are questions over the
    # typed lines, not folded state.

    @property
    def isolation(self) -> str:
        return str(self.start.isolation) if self.start is not None else ""

    @property
    def cassetted_kinds(self) -> frozenset[str]:
        """Which kinds were read from recorded results rather than re-executed, so "state
        verified" is never mistaken for "re-run"."""
        kinds = {"exec": ExecP, "mcp_call": McpCallP}
        return frozenset(kind for kind, model in kinds.items() if self.lines(model))

    @property
    def external_effects(self) -> tuple[McpCallP, ...]:
        """MC-3: the handoff prints these as the only available undo list."""
        return tuple(line.payload for line in self.lines(McpCallP))

    @property
    def untrusted_reads(self) -> int:
        return len(self.lines(UntrustedReadP))

    @property
    def secret_denials(self) -> int:
        return len(self.lines(SecretDeniedP))

    @property
    def redactions(self) -> int:
        return sum(int(line.payload.count) for line in self.lines(RedactionP))


class Fold:
    """The one fold. Pure: no file, no clock, no randomness. Lines go in one at a time, so a
    reader can stop at damage and keep what it read, and the `Journal` folds each line after
    its durable write."""

    def __init__(self) -> None:
        self._run_id = ""
        self._last_seq = 0
        self._start: RunStartP | None = None
        self._end: RunEndP | None = None
        self._tasks: dict[str, list[Placed[BaseModel]]] = {}
        self._operations: dict[str, Operation] = {}
        self._by_type: dict[type[BaseModel], list[Placed[BaseModel]]] = {}
        self._defects: list[Defect] = []
        self._tokens_in = 0
        self._tokens_out = 0
        self._cost = 0.0
        self._by_rung: dict[str, float] = {}
        self._by_role: dict[str, float] = {}
        self._by_task: dict[str, tuple[int, float]] = {}
        self._floor = 0.0
        self._held: RunProjection | None = None

    def add(self, event: JournalEvent) -> None:
        self._held = None
        if self._end is not None:
            self._defects.append(Defect(event.seq, "event follows run_end"))
        if event.seq != self._last_seq + 1:
            self._defects.append(Defect(self._last_seq + 1, f"seq gap: got {event.seq}"))
        self._last_seq = event.seq
        self._run_id = self._run_id or event.run_id
        p = event.payload
        place: Placed[BaseModel] = Placed(event.seq, p, event.task_id, event.round)
        self._by_type.setdefault(type(p), []).append(place)
        if event.task_id is not None:
            self._tasks.setdefault(event.task_id, []).append(place)
        if isinstance(p, RunStartP):
            self._start = self._start or p
        elif isinstance(p, RunEndP):
            self._end = p
        elif isinstance(p, EffectIntendedP | EffectStartedP | EffectCompletedP):
            self._effect(event.seq, p, event.task_id)
        elif isinstance(p, ModelCallP):
            self._spend(event.task_id, p)
        elif isinstance(p, BudgetP) and p.scope == "run":
            self._floor = max(self._floor, float(p.spent_usd))

    def _effect(
        self,
        seq: int,
        p: EffectIntendedP | EffectStartedP | EffectCompletedP,
        task_id: str | None = None,
    ) -> None:
        oid = str(p.operation_id)
        op = self._operations.get(oid)
        if isinstance(p, EffectIntendedP):
            if op is None:
                self._operations[oid] = Operation(oid, p, task_id=task_id)
            elif op.intended is None:
                # Written after its start: out of order, a defect recorded at the start, and
                # still what the operation declared.
                self._operations[oid] = replace(op, intended=p)
            else:
                self._defects.append(Defect(seq, f"operation_id {oid} reused"))
            return
        # A line with no intent before it still opens the operation, so a damaged journal's
        # effect is seen by every reader; the defect makes the refusers refuse.
        if op is None:
            op = Operation(oid, None, task_id=task_id)
        if isinstance(p, EffectStartedP):
            if op.intended is None:
                self._defects.append(Defect(seq, "effect_started without intended"))
            self._operations[oid] = replace(op, started=True)
        else:
            if not op.started:
                self._defects.append(Defect(seq, "effect_completed without started"))
            self._operations[oid] = replace(op, completed=p)

    def _spend(self, task_id: str | None, p: ModelCallP) -> None:
        self._tokens_in += int(p.tokens_in)
        self._tokens_out += int(p.tokens_out)
        self._cost += float(p.cost_usd)
        rung = p.rung or p.role
        self._by_rung[rung] = self._by_rung.get(rung, 0.0) + float(p.cost_usd)
        self._by_role[p.role] = self._by_role.get(p.role, 0.0) + float(p.cost_usd)
        if task_id:
            tokens, cost = self._by_task.get(task_id, (0, 0.0))
            self._by_task[task_id] = (
                tokens + int(p.tokens_in) + int(p.tokens_out),
                cost + float(p.cost_usd),
            )

    def projection(self) -> RunProjection:
        """What the fold holds so far, immutable: a projection handed out never changes."""
        if self._held is None:
            self._held = RunProjection(
                run_id=self._run_id,
                last_seq=self._last_seq,
                start=self._start,
                end=self._end,
                operations=MappingProxyType(dict(self._operations)),
                spend=RunSpend(
                    tokens_in=self._tokens_in,
                    tokens_out=self._tokens_out,
                    cost_usd=self._cost,
                    by_rung=MappingProxyType(dict(self._by_rung)),
                    by_role=MappingProxyType(dict(self._by_role)),
                    by_task=MappingProxyType(dict(self._by_task)),
                    floor_usd=self._floor,
                ),
                defects=tuple(self._defects),
                by_type=MappingProxyType({t: tuple(ls) for t, ls in self._by_type.items()}),
                tasks=MappingProxyType({t: tuple(ls) for t, ls in self._tasks.items()}),
            )
        return self._held


def project(events: Iterable[JournalEvent]) -> RunProjection:
    fold = Fold()
    for event in events:
        fold.add(event)
    return fold.projection()


def project_before(events: Iterable[JournalEvent], seq: int) -> RunProjection:
    """The run as it stood before line `seq`: the prefix, folded (resume, at a boundary)."""
    return project(e for e in events if e.seq < seq)


def project_readable(events: Iterable[JournalEvent]) -> RunProjection:
    """Every line the stream yields before it raises, folded: a damaged journal still counts
    what it recorded (`budget/family.py`'s "up to any damage"). A torn UTF-8 tail is a
    `UnicodeDecodeError`, a `ValueError` that `read_events`' torn-tail repair does not catch,
    so it is caught here too."""
    fold = Fold()
    try:
        for event in events:
            fold.add(event)
    except (JournalError, OSError, ValueError):
        pass
    return fold.projection()
