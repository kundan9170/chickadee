"""Layer 2 — a completed work unit, rendered from typed fields.

> Render facts from typed fields. Quote humans verbatim. Generate nothing.

No model call, and the same unit renders to the same bytes every time: the journal is
append-only and a finished run's events never change. An absent field renders absent —
never guessed — and a journal that cannot be read is said to be unavailable, never filled in.

**A unit runs from the previous RUN_END to this RUN_END**, so units are contiguous and
every line belongs to exactly one: nothing can fall between two blocks and be lost. A unit
is located in SESSION seqs. A run journal's seqs order one run's events and say nothing about
where in the conversation they fell, so the session lines that bracket the run are what
place it:

    (seq_open....... seq_decision seq_run_start....... seq_end]
     previous the turn that its `episode` the line where the
     unit's end called `coder` line run's result arrived

The design half ends at the dispatch and holds the goal; the execution half runs to the result.
The two halves are two renderers feeding ONE block — never two cuts, which would pay the rebuild
twice on the same suffix.

**The execution half is a stub since the rebuild**: the run's outcome, its
counts and its cost, and the handles. The packet loop's rows (route, plan decision, packets,
checkpoints, escalations) are parked as they stood, and the task loop's own rows
wait for the compaction that would read them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from chickadee.context.ledger import rejected_in
from chickadee.context.standing import turns_phrase
from chickadee.journal.errors import JournalError
from chickadee.journal.paths import blobs_dir, runs_root
from chickadee.journal.projection import RunProjection
from chickadee.journal.reader import read_run
from chickadee.journal.runs import RunProjections
from chickadee.schema.journal import RunEndP, RunStartP

if TYPE_CHECKING:
    from chickadee.context.lines import EpisodeLine
    from chickadee.context.session import SessionState
    from chickadee.context.spans import Span
    from chickadee.state.roots import Roots

__all__ = [
    "JournalRecords",
    "RunRecords",
    "Unit",
    "completed_units",
    "paused_runs",
    "project_unit",
    "run_ended",
    "unit_episodes",
    "unit_handles",
]


class RunRecords(Protocol):
    """Where a run's journal is read from. `projection` raises `FileNotFoundError` when the
    run has no journal, so an absent journal is never mistaken for an empty one, and
    `journal.errors.JournalError` when it will not read; `blob` raises
    `journal.errors.BlobMissing`, whose `why` says whether a person pruned it."""

    def projection(self, run_id: str) -> RunProjection:
        """The run's journal folded once: what a reader of a finished run reads."""
        ...

    def blob(self, run_id: str, sha: str) -> bytes: ...


class JournalRecords:
    """`RunRecords` over the repo-local run journals. A run's projection
    is the engine's `RunProjections`', each journal folded once."""

    def __init__(self, roots: Roots, runs: RunProjections) -> None:
        self._roots = roots
        self._runs = runs

    def projection(self, run_id: str) -> RunProjection:
        return self._runs.projection(run_id)

    def blob(self, run_id: str, sha: str) -> bytes:
        from chickadee.journal.gc import blob_missing

        target = blobs_dir(self._roots, run_id) / sha
        try:
            return target.read_bytes()
        except FileNotFoundError:
            raise blob_missing(runs_root(self._roots), run_id, sha) from None

    def journal_text(self, run_id: str, seqs: Span | None) -> bytes:
        """A journal handle's bytes: the events, one JSON line each, in the range asked.
        The lines as written, not a fold of them: a handle shows the journal itself."""
        events = tuple(read_run(self._roots, run_id))
        if seqs is not None:
            events = tuple(e for e in events if seqs.start <= e.seq <= seqs.end)
        return "".join(e.model_dump_json() + "\n" for e in events).encode("utf-8")


# -- units ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """One completed work unit: `(previous RUN_END | session start, this RUN_END]`."""

    run_id: str
    seq_open: int
    """The previous unit's end, or 0 at session start."""
    seq_run_start: int
    """The `episode` line — the design half ends here."""
    seq_decision: int
    """The turn that called `coder`: the design or the one task is approved at dispatch, at
    stop 1, inside the design half."""
    seq_end: int
    """The line where the run's result arrived."""
    turn_start: int
    """The unit's first line: `seq_open + 1`."""
    turn_end: int
    """Its last: `seq_end`."""


def unit_episodes(state: SessionState) -> tuple[EpisodeLine, ...]:
    """The episodes that open a unit of the conversation the model can see: Zeus's, from a
    turn no rewind hides. A host's run is no unit of Zeus's, and a run whose turn was
    taken back still happened, so its episode is kept, but it opens no unit."""
    return tuple(
        e
        for e in state.episodes()
        if e.opened_at_seq is not None and state.turn(e.opened_at_seq) is not None
    )


def completed_units(state: SessionState, *, after: int = 0) -> tuple[Unit, ...]:
    """Every completed unit whose end falls at or after `after`.

    A unit is complete when its run's result has arrived in the session — the `coder`
    result line, which carries the run's id — and every call made before that line has its
    own result, so a unit only ever ends where a turn could begin. One not yet
    projected is one whose end has not entered a cut; a unit a fallback cut split keeps its
    facts for the cut that reaches its end, since those come from the journal rather than
    from the turns it lost."""
    units: list[Unit] = []
    previous_end = 0
    ends = _result_seqs(state)
    for episode in unit_episodes(state):
        assert episode.opened_at_seq is not None  # unit_episodes keeps Zeus's alone
        result = ends.get(episode.run_id)
        end = _settled(state, previous_end, result) if result is not None else None
        if end is None:
            break  # a unit is complete only once every earlier one is
        unit = Unit(
            run_id=episode.run_id,
            seq_open=previous_end,
            seq_run_start=episode.seq,
            seq_decision=episode.opened_at_seq,
            seq_end=end,
            turn_start=previous_end + 1,
            turn_end=end,
        )
        previous_end = end
        if unit.seq_end >= after:
            units.append(unit)
    return tuple(units)


def _settled(state: SessionState, after: int, seq: int) -> int | None:
    """The first seq at or after `seq` with no call made in `(after, it]` still waiting —
    None while one still waits."""
    end = seq
    moved = True
    while moved:
        moved = False
        for call in state.calls_upto(end):
            if call.seq <= after:
                continue
            result = state.result(call.call_id)
            if result is None:
                return None
            if result.seq > end:
                end, moved = result.seq, True
    return end


def paused_runs(state: SessionState) -> tuple[str, ...]:
    """The runs whose `coder` results all said only that the run paused, oldest first:
    none of them carries a typed outcome. A run that reported one (a harness failure's
    `BLOCKED`) is not paused. Whether each has ended since is `run_ended`'s."""
    from chickadee.context.session import ToolLine

    reported: dict[str, bool] = {}
    for _event, record in state.lines():
        if isinstance(record, ToolLine) and record.run_id is not None:
            told = record.outcome is not None
            reported[record.run_id] = reported.get(record.run_id, False) or told
    return tuple(run_id for run_id, told in reported.items() if not told)


def run_ended(records: RunRecords, run_id: str) -> bool | None:
    """Whether run `run_id`'s journal has its `run_end`; None when it cannot be read, which
    says nothing either way (`resume` reports a damaged one)."""
    try:
        return records.projection(run_id).end is not None
    except (FileNotFoundError, JournalError, OSError, ValueError):
        return None


def _result_seqs(state: SessionState) -> dict[str, int]:
    """run id → the seq of the `coder` result that carried it."""
    from chickadee.context.session import CODER, ToolLine

    ends: dict[str, int] = {}
    for _event, record in state.lines():
        if isinstance(record, ToolLine) and record.run_id is not None:
            call = state.call(record.call_id)
            if call is not None and call.tool == CODER:
                ends.setdefault(record.run_id, record.seq)
    return ends


def unit_handles(state: SessionState, unit: Unit) -> tuple[tuple[str, str], ...]:
    """`(kind, handle id)` for every handle the unit minted that must be carried: all but
    file handles, which re-derive from disk. A `log` or `output` blob existed
    nowhere else, and an exchange this projection subsumes writes no receipt — so without
    this row it would be real, on disk, and unnameable."""
    from chickadee.context.session import ToolLine

    seen: dict[str, str] = {}
    for _event, record in state.lines(unit.turn_start, unit.turn_end):
        handle = record.handle if isinstance(record, ToolLine) else None
        if handle is not None and handle.kind != "file":
            seen.setdefault(handle.id, handle.kind)
    return tuple((kind, hid) for hid, kind in seen.items())


# -- rendering ------------------------------------------------------------------------------

_LABEL_WIDTH = 12


def _row(label: str, text: str) -> str:
    """One labelled row; a multi-line value keeps every line, indented under the first."""
    pad = " " * (4 + _LABEL_WIDTH)
    lines = text.split("\n")
    head = f"    {label:<{_LABEL_WIDTH}}{lines[0]}"
    return "\n".join([head, *(pad + line for line in lines[1:])])


def _quote(text: str) -> str:
    return f'"{text}"'


def _money(usd: float) -> str:
    return f"${usd:,.2f}"


def _seconds(ms: int) -> str:
    return f"{ms / 1000:,.1f}s"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def project_unit(state: SessionState, unit: Unit, records: RunRecords | None) -> str:
    """The unit as one FACT block's worth of rows. Deterministic: every value comes from
    a typed field or a verbatim blob, in a fixed order."""
    rows = [f"▸ run {unit.run_id} · seq {unit.turn_start}-{unit.turn_end}"]
    run, why = _journal(unit.run_id, records)
    if run is None:
        rows.append(_row("journal", f"unavailable ({why}) — its facts are not shown"))
    else:
        rows.extend(_design_half(run))
    rows.extend(_session_rows(state, unit))
    if run is not None:
        rows.extend(_execution_half(run))
    handles = unit_handles(state, unit)
    if handles:
        rows.append(_row("handles", " · ".join(f"{kind} {hid}" for kind, hid in handles)))
    return "\n".join(rows)


def _journal(run_id: str, records: RunRecords | None) -> tuple[RunProjection | None, str]:
    if records is None:
        return None, "its journal was not consulted"
    try:
        return records.projection(run_id), ""
    except FileNotFoundError:
        return None, "no journal"
    except (JournalError, OSError) as exc:
        return None, f"journal unreadable: {exc}"


def _design_half(run: RunProjection) -> list[str]:
    start = run.first(RunStartP)
    return [_row("goal", _quote(start.goal))] if start is not None else []


def _session_rows(state: SessionState, unit: Unit) -> list[str]:
    """What the session log holds about the unit: the constraints its dispatch bound, the
    rejections made in it, the concerns raised in it."""
    rows: list[str] = []
    episode = state.episode(unit.run_id)
    if episode is not None:
        # The ledger rows, not their ids: standing restates only what still binds,
        # so a constraint superseded later would otherwise leave this block naming an id
        # whose text and source turns no longer appear anywhere in context.
        for n, cid in enumerate(episode.constraints):
            rows.append(_row("constraints" if n == 0 else "", _constraint_row(state, cid)))
    rejected, total = rejected_in(state, unit.seq_open, unit.seq_end)
    for r in rejected:
        mark = f"  [reconsidered at seq {r.reconsidered_at}]" if r.reconsidered_at else ""
        text = f"{r.alt.id} at seq {r.alt.at}: {_quote(r.alt.text)} — {_quote(r.alt.reason)}"
        rows.append(_row("rejected", text + mark))
    if total > len(rejected):
        rows.append(_row("", f"+{total - len(rejected)} more · rejected(all=True)"))
    # The id, always and only: the full text in a frozen block would be a permanent
    # decision made on a temporary condition.
    raised = [c.id for c in state.concerns.all() if unit.seq_open < c.raised_at <= unit.seq_end]
    if raised:
        rows.append(_row("raised", " · ".join(raised)))
    return rows


def _constraint_row(state: SessionState, cid: str) -> str:
    c = state.ledger.get(cid)
    if c is None:
        return f"{cid} [not in the ledger]"
    status = f" · superseded by {c.superseded_by}" if c.superseded_by is not None else ""
    return f"{c.id} ({c.author}, from {turns_phrase(c.sources)}): {_quote(c.text)}{status}"


def _execution_half(run: RunProjection) -> list[str]:
    end = run.first(RunEndP)
    if end is None:
        return []
    rows = [
        _row(
            "outcome",
            f"{end.outcome} · {_count(end.tasks, 'task')} ({end.committed} committed) · "
            f"{_money(end.cost_usd)} · {_seconds(end.wall_ms)}",
        )
    ]
    if end.why is not None:
        rows.append(_row("why", _quote(end.why)))
    return rows
