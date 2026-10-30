"""One cut's block, built: the region before the boundary,
each element put into exactly one bucket (lifted, layer 2, dropped, layer 1, layer 3) and
rendered once.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chickadee.context.compaction.boundary import (
    Boundary,
    Decline,
    choose_boundary,
    fallback_boundary,
)
from chickadee.context.compaction.receipts import (
    Receipt,
    ToolExchange,
    collapse,
    dropped_text,
    exchanges_in,
    kept_text,
    recovery_line,
    supersede,
    superseded_text,
)
from chickadee.context.compaction.residue import Job, ProseTurn, ResidueOutcome, residue_text
from chickadee.context.compaction.trigger import Trigger, TriggerKind
from chickadee.context.freshness import stale_check
from chickadee.context.project import completed_units, project_unit
from chickadee.context.recovery import actual_recovery
from chickadee.context.request import (
    Piece,
    est_call,
    est_text,
    tool_line,
    tool_piece,
    turn_header,
    turn_piece,
)
from chickadee.context.session import ToolLine, Turn
from chickadee.context.standing import NoticeTrigger, escape_frames
from chickadee.errors import HarnessBug

if TYPE_CHECKING:
    from chickadee.context.project import RunRecords, Unit
    from chickadee.context.resolver import HandleResolver
    from chickadee.context.session import SessionState

__all__ = [
    "Cutter",
    "framed_block",
    "region_of",
]


@dataclass(frozen=True)
class _Region:
    previous: int
    boundary: int
    units: tuple[Unit, ...]
    job: Job
    declined: Decline | None


def region_of(state: SessionState, trigger: Trigger) -> _Region | Decline:
    """Steps 2-3. A normal boundary projects every earlier unit; a decline falls back to
    everything older than the recent turns. BOUNDARY_TOO_RECENT falls back only under
    MUST_FIT, which cannot decline: its alternative is no request.

    A fallback region projects the completed units that END inside it, by the rule a normal
    boundary uses. As §5.12 step 3 first read it projected none, so a finished run
    that happened to lie inside a fallback region lost its typed facts — goal, outcome,
    cost, the user's verbatim note — to receipts and a model's recollection, when layer 2
    renders them exactly and with no model call."""
    previous = state.boundary
    chosen = choose_boundary(state, previous)
    if isinstance(chosen, Boundary):
        return _Region(previous, chosen.turn, chosen.units, "residue", None)
    if chosen is Decline.BOUNDARY_TOO_RECENT and trigger.kind is not TriggerKind.MUST_FIT:
        return chosen
    turn = fallback_boundary(state, previous)
    if turn is None:
        return chosen
    units = tuple(u for u in completed_units(state, after=previous) if u.turn_end < turn)
    return _Region(previous, turn, units, "fallback", chosen)


def _lifted(state: SessionState, pinned: Collection[int], boundary: int) -> tuple[int, ...]:
    """The pinned bodies the cut re-emits: the CURRENTLY pinned set before the boundary,
    re-read at every cut and never carried forward. Those past it are still in the tail."""
    for seq in pinned:
        tool_line(state, seq)
    return tuple(sorted(s for s in set(pinned) if s < boundary))


class Cutter:
    """Steps 4-8: every element of the region lands in exactly one bucket, and each bucket's
    tokens freed are counted apart — one total would hide a model call that
    freed four hundred tokens. A bucket may free a negative amount, and says so."""

    def __init__(
        self,
        state: SessionState,
        region: _Region,
        *,
        pinned: Collection[int],
        resolver: HandleResolver,
        records: RunRecords | None,
        blob: Callable[[str], bytes],
    ) -> None:
        self.state = state
        self.region = region
        self.lo = max(region.previous, 1)
        self.hi = region.boundary - 1
        self.records = records
        self.blob = blob
        self.lifted = _lifted(state, pinned, region.boundary)
        self.freed = {"l1": 0, "l2": 0, "l3": 0, "dropped": 0}
        self.projections: list[str] = []
        self.entries: list[tuple[int, str]] = []
        self.prose: list[ProseTurn] = []
        self.objects = 0
        self.notices = 0
        self.elided: list[ProseTurn] = []
        self.refused: list[dict[str, object]] = []
        self.superseded: list[str] = []
        self.stale: set[str] = set()
        self._check = stale_check(state.handles.get, resolver)

    def run(self) -> None:
        self._project()
        self._walk()
        self._exchanges()

    def _subsumed(self, seq: int) -> bool:
        return any(u.turn_start <= seq <= u.turn_end for u in self.region.units)

    def _is_stale(self, handle_id: str) -> bool:
        # Step 4, evaluated where step 8 needs it: a stale handle is NONE, never EXACT.
        stale = self._check(handle_id)
        if stale:
            self.stale.add(handle_id)
        return stale

    def _project(self) -> None:
        for unit in self.region.units:
            text = project_unit(self.state, unit, self.records)
            self.projections.append(text)
            self.freed["l2"] -= est_text(text)

    def _walk(self) -> None:
        for _event, record in self.state.lines(self.lo, self.hi):
            if isinstance(record, Turn):
                if record.role == "harness":
                    self._notice(record)
                else:
                    self._prose(record)
            elif isinstance(record, ToolLine) and self.state.is_user_words(record.seq):
                # An ask_user answer is the user's own words: its exchange is
                # collapsed like any other, and layer 3 reads it as the prose it is.
                self.prose.append(ProseTurn(record.seq, "user", record.text))

    def _prose(self, turn: Turn) -> None:
        if turn.role == "user":
            self.prose.append(ProseTurn(turn.seq, "user", turn.text))
            shown = f"{turn_header(turn.seq)}\n{turn.text}"
            self._attachments(turn)
        elif turn.text.strip():
            self.prose.append(ProseTurn(turn.seq, "assistant", turn.text))
            shown = turn.text
        else:
            return
        self.objects += 1
        if not self._subsumed(turn.seq):
            # A unit's projection stands for its own turns; any other turn leaves the
            # block, and invariant 16 says the reader is told so.
            self.elided.append(self.prose[-1])
        self.freed["l2" if self._subsumed(turn.seq) else "l3"] += est_text(shown)

    def _notice(self, turn: Turn) -> None:
        """Notices go with the cut; the snapshot restates what still binds. A
        fork's effects are the exception: the snapshot restates constraints and concerns,
        and nothing restates what a fork could not undo — dropping it would be the lie by
        omission §5.1.1 forbids."""
        n = turn.notice
        if n is None:
            raise HarnessBug(f"harness turn {turn.seq} carries no notice")
        self.objects += 1
        tokens = est_text(n.framed())
        if n.trigger is NoticeTrigger.FORK_EFFECTS:
            text = f"[notice · seq {turn.seq} · kept, since nothing restates it]\n{n.text}"
            self.entries.append((turn.seq, text))
            self.freed["dropped"] += tokens - est_text(text)
        else:
            self.notices += 1
            self.freed["dropped"] += tokens

    def _attachments(self, turn: Turn) -> None:
        """A user turn's attachments leave with it, the first thing collapsed (as D110's images
        are): each keeps one line, its handle, so `read` shows it again."""
        if not turn.attachments:
            return
        sent = turn_piece(turn, self.blob).tokens - est_text(
            f"{turn_header(turn.seq)}\n{turn.text}"
        )
        rows = [
            f"[attached at seq {turn.seq} · {a.label} · "
            f'read(handle="{a.handle.id}", depth="full") shows it again]'
            for a in turn.attachments
        ]
        text = "\n".join(rows)
        self.entries.append((turn.seq, text))
        self.freed["dropped"] += sent - est_text(text)

    def _exchanges(self) -> None:
        """Lifted first, then layer 2, then supersede, then layer 1. Every exchange in the
        region is taken in this one cut, so D110's images-first order has nothing left to
        decide: an image is collapsed like any exchange, its pixels dropped whole with the
        receipt and never summarised."""
        loose: list[tuple[ToolExchange, int]] = []
        for ex in exchanges_in(self.state, self.lo, self.hi):
            if ex.result_seq in self.lifted:
                continue  # re-emitted intact after the snapshot; its call goes with the block
            self.objects += 1
            current = est_call(ex.tool, ex.args) + self._piece(ex).tokens
            if self._subsumed(ex.seq):
                self.freed["l2"] += current
            else:
                loose.append((ex, current))
        _kept, older = supersede([ex for ex, _ in loose])
        for ex, current in loose:
            newer = older.get(ex.call_id)
            if newer is not None:
                self._supersede(ex, newer, current)
            else:
                self._collapse(ex, current)

    def _piece(self, ex: ToolExchange) -> Piece:
        return tool_piece(self.state, tool_line(self.state, ex.result_seq), self.blob)

    def _supersede(self, ex: ToolExchange, newer: ToolExchange, current: int) -> None:
        text = superseded_text(ex, newer)
        self.superseded.append(ex.call_id)
        self.entries.append((ex.seq, text))
        self.freed["dropped"] += current - est_text(text)

    def _collapse(self, ex: ToolExchange, current: int) -> None:
        handle = self.state.handles.get(ex.handle) if ex.handle is not None else None
        recovery = actual_recovery(ex.tool, ex.handle, is_stale=self._is_stale)
        outcome = collapse(ex, recovery, handle)
        if isinstance(outcome, Receipt):
            text, bucket = outcome.render(), "l1"
        else:
            kept = "dropped" if ex.image else "verbatim"
            self.refused.append({"call_id": ex.call_id, "kept": kept})
            if ex.image:
                line = recovery_line(ex, recovery, handle)
                text, bucket = dropped_text(ex, line), "dropped"
            else:
                shown = self._piece(ex).message.content
                text = kept_text(ex, shown if isinstance(shown, str) else "")
                bucket = "l1"
        self.entries.append((ex.seq, text))
        self.freed[bucket] += current - est_text(text)

    def body(self, outcome: ResidueOutcome) -> str:
        runs = ", ".join(u.run_id for u in self.region.units)
        head = f"[earlier in this session · seq {self.lo}-{self.hi}"
        if runs:
            head += f" · runs {runs}"
        if self.notices:
            noun = "notice" if self.notices == 1 else "notices"
            head += f" · {self.notices} {noun} elided"
        rows = [head + "]", *self.projections]
        if self.elided:
            rows.append(_elided_turns(self.elided))
        rows.extend(text for _seq, text in sorted(self.entries, key=lambda e: e[0]))
        if outcome.items:
            section = residue_text(outcome)
            rows.append(section)
            self.freed["l3"] -= est_text(section)
        return "\n".join(rows)


def _elided_turns(turns: Sequence[ProseTurn]) -> str:
    """The omission marker for conversation turns the block does not carry. Their words are
    in the session log, verbatim; what this block says of them is only the residue's claim."""
    parts: list[str] = []
    for role in ("user", "assistant"):
        seqs = [t.seq for t in turns if t.role == role]
        if not seqs:
            continue
        shown = ", ".join(str(n) for n in seqs[:12]) + (", …" if len(seqs) > 12 else "")
        noun = "turn" if len(seqs) == 1 else "turns"
        parts.append(f"{len(seqs)} {role} {noun} (seq {shown})")
    return f"[{' and '.join(parts)} elided: not shown here, kept verbatim in the session log]"


def framed_block(body: str, objects: int, freed: int, boundary: int) -> str:
    """ONE message in the harness frame: without it, a projected line from seq 412
    reads as a fresh instruction at turn 41. The last line is what the model is told about
    the cut; it names the rebuild point, the one fact that explains an expensive next
    request."""
    noun = "object" if objects == 1 else "objects"
    footer = (
        f"[compacted · {objects} {noun} · est. {freed:,} tokens freed · "
        f"prefix rebuilt from turn {boundary}]"
    )
    return f"<harness_context>\n{escape_frames(body)}\n{footer}\n</harness_context>"
