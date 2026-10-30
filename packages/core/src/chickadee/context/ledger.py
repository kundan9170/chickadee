"""The constraint ledger and the rejected-alternatives record.

Two durable kinds of the session log, and the reason both are exempt from every compaction
layer: they are not messages at all. A constraint is a `constraint` line and a rejection is a
`rejected` line, so no layer can fold them away — *exempt* is a property of where they live,
not a protection on a message.

**A constraint binds; a rejection informs.** That is why they have different machinery. A
constraint carries validity endpoints and a supersession link, and the harness checks work
against it. A rejection carries a required reason and its age, binds nothing, and nothing
checks against it.

**Zeus writes the constraint; the user's turn is the authority.** `note_constraint`
refuses a constraint that cites no turn, or cites a turn the user did not say, so a model can
word a constraint but cannot invent one. The ledger stores the pointer, never a copy of the
user's words, which stay at their seq in the log. The one constraint whose text is the
user's own is a refusal's note, stored verbatim and rendered *(your words)*.

Every writer here reads the same way — validate, durable write, fold, then speech.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

from chickadee.context import standing
from chickadee.context.lines import (
    AUTHORS,
    Author,
    ConstraintP,
    PayloadError,
    Reconsideration,
    RejectedAlt,
    Scope,
)
from chickadee.context.results import InvalidArgument
from chickadee.limits import limit

if TYPE_CHECKING:
    from chickadee.context.session import Session, SessionState

__all__ = [
    "REJECTED_RENDER_MAX",
    "ZEUS_SCOPES",
    "Constraint",
    "LedgerLine",
    "LedgerState",
    "RejectedProposal",
    "RejectedRow",
    "binding_constraints",
    "ingest_rejected_proposals",
    "note_constraint",
    "note_rejected",
    "reconsider",
    "record_refusal_note",
    "rejected",
    "rejected_in",
    "unapplied_constraints",
]

ZEUS_SCOPES: Final[frozenset[str]] = frozenset({"session", "design"})
"""What `note_constraint` accepts. Zeus speaks in components and contracts, never in
packets, so it has no packet set to name."""

REJECTED_RENDER_MAX: Final = 6
"""Rejected-alternative rows rendered per unit by §5.9. A bound, not a target."""


def _number(record_id: str) -> int:
    """`C-12` → 12, so ids sort as numbers rather than as strings (`C-10` after `C-9`)."""
    return int(record_id.split("-", 1)[1])


# -- constraints --------------------------------------------------------------


@dataclass(frozen=True)
class Constraint:
    id: str
    text: str
    """`author == "zeus"`: Zeus's technical statement. `author == "user"`: the
    person's refusal note, VERBATIM."""
    sources: tuple[int, ...]
    """Session-log seqs of the user turns it came from; never empty. The user's own words
    stay retrievable at those seqs."""
    author: Author
    scope: Scope
    scope_id: str | None
    """Set only on D105's path, from the question's scope; None for every constraint Zeus
    writes."""
    valid_from: int
    valid_until: int | None
    """None = still binding."""
    superseded_by: str | None

    @property
    def live(self) -> bool:
        return self.valid_until is None

    @staticmethod
    def of(line: ConstraintP) -> Constraint:
        """The ledger's row for a `constraint` line: the line without its `supersedes`."""
        return Constraint(
            id=line.id,
            text=line.text,
            sources=line.sources,
            author=line.author,
            scope=line.scope,
            scope_id=line.scope_id,
            valid_from=line.valid_from,
            valid_until=line.valid_until,
            superseded_by=line.superseded_by,
        )

    def line(self, supersedes: str | None = None) -> ConstraintP:
        """The `constraint` line that records this row, ending `supersedes` if it names one."""
        return ConstraintP(
            id=self.id,
            text=self.text,
            sources=self.sources,
            author=self.author,
            scope=self.scope,
            scope_id=self.scope_id,
            valid_from=self.valid_from,
            valid_until=self.valid_until,
            superseded_by=self.superseded_by,
            supersedes=supersedes,
        )


# -- rejected alternatives -------------------------------------------------------


LedgerLine = Constraint | RejectedAlt | Reconsideration


@dataclass(frozen=True)
class RejectedRow:
    """A rejection as §5.9 renders it: the record, and whether it was later reconsidered —
    a mark on the row, never a removal."""

    alt: RejectedAlt
    reconsidered_at: int | None


# -- the fold -----------------------------------------------------------------------------


class LedgerState:
    """The constraint and rejected lines of one session, folded.

    A constraint line either introduces an id or, carrying `supersedes`, also ends the live
    constraint it names — one line, so a crash can never leave the old and the new both
    binding. Nothing is ever deleted: a superseded row keeps its text, gains `valid_until`
    and `superseded_by`, and still explains why earlier work avoided what it forbade.
    """

    def __init__(self) -> None:
        self._constraints: dict[str, Constraint] = {}
        self._rejected: dict[str, RejectedAlt] = {}
        self._reconsidered: dict[str, list[Reconsideration]] = {}

    # -- checking (pure) ------------------------------------------------------------------

    def check_constraint(self, seq: int, line: ConstraintP) -> tuple[Constraint, str | None]:
        """The new row and the id it supersedes, validated against the ledger as it stands.
        The same checks run before a write and on replay, so a line this refuses is never
        written."""
        c = Constraint.of(line)
        supersedes = line.supersedes
        if c.id in self._constraints:
            raise PayloadError(f"constraint {c.id} is already recorded")
        if not c.text:
            raise PayloadError("a constraint says something")
        if not c.sources or list(c.sources) != sorted(set(c.sources)):
            raise PayloadError("sources are a non-empty, sorted set of seqs")
        if max(c.sources) >= seq:
            raise PayloadError("a constraint cites turns before it, never itself or later")
        if c.valid_from != seq or c.valid_until is not None or c.superseded_by is not None:
            raise PayloadError("a constraint begins binding at its own line, and is live")
        if c.author == "zeus" and (c.scope not in ZEUS_SCOPES or c.scope_id is not None):
            raise PayloadError("Zeus writes session or design constraints, with no scope id")
        if supersedes is not None:
            old = self._constraints.get(supersedes)
            if old is None or not old.live:
                raise PayloadError(f"{supersedes} is not a live constraint to supersede")
        return c, supersedes

    def check_rejected(
        self, seq: int, line: RejectedAlt | Reconsideration
    ) -> RejectedAlt | Reconsideration:
        if isinstance(line, RejectedAlt):
            if line.id in self._rejected:
                raise PayloadError(f"rejection {line.id} is already recorded")
            if not line.text.strip() or not line.reason.strip():
                raise PayloadError("a rejection names what was declined and why")
            if not 1 <= line.at < seq:
                raise PayloadError(f"a rejection cites a turn before its line, not {line.at}")
            return line
        if line.rejected_id not in self._rejected:
            raise PayloadError(f"no rejection {line.rejected_id} to reconsider")
        if not line.reason.strip():
            raise PayloadError("a reconsideration says why")
        return replace(line, seq=seq)

    # -- folding --------------------------------------------------------------------------

    def apply(self, line: LedgerLine, *, supersedes: str | None = None) -> None:
        if isinstance(line, Constraint):
            self._constraints[line.id] = line
            if supersedes is not None:
                self._constraints[supersedes] = replace(
                    self._constraints[supersedes],
                    valid_until=line.valid_from,
                    superseded_by=line.id,
                )
        elif isinstance(line, RejectedAlt):
            self._rejected[line.id] = line
        else:
            self._reconsidered.setdefault(line.rejected_id, []).append(line)

    # -- reading --------------------------------------------------------------------------

    def get(self, constraint_id: str) -> Constraint | None:
        return self._constraints.get(constraint_id)

    def constraints(self) -> tuple[Constraint, ...]:
        """Every constraint ever recorded, live or superseded, by id."""
        return tuple(sorted(self._constraints.values(), key=lambda c: _number(c.id)))

    def live_constraints(self) -> tuple[Constraint, ...]:
        return tuple(c for c in self.constraints() if c.live)

    def next_constraint_id(self) -> str:
        return f"C-{len(self._constraints) + 1}"

    def rejections(self) -> tuple[RejectedAlt, ...]:
        return tuple(sorted(self._rejected.values(), key=lambda r: _number(r.id)))

    def get_rejected(self, rejected_id: str) -> RejectedAlt | None:
        return self._rejected.get(rejected_id)

    def reconsiderations(self, rejected_id: str) -> tuple[Reconsideration, ...]:
        return tuple(self._reconsidered.get(rejected_id, ()))

    def next_rejected_id(self) -> str:
        return f"R-{len(self._rejected) + 1}"


# -- note_constraint ---------------------------------------------------------


def note_constraint(
    session: Session,
    text: str,
    sources: Sequence[int],
    scope: str,
    supersedes: str | None = None,
) -> str | InvalidArgument:
    """The one way Zeus writes a constraint. Every refusal is `INVALID_ARGUMENTS` naming the
    field — nothing is written unless all of it holds.

    `sources` is the tool's `from`, renamed only because `from` is a Python keyword."""
    text = text.strip()
    cap = int(limit("zeus.constraint_text_max").value)
    if not text or len(text) > cap:
        return InvalidArgument(
            "text",
            f"a constraint is 1 to {cap} characters; this one is {len(text)}",
            do=f"send it again as one rule of at most {cap} characters",
        )
    if not sources:
        return InvalidArgument(
            "from",
            "a constraint must come from something the user said",
            do="send from= with the number of the user's turn that states it",
        )
    with session.lock:
        state = session.state
        for n in sources:
            if not state.has_line(n):
                return InvalidArgument(
                    "from",
                    f"turn {n} does not exist",
                    do="send from= with the number of a turn the user said",
                )
            if not state.is_user_words(n):
                return InvalidArgument(
                    "from",
                    f"turn {n} was not said by the user",
                    do="cite the user's turn that states it; a constraint you inferred is "
                    "not one",
                )
        if scope not in ZEUS_SCOPES:
            return InvalidArgument(
                "scope",
                "a constraint Zeus writes is session or design scope",
                do='send scope as "session" or "design"',
            )
        cited = tuple(sorted(set(sources)))
        if supersedes is not None:
            old = state.ledger.get(supersedes)
            if old is None or not old.live:
                return InvalidArgument(
                    "supersedes",
                    "no such live constraint",
                    do="name a live constraint's id, or leave supersedes out to add a new one",
                )
            if max(cited) <= max(old.sources):
                return InvalidArgument(
                    "supersedes",
                    "only a later turn can change it",
                    do="cite a user turn later than the ones the constraint came from",
                )
        c = Constraint(
            id=state.ledger.next_constraint_id(),
            text=text,
            sources=cited,
            author="zeus",
            scope=cast("Scope", scope),
            scope_id=None,
            valid_from=session.next_seq,
            valid_until=None,
            superseded_by=None,
        )
        session.append(c.line(supersedes))
        standing.notice(session, standing.constraint_recorded(c, supersedes=supersedes))
        return c.id


def record_refusal_note(
    session: Session,
    note: str,
    answer_seq: int,
    *,
    scope: Scope = "session",
    scope_id: str | None = None,
) -> str | None:
    """A refusal (`skip`, `stop`, `deny`, `cancel`) carrying a note becomes a
    constraint in the person's own words, appended BEFORE the typed refusal returns.

    Not a paraphrase and not an extraction — the person typed it at the prompt, so the
    text is stored as typed and no length cap applies: cutting it would reword it. An empty
    note binds nothing and returns None. A note on an approval is recorded on the answer and
    never reaches here."""
    if not note.strip():
        return None
    with session.lock:
        if not session.state.has_line(answer_seq):
            raise ValueError(f"answer seq {answer_seq} is not a line of this session")
        c = Constraint(
            id=session.state.ledger.next_constraint_id(),
            text=note,
            sources=(answer_seq,),
            author="user",
            scope=scope,
            scope_id=scope_id,
            valid_from=session.next_seq,
            valid_until=None,
            superseded_by=None,
        )
        session.append(c.line())
        standing.notice(session, standing.constraint_recorded(c))
        return c.id


def binding_constraints(
    state: SessionState, cited_ids: Iterable[str]
) -> tuple[Constraint, ...] | InvalidArgument:
    """The set a `coder` call carries, assembled by the harness, not Zeus.

    Session-scope constraints bind every design and every T0 task without being cited;
    design-scope ones bind when the work cites them. A cited id that is not a live
    constraint is refused. Ordered by id."""
    ledger = state.ledger
    bound = {c.id: c for c in ledger.live_constraints() if c.scope == "session"}
    for cid in cited_ids:
        c = ledger.get(cid)
        if c is None or not c.live:
            return InvalidArgument(
                "constraints",
                f"{cid} is not a live constraint",
                do="name only live constraints' ids, or leave it out",
            )
        if c.scope == "design":
            bound[c.id] = c
    return tuple(sorted(bound.values(), key=lambda c: _number(c.id)))


def unapplied_constraints(
    state: SessionState, bound: Iterable[Constraint]
) -> tuple[Constraint, ...]:
    """Live design-scope constraints the work does not cite — named *not applied* on the
    approval prompt or the `coder` transcript line. Visible, never refused."""
    held = {c.id for c in bound}
    return tuple(
        c for c in state.ledger.live_constraints() if c.scope == "design" and c.id not in held
    )


# -- note_rejected -------------------------------------------------------------


def note_rejected(
    session: Session,
    *,
    text: str,
    reason: str,
    by: str,
    at: int,
    scope: Scope = "session",
    scope_id: str | None = None,
    recorded_by: Literal["explicit", "checkpoint"] = "explicit",
) -> str | InvalidArgument:
    """Record that an alternative was declined, and why. `at` is the turn it came from —
    for the explicit path, the turn that made the call. Never judges whether the rejection
    was right; it records that it happened."""
    if not reason.strip():
        return InvalidArgument(
            "reason",
            "a rejection without a reason is a preference stated as history",
            do="send reason: why it was rejected",
        )
    if not text.strip():
        return InvalidArgument(
            "text",
            "the text is empty",
            do="send text naming the alternative that was considered",
        )
    if by not in AUTHORS:
        return InvalidArgument(
            "by", "a rejection is noted by Zeus or the user", do='send by as "zeus" or "user"'
        )
    with session.lock:
        if not session.state.has_line(at):
            return InvalidArgument(
                "at",
                f"turn {at} does not exist",
                do="send at= with the turn it was rejected in",
            )
        alt = RejectedAlt(
            id=session.state.ledger.next_rejected_id(),
            at=at,
            by=by,
            text=text,
            reason=reason,
            scope=scope,
            scope_id=scope_id,
            recorded_by=recorded_by,
        )
        session.append(alt)
        return alt.id


class RejectedProposal(Protocol):
    """A checkpoint's `RejectedAltProposal`, structurally: the checkpoint module
    owns the type, and this only reads its three fields."""

    @property
    def text(self) -> str: ...
    @property
    def reason(self) -> str: ...
    @property
    def source_seq(self) -> int: ...


def ingest_rejected_proposals(
    session: Session, proposals: Iterable[RejectedProposal]
) -> tuple[str, ...]:
    """The extraction path converges on the explicit one: same record, `recorded_by` the
    only difference. A proposal that cannot say WHY is dropped — rule 1 binds the cheap
    rung too, and a checkpoint with no reason has found a topic, not a rejection."""
    ids: list[str] = []
    for p in proposals:
        if not p.reason.strip():
            continue
        noted = note_rejected(
            session,
            text=p.text,
            reason=p.reason,
            by="zeus",
            at=p.source_seq,
            recorded_by="checkpoint",
        )
        if isinstance(noted, str):
            ids.append(noted)
    return tuple(ids)


def reconsider(
    session: Session, rejected_id: str, reason: str, by: str
) -> int | InvalidArgument:
    """The only thing that changes a rejection, and only Zeus or the user may: judging
    that a past rejection no longer applies is a design decision, never a cheap rung's.
    Returns the seq of the appended line."""
    if by not in AUTHORS:
        return InvalidArgument(
            "by",
            "only Zeus or the user reconsiders a rejection",
            do='send by as "zeus" or "user"',
        )
    if not reason.strip():
        return InvalidArgument(
            "reason", "the reason is empty", do="send reason: why it is being reopened"
        )
    with session.lock:
        if session.state.ledger.get_rejected(rejected_id) is None:
            return InvalidArgument(
                "id", f"no rejection {rejected_id}", do="name a rejection's id as it was noted"
            )
        line = Reconsideration(rejected_id, reason, by)
        return session.append(line).seq


def rejected_in(state: SessionState, lo: int, hi: int) -> tuple[tuple[RejectedRow, ...], int]:
    """§5.15.3 step 4, for §5.9: the rejections whose `at` falls in `(lo, hi]`, at most
    `REJECTED_RENDER_MAX`, and how many there were in all — so a cut row is counted, never
    silently missing (invariant 16). A later reconsideration is a mark on its row."""
    ledger = state.ledger
    inside = [r for r in ledger.rejections() if lo < r.at <= hi]
    rows = tuple(
        RejectedRow(r, next((x.seq for x in ledger.reconsiderations(r.id)), None))
        for r in inside[:REJECTED_RENDER_MAX]
    )
    return rows, len(inside)


def rejected(state: SessionState, *, scope: Scope | None = None) -> tuple[RejectedRow, ...]:
    """`rejected(all=True)` or `rejected(scope=…)`: enumeration, not search — you cannot
    search for what you have forgotten."""
    ledger = state.ledger
    return tuple(
        RejectedRow(r, next((x.seq for x in ledger.reconsiderations(r.id)), None))
        for r in ledger.rejections()
        if scope is None or r.scope == scope
    )
