"""Concerns: advisory findings with a durable, anchored home.

The architecture says advisories are *kept and kept powerless*, and gave them one
line in a report that compaction folds away. This is the *kept* half: a `concern` line in the
session log, never in `messages[]`.

Two rules define the category:

1. **Anchored, or it is not a concern.** An advisory that names no path can never surface,
   so `note_concern` with an empty anchor is refused rather than stored.
2. **A concern is never created by reading the conversation back.** Only an explicit call
   creates one — there is no extractor here, so the re-extraction loop that filled a market
   memory layer with 808 copies of one hallucination cannot form.

**There is no status field, and nothing is mutated**. A concern's state is four kinds
of append-only line — raised, touched, promoted, dismissed — and every question is computed
from them. *"Is this resolved?"* has no answerable form; *"is it relevant to what we are
touching?"* is a path intersection, and that is what surfaces it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Final, Literal, Protocol

from chickadee.context import standing
from chickadee.context.lines import (
    RAISERS,
    Concern,
    ConcernLine,
    Dismissed,
    PayloadError,
    Promoted,
    Touched,
)
from chickadee.context.results import InvalidArgument
from chickadee.errors import HarnessBug

if TYPE_CHECKING:
    from chickadee.context.session import Session, SessionState

__all__ = [
    "CODER_CONCERN_CAP",
    "BoxConcernLike",
    "ConcernState",
    "CriticVariant",
    "accrue",
    "concern_anchors",
    "concerns",
    "concerns_for_critic",
    "dismiss",
    "ingest_report_concerns",
    "note_concern",
    "promote",
]


CriticVariant = Literal["pre_code", "post_code"]

CODER_CONCERN_CAP: Final = 3
"""`coder_concern_cap`'s default. The value is config (`.chickadee/config`), and it is
the BOX that enforces it before the report leaves; the harness only checks that it
did."""


# -- the fold -----------------------------------------------------------------------------


class ConcernState:
    def __init__(self) -> None:
        self._raised: dict[str, Concern] = {}
        self._touched: dict[str, list[Touched]] = {}
        self._promoted: dict[str, Promoted] = {}
        self._dismissed: dict[str, Dismissed] = {}

    def check(
        self, seq: int, run_id: str | None, line: Concern | Touched | Promoted | Dismissed
    ) -> Concern | Touched | Promoted | Dismissed:
        if isinstance(line, Concern):
            if line.id in self._raised:
                raise PayloadError(f"concern {line.id} is already raised")
            anchored = bool(line.anchor) and all(a.strip() for a in line.anchor)
            if not line.text.strip() or not anchored:
                raise PayloadError("a concern says something, about at least one path")
            return replace(line, raised_at=seq, run_id=run_id)
        cid = line.concern_id
        if cid not in self._raised:
            raise PayloadError(f"no concern {cid}")
        if isinstance(line, Touched) and line.path not in self._raised[cid].anchor:
            raise PayloadError(f"{line.path} is not an anchor of {cid}")
        if isinstance(line, Promoted) and cid in self._promoted:
            raise PayloadError(f"{cid} is already watched by a check")
        if isinstance(line, Dismissed) and cid in self._dismissed:
            raise PayloadError(f"{cid} is already dismissed")
        return replace(line, seq=seq)

    def apply(self, line: ConcernLine) -> None:
        if isinstance(line, Concern):
            self._raised[line.id] = line
        elif isinstance(line, Touched):
            self._touched.setdefault(line.concern_id, []).append(line)
        elif isinstance(line, Promoted):
            self._promoted[line.concern_id] = line
        else:
            self._dismissed[line.concern_id] = line

    # -- reading --------------------------------------------------------------------------

    def get(self, concern_id: str) -> Concern | None:
        return self._raised.get(concern_id)

    def all(self) -> tuple[Concern, ...]:
        """Every concern ever raised, in the order raised."""
        return tuple(sorted(self._raised.values(), key=lambda c: c.raised_at))

    def is_live(self, concern_id: str) -> bool:
        """`raised - dismissed - promoted`."""
        return (
            concern_id in self._raised
            and concern_id not in self._dismissed
            and concern_id not in self._promoted
        )

    def live(self) -> tuple[Concern, ...]:
        return tuple(c for c in self.all() if self.is_live(c.id))

    def anchors(self) -> tuple[str, ...]:
        """The distinct anchor paths of live concerns, in the order first raised — so the
        same paths render in the same order across many turns and the notice stays
        byte-identical. Every path is returned; the renderer bounds them at `SURFACE_MAX`
        and counts the rest, since an elision nobody can see is a hole (invariant 16)."""
        seen: dict[str, None] = {}
        for c in self.live():
            for path in c.anchor:
                seen.setdefault(path, None)
        return tuple(seen)

    def touched(self, concern_id: str) -> tuple[Touched, ...]:
        return tuple(self._touched.get(concern_id, ()))

    def promoted(self, concern_id: str) -> Promoted | None:
        return self._promoted.get(concern_id)

    def dismissed(self, concern_id: str) -> Dismissed | None:
        return self._dismissed.get(concern_id)

    def next_id(self) -> str:
        return f"K-{len(self._raised) + 1}"


# -- 1 · raise ----------------------------------------------------------------------------


def note_concern(
    session: Session,
    text: str,
    anchor: Sequence[str],
    by: str,
    *,
    run_id: str | None = None,
) -> str | InvalidArgument:
    """§5.13.4 step 1. The text is stored exactly as given — the raiser's own words.

    Refused, not stored, when the anchor is empty: the finding stays advisory-only. A
    notice follows only for anchor paths no live concern already surfaces (trigger 2), so
    ten concerns on one file produce one line."""
    paths = tuple(dict.fromkeys(a.strip() for a in anchor if a.strip()))
    if not paths:
        return InvalidArgument(
            "anchor",
            "an advisory naming no path can never surface",
            do="send anchor with the path the concern is about",
        )
    if not text.strip():
        return InvalidArgument(
            "text", "the text is empty", do="send text saying what the worry is"
        )
    if by not in RAISERS:
        return InvalidArgument(
            "by",
            f"a concern is raised by one of {list(RAISERS)}",
            do="send by as one of them",
        )
    with session.lock:
        state = session.state
        surfaced = set(state.concerns.anchors())
        c = Concern(
            id=state.concerns.next_id(),
            raised_by=by,
            text=text,
            anchor=paths,
            raised_at=session.next_seq,
            run_id=run_id,
        )
        session.append(c, run_id=run_id)
        new_paths = [p for p in paths if p not in surfaced]
        if new_paths:
            standing.notice(session, standing.concern_raised(c.id, new_paths))
        return c.id


# -- 2 · ingest from the box ---------------------------------------------------------------


class BoxConcernLike(Protocol):
    """`CoderReport.concerns`' element, structurally."""

    @property
    def text(self) -> str: ...
    @property
    def anchor(self) -> Sequence[str]: ...


def ingest_report_concerns(
    session: Session,
    report_concerns: Sequence[BoxConcernLike],
    *,
    run_id: str,
    cap: int = CODER_CONCERN_CAP,
) -> tuple[str | InvalidArgument, ...]:
    """The HARNESS appends the box's concerns on receipt, not Zeus — requiring Zeus to
    re-emit them adds a step Zeus can silently skip.

    The cap is the box's to enforce (bounded exactly once, by the component that
    produced it). A report over it is the box's bug, raised rather than trimmed — trimming
    here would be the second cut D78 forbids. An unanchored one is refused like any other,
    and its refusal is returned in its place."""
    if len(report_concerns) > cap:
        raise HarnessBug(
            f"run {run_id} reported {len(report_concerns)} concerns; the box caps them at {cap}"
        )
    return tuple(
        note_concern(session, c.text, c.anchor, "coder", run_id=run_id) for c in report_concerns
    )


# -- 3 · accrue ---------------------------------------------------------------------------


def accrue(
    session: Session, paths: Iterable[str], *, run_id: str | None = None
) -> tuple[int, ...]:
    """A run modified these paths: one `touched` line per live concern anchored on each.
    No model, no judgment — whether the change fixed the worry is not knowable here.
    Returns the seqs written."""
    written: list[int] = []
    with session.lock:
        for path in dict.fromkeys(paths):
            for c in session.state.concerns.live():
                if path in c.anchor:
                    line = Touched(c.id, path, run_id)
                    written.append(session.append(line, run_id=run_id).seq)
    return tuple(written)


# -- 4 · surface --------------------------------------------------------------------------


def concern_anchors(state: SessionState) -> tuple[str, ...]:
    """What `standing_snapshot` carries: paths, never text. No scope filter — the snapshot
    carries the index and Zeus pulls the text with `concerns(path=…)` when it touches one."""
    return state.concerns.anchors()


# -- 5 · feed the Critic ------------------------------------------------------------------


def concerns_for_critic(
    state: SessionState, variant: CriticVariant, paths: Iterable[str] = ()
) -> tuple[Concern, ...]:
    """The pre-code Critic receives only the user's concerns: its blindness to code is what
    makes it worth running, and a model-raised concern is code knowledge in prose clothing.
    The post-code Critic receives the live concerns anchored on what the work touched."""
    live = state.concerns.live()
    if variant == "pre_code":
        return tuple(c for c in live if c.raised_by == "user")
    touched = set(paths)
    return tuple(c for c in live if touched.intersection(c.anchor))


# -- 6 · promote, 7 · dismiss -------------------------------------------------------------


def promote(session: Session, concern_id: str, check_id: str) -> int | InvalidArgument:
    """The only non-user retirement: a check registered in `Checkpoint.checks` now watches
    it. It stops surfacing and is NOT resolved — a passing check means nothing broke under
    that check (invariants 7 and 20). Where the check is registered is the caller's
    precondition, never `Packet.done_when`."""
    if not check_id.strip():
        return InvalidArgument(
            "check_id",
            "promotion names the check that watches it",
            do="send check_id with that check's id",
        )
    with session.lock:
        state = session.state.concerns
        if state.get(concern_id) is None:
            return InvalidArgument(
                "id", f"no concern {concern_id}", do="name a concern's id as it was raised"
            )
        if state.promoted(concern_id) is not None:
            return InvalidArgument(
                "id",
                f"{concern_id} is already watched by a check",
                do="nothing more to do for it; go on",
            )
        return session.append(Promoted(concern_id, check_id)).seq


def dismiss(session: Session, concern_id: str, *, by: str) -> int | InvalidArgument:
    """Stop displaying it. The user's alone: a model marking its own earlier worry closed is
    guessing, with an incentive to tidy."""
    if by != "user":
        return InvalidArgument(
            "by",
            "only the user dismisses a concern",
            do="leave it; tell the user if you think it no longer applies",
        )
    with session.lock:
        state = session.state.concerns
        if state.get(concern_id) is None:
            return InvalidArgument(
                "id", f"no concern {concern_id}", do="name a concern's id as it was raised"
            )
        if state.dismissed(concern_id) is not None:
            return InvalidArgument(
                "id",
                f"{concern_id} is already dismissed",
                do="nothing more to do for it; go on",
            )
        return session.append(Dismissed(concern_id)).seq


# -- 8 · query ----------------------------------------------------------------------------


def concerns(
    state: SessionState, *, path: str | None = None, include_retired: bool = False
) -> tuple[Concern, ...]:
    """`concerns(path=…)` · `concerns(all=True)`: enumeration, not search — you cannot
    search for what you have forgotten. Live concerns by default; `include_retired` adds
    the promoted and the dismissed, which are history rather than current worries."""
    pool = state.concerns.all() if include_retired else state.concerns.live()
    if path is None:
        return pool
    return tuple(c for c in pool if path in c.anchor)
