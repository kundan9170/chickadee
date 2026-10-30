"""One session's driver of `compact`.

`compact` is the transaction; this is what calls it, before every request a session's loop
makes, and holds what that needs and the loop does not: the endpoint's window and reserve, its
price row and cache mode, the resolver and the run records a cut reads, the role's trigger, and
the residue model with what charges its calls. It also keeps the two inputs that only exist
across requests:

- **the cache observations**, the last two responses' `CacheObs`, since CX-1 reads a
  *transition*, never one observation;
- **the gap history**, the time since the last response, which is what says a cached prefix
  expired, and whether the model moved since (`/rung`).

Three calls, each one `compact`: `before` a request, `observed` after an answer, and
`overflowed` after a provider refused a request as too long.

**Concurrency.** A `SessionCompaction` belongs to one conversation, whose worker makes one
request at a time; `compact` takes the session's own lock for the cut.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING

from chickadee.context.compaction.cut import CacheState, Compaction, compact
from chickadee.context.compaction.residue import NO_RESIDUE, ModelResidue, ResidueSource
from chickadee.context.compaction.trigger import COMPACT_TRIGGER_FRACTION, RebuildSignal
from chickadee.context.request import Ctx, assemble
from chickadee.errors import HarnessBug
from chickadee.schema.adapter import CacheMode, CacheObs, Usage

if TYPE_CHECKING:
    from chickadee.config import PriceRow, RungEndpoint
    from chickadee.context.project import RunRecords
    from chickadee.context.resolver import HandleResolver
    from chickadee.context.session import Session

__all__ = ["SessionCompaction"]


class SessionCompaction:
    """`compact` for one session on one endpoint. `trigger_fraction` is the role's resolved
    `compact_trigger`, as a fraction of the window; `clock` measures gaps."""

    def __init__(
        self,
        endpoint: RungEndpoint,
        price_row: PriceRow,
        cache_mode: CacheMode,
        *,
        resolver: Callable[[], HandleResolver],
        records: RunRecords | None = None,
        residue: ModelResidue | None = None,
        charge: Callable[[Usage], object] | None = None,
        residue_off: str | None = None,
        trigger_fraction: float = COMPACT_TRIGGER_FRACTION,
        clock: Callable[[], float] = time.monotonic,
        pinned: Callable[[Session], Collection[int]] = lambda session: (),
        on_cut: Callable[[int], None] | None = None,
    ) -> None:
        if residue is not None and charge is None:
            raise HarnessBug("a residue model's calls are charged (D76); pass charge=")
        self._endpoint = endpoint
        self._price_row = price_row
        self._mode = cache_mode
        self._resolver = resolver
        """Asked at each cut: Zeus's view may have moved to the session's worktree since the
        last one, and a file handle's freshness is read where the view is."""
        self._records = records
        self._model = residue
        self._residue = (
            ResidueSource(model=residue)
            if residue is not None
            else ResidueSource(off=residue_off)
            if residue_off is not None
            else NO_RESIDUE
        )
        self._charge = charge
        self._fraction = trigger_fraction
        # The tool lines a cut lifts: asked at every cut, never held,
        # so an unloaded skill is simply absent from the next one.
        self._pinned = pinned
        self._on_cut = on_cut
        """Told the tokens held as a cut starts: the status line's
        `compacting…`."""
        self._clock = clock
        self._obs = CacheObs(mode=cache_mode)
        """No response yet: nothing observed, so `warm` is unknowable, never cold."""
        self._prev: CacheObs | None = None
        self._answered_at: float | None = None
        self._model_changed = False

    def before(self, session: Session, prefix_tokens: int) -> Compaction:
        """Before a request is assembled. Nothing is pending beside the log: the person's
        words and every tool result are already on it. May raise `TurnTooLarge`."""
        return self._compact(session, prefix_tokens, overflow=False)

    def overflowed(self, session: Session, prefix_tokens: int) -> Compaction:
        """After the provider refused the request as too long: a MUST_FIT cut on the
        provider's word, since the estimate that let the request go was wrong about it.
        May raise `TurnTooLarge`."""
        return self._compact(session, prefix_tokens, overflow=True)

    def observed(self, usage: Usage) -> None:
        """After an answer: its cache observation, and the time the gap is measured from."""
        self._prev, self._obs = self._obs, CacheObs.from_usage(usage, self._mode)
        self._answered_at = self._clock()
        self._model_changed = False

    def model_changed(self) -> None:
        """`/rung zeus` moved the model: the next request rebuilds the cache whatever is
        done, so a cut before it owes no premium."""
        self._model_changed = True

    def _compact(self, session: Session, prefix_tokens: int, *, overflow: bool) -> Compaction:
        held = assemble(session.state, session.blob).tokens
        idle = None if self._answered_at is None else self._clock() - self._answered_at
        cache = CacheState(
            self._price_row,
            self._obs,
            self._prev,
            RebuildSignal(idle_s=idle, model_changed=self._model_changed),
        )
        try:
            return compact(
                session,
                Ctx.of(self._endpoint, prefix_tokens + held),
                cache,
                0,
                resolver=self._resolver(),
                records=self._records,
                layer3=self._residue,
                pinned=self._pinned(session),
                provider_overflow=overflow,
                trigger_fraction=self._fraction,
                on_cut=self._on_cut,
            )
        finally:
            # A MUST_FIT cut asks the residue model and may then raise `TurnTooLarge`: the
            # call was made, so it is charged now, not at a later cut that may never come.
            self._charge_residue()

    def _charge_residue(self) -> None:
        """The residue model's spend, on this thread (`ModelResidue`'s docstring says why
        not on its own), every answer once, including one that outlived its timeout."""
        if self._model is None or self._charge is None:
            return
        for usage in self._model.drain():
            self._charge(usage)
