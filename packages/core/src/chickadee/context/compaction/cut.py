"""The one transaction: `compact`, measured, committed and
settled. **Commit, then emit**: the cut's line carries the block, the standing
snapshot and the boundary together, so a crash leaves the whole cut or none of it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

from pydantic import JsonValue

from chickadee.context.compaction.boundary import Decline
from chickadee.context.compaction.cutter import Cutter, framed_block, region_of
from chickadee.context.compaction.residue import (
    NO_RESIDUE,
    ResidueOutcome,
    ResidueSource,
    residue,
)
from chickadee.context.compaction.trigger import (
    COMPACT_TRIGGER_FRACTION,
    NO_REBUILD,
    RebuildSignal,
    Trigger,
    TriggerKind,
    payback,
    should_compact,
)
from chickadee.context.lines import CutLine
from chickadee.context.request import Ctx, WindowBasis, assemble, est_text, fits, pieces_of
from chickadee.context.standing import standing_snapshot
from chickadee.context.store import SessionStoreError

if TYPE_CHECKING:
    from chickadee.config import PriceRow
    from chickadee.context.project import RunRecords
    from chickadee.context.resolver import HandleResolver
    from chickadee.context.session import Session
    from chickadee.schema.adapter import CacheObs

__all__ = [
    "OVERFLOW_RETRIES",
    "CacheState",
    "Compaction",
    "TurnTooLarge",
    "compact",
]


OVERFLOW_RETRIES: Final = 1
"""One compact-and-retry after a provider overflow. A second identical
rejection means the estimate is wrong about the object, and retrying again is a loop."""


@dataclass(frozen=True)
class CacheState:
    """What §5.4 reads about the cache."""

    price_row: PriceRow
    obs: CacheObs
    """The last response's observation."""
    prev: CacheObs | None = None
    """The one before it: a transition needs two points."""
    rebuild: RebuildSignal = NO_REBUILD


@dataclass(frozen=True)
class Compaction:
    """What one `compact` call did."""

    trigger: Trigger | None
    """None: `should_compact` said no, and nothing else happened."""
    cut: CutLine | None
    """The committed cut, or None when nothing was cut — including an economic cut that
    would have freed nothing, which is not taken (its rebuild would buy no room)."""
    declined: Decline | None
    """Why `choose_boundary` declined. With a cut, the cut is the fallback it led to; without
    one, nothing past the previous boundary was old enough to take."""
    aborted: str | None
    """The store's failure when the commit did not land: nothing was said, and the
    boundary did not move (invariant 4)."""
    ctx: Ctx
    """The context after — unchanged unless a cut committed."""
    skipped: Mapping[str, object] | None = None
    """The facts of an economic cut not taken because it would free nothing, for the
    caller to journal. Nothing is written to the session: a line would be the very prefix
    edit the skip declines to pay for. Without these the skip would be silent, and
    a later measurement, which reads what each layer frees, would lose the one case
    it most needs to see: a cut that does not pay for itself."""


class TurnTooLarge(Exception):
    """§5.12 step 12b: after a full cut the pending request still does not fit. A fact about
    the request, never a reason to shorten it — inventing a shorter version of what someone
    typed is what invariant 19 forbids. The cut, when one committed, stays committed, and
    its line records every term of the comparison."""

    def __init__(
        self,
        est_tokens: int,
        headroom: int,
        window_basis: WindowBasis,
        *,
        held: int,
        cut: CutLine | None,
    ) -> None:
        self.est_tokens = est_tokens
        self.headroom = headroom
        self.window_basis = window_basis
        self.held = held
        self.cut = cut
        super().__init__(
            f"the pending request (~{est_tokens:,} tokens) does not fit beside {held:,} "
            f"held tokens: {headroom:,} tokens of headroom on the {window_basis}, after "
            f"{'a full cut' if cut is not None else 'finding nothing left to cut'}"
        )


def compact(
    session: Session,
    ctx: Ctx,
    cache: CacheState,
    pending_tokens: int,
    *,
    resolver: HandleResolver,
    records: RunRecords | None = None,
    layer3: ResidueSource = NO_RESIDUE,
    pinned: Collection[int] = (),
    provider_overflow: bool = False,
    trigger_fraction: float = COMPACT_TRIGGER_FRACTION,
    on_cut: Callable[[int], None] | None = None,
) -> Compaction:
    """§5.12, the one transaction. `ctx` measures everything assembled — the caller's prefix
    and this session's messages; `pending_tokens` is what the request adds beyond it (zero
    when that is already recorded). `pinned` holds the seqs of tool lines whose bodies are
    pinned now. `on_cut` is told the tokens held once a cut is due, before it is made.

    The constraint ledger is exempt from every layer by construction: it is folded from
    this same log into records no layer reads, and the snapshot restates it after the block.
    The journals are never edited — a cut rewrites prompt assembly only."""
    trigger = should_compact(
        ctx,
        cache.price_row,
        cache.obs,
        cache.prev,
        pending_tokens,
        rebuild=cache.rebuild,
        provider_overflow=provider_overflow,
        trigger_fraction=trigger_fraction,
    )
    if trigger is None:
        return Compaction(None, None, None, None, ctx)
    if on_cut is not None:
        # The status line says `compacting…` before the cut's slow part.
        on_cut(ctx.tokens)
    with session.lock:
        region = region_of(session.state, trigger)
        if isinstance(region, Decline):
            if trigger.kind is TriggerKind.MUST_FIT:
                raise TurnTooLarge(
                    pending_tokens, ctx.headroom, ctx.window_basis, held=ctx.tokens, cut=None
                )
            return Compaction(trigger, None, region, None, ctx)
        cutter = Cutter(
            session.state,
            region,
            pinned=pinned,
            resolver=resolver,
            records=records,
            blob=session.blob,
        )
        cutter.run()
        if trigger.kind is not TriggerKind.MUST_FIT:
            # A residue only adds to the block, so a cut that frees nothing without one frees
            # nothing with one: it is not taken, and its model call is never made.
            unfired = ResidueOutcome(region.job, (), fired=False, failure=None)
            freed = _measure(session, cutter, unfired).freed
            if freed <= 0:
                skipped = _skipped(trigger, cutter, freed)
                return Compaction(trigger, None, region.declined, None, ctx, skipped)
        outcome = residue(
            region.job, cutter.prose, (*cutter.projections, *layer3.checkpoint_rows), layer3
        )
        return _commit(session, ctx, trigger, cutter, outcome, cache, pending_tokens)


@dataclass(frozen=True)
class _Measured:
    """The block a cut would append, and what it would free — before anything is written."""

    standing: str
    block: str
    freed: int
    rest: int
    """Everything the new assembly holds but the block."""


def _measure(session: Session, cutter: Cutter, outcome: ResidueOutcome) -> _Measured:
    """Measured once per residue answer: `cutter.body` books an answer's own text against
    layer 3, so only an outcome with no items may be measured again."""
    state, region = session.state, cutter.region
    seq = session.next_seq
    standing = standing_snapshot(state, seq).framed()
    before = assemble(state, session.blob).tokens
    # Everything the new assembly holds but the block, so the block can state what it frees.
    probe = CutLine(
        boundary=region.boundary,
        previous=region.previous,
        block="",
        standing=standing,
        lifted=cutter.lifted,
        facts=MappingProxyType({}),
        seq=seq,
    )
    rest = sum(p.tokens for p in pieces_of(state, (*state.cuts(), probe), session.blob))
    block, freed = _settle(cutter.body(outcome), cutter.objects, region.boundary, before - rest)
    return _Measured(standing, block, freed, rest)


def _commit(
    session: Session,
    ctx: Ctx,
    trigger: Trigger,
    cutter: Cutter,
    outcome: ResidueOutcome,
    cache: CacheState,
    pending_tokens: int,
) -> Compaction:
    """Steps 10-14: assemble the block, measure, append ONE line, and only then answer."""
    state, region = session.state, cutter.region
    prior = state.cuts()
    measured = _measure(session, cutter, outcome)
    standing, block, freed = measured.standing, measured.block, measured.freed
    if freed <= 0 and trigger.kind is not TriggerKind.MUST_FIT:
        # An economic cut is paid for with a prefix rebuild in exchange for room. One that
        # frees nothing, or grows the context, buys none: nothing is appended. Reached only
        # when the residue's own text tipped it, since `compact` checked the cut without it.
        skipped = _skipped(trigger, cutter, freed)
        return Compaction(trigger, None, region.declined, None, ctx, skipped)
    after_session = measured.rest + est_text(block)
    new_ctx = replace(ctx, tokens=ctx.tokens - freed)
    fits_after = fits(new_ctx, pending_tokens)
    if trigger.kind is not TriggerKind.FREE_REBUILD:
        # S is what the cut makes the cache write again: everything from the new block on.
        suffix = after_session - sum(est_text(c.block) + est_text(c.standing) for c in prior)
        h = outcome.input_tokens if outcome.fired else 0
        trigger = replace(trigger, payback_requests=payback(cache.price_row, suffix, freed, h))
    facts = _facts(trigger, cutter, outcome, freed, new_ctx, pending_tokens, fits_after)
    line = CutLine(
        boundary=region.boundary,
        previous=region.previous,
        block=block,
        standing=standing,
        lifted=tuple(cutter.lifted),
        facts=cast("Mapping[str, JsonValue]", MappingProxyType(facts)),
    )
    try:
        session.append(line)
    except SessionStoreError as exc:
        # A cut that cannot be recorded did not happen: nothing said, no boundary moved.
        return Compaction(trigger, None, region.declined, f"{type(exc).__name__}: {exc}", ctx)
    cut = state.cuts()[-1]
    if trigger.kind is TriggerKind.MUST_FIT and not fits_after:
        raise TurnTooLarge(
            pending_tokens, new_ctx.headroom, ctx.window_basis, held=new_ctx.tokens, cut=cut
        )
    return Compaction(trigger, cut, region.declined, None, new_ctx)


def _skipped(trigger: Trigger, cutter: Cutter, freed: int) -> dict[str, object]:
    """What a skipped cut would have done: enough to count skips per trigger, see which
    layer failed to pay, and tell a region of receipts that could not collapse (no
    finding) from one that was simply small."""
    region = cutter.region
    return {
        "trigger": trigger.kind.value,
        "basis": trigger.basis.value,
        "region": [cutter.lo, cutter.hi],
        "declined": region.declined.value if region.declined is not None else None,
        "units": [u.run_id for u in region.units],
        "objects": cutter.objects,
        "est_freed": freed,
        "freed": dict(cutter.freed),
        "refused": len(cutter.refused),
    }


def _settle(body: str, objects: int, boundary: int, room: int) -> tuple[str, int]:
    """The block and the tokens it frees, agreeing: the footer states the figure, and its own
    digits are part of what it measures. `room` is what everything but the block frees."""
    freed = room
    block = framed_block(body, objects, freed, boundary)
    for _ in range(3):
        now = room - est_text(block)
        if now == freed:
            break
        freed = now
        block = framed_block(body, objects, freed, boundary)
    return block, freed


def _facts(
    trigger: Trigger,
    cutter: Cutter,
    outcome: ResidueOutcome,
    freed: int,
    after: Ctx,
    pending_tokens: int,
    fits_after: bool,
) -> dict[str, object]:
    """What step 12 records beside the boundary — for the reader of the log; assembly never
    reads it. The provider's actual count arrives with the next response and is journaled by
    the client that receives it, never written here as if measured."""
    region = cutter.region
    return {
        "trigger": trigger.kind.value,
        "basis": trigger.basis.value,
        "cache_mode": trigger.cache_mode.value,
        "premium": trigger.premium,
        "payback_requests": trigger.payback_requests,
        "region": [cutter.lo, cutter.hi],
        "declined": region.declined.value if region.declined is not None else None,
        "units": [u.run_id for u in region.units],
        "objects": cutter.objects,
        "est_freed": freed,
        "freed": dict(cutter.freed),
        "residue": {
            "job": outcome.job,
            "fired": outcome.fired,
            "items": len(outcome.items),
            "failure": outcome.failure,
        },
        "refused": cutter.refused,
        "superseded": cutter.superseded,
        "stale": sorted(cutter.stale),
        "notices_elided": cutter.notices,
        "turns_elided": [t.seq for t in cutter.elided],
        "fit": {
            "window": after.window,
            "window_basis": after.window_basis,
            "response_reserve": after.max_output,
            "pending_tokens": pending_tokens,
            "tokens_after": after.tokens,
            "fits_after": fits_after,
        },
    }
