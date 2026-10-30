"""The harness's own hooks: in-process code, named, each with the points it
fires at, the verdicts it can answer and the parameters it takes, checked when it is bound.

A built-in's words are the harness's, kept beside it (`review_owed`, the watchdog's templates),
so the framework imports no role. `DEFAULTS` are bound unless a layer switches one off by name:
today, `owed_review` alone, which is the behaviour Athena had in older versions. The progress
watchdog's two, and evidence at handoff's two, are off until a layer binds them
for the worker.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict

from chickadee.errors import ConfigError
from chickadee.hooks.evidence import EvidenceParams, evidence_check, evidence_report
from chickadee.hooks.fire import BoundHook, HookFn, HookLimits, HookSet
from chickadee.hooks.payloads import BeforeStop, HookEvent
from chickadee.hooks.points import HookPoint, spec_of
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import Continue, Halt, HookVerdict, Note, Proceed
from chickadee.hooks.watchdog import WatchdogParams, progress_report, progress_watchdog
from chickadee.tools.builtin.review import ANSWERS

__all__ = [
    "BUILTINS",
    "DEFAULTS",
    "REVIEW_OWED_HEAD",
    "Builtin",
    "bind_builtin",
    "check_shared",
    "default_set",
    "owed_review",
    "review_owed",
]


class NoParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class Builtin:
    name: str
    points: frozenset[HookPoint]
    returns: frozenset[type[HookVerdict]]
    """Every verdict it can answer, checked against its point's when it is bound."""
    make: Callable[[BaseModel], HookFn]
    """Its run, from its parameters as `params` read them."""
    params: type[BaseModel] = NoParameters
    steers: bool = True
    """False for the harness's own behaviour moved onto the framework (`BoundHook.steers`)."""
    with_of: str | None = None
    """Another built-in whose `with` this one must be given too: the two read one set of
    thresholds (`check_shared`)."""


REVIEW_OWED_HEAD: Final = "Not finished:"
"""How the reminder begins. That it was said is the `owed_review` hook's line."""


def review_owed(task: str, title: str, *, taken_over: bool) -> str:
    """What Athena is told when she replies with `task` still open: nothing of it is
    committed until she reviews it. Names her tools by their names, and no shell."""
    # From the review tool's own list, so a renamed answer is never named here; not
    # `continue`, which answers only a round held for a message, and its message says so.
    named = ("accept", "drop") if taken_over else tuple(a for a in ANSWERS if a != "continue")
    answers = f"{', '.join(named[:-1])} or {named[-1]}"
    return (
        f"{REVIEW_OWED_HEAD} {task} {title!r} is still open and nothing of it is committed. "
        f"Review it with `review` ({answers}), then reply with your report."
    )


def owed_review(event: HookEvent) -> HookVerdict:
    """Athena replied with a task reported and unreviewed: nothing of it is committed until she
    reviews it, so she is told once, with her tools open. Moved from `box/athena.py`
    with its words (`review_owed`)."""
    stop = event.payload
    if not isinstance(stop, BeforeStop) or stop.ended or stop.open_task is None:
        return Proceed()
    task = stop.open_task
    if task.state not in ("reported", "held", "taken_over"):
        return Proceed()
    return Continue(review_owed(task.id, task.title, taken_over=task.state == "taken_over"))


def _watched(make: Callable[[WatchdogParams], HookFn]) -> Callable[[BaseModel], HookFn]:
    def made(params: BaseModel) -> HookFn:
        if not isinstance(params, WatchdogParams):
            raise TypeError(f"the watchdog takes WatchdogParams, not {type(params).__name__}")
        return make(params)

    return made


def _evidenced(make: Callable[[EvidenceParams], HookFn]) -> Callable[[BaseModel], HookFn]:
    def made(params: BaseModel) -> HookFn:
        if not isinstance(params, EvidenceParams):
            raise TypeError(f"evidence takes EvidenceParams, not {type(params).__name__}")
        return make(params)

    return made


BUILTINS: Final[Mapping[str, Builtin]] = {
    "owed_review": Builtin(
        "owed_review",
        frozenset({HookPoint.BEFORE_STOP}),
        frozenset({Continue, Proceed}),
        lambda params: owed_review,
        steers=False,
    ),
    "progress_watchdog": Builtin(
        "progress_watchdog",
        frozenset({HookPoint.AFTER_BATCH}),
        frozenset({Note, Halt, Proceed}),
        _watched(progress_watchdog),
        params=WatchdogParams,
    ),
    "progress_report": Builtin(
        "progress_report",
        frozenset({HookPoint.ROUND_ENDED}),
        frozenset({Note, Proceed}),
        _watched(progress_report),
        params=WatchdogParams,
        with_of="progress_watchdog",
    ),
    "evidence_check": Builtin(
        "evidence_check",
        frozenset({HookPoint.BEFORE_STOP}),
        frozenset({Continue, Proceed}),
        _evidenced(evidence_check),
        params=EvidenceParams,
    ),
    "evidence_report": Builtin(
        "evidence_report",
        frozenset({HookPoint.ROUND_ENDED}),
        frozenset({Note, Proceed}),
        _evidenced(evidence_report),
        params=EvidenceParams,
        with_of="evidence_check",
    ),
}

DEFAULTS: Final[tuple[HookSpec, ...]] = (
    HookSpec(event=HookPoint.BEFORE_STOP, name="owed_review", use="owed_review", role="athena"),
)
"""Bound unless a layer sets `enabled = false` under the same `(event, name)`."""


def bind_builtin(spec: HookSpec) -> BoundHook:
    """`spec`'s built-in, its parameters checked, or `ConfigError` naming what is wrong."""
    name = spec.use
    found = BUILTINS.get(name) if name is not None else None
    if found is None:
        raise ConfigError(
            f"hook {spec.name!r} uses {name!r}, which is no built-in; the built-ins are "
            f"{sorted(BUILTINS)}"
        )
    if spec.event not in found.points:
        raise ConfigError(
            f"hook {spec.name!r}: the built-in {name!r} fires at "
            f"{', '.join(sorted(p.value for p in found.points))}, not {spec.event.value}"
        )
    beyond = found.returns - spec_of(spec.event).accepts
    if beyond:
        raise ConfigError(
            f"hook {spec.name!r}: the built-in {name!r} can answer "
            f"{', '.join(sorted(k.__name__ for k in beyond))}, which {spec.event.value} does "
            "not take"
        )
    try:
        params = found.params.model_validate(spec.with_)
    except ValueError as exc:
        raise ConfigError(f"hook {spec.name!r}'s `with`: {exc}") from None
    digest = hashlib.sha256(spec.model_dump_json(by_alias=True).encode()).hexdigest()
    return BoundHook(spec, found.make(params), digest, steers=found.steers)


def check_shared(hooks: Sequence[BoundHook]) -> None:
    """`ConfigError` where a built-in is given another `with` than the built-in it shares its
    thresholds with (`Builtin.with_of`): the progress report would read a ladder the watchdog
    did not climb. Whatever their roles: the watchdog is the worker's, and the report
    is read at Athena's desk (`round_ended`)."""
    for hook in hooks:
        found = BUILTINS.get(hook.spec.use or "")
        if found is None or found.with_of is None or not hook.spec.enabled:
            continue
        for other in hooks:
            if (
                other.spec.enabled
                and other.spec.use == found.with_of
                and other.spec.with_ != hook.spec.with_
            ):
                raise ConfigError(
                    f"hook {hook.spec.name!r} ({found.name}) and hook {other.spec.name!r} "
                    f"({found.with_of}) read one set of thresholds; give both the same `with`"
                )


def default_set(limits: HookLimits | None = None) -> HookSet:
    """The defaults alone, bound: what an episode runs with where nothing was configured."""
    return HookSet(tuple(bind_builtin(s) for s in DEFAULTS), limits=limits or HookLimits())
