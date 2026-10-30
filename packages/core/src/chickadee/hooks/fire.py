"""`Hooks`: what every fire site holds.

`fire` runs the bound hooks that match an event, in configuration order, one at a time (the
dispatcher is one thread, D164 decision 6b, and a fixed order is what makes a run reproducible),
and combines their verdicts. It journals every verdict but a silent `Proceed`, and every error,
as a `hook` line, before any fire site acts on it: resume reads those lines and never runs a
decided hook again. A hook that raises, times out, or answers what its point does not take has
failed, and its `on_error` decides: `allow` journals the error and goes on as `Proceed`; `deny`
fails closed, as the point's blocking verdict.

`has` is read before a fire site builds a payload (OpenCode's `has`), so a point no hook is
bound at costs a lookup, and `Hooks.none()` is what every site takes by default: with no hook
bound, a run journals exactly what it did before (decision 1).

**Concurrency.** A `Hooks` is shared by an episode's roles, which run one at a time on the
episode's thread, and by a session's turn and its job waiters, on their own threads. `_lock`
makes one firing at a time, so the lines of a firing are never interleaved with another's,
and guards `_said`, the index of the hook lines the sink holds: read from the sink once, then
kept as this `Hooks` writes, so a bound is counted without folding the journal again (one
`Hooks` writes a run's or a session's lines; `write_bound` writes before one exists).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Final, Protocol

from chickadee.hooks.failures import HookDenied
from chickadee.hooks.match import matches
from chickadee.hooks.payloads import HookEvent, HookPayload, HookRole, PermissionAsked
from chickadee.hooks.points import HookPoint, blocking, spec_of
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import (
    KINDS,
    NOTHING,
    Ask,
    Combined,
    Continue,
    Deny,
    Halt,
    HookVerdict,
    Note,
    Proceed,
    RetryTask,
    Warn,
    combine,
    kind_of,
    labelled,
)
from chickadee.journal.previews import preview
from chickadee.journal.projection import Placed
from chickadee.schema.journal import HookP
from chickadee.tools.rules import AskPerson, Pending, Refuse

__all__ = [
    "BoundHook",
    "FiredLine",
    "HookError",
    "HookFn",
    "HookLimits",
    "HookSet",
    "HookSink",
    "Hooks",
    "RoleHooks",
    "text_of",
]

_log = logging.getLogger(__name__)

HookFn = Callable[[HookEvent], HookVerdict]
"""A hook's run: a built-in's function, or the closure that runs a script."""


class HookError(Exception):
    """A hook failed: its script exited badly, timed out, or answered what cannot be read."""

    def __init__(self, why: str, *, timed_out: bool = False) -> None:
        super().__init__(why)
        self.why = why
        self.timed_out = timed_out


@dataclass(frozen=True)
class BoundHook:
    spec: HookSpec
    run: HookFn
    digest: str
    """What this hook is, for the set's digest: its entry, and a script's file's bytes."""
    steers: bool = True
    """Whether what it takes counts as the run steering itself (`hooks.max_steering`), and so
    is held by that bound. False for a built-in that is the harness's own behaviour, moved
    onto the framework (`owed_review`), which a person's hooks must not crowd out."""


@dataclass(frozen=True)
class FiredLine:
    """One hook line, as `Hooks` counts and reads it back."""

    seq: int
    event: str
    name: str
    decision: str
    role: str | None
    bounded: bool
    task_id: str | None
    round: int | None
    call_id: str | None
    input_sha: str | None
    text_sha: str | None

    @classmethod
    def of(cls, placed: Placed[HookP]) -> FiredLine:
        line = placed.payload
        return cls(
            placed.seq,
            line.event,
            line.name,
            line.decision,
            line.role,
            line.bounded,
            placed.task_id,
            placed.round,
            line.call_id,
            line.input_sha,
            line.text_sha,
        )


@dataclass(frozen=True)
class HookLimits:
    """The bounds on what hooks may steer (decision 8), the register's `hooks.*` rows."""

    max_continues: int = 1
    max_retries: int = 1
    max_steering: int = 3
    recheck_wait_s: int = 30
    """How long a worker's stop waits on the background re-check."""


@dataclass(frozen=True)
class HookSet:
    """The hooks an episode or a session bound (decision 14), in configuration order, the ones
    it skipped, each with why (an unaccepted script in an unattended run, say), and the bounds
    they run under."""

    hooks: tuple[BoundHook, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    limits: HookLimits = HookLimits()
    _by_point: Mapping[HookPoint, tuple[BoundHook, ...]] = field(
        default_factory=dict[HookPoint, tuple[BoundHook, ...]], init=False, repr=False
    )

    def __post_init__(self) -> None:
        by: dict[HookPoint, list[BoundHook]] = {}
        for hook in self.hooks:
            if hook.spec.enabled:
                by.setdefault(hook.spec.event, []).append(hook)
        object.__setattr__(self, "_by_point", {p: tuple(h) for p, h in by.items()})

    def at(self, point: HookPoint) -> tuple[BoundHook, ...]:
        return self._by_point.get(point, ())

    @property
    def digest(self) -> str:
        """The set as one sha: two runs under different hooks are told apart (`config_sha`)."""
        lim = self.limits
        bounds = f"{lim.max_continues}/{lim.max_retries}/{lim.max_steering}"
        joined = "\n".join((bounds, *(h.digest for h in self.hooks)))
        return hashlib.sha256(joined.encode()).hexdigest()


class HookSink(Protocol):
    """Where `hook` lines go, and are read back from: the run journal, or Zeus's session log."""

    def hook(self, line: HookP, *, task_id: str | None, round: int | None) -> int:
        """`line` written; its seq."""
        ...

    def blob(self, content: bytes) -> str: ...

    def read(self, sha: str) -> bytes:
        """A blob `blob` stored: a `bound` line's entries, read back on resume."""
        ...

    def lines(self) -> tuple[Placed[HookP], ...]: ...


_ERROR_TEXT: Final = "hook {name} failed: {why}"


class Hooks:
    def __init__(self, bound: HookSet, sink: HookSink | None) -> None:
        self.bound = bound
        self._sink = sink
        self._lock = threading.RLock()
        """One firing at a time; guards `_said`."""
        self._said: list[FiredLine] | None = None
        """The sink's hook lines, read once and then kept as they are written."""

    @classmethod
    def none(cls) -> Hooks:
        """No hook bound: every point is a lookup and nothing more."""
        return cls(HookSet(), None)

    def has(self, point: HookPoint) -> bool:
        return bool(self.bound.at(point))

    def has_for(self, point: HookPoint, role: str) -> bool:
        """Whether any hook is bound at `point` for `role`: what a fire site asks before it
        builds a payload."""
        return any(h.spec.role in (None, role) for h in self.bound.at(point))

    def uses_for(self, builtin: str, role: str) -> bool:
        """Whether the built-in `builtin` is bound for `role`, at any point: what a role asks
        when another of its rules gives way to it."""
        return any(
            h.spec.enabled and h.spec.use == builtin and h.spec.role in (None, role)
            for h in self.bound.hooks
        )

    def fire(
        self,
        event: HookEvent,
        *,
        bounded: frozenset[type[HookVerdict]] = frozenset(),
        steering: frozenset[type[HookVerdict]] = frozenset(),
    ) -> Combined:
        """Every matching hook's verdict at `event`, journaled, then combined. A verdict of a
        kind in `bounded` is past its bound (decision 8), and one in `steering` is for a hook
        that steers (`BoundHook.steers`): its line says so, and it is not taken."""
        hooks = [h for h in self.bound.at(event.point) if matches(h.spec, event)]
        if not hooks:
            return NOTHING
        input_sha = hashlib.sha256(event.payload.model_dump_json().encode()).hexdigest()
        said: list[tuple[str, HookVerdict]] = []
        with self._lock:
            for hook in hooks:
                held = steering if hook.steers else frozenset[type[HookVerdict]]()
                over = bounded | held
                said.append((hook.spec.name, self._run(hook, event, input_sha, over)))
        return combine(said)

    def select(
        self,
        point: HookPoint,
        *,
        decision: type[HookVerdict] | None = None,
        role: str | None = None,
        task_id: str | None = None,
        round: int | None = None,
        call_id: str | None = None,
        between: tuple[int, int] | None = None,
        taken: bool = False,
    ) -> tuple[FiredLine, ...]:
        """The lines `point`'s hooks wrote, narrowed by each filter given: the one reading of
        the journal's hook lines, so every reader spells a point and a verdict the same way.
        `between` is a seq range, both ends excluded; `taken` drops the lines past a bound."""
        kind = None if decision is None else KINDS[decision]
        with self._lock:
            return tuple(
                said
                for said in self._lines()
                if said.event == point.value
                and (kind is None or said.decision == kind)
                and (role is None or said.role == role)
                and (task_id is None or said.task_id == task_id)
                and (round is None or said.round == round)
                and (call_id is None or said.call_id == call_id)
                and (between is None or between[0] < said.seq < between[1])
                and not (taken and said.bounded)
            )

    def used(
        self,
        point: HookPoint,
        decision: type[HookVerdict],
        *,
        role: str,
        task_id: str | None = None,
        round: int | None = None,
        steering: bool = False,
    ) -> int:
        """How many firings at `point` took `decision` for `role` (and the task and round, when
        given): what a bound counts. A firing is counted once however many of its hooks
        answered it, as `combine` takes one; with `steering`, only what hooks that steer took.
        Read from the journal, so a resumed run counts what it took before the crash."""
        quiet = {h.spec.name for h in self.bound.hooks if steering and not h.steers}
        lines = self.select(
            point, decision=decision, role=role, task_id=task_id, round=round, taken=True
        )
        return len({(s.input_sha, s.task_id, s.round) for s in lines if s.name not in quiet})

    def steering_used(self) -> int:
        """The level-2 verdicts the episode's hooks that steer took: retries, and continues at
        Athena's stop, both of which are hers (decision 8). The harness's own (`owed_review`)
        are not counted."""
        retries = self.used(HookPoint.ROUND_ENDED, RetryTask, role="athena", steering=True)
        continues = self.used(HookPoint.BEFORE_STOP, Continue, role="athena", steering=True)
        return retries + continues

    def said(self, point: HookPoint, call_id: str) -> tuple[str, ...] | None:
        """The notes `point`'s hooks already gave about `call_id`, or None where no line says
        they fired: what a resumed call, acting again, reads back rather than asking again."""
        lines = self.select(point, call_id=call_id)
        if not lines:
            return None
        return tuple(self.text(said) for said in lines if said.decision == KINDS[Note])

    def verdicts(self, lines: Sequence[FiredLine]) -> list[tuple[str, HookVerdict]]:
        """The notes and continues `lines` hold, as the verdicts they were, for `combine`: what
        a resume reads back sends what the live run sent."""
        out: list[tuple[str, HookVerdict]] = []
        for said in lines:
            if said.bounded:
                continue
            if said.decision == KINDS[Continue]:
                out.append((said.name, Continue(self.text(said))))
            elif said.decision == KINDS[Note]:
                out.append((said.name, Note(self.text(said))))
        return out

    def text(self, said: FiredLine) -> str:
        """A line's words, from the blob it names."""
        if self._sink is None or said.text_sha is None:
            return ""
        return self._sink.read(said.text_sha).decode("utf-8")

    def _lines(self) -> list[FiredLine]:
        """Under `_lock`."""
        if self._said is None:
            lines = () if self._sink is None else self._sink.lines()
            self._said = [FiredLine.of(p) for p in lines]
        return self._said

    def _run(
        self,
        hook: BoundHook,
        event: HookEvent,
        input_sha: str,
        bounded: frozenset[type[HookVerdict]],
    ) -> HookVerdict:
        name = hook.spec.name
        started = perf_counter()
        try:
            verdict = hook.run(event)
            accepts = spec_of(event.point).accepts
            if type(verdict) not in accepts:
                raise HookError(
                    f"answered {kind_of(verdict)}, which {event.point} does not take; it "
                    f"takes {', '.join(sorted(KINDS[k] for k in accepts))}"
                )
        except HookError as exc:
            return self._failed(hook, event, input_sha, exc, _ms(started), bounded)
        except Exception as exc:
            # A built-in is harness code, and its raising is a harness bug; it is still the
            # hook's failure, never the run's, and its `on_error` decides as a script's does.
            _log.exception("hook %s raised at %s", name, event.point)
            failed = HookError(f"{type(exc).__name__}: {exc}")
            return self._failed(hook, event, input_sha, failed, _ms(started), bounded)
        if hook.spec.kind == "script":
            verdict = labelled(name, verdict)
        if isinstance(verdict, Proceed):
            return verdict
        over = type(verdict) in bounded
        self._line(
            event, name, verdict, wall_ms=_ms(started), input_sha=input_sha, bounded=over
        )
        return Proceed() if over else verdict

    def _failed(
        self,
        hook: BoundHook,
        event: HookEvent,
        input_sha: str,
        exc: HookError,
        wall_ms: int,
        bounded: frozenset[type[HookVerdict]],
    ) -> HookVerdict:
        """`allow`: the error journaled, and a warning. `deny`: the point's blocking verdict,
        journaled as that verdict with the error beside it, so it is counted, and held, by the
        same bound as the verdict a hook gives (decision 8)."""
        name = hook.spec.name
        words = _ERROR_TEXT.format(name=name, why=exc.why)
        if hook.spec.on_error == "allow":
            self._line(
                event,
                name,
                None,
                wall_ms=wall_ms,
                input_sha=input_sha,
                error=exc.why,
                timed_out=exc.timed_out,
            )
            return Warn(words)
        verdict = blocking(
            event.point, words, "go on without this; the harness could not check it"
        )
        over = type(verdict) in bounded
        self._line(
            event,
            name,
            verdict,
            wall_ms=wall_ms,
            input_sha=input_sha,
            bounded=over,
            error=exc.why,
            timed_out=exc.timed_out,
        )
        return Proceed() if over else verdict

    def _line(
        self,
        event: HookEvent,
        name: str,
        verdict: HookVerdict | None,
        *,
        wall_ms: int,
        input_sha: str | None,
        bounded: bool = False,
        error: str | None = None,
        timed_out: bool = False,
    ) -> None:
        """Under `_lock`: the line written, then kept in `_said` once that has been read."""
        if self._sink is None:
            return
        text = None if verdict is None else text_of(verdict)
        line = HookP(
            event=event.point.value,
            name=name,
            decision="error" if verdict is None else kind_of(verdict),
            wall_ms=wall_ms,
            message=None if (said := error or text) is None else preview(said),
            timed_out=timed_out,
            role=event.role,
            call_id=event.call_id,
            text_sha=None if text is None else self._sink.blob(text.encode("utf-8")),
            input_sha=input_sha,
            bounded=bounded,
            error=error,
        )
        seq = self._sink.hook(line, task_id=event.task_id, round=event.round)
        if self._said is not None:
            self._said.append(FiredLine.of(Placed(seq, line, event.task_id, event.round)))


@dataclass(frozen=True)
class RoleHooks:
    """`Hooks` as one role holds them: who it is, on which model, and where in the run it is, so
    a fire site names only the point and its payload. `place` reads the task and round the role
    is at when it fires, since a box role moves between tasks."""

    hooks: Hooks
    role: HookRole
    model: str = ""
    run_id: str | None = None
    session: str | None = None
    place: Callable[[], tuple[str | None, int | None]] = field(default=lambda: (None, None))

    @classmethod
    def none(cls, role: HookRole) -> RoleHooks:
        return cls(Hooks.none(), role)

    def has(self, point: HookPoint) -> bool:
        return self.hooks.has_for(point, self.role)

    def uses(self, builtin: str) -> bool:
        return self.hooks.uses_for(builtin, self.role)

    @property
    def limits(self) -> HookLimits:
        return self.hooks.bound.limits

    def steering_room(self) -> bool:
        """Whether the episode may still steer itself (`hooks.max_steering`, decision 8)."""
        return self.hooks.steering_used() < self.limits.max_steering

    def select(
        self,
        point: HookPoint,
        *,
        task_id: str | None = None,
        round: int | None = None,
        between: tuple[int, int] | None = None,
    ) -> tuple[FiredLine, ...]:
        """`Hooks.select`, every role's lines: a level-2 reader reads the worker's too."""
        return self.hooks.select(point, task_id=task_id, round=round, between=between)

    def text(self, said: FiredLine) -> str:
        return self.hooks.text(said)

    def combined(self, lines: Sequence[FiredLine]) -> Combined:
        """What `lines` said, combined as the live firing combined it (`Hooks.verdicts`)."""
        return combine(self.hooks.verdicts(lines))

    def event(
        self,
        point: HookPoint,
        payload: HookPayload,
        *,
        call_id: str | None = None,
        task_id: str | None = None,
        round: int | None = None,
    ) -> HookEvent:
        """`point`'s event, at the task and round the role is at, or those given."""
        here_task, here_round = self.place()
        task_id = task_id if task_id is not None else here_task
        round_ = round if round is not None else here_round
        return HookEvent(
            point,
            self.role,
            payload,
            model=self.model,
            run_id=self.run_id,
            session=self.session,
            task_id=task_id,
            round=round_,
            call_id=call_id,
        )

    def fire(
        self,
        point: HookPoint,
        payload: HookPayload,
        *,
        call_id: str | None = None,
        task_id: str | None = None,
        round: int | None = None,
        bounded: frozenset[type[HookVerdict]] = frozenset(),
        steering: frozenset[type[HookVerdict]] = frozenset(),
    ) -> Combined:
        if not self.has(point):
            return NOTHING
        event = self.event(point, payload, call_id=call_id, task_id=task_id, round=round)
        return self.hooks.fire(event, bounded=bounded, steering=steering)

    def used(
        self,
        point: HookPoint,
        decision: type[HookVerdict],
        *,
        task_id: str | None = None,
        round: int | None = None,
        steering: bool = False,
    ) -> int:
        return self.hooks.used(
            point, decision, role=self.role, task_id=task_id, round=round, steering=steering
        )

    def ask_refused(self, pending: Pending, ask: AskPerson) -> Refuse | None:
        """`permission_asked` (decision 2), before the person is asked: a hook may refuse the
        call unasked, never approve it. None, and the person is asked as before. The
        dispatcher's `AskGate`, so it asks without importing the framework."""
        if not self.has(HookPoint.PERMISSION_ASKED):
            return None
        payload = PermissionAsked(question=ask.question, tier=ask.tier.value, tool=pending.tool)
        said = self.fire(HookPoint.PERMISSION_ASKED, payload, call_id=pending.call.id)
        if isinstance(said.decided, Deny):
            return Refuse(HookDenied(said.by or "hook", said.decided.reason, said.decided.do))
        return None

    def noted_once(
        self,
        point: HookPoint,
        payload: HookPayload,
        *,
        call_id: str,
        task_id: str | None = None,
    ) -> str | None:
        """`point`'s note about `call_id`: read back where its lines hold it, so a call acting
        again on resume says what it said, else fired. A point whose only verdict is a note,
        level 2's after the fact (`task_reviewed`, `task_committed`)."""
        if not self.has(point):
            return None
        said = self.hooks.said(point, call_id)
        if said is None:
            return self.fire(point, payload, call_id=call_id, task_id=task_id).note
        return "\n".join(said) or None


def text_of(verdict: HookVerdict) -> str | None:
    """A verdict's words, as its line keeps them."""
    match verdict:
        case Proceed():
            return None
        case Deny(reason=reason, do=do):
            return reason if do is None else f"{reason}. Do: {do}"
        case Ask(question=question):
            return question
        case Halt(reason=reason):
            return reason
        case Note(text=text) | Continue(text=text) | RetryTask(text=text) | Warn(text=text):
            return text


def _ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)
