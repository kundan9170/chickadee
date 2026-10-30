"""The progress watchdog: two
meters over the worker's tagged calls, and the ladder they climb.

- **Looping**, in calls: a change sets it to 0, a call that showed something new lowers it by
  1, a call that did nothing raises it by 1. It fires at `looping`.
- **Wandering**, in model responses: 0 on a response with a change, else 1 more, unless every
  call of the response read on in a file the last response read (a continuation). It fires at
  `W` responses, or at `calls_per_response` times `W` calls, continuations counted, so neither
  batching nor paging through one file can run past it.
- **Unchecked**, in model responses: from the first change after the last
  test run (or the round's start), every response, until a test run. It fires at `unchecked`
  responses, or `calls_per_response` times that in calls, while a runner exists for a changed
  file and the worker may run commands. A test run, not a change, ends its stall. Its measure is
  `unchecked`, the one hook 2 reads too.

A test run that a later run confirms broke a test, after more changes, carries a note on that
run's answer: once a test a round, never a rung.

A meter that fires starts a stall: a **nudge**; then, if the meter rises half its threshold
again with no change, **stop and report** (call `finish` with four parts); then, `backstop`
calls later, a **halt**. A change ends a stall short of its report. After one, a new problem is
nudged again; the same problem (its evidence contained in an earlier stall's, either way) or
the third stall of the round goes straight to the report. A round starts from zero.

**A pure function of the payload.** A built-in keeps nothing between firings, and a resumed
round must reach the rung it had, so each firing replays the round's calls through the ladder
(`replay`) and answers what is due at the last response. The same replay tells the reader of
the worker's report (`progress_report`) whether the round was handed up.

The words are templates (`adapters/shared/templates/hooks/progress_*.md`): the facts first,
the evidence the worker has not acted on quoted back, and what to do last.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from chickadee.adapters.base import SHARED_TEMPLATES, load_template
from chickadee.hooks.fire import HookFn
from chickadee.hooks.payloads import (
    AfterBatch,
    HookEvent,
    ProgressCall,
    Regression,
    RoundEnded,
    TestsSeen,
    WorkerProgress,
)
from chickadee.hooks.verdicts import Halt, HookVerdict, Note, Proceed

__all__ = [
    "HEAD",
    "REPORT_HEAD",
    "Due",
    "Ladder",
    "Unchecked",
    "WatchdogParams",
    "fill",
    "listed",
    "progress_report",
    "progress_watchdog",
    "regression_note",
    "replay",
    "summary",
    "tests_line",
    "unchecked",
    "wandering_threshold",
]

Meter = Literal["looping", "wandering", "unchecked"]
Rung = Literal["nudge", "report", "backstop"]

_TEMPLATES: Final = SHARED_TEMPLATES / "hooks"

HEAD: Final = "[progress watchdog]"
"""How each of the watchdog's texts to the worker begins, as its templates write it: what
the progress record knows its words by on an answer they ride on (`box/progress.py`)."""
REPORT_HEAD: Final = f"{HEAD} The task is still not moving"
"""How the report's request begins (`progress_report.md`): what the worker reads in its round's
hook lines to know it was asked to stop and report (`BeforeStop.handed_up`)."""
_SHOWN_MAX: Final = 6
"""At most this many calls or files listed in one message."""


class WatchdogParams(BaseModel):
    """The thresholds, each a reasoned heuristic: checked against
    recorded runs for false alarms (`box/progress_replay.py`), never fitted to them. Set under
    the hook's `with`; give `progress_report` the same, so it reads the ladder the watchdog
    climbed. A nudge is a note and cheap, so it comes early; a report starts a round of Athena's
    on the frontier model, so it needs as much evidence again."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    looping: int = Field(default=5, ge=2)
    """Calls that did nothing, net, before the nudge. The `Repetition` rule notes at 2
    identical calls and other harnesses act at 3 to 5; looping counts any call that did
    nothing, which is broader, so it is not lower. One refusal and its correction, or a re-run
    for flakiness, stay below."""
    report_looping: float = Field(default=2.0, gt=1)
    """The report comes at this times the threshold: after the nudge the meter must rise by
    `(report_looping - 1) * looping` more, at least 2, so 5 more calls that did nothing, 10 in
    all. A weak model re-checks (`git status` twice, held lines re-read, a re-run to be
    sure), and a report costs a frontier round."""
    wandering_base: int = Field(default=20, ge=1)
    """Responses to orient, with one call a response as a small model makes them: a `glob`
    or two, `tests_for`, a baseline run, a `grep`, and on a medium codebase 15 to 25 files read
    before the first edit."""
    wandering_per_file: int = Field(default=2, ge=0)
    """Responses per repository file the brief names: about two reads each."""
    wandering_min: int = Field(default=20, ge=2)
    """Room on a brief that names nothing."""
    wandering_max: int = Field(default=40, ge=2)
    """Responses with nothing written past which no brief is a reason to go on: a fifth of
    the default call cap (200)."""
    report_wandering: float = Field(default=1.5, gt=1)
    """The report comes at this times the threshold, as `report_looping` reads it: half the
    threshold again after the nudge, at least 2 more."""
    wandering_share: float = Field(default=0.2, gt=0, le=1)
    """At most this share of the calls the round may make, so the ladder fits well inside
    them, in a late round as in the first."""
    calls_per_response: int = Field(default=3, ge=1)
    """Wandering also fires at this many times its threshold in calls."""
    backstop: int = Field(default=3, ge=1)
    """Calls allowed after the report was asked for: one last read, then `finish`."""
    nudges: int = Field(default=2, ge=0)
    """Nudges at most in a round; a stall past them goes straight to the report."""
    unchecked: int = Field(default=25, ge=2)
    """Responses since the first change after the last test run: about six
    edit-and-look cycles of four responses with no run, where the tests would have said
    something. Generous, as D238: a worker that edits several files before its first run stays
    below."""
    report_unchecked: float = Field(default=1.5, gt=1)
    """The report comes at this times the threshold: 38 responses at the default."""


def wandering_threshold(params: WatchdogParams, brief_files: int, calls: int | None) -> int:
    """`W = base + per_file·n`, at least `wandering_min`, at most `wandering_max` and
    `wandering_share` of the `calls` the round may make: fewer calls lower it."""
    w = max(
        params.wandering_base + params.wandering_per_file * brief_files, params.wandering_min
    )
    w = min(w, params.wandering_max)
    if calls is not None:
        w = min(w, math.floor(params.wandering_share * calls))
    return max(w, 2)


def _rise(factor: float, threshold: int) -> int:
    """How far a meter must rise after the nudge before the report: a worker that recovers
    after the nudge never reaches it."""
    return max(math.ceil(threshold * (factor - 1)), 2)


@dataclass(frozen=True)
class Unchecked:
    """The calls since the last test run, counted from the first change after it (or the
    round's start): the one measure of "changed and not tested"."""

    responses: int = 0
    calls: int = 0
    files: tuple[str, ...] = ()
    """The files changed since, each once, in order."""


def unchecked(calls: Sequence[ProgressCall]) -> Unchecked:
    """`calls`, in order, measured since the last test run: nothing when no change came after
    it."""
    last = max((i for i, c in enumerate(calls) if c.run is not None), default=-1)
    after = calls[last + 1 :]
    first = next((i for i, c in enumerate(after) if c.tag == "changed"), None)
    if first is None:
        return Unchecked()
    tail = after[first:]
    files = dict.fromkeys(path for c in tail for path in c.changed)
    return Unchecked(len({c.response for c in tail}), sum(c.weight for c in tail), tuple(files))


@dataclass(frozen=True)
class Due:
    """One rung, given at the end of `response`: by which meter, and its evidence."""

    response: int
    rung: Rung
    meter: Meter
    calls: int
    """Calls since the last change when it was given."""
    responses: int
    """Responses counted toward wandering when it was given."""
    idle: tuple[ProgressCall, ...] = ()
    """The calls that did nothing (looping), or every call since the last change."""
    read: tuple[str, ...] = ()
    """The files read since the last change."""
    fresh_from: int = 1
    """The first call of the stall, from 1: a failure before it is not quoted."""
    unchecked: Unchecked = Unchecked()
    """The calls since the last test run, when it was given."""


@dataclass(frozen=True)
class Ladder:
    """What `replay` found of one round."""

    due: tuple[Due, ...]
    changes: int
    longest: int
    """The most calls in a row with no change."""
    since: tuple[ProgressCall, ...]
    """The calls since the last change, at the round's end."""

    @property
    def handed_up(self) -> Due | None:
        """The rung that handed the round up, the backstop over the report, or None."""
        found = [d for d in self.due if d.rung != "nudge"]
        return found[-1] if found else None

    @property
    def nudges(self) -> int:
        return sum(1 for d in self.due if d.rung == "nudge")


@dataclass
class _Stall:
    meter: Meter
    loop: int
    idle: int
    idle_calls: int
    unchecked: Unchecked
    """The meters when it was nudged: the report needs them to rise again."""


@dataclass
class _Meters:
    loop: int = 0
    nothing: list[ProgressCall] = field(default_factory=list[ProgressCall])
    """The calls that did nothing since the loop meter was last 0."""
    idle: int = 0
    idle_calls: int = 0
    since: list[ProgressCall] = field(default_factory=list[ProgressCall])
    """Every call since the last change."""
    last_files: frozenset[str] = frozenset()
    """The files the last response read."""
    changes: int = 0
    longest: int = 0
    seen: int = 0
    """The round's calls counted so far."""
    fresh_from: int = 1
    """The first call after the last change, from 1: where the stall began."""

    def count(self, calls: Sequence[ProgressCall]) -> bool:
        """One response's calls counted; whether one of them was a change."""
        changed = False
        for call in calls:
            self.seen += 1
            if call.tag == "changed":
                changed = True
                self.changes += 1
                self.loop, self.nothing, self.since = 0, [], []
                self.fresh_from = self.seen + 1
                continue
            self.since.append(call)
            self.longest = max(self.longest, len(self.since))
            if call.tag == "learned":
                self.loop = max(self.loop - 1, 0)
                if self.loop == 0:
                    self.nothing = []
            else:
                self.loop += 1
                self.nothing.append(call)
        # A call names a file only when it read one (`ProgressCall.file`).
        continuation = all(c.tag == "learned" and c.file in self.last_files for c in calls)
        # Calls count as the call cap counts them: a Python block by the calls it made.
        if changed:
            # The calls after the change in this response start the next stretch.
            self.idle, self.idle_calls = 0, sum(c.weight for c in self.since)
        else:
            # Reading one file on is one reading, not a response each; its calls still count,
            # so paging through a file with nothing written ends at the calls ceiling.
            self.idle += 0 if continuation else 1
            self.idle_calls += sum(c.weight for c in calls)
        self.last_files = frozenset(c.file for c in calls if c.file is not None)
        return changed


class _Climb:
    """The ladder, one response at a time."""

    def __init__(self, progress: WorkerProgress, params: WatchdogParams) -> None:
        self.params = params
        self.looping = params.looping
        self.wander = wandering_threshold(
            params, len(progress.brief_files), progress.calls_left
        )
        self.wander_calls = params.calls_per_response * self.wander
        self.runnable = progress.runnable
        self.unchecked_calls = params.calls_per_response * params.unchecked
        self.calls: list[ProgressCall] = []
        self.u = Unchecked()
        self.m = _Meters()
        self.due: list[Due] = []
        self.stalls: list[tuple[Meter, frozenset[str]]] = []
        self.active: _Stall | None = None
        self.reported: Meter | None = None
        """The meter whose report was asked for: the round's last rung but the backstop."""
        self.after_report = 0
        self.halted = False

    def step(self, response: int, calls: Sequence[ProgressCall]) -> None:
        changed = self.m.count(calls)
        self.calls.extend(calls)
        self.u = unchecked(self.calls)
        tested = any(c.run is not None for c in calls)
        if self.reported is not None:
            self.after_report += sum(c.weight for c in calls)
            if not self.halted and self.after_report >= self.params.backstop:
                self.halted = True
                self._give(response, "backstop", self.reported)
            return
        if self.active is not None and (
            tested if self.active.meter == "unchecked" else changed
        ):
            # Recovered: a change ends a stall short of its report; a test run, a stall of
            # changes without one.
            self.active = None
        if self.active is not None:
            if self._risen(self.active):
                self._give(response, "report", self.active.meter)
            return
        fired = self._fired()
        if fired is None:
            return
        evidence = _evidence(fired, self.m, self.u)
        same = any(m == fired and (e <= evidence or evidence <= e) for m, e in self.stalls)
        self.stalls.append((fired, evidence))
        nudged = sum(1 for d in self.due if d.rung == "nudge")
        if same or len(self.stalls) > 2 or nudged >= self.params.nudges:
            self._give(response, "report", fired)
        else:
            m = self.m
            self.active = _Stall(fired, m.loop, m.idle, m.idle_calls, self.u)
            self._give(response, "nudge", fired)

    def _fired(self) -> Meter | None:
        m = self.m
        if m.loop >= self.looping:
            return "looping"
        if m.idle >= self.wander or m.idle_calls >= self.wander_calls:
            return "wandering"
        u = self.u
        if self.runnable and (
            u.responses >= self.params.unchecked or u.calls >= self.unchecked_calls
        ):
            return "unchecked"
        return None

    def _risen(self, stall: _Stall) -> bool:
        """Whether the stall has gone on to its report: the meter that fired has risen by its
        report's margin since the nudge, or the other meter has reached its own report level.
        Not the other meter's margin from where the nudge found it: a worker nudged for a
        short loop that then reads to find the cause would be handed up sooner than one that
        only read."""
        m, params = self.m, self.params
        loop_rise = _rise(params.report_looping, self.looping)
        idle_rise = _rise(params.report_wandering, self.wander)
        calls_rise = _rise(params.report_wandering, self.wander_calls)
        looping = m.loop >= (
            stall.loop + loop_rise if stall.meter == "looping" else self.looping + loop_rise
        )
        if stall.meter == "wandering":
            wandering = (
                m.idle >= stall.idle + idle_rise
                or m.idle_calls >= stall.idle_calls + calls_rise
            )
        else:
            wandering = (
                m.idle >= self.wander + idle_rise
                or m.idle_calls >= self.wander_calls + calls_rise
            )
        u, at = self.u, params.unchecked
        u_rise = _rise(params.report_unchecked, at)
        u_calls_rise = _rise(params.report_unchecked, self.unchecked_calls)
        if stall.meter == "unchecked":
            checks = (
                u.responses >= stall.unchecked.responses + u_rise
                or u.calls >= stall.unchecked.calls + u_calls_rise
            )
        else:
            checks = self.runnable and (
                u.responses >= at + u_rise or u.calls >= self.unchecked_calls + u_calls_rise
            )
        return looping or wandering or checks

    def _give(self, response: int, rung: Rung, meter: Meter) -> None:
        if rung == "report":
            self.reported, self.after_report = meter, 0
        self.due.append(_due(response, rung, meter, self.m, self.u))


def replay(progress: WorkerProgress, params: WatchdogParams) -> Ladder:
    """The round's calls through the meters and the ladder, in order: deterministic, so every
    firing, a resumed round's included, finds the rungs the earlier firings gave."""
    climb = _Climb(progress, params)
    for response, calls in _responses(progress.calls):
        climb.step(response, calls)
    m = climb.m
    return Ladder(tuple(climb.due), m.changes, m.longest, tuple(m.since))


def _responses(
    calls: Sequence[ProgressCall],
) -> list[tuple[int, tuple[ProgressCall, ...]]]:
    grouped: list[tuple[int, list[ProgressCall]]] = []
    for call in calls:
        if not grouped or grouped[-1][0] != call.response:
            grouped.append((call.response, []))
        grouped[-1][1].append(call)
    return [(r, tuple(c)) for r, c in grouped]


def _evidence(meter: Meter, m: _Meters, u: Unchecked) -> frozenset[str]:
    """What makes a stall the same problem as an earlier one: the calls that did nothing, for
    looping; the files read without a write (or, reading none, the calls), for wandering; the
    files changed without a test run, for unchecked."""
    if meter == "unchecked":
        return frozenset(u.files)
    if meter == "looping":
        return frozenset(c.key for c in m.nothing)
    files = frozenset(c.file for c in m.since if c.file is not None)
    return files or frozenset(c.key for c in m.since)


def _due(response: int, rung: Rung, meter: Meter, m: _Meters, u: Unchecked) -> Due:
    idle = tuple(m.nothing) if meter == "looping" else tuple(m.since)
    read = tuple(dict.fromkeys(c.file for c in m.since if c.file is not None))
    return Due(response, rung, meter, len(m.since), m.idle, idle, read, m.fresh_from, u)


# -- the words ---------------------------------------------------------------------------------


def fill(name: str, **fields: str) -> str:
    """A template with its `{{field}}`s filled: a replace, never `str.format`, since what is
    quoted back (a traceback, a command) holds braces of its own."""
    text = load_template(_TEMPLATES, name).strip()
    for key, value in fields.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def listed(names: Sequence[str]) -> str:
    shown = ", ".join(names[:_SHOWN_MAX])
    more = len(names) - _SHOWN_MAX
    return f"{shown} and {more} more" if more > 0 else shown


def _repeated(calls: Sequence[ProgressCall], indent: str = "- ") -> str:
    counted = Counter(c.about for c in calls)
    rows = [
        f"{indent}{about}" + (f" ({n} times)" if n > 1 else "")
        for about, n in counted.most_common(_SHOWN_MAX)
    ]
    return "\n".join(rows)


def _within(progress: WorkerProgress, fresh_from: int) -> str | None:
    """The last failing output, if it came within the stall: one before the last change may
    be one the worker already answered, and quoting it would point it back there."""
    at = progress.failure_call
    return progress.last_failure if at is not None and at >= fresh_from else None


def _failure(progress: WorkerProgress, fresh_from: int) -> str:
    """The last failing output, quoted, and the repository files it names that the worker has
    not opened: the evidence most often ignored."""
    failed = _within(progress, fresh_from)
    if failed is None:
        return ""
    said = f"The last failure said:\n```\n{failed}\n```\n"
    unopened = [f for f in progress.failure_files if f not in progress.opened]
    if unopened:
        said += f"It names {listed(unopened)}, which you have not opened.\n"
    return said


def _nudge(due: Due, progress: WorkerProgress) -> str:
    if due.meter == "unchecked":
        return fill(
            "progress_nudge_unchecked.md",
            responses=str(due.unchecked.responses),
            files=listed(due.unchecked.files) or "none named",
        )
    if due.meter == "looping":
        return fill(
            "progress_nudge_looping.md",
            calls=str(due.calls),
            repeated=_repeated(due.idle),
            failure=_failure(progress, due.fresh_from),
        )
    opened = set(progress.opened)
    brief = [f"{f} ({'opened' if f in opened else 'not opened'})" for f in progress.brief_files]
    return fill(
        "progress_nudge_wandering.md",
        responses=str(due.responses),
        read=listed(due.read) or "none",
        brief=listed(brief) or "none named",
    )


def _words(due: Due, progress: WorkerProgress) -> HookVerdict:
    if due.rung == "nudge":
        return Note(_nudge(due, progress))
    if due.rung == "report":
        return Note(fill("progress_report.md", calls=str(due.calls)))
    return Halt(fill("progress_backstop.md", calls=str(due.calls)))


def _measured(ladder: Ladder, progress: WorkerProgress) -> str:
    """The section the reader of a handed-up round reads under the worker's report: what the
    worker cannot be trusted to retype."""
    up = ladder.handed_up
    if up is None:
        return ""
    why = (
        "the round was handed up by the backstop: asked to stop and report, the worker went "
        "on calling tools"
        if up.rung == "backstop"
        else f"the worker was asked to stop and report ({up.meter})"
    )
    repeated = _repeated([c for c in ladder.since if c.tag == "nothing"], indent="  - ")
    read = (
        f"- files read without a write: {listed(up.read)}\n"
        if up.meter == "wandering" and up.read
        else ""
    )
    failed = _within(progress, up.fresh_from)
    failure = f"- the last failing output:\n```\n{failed}\n```" if failed is not None else ""
    return fill(
        "progress_measured.md",
        why=why,
        since=str(len(ladder.since)),
        repeated=f"; the calls that did nothing:\n{repeated}" if repeated else "",
        read=read,
        failure=failure,
    ).strip()


def summary(ladder: Ladder, tests: TestsSeen | None = None) -> str:
    """The one line every report gains (the specification's §2.8), with the tests fixed and
    broken once a run was read."""
    line = (
        f"Progress (harness): {ladder.changes} change(s) to the files this round; the longest "
        f"stretch without one, {ladder.longest} call(s); {ladder.nudges} nudge(s) given"
    )
    if tests is not None:
        line += f"; tests fixed {len(tests.fixed)}, broken {len(tests.broken)}"
    return f"{line}."


def tests_line(tests: TestsSeen | None) -> str:
    """The task's tests as its runs left them, for the reader of the report; nothing before a
    run was read."""
    if tests is None:
        return ""
    edited = set(tests.edited)
    fixed = [f"{t} (its file changed by the task)" if t in edited else t for t in tests.fixed]
    return fill(
        "progress_tests.md",
        fixed=listed(fixed) or "none",
        broken=listed(tests.broken) or "none",
        failing=listed(tests.failing) or "none",
        already=listed(tests.already_failing) or "none",
    )


def regression_note(regressed: Regression) -> str:
    """The note a run that confirmed a test broken carries."""
    return fill(
        "progress_regression.md",
        test=regressed.test,
        passed_at=regressed.passed_at,
        changed=listed(regressed.changed) or "no file named",
    )


def progress_watchdog(params: WatchdogParams) -> HookFn:
    """At `after_batch`: what is due at the batch's response, if anything."""

    def run(event: HookEvent) -> HookVerdict:
        batch = event.payload
        if not isinstance(batch, AfterBatch) or batch.progress is None:
            return Proceed()
        progress = batch.progress
        if not progress.calls:
            return Proceed()
        last = progress.calls[-1].response
        now = [d for d in replay(progress, params).due if d.response == last]
        said = _words(now[-1], progress) if now else None
        noted = "\n".join(
            regression_note(r)
            for c in progress.calls
            if c.response == last
            for r in c.regressed
        )
        if not noted:
            return said if said is not None else Proceed()
        if said is None:
            return Note(noted)
        # A note rides with a rung's note; a halt ends the round, and says only itself.
        return Note(f"{said.text}\n\n{noted}") if isinstance(said, Note) else said

    return run


def progress_report(params: WatchdogParams) -> HookFn:
    """At `round_ended`, once a round, however it ended: the summary line for the report's
    reader, and what the harness
    measured when the round was handed up."""

    def run(event: HookEvent) -> HookVerdict:
        stop = event.payload
        if not isinstance(stop, RoundEnded) or stop.progress is None:
            return Proceed()
        ladder = replay(stop.progress, params)
        measured = _measured(ladder, stop.progress)
        tests = stop.progress.tests
        return Note(
            "\n".join(t for t in (summary(ladder, tests), tests_line(tests), measured) if t)
        )

    return run
