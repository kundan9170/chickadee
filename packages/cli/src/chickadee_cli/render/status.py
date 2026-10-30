"""The REPL's status line: always at the bottom,
`qwen3 · WORK 2/3 · $0.72 / $5.00 · context 41% (58k left)`.

Every field is folded from what the wire carries, never read from the engine:

- **the model:** the catalog model of Zeus's last call, from his `context` frames;
- **the stage:** the run in flight, folded by the board's own fold (`board.apply_record`):
  `PLAN` before any task, `WORK done/total` once tasks are delegated; nothing between runs;
- **spend and cap:** the session family's, from `/budget`, asked again after a charge at most
  every `BUDGET_EVERY_S`;
- **context:** Zeus's `context` frames: used of his window, amber past `CONTEXT_AMBER`, and
  `compacting…` while a cut runs;
- **jobs:** how many of the session's jobs run (`/jobs`), asked as the REPL opens, after
  each turn and on each job's exit turn, and held between: no frame says a job started, so a
  count folded from frames alone would miss one. Left out when none runs.

A field nothing has said yet is left out. Fed from the REPL's reader threads, drawn on the
prompt's: one lock guards it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Final

from pydantic import ValidationError

from chickadee_cli.render.board import RunView, apply_record
from chickadee_cli.render.stream import JOB_EXITED
from chickadee_cli.render.style import Style, Tone
from chickadee_serve.frames import ProgressFrame, RecordFrame
from chickadee_wire.errors import WireError
from chickadee_wire.models import (
    EpisodeRecord,
    HarnessTurnP,
    JobView,
    ModelCallP,
    SessionBudget,
    SpendRecord,
)
from chickadee_wire.records import is_run_record, payload

__all__ = ["BUDGET_EVERY_S", "CONTEXT_AMBER", "StatusLine"]

CONTEXT_AMBER: Final = 0.8
"""Past this share of Zeus's window, context is amber: a compaction is coming."""
BUDGET_EVERY_S: Final = 1.0
"""The most often spend is asked again, however fast the charges come."""


class StatusLine:
    """`budget` asks the server for the family's spend and cap (`Client.budget`); `jobs`, for
    the session's jobs (`Client.jobs`)."""

    def __init__(
        self,
        style: Style,
        budget: Callable[[], SessionBudget],
        jobs: Callable[[], Sequence[JobView]] = lambda: (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._style = style
        self._budget = budget
        self._jobs = jobs
        self._clock = clock
        self._lock = threading.Lock()
        """Guards the fields below."""
        self._model: str | None = None
        self._run: RunView | None = None
        self._spent: SessionBudget | None = None
        self._asked = float("-inf")
        self._used = 0
        self._window = 0
        self._compacting = False
        self._running = 0

    def progress(self, frame: ProgressFrame) -> None:
        if frame.kind in ("answer", "error"):
            self.recount()  # a turn ended: a job it started, or ended, is counted
            return
        if frame.kind != "context":
            return
        used, window = frame.payload.get("used"), frame.payload.get("window")
        model = frame.payload.get("model")
        with self._lock:
            if isinstance(used, int) and isinstance(window, int) and window > 0:
                self._used, self._window = used, window
            if isinstance(model, str) and model:
                self._model = model
            self._compacting = frame.payload.get("compacting") is True

    def record(self, frame: RecordFrame) -> None:
        try:
            typed = payload(frame)
        except ValidationError:
            return
        charged = exited = False
        with self._lock:
            match typed:
                case SpendRecord() | ModelCallP():
                    charged = True
                case HarnessTurnP() if typed.trigger == JOB_EXITED:
                    exited = True
                case EpisodeRecord():
                    self._run = RunView(typed.run_id)
                case _:
                    pass
            if (
                is_run_record(frame)
                and self._run is not None
                and frame.scope == self._run.run_id
            ):
                apply_record(self._run, frame)
        if charged:
            self.refresh()
        if exited:
            self.recount()

    def recount(self) -> None:
        """Ask how many jobs run. A server that cannot say leaves the last count shown."""
        try:
            running = sum(1 for job in self._jobs() if job.running)
        except WireError:
            return
        with self._lock:
            self._running = running

    def refresh(self, *, now: bool = False) -> None:
        """Ask for spend again, unless it was asked within `BUDGET_EVERY_S`. A server that
        cannot say leaves the last figure shown."""
        with self._lock:
            if not now and self._clock() - self._asked < BUDGET_EVERY_S:
                return
            self._asked = self._clock()
        try:
            spent = self._budget()
        except WireError:
            return
        with self._lock:
            self._spent = spent

    def line(self) -> str:
        with self._lock:
            segments = [(t, Tone.DIM) for t in (self._model, self._stage(), self._money()) if t]
            context, tone = self._context()
            running = self._running
        if context:
            segments.append((context, tone))
        if running:
            segments.append((f"{running} job{'' if running == 1 else 's'}", Tone.DIM))
        sep = self._style.verbatim(" · ", Tone.DIM)
        return sep.join(self._style.verbatim(text, tone) for text, tone in segments)

    # -- under the lock ------------------------------------------------------------------------

    def _stage(self) -> str | None:
        run = self._run
        if run is None or run.end is not None:
            return None
        if not run.tasks:
            return "PLAN"
        done = sum(1 for t in run.tasks.values() if t.state == "committed")
        return f"WORK {done}/{len(run.tasks)}"

    def _money(self) -> str | None:
        spent = self._spent
        if spent is None:
            return None
        cap = f" / ${spent.cap_usd:.2f}" if spent.cap_usd is not None else ""
        return f"${spent.spent_usd:.2f}{cap}"

    def _context(self) -> tuple[str, Tone]:
        if self._compacting:
            return "compacting…", Tone.ASK
        if self._window <= 0:
            return "", Tone.DIM
        share = self._used / self._window
        left = max(self._window - self._used, 0)
        text = f"context {share:.0%} ({_thousands(left)} left)"
        return text, Tone.ASK if share > CONTEXT_AMBER else Tone.DIM


def _thousands(tokens: int) -> str:
    return f"{tokens // 1000}k" if tokens >= 1000 else str(tokens)
