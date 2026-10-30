"""`--replay`: a recorded session served over the same wire.

A frontend can be built without a live engine: the wire is the contract, a recording is the
data, and the first live session is a change of address. It is also the admission test run
backwards. A frontend that renders a replayed run as it renders the live one has needed
nothing that was not on the record stream.

**What it serves.** `ReplayFacade` has the reading half of the facade over a scratch copy of
the recording, a directory laid out as a repository's local state (`sessions/`, and `runs/`
when the recording kept its runs). Its records are revealed at the recorded pace, divided by
`speed`: `0` reveals as fast as a client reads. Progress is synthesised from the records it
precedes: an assistant turn is preceded by `thinking`, a tool result by `tool_call`. Nothing
the facade lacks is pretended; those routes answer `NOT_BUILT`, and the mutating ones
`REPLAY_READ_ONLY`.

**Questions wait as live**. When the replay reaches a `question_asked`, it waits for a
client's answer, or the timeout, before revealing anything after it. An answer is appended to
`answers.jsonl` in the scratch directory, never to the recording, and the recording's own
`question_answered` follows as it was recorded.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from chickadee import clock
from chickadee.config import InteractionPolicy
from chickadee.context.lines import AssistantTurnP, ToolP, parse
from chickadee.context.project import JournalRecords
from chickadee.context.questions import LINES, QuestionAskedLine
from chickadee.context.session import SessionState
from chickadee.context.store import SessionEvent, SessionLog, sessions_root
from chickadee.durable import AppendLog
from chickadee.journal.paths import journal_path
from chickadee.journal.runs import RunProjections
from chickadee.run_reads import RunBlob, run_blob
from chickadee.scopes import Cursor, Page, run_page, session_page
from chickadee.state.roots import Roots
from chickadee_serve.frames import (
    STOP_CHOICES,
    ProgressFrame,
    ProgressKind,
    approval_choices,
)
from chickadee_serve.interaction import PendingQuestions
from chickadee_serve.progress import ProgressHub

if TYPE_CHECKING:
    from chickadee.engine import SessionInfo

__all__ = ["ANSWERS_NAME", "REPLAY_MAX_PAUSE_S", "ReplayFacade"]

log = logging.getLogger(__name__)

ANSWERS_NAME: Final = "answers.jsonl"

REPLAY_MAX_PAUSE_S: Final = 3.0
"""The longest a replay waits between two records, whatever the recording says. A replay
is for building and checking a frontend; an idle hour in the recording shows nothing."""


@dataclass(frozen=True)
class _Step:
    event: SessionEvent
    line: object | None
    """The line parsed by its kind's type (`context.lines.parse`); None for a kind another
    package owns, which a replay only reveals."""


class ReplayFacade:
    """The facade's reading half over a recorded session, revealed over time."""

    def __init__(
        self,
        recording: Path,
        session_id: str,
        scratch: Path,
        *,
        speed: float = 1.0,
        hub: ProgressHub | None = None,
        pending: PendingQuestions | None = None,
        timeout_s: float = InteractionPolicy().question_timeout_s,
    ) -> None:
        """`recording` is a repository-local state directory holding `sessions/<session_id>`.
        It is copied to `scratch`, which this replay owns, and never written."""
        if speed < 0:
            raise ValueError(f"a replay's speed is 0 or more, not {speed}")
        shutil.copytree(recording, scratch, dirs_exist_ok=True)
        self.session_id = session_id
        self.speed = speed
        self.hub = hub if hub is not None else ProgressHub()
        self.pending = pending if pending is not None else PendingQuestions()
        self.timeout_s = timeout_s
        self._scratch = scratch
        self._roots = Roots(
            project=scratch, repo_local=scratch, checkout_local=scratch, user=scratch
        )
        self._root = sessions_root(self._roots)
        self._records = JournalRecords(self._roots, RunProjections(self._roots))
        """The recording's run journals and blobs, read as the engine reads its own."""
        log_ = SessionLog.read(self._root, session_id)
        self._log = log_
        self._steps = tuple(_Step(e, _parsed(e, log_.payload(e))) for e in log_.events())
        self._lock = threading.Lock()
        self._horizon = 0
        """The last seq revealed. Guarded by `_lock`: the pacing thread writes it, request
        threads read it."""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- pacing ---------------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._play, name="chickadee-replay", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def horizon(self) -> int:
        with self._lock:
            return self._horizon

    @property
    def finished(self) -> bool:
        return self.horizon == (self._steps[-1].event.seq if self._steps else 0)

    def _play(self) -> None:
        previous: str | None = None
        for step in self._steps:
            if self._stop.wait(self._pause(previous, step.event.at)):
                return
            previous = step.event.at
            self._progress_before(step)
            with self._lock:
                self._horizon = step.event.seq
            if isinstance(step.line, QuestionAskedLine):
                self._wait_for_answer(step.line)
                if self._stop.is_set():
                    return

    def _pause(self, previous: str | None, at: str) -> float:
        if previous is None or self.speed == 0:
            return 0.0
        before, now = clock.parse(previous), clock.parse(at)
        if before is None or now is None:
            return 0.0
        gap = max(0.0, (now - before).total_seconds()) / self.speed
        return min(gap, REPLAY_MAX_PAUSE_S)

    def _progress_before(self, step: _Step) -> None:
        kind = _synthesised(step)
        if kind is not None:
            self.hub.publish(
                ProgressFrame(scope=self.session_id, kind=kind, payload={"seq": step.event.seq})
            )

    def _wait_for_answer(self, asked: QuestionAskedLine) -> None:
        question_id = asked.question_id
        if asked.question is not None:
            choices = tuple(o.key for o in asked.question.options)
        elif asked.stop is not None or asked.gate is not None:
            # A design gate an older log left open is answered as a stop would be.
            choices = STOP_CHOICES
        else:
            approval = asked.approval
            choices = approval_choices(
                grant=approval is not None and approval.grant is not None,
                exact=approval is not None and approval.exact is not None,
                offer_repo=approval is None or approval.offer_repo,
            )
        self.pending.open(self.session_id, question_id, choices)
        given = self.pending.wait(self.session_id, question_id, self.timeout_s, self._stop)
        if given is None:
            return
        line = {
            "id": question_id,
            "choice": given.choice,
            "token_class": given.token_class.value,
            "note": given.note,
            "at": clock.stamp(),
        }
        answers = AppendLog.open(self._scratch / ANSWERS_NAME, create=True)
        try:
            answers.append(json.dumps(line, sort_keys=True).encode() + b"\n", sync=True)
        finally:
            answers.close()

    # -- the facade's reading half --------------------------------------------------------

    def journal(self, scope_id: str, cursor: Cursor, limit: int) -> Page:
        """A scope's records past `cursor`, up to what has been revealed. A run's journal, if
        the recording kept it, is served whole: its pace is the session's `episode` record."""
        if scope_id != self.session_id:
            return run_page(journal_path(self._roots, scope_id), scope_id, cursor, limit)
        horizon = self.horizon
        if cursor.after >= horizon:
            return Page((), Cursor(cursor.after))
        page = session_page(self._log, Cursor(cursor.after), limit)
        shown = tuple(r for r in page.records if r.seq <= horizon)
        after = shown[-1].seq if shown else cursor.after
        return Page(shown, Cursor(after))

    def run_blob(self, run_id: str, sha: str) -> RunBlob:
        """A blob a recorded run's record names, as the engine serves its own. A run's diff is
        not served: its trees are in a repository a recording lacks."""
        return run_blob(self._records, run_id, sha)

    def sessions(self) -> tuple[SessionInfo, ...]:
        return (self.session_info(self.session_id),)

    def session_info(self, session_id: str) -> SessionInfo:
        from chickadee.engine import SessionInfo
        from chickadee.errors import SessionError

        if session_id != self.session_id:
            raise SessionError(
                f"this replay serves session {self.session_id}, not {session_id}"
            )
        horizon = self.horizon
        shown = [s.event for s in self._steps if s.event.seq <= horizon]
        return SessionInfo(
            session_id=session_id,
            last_seq=horizon,
            last_run_id=next(
                (
                    s.event.run_id
                    for s in reversed(self._steps)
                    if s.event.kind == "episode" and s.event.seq <= horizon
                ),
                None,
            ),
            created_at=shown[0].at if shown else "",
            updated_at=shown[-1].at if shown else "",
        )

    def hidden(self, session_id: str) -> tuple[int, ...]:
        """What the recording's rewinds hide, as far as it has been revealed."""
        self.session_info(session_id)
        horizon = self.horizon
        log_ = SessionLog.read(self._root, self.session_id, upto=horizon) if horizon else None
        if log_ is None:
            return ()
        state = SessionState.replay(self.session_id, self._root, log_.events(), log_.payload)
        return tuple(sorted(state.hidden))


def _parsed(event: SessionEvent, raw: bytes) -> object | None:
    """A line by its kind's type: the session fold's own (`context.lines`), or the question
    kinds' (`context.questions`), whose writer owns them."""
    owned = LINES.get(event.kind)
    if owned is not None:
        return TypeAdapter(owned).validate_json(raw)
    return parse(event.kind, raw)


def _synthesised(step: _Step) -> ProgressKind | None:
    """The progress a live turn would have streamed before this record committed."""
    if isinstance(step.line, AssistantTurnP):
        return "thinking"
    if isinstance(step.line, ToolP):
        return "tool_call"
    return None
