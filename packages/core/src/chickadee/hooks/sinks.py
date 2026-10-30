"""Where `hook` lines are written and read back: a run's journal for the
box's roles, Zeus's session log for his. One payload, `HookP`, in both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final

from pydantic import TypeAdapter, with_config

from chickadee.context.lines import LINE, SessionKind
from chickadee.context.session import Session
from chickadee.journal.port import JournalPort
from chickadee.journal.projection import Placed
from chickadee.schema.journal import EventKind, HookP

__all__ = ["HookLine", "RunHookSink", "SessionHookSink"]


class RunHookSink:
    """`HookSink` over a run's journal: each line at the task and round it is about."""

    def __init__(self, journal: JournalPort) -> None:
        self._journal = journal

    def hook(self, line: HookP, *, task_id: str | None, round: int | None) -> int:
        return self._journal.append(EventKind.HOOK, line, task_id=task_id, round=round)

    def blob(self, content: bytes) -> str:
        return self._journal.put_blob(content)

    def read(self, sha: str) -> bytes:
        return self._journal.get_blob(sha)

    def lines(self) -> tuple[Placed[HookP], ...]:
        return self._journal.projection().lines(HookP)


@with_config(LINE)
@dataclass(frozen=True)
class HookLine:
    """The session log's `hook` line: the run journal's `HookP`, whole. A session line has no
    task or round, so a Zeus hook's line names none."""

    KIND: ClassVar[SessionKind] = "hook"
    hook: HookP


_LINE: Final = TypeAdapter(HookLine)


class SessionHookSink:
    """`HookSink` over Zeus's session log."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def hook(self, line: HookP, *, task_id: str | None, round: int | None) -> int:
        del task_id, round  # a session line has neither
        return self._session.append(HookLine(line)).seq

    def blob(self, content: bytes) -> str:
        return self._session.put_blob(content)

    def read(self, sha: str) -> bytes:
        return self._session.blob(sha)

    def lines(self) -> tuple[Placed[HookP], ...]:
        session = self._session
        with session.lock:
            events = [e for e, _ in session.state.lines() if e.kind == HookLine.KIND]
        return tuple(
            Placed(e.seq, _LINE.validate_json(session.blob(e.payload_sha)).hook) for e in events
        )
