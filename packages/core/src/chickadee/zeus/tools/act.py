"""Family C — act: what Zeus's `shell` and `coder` are given (`33-tools.md` §3 rows 12-13).

The two tools that change anything, and neither can change tracked source from here: `shell`
runs sealed against a read-only workspace (`shell.py`, the `exec` port), and `coder`
hands the change to the box, which is the only writer (`zeus/delegate.py`). `shell`
serialises; `coder` is exclusive; the dispatcher schedules both from `ToolSpec.parallel`.

`coder` streams, so Zeus's loop runs it (`RoleExecutors`): `Delegations.open` names the run
and writes its episode line, the loop emits the delegation, and `finish` runs it and waits for
the report. A stream that ends between the two abandons the run, which releases its lease.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Final

from chickadee.context.results import CallCancelled, Interrupted, InvalidArgument, ToolResult
from chickadee.errors import (
    ConfigError,
    EngineBusy,
    HarnessBug,
    InfraError,
    RepoLeaseBusy,
    RepoNotIndexed,
)
from chickadee.tools.tool import Answer, Failure
from chickadee.zeus.delegate import BoxPort, OpenedEpisode, open_episode, parse

if TYPE_CHECKING:
    from chickadee.context.session import Session
    from chickadee.tools.validate import Arguments

__all__ = ["Delegations", "NoShell"]

NOT_OPENED: Final = (ConfigError, EngineBusy, RepoLeaseBusy, RepoNotIndexed, InfraError)
"""What the box refuses with before it opens anything: a request this engine has no path for,
an episode already in flight, a lease another process holds, no index, a missing tool."""


class NoShell:
    """The `exec` port of a loop built without a shell. `shell` is still offered, as the
    prefix lists it, so being asked to run one is a harness bug, never the model's."""

    def run(
        self, args: Arguments, *, operation_id: str, cancel: threading.Event | None
    ) -> ToolResult | Failure:
        raise HarnessBug("a Zeus loop without a shell was asked to run one")


class Delegations:
    """`coder`'s two steps over one session and its box."""

    def __init__(self, session: Session, box: BoxPort | None) -> None:
        self._session = session
        self._box = box

    def open(
        self,
        args: Arguments,
        *,
        opened_at_seq: int,
        cancel: threading.Event | None,
    ) -> OpenedEpisode | Failure:
        """Name the run and write its episode line; or the refusal, when nothing opened."""
        if self._box is None:
            raise HarnessBug("a Zeus loop without a box was asked to delegate")
        if cancel is not None and cancel.is_set():
            return CallCancelled("coder")
        delegation = parse(args, self._session)
        if isinstance(delegation, InvalidArgument):
            return delegation
        try:
            return open_episode(
                self._session,
                self._box,
                delegation,
                opened_at_seq=opened_at_seq,
                cancel=cancel,
            )
        except NOT_OPENED as exc:
            # Refused before anything opened: no lease, no run, no episode line. Known now,
            # so it is the call's answer, as a shell refused before it spawns is.
            return Interrupted(
                "coder",
                f"not run: {exc}",
                do="nothing started: fix what the reason names and call coder again, or tell "
                "the person why it cannot start",
            )

    @staticmethod
    def finish(opened: OpenedEpisode) -> Answer:
        """Run the box — its episode line is already durable — and return its report: the
        call's result. The run it opened is the episode line's."""
        return opened.run.report()
