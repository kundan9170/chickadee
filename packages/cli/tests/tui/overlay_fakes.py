"""A wire for the overlays' tests: the routes an overlay reads once, answered from fixed values
on the screen's own loop, and the lines its actions send, recorded rather than run.

An overlay reads through the app (`ChickadeeApp.read`, the `SessionReader.ask` it calls) and
acts through it (`command`, `submit`); these stand in for the reader and the REPL's thread, so
what an overlay shows and sends is tested without a server, as the panes are.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from recordings import recorded_run_id, run_frames
from tui_frames import SID

from chickadee_cli.tui.app import ChickadeeApp
from chickadee_serve.frames import RecordFrame
from chickadee_wire.models import (
    HandoffResult,
    HandoffSummary,
    Liveness,
    RunBlob,
    SessionBudget,
    SessionInfo,
    Undoable,
)

__all__ = ["FakeReader", "Wire", "handoff_summary", "hosted"]


@dataclass
class Wire:
    """The routes' answers, by route; a missing one is refused, as an unbuilt route is."""

    listed: tuple[SessionInfo, ...] = ()
    budgets: dict[str, SessionBudget] = field(default_factory=dict[str, SessionBudget])
    lives: dict[str, Liveness] = field(default_factory=dict[str, Liveness])
    undoables: tuple[Undoable, ...] = ()
    summary: HandoffSummary | None = None
    result: HandoffResult | None = None
    finalized: list[str] = field(default_factory=list[str])
    blobs: dict[str, str] = field(default_factory=dict[str, str])
    """Run blob texts by sha; one not here is refused, as an unnamed sha is."""

    def sessions(self) -> tuple[SessionInfo, ...]:
        return self.listed

    def budget(self, sid: str) -> SessionBudget:
        return self.budgets[sid]

    def liveness(self, sid: str) -> Liveness:
        return self.lives[sid]

    def run_journal(self, run_id: str) -> tuple[RecordFrame, ...]:
        assert run_id == recorded_run_id()
        return tuple(run_frames())

    def run_blob(self, run_id: str, sha: str) -> RunBlob:
        text = self.blobs[sha]
        return RunBlob(run_id=run_id, sha=sha, field="sha", size=len(text.encode()), text=text)

    def undoable(self, _sid: str) -> tuple[Undoable, ...]:
        return self.undoables

    def handoff(self, _sid: str) -> HandoffSummary:
        assert self.summary is not None
        return self.summary

    def finalize(self, _sid: str, choice: str) -> HandoffResult:
        self.finalized.append(choice)
        assert self.result is not None
        return self.result


class FakeReader:
    """`SessionReader`'s overlay half: `ask` answers at once; `refresh` is recorded."""

    def __init__(self, wire: Wire) -> None:
        self.wire = wire
        self.refreshed: list[str] = []

    def ask(
        self,
        read: Callable[[Any], Any],
        deliver: Callable[[Any], None],
        failed: Callable[[str], None],
    ) -> None:
        try:
            value = read(self.wire)
        except (KeyError, AssertionError) as exc:
            failed(f"refused: {exc}")
            return
        deliver(value)

    def refresh(self, routes: Iterable[str]) -> None:
        self.refreshed.extend(routes)

    def close(self) -> None:
        pass


@dataclass
class SentLines:
    commands: list[str] = field(default_factory=list[str])
    submitted: list[str] = field(default_factory=list[str])


def hosted(app: ChickadeeApp, wire: Wire) -> tuple[FakeReader, SentLines]:
    """`app` reading `wire` and recording what its overlays send."""
    reader = FakeReader(wire)
    sent = SentLines()
    app.live = reader  # type: ignore[assignment] # the reader's overlay half, faked
    app.command = sent.commands.append  # type: ignore[method-assign]
    app.submit = sent.submitted.append  # type: ignore[method-assign]
    return reader, sent


def handoff_summary(
    *,
    commits: int = 2,
    refused: dict[str, str] | None = None,
    diff: str = "",
    handoff: str = "pending",
) -> HandoffSummary:
    """A session's worktree with `commits` commits not yet handed off."""
    spend = {"scope": "session", "tokens_in": 0, "tokens_out": 0, "cached_in": 0,
             "cost_usd": 0.0, "wall_ms": 0}  # fmt: skip
    git = {
        "base": "b" * 40, "branch": "ccr-1", "commits": [f"{n}" * 40 for n in range(commits)],
        "drift": None, "lease_id": "L1", "mode": "worktree", "source_branch": "main",
        "source_clean": True, "source_head": "a" * 40, "tip": "c" * 40,
        "worktree_path": "/wt/s-1",
    }  # fmt: skip
    workspace = {
        "handoff": handoff, "lease_id": "L1", "mode": "worktree", "source_head": "a" * 40,
        "source_ref": "refs/heads/main", "branch": "ccr-1", "worktree_path": "/wt/s-1",
    }  # fmt: skip
    return HandoffSummary.model_validate(
        {
            "session_id": SID, "git": git, "workspace": workspace, "spend": spend,
            "episodes": ["R-1"], "last_outcome": "COMPLETED", "unverified": [], "diff": diff,
            "refused": refused, "notes": [],
            "offers": {
                "bring": f"write the {commits} commit(s)' changes into your checkout",
                "apply": f"fast-forward main onto the {commits} commit(s), each kept",
                "squash": f"put the {commits} commit(s) on main as one commit",
                "keep": "change nothing; the next coder call reattaches this worktree",
                "discard": "delete the worktree",
            },
        }
    )  # fmt: skip
