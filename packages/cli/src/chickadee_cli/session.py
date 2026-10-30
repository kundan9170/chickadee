"""Which session the REPL holds, and what the person saw of it.

**Opening.** New by default; `--session <id>` names one; `--continue` takes the one updated
most recently. The session log is the server's and the truth: this module only chooses.

**A new session starts where the work is**. It reads the repository's open work first: none, it
starts fresh; one, it continues it;
several, `choose` picks (the REPL asks once, the newest the default), and with no `choose` the
newest is taken, as a batch run takes it. `--new` (`fresh`) continues nothing. The start header
(`continuing_line`) says once what was continued.

**The transcript** is every line the REPL printed, appended to
`<state_dir>/transcripts/<session>.log`: what the person saw, at the verbosity they chose,
which the session log cannot say. It is a convenience, not a record: a lost line costs a
reader nothing the session log does not still hold, so it is a plain append.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from chickadee_serve.frames import TurnWorkspace
from chickadee_wire.client import Client
from chickadee_wire.errors import WireError
from chickadee_wire.models import OpenWorkInfo, SessionInfo

__all__ = [
    "Choose",
    "NoSessionToContinue",
    "Transcript",
    "ask_which",
    "continuing_line",
    "open_session",
    "transcript_path",
]

Choose = Callable[[Sequence[OpenWorkInfo]], str | None]
"""Picks the open work a new session continues, by its key, or None to start fresh."""


class NoSessionToContinue(Exception):
    """`--continue` in a repository with no session yet."""


def open_session(
    client: Client,
    *,
    session_id: str | None,
    resume_latest: bool,
    fresh: bool = False,
    choose: Choose | None = None,
    only: TurnWorkspace | None = None,
) -> SessionInfo:
    """The session the REPL holds: `session_id` when named (it must exist), the most recently
    updated with `resume_latest`, else a new one, continuing the repository's open work unless
    `fresh`: one is continued, several are `choose`n from, the newest when
    nothing chooses. `only`, the person's flag, continues open work of that kind alone."""
    if session_id is not None:
        return client.session_info(session_id)
    if resume_latest:
        known = client.sessions()
        if not known:
            raise NoSessionToContinue("no session to continue in this repository yet")
        return max(known, key=lambda s: (s.updated_at, s.last_seq))
    if fresh:
        return client.create_session()
    try:
        works = client.open_works()
    except WireError as refused:
        if refused.status != 404:
            raise
        # A server of an earlier 4.x wire has no `/works` route (minor versions are
        # compatible, `chickadee_wire.client`): it has no open work to offer, so start fresh.
        works = ()
    if only is not None:
        # The person's `--worktree` or `--in-place`: open work of
        # the other kind is not continued.
        works = tuple(w for w in works if w.mode == only)
    if not works:
        return client.create_session()
    key = works[0].key if len(works) == 1 or choose is None else choose(works)
    return client.create_session() if key is None else client.create_session(adopt=key)


def _named(work: OpenWorkInfo) -> str:
    return f'"{work.title}"' if work.title else (work.work or work.worktree_path or work.key)


def _where(work: OpenWorkInfo) -> str:
    if work.mode == "worktree":
        return f"in its worktree {work.worktree_path}"
    return "in your checkout"


def ask_which(
    works: Sequence[OpenWorkInfo],
    *,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> str | None:
    """Several pieces of open work: asked once, the newest the default.
    Enter takes the newest; `n` starts fresh; anything else unknown takes the default. An end
    of input (Ctrl-D) starts fresh, as `n` does; Ctrl-C raises `KeyboardInterrupt`, the
    person stopping, for the caller to end on."""
    write("Open work in this repository:")
    for n, work in enumerate(works, start=1):
        write(f"  {n}. {_named(work)}, {_where(work)}")
    write("  n. start fresh")
    try:
        answer = read("Continue which? [1] ").strip().lower()
    except EOFError:
        return None
    if answer == "n":
        return None
    if answer.isdigit() and 1 <= int(answer) <= len(works):
        return works[int(answer) - 1].key
    return works[0].key


def continuing_line(info: SessionInfo, *, repl: bool = True) -> str | None:
    """The start header's one line about the work a new session continues, or None. A batch
    run (`repl=False`) is told only what a batch run can do: no REPL command."""
    work = info.continuing
    if work is None:
        return None
    if not repl:
        return f"Continuing {_named(work)}, {_where(work)}; --new continues none of it."
    return (
        f"Continuing {_named(work)}, {_where(work)}. "
        "/handoff to land it; start with --new to continue none of it."
    )


def transcript_path(state_dir: Path, session_id: str) -> Path:
    return state_dir / "transcripts" / f"{session_id}.log"


class Transcript:
    """Appends each printed line to the session's transcript, opened once."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh: TextIO = path.open("a", encoding="utf-8")

    def write(self, line: str) -> None:
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
