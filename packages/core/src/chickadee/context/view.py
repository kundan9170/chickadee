"""Zeus's view of the session's workspace.

The box writes the session's worktree. Zeus reads through a `SessionView`:
one root, and the parts that read it — the `Reader` and its `HandleResolver`, the `Shell` with
its deny-list masks, the graph's navigator and `locate` — built together over that root as a
`Rooted`. It starts at the checkout and moves when the session's `workspace` line names a
worktree the session still writes, and only then: a reattach moves nothing.

The view is the ports Zeus's tools are given, so a turn's `ToolContext`, built once, still sees
a move made by a `coder` call in the middle of it. Each call reads the current `Rooted` once,
at its start, under the lock, and finishes against it. Every call of a turn runs on the
conversation's worker, so the parts are built and used on one thread, as the graph's SQLite
handle needs.

After the move, a path or working directory in the person's checkout still runs, through the
checkout's own parts, and its result opens with one line saying whose files those are (the
owner's call: label, never refuse). The worktree itself lives under the checkout's `.git`, so
"in the checkout" means inside the checkout and not inside the view's root.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from chickadee.context.results import ToolResult
from chickadee.context.session import WorkspaceLine
from chickadee.paths import Place, within

if TYPE_CHECKING:
    from chickadee.context.attachment import CheckedFile, Minted
    from chickadee.context.handle import Handle, HandleKind
    from chickadee.context.reader import Reader
    from chickadee.context.resolver import HandleResolver
    from chickadee.context.results import CallCancelled, ToolFailure
    from chickadee.context.session import SessionState
    from chickadee.context.workspace_line import WorkspaceRef
    from chickadee.schema.isolation import WorkspaceMode
    from chickadee.tools.context import ExecPort, HandleStore, Locator, Navigator, SearchPort
    from chickadee.tools.tool import Failure
    from chickadee.tools.validate import Arguments

__all__ = [
    "REATTACHED",
    "Rooted",
    "SessionView",
    "checkout_label",
    "held_worktree",
    "last_worktree",
    "latest_workspace",
    "run_workspace",
    "run_worktree",
    "session_mode",
    "still_written",
]

REATTACHED: Final = frozenset({"pending", "keep"})
"""The handoffs after which the session still writes its worktree: none yet,
or `keep`, the person's *not yet*. After `apply`, `squash` or `discard` the next call opens a
new one, and the view goes back to the checkout."""


def still_written(ref: WorkspaceRef | None) -> WorkspaceRef | None:
    """`ref`, if it names a worktree the session still writes: the one its next `coder` call
    reattaches, the one a handoff acts on, and the one Zeus's view is in. The one home of that
    rule. The line is a reference that may outlive its directory: the box adds a gone one back
    from its branch (`workspace.reattach_worktree`), and the view stays where it is until
    then."""
    if ref is None or ref.mode != "worktree" or ref.handoff not in REATTACHED:
        return None
    return ref if ref.worktree_path is not None and ref.branch is not None else None


def latest_workspace(state: SessionState) -> WorkspaceRef | None:
    """The session's latest `workspace` line: where the session is now, its worktree or, after
    `worktree leave`, the checkout."""
    return state.workspace_at(state.last_seq)


def last_worktree(
    state: SessionState, *, upto: int | None = None, branch: str | None = None
) -> WorkspaceRef | None:
    """The latest `workspace` line naming a worktree, whatever lines naming the checkout came
    after it (a session may leave its worktree and keep it); as of `seq <=
    upto` when given, and of the worktree on `branch` when given: its branch, since a
    discard's line keeps the branch and drops the path. Every line is read, hidden ones too,
    as `SessionState.workspace_at` reads them."""
    found: WorkspaceRef | None = None
    for _event, record in state.lines(1, upto, with_hidden=True):
        if (
            isinstance(record, WorkspaceLine)
            and record.ref.mode == "worktree"
            and (branch is None or record.ref.branch == branch)
        ):
            found = record.ref
    return found


def held_worktree(state: SessionState) -> WorkspaceRef | None:
    """The worktree the session holds: its latest line naming one, while it still writes it
    (`still_written`), even after the session left it for the checkout. What
    a handoff acts on, what `worktree enter` and a `coder` call in a worktree reattach, and what
    a later session continues (`open_works`)."""
    return still_written(last_worktree(state))


def session_mode(state: SessionState, configured: WorkspaceMode) -> WorkspaceMode:
    """Where work with no `workspace` of its own runs: where the later of
    two things put the session, its latest `workspace` line (a worktree it still writes, or the
    checkout it was moved back to) and the person's latest `--worktree` or `--in-place`; else
    `configured`, `[workspace] mode`. The one home of that rule, read by `coder`'s dispatch and
    by where a design's work folder opens."""
    placed: tuple[int, WorkspaceMode] | None = None
    for _event, record in state.lines(1, None, with_hidden=True):
        if isinstance(record, WorkspaceLine):
            ref = record.ref
            if still_written(ref) is not None or ref.mode == "in_place":
                placed = (record.seq, ref.mode)
            else:
                placed = None  # handed off: the session is nowhere the person put it
    flagged = next(
        (t for t in reversed(state.turns()) if t.role == "user" and t.workspace is not None),
        None,
    )
    if placed is not None and (flagged is None or placed[0] > flagged.seq):
        return placed[1]
    if flagged is not None and flagged.workspace is not None:
        return flagged.workspace
    return configured


def run_workspace(state: SessionState, run_id: str) -> WorkspaceRef | None:
    """The session's worktree run `run_id` wrote in: the `workspace` line at
    its `episode` line, since a new worktree is recorded while its run is opened, before the
    episode line names the run, if that line was still written (`still_written`). A line
    already handed off when the run opened names a worktree the run could not write: a `coder`
    call after a handoff opens a new one, and an in-place run writes no line. None for that, for
    a run that ran in place, and for one the session does not name. The caller holds the
    session's lock."""
    episode = state.episode(run_id)
    return still_written(state.workspace_at(episode.seq)) if episode is not None else None


def run_worktree(state: SessionState, run_id: str) -> Path | None:
    """`run_workspace`'s directory: where the run's code is."""
    ref = run_workspace(state, run_id)
    return Path(ref.worktree_path) if ref is not None and ref.worktree_path else None


def checkout_label(branch: str | None) -> str:
    """The one line a result from the person's checkout opens with, after the move."""
    named = f" `{branch}`" if branch else ""
    return (
        f"this is your checkout, not the session's worktree{named}; the coder's edits are not "
        "here until the handoff"
    )


@dataclass(frozen=True)
class Rooted:
    """What Zeus's tools read through, over one root, built together so they move together:
    the read root, the masks the shell resolved for it, and the graph over it."""

    root: Path
    resolver: HandleResolver
    files: Reader
    shell: ExecPort
    navigator: Navigator
    locator: Locator
    search: SearchPort
    """`grep` and `glob` over this root, masked as the shell is."""
    agent_shell: Callable[[HandleStore], ExecPort] | None = None
    """A review agent's shell over this root: T0 only, never asked, storing its output
    in the agent's own reader. None where no executor was composed."""
    masked: tuple[Path, ...] = ()
    """What a process that reads none of the repository is masked from (the
    `lightpanda` engine): this root, the checkout, and every secret the shell's masks name."""


class SessionView:
    """`FileView`, `HandleStore`, `Navigator`, `ExecPort` and `Locator` over the session's
    current root (`tools.context`'s ports, structurally).

    `build` makes a `Rooted` over a root; it is called on the conversation's worker, at
    composition and at a move, once per root (`_built`), and whatever it opens lives until the
    session closes, since a call that began before a move may still hold the parts it began
    with. `_lock` guards `_where`, `_current` and `_built`, the only state that changes."""

    def __init__(
        self, checkout: Path, build: Callable[[Path], Rooted], at: WorkspaceRef | None = None
    ) -> None:
        self._build = build
        self._checkout = build(checkout)
        self._lock = threading.Lock()
        self._where: WorkspaceRef | None = None
        self._current = self._checkout
        self._built: dict[Path, Rooted] = {self._checkout.root: self._checkout}
        self.follow(at)

    # -- the move ------------------------------------------------------------------------------

    @property
    def root(self) -> Path:
        with self._lock:
            return self._current.root

    @property
    def where(self) -> WorkspaceRef | None:
        """The worktree the view is in, or None for the checkout."""
        with self._lock:
            return self._where

    def resolver(self) -> HandleResolver:
        """The current root's resolver, for compaction's freshness check."""
        with self._lock:
            return self._current.resolver

    def current(self) -> Rooted:
        """The current root's parts, for a call that builds its own over them: an agent's
        reader and shell."""
        with self._lock:
            return self._current

    def follow(self, ref: WorkspaceRef | None) -> WorkspaceRef | None:
        """Move to where the session's latest `workspace` line puts the view: the worktree it
        names while the session still writes one, else the checkout. A line naming where the
        view already is moves nothing, and nor does one whose worktree directory is gone: the
        view stays where it is until the box adds it back (`still_written`). Returns the
        worktree the view just entered, for the `coder` result to say so; None when it did not
        enter one. A part that cannot be built raises, and the view stays where it was."""
        target = still_written(ref)
        if target is not None and not Path(str(target.worktree_path)).is_dir():
            return None
        with self._lock:
            if _key(target) == _key(self._where):
                return None
            root = self._checkout.root if target is None else Path(str(target.worktree_path))
            rooted = self._built.get(root)
        if rooted is None:
            rooted = self._build(root)
        with self._lock:
            self._built[root] = rooted
            self._where, self._current = target, rooted
        return target

    def _at_call(self, path: str | None) -> tuple[Rooted, _Elsewhere | None]:
        """The parts a call runs against, read once at its start. After the move, a path that
        lands in the checkout and not in the view's root, absolute or climbing out of it, runs
        through the checkout's own parts, as the absolute path it lands on, and its result
        opens with the label; anything else runs through the current parts as given."""
        with self._lock:
            current, where = self._current, self._where
        if where is None or path is None:
            return current, None
        if within(current.root, path) is not None:
            return current, None
        landed = within(self._checkout.root, current.root / path)
        if landed is None:
            return current, None
        return self._checkout, _Elsewhere(str(landed), checkout_label(where.branch))

    # -- FileView and HandleStore --------------------------------------------------------------

    def read(
        self,
        target: str,
        *,
        lines: str | None = None,
        at: str | None = None,
        depth: str | None = None,
        max_bytes: int | None = None,
        call_id: str | None = None,
    ) -> ToolResult | ToolFailure:
        parts, there = self._at_call(target)
        target = target if there is None else there.path
        got = parts.files.read(
            target, lines=lines, at=at, depth=depth, max_bytes=max_bytes, call_id=call_id
        )
        return _labelled(got, there)

    def repeat_key(
        self,
        target: str,
        *,
        lines: str | None = None,
        at: str | None = None,
        depth: str | None = None,
    ) -> Hashable | None:
        parts, there = self._at_call(target)
        target = target if there is None else there.path
        return parts.files.repeat_key(target, lines=lines, at=at, depth=depth)

    def grep_in(
        self, handle: str, pattern: str, *, context: int = 2
    ) -> ToolResult | ToolFailure:
        parts, _ = self._at_call(None)
        return parts.files.grep_in(handle, pattern, context=context)

    def place(self, sent: str) -> Place | None:
        return self._at_call(None)[0].files.place(sent)

    def store(self, kind: HandleKind, data: bytes) -> Handle:
        parts, _ = self._at_call(None)
        return parts.files.store(kind, data)

    def attach(self, checked: CheckedFile, *, inline_bytes: int) -> Minted:
        """A file the person sent, kept by the reader the view is on now."""
        parts, _ = self._at_call(None)
        return parts.files.attach(checked, inline_bytes=inline_bytes)

    # -- Navigator -----------------------------------------------------------------------------

    def atlas(self) -> str | ToolFailure:
        return self._at_call(None)[0].navigator.atlas()

    def symbol(self, name: str) -> str | ToolFailure:
        return self._at_call(None)[0].navigator.symbol(name)

    def neighbors(self, target: str, depth: int) -> str | ToolFailure:
        parts, there = self._at_call(target)
        got = parts.navigator.neighbors(target if there is None else there.path, depth)
        return f"{there.label}\n{got}" if there is not None and isinstance(got, str) else got

    def impact(self, symbols: Sequence[str]) -> str | ToolFailure:
        return self._at_call(None)[0].navigator.impact(symbols)

    def tests_for(self, paths: Sequence[str]) -> str | ToolFailure:
        return self._at_call(None)[0].navigator.tests_for(paths)

    # -- ExecPort and Locator ------------------------------------------------------------------

    def run(
        self, args: Arguments, *, operation_id: str, cancel: threading.Event | None
    ) -> ToolResult | Failure:
        parts, there = self._at_call(args.text_or_none("cwd"))
        if there is not None:
            args = dataclasses.replace(args, sent={**args.sent, "cwd": there.path})
        got = parts.shell.run(args, operation_id=operation_id, cancel=cancel)
        return _labelled(got, there)

    def locate(
        self, question: str, *, cancel: threading.Event | None
    ) -> str | ToolFailure | CallCancelled:
        return self._at_call(None)[0].locator.locate(question, cancel=cancel)

    # -- SearchPort, as a bound method (`ZeusTools.search`) -----------------------------------

    def search(
        self,
        query: str,
        *,
        kind: Literal["text", "path"] = "text",
        paths: Sequence[str] = (),
        cancel: threading.Event | None = None,
    ) -> str:
        return self._at_call(None)[0].search(query, kind=kind, paths=paths, cancel=cancel)


def _key(ref: WorkspaceRef | None) -> tuple[str | None, str | None]:
    """What a move is keyed on: the worktree's path and branch, never its head."""
    return (None, None) if ref is None else (ref.worktree_path, ref.branch)


@dataclass(frozen=True)
class _Elsewhere:
    """A call that lands in the person's checkout after the move: where, and what it says."""

    path: str
    label: str


def _labelled[R](got: R, there: _Elsewhere | None) -> R:
    if there is None or not isinstance(got, ToolResult):
        return got
    label = there.label
    header = f"{label}\n{got.header}" if got.header else label
    return dataclasses.replace(got, header=header)  # pyright: ignore[reportReturnType]
