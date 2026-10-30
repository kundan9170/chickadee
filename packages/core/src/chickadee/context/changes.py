"""`at="@changed"`'s lines: where a file differs from the run's base (architecture §7.4.3).

`GitChanges` is the reader's `ChangedLines` over one worktree and the commit its changes are
measured from: the box's run base (`box/episode.py`), or the source head of Zeus's session
worktree. Git computes it, through `chickadee.process`, never a direct spawn; the diff is
read-only and takes no optional lock, so a read stays parallel-safe beside an index pass or a
command in the same worktree.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from chickadee.context.spans import Span, coalesce
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.edits.formats.unified_diff import new_sides
from chickadee.process.harness import DIFF_FLAGS, run_git

__all__ = ["GitChanges", "changed_spans", "worktree_changes"]


_TO_THE_END: Final = sys.maxsize
"""A file new since the base differs on every line; the reader clips the span to its end."""


def changed_spans(diff: str) -> tuple[Span, ...]:
    """The new side's lines of each hunk of a `-U0` diff. A hunk that only deletes holds no
    line on the new side, and is marked on the line its deletion follows (the first, when it
    deleted the file's head), so a file whose only change is a deletion still reads changed."""
    spans: list[Span] = []
    for start, count in new_sides(diff):
        if count == 0:
            spans.append(Span(max(start, 1), max(start, 1)))
        else:
            spans.append(Span(start, start + count - 1))
    return coalesce(spans)


class GitChanges:
    """`ChangedLines` over the worktree at `root`, against the commit `base`."""

    def __init__(self, root: Path, base: str) -> None:
        self.root = root
        self.base = base

    def changed_lines(self, path: str) -> Sequence[Span] | None:
        """`path`'s lines that differ from `base`, as the worktree holds it: committed, staged
        or not. A file the base did not hold differs on every line. None when git cannot say."""
        done = run_git(
            self.root,
            "--no-optional-locks",
            "diff",
            *DIFF_FLAGS,
            "-U0",
            self.base,
            "--",
            path,
        )
        if done.timed_out or done.exit_code != 0:
            return None
        spans = changed_spans(done.stdout.decode("utf-8", "replace"))
        if spans:
            return spans
        # No hunk is either a file unchanged since the base or one git does not track yet (a
        # file the worker created): the base says which.
        held = run_git(self.root, "cat-file", "-e", f"{self.base}:{path}")
        if held.timed_out:
            return None
        return () if held.exit_code == 0 else (Span(1, _TO_THE_END),)


def worktree_changes(root: Path, ref: WorkspaceRef | None) -> GitChanges | None:
    """The changes of the session worktree `ref` names, against the head it was made from, when
    `root` is that worktree; None for any other root, the person's checkout among them, which
    has no base."""
    if ref is None or ref.worktree_path is None or Path(ref.worktree_path) != root:
        return None
    return GitChanges(root, ref.source_head)
