"""`/diff`: what each task of a run changed, from `board.fold_run`; and
`/handoff`'s review: the session's worktree, its diff from the base, and the
choices the engine accepts.

A task's committed files are its commit effect's observed surface: what the run kept. One not
committed shows the files its last round changed, by its facts.

**Files, not lines.** No route serves a blob, so the line diff is not on the wire yet; the
output says so rather than showing what it does not have.
"""

from __future__ import annotations

from typing import Final

from chickadee_cli.render.board import BoardTask, RunView
from chickadee_cli.render.style import Style, Tone
from chickadee_wire.models import HandoffResult, HandoffSummary

__all__ = [
    "CHOICES",
    "NO_LINES",
    "asked_on_leaving",
    "diff_lines",
    "finalize_lines",
    "handoff_lines",
    "offered",
    "shown_line",
    "task_files",
]

CHOICES: Final = ("bring", "apply", "squash", "keep", "discard")
"""The five choices, in the order the review offers them: `bring`
first, the choice that leaves the work uncommitted for the person to review."""

NO_LINES: Final = "the line diff is not on the wire yet; these are the files"


def task_files(task: BoardTask) -> tuple[tuple[str, ...], bool]:
    """The files a task shows, and whether they are committed: its commit's observed surface
    when it has one (what the run kept), else the files its last round changed, by its facts.
    The one rule for it, `/diff`'s and the TUI's diff screen's."""
    if task.committed:
        return task.committed, True
    return task.changed, False


def diff_lines(view: RunView, style: Style) -> list[str]:
    lines = style.lines(f"run {view.run_id}: {view.goal or '(no run_start yet)'}")
    changed = False
    for t in view.tasks.values():
        files, committed = task_files(t)
        if not files:
            continue
        if committed:
            what, tone = "committed", Tone.GOOD
        else:
            what, tone = f"changed, not committed ({t.state})", Tone.DIM
        changed = True
        lines += style.lines(f"{t.task_id}  {what}", tone, indent="  ")
        for path in files:
            lines += style.lines(path, Tone.PLAIN, indent="      ")
    if not changed:
        lines += style.lines("no file changed", Tone.DIM, indent="  ")
    else:
        lines += style.lines(NO_LINES, Tone.DIM, indent="  ")
    return lines


def offered(summary: HandoffSummary) -> tuple[str, ...]:
    """The choices the engine would carry out now: none without a worktree, and never one
    `HandoffPreview.refusal` refuses (`summary.refused`), so the review offers no choice that
    then fails."""
    if summary.git is None:
        return ()
    refused = summary.refused or {}
    return tuple(c for c in CHOICES if c not in refused)


def asked_on_leaving(summary: HandoffSummary) -> bool:
    """Whether leaving asks for a choice: only over a worktree with work not
    yet handed off. One kept earlier was the person's *not yet*, and one without commits holds
    nothing to choose over. The REPL's leaving step and the TUI's both ask it."""
    git, ref = summary.git, summary.workspace
    return (
        git is not None and ref is not None and ref.handoff == "pending" and bool(git.commits)
    )


def shown_line(line: str) -> str:
    """A diff line with its control characters made visible: the bytes are the coder's, and
    an escape sequence in them would move the cursor or hide text on the one screen the
    person decides from. A tab stays a tab."""
    return "".join(
        ch if ch == "\t" or (ch >= " " and ch != "\x7f") else f"\\x{ord(ch):02x}" for ch in line
    )


def _diff_tone(line: str) -> Tone:
    if line.startswith(("+++", "---")):
        return Tone.PERSON
    if line.startswith("+"):
        return Tone.GOOD
    if line.startswith("-"):
        return Tone.BAD
    return Tone.DIM if line.startswith("@@") else Tone.PLAIN


def handoff_lines(summary: HandoffSummary, style: Style) -> list[str]:
    """What the session leaves behind, for the person to choose over: its episodes and spend,
    the worktree against its base, the diff, what bears on the choice (`notes`), and each
    choice with why one is refused. The record's `unverified` lines (the run header, behaviour
    not verified) are not shown here: the owner's call (2026-10-13), they are noise at the
    moment of choosing, and `/status` states the header."""
    runs = len(summary.episodes)
    last = f", the last {summary.last_outcome}" if summary.last_outcome else ""
    lines = style.lines(
        f"session {summary.session_id}: {runs} coder run(s){last}; "
        f"spent ${summary.spend.cost_usd:.4f}"
    )
    git, ref = summary.git, summary.workspace
    if git is None or ref is None:
        lines += style.lines("no worktree to hand off", Tone.DIM, indent="  ")
    else:
        kept = " (kept earlier)" if ref.handoff == "keep" else ""
        lines += style.lines(
            f"worktree {git.branch} at {git.worktree_path}{kept}", Tone.PLAIN, indent="  "
        )
        onto = git.source_branch or "a detached HEAD"
        lines += style.lines(
            f"{len(git.commits)} commit(s) from {git.base[:12]}, onto {onto}",
            Tone.PLAIN,
            indent="  ",
        )
        lines += [
            style.verbatim(shown_line(line), _diff_tone(line))
            for line in summary.diff.split("\n")
        ]
        if not summary.diff:
            lines += style.lines("no change from the base", Tone.DIM, indent="  ")
    for note in summary.notes:
        lines += style.lines(note, Tone.BAD, indent="  ")
    for choice, why in (summary.refused or {}).items():
        lines += style.lines(f"{choice} refused: {why}", Tone.BAD, indent="  ")
    offers = summary.offers or {}
    for choice in offered(summary):
        # A server one minor older sends no words for a choice: its name alone, never "name: ".
        said = f"{choice}: {offers[choice]}" if offers.get(choice) else choice
        lines += style.lines(said, Tone.DIM, indent="  ")
    return lines


def finalize_lines(result: HandoffResult, style: Style) -> list[str]:
    """What a handoff did: the choice and its detail, the commits it put on the source branch,
    and that nothing was pushed. `/handoff <choice>` prints it; the TUI's handoff screen shows
    it."""
    lines = style.lines(f"{result.choice}: {result.detail}", Tone.GOOD)
    if result.commits_applied:
        shas = ", ".join(c[:12] for c in result.commits_applied)
        lines += style.lines(f"on {result.source_branch}: {shas}", Tone.DIM, indent="  ")
    return lines + style.lines("nothing was pushed", Tone.DIM, indent="  ")
