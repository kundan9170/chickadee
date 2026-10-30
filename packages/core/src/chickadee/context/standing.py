"""Standing: what still binds, told as appended notices.

> **Check always. Tell only on change. Push only what Zeus cannot ask for.**

Every other layer preserves the past; this one preserves what still binds — and it is not a
block that gets rewritten, because rewriting is an edit even when the text is identical: the
position moves, so the cache never matches. A notice is **appended**, once, and never edited,
corrected or removed; only a later cut removes it.

**A notice states an event, never current state.** "C-9 recorded" is true forever, which is
what makes append-only safe. The one exception is the snapshot a compaction appends, and its
stamp — *as of seq N · later notices amend this* — is what keeps it honest.

**The trigger list is closed**, as `QuestionKind` is. Ten entries,
each admitted on the test; an eleventh is a change to §5.11.2, never a commit.
"""

from __future__ import annotations

import re
from collections.abc import Sequence, Set
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Final, cast

from chickadee.journal.effects import KINDS, EffectKind, sealed
from chickadee.schema.interaction import network_opened
from chickadee.schema.journal import EffectCompletedP

if TYPE_CHECKING:
    from chickadee.context.handle import Handle
    from chickadee.context.ledger import Constraint
    from chickadee.context.project import RunRecords
    from chickadee.context.session import Session, SessionState
    from chickadee.context.spans import Span

__all__ = [
    "FRAME_TAGS",
    "NOTICE_TRIGGERS",
    "SURFACE_MAX",
    "ForkEffect",
    "Notice",
    "NoticeTrigger",
    "code_kept",
    "concern_raised",
    "constraint_proposed",
    "constraint_recorded",
    "constraint_row",
    "escape_frames",
    "fork_effects",
    "job_exited",
    "journaled_effects",
    "notice",
    "paused_run_ended",
    "run_ended",
    "run_reported",
    "standing_snapshot",
    "turns_phrase",
    "view_stale",
]

SURFACE_MAX: Final = 2
"""Distinct anchor PATHS carried in a notice. A bound on the notice, never on how many
concerns may exist — `concerns(all=True)` always enumerates them."""

FRAME_TAGS: Final = ("harness_context", "harness_notice")
"""The two frames harness text travels in. Content never contains one unescaped, or
a quoted line could close the frame and read as a fresh user turn."""


def _worktree_only(kind: str) -> bool:
    """An effect kind confined to the code a run wrote: `KINDS`' column. A fork never
    inherits a worktree run's worktree, and a turn snapshot restores or names an
    in-place run's; `journaled_effects` says which. Every other completed
    effect already happened in the world; a kind the table does not know is one of those."""
    try:
        return KINDS[EffectKind(kind)].worktree_only
    except ValueError:
        return False


class NoticeTrigger(IntEnum):
    """The table, by its own numbers."""

    CONSTRAINT = 1
    """A constraint recorded or superseded, or a checkpoint pointing at a turn."""
    CONCERN = 2
    """A concern raised on a NEW anchor path — the path is what surfaces, not the text."""
    STALE_VIEW = 3
    """An expanded view in context went stale."""
    RUN_ENDED = 4
    """A run ended leaving work that was not applied, or a run resumed after a pause ended:
    no `coder` call waits on it to say so."""
    SNAPSHOT = 5
    """A compaction committed: the surviving state, restated once."""
    FORK_EFFECTS = 6
    """A fork arrived carrying effects it cannot undo (at most once in a session's life), or a
    fork or rewind kept code the model no longer remembers changing (once per rewind)."""
    VIEW_MOVED = 7
    """Zeus's view moved into the session's worktree: said once per workspace
    change, in the `coder` result that made it, since a notice cannot sit between a call and its
    result."""
    JOB_EXITED = 8
    """A job ended and nobody asked for it, or saw it end:
    written by its waiter, held by `assemble` until any waiting call's result is in."""
    HOOK = 9
    """A hook the person bound spoke to Zeus where no answer of his carries it: at a turn's
    start, at his stop, after a batch. None unless a hook is bound to say it."""
    TEAM_REPLIED = 10
    """The team replied to one of his tells: once per reply. A reply that
    needs him (`needs_design`, `declined`) wakes him; one that does not waits for his turn."""


NOTICE_TRIGGERS: Final[tuple[NoticeTrigger, ...]] = tuple(NoticeTrigger)


def escape_frames(text: str) -> str:
    """Neutralise any frame tag inside harness-framed content, so nothing quoted can end the
    frame it sits in."""
    return _FRAME_TAG_RE.sub(r"&lt;\1", text)


_FRAME_TAG_RE: Final = re.compile(
    r"<(\s*/?\s*(?:" + "|".join(FRAME_TAGS) + r"))(?=[\s/>]|$)", re.IGNORECASE
)
"""Any spelling a model might read as a frame tag, not only the exact one: a space before the
`>`, another case, an attribute, a space after the `<` or the `/`."""


@dataclass(frozen=True)
class Notice:
    """One appended line. `handle` names the view a staleness notice is about, so the fold
    knows it was told and the sweep never says it twice."""

    trigger: NoticeTrigger
    text: str
    handle: str | None = None
    carries: Handle | None = None
    """The handle the notice's text names for Zeus to read (a run's report as a
    trigger-4 notice carries its diff's or journal's), registered in the session's table as a
    result's handle is."""

    def __post_init__(self) -> None:
        if self.trigger not in NOTICE_TRIGGERS:
            raise ValueError(f"{self.trigger!r} is not one of §5.11.2's notice triggers")
        if not self.text:
            raise ValueError("a notice says something")

    def framed(self) -> str:
        return f"<harness_notice>\n{escape_frames(self.text)}\n</harness_notice>"


def notice(session: Session, n: Notice) -> int:
    """Append ONE framed line to the context, as a durable harness line of the
    session log. Whatever the notice reports is committed first — the caller's line precedes
    this one — and nothing here is ever rewritten."""
    return session.record_notice(n)


# -- the builders: one per trigger ------------------------------------------------------


def turns_phrase(seqs: Sequence[int]) -> str:
    """ "turn 22", or "turns 22, 30" — a turn is named by its seq."""
    if len(seqs) == 1:
        return f"turn {seqs[0]}"
    return "turns " + ", ".join(str(s) for s in seqs)


def _one_line(text: str) -> str:
    return " ".join(text.splitlines())


def constraint_recorded(c: Constraint, *, supersedes: str | None = None) -> Notice:
    """Trigger 1: `constraint C-9 recorded · <text> · from turn 22`."""
    words = " (your words)" if c.author == "user" else ""
    source = turns_phrase(c.sources)
    text = f"constraint {c.id} recorded · {_one_line(c.text)}{words} · from {source}"
    if supersedes is not None:
        text += f" · supersedes {supersedes}"
    return Notice(NoticeTrigger.CONSTRAINT, text)


def constraint_proposed(source_seq: int, scope: str) -> Notice:
    """Trigger 1's widened row: a checkpoint may only POINT at a turn; nothing binds
    until Zeus writes a constraint citing it."""
    return Notice(
        NoticeTrigger.CONSTRAINT,
        f"turn {source_seq} may hold an unrecorded constraint (checkpoint) · {scope}",
    )


def _paths(paths: Sequence[str]) -> str:
    shown = " · ".join(paths[:SURFACE_MAX])
    more = len(paths) - SURFACE_MAX
    return f"{shown} · +{more} more" if more > 0 else shown


def concern_raised(concern_id: str, new_paths: Sequence[str]) -> Notice:
    """Trigger 2: `concern K-9 raised · db/pool.py` — only for anchor paths no live concern
    already surfaces, so ten concerns on one file produce one line."""
    return Notice(NoticeTrigger.CONCERN, f"concern {concern_id} raised · {_paths(new_paths)}")


def view_stale(handle: Handle, spans: Sequence[Span]) -> Notice:
    """Trigger 3: `view h_7f2a (applier.py 119-224) is stale`."""
    where = ", ".join(str(s) for s in spans)
    label = f"{handle.label} {where}" if where else handle.label
    return Notice(NoticeTrigger.STALE_VIEW, f"view {handle.id} ({label}) is stale", handle.id)


def run_ended(run_id: str, commits: int) -> Notice:
    """Trigger 4: `run r7 ended · worktree has 4 commits, not applied`."""
    noun = "commit" if commits == 1 else "commits"
    return Notice(
        NoticeTrigger.RUN_ENDED,
        f"run {run_id} ended · worktree has {commits} {noun}, not applied",
    )


def team_replied(tell: str, outcome: str, text: str) -> Notice:
    """Trigger 10: `tell-2 · needs_design: new behaviour and a dependency`.
    A reply that needs Zeus (`needs_design`, `declined`) names its tell in `handle`, and wakes
    him (`zeus/jobs`); one that does not (`applied`, `planned`) names none, and is read at his
    next turn."""
    needs_him = outcome in ("needs_design", "declined")
    return Notice(
        NoticeTrigger.TEAM_REPLIED,
        f"{tell} · {outcome}: {text}",
        handle=tell if needs_him else None,
    )


def run_reported(run_id: str, report: str, carries: Handle | None) -> Notice:
    """Trigger 4: a run handed over in the background ended, and this is its
    report as Zeus reads it (`render_report`), with the handle it names. `handle` names the run,
    so the fold knows it was told."""
    return Notice(NoticeTrigger.RUN_ENDED, report, handle=run_id, carries=carries)


def paused_run_ended(run_id: str, outcome: str) -> Notice:
    """Trigger 4: `run r7 ended COMPLETED: 1 of 1 task(s) committed · resumed after a
    pause`. Its `coder` call was answered `PAUSED`, so he hears the end here; `handle` names the
    run, so the fold knows it was told."""
    return Notice(NoticeTrigger.RUN_ENDED, f"{outcome} · resumed after a pause", handle=run_id)


def view_moved(branch: str | None, source_head: str) -> Notice:
    """Trigger 7: `view moved to the session's worktree <branch>, created from
    <source_head>; the checkout is unchanged until /handoff`."""
    named = f" {branch}" if branch else ""
    return Notice(
        NoticeTrigger.VIEW_MOVED,
        f"view moved to the session's worktree{named}, created from {source_head[:8]}; "
        "the checkout is unchanged until /handoff",
    )


def moved_by_flag(said: str) -> Notice:
    """Trigger 7: the person's `--in-place` moved the session
    back to the checkout as their turn opened; `said` is what `worktree leave` says."""
    return Notice(NoticeTrigger.VIEW_MOVED, said)


def job_exited(
    *,
    job_id: str,
    kind: str,
    line: str,
    how: str,
    moved: bool,
    wrote: Sequence[str],
    overwrote: Sequence[str],
    alongside: Sequence[str],
    tail: str,
    handle: str | None,
    dropped: int = 0,
    git_restored: Sequence[str] = (),
) -> Notice:
    """Trigger 8: `job j2 · task · pytest -q · exited 1 after 4m12s`, then
    what the harness knows about what it touched, the end of its output, and the handle that
    holds all of it. `how` is the exit as `zeus/jobs.py` words it."""
    lines = [f"job {job_id} · {kind} · {_one_line(line)} · {how}"]
    if moved:
        lines.append(
            "the workspace changed while it ran; its result may not cover the latest edits"
        )
    if wrote:
        # Beside another job in the same workspace, a change no step recorded is either's.
        who = job_id if not alongside else f"{job_id} or {', '.join(alongside)}, run beside it,"
        lines.append(
            f"{who} wrote {len(wrote)} file(s) no step recorded; they are not recorded and "
            f"cannot be undone: {', '.join(wrote)}"
        )
    if overwrote:
        lines.append(f"{job_id} wrote {', '.join(overwrote)} after a step had; check them")
    if git_restored:
        # What git runs outside any sandbox is never a command's to change.
        lines.append(
            f"{job_id} changed git's own settings ({', '.join(git_restored)}); they were put "
            "back, since git runs them outside any sandbox"
        )
    if tail:
        lines.append(tail.rstrip("\n"))
    if handle is not None:
        lines.append(
            f"(output kept: handle {handle}; {dropped} bytes from its middle not kept)"
            if dropped
            else f"(whole output: handle {handle})"
        )
    return Notice(NoticeTrigger.JOB_EXITED, "\n".join(lines), handle)


def hook_noted(text: str) -> Notice:
    """Trigger 9: a hook's words, already labelled with its name (`hooks.verdicts.label`)."""
    return Notice(NoticeTrigger.HOOK, text)


@dataclass(frozen=True)
class ForkEffect:
    """An effect a fork cannot undo: something that already happened in the world."""

    seq: int
    """The session seq that caused it — the `shell` call, or the `coder` call whose run did."""
    kind: str
    detail: str


def fork_effects(
    parent_id: str,
    at_seq: int,
    reason: str,
    effects: Sequence[ForkEffect],
    source_head: str | None,
) -> Notice:
    """Trigger 6: the lineage line, then every effect at or before the fork point.
    A fork branches the conversation and the workspace, never the effects, and a child
    that is not told will plan against a world state that does not exist."""
    count = len(effects)
    noun, verb = ("effect", "is") if count == 1 else ("effects", "are")
    rows = "\n".join(f"   seq {e.seq}  {e.kind:<10}  {_one_line(e.detail)}" for e in effects)
    fresh = f"fresh from {source_head[:6]}" if source_head else "fresh"
    text = (
        f'[forked from {parent_id} at seq {at_seq} · "{_one_line(reason)}"]\n'
        f"[{count} {noun} at or before seq {at_seq} {verb} NOT undone by this fork:\n"
        f"{rows}\n"
        f" the workspace is {fresh}; the world is not]"
    )
    return Notice(NoticeTrigger.FORK_EFFECTS, text)


def code_kept(paths: Sequence[str], runs: Sequence[str]) -> Notice:
    """Trigger 6, widened: a rewind took the conversation back and kept the
    code, so the model is told what changed in turns it can no longer read."""
    shown = ", ".join(paths)
    if runs:
        noun = "run" if len(runs) == 1 else "runs"
        shown += f" ({noun} {', '.join(runs)})"
    return Notice(
        NoticeTrigger.FORK_EFFECTS, f"code changed in turns you no longer see: {shown}"
    )


def journaled_effects(
    state: SessionState,
    records: RunRecords | None,
    *,
    hi: int,
    lo: int = 1,
    accounted: Set[int] = frozenset(),
) -> tuple[ForkEffect, ...]:
    """Every effect journaled with `lo <= seq <= hi` that a rewind cannot take back: the one
    reader for a fork (up to its point) and an undo (from its turn on), D162 decision 12.

    Two sources. A `shell` call that opened the network (`network: true`, or a host list before
    D199) or declared `effects` outside the writable set, and ran (a refused call did nothing).
    And each run's own journal: every `effect_completed` that succeeded, or may have
    (`ambiguous`), whose kind is not confined to the worktree. A run whose journal cannot be
    read is listed as such — a gap in this list must be visible, since silence here is a lie by
    omission.

    `accounted` names the user turns whose code the rewind restores, or names in the kept-code
    notice. A worktree run never lists its worktree kinds: its code lives on
    a branch nobody inherits. An in-place run wrote the person's checkout, so it lists
    them only in a turn outside `accounted`: one with no snapshot to restore from."""
    found: list[ForkEffect] = []
    for call in state.calls_upto(hi):
        if call.seq < lo or call.tool != "shell" or not state.ran(call.call_id):
            continue
        command = str(call.args.get("command", ""))
        if network_opened(call.args.get("network")) or _opened(state, call.call_id):
            found.append(ForkEffect(call.seq, "network", command))
        if _declared(call.args.get("effects")):
            found.append(ForkEffect(call.seq, "writes", command))
    users = [t.seq for t in state.turns() if t.role == "user"]
    for episode in state.episodes_upto(hi):
        # A host's run has no turn: it is placed where its line is.
        at = episode.opened_at_seq if episode.opened_at_seq is not None else episode.seq
        if at < lo:
            continue
        turn = max((u for u in users if u <= at), default=None)
        found.extend(_run_effects(episode.run_id, at, records, turn in accounted))
    return tuple(sorted(found, key=lambda e: e.seq))


_OPENED: Final = re.compile(r"· sandbox (?!none\b)\S+ · network on ·")


def _opened(state: SessionState, call_id: str) -> bool:
    """Whether a shell call's result says it ran with the network opened for it: what an
    approved push, fetch or install does with no `network` argument. Read from
    the result's header, the fact the shell recorded, not re-derived from the command. At
    `sandbox none` every command had the network, which the run header says once; listing each
    here would bury the ones that used it."""
    line = state.result(call_id)
    return line is not None and _OPENED.search(line.text) is not None


def _declared(value: object) -> bool:
    return isinstance(value, list) and len(cast("list[object]", value)) > 0


def _run_effects(
    run_id: str, seq: int, records: RunRecords | None, accounted: bool
) -> list[ForkEffect]:
    from chickadee.journal.errors import JournalError

    if records is None:
        return [ForkEffect(seq, "unknown", f"run {run_id}: its journal was not consulted")]
    try:
        run = records.projection(run_id)
    except (JournalError, OSError) as exc:
        return [ForkEffect(seq, "unknown", f"run {run_id}: journal unreadable ({exc})")]
    out: list[ForkEffect] = []
    code_answered = accounted or run.isolation == "worktree"
    # In the order each completed: the operation's kind is its intent's.
    for line in run.lines(EffectCompletedP):
        done = line.payload
        op = run.operations.get(done.operation_id)
        kind = (op.kind if op is not None else None) or "unknown"
        if done.outcome == "failed" or (code_answered and _worktree_only(kind)) or sealed(kind):
            # A sealed run (the worker's shell) changed nothing a fork could inherit.
            continue
        note = " (ambiguous: may have happened)" if done.outcome == "ambiguous" else ""
        out.append(ForkEffect(seq, kind, f"run {run_id} · {done.operation_id}{note}"))
    return out


# -- the snapshot -------------------------------------------------------------


def standing_snapshot(state: SessionState, seq: int) -> Notice:
    """Called once, immediately after a compaction commits. Recomputed from the ledger and
    the log — never copied from the previous snapshot, nor from anything being removed.

    Only rows with something in them render. The constraints row never says "no
    constraints": absence of a row is not evidence of absence, because capture coverage is
    not total (invariant 20)."""
    lines = [f"[standing · as of seq {seq} · later notices amend this]"]
    live = state.ledger.live_constraints()
    for i, c in enumerate(live):
        head = "  constraints   " if i == 0 else "                "
        lines.append(f"{head}{constraint_row(c)}")
    anchors = state.concerns.anchors()
    if anchors:
        lines.append(f"  concerns      {_paths(anchors)}  →  concerns(path=…)")
    return Notice(NoticeTrigger.SNAPSHOT, "\n".join(lines))


def constraint_row(c: Constraint) -> str:
    words = " (your words)" if c.author == "user" else ""
    return f"{c.id}  {_one_line(c.text)}{words} · from {turns_phrase(c.sources)}"
