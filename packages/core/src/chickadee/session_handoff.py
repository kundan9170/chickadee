"""The session's handoff: what the session leaves
behind, and the person's one choice of `apply`, `squash`, `keep`, `discard` or `bring` over its
worktree.

**The git is `handoff.py`'s; this module is the session's half.** `handoff.describe` and
`handoff.finalize` act on one `IsolationSpec` and a base, and never push. What a session
adds is which worktree, from which base, and a record of the choice that survives a crash:

- **Which worktree.** The session's latest `workspace` line, when it names a
  worktree the session still writes (`still_written`): none handed off yet, or `keep`. Its
  `source_head` is the base, so `apply` refuses when the person's branch moved underneath.
- **Write-ahead on the session log.** The session log has no effect kinds, so a handoff
  is two line kinds, typed here beside their writers as `wiring.py` types its own:
  `handoff_intended`, written and fsynced before any git change, and `handoff_completed`. A
  new `workspace` line then records the choice, with `worktree_path` cleared after `discard`,
  and the next `coder` call reads it: after `apply`, `squash` or `discard` it opens a new
  worktree, after `keep` it reattaches this one.
- **Settled by observation, never repeated on a guess.** An intent with no completion is
  a crash between the two. `settle` asks `handoff.reconcile` what the repository shows and
  writes the completion it observed; a handoff that did not land is reported, not retried, and
  the person chooses again.

**One refusal rule.** `HandoffPreview.refusal` is the only place a choice is refused: the
summary carries its answers (`HandoffSummary.refused`) and `finalize` asks it again under the
lease, so a host is never offered a choice that then fails.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import ClassVar, Final, Literal

from pydantic import TypeAdapter, with_config

from chickadee.box.desk import known_good
from chickadee.budget.spend import Spend
from chickadee.context.lines import EpisodeLine
from chickadee.context.session import Session, SessionState
from chickadee.context.store import SessionEvent
from chickadee.context.view import (
    held_worktree,
    last_worktree,
    latest_workspace,
    still_written,
)
from chickadee.context.vocabulary import LINE, SessionKind
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.errors import ConfigError, HandoffError, InfraError, RepoNotIndexed
from chickadee.executor.reconcile import Landed
from chickadee.graph.store import graph_failures_are_infra
from chickadee.handoff import (
    HandoffPreview,
    HandoffResult,
    describe,
    diff_of,
    finalize,
    landed_paths,
    reconcile,
    record_works,
    squash_message,
)
from chickadee.handoff_summary import HandoffSummary
from chickadee.ids import handoff_operation
from chickadee.init.detect import resolve
from chickadee.init.lease import take_write_lease
from chickadee.journal.projection import RunProjection
from chickadee.journal.runs import RunProjections
from chickadee.process import ProcessSupervisor
from chickadee.schema.isolation import GitHandoffChoice, IsolationSpec
from chickadee.schema.journal import Outcome as RunOutcome
from chickadee.work import open_work, record_work
from chickadee.zeus.writes import not_gated, shell_effects

_log = logging.getLogger(__name__)

__all__ = [
    "HANDOFF_COMPLETED",
    "HANDOFF_INTENDED",
    "NOT_VERIFIED",
    "HandoffCompleted",
    "HandoffIntended",
    "dangling",
    "finalize_session",
    "handoff_choice",
    "paused_run",
    "refusal",
    "settle",
    "summarize",
    "unfinished_runs",
]

HANDOFF_INTENDED: Final[SessionKind] = "handoff_intended"
HANDOFF_COMPLETED: Final[SessionKind] = "handoff_completed"


NOT_VERIFIED: Final = (
    "behaviour not verified: no checkpoint result reaches the handoff yet, so nothing here "
    "says the tests passed"
)
"""Invariant 7: until the handoff carries checkpoint results, it says so."""

Choice = Literal["apply", "squash", "keep", "discard", "bring"]
Outcome = Literal["done", "not_landed", "unknown"]


def handoff_choice(choice: GitHandoffChoice | str) -> GitHandoffChoice:
    """The person's word as a choice, or `ConfigError` naming the five."""
    try:
        return GitHandoffChoice(choice)
    except ValueError:
        known = ", ".join(c.value for c in GitHandoffChoice)
        raise ConfigError(f"{choice!r} is not a handoff choice: {known}") from None


@with_config(LINE)
@dataclass(frozen=True)
class HandoffIntended:
    """A handoff about to change the repository: written and fsynced before any git call."""

    KIND: ClassVar[SessionKind] = HANDOFF_INTENDED
    operation_id: str
    choice: Choice
    base: str
    """The worktree's `source_head`, which the choice is checked against."""
    tip: str | None
    """The worktree branch's commit when the choice was made: what `apply` lands."""


@with_config(LINE)
@dataclass(frozen=True)
class HandoffCompleted:
    """How the intent with the same `operation_id` ended: `done`, or, when the git work
    failed or a crash cut it short, `done`, `not_landed` or `unknown` as the repository showed
    it afterwards (`reconciled`). A failure is never taken at its word, since one git call of
    several may already have run."""

    KIND: ClassVar[SessionKind] = HANDOFF_COMPLETED
    operation_id: str
    outcome: Outcome
    detail: str
    reconciled: bool


_INTENDED: Final = TypeAdapter(HandoffIntended)
_COMPLETED: Final = TypeAdapter(HandoffCompleted)

Payload = Callable[[SessionEvent], bytes]


# -- reading the session --------------------------------------------------------------------


def _spec(ref: WorkspaceRef) -> IsolationSpec:
    return IsolationSpec(
        mode="worktree",
        lease_id=ref.lease_id,
        branch=ref.branch,
        worktree_path=ref.worktree_path,
    )


def _intents(
    state: SessionState, payload: Payload
) -> tuple[list[tuple[int, HandoffIntended]], set[str]]:
    """Every intent with its seq, in order, and the operation ids that have a completion."""
    intents: list[tuple[int, HandoffIntended]] = []
    completed: set[str] = set()
    for event, _record in state.lines():
        if event.kind == HANDOFF_INTENDED:
            intents.append((event.seq, _INTENDED.validate_json(payload(event))))
        elif event.kind == HANDOFF_COMPLETED:
            completed.add(_COMPLETED.validate_json(payload(event)).operation_id)
    return intents, completed


def dangling(state: SessionState, payload: Payload) -> tuple[int, HandoffIntended] | None:
    """The session's handoff with an intent and no completion, with the intent's seq: a crash
    between the two. At most one, since `finalize_session` settles it before another."""
    intents, completed = _intents(state, payload)
    open_ = [(seq, i) for seq, i in intents if i.operation_id not in completed]
    return open_[-1] if open_ else None


def _payload(session: Session) -> Payload:
    return lambda event: session.blob(event.payload_sha)


# -- the summary ----------------------------------------------------------------------------


def _last_outcome(runs: RunProjections, episodes: Sequence[EpisodeLine]) -> RunOutcome | None:
    """The last episode's outcome; None while it is live, or with no episode."""
    if not episodes:
        return None
    try:
        end = runs.readable(episodes[-1].run_id).end
    except FileNotFoundError:
        return None
    return end.outcome if end is not None else None


def unfinished_runs(
    state: SessionState, read: Callable[[str], RunProjection]
) -> tuple[str, ...]:
    """The session's runs that started and have not ended, oldest first: each one a Ctrl-C
    paused at a task boundary or a crash cut off, which `resume` continues in its worktree. A
    run that ended (an `ABORTED` hard stop is an end), or never wrote a journal or its
    `run_start`, is not one. `read` is the journal's fold, strict or readable as the caller
    needs (`RunProjections.projection` or `.readable`)."""
    found: list[str] = []
    for episode in state.episodes():
        try:
            run = read(episode.run_id)
        except FileNotFoundError:
            continue
        if run.start is not None and run.end is None:
            found.append(episode.run_id)
    return tuple(found)


def paused_run(state: SessionState, runs: RunProjections) -> str | None:
    """The run `resume` would continue, or None: the latest of `unfinished_runs`."""
    found = unfinished_runs(state, runs.readable)
    return found[-1] if found else None


def refusal(
    choice: GitHandoffChoice, *, paused: str | None, git: HandoffPreview | None
) -> str | None:
    """Why the handoff `choice` is refused, in one sentence, or None: the one answer the review
    offers by and `finalize_session` refuses by, so the review never offers a choice that would
    fail. A paused run is still writing its worktree (`resume` continues it there), so nothing
    lands, and nothing is discarded, from under it: only `keep`, which leaves the worktree
    pending for it. Then the worktree's own state
    (`HandoffPreview.refusal`)."""
    if paused is not None and choice is not GitHandoffChoice.KEEP:
        return (
            f"run {paused} is paused, not finished: /resume to let it finish, or stop it for "
            "good (Ctrl-C twice while it runs), then hand off; keep leaves the worktree "
            "pending for it"
        )
    return git.refusal(choice) if git is not None else None


def summarize(
    session_id: str,
    state: SessionState,
    payload: Payload,
    repo: Path,
    *,
    spend: Spend,
    header: Iterable[str],
    runs: RunProjections,
    blob: Callable[[str, str], bytes] | None = None,
    supervisor: ProcessSupervisor | None = None,
) -> HandoffSummary:
    """`engine.handoff()`'s answer: the worktree's git state and diff, the choices it accepts,
    and every claim the session could not make. Reads only; settles nothing."""
    ref = held_worktree(state)
    episodes = state.episodes()
    # A declared shell write committed in the worktree is in the diff, and
    # nothing gated it: a claim the session could not make.
    shell = tuple(
        not_gated(effect)
        for effect in shell_effects(state, payload)
        if effect.post_state is not None and (effect.written or not effect.is_job)
    )
    unverified = (NOT_VERIFIED, *header, *shell)
    notes: list[str] = []
    open_ = dangling(state, payload)
    if open_ is not None:
        _seq, intent = open_
        notes.append(
            f"the {intent.choice} handoff {intent.operation_id} was interrupted and is not "
            "settled yet; the next /handoff or resume reads what landed"
        )
    git: HandoffPreview | None = None
    diff = ""
    if ref is not None:
        sup = supervisor or ProcessSupervisor()
        try:
            git = describe(repo, _spec(ref), base=ref.source_head, supervisor=sup)
        except HandoffError as exc:
            # The branch is gone (deleted by hand, or a discard cut short): the summary still
            # says what the session spent and did, and why there is nothing to choose over.
            notes.append(f"the session's worktree cannot be described: {exc}")
        else:
            # From the seed, when there is one: the person's own changes are not the run's.
            diff = diff_of(repo, git.start, git.tip, supervisor=sup)
            if git.seed is not None:
                seeded = landed_paths(repo, git.base, git.seed, supervisor=sup)
                notes.append(
                    "the worktree started from your uncommitted changes, which stay yours and "
                    "are not in the diff: " + ", ".join(seeded)
                )
            if git.uncommitted:
                notes.append(
                    "the worktree holds work no commit does, which no choice lands: "
                    + ", ".join(git.uncommitted_paths)
                )
    paused = paused_run(state, runs)
    refused = {
        c.value: why
        for c in GitHandoffChoice
        if git is not None and (why := refusal(c, paused=paused, git=git)) is not None
    }
    return HandoffSummary(
        session_id=session_id,
        git=git,
        spend=spend,
        episodes=tuple(e.run_id for e in episodes),
        last_outcome=_last_outcome(runs, episodes),
        unverified=unverified,
        workspace=ref,
        refused=refused,
        offers={
            c.value: git.says(c) for c in GitHandoffChoice if git and c.value not in refused
        },
        diff=diff,
        message=_offered(runs, episodes, blob) if ref is None else "",
        notes=tuple(notes),
    )


def _offered(
    runs: RunProjections,
    episodes: Sequence[EpisodeLine],
    blob: Callable[[str, str], bytes] | None,
) -> str:
    """The message offered in place: every task the session's runs accepted
    in place, oldest run first; empty when there is none, or no reader of the runs' blobs."""
    if blob is None:
        return ""
    logged: list[tuple[str, str]] = []
    for episode in episodes:
        try:
            run = runs.projection(episode.run_id)
        except (OSError, ValueError):
            continue
        logged += known_good(run, partial(blob, episode.run_id))
    return squash_message("your checkout", logged, None) if logged else ""


# -- the writers ----------------------------------------------------------------------------


def _append(session: Session, line: HandoffIntended | HandoffCompleted | WorkspaceRef) -> None:
    with session.lock:
        session.append(line)


def _record_after(session: Session, ref: WorkspaceRef, choice: GitHandoffChoice) -> None:
    """The `workspace` line the choice leaves (`_after`), written where the session is left
    where it was: a session that left its worktree for the checkout and
    keeps it is still in the checkout, so the checkout's line is written again after it. A
    landing or a discard ends the worktree, and the session is in the checkout either way."""
    with session.lock:
        here = latest_workspace(session.state)
    _append(session, _after(ref, choice))
    if choice is GitHandoffChoice.KEEP and here is not None and here.mode == "in_place":
        _append(session, here)


def _after(ref: WorkspaceRef, choice: GitHandoffChoice) -> WorkspaceRef:
    """The `workspace` line a landed choice leaves: the same worktree, its handoff named, and
    no path once `discard` removed it."""
    path = None if choice is GitHandoffChoice.DISCARD else ref.worktree_path
    return replace(ref, handoff=choice.value, worktree_path=path)


def settle(
    session: Session, repo: Path, *, supervisor: ProcessSupervisor | None = None
) -> HandoffCompleted | None:
    """Close a handoff a crash left between its intent and its completion, by what the
    repository shows (`handoff.reconcile`): landed, the completion and the `workspace` line it
    would have written; not landed or unknown, a completion saying so and the worktree still
    pending, for the person to choose again. Never repeats the handoff. None when there
    is nothing to settle."""
    with session.lock:
        found = dangling(session.state, _payload(session))
        # The worktree the session held when the intent was written, and the one it holds now:
        # a line naming the checkout after either (`worktree leave`) moves neither.
        ref = last_worktree(session.state, upto=found[0]) if found is not None else None
        latest = last_worktree(session.state)
    if found is None:
        return None
    _seq, intent = found
    choice = GitHandoffChoice(intent.choice)
    written = still_written(ref)
    if written is None:
        done = HandoffCompleted(
            intent.operation_id, "unknown", "the session names no worktree it wrote", True
        )
        _append(session, done)
        return done
    seen = reconcile(
        repo,
        _spec(written),
        choice,
        operation_id=intent.operation_id,
        tip=intent.tip,
        base=intent.base,
        supervisor=supervisor,
    )
    outcome: Outcome = "unknown"
    if seen.landed is Landed.YES:
        outcome = "done"
    elif seen.landed is Landed.NO:
        outcome = "not_landed"
    done = HandoffCompleted(intent.operation_id, outcome, seen.evidence, True)
    _append(session, done)
    # Only over the line the intent acted on: a later call may have recorded a newer worktree
    # since, and naming the old one's handoff after it would orphan the newer.
    if outcome == "done" and latest == written:
        _record_after(session, written, choice)
    return done


_CLOSED: Final[Mapping[GitHandoffChoice, Literal["landed", "kept", "dropped"]]] = {
    GitHandoffChoice.APPLY: "landed",
    GitHandoffChoice.SQUASH: "landed",
    GitHandoffChoice.KEEP: "kept",
    GitHandoffChoice.DISCARD: "dropped",
    GitHandoffChoice.BRING: "landed",
}


def _close_work(
    session: Session,
    repo: Path,
    choice: GitHandoffChoice,
    work_commit: bool,
    sup: ProcessSupervisor,
) -> None:
    with session.lock:
        work = open_work(session.state)
    if work is None:
        return
    state = _CLOSED[choice]
    # Closed first: the handoff is done, and the work with it, whatever the index does.
    record_work(session, replace(work, state=state))
    if state == "landed" and work_commit and choice is not GitHandoffChoice.BRING:
        # The row is bookkeeping over what already landed; the next landing's `record_works`
        # writes every row the index lacks, so one missed here, or by a crash, is written
        # then. Raising would report a done handoff as failed. Not after `bring`, which
        # commits nothing: the person's own commit lands the work, and the
        # next landing records it.
        with suppress(HandoffError):
            record_works(repo, work.dir, supervisor=sup)


def _reindex_landed(
    repo: Path,
    preview: HandoffPreview,
    result: HandoffResult,
    reindex: Callable[[Sequence[Path]], object] | None,
    sup: ProcessSupervisor,
) -> None:
    """The checkout's index refreshed for what `result` landed: the commits a choice put on the
    branch, or the files `bring` wrote; nothing for a choice that landed neither. A checkout
    never indexed has nothing to refresh."""
    if reindex is None:
        return
    if result.choice is GitHandoffChoice.BRING:
        paths = landed_paths(repo, preview.start, preview.tip, supervisor=sup)
    elif result.commits_applied:
        paths = landed_paths(repo, preview.source_head, result.source_head, supervisor=sup)
    else:
        return
    if not paths:
        return
    try:
        with graph_failures_are_infra("the checkout's reindex after a handoff"):
            reindex([repo / p for p in paths])
    except RepoNotIndexed:
        return
    except InfraError:
        _log.warning("the checkout's index was not refreshed after a handoff", exc_info=True)


def finalize_session(
    session: Session,
    repo: Path,
    choice: GitHandoffChoice,
    *,
    runs: RunProjections,
    work_commit: bool = True,
    supervisor: ProcessSupervisor | None = None,
    reindex: Callable[[Sequence[Path]], object] | None = None,
) -> HandoffResult:
    """Carry out the person's `choice` over the session's worktree, under the repository WRITE
    lease: settle an interrupted handoff first, check the choice (`refusal`: a paused run's,
    read from `runs`, or the worktree's own; nothing written when refused), write the intent,
    run `handoff.finalize`, and write its completion and the new `workspace` line. Never
    pushes.

    The session's open work is closed by the choice, and one that lands is
    appended to the index on the person's branch (B1), unless `work_commit` is off (B7). A
    row missed by a crash is written by the next landing, which appends every row the index
    lacks (`handoff.record_works`); a session's start writes none.

    `reindex` refreshes the checkout's code index for the paths a landing choice put on the
    branch, in this hold of the lease (`init.orchestrate.reindex_held`): Zeus's view moves back
    to the checkout, and its graph tools read that index. A failure is logged, never raised: the
    handoff landed, and a read still checks each file against the index's sha.

    `HandoffError` when there is no worktree to hand off or the choice is refused, with nothing
    written. A git failure after the intent is settled at once by observation (`settle`) and
    raised: one call of several may already have run. Any other failure, a crash among them,
    leaves the intent open for `settle`."""
    sup = supervisor or ProcessSupervisor()
    lease = take_write_lease(resolve(repo))
    try:
        settle(session, repo, supervisor=sup)
        with session.lock:
            ref = held_worktree(session.state)
            n = len(_intents(session.state, _payload(session))[0]) + 1
            paused = paused_run(session.state, runs)
        if ref is None:
            raise HandoffError(
                "the session has no worktree to hand off: no coder call has written one, or "
                "its last handoff already applied, squashed or discarded it"
            )
        spec, base = _spec(ref), ref.source_head
        preview = describe(repo, spec, base=base, supervisor=sup)
        refused = refusal(choice, paused=paused, git=preview)
        if refused is not None:
            raise HandoffError(f"handoff refused, nothing was changed: {refused}")
        operation_id = handoff_operation(session.session_id, n)
        _append(session, HandoffIntended(operation_id, choice.value, base, preview.tip))
        try:
            result = finalize(
                repo, spec, choice, operation_id=operation_id, base=base, supervisor=sup
            )
        except HandoffError:
            settle(session, repo, supervisor=sup)
            raise
        _append(session, HandoffCompleted(operation_id, "done", result.detail, False))
        _record_after(session, ref, choice)
        _close_work(session, repo, choice, work_commit, sup)
        _reindex_landed(repo, preview, result, reindex, sup)
        return result
    finally:
        lease.release()
