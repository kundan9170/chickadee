"""Undo by turn, redo, and task and operation undo: the facade's one home for each.

An undo picks a turn N and returns to just before it. The person chooses the code, the
conversation, or both (decision 1). The code is put back per workspace, by where it now
lives (decision 4):

- in the run's own worktree, before the handoff: the worktree branch moves back without
  touching a file, then the files the harness wrote are restored;
- in the person's checkout, after `apply` or `squash`: the files are restored, and committed
  only under `undo.on_your_branch = "commit"` or when the person asks, never on a conflict;
- in place: the files are restored, and never committed.

Every part has one home elsewhere, and this module calls it rather than doing it again:

| Part | Home |
|---|---|
| the plan | `workspace.plan_undo` |
| the restore | `workspace.restore_tree` |
| the redo snapshot | `workspace.snapshot_for_redo` |
| whether a handoff landed | `handoff.landed` |
| the safe point | `rewind.safe_point_refusal` |
| the line and its kept-code notice | `rewind.record_rewind` |
| what could not be undone | `standing.journaled_effects` |
| the conflict question | `InteractionPort.ask`, `QuestionKind.UNDO_CONFLICT` |

**Order.** The redo snapshot is parked before anything is written, and the line is appended
last. A crash between them leaves restored files and a findable `undo-<seq>` ref, never a line
claiming a restore that did not happen. The session's lock is held throughout, so the seq the
redo ref names is the line's. The repository WRITE lease is held across every write, so no run
writes underneath.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from chickadee.box.commit import commit_message
from chickadee.box.steps import beside_jobs
from chickadee.context.lines import RedoLine, RewoundCode, UndoLine
from chickadee.context.rewind import (
    CodeSnapshotLine,
    ForkRefused,
    moved_on,
    record_rewind,
    safe_point_refusal,
    taken,
)
from chickadee.context.session import Session, SessionState, Turn
from chickadee.context.standing import ForkEffect, journaled_effects
from chickadee.context.view import run_worktree
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.errors import SessionError
from chickadee.executor.reconcile import operations_after, trailer_commit
from chickadee.handoff import landed
from chickadee.ids import new_session_id, split_task_key, task_of_commit
from chickadee.init.detect import resolve
from chickadee.init.lease import take_write_lease
from chickadee.journal.effects import EffectKind
from chickadee.schema.interaction import (
    AnswerSource,
    InteractionPort,
    Option,
    Question,
    QuestionKind,
)
from chickadee.workspace import (
    TurnTrees,
    acquire,
    changed_paths,
    commit_of,
    commit_paths,
    current_tree,
    dirty_paths,
    harness_paths,
    head_and_branch,
    is_ancestor,
    move_branch,
    plan_undo,
    reattach,
    ref_exists,
    restore_operation,
    restore_tree,
    snapshot_for_redo,
    tree_of,
    turn_refs,
)
from chickadee.zeus.writes import ShellEffect, shell_effects

if TYPE_CHECKING:
    from chickadee.config import UndoPolicy
    from chickadee.context.project import RunRecords
    from chickadee.journal.projection import Operation, RunProjection
    from chickadee.workspace import UndoPlan

__all__ = [
    "CodeOutcome",
    "ForkChoice",
    "Forked",
    "Restored",
    "UndoOption",
    "UndoReport",
    "Undoable",
    "ask_fork_code",
    "fork",
    "redo",
    "undo",
    "undo_operation",
    "undo_task",
    "undoable",
]

UndoOption = Literal["both", "conversation", "code"]
Where = Literal["worktree", "checkout", "in_place"]

_DEFAULTED = frozenset({AnswerSource.DEFAULT, AnswerSource.TIMEOUT_DEFAULT})


@dataclass(frozen=True)
class Undoable:
    """A turn a person can take back, and how (decision 1)."""

    turn: int
    prompt: str
    """The turn's first line, as the person typed it."""
    options: tuple[UndoOption, ...]
    no_code: str | None = None
    """Why the two code options are missing: no turn from here on changed code, or a snapshot
    it needs is gone. None when they are offered."""


@dataclass(frozen=True)
class CodeOutcome:
    """What an undo or redo did to one workspace's code."""

    key: str
    where: Where
    root: str
    restored: tuple[str, ...] = ()
    kept: tuple[str, ...] = ()
    """Conflicting paths left as the person had them."""
    kept_by_default: tuple[str, ...] = ()
    """The kept paths nobody answered for: an unattended session's default."""
    skipped: tuple[str, ...] = ()
    """Untracked paths a snapshot left out for their size, which no undo can protect."""
    committed: str | None = None
    note: str | None = None
    """Why something the person might expect did not happen: no commit, or no restore."""


@dataclass(frozen=True)
class UndoReport:
    """What an undo or a redo did, and what it could not do (decision 3)."""

    seq: int
    """The `undo` or `redo` line."""
    to_turn: int
    code: bool
    conversation: bool
    workspaces: tuple[CodeOutcome, ...] = ()
    not_undone: tuple[ForkEffect, ...] = ()
    """Every effect from turn N on that reached past the code: a command, a network call."""
    prompt: str | None = None
    """Turn N's words, offered back for editing when its conversation was taken back."""
    withheld: tuple[str, ...] = ()
    """A redo's: each half of the undo it could no longer take back, and why (decision 7)."""


@dataclass(frozen=True)
class Restored:
    """What a task or operation undo restored, in which workspace."""

    root: str
    restored: tuple[str, ...]
    kept: tuple[str, ...] = ()
    kept_by_default: tuple[str, ...] = ()
    beside: tuple[str, ...] = ()
    """The jobs the operation ran beside: what it restored may hold some of
    their writes, which no step recorded."""


@dataclass(frozen=True)
class _Pair:
    before: CodeSnapshotLine
    after: CodeSnapshotLine

    def trees(self) -> TurnTrees:
        skipped = tuple(sorted({*self.before.skipped, *self.after.skipped}))
        return TurnTrees(self.before.turn, self.before.tree, self.after.tree, skipped)


@dataclass
class _Answers:
    kept: list[str] = field(default_factory=list[str])
    by_default: list[str] = field(default_factory=list[str])


# -- the list ---------------------------------------------------------------------------------


def undoable(session: Session, repo: Path) -> tuple[Undoable, ...]:
    """Every visible user turn, oldest first, with the options it offers. The conversation can
    always be taken back; the code only when a turn from there on changed it and every snapshot
    that needs is still there (decision 7)."""
    out: list[Undoable] = []
    with session.lock:
        shots = taken(session)
        alive = turn_refs(repo, session.session_id)
        for turn in _user_turns(session.state):
            groups, why = _pairs(session.state, shots, alive, turn.seq)
            if why is None and groups:
                out.append(
                    Undoable(turn.seq, _first_line(turn), ("both", "conversation", "code"))
                )
            else:
                no_code = why or "no turn from here on changed code"
                out.append(Undoable(turn.seq, _first_line(turn), ("conversation",), no_code))
    return tuple(out)


# -- undo -------------------------------------------------------------------------------------


def undo(
    session: Session,
    records: RunRecords,
    repo: Path,
    policy: UndoPolicy,
    port: InteractionPort,
    *,
    to_turn: int,
    code: bool,
    conversation: bool,
    commit: bool | None = None,
) -> UndoReport:
    """Take back turns `to_turn` onward: the code, the conversation, or both (decisions 1-5)."""
    if not (code or conversation):
        raise SessionError("an undo takes back the code, the conversation, or both")
    with session.lock:
        state = session.state
        turn = state.turn(to_turn)
        if turn is None or turn.role != "user":
            raise SessionError(f"turn {to_turn} is not a user turn this session can see")
        why = safe_point_refusal(state, records)
        if why is not None:
            raise SessionError(f"cannot undo now: {why}")
        alive = turn_refs(repo, session.session_id)
        groups, missing = _pairs(state, taken(session), alive, to_turn)
        if code and missing is not None:
            raise SessionError(f"the code from turn {to_turn} on cannot be undone: {missing}")
        if code and not groups:
            raise SessionError(f"no turn from {to_turn} on changed code")
        # Read before the line, which hides these turns from the fold.
        effects = journaled_effects(
            state, records, lo=to_turn, hi=state.last_seq, accounted=_snapshotted(groups)
        )
        runs = _runs_from(state, to_turn)
        outcomes: list[CodeOutcome] = []
        rewound: list[RewoundCode] = []
        # The seq this undo begins at names its questions and its redo refs: unique per undo,
        # and known before anything is asked. Its own line lands later, after the questions
        # and carries each ref by name, so nothing reads a seq back out of one.
        began = session.next_seq
        if code:
            with _lease(repo):
                # One workspace at a time, planned, asked and restored together: a plan made
                # before another workspace's questions could go stale while they wait.
                for key, pairs in groups.items():
                    prepared = _prepare(
                        repo, policy, port, key, pairs, qid=f"undo-{began}-{key}"
                    )
                    done = _undo_workspace(
                        session.session_id, policy, prepared, seq=began, commit=commit
                    )
                    outcomes.append(done[0])
                    if done[1] is not None:
                        rewound.append(done[1])
        kept_paths = () if code else _harness_paths(repo, groups)
        line = UndoLine(to_turn, code, conversation, tuple(rewound))
        written = record_rewind(session, line, kept_paths, runs)
        session.sync()
    return UndoReport(
        seq=written,
        to_turn=to_turn,
        code=code,
        conversation=conversation,
        workspaces=tuple(outcomes),
        not_undone=effects,
        prompt=turn.text if conversation else None,
    )


@dataclass(frozen=True)
class _Prepared:
    """One workspace's undo, planned and its conflicts answered, applied straight after by
    `_undo_workspace`: nothing is asked of the person between its plan and its restore."""

    key: str
    first: _Pair
    turns: str
    where: Where
    root: Path | None
    plan: UndoPlan | None = None
    answers: _Answers = field(default_factory=_Answers)


def _prepare(
    repo: Path,
    policy: UndoPolicy,
    port: InteractionPort,
    key: str,
    pairs: Sequence[_Pair],
    *,
    qid: str,
) -> _Prepared:
    first, last = pairs[0], pairs[-1]
    where, root = _where(repo, first, last)
    turns = f"{first.before.turn}-{last.before.turn}"
    if root is None:
        return _Prepared(key, first, turns, where, None)
    trees = [p.trees() for p in pairs]
    plan = plan_undo(root, trees, untracked_max_bytes=policy.untracked_max_bytes)
    answers = _ask(port, plan.conflicts, qid, f"before turn {turns}")
    return _Prepared(key, first, turns, where, root, plan, answers)


def _undo_workspace(
    session_id: str,
    policy: UndoPolicy,
    prepared: _Prepared,
    *,
    seq: int,
    commit: bool | None,
) -> tuple[CodeOutcome, RewoundCode | None]:
    key, first, turns, where = prepared.key, prepared.first, prepared.turns, prepared.where
    root, plan, answers = prepared.root, prepared.plan, prepared.answers
    if root is None or plan is None:
        note = f"the worktree at {first.before.root} is gone, so its code cannot be put back"
        return CodeOutcome(key, where, first.before.root, note=note), None
    redo = snapshot_for_redo(
        root,
        session_id=session_id,
        undo_seq=seq,
        key=key,
        untracked_max_bytes=policy.untracked_max_bytes,
    )
    restore = tuple(p for p in plan.restore if p not in answers.kept)
    branch_head = None
    if where == "worktree" and first.before.head is not None:
        branch_head = redo.head
        move_branch(root, first.before.head)
    restored = restore_tree(root, plan.target, restore)
    committed, note = None, None
    if where == "checkout":
        wanted = policy.on_your_branch == "commit" if commit is None else commit
        if wanted and plan.conflicts:
            note = "not committed: a conflict leaves the restore for you to review"
        elif wanted and restored:
            message = f"Undo turns {turns} ({session_id})"
            committed = commit_paths(root, restored, message)
    outcome = CodeOutcome(
        key=key,
        where=where,
        root=str(root),
        restored=restored,
        kept=tuple(answers.kept),
        kept_by_default=tuple(answers.by_default),
        skipped=plan.skipped,
        committed=committed,
        note=note,
    )
    line = RewoundCode(
        key=key,
        mode=first.before.mode,
        root=str(root),
        target=plan.target,
        redo_tree=redo.tree,
        redo_ref=redo.ref,
        branch_head=branch_head,
        restored=restored,
        conflicts_kept=tuple(answers.kept),
        committed=committed,
    )
    return outcome, line


def _where(repo: Path, first: _Pair, last: _Pair) -> tuple[Where, Path | None]:
    """Where a workspace's code lives now (decision 4). Read from git, since nothing records a
    handoff the person made: `handoff.landed` answers it by observation."""
    if first.before.mode == "in_place":
        return "in_place", Path(first.before.root)
    start, tip = first.before.head, last.after.head
    if start is not None and tip is not None and landed(repo, start, tip) is not None:
        return "checkout", repo
    worktree = Path(first.before.root)
    return "worktree", worktree if worktree.is_dir() else None


# -- redo -------------------------------------------------------------------------------------


def redo(
    session: Session,
    records: RunRecords,
    repo: Path,
    policy: UndoPolicy,
    port: InteractionPort,
) -> UndoReport:
    """Take the newest undo back, as much of it as is still offered (decision 7): the code while
    its redo snapshot exists, the conversation until the conversation moves on."""
    with session.lock:
        state = session.state
        found = _newest_undo(state)
        if found is None:
            raise SessionError("there is no undo to take back")
        line, code, conversation = found
        gone: list[str] = []
        later = moved_on(state, line.seq) if conversation else None
        if later is not None:
            conversation = False
            gone.append(f"its conversation moved on at seq {later}")
        if code and not all(ref_exists(repo, w.redo_ref) for w in line.workspaces):
            code = False
            gone.append("its code's redo snapshot was removed by retention")
        if not (code or conversation):
            raise SessionError(
                f"the undo at seq {line.seq} can no longer be taken back: {'; '.join(gone)}"
            )
        why = safe_point_refusal(state, records)
        if why is not None:
            raise SessionError(f"cannot redo now: {why}")
        seq = session.next_seq
        outcomes: list[CodeOutcome] = []
        if code:
            with _lease(repo):
                outcomes = [
                    _redo_workspace(session, policy, port, w, line, seq=seq)
                    for w in line.workspaces
                ]
        written = session.append(RedoLine(line.seq, code, conversation)).seq
        session.sync()
    return UndoReport(
        seq=written,
        to_turn=line.to_turn,
        code=code,
        conversation=conversation,
        workspaces=tuple(outcomes),
        withheld=tuple(gone),
    )


def _redo_workspace(
    session: Session,
    policy: UndoPolicy,
    port: InteractionPort,
    w: RewoundCode,
    line: UndoLine,
    *,
    seq: int,
) -> CodeOutcome:
    """Put back what the undo restored, asking about any of it the person changed since: a
    path whose content now differs from the undo's target (decision 7)."""
    root = Path(w.root)
    now, skipped = current_tree(root, untracked_max_bytes=policy.untracked_max_bytes)
    since = set(changed_paths(root, w.target, now)) | set(skipped)
    conflicts = tuple(p for p in w.restored if p in since)
    answers = _ask(port, conflicts, f"redo-{seq}-{w.key}", "as it was before the undo")
    restore = tuple(p for p in w.restored if p not in answers.kept)
    if w.branch_head is not None:
        move_branch(root, w.branch_head)
    restored = restore_tree(root, w.redo_tree, restore)
    committed, note = None, None
    if w.committed is not None:
        if conflicts:
            note = "not committed: a conflict leaves the restore for you to review"
        elif restored:
            message = f"Redo turns from {line.to_turn} ({session.session_id})"
            committed = commit_paths(root, restored, message)
    return CodeOutcome(
        key=w.key,
        where="in_place" if w.mode == "in_place" else _where_of(w),
        root=w.root,
        restored=restored,
        kept=tuple(answers.kept),
        kept_by_default=tuple(answers.by_default),
        committed=committed,
        note=note,
    )


def _where_of(w: RewoundCode) -> Where:
    return "worktree" if w.branch_head is not None else "checkout"


def _newest_undo(state: SessionState) -> tuple[UndoLine, bool, bool] | None:
    """The newest undo with a half not yet redone, and which halves remain."""
    redone: dict[int, tuple[bool, bool]] = {}
    for _, record in state.lines():
        if isinstance(record, RedoLine):
            code, conversation = redone.get(record.undo_seq, (False, False))
            redone[record.undo_seq] = (code or record.code, conversation or record.conversation)
    for _, record in reversed(list(state.lines())):
        if isinstance(record, UndoLine):
            code_done, conversation_done = redone.get(record.seq, (False, False))
            code = record.code and not code_done
            conversation = record.conversation and not conversation_done
            if code or conversation:
                return record, code, conversation
    return None


# -- fork: restore the code, or keep it -------------------------------------


ForkChoice = Literal["restore", "keep"]


@dataclass(frozen=True)
class Forked:
    """What a fork made: the child, what it was told on arrival, and its code."""

    child_id: str
    code: ForkChoice
    notices: tuple[str, ...]
    """The child's arrival notices (the kept-code notice), as the model reads them."""
    workspaces: tuple[CodeOutcome, ...] = ()
    """In place, what a `restore` put back in the checkout."""
    worktree: str | None = None
    """The child's own worktree, when the parent's code was in worktrees."""


def ask_fork_code(port: InteractionPort, session_id: str, at_seq: int) -> ForkChoice:
    """Restore the code to the fork point, or keep it; `keep` unattended."""
    question = Question(
        id=f"fork-{session_id}-{at_seq}",
        kind=QuestionKind.FORK_CODE,
        text=(
            f"Fork {session_id} at seq {at_seq}: put the code back as it was then, or keep it "
            "as it is now?"
        ),
        options=(
            Option(key="restore", label="restore the code to the fork point"),
            Option(key="keep", label="keep the code as it is"),
        ),
        default="keep",
    )
    return "restore" if port.ask(question).choice == "restore" else "keep"


def fork(
    parent: Session,
    records: RunRecords,
    repo: Path,
    policy: UndoPolicy,
    port: InteractionPort,
    *,
    at_seq: int,
    reason: str,
    code: ForkChoice,
) -> Forked:
    """Branch `parent` at `at_seq` into a new session, its code restored or kept. The safe point
    first, so a refused fork touches nothing; then the code, its
    refs before its files; then the child, told on arrival what it cannot undo and, on `keep`,
    what code it no longer remembers."""
    child_id = new_session_id()
    if not 1 <= at_seq <= parent.state.last_seq:
        # A seq the log does not have is the caller's mistake, said as one: read, it would
        # surface as `SessionCorrupt`, which tells the person their log is damaged.
        raise SessionError(f"session {parent.session_id} has no line {at_seq} to fork at")
    at = Session.state_at(parent.root, parent.session_id, at_seq)
    why = safe_point_refusal(at, records)
    if why is not None:
        raise ForkRefused(parent.session_id, at_seq, why)
    with parent.lock:
        alive = turn_refs(repo, parent.session_id)
        shots = taken(parent)
        groups, missing = _pairs(parent.state, shots, alive, at_seq + 1)
        runs = _runs_from(parent.state, at_seq + 1)
    outcomes: list[CodeOutcome] = []
    in_place = {k: v for k, v in groups.items() if v[0].before.mode == "in_place"}
    if code == "restore" and in_place:
        if missing is not None:
            raise SessionError(f"the code after seq {at_seq} cannot be restored: {missing}")
        with _lease(repo):
            for key, pairs in in_place.items():
                prepared = _prepare(
                    repo, policy, port, key, pairs, qid=f"fork-{child_id}-{key}"
                )
                done, _ = _undo_workspace(child_id, policy, prepared, seq=0, commit=False)
                outcomes.append(done)
    worktree = _child_worktree(repo, shots, alive, at_seq, child_id, reason, code)
    kept = _harness_paths(repo, groups) if code == "keep" else ()
    child = Session.fork(
        parent.root,
        parent.session_id,
        at_seq,
        reason,
        child_id,
        code=code,
        records=records,
        kept_paths=kept,
        runs=runs,
    )
    with child:
        if worktree is not None:
            child.record_workspace(worktree)
        told = tuple(
            r.text
            for e, r in child.state.lines()
            if e.session_id == child_id and isinstance(r, Turn) and r.role == "harness"
        )
    path = worktree.worktree_path if worktree is not None else None
    return Forked(child_id, code, told, tuple(outcomes), path)


def _child_worktree(
    repo: Path,
    shots: Sequence[CodeSnapshotLine],
    alive: frozenset[str],
    at_seq: int,
    child_id: str,
    reason: str,
    code: ForkChoice,
) -> WorkspaceRef | None:
    """When the parent's code was in worktrees, the child's own, from the newest `after`
    snapshot at the fork point (`restore`) or of all (`keep`); from `HEAD` when none is left.
    None when the parent never wrote in a worktree."""
    worktree = [s for s in shots if s.mode == "worktree" and s.phase == "after"]
    if not worktree:
        return None
    usable = [s for s in worktree if s.ref in alive and (code == "keep" or s.turn <= at_seq)]
    start = commit_of(repo, usable[-1].ref) if usable else None
    _, source_branch = head_and_branch(repo)
    with acquire(repo, run_id=f"fork-{child_id}", goal=reason, start=start) as ws:
        return WorkspaceRef(
            mode="worktree",
            lease_id=ws.spec.lease_id,
            source_ref=source_branch or "HEAD",
            source_head=start or commit_of(repo, "HEAD"),
            branch=ws.spec.branch,
            worktree_path=ws.spec.worktree_path,
            handoff="pending",
        )


# -- task and operation undo (decision 12) ----------------------------------------------------


def undo_task(
    session: Session,
    records: RunRecords,
    repo: Path,
    policy: UndoPolicy,
    port: InteractionPort,
    *,
    run_id: str,
    task_id: str,
) -> Restored:
    """Roll a worktree run back to just before one task's commit (`/undo task`): the branch
    moves to its parent, and every path changed from there to the tip is restored, asking about
    any the person has edited in the worktree since. In place a task is no commit, so it is
    refused, pointing at the turn undo."""
    with session.lock:
        why = safe_point_refusal(session.state, records)
        if why is not None:
            raise SessionError(f"cannot undo a task now: {why}")
        # A run of this session, as `undo_operation` asks: a mistyped or foreign run id is the
        # person's to correct, never a missing journal surfacing as an internal error.
        if run_id not in {episode.run_id for episode in session.state.episodes()}:
            raise SessionError(f"no run {run_id} in this session; /status names its runs")
        run = records.projection(run_id)
        if run.isolation != "worktree":
            raise SessionError(
                f"run {run_id} ran in place, where a task is not a commit; take back "
                "the turn instead (`undo`)"
            )
        ops = [op for op in _committed(run) if _names(task_of_commit(op.operation_id), task_id)]
        if not ops:
            raise SessionError(f"run {run_id} committed no task {task_id}")
        worktree = run_worktree(session.state, run_id)
        with reattach(repo, run_id=run_id, worktree=worktree) as workspace:
            root = workspace.root
            since = run.start.repo_head if run.start is not None else None
            until, later = _run_end(session.state, records, root, run_id, worktree)
            commit = trailer_commit(
                root, ops[-1].operation_id, since=since or None, until=until
            )
            if commit is None:
                raise SessionError(f"task {task_id}'s commit is not on run {run_id}'s branch")
            _nothing_above(root, run, run_id, since or None, until, later)
            parent = f"{commit}^"
            target, tip = tree_of(root, parent), tree_of(root, "HEAD")
            paths = changed_paths(root, target, tip)
            now, skipped = current_tree(root, untracked_max_bytes=policy.untracked_max_bytes)
            edited = set(changed_paths(root, tip, now)) | set(skipped)
            conflicts = tuple(p for p in paths if p in edited)
            answers = _ask(port, conflicts, f"task-{run_id}-{task_id}", "before the task")
            move_branch(root, parent)
            restored = restore_tree(
                root, target, tuple(p for p in paths if p not in answers.kept)
            )
    return Restored(str(root), restored, tuple(answers.kept), tuple(answers.by_default))


def _committed(run: RunProjection) -> list[Operation]:
    """The run's task commits that landed, in the order they were intended."""
    return [
        op
        for op in run.operations.values()
        if op.kind == EffectKind.TASK_COMMIT.value
        and op.completed is not None
        and op.completed.outcome == "succeeded"
    ]


def _run_end(
    state: SessionState, records: RunRecords, root: Path, run_id: str, worktree: Path | None
) -> tuple[str | None, tuple[str, ...]]:
    """Where run `run_id`'s commits end on the session's shared branch: the head the next run
    in the same worktree started from, None when no later run wrote there;
    and the later runs in that worktree. Operation ids repeat across runs, so a search past
    this point finds a later run's task under the same id."""
    if worktree is None:
        return None, ()
    episodes = state.episodes()
    at = next(i for i, e in enumerate(episodes) if e.run_id == run_id)
    later = [e.run_id for e in episodes[at + 1 :] if run_worktree(state, e.run_id) == worktree]
    for later_id in later:
        try:
            start = records.projection(later_id).start
        except FileNotFoundError:
            continue
        if start is None or not start.repo_head:
            continue
        if not is_ancestor(root, start.repo_head, "HEAD"):
            return None, ()  # the branch was moved back below it; nothing of it is on top
        return start.repo_head, tuple(later)
    return None, ()


def _nothing_above(
    root: Path,
    run: RunProjection,
    run_id: str,
    since: str | None,
    until: str | None,
    later: tuple[str, ...],
) -> None:
    """A task undo moves the branch back, taking everything above the task with it. The run's
    own later tasks are what the undo means; anything else on top, a later run's commits or a
    declared shell write's, nobody asked for: refused, named, so the person undoes it
    first or takes back the turn."""
    committed = _committed(run)
    last = trailer_commit(root, committed[-1].operation_id, since=since, until=until)
    if last is None or commit_of(root, "HEAD") == last:
        return
    above = operations_after(root, last)
    named = ", ".join(dict.fromkeys([*later, *above]))
    raise SessionError(
        f"later commits sit on top of run {run_id}'s last task in the session's worktree "
        f"({named}); undo them first, or take back the turn instead (`undo`)"
    )


def undo_operation(
    session: Session,
    records: RunRecords,
    repo: Path,
    operation_id: str,
    *,
    run_id: str | None = None,
) -> Restored:
    """Put an operation's declared paths back from its snapshot ref, through
    `restore_operation`, in the workspace of the run that journaled it, or, for a declared
    Zeus write, where the session's log says it wrote.

    Operation ids are unique within a run, not across runs, so `run_id` names which run's.
    Without it, the one run of the session that journaled the id is taken; when several did,
    the undo is refused, naming them, rather than guessing which one the person meant."""
    with session.lock:
        why = safe_point_refusal(session.state, records)
        if why is not None:
            raise SessionError(f"cannot undo an operation now: {why}")
        if run_id is None:
            shell = [
                e
                for e in shell_effects(session.state, lambda e: session.blob(e.payload_sha))
                if e.operation_id == operation_id
            ]
            if shell and shell[-1].is_job:
                # What a job wrote is not recorded, so nothing can put it back.
                raise SessionError(
                    f"operation {operation_id} is a job: what it wrote after its yield is not "
                    "recorded, so it cannot be undone; its step, the call's own operation id, "
                    "can"
                )
            if shell:
                return _undo_shell_write(session, repo, shell[-1])
        runs = [episode.run_id for episode in session.state.episodes()]
        if run_id is not None and run_id not in runs:
            raise SessionError(f"no run {run_id} in this session; /status names its runs")
        found: list[tuple[str, Operation]] = []
        for candidate in runs if run_id is None else [run_id]:
            try:
                op = records.projection(candidate).operations.get(operation_id)
            except FileNotFoundError:
                continue
            if op is not None:
                found.append((candidate, op))
        if not found and run_id is not None:
            raise SessionError(f"run {run_id} journaled no operation {operation_id}")
        if not found:
            raise SessionError(f"no run of this session journaled operation {operation_id}")
        if len(found) > 1:
            named = ", ".join(r for r, _ in found)
            raise SessionError(
                f"operation {operation_id} was journaled by more than one run of this session "
                f"({named}); name the run: /undo {operation_id} <run>"
            )
        ((owner, op),) = found
        if not op.surface:
            raise SessionError(f"operation {operation_id} declared no paths to restore")
        worktree = run_worktree(session.state, owner)
        with reattach(repo, run_id=owner, worktree=worktree) as workspace:
            restored = restore_operation(workspace.root, operation_id, op.surface, run_id=owner)
        return Restored(str(workspace.root), restored)


def _undo_shell_write(session: Session, repo: Path, effect: ShellEffect) -> Restored:
    """A declared Zeus write put back from its snapshot, under the repository lease. One the
    session committed in its worktree has its undo committed too, so the tree stays clean for
    the next `coder` call (the task's design, decision 3)."""
    root = effect.root
    if root is None or not root.is_dir():
        raise SessionError(f"operation {effect.operation_id} wrote where nothing is now")
    if effect.outcome != "succeeded":
        # A failed write was put back when it failed; there is nothing of it left to undo.
        raise SessionError(f"operation {effect.operation_id} failed and was put back then")
    if not effect.written:
        raise SessionError(f"operation {effect.operation_id} changed nothing to restore")
    lease = take_write_lease(resolve(repo))
    try:
        # What it wrote, not what it declared: a glob is no path to restore, and a restore
        # bounded to `**` would reach the person's work beside it.
        restored = restore_operation(
            root, effect.operation_id, effect.written, run_id=session.session_id
        )
        changed = [p for p in dirty_paths(root) if p in restored]
        if effect.post_state is not None and changed:
            undo_op = f"undo:{effect.operation_id}"
            commit_paths(
                root,
                changed,
                commit_message(f"chickadee: undo {effect.operation_id}", undo_op),
                own_unit=True,
            )
    finally:
        lease.release()
    return Restored(str(root), restored, beside=beside_jobs(effect.preconditions))


# -- shared -----------------------------------------------------------------------------------


def _ask(port: InteractionPort, conflicts: Sequence[str], qid: str, back_to: str) -> _Answers:
    """One D180 question per conflicting path. The default keeps the person's version, so an
    unattended session resolves it and the report says it did."""
    answers = _Answers()
    for n, path in enumerate(conflicts, start=1):
        question = Question(
            id=f"{qid}-{n}",
            kind=QuestionKind.UNDO_CONFLICT,
            text=(
                f"{path} was written by the harness and has been edited by you since. Keep "
                f"your version, or restore it as it was {back_to}?"
            ),
            options=(
                Option(key="keep", label="keep your version"),
                Option(key="restore", label="restore it"),
            ),
            default="keep",
            evidence=(path,),
        )
        answer = port.ask(question)
        if answer.choice != "restore":
            answers.kept.append(path)
            if answer.source in _DEFAULTED:
                answers.by_default.append(path)
    return answers


@contextmanager
def _lease(repo: Path) -> Generator[None]:
    """The repository WRITE lease across an undo's writes: `RepoLeaseBusy` while a run
    holds it, which the safe-point check has normally refused already."""
    lease = take_write_lease(resolve(repo))
    try:
        yield
    finally:
        lease.release()


def _user_turns(state: SessionState) -> list[Turn]:
    return [t for t in state.turns() if t.role == "user"]


def _first_line(turn: Turn) -> str:
    return turn.text.splitlines()[0] if turn.text else ""


def _pairs(
    state: SessionState, shots: Sequence[CodeSnapshotLine], alive: frozenset[str], to_turn: int
) -> tuple[dict[str, list[_Pair]], str | None]:
    """Each workspace's snapshot pairs from `to_turn` on, oldest first, less the turns a
    standing code undo already took back; and why the code cannot be undone, if it cannot."""
    undone = _code_undone(state)
    # By turn, then by part: a wake turn's work is its person's turn's next part.
    phases: dict[tuple[str, int, int], dict[str, CodeSnapshotLine]] = {}
    for shot in shots:
        if shot.turn >= to_turn and shot.turn not in undone:
            phases.setdefault((shot.key, shot.turn, shot.part), {})[shot.phase] = shot
    groups: dict[str, list[_Pair]] = {}
    ordered = sorted(phases.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0]))
    for (key, turn, _part), held in ordered:
        before, after = held.get("before"), held.get("after")
        if before is None or after is None:
            return groups, f"turn {turn}'s snapshot of {key} is incomplete"
        for shot in (before, after):
            if shot.ref not in alive:
                return groups, f"turn {turn}'s snapshot was removed by retention ({shot.ref})"
        groups.setdefault(key, []).append(_Pair(before, after))
    return groups, None


def _code_undone(state: SessionState) -> set[int]:
    """The turns whose code a standing code undo took back: a snapshot of each is behind it."""
    redone = {r.undo_seq for _, r in state.lines() if isinstance(r, RedoLine) and r.code}
    out: set[int] = set()
    for _, record in state.lines():
        if isinstance(record, UndoLine) and record.code and record.seq not in redone:
            out.update(range(record.to_turn, record.seq))
    return out


def _snapshotted(groups: dict[str, list[_Pair]]) -> frozenset[int]:
    return frozenset(p.before.turn for pairs in groups.values() for p in pairs)


def _harness_paths(repo: Path, groups: dict[str, list[_Pair]]) -> tuple[str, ...]:
    """The paths the harness wrote, for the kept-code notice when only the conversation goes
    back (decision 5)."""
    paths: set[str] = set()
    for pairs in groups.values():
        paths.update(harness_paths(repo, [p.trees() for p in pairs]))
    return tuple(sorted(paths))


def _runs_from(state: SessionState, to_turn: int) -> tuple[str, ...]:
    return tuple(
        e.run_id
        for e in state.episodes()
        if (e.opened_at_seq if e.opened_at_seq is not None else e.seq) >= to_turn
    )


def _names(key: str | None, task: str) -> bool:
    """Whether a commit's task key is `task`: its key (`T3@2`), or the task by its plan's id,
    whose one committed attempt it is."""
    return key is not None and (key == task or split_task_key(key)[0] == task)
