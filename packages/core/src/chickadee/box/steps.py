"""Steps: every writing call of a box role is one recorded, undoable effect.

`StepRecord` is a dispatcher `around` wrapper (`tools/rules.Wrapper`): a call whose spec mutates
something (an edit tool, `shell`) runs inside one `worker_step` effect. The intent names the
tree before the call, its pre-image: the last step's tree after, or the task's start tree. The
completion names the tree after as `post_state`, the paths that changed between the two as
`observed`, and the answer's sha as `result`. A command that opened the network says so among
the intent's preconditions, which resume reads (a networked step is never re-run). The
changed paths are reindexed, so the graph and every navigation tool see the worktree as it now
is. A read passes through: no step. A call that raises is put back to the tree before it and its
step completed `failed` (`put_back`), the same undo resume gives a step a crash cut off. A path
the person's `deny` globs name that a call changed is put back from the tree before, and the
answer says so: the edit port refuses such a path, so a command is what changes one.

Each step's tree after is committed on its task's step chain (`workspace.park_step`), so
`git gc` keeps every one; `workspace.sweep_snapshots` sweeps the chains.

Its callers are the worker and the desk (a task's trees, and the desk's take-over), composed
by the episode (`box/compose.compose_episode`).

**Concurrency.** A `StepTrees` belongs to one task, and its role serves one call at a time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from chickadee.approval.prefs import denied, path_rule
from chickadee.approval.tier import Preferences
from chickadee.box.commit import Reindexer
from chickadee.box.edits import writes_seen
from chickadee.box.shell import BoxShell
from chickadee.box.tells import CUT, RoundToken
from chickadee.context.results import Interrupted
from chickadee.hooks.payloads import AfterStep
from chickadee.hooks.points import HookPoint
from chickadee.journal.effects import EffectKind, EffectLog, Intent, perform
from chickadee.journal.projection import Operation, RunProjection
from chickadee.schema.interaction import network_opened
from chickadee.tools.catalogue import args_sha, failed
from chickadee.tools.context import ToolContext
from chickadee.tools.rules import Around, Pending
from chickadee.tools.tool import Answer, noted
from chickadee.work import in_work
from chickadee.workspace import changed_paths, current_tree, park_step, restore_tree

__all__ = [
    "NETWORK",
    "Beside",
    "StepRecord",
    "StepSnapshot",
    "StepTrees",
    "beside_jobs",
    "beside_note",
    "beside_precondition",
    "caught_note",
    "networked",
    "put_back",
    "reached",
    "task_steps",
]

NETWORK = "network=on"
"""How a step's intent says its command opened the network (`shell`'s `network: true`):
resume reads it (`networked`), and never runs such a step again."""

_HOST = "host="
"""A step intended by an older version named each host its command declared, one
precondition each."""


_BESIDE: Final = "beside="
"""How a step's intent names the jobs that ran beside it."""


class Beside(Protocol):
    """The session's running jobs, as a step sees them (`zeus/jobs.JobTable`):
    which are running, and what each step recorded, so a job's exit can say what it wrote that
    no step did and what it changed after a step had."""

    def running_in(self, root: Path) -> tuple[str, ...]:
        """The ids of the jobs running in `root` now: a job in another workspace cannot write
        this one."""
        ...

    def started_in(self, root: Path) -> int:
        """How many jobs have ever started in `root`: one that started and ended between two
        steps wrote between them too."""
        ...

    def stepped(self, root: Path, step: str, paths: tuple[str, ...], tree: str) -> None:
        """The step `step` in `root` recorded `paths`, as they are in `tree`, the tree it left
        once it had put back what it does."""
        ...


def beside_precondition(running: Sequence[str]) -> tuple[str, ...]:
    """The precondition naming the jobs a step ran beside, durable in its intent, or none."""
    return (f"{_BESIDE}{','.join(running)}",) if running else ()


def beside_jobs(preconditions: Sequence[str]) -> tuple[str, ...]:
    """The jobs a step's intent says it ran beside: what `undo` warns of."""
    for p in preconditions:
        if p.startswith(_BESIDE):
            return tuple(p[len(_BESIDE) :].split(","))
    return ()


def beside_note(running: Sequence[str]) -> tuple[str, ...]:
    """What a command's step beside a running job says: the job writes
    unrecorded, and a command has no list of its own writes, so what this step recorded may
    hold some of the job's."""
    if not running:
        return ()
    jobs = ", ".join(running)
    return (f"[{jobs} was running: this record may include {jobs}'s writes]",)


def caught_note(running: Sequence[str], caught: Sequence[str]) -> tuple[str, ...]:
    """What an edit step beside a running job says of the paths that changed during it that it
    did not write: they were the job's, and are not this step's."""
    if not caught:
        return ()
    jobs = ", ".join(running)
    return (f"[caught from {jobs}: {', '.join(caught)} changed during this call; not its own]",)


def networked(preconditions: Sequence[str]) -> bool:
    """Whether a step's command opened the network, as its intent says: the one reading, for
    resume, of this and of a step intended by an older version."""
    return any(p == NETWORK or p.startswith(_HOST) for p in preconditions)


@dataclass(frozen=True)
class StepSnapshot:
    """The tree after one step, what changed since the tree before, and what was left out."""

    tree: str
    changed: tuple[str, ...]
    skipped: tuple[str, ...]
    """Untracked files over the cap, named and not stored."""


class StepTrees:
    """The last tree a task's steps reached: the "before" of the next step. Set
    at delegation from the task's start tree; on resume, from the last completed step's tree
    (`before`).
    `untracked_max_bytes` is the turn snapshots' bound."""

    def __init__(
        self,
        start: str,
        untracked_max_bytes: int,
        *,
        run_id: str,
        task: str | None,
        before: str | None = None,
    ) -> None:
        self._before = start if before is None else before
        self.start = start
        """The task's start tree: what its files are compared with at its stop."""
        self._cap = untracked_max_bytes
        self._run_id = run_id
        self._task = task

    def before(self) -> str:
        return self._before

    @property
    def cap(self) -> int:
        return self._cap

    def after(
        self, root: Path, measured: tuple[str, tuple[str, ...]] | None = None
    ) -> StepSnapshot:
        """The tree of `root` now, parked on the task's step chain, and what changed since
        the last step: the next step's "before". `measured` is `current_tree`'s answer when
        the caller already took it with nothing written since, so it is not taken twice."""
        tree, skipped = (
            measured
            if measured is not None
            else current_tree(root, untracked_max_bytes=self._cap)
        )
        changed = changed_paths(root, self._before, tree) if tree != self._before else ()
        chain = self._task if self._task is not None else "athena"
        park_step(root, self._run_id, chain, tree, f"chickadee step {self._run_id} {chain}")
        self._before = tree
        return StepSnapshot(tree, changed, skipped)


class StepRecord:
    """The `around` wrapper that records each writing call as a `worker_step`."""

    def __init__(
        self,
        root: Path,
        log: EffectLog,
        trees: StepTrees,
        indexer: Reindexer | None,
        *,
        prefs: Callable[[], Preferences] | None = None,
        work_dir: str | None = None,
        beside: Beside | None = None,
        cut: RoundToken | None = None,
    ) -> None:
        self._root = root
        self._log = log
        self._trees = trees
        self._cut = cut
        """The round's token: a command that opened no network, or any command
        while its question waits, is cut by a tell, and put back. None where no tell reaches
        the calls."""
        self._beside = beside
        """The session's running jobs; `None` where nothing runs beside."""
        self._indexer = indexer
        self._prefs = prefs
        self._work_dir = work_dir
        """`[work] dir`: a command's write into it is put back; `doc`'s is
        the step's own write."""
        self.last: StepSnapshot | None = None
        """The last step's snapshot: what the facts read for the skipped files."""
        self.last_beside: tuple[str, ...] = ()
        """The jobs the last step ran beside: its changes may be theirs."""
        self.last_operation: str | None = None
        """The last step's operation id: its `exec` line's, when it ran a command."""
        self._jobs_seen = 0
        """`Beside.started_in` at the last step: a job started since may have written since."""

    def around(self, pending: Pending, run: Around, ctx: ToolContext) -> Answer:
        spec = pending.spec
        if spec is None or spec.mutates == "nothing":
            return run()
        # A command that ran with the network says so, so resume never runs it again: hinted,
        # its class needing it, unsealed, where every command has it, or asked,
        # since an approved command runs outside the sandbox.
        hint = network_opened(pending.raw.get("network"))
        outside = pending.raw.get("sandbox") is False
        command = pending.raw.get("command")
        network = spec.mutates == "environment" and (
            ctx.exec.opens_network(command, hint, outside)
            if isinstance(ctx.exec, BoxShell) and isinstance(command, str)
            else hint
        )
        running = self._beside.running_in(self._root) if self._beside is not None else ()
        started = self._beside.started_in(self._root) if self._beside is not None else 0
        # Beside a job, running now or run since the last step, the "before"
        # is the disk's, not the last step's "after", so what the job wrote between two steps
        # is never this step's.
        fresh = bool(running) or started != self._jobs_seen
        self._jobs_seen = started
        drift = self._trees.after(self._root).changed if fresh else ()
        before = self._trees.before()
        intent = Intent(
            operation_id=pending.call.operation_id,
            kind=EffectKind.WORKER_STEP,
            approved_sha=args_sha(pending.raw),
            preconditions=(
                f"before={before}",
                *((NETWORK,) if network else ()),
                *beside_precondition(running),
            ),
            idempotent=False,
            surface=(spec.name,),
            preimages=before,
        )
        command = spec.mutates == "environment"
        # A networked command is cut only while its question waits (`RoundToken.asking`).
        cut = self._cut if command else None
        if cut is not None:
            cut.begin_call()
        whole = cut.cuttable() if cut is not None and not network else nullcontext()
        with perform(self._log, intent) as effect:
            try:
                with writes_seen() as own, whole:
                    answer = run()
                if cut is not None and cut.tripped:
                    # A tell cut it: all or none, as a call that raised, and said.
                    restored = put_back(self._root, before, self._trees.cap, self._indexer)
                    effect.complete("failed", observed=restored, post_state=before)
                    return Interrupted(
                        spec.name,
                        CUT,
                        do="the files are as before this call; read the message that follows, "
                        "and call it again only if the work still needs it",
                    )
                # Inside the guard: the person's file may not parse, or git may refuse, and
                # the step is still all or none.
                notes, measured = self._put_back(before, command=command)
            except Exception:
                # The call is all or none: what it wrote before it raised is put back, as
                # resume puts back a step the crash cut off, and the step says so.
                restored = put_back(self._root, before, self._trees.cap, self._indexer)
                effect.complete("failed", observed=restored, post_state=before)
                raise
            snap = self._trees.after(self._root, measured)
            # An anchored edit over a file the job changed is refused stale
            # by its anchor; a whole-file write is not, so it is said.
            replaced = tuple(p for p in snap.changed if p in drift)
            if replaced:
                jobs = ", ".join(running)
                notes = (*notes, f"[this replaced {jobs}'s change to {', '.join(replaced)}]")
            # An edit knows what it wrote, so a path that changed during it
            # and that it did not write was the job's: said here, and left to the job's `wrote`.
            # A command knows no such list, so its record may include the job's writes.
            caught = (
                () if command or not running else tuple(p for p in snap.changed if p not in own)
            )
            label = beside_note(running) if command else caught_note(running, caught)
            for note in (*notes, *label, *_after_step(ctx, pending, snap.changed)):
                answer = noted(answer, note)
            if snap.changed and self._indexer is not None:
                self._indexer.reindex([self._root / p for p in snap.changed])
            self.last, self.last_beside = snap, tuple(running)
            self.last_operation = pending.call.operation_id
            effect.complete(
                "failed" if failed(answer) else "succeeded",
                observed=snap.changed,
                post_state=snap.tree,
                result=hashlib.sha256(answer.render().encode()).hexdigest(),
            )
            if self._beside is not None:
                mine = tuple(p for p in snap.changed if p not in caught)
                self._beside.stepped(self._root, pending.call.operation_id, mine, snap.tree)
            return answer

    def _put_back(
        self, before: str, *, command: bool
    ) -> tuple[tuple[str, ...], tuple[str, tuple[str, ...]] | None]:
        """Put back from `before` every changed path the person's `deny` globs name,
        and for a command every one in the work folder (`doc` writes it),
        one note per path, and the tree it measured when nothing was put back, for
        `StepTrees.after` to reuse. Read only when there is something to judge, so a step pays
        no second tree otherwise. A path in no tree, git-ignored or over the cap, cannot be put
        back: no tree holds it, and its note says so rather than that it was."""
        prefs = self._prefs() if self._prefs is not None else None
        deny = prefs is not None and bool(prefs.deny)
        work = self._work_dir if command else None
        if not deny and work is None:
            return (), None
        measured = current_tree(self._root, untracked_max_bytes=self._trees.cap)
        now = measured[0]
        if now == before:
            return (), measured
        changed = changed_paths(self._root, before, now)
        hits: list[tuple[str, str]] = []
        for p in changed:
            glob = denied(prefs, p) if prefs is not None and deny else None
            if glob is not None and prefs is not None:
                hits.append((p, f"{path_rule(prefs, glob)} denies it"))
            elif work is not None and in_work(p, work):
                hits.append((p, "the work folder is written with doc"))
        if not hits:
            return (), measured
        restored = set(restore_tree(self._root, before, tuple(p for p, _ in hits)))
        if restored and self._indexer is not None:
            self._indexer.reindex([self._root / p for p in sorted(restored)])
        notes = tuple(
            f"[put back {p}: {why}]"
            if p in restored
            else f"[{p} changed and {why}, but no tree holds it to put back]"
            for p, why in hits
        )
        return notes, None if restored else measured


def _after_step(
    ctx: ToolContext, pending: Pending, changed: tuple[str, ...]
) -> tuple[str, ...]:
    """`after_step`: the role's hooks told what the step changed, whatever
    wrote it; their note rides on the call's answer, before its record is completed."""
    hooks = ctx.hooks
    if hooks is None or not hooks.has(HookPoint.AFTER_STEP) or pending.spec is None:
        return ()
    payload = AfterStep(tool=pending.spec.name, changed=changed)
    note = hooks.fire(HookPoint.AFTER_STEP, payload, call_id=pending.call.id).note
    return (note,) if note else ()


def put_back(
    root: Path, before: str, untracked_max_bytes: int, indexer: Reindexer | None
) -> tuple[str, ...]:
    """The paths changed since tree `before` put back to it (`workspace.restore_tree`) and
    reindexed: a step that did not complete, undone. Returns the paths it touched."""
    now, _ = current_tree(root, untracked_max_bytes=untracked_max_bytes)
    changed = changed_paths(root, before, now) if now != before else ()
    restored = restore_tree(root, before, changed)
    if restored and indexer is not None:
        indexer.reindex([root / p for p in restored])
    return restored


def task_steps(run: RunProjection, task: str | None) -> tuple[Operation, ...]:
    """`task`'s steps in the order they were intended (Athena's own, `task` None)."""
    return tuple(
        op
        for op in run.operations.values()
        if op.kind == EffectKind.WORKER_STEP.value and op.task_id == task
    )


def reached(steps: Sequence[Operation], start: str) -> str:
    """The tree the last completed step reached, or `start` when none did: what the next
    step starts from, and where a task's facts end."""
    for op in reversed(steps):
        if op.completed is not None and op.completed.post_state_sha:
            return op.completed.post_state_sha
    return start
