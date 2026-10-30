"""The desk: the
harness side of Athena's three task tools, and the only writer of the episode's task lines.

- **`plan`**: her first plan is asked at stop 2 (`approval/stops`, through the `StopAsker`
  the composition binds); approved, it is a `plan_written` line, in the person's text if they
  edited it. A note answers her with the note, verbatim, to revise by; a cancel ends her tool
  use and the episode (`ended`). Once a plan is approved, every later `plan` is a revision:
  journaled, shown, never asked.
- **`delegate`**: one writer at a time, so it is refused while a task is open, and before a
  plan is approved. A task is named by its plan: `delegate(task="T3")` runs the pinned
  plan's `T3`, its next attempt if an earlier one was dropped; with no plan the harness names it
  by lineage (`tasks.next_folderless`). It journals `task_delegated` with the tree and commit it
  starts from, and runs the worker's first round; its answer is the round's report and facts
  (`render_round`).
- **`review`** names the open task, or is refused. Each answer is a `task_reviewed` line,
  then: `accept` commits the task through `Projector.commit` (`task_commit`; in place,
  `task_known_good`); `feedback` is the worker's next round on the same conversation;
  `take_over` makes the task hers; `re_cut` replaces it with the tasks she delegates next,
  refused past `athena.max_recut_depth`; `drop` (once `abandon`) gives it up. What stays is
  said apart: `fresh` on `feedback`, `take_over` or `re_cut` puts every file
  back to the task's start, or only those it lists, under its own `worker_step`; omitted, the
  changes are kept; a `drop` always puts every file back.

**A pause** (at a task boundary): once `pause` is set, the next `delegate` starts no
task; it answers that the episode paused and sets `paused`, and her request ends before its
next model call (`box/athena.py`). On resume she reads that answer, and delegates again.

**Her edits and her steps.** Her edit tools are always offered, and the desk refuses
them unless she holds a task (`edits`); her writing calls are recorded under the task she
holds, or under none (`around`, `operation_id`), each a `worker_step` like the worker's.

**Resume.** Each line the desk writes carries the `call_id` of the call that made it.
A call a crash cut off is dispatched again, and the desk, finding its own line for that call,
continues from it rather than doing it twice: a `delegate` whose task was delegated resumes or
answers that task; a `review` whose line was written finishes what it began (the commit
reconciled by its trailer, the restore done again, the feedback round resumed); a `plan`
whose line was written answers as it did.

Composed by `box/compose.compose_episode`; `box/loop.TaskLoop` calls it.

**Concurrency.** One desk per episode, called from Athena's thread, one call at a time. Its
state is the run journal; it holds nothing it cannot rebuild from it but `ended`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from chickadee.approval.stops import Reread, StopAsker, StopOutcome, WorkspacePreview, stop_of
from chickadee.approval.tier import Preferences
from chickadee.box.carry import Carried, PriorTask
from chickadee.box.commit import Projector, Reindexer, commit_effect, task_body
from chickadee.box.edits import WorktreeEdits
from chickadee.box.facts import last_attempt, measure, render_facts
from chickadee.box.report import CoderReport
from chickadee.box.resume import Asking, never_asking, task_trees
from chickadee.box.steps import Beside, StepRecord, StepTrees
from chickadee.box.tasks import TaskBook, TaskEntry, growth, lineage, next_folderless, one_task
from chickadee.box.tells import RoundToken, Tells, told_text
from chickadee.box.worker import RoundReport, WorkerRun
from chickadee.boxctx.records import SymbolTable, TaskRecord, flat, task_record
from chickadee.context.results import InvalidArgument, ToolFailure
from chickadee.context.work_line import WorkLine
from chickadee.edits.draft import DraftRefused
from chickadee.edits.formats.base import Targets
from chickadee.edits.lines import AnchorKind
from chickadee.edits.match import Region
from chickadee.edits.symbols import SymbolSpan
from chickadee.errors import HarnessBug
from chickadee.hooks.fire import RoleHooks
from chickadee.hooks.payloads import RoundEnded, TaskCommitted, TaskReviewed
from chickadee.hooks.points import HookPoint
from chickadee.hooks.steer import harness_notes, hook_review, retry_line, round_ended
from chickadee.ids import (
    ATHENA_ROUND,
    in_task,
    split_task_key,
    step_id,
    step_prefix,
    task_commit_id,
    task_depth,
    task_key,
)
from chickadee.journal.effects import EffectKind, Intent, on_journal, perform, reopen
from chickadee.journal.port import JournalPort
from chickadee.journal.projection import Operation, RunProjection
from chickadee.profiles.model import LanguageProfile
from chickadee.schema.ids import sha256_hex
from chickadee.schema.interaction import StopAnswer, StopKind
from chickadee.schema.journal import (
    EventKind,
    Fresh,
    PlanOutgrownP,
    PlanTaskP,
    PlanWrittenP,
    ReviewAnswer,
    TaskDelegatedP,
    TaskReportedP,
    TaskReviewedP,
    TellOutcome,
    TellReplyP,
)
from chickadee.tools.context import DraftCall, EditScope, EditsPort, ToolContext
from chickadee.tools.rules import Around, Pending
from chickadee.tools.tool import Answer
from chickadee.work import (
    approved_plans,
    part_dir,
    pin_plan,
    plan_errors,
    plan_ids,
    plan_pin,
    plan_tasks,
)
from chickadee.work_records import keep_private, records
from chickadee.workspace import (
    changed_paths,
    current_tree,
    folder_tree,
    pinned_text,
    restore_tree,
    tree_diff,
)

__all__ = [
    "ENDED",
    "NO_TASK_HELD",
    "OUTGROWN",
    "TAKEN_OVER",
    "ClosedDraft",
    "DeskParts",
    "Ended",
    "TaskDesk",
    "render_round",
]


def _listed_tasks(text: str) -> tuple[PlanTaskP, ...]:
    """The tasks a plan's text lists, as its `plan_written` line records them:
    `work.plan_tasks`' reading, the one `delegate(task)` resolves ids through."""
    return tuple(PlanTaskP(id=task, title=title) for task, title, _file in plan_tasks(text))


TAKEN_OVER: Final = (
    "you hold {task}: your edit tools now work on it; then review {task} accept or drop"
)
"""`take_over`'s answer. It names no shell: she may hold none."""

NO_TASK_HELD: Final = DraftRefused(
    "DENIED",
    "you hold no task, so you change no file",
    do="delegate the change, or take the open task over with review(task, take_over)",
)


@dataclass(frozen=True)
class DeskParts:
    """What the desk reads of its run besides the journal and the worker."""

    root: Path
    workspace: WorkspacePreview
    """Where the work runs, for the stop's display."""
    profiles: tuple[LanguageProfile, ...]
    untracked_max_bytes: int
    recut_cap: int
    """`athena.max_recut_depth`: a `re_cut` of a task at this depth is refused."""
    one_task_max: int
    """`athena.one_task_max_tasks`: the tasks a one-task plan's episode may
    delegate, its own counted; a design's is not capped."""
    brief_template_sha: str | None = None
    """The brief template in force (`handoffs.resolve`), recorded on each delegation."""
    design_layer: int = 4000
    """The worker's "done so far" budget, in tokens."""
    prefs: Callable[[], Preferences] | None = None
    """The person's approval.toml: a step puts back a path its `deny` globs name."""
    work_dir: str | None = None
    """`[work] dir`: a command's write into it is put back."""
    beside: Beside | None = None
    """The session's running jobs: each step beside one says so."""
    work: WorkLine | None = None
    """The work folder the design is in: her plan is its `plan/`, checked, pinned at
    stop 2 and read from the pin. None for a one-task plan or a host's design text."""
    plan_template: str = ""
    """The plan template in force: what `plan/README.md` is checked against (A1)."""
    brief_template: str = ""
    """The brief template in force: what each `plan/tasks/` file is checked against (A6)."""
    design_files: tuple[tuple[str, str], ...] = ()
    """The pinned design's other files and their lines (A4), named in each brief (A7)."""
    design_tree: str | None = None
    """The `design/` tree stop 1 pinned, which its approval commit names (decision 6)."""
    work_commit: bool = True
    """`[work] commit` (B7): false, and the folder is never committed, only kept privately."""
    records: WorktreeEdits | None = None
    """The edit port the work's records are written through: the worktree's, unfenced, since
    they are the harness's files of the folder both roles' ports refuse."""
    private: Path | None = None
    """Where the folder is kept privately when `work_commit` is off: the clone's local root,
    inside the git dir, so never in a tree."""
    carried: Carried = field(default_factory=Carried.none)
    """What this run carries from the work's earlier runs: none when it
    starts the work."""


@dataclass(frozen=True)
class Ended:
    """How stop 2 ended the episode: cancelled by the person, with their words, or by the
    rounds bound, with the harness's reason in `bound` (never the person's)."""

    note: str | None
    bound: str | None = None

    @property
    def why(self) -> str:
        """The episode's `why`, saying who ended it."""
        if self.bound is not None:
            return f"the plan was not approved: {self.bound}"
        return "the user cancelled the plan" + (f": {self.note}" if self.note else "")


class TaskDesk:
    """`tools/context.DeskPort` over the run journal, the worker and the commit."""

    def __init__(
        self,
        journal: JournalPort,
        worker: WorkerRun,
        projector: Projector,
        stops: StopAsker,
        parts: DeskParts,
        *,
        indexer: Reindexer | None = None,
        symbols: Callable[[], SymbolTable | None] | None = None,
        pause: threading.Event | None = None,
        asking: Asking = never_asking,
        hooks: RoleHooks | None = None,
        tells: Tells | None = None,
    ) -> None:
        self._journal = journal
        self._tells = tells
        """Zeus's messages to the team: delivered with a held round, and
        each owed a `reply`. None where no tell reaches the run."""
        self._hooks = hooks or RoleHooks.none("athena")
        """Athena's hooks: the desk's points are hers."""
        self._worker = worker
        self._projector = projector
        self._stops = stops
        self._parts = parts
        self._indexer = indexer
        self._symbols = symbols
        self._pause = pause
        self.asking = asking
        """Whether a step's permission question is still waiting on the session log:
        read by every resume this episode makes, the worker's and Athena's (`box/loop.py`)."""
        self._book: tuple[int, TaskBook] | None = None
        """The book folded at one journal position: `held` is read several times a call.
        Valid while the journal's `last_seq` is unchanged, since the fold is pure."""
        self.paused = False
        """Set when a `delegate` found `pause` set: no task was started, and her request ends
        before its next model call."""

    # -- what the desk knows ----------------------------------------------------------------

    def book(self) -> TaskBook:
        run = self._journal.projection()
        cached = self._book
        if cached is None or cached[0] != run.last_seq:
            cached = (run.last_seq, TaskBook(run))
            self._book = cached
        return cached[1]

    @property
    def ended(self) -> Ended | None:
        """The person's cancel at stop 2, from its journaled line: her tool use ends, and so
        does the episode (`ABORTED`, the person's stop), resumed or not."""
        for line in self._journal.projection().lines(PlanWrittenP):
            if line.payload.outcome == "cancelled":
                sha = line.payload.note_sha
                note = self._journal.get_blob(sha).decode("utf-8") if sha else None
                return Ended(note, line.payload.bound)
        return None

    @property
    def held(self) -> str | None:
        """The task she holds after `take_over`, or None."""
        entry = self.book().open()
        return entry.id if entry is not None and entry.state == "taken_over" else None

    @property
    def outgrown(self) -> bool:
        """Whether she ended the one-task plan `OUTGROWN`: its journaled
        `plan_outgrown` line, so a resume ends the episode as she did."""
        return bool(self._journal.projection().lines(PlanOutgrownP))

    def plans(self) -> tuple[PlanWrittenP, ...]:
        """Every plan approved or revised, in order; a cancelled one is not a plan."""
        return approved_plans(self._journal.projection())

    # -- plan ------------------------------------------------------------------------------

    def plan(self, text: str | None, *, call_id: str) -> str | ToolFailure:
        """Her plan: with a work folder, `plan/` as it stands, `text` left
        out; without one, the plan's `text`, as before (a one-task plan's re-cut)."""
        if self.ended is not None:
            return ENDED
        written = any(
            p.payload.call_id == call_id for p in self._journal.projection().lines(PlanWrittenP)
        )
        owed = None if written else self._owed()
        if owed is not None:
            return owed
        work = self._parts.work
        if work is not None:
            if text is not None:
                return InvalidArgument(
                    "text",
                    "this work's plan is its folder's plan/, not text",
                    do="write plan/ with doc, then call plan() with no text",
                )
            return self._plan_folder(work, call_id)
        if text is None:
            return InvalidArgument(
                "text",
                "this work has no folder",
                do="call plan again with the whole plan as text",
            )
        plans = self.plans()
        mine = next((p for p in plans if p.call_id == call_id), None)
        if mine is not None:
            return self._planned(mine, text)
        if plans:
            return self._planned(self._write_plan(text, call_id, asked=False), text)
        stop = stop_of(StopKind.PLAN, "athena", text, self._parts.workspace)
        out = self._stops(stop)
        return self._asked(out, text, call_id)

    def _plan_folder(self, work: WorkLine, call_id: str) -> str | ToolFailure:
        """`plan/` checked (decision 13, A4, A6), refused naming what is wrong; the first plan
        that passes asked at stop 2 with its folder, re-read at every answer; a later one a
        revision, shown and not asked. Each version is pinned as `plan-v<n>` under the run."""
        folder = self._plan_dir(work)
        plans = self.plans()
        mine = next((p for p in plans if p.call_id == call_id), None)
        if mine is not None:
            text = self._journal.get_blob(mine.sha).decode("utf-8")
            return self._planned(mine, text)
        errors = self._plan_errors(folder)
        if errors:
            return InvalidArgument(
                "plan",
                "plan/ is not in shape: " + "; ".join(errors),
                do="fix each problem named in plan/ with doc, then call plan() again",
            )
        text = (self._parts.root / folder / "README.md").read_text(encoding="utf-8")
        tree = folder_tree(self._parts.root, folder) if plans else None
        stands = plans and tree is not None and tree == self._plan_pin(len(plans))
        if plans and (stands or not self._asks_again(plans[-1])):
            version = len(plans) + 1
            if tree is None:
                raise HarnessBug(f"{folder} is not in the tree; its shape was checked first")
            self._pin_plan(tree, folder, version)
            said = "stands" if stands else "revised"
            self._approval_commit(
                f"approved:plan:{version}",
                f"chickadee: plan {said} (version {version})",
                (folder,),
                f"Tree: {tree}",
            )
            line = self._write_plan(text, call_id, asked=False)
            if stands:
                # The same plan, approved under this run's design: said as standing, and
                # pinned again so a later revision is bound to this design.
                return (
                    f"the plan (version {len(plans)}) stands unchanged for this design; carry "
                    "on with it"
                )
            return self._planned(line, text)

        def reread() -> Reread:
            now = (self._parts.root / folder / "README.md").read_text(encoding="utf-8")
            version = folder_tree(self._parts.root, folder) or ""
            return Reread(now, version, tuple(self._plan_errors(folder)))

        version = len(plans) + 1
        stop = stop_of(
            StopKind.PLAN,
            "athena",
            text,
            self._parts.workspace,
            folder=str(self._parts.root / folder),
            notice=self._changes_since(len(plans), folder) if plans else None,
        )
        out = self._stops(stop, reread=reread)
        if out.answer is StopAnswer.APPROVE:
            tree = out.version
            if not tree:
                raise HarnessBug(f"the stop approved {folder} with no tree")
            self._pin_plan(tree, folder, version)
            self._approval_commit(
                f"approved:plan:{version}",
                "chickadee: plan approved"
                if version == 1
                else f"chickadee: plan approved (version {version})",
                (folder,),
                f"Tree: {tree}",
            )
        return self._asked(out, text, call_id)

    def _asks_again(self, latest: PlanWrittenP) -> bool:
        """Whether a revision of the plan is asked at stop 2: only when the latest
        approval was given under another design than this run's, which an amended design
        makes so. Every other revision is shown, not asked."""
        if latest.inherited_from is None:
            return False
        continues = self._parts.carried.continues
        bound = continues.plan_bound_to if continues is not None else None
        return bound != self._parts.design_tree

    def _plan_pin(self, version: int) -> str | None:
        return plan_pin(self._parts.root, self._journal.run_id, version)

    def _changes_since(self, version: int, folder: str) -> str | None:
        """What changed in `plan/` since `version`, shown with the stop."""
        before = self._plan_pin(version)
        now = folder_tree(self._parts.root, folder)
        if before is None or now is None:
            return None
        return tree_diff(self._parts.root, before, now, f"plan version {version}")

    def seed_plan(self) -> None:
        """The approved plan this run inherits, as its first `plan_written`
        line and pin, before Athena's first call: approved already, so never asked again.
        Once: a resumed run finds its line."""
        carried, work = self._parts.carried, self._parts.work
        continues = carried.continues
        if continues is None or continues.plan_run is None or work is None or self.plans():
            return
        tree = carried.plan_tree(self._parts.root)
        if tree is None or continues.plan_sha is None:
            raise HarnessBug(f"run {continues.plan_run}'s approved plan was never pinned")
        folder = self._plan_dir(work)
        self._pin_plan(tree, folder, 1)
        readme = pinned_text(self._parts.root, tree, "README.md") or ""
        self._journal.append(
            EventKind.PLAN_WRITTEN,
            PlanWrittenP(
                sha=continues.plan_sha,
                by="athena",
                version=1,
                asked=False,
                inherited_from=continues.plan_run,
                tasks=_listed_tasks(readme),
            ),
        )

    def carried_committed(self) -> bool:
        """Whether an earlier run of the work committed a task: a run that continues
        it and accepts nothing more has still ended with work committed."""
        return bool(self._parts.carried.committed())

    def standing(self) -> str | None:
        """Where the work stands, for Athena's head in a run that continues it:
        what the earlier runs did, the plan she inherits and whether it is asked again, what is
        unfinished, and what is not started. None when the run starts the work."""
        carried = self._parts.carried
        continues = carried.continues
        if continues is None:
            return None
        runs = ", ".join(continues.prior_runs)
        out = [
            "## Where the work stands",
            f"This run continues the work after its earlier runs stopped ({runs}). Nothing "
            "they committed is done again.",
        ]
        if continues.mode == "AMEND" and continues.amended_from and self._parts.design_tree:
            diff = tree_diff(
                self._parts.root,
                continues.amended_from,
                self._parts.design_tree,
                "the design approved before",
            )
            out += ["", "The person approved a change to the design.", diff or ""]
        plans = self.plans()
        if not plans:
            out += [
                "",
                "No plan was approved yet. plan/ holds what was written of it: finish it to "
                "the plan template and call plan.",
            ]
        elif self._asks_again(plans[-1]):
            out += [
                "",
                "The plan (plan/README.md as pinned) was approved under the design before this "
                "change. Decide whether it still covers the design: call plan unchanged if it "
                "does, or revise plan/ first; a revision is asked of the person.",
            ]
        else:
            out += [
                "",
                "The plan is approved (plan/README.md as pinned): do not plan again; call plan "
                "only to revise it.",
            ]
        out += ["", *self._stood()]
        out += ["", "Delegate the next task."]
        return "\n".join(out)

    def _stood(self) -> list[str]:
        """Each plan task as the earlier runs left it."""
        carried = self._parts.carried
        listed = [t.id for t in self.plans()[-1].tasks] if self.plans() else []
        done: list[str] = []
        unfinished: list[str] = []
        settled: list[str] = []
        lost: list[str] = []
        now: str | None = None
        for prior in carried.tasks():
            name = prior.entry.name
            if prior.done:
                commit = prior.entry.commit
                at = f" {commit[:7]}" if commit and not prior.in_place else ""
                done.append(f"{name}{at}")
            elif prior.lost:
                lost.append(name)
            elif not prior.unfinished:
                said = "dropped" if prior.entry.state == "abandoned" else "re-cut"
                settled.append(f"{name} ({said})")
            else:
                # The tree is snapshot once, and only when a task was left unfinished.
                now = now or self._tree_now()
                left = self._carried_paths(prior, now)
                unfinished.append(
                    f"{name} (left uncommitted: {', '.join(left)})"
                    if left
                    else f"{name} (nothing of it is left)"
                )
        named = {t.entry.name for t in carried.tasks()}
        rest = [task for task in listed if task not in named]
        lines = [f"Committed: {', '.join(done) or 'none'}."]
        if unfinished:
            lines.append(
                f"Not finished: {'; '.join(unfinished)}. Each was started and stopped before "
                "its review; delegating it again goes on from what it left."
            )
        if lost:
            lines.append(
                f"Lost: {', '.join(lost)}. Each was committed, and its commit has left the "
                "branch since: delegate it again."
            )
        if settled:
            lines.append(f"Settled without a commit: {', '.join(settled)}.")
        lines.append(f"Not started: {', '.join(rest) or 'none'}.")
        return lines

    def record_design(self) -> None:
        """The approved design as its own commit in the worktree, first in
        the run, naming the pinned tree; nothing in place or without a folder. Made once:
        a resumed run finds it by its lifecycle or its trailer."""
        work, tree = self._parts.work, self._parts.design_tree
        if work is None or tree is None:
            return
        folder = part_dir(work.dir, work.name, "design")
        continues = self._parts.carried.continues
        amended = continues is not None and continues.mode == "AMEND"
        self._approval_commit(
            "approved:design",
            "chickadee: design amended" if amended else "chickadee: design approved",
            (folder,),
            f"Tree: {tree}",
        )

    def record_report(self, report: CoderReport) -> bool:
        """The work's README and `report.md` as the run ends, projected
        from `report`, and in a worktree committed as one unit (decision 6). In place they are
        written and left for the person's commit. Nothing without a folder. Returns
        whether anything was written, so the caller re-reads what the run changed."""
        work = self._parts.work
        if work is None or self._parts.design_tree is None:
            return False
        folder = f"{work.dir}/{work.name}"
        texts = records(self._parts.root, folder, work.name, report)
        self._write_records(texts)
        paths = tuple(texts)
        if not self._parts.work_commit and self._parts.private is not None:
            keep_private(self._parts.root / folder, self._parts.private / work.name)
        self._approval_commit("records:report", f"chickadee: report {work.name}", paths, "")
        return True

    def _write_records(self, texts: Mapping[str, str]) -> None:
        """The harness's own files of the work folder, through the edit port (invariant 1), all
        or none; not a model's write, so no step. Unchanged, nothing."""
        port = self._parts.records
        if port is None:
            raise HarnessBug("a work folder's records were written with no port for them")
        with port.edit() as draft:
            for path, text in texts.items():
                refused = draft.apply(path, None, text)
                if refused is not None:
                    raise HarnessBug(f"the work's records were refused: {refused.reason}")
            said = draft.answer()
            if isinstance(said, DraftRefused):
                raise HarnessBug(f"the work's records were refused: {said.reason}")

    def _approval_commit(
        self, operation_id: str, subject: str, paths: Sequence[str], body: str
    ) -> None:
        """Harness-owned files of the work folder committed as their own unit; once. Never in
        place, nor with `work.commit` off (B7)."""
        if self._projector.in_place or not self._parts.work_commit:
            return
        run = self._journal.projection()
        op = run.operations.get(operation_id)
        if op is not None and op.completed is not None:
            return
        # Searched only since this run began: an operation id is this run's alone, and an
        # earlier run on the branch (another work's, or this work's before it continued) has
        # committed under the same one (`Projector.landed`).
        start = run.start
        landed = self._projector.landed(
            operation_id,
            since=start.repo_head if start is not None and start.repo_head else None,
        )
        commit_effect(
            self._projector,
            on_journal(self._journal),
            operation_id,
            subject,
            paths,
            kind=EffectKind.APPROVAL_COMMIT,
            in_place=False,
            landed=landed,
            body=body,
        )

    def _plan_dir(self, work: WorkLine) -> str:
        return part_dir(work.dir, work.name, "plan")

    def _plan_errors(self, folder: str) -> list[str]:
        return plan_errors(
            self._parts.root / folder,
            self._parts.plan_template,
            self._parts.brief_template,
            approved=self._approved_ids(),
            delegated={e.name for e in self.book().entries() if e.plan_task is not None}
            | {t.entry.name for t in self._parts.carried.tasks()},
        )

    def _approved_ids(self) -> dict[str, str]:
        """Every id an approved version of the plan held, by its file: read
        from each version's pin, oldest first, so an id keeps the file it was first approved
        with."""
        held = self._parts.carried.approved_ids(self._parts.root)
        mine = plan_ids(
            self._parts.root,
            [self._plan_pin(version) for version in range(1, len(self.plans()) + 1)],
        )
        for task, file in mine.items():
            held.setdefault(task, file)
        return held

    def _pin_plan(self, tree: str, folder: str, version: int) -> None:
        pin_plan(self._parts.root, tree, folder, self._journal.run_id, version)

    def _asked(self, out: StopOutcome, text: str, call_id: str) -> str | ToolFailure:
        """How stop 2 ended, as her answer, its `plan_written` line written."""
        if out.answer is StopAnswer.APPROVE:
            return self._planned(self._write_plan(out.text, call_id, asked=True), text)
        if out.answer is StopAnswer.NOTE:
            return (
                "the person did not approve this plan; their note, verbatim:\n"
                f"{out.note}\nRevise the plan by it and call plan again."
            )
        # The text last shown, which is what was cancelled, stored so its sha resolves: an
        # edit at an earlier round may have replaced Athena's first text.
        self._journal.append(
            EventKind.PLAN_WRITTEN,
            PlanWrittenP(
                call_id=call_id,
                sha=self._journal.put_blob(out.text.encode("utf-8")),
                by="athena",
                version=1,
                asked=True,
                outcome="cancelled",
                note_sha=self._journal.put_blob(out.note.encode("utf-8")) if out.note else None,
                bound=out.bound,
                tasks=_listed_tasks(out.text),
            ),
        )
        ended = Ended(out.note, out.bound)
        return f"{ended.why}. Nothing has run. Stop here and reply with your report."

    def _write_plan(self, text: str, call_id: str | None, *, asked: bool) -> PlanWrittenP:
        sha = self._journal.put_blob(text.encode("utf-8"))
        line = PlanWrittenP(
            call_id=call_id,
            sha=sha,
            by="athena" if call_id is not None else "zeus",
            version=len(self.plans()) + 1,
            asked=asked,
            tasks=_listed_tasks(text),
        )
        self._journal.append(EventKind.PLAN_WRITTEN, line)
        return line

    def _planned(self, line: PlanWrittenP, sent: str) -> str:
        if not line.asked:
            return (
                f"plan revised (version {line.version}): shown to the person, not asked; carry "
                "on with it"
            )
        if line.sha == sha256_hex(sent):
            return "the person approved the plan. Delegate its first task."
        text = self._journal.get_blob(line.sha).decode("utf-8")
        return (
            "the person approved the plan as they edited it; this is the plan now:\n\n"
            f"{text}\n\nDelegate its first task."
        )

    # -- delegate --------------------------------------------------------------------------

    def delegate(
        self,
        title: str | None = None,
        brief: str | None = None,
        *,
        task: str | None = None,
        call_id: str,
    ) -> str | ToolFailure:
        """A task to the worker: with a work folder, one of the pinned plan's `Tasks` by id,
        its brief its pinned file; without one, a `title` and `brief`, as
        before (a one-task plan's re-cut)."""
        if self.ended is not None:
            return ENDED
        book = self.book()
        mine = next((e for e in book.entries() if e.call_id == call_id), None)
        if mine is not None:
            return self._continue(mine)
        owed = self._owed()
        if owed is not None:
            return owed
        if self.outgrown:
            return InvalidArgument(
                "task",
                "you ended this plan OUTGROWN: nothing more runs",
                do="stop calling tools and reply with your report",
            )
        if not self.plans():
            return InvalidArgument(
                "plan", "no plan is approved yet", do="call plan first, then delegate its tasks"
            )
        cap = self._parts.one_task_max
        if one_task(book) and len(book.entries()) >= cap:
            return InvalidArgument(
                "task",
                f"this one-task plan has delegated {len(book.entries())} tasks, its cap "
                f"(athena.one_task_max_tasks = {cap})",
                do="finish the open task, take it over with "
                'review(task=..., answer="take_over"), '
                "or end the plan with outgrown",
            )
        work = self._parts.work
        if work is not None:
            if task is None or title is not None or brief is not None:
                return InvalidArgument(
                    "task",
                    "this work's tasks are the approved plan's",
                    do='delegate one of them by its id: delegate(task="T1")',
                )
            ran = book.of_plan(task)
            if ran is not None:
                return InvalidArgument(
                    "task",
                    f"the plan's {task} was delegated, and is {ran.state}: a plan task runs "
                    "once",
                    do=f"delegate another task; to run {task} again, drop it first",
                )
            prior = self._parts.carried.task(task)
            done = self._done_before(prior)
            if done is not None:
                return done
            pinned = self._pinned_task(work, task)
            if isinstance(pinned, InvalidArgument):
                return pinned
            title, brief = pinned
            key = self._next_key(book, task, prior)
        elif title is None or brief is None or task is not None:
            return InvalidArgument(
                "title",
                "this work has no folder, so a task is not named by id",
                do="call delegate with a title and a brief",
            )
        else:
            key = next_folderless(self._journal.projection(), book)
        held = book.open()
        if held is None and self._pause is not None and self._pause.is_set():
            self.paused = True
            return InvalidArgument(
                "task",
                "the person paused the episode at this task boundary: nothing was delegated",
                do="stop calling tools and reply with where the work stands; when the episode "
                "resumes, delegate it again",
            )
        if held is not None:
            return InvalidArgument(
                "task",
                f"{held.id} is open; one task at a time",
                do=f"review {held.id} first (accept, feedback, take_over, re_cut or drop), "
                "then delegate the next",
            )
        started = self._start(key, title, brief, call_id=call_id, by="athena", plan_task=task)
        return self._render(self._steered(started))

    def _pinned_task(self, work: WorkLine, task: str) -> tuple[str, str] | InvalidArgument:
        """`task`'s title and brief from the latest pinned plan (decision 14): refused unless
        the plan lists it and its file on disk is the pinned one, so a task is never started
        from a version nobody approved. The brief is the pinned file, then the work's files by
        path, read-only to the worker (decision 15, A7)."""
        folder = self._plan_dir(work)
        plans = self.plans()
        tree = self._plan_pin(len(plans))
        if tree is None:
            raise HarnessBug(f"plan version {len(plans)} was approved and never pinned")
        readme = pinned_text(self._parts.root, tree, "README.md") or ""
        listed = {t: (title, file) for t, title, file in plan_tasks(readme)}
        if task not in listed:
            named = ", ".join(listed) or "none"
            return InvalidArgument(
                "task",
                f"{task} is not in the approved plan: {named}",
                do="delegate one of the plan's tasks by the id above",
            )
        title, file = listed[task]
        pinned = pinned_text(self._parts.root, tree, file)
        disk = self._parts.root / folder / file
        now = disk.read_text(encoding="utf-8") if disk.is_file() else None
        if pinned is None or now != pinned:
            return InvalidArgument(
                "task",
                f"{task}'s file plan/{file} changed after the plan was approved",
                do=f"call plan to approve the change, then delegate {task}",
            )
        return title, _with_files(pinned, work, readme, self._parts.design_files)

    def end_outgrown(self, *, call_id: str) -> str | ToolFailure:
        """She ends the one-task plan `OUTGROWN`: its finishing needs a
        decision nobody made, a different problem or a wider scope. Refused in a design's
        episode, whose scope is its design, and while a task is open, which she reviews first;
        her reply after it is her report."""
        if self.ended is not None:
            return ENDED
        book = self.book()
        if not one_task(book):
            return InvalidArgument(
                "plan",
                "outgrown ends a one-task plan; this episode runs a design, whose scope is the "
                "design",
                do="do not call outgrown; say in your report what the design does not cover",
            )
        held = book.open()
        if held is not None:
            return InvalidArgument(
                "task",
                f"{held.id} is open",
                do=f"review {held.id} first (accept what is done, or drop it), then end the "
                "plan with outgrown",
            )
        if not self.outgrown:
            self._journal.append(EventKind.PLAN_OUTGROWN, PlanOutgrownP(call_id=call_id))
        return OUTGROWN

    def paused_before_start(self) -> bool:
        """Whether the person paused the episode before its one task was delegated: nothing
        has started, so it pauses now, at the first task boundary, as a design's first
        `delegate` does. Sets `paused`."""
        if self._pause is None or not self._pause.is_set():
            return False
        if any(e.by == "zeus" for e in self.book().entries()):
            return False
        self.paused = True
        return True

    def delegate_one(self, title: str, brief: str) -> RoundReport:
        """Zeus's one-task plan, approved at stop 2: its `plan_written` and its
        `task_delegated` lines, both by Zeus, then the worker's first round, before Athena is
        called. On resume, the task already delegated is continued instead."""
        book = self.book()
        first = next((e for e in book.entries() if e.by == "zeus"), None)
        if first is not None:
            got = self._round_of(first)
            return self._steered(got if got is not None else self._first_report(first.id))
        if not self.plans():
            self._write_plan(brief, None, asked=True)
        return self._steered(self._start("T1", title, brief, call_id=None, by="zeus"))

    def _start(
        self,
        task: str,
        title: str,
        brief: str,
        *,
        call_id: str | None,
        by: Literal["zeus", "athena"],
        plan_task: str | None = None,
    ) -> RoundReport:
        """`task` is the attempt's key: the plan's id, its next attempt, or the name
        `next_folderless` gave. Every attempt starts from the tree as it is: one that goes on
        from an earlier run's unfinished attempt (`_carried_paths`) is told what that one left,
        and commits it, but never starts from that one's start, which would put back what was
        committed and written since."""
        start, _ = current_tree(
            self._parts.root, untracked_max_bytes=self._parts.untracked_max_bytes
        )
        started_at = self._projector.head()
        self._journal.append(
            EventKind.TASK_DELEGATED,
            TaskDelegatedP(
                task_id=task,
                call_id=call_id,
                title=flat(title),
                brief_sha=self._journal.put_blob(brief.encode("utf-8")),
                by=by,
                start_tree=start,
                start_commit=started_at,
                depth=task_depth(split_task_key(task)[0]),
                template_sha=self._parts.brief_template_sha,
                plan_task=plan_task,
            ),
            task_id=task,
        )
        before = self._symbols() if self._symbols is not None else None
        return self._worker.start(
            task,
            brief,
            start,
            done=self._done(),
            design_layer=self._parts.design_layer,
            before=before,
            learned=self._first_note(task),
        )

    def _done_before(self, prior: PriorTask | None) -> InvalidArgument | None:
        """A plan task an earlier run of the work committed is never delegated again,
        unless its commit has left the branch (`lost`); nor one it re-cut, whose pieces run in
        its place, as `book.of_plan` refuses within a run."""
        if prior is not None and prior.entry.state == "replaced":
            name = prior.entry.name
            return InvalidArgument(
                "task",
                f"{name} was re-cut in an earlier run of this work ({prior.run_id}): its "
                "pieces run in its place",
                do=f"delegate its pieces ({name}.1, {name}.2, …) as the plan lists them",
            )
        if prior is None or not prior.done:
            return None
        commit = prior.entry.commit
        name = prior.entry.name
        at = f" as {commit[:7]}" if commit and not prior.in_place else ""
        return InvalidArgument(
            "task",
            f"{name} was committed in an earlier run of this work ({prior.run_id}){at}",
            do="delegate a task not done yet; rework of a committed task is a new task in the "
            "plan",
        )

    def _next_key(self, book: TaskBook, task: str, prior: PriorTask | None) -> str:
        """`task`'s next attempt across the work's runs, so no two of its attempts share a
        commit's operation id or a step chain."""
        here = book.latest(task)
        before = prior.attempts if prior is not None else 0
        return task_key(task, max(here.attempt if here is not None else 0, before) + 1)

    def _first_note(self, task: str) -> str | None:
        """What an attempt's worker is told first: where an earlier run left the task, for the
        first attempt in this run that goes on from it; else what the attempt it replaces
        learned. The same at delegation and on a resume of the first round."""
        entry = TaskBook(self._journal.projection()).get(task)
        prior = self._carried_for(entry)
        if entry is not None and prior is not None and self._first_of(entry, prior):
            return _carried_note(prior, self._carried_paths(prior, self._tree_now()))
        return self._learned(task)

    def _carried_for(self, entry: TaskEntry | None) -> PriorTask | None:
        """The unfinished attempt of an earlier run that `entry` goes on from: the plan task's
        own, for this run's attempts of it and its pieces, until one of them commits. A drop
        in this run puts back only this run's changes, so what the earlier run left stays for
        the next attempt to commit."""
        if entry is None:
            return None
        name = entry.name
        prior = next(
            (
                p
                for p in self._parts.carried.tasks()
                if p.unfinished and in_task(name, p.entry.name)
            ),
            None,
        )
        if prior is None:
            return None
        family = prior.entry.name
        for other in self.book().entries():
            if other.state == "committed" and in_task(other.name, family):
                return None
        return prior

    def _first_of(self, entry: TaskEntry, prior: PriorTask) -> bool:
        family = prior.entry.name
        first = next((e for e in self.book().entries() if in_task(e.name, family)), None)
        return first is not None and first.id == entry.id

    def _tree_now(self) -> str:
        now, _ = current_tree(
            self._parts.root, untracked_max_bytes=self._parts.untracked_max_bytes
        )
        return now

    def _carried_paths(self, prior: PriorTask, now: str) -> tuple[str, ...]:
        """What the unfinished attempt `prior` left: each file its steps recorded touching
        (`worker_step`'s observed surface) that differs between its start and `now`. A file
        it changed that was put back since is not one."""
        run = dict(self._parts.carried.prior).get(prior.run_id)
        if run is None:
            return ()
        touched = {
            path
            for op in run.operations.values()
            if op.task_id == prior.entry.id and op.completed is not None
            for path in op.completed.observed_surface
        }
        if not touched:
            return ()
        start = prior.entry.start_tree
        changed = changed_paths(self._parts.root, start, now) if now != start else ()
        return tuple(p for p in changed if p in touched)

    def _continue(self, entry: TaskEntry) -> str:
        """A `delegate` call cut off after its task was delegated: the task's open round
        resumed, its first round run if none began, or its last report answered."""
        got = self._round_of(entry)
        if got is None:
            got = self._reported(entry.id, _last(self._journal.projection(), entry.id))
        return self._render(self._steered(got))

    def _round_of(self, entry: TaskEntry) -> RoundReport | None:
        """The first round of a delegated task, run or resumed; None once it reported."""
        if entry.rounds > 0:
            return None
        resumed = self._worker.resume(entry.id, asking=self.asking)
        if resumed is not None:
            return resumed
        line = _delegated(self._journal.projection(), entry.id)
        brief = self._journal.get_blob(line.brief_sha).decode("utf-8")
        # Nothing of the round ran, so the tree is still the task's start: the table can be
        # taken now, as at delegation.
        return self._worker.start(
            entry.id,
            brief,
            entry.start_tree,
            done=self._done(),
            design_layer=self._parts.design_layer,
            before=self._symbols() if self._symbols is not None else None,
            learned=self._first_note(entry.id),
        )

    def _learned(self, task: str) -> str | None:
        """What the attempt `task` replaces learned: a piece of a `re_cut`,
        or a dropped task delegated again (`tasks.lineage`). None for a task that replaces
        nothing, and for an attempt that never reported."""
        run = self._journal.projection()
        settled = lineage(run).replaces.get(task)
        if settled is None:
            return None
        last = _last(run, settled.task_id)
        if last is None:
            return None
        cut = settled.answer == "re_cut"
        put_back: bool | tuple[str, ...] = (settled.fresh or False) if cut else True
        return last_attempt(
            settled.task_id,
            "re-cut" if cut else "dropped",
            self._text_of(settled),
            last.stopped,
            last.facts,
            put_back,
        )

    def _done(self) -> tuple[TaskRecord, ...]:
        """Each committed task's record, oldest first: the next task's "done so far". From its
        last report's facts; a task she took over is measured again, since her changes came
        after its report."""
        run = self._journal.projection()
        out: list[TaskRecord] = []
        prior = dict(self._parts.carried.prior)
        for done in self._parts.carried.committed():
            earlier = prior.get(done.run_id)
            last = _last(earlier, done.entry.id) if earlier is not None else None
            facts = last.facts if last is not None else None
            name = done.entry.name
            out.append(task_record(name, done.entry.title, facts, done.entry.commit))
        for entry in TaskBook(run).entries():
            if entry.state != "committed":
                continue
            last = _last(run, entry.id)
            if last is not None and "take_over" not in _answers(run, entry.id):
                facts = last.facts
            else:
                facts = measure(
                    self._parts.root,
                    run,
                    entry.id,
                    self._parts.profiles,
                    None,
                    None,
                    self._journal.get_blob,
                )
            out.append(task_record(entry.id, entry.title, facts, entry.commit))
        return tuple(out)

    # -- review ----------------------------------------------------------------------------

    def review(
        self,
        task: str,
        answer: ReviewAnswer,
        text: str | None,
        *,
        call_id: str,
        fresh: Fresh = None,
    ) -> str | ToolFailure:
        if self.ended is not None:
            return ENDED
        run = self._journal.projection()
        mine = next(
            (p.payload for p in run.lines(TaskReviewedP) if p.payload.call_id == call_id), None
        )
        if mine is not None:
            entry = TaskBook(run).get(mine.task_id)
            if entry is None:
                raise ValueError(f"{mine.task_id} was reviewed and never delegated")
            return self._act(entry, mine, call_id)
        book = TaskBook(run)
        entry = book.open()
        if entry is None:
            return InvalidArgument(
                "task", "no task is open", do="delegate a task first, then review its report"
            )
        if task not in (entry.id, entry.name):
            return InvalidArgument(
                "task",
                f"review names the open task, {entry.name}, not {task}",
                do=f"send review again with task {entry.name}",
            )
        task = entry.id
        refused = self._refused(entry, answer, text, fresh)
        if refused is not None:
            return refused
        text_sha = self._journal.put_blob(text.encode("utf-8")) if text else None
        line = TaskReviewedP(
            task_id=task,
            call_id=call_id,
            answer=answer,
            text_sha=text_sha,
            fresh=_fresh(answer, fresh),
        )
        self._journal.append(EventKind.TASK_REVIEWED, line, task_id=task)
        return self._act(entry, line, call_id)

    def _refused(
        self, entry: TaskEntry, answer: ReviewAnswer, text: str | None, fresh: Fresh
    ) -> InvalidArgument | None:
        """Why this review of the open task cannot be given, or None."""
        task = entry.id
        if entry.state == "accepted" and answer not in ("accept", "drop"):
            # Its commit completed unsucceeded: accept again (a new attempt, asking the
            # trailer first) or drop it.
            return InvalidArgument(
                "answer",
                f"{task} is accepted and its commit did not land",
                do=f'send review(task="{task}", answer="accept") again to land it, or drop it',
            )
        if entry.state not in ("reported", "held", "taken_over", "accepted"):
            return InvalidArgument(
                "task",
                f"{task} has no report to review yet",
                do="wait for the worker's report; review it when it arrives",
            )
        if answer == "take_over" and entry.state == "taken_over":
            return InvalidArgument(
                "answer",
                f"you already hold {task}",
                do=f'finish it yourself, then review(task="{task}", answer="accept"), '
                "or drop it",
            )
        if answer == "continue" and entry.state != "held":
            return InvalidArgument(
                "answer",
                f"continue resumes a round held for a message, and {task}'s last round was not",
                do="accept, give feedback, take it over, re-cut or drop it",
            )
        if answer == "feedback" and not (text and text.strip()):
            return InvalidArgument(
                "text",
                "feedback is the worker's next message, and it is empty",
                do="send review again with text: what to change, and why",
            )
        if answer == "re_cut" and entry.depth >= self._parts.recut_cap:
            return InvalidArgument(
                "answer",
                f"re-cut limit ({self._parts.recut_cap}) reached for {task}",
                do="take it over, give feedback, or drop it",
            )
        if answer == "accept" and fresh:
            return InvalidArgument(
                "fresh",
                "accept commits the work as it is, so nothing is put back",
                do="to start over, send feedback, take_over or re_cut with fresh, or drop the "
                "task; to keep the work, accept without fresh",
            )
        if isinstance(fresh, tuple) and answer != "drop":
            changed = self._changed(entry)
            stray = [p for p in fresh if p not in changed]
            if stray:
                named = ", ".join(changed) or "nothing"
                return InvalidArgument(
                    "fresh",
                    f"{task} did not change {', '.join(stray)}; it changed {named}",
                    do="send fresh again naming only the paths the task changed",
                )
        return None

    def _changed(self, entry: TaskEntry) -> tuple[str, ...]:
        """What `entry`'s task changed since its start, as the tree is now."""
        root, cap = self._parts.root, self._parts.untracked_max_bytes
        now, _ = current_tree(root, untracked_max_bytes=cap)
        return changed_paths(root, entry.start_tree, now) if now != entry.start_tree else ()

    def _text_of(self, line: TaskReviewedP) -> str | None:
        return self._journal.get_blob(line.text_sha).decode("utf-8") if line.text_sha else None

    def _act(self, entry: TaskEntry, line: TaskReviewedP, call_id: str) -> str:
        """What a review does, from its line: the first time, or again on resume, each step
        safe to repeat or found already done. A put-back comes first, so what the answer then
        does starts from the tree it leaves."""
        task = entry.id
        text = self._text_of(line)
        back = self._restore(task, call_id, only=line.fresh) if line.fresh else ()
        # `task_reviewed`: its note is the worker's on feedback, else hers.
        noted = self._hooks.noted_once(
            HookPoint.TASK_REVIEWED,
            TaskReviewed(task=task, answer=line.answer, text=text, fresh=line.fresh),
            call_id=call_id,
            task_id=task,
        )
        match line.answer:
            case "accept":
                return _noted(self._accepted(task, call_id), noted)
            case "feedback":
                said = _noted(_with_put_back(text or "", back, worker=True), noted)
                return self._feedback(task, said)
            case "continue":
                return _noted(
                    self._render(self._steered(self._feedback_round(task, None))), noted
                )
            case "take_over":
                said = _with_put_back(TAKEN_OVER.format(task=task), back, worker=False)
                return _noted(said, noted)
            case "drop" | "abandon":
                self._restore(task, call_id)
                return _noted(f"{task} dropped; the tree is as before it", noted)
            case "re_cut":
                kept = "" if line.fresh is True else "; what it changed and you kept stays"
                said = _recut(entry.name, folder=self._parts.work is not None) + kept
                return _noted(_with_put_back(said, back, worker=False), noted)

    def _accepted(self, task: str, call_id: str) -> str:
        """The accepted task committed, and what `task_committed`'s hooks say of it."""
        said = self._commit(task)
        if not self._hooks.has(HookPoint.TASK_COMMITTED):
            return said
        entry = TaskBook(self._journal.projection()).get(task)
        if entry is None or entry.state != "committed":
            return said
        payload = TaskCommitted(task=task, sha=entry.commit or None, files=self._changed(entry))
        noted = self._hooks.noted_once(
            HookPoint.TASK_COMMITTED, payload, call_id=call_id, task_id=task
        )
        return _noted(said, noted)

    def _feedback(self, task: str, text: str) -> str:
        return self._render(self._steered(self._feedback_round(task, text)))

    def _feedback_round(self, task: str, text: str | None) -> RoundReport:
        """The next round, on the same conversation, `text` its next message, or none after a
        held round (`continue`); on resume, the round the review opened resumed,
        or its report once it reported."""
        run = self._journal.projection()
        entry = TaskBook(run).get(task)
        if entry is None:
            raise ValueError(f"{task} was never delegated")
        reviewed_at = _reviewed_round(run, task)
        if entry.rounds > reviewed_at:
            return self._reported(task, _last(run, task))
        resumed = self._worker.resume(task, asking=self.asking)
        if resumed is not None:
            return resumed
        return self._worker.cont(task, text, entry.rounds)

    def _steered(self, report: RoundReport) -> RoundReport:
        """`report`'s round past its `round_ended` hooks: while one gives the
        task another round, the harness writes Athena's own `feedback` line for it and the
        round runs, as hers would; then the report she reads. A retry already written for a
        round (a resume) is acted on, never asked again."""
        while True:
            if report.stopped == "held":
                # A held round is no round's end to anything else: no hook adds a
                # retry on top of a tell, and it counts toward no steering bound.
                return report
            task, round_ = report.task, report.round
            line = hook_review(self._journal.projection(), task, round_)
            if line is None:
                taken = round_ended(
                    self._hooks,
                    RoundEnded(
                        task=task,
                        round=round_,
                        stopped=report.stopped,
                        report=report.report,
                        facts=report.facts,
                        calls_left=self._worker.calls_left(task),
                        progress=self._worker.progress(task, round_)
                        if self._hooks.has(HookPoint.ROUND_ENDED)
                        else None,
                    ),
                    room=self._worker.room(task),
                )
                if taken is None:
                    return report
                by, retry = taken
                text_sha = self._journal.put_blob(retry.text.encode("utf-8"))
                line = retry_line(by, task, round_, text_sha, retry)
                self._journal.append(EventKind.TASK_REVIEWED, line, task_id=task)
            back = self._restore(task, line.call_id, only=line.fresh) if line.fresh else ()
            said = _with_put_back(self._text_of(line) or "", back, worker=True, by=line.by)
            report = self._feedback_round(task, said)

    def _commit(self, task: str) -> str:
        """`task_commit` (`task_known_good` in place), never made twice. A commit a
        crash cut off is found by its trailer, or made now (`_reconcile_commit`). An attempt
        that completed unsucceeded (git raised: `ambiguous`, or `failed`) keeps its lifecycle,
        so a new accept is a new attempt under its own operation id, which first asks each
        earlier attempt's trailer, and records a commit found that way rather than making a
        second."""
        run = self._journal.projection()
        entry = TaskBook(run).get(task)
        if entry is None:
            raise ValueError(f"{task} was never delegated")
        if entry.state == "committed":
            return self._committed(task, entry.commit)
        line = _delegated(run, task)
        attempts = _commit_ops(run, task)
        open_ = next((op for op in attempts if op.completed is None), None)
        if open_ is not None:
            return self._reconcile_commit(task, open_, line)
        in_place = self._projector.in_place
        landed = None
        if not in_place:
            landed = next(
                (
                    found
                    for op in attempts
                    if (
                        found := self._projector.landed(
                            op.operation_id, since=line.start_commit
                        )
                    )
                ),
                None,
            )
        root = self._parts.root
        now, _ = current_tree(root, untracked_max_bytes=self._parts.untracked_max_bytes)
        changed = changed_paths(root, entry.start_tree, now) if now != entry.start_tree else ()
        carried = self._carried_for(entry)
        if carried is not None:
            # What the earlier run's attempt left uncommitted is this task's too.
            changed = tuple(sorted({*changed, *self._carried_paths(carried, now)}))
        sha = commit_effect(
            self._projector,
            on_journal(self._journal, task_id=task),
            task_commit_id(task, len(attempts) + 1),
            _subject(task, entry.title),
            changed,
            kind=EffectKind.TASK_KNOWN_GOOD if in_place else EffectKind.TASK_COMMIT,
            in_place=in_place,
            landed=landed,
            body=self._body(run, task, line),
        )
        return self._committed(task, sha or None)

    def _reconcile_commit(self, task: str, op: Operation, line: TaskDelegatedP) -> str:
        """An accept whose commit did not complete: in place there is nothing to find; else
        the commit by its trailer since the task's start commit, recorded if found, else made
        now."""
        log = on_journal(self._journal, task_id=task)
        effect = reopen(log, op)
        if self._projector.in_place:
            head = self._projector.head()
            effect.complete("succeeded", observed=op.surface, post_state=head)
            return self._committed(task, head)
        found = self._projector.landed(op.operation_id, since=line.start_commit)
        if found is None:
            run = self._journal.projection()
            found = self._projector.commit(
                _subject(task, line.title),
                op.operation_id,
                op.surface,
                self._body(run, task, line),
            )
        effect.complete("succeeded", observed=op.surface, post_state=found)
        return self._committed(task, found)

    def _body(self, run: RunProjection, task: str, line: TaskDelegatedP) -> str:
        return _task_body(run, task, line, self._journal.get_blob)

    def _committed(self, task: str, sha: str | None) -> str:
        if self._projector.in_place or not sha:
            return f"{task} accepted; in place nothing is committed (D39)"
        return f"committed {task} as {sha[:7]}"

    def _restore(
        self, task: str, call_id: str, *, only: bool | tuple[str, ...] | None = True
    ) -> tuple[str, ...]:
        """The tree put back to `task`'s start, every file or `only` those, under its own
        `worker_step`: finished if a crash cut it off, done again under a new step if it
        completed unsucceeded, since putting a tree back is idempotent; nothing more once one
        succeeded. Returns the paths it put back, from the step's record when it already ran."""
        run = self._journal.projection()
        entry = TaskBook(run).get(task)
        if entry is None:
            raise ValueError(f"{task} was never delegated")
        prefix = step_prefix(task, ATHENA_ROUND, call_id)
        made = [op for op in run.operations.values() if op.operation_id.startswith(prefix)]
        done = next(
            (
                op
                for op in made
                if op.completed is not None and op.completed.outcome == "succeeded"
            ),
            None,
        )
        if done is not None and done.completed is not None:
            return done.completed.observed_surface
        root, cap = self._parts.root, self._parts.untracked_max_bytes
        log = on_journal(self._journal, task_id=task)
        now, _ = current_tree(root, untracked_max_bytes=cap)
        changed = changed_paths(root, entry.start_tree, now) if now != entry.start_tree else ()
        if isinstance(only, tuple):
            changed = tuple(p for p in changed if p in only)
        open_ = next((op for op in made if op.completed is None), None)
        if open_ is not None:
            restored = restore_tree(root, entry.start_tree, changed)
            self._reindexed(restored)
            reopen(log, open_).complete(
                "succeeded", observed=restored, post_state=self._after(entry, only)
            )
            return restored
        intent = Intent(
            step_id(task, ATHENA_ROUND, call_id),
            EffectKind.WORKER_STEP,
            approved_sha=sha256_hex(f"restore {task} to {entry.start_tree}: {only}"),
            preconditions=(f"before={now}",),
            idempotent=True,
            surface=("review",),
            preimages=now,
        )
        with perform(log, intent) as fx:
            restored = restore_tree(root, entry.start_tree, changed)
            self._reindexed(restored)
            fx.complete("succeeded", observed=restored, post_state=self._after(entry, only))
        return restored

    def _after(self, entry: TaskEntry, only: bool | tuple[str, ...] | None) -> str:
        """The tree a put-back leaves: the task's start for every file, else as it is now."""
        if not isinstance(only, tuple):
            return entry.start_tree
        now, _ = current_tree(
            self._parts.root, untracked_max_bytes=self._parts.untracked_max_bytes
        )
        return now

    def _reindexed(self, paths: Sequence[str]) -> None:
        if paths and self._indexer is not None:
            self._indexer.reindex([self._parts.root / p for p in paths])

    # -- her edits and her steps -------------------------------------------------------------

    def edits(self, port: EditsPort) -> EditsPort:
        """Her `edits` port: `port` while she holds a task, a draft that takes no edit
        otherwise."""
        return _HeldEdits(self, port)

    def operation_id(self, call_id: str) -> str:
        """Her call's step id: under the task she holds, or her own (`athena:0`)."""
        held = self.held
        return step_id(held, ATHENA_ROUND if held is not None else None, call_id)

    def around(self, pending: Pending, run: Around, ctx: ToolContext) -> Answer:
        """Her writing call as one `worker_step`, under the task she holds, its trees the
        task's from the journal; outside a task, under none, from the tree as it is now."""
        spec = pending.spec
        if spec is None or spec.mutates == "nothing":
            return run()
        held = self.held
        root, cap = self._parts.root, self._parts.untracked_max_bytes
        run_id = self._journal.run_id
        if held is not None:
            start, last = task_trees(self._journal.projection(), held)
            trees = StepTrees(start, cap, run_id=run_id, task=held, before=last)
        else:
            now, _ = current_tree(root, untracked_max_bytes=cap)
            trees = StepTrees(now, cap, run_id=run_id, task=None)
        log = on_journal(self._journal, task_id=held)
        steps = StepRecord(
            root,
            log,
            trees,
            self._indexer,
            prefs=self._parts.prefs,
            work_dir=self._parts.work_dir,
            beside=self._parts.beside,
            # Her own command is cut by a tell as the worker's is.
            cut=ctx.cancel if isinstance(ctx.cancel, RoundToken) else None,
        )
        return steps.around(pending, run, ctx)

    # -- what she reads ----------------------------------------------------------------------

    def _render(self, report: RoundReport) -> str:
        if report.stopped == "held":
            return self._held_message(report)
        notes = harness_notes(self._hooks, task=report.task, round_=report.round)
        return render_round(report, notes=notes)

    def _held_message(self, report: RoundReport) -> str:
        """A round held for a message, as she reads it: the round so far, then
        Zeus's messages, each delivered now; on resume, the ones delivered at that boundary."""
        tells = self._tells
        due = (
            ()
            if tells is None
            else tells.delivered_at(report.task, report.round)
            or tells.deliver(task=report.task, round_=report.round)
        )
        entry = TaskBook(self._journal.projection()).get(report.task)
        name = entry.name if entry is not None else report.task
        title = f" {entry.title!r}" if entry is not None else ""
        lines = [
            f"{name}{title} held for a message (round {report.round}, unfinished; the worker "
            "wrote no report).",
            render_facts(report.facts, report.stopped),
        ]
        if due:
            lines.append(told_text(due))
        lines.append(
            f"Decide what {name} needs (review: continue leaves the worker undisturbed, "
            "feedback gives it your words) and answer each message (reply)."
        )
        return "\n\n".join(lines)

    def _owed(self) -> InvalidArgument | None:
        """A delivered tell she has not replied to refuses her `delegate` and `plan`, naming
        it."""
        owed = self._tells.owed() if self._tells is not None else ()
        if not owed:
            return None
        return InvalidArgument(
            "tell",
            f"reply to {owed[0]} first",
            do=f'reply(tell="{owed[0]}", outcome=..., text=...): applied, planned, '
            "needs_design or declined",
        )

    def reply(
        self, tell: str, outcome: TellOutcome, text: str, *, call_id: str
    ) -> str | ToolFailure:
        """Her answer to a delivered tell: `tell_reply` on the journal, and Zeus
        told; a reply needing him wakes him. One per tell."""
        tells = self._tells
        if tells is None:
            return InvalidArgument("tell", "no message reached this run", do="carry on")
        replies = self._journal.projection().lines(TellReplyP)
        done = next((p.payload for p in replies if p.payload.call_id == call_id), None)
        if done is not None:
            return f"replied to {done.tell}: {done.outcome}"
        if tell not in tells.owed():
            if any(p.payload.tell == tell for p in replies):
                why = f"{tell} was answered already"
            elif any(t.id == tell for t in tells.pending()):
                why = f"{tell} has not reached you yet; it comes at your next step"
            else:
                why = f"{tell} is no message of this run"
            return InvalidArgument(
                "tell", why, do="reply to a message delivered to you, by its id"
            )
        if not text.strip():
            return InvalidArgument("text", "a reply says what you did", do="send it again")
        tells.reply(tell, outcome, text, call_id=call_id)
        return f"replied to {tell}: {outcome}"

    def _last_report(self, task: str) -> str:
        return self._render(self._reported(task, _last(self._journal.projection(), task)))

    def _first_report(self, task: str) -> RoundReport:
        found = [
            p.payload
            for p in self._journal.projection().tasks.get(task, ())
            if isinstance(p.payload, TaskReportedP)
        ]
        return self._reported(task, found[0] if found else None)

    def _reported(self, task: str, line: TaskReportedP | None) -> RoundReport:
        if line is None:
            raise ValueError(f"{task} has no report")
        blob = self._journal.get_blob
        words = blob(line.report_sha).decode("utf-8") if line.report_sha else None
        return RoundReport(task, line.round, line.stopped, words, line.facts)


def render_round(report: RoundReport, *, notes: tuple[str, ...] = ()) -> str:
    """A round as she reads it (step 6): the worker's words, then the facts, then what
    the harness's hooks said of the round, under their own heading."""
    words = report.report if report.report is not None else "(the worker did not reply)"
    said = (
        f"{report.task} · round {report.round}\n"
        f"REPORT (worker): {words}\n\n"
        f"{render_facts(report.facts, report.stopped)}"
    )
    return said if not notes else said + "\n\nharness:\n" + "\n".join(notes)


class ClosedDraft:
    """A draft that takes no edit: every call is refused with `refusal`. Hers while she holds
    no task."""

    def __init__(self, refusal: DraftRefused) -> None:
        self._refusal = refusal

    @property
    def targets(self) -> Targets:
        return Targets(writable=(), deletes=())

    def text(self, path: str) -> str | DraftRefused:
        return self._refusal

    def crlf(self, path: str) -> bool:
        return False

    def line(self, path: str, anchor: int | str, kind: AnchorKind) -> int | DraftRefused:
        return self._refusal

    def symbol(self, path: str, name: str) -> SymbolSpan | DraftRefused:
        return self._refusal

    def apply(self, path: str, region: Region | None, content: str) -> DraftRefused | None:
        return self._refusal

    def delete(self, path: str) -> DraftRefused | None:
        return self._refusal

    def move(self, source: str, target: str) -> DraftRefused | None:
        return self._refusal

    def answer(self) -> str | DraftRefused:
        return self._refusal


_NO_TASK: Final = ClosedDraft(NO_TASK_HELD)


class _HeldEdits:
    def __init__(self, desk: TaskDesk, port: EditsPort) -> None:
        self._desk = desk
        self._port = port

    def edit(self) -> EditScope:
        if self._desk.held is None:
            closed: DraftCall = _NO_TASK
            return nullcontext(closed)
        return self._port.edit()


ENDED: Final = InvalidArgument(
    "plan",
    "the user cancelled the plan: nothing more runs",
    do="stop calling tools and reply with your report",
)

OUTGROWN: Final = (
    "the plan ends OUTGROWN when you reply: call no more tools, and reply with your report: "
    "what is missing (the decision, the problem or the scope the brief does not hold), what is "
    "done and committed, and what you propose next"
)
"""`outgrown`'s answer: Zeus reads her reply as the report."""


def _subject(task: str, title: str) -> str:
    """A task's commit subject: its id and its title, on one line. `task` is the
    attempt's key; the subject names the task as people read it."""
    return f"chickadee: {split_task_key(task)[0]} {flat(title)}"


def _recut(task: str, *, folder: bool) -> str:
    """`re_cut`'s answer: its pieces are the task's children."""
    if folder:
        return (
            f"{task} replaced; add its pieces to plan/ as {task}.1, {task}.2, … (each with its "
            f"file tasks/{task}.1-<slug>.md), call plan, then delegate them"
        )
    return f"{task} replaced; delegate its pieces, which are named {task}.1, {task}.2, …"


_COMMIT_KINDS: Final = frozenset(
    {EffectKind.TASK_COMMIT.value, EffectKind.TASK_KNOWN_GOOD.value}
)


def known_good(run: RunProjection, blob: Callable[[str], bytes]) -> tuple[tuple[str, str], ...]:
    """Each task accepted in place, as `(subject, body)` its commit would have had: what the
    handoff offers for the person's own commit, since in place nothing is
    committed. In the order delegated."""
    out: list[tuple[str, str]] = []
    for entry in TaskBook(run).entries():
        done = [
            op
            for op in _commit_ops(run, entry.id)
            if op.kind == EffectKind.TASK_KNOWN_GOOD.value
            and op.completed is not None
            and op.completed.outcome == "succeeded"
        ]
        if not done:
            continue
        body = _task_body(run, entry.id, _delegated(run, entry.id), blob)
        out.append((_subject(entry.id, entry.title), body))
    return tuple(out)


def _task_body(
    run: RunProjection, task: str, line: TaskDelegatedP, blob: Callable[[str], bytes]
) -> str:
    """The task's commit body, one assembly for the commit made and the one
    offered in place: from its brief as delegated, the facts of its last report, under the round
    it was accepted in, and the plan's growth (decision 19)."""
    brief = blob(line.brief_sha).decode("utf-8")
    last = _last(run, task)
    commands = last.facts.commands if last is not None else ()
    book = TaskBook(run)
    entry = book.get(task)
    rounds = entry.rounds if entry is not None else 0
    return task_body(brief, rounds, commands, growth=growth(run, book))


def _commit_ops(run: RunProjection, task: str) -> list[Operation]:
    """`task`'s commit attempts, in the order intended."""
    return [
        op for op in run.operations.values() if op.task_id == task and op.kind in _COMMIT_KINDS
    ]


def _delegated(run: RunProjection, task: str) -> TaskDelegatedP:
    for line in run.tasks.get(task, ()):
        if isinstance(line.payload, TaskDelegatedP):
            return line.payload
    raise ValueError(f"{task} was never delegated")


def _last(run: RunProjection, task: str) -> TaskReportedP | None:
    found = [p.payload for p in run.tasks.get(task, ()) if isinstance(p.payload, TaskReportedP)]
    return found[-1] if found else None


def _answers(run: RunProjection, task: str) -> tuple[ReviewAnswer, ...]:
    return tuple(
        p.payload.answer
        for p in run.tasks.get(task, ())
        if isinstance(p.payload, TaskReviewedP)
    )


def _reviewed_round(run: RunProjection, task: str) -> int:
    """The rounds `task` had reported when it was last reviewed."""
    rounds = 0
    at = 0
    for line in run.tasks.get(task, ()):
        if isinstance(line.payload, TaskReportedP):
            rounds = line.payload.round
        elif isinstance(line.payload, TaskReviewedP):
            at = rounds
    return at


def _fresh(answer: ReviewAnswer, fresh: Fresh) -> Fresh:
    """`fresh` as the line keeps it: only where it means something, `False` as None, and a
    list as sorted unique paths, so a resume reads one answer."""
    if answer not in ("feedback", "take_over", "re_cut") or not fresh:
        return None
    return True if fresh is True else tuple(sorted(set(fresh)))


def _noted(said: str, note: str | None) -> str:
    """`said`, then a hook's note about it."""
    return said if note is None else f"{said}\n\n{note}"


def _with_put_back(
    said: str, back: Sequence[str], *, worker: bool, by: str | None = None
) -> str:
    """`said`, then each file a review's `fresh` put back: the worker's next
    message names them as the reviewer's, or the harness's for a hook's retry, her
    answer as her own."""
    if not back:
        return said
    who = ("The harness put" if by else "The reviewer put") if worker else "Put"
    return f"{said}\n\n{who} back to the task's start: {', '.join(back)}."


def _with_files(
    brief: str, work: WorkLine, plan: str, design: Sequence[tuple[str, str]]
) -> str:
    """A task's pinned brief, then where the work's documents are (decision 15, A7): the
    design's README and files, and the plan's, each by its path, read-only to the worker."""
    base = f"{work.dir}/{work.name}"
    rows = [f"- `{base}/design/README.md`: the approved design"]
    rows += [f"- `{base}/design/{file}`: {line}" for file, line in design]
    rows += [f"- `{base}/plan/README.md`: the approved plan"]
    rows += [
        f"- `{base}/plan/{file}`: {task}, {title}" for task, title, file in plan_tasks(plan)
    ]
    return "\n\n".join(
        (
            brief.strip(),
            "## The work's documents (read them as you need; never write them)",
            "\n".join(rows),
        )
    )


def _carried_note(carried: PriorTask, changed: Sequence[str]) -> str:
    """What the worker of a task an earlier run left unfinished is told first."""
    name = carried.entry.name
    head = (
        "## Where this task stands\n"
        f"{name} was started in an earlier run of this work ({carried.run_id}), which stopped "
        "before it was reviewed."
    )
    if not changed:
        return f"{head} Nothing it changed is left: start it from the brief."
    return (
        f"{head} What it changed is still here, uncommitted: {', '.join(changed)}. Go on from "
        "it: read those files first."
    )
